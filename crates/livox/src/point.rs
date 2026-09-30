//! Point-cloud / IMU data packets (the LiDAR's UDP stream to the host data port).
//!
//! Header (18 bytes): `version, slot, id, rsvd, status:u32, timestamp_type, data_type, timestamp[8]`.
//! Records follow; their layout depends on `data_type` (livox_def.h `PointDataType`).

use crate::proto::DeviceType;
use std::fmt;

pub const DATA_HEADER_LEN: usize = 18;

/// Records in every Mid-40 point packet (both the standard firmware and the dual-return one).
pub const MID40_RECORDS_PER_PACKET: usize = 100;

/// How point records map to firings.
///
/// `Mid40Dual`: the Mid-40 dual-return firmware (03.03.0001 and 03.03.0006) sends the 100 plain
/// Cartesian or spherical records per packet that a standard Mid-40 does, but they are really 50
/// firings x 2 echoes, interleaved (first echo, second echo, first echo, ...). The SDK's
/// extended dual-return record types are not involved. Stored as a byte in LVXR2 recordings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamProfile {
    #[default]
    Standard = 0,
    Mid40Dual = 1,
}

impl StreamProfile {
    pub fn from_firmware(dev: DeviceType, fw: [u8; 4]) -> Self {
        if dev == DeviceType::Mid40 && matches!(fw, [3, 3, 0, 1] | [3, 3, 0, 6]) {
            Self::Mid40Dual
        } else {
            Self::Standard
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Mid40Dual => "mid40-dual",
        }
    }
}

/// A profile byte (as stored in an LVXR2 recording) that names no known [`StreamProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownProfile(pub u8);

impl fmt::Display for UnknownProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown stream profile {}", self.0)
    }
}

impl std::error::Error for UnknownProfile {}

impl TryFrom<u8> for StreamProfile {
    type Error = UnknownProfile;

    fn try_from(v: u8) -> Result<Self, UnknownProfile> {
        match v {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Mid40Dual),
            other => Err(UnknownProfile(other)),
        }
    }
}

/// [`for_each_point`] for a stream of the given profile.
///
/// For `Mid40Dual` the interleaved records are re-labelled so that `record` is the firing index
/// (0-49) and `ret` the echo (0 or 1). A dual packet that is not a full
/// [`MID40_RECORDS_PER_PACKET`] plain records is rejected, since the pairing would be off.
pub fn for_each_point_profile(
    buf: &[u8],
    profile: StreamProfile,
    mut f: impl FnMut(Point),
) -> Option<DataHeader> {
    if profile == StreamProfile::Standard {
        return for_each_point(buf, f);
    }
    let h = DataHeader::parse(buf)?;
    if !matches!(h.data_type, data_type::CARTESIAN | data_type::SPHERICAL)
        || records_in(buf, h.data_type) != MID40_RECORDS_PER_PACKET
    {
        return None;
    }
    for_each_point(buf, |mut p| {
        // First echoes can also have reflectivity 200. Identify returns by wire slot.
        p.ret = (p.record % 2) as u8;
        p.record /= 2;
        f(p);
    })
}

pub mod data_type {
    pub const CARTESIAN: u8 = 0;
    pub const SPHERICAL: u8 = 1;
    pub const EXT_CARTESIAN: u8 = 2;
    pub const EXT_SPHERICAL: u8 = 3;
    pub const DUAL_CARTESIAN: u8 = 4;
    pub const DUAL_SPHERICAL: u8 = 5;
    pub const IMU: u8 = 6;
    pub const TRIPLE_CARTESIAN: u8 = 7;
    pub const TRIPLE_SPHERICAL: u8 = 8;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DataHeader {
    pub version: u8,
    pub slot: u8,
    pub id: u8,
    pub status: u32,
    pub timestamp_type: u8,
    pub data_type: u8,
    pub timestamp: [u8; 8],
}

impl DataHeader {
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < DATA_HEADER_LEN {
            return None;
        }
        let mut ts = [0u8; 8];
        ts.copy_from_slice(&buf[10..18]);
        Some(Self {
            version: buf[0],
            slot: buf[1],
            id: buf[2],
            status: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            timestamp_type: buf[8],
            data_type: buf[9],
            timestamp: ts,
        })
    }

    /// Timestamp in ns. Types 0/1/4 (none, PTP, PPS) are already ns; type 3 is GPS UTC
    /// (year, month, day, hour, then microseconds within the hour), folded to ns-of-day.
    pub fn timestamp_ns(&self) -> u64 {
        let t = &self.timestamp;
        if self.timestamp_type == 3 {
            let hour = t[3] as u64;
            let us = u32::from_le_bytes([t[4], t[5], t[6], t[7]]) as u64;
            (hour * 3_600_000_000 + us) * 1_000
        } else {
            u64::from_le_bytes(*t)
        }
    }
}

/// Bytes per record and returns per record, or None for an unknown data type.
pub fn record_layout(data_type: u8) -> Option<(usize, usize)> {
    Some(match data_type {
        data_type::CARTESIAN => (13, 1),
        data_type::SPHERICAL => (9, 1),
        data_type::EXT_CARTESIAN => (14, 1),
        data_type::EXT_SPHERICAL => (10, 1),
        data_type::DUAL_CARTESIAN => (28, 2),
        data_type::DUAL_SPHERICAL => (16, 2),
        data_type::IMU => (24, 0),
        data_type::TRIPLE_CARTESIAN => (42, 3),
        data_type::TRIPLE_SPHERICAL => (22, 3),
        _ => return None,
    })
}

/// One return. Metres, LiDAR frame (x forward, y left, z up).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub reflectivity: u8,
    /// Extended types only: bits 0-1 spatial noise confidence, 2-3 intensity noise confidence,
    /// 4-5 return number. Zero for the plain Mid-40 types.
    pub tag: u8,
    /// Which return of a dual/triple record this is (0-based); 0 for single-return types.
    pub ret: u8,
    /// Record (firing) index inside the packet; firing time = packet time + record * interval.
    pub record: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Imu {
    /// rad/s
    pub gyro: [f32; 3],
    /// g
    pub acc: [f32; 3],
}

fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn cartesian(b: &[u8], o: usize) -> (f32, f32, f32) {
    (
        i32_at(b, o) as f32 * 1e-3,
        i32_at(b, o + 4) as f32 * 1e-3,
        i32_at(b, o + 8) as f32 * 1e-3,
    )
}

/// depth in mm, zenith theta and azimuth phi in 0.01 degree.
fn spherical(depth_mm: u32, theta: u16, phi: u16) -> (f32, f32, f32) {
    let d = depth_mm as f32 * 1e-3;
    let th = (theta as f32 * 0.01).to_radians();
    let ph = (phi as f32 * 0.01).to_radians();
    (
        d * th.sin() * ph.cos(),
        d * th.sin() * ph.sin(),
        d * th.cos(),
    )
}

/// Decode every real return in a point packet, skipping the (0,0,0) "no return" records.
/// Returns the header, or None if the packet is malformed or is not point data.
pub fn for_each_point(buf: &[u8], mut f: impl FnMut(Point)) -> Option<DataHeader> {
    let h = DataHeader::parse(buf)?;
    let (size, returns) = record_layout(h.data_type)?;
    if returns == 0 {
        return None;
    }
    let body = &buf[DATA_HEADER_LEN..];
    if body.is_empty() || !body.len().is_multiple_of(size) {
        return None;
    }
    let records = body.len() / size;
    for r in 0..records {
        let o = r * size;
        let rec = &body[o..o + size];
        let mut emit = |(x, y, z): (f32, f32, f32), reflectivity: u8, tag: u8, ret: u8| {
            if x != 0.0 || y != 0.0 || z != 0.0 {
                f(Point {
                    x,
                    y,
                    z,
                    reflectivity,
                    tag,
                    ret,
                    record: r as u16,
                });
            }
        };
        match h.data_type {
            data_type::CARTESIAN => emit(cartesian(rec, 0), rec[12], 0, 0),
            data_type::SPHERICAL => emit(
                spherical(u32_at(rec, 0), u16_at(rec, 4), u16_at(rec, 6)),
                rec[8],
                0,
                0,
            ),
            data_type::EXT_CARTESIAN => {
                emit(cartesian(rec, 0), rec[12], rec[13], (rec[13] >> 4) & 3)
            }
            data_type::EXT_SPHERICAL => emit(
                spherical(u32_at(rec, 0), u16_at(rec, 4), u16_at(rec, 6)),
                rec[8],
                rec[9],
                (rec[9] >> 4) & 3,
            ),
            data_type::DUAL_CARTESIAN | data_type::TRIPLE_CARTESIAN => {
                for k in 0..returns {
                    let b = k * 14;
                    emit(cartesian(rec, b), rec[b + 12], rec[b + 13], k as u8);
                }
            }
            data_type::DUAL_SPHERICAL | data_type::TRIPLE_SPHERICAL => {
                let (theta, phi) = (u16_at(rec, 0), u16_at(rec, 2));
                for k in 0..returns {
                    let b = 4 + k * 6;
                    emit(
                        spherical(u32_at(rec, b), theta, phi),
                        rec[b + 4],
                        rec[b + 5],
                        k as u8,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
    Some(h)
}

/// Records per packet for this data type and datagram length (for loss accounting).
pub fn records_in(buf: &[u8], data_type: u8) -> usize {
    match record_layout(data_type) {
        Some((size, _)) if buf.len() >= DATA_HEADER_LEN => (buf.len() - DATA_HEADER_LEN) / size,
        _ => 0,
    }
}

pub fn parse_imu(buf: &[u8]) -> Option<Imu> {
    let h = DataHeader::parse(buf)?;
    if h.data_type != data_type::IMU || buf.len() < DATA_HEADER_LEN + 24 {
        return None;
    }
    let b = &buf[DATA_HEADER_LEN..];
    Some(Imu {
        gyro: [f32_at(b, 0), f32_at(b, 4), f32_at(b, 8)],
        acc: [f32_at(b, 12), f32_at(b, 16), f32_at(b, 20)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(data_type: u8, ts: u64) -> Vec<u8> {
        let mut b = vec![5, 0, 1, 0, 0, 0, 0, 0, 0, data_type];
        b.extend_from_slice(&ts.to_le_bytes());
        b
    }

    #[test]
    fn extended_cartesian_skips_empty_returns() {
        let mut b = header(data_type::EXT_CARTESIAN, 1_000);
        for (x, refl, tag) in [(1500i32, 40u8, 0x10u8), (0, 0, 0), (-2500, 200, 0)] {
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&250i32.to_le_bytes());
            b.extend_from_slice(&(-100i32).to_le_bytes());
            b.push(refl);
            b.push(tag);
        }
        // record 1 is all-zero except y/z: patch it to a true no-return
        let r1 = DATA_HEADER_LEN + 14;
        b[r1..r1 + 12].fill(0);
        let mut pts = Vec::new();
        let h = for_each_point(&b, |p| pts.push(p)).unwrap();
        assert_eq!(h.timestamp_ns(), 1_000);
        assert_eq!(pts.len(), 2);
        assert!((pts[0].x - 1.5).abs() < 1e-6 && (pts[0].y - 0.25).abs() < 1e-6);
        assert_eq!((pts[0].reflectivity, pts[0].ret, pts[0].record), (40, 1, 0));
        assert_eq!((pts[1].reflectivity, pts[1].record), (200, 2));
    }

    #[test]
    fn dual_spherical_shares_direction() {
        let mut b = header(data_type::DUAL_SPHERICAL, 0);
        b.extend_from_slice(&9000u16.to_le_bytes()); // theta 90 deg: horizontal
        b.extend_from_slice(&0u16.to_le_bytes()); // phi 0: +x
        for d in [2000u32, 4000] {
            b.extend_from_slice(&d.to_le_bytes());
            b.push(10);
            b.push(0);
        }
        let mut pts = Vec::new();
        for_each_point(&b, |p| pts.push(p)).unwrap();
        assert_eq!(pts.len(), 2);
        assert!((pts[0].x - 2.0).abs() < 1e-4 && pts[0].z.abs() < 1e-4);
        assert!((pts[1].x - 4.0).abs() < 1e-4 && pts[1].ret == 1);
    }

    #[test]
    fn mid40_dual_preserves_slots_across_empty_returns() {
        let mut b = header(data_type::CARTESIAN, 0);
        for slot in 0..100 {
            let x: i32 = if slot == 1 { 0 } else { 1000 + slot * 100 };
            b.extend_from_slice(&x.to_le_bytes());
            b.extend_from_slice(&[0; 8]);
            b.push(200); // Both first and second can be 200: intensity is not a discriminator.
        }
        let mut points = Vec::new();
        for_each_point_profile(&b, StreamProfile::Mid40Dual, |p| points.push(p)).unwrap();
        assert_eq!(points.len(), 99);
        assert_eq!((points[0].ret, points[0].record), (0, 0));
        assert_eq!((points[1].ret, points[1].record), (0, 1));
        assert_eq!((points[2].ret, points[2].record), (1, 1));
        assert_eq!((points[98].ret, points[98].record), (1, 49));
        b.pop();
        assert!(
            for_each_point_profile(&b, StreamProfile::Mid40Dual, |_| panic!(
                "partial packet emitted"
            ))
            .is_none()
        );
    }

    #[test]
    fn profile_byte_roundtrip() {
        for p in [StreamProfile::Standard, StreamProfile::Mid40Dual] {
            assert_eq!(StreamProfile::try_from(p as u8), Ok(p));
        }
        assert_eq!(StreamProfile::try_from(7), Err(UnknownProfile(7)));
    }

    #[test]
    fn mid40_dual_spherical_and_firmware_detection() {
        assert_eq!(
            StreamProfile::from_firmware(DeviceType::Mid40, [3, 3, 0, 6]),
            StreamProfile::Mid40Dual
        );
        assert_eq!(
            StreamProfile::from_firmware(DeviceType::Mid40, [3, 3, 0, 4]),
            StreamProfile::Standard
        );
        assert_eq!(
            StreamProfile::from_firmware(DeviceType::Avia, [3, 3, 0, 6]),
            StreamProfile::Standard
        );
        let mut b = header(data_type::SPHERICAL, 0);
        for slot in 0..100 {
            b.extend_from_slice(&(1000u32 + slot * 100).to_le_bytes());
            b.extend_from_slice(&9000u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.push(200);
        }
        let mut points = Vec::new();
        for_each_point_profile(&b, StreamProfile::Mid40Dual, |p| points.push(p)).unwrap();
        assert_eq!(points.len(), 100);
        assert_eq!((points[1].ret, points[1].record), (1, 0));
        assert!((points[1].x - 1.1).abs() < 1e-5);
    }
}
