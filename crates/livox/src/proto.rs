//! SDK1 control-protocol framing, command ids and payload layouts.
//!
//! Frame: `sof(0xAA) ver(1) len:u16 type:u8 seq:u16 crc16:u16 | cmd_set cmd_id payload | crc32:u32`.
//! Integers are little-endian. IPv4 addresses travel as their four dotted bytes, in order.
//! Layouts follow Livox-SDK `livox_def.h` / `command_impl.h` (all structs are `#pragma pack(1)`).

use crate::crc::{crc16, crc32};
use std::fmt;
use std::net::Ipv4Addr;

pub const SOF: u8 = 0xAA;
pub const VERSION: u8 = 1;
pub const PREAMBLE_LEN: usize = 9;
pub const HEADER_LEN: usize = PREAMBLE_LEN + 2;
pub const WRAPPER_LEN: usize = HEADER_LEN + 4;

/// Port every host receives LiDAR discovery broadcasts on.
pub const BROADCAST_PORT: u16 = 55000;
/// Port the LiDAR listens for commands on.
pub const LIDAR_CMD_PORT: u16 = 65000;

pub mod ptype {
    pub const CMD: u8 = 0;
    pub const ACK: u8 = 1;
    pub const MSG: u8 = 2;
}

pub mod cmd_set {
    pub const GENERAL: u8 = 0;
    pub const LIDAR: u8 = 1;
}

pub mod general {
    pub const BROADCAST: u8 = 0x00;
    pub const HANDSHAKE: u8 = 0x01;
    pub const DEVICE_INFO: u8 = 0x02;
    pub const HEARTBEAT: u8 = 0x03;
    pub const SAMPLING: u8 = 0x04;
    pub const COORDINATE: u8 = 0x05;
    pub const DISCONNECT: u8 = 0x06;
    pub const ABNORMAL_STATUS: u8 = 0x07;
    pub const SET_IP: u8 = 0x08;
    pub const GET_IP: u8 = 0x09;
    pub const REBOOT: u8 = 0x0A;
}

pub mod lidar {
    pub const SET_MODE: u8 = 0x00;
    pub const SET_EXTRINSIC: u8 = 0x01;
    pub const GET_EXTRINSIC: u8 = 0x02;
    pub const RAIN_FOG: u8 = 0x03;
    pub const SET_FAN: u8 = 0x04;
    pub const GET_FAN: u8 = 0x05;
    pub const SET_RETURN_MODE: u8 = 0x06;
    pub const GET_RETURN_MODE: u8 = 0x07;
    pub const SET_IMU_RATE: u8 = 0x08;
    pub const GET_IMU_RATE: u8 = 0x09;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    pub ptype: u8,
    pub seq: u16,
    pub cmd_set: u8,
    pub cmd_id: u8,
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    Short,
    Sof,
    Length,
    Crc16,
    Crc32,
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FrameError::Short => "frame shorter than the 15-byte wrapper",
            FrameError::Sof => "missing 0xAA start byte",
            FrameError::Length => "length field disagrees with the datagram",
            FrameError::Crc16 => "preamble CRC-16 mismatch",
            FrameError::Crc32 => "frame CRC-32 mismatch",
        };
        f.write_str(s)
    }
}

impl std::error::Error for FrameError {}

pub fn encode(ptype: u8, seq: u16, cmd_set: u8, cmd_id: u8, payload: &[u8]) -> Vec<u8> {
    let len = WRAPPER_LEN + payload.len();
    debug_assert!(
        len <= u16::MAX as usize,
        "frame of {len} bytes overflows the u16 length field"
    );
    let mut b = Vec::with_capacity(len);
    b.push(SOF);
    b.push(VERSION);
    b.extend_from_slice(&(len as u16).to_le_bytes());
    b.push(ptype);
    b.extend_from_slice(&seq.to_le_bytes());
    let c16 = crc16(&b[..7]);
    b.extend_from_slice(&c16.to_le_bytes());
    b.push(cmd_set);
    b.push(cmd_id);
    b.extend_from_slice(payload);
    let c32 = crc32(&b);
    b.extend_from_slice(&c32.to_le_bytes());
    b
}

pub fn decode(buf: &[u8]) -> Result<Frame<'_>, FrameError> {
    if buf.len() < WRAPPER_LEN {
        return Err(FrameError::Short);
    }
    if buf[0] != SOF {
        return Err(FrameError::Sof);
    }
    let len = u16::from_le_bytes([buf[2], buf[3]]) as usize;
    if len < WRAPPER_LEN || len > buf.len() {
        return Err(FrameError::Length);
    }
    if crc16(&buf[..7]) != u16::from_le_bytes([buf[7], buf[8]]) {
        return Err(FrameError::Crc16);
    }
    let want = u32::from_le_bytes([buf[len - 4], buf[len - 3], buf[len - 2], buf[len - 1]]);
    if crc32(&buf[..len - 4]) != want {
        return Err(FrameError::Crc32);
    }
    Ok(Frame {
        ptype: buf[4],
        seq: u16::from_le_bytes([buf[5], buf[6]]),
        cmd_set: buf[9],
        cmd_id: buf[10],
        payload: &buf[HEADER_LEN..len - 4],
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Hub,
    Mid40,
    Tele15,
    Horizon,
    Mid70,
    Avia,
    Unknown(u8),
}

impl DeviceType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Hub,
            1 => Self::Mid40,
            2 => Self::Tele15,
            3 => Self::Horizon,
            6 => Self::Mid70,
            7 => Self::Avia,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Hub => 0,
            Self::Mid40 => 1,
            Self::Tele15 => 2,
            Self::Horizon => 3,
            Self::Mid70 => 6,
            Self::Avia => 7,
            Self::Unknown(v) => v,
        }
    }

    /// Time between successive firings (all returns of one firing share it).
    pub fn sample_interval_ns(self) -> u64 {
        match self {
            Self::Tele15 | Self::Horizon | Self::Avia => 4_167, // 240k samples/s
            _ => 10_000,                                        // 100k samples/s
        }
    }
}

/// The marketing name; unknown codes print as `type N`. Honours width and alignment.
impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hub => f.pad("Hub"),
            Self::Mid40 => f.pad("Mid-40"),
            Self::Tele15 => f.pad("Tele-15"),
            Self::Horizon => f.pad("Horizon"),
            Self::Mid70 => f.pad("Mid-70"),
            Self::Avia => f.pad("Avia"),
            Self::Unknown(v) => f.pad(&format!("type {v}")),
        }
    }
}

/// Parsed discovery broadcast (general 0x00, sent as a MSG from the LiDAR).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Broadcast {
    pub code: String,
    pub dev_type: DeviceType,
}

pub fn parse_broadcast(f: &Frame<'_>) -> Option<Broadcast> {
    if f.cmd_set != cmd_set::GENERAL || f.cmd_id != general::BROADCAST || f.payload.len() < 17 {
        return None;
    }
    Some(Broadcast {
        code: cstr(&f.payload[..16]),
        dev_type: DeviceType::from_u8(f.payload[16]),
    })
}

/// The [`StatusBits`] word of an abnormal-status push (general 0x07, sent as a MSG).
pub fn parse_abnormal_status(f: &Frame<'_>) -> Option<u32> {
    if f.cmd_set != cmd_set::GENERAL || f.cmd_id != general::ABNORMAL_STATUS {
        return None;
    }
    let [a, b, c, d, ..] = *f.payload else {
        return None;
    };
    Some(u32::from_le_bytes([a, b, c, d]))
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// HandshakeRequest: where the LiDAR should send acks (cmd), points (data) and IMU (sensor).
pub fn handshake_payload(host: Ipv4Addr, data_port: u16, cmd_port: u16, imu_port: u16) -> Vec<u8> {
    let mut p = host.octets().to_vec();
    p.extend_from_slice(&data_port.to_le_bytes());
    p.extend_from_slice(&cmd_port.to_le_bytes());
    p.extend_from_slice(&imu_port.to_le_bytes());
    p
}

/// SetDeviceIpExtendModeRequest; every SDK path (Mid-40 included) sends this 13-byte form.
pub fn set_ip_payload(ip: &IpInfo) -> Vec<u8> {
    let mut p = vec![if ip.dynamic { 0 } else { 1 }];
    p.extend_from_slice(&ip.ip.octets());
    p.extend_from_slice(&ip.mask.octets());
    p.extend_from_slice(&ip.gateway.octets());
    p
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpInfo {
    pub dynamic: bool,
    pub ip: Ipv4Addr,
    pub mask: Ipv4Addr,
    pub gateway: Ipv4Addr,
}

impl IpInfo {
    /// GetDeviceIpModeResponse minus the leading ret_code. Mid-40 firmware sends only mode + ip;
    /// like the SDK (command_handler.cpp), assume a /24 then, and report the gateway as 0.0.0.0.
    pub fn parse(p: &[u8]) -> Option<Self> {
        if p.len() < 5 {
            return None;
        }
        let ip4 = |o: usize| Ipv4Addr::new(p[o], p[o + 1], p[o + 2], p[o + 3]);
        let (mask, gateway) = if p.len() >= 13 {
            (ip4(5), ip4(9))
        } else {
            (Ipv4Addr::new(255, 255, 255, 0), Ipv4Addr::UNSPECIFIED)
        };
        Some(Self {
            dynamic: p[0] == 0,
            ip: ip4(1),
            mask,
            gateway,
        })
    }
}

impl fmt::Display for IpInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = if self.dynamic { "dynamic" } else { "static" };
        write!(
            f,
            "{mode} {} mask {} gw {}",
            self.ip, self.mask, self.gateway
        )
    }
}

/// LiDAR mounting pose stored on the device (degrees, millimetres).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Extrinsic {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Extrinsic {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(24);
        for v in [self.roll, self.pitch, self.yaw] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        for v in [self.x, self.y, self.z] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p
    }

    /// LidarGetExtrinsicParameterResponse minus the leading ret_code.
    pub fn parse(p: &[u8]) -> Option<Self> {
        if p.len() < 24 {
            return None;
        }
        let f = |o: usize| f32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
        let i = |o: usize| i32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
        Some(Self {
            roll: f(0),
            pitch: f(4),
            yaw: f(8),
            x: i(12),
            y: i(16),
            z: i(20),
        })
    }
}

/// Heartbeat ack body (after ret_code): work state, feature flags, status/progress word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Heartbeat {
    pub state: u8,
    pub feature: u8,
    pub status: u32,
}

impl Heartbeat {
    pub fn parse(p: &[u8]) -> Option<Self> {
        if p.len() < 6 {
            return None;
        }
        Some(Self {
            state: p[0],
            feature: p[1],
            status: u32::from_le_bytes([p[2], p[3], p[4], p[5]]),
        })
    }

    pub fn state_name(&self) -> &'static str {
        match self.state {
            0 => "initializing",
            1 => "normal",
            2 => "power-saving",
            3 => "standby",
            4 => "error",
            _ => "unknown",
        }
    }
}

/// LidarErrorCode bitfield, carried by heartbeats, abnormal-status pushes and every data packet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StatusBits(pub u32);

impl StatusBits {
    fn field(&self, shift: u32, bits: u32) -> u32 {
        (self.0 >> shift) & ((1 << bits) - 1)
    }
    pub fn temperature(&self) -> u32 {
        self.field(0, 2)
    }
    pub fn voltage(&self) -> u32 {
        self.field(2, 2)
    }
    pub fn motor(&self) -> u32 {
        self.field(4, 2)
    }
    pub fn dirty(&self) -> u32 {
        self.field(6, 2)
    }
    pub fn firmware_error(&self) -> bool {
        self.field(8, 1) != 0
    }
    pub fn pps(&self) -> bool {
        self.field(9, 1) != 0
    }
    pub fn end_of_life(&self) -> bool {
        self.field(10, 1) != 0
    }
    pub fn fan_warning(&self) -> bool {
        self.field(11, 1) != 0
    }
    pub fn self_heating(&self) -> bool {
        self.field(12, 1) != 0
    }
    pub fn ptp(&self) -> bool {
        self.field(13, 1) != 0
    }
    pub fn time_sync(&self) -> u32 {
        self.field(14, 3)
    }
    pub fn system(&self) -> u32 {
        self.field(30, 2)
    }

    /// Human-readable list of everything that is not nominal.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        let level = |v: u32| if v >= 2 { "critical" } else { "warning" };
        if self.temperature() != 0 {
            out.push(format!("temperature {}", level(self.temperature())));
        }
        if self.voltage() != 0 {
            out.push(format!("voltage {}", level(self.voltage())));
        }
        if self.motor() != 0 {
            out.push(format!("motor {}", level(self.motor())));
        }
        if self.dirty() != 0 {
            out.push("window dirty or blocked".into());
        }
        if self.firmware_error() {
            out.push("firmware abnormal, needs upgrade".into());
        }
        if self.end_of_life() {
            out.push("approaching end of service life".into());
        }
        if self.fan_warning() {
            out.push("fan warning".into());
        }
        if self.self_heating() {
            out.push("low-temperature self heating".into());
        }
        if self.time_sync() == 4 {
            out.push("time sync abnormal".into());
        }
        if self.system() != 0 {
            out.push(format!("system {}", level(self.system())));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let f = encode(
            ptype::CMD,
            0x1234,
            cmd_set::GENERAL,
            general::SAMPLING,
            &[1],
        );
        assert_eq!(f.len(), WRAPPER_LEN + 1);
        let d = decode(&f).unwrap();
        assert_eq!(
            (d.ptype, d.seq, d.cmd_set, d.cmd_id, d.payload),
            (0, 0x1234, 0, 4, &[1u8][..])
        );
    }

    #[test]
    fn corrupt_payload_fails_crc32() {
        let mut f = encode(ptype::CMD, 7, cmd_set::LIDAR, lidar::SET_MODE, &[1]);
        f[11] ^= 0xFF;
        assert_eq!(decode(&f), Err(FrameError::Crc32));
    }

    #[test]
    fn broadcast_layout() {
        let mut payload = b"TESTSERIAL00001\0".to_vec();
        payload.extend_from_slice(&[6, 0, 0]);
        let f = encode(ptype::MSG, 0, 0, 0, &payload);
        assert_eq!(f.len(), 34); // matches the 34-byte datagrams seen on the wire
        let b = parse_broadcast(&decode(&f).unwrap()).unwrap();
        assert_eq!(b.code, "TESTSERIAL00001");
        assert_eq!(b.dev_type, DeviceType::Mid70);
    }

    #[test]
    fn abnormal_status_needs_a_whole_word() {
        let f = encode(
            ptype::MSG,
            0,
            cmd_set::GENERAL,
            general::ABNORMAL_STATUS,
            &[1, 0, 0, 0x40],
        );
        assert_eq!(
            parse_abnormal_status(&decode(&f).unwrap()),
            Some(0x4000_0001)
        );
        let short = encode(
            ptype::MSG,
            0,
            cmd_set::GENERAL,
            general::ABNORMAL_STATUS,
            &[1, 0],
        );
        assert_eq!(parse_abnormal_status(&decode(&short).unwrap()), None);
        let other = encode(
            ptype::MSG,
            0,
            cmd_set::GENERAL,
            general::HEARTBEAT,
            &[1, 0, 0, 0],
        );
        assert_eq!(parse_abnormal_status(&decode(&other).unwrap()), None);
    }

    #[test]
    fn device_type_display_pads() {
        assert_eq!(format!("{:<9}|", DeviceType::Mid40), "Mid-40   |");
        assert_eq!(DeviceType::Unknown(9).to_string(), "type 9");
    }

    #[test]
    fn set_ip_is_13_bytes_dotted_order() {
        let p = set_ip_payload(&IpInfo {
            dynamic: false,
            ip: Ipv4Addr::new(10, 1, 2, 60),
            mask: Ipv4Addr::new(255, 255, 255, 0),
            gateway: Ipv4Addr::new(10, 1, 2, 1),
        });
        assert_eq!(p, [1, 10, 1, 2, 60, 255, 255, 255, 0, 10, 1, 2, 1]);
    }
}
