//! Raw data-port capture, so a session can be replayed bit-for-bit without the sensor.
//!
//! File: `LVXR2\n`, dev_type:u8, profile:u8, then `host_ns:u64 len:u16 datagram[len]`, where
//! host_ns is monotonic time since the recording started.
//! Legacy LVXR1 files omit the profile byte and are read as standard streams.

use crate::point::StreamProfile;
use crate::proto::DeviceType;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

pub const MAGIC_V1: &[u8; 6] = b"LVXR1\n";
pub const MAGIC_V2: &[u8; 6] = b"LVXR2\n";

/// Writes an LVXR2 recording.
pub struct Writer {
    w: BufWriter<File>,
    start: Instant,
    pub packets: u64,
}

impl Writer {
    pub fn create(
        path: impl AsRef<Path>,
        dev_type: DeviceType,
        profile: StreamProfile,
    ) -> io::Result<Self> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(MAGIC_V2)?;
        w.write_all(&[dev_type.as_u8(), profile as u8])?;
        Ok(Self {
            w,
            start: Instant::now(),
            packets: 0,
        })
    }

    pub fn write(&mut self, datagram: &[u8]) -> io::Result<()> {
        let len = u16::try_from(datagram.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "datagram longer than 65535 bytes",
            )
        })?;
        let ns = self.start.elapsed().as_nanos() as u64;
        self.w.write_all(&ns.to_le_bytes())?;
        self.w.write_all(&len.to_le_bytes())?;
        self.w.write_all(datagram)?;
        self.packets += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<u64> {
        self.w.flush()?;
        Ok(self.packets)
    }
}

/// Reads an LVXR1 or LVXR2 recording.
pub struct Reader {
    r: BufReader<File>,
    pub dev_type: DeviceType,
    pub stream_profile: StreamProfile,
}

impl Reader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut head = [0u8; 7];
        r.read_exact(&mut head)?;
        let [magic @ .., dev_type] = head;
        let dev_type = DeviceType::from_u8(dev_type);
        let stream_profile = if magic == *MAGIC_V1 {
            StreamProfile::Standard
        } else if magic == *MAGIC_V2 {
            let mut profile = [0u8];
            r.read_exact(&mut profile)?;
            let profile =
                StreamProfile::try_from(profile[0]).map_err(|e| invalid(e.to_string()))?;
            if profile == StreamProfile::Mid40Dual && dev_type != DeviceType::Mid40 {
                return Err(invalid(format!(
                    "profile {} is only valid for a Mid-40, not a {dev_type}",
                    profile.name()
                )));
            }
            profile
        } else {
            return Err(invalid("not an LVXR recording".into()));
        };
        Ok(Self {
            r,
            dev_type,
            stream_profile,
        })
    }

    /// Next `(host_ns, datagram)`, or None at end of file.
    pub fn next_packet(&mut self) -> io::Result<Option<(u64, Vec<u8>)>> {
        let mut head = [0u8; 10];
        match self.r.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let [ns @ .., len_lo, len_hi] = head;
        let ns = u64::from_le_bytes(ns);
        let len = u16::from_le_bytes([len_lo, len_hi]) as usize;
        let mut buf = vec![0u8; len];
        self.r.read_exact(&mut buf)?;
        Ok(Some((ns, buf)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_roundtrip_and_legacy_compatibility() {
        let path = std::env::temp_dir().join(format!("livox-profile-{}.lvxr", std::process::id()));
        let mut writer =
            Writer::create(&path, DeviceType::Mid40, StreamProfile::Mid40Dual).unwrap();
        writer.write(&[1, 2, 3]).unwrap();
        writer.finish().unwrap();
        let mut reader = Reader::open(&path).unwrap();
        assert_eq!(reader.stream_profile, StreamProfile::Mid40Dual);
        assert_eq!(reader.next_packet().unwrap().unwrap().1, [1, 2, 3]);
        assert!(reader.next_packet().unwrap().is_none());
        drop(reader);
        std::fs::write(&path, b"LVXR1\n\x01").unwrap();
        let reader = Reader::open(&path).unwrap();
        assert_eq!(reader.stream_profile, StreamProfile::Standard);
        drop(reader);
        std::fs::write(&path, b"LVXR2\n\x01\xff").unwrap();
        assert!(Reader::open(&path).is_err(), "unknown profile byte");
        std::fs::write(&path, b"LVXR2\n\x07\x01").unwrap();
        assert!(Reader::open(&path).is_err(), "dual profile on a non-Mid-40");
        std::fs::remove_file(path).unwrap();
    }
}
