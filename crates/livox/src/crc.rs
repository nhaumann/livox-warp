//! The two checksums every SDK1 frame carries.
//!
//! Taken from Livox-SDK `sdk_core/src/comm/comm_port.cpp` (`SdkProtocol(0x4c49, 0x564f580a)`)
//! and its FastCRC fork: CRC-16/MCRF4XX over the 7-byte preamble, and the standard reflected
//! CRC-32 over the whole frame, each started from a Livox-specific seed.

pub const CRC16_SEED: u16 = 0x4C49;
pub const CRC32_SEED: u32 = 0x564F_580A;

const fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u16;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x8408
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC16_TABLE: [u16; 256] = crc16_table();
static CRC32_TABLE: [u32; 256] = crc32_table();

/// CRC-16/MCRF4XX (reflected 0x1021, no final xor) seeded with [`CRC16_SEED`].
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = CRC16_SEED;
    for &b in data {
        crc = (crc >> 8) ^ CRC16_TABLE[((crc ^ b as u16) & 0xFF) as usize];
    }
    crc
}

/// Reflected CRC-32 (0xEDB88320, final xor) seeded with [`CRC32_SEED`].
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = CRC32_SEED ^ 0xFFFF_FFFF;
    for &b in data {
        crc = (crc >> 8) ^ CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32_standard(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc = (crc >> 8) ^ CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize];
        }
        crc ^ 0xFFFF_FFFF
    }

    #[test]
    fn crc32_table_is_the_standard_one() {
        // The seed is Livox's; the table and reflection are plain CRC-32 ("123456789" check value).
        assert_eq!(crc32_standard(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc16_residue_is_zero() {
        // SdkProtocol::CheckPreamble relies on this: CRC over data + its own LE CRC == 0.
        let data = [0xAA, 0x01, 0x0F, 0x00, 0x00, 0x34, 0x12];
        let c = crc16(&data);
        let mut with = data.to_vec();
        with.extend_from_slice(&c.to_le_bytes());
        assert_eq!(crc16(&with), 0);
    }
}
