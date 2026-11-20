//! Container and sample formats for file output (spec §3.1, §10.2).

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileFormat {
    Wave,
    Aiff,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataFormat {
    pub endian: Endian,
    pub float: bool,
    pub bits: u16,
    /// Output sample rate; `None` = the backend's rate.
    pub rate: Option<u32>,
}

pub const FILE_FORMATS: &[(&str, &str, FileFormat)] = &[
    (
        "WAVE",
        "WAVE (.wav) [LEI16 LEI24 LEI32 LEF32]",
        FileFormat::Wave,
    ),
    (
        "AIFF",
        "AIFF (.aiff, .aif) [BEI16 BEI24 BEI32]",
        FileFormat::Aiff,
    ),
    (
        "raw",
        "headerless PCM (.raw, .pcm) [BEI16 BEI24 BEI32 BEF32 LEI16 LEI24 LEI32 LEF32]",
        FileFormat::Raw,
    ),
];

const UNSUPPORTED_DATA: &[&str] = &[
    "aac", "alac", "aach", "aacl", "ima4", "ulaw", "alaw", "opus", "flac", "mp3",
];

impl FileFormat {
    pub fn parse(s: &str) -> Result<FileFormat> {
        FILE_FORMATS
            .iter()
            .find(|f| {
                f.0.eq_ignore_ascii_case(s)
                    || (s.eq_ignore_ascii_case("wav") && f.2 == FileFormat::Wave)
            })
            .map(|f| f.2)
            .ok_or_else(|| {
                Error::usage(format!(
                    "--file-format {s}: expected WAVE, AIFF or raw. Run speak --file-format='?'"
                ))
            })
    }

    /// From the extension; no or unknown extension → AIFF, as `say`.
    pub fn from_path(path: &std::path::Path) -> FileFormat {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("wav" | "wave") => FileFormat::Wave,
            Some("raw" | "pcm") => FileFormat::Raw,
            _ => FileFormat::Aiff,
        }
    }

    pub fn default_data(self) -> DataFormat {
        let endian = if self == FileFormat::Aiff {
            Endian::Big
        } else {
            Endian::Little
        };
        DataFormat {
            endian,
            float: false,
            bits: 16,
            rate: None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FileFormat::Wave => "WAVE",
            FileFormat::Aiff => "AIFF",
            FileFormat::Raw => "raw",
        }
    }

    /// Checks the pair; a rate given without a sample type keeps the container default.
    pub fn check(self, d: &DataFormat) -> Result<()> {
        let ok = match self {
            FileFormat::Wave => d.endian == Endian::Little && (!d.float || d.bits == 32),
            FileFormat::Aiff => d.endian == Endian::Big && !d.float,
            FileFormat::Raw => !d.float || d.bits == 32,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::usage(format!(
                "{} cannot hold {}. Run speak --data-format='?'",
                self.name(),
                d.name()
            )))
        }
    }
}

impl DataFormat {
    /// `LEI16@22050`, `BEF32`, `@48000` (rate only, with `default` for the rest).
    pub fn parse(s: &str, default: DataFormat) -> Result<DataFormat> {
        let bad = || {
            Error::usage(format!(
                "--data-format {s}: expected e.g. LEI16, BEI24@22050, LEF32. Run speak --data-format='?'"
            ))
        };
        if UNSUPPORTED_DATA.contains(&s.to_ascii_lowercase().as_str()) {
            return Err(Error::usage(format!("--data-format {s} is not supported")));
        }
        let (kind, rate) = match s.split_once('@') {
            Some((k, r)) => (
                k,
                Some(
                    r.parse::<u32>()
                        .ok()
                        .filter(|r| (4000..=192_000).contains(r))
                        .ok_or_else(bad)?,
                ),
            ),
            None => (s, None),
        };
        if kind.is_empty() {
            return Ok(DataFormat { rate, ..default });
        }
        let upper = kind.to_ascii_uppercase();
        let (endian, rest) = match upper.get(..2) {
            Some("BE") => (Endian::Big, &upper[2..]),
            Some("LE") => (Endian::Little, &upper[2..]),
            _ => return Err(bad()),
        };
        let (float, bits) = match rest {
            "I16" => (false, 16),
            "I24" => (false, 24),
            "I32" => (false, 32),
            "F32" => (true, 32),
            _ => return Err(bad()),
        };
        Ok(DataFormat {
            endian,
            float,
            bits,
            rate,
        })
    }

    pub fn name(&self) -> String {
        format!(
            "{}{}{}",
            if self.endian == Endian::Big {
                "BE"
            } else {
                "LE"
            },
            if self.float { "F" } else { "I" },
            self.bits
        )
    }

    pub fn bytes(&self) -> usize {
        self.bits as usize / 8
    }

    /// Float → the format's bytes: clamp to [−1, 1], scale, round; no dither.
    pub fn encode(&self, sample: f32, out: &mut Vec<u8>) {
        let s = sample.clamp(-1.0, 1.0);
        let big = self.endian == Endian::Big;
        if self.float {
            out.extend_from_slice(&if big {
                s.to_be_bytes()
            } else {
                s.to_le_bytes()
            });
            return;
        }
        match self.bits {
            16 => {
                let v = (s * 32767.0).round() as i16;
                out.extend_from_slice(&if big {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
            24 => {
                // Three bytes of a 32-bit two's complement value.
                let v = (s as f64 * 8_388_607.0).round() as i32;
                let b = v.to_le_bytes();
                if big {
                    out.extend_from_slice(&[b[2], b[1], b[0]])
                } else {
                    out.extend_from_slice(&b[..3])
                }
            }
            _ => {
                let v = (s as f64 * 2_147_483_647.0).round() as i32;
                out.extend_from_slice(&if big {
                    v.to_be_bytes()
                } else {
                    v.to_le_bytes()
                });
            }
        }
    }
}

pub fn file_format_listing() -> String {
    FILE_FORMATS
        .iter()
        .map(|f| format!("{:<6} {}\n", f.0, f.1))
        .collect()
}

pub fn data_format_listing() -> String {
    let mut s = String::from("LPCM formats: [BE|LE][I|F][16|24|32][@rate]\n");
    for f in FILE_FORMATS {
        s.push_str(&format!("{}\n", f.1));
    }
    s.push_str("Defaults: WAVE LEI16, AIFF BEI16, raw LEI16; rate 16000\n");
    s
}

/// 80-bit IEEE 754 extended, big-endian (AIFF sample rate).
pub fn extended_80(rate: u32) -> [u8; 10] {
    let mut out = [0u8; 10];
    if rate == 0 {
        return out;
    }
    let msb = 31 - rate.leading_zeros();
    let exponent = (16383 + msb) as u16;
    let mantissa = (rate as u64) << (63 - msb);
    out[..2].copy_from_slice(&exponent.to_be_bytes());
    out[2..].copy_from_slice(&mantissa.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing() {
        let d = DataFormat::parse("LEI16@22050", FileFormat::Wave.default_data()).unwrap();
        assert_eq!(
            (d.endian, d.float, d.bits, d.rate),
            (Endian::Little, false, 16, Some(22050))
        );
        assert!(
            DataFormat::parse("LEF32", FileFormat::Wave.default_data())
                .unwrap()
                .float
        );
        assert_eq!(
            DataFormat::parse("@48000", FileFormat::Aiff.default_data())
                .unwrap()
                .name(),
            "BEI16"
        );
        for bad in ["XX16", "LEI8", "LEF64", "LEI16@x", "aac", "alac", ""] {
            assert!(
                DataFormat::parse(bad, FileFormat::Wave.default_data()).is_err() || bad.is_empty(),
                "{bad}"
            );
        }
        assert!(
            FileFormat::Wave
                .check(&DataFormat::parse("BEI16", FileFormat::Wave.default_data()).unwrap())
                .is_err()
        );
        assert!(
            FileFormat::Aiff
                .check(&DataFormat::parse("LEI16", FileFormat::Aiff.default_data()).unwrap())
                .is_err()
        );
        assert!(
            FileFormat::Aiff
                .check(&DataFormat::parse("BEI24", FileFormat::Aiff.default_data()).unwrap())
                .is_ok()
        );
        assert!(
            FileFormat::Raw
                .check(&DataFormat::parse("BEF32", FileFormat::Raw.default_data()).unwrap())
                .is_ok()
        );
        assert_eq!(
            FileFormat::from_path(std::path::Path::new("a.WAV")),
            FileFormat::Wave
        );
        assert_eq!(
            FileFormat::from_path(std::path::Path::new("a.aif")),
            FileFormat::Aiff
        );
        assert_eq!(
            FileFormat::from_path(std::path::Path::new("noext")),
            FileFormat::Aiff
        );
        assert_eq!(
            FileFormat::from_path(std::path::Path::new("x.pcm")),
            FileFormat::Raw
        );
    }

    #[test]
    fn extended_float() {
        // Known encodings of common rates.
        assert_eq!(
            extended_80(44100),
            [0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            extended_80(48000),
            [0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            extended_80(16000),
            [0x40, 0x0C, 0xFA, 0x00, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            extended_80(22050),
            [0x40, 0x0D, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            extended_80(24000),
            [0x40, 0x0D, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn encoding() {
        let enc = |name: &str, v: f32| {
            let mut out = Vec::new();
            DataFormat::parse(name, FileFormat::Raw.default_data())
                .unwrap()
                .encode(v, &mut out);
            out
        };
        assert_eq!(enc("LEI16", 1.0), [0xFF, 0x7F]);
        assert_eq!(enc("BEI16", -1.0), [0x80, 0x01]);
        assert_eq!(enc("LEI24", -1.0), [0x01, 0x00, 0x80]);
        assert_eq!(enc("BEI24", 1.0), [0x7F, 0xFF, 0xFF]);
        assert_eq!(enc("BEI32", 2.0), [0x7F, 0xFF, 0xFF, 0xFF]);
        assert_eq!(enc("LEF32", 0.5), 0.5f32.to_le_bytes());
    }
}
