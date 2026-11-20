//! The `.sdv` voice file (spec §8.2.4). Little-endian, version in the magic.
//!
//! ```text
//! magic "SDV1" | u32 sample_rate | u16 lpc_order | u16 frame_us | u32 n_diphones | u32 flags
//! 256-byte UTF-8 description
//! index: n × { u8 a, u8 b, u16 n_frames, u16 boundary_frame, u32 frame_offset, u32 residual_offset, u32 n_epochs }
//! frames: f32 × (lpc_order + 3) per frame            (k[1..p], gain, f0, voicing)
//! residuals: per diphone i16 × (n_frames × hop), then u32 × n_epochs
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::diphone::{HOP, SR};
use crate::dsp::lpc::{LpcFrame, ORDER};
use crate::error::{Error, Result};
use crate::phone::Phone;

pub const MAGIC: &[u8; 4] = b"SDV1";
pub const FLAG_RESIDUAL: u32 = 1;
const DESCRIPTION_BYTES: usize = 256;
const INDEX_ENTRY: usize = 18;
const HEADER: usize = 20;

#[derive(Clone, Debug, PartialEq)]
pub struct Diphone {
    pub a: Phone,
    pub b: Phone,
    pub frames: Vec<LpcFrame>,
    /// Frame index of the A/B phone boundary.
    pub boundary: usize,
    /// LPC residual, `frames.len() × HOP` samples; empty when the voice has none.
    pub residual: Vec<i16>,
    /// Pitch marks: sample positions within `residual`.
    pub epochs: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct VoiceFile {
    pub description: String,
    pub diphones: Vec<Diphone>,
    index: HashMap<(u8, u8), usize>,
}

impl VoiceFile {
    pub fn new(description: String, diphones: Vec<Diphone>) -> VoiceFile {
        let index = diphones
            .iter()
            .enumerate()
            .map(|(i, d)| ((d.a as u8, d.b as u8), i))
            .collect();
        VoiceFile {
            description,
            diphones,
            index,
        }
    }

    pub fn get(&self, a: Phone, b: Phone) -> Option<&Diphone> {
        self.index
            .get(&(a as u8, b as u8))
            .map(|i| &self.diphones[*i])
    }

    pub fn index_of(&self, a: Phone, b: Phone) -> Option<usize> {
        self.index.get(&(a as u8, b as u8)).copied()
    }

    pub fn has_residual(&self) -> bool {
        self.diphones.iter().any(|d| !d.residual.is_empty())
    }

    pub fn frame_count(&self) -> usize {
        self.diphones.iter().map(|d| d.frames.len()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let residual = self.has_residual();
        let mut out =
            Vec::with_capacity(HEADER + DESCRIPTION_BYTES + self.frame_count() * (ORDER + 3) * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SR.to_le_bytes());
        out.extend_from_slice(&(ORDER as u16).to_le_bytes());
        out.extend_from_slice(&5000u16.to_le_bytes());
        out.extend_from_slice(&(self.diphones.len() as u32).to_le_bytes());
        out.extend_from_slice(&(if residual { FLAG_RESIDUAL } else { 0 }).to_le_bytes());
        let mut desc = [0u8; DESCRIPTION_BYTES];
        // Cut on a character boundary.
        let mut n = self.description.len().min(DESCRIPTION_BYTES);
        while !self.description.is_char_boundary(n) {
            n -= 1;
        }
        desc[..n].copy_from_slice(&self.description.as_bytes()[..n]);
        out.extend_from_slice(&desc);
        let (mut frame_offset, mut residual_offset) = (0u32, 0u32);
        for d in &self.diphones {
            out.push(d.a as u8);
            out.push(d.b as u8);
            out.extend_from_slice(&(d.frames.len() as u16).to_le_bytes());
            out.extend_from_slice(&(d.boundary as u16).to_le_bytes());
            out.extend_from_slice(&frame_offset.to_le_bytes());
            out.extend_from_slice(&residual_offset.to_le_bytes());
            out.extend_from_slice(&(d.epochs.len() as u32).to_le_bytes());
            frame_offset += d.frames.len() as u32;
            residual_offset += (d.residual.len() * 2 + d.epochs.len() * 4) as u32;
        }
        for d in &self.diphones {
            for f in &d.frames {
                for k in &f.k {
                    out.extend_from_slice(&k.to_le_bytes());
                }
                out.extend_from_slice(&f.gain.to_le_bytes());
                out.extend_from_slice(&f.f0.to_le_bytes());
                out.extend_from_slice(&f.voicing.to_le_bytes());
            }
        }
        if residual {
            for d in &self.diphones {
                for s in &d.residual {
                    out.extend_from_slice(&s.to_le_bytes());
                }
                for e in &d.epochs {
                    out.extend_from_slice(&e.to_le_bytes());
                }
            }
        }
        out
    }

    pub fn from_bytes(b: &[u8], file: &str) -> Result<VoiceFile> {
        let bad = |why: &str| Error::invalid(file, why);
        let u16_at = |at: usize| b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]]));
        let u32_at = |at: usize| {
            b.get(at..at + 4)
                .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        if b.len() < HEADER + DESCRIPTION_BYTES {
            return Err(bad("truncated header"));
        }
        if &b[..4] != MAGIC {
            return Err(bad("not a speak diphone voice (bad magic)"));
        }
        let (rate, order, frame_us) = (
            u32_at(4).unwrap_or(0),
            u16_at(8).unwrap_or(0),
            u16_at(10).unwrap_or(0),
        );
        if rate != SR {
            return Err(bad(&format!("sample rate {rate}, expected {SR}")));
        }
        if order as usize != ORDER || frame_us != 5000 {
            return Err(bad(&format!(
                "LPC order {order} / frame {frame_us} µs, expected {ORDER} / 5000"
            )));
        }
        let n = u32_at(12).unwrap_or(0) as usize;
        let residual = u32_at(16).unwrap_or(0) & FLAG_RESIDUAL != 0;
        let desc = &b[HEADER..HEADER + DESCRIPTION_BYTES];
        let end = desc
            .iter()
            .position(|c| *c == 0)
            .unwrap_or(DESCRIPTION_BYTES);
        let description = String::from_utf8_lossy(&desc[..end]).to_string();
        let index_at = HEADER + DESCRIPTION_BYTES;
        let frames_at = index_at
            + n.checked_mul(INDEX_ENTRY)
                .ok_or_else(|| bad("bad diphone count"))?;
        if frames_at > b.len() {
            return Err(bad("truncated index"));
        }
        let frame_bytes = (ORDER + 3) * 4;
        let mut entries = Vec::with_capacity(n);
        let mut total_frames = 0usize;
        for i in 0..n {
            let e = index_at + i * INDEX_ENTRY;
            let (a, bb) = (
                Phone::from_index(b[e] as usize),
                Phone::from_index(b[e + 1] as usize),
            );
            let (Some(a), Some(bb)) = (a, bb) else {
                return Err(bad("unknown phone in the index"));
            };
            let n_frames = u16_at(e + 2).unwrap_or(0) as usize;
            let boundary = u16_at(e + 4).unwrap_or(0) as usize;
            let frame_offset = u32_at(e + 6).unwrap_or(0) as usize;
            let residual_offset = u32_at(e + 10).unwrap_or(0) as usize;
            let n_epochs = u32_at(e + 14).unwrap_or(0) as usize;
            if boundary > n_frames {
                return Err(bad("boundary beyond the diphone"));
            }
            total_frames = total_frames.max(frame_offset + n_frames);
            entries.push((
                a,
                bb,
                n_frames,
                boundary,
                frame_offset,
                residual_offset,
                n_epochs,
            ));
        }
        let residual_at = frames_at + total_frames * frame_bytes;
        if residual_at > b.len() {
            return Err(bad("truncated frames"));
        }
        let mut diphones = Vec::with_capacity(n);
        for (a, bb, n_frames, boundary, frame_offset, residual_offset, n_epochs) in entries {
            let mut frames = Vec::with_capacity(n_frames);
            for f in 0..n_frames {
                let at = frames_at + (frame_offset + f) * frame_bytes;
                let v = |j: usize| {
                    f32::from_le_bytes([
                        b[at + j * 4],
                        b[at + j * 4 + 1],
                        b[at + j * 4 + 2],
                        b[at + j * 4 + 3],
                    ])
                };
                let mut frame = LpcFrame::SILENT;
                for (j, k) in frame.k.iter_mut().enumerate() {
                    *k = v(j);
                }
                frame.gain = v(ORDER);
                frame.f0 = v(ORDER + 1);
                frame.voicing = v(ORDER + 2);
                if !frame.k.iter().all(|k| k.is_finite()) || !frame.gain.is_finite() {
                    return Err(bad("non-finite frame data"));
                }
                frames.push(frame);
            }
            let (mut res, mut epochs) = (Vec::new(), Vec::new());
            if residual {
                let samples = n_frames * HOP;
                let at = residual_at + residual_offset;
                let need = samples * 2 + n_epochs * 4;
                let data = b
                    .get(at..at + need)
                    .ok_or_else(|| bad("truncated residual"))?;
                res = data[..samples * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect();
                epochs = data[samples * 2..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
            }
            diphones.push(Diphone {
                a,
                b: bb,
                frames,
                boundary,
                residual: res,
                epochs,
            });
        }
        Ok(VoiceFile::new(description, diphones))
    }

    pub fn read(path: &Path) -> Result<VoiceFile> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::invalid(path.display().to_string(), e.to_string()))?;
        VoiceFile::from_bytes(&bytes, &path.display().to_string())
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.to_bytes()).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        })
    }
}

/// Reads only the header: (description, diphone count, has residual).
pub fn read_header(path: &Path) -> Result<(String, usize, bool)> {
    use std::io::Read;
    let mut buf = vec![0u8; HEADER + DESCRIPTION_BYTES];
    let mut f = std::fs::File::open(path)
        .map_err(|e| Error::invalid(path.display().to_string(), e.to_string()))?;
    f.read_exact(&mut buf)
        .map_err(|_| Error::invalid(path.display().to_string(), "truncated header"))?;
    if &buf[..4] != MAGIC {
        return Err(Error::invalid(
            path.display().to_string(),
            "not a speak diphone voice (bad magic)",
        ));
    }
    let n = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let flags = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let desc = &buf[HEADER..];
    let end = desc
        .iter()
        .position(|c| *c == 0)
        .unwrap_or(DESCRIPTION_BYTES);
    Ok((
        String::from_utf8_lossy(&desc[..end]).to_string(),
        n,
        flags & FLAG_RESIDUAL != 0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(residual: bool) -> VoiceFile {
        let frame = |x: f32| {
            let mut f = LpcFrame::SILENT;
            f.k[0] = x;
            f.k[17] = -x / 2.0;
            f.gain = x.abs();
            f.f0 = 110.0;
            f.voicing = 0.9;
            f
        };
        let d = |a, b, n: usize| Diphone {
            a,
            b,
            frames: (0..n).map(|i| frame(0.1 * i as f32 - 0.3)).collect(),
            boundary: n / 2,
            residual: if residual {
                (0..n * HOP).map(|i| (i % 300) as i16 - 150).collect()
            } else {
                Vec::new()
            },
            epochs: if residual {
                vec![10, 155, 300]
            } else {
                Vec::new()
            },
        };
        VoiceFile::new(
            "boot; source klatt-male; test".into(),
            vec![
                d(Phone::AA, Phone::B, 6),
                d(Phone::SIL, Phone::HH, 3),
                d(Phone::IY, Phone::SIL, 9),
            ],
        )
    }

    #[test]
    fn round_trip_with_and_without_residual() {
        for residual in [false, true] {
            let v = sample(residual);
            let back = VoiceFile::from_bytes(&v.to_bytes(), "t.sdv").unwrap();
            assert_eq!(back.description, v.description);
            assert_eq!(back.diphones, v.diphones);
            assert_eq!(back.has_residual(), residual);
            assert_eq!(back.get(Phone::SIL, Phone::HH).unwrap().frames.len(), 3);
            assert!(back.get(Phone::HH, Phone::SIL).is_none());
        }
    }

    #[test]
    fn rejects_bad_files() {
        let bytes = sample(true).to_bytes();
        let code = |b: &[u8]| VoiceFile::from_bytes(b, "x.sdv").unwrap_err().exit_code();
        assert_eq!(code(&bytes[..bytes.len() - 5]), 3);
        assert_eq!(code(&bytes[..100]), 3);
        assert_eq!(code(&bytes[..600]), 3);
        let mut wrong = bytes.clone();
        wrong[..4].copy_from_slice(b"SDV2");
        assert_eq!(code(&wrong), 3);
        let mut rate = bytes.clone();
        rate[4..8].copy_from_slice(&22050u32.to_le_bytes());
        assert!(
            VoiceFile::from_bytes(&rate, "x.sdv")
                .unwrap_err()
                .to_string()
                .contains("sample rate")
        );
        let mut order = bytes;
        order[8..10].copy_from_slice(&10u16.to_le_bytes());
        assert_eq!(code(&order), 3);
    }
}
