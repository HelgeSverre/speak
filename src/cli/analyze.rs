//! `speak-lab analyze` and `speak-lab spectrogram` (spec §3.6).

use std::path::Path;

use crate::dsp::{lpc, pitch};
use crate::error::{Error, Result};

/// Reads a WAV file as mono f32.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let bad = |e: hound::Error| Error::invalid(path.display().to_string(), e.to_string());
    let mut r = hound::WavReader::open(path).map_err(bad)?;
    let spec = r.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(bad)?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(bad)?
        }
    };
    let mono = samples
        .chunks(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

/// TSV tracks, one row per 5 ms.
pub fn analyze(pcm: &[f32], sample_rate: u32, f0: bool, formants: bool, lpc_out: bool) -> String {
    let sr = sample_rate as f32;
    let hop = (sr * 0.005) as usize;
    let win = (sr * 0.025) as usize;
    let order = (2 + sample_rate / 1000).clamp(8, 30) as usize;
    let pitch = if f0 {
        pitch::track(pcm, sample_rate)
    } else {
        Vec::new()
    };
    let mut out = String::from("t_ms");
    if f0 {
        out.push_str("\tf0\tvoicing");
    }
    if formants {
        out.push_str("\tf1\tf2\tf3\tf4");
    }
    if lpc_out {
        for k in 1..=order {
            out.push_str(&format!("\tk{k}"));
        }
        out.push_str("\tgain");
    }
    out.push('\n');
    let frames = pcm.len() / hop.max(1);
    for i in 0..frames {
        let centre = i * hop;
        let start = centre.saturating_sub(win / 2);
        let end = (start + win).min(pcm.len());
        let frame = &pcm[start..end];
        out.push_str(&format!("{}", i * 5));
        if f0 {
            let p = pitch.get(i).copied().unwrap_or(pitch::Pitch {
                f0: 0.0,
                voicing: 0.0,
            });
            out.push_str(&format!("\t{:.1}\t{:.2}", p.f0, p.voicing));
        }
        if formants || lpc_out {
            let quiet = crate::dsp::rms(frame) < 1e-4 || frame.len() < order * 2;
            let x = lpc::pre_emphasis(frame, 0.97);
            let a = if quiet {
                None
            } else {
                Some(lpc::analyze_frame(&x, order))
            };
            if formants {
                let peaks = a
                    .as_ref()
                    .map(|a| lpc::spectral_peaks(&a.a, sr))
                    .unwrap_or_default();
                for k in 0..4 {
                    out.push_str(&format!(
                        "\t{:.0}",
                        peaks.get(k).map(|p| p.0).unwrap_or(0.0)
                    ));
                }
            }
            if lpc_out {
                for k in 0..order {
                    out.push_str(&format!(
                        "\t{:.4}",
                        a.as_ref().and_then(|a| a.k.get(k)).copied().unwrap_or(0.0)
                    ));
                }
                let gain = a
                    .as_ref()
                    .map(|a| (a.error.max(0.0) / frame.len().max(1) as f32).sqrt())
                    .unwrap_or(0.0);
                out.push_str(&format!("\t{gain:.5}"));
            }
        }
        out.push('\n');
    }
    out
}

/// Log-magnitude spectrogram as a binary PGM (P5): time left to right, low frequencies at the bottom.
pub fn spectrogram_pgm(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = 512usize;
    let hop = ((sample_rate as f32 * 0.005) as usize).max(1);
    let stft = crate::dsp::stft::Stft::new(n, hop);
    let bins = stft.bins();
    let spectra = stft.forward(pcm);
    let frames = spectra.len();
    let mut db = vec![-120.0f32; frames * bins];
    for (f, spec) in spectra.iter().enumerate() {
        for (k, c) in spec.iter().enumerate() {
            db[f * bins + k] = 10.0 * c.norm_sqr().max(1e-12).log10();
        }
    }
    let top = db.iter().cloned().fold(f32::MIN, f32::max);
    let range = 70.0;
    let mut out = format!("P5\n{frames} {bins}\n255\n").into_bytes();
    for row in 0..bins {
        let k = bins - 1 - row;
        for f in 0..frames {
            let v = ((db[f * bins + k] - (top - range)) / range).clamp(0.0, 1.0);
            out.push((v * 255.0) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone() -> Vec<f32> {
        (0..8000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect()
    }

    #[test]
    fn analyze_has_one_row_per_5_ms_and_the_requested_columns() {
        let tsv = analyze(&tone(), 16000, true, true, true);
        let mut lines = tsv.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        assert_eq!(
            &header[..7],
            ["t_ms", "f0", "voicing", "f1", "f2", "f3", "f4"]
        );
        assert_eq!(header.len(), 7 + 18 + 1);
        assert_eq!(lines.clone().count(), 100);
        assert!(lines.all(|l| l.split('\t').count() == header.len()));
        assert_eq!(
            analyze(&tone(), 16000, false, false, false).lines().next(),
            Some("t_ms")
        );
    }

    #[test]
    fn spectrogram_is_a_valid_pgm_with_the_tone_in_the_right_row() {
        let pgm = spectrogram_pgm(&tone(), 16000);
        assert_eq!(&pgm[..2], b"P5");
        let header_end = pgm
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == b'\n')
            .nth(2)
            .unwrap()
            .0
            + 1;
        let dims = String::from_utf8_lossy(&pgm[3..header_end]).to_string();
        let mut it = dims.split_whitespace();
        let (w, h): (usize, usize) = (
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
        );
        assert_eq!(pgm.len(), header_end + w * h);
        // 1000 Hz is bin 32 of 257: row (256 − 32) from the top is the brightest.
        let column: Vec<u8> = (0..h).map(|r| pgm[header_end + r * w + w / 2]).collect();
        let brightest = column
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| **v)
            .unwrap()
            .0;
        assert_eq!(brightest, 256 - 32);
    }
}
