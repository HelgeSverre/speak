//! Automatic labelling of recorded prompts (spec §8.2.6 step 3): MFCCs of the recording are
//! aligned to MFCCs of the guide by dynamic time warping, and the guide's known phone
//! boundaries are mapped through the warp path.

use std::path::Path;

use crate::diphone::build::residual;
use crate::diphone::format::{Diphone, VoiceFile};
use crate::diphone::prompts::{self, Prompt};
use crate::diphone::{FRAME_MS, HOP, SR};
use crate::dsp::mel::{MelBank, deltas, log_mel, mfcc};
use crate::dsp::stft::Stft;
use crate::dsp::{dtw, lpc, pitch, psola};
use crate::error::{Error, Result};
use crate::klatt::KlattSynth;
use crate::traits::Synthesizer;

/// Sakoe-Chiba band: ± 300 ms.
const BAND_FRAMES: usize = 60;
/// A prompt whose warp cost is this many times the median is rejected.
const REJECT_FACTOR: f32 = 1.8;
const N_MFCC: usize = 13;

/// 12 MFCCs (C0 dropped) with cepstral mean normalization, plus deltas; one vector per 5 ms.
pub fn features(pcm: &[f32]) -> Vec<Vec<f32>> {
    let stft = Stft::new(400, HOP);
    let bank = MelBank::new(40, 400, SR, 0.0, 8000.0, false);
    let mut frames: Vec<Vec<f32>> = stft
        .forward(pcm)
        .iter()
        .map(|spec| {
            let power: Vec<f32> = spec.iter().map(|c| c.norm_sqr()).collect();
            // C0 carries overall level, which differs between guide and speaker: drop it.
            mfcc(&log_mel(&bank.apply(&power)), N_MFCC)[1..].to_vec()
        })
        .collect();
    if frames.is_empty() {
        return frames;
    }
    // Cepstral mean normalization removes the channel and the speaker's average spectrum.
    let dims = frames[0].len();
    let mut mean = vec![0.0f32; dims];
    for f in &frames {
        for (m, v) in mean.iter_mut().zip(f) {
            *m += v / frames.len() as f32;
        }
    }
    for f in &mut frames {
        for (v, m) in f.iter_mut().zip(&mean) {
            *v -= m;
        }
    }
    let d = deltas(&frames);
    frames
        .iter()
        .zip(d)
        .map(|(a, b)| a.iter().chain(b.iter()).copied().collect())
        .collect()
}

/// Landmarks of a prompt in a recording, in ms, with the warp cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Labels {
    pub mid_a: f32,
    pub boundary: f32,
    pub mid_b: f32,
    pub cost: f32,
}

/// The guide rendering of a prompt (deterministic, so it need not be read from disk).
pub fn render_guide(p: &Prompt) -> Result<Vec<f32>> {
    let mut synth = KlattSynth::new(crate::diphone::build::source_voice());
    let mut pcm = Vec::new();
    synth.synthesize(&prompts::guide(p), &mut pcm)?;
    Ok(pcm)
}

/// Aligns a recording to the guide and maps the guide's landmarks onto it.
pub fn label(p: &Prompt, guide: &[f32], recording: &[f32]) -> Option<Labels> {
    let (g, r) = (features(guide), features(recording));
    let warp = dtw::dtw(&g, &r, BAND_FRAMES)?;
    let (mid_a, boundary, mid_b) = p.landmarks(&p.times(&|_| 1.0));
    let map = |ms: f32| -> f32 {
        // Interpolate between the two guide frames around the landmark.
        let pos = (ms / FRAME_MS).clamp(0.0, (g.len() - 1) as f32);
        let (i, t) = (pos as usize, pos.fract());
        let (lo, hi) = (warp.map(i), warp.map((i + 1).min(g.len() - 1)));
        (lo + (hi - lo) * t) * FRAME_MS
    };
    Some(Labels {
        mid_a: map(mid_a),
        boundary: map(boundary),
        mid_b: map(mid_b),
        cost: warp.cost,
    })
}

/// Analyses a recording and cuts the diphone at the given landmarks.
pub fn cut_recording(
    p: &Prompt,
    recording: &[f32],
    labels: &Labels,
) -> Option<(Diphone, Vec<f32>)> {
    let frames = lpc::analyze(recording, SR);
    let index = |ms: f32| ((ms / FRAME_MS).round() as usize).min(frames.len());
    let (start, end) = (index(labels.mid_a), index(labels.mid_b));
    if end <= start + 1 {
        return None;
    }
    let boundary = index(labels.boundary).clamp(start, end) - start;
    let res = residual(recording, &frames);
    let (s0, s1) = (start * HOP, (end * HOP).min(res.len()));
    let stored: Vec<i16> = (s0..start * HOP + (end - start) * HOP)
        .map(|n| {
            (res.get(n).copied().unwrap_or(0.0) * crate::diphone::build::RESIDUAL_SCALE)
                .round()
                .clamp(-32767.0, 32767.0) as i16
        })
        .collect();
    // Pitch marks on the low-passed residual's energy peaks, guided by F0.
    let track = pitch::track(recording, SR);
    let (marks, voiced) = psola::epochs(recording, SR, &track);
    let epochs = marks
        .iter()
        .zip(&voiced)
        .filter(|(m, v)| **v && **m >= s0 && **m < s1)
        .map(|(m, _)| (*m - s0) as u32)
        .collect();
    let f0s: Vec<f32> = frames[start..end]
        .iter()
        .map(|f| f.f0)
        .filter(|f| *f > 0.0)
        .collect();
    Some((
        Diphone {
            a: p.a,
            b: p.b,
            frames: frames[start..end].to_vec(),
            boundary,
            residual: stored,
            epochs,
        },
        f0s,
    ))
}

#[derive(Debug, Default)]
pub struct Report {
    pub prompts: usize,
    pub recorded: usize,
    pub built: usize,
    pub rejected: Vec<String>,
    /// Boundary errors against `truth.tsv`, when the directory has one (ms).
    pub boundary_errors: Vec<f32>,
    pub median_f0: f32,
}

impl Report {
    pub fn percentile(&self, p: f32) -> Option<f32> {
        let mut v = self.boundary_errors.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f32::total_cmp);
        Some(v[((v.len() - 1) as f32 * p).round() as usize])
    }
}

fn read_recording(path: &Path) -> Result<Vec<f32>> {
    crate::cli::resynth::load(path)
}

/// Builds a voice from `dir`: `prompts.tsv` and `rec/<id>.wav`. A partial set is fine.
pub fn build_from_dir(
    dir: &Path,
    name: &str,
    mut progress: impl FnMut(usize, usize),
) -> Result<(VoiceFile, Report)> {
    let tsv_path = dir.join("prompts.tsv");
    let tsv = std::fs::read_to_string(&tsv_path)
        .map_err(|e| Error::invalid(tsv_path.display().to_string(), e.to_string()))?;
    let prompts = prompts::parse_tsv(&tsv);
    if prompts.is_empty() {
        return Err(Error::invalid(tsv_path.display().to_string(), "no prompts"));
    }
    // Known boundaries, when the recordings were simulated.
    let truth: std::collections::HashMap<String, (f32, f32, f32)> =
        std::fs::read_to_string(dir.join("truth.tsv"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let mut it = l.split('\t');
                let id = it.next()?.to_string();
                let v: Vec<f32> = it.filter_map(|x| x.parse().ok()).collect();
                (v.len() == 3).then(|| (id, (v[0], v[1], v[2])))
            })
            .collect();
    let mut report = Report {
        prompts: prompts.len(),
        ..Report::default()
    };
    let mut labelled: Vec<(usize, Vec<f32>, Labels)> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        progress(i + 1, prompts.len());
        let path = dir.join("rec").join(format!("{}.wav", p.id));
        if !path.is_file() {
            continue;
        }
        report.recorded += 1;
        let recording = read_recording(&path)?;
        let guide = render_guide(p)?;
        match label(p, &guide, &recording) {
            Some(l) => labelled.push((i, recording, l)),
            None => report.rejected.push(p.id.clone()),
        }
    }
    if labelled.is_empty() {
        return Err(Error::invalid(
            dir.display().to_string(),
            "no recordings found under rec/",
        ));
    }
    let mut costs: Vec<f32> = labelled.iter().map(|l| l.2.cost).collect();
    costs.sort_by(f32::total_cmp);
    let limit = costs[costs.len() / 2] * REJECT_FACTOR;
    let mut diphones = Vec::new();
    let mut f0s: Vec<f32> = Vec::new();
    for (i, recording, labels) in &labelled {
        let p = &prompts[*i];
        if labels.cost > limit {
            report.rejected.push(p.id.clone());
            continue;
        }
        let Some((d, f0)) = cut_recording(p, recording, labels) else {
            report.rejected.push(p.id.clone());
            continue;
        };
        if let Some((_, boundary, _)) = truth.get(&p.id) {
            report
                .boundary_errors
                .push((labels.boundary - boundary).abs());
        }
        f0s.extend(f0);
        diphones.push(d);
    }
    report.built = diphones.len();
    f0s.sort_by(f32::total_cmp);
    report.median_f0 = f0s.get(f0s.len() / 2).copied().unwrap_or(110.0);
    let description = format!(
        "name={name}; LPC diphones from recordings; source=recorded; f0={:.0}; diphones={}",
        report.median_f0, report.built
    );
    Ok((VoiceFile::new(description, diphones), report))
}

/// Test helper: "recordings" rendered by another Klatt voice with perturbed timing, plus the
/// true landmarks (spec §8.2.6). Returns the number of files written.
pub fn simulate(
    dir: &Path,
    voice: &str,
    rate: f32,
    jitter: f32,
    seed: u32,
    limit: Option<usize>,
) -> Result<usize> {
    use crate::audio::FileSink;
    use crate::traits::AudioSink;
    let tsv_path = dir.join("prompts.tsv");
    let tsv = std::fs::read_to_string(&tsv_path)
        .map_err(|e| Error::invalid(tsv_path.display().to_string(), e.to_string()))?;
    let prompts = prompts::parse_tsv(&tsv);
    let speaker = crate::klatt::Voice::find(voice)
        .ok_or_else(|| Error::usage(format!("unknown voice '{voice}'")))?;
    std::fs::create_dir_all(dir.join("rec"))?;
    let mut truth = String::new();
    let n = limit.unwrap_or(prompts.len()).min(prompts.len());
    for (i, p) in prompts.iter().take(n).enumerate() {
        // Per-phone duration factors: overall rate, plus seeded jitter.
        let mut noise =
            crate::dsp::Noise::new(seed.wrapping_mul(2_654_435_761).wrapping_add(i as u32 + 1));
        let factors: Vec<f32> = p
            .phones
            .iter()
            .map(|_| (1.0 + jitter * noise.sample()) / rate.max(0.1))
            .collect();
        let scale = |k: usize| factors[k];
        let mut synth = KlattSynth::new(speaker.clone());
        let mut sink = FileSink::create(&dir.join("rec").join(format!("{}.wav", p.id)))?;
        sink.begin(SR)?;
        synth.synthesize(&p.chunk(speaker.base_hz, &scale), &mut sink)?;
        sink.finish()?;
        let (a, b, c) = p.landmarks(&p.times(&scale));
        truth.push_str(&format!("{}\t{a:.1}\t{b:.1}\t{c:.1}\n", p.id));
    }
    std::fs::write(dir.join("truth.tsv"), truth)?;
    Ok(n)
}
