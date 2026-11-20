//! `--timing` report (spec §11.3).

use crate::chunk::WordSource;

#[derive(Clone, Debug, Default)]
pub struct TimingReport {
    pub backend: String,
    pub voice: String,
    pub sample_rate: u32,
    pub chunks: usize,
    pub startup_ms: f64,
    pub frontend_init_ms: f64,
    pub voice_open_ms: f64,
    pub frontend_ms: f64,
    pub ttfa_ms: f64,
    pub synth_total_ms: f64,
    pub audio_s: f64,
    pub underruns: u64,
    /// Indexed like `WordSource::ALL`.
    pub lexicon_hits: [usize; 6],
}

impl TimingReport {
    pub fn rtf(&self) -> f64 {
        if self.audio_s > 0.0 {
            self.synth_total_ms / 1000.0 / self.audio_s
        } else {
            0.0
        }
    }

    fn hits(&self) -> String {
        WordSource::ALL
            .iter()
            .zip(self.lexicon_hits)
            .map(|(s, n)| format!("{} {}", s.name(), n))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let mut line = |k: &str, v: String| s.push_str(&format!("{k:<17} {v}\n"));
        line("backend", self.backend.clone());
        line("voice", self.voice.clone());
        line("sample_rate", self.sample_rate.to_string());
        line("chunks", self.chunks.to_string());
        line("startup_ms", format!("{:.1}", self.startup_ms));
        line("frontend_init_ms", format!("{:.1}", self.frontend_init_ms));
        line("voice_open_ms", format!("{:.1}", self.voice_open_ms));
        line("frontend_ms", format!("{:.1}", self.frontend_ms));
        line("ttfa_ms", format!("{:.1}", self.ttfa_ms));
        line("synth_total_ms", format!("{:.1}", self.synth_total_ms));
        line("audio_s", format!("{:.2}", self.audio_s));
        line("rtf", format!("{:.4}", self.rtf()));
        line("underruns", self.underruns.to_string());
        line("lexicon_hits", self.hits());
        s
    }

    pub fn to_json(&self) -> String {
        let hits = WordSource::ALL
            .iter()
            .zip(self.lexicon_hits)
            .map(|(s, n)| format!("\"{}\":{}", s.name(), n))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"backend\":\"{}\",\"voice\":\"{}\",\"sample_rate\":{},\"chunks\":{},\"startup_ms\":{:.3},\"frontend_init_ms\":{:.3},\"voice_open_ms\":{:.3},\"frontend_ms\":{:.3},\"ttfa_ms\":{:.3},\"synth_total_ms\":{:.3},\"audio_s\":{:.4},\"rtf\":{:.5},\"underruns\":{},\"lexicon_hits\":{{{}}}}}",
            self.backend,
            self.voice,
            self.sample_rate,
            self.chunks,
            self.startup_ms,
            self.frontend_init_ms,
            self.voice_open_ms,
            self.frontend_ms,
            self.ttfa_ms,
            self.synth_total_ms,
            self.audio_s,
            self.rtf(),
            self.underruns,
            hits
        )
    }
}
