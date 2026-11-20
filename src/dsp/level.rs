//! Level measurements.

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

/// RMS in dB relative to full scale (1.0). Silence is −120.
pub fn rms_dbfs(x: &[f32]) -> f32 {
    let r = rms(x);
    if r <= 1e-6 { -120.0 } else { 20.0 * r.log10() }
}

pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}
