//! `config.toml` (spec §13). Precedence: command line > environment > config file > default.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub backend: Option<String>,
    pub rate: Option<f32>,
    pub volume: Option<f32>,
    pub lts: Option<String>,
    /// Drive a connected SidePulse LED strip (default true).
    pub lights: Option<bool>,
    #[serde(default)]
    pub klatt: BackendConfig,
    #[serde(default)]
    pub diphone: BackendConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub voices: BTreeMap<String, Alias>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub voice: Option<String>,
    pub excitation: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    pub device: Option<String>,
}

/// `[voices.name]`: usable as `-v name`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alias {
    pub backend: Option<String>,
    pub voice: Option<String>,
    pub rate: Option<f32>,
    #[serde(default)]
    pub params: BTreeMap<String, toml::Value>,
}

impl Alias {
    /// Params as the strings `-P` would carry.
    pub fn params(&self) -> Vec<(String, String)> {
        self.params
            .iter()
            .map(|(k, v)| {
                let text = match v {
                    toml::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), text)
            })
            .collect()
    }
}

pub fn config_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    let base = if cfg!(target_os = "macos") {
        home.join(".config")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"))
    };
    base.join("speak")
}

pub fn parse(text: &str, path: &Path) -> Result<Config> {
    toml::from_str::<Config>(text).map_err(|e| {
        // toml's message names the key and the line.
        Error::Config(format!("{}: {}", path.display(), e.to_string().trim_end()))
    })
}

/// `explicit` (from `--config` or `SPEAK_CONFIG`) must exist; the default path may be absent.
pub fn load(explicit: Option<&Path>) -> Result<Config> {
    let env = std::env::var_os("SPEAK_CONFIG")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let (path, required) = match explicit.map(Path::to_path_buf).or(env) {
        Some(p) => (p, true),
        None => (config_dir().join("config.toml"), false),
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => parse(&text, &path),
        Err(e) if !required && e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(e) => Err(Error::Config(format!("{}: {e}", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_example_parses() {
        let c = parse(
            r#"
backend = "klatt"
rate = 175
volume = 1.0
lts = "cart"

[klatt]
voice = "klatt-male"

[diphone]
voice = "boot"
excitation = "glottal"

[audio]
device = ""

[voices.narrator]
backend = "klatt"
voice = "klatt-female"
rate = 165
params = { tl = 12, oq = 0.65 }
"#,
            Path::new("config.toml"),
        )
        .unwrap();
        assert_eq!(c.rate, Some(175.0));
        let n = &c.voices["narrator"];
        assert_eq!(n.voice.as_deref(), Some("klatt-female"));
        assert_eq!(
            n.params(),
            [
                ("oq".to_string(), "0.65".to_string()),
                ("tl".to_string(), "12".to_string())
            ]
        );
    }

    #[test]
    fn unknown_key_names_key_and_line() {
        let e = parse("rate = 100\nnonsense = 1\n", Path::new("bad.toml")).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("nonsense") && msg.contains("line 2"), "{msg}");
        assert_eq!(e.exit_code(), 2);
        assert!(parse("[voices.x]\nbogus = 1\n", Path::new("b.toml")).is_err());
    }
}
