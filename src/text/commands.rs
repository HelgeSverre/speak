//! Embedded commands `[[cmd arg]]` (spec §7.2).

use crate::traits::EmbeddedCommand;

#[derive(Clone, Debug, PartialEq)]
pub enum Parsed {
    Command(EmbeddedCommand),
    /// `[[inpt PHON]]` / `[[inpt TEXT]]`.
    PhonemeMode(bool),
    /// Any other content: inline ARPAbet.
    Phonemes(String),
}

pub fn parse(content: &str) -> Parsed {
    let mut it = content.split_whitespace();
    let (cmd, arg) = (it.next().unwrap_or(""), it.next());
    let num = arg
        .and_then(|a| a.parse::<f32>().ok())
        .filter(|v| v.is_finite());
    let single = it.next().is_none();
    match (cmd.to_ascii_lowercase().as_str(), num, arg) {
        ("slnc", Some(n), _) if single => {
            Parsed::Command(EmbeddedCommand::Silence(n.clamp(0.0, 60_000.0)))
        }
        ("rate", Some(n), _) if single => {
            Parsed::Command(EmbeddedCommand::Rate(n.clamp(80.0, 500.0)))
        }
        ("volm", Some(n), _) if single => {
            Parsed::Command(EmbeddedCommand::Volume(n.clamp(0.0, 1.0)))
        }
        ("pbas", Some(n), _) if single => {
            Parsed::Command(EmbeddedCommand::PitchBase(n.clamp(40.0, 500.0)))
        }
        ("inpt", _, Some(mode)) if single && mode.eq_ignore_ascii_case("PHON") => {
            Parsed::PhonemeMode(true)
        }
        ("inpt", _, Some(mode)) if single && mode.eq_ignore_ascii_case("TEXT") => {
            Parsed::PhonemeMode(false)
        }
        _ => Parsed::Phonemes(content.trim().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands() {
        assert_eq!(
            parse("slnc 500"),
            Parsed::Command(EmbeddedCommand::Silence(500.0))
        );
        assert_eq!(
            parse("rate 300"),
            Parsed::Command(EmbeddedCommand::Rate(300.0))
        );
        assert_eq!(
            parse("volm 0.5"),
            Parsed::Command(EmbeddedCommand::Volume(0.5))
        );
        assert_eq!(
            parse("pbas 140"),
            Parsed::Command(EmbeddedCommand::PitchBase(140.0))
        );
        assert_eq!(parse("inpt PHON"), Parsed::PhonemeMode(true));
        assert_eq!(parse("inpt TEXT"), Parsed::PhonemeMode(false));
        assert_eq!(
            parse(" HH AX L OW1 "),
            Parsed::Phonemes("HH AX L OW1".into())
        );
    }
}
