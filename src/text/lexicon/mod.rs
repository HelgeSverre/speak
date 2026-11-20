//! Lexicon lookup (spec §7.4): user lexicon → built-in exceptions → CMUdict (optional).

pub mod builtin;
pub mod compiled;
pub mod homograph;
pub mod user;

use std::collections::HashMap;
use std::path::Path;

use crate::chunk::WordSource;
use crate::phone::{Phone, Stress, parse_phone_token};

pub type Pron = Vec<(Phone, Stress)>;

/// Parses `PH PH ...`; `None` when a token is not a phone.
pub fn parse_pron(text: &str) -> Option<Pron> {
    let pron: Option<Pron> = text
        .split_whitespace()
        .map(|t| parse_phone_token(t).ok().filter(|p| p.0 != Phone::SIL))
        .collect();
    pron.filter(|p| !p.is_empty())
}

#[derive(Default)]
pub struct Lexicons {
    user: HashMap<String, Pron>,
    builtin: HashMap<&'static str, Pron>,
    letters: HashMap<char, Pron>,
    /// CMUdict, when `<data>/lexicon/en-us.lex` exists.
    compiled: Option<compiled::Compiled>,
}

impl Lexicons {
    /// Built-in lexicon plus the user lexicon at `user_path` and the compiled CMUdict under
    /// `data_dir`, for whichever of those files exist.
    pub fn load(user_path: Option<&Path>, data_dir: Option<&Path>) -> Lexicons {
        let (builtin, letters) = builtin::parse();
        let user = user_path.map(user::load).unwrap_or_default();
        let compiled = data_dir
            .and_then(|d| compiled::Compiled::open(&d.join("lexicon").join(compiled::FILE_NAME)));
        Lexicons {
            user,
            builtin,
            letters,
            compiled,
        }
    }

    pub fn has_cmudict(&self) -> bool {
        self.compiled.is_some()
    }

    /// Lookup of a lowercase word.
    pub fn lookup(&self, lower: &str) -> Option<(Pron, WordSource)> {
        if let Some(p) = self.user.get(lower) {
            return Some((p.clone(), WordSource::User));
        }
        if let Some(p) = self.builtin.get(lower) {
            return Some((p.clone(), WordSource::Builtin));
        }
        self.compiled
            .as_ref()?
            .lookup(lower)
            .map(|p| (p, WordSource::Cmudict))
    }

    pub fn contains(&self, lower: &str) -> bool {
        self.user.contains_key(lower)
            || self.builtin.contains_key(lower)
            || self.compiled.as_ref().is_some_and(|c| c.contains(lower))
    }

    pub fn letter(&self, c: char) -> Option<&Pron> {
        self.letters.get(&c.to_ascii_lowercase())
    }

    pub fn builtin_len(&self) -> usize {
        self.builtin.len()
    }
}
