//! User lexicon: one `word PH PH ...` per line; overrides everything (spec §7.4 row 1).

use std::collections::HashMap;
use std::path::Path;

use crate::text::lexicon::{Pron, parse_pron};

pub fn parse(text: &str) -> HashMap<String, Pron> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some((word, rest)) = line.split_once(char::is_whitespace)
            && let Some(pron) = parse_pron(rest)
        {
            map.insert(word.to_lowercase(), pron);
        }
    }
    map
}

/// A missing or unreadable file is an empty lexicon.
pub fn load(path: &Path) -> HashMap<String, Pron> {
    std::fs::read_to_string(path)
        .map(|t| parse(&t))
        .unwrap_or_default()
}
