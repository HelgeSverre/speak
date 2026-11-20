//! The compiled CMUdict: sorted `word\tphones\n` lines, read whole and binary-searched on the
//! raw bytes. Nothing is parsed at startup (spec §7.4).

use std::path::Path;

use crate::text::lexicon::{Pron, parse_pron};

pub const FILE_NAME: &str = "en-us.lex";

pub struct Compiled {
    bytes: Vec<u8>,
}

/// CMUdict phones → ours: `AH0 → AX`, `ER0 → AXR`, `IH0 → IX`; a trailing `0` is dropped.
pub fn map_phone(p: &str) -> String {
    match p {
        "AH0" => "AX".into(),
        "ER0" => "AXR".into(),
        "IH0" => "IX".into(),
        other => other.strip_suffix('0').unwrap_or(other).to_string(),
    }
}

/// Builds the compiled text from `cmudict.dict`: first variant only, comments removed.
/// Returns (file contents, entry count).
pub fn build(dict: &str) -> (String, usize) {
    let mut entries: Vec<(String, String)> = Vec::new();
    for line in dict.lines() {
        let line = line.split(" #").next().unwrap_or("");
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        if word.contains('(') || word.contains('\t') {
            continue;
        }
        let phones: Vec<String> = it.map(map_phone).collect();
        // Entries we cannot parse into our inventory are left out.
        if phones.is_empty() || parse_pron(&phones.join(" ")).is_none() {
            continue;
        }
        entries.push((word.to_lowercase(), phones.join(" ")));
    }
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    entries.dedup_by(|a, b| a.0 == b.0);
    let mut out = String::with_capacity(dict.len());
    for (w, p) in &entries {
        out.push_str(w);
        out.push('\t');
        out.push_str(p);
        out.push('\n');
    }
    (out, entries.len())
}

impl Compiled {
    /// `None` when the file is absent or unreadable: CMUdict is optional.
    pub fn open(path: &Path) -> Option<Compiled> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.is_empty() || bytes.last() != Some(&b'\n') {
            return None;
        }
        Some(Compiled { bytes })
    }

    pub fn from_text(text: String) -> Compiled {
        Compiled {
            bytes: text.into_bytes(),
        }
    }

    /// The line containing byte `at`: (start, end without the newline).
    fn line_at(&self, at: usize) -> (usize, usize) {
        let b = &self.bytes;
        let start = b[..at]
            .iter()
            .rposition(|c| *c == b'\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let end = b[at..]
            .iter()
            .position(|c| *c == b'\n')
            .map(|p| at + p)
            .unwrap_or(b.len());
        (start, end)
    }

    fn find(&self, word: &[u8]) -> Option<&[u8]> {
        let b = &self.bytes;
        let (mut lo, mut hi) = (0usize, b.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (start, end) = self.line_at(mid);
            let line = &b[start..end];
            let tab = line.iter().position(|c| *c == b'\t')?;
            match line[..tab].cmp(word) {
                std::cmp::Ordering::Equal => return Some(&line[tab + 1..]),
                std::cmp::Ordering::Less => lo = end + 1,
                std::cmp::Ordering::Greater => hi = start,
            }
        }
        None
    }

    pub fn contains(&self, lower: &str) -> bool {
        self.find(lower.as_bytes()).is_some()
    }

    pub fn lookup(&self, lower: &str) -> Option<Pron> {
        parse_pron(std::str::from_utf8(self.find(lower.as_bytes())?).ok()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phone::{Phone, Stress};

    const DICT: &str = "zebra Z IY1 B R AH0\napple AE1 P AH0 L\napple(2) AE1 P L\nbutter B AH1 T ER0 # comment\nit's IH1 T S\nlimit L IH1 M IH0 T\na AH0\n'bout B AW1 T\nbogus XX1 YY\n";

    #[test]
    fn build_and_lookup() {
        let (text, n) = build(DICT);
        assert_eq!(n, 7);
        assert!(text.starts_with("'bout\tB AW1 T\na\tAX\napple\tAE1 P AX L\n"));
        let c = Compiled::from_text(text);
        assert_eq!(
            c.lookup("butter").unwrap().last().unwrap(),
            &(Phone::AXR, Stress::None)
        );
        assert_eq!(c.lookup("limit").unwrap()[3], (Phone::IX, Stress::None));
        assert_eq!(c.lookup("zebra").unwrap().len(), 5);
        assert_eq!(c.lookup("a").unwrap(), [(Phone::AX, Stress::None)]);
        assert!(c.lookup("it's").is_some());
        for miss in ["", "aardvark", "zzz", "appl", "apples", "bogus", "b"] {
            assert!(c.lookup(miss).is_none(), "{miss}");
        }
    }

    #[test]
    fn absent_file_is_not_an_error() {
        assert!(Compiled::open(Path::new("/nonexistent/en-us.lex")).is_none());
    }
}
