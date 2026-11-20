//! `EnglishNormalizer`: raw text → words, punctuation and commands (spec §7.1).
//! All number and symbol expansion happens here, so every G2P sees only words.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::text::abbrev::{MONTHS, abbreviation_in_context, is_letter_expansion, month};
use crate::text::commands::{self, Parsed};
use crate::text::numbers;
use crate::text::units::{is_unit_word, needs_glue, unit};
use crate::traits::{Normalizer, Token};

type Span = (usize, usize);

pub struct EnglishNormalizer {
    /// True when the lexicon has this (lowercase) word: decides acronyms.
    known: Box<dyn Fn(&str) -> bool + Send + Sync>,
    dropped: AtomicUsize,
}

/// Step 1: a hand-written subset of NFKC plus quote/dash folding. `None` drops the character.
fn fold(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' | '`' | '\u{00B4}' => "'",
        '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' | '\u{00AB}' | '\u{00BB}' => "\"",
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2212}' => "-",
        '\u{2015}' => "—",
        '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\t' => " ",
        '\u{2028}' => "\n",
        '\u{2029}' => "\n\n",
        '\u{200B}'..='\u{200D}' | '\u{FEFF}' | '\u{00AD}' | '\r' => return None,
        '\u{FB00}' => "ff",
        '\u{FB01}' => "fi",
        '\u{FB02}' => "fl",
        '\u{FB03}' => "ffi",
        '\u{FB04}' => "ffl",
        '½' => "1/2",
        '¼' => "1/4",
        '¾' => "3/4",
        '²' => "2",
        '³' => "3",
        '¹' => "1",
        '×' => "x",
        '÷' => "/",
        '™' | '®' | '©' => return None,
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' => "a",
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' => "A",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ę' => "e",
        'È' | 'É' | 'Ê' | 'Ë' | 'Ē' => "E",
        'ì' | 'í' | 'î' | 'ï' | 'ī' => "i",
        'Ì' | 'Í' | 'Î' | 'Ï' => "I",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ō' | 'ø' => "o",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => "O",
        'ù' | 'ú' | 'û' | 'ü' | 'ū' => "u",
        'Ù' | 'Ú' | 'Û' | 'Ü' => "U",
        'ç' | 'ć' | 'č' => "c",
        'Ç' | 'Č' => "C",
        'ñ' | 'ń' => "n",
        'Ñ' => "N",
        'ß' => "ss",
        'æ' => "ae",
        'Æ' => "AE",
        'œ' => "oe",
        'š' | 'ś' => "s",
        'Š' => "S",
        'ž' | 'ź' | 'ż' => "z",
        'Ž' => "Z",
        'ý' | 'ÿ' => "y",
        'ł' => "l",
        c if c.is_control() && c != '\n' => return None,
        _ => return Some(""),
    })
}

/// Normalized characters with the byte offset of their source character.
fn fold_text(text: &str) -> Vec<(char, usize)> {
    let mut out = Vec::with_capacity(text.len());
    for (at, c) in text.char_indices() {
        // Fullwidth ASCII forms.
        if ('\u{FF01}'..='\u{FF5E}').contains(&c) {
            if let Some(a) = char::from_u32(c as u32 - 0xFF01 + 0x21) {
                out.push((a, at));
            }
            continue;
        }
        match fold(c) {
            None => {}
            Some("") => out.push((c, at)),
            Some(rep) => out.extend(rep.chars().map(|r| (r, at))),
        }
    }
    out
}

/// A whitespace-delimited token, or something already final.
enum Pre {
    Raw { text: String, span: Span },
    Tok(Token),
}

#[derive(Clone, Debug, PartialEq)]
struct Num {
    neg: bool,
    int: String,
    frac: Option<String>,
}

impl Num {
    fn words(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.neg {
            out.push("minus".to_string());
        }
        match &self.frac {
            Some(f) => out.extend(numbers::decimal(&self.int, f)),
            None => out.extend(numbers::cardinal_str(&self.int)),
        }
        out
    }

    fn is_one(&self) -> bool {
        !self.neg && self.frac.is_none() && self.int.trim_start_matches('0') == "1"
    }

    fn value(&self) -> Option<u64> {
        if self.frac.is_some() || self.neg {
            None
        } else {
            self.int.parse().ok()
        }
    }
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `1,234.5`, `-7`, `.5`, `42`. The whole string must match.
fn parse_number(s: &str) -> Option<Num> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let (int_part, frac) = match body.split_once('.') {
        Some((i, f)) if all_digits(f) => (i, Some(f.to_string())),
        Some(_) => return None,
        None => (body, None),
    };
    let int = if int_part.contains(',') {
        let groups: Vec<&str> = int_part.split(',').collect();
        let ok = all_digits(groups[0])
            && groups[0].len() <= 3
            && groups[1..].iter().all(|g| g.len() == 3 && all_digits(g));
        if !ok {
            return None;
        }
        groups.concat()
    } else if int_part.is_empty() && frac.is_some() {
        String::new()
    } else if all_digits(int_part) {
        int_part.to_string()
    } else {
        return None;
    };
    Some(Num { neg, int, frac })
}

/// Splits a leading number from a suffix: `5km` → (5, "km").
fn split_number(s: &str) -> Option<(Num, &str)> {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit()
            || (c == '-' && i == 0)
            || ((c == '.' || c == ',') && s[i + 1..].starts_with(|d: char| d.is_ascii_digit()))
        {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    parse_number(&s[..end]).map(|n| (n, &s[end..]))
}

const TRAILING: &[char] = &[
    '.', ',', ';', ':', '!', '?', '…', '—', '"', '\'', ')', ']', '}', '>',
];
const LEADING: &[char] = &['"', '\'', '(', '[', '{', '<'];
const MAGNITUDES: &[&str] = &["hundred", "thousand", "million", "billion", "trillion"];
const TLDS: &[&str] = &[
    "com", "org", "net", "io", "dev", "no", "edu", "gov", "co", "uk", "app", "ai", "de", "se",
    "us", "info", "rs",
];

fn is_letter_dots(lower: &str) -> bool {
    // u.s. / e.g. shaped: single letters each followed by a period, at least two.
    let b = lower.as_bytes();
    b.len() >= 4
        && b.len().is_multiple_of(2)
        && b.chunks(2)
            .all(|p| p[0].is_ascii_lowercase() && p[1] == b'.')
}

/// Splits trailing punctuation off a token, keeping periods that belong to an abbreviation,
/// an initial or a letter sequence. Returns (core, trailing marks in order).
fn split_trailing<'a>(raw: &'a str, next: Option<&str>) -> (&'a str, Vec<char>) {
    let mut core = raw;
    let mut trail = Vec::new();
    loop {
        let lower = core.to_lowercase();
        if abbreviation_in_context(core, next).is_some() || is_letter_dots(&lower) {
            break;
        }
        // An initial: one capital letter and a period.
        let mut cs = core.chars();
        if let (Some(a), Some('.'), None) = (cs.next(), cs.next(), cs.next())
            && a.is_ascii_uppercase()
        {
            break;
        }
        match core.chars().last() {
            Some(c) if TRAILING.contains(&c) => {
                core = &core[..core.len() - c.len_utf8()];
                trail.insert(0, c);
            }
            _ => break,
        }
    }
    (core, trail)
}

fn strip_leading(raw: &str) -> (&str, Vec<char>) {
    let mut core = raw;
    let mut lead = Vec::new();
    while let Some(c) = core.chars().next() {
        // A leading apostrophe followed by a letter is kept ('tis, 'em).
        if !LEADING.contains(&c)
            || (c == '\''
                && core[1..].starts_with(|x: char| x.is_alphabetic())
                && core.matches('\'').count() == 1)
        {
            break;
        }
        core = &core[c.len_utf8()..];
        lead.push(c);
    }
    (core, lead)
}

struct Out<'a> {
    tokens: &'a mut Vec<Token>,
}

impl Out<'_> {
    fn word(&mut self, text: &str, span: Span) {
        if !text.is_empty() {
            self.tokens.push(Token::Word {
                text: text.to_string(),
                span,
                spell: false,
            });
        }
    }

    fn words<S: AsRef<str>>(&mut self, words: impl IntoIterator<Item = S>, span: Span) {
        for w in words {
            for part in w.as_ref().split(' ') {
                self.word(part, span);
            }
        }
    }

    fn letters(&mut self, text: &str, span: Span) {
        for c in text.chars().filter(|c| c.is_ascii_alphabetic()) {
            self.tokens.push(Token::Word {
                text: c.to_ascii_lowercase().to_string(),
                span,
                spell: true,
            });
        }
    }

    fn punct(&mut self, c: char) {
        let c = match c {
            ')' | ']' | '}' | '(' | '[' | '{' => ',',
            '.' | ',' | ';' | ':' | '!' | '?' | '…' | '—' => c,
            _ => return,
        };
        self.tokens.push(Token::Punct(c));
    }
}

impl EnglishNormalizer {
    pub fn new(known: Box<dyn Fn(&str) -> bool + Send + Sync>) -> EnglishNormalizer {
        EnglishNormalizer {
            known,
            dropped: AtomicUsize::new(0),
        }
    }

    /// Characters dropped so far (emoji, CJK, …).
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    fn pre_tokenize(&self, chars: &[(char, usize)], end_offset: usize) -> Vec<Pre> {
        let mut pres = Vec::new();
        let offset = |i: usize| chars.get(i).map(|c| c.1).unwrap_or(end_offset);
        let mut i = 0;
        let mut phon_mode = false;
        let mut phon_start: Option<usize> = None;
        let mut cur = String::new();
        let mut cur_start = 0usize;
        let mut newlines = 0usize;
        let flush = |cur: &mut String, start: usize, end: usize, pres: &mut Vec<Pre>| {
            if !cur.is_empty() {
                pres.push(Pre::Raw {
                    text: std::mem::take(cur),
                    span: (start, end),
                });
            }
        };
        while i < chars.len() {
            let c = chars[i].0;
            // Embedded command.
            if c == '[' && chars.get(i + 1).map(|x| x.0) == Some('[') {
                let close = (i + 2..chars.len().saturating_sub(1))
                    .find(|j| chars[*j].0 == ']' && chars[*j + 1].0 == ']');
                if let Some(close) = close {
                    let content: String = chars[i + 2..close].iter().map(|x| x.0).collect();
                    let parsed = commands::parse(&content);
                    if phon_mode {
                        if parsed == Parsed::PhonemeMode(false) {
                            let s: String = chars[phon_start.unwrap_or(i)..i]
                                .iter()
                                .map(|x| x.0)
                                .collect();
                            if !s.trim().is_empty() {
                                pres.push(Pre::Tok(Token::PhonemeLiteral(s.trim().to_string())));
                            }
                            phon_mode = false;
                        }
                        // Anything else inside phoneme mode is ignored.
                        if phon_mode {
                            i = close + 2;
                            continue;
                        }
                    } else {
                        flush(&mut cur, cur_start, offset(i), &mut pres);
                        match parsed {
                            Parsed::Command(cmd) => pres.push(Pre::Tok(Token::Command(cmd))),
                            Parsed::PhonemeMode(true) => {
                                phon_mode = true;
                                phon_start = Some(close + 2);
                            }
                            Parsed::PhonemeMode(false) => {}
                            Parsed::Phonemes(p) => pres.push(Pre::Tok(Token::PhonemeLiteral(p))),
                        }
                    }
                    i = close + 2;
                    if phon_mode && phon_start.is_none() {
                        phon_start = Some(i);
                    }
                    continue;
                }
            }
            if phon_mode {
                i += 1;
                continue;
            }
            if c.is_whitespace() {
                flush(&mut cur, cur_start, offset(i), &mut pres);
                if c == '\n' {
                    newlines += 1;
                }
            } else {
                if cur.is_empty() {
                    match newlines {
                        0 => {}
                        1 => pres.push(Pre::Tok(Token::Newline)),
                        _ => pres.push(Pre::Tok(Token::ParagraphBreak)),
                    }
                    newlines = 0;
                    cur_start = offset(i);
                }
                cur.push(c);
            }
            i += 1;
        }
        if phon_mode {
            let s: String = chars[phon_start.unwrap_or(chars.len()).min(chars.len())..]
                .iter()
                .map(|x| x.0)
                .collect();
            if !s.trim().is_empty() {
                pres.push(Pre::Tok(Token::PhonemeLiteral(s.trim().to_string())));
            }
        }
        flush(&mut cur, cur_start, end_offset, &mut pres);
        pres
    }

    fn raw_text(pres: &[Pre], i: usize) -> Option<&str> {
        match pres.get(i) {
            Some(Pre::Raw { text, .. }) => Some(text.as_str()),
            _ => None,
        }
    }

    /// Core (lowercased) and trailing marks of the raw token at `i`, if there is one.
    fn peek(pres: &[Pre], i: usize) -> Option<(&str, Vec<char>, Span)> {
        match pres.get(i) {
            Some(Pre::Raw { text, span }) => {
                let (core, trail) = split_trailing(text, Self::raw_text(pres, i + 1));
                Some((core, trail, *span))
            }
            _ => None,
        }
    }

    fn next_is_capitalized(pres: &[Pre], i: usize) -> bool {
        match pres.get(i) {
            Some(Pre::Raw { text, .. }) => strip_leading(text)
                .0
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase()),
            Some(Pre::Tok(Token::Newline | Token::ParagraphBreak)) => true,
            // At the end of the input the chunk ends anyway.
            _ => false,
        }
    }

    /// A plain or leftover word: letters and apostrophes survive; anything else splits or drops.
    fn plain(&self, core: &str, span: Span, out: &mut Out) {
        let mut cur = String::new();
        let mut digits = String::new();
        let flush_word = |cur: &mut String, out: &mut Out, known: &dyn Fn(&str) -> bool| {
            let w = cur.trim_matches('\'').to_string();
            cur.clear();
            if w.is_empty() {
                return;
            }
            let letters = w.chars().filter(|c| c.is_alphabetic()).count();
            let caps = w.chars().all(|c| c.is_ascii_uppercase());
            if caps && (2..=5).contains(&letters) && !known(&w.to_lowercase()) {
                out.letters(&w, span);
            } else {
                out.word(&w, span);
            }
        };
        let flush_digits = |digits: &mut String, out: &mut Out| {
            if !digits.is_empty() {
                out.words(numbers::cardinal_str(digits), span);
                digits.clear();
            }
        };
        let chars: Vec<char> = core.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            if c.is_ascii_digit() {
                flush_word(&mut cur, out, &*self.known);
                digits.push(c);
            } else if c.is_ascii_alphabetic() || (c == '\'' && !cur.is_empty()) {
                flush_digits(&mut digits, out);
                // camelCase boundary: lower → Upper, or the last capital of a run before a lowercase.
                let prev = i.checked_sub(1).map(|j| chars[j]);
                let next = chars.get(i + 1).copied();
                let boundary = c.is_ascii_uppercase()
                    && prev.is_some_and(|p| {
                        p.is_ascii_lowercase()
                            || (p.is_ascii_uppercase()
                                && next.is_some_and(|n| n.is_ascii_lowercase()))
                    });
                if boundary {
                    flush_word(&mut cur, out, &*self.known);
                }
                cur.push(c);
            } else {
                flush_word(&mut cur, out, &*self.known);
                flush_digits(&mut digits, out);
                let symbol = match c {
                    '&' => "and",
                    '+' => "plus",
                    '=' => "equals",
                    '@' => "at",
                    '#' => "hash",
                    '°' => "degrees",
                    '~' => "approximately",
                    '%' => "percent",
                    '$' => "dollar",
                    '€' => "euro",
                    '£' => "pound",
                    '<' => "less than",
                    '>' => "greater than",
                    '*' => "star",
                    '^' => "caret",
                    '|' => "pipe",
                    '\\' => "backslash",
                    '/' if i > 0 && i + 1 < chars.len() => "slash",
                    '_' | '-' | '.' | '/' | ',' | ':' | ';' | '\'' | '"' | '(' | ')' | '['
                    | ']' | '{' | '}' | '!' | '?' => "",
                    _ => {
                        self.dropped.fetch_add(1, Ordering::Relaxed);
                        ""
                    }
                };
                out.words(symbol.split(' '), span);
            }
        }
        flush_word(&mut cur, out, &*self.known);
        flush_digits(&mut digits, out);
    }

    fn url(&self, core: &str, span: Span, out: &mut Out) -> bool {
        let lower = core.to_lowercase();
        let (rest, had_scheme) = match lower.split_once("://") {
            Some((scheme, rest)) if scheme.chars().all(|c| c.is_ascii_alphabetic()) => {
                (rest.to_string(), true)
            }
            _ => (lower.clone(), false),
        };
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        let host = host.rsplit('@').next().unwrap_or(host);
        let tld = host.rsplit('.').next().unwrap_or("");
        let dotted = host.contains('.')
            && host
                .split('.')
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        let is_email = rest.contains('@') && dotted;
        let is_url = had_scheme
            || rest.starts_with("www.")
            || (dotted && TLDS.contains(&tld) && !all_digits(&host.replace('.', "")));
        if !(is_email || is_url) {
            return false;
        }
        let rest = rest.trim_end_matches('/');
        let mut part = String::new();
        let flush = |part: &mut String, out: &mut Out| {
            if part.is_empty() {
                return;
            }
            let alpha = part.chars().all(|c| c.is_ascii_alphabetic());
            if all_digits(part) {
                out.words(numbers::cardinal_str(part), span);
            } else if alpha && (part.len() == 1 || (part.len() <= 3 && !(self.known)(part))) {
                out.letters(part, span);
            } else {
                self.plain(part, span, out);
            }
            part.clear();
        };
        for c in rest.chars() {
            let name = match c {
                '.' => "dot",
                '/' => "slash",
                '@' => "at",
                '-' => "dash",
                '_' => "underscore",
                ':' => "colon",
                '?' => "question mark",
                '=' => "equals",
                '&' => "and",
                '#' => "hash",
                '~' => "tilde",
                '%' => "percent",
                c if c.is_ascii_alphanumeric() => {
                    // Letter/digit boundary splits the part.
                    if part
                        .chars()
                        .last()
                        .is_some_and(|p| p.is_ascii_digit() != c.is_ascii_digit())
                    {
                        flush(&mut part, out);
                    }
                    part.push(c);
                    continue;
                }
                _ => "",
            };
            flush(&mut part, out);
            out.words(name.split(' '), span);
        }
        flush(&mut part, out);
        true
    }

    /// Classifies the raw token at `i`. Returns how many following raw tokens were consumed.
    fn classify(
        &self,
        core: &str,
        span: Span,
        pres: &[Pre],
        i: usize,
        out: &mut Out,
        trail: &mut Vec<char>,
    ) -> usize {
        let lower = core.to_lowercase();
        let next = Self::peek(pres, i + 1);
        let next_lower = next.as_ref().map(|n| n.0.to_lowercase());
        // Consuming the next token moves its trailing punctuation to this one.
        let take_next = |trail: &mut Vec<char>| {
            if let Some((_, t, _)) = &next {
                trail.extend(t.iter().copied());
            }
            1
        };
        let no_trail = trail.is_empty();

        // Currency.
        let currency = [
            ('$', "dollar", "dollars"),
            ('€', "euro", "euros"),
            ('£', "pound", "pounds"),
            ('¥', "yen", "yen"),
        ];
        if let Some(first) = core.chars().next()
            && let Some((_, one, many)) = currency.iter().find(|c| c.0 == first)
            && let Some((num, suffix)) = split_number(&core[first.len_utf8()..])
        {
            let glued = match suffix {
                "k" | "K" => Some("thousand"),
                "m" | "M" => Some("million"),
                "b" | "B" | "bn" => Some("billion"),
                "" => None,
                _ => Some(""),
            };
            if glued != Some("") {
                let spaced = next_lower
                    .as_deref()
                    .filter(|n| no_trail && glued.is_none() && MAGNITUDES.contains(n));
                let mut consumed = 0;
                if let Some(mag) = glued.or(spaced) {
                    out.words(num.words(), span);
                    out.word(mag, span);
                    out.word(many, span);
                    if spaced.is_some() {
                        consumed = take_next(trail);
                    }
                } else {
                    let cents = num
                        .frac
                        .as_deref()
                        .filter(|f| f.len() == 2)
                        .and_then(|f| f.parse::<u64>().ok());
                    let whole = Num {
                        frac: None,
                        ..num.clone()
                    };
                    let zero_dollars = whole.int.trim_start_matches('0').is_empty();
                    match cents {
                        Some(c) if zero_dollars && c > 0 => {
                            out.words(numbers::cardinal(c), span);
                            out.word(if c == 1 { "cent" } else { "cents" }, span);
                        }
                        Some(c) => {
                            out.words(whole.words(), span);
                            out.word(if whole.is_one() { one } else { many }, span);
                            if c > 0 {
                                out.words(numbers::cardinal(c), span);
                            }
                        }
                        None => {
                            out.words(num.words(), span);
                            out.word(if num.is_one() { one } else { many }, span);
                        }
                    }
                }
                return consumed;
            }
        }

        // Time: 17:30, 9:05 a.m., 5pm, 5 p.m.
        let meridiem = |s: &str| match s {
            "am" | "a.m." | "a.m" => Some("a"),
            "pm" | "p.m." | "p.m" => Some("p"),
            _ => None,
        };
        {
            let (clock, glued) = match lower.find(|c: char| c.is_ascii_alphabetic()) {
                Some(at) => (&lower[..at], meridiem(&lower[at..])),
                None => (lower.as_str(), None),
            };
            let has_suffix = clock.len() != lower.len();
            let parts: Vec<&str> = clock.split(':').collect();
            let hm = match parts.as_slice() {
                [h, m] if all_digits(h) && h.len() <= 2 && m.len() == 2 && all_digits(m) => Some((
                    h.parse::<u64>().unwrap_or(99),
                    m.parse::<u64>().unwrap_or(99),
                )),
                [h] if all_digits(h) && h.len() <= 2 => Some((h.parse::<u64>().unwrap_or(99), 100)),
                _ => None,
            };
            let spaced = if !has_suffix && no_trail {
                next_lower.as_deref().and_then(meridiem)
            } else {
                None
            };
            if let Some((h, m)) = hm
                && h <= 24
                && (m < 60 || m == 100)
                && (!has_suffix || glued.is_some())
                && (m != 100 || glued.is_some() || spaced.is_some())
            {
                let mer = glued.or(spaced);
                out.words(numbers::cardinal(h), span);
                match m {
                    100 => {}
                    0 if mer.is_none() && h <= 12 => out.word("o'clock", span),
                    0 if mer.is_none() => out.word("hundred", span),
                    0 => {}
                    m => out.words(numbers::two_digit(m, true), span),
                }
                if let Some(letter) = mer {
                    out.letters(letter, span);
                    out.letters("m", span);
                }
                // A sentence-final "p.m." keeps its period (spec §7.1).
                if spaced.is_some() {
                    let ends = next.as_ref().is_some_and(|n| n.0.ends_with('.'))
                        && Self::next_is_capitalized(pres, i + 2);
                    let consumed = take_next(trail);
                    if ends && trail.is_empty() {
                        trail.push('.');
                    }
                    return consumed;
                }
                return 0;
            }
        }

        // Dates: 2024-12-03, 12/03/2024, Dec 3, 2024.
        let date = |m: u64, d: u64, y: Option<u64>, out: &mut Out| {
            out.word(MONTHS[(m - 1) as usize], span);
            out.words(numbers::ordinal(d), span);
            if let Some(y) = y {
                out.words(numbers::year(y), span);
            }
        };
        {
            let iso: Vec<&str> = core.split('-').collect();
            if let [y, m, d] = iso.as_slice()
                && y.len() == 4
                && m.len() == 2
                && d.len() == 2
                && all_digits(y)
                && all_digits(m)
                && all_digits(d)
            {
                let (y, m, d) = (
                    y.parse::<u64>().unwrap_or(0),
                    m.parse::<u64>().unwrap_or(0),
                    d.parse::<u64>().unwrap_or(0),
                );
                if (1..=12).contains(&m) && (1..=31).contains(&d) {
                    date(m, d, Some(y), out);
                    return 0;
                }
            }
            let us: Vec<&str> = core.split('/').collect();
            if let [m, d, y] = us.as_slice()
                && all_digits(m)
                && all_digits(d)
                && all_digits(y)
                && y.len() == 4
                && m.len() <= 2
                && d.len() <= 2
            {
                let (m, d, y) = (
                    m.parse::<u64>().unwrap_or(0),
                    d.parse::<u64>().unwrap_or(0),
                    y.parse::<u64>().unwrap_or(0),
                );
                if (1..=12).contains(&m) && (1..=31).contains(&d) {
                    date(m, d, Some(y), out);
                    return 0;
                }
            }
            if let Some(m) = month(core)
                && no_trail
                && let Some((day_core, day_trail, _)) = &next
            {
                let day_digits = day_core.trim_end_matches(|c: char| c.is_ascii_alphabetic());
                let suffix = &day_core[day_digits.len()..];
                let day = day_digits
                    .parse::<u64>()
                    .ok()
                    .filter(|d| (1..=31).contains(d) && day_digits.len() <= 2);
                if let Some(d) = day
                    && matches!(suffix, "" | "st" | "nd" | "rd" | "th")
                {
                    // Optional year after "3," or "3".
                    let after = Self::peek(pres, i + 2);
                    let year = after.as_ref().and_then(|a| {
                        a.0.parse::<u64>()
                            .ok()
                            .filter(|y| a.0.len() == 4 && (1100..=2099).contains(y))
                    });
                    let year_ok =
                        year.is_some() && (day_trail.is_empty() || day_trail.as_slice() == [',']);
                    if year_ok {
                        date(m as u64, d, year, out);
                        if let Some(a) = after {
                            trail.extend(a.1);
                        }
                        return 2;
                    }
                    date(m as u64, d, None, out);
                    trail.extend(day_trail.iter().copied());
                    return 1;
                }
            }
        }

        // Ordinal.
        for suffix in ["st", "nd", "rd", "th"] {
            if let Some(d) = lower.strip_suffix(suffix)
                && all_digits(d)
                && let Ok(n) = d.parse::<u64>()
            {
                out.words(numbers::ordinal(n), span);
                return 0;
            }
        }

        // Percent, units, cardinals, decimals, years.
        if let Some((num, suffix)) = split_number(core) {
            if suffix == "%" {
                out.words(num.words(), span);
                out.word("percent", span);
                return 0;
            }
            if let Some((one, many)) = unit(suffix) {
                out.words(num.words(), span);
                out.words((if num.is_one() { one } else { many }).split(' '), span);
                return 0;
            }
            if suffix == "°" {
                out.words(num.words(), span);
                out.word(if num.is_one() { "degree" } else { "degrees" }, span);
                return 0;
            }
            if suffix.is_empty() {
                // Unit as the next token: 10 kg.
                if no_trail
                    && let Some((ncore, _, _)) = &next
                    && !needs_glue(ncore)
                    && let Some((one, many)) = unit(ncore)
                {
                    out.words(num.words(), span);
                    out.words((if num.is_one() { one } else { many }).split(' '), span);
                    return take_next(trail);
                }
                let followed_by_unit = no_trail && next_lower.as_deref().is_some_and(is_unit_word);
                if let Some(y) = num.value()
                    && core.len() == 4
                    && (1100..=2099).contains(&y)
                    && !followed_by_unit
                {
                    out.words(numbers::year(y), span);
                } else {
                    out.words(num.words(), span);
                }
                return 0;
            }
        }

        // Phone-like digit strings, ranges and fractions.
        {
            let digit_count = core.chars().filter(|c| c.is_ascii_digit()).count();
            let only = core
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '-' | '.' | '(' | ')' | '+'));
            let groups: Vec<&str> = core
                .split(['-', '.', '(', ')', '+'])
                .filter(|g| !g.is_empty())
                .collect();
            if only && digit_count >= 7 && groups.len() >= 2 {
                for (gi, g) in groups.iter().enumerate() {
                    if gi > 0 {
                        out.punct(',');
                    }
                    out.words(numbers::digits(g), span);
                }
                return 0;
            }
            if let Some((a, b)) = core.split_once('-')
                && all_digits(a)
                && all_digits(b)
            {
                out.words(numbers::cardinal_str(a), span);
                out.word("to", span);
                out.words(numbers::cardinal_str(b), span);
                return 0;
            }
            if let Some((a, b)) = core.split_once('/')
                && let (Ok(n), Ok(d)) = (a.parse::<u64>(), b.parse::<u64>())
                && all_digits(a)
                && all_digits(b)
            {
                if d >= 2 && n < d && d <= 100 {
                    out.words(numbers::fraction(n, d), span);
                } else {
                    out.words(numbers::cardinal(n), span);
                    out.word("over", span);
                    out.words(numbers::cardinal(d), span);
                }
                return 0;
            }
        }

        // Abbreviations and letter sequences.
        if let Some((expansion, title)) = abbreviation_in_context(core, Self::raw_text(pres, i + 1))
        {
            let capitalized_next = Self::next_is_capitalized(pres, i + 1);
            let expansion = if lower == "st."
                && capitalized_next
                && pres
                    .get(i + 1)
                    .is_some_and(|p| matches!(p, Pre::Raw { .. }))
            {
                "saint"
            } else {
                expansion
            };
            if is_letter_expansion(expansion) {
                out.letters(&expansion.replace(' ', ""), span);
            } else {
                out.words(expansion.split(' '), span);
            }
            if !title && capitalized_next && trail.is_empty() {
                trail.push('.');
            }
            return 0;
        }
        if is_letter_dots(&lower) {
            out.letters(&lower.replace('.', ""), span);
            if Self::next_is_capitalized(pres, i + 1)
                && trail.is_empty()
                && lower != "u.s."
                && lower != "u.k."
            {
                trail.push('.');
            }
            return 0;
        }
        {
            let mut cs = core.chars();
            if let (Some(a), Some('.'), None) = (cs.next(), cs.next(), cs.next())
                && a.is_ascii_uppercase()
            {
                out.letters(&a.to_string(), span);
                return 0;
            }
        }

        if self.url(core, span, out) {
            return 0;
        }

        // Symbols standing alone.
        let symbol = match core {
            "&" => Some("and"),
            "+" => Some("plus"),
            "=" => Some("equals"),
            "@" => Some("at"),
            "#" => Some("hash"),
            "~" => Some("approximately"),
            "/" => Some("slash"),
            "%" => Some("percent"),
            "<" => Some("less than"),
            ">" => Some("greater than"),
            "°" => Some("degrees"),
            "-" | "--" | "---" => {
                out.punct('—');
                return 0;
            }
            _ => None,
        };
        if let Some(s) = symbol {
            out.words(s.split(' '), span);
            return 0;
        }
        // "#5" → number five; "~5" → approximately five.
        if let Some(rest) = core.strip_prefix('#')
            && all_digits(rest)
        {
            out.word("number", span);
            out.words(numbers::cardinal_str(rest), span);
            return 0;
        }

        // Words, identifiers, acronyms and everything left over.
        self.plain(core, span, out);
        0
    }
}

impl Normalizer for EnglishNormalizer {
    fn normalize(&self, text: &str) -> Vec<Token> {
        let chars = fold_text(text);
        let pres = self.pre_tokenize(&chars, text.len());
        let mut tokens: Vec<Token> = Vec::new();
        let mut i = 0;
        while i < pres.len() {
            match &pres[i] {
                Pre::Tok(t) => {
                    tokens.push(t.clone());
                    i += 1;
                }
                Pre::Raw { text: raw, span } => {
                    let (rest, lead) = strip_leading(raw);
                    let (core, mut trail) = split_trailing(rest, Self::raw_text(&pres, i + 1));
                    let mut out = Out {
                        tokens: &mut tokens,
                    };
                    for c in lead {
                        out.punct(c);
                    }
                    let mut consumed = 0;
                    if !core.is_empty() {
                        consumed = self.classify(core, *span, &pres, i, &mut out, &mut trail);
                    }
                    // "..." is an ellipsis; of several marks the first sentence mark wins.
                    let dots = trail.iter().filter(|c| **c == '.').count();
                    if dots >= 3 {
                        out.punct('…');
                    } else {
                        let pick = trail
                            .iter()
                            .find(|c| matches!(c, '.' | '!' | '?' | '…'))
                            .or_else(|| trail.iter().find(|c| !matches!(c, '"' | '\'')));
                        if let Some(c) = pick {
                            out.punct(*c);
                        }
                    }
                    i += 1 + consumed;
                }
            }
        }
        // Collapse runs of punctuation: a sentence mark beats a clause mark.
        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        for t in tokens {
            if let (Token::Punct(new), Some(Token::Punct(old))) = (&t, out.last()) {
                let strong = |c: char| matches!(c, '.' | '!' | '?' | '…');
                if strong(*new) && !strong(*old) {
                    out.pop();
                    out.push(t);
                }
                continue;
            }
            out.push(t);
        }
        // Leading punctuation carries nothing.
        while matches!(out.first(), Some(Token::Punct(_))) {
            out.remove(0);
        }
        out
    }
}

/// Normalized text of a token run, as `--dump text` prints it.
pub fn tokens_to_text(tokens: &[Token]) -> String {
    let mut s = String::new();
    for t in tokens {
        match t {
            Token::Word { text, .. } => {
                if !s.is_empty() && !s.ends_with('\n') {
                    s.push(' ');
                }
                s.push_str(text);
            }
            Token::Punct(c) => {
                if *c == '—' {
                    s.push(' ');
                }
                s.push(*c);
            }
            Token::PhonemeLiteral(p) => {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&format!("[[{p}]]"));
            }
            Token::Command(_) | Token::Newline => {}
            Token::ParagraphBreak => s.push('\n'),
        }
    }
    s
}
