//! Abbreviations and month names (spec §7.1). Keys are lowercase, with their periods.

/// `(abbreviation, expansion, is a title that never ends a sentence)`.
pub const ABBREVIATIONS: &[(&str, &str, bool)] = &[
    ("dr.", "doctor", true),
    ("mr.", "mister", true),
    ("mrs.", "missus", true),
    ("ms.", "miz", true),
    ("prof.", "professor", true),
    ("st.", "street", true),
    ("sr.", "senior", false),
    ("jr.", "junior", false),
    ("gen.", "general", true),
    ("col.", "colonel", true),
    ("capt.", "captain", true),
    ("lt.", "lieutenant", true),
    ("sgt.", "sergeant", true),
    ("gov.", "governor", true),
    ("sen.", "senator", true),
    ("rep.", "representative", true),
    ("rev.", "reverend", true),
    ("hon.", "honorable", true),
    ("pres.", "president", true),
    ("vs.", "versus", true),
    ("v.", "versus", true),
    ("e.g.", "for example", true),
    ("i.e.", "that is", true),
    ("etc.", "et cetera", false),
    ("no.", "number", true),
    ("nos.", "numbers", true),
    ("vol.", "volume", true),
    ("ch.", "chapter", true),
    ("fig.", "figure", true),
    ("eq.", "equation", true),
    ("sec.", "section", true),
    ("p.", "page", true),
    ("pp.", "pages", true),
    ("ed.", "edition", false),
    ("est.", "established", true),
    ("approx.", "approximately", true),
    ("dept.", "department", false),
    ("inc.", "incorporated", false),
    ("ltd.", "limited", false),
    ("corp.", "corporation", false),
    ("co.", "company", false),
    ("ave.", "avenue", false),
    ("blvd.", "boulevard", false),
    ("rd.", "road", false),
    ("mt.", "mount", true),
    ("ft.", "feet", false),
    ("min.", "minutes", false),
    ("max.", "maximum", false),
    ("misc.", "miscellaneous", false),
    ("tel.", "telephone", true),
    ("ext.", "extension", true),
    ("a.m.", "a m", false),
    ("p.m.", "p m", false),
    ("u.s.", "u s", true),
    ("u.k.", "u k", true),
    ("u.s.a.", "u s a", false),
    ("ph.d.", "p h d", false),
    ("b.c.", "b c", false),
    ("a.d.", "a d", false),
    ("jan.", "january", true),
    ("feb.", "february", true),
    ("mar.", "march", true),
    ("apr.", "april", true),
    ("jun.", "june", true),
    ("jul.", "july", true),
    ("aug.", "august", true),
    ("sep.", "september", true),
    ("sept.", "september", true),
    ("oct.", "october", true),
    ("nov.", "november", true),
    ("dec.", "december", true),
    ("mon.", "monday", false),
    ("tue.", "tuesday", false),
    ("wed.", "wednesday", false),
    ("thu.", "thursday", false),
    ("fri.", "friday", false),
    ("sat.", "saturday", false),
    ("sun.", "sunday", false),
];

/// Expansions that are letter sequences: each word is a letter name.
pub fn is_letter_expansion(expansion: &str) -> bool {
    expansion.split(' ').all(|w| w.len() == 1)
}

/// Abbreviations that are also ordinary words: "The cat sat." must not become Saturday.
const AMBIGUOUS: &[&str] = &[
    "sat.", "sun.", "wed.", "mon.", "mar.", "jan.", "dec.", "no.", "nos.", "min.", "max.", "ed.",
    "co.", "est.", "fig.", "p.", "pp.", "v.", "ch.", "sec.", "inc.", "ext.", "tel.", "rep.",
    "sen.", "gen.", "pres.", "hon.", "rev.", "col.", "ft.", "rd.", "mt.", "misc.", "dept.",
    "corp.", "ltd.", "vol.", "eq.", "st.",
];

/// The abbreviation `token` (with its period, original case) stands for in this context.
/// `next`: the following token, if any. Ambiguous ones need support from the context:
/// a capital letter and something after them, or (for `no.`, `p.`, `fig.`, …) a number.
pub fn abbreviation_in_context(token: &str, next: Option<&str>) -> Option<(&'static str, bool)> {
    let lower = token.to_lowercase();
    let hit = abbreviation(&lower)?;
    if !AMBIGUOUS.contains(&lower.as_str()) {
        return Some(hit);
    }
    let capital = token.chars().next().is_some_and(|c| c.is_uppercase());
    let next_digit = next.is_some_and(|n| {
        n.trim_start_matches(['(', '#'])
            .starts_with(|c: char| c.is_ascii_digit())
    });
    let needs_number = matches!(
        lower.as_str(),
        "no." | "nos." | "p." | "pp." | "fig." | "ch." | "sec." | "vol." | "eq." | "ext." | "tel."
    );
    let ok = if needs_number {
        next_digit
    } else {
        (capital && next.is_some()) || next_digit
    };
    ok.then_some(hit)
}

pub fn abbreviation(lower: &str) -> Option<(&'static str, bool)> {
    ABBREVIATIONS
        .iter()
        .find(|a| a.0 == lower)
        .map(|a| (a.1, a.2))
}

pub const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Month number (1–12) from a full name or a three-letter abbreviation, with or without a period.
pub fn month(word: &str) -> Option<usize> {
    let w = word.trim_end_matches('.').to_lowercase();
    if w.len() < 3 {
        return None;
    }
    MONTHS
        .iter()
        .position(|m| {
            *m == w || (w.len() == 3 && m.starts_with(&w)) || (w == "sept" && *m == "september")
        })
        .map(|i| i + 1)
}
