//! Postlexical rules, G2P and the frontend as a whole (spec §16.3).

use speak::phone::print_arpabet;
use speak::text::Frontend;
use speak::{Controls, Unit, WordSource};

fn phones(text: &str) -> String {
    let fe = Frontend::new(None, None, Some("rules")).unwrap();
    fe.chunks(text, Controls::default())
        .unwrap()
        .iter()
        .map(|c| print_arpabet(&c.units))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn flapping() {
    assert_eq!(phones("butter"), "B AH1 DX AXR");
    assert!(phones("get it").contains("G EH1 DX / IH"));
    // Not before a stressed vowel in the same word, not word-initially.
    assert!(phones("attack").contains("T AE1 K"));
    assert!(phones("a tomato").starts_with("AX / T"));
}

#[test]
fn the_before_vowels() {
    assert!(phones("the apple").starts_with("DH IY / "));
    assert!(phones("the cat").starts_with("DH AX / "));
}

#[test]
fn a_and_to() {
    assert_eq!(phones("a"), "Q EY1");
    assert!(phones("a cat").starts_with("AX / "));
    assert!(phones("to go").starts_with("T AX / "));
    assert!(phones("to eat").starts_with("T UW / "));
}

#[test]
fn glottal_onset() {
    assert!(phones("Apples fall.").starts_with("Q AE1"));
    assert!(!phones("The apples fall.").contains('Q'));
}

#[test]
fn geminates_merge() {
    let p = phones("big game");
    assert_eq!(p, "B IH1 / G EY1 M");
    let fe = Frontend::new(None, None, None).unwrap();
    let c = &fe.chunks("big game", Controls::default()).unwrap()[0];
    assert!(c.segments().any(|s| (s.dur_scale - 1.3).abs() < 1e-6));
}

#[test]
fn function_words_lose_stress() {
    let p = phones("I can see that it was good.");
    assert!(p.contains("AY / K AE N / S IY1"), "{p}");
    assert!(p.contains("W AH Z / G UH1 D"), "{p}");
}

#[test]
fn syllabic_reduction_marks_the_schwa_short() {
    let fe = Frontend::new(None, None, None).unwrap();
    let c = &fe.chunks("little button", Controls::default()).unwrap()[0];
    assert!(
        c.segments()
            .filter(|s| (s.dur_scale - 0.6).abs() < 1e-6)
            .count()
            >= 2
    );
}

#[test]
fn sources_are_recorded() {
    let fe = Frontend::new(None, None, None).unwrap();
    let c = &fe
        .chunks("The blorptastic FBI cats ran", Controls::default())
        .unwrap()[0];
    let src: Vec<WordSource> = c.words.iter().map(|w| w.source).collect();
    assert_eq!(src[0], WordSource::Builtin);
    assert!(matches!(src[1], WordSource::Cart | WordSource::Rules));
    assert_eq!(&src[2..5], [WordSource::Spelled; 3]);
}

#[test]
fn morphology_and_compounds() {
    assert_eq!(phones("dogs"), "D AO1 G Z");
    assert_eq!(phones("tests"), "T EH1 S T S");
    assert_eq!(phones("boxes"), "B AA1 K S IX Z");
    assert_eq!(phones("tested"), "T EH1 S T IX D");
    assert_eq!(phones("jumped"), "JH AH1 M P T");
    assert_eq!(phones("speaking"), "S P IY1 K IX NG");
    assert_eq!(phones("dog's"), "D AO1 G Z");
    assert_eq!(phones("quickly"), "K W IH1 K L IY");
    assert_eq!(phones("doghouse"), "D AO1 G HH AW2 S");
}

#[test]
fn nothing_is_dropped_and_no_digits_survive() {
    let corpus = include_str!("fixtures/sentences.txt").to_string()
        + "\nqzxwv blorptastic 1969 x86 $5 3rd 99% a.b@c.io 2024-13-45 12:99 ٣ 😀 naïve\n";
    let fe = Frontend::new(None, None, None).unwrap();
    for line in corpus.lines() {
        for c in fe.chunks(line, Controls::default()).unwrap() {
            let text = print_arpabet(&c.units);
            assert!(
                !text
                    .chars()
                    .any(|ch| ch.is_ascii_digit() && !"012".contains(ch)),
                "{text}"
            );
            for w in &c.words {
                assert!(
                    c.units.iter().any(|u| matches!(u, Unit::Seg(_))),
                    "word {} lost in {line}",
                    w.text
                );
            }
            // Every word has at least one segment.
            for wi in 0..c.words.len() {
                assert!(
                    c.segments().any(|s| s.word as usize == wi)
                        || c.words[wi].text.starts_with("[["),
                    "'{}' has no phones in: {line}",
                    c.words[wi].text
                );
            }
        }
    }
}

#[test]
fn inline_phonemes_equal_phoneme_input() {
    let p = phones("[[inpt PHON]] HH AX L OW1 [[inpt TEXT]]");
    assert_eq!(p, "HH AX L OW1");
    assert_eq!(
        phones("say [[ W ER1 L D ]] now")
            .matches("W ER1 L D")
            .count(),
        1
    );
}

#[test]
fn homographs_follow_their_context() {
    let p = phones("I read it yesterday. I will read it tomorrow.");
    let parts: Vec<&str> = p.split(" | ").collect();
    assert!(parts[0].contains("R EH1"), "{p}");
    assert!(parts[1].contains("R IY1"), "{p}");
    let p = phones("They record a record.");
    // Verb first (stress on the second syllable), then the noun; the verb's D flaps before "a".
    assert!(p.contains("R IX K AO1 R DX / AX / R EH1 K AXR D"), "{p}");
}

#[test]
fn compiled_lexicon_is_used_when_present_and_optional() {
    use speak::text::lexicon::compiled;
    let dir = std::env::temp_dir().join(format!("speak-lex-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("lexicon")).unwrap();
    // Without the file everything still works.
    let fe = Frontend::new(None, Some(&dir), None).unwrap();
    assert!(!fe.lexicons.has_cmudict());
    assert!(
        !fe.chunks("Hello zyzzyva.", Controls::default())
            .unwrap()
            .is_empty()
    );
    // With it, lookups hit and are reported as cmudict.
    let words = "zyzzyva Z IH1 Z IH0 V AH0\nquark K W AA1 R K\naardvark AA1 R D V AA2 R K\n";
    let (text, n) = compiled::build(words);
    assert_eq!(n, 3);
    std::fs::write(dir.join("lexicon").join(compiled::FILE_NAME), text).unwrap();
    let t = std::time::Instant::now();
    let fe = Frontend::new(None, Some(&dir), None).unwrap();
    assert!(t.elapsed().as_millis() < 200);
    assert!(fe.lexicons.has_cmudict());
    let c = &fe
        .chunks("zyzzyva quark blorp", Controls::default())
        .unwrap()[0];
    assert_eq!(c.words[0].source, WordSource::Cmudict);
    assert_eq!(c.words[1].source, WordSource::Cmudict);
    assert_ne!(c.words[2].source, WordSource::Cmudict);
    assert!(print_arpabet(&c.units).starts_with("Z IH1 Z IX V AX / K W AA1 R K"));
    let _ = std::fs::remove_dir_all(&dir);
}
