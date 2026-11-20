//! Letter-to-sound accuracy on the hand-checked word list (spec §16.3).

use speak::text::lts::{CartLts, RuleLts, evaluate};

const WORDS: &str = include_str!("fixtures/lts_words.txt");

#[test]
fn rule_lts_per() {
    let ev = evaluate(&RuleLts::new(), WORDS);
    if std::env::var_os("LTS_VERBOSE").is_some() {
        for (w, r, h) in &ev.misses {
            println!("{w:<14} ref {r:<32} got {h}");
        }
    }
    println!(
        "rules: PER {:.1} %, words correct {}/{}",
        ev.per() * 100.0,
        ev.words_correct,
        ev.words
    );
    assert!(ev.words >= 500);
    assert!(ev.per() <= 0.35, "PER {:.1} %", ev.per() * 100.0);
}

#[test]
fn cart_beats_rules() {
    let Some(cart) = CartLts::embedded() else {
        println!("skipped: this build has no tree");
        return;
    };
    let (c, r) = (evaluate(&cart, WORDS), evaluate(&RuleLts::new(), WORDS));
    println!(
        "cart: PER {:.1} %, words correct {}/{}",
        c.per() * 100.0,
        c.words_correct,
        c.words
    );
    assert!(
        c.per() < r.per(),
        "cart {:.3} vs rules {:.3}",
        c.per(),
        r.per()
    );
}

#[test]
fn cart_never_returns_nothing() {
    let Some(cart) = CartLts::embedded() else {
        return;
    };
    use speak::traits::LetterToSound;
    for w in [
        "qzxwv",
        "blorptastic",
        "x",
        "zzz",
        "a",
        "bcdfg",
        "rhythm",
        "supercalifragilistic",
    ] {
        let segs = cart.pronounce(w);
        assert!(!segs.is_empty(), "{w}");
        assert!(
            segs.iter()
                .filter(|s| s.stress == speak::Stress::Primary)
                .count()
                <= 1,
            "{w}"
        );
    }
}
