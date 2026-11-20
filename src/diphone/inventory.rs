//! The diphone inventory: ordered pairs over the 44 phones plus silence (spec §8.2.4).

use crate::phone::Phone;

/// Pairs that do not occur and are not built.
pub fn impossible(a: Phone, b: Phone) -> bool {
    use Phone::*;
    matches!(
        (a, b),
        (SIL, SIL)
            | (SIL, NG)
            | (SIL, DX)
            | (DX, SIL)
            | (Q, Q)
            | (DX, DX)
            | (NG, NG)
            | (HH, SIL)
            | (HH, HH)
            | (DX, Q)
            | (Q, DX)
            | (SIL, ZH)
    )
}

/// Every diphone of the inventory, in a stable order.
pub fn all() -> Vec<(Phone, Phone)> {
    let mut out = Vec::with_capacity(Phone::COUNT * Phone::COUNT);
    for a in Phone::ALL {
        for b in Phone::ALL {
            if !impossible(*a, *b) {
                out.push((*a, *b));
            }
        }
    }
    out
}

pub fn name(a: Phone, b: Phone) -> String {
    format!("{}-{}", a.text(), b.text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_and_contents() {
        let inv = all();
        assert!(inv.len() >= 1900 && inv.len() < 2025, "{}", inv.len());
        assert!(inv.contains(&(Phone::SIL, Phone::HH)));
        assert!(inv.contains(&(Phone::AA, Phone::AA)));
        assert!(!inv.contains(&(Phone::SIL, Phone::NG)));
        assert_eq!(name(Phone::SIL, Phone::HH), "_-HH");
    }
}
