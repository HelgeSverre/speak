//! `-v '?'` and `--list-backends` (spec §3.4).

use crate::traits::{Backend, DataDirs};

pub fn voices(backends: &[Box<dyn Backend>], filter: Option<&str>, data: &DataDirs) -> String {
    let mut rows = Vec::new();
    for b in backends {
        if filter.is_some_and(|f| f != b.id()) {
            continue;
        }
        let mut vs = b.voices(data);
        vs.sort_by(|a, b| a.name.cmp(&b.name));
        rows.extend(vs);
    }
    rows.sort_by(|a, b| a.backend.cmp(b.backend).then(a.name.cmp(&b.name)));
    let width = rows.iter().map(|v| v.name.len()).max().unwrap_or(0).max(16);
    rows.iter()
        .map(|v| {
            format!(
                "{:<width$}  {}  {:<7}  {:<9}  # {}\n",
                v.name,
                v.locale,
                v.backend,
                v.state.label(),
                v.description
            )
        })
        .collect()
}

pub fn backend_table(backends: &[Box<dyn Backend>], data: &DataDirs) -> String {
    let mut s =
        String::from("id       technique                                    rate   voices\n");
    for b in backends {
        s.push_str(&format!(
            "{:<8} {:<44} {:<6} {}\n",
            b.id(),
            b.technique(),
            b.sample_rate(),
            b.voices(data).len()
        ));
    }
    s
}
