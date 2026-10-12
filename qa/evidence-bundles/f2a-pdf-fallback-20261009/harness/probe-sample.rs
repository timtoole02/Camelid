use std::panic::{catch_unwind, AssertUnwindSafe};

fn lopdf_text(bytes: &[u8]) -> String {
    let doc = lopdf::Document::load_mem(bytes).unwrap();
    let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    let mut out = String::new();
    for page in pages {
        if let Ok(Ok(text)) = catch_unwind(AssertUnwindSafe(|| doc.extract_text(&[page]))) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    out
}

fn sample(text: &str, from: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    flat.chars().skip(from).take(320).collect()
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap();
    for name in args {
        let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
        let extract = catch_unwind(|| pdf_extract::extract_text_from_mem(&bytes)).ok().and_then(|r| r.ok());
        let lo = lopdf_text(&bytes);
        println!("===== {name}");
        match &extract {
            Some(text) => println!("pdf-extract: {}", sample(text, 2000)),
            None => println!("pdf-extract: (failed)"),
        }
        println!("lopdf:       {}", sample(&lo, 2000));
        let long_tokens = lo.split_whitespace().filter(|w| w.chars().count() > 25).count();
        let replacement = lo.chars().filter(|c| *c == '\u{fffd}').count();
        println!("lopdf: {} words, {} tokens over 25 chars, {} replacement chars", lo.split_whitespace().count(), long_tokens, replacement);
    }
}
