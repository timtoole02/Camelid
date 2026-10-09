use std::panic::{catch_unwind, AssertUnwindSafe};

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

fn lopdf_text(bytes: &[u8]) -> Result<String, String> {
    let doc = lopdf::Document::load_mem(bytes).map_err(|e| e.to_string())?;
    let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    let mut out = String::new();
    for page in pages {
        if let Ok(Ok(text)) = catch_unwind(AssertUnwindSafe(|| doc.extract_text(&[page]))) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    Ok(out)
}

fn show<T: std::fmt::Display>(result: &Result<Result<T, String>, ()>) -> String {
    match result {
        Err(()) => "PANIC".to_string(),
        Ok(Err(e)) => format!("ERR {}", &e[..e.len().min(40)]),
        Ok(Ok(n)) => n.to_string(),
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap();
    let list = std::fs::read_to_string(args.next().unwrap()).unwrap();
    println!("file\tpdf_extract_words\tlopdf_words");
    for name in list.lines().filter(|line| !line.is_empty()) {
        let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
        let extract = catch_unwind(|| pdf_extract::extract_text_from_mem(&bytes))
            .map(|r| r.map(|t| words(&t)).map_err(|e| e.to_string()))
            .map_err(|_| ());
        let lo = catch_unwind(AssertUnwindSafe(|| lopdf_text(&bytes)))
            .map(|r| r.map(|t| words(&t)))
            .map_err(|_| ());
        println!("{name}\t{}\t{}", show(&extract), show(&lo));
    }
}
