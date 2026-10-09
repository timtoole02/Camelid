use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::catch_unwind;

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let dir = std::env::args().nth(1).unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap()
        .filter_map(|entry| entry.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|name| name.ends_with(".pdf"))
        .collect();
    names.sort();
    for name in names {
        let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
        let result = catch_unwind(|| pdf_extract::extract_text_from_mem(&bytes));
        let line = match result {
            Err(_) => "PANIC".to_string(),
            Ok(Err(_)) => "ERR".to_string(),
            Ok(Ok(text)) => {
                let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
                let mut hasher = DefaultHasher::new();
                normalized.hash(&mut hasher);
                format!("{:016x} {}", hasher.finish(), normalized.split(' ').count())
            }
        };
        println!("{name}\t{line}");
    }
}
