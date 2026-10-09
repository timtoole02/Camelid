use std::panic::catch_unwind;

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    match catch_unwind(|| pdf_extract::extract_text_from_mem(&bytes)) {
        Ok(Ok(text)) => {
            for word in text.split_whitespace() {
                println!("{word}");
            }
        }
        _ => println!("<<FAILED>>"),
    }
}
