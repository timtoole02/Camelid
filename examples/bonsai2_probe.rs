//! Reproducible packed-weight text/image probe. Supply a pre-rendered ChatML prompt.
//! cargo run --release --example bonsai2_probe -- MODEL PROMPT_FILE TOKENS [MMPROJ IMAGE]
use camelid::{
    gguf,
    runnable::{PrismVisionProjector, RunnableModel},
    tokenizer::Tokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 && args.len() != 6 {
        return Err("expected MODEL PROMPT_FILE TOKENS [MMPROJ IMAGE]".into());
    }
    let metadata = gguf::read_metadata(&args[1])?;
    let tokenizer = Tokenizer::from_gguf(&metadata)?;
    let prompt = std::fs::read_to_string(&args[2])?;
    let model = RunnableModel::load(&args[1])?;
    let max = args[3].parse()?;
    let stop = tokenizer.special.eog.iter().copied().collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let ids = if args.len() == 6 {
        let (prefix, suffix) = prompt
            .split_once("<|image_pad|>")
            .ok_or("image prompt needs <|image_pad|>")?;
        let projector = PrismVisionProjector::load(&args[4])?;
        let image = projector.encode_image(&args[5], 4, 4)?;
        if let Ok(path) = std::env::var("CAMELID_PROBE_EMBEDDINGS") {
            std::fs::write(path, serde_json::to_vec(&image.embeddings)?)?;
        }
        model.generate_vision_stopping_streaming(
            &tokenizer.encode(prefix, false, true)?,
            &image,
            &tokenizer.encode(suffix, false, true)?,
            max,
            &stop,
            &mut |_| {},
        )?
    } else {
        model.generate_stopping(&tokenizer.encode(&prompt, false, true)?, max, &stop)?
    };
    println!(
        "{}",
        serde_json::json!({"ids": ids, "text":tokenizer.decode(&ids, true)?, "seconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}
