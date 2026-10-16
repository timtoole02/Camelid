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
    let load_start = std::time::Instant::now();
    let model = RunnableModel::load(&args[1])?;
    let model_load_seconds = load_start.elapsed().as_secs_f64();
    let max = args[3].parse()?;
    let stop = tokenizer.special.eog.iter().copied().collect::<Vec<_>>();
    let mut start = std::time::Instant::now();
    let mut warmup_seconds = Vec::new();
    let mut token_times = Vec::new();
    let ids = if args.len() == 6 {
        let (prefix, suffix) = prompt
            .split_once("<|image_pad|>")
            .ok_or("image prompt needs <|image_pad|>")?;
        let projector = PrismVisionProjector::load(&args[4])?;
        let projector = if metadata.metadata.contains_key("prism.hadamard.version") {
            projector.with_portable_decode()
        } else {
            projector
        };
        let min = std::env::var("CAMELID_PROBE_IMAGE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let image_max = std::env::var("CAMELID_PROBE_IMAGE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(min);
        let image = projector.encode_image(&args[5], min, image_max)?;
        if let Ok(path) = std::env::var("CAMELID_PROBE_EMBEDDINGS") {
            std::fs::write(path, serde_json::to_vec(&image.embeddings)?)?;
        }
        model.generate_vision_stopping_streaming(
            &tokenizer.encode(prefix, false, true)?,
            &image,
            &tokenizer.encode(suffix, false, true)?,
            max,
            &stop,
            &mut |_| token_times.push(start.elapsed().as_secs_f64()),
        )?
    } else {
        let tokens = tokenizer.encode(&prompt, false, true)?;
        let warmups = std::env::var("CAMELID_PROBE_WARMUPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        for _ in 0..warmups {
            let warmup = std::time::Instant::now();
            model.generate_stopping_streaming(&tokens, 8, &stop, &mut |_| {})?;
            warmup_seconds.push(warmup.elapsed().as_secs_f64());
        }
        start = std::time::Instant::now();
        model.generate_stopping_streaming(&tokens, max, &stop, &mut |_| {
            token_times.push(start.elapsed().as_secs_f64())
        })?
    };
    let seconds = start.elapsed().as_secs_f64();
    let cuda_active = model.cuda_language_active();
    if std::env::var("CAMELID_PROBE_REQUIRE_CUDA").as_deref() == Ok("1") && !cuda_active {
        return Err(
            "CUDA language execution was required, but no CUDA graph was initialized".into(),
        );
    }
    println!(
        "{}",
        serde_json::json!({"ids": ids, "text":tokenizer.decode(&ids, true)?,
            "seconds":seconds, "token_seconds": token_times,
            "backend": if cuda_active { "cuda" } else { "cpu" },
            "model_load_seconds": model_load_seconds, "warmup_seconds": warmup_seconds})
    );
    Ok(())
}
