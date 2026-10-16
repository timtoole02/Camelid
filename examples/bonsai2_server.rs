//! Small library-server harness for Bonsai GUI/API qualification.
//! Uses the production API and frontend. Run with an isolated models/data directory.
//! cargo run --release --example bonsai2_server -- MODELS_DIR [127.0.0.1:28182]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let models = args.get(1).ok_or("MODELS_DIR required")?;
    let addr = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("127.0.0.1:28182")
        .parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    camelid::api::serve(
        addr,
        Some(8),
        None,
        false,
        false,
        Some(models.into()),
        Default::default(),
    )
    .await?;
    Ok(())
}
