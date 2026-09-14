// CLI-only experiment. Source Q4 weights are retained, but numerical arithmetic
// differs from V4. Perfect references are measurements, never usable drafts.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_fp16_coding_probe(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    input_text: &str,
    prompt_text: &str,
    workload: &str,
    model_sha256: &str,
    max_tokens: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(prompt.len() + max_tokens <= 2048, "FP16 probe context cap is 2048");
    let widths: Vec<usize> = match std::env::var("CAMELID_BENCH_ORACLE_WIDTHS") {
        Ok(value) => value.split(',').map(|v| v.trim().parse::<usize>()).collect::<Result<_, _>>()?,
        Err(std::env::VarError::NotPresent) => vec![8, 16, 32, 64],
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(!widths.is_empty() && widths.iter().all(|v| [1,2,4,8,16,32,64].contains(v)),
        "FP16 probe widths must be selected from 1,2,4,8,16,32,64");
    anyhow::ensure!(widths.iter().collect::<std::collections::BTreeSet<_>>().len() == widths.len(),
        "FP16 probe widths must be unique");
    let session = LlamaInferenceSession::new(config.clone(), Arc::clone(weights))?;
    let mut probe = session.create_fp16_probe(2048)?
        .ok_or_else(|| anyhow::anyhow!("FP16 target probe not admitted"))?;
    let prefill = |probe: &mut camelid::metal::ResidentFp16Probe| -> anyhow::Result<u32> {
        probe.reset();
        let mut last = None;
        for chunk in prompt.chunks(64) {
            let predictions = session.forward_fp16_probe_tokens(probe, chunk)?
                .ok_or_else(|| anyhow::anyhow!("FP16 prompt left diagnostic GPU route"))?;
            anyhow::ensure!(predictions.len() == chunk.len(), "FP16 prompt row count mismatch");
            last = predictions.last().copied();
        }
        last.ok_or_else(|| anyhow::anyhow!("empty FP16 prompt"))
    };
    let prefill_started = Instant::now();
    let first = prefill(&mut probe)?;
    let reference_prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut reference = vec![first];
    let reference_started = Instant::now();
    while reference.len() < max_tokens && !tokenizer.special.eog.contains(reference.last().unwrap()) {
        let predictions = session.forward_fp16_probe_tokens(&mut probe, &[*reference.last().unwrap()])?
            .ok_or_else(|| anyhow::anyhow!("FP16 serial reference left diagnostic GPU route"))?;
        anyhow::ensure!(predictions.len() == 1, "FP16 serial row count mismatch");
        reference.push(predictions[0]);
    }
    let reference_decode_ms = reference_started.elapsed().as_secs_f64() * 1000.0;
    let max_width = *widths.iter().max().unwrap();
    let count = reference.len().saturating_sub(1) / max_width * max_width;
    anyhow::ensure!(count >= max_width, "FP16 reference too short for selected widths");
    eprintln!("[fp16-coding] reference_tokens={} text={:?}", reference.len(), tokenizer.decode(&reference, false)?);
    let mut copy_records = Vec::new();
    let mut proposal_sha256 = None;
    if let Some(path) = std::env::var_os("CAMELID_BENCH_FP16_PROPOSAL") {
        let proposal = std::fs::read_to_string(path)?;
        proposal_sha256 = Some(camelid::receipt::sha256_hex(proposal.as_bytes()));
        let rows = match std::env::var("CAMELID_BENCH_FP16_COPY_ROWS") {
            Ok(value) => value.parse::<usize>()?,
            Err(std::env::VarError::NotPresent) => 64,
            Err(error) => return Err(error.into()),
        };
        anyhow::ensure!([8,16,32,64].contains(&rows), "invalid causal copy width");
        for rep in 0..=3 {
            let attention_before = (probe.attention_batch_chunks(), probe.attention_batch_rows());
            let mm_before = probe.attention_mm_layers();
            let run = run_fp16_copy_probe(
                &session, &mut probe, tokenizer, prompt, &proposal, "rust", rows, max_tokens,
            )?;
            let exact = run.generated_ids == reference;
            eprintln!("[fp16-copy] rep={rep} exact={exact} stats={}", serde_json::to_string(&run.stats)?);
            copy_records.push(serde_json::json!({
                "rep": rep, "warmup": rep == 0, "exact_serial_reference": exact,
                "generated_token_ids": run.generated_ids, "stats": run.stats,
                "attention_batch_chunks_including_prefill": probe.attention_batch_chunks() - attention_before.0,
                "attention_batch_rows_including_prefill": probe.attention_batch_rows() - attention_before.1,
                "attention_mm_layers_including_prefill": probe.attention_mm_layers() - mm_before,
            }));
        }
    }
    let mut records = Vec::new();
    for rep in 0..=3 {
        let order: Vec<usize> = if rep % 2 == 0 { widths.iter().copied().rev().collect() } else { widths.clone() };
        for width in order {
            anyhow::ensure!(prefill(&mut probe)? == first, "FP16 prompt anchor nondeterministic");
            let before = probe.projection_count();
            let matrix_before = probe.actual_matrix_dispatch_count();
            let attention_before = (probe.attention_batch_chunks(), probe.attention_batch_rows());
            let mm_before = probe.attention_mm_layers();
            let mut round_ms = Vec::new();
            let mut mismatches = Vec::new();
            for offset in (1..=count).step_by(width) {
                let started = Instant::now();
                let predictions = session.forward_fp16_probe_tokens(
                    &mut probe, &reference[offset - 1..offset + width - 1],
                )?.ok_or_else(|| anyhow::anyhow!("FP16 width {width} left diagnostic GPU route"))?;
                round_ms.push(started.elapsed().as_secs_f64() * 1000.0);
                anyhow::ensure!(predictions.len() == width, "FP16 wide row count mismatch");
                for (j, (&actual, &expected)) in predictions.iter().zip(&reference[offset..offset + width]).enumerate() {
                    if actual != expected {
                        mismatches.push(serde_json::json!({"index": offset+j, "expected": expected, "actual": actual}));
                    }
                }
            }
            if rep > 0 {
                let total_ms: f64 = round_ms.iter().sum();
                let projections = probe.projection_count() - before;
                let matrices = probe.actual_matrix_dispatch_count() - matrix_before;
                anyhow::ensure!(projections == (197 * round_ms.len()) as u64, "FP16 route count mismatch");
                anyhow::ensure!(matrices == (if probe.gate_up_fused() {169} else {197}) * round_ms.len() as u64, "FP16 matrix dispatch count mismatch");
                eprintln!("[fp16-coding] rep={rep} width={width} rows={count} rate={:.2} mismatches={}",
                    count as f64 * 1000.0 / total_ms, mismatches.len());
                records.push(serde_json::json!({
                    "rep": rep, "width": width, "verified_rows": count,
                    "rounds": round_ms.len(), "round_ms": round_ms,
                    "total_verify_ms": total_ms, "mean_round_ms": total_ms / round_ms.len() as f64,
                    "verified_rows_per_second": count as f64 * 1000.0 / total_ms,
                    "exact_reference_match": mismatches.is_empty(), "mismatches": mismatches,
                    "logical_projection_count": projections, "projection_dispatches": matrices, "cpu_fallbacks": 0,
                    "attention_batch_chunks": probe.attention_batch_chunks() - attention_before.0,
                    "attention_batch_rows": probe.attention_batch_rows() - attention_before.1,
                    "attention_mm_layers": probe.attention_mm_layers() - mm_before,
                }));
            }
        }
    }
    let all_exact = records.iter().all(|r| r["exact_reference_match"] == true)
        && copy_records.iter().all(|r| r["exact_serial_reference"] == true);
    let binary = std::env::current_exe()?;
    println!("{}", serde_json::json!({
        "schema": if copy_records.is_empty() { "camelid.coding-fp16-perfect-draft-cost.v1" }
            else { "camelid.coding-fp16-causal-and-oracle.v1" },
        "measurement_only": true,
        "contains_actual_causal_generation": !copy_records.is_empty(),
        "oracle_is_achieved_generation_throughput": false,
        "arithmetic": "canonical-Q4K/Q6K-to-half; half activations; persistent f32 MMA; independent f16 KV",
        "projection_backend": probe.projection_backend(), "attention_chunk": probe.attention_chunk(),
        "gate_up_fused": probe.gate_up_fused(),
        "attention_backend": probe.attention_backend(),
        "attention_mm_peak_scratch_bytes": probe.attention_mm_peak_scratch_bytes(),
        "attention_mm_peak_physical_rows": probe.attention_mm_peak_physical_rows(),
        "same_as_v4_target": false, "all_exact_serial_reference": all_exact,
        "scope": "records measure perfect-reference teacher forcing; causal_copy_records measure source-only proposal generation; prefill/setup/reference generation reported separately",
        "commit": benchmark_commit(),
        "binary_sha256": camelid::receipt::sha256_file_hex(&binary).map_err(anyhow::Error::msg)?,
        "model_sha256": model_sha256, "workload": workload,
        "input_sha256": camelid::receipt::sha256_hex(input_text.as_bytes()),
        "prompt_sha256": camelid::receipt::sha256_hex(prompt_text.as_bytes()),
        "prompt_token_ids": prompt, "plain_token_ids": reference,
        "prompt_tokens": prompt.len(), "verified_rows_per_arm": count,
        "reference_prefill_ms": reference_prefill_ms, "reference_decode_ms": reference_decode_ms,
        "mirror_bytes": probe.mirror_bytes(), "setup_ms": probe.setup_ms(),
        "effective_env": speculative_effective_env(), "records": records,
        "causal_copy_records": copy_records, "proposal_sha256": proposal_sha256,
        "target_text": tokenizer.decode(&reference, false)?,
    }));
    Ok(())
}
