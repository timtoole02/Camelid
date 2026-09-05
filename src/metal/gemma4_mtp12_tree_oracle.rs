//! Explicit local-model gate: compare device tree forwards with repeated K1
//! and the frozen original primary/alternate continuation captures.
use super::*;

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn u32_at(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap())
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn snapshot_view<'a>(
    key: &'a Buffer,
    value: &'a Buffer,
    layer: usize,
    heads: usize,
    dim: usize,
    capacity: usize,
) -> Gemma4Mtp12DeviceKv<'a> {
    let view = |buffer: &'a Buffer| Gemma4Mtp12MetalBufferView {
        buffer,
        byte_offset: 0,
        byte_len: buffer.length(),
    };
    Gemma4Mtp12DeviceKv {
        key: view(key),
        value: view(value),
        source_layer: layer,
        kv_heads: heads,
        head_dim: dim,
        max_positions: capacity,
    }
}

#[test]
#[ignore = "explicit local model, snapshot and captured oracle paths required"]
fn tree_snapshot_primary_and_fork_bits() {
    // These frozen captures and alternate-selection assertions qualify only
    // the original threshold. An explicit experiment must use a new oracle.
    assert_eq!(max_margin_from_env().unwrap().to_bits(), DEFAULT_MAX_MARGIN.to_bits(),
        "captured tree replay requires the default max margin 2");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("CAMELID_MTP12_TREE_ORACLE_CONFIG").unwrap()).unwrap(),
    )
    .unwrap();
    let path = |key: &str| PathBuf::from(config[key].as_str().unwrap());
    let snapshot_path = path("snapshot");
    let directory = snapshot_path.parent().unwrap();
    let snapshot: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&snapshot_path).unwrap()).unwrap();
    let baseline: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path("baseline_response")).unwrap()).unwrap();
    assert_eq!(snapshot["format"], "camelid-mtp12-final-kv-v1");
    assert_eq!(
        snapshot["generated_token_ids"],
        baseline["camelid"]["generated_token_ids"]
    );
    assert_eq!(snapshot["target_sha256"], GEMMA4_12B_QAT_Q4_0_TARGET_SHA256);
    assert_eq!(
        snapshot["assistant_sha256"],
        GEMMA4_12B_MTP_ASSISTANT_SHA256
    );
    let single = snapshot["environment"]["CAMELID_GEMMA4_MTP12_SINGLE_POSITION"] == "1";
    let prefix = snapshot["committed_prefix"].as_u64().unwrap() as usize;
    let prompt_len = snapshot["prompt_token_ids"].as_array().unwrap().len();
    let golden: Vec<u32> = serde_json::from_value(snapshot["generated_token_ids"].clone()).unwrap();
    let mapping: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path("mapping")).unwrap()).unwrap();
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path("continuation_oracle")).unwrap()).unwrap();
    assert_eq!(
        original["primary_queries_bit_exact"].as_u64().unwrap() as usize * 4112,
        std::fs::metadata(path("queries")).unwrap().len() as usize
    );
    let queries = std::fs::read(path("queries")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&queries)),
        mapping["dump_sha256"].as_str().unwrap()
    );
    let mut seed_path = path("queries").into_os_string();
    seed_path.push(".seeds");
    let seeds = std::fs::read(PathBuf::from(seed_path)).unwrap();
    let rounds = mapping["rounds"].as_array().unwrap();
    assert_eq!(seeds.len(), rounds.len() * (16 + TARGET_HIDDEN * 4));
    let device = Device::system_default().unwrap();
    let capacity = prefix + 32;
    let load = |layer: usize, kind: &str, heads: usize, dim: usize| {
        let filename = format!("layer{layer}-{kind}.f32le");
        let raw = std::fs::read(directory.join(&filename)).unwrap();
        let entry = snapshot["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["file"] == filename)
            .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            entry["sha256"].as_str().unwrap()
        );
        assert_eq!(raw.len(), heads * prefix * dim * 4);
        let packed = floats(&raw);
        let mut padded = vec![12345.0; heads * capacity * dim];
        for h in 0..heads {
            padded[h * capacity * dim..h * capacity * dim + prefix * dim]
                .copy_from_slice(&packed[h * prefix * dim..(h + 1) * prefix * dim]);
        }
        f32_buffer(&device, &padded).unwrap()
    };
    let sk = load(46, "key", 8, 256);
    let sv = load(46, "value", 8, 256);
    let fk = load(47, "key", 1, 512);
    let fv = load(47, "value", 1, 512);
    let sliding = snapshot_view(&sk, &sv, 46, 8, 256, capacity);
    let full = snapshot_view(&fk, &fv, 47, 1, 512, capacity);
    let target = path("target");
    let digest = std::process::Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .arg(&target)
        .output()
        .unwrap();
    assert!(digest.status.success());
    assert_eq!(
        String::from_utf8(digest.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        GEMMA4_12B_QAT_Q4_0_TARGET_SHA256
    );
    let metadata = crate::gguf::read_metadata(&target).unwrap();
    let embedding = metadata
        .tensors
        .iter()
        .find(|v| v.name == "token_embd.weight")
        .unwrap();
    assert_eq!(embedding.dimensions, [TARGET_HIDDEN as u64, VOCAB as u64]);
    assert_eq!(embedding.tensor_type, crate::gguf::GgufTensorType::Q6K);
    let mapping_target = GgufWireMmap::map(&target).unwrap();
    let raw = mapping_target
        .bytes(
            embedding.absolute_offset,
            GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES as usize,
        )
        .unwrap();
    let table_buffer = device.new_buffer_with_data(
        raw.as_ptr().cast(),
        raw.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    drop(mapping_target);
    let table = Gemma4Mtp12Q6KEmbeddingTable {
        wire: Gemma4Mtp12MetalBufferView {
            buffer: &table_buffer,
            byte_offset: 0,
            byte_len: table_buffer.length(),
        },
        hidden: TARGET_HIDDEN,
        vocab: VOCAB,
        target_model_sha256: GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
    };
    let mut assistant = Gemma4Mtp12AssistantMetal::load(&path("assistant")).unwrap();
    assistant.single_position = single;
    assert!(assistant.dense_bf16.is_none());
    let shortlist = assistant.shortlist.is_some();
    let invoke = |assistant: &mut Gemma4Mtp12AssistantMetal,
                  token: u32,
                  hidden: &[f32],
                  p: usize,
                  step: usize| {
        let mut answer = assistant
            .propose_k1_device_at_position(
                Gemma4Mtp12Q6KEmbeddingRow {
                    wire: Gemma4Mtp12MetalBufferView {
                        buffer: &table_buffer,
                        byte_offset: u64::from(token) * GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES,
                        byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES,
                    },
                    hidden: TARGET_HIDDEN,
                },
                hidden,
                sliding,
                full,
                p,
                if single { p } else { p + step },
            )
            .unwrap();
        if shortlist {
            // K1 deliberately exposes the full tied head. Reapply the selected
            // production shortlist to that same query to obtain its independent oracle.
            let command = assistant.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            assistant.encode_draft_head(encoder);
            assistant.encode_vocab_argmax(encoder, 0);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            answer.token = unsafe { *assistant.scratch.output_token.contents().cast::<u32>() };
            read_buffer_f32(&assistant.scratch.logits, &mut answer.logits).unwrap();
        }
        answer
    };
    let mut records = Vec::new();
    for (ri, round) in rounds.iter().enumerate() {
        let k = round["draft_k"].as_u64().unwrap() as usize;
        if k != 7 {
            continue;
        }
        let p = round["position"].as_u64().unwrap() as usize;
        let first = round["first_query_index"].as_u64().unwrap() as usize;
        let seed = &seeds[ri * (16 + TARGET_HIDDEN * 4)..(ri + 1) * (16 + TARGET_HIDDEN * 4)];
        assert_eq!(
            [u32_at(seed, 4), u32_at(seed, 8), u32_at(seed, 12)],
            [p as u32, p as u32, 7]
        );
        let anchor = u32_at(seed, 0);
        assert_eq!(anchor, golden[p - prompt_len]);
        let initial = floats(&seed[16..]);
        let tree = assistant
            .propose_tree_w8_from_cpu_hidden(anchor, &initial, table, sliding, full, p, p)
            .unwrap();
        let mut tree_queries = vec![0.0; tree.assistant_steps * ASSISTANT_HIDDEN];
        let mut tree_hidden = vec![0.0; tree.assistant_steps * TARGET_HIDDEN];
        read_buffer_f32(&assistant.scratch.chain_final_normalized, &mut tree_queries).unwrap();
        read_buffer_f32(&assistant.scratch.chain_recurrent_hidden, &mut tree_hidden).unwrap();
        let mut token = anchor;
        let mut hidden = initial;
        let mut parents = Vec::new();
        let mut alternatives = Vec::new();
        let mut margins = Vec::new();
        for step in 0..4 {
            let answer = invoke(&mut assistant, token, &hidden, p, step);
            assert_eq!(
                tree.tokens[step + 1],
                answer.token,
                "round{ri} primary{step}"
            );
            let mut query = vec![0.0; ASSISTANT_HIDDEN];
            read_buffer_f32(&assistant.scratch.final_normalized, &mut query).unwrap();
            assert_eq!(
                bits(&tree_queries[step * ASSISTANT_HIDDEN..(step + 1) * ASSISTANT_HIDDEN]),
                bits(&query)
            );
            assert_eq!(
                bits(&tree_hidden[step * TARGET_HIDDEN..(step + 1) * TARGET_HIDDEN]),
                bits(&answer.recurrent_hidden)
            );
            if !shortlist {
                let captured = &queries[(first + step) * 4112..(first + step + 1) * 4112];
                assert_eq!(answer.token, u32_at(captured, 0));
                assert_eq!(bits(&query), bits(&floats(&captured[16..])));
            }
            let mut second = usize::MAX;
            let mut second_value = f32::NEG_INFINITY;
            for (id, &value) in answer.logits.iter().enumerate() {
                if id == answer.token as usize || value.is_nan() {
                    continue;
                }
                if value > second_value || (value == second_value && id < second) {
                    second = id;
                    second_value = value;
                }
            }
            assert!(second < VOCAB);
            let margin = answer.logits[answer.token as usize] - second_value;
            assert_eq!(tree.primary_margins[step].to_bits(), margin.to_bits());
            alternatives.push(second as u32);
            margins.push(margin);
            parents.push(answer.recurrent_hidden.clone());
            token = answer.token;
            hidden = answer.recurrent_hidden;
        }
        let selected = margins
            .iter()
            .position(|v| v.is_finite() && (0.0..=2.0).contains(v));
        assert_eq!(tree.branch_primary_step, selected);
        if let Some(step) = selected {
            token = alternatives[step];
            hidden = parents[step].clone();
            assert_eq!(tree.tokens[5], token);
            for c in 0..2 {
                let answer = invoke(&mut assistant, token, &hidden, p, step + 1 + c);
                assert_eq!(
                    tree.tokens[6 + c],
                    answer.token,
                    "round{ri} branch{step} continuation{c}"
                );
                assert_eq!(
                    bits(&tree_hidden[(4 + c) * TARGET_HIDDEN..(5 + c) * TARGET_HIDDEN]),
                    bits(&answer.recurrent_hidden)
                );
                let mut query = vec![0.0; ASSISTANT_HIDDEN];
                read_buffer_f32(&assistant.scratch.final_normalized, &mut query).unwrap();
                assert_eq!(
                    bits(&tree_queries[(4 + c) * ASSISTANT_HIDDEN..(5 + c) * ASSISTANT_HIDDEN]),
                    bits(&query)
                );
                token = answer.token;
                hidden = answer.recurrent_hidden;
            }
            if !shortlist {
                let expected = &original["rounds"][ri]["branches"][step];
                assert_eq!(
                    tree.tokens[5] as u64,
                    expected["alternate_token"].as_u64().unwrap()
                );
                assert_eq!(
                    tree.tokens[6] as u64,
                    expected["continuation_tokens"][0].as_u64().unwrap()
                );
                assert_eq!(
                    tree.tokens[7] as u64,
                    expected["continuation_tokens"][1].as_u64().unwrap()
                );
            }
        } else {
            for step in 4..7 {
                let answer = invoke(&mut assistant, token, &hidden, p, step);
                assert_eq!(tree.tokens[step + 1], answer.token);
                assert_eq!(
                    bits(&tree_hidden[step * TARGET_HIDDEN..(step + 1) * TARGET_HIDDEN]),
                    bits(&answer.recurrent_hidden)
                );
                let mut query = vec![0.0; ASSISTANT_HIDDEN];
                read_buffer_f32(&assistant.scratch.final_normalized, &mut query).unwrap();
                assert_eq!(
                    bits(&tree_queries[step * ASSISTANT_HIDDEN..(step + 1) * ASSISTANT_HIDDEN]),
                    bits(&query)
                );
                if !shortlist {
                    assert_eq!(answer.token, u32_at(&queries[(first + step) * 4112..], 0));
                }
                token = answer.token;
                hidden = answer.recurrent_hidden;
            }
        }
        records.push(serde_json::json!({"round":ri,"position":p,"tokens":tree.tokens,"parents":tree.parents,
            "depths":tree.depths,"branch_primary_step":tree.branch_primary_step,"primary_margins":tree.primary_margins,
            "assistant_steps":tree.assistant_steps,"gpu_us":tree.timing.gpu_us,"wall_us":tree.timing.wall_us}));
        eprintln!(
            "tree oracle: {} rounds passed; shortlist={shortlist} single={single}",
            records.len()
        );
    }
    std::fs::write(path("output"),serde_json::to_vec_pretty(&serde_json::json!({
        "primary_and_fork_query_recurrence_bits_exact":true,"shortlist":shortlist,"single_position":single,"rounds":records,
        "timing_qualification":false,"note":"Original committed round starts; altered target-tree generation not measured."})).unwrap()).unwrap();
}
