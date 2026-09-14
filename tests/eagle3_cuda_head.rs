#![cfg(feature = "cuda")]
use camelid::{eagle3::Eagle3DraftModel, eagle3_cuda::CudaEagle3Head};
use std::path::PathBuf;

fn row(n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        (0..7680)
            .map(|i| (((i + n * 7) as f64 * 0.013).sin() as f32) * 0.1)
            .collect(),
        (0..2560)
            .map(|i| (((i + n * 11) as f64 * 0.017).cos() as f32) * 0.09)
            .collect(),
    )
}

#[test]
#[ignore = "requires pinned EAGLE checkpoint, CUDA, and independent NumPy reference"]
fn cuda_learned_head_matches_numpy_and_resets_cache() {
    let path =
        PathBuf::from(std::env::var_os("CAMELID_EAGLE3_MODEL").expect("set CAMELID_EAGLE3_MODEL"));
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("CAMELID_EAGLE3_REFERENCE").expect("set CAMELID_EAGLE3_REFERENCE"),
        )
        .unwrap(),
    )
    .unwrap();
    let model = Eagle3DraftModel::load(&path).unwrap();
    let mut head = CudaEagle3Head::new(&model, 4).unwrap();
    let mut max_error = 0f64;
    for n in 0..2 {
        let (f, e) = row(n);
        head.append(&f, &e, true).unwrap();
        let logits = head.logits().unwrap();
        let expected = reference["rows"][n]["logits"].as_array().unwrap();
        assert_eq!(logits.len(), expected.len());
        for (a, b) in logits.iter().zip(expected) {
            let b = b.as_f64().unwrap();
            let error = (*a as f64 - b).abs();
            max_error = max_error.max(error);
            assert!(
                a.is_finite() && error <= 0.005 + 0.001 * b.abs(),
                "row {n}: {a} versus {b}"
            );
        }
        assert_eq!(
            u64::from(head.next_token().unwrap()),
            reference["rows"][n]["target_token"].as_u64().unwrap()
        );
    }
    let stable_logits = head.logits().unwrap();
    head.reset();
    assert!(head.next_token().is_err());
    let (f, e) = row(0);
    head.append(&f, &e, false).unwrap();
    assert!(head.next_token().is_err());
    let (f, e) = row(1);
    head.append(&f, &e, true).unwrap();
    assert_eq!(
        u64::from(head.next_token().unwrap()),
        reference["rows"][1]["target_token"].as_u64().unwrap()
    );
    assert_eq!(head.filled(), 2);
    assert_eq!(head.logits().unwrap(), stable_logits);
    head.append(&f, &e, false).unwrap();
    head.append(&f, &e, false).unwrap();
    assert!(head.append(&f, &e, true).is_err());
    assert_eq!(head.filled(), 4);
    eprintln!("full learned head versus independent NumPy: 64000 logits, max absolute error {max_error}; target mappings and reset/KV-only catch-up agree");
}
