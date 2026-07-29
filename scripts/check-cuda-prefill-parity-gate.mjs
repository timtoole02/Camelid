#!/usr/bin/env node
// CI guard for the CUDA-resident batched-prefill parity gate.
//
// CI runners have no GPU, so the token-identical parity itself runs locally
// (scripts/validate-cuda-prefill-row.sh, on a CUDA host). This guard runs in CI
// without a GPU and fails if the optimization or its parity gate is silently removed
// or weakened — the failure mode the brief calls out: "the earlier regression (a
// CPU-optimization commit silently broke the GPU path) recurs unless the GPU parity
// gate runs automatically." It cannot prove parity; it proves the parity machinery is
// still wired so a human/GPU run will catch a real divergence.
//
// Pairs with the compile gate already in the `rust` job: `cargo test --all-targets
// --all-features` builds the `cuda` path and the #[ignore]d CUDA tests on every push
// (they self-skip without a device), so API/refactor drift fails CI at compile time.

import { readFileSync } from 'node:fs'

const checks = [
  {
    file: 'src/cuda_resident.rs',
    label: 'batched-prefill kernels present',
    needs: [
      // The batched prefill entry, the shared layer stack, and its scratch.
      'pub fn prefill_batched',
      'fn run_batched_layer_stack',
      'fn ensure_verify_scratch',
      // The shared stack MUST keep per-head QK-norm (the original Qwen3 GPU defect).
      'launch_rms_norm_per_head',
      // Batched prefill MUST fall back to serial for offloaded models (the batched
      // stack reads VRAM slices directly and has no offload streaming — 8B would read
      // placeholder bytes otherwise).
      'fn is_offloaded',
      'fn supports_batched_layer_stack',
      '!self.is_offloaded()',
      'pub fn supports_batched_prefill',
      'if !self.supports_batched_prefill() {',
    ],
  },
  {
    file: 'src/cuda_resident.rs',
    label: 'verify_batch reuses the shared stack (single source of truth)',
    // verify_batch must call the shared helper, not carry its own divergent copy. The trailing
    // `false` is the SIROCCO Phase P M1 flash_ok flag: verify MUST pass false so it stays on the
    // bit-identical attention_batched path (flash prefill is opt-in, prefill-only, token-parity).
    needs: ['self.run_batched_layer_stack(&mut sc, &s, base_position, k, scale, false)'],
  },
  {
    file: 'src/inference.rs',
    label: 'server routes GPU prefill through the batched path',
    needs: [
      'prefill_batched(&embeddings.data',
      // The serial path stays as an A/B escape hatch for parity bisection.
      'CAMELID_CUDA_RESIDENT_PREFILL_BATCHED',
    ],
  },
  {
    file: 'src/cuda_resident/tests.rs',
    label: 'batched-prefill parity test present and asserts token-identity',
    needs: [
      'fn prefill_then_decode_matches_sequential',
      '.prefill_batched(',
      'batched prefill+decode produced a different token than sequential forwards',
      // The speculative batched path keeps its own equivalence test.
      'fn verify_batch_matches_sequential',
    ],
  },
]

let failed = false
for (const { file, label, needs } of checks) {
  let src
  try {
    src = readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
  } catch (e) {
    console.error(`FAIL [${file}] ${label}: cannot read file (${e.message})`)
    failed = true
    continue
  }
  for (const token of needs) {
    if (!src.includes(token)) {
      console.error(`FAIL [${file}] ${label}: missing required marker:\n        ${token}`)
      failed = true
    }
  }
}

if (failed) {
  console.error(
    '\nThe CUDA batched-prefill optimization or its parity gate was removed or changed.\n' +
      'If this is intentional, update scripts/check-cuda-prefill-parity-gate.mjs AND re-run the\n' +
      'local GPU parity gate (scripts/validate-cuda-prefill-row.sh) before promoting any number.',
  )
  process.exit(1)
}
console.log('CUDA batched-prefill parity gate wiring intact.')
