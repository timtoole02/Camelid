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
    label: 'server routes GPU prefill through transactional paged paths',
    needs: [
      // Phase 7 replaced the contiguous `_from` serving call with generation-safe
      // paged append. Keep both the single-sequence and cross-request routes wired.
      'slot.prefill_paged_append(',
      'slot.prefill_paged_batch_append(',
      'fn prefill_paged_append(',
      'fn prefill_paged_batch_append(',
      // The serial path stays as an A/B escape hatch for parity bisection.
      'CAMELID_CUDA_RESIDENT_PREFILL_BATCHED',
    ],
  },
  {
    file: 'src/cuda_resident.rs',
    label: 'prefix continuation stops vouching for KV written by any other route',
    // Continuation lets a prefill SKIP rows it believes already hold this prompt's
    // tokens. That belief is only as good as its invalidation: a stale record does
    // not fail loudly, it answers from the wrong KV. Every path that writes the KV
    // cache by a route other than a prefill must therefore drop the record, and a
    // reviewer adding the next such path needs to be told. These markers are the
    // structural half; `resident_prefix_len_is_bounded_by_the_record_and_the_watermark`
    // is the behavioural half.
    needs: [
      'fn invalidate_resident_tokens',
      // One call for each non-prefill KV writer: seed_layer (reseeds from f16-rounded
      // host history — NOT bit-identical to a fresh prefill, the exact hazard that
      // keeps the prompt-prefix cache off this lane), compact_tree_kv_path (relocates
      // rows), sparsify_kv (drops a layer's cache), reset_qwen35_state (zeroes it).
      // Adding a fifth writer without invalidating is the silent-wrong-output bug this
      // count exists to catch.
      { token: 'self.invalidate_resident_tokens();', atLeast: 4 },
      // A rewind must shorten the record; `set_filled` is the single choke point.
      'self.resident_tokens.truncate(filled)',
    ],
  },
  {
    file: 'src/inference.rs',
    label: 'every CPU KV-history reader materializes the GPU mirror on demand',
    // A CUDA-resident prefill no longer mirrors its KV back eagerly (except for
    // speculating sessions), so the CPU history exists only when someone asks for it.
    // Each reader must ask BEFORE touching `kv_cache`, or it silently attends over a
    // zero-filled prefix — degraded output with one stderr warning, not a failure.
    //
    // Four callers today: the three CPU forward readers
    // (forward_layer_range_from_hidden, forward_single_token_timed_internal, the verify
    // path) plus rollback_to_position, which gates on cpu_kv_authoritative().
    //
    // HONEST LIMIT: this pins the count, so DELETING a call fails the gate. It cannot
    // detect a NEW reader added without one — grep cannot know what reads the history.
    // If you are adding a CPU path that attends over `kv_cache`, call
    // `ensure_cpu_kv_materialized()` first and raise this number.
    needs: [
      { token: 'self.ensure_cpu_kv_materialized()', atLeast: 4 },
      // The safe default: sessions start eager, and only an audited caller opts out.
      'cpu_kv_mirror_eager: true',
      'fn set_cpu_kv_mirror_eager',
    ],
  },
  {
    file: 'src/api/mod.rs',
    label: 'only non-speculating requests opt into the lazy KV mirror',
    // Speculation reaches rollback_to_position, which the lazy recovery cannot always
    // satisfy. Widening this to unconditional lazy re-opens speculative_rollback_failed.
    needs: ['session.set_cpu_kv_mirror_eager(speculative.is_some())'],
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
  for (const need of needs) {
    // A marker is either a string that must appear, or `{ token, atLeast }` when the
    // point is that it appears at EVERY site it has to (presence alone would pass
    // with three of four call sites deleted).
    const token = typeof need === 'string' ? need : need.token
    const atLeast = typeof need === 'string' ? 1 : need.atLeast
    const count = src.split(token).length - 1
    if (count < atLeast) {
      console.error(
        `FAIL [${file}] ${label}: required marker appears ${count}x, need >=${atLeast}:\n        ${token}`,
      )
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
