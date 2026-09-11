# W8 tree prototype — not throughput-qualified

Base is the selected mini2 V8 source `03a307b629616abc93b8193a70477c1a6af071dc`.
The default remains the existing linear lane. Set `CAMELID_GEMMA4_MTP12_TREE_W8=1`
only for experimental full logical/physical W8 rounds. Other widths fail closed;
normal short/padded budget tails stay linear. Legacy draft-query/seed capture is
incompatible; final committed KV snapshots alone remain supported.

The proposer uses anchor + four primary drafts + a top2 alternative + two
continuations, branching at the earliest of the four head margins <=2. If none
qualifies, it continues all seven linear drafts. Actual assistant evaluations
are six or seven, respectively. Target rows use physical slots independently
from semantic depths; attention sees only the committed prefix and the node's
logical ancestor path. The target's existing SPEC50 argmax remains authoritative.

Tree commits bind topology to the completed ticket, select the accepted physical
leaf's own raw hidden, stage K/V as uint in threadgroup memory, and compact the
selected path before cursor advancement. Prefix commit cannot consume a tree
ticket except for zero-row abort. Stop inputs are neither emitted nor committed.

## Current evidence

Integrated library and test compilation pass. CPU policy/state tests pass.
Source review covered paired HD256 context, scalar HD512 context, ragged bounds,
logical score layout, fork recurrence, stop/budget/capacity, failure and rollback.
Device top2 fixtures passed on the earlier unrestricted host before integration.

The task changed to restricted permissions during implementation. Metal device
creation now returns null; the two required raw GPU tests fail at device setup.
Do not interpret other tests' early-return behavior as GPU qualification. The
full-model, captured-assistant and mini2 end-to-end tree gates remain pending.
No tree throughput is claimed. The selected release's sample median remains
27.409857 decode tok/s for the 529-prompt/192-output-budget benchmark.

## Required gates after access restoration

Run one heavy job per host through `~/bin/cam-lock.sh`.
For native exact comparisons keep Rust1.94.1, release LTO off, codegen-units16,
build-jobs1. Do not substitute the default fat-LTO profile: it changed old target
IDs earlier in this optimization. Use all selected V8 selectors (W8,
fm_bits_u4/SG1, V2 attention2,3, exact MMA head, shortlist384, advancing position,
2048 positions, padded tail) and the exact official target/assistant artifacts.

1. Run raw `metal_gemma4_tree_attention_matches_independent_linear_paths` and
   `metal_gemma4_tree_compaction_stages_overlapping_sources_and_preserves_bits`.
2. Run ignored `tree_snapshot_primary_and_fork_bits` with the explicit frozen
   oracle configuration, both full head/shortlist384 and advancing/fixed modes.
3. Run ignored `target_tree_w8_model_paths_logits_kv_and_compaction_are_bit_exact`.
   Set `CAMELID_MTP12_TEST_MODEL`, `CAMELID_TREE_TEST_TRACE`, and the native
   sidecar selector. This gate compares160 nodes (raw hidden, all logits,
   all48-layer K/V), poisoned wrong paths, real compaction and the next K1 step.
4. Build the native server with the original release profile. First check
   tree OFF against archived old token IDs; then tree ON on the exact user
   request. Use native decode time, excluding prefill and the first free token.
   Never use the nested historical `native_receipt_qualification` rate.
5. If it wins, run interleaved old/candidate pairs, held-out prompts, output
   tails/stop/capacity and SSE text/count/finish checks before selecting it.

Trace fields preserve physical parents/depths, accepted path, branch decision,
assistant forward count and compaction wall. Aggregate tree counters are additive;
the existing assistant byte ledger is a nominal full-head estimate with shortlist.
