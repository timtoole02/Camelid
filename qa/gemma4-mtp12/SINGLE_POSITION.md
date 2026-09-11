# Fixed-position assistant draft chains

`CAMELID_GEMMA4_MTP12_SINGLE_POSITION=1` makes every assistant query in a
resident draft chain use the round's initial RoPE position and sliding-window
crop. The previous advancing-position behavior remains the default control.
The token embedding and recurrent hidden still advance between draft steps;
the target decoder and verifier are unchanged.

## Source discrepancy

The official HF implementation calls this
[`SinglePositionMultiTokenCandidateGenerator`](https://github.com/huggingface/transformers/blob/0c92811846/src/transformers/generation/candidate_generator.py#L1230).
Its class documentation explicitly specifies a constant position for the entire
draft round. In the same pinned source, [line 1373](https://github.com/huggingface/transformers/blob/0c92811846/src/transformers/generation/candidate_generator.py#L1373)
constructs `position_ids` before the loop, and
[line 1388](https://github.com/huggingface/transformers/blob/0c92811846/src/transformers/generation/candidate_generator.py#L1388)
passes the unchanged tensor on every iteration. Token IDs and projected hidden
states change inside that loop. This contract also holds in Transformers
`f62dc9bf2c90353b442a56e74391fbb8c689b55e`, including the unified 12B assistant.

The correct round anchor is the existing `proposal_position=P`: after the
target samples its bonus token, `input_ids` has length `P+1` but only `P`
positions have been forwarded into the shared target KV. HF uses
`input_ids.shape[1]-1=P`. The bug was adding the draft step to that anchor,
not the initial anchor itself.

The assistant's mask is created from the fixed shared-KV prefix with no
assistant cache. Thus its local crop also stays fixed during the chain. The
existing crop width and all attention arithmetic are preserved by this switch.
Diagnostic query-dump positions continue to describe absolute draft sequence
positions, rather than RoPE anchors. The KV-read ledger follows the selected
crop behavior.

## Validation and measurement

The original advancing-position K1-through-K15 test remains. New tests compare
every token and every recurrent f32 bit against repeated K1 calls at the same
position, for both a short prefix and a 1,031-row prefix beyond the sliding
window. They also poison all unverified physical KV slots. A CPU test covers
the frozen position/crop contract around the window boundary.

Local Metal validation passed: all three chain-oracle variants, the CPU
position/crop assertion, and both fixed-position chain variants with
`CAMELID_GEMMA4_MTP12_ATTN_V2=1`.

Run focused local tests with:

```sh
cargo test --lib single_position -- --nocapture --test-threads=1
cargo test --lib resident_chain_k1_through_k15_matches_repeated_device_k1_bit_for_bit -- --nocapture --test-threads=1
```

The synthetic parity tests do not establish HF end-to-end numerical parity or
throughput. Measure the full mini2 prompt suite with the switch off and on,
preserving the same release flags and all other selectors. Verify exact target
token IDs and compare draft acceptance, rounds, and generated tokens/second.
