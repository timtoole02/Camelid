# Exact-Q4 EAGLE-3 warm-start training on Apple Silicon

This directory is a real training path for Camelid's existing 243,019,776-parameter SW512
EAGLE-3 head. It does not load the 3B target into the MLX process and it does not memorize a
prepared Pitch response. The teacher signal is captured by Camelid's production Q4 target and
the trainer learns from a broad corpus of target-authoritative continuations.

The default objective is SpecForge's official seven-depth soft-target cross entropy after
renormalizing the exact Q4 teacher logits over the checkpoint's fixed 32K `d2t` rows. It is KL
distillation up to the teacher-entropy constant. `--objective soft_kl` subtracts that constant and
has identical gradients when a true KL scalar is desired. `target_argmax` is used only for mapped
top-1 accuracy. Hard-label cross entropy remains the explicit `--objective hard_ce` ablation; it
is not the default.

## Pinned serving contract

The loader and exporter fail closed on Camelid's 15-tensor ABI:

- 13 trainable BF16 serving tensors: `fc`, one Llama decoder cell, `norm`, and the 32K
  `lm_head`.
- The original `d2t` offset vector and exact `t2d` membership mask are immutable.
- Hidden size 3072, intermediate size 8192, 24 query heads, 8 KV heads, head dimension 128,
  one layer, target vocabulary 128256, draft vocabulary 32000.
- The source `config.json` is copied byte-for-byte. RoPE and SW512 attention semantics cannot
  drift during export.

The MLX parameter tree uses the serving names directly. Export converts trained parameters to
BF16, reattaches the original mapping tensors, validates dense Safetensors coverage, reloads the
mapping, and records all SHA-256 values.

## Training alignment

The required positional contract is named
`eagle3-aux-p-next-token-teacher-p1-v1`. Exporter base row `P` contains target auxiliary state
from capture row `P`, the embedding of token `P+1`, and the target distribution captured after
row `P+1` (the distribution predicting token `P+2`). Thus at TTT step `j`, backbone row `P` uses
base embedding, teacher distribution and loss mask at `P+j`, while recurrence begins from
`aux[P]`. The loader rejects stores without this versioned contract.

The trainer implements the same recurrence as the official EAGLE-3 SDPA path:

1. Fuse pre-layer residual taps `[2,14,25]` with `fc.weight`.
2. At depth zero, use ordinary causal attention over the first EAGLE K/V stream.
3. At later depths, attend to that causal stream plus one same-position recurrent K/V entry per
   earlier depth.
4. Shift the already-left-shifted embedding, teacher and loss-mask rows once per depth.
5. Apply SpecForge's depth weights `0.8 ** d`.

The common retained prefix has `T - ttt_length` rows. Dropping only the padded tail is exactly
equivalent for those causal rows and avoids wasting unified memory on invalid logits.

## Feature schema

The reader consumes `camelid-eagle3-q4-features-v1` from Camelid's
`export-eagle3-features` command. Each sample is hash-verified and contains:

```text
input_ids.u32le                  [T]
labels.u32le                     [T]
target_argmax.u32le              [T]
loss_mask.u8                     [T]
aux_layer_inputs.bf16le          [T,9216]
hidden_state.bf16le              [T,3072]
input_embedding.bf16le           [T,3072]
next_token_embedding.bf16le      [T,3072]
teacher_draft_logits.bf16le      [T,32000]
teacher_logsumexp.f32le          [T]          # optional for KL training
```

`next_token_embedding[P]` must be bit-identical to `input_embedding[P+1]`. The trainer verifies
that relationship and `labels[P] == input_ids[P+2]`. `teacher_draft_logits[P]` is gathered from
the exact target capture at `P+1`, in the warm checkpoint's immutable `d2t` order. Both manifest
and sample pin the root mapping payload SHA-256; the loader then compares every decoded target ID
against the warm checkpoint rather than assuming two different serialization hashes are equal.
`UINT32_MAX` bootstrap/final sentinels are never treated as labels. The target embedding table is
not part of the 13 trainable tensors:
all exported embeddings are immutable inputs, so embedding weights are frozen by construction.

The required files occupy 100,877 bytes per token: 51.65 MB (49.26 MiB) for one 512-row sample,
plus four bytes per token when the exporter includes full-vocabulary log-sum-exp. That is about
413 MB for eight samples. The 32K teacher file stays memory-mapped; only supervised
rows are BF16-decoded. `--loss-row-chunk 16` limits each teacher/draft logit surface to 2.05 MB
before MLX graph overhead. FP32 parameters, gradients and two Adam moments have a rough 3.89 GB
upper-bound before activations and allocator workspaces, which is why the smoke run starts with
short sequences and serialized shards on a 16 GB machine.

## Environment

Use a dedicated environment on the Apple-Silicon training host. The scaffold targets the current
documented MLX API and pins it for reproducibility:

```bash
python3.12 -m venv .venv-eagle3-mlx
source .venv-eagle3-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mlx==0.32.2' 'numpy==2.3.2'
```

No package was installed while developing this branch. The current host did not already contain
MLX, so the device training step still needs its first controlled smoke run. The format, mapping,
shift, checksum, BF16 and independent NumPy KL parity tests run without MLX. One additional MLX
versus NumPy parity test automatically runs when MLX is present and is skipped otherwise.

## One-shard smoke test

Build the feature-export branch first, then serialize target capture under the Mini2 lock. The
input JSONL records are already tokenized with the Llama 3.2 Instruct tokenizer and carry an
assistant-only `loss_mask`.

```bash
CAM_SESSION_PID=$$ /Users/timtoole/bin/cam-lock.sh env \
  CAMELID_METAL=1 \
  CAMELID_METAL_RESIDENT_DECODE=1 \
  /path/to/camelid export-eagle3-features \
  /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --eagle3 /path/to/Llama-3.2-3B-Instruct-Eagle3-ShareGPT-SW512 \
  --input /path/to/train-shard-0000.jsonl \
  --output /path/to/spool/train-shard-0000 \
  --limit 8
```

Validate the checkpoint, every payload hash, the fixed mapping, alignment, and mapping coverage
before allocating the MLX model:

```bash
PYTHONPATH=. python -m tools.eagle3_mlx.validate \
  --warm-start /path/to/Llama-3.2-3B-Instruct-Eagle3-ShareGPT-SW512 \
  --features /path/to/spool/train-shard-0000 \
  --max-length 512 \
  --ttt-length 7
```

Then run a two-update smoke with a separately captured, locked evaluation store:

```bash
PYTHONPATH=. python -m tools.eagle3_mlx.train \
  --warm-start /path/to/Llama-3.2-3B-Instruct-Eagle3-ShareGPT-SW512 \
  --features /path/to/spool/train-shard-0000 \
  --eval-features /path/to/eval-features-locked \
  --output-dir /path/to/exports/smoke-0000 \
  --state-output /path/to/states/state-0000 \
  --max-steps 2 \
  --max-length 256 \
  --ttt-length 7 \
  --objective soft_ce \
  --loss-row-chunk 16 \
  --learning-rate 1e-5 \
  --parameter-dtype float32
```

The run prints JSONL memory, loss, gradient-norm and per-depth evaluation metrics. It emits both:

- a strict 15-tensor BF16 serving checkpoint in `--output-dir`;
- FP32/BF16 training parameters plus Adam state in the new `--state-output` directory.

## Serialized streaming without a 25 GB feature lake

Mini2 can alternate target export and MLX training without keeping both models resident:

1. Exit the exporter after writing one small shard.
2. Train that shard and write a new state directory.
3. Validate the serving export and `state.json` hashes.
4. Delete that exact consumed feature-shard directory.
5. Export the next shard only after the MLX process exits.
6. Resume from the previous state, writing a different state/output directory.

Example second invocation:

```bash
PYTHONPATH=. python -m tools.eagle3_mlx.train \
  --warm-start /path/to/Llama-3.2-3B-Instruct-Eagle3-ShareGPT-SW512 \
  --resume-state /path/to/states/state-0000 \
  --features /path/to/spool/train-shard-0001 \
  --eval-features /path/to/eval-features-locked \
  --output-dir /path/to/exports/shard-0001 \
  --state-output /path/to/states/state-0001 \
  --max-steps 8 \
  --max-length 512 \
  --ttt-length 7 \
  --objective soft_ce \
  --loss-row-chunk 16 \
  --learning-rate 1e-5 \
  --parameter-dtype float32
```

State/output paths must be new and empty. That refusal is intentional: a partial or stale state
must never be mistaken for a resume point. After `state-0001` validates, the older state and the
consumed feature shard can be removed explicitly. This keeps storage bounded to one small feature
shard, one retained evaluation store, and roughly one current model/Adam state instead of tens of
gigabytes of target activations.

## Corpus and gates

Train on target-generated, non-Pitch data. A credible pilot is 2,000-5,000 examples streamed in
small shards, with a source/project/template-family split and a locked 100-300-example evaluation
suite. Suggested mix:

- 45% unseen technical requirements, architecture plans, APIs, Bluetooth, mobile, testing and
  systems work;
- 25% Swift/Rust/Python/code instructions;
- 30% general conversational retention.

Exclude the Pitch prompt, its repository URLs, distinctive requirement list, response, and close
paraphrases. Hash the split before target capture. The exact Pitch prompt remains a final canary,
not a model-selection metric.

The MLX pilot gate is at least 10% relative mapped top-1 improvement with no material general/code
regression. That metric is not a throughput claim. The decisive Camelid gates are:

- ordered output token equality against target-only decode;
- no increase in EAGLE round latency;
- roughly 6.5 emitted tokens/round on X7, which means about 80% offered-node acceptance on held-out
  technical prose at the measured JSON/X7 round cost;
- repeated Pitch throughput above 100 tok/s, plus a diverse prose suite that rules out leakage.

Run the runtime gate with the existing pinned `bench-eagle3` harness and preserve the raw receipt,
binary SHA, head SHA, mapping SHA and prompt SHA. A training-loss win alone is not a win.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tools/eagle3_mlx/tests -v
python -m py_compile tools/eagle3_mlx/*.py
```

The synthetic suite checks the exact 15-key geometry, delta-coded mapping inversion, dense
Safetensors layout, payload hashes, BF16 RNE conversion, feature alignment, bootstrap masking and
the seven-depth shift contract.

Architecture references (audited against SpecForge commit
`c439546983863facd8126f505c2d291d0ab31faf`):

- SpecForge EAGLE-3 model/training path: <https://github.com/sgl-project/SpecForge>
- Official EAGLE implementation: <https://github.com/SafeAILab/EAGLE>
- MLX optimizer and serialization API: <https://ml-explore.github.io/mlx/>
