# Fleet receipts: Camelid v0.6.1 on six machines

External platform validation of [Camelid](https://github.com/timtoole02/Camelid)
run across a six-machine heterogeneous fleet, 2026-08-22 to 2026-08-24.

Every claim below is tied to a dated record and a procedure a stranger can run.
Claims are **evidence-gated**: anything not listed here with a receipt is not
claimed. Boundaries, meaning what is explicitly *not* claimed, are part of each
entry.

Two provenance grades are used and marked on every entry:

- **Reproducible.** You can run the committed procedure yourself against the
  public repo and get the same class of result.
- **Attested.** Verified on dated hardware with sealed local run records, whose
  digests are listed; the procedure to re-derive it is included.

Raw evidence is under `fleet-evidence-clean/`: 19 platform bundles plus 18
run-to-run repeat bundles under `repeats/`, each self-verifying through its own
`SHA256SUMS`. Two superseded bundles are retained under `superseded/` with the
reason they were replaced.

Every bundle passes Camelid's own `scripts/audit-evidence-bundle-privacy.mjs
--strict` with `finding_count: 0`.

**Two publication-time transforms, both documented in place.** Bundles were
passed through a redaction pass that replaces operator paths in `raw/` streams,
and 20 oversized `inspect` artifacts were excerpted, because Camelid's `inspect`
dumps the full tokenizer vocabulary (7.1 MB per model row) which is reproducible
from the model file and carries no evidentiary value. Each excerpt states the
original byte count, line count, and sha256 of the untruncated artifact, and the
command to regenerate it. Both transforms change file contents, so every
`SHA256SUMS` was recomputed afterwards and each bundle is internally consistent
as published. The unredacted originals remain on the machines that produced
them.

Entries are numbered `F-NNN` (fleet receipt), independent of pxx's `R-NNN`
series, which this work cross-references but does not extend.

---

## Fleet

| Host | OS | CPU | ISA | Accelerator | Accel memory |
|---|---|---|---|---|---|
| Dell XPS 15 9510 | Gentoo 6.18.41 | i7-11800H, 16T, AVX-512 | x86_64 | RTX 3050 Ti | 4096 MB discrete |
| ASRock B550 | Gentoo 6.18.21 | Ryzen 9 5950X, 32T, AVX2 | x86_64 | RTX 5060 Ti, **Blackwell** | 16311 MB discrete |
| Dell Precision T5810 | Gentoo 6.18.16 | Xeon E5-2699 v4, 44T, AVX2 | x86_64 | 2x RTX A4500 | 20470 MB discrete |
| Intel NUC11 | Gentoo | i5-1135G7, 8T, AVX-512 | x86_64 | none (Iris Xe, no CUDA) | n/a |
| Mac Mini M4 | macOS | Apple M4, 10 cores, NEON | arm64 | Metal | 16384 MB unified |
| MacBook Neo | macOS | Apple A18 Pro, 6 cores, NEON | arm64 | Metal | 8192 MB unified |

For contrast, every CUDA bundle in Camelid's `COMPATIBILITY.md` is Windows
x86_64, RTX 3060 Laptop 6 GB, driver 576.83, CUDA 12.9. This fleet differs on
OS, GPU generation, VRAM, driver, CPU vendor, and instruction set.

All hosts built from source at `db283b69` (`v0.6.1-191-gdb283b69`), except
MacBook Neo which built `0.6.1` from a shallow clone of the same tag. Model
files were verified identical by sha256 across hosts.

---

## F-001: the x86 Q8_0 lane is bit-identical across AVX-512 and AVX2 kernels; the ARM lane differs from it, as predicted

**Prior art, and what is actually new.** Camelid already owns most of this
question. `qa/determinism/determinism-baseline-20260614T063455Z.md` establishes
on an Apple M4 that thread count does not change output ("`--threads 1` vs
default (10 threads) is byte-identical. This is the decisive test"), explains
the mechanism (rayon partitions the output space, so no cross-thread float
combine exists), and states the caveat directly: "the exact logit values are
machine/ISA-specific (i8mm here on M4; an M1 runner without i8mm takes a
different-but-internally-deterministic kernel). The portable invariant is
run-to-run / thread-count byte-identity."

This receipt does not claim that finding. It contributes three things to it:

1. **Replication and extension of the thread-count invariant** from one M4 at
   1 versus 10 threads to six hosts across four x86 microarchitectures and two
   Apple silicon generations, up to 44 workers.
2. **First measurement of the ISA divergence** the baseline predicts but does
   not measure, with the divergence point located exactly.
3. **A result the caveat model does not predict**: two hosts taking the
   **AVX-512** kernel and two taking **AVX2** produce byte-identical output.

Item 3 is the reason this receipt exists.

**Claim.** With greedy decode fixed (temperature 0, top_p 1, seed 0), the
`Qwen3-0.6B-Q8_0` CPU lane is byte-identical across the AVX-512 and AVX2 kernel
paths, and differs from the NEON path by one token. Thread count changes nothing
anywhere.

**Grade.** Attested (2026-08-22, six hosts; n=3 repeats added 2026-08-24 on
three of them) and Reproducible (`cpu-thread-determinism.sh`).

**Environment.** `Qwen3-0.6B-Q8_0.gguf`, sha256 verified identical on every
host, 48 tokens, `CAMELID_LAZY_Q8_0_LINEAR=1`, `v0.6.1-191-gdb283b69`.

**Run-to-run repeats.** The initial sweeps were one per configuration. Every
host was then re-run three times, following the `-r2`/`-r3` convention in
Camelid's own measurement bundles, so run-to-run stability sits on top of
thread-count stability. Each repeat spawns a fresh process per thread count, so
these are independent processes, not in-process iterations:

| Host | Engine-reported lane | Total runs | Unique digests |
|---|---|---|---|
| XPS 9510 | AVX-512 | 15 | **1** (`d5b946bd...`) |
| NUC11 | AVX-512 | 12 | **1** (`d5b946bd...`) |
| ASRock B550 | AVX2 | 9 | **1** (`d5b946bd...`) |
| Precision T5810 | AVX2 | 9 | **1** (`d5b946bd...`) |
| Mac Mini M4 | NEON | 9 | **1** (`15593cfb...`) |
| MacBook Neo | NEON | 12 | **1** (`15593cfb...`) |

**66 independent runs, two digests, and the split falls exactly on instruction
set.** 45 x86_64 runs across four microarchitectures and both SIMD lanes produced
one digest; 21 ARM64 runs across two Apple generations produced another. This
replicates the baseline's fresh-process finding on x86 as well as ARM, and gives
the AVX-512 lane (27 runs) more coverage than the AVX2 lane (18), so the claim in
item 3 does not rest on the thinner side of its own comparison.

**Observed.** The SIMD lane column is Camelid's own capability reporting, taken
verbatim from each run's serve stderr, not inferred from the CPU model:

| Host | CPU | Engine-reported lane | Threads swept | Digest |
|---|---|---|---|---|
| XPS 9510 | i7-11800H | `AVX-512 (avx512f=true)` | 1, 2, 4, 8, 16 | `d5b946bd...` |
| NUC11 | i5-1135G7 | `AVX-512 (avx512f=true)` | 1, 2, 4, 8 | `d5b946bd...` |
| Precision T5810 | Xeon E5-2699 v4 | `AVX2 (avx512f=false)` | 1, 2, 4, 8, 16, 22, 44 | `d5b946bd...` |
| ASRock B550 | Ryzen 9 5950X | `AVX2 (avx512f=false)` | 1, 2, 4, 8, 16, 32 | `d5b946bd...` |
| Mac Mini M4 | Apple M4 | `NEON` | 1, 2, 4, 8, 10 | `15593cfb...` |
| MacBook Neo | Apple A18 Pro | `NEON` | 1, 2, 4, 6 | `15593cfb...` |

**The AVX-512 versus AVX2 result.** The engine reports two distinct x86 kernel
lanes across these four hosts, and `src/` carries genuinely separate
implementations selected at runtime (`is_x86_feature_detected!("avx512f")`,
`avx512bw`, `avx512vnni`, `avx512vnni_dpwssd`). Under the baseline's model, hosts
on different kernels would be expected to produce different-but-internally-
deterministic logits, exactly as an M1 without i8mm is expected to differ from an
M4 with it. They do not differ here. Two CPU vendors and four microarchitectures
spanning Broadwell-EP to Zen 3 to Tiger Lake, on two different SIMD kernels,
produce the same bytes.

**The ARM divergence.** Both Apple generations agree with each other and differ
from x86 by a single token. The sequences are identical for 21 tokens and then
split at one decision:

```
x86_64 : ... because 1 is not considered a prime
arm64  : ... because it is not a prime number.
```

Both answers are correct. This is a near-tie resolving differently because f32
reduction order differs between the AVX and NEON paths, after which the
sequences naturally diverge. It confirms the baseline's stated caveat with a
measurement and a located divergence point.

**Why item 3 matters.** If AVX-512 and AVX2 are genuinely bit-identical for this
row, then an x86 CPU parity reference is portable across the x86 fleet and does
not need a SIMD level pinned alongside it, which is a stronger and more useful
guarantee than the baseline currently claims. If instead the AVX-512 kernel is
not being selected for this particular row despite being detected, that is worth
knowing for a different reason. This receipt cannot distinguish those two cases
from the outside; it records that the outputs match and that the engine reports
different lanes. Resolving which is true is a question for someone who can
instrument the kernel selection.

**Boundary, explicitly not claimed.** One model, one prompt, one build, greedy
only, one run per configuration. Not a parity test against llama.cpp and no
parity claim; this tests self-consistency of Camelid's own CPU lane. No claim
that the AVX-512 kernel was exercised, only that the engine reported the
capability and that outputs matched. No claim about i8mm-versus-not on ARM, which
the baseline raises and this fleet cannot test, since both Apple hosts are recent
enough to have it. Longer generations, other quantizations, and other model
families are untested. Neither output is claimed more correct than the other.

**Reproduce.**

```bash
./cpu-thread-determinism.sh --model models/Qwen3-0.6B-Q8_0.gguf \
  --threads "1 2 4 8 16" --tokens 48
```

---

## F-002: pxx receipt R-007 finding 3 is resolved, shown as a controlled before and after on the original platform

**Claim.** pxx `R-007` recorded against Camelid **v0.4.4** that the OpenAI
`tools` surface accepted the parameter but never executed: `tool_calls` always
null, and the model's Qwen-native `<tool_call>` block returned verbatim as
`content`. On **v0.6.1** the same surface returns a structured `tool_calls`
array with `finish_reason: "tool_calls"` and empty content.

**Grade.** Attested (2026-08-22, five hosts including a same-machine version
pair) and Reproducible (`r007-retest.sh`).

**Environment.** `Qwen3-1.7B-Q8_0.gguf`. The decisive pair ran on one Mac Mini
M4, same model file, same prompt, same day, with only the binary differing:
`v0.6.1-191-gdb283b69` built from source, against the preserved `v0.4.4`
release binary that R-007 was originally filed against.

**Observed.** The controlled pair:

| Binary | Platform | F3 tools surface |
|---|---|---|
| v0.4.4 | macOS arm64 | `UNCHANGED_raw_marker_in_content` |
| v0.6.1 | macOS arm64 | `FIXED_tool_calls_populated` |

The defect reproduces on the original platform and is then resolved on the same
machine. Independently confirmed on four more hosts, all `FIXED`:

| Host | Accelerator |
|---|---|
| XPS 9510 | RTX 3050 Ti, Ampere |
| ASRock B550 | RTX 5060 Ti, Blackwell |
| Precision T5810 | RTX A4500, Ampere |
| MacBook Neo | Apple A18 Pro, Metal |

Payload on v0.6.1:

```json
{
  "finish_reason": "tool_calls",
  "content": "",
  "tool_calls": [{
    "id": "call_927383653e48448c955a8c2d6353dab0",
    "type": "function",
    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
  }]
}
```

Correct OpenAI semantics: structured call, parsed arguments, empty content,
correct `finish_reason`.

**Boundary, explicitly not claimed.** One model, one request per host. R-007
finding 1 was a Metal panic at `metal_resident.rs:27:41`; serve survived a plain
completion on every host here, but this run does not claim that specific panic
path is fixed, only that the observable lane no longer fails. R-007 finding 2
was a throughput measurement and is not re-measured; no throughput claim is made
anywhere in this document.

**A trap worth publishing.** An earlier run of this probe sent `"model": "local"`
and received `model_not_found`, which looks like a lane failure and is a client
mistake. The harness now reads the served id from `/v1/models`. Anyone
re-testing should check for that before reporting a regression.

---

## F-003: Camelid builds and runs on Blackwell (compute capability 12.0)

**Claim.** Camelid builds from source and runs its standard lanes on an NVIDIA
Blackwell GPU, which `COMPATIBILITY.md` records as
`Phase 5 (Blackwell) BLOCKED-HW`.

**Grade.** Attested (2026-08-22, ASRock B550).

**Observed.**

```
[hw] GPU: NVIDIA GeForce RTX 5060 Ti (x1) | compute 12.0 | tensor-cores yes
     | VRAM 15.3 GiB free / 15.5 GiB total
```

Detected correctly, tensor cores recognised, full step battery completed on all
three model rows with outcomes identical to the Ampere hosts.

**Boundary, explicitly not claimed.** `BLOCKED-HW` refers to the **NVFP4 lane**.
NVFP4 was not exercised. This receipt says the engine builds, detects sm_120, and
runs its standard lanes there. It does not promote any support row and does not
speak to NVFP4.

---

## F-004: Offload planning is exercised at three residency regimes, including two the existing evidence base does not reach

**Claim.** The `Qwen3-4B-Q8_0` layer split was measured at three distinct VRAM
conditions, bracketing the single 6 GB datapoint in `COMPATIBILITY.md` from both
sides.

**Grade.** Attested (2026-08-22).

**Observed.**

| Condition | Free VRAM | 36-layer split |
|---|---|---|
| RTX 3050 Ti | 3.6 GiB | 17 VRAM / 19 host |
| RTX 5060 Ti, contended | 2.6 GiB | 0 resident, planner refused |
| RTX 5060 Ti, free | 15.3 GiB | **36 / 36 fully resident** |

Fully resident, on the free Blackwell:

```
[offload] 36 layers: 36 resident in VRAM, 0 offloaded to host
[offload] budget (MiB): free 15684 = KV 1152 + ends 394 + safety 256
                        + scratch 0 + resident-layers 3682 (of total 3682)
[offload] layer map (V=VRAM, H=host): VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
```

Partial split, on 4 GB:

```
[offload] 36 layers: 17 resident in VRAM, 19 offloaded to host
[offload] layer map (V=VRAM, H=host):
          VVVVVVVVVVVVVVVVVHHHHHHHHHHHHHHHHHHH
```

The 6 GB record in `COMPATIBILITY.md` never exercises the split for this row.
The 4 GB case tests the boundary a 6 GB card does not, and the free 16 GB case
tests full residency.

**Observation on the contended run.** Freeing 13 GB of VRAM on the Blackwell
host changed the plan from 0 resident to 36 resident, and changed
`runnable-smoke` wall clock by nothing measurable (258 s contended, 255 s free).
That is recorded as an observation, not a performance claim; see the throughput
boundary below.

**Boundary, explicitly not claimed.** No throughput claim. The contended figure
was taken while the host served production traffic and is recorded to describe
the condition, not to compare.

---

## F-005: Refusals are fail-closed, typed, and platform-aware

**Claim.** Where Camelid cannot run a configuration it refuses with a typed
error naming the limit, the measured value, and a documented remedy, rather than
entering an unvalidated lane. Two distinct refusal classes were observed, each
correct for its platform.

**Grade.** Attested (2026-08-22, six hosts).

**Observed, refusal class 1: CPU weight materialization.** `agent-eval` on the
4B Q8_0:

```
model error: unsupported tensor type: estimated CPU weight materialization/
retention is 9959225344 bytes, above safety limit 6442450944 bytes; dense Q8_0
linears retain expanded in-memory blocks by default ... set
CAMELID_LAZY_Q8_0_LINEAR=1 only if deliberately accepting the slower
file-backed Q8 path, or raise CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES
deliberately for a controlled run
```

The outcome is not uniform, and the pattern is the interesting part:

| Host | Accel memory | System RAM | 4B `runnable-smoke` | 4B `agent-eval` |
|---|---|---|---|---|
| XPS 9510 | 4 GB discrete | 31 GiB | passed, 338 s | refused |
| NUC11 | none | 15 GiB | passed, 465 s | refused |
| ASRock B550 | 16 GB discrete | 63 GiB | passed, 255 s | refused |
| Precision T5810 | 20 GB discrete | 252 GiB | passed, 1210 s | **passed**, 69 s |
| Mac Mini M4 | 16 GB unified | 16 GiB | passed, 211 s | **passed**, 75 s |
| MacBook Neo | 8 GB unified | 8 GiB | passed, 220 s | refused |

Two things separate cleanly here. `runnable-smoke` passed the 4B row on **every**
host in the fleet, including the 8 GB unified one, so the model loads and
generates everywhere. Only `agent-eval` refuses, and only on some hosts, which
places the refusal in the tool-capability harness's memory budget rather than in
the ability to run the model.

The two Apple hosts bracket the unified-memory boundary directly: same
architecture, same binary, same model file, 16 GiB passes and 8 GiB refuses.

Camelid sealed its own receipt for each refusal, for example
`qa/agent-eval/Qwen3-4B-Q8_0-1787387512-FAIL.json`.

**Observed, refusal class 2: offload planning on unified memory.**
`plan-offload --detected` exits 1 on every Metal host and on the CUDA-less
NUC11, and exits 0 on every CUDA host:

```
Error: no CUDA device, offloading is a no-op; the CPU backend already
holds all weights in system RAM
```

This is correct behaviour, not a defect. There is nothing to offload between on
a unified-memory or CPU-only host, and the engine says so rather than
fabricating a plan.

**Boundary, explicitly not claimed.** The 4B refusal is a **memory-budget**
refusal, not a statement about the model's tool capability. This receipt does not
explain why the T5810 and M4 pass where the ASRock does not; it records that the
outcome varies with host memory configuration and that every outcome was a clean
pass or a typed refusal, never a crash. Characterising the threshold would need
a designed experiment, which this was not.

---

## F-006: `verify` abstains on every row, and says so

**Claim.** `camelid verify` exited 2 on all three model rows on all six hosts.

**Grade.** Attested (2026-08-22, six hosts, 19 bundles).

**Observed.** Exit 2 uniformly. The subcommand documents this: "Verification
abstains when no exact-hash profile exists." These files have no committed
profile, so abstention is correct and uniform across every platform tested.

**Boundary, explicitly not claimed.** No verification claim is made for any file
in this evidence set. An abstention is not a pass.

---

## F-007: a distributed pipeline runs across x86_64+CUDA and ARM64+Metal, and on this prompt matches the homogeneous x86 result byte for byte

**Claim.** Camelid's `distribute-master` / `distribute-worker` layer-sharding
pipeline forms a working ring between an x86_64 Linux CUDA host and an ARM64
macOS Metal host, and for one 47-token greedy completion produced output
byte-identical to the same split run entirely on x86_64.

**Grade.** Attested (2026-08-24, three hosts).

**Prior art.** `qa/distributed/` records two topologies: `two-mac-tinyllama-q8`
(Mac plus Mac) and `hetero-mac-pi-tinyllama-q8` (Mac plus Raspberry Pi). Both
are ARM plus ARM. No x86_64 node appears in either, and no CUDA plus Metal pair.
Notably the existing hetero receipt records
`"generated_token_ids_match": false` with
`"first_divergent_generated_token_index": 25`, so divergence in a heterogeneous
ring is already documented as possible.

**Design.** Run as a controlled pair rather than a single observation. Same
master, same model file (sha256 identical on all three hosts), same 0..14 /
14..28 split of 28 layers, same prompt, same 48-token greedy budget. The only
variable is the worker's ISA.

| Run | Master | Worker | Result |
|---|---|---|---|
| Control | ASRock B550, x86_64, CUDA Blackwell | Precision T5810, x86_64 | 47 tokens, 9.51 tok/s, exit 0 |
| Variable | ASRock B550, x86_64, CUDA Blackwell | Mac Mini M4, arm64, Metal | 47 tokens, 13.76 tok/s, exit 0 |

**Observed.** Both completions are byte-identical:

```
control (x86 + x86) sha256 ecedad706a5c0658bfabdfcc3e5605e1
hetero  (x86 + ARM) sha256 ecedad706a5c0658bfabdfcc3e5605e1
222 bytes each, cmp reports no difference
```

**Why this is not a bit-exactness claim.** F-001 in this same document shows
that single-node x86 and ARM **do** diverge for this model, at one near-tie token.
The two results are not in conflict. Divergence is near-tie dependent: it occurs
when two candidate tokens are close enough that a last-bit difference in f32
reduction order flips the argmax. F-001's prompt hits such a tie; these 47 tokens
apparently do not. The correct reading is "no divergence was observed on this
prompt", not "cross-ISA sharding is bit-exact". The existing Mac plus Pi receipt
diverging at token 25 is the same phenomenon landing the other way.

**Startup ordering matters, and cost real time.** Starting the worker first left
it unable to reach the master's feedback port, which is not yet listening. On
macOS that first attempt returned `EHOSTUNREACH` (os error 65) rather than a
connection refusal, and the worker did not recover from it, reporting
`Failed to connect ... after 600 seconds` and exiting. The same ordering works
between two Linux hosts. Starting the **master first** made the ring form
immediately. Network reachability was verified independently in both directions
(ping 0% loss, port open) and both hosts' firewalls were stopped, so this is a
retry-behaviour difference, not a network fault.

**Boundary, explicitly not claimed.** One prompt, one model, one layer split, one
run per configuration, greedy only. No bit-exactness claim for cross-ISA
sharding; see above. No throughput claim: the two runs used different worker
hardware, so 9.51 versus 13.76 tok/s compares two different machines and not two
ISAs. No claim that other splits, longer generations, or three-node rings behave
the same. No claim about the `EHOSTUNREACH` retry behaviour beyond what was
observed on this one macOS host; whether it is a defect worth fixing is a
maintainer call, and it is recorded in `FLEET-DEFECTS.md` as an observation
rather than a bug.

**Reproduce.**

```bash
# on the worker host, second half of the layers
camelid distribute-worker --layers "14..28" --addr 0.0.0.0:5005 \
  --master-addr <master>:5006 models/Qwen3-0.6B-Q8_0.gguf

# on the master host, first half. Start the MASTER first if either node is macOS.
camelid distribute-master --worker-addr <worker>:5005 --layers "0..14" \
  --addr 0.0.0.0:5006 --max-tokens 48 models/Qwen3-0.6B-Q8_0.gguf
```

---

## Method notes

### A harness defect found and fixed mid-run

Both probes originally waited for `GET /v1/models` to return HTTP 200 and treated
that as readiness. Camelid's serve answers 200 with an empty `{"data":[]}` while
the model loads, so on a host where loading outran the poll the probe fired at a
server with no model and every request returned `model_not_loaded`.

This produced two artifacts that would have been actively misleading if
aggregated. The determinism receipt reported `diverged_count: 0` while all seven
of its runs had `status: no_completion`, which reads as "determinism held" when
nothing was compared. The R-007 receipt reported `no_choices_in_response`, which
looks like a platform difference and was not.

Both are fixed at root. Readiness now means the model is loaded, and the served
model id is read from the same response so the two cannot disagree. The
determinism receipt carries `completed_runs` and an explicit `verdict`, and fewer
than two completed runs reports `inconclusive_insufficient_runs`, never a pass.
The affected bundles are retained under `fleet-evidence-clean/superseded/` with
this explanation, and their corrected replacements are named there.

### Thermal envelope

One host is a laptop and is thermally bounded. Every bundle now records PL1, PL2,
CPU governor, energy performance preference, and chassis type, plus an
operator-supplied cooling description, because external cooling reports nothing
to the host. The XPS ran throughout at PL1 35 W, PL2 60 W, governor and EPP
`performance`, on an external pad at its low setting of 1000 rpm, confirmed by
the operator and sustained overnight. That is the conservative end of that
machine's range.

### Privacy gate

Bundle manifests are public-safe by construction: OS, CPU, core counts, RAM,
accelerator name and driver, file hashes, and no hostname, address, port, or
process name. The `raw/` streams do contain absolute paths, so every bundle was
passed through `scrub-bundle.sh`, which applies a superset of Camelid's own
`scripts/check-public-scrub.sh` patterns and re-seals `SHA256SUMS` after
redaction. Twenty of twenty-one bundles required redaction; all twenty-one pass
the gate on a second pass. Redaction changes file contents, so published digests
differ from the original run by design, and the bundles are self-consistent.

---

## Boundaries across the whole document

- **No parity claim.** `runnable-smoke` attests deterministic execution, and its
  own receipt says so. Parity belongs to Camelid's ledger.
- **No throughput claim.** Timings are whole-step wall clock including model
  load, on unpinned machines, one thermally constrained and one that was serving
  production traffic during part of the run. They must not be quoted as
  throughput.
- **No support-contract change.** Nothing here promotes a model row or extends
  an existing claim. `COMPATIBILITY.md` is untouched by design; support is earned
  per exact row and that is the maintainer's call.
- **No NVFP4 or Blackwell lane claim.** See F-003.
- **One run per configuration.** No variance repeats. A single divergence would
  need re-running before counting as a finding.

---

## Reproducing this

Every step is scripted, and the scripts run unmodified on macOS and Linux.

```bash
git clone https://github.com/timtoole02/Camelid && cd Camelid
cargo build --release
./target/release/camelid pull qwen3_0_6b_instruct_q8_0
./target/release/camelid pull qwen3_1_7b_instruct_q8_0
./target/release/camelid pull qwen3_4b_instruct_q8_0

./camelid-receipts.sh                                   # platform bundle
./r007-retest.sh --model models/Qwen3-1.7B-Q8_0.gguf    # R-007 probes
./cpu-thread-determinism.sh --model models/Qwen3-0.6B-Q8_0.gguf
./scrub-bundle.sh evidence-out/<bundle>                 # privacy gate
```

For several hosts at once, `fleet-receipts.sh` pins git HEAD and model sha256
across them and skips any host that diverges, with the reason stated. It never
uses sudo and never builds.

Procedure and rationale: `CROSS-PLATFORM-RECEIPTS.md`.
