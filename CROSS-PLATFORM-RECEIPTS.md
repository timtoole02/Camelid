# Cross-platform receipt procedure

External-contributor evidence for Camelid on platforms outside the recorded
validation set. Written so the same procedure runs unmodified on Linux + CUDA
and on Apple Silicon + Metal, and so the resulting bundles can be laid side by
side.

## Why these hosts

`COMPATIBILITY.md` records the CUDA parity work on **Windows x86_64, RTX 3060
Laptop (6 GB), driver 576.83, CUDA 12.9**, and states the limit plainly:

> Results are **specific to the recorded GPU / driver / CUDA version** - f32
> reduction order is GPU-specific, so other GPUs/drivers/CUDA versions are not
> covered by these bundles.

The hosts below differ from that record on every axis at once, which is the
point. None of them is covered by an existing bundle.

| | Recorded | Host A | Host B | Host C |
|---|---|---|---|---|
| OS | Windows | Linux (Gentoo) | macOS | macOS |
| Arch | x86_64 | x86_64 | arm64 | arm64 |
| Accelerator | RTX 3060 Laptop | RTX 3050 Ti Laptop | Apple Silicon | Apple Silicon |
| Accelerator memory | 6 GB discrete | **4 GB discrete** | **8 GB unified** | **8 GB unified** |
| Driver / runtime | 576.83 / CUDA 12.9 | 610.57.04 / CUDA 13 | Metal | Metal |

Host A is the tightest CUDA memory case: the recorded 4B row was *fully
VRAM-resident* on 6 GB, so on 4 GB the same file must take the offload path.

Hosts B and C are the architectural contrast. Discrete VRAM is a hard wall;
unified memory is not. A 4.0 GB model that cannot stay resident on a 4 GB
discrete card can plausibly stay resident in an 8 GB unified pool, on a weaker
GPU. That is a testable claim, and `plan-offload` answers it without loading a
single weight.

## Two kinds of deliverable

These hosts serve two distinct purposes, and conflating them would overstate
both. Every bundle below is labelled as one or the other.

### A. Validation: does his recorded work hold elsewhere?

The CUDA parity rows are recorded on Windows / RTX 3060 6 GB / driver 576.83 /
CUDA 12.9, with the scope limited to that exact combination. Re-running the same
rows on Linux / RTX 3050 Ti 4 GB / driver 610.57.04 / CUDA 13 either extends the
claim to a second platform or finds a divergence. Both outcomes are useful, and
neither is available without hardware he does not have.

This is confirmatory work. It adds a row to his matrix; it does not open a new
question.

### B. New angles: questions the recorded set cannot answer

Three of these, each turning on a hardware difference rather than an opinion.

**B1. The discrete offload boundary at 4 GB.**
His 4B Q8_0 row was *fully VRAM-resident* on 6 GB. On 4 GB the planner splits it
17 resident / 19 streamed (measured, layer map
`VVVVVVVVVVVVVVVVVHHHHHHHHHHHHHHHHHHH`). The 6 GB record never exercises the
split for this row, so the 4 GB host is the only one that tests it.

**B2. Unified versus discrete at the same model size.**
The 4B needs 3682 MiB of layer weights plus 1152 MiB of KV at context 4096.
A 4 GB discrete card cannot hold that and must stream. An 8 GB unified pool
should hold all of it. If the Macs plan full residency where the CUDA host plans
a 17/19 split, that is unified memory beating discrete at identical model size,
demonstrated with his own planner rather than asserted.

**B3. Bandwidth isolated as a single variable.**
This is the angle the two Macs make possible and nothing else does. They have
the **same 8 GB unified capacity and different memory bandwidth**. Capacity,
architecture, OS, and engine are held constant; only bandwidth moves.

That matters because the offload path is bandwidth-bound by construction: when
19 of 36 layers stream every forward pass, the limit is how fast weights arrive,
not how fast the GPU computes. Three regimes are available across these hosts:

| Host | Path for non-resident weights | Nominal bandwidth |
|---|---|---|
| Linux + RTX 3050 Ti | host RAM over PCIe Gen 4 x8 | ~16 GB/s link, DDR4-3200 behind it |
| Mac, lower-bandwidth | unified pool, no bus hop | lower |
| Mac, higher-bandwidth | unified pool, no bus hop | higher |

A same-capacity pair separated only by bandwidth is a clean natural experiment,
and it is not reproducible on a single machine.

## The hypothesis

> For a model larger than the discrete card's VRAM but smaller than the Mac's
> usable unified pool, the Apple Silicon host keeps it resident while the CUDA
> host must split layers to host RAM.

This is falsifiable and cheap to test. `plan-offload` prints the planned split
per host and loads no weights, so it costs seconds. If both hosts plan a
full-resident layout, the hypothesis is wrong and B2 collapses.

Measured on Host A already: 17 resident / 19 streamed, so the CUDA half holds.
The Mac half is untested until the bundles come back.

A second hypothesis follows for B3, and only the Mac pair can test it:

> Between two hosts with identical unified capacity, the higher-bandwidth one
> shows a larger advantage on a model that must stream than on one that stays
> resident.

If both Macs hold the 4B fully resident, streaming never happens and the two
should differ only modestly. The 8B row (8.7 GB Q8_0) would then be the file
that forces streaming on an 8 GB pool and exposes the bandwidth difference.

## What the steps prove, and what they do not

These scope limits are Camelid's own. They are repeated here rather than
widened, because an unstated limit reads as a claim.

| Step | Proves | Does not prove |
|---|---|---|
| `inspect` | The file parses and its tensors are described on this host | Anything about execution |
| `plan-offload` | The planned VRAM/host layer split for this memory budget | That the plan executes; no weights are loaded |
| `runnable-smoke` | Admission, load, greedy forward sanity, coherence. **Deterministic execution** | **Parity.** The receipt says so itself |
| `verify` | One request, one exact file, digest-sealed | Broad support for the model or family |
| `agent-eval` | A clean tool-call round trip, or INCONCLUSIVE | That INCONCLUSIVE is a failure; a contended host yields it legitimately |

No bundle produced by this procedure promotes a support claim or extends an
existing one. It is a platform data point.

## Procedure

Identical on every host.

### 1. Checkout and build

```bash
git clone https://github.com/timtoole02/Camelid
cd Camelid
cargo build --release            # CUDA is enabled by default on x86_64 Linux
git rev-parse --short HEAD       # record this; it goes in the bundle name
```

On Apple Silicon the Metal lane is the default GPU path; no feature flag is
needed. On x86_64 Linux `build.rs` enables the `cuda` cfg automatically.

### 2. Pull the same three models on every host

Sizes chosen to straddle the 4 GB discrete boundary:

```bash
./target/release/camelid pull qwen3_0_6b_instruct_q8_0   # 0.6 GB, resident everywhere
./target/release/camelid pull qwen3_1_7b_instruct_q8_0   # 1.8 GB, resident everywhere
./target/release/camelid pull qwen3_4b_instruct_q8_0     # 4.0 GB, the crossover case
```

The 4B row is the one that matters. Everything else is a control.

Verify the files match across hosts before comparing anything:

```bash
sha256sum models/*.gguf      # Linux
shasum -a 256 models/*.gguf  # macOS
```

Different bytes make the comparison meaningless. The harness records these
hashes in the manifest so a mismatch is detectable after the fact.

### 3. Run the harness

```bash
./camelid-receipts.sh
```

Options:

- `--quick` skips `verify` and `agent-eval`, the two slow lanes. Useful for a
  first pass or a memory-constrained host.
- `--label TEXT` overrides the auto-generated bundle label.
- `--models-dir`, `--out`, `--binary` for non-default layouts.

The harness detects OS, CPU, core counts, RAM, and accelerator itself. Every
platform-specific call (`sha256sum` vs `shasum`, `nproc` vs `sysctl`,
`nvidia-smi` vs `system_profiler`) is behind a helper, so nothing needs editing
between hosts.

Expect roughly 10 to 40 minutes depending on host and whether `--quick` is set.
The 4B row on a 4 GB discrete card is the slowest step by a wide margin.

### 4. What you get

```
evidence-out/<label>-<UTC>-head-<commit>/
  manifest.json    host, scope, privacy, per-model per-step exit codes and timings
  raw/             verbatim stdout and stderr of every step
  SHA256SUMS       tamper evidence over everything above
```

The label is auto-derived and encodes the regime, for example
`linux-x86_64-cuda-4096mb-...` or `apple-silicon-metal-8gb-unified-...`, so
bundle names remain self-describing once separated from their host.

## Comparing bundles

Read these three fields first. They determine whether a comparison is valid at
all:

1. `models[].sha256` - identical across hosts, or stop
2. `git_head` - identical, or you are comparing two versions of the engine
3. `host.accelerator.memory_kind` - `discrete` vs `unified`, the axis under test

Then compare `raw/*.plan-offload.*` between hosts. The forced-budget runs
(`--budget-mb 4096` and `8192`) exist so both hosts can be evaluated at the
*same* nominal budget, separating the planner's behaviour from the hardware's.

For execution results, compare `runnable-smoke` greedy output. Divergence there
between a CUDA host and a Metal host is expected and documented upstream: f32
reduction order is GPU-specific, and near-ties can flip. A divergence is a data
point about reduction order, not automatically a bug. `verify` is the stricter
lane and is where a real defect would surface.

## Privacy

The manifest carries a `privacy` field and the harness honours it: only file
basenames and hashes appear. Operator paths, hostnames, addresses, ports, and
key material are omitted. Check `raw/` before publishing if your host prints
paths in an error message.

## Known limitations of this procedure

- **No throughput claim.** Timings in the manifest are wall-clock for the whole
  step, including model load, on an unpinned machine. They are not benchmarks
  and must not be quoted as tokens/sec.
- **Host A is thermally constrained.** It is a laptop with a measured sustained
  package power limit. Under a combined CPU+GPU load its sustained clocks are
  materially lower than a desktop's. Any timing comparison against a desktop
  host is invalid without recording that limit.
- **`agent-eval` is load-sensitive.** It returns INCONCLUSIVE rather than FAIL
  on a contended box, by design. Run it on an otherwise idle host or accept the
  INCONCLUSIVE.
- **One run is one run.** Nothing here is repeated for variance. A single
  divergence should be re-run before it is reported as a finding.
