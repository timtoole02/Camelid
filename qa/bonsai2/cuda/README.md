# Bonsai 2 Windows CUDA qualification

This directory keeps CUDA evidence separate from the earlier CPU qualification.
All four API/GUI and seeded-draft checks have passed, including fresh and
reloaded 27B GUI requests. Native and reference benchmarks are complete; see
[`comparison.json`](comparison.json) and the
[full twelve-case table](../../../docs/BONSAI2.md#cuda-measurements-and-qualification).
The final rebuilt suite passed all twelve CUDA tests and both host memory
regressions; [`cache-parity.json`](cache-parity.json) retains the results.
The completed [`manifest.json`](manifest.json) binds all qualified models,
projectors, binaries, and receipts after `check-cuda-evidence.mjs` passed.

The measured host is Windows x86_64, an Intel i7-11800H with 16 GB RAM, and an
RTX 3060 Laptop GPU with 6 GiB VRAM, driver 576.83. Run one model workload at a
time. Stop test servers before benchmarking, and keep builds separate from timed
inference. Apple Silicon qualification is separate.

## Artifacts and builds

Run commands from the repository root with Node.js, the frontend dependencies,
Rust, and CUDA available. Keep all four official PTQ1_0 language packs and their
matching Q8_0 projectors in `output/bonsai2-models`. Filenames and SHA-256 pins are
defined in `src/bonsai2.rs`; each benchmark hashes the actual language files.

```powershell
npm --prefix frontend run build
cargo build --release --features cuda --example bonsai2_probe --example bonsai2_server --jobs 2 --config 'profile.release.lto=false' --config 'profile.release.codegen-units=16'
$buildRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path (Get-Location) 'target' }
$probe = Join-Path $buildRoot 'release/examples/bonsai2_probe.exe'
$server = Join-Path $buildRoot 'release/examples/bonsai2_server.exe'
$models = (Resolve-Path 'output/bonsai2-models').Path
$draft = Join-Path $models 'Ternary-Bonsai-2-2B-PTQ1_0.gguf'
```

The reference is the official
[PrismML-Eng/llama.cpp prism-b10777-e3de715 release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715),
Windows CUDA 12.4 package. Its archive SHA-256 is
`fa5874a1412e251a6efb13f289e97bacbbb4e7fc6714c1836b4fc6c1dc863462`.
Extract the server and companion DLLs into
`output/bonsai2-reference/oracle-cuda12`. CUDA runtime DLLs must be available;
these receipts used the installed CUDA 12.9 runtime. The earlier `oracle`
directory contains a CPU-only binary and is unsuitable for this comparison.
The harness rejects a reference binary that exposes no CUDA device.

## Matched language benchmarks

Both engines receive the exact `count.chatml`, `code.chatml`, and
`explain.chatml` bytes, greedy sampling, 32 generated tokens, eight CPU threads,
2048 context slots, f16 KV, and an eight-token warmup. Three measured repetitions
are retained; the middle request duration supplies the published rate. The
primary metric is generated tokens divided by complete request wall time,
including uncached prefill. Target/drafter loading and warmup are excluded.
The native timer surrounds its Rust generation call after tokenization; the
reference timer surrounds a localhost HTTP completion request and JSON parsing,
so the port also includes transport, serialization, and tokenization overhead.
The native receipt fields `decode_tokens_per_second` and `decode_speedup` are
legacy callback-delivery diagnostics derived from token callback timestamps.
CUDA execution and speculative verification can deliver tokens in bursts, so
those fields do not measure the GPU decode clock and are not used for the
cross-engine comparison. Original observed timestamps are retained.

The proposal model is the same Bonsai 2 2B pack, with a three-token draft window.
The port uses `draft-simple`; the previous-generation DSpark assistants are not
qualified for Bonsai 2. See the publisher's
[speculative decoding notes](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md).

```powershell
$reference = 'output/bonsai2-reference/oracle-cuda12/llama-server.exe'
$env:CAMELID_REFERENCE_CUDA_BIN = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin'
$env:CAMELID_REFERENCE_RUNTIME_NOTE = 'CUDA 12.4 package with installed CUDA 12.9 runtime DLLs'
$env:CAMELID_REFERENCE_LOG_DIR = 'output/bonsai2-reference/cuda-logs'
$env:CAMELID_REFERENCE_SCORES = '0'
$env:CAMELID_BENCH_REPETITIONS = '3'
$env:CAMELID_SPEC_DRAFT_TOKENS = '3'
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    node qa/bonsai2/cuda-reference-benchmark.mjs $reference $target $draft "qa/bonsai2/cuda/${size}b-reference-benchmark.json" 32
    if ($LASTEXITCODE -ne 0) { throw "Reference ${size}B failed" }
}
$env:CAMELID_BENCH_CUDA = '1'
$env:CAMELID_BENCH_BUILD = 'release, cuda, lto=false, codegen-units=16, x86-64-v3'
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    $env:CAMELID_BENCH_REFERENCE = "qa/bonsai2/cuda/${size}b-reference-benchmark.json"
    node qa/bonsai2/spec-benchmark.mjs $probe $target $draft "qa/bonsai2/cuda/${size}b-native-benchmark.json" 32
    if ($LASTEXITCODE -ne 0) { throw "Native ${size}B failed" }
}
node qa/bonsai2/compare-cuda-benchmarks.mjs
```

Set `CAMELID_REFERENCE_RESUME=1` only when continuing an interrupted reference
receipt with the same settings/binaries. Completed timing rows are retained;
new startup logs get separate attempt filenames. Log-probability collection is
optional (`CAMELID_REFERENCE_SCORES=1`), untimed, and best restricted to a case
whose output diverged. It can be substantially slower than ordinary generation.
Raw logs stay in ignored `output`; public JSON includes sanitized offload
excerpts, binary/model hashes, and raw-log digests.

The [complete comparison](comparison.json) keeps all twelve prompt/model rows.
Native drafting improves 4B counting/code by 1.20–1.21x, 9B counting/code by
1.59x, and all 27B cases by 1.38–2.19x. It slows all 2B cases and 4B/9B prose;
4B prose falls from 13.62 to 6.17 tok/s. The port is faster on every measured
2B/4B/9B case. Camelid is 1.18–1.25x faster for ordinary 27B and 3.29–3.78x
faster for draft 27B on this host. Drafting is opt-in.

The first three target sizes fit all packed weights on CUDA. Native 27B streams
weights for 19/64 trailing layers in ordinary mode and 29/64 with drafting;
all layers still compute on CUDA. The port's automatic fit places 45/65 target
layers on CUDA for ordinary 27B and 31/65 with drafting, plus all 25/25 drafter
layers; its remaining target layers compute on CPU. Those are complete
operating configurations on a constrained GPU, with engine-specific layer
counts. Native capacity-plan excerpts are retained in each benchmark receipt.
This does not compare both engines with an entirely resident 27B target.

All twelve native ordinary/draft outputs match, and native ordinary IDs match
the port baseline at all 32 output tokens. All repetitions are stable within
mode. The port's 2B and 9B prose continuations differ between its ordinary and
draft modes; those observed differences remain in the receipts.

## API and GUI

Use a dedicated server on `127.0.0.1:28182`, with isolated data paths. Start it in
one terminal, and run the clients in another. Use ordinary decoding for this
first pass. Do not benchmark while the server is running.

```powershell
$data = Join-Path (Get-Location) 'output/bonsai2-reference/cuda-gui-data'
New-Item -ItemType Directory -Force -Path $data | Out-Null
$env:CAMELID_MODELS_DIR = $models
$env:CAMELID_DATA_DIR = $data
$env:CAMELID_RESPONSES_DB = Join-Path $data 'responses.sqlite'
$env:CAMELID_WORKSPACE_MEMORY_DB = Join-Path $data 'workspace.sqlite'
$env:CAMELID_CUDA_DEVICE = '0'
$env:CAMELID_QWEN35_CUDA = '1'
$env:CAMELID_QWEN35_CUDA_MAXPOS = '8192'
$env:CAMELID_SPEC_DECODE = $null
& $server $models '127.0.0.1:28182'
```

Use the newly started ordinary server for the fresh 27B GUI run first. The
following 2B/4B/9B/27B sequence then checks switching away from and reloading 27B.
API/GUI qualification allocates 8192 context slots; its semantic recall check is
1044 prompt tokens. This is separate from the benchmark's 2048-slot allocation.

Client terminal:

```powershell
$models = (Resolve-Path 'output/bonsai2-models').Path
$env:CAMELID_QUALIFY_REQUIRE_CUDA = '1'
$fresh27 = Join-Path $models 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
node qa/bonsai2/gui-smoke.mjs $fresh27 qa/bonsai2/cuda/27b-gui-fresh http://127.0.0.1:28182 --send --default-image --require-supported --require-cuda --require-request-rate
if ($LASTEXITCODE -ne 0) { throw 'Fresh 27B GUI failed' }
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    node qa/bonsai2/api-qualification.mjs $target "qa/bonsai2/cuda/${size}b-api-qualified.json" http://127.0.0.1:28182 'text,tools,context,sampling,sampling_long,vision'
    if ($LASTEXITCODE -ne 0) { throw "API ${size}B failed" }
    node qa/bonsai2/gui-smoke.mjs $target "qa/bonsai2/cuda/${size}b-gui-supported" http://127.0.0.1:28182 --send --default-image --require-supported --require-cuda --require-request-rate
    if ($LASTEXITCODE -ne 0) { throw "GUI ${size}B failed" }
}
```

The GUI smoke submits both text and an actual image attachment through the
production frontend served by the native library-server harness, requires the
CUDA execution plan and Verified answers, and
saves JSON plus a screenshot. `--default-image` retains normal GUI image bounds.
The API pass covers text, tool calls/results, 1044-token context recall, seeded
sampling, a 32-token seeded continuation, and spatial vision.
The GUI receipt also checks request-rate provenance. Client timing is labelled
`tok/s request`; callback bursts are not reported as engine decode throughput.
These hardware runs do not exercise the installed Tauri desktop shell. Its CI
checks are separate from the model qualification documented here.

Stop the dedicated server, then restart it with the same data paths and:

```powershell
$env:CAMELID_SPEC_DECODE = 'draft'
$env:CAMELID_SPEC_DRAFT_MODEL = $draft
$env:CAMELID_SPEC_DRAFT_TOKENS = '3'
$draftLog = 'output/bonsai2-reference/cuda-server-qualified-draft.log'
& $server $models '127.0.0.1:28182' 2>&1 |
    ForEach-Object { $_.ToString() } |
    Out-File -FilePath $draftLog -Encoding utf8 -Width 4096
```

This terminal captures the dedicated server log as UTF-8. In the client terminal:

```powershell
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    node qa/bonsai2/api-qualification.mjs $target "qa/bonsai2/cuda/${size}b-sampled-draft.json" http://127.0.0.1:28182 sampling_long
    if ($LASTEXITCODE -ne 0) { throw "Draft sampling ${size}B failed" }
}
```

The checker compares all 32 generated IDs against the ordinary API continuation.
[`sampled-draft-runtime.json`](sampled-draft-runtime.json) also preserves the
complete qualified draft-server log and its digest, the four proposal/acceptance
summaries, and the API receipt hashes/timestamps. Model associations follow the
recorded sequential 2B/4B/9B/27B requests; the server log itself has no model
names or timestamps. This confirms active drafting separately from output
parity. Stop this dedicated server, then preserve its log and associations:

```powershell
node qa/bonsai2/record-sampled-draft-runtime.mjs output/bonsai2-reference/cuda-server-qualified-draft.log qa/bonsai2/cuda qa/bonsai2/cuda/sampled-draft-runtime.json 2,4,9,27
```

The final argument records the actual request order. The helper verifies the
four completed API receipt timestamps and hashes without changing the source
log or those receipts. Keep all servers stopped for the device-state tests.

## Device-state and rollback checks

Build the CUDA library test executable with the same release options. Cargo's
JSON output identifies the actual test binary, avoiding accidentally selecting
an older CPU executable from the build cache:

```powershell
$artifacts = cargo test --release --features cuda --lib --no-run --message-format=json --jobs 2 --config 'profile.release.lto=false' --config 'profile.release.codegen-units=16' | ForEach-Object { $_ | ConvertFrom-Json }
if ($LASTEXITCODE -ne 0) { throw 'CUDA library test build failed' }
$testBinary = ($artifacts | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.profile.test -and $_.executable -and $_.target.name -eq 'camelid' } | Select-Object -Last 1).executable
if (-not $testBinary) { throw 'Cargo did not report the Camelid library test executable' }
node qa/bonsai2/cuda-cache-qualification.mjs $testBinary $models qa/bonsai2/cuda/cache-parity.json
```

The controller first runs two host memory regression tests: shared read-only
PTQ mapping bounds/lifetime, and borrowing unchanged packed bytes for upload.
It then runs twelve CUDA tests: the PTQ/Hadamard/embedding and BF16 fixtures,
real-model cache/rollback comparisons for all four sizes, the 2B graph
group/split-K boundary cases, and greedy/seeded/stop/cancel draft parity for all
four sizes. It records each test's exit status, model hash, environment, and log
digest. The checker requires both host rows and all twelve CUDA rows; an
aggregate success flag alone is insufficient.

After all native receipts exist, record the exact build revisions, build
options, retained probe/test/server binary paths, and server receipt build IDs
in `output/bonsai2-reference/cuda-build-record.json`. The checked-in
[`build-record.json`](build-record.json) is the example used for this bundle;
replace its paths/revisions with the binaries from your own run. Use
`node qa/bonsai2/cuda-manifest.mjs --help` for its schema. The build record must
identify the binaries that actually produced the receipts, including any older
server still referenced by a receipt.

Generate the comparison and manifest, then validate the complete bundle:

```powershell
node qa/bonsai2/compare-cuda-benchmarks.mjs
node qa/bonsai2/cuda-manifest.mjs output/bonsai2-reference/cuda-build-record.json
node qa/bonsai2/check-cuda-evidence.mjs
```

Manifest generation also runs the receipt checker. It hashes the actual model,
projector, and retained binary files; records source blobs at the stated build
commits; and checks each API/GUI receipt's server build association. The saved
`manifest.json` links the complete CUDA evidence without replacing the earlier
CPU manifest. No inference or compilation is performed by these three commands.
