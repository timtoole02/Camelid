# Bonsai 2 Windows CUDA qualification

This directory keeps CUDA evidence separate from the earlier CPU qualification.
The reference measurements are complete. Native benchmark, API, GUI, and cache
receipts must be generated and pass `check-cuda-evidence.mjs` before this directory
establishes a completed native qualification.

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
Callback emission timing can be bursty during speculation and is not the
cross-engine comparison metric.

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

On this host, reference ordinary request rates were 205–221 tok/s (2B),
113–118 (4B), 77–80 (9B), and 1.68–1.76 (27B). The same 2B drafter reduced those
rates to 57–86, 47–70, 38–62, and 0.89–1.19 respectively. The first three target
sizes fit all layers on CUDA. The port's automatic fit places 45/65 target layers
on CUDA for ordinary 27B and 31/65 with drafting, plus all 25/25 drafter layers.
Those are complete operating configurations on a constrained GPU, not a pure
comparison of kernel arithmetic. Native 27B streams packed weights while
executing the layer computations on CUDA; its capacity-plan excerpts are
recorded in the native receipts.

The port's 2B and 9B prose continuations differ between ordinary and draft
decoding. These observed parity failures remain in the receipts. They do not
waive the native requirement that ordinary and draft outputs match exactly.

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
$env:CAMELID_SPEC_DECODE = $null
& $server $models '127.0.0.1:28182'
```

Client terminal:

```powershell
$models = (Resolve-Path 'output/bonsai2-models').Path
$env:CAMELID_QUALIFY_REQUIRE_CUDA = '1'
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    node qa/bonsai2/api-qualification.mjs $target "qa/bonsai2/cuda/${size}b-api-qualified.json" http://127.0.0.1:28182 'text,tools,context,sampling,sampling_long,vision'
    if ($LASTEXITCODE -ne 0) { throw "API ${size}B failed" }
    node qa/bonsai2/gui-smoke.mjs $target "qa/bonsai2/cuda/${size}b-gui-supported" http://127.0.0.1:28182 --send --default-image --require-supported --require-cuda
    if ($LASTEXITCODE -ne 0) { throw "GUI ${size}B failed" }
}
```

The GUI smoke submits both text and an actual image attachment through the
packaged frontend, requires the CUDA execution plan and Verified answers, and
saves JSON plus a screenshot. `--default-image` retains normal GUI image bounds.
The API pass covers text, tool calls/results, 1044-token context recall, seeded
sampling, a 32-token seeded continuation, and spatial vision.

Stop the dedicated server, then restart it with the same data paths and:

```powershell
$env:CAMELID_SPEC_DECODE = 'draft'
$env:CAMELID_SPEC_DRAFT_MODEL = $draft
$env:CAMELID_SPEC_DRAFT_TOKENS = '3'
& $server $models '127.0.0.1:28182'
```

In the client terminal:

```powershell
foreach ($size in 2, 4, 9, 27) {
    $target = Join-Path $models "Ternary-Bonsai-2-${size}B-PTQ1_0.gguf"
    node qa/bonsai2/api-qualification.mjs $target "qa/bonsai2/cuda/${size}b-sampled-draft.json" http://127.0.0.1:28182 sampling_long
    if ($LASTEXITCODE -ne 0) { throw "Draft sampling ${size}B failed" }
}
```

The checker compares all 32 generated IDs against the ordinary API continuation.
Stop this server before running the device-state tests.

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

The controller runs the synthetic PTQ/Hadamard/embedding and BF16 fixtures, the real-model
cache/rollback comparison for all four sizes, the 2B graph group/split-K boundary
cases, and greedy/seeded/stop/cancel draft parity for all four sizes. It records
each test's exit status, model hash, environment, and log digest. The checker
requires all twelve rows; an aggregate success flag alone is insufficient.

After all native receipts exist, record the exact build revisions, build
options, retained probe/test/server binary paths, and server receipt build IDs
in `output/bonsai2-reference/cuda-build-record.json`. Use
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
