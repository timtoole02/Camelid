# Camelid benchmark system foundations

This internal harness implements the first benchmark-system phase: a local,
informational, same-host base/head comparison over Camelid's hidden
`bench-generate` command.

It also implements the model-free Phase 2 task/scorer foundation. Phase 2 task
packages contain a small fixture, but only that fixture is copied into the
writable attempt root. The task manifest, hidden scorer, expected control
overlays, and outside canary remain controller-owned. Package, fixture, scorer,
and canary identities are checked before execution; task, fixture, and scorer
identities are checked again after scoring.

Phase 3 starts the native Camelid adapter around the shipped `camelid agent
exec` CLI. The adapter verifies candidate/model bytes before materialization,
reserves an ephemeral loopback address, constructs bounded shared-task flags,
strips credentials and Camelid environment overrides, preserves exit `0/1/3`,
kills the owned process tree on timeout, and invokes the independent scorer only
after cleanup. Unattended execution requires an explicit disposable-boundary
implementation; the production CLI exposes only WSL bubblewrap, with network
unshared, Windows mounts absent, system/runtime/model read-only, and only the
task attempt writable. Synthetic mode exists only inside canned tests.

It does not set a regression threshold, run model-backed work in CI, compare
external runtimes, execute a real model in hosted CI, or create public evidence.
O9 is implemented through `agent exec --benchmark-events`: a create-new,
secret-safe typed trace carries terminal state, per-step token/timing data, and
hashed tool audit events. Human stderr is never parsed into evidence.

Phase 4 pins Pi `0.84.3` by release archive hash and source commit. The Pi
adapter verifies the archive, extracted Pi executable, Camelid binary, and
model through both Windows-visible and WSL paths before task materialization.
Pi and Camelid then run as siblings inside one network-unshared bubblewrap
namespace. Pi receives an isolated ephemeral config/home, no session, no
project trust, no context/resource discovery, offline startup, and an exact
shared-tool allowlist. Pi JSON stream v3 is parsed from full stdout after both
processes are terminal; repository scoring remains independent of final prose.
The checked compatibility fixture records the real pinned release sending
`max_tokens`, SSE usage frames, non-strict function tools, and a nameless tool
result continuation. `developer`, `max_completion_tokens`, `reasoning_effort`,
`store`, strict tools, and grammar tools remain explicitly unqualified.
The only explicit extension is controller-owned and provider-scoped: it maps
Camelid's typed prompt-limit error to Pi's documented
`context_length_exceeded` marker. Extension discovery remains disabled.

## Safety and validity

- Source SHAs, `Cargo.lock`, model bytes, prompt bytes, and built binaries are
  hashed or checked before measurement.
- Base and head build into separate target directories.
- The planner owns balanced arm order; a campaign cannot submit all-base then
  all-head ordering.
- Every process block gets a unique marker at the front of the prompt, shared by
  the paired base/head arms.
- Phase 1 supports only `cpu_deterministic`, asserted by `--deterministic` plus
  the absence of Camelid's structured GPU offload status.
- Cross-arm output-token divergence invalidates the performance result.
- Invalid and unfavorable samples remain in the sealed local bundle.
- Numeric verdicts remain `INCONCLUSIVE_NOISE` until a later calibration phase
  approves practical margins and sample counts.

## Commands

Print the source-manifest digest for the exact harness code and schemas:

```sh
node tools/bench/system/cli.mjs digest
```

Copy `examples/campaign.phase1.json` outside the tracked tree, replace every
placeholder, and pin the digest above. Audit the resolved plan without building:

```sh
node tools/bench/system/cli.mjs plan --config <campaign.json> --out <plan.json>
```

Run the complete local campaign. This builds both arms serially unless a local
ablation campaign explicitly supplies `--prepared` identities:

```sh
node tools/bench/system/cli.mjs run --config <campaign.json> --out-root <output-root>
```

The output directory is refused if it already contains files. A complete bundle
contains the resolved plan, prepared binary identities, raw stdout/stderr,
materialized block prompts, per-sample records, comparison, summary, manifest,
and `SHA256SUMS`.

Builds set `CARGO_NET_OFFLINE=true` when the campaign network policy is `deny`.
Provision the pinned toolchain and dependency cache before starting; a missing
crate is a preparation failure, not permission to resolve mutable dependencies
during measurement.

One lock file serializes campaigns under the selected output root. The harness
does not remove a stale lock automatically. After an interrupted controller,
prove the recorded PID and all campaign-owned children are gone before manually
removing the lock. A run that fails after creating its directory writes
`failure.json` with state `INCOMPLETE`; reruns use a new campaign ID.

Verify a Phase 2 task package and its pinned fixture/scorer/canary identities:

```sh
node tools/bench/system/cli.mjs task-verify --task qa/benchmarks/agent/tasks/agent_local_logic_fix
```

Materialize a fresh writable attempt plus an outside canary, then score the
terminal repository state independently of any agent prose:

```sh
node tools/bench/system/cli.mjs task-materialize --task <task-dir> --workspace <new-workspace>
node tools/bench/system/cli.mjs task-score --task <task-dir> --workspace <workspace> --out <score.json>
```

Run one real native attempt only through the WSL bubblewrap boundary, score it,
seal the local evidence bundle, verify checksums, and remove the disposable
workspace after success:

```sh
node tools/bench/system/cli.mjs native-run \
  --task <task-dir> --workspace <new-workspace> \
  --binary <windows-visible-linux-binary> --linux-binary <linux-path> \
  --model <windows-model> --linux-model <linux-path> \
  --source-sha <sha> --campaign-id <id> --timeout-ms <ms> --out <bundle-dir> \
  --wsl-gpu true --max-tokens-per-step 256
```

Run one pinned Pi attempt through the same task/scorer and WSL boundary. The
release archive, extracted executable, Camelid binary, model, exact served
model ID, and context window are all explicit inputs:

```sh
node tools/bench/system/cli.mjs pi-run \
  --task <task-dir> --workspace <new-workspace> --out <bundle-dir> \
  --pi-archive <windows-archive> --pi <windows-visible-linux-pi> \
  --linux-pi-archive <linux-archive> --linux-pi-dir <linux-pi-dir> \
  --binary <windows-visible-linux-camelid> --linux-binary <linux-camelid> \
  --model <windows-model> --linux-model <linux-model> \
  --model-id <exact-served-id> --context-window <tokens> \
  --source-sha <sha> --campaign-id <id> --timeout-ms <ms> \
  --wsl-gpu true --max-tokens-per-step 256
```

`pi-run` writes and independently verifies a sealed local bundle before it
removes the disposable workspace. A failed scorer or model outcome is retained
as evidence and returns nonzero; it is never retried until it passes.

Pi's named `find` and `grep` tools are not advertised because the pinned release
implements them through `fd` and `rg`, which are absent and cannot be downloaded
inside the offline boundary. The allowed `bash` tool retains search capability
through preflight-verified `/usr/bin/find` and `/usr/bin/grep`.

The controller appends task-independent benchmark workflow rules to Pi's stock
system prompt: inspect and read before editing, never invent workspace facts,
recover from tool errors, verify changes, and continue until done or genuinely
blocked. The appended text contains no task path, source text, or expected answer.

CPU remains the default. `--wsl-gpu true` adds only the WSL GPU device and
driver-library search path to the otherwise unchanged network-unshared
bubblewrap boundary, requires a successful in-boundary GPU preflight, and is
recorded in `adapter.json` and `manifest.json`.

`--max-tokens-per-step` may tighten but never exceed the task package's declared
per-step ceiling. The effective value is recorded in the same evidence files.

Trace-emitting disposable benchmark runs disable Camelid's user-facing undo
checkpoints. Ordinary agent sessions retain checkpoints; benchmark bundles
record `checkpoints_enabled: false` so strict repository scoring sees only task
mutations rather than adapter-owned `.camelid/` state.

The same path uses the recorded `benchmark_shared` tool profile: exactly
`read_file`, `list_dir`, `search`, `write_file`, `edit_file`, and `run_shell`.
Native-only planning, GUI, MCP, network, system-inspection, and subagent tools
are neither advertised nor accepted for shared-task evidence.

The workspace path must not already exist. This prevents materialization from
overwriting an unrelated directory or a pre-existing canary.

The initial task packages name Windows and Linux; exact-head hosted validation
proves both. macOS remains unclaimed until the same model-free suite runs there.
Phase 2 paths use forward slashes, are
case-sensitive, and allow either an exact
relative path or a trailing recursive `/**`. Parent traversal, absolute paths,
backslashes, other wildcard forms, symlinks, special files, and case-folding
collisions are refused. Model-free setup/check commands are restricted to
`node <relative-script>` or `node --check <relative-script>` with an isolated
environment and no shell interpolation.

## Self-tests

```sh
node tools/bench/system/test-schemas.mjs
node tools/bench/system/test-bench-generate-parser.mjs
node tools/bench/system/test-stats.mjs
node tools/bench/system/test-planner.mjs
node tools/bench/system/test-prepare.mjs
node tools/bench/system/test-process-runner.mjs
node tools/bench/system/test-runtime-adapter.mjs
node tools/bench/system/test-bundle.mjs
node tools/bench/system/test-cli.mjs
node tools/bench/system/test-pi-contract.mjs
node tools/bench/system/test-pi-openai-contract.mjs
node tools/bench/system/test-pi-provider-extension.mjs
node tools/bench/system/test-pi-adapter.mjs
node tools/bench/system/test-pi-bundle.mjs
node tools/bench/system/test-pi-cli.mjs
node tools/bench/system/test-safety.mjs
node tools/bench/test-v0.1-benchmark-harness.mjs
node scripts/test-benchmark-system-phase2.mjs
node scripts/test-benchmark-system-phase3.mjs
node scripts/test-benchmark-system-phase4.mjs
```

The existing `validation-scripts` CI job runs the same set through
`scripts/test-benchmark-system-phase1.mjs` and
`scripts/test-benchmark-system-phase2.mjs`. The model-free Phase 3 adapter test
is discovered through `scripts/test-benchmark-system-phase3.mjs`; the Phase 4
Pi contract, provider extension, adapter, bundle, and CLI tests are grouped by
`scripts/test-benchmark-system-phase4.mjs`.
