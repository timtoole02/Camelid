# Uniform Hot40 K8 benchmark

This directory contains the isolated, default-off benchmark harness for the
uniform Hot40 experiment. It does not modify the existing Hot32 or Hot48
runners.

The runner admits one deterministic 48-token request on benchmark port 8189
with:

- exactly 40 anonymous physical hot slots on each of 30 layers;
- 128 logical mapped-cold slots per layer;
- K8 speculative verification and full-Q4 MTP;
- mapped readahead omitted from the child environment, with disabled/zero
  geometry and counters;
- no victim cache, overflow, slot pinning, prediction, or sparse-predict probe.

It only starts an absolute prebuilt binary supplied through
`CAMELID_HOT40_BINARY`; it never builds. Both ports 8181 and 8189 must be clear
before the run. Port 8181 is observed with `lsof` through startup, readiness,
the measured request, and shutdown: the runner never binds, connects to, or
signals anything through that port. The benchmark child binds only
`127.0.0.1:8189` and is supervised in its own process group.

## Memory admission

By default, the schema-v3 watchdog requires 60 seconds at pressure level 1
and at least 7.5 GiB reclaimable before spawn. Setting the exact opt-in
`CAMELID_HOT40_ALLOW_WARNING_PRESSURE=1` admits pressure levels 1 or 2 and
lowers only that pre-spawn floor to 4.5 GiB. Critical/unknown pressure is always
rejected. Both policies retain the 2 GiB runtime floor, 7.5 GiB child physical
footprint ceiling, 8 GiB wired-memory ceiling, and hard abort on any current-run
swap-in or swap-out counter change. Existing swap is allowed.

Mapped readahead is omitted by default rather than set to zero. The exact
`CAMELID_HOT40_PAGER_RDADVISE=1` variant enables bounded Darwin `F_RDADVISE`
on exact expert-record ranges: at most 32 early and 64 total records per round,
with zero anonymous pager capacity. It suppresses the legacy `MADV_WILLNEED`
path. A PASS requires zero advisory refusals, exact byte accounting, at least
one accepted record, and all legacy readahead counters to remain zero.

## Page-cache-safe provenance

The runner deliberately does **not** hash, copy, mmap, or otherwise read the
model, cghost, or assistant contents before child spawn. Freshly hashing those
large files previously populated the macOS file cache and invalidated memory
admission.

Instead, their historical SHA-256 claims from
`run-hot48-301da730-v1/intent.json` are recorded with each file's live
device/inode/size/mtime identity. The binary, fixtures, analyzer, runner,
memory sampler, hasher, and watchdog are freshly SHA-256 hashed. This
trade-off is explicit in `intent.json` and `verdict.json`: historical hash plus
live stat is weaker provenance than a fresh content hash, but it avoids
pre-spawn page-cache pollution.

Custom large-input paths require the matching
`CAMELID_HOT40_{MODEL,CGHOST,ASSISTANT}_PREVERIFIED_SHA256` value and a
nonempty `CAMELID_HOT40_PREVERIFIED_PROVENANCE` reference.

## Run

From this directory, after committing the exact source and building the
release binary separately:

```sh
CAMELID_HOT40_BINARY=/absolute/path/to/camelid ./run_hot40.zsh
```

For a QA-only harness commit that did not change runtime sources, an existing
binary may be reused by naming its source commit:

```sh
CAMELID_HOT40_BINARY=/private/tmp/camelid-hot40-49757102 \
CAMELID_HOT40_BINARY_SOURCE_COMMIT=49757102 \
./run_hot40.zsh
```

For the authorized pressure-warning measurement path, add the exact opt-in:

```sh
CAMELID_HOT40_BINARY=/private/tmp/camelid-hot40-49757102 \
CAMELID_HOT40_BINARY_SOURCE_COMMIT=49757102 \
CAMELID_HOT40_ALLOW_WARNING_PRESSURE=1 \
./run_hot40.zsh
```

To run the bounded pager comparison against the frozen 9.857 tok/s Hot40
receipt, add both exact harness opt-ins:

```sh
CAMELID_HOT40_BINARY=/private/tmp/camelid-hot40-rdadvise-2b0c4452 \
CAMELID_HOT40_BINARY_SOURCE_COMMIT=2b0c4452 \
CAMELID_HOT40_ALLOW_WARNING_PRESSURE=1 \
CAMELID_HOT40_PAGER_RDADVISE=1 \
./run_hot40.zsh
```

The resulting speed delta is observational: the baseline and candidate do not
prove identical macOS file-cache state, and speedup is not a correctness gate.

The revision is canonicalized to a full 40-character commit and must be an
ancestor of the clean harness HEAD. Admission also requires an empty Git diff
between those commits for `src`, `Cargo.toml`, `Cargo.lock`, and `build.rs`.
The receipt records `binary_source_commit` separately from `harness_commit`;
the server health check must report the binary source commit.

Optional path/output overrides are `CAMELID_HOT40_MODEL`,
`CAMELID_HOT40_CGHOST`, `CAMELID_HOT40_ASSISTANT`, and
`CAMELID_HOT40_RECEIPT_ROOT`. The receipt root must be a fresh path on the
internal Data volume.

A PASS proves exact 48-token IDs, exact 40-by-30 telemetry, the selected pager
policy, K8/full-Q4 execution, pressure within the selected policy, no
current-run swap delta, bounded memory, and clean ports afterward. It reports
throughput; it is not by itself a claim that the 50 tok/s target has been
reached.

Run model-free contract tests with:

```sh
python3 -m unittest discover -s tests -v
```
