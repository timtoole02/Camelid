# Uniform Hot40 K8 benchmark

This directory contains the isolated, default-off benchmark harness for the
uniform Hot40 experiment. It does not modify the existing Hot32 or Hot48
runners.

The runner admits one deterministic 48-token request on benchmark port 8189
with:

- exactly 40 anonymous physical hot slots on each of 30 layers;
- 128 logical mapped-cold slots per layer;
- K8 speculative verification and full-Q4 MTP;
- no victim cache, overflow, slot pinning, prediction, or sparse-predict probe.

It only starts an absolute prebuilt binary supplied through
`CAMELID_HOT40_BINARY`; it never builds. Both ports 8181 and 8189 must be clear
before the run. Port 8181 is only observed with `lsof`: the runner never binds,
connects to, or signals anything through that port. The benchmark child binds
only `127.0.0.1:8189` and is supervised in its own process group.

## Memory admission

The schema-v3 watchdog requires 60 seconds of normal-pressure baseline
samples, at least 8 GiB reclaimable before spawn, at least 2 GiB during the
run, no more than 7.5 GiB child physical footprint, and no more than 8 GiB host
wired memory. Existing swap is allowed. Any current-run swap-in or swap-out
counter change—from the preflight sample through the final watchdog sample—is
a failure. The watchdog always rejects swap-out changes; `--reject-swapin-growth`
adds the matching swap-in rule.

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

Optional path/output overrides are `CAMELID_HOT40_MODEL`,
`CAMELID_HOT40_CGHOST`, `CAMELID_HOT40_ASSISTANT`, and
`CAMELID_HOT40_RECEIPT_ROOT`. The receipt root must be a fresh path on the
internal Data volume.

A PASS proves exact 48-token IDs, exact 40-by-30 telemetry, K8/full-Q4
execution, normal pressure, no current-run swap delta, bounded memory, and
clean ports afterward. It reports throughput; it is not by itself a claim
that the 50 tok/s target has been reached.

Run model-free contract tests with:

```sh
python3 -m unittest discover -s tests -v
```
