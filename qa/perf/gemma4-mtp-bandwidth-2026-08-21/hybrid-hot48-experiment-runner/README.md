# Uniform Hot48 exact benchmark

This directory is an isolated A/B harness for the runtime-only uniform Hot48
experiment. It does not import, rewrite, or relax the frozen Hot32/V5 runners
in `../hybrid-hot48-runner`.

The runner admits exactly one 48-token K8 request. It starts only a prebuilt
binary, refuses listeners on the user WebUI port 8181 and benchmark port 8189,
and runs the existing schema-v3 macOS watchdog on its fixed 250 ms schedule.
The watchdog requires a 60 second clean baseline, at least 8 GiB reclaimable
baseline headroom, at least 2 GiB at runtime, no more than 7.5 GiB child
physical footprint, and no more than 8 GiB host wired memory. Existing swap is
allowed; any new swap-in or swap-out activity is rejected.

The target is launched with the exact opt-in
`CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT48_EXPERIMENT=1`, canonical parent
admission `CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS=32`, full-Q4 MTP, and
the one-command-buffer device chain. Any inherited per-layer slot override is
a pre-spawn refusal.

Build the optimized binary separately, then run:

```sh
CAMELID_HOT48_BINARY=/absolute/path/to/camelid \
  ./run_hot48.zsh
```

Optional input overrides are `CAMELID_HOT48_MODEL`, `CAMELID_HOT48_CGHOST`,
`CAMELID_HOT48_ASSISTANT`, and `CAMELID_HOT48_RECEIPT_ROOT`. Every override
must resolve to a regular non-symlink file or a fresh output directory on the
internal system data volume. The runner hashes the binary, all model inputs,
fixtures, and tooling into `intent.json` and `baseline.txt` before launch.
Large-file integrity hashing disables macOS read-ahead and enables
`F_NOCACHE`, then allows 60 seconds for transient read pages to age out, so
receipt creation does not turn model files into artificial baseline pressure.
Canonical watchdog telemetry is captured immediately before and after that
hash phase; both samples must retain 8 GiB headroom, normal pressure, bounded
wired memory, one boot identity, and unchanged swap-in/swap-out counters.
It also refuses a dirty or untracked source checkout and requires the running
binary to report the exact clean commit through its health receipt.

`verdict.json` is a measurement report, not a 50 tok/s promotion claim. A PASS
proves exact token identity, a uniform 48 x 30 anonymous-hot geometry, bounded
memory, no safety abort, full-Q4/device-chain execution, and coherent stage
timings. It reports effective receipt TPS, verifier/assistant timing summaries,
Metal stage summaries, and watchdog memory peaks.

Run the model-free contract tests with:

```sh
python3 -m unittest discover -s tests -v
```
