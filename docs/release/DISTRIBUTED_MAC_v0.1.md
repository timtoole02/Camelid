# Distributed Mac Baseline for Camelid v0.1

Status: excluded from `v0.1.0-rc1` unless a fresh two-Mac evidence bundle is captured and release-captain-approved.

The release directive says Distributed Mac Mini is included only if stable; otherwise it must be explicitly excluded. Current repo evidence supports loopback/tiny-fixture distributed plumbing and raw TCP benchmark code, but no stable two-Mac, real-model, release-SHA baseline is present.

## Current Evidence

Code surfaces inspected:

- `src/distributed.rs`: TCP worker/coordinator tensor forwarding and network benchmark helpers
- `src/cluster.rs`: lower-level activation/token packet helpers
- `src/main.rs`: `serve-distributed` and `bench-network` commands
- `tests/distributed_tests.rs`: local loopback tiny GGUF pipeline and local loopback network benchmark

Current evidence boundary:

- Local tests can validate protocol plumbing on loopback.
- There is no committed v0.1 evidence bundle showing two physical Macs, Thunderbolt/private-link addressing, a real release model, timing, memory, correctness, failover behavior, or stable repeated runs.
- Therefore Distributed Mac must not appear as a v0.1 performance or support claim.

## Minimal Local Plumbing Check

This check is useful before a real two-Mac run, but it is not a distributed release baseline:

```sh
cd <repo>
cargo test --test distributed_tests -- --nocapture
```

Expected scope: local loopback only.

## Two-Mac Network Baseline

Worker Mac:

```sh
./target/release/camelid bench-network \
  --role worker \
  --addr "$WORKER_THUNDERBOLT_IP:8182"
```

Coordinator Mac:

```sh
./target/release/camelid bench-network \
  --role coordinator \
  --addr "$WORKER_THUNDERBOLT_IP:8182" \
  --ping-count 1000 \
  --payload-size 16384 \
  --bandwidth-mb 100
```

Required output: latency, bandwidth, raw stdout/stderr, both host facts, and pass/fail status.

## Two-Mac Real-Model Baseline

Worker Mac:

```sh
./target/release/camelid serve-distributed \
  --role worker \
  --addr "$WORKER_THUNDERBOLT_IP:8089" \
  --layer-range 16..32 \
  --model "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf" \
  --threads 8
```

Coordinator Mac:

```sh
./target/release/camelid serve-distributed \
  --role coordinator \
  --addr "$COORDINATOR_THUNDERBOLT_IP:8181" \
  --worker-addr "$WORKER_THUNDERBOLT_IP:8089" \
  --layer-range 0..16 \
  --model "$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf" \
  --threads 8
```

Then run the same marker prompt through the coordinator API and record timing/memory:

```sh
/usr/bin/time -lp curl -sS "http://$COORDINATOR_THUNDERBOLT_IP:8181/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama32-3b-q8",
    "messages": [
      {"role": "system", "content": "You are Camelid benchmark mode. Reply with the exact requested text and nothing else."},
      {"role": "user", "content": "Reply with exactly this single line and nothing else: CMLD-BENCH"}
    ],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

## Required Evidence Field Ledger

- Camelid commit SHA: pending; expected release branch HEAD on both Macs
- Comparator commit or version: not applicable; this is a Camelid distributed-mode baseline, so record `Camelid distributed mode`
- Model name: pending; recommended `Llama 3.2 3B Instruct Q8_0`
- Model path: pending; sanitized `$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf` on both Macs
- Model SHA256 hash: pending; same SHA must be recorded on both Macs
- Quantization: pending; expected `GGUF Q8_0`
- Prompt: pending; recommended marker prompt above
- Context size: pending; record configured context/defaults
- Max generated tokens: pending; recommended 16
- Thread count: pending; record coordinator and worker thread counts separately
- Batch settings: pending; record any coordinator/worker batching settings or state defaults
- Runtime flags: pending; record `serve-distributed`, `layer-range`, `worker-addr`, and acceleration flags
- Environment variables: pending; record relevant `CAMELID_*` values on both Macs
- Hardware details: pending; record CPU, RAM, network interface, link speed, and Metal availability for both Macs
- OS version: pending; record `sw_vers` on both Macs
- Raw command: pending; preserve worker, coordinator, network benchmark, and API commands
- Raw output: pending; preserve stdout/stderr from both processes and API response
- Timing data: pending; include network latency/bandwidth plus generation timing
- Memory data: pending; record worker and coordinator RSS/VSZ during load and generation
- Pass/fail status: pending; pass requires stable startup, marker response, no worker crash, and repeated measured runs

## Blockers

- No two-Mac release-SHA run is recorded.
- No real-model distributed correctness artifact is present.
- No repeated timing/RSS evidence exists for coordinator and worker processes.
- Current evidence is loopback/tiny-fixture plumbing only, so Distributed Mac is excluded from v0.1 claims until a fresh evidence bundle proves stability.
