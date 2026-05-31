# Camelid v0.1 Benchmark Commands

- Generated UTC: 2026-05-31T18:42:31.916Z
- Config: `$CAMELID_WORKTREE/target/v0.1-20260531T184150Z.config.json`
- Bundle: `$CAMELID_WORKTREE/qa/evidence-bundles/v0.1/20260531T184150Z-real-local-raw`
- Dry run: false

## llamacpp-cpu-samehost-llama32-3b

- Engine: llama.cpp
- Model: llama32-3b-instruct-q8-local
- Repetitions: 1
- Timeout ms: 1200000
- CWD: `$CAMELID_WORKTREE`
- Env overrides: `{"CAMELID_STREAM_TIMING_DIAGNOSTICS":"on"}`

```sh
node scripts/bench-llama3-same-host.mjs --backend http://canonical-private-ubuntu-validation-host:19781 --llama-url http://canonical-private-ubuntu-validation-host:19783 --model '$CAMELID_MODEL_DIR/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct-Q8_0.gguf' --model-id llama32-3b-q8-v01-rc1-20260531T184150Z --row-id llama32_3b_instruct_q8_0 --backend-bin '$CAMELID_RELEASE_BUILD_ROOT/cargo-target/release/camelid' --llama-server '$CAMELID_RELEASE_BUILD_ROOT/llama.cpp-v0.1-rc1/build/bin/llama-server' --max-tokens 16 --warmup 1 --repeats 3 --threads 8 --llama-context 512 --unique-prompt --require-marker --expected-marker CMLD-BENCH --out qa/evidence-bundles/v0.1/20260531T184150Z-real-local-raw/same-host-llama32-3b-q8.json
```
