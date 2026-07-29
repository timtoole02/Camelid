# Projects 3 and 8 validation report

Date: 2026-07-28

Workspace: local Camelid checkout

Branch: `codex/mixtral-server-hardening`

Pre-change baseline: `89409ec9cf4b0931396314a8a12e09d987d2916a`

## Outcome

- Project 8, production server hardening: implemented and default-enabled.
- Project 3, Mixtral/MoE completion: diagnostics and performance safeguards implemented, but exact-row long-generation promotion remains fail-closed because the required Mixtral 8x7B Q8 GGUF and llama.cpp oracle are not present on this host.
- The existing one-token Mixtral path and default file-backed expert policy remain unchanged.

## Project 3 changes

- Per-layer router logits, selected expert IDs/normalized weights, and per-expert gate/up/activation/down/weighted-down checkpoints.
- Generated-index diagnostic capture suitable for the known index-9 divergence.
- A watchdog script that samples health, slot progress, metrics, and the diagnostic response.
- Multi-token Mixtral generation rejected by default unless `CAMELID_MIXTRAL_LONG_GENERATION=1`.
- Optional `CAMELID_MOE_EXPERT_STORAGE=resident_q8` path:
  - zero-copy expert range selection;
  - no f32 expert materialization;
  - no per-token resident-block clone;
  - no `Vec` to `Arc<[T]>` multi-GiB promotion copy;
  - live-RAM preflight covering expanded resident blocks, peak loader scratch, and host headroom.
- Normal non-diagnostic MoE execution remains allocation-free for the new trace fields.

## Project 8 changes

- Bearer and `X-API-Key` authentication, including key-file support and constant-time comparison.
- Non-loopback listeners fail closed without authentication unless the explicit unsafe override is supplied.
- Exact-origin CORS allowlists; wildcard, `null`, path, and query origins are refused.
- Optional rustls TLS with paired certificate/key validation.
- Request-body, prompt-token, generation-token, and model-download ceilings.
- Completed model downloads are rechecked on disk before `.part` promotion, independently of curl metadata/behavior.
- Prometheus `/metrics` with HTTP/generation/token/cache/queue/slot/progress/RSS/VRAM telemetry and no model, prompt, path, secret, user, or unbounded labels.
- Real active-task elapsed/stalled/progress reporting in health and slot snapshots.
- Anonymous loopback behavior remains the default.

## Correctness and build gates

- Final serial library suite: **1,259 passed, 0 failed, 69 ignored** (fixture/platform gates).
- API vertical slice: **90 passed, 0 failed**.
- Earlier all-integration rollup: **1,592 passed, 0 failed, 72 ignored** including the then-current library suite; final additive changes were subsequently covered by the final serial library suite and API vertical slice.
- Strict Clippy: `cargo clippy --all-targets --all-features -- -D warnings` passed.
- `cargo fmt --check`, `cargo check --all-targets --all-features`, and `git diff --check` passed.
- Final release build: `cargo build --release --bin camelid --all-features` passed.
- Live final-binary smoke:
  - `/health`: HTTP 200
  - `/metrics`: HTTP 200, Prometheus text content type
  - runtime project 8: `implemented`, `default_enabled=true`
  - unauthenticated `0.0.0.0` bind: refused with exit code 1

## Dense decode A/B regression gate

Exact model:

- local `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` fixture
- SHA-256: `A4C9BB1DBAA372F6381A035FA5C02EF087AAA1FF1F843A56A22328114F03FC59`
- Deterministic CPU, 8 threads, 48 generated tokens, seven alternating before/after pairs.

| Measure | Before | After | Result |
|---|---:|---:|---|
| Absolute median tok/s | 25.357 | 25.193 | -0.649% (within observed run noise) |
| Median paired after/before ratio | 1.000 | 1.006 | **+0.609%** |
| Median peak RSS | 1,286,582,272 B | 1,286,811,648 B | +229,376 B / +0.018% |
| Generated-token hash | `B3ECD15506DA` | `B3ECD15506DA` | exact match in every run |

Verdict: no measurable dense decode or memory regression.

## API decode B-A-B order/thermal check

The first baseline/final sequence showed a small later-run slowdown, so the unchanged baseline was run again after the final binary.

| Run order | Median internal generation time, 48 tokens |
|---|---:|
| Baseline before | 2,251 ms |
| Final | 2,279 ms |
| Baseline after | 2,284 ms |

The final server lies inside the unchanged baseline's order/thermal bracket. All requests produced the identical 48-token sequence. Verdict: no measurable API decode regression from progress/metrics accounting.

## Honest Mixtral boundary

No Mixtral GGUF exists in the checked local model directories, and this host has 15.7 GiB of physical RAM. Therefore:

- exact Mixtral 8x7B Q8 token parity at and beyond generated index 9 was not rerun;
- resident-Q8 Mixtral throughput/RSS was not measured;
- no broad Mixtral, API/WebUI, long-context, or production-throughput promotion is made;
- multi-token Mixtral remains fail-closed by default until the exact model and compatible llama.cpp oracle can run the watchdog/parity matrix.
