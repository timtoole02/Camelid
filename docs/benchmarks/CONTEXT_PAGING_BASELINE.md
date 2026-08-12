# Context Paging baseline

This is the deterministic fixture exercised by
`chat::context_paging::tests::benchmark_reports_existing_vs_paged_context`.
It compares two growing prompt requests with two bounded capsule requests using
the conservative fallback estimator. It is a regression baseline, not a claim
about live-model speed.

| Metric | Existing prompt | Context Paging |
|---|---:|---:|
| Total input tokens | 7,002 | 200 |
| Peak request tokens | 4,001 | 100 |
| Model calls | 2 | 2 |
| Task success | yes | yes |
| Verification retries | 1 | 1 |
| Fixture wall-clock | 42 ms | 42 ms |

The acceptance suite separately builds real capsules, includes exact source and
phase-filtered schemas, runs a hash-checked edit plus verification across three
fresh model requests, and asserts that an oversized prior transcript never
appears in any request. Live wall-clock and tokenizer numbers depend on the
loaded model and are recorded in the runtime metrics files under
`.camelid/context-paging/runtime-state-<task-id>.json`.

## Live restart acceptance (2026-08-12)

The Windows live gate resumed a persisted, already-verified Web Code task using
`Qwen3-4B-Q4_K_M.gguf`. The task had previously created a graphical Tkinter
tic-tac-toe game. This restart specifically guards against a completed ledger
falling back into Modify and changing the workspace again.

| Metric | Observed |
|---|---:|
| Execution lane | CUDA resident K-quant |
| Exact prompt tokens | 438 |
| Capsule estimate / configured maximum | 598 / 5,500 |
| Exact source pages included | 0 |
| Tools exposed | 0 |
| Generation allowance | 256 tokens |
| Time to first token | 20,200 ms |
| Total model step | 21,352 ms |
| Model actions | 1 typed `COMPLETE` |
| Session outcome | `answered` |
| Persisted ledger | revision 25, `complete` |
| Final source verification | `py -m py_compile tic_tac_toe.py` (exit 0) |

The frontend activity endpoint ended at `idle / answered / finished` and marked
the main agent `completed`. The generated file remained byte-identical across
the restart (`sha256 0f1e1b5d24cf0aff9d304c78735ac172a92636543e205a04ba7cbed375d6abde`).

An intentionally CPU-only debug-binary probe received the same 438-token
capsule but was stopped after its 600-second client bound before a first token.
That is a performance limitation of the unoptimized CPU lane, not a successful
latency result; the CUDA gate above is the completed end-to-end receipt.
