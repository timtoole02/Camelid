# P5 remainder — mixed-engine placement: receipts

Live receipts for `fabric serve --allow-mixed-engines` (AGENT.md §6 R2), taken on the M4
benchmark node on 2026-09-11 against real software: a Camelid node and an Ollama node serving the
same Llama 3.2 1B Q8_0 weights, plus a logging stand-in engine for the refusal cases. Everything
under `raw/` is the unfiltered output it names (home directories replaced with `~` for the public
scrub); the screenshots are in `screens/`.

| | |
|---|---|
| Code under test | `42561ea8` (the proxy binary: `camelid v0.7.3-29-g42561ea8`, sha256 `eead05a2…`, built from a clean clone) |
| Web UI in the final captures | `74b3a285` (a CSS/wording fix to Screen E found in the first capture; no proxy change) |
| Camelid | `camelid serve --model Llama-3.2-1B-Instruct-Q8_0.gguf`, GGUF sha256 `432f310a…`, active id `Llama 3.2 1B Instruct` |
| Ollama | 0.33.2, `llama32-1b-q8-r1:latest`, created from that GGUF (Ollama stores a re-serialized copy of it) |
| Versions and digests | `raw/versions.txt` |

No throughput or latency is claimed here (I10). The only numbers are request counts.

## How this was taken

Four processes on loopback, each with its own request log:

| Process | Address | Request log |
|---|---|---|
| Camelid (also serves the web UI) | `:8181` | `raw/relay-camelid.log`, from a byte-for-byte TCP relay on `:18181` that the fabric addresses as `a-camelid` |
| Ollama 0.33.2 | `:11434` | `raw/ollama.log` (Ollama's own `[GIN]` request lines) |
| A stand-in that answers Ollama's three listings and refuses every chat request | `:11500` | `raw/standin.log` |
| `camelid fabric serve`, without and with the flag | `:8282` (`:8283` for the refusal cases) | `raw/*.stderr` (`RUST_LOG=camelid=info`) |

```
a-camelid=127.0.0.1:18181
b-ollama=ollama://127.0.0.1:11434
alias llama-1b=a-camelid:Llama 3.2 1B Instruct
alias llama-1b=b-ollama:llama32-1b-q8-r1:latest
```

Counters are line counts in those logs before and after a step. Every client answer is the
`curl -si` output in the file named below. Where a step needed the Camelid node busy, four long
generations were sent **directly** to `:8181`, not through the proxy, and the node's
`engine_queue_depth` at the moment each probe was sent is in the file name (`-depthN`). The driver
is `live3.sh` next to the raw receipts on the node; the step-by-step transcript is
`raw/steps.log`.

## (c) Without the flag, the proxy never sends Ollama a request

Started with `CAMELID_ALLOW_MIXED_ENGINES=1` in its environment. Its startup line still reads
`placement: Camelid engines only` (`raw/refused.stdout`): the flag has no environment variable.

| Request | Answer | File |
|---|---|---|
| `llama-1b` (alias) | 200 from `a-camelid`, `model: Llama 3.2 1B Instruct`, `model-identity: asserted_by_operator` | `raw/c1-alias-llama-1b.txt` |
| Camelid's own id | 200 from `a-camelid`, with exactly the headers a proxy answered with before this work (node, engine, reason, attempts) | `raw/c2-camelid-id.txt` |
| `llama32-1b-q8-r1:latest`, which only Ollama holds | **404 `model_not_found`**: "…b-ollama holds it but runs an engine this fabric does not place on (accept that with --allow-mixed-engines)" | `raw/c3-ollama-only.txt`, `raw/c-model-ollama-only.txt` |
| the same, carrying tools | 404 | `raw/c4-ollama-only-tools.txt` |
| no model | 200 from `a-camelid` | `raw/c5-no-model.txt` |
| six concurrent `llama-1b` | 6 × 200 from `a-camelid` | `raw/c6-burst-*.txt` |

**Ollama received 0 `POST /v1/chat/completions`** in this stage; the Camelid relay received 9, one
per inference request above. What the proxy did send Ollama was reads of its listings (30
`GET /api/version|tags|ps`), which is how a node it does not place on is still reported.
`/v1/models` lists Camelid's id and `llama-1b` (`"x_camelid_identity":"asserted_by_operator"`),
not the Ollama-only model (`raw/c-models.txt`).

## (a) With the flag, answers come from both engines and each names its engine and model

Startup (`raw/mixed.stdout`):

```
placement: ALSO placing on other engines (--allow-mixed-engines)
  b-ollama (ollama 0.33.2): accepted without asking: publishes no load to rank on; a full queue is indistinguishable from a failure; cannot attest a warm prefix, so affinity would be a guess. requests carrying tools are never placed here: tool calls have not been measured on ollama 0.33.2; rerank requests are never placed here: its API has no rerank route.
  note: nodes of other engines added to the nodes file later are accepted too, and are announced here.
```

| Request | Answer |
|---|---|
| `llama32-1b-q8-r1:latest` | 200 from `b-ollama`, `engine: ollama`, `model: llama32-1b-q8-r1:latest`, `residency-observed: resident` |
| Camelid's id, `llama-1b` idle, no model | 200 from `a-camelid`, each with `engine: camelid` and the model id sent |
| six concurrent `llama-1b` | 4 from `a-camelid`, 2 from `b-ollama`, each answer naming its engine, its model and `asserted_by_operator` |

10 answers named `ollama` 3 times and `camelid` 7 times; Ollama's own log grew by exactly 3 chat
requests and the Camelid relay by exactly 7 (`raw/a*.txt`, `raw/steps.log`).

## (b) A request carrying tools never lands on the unmeasured Ollama

1. **The primary proof**, which no ranking can produce: tools for the model only Ollama holds →
   **400**, Ollama +0 (`raw/b1-tools-ollama-only.txt`):

   ```
   {"error":{"message":"a request carrying tools is placed only on a node measured to handle them; b-ollama (ollama 0.33.2) has not been measured handling tool calls here; a-camelid can take tools but does not hold `llama32-1b-q8-r1:latest`","type":"invalid_request_error","code":"capability_unavailable","param":"tools"}}
   ```

2. **Interleaved, with Camelid held busy** (queue depth 4 → 9 by direct requests), `P,T,P,T,P,T,P`,
   where P is a plain `llama-1b` request and T the same with `tools`, buffered and then streamed:
   every P went to `b-ollama` (Ollama +1 each) and every T to `a-camelid`. Across both
   interleaves Ollama received **8 chat requests, exactly the 8 P's** (`raw/b2-*`, `raw/b3-*`). One
   streamed T was answered 503 `engine_queue_full` by Camelid itself, its queue being past its
   bound at depth 9; it was relayed, not sent to Ollama, which is the guard doing its job.
3. `fabric route --allow-mixed-engines --with-tools --model llama-1b --json` → `a-camelid`
   (`raw/b4-route-with-tools.json`).

## (d) An untyped refusal from another engine is relayed once and never retried on a sibling

A fabric of `a-camelid` (held busy) and `c-standin` (`ollama://`, the stand-in), both holding
`llama-1b` by alias, `--max-forward-attempts 2`:

| Stand-in answers | Buffered | Streaming | Stand-in POSTs | Camelid POSTs |
|---|---|---|---|---|
| `503 {"error":"server busy, please try again.  maximum pending requests exceeded"}` | 503 relayed, `attempts: 1`, `engine: ollama` | the same | 2 | **0** |
| `503 {"error":{"code":"engine_queue_full"}}` (our own code, from an engine that does not declare it) | 503 relayed, `attempts: 1` | the same | 2 | **0** |

(`raw/d1-*`, `raw/d2-*`, `raw/standin.log`, `raw/relay-*.stderr`.) The stand-in shows as
unreachable in each proxy's startup line only because the proxy started a moment before the
stand-in was listening; it was ready before the requests were sent.

## (e) `/v1/health` and Screen E, from the real proxy

`raw/c-health.txt` / `raw/final-refused-health.txt` (flag off) and `raw/a-health.txt` /
`raw/final-allowed-health.txt` (flag on) carry the `placement` block: the mode, the flag's
spelling, both prices, `models_if_mixed`, `covers_nodes_added_later`, the standing-grant sentence,
and per node `placement_blocker_detail` and `requirement_limits`.

The release binary served the web UI with the Routing section in it (`raw/embed-asset.txt`,
`raw/steps.log`). The final captures were taken from the fixed build, served locally, reading the
same real proxy:

| Screen | File |
|---|---|
| Flag off: the confirmation, in the proxy's own words, naming `b-ollama` / `ollama 0.33.2`, both limits, the standing grant and what it would start serving | `screens/final-refused-2-confirm.png` |
| Flag off: the command, shown only after the confirmation | `screens/final-refused-3-command.png` |
| Flag on: what the proxy is accepting now, and the node added since it started | `screens/final-allowed-1-routing.png` |
| Flag on: the node drawer, routed to with its limits | `screens/final-allowed-2-node-drawer.png` |

Across each flow the page made only `GET` requests, and its only request to the proxy was
`GET /v1/health` (`raw/final-*-report.json`).

## The standing grant, live

With the flag on and the nodes file re-read, appending `c-ollama2=ollama://127.0.0.1:11434`
printed

```
fabric: c-ollama2 (ollama) added while placing on other engines; accepted without asking: publishes no load to rank on; a full queue is indistinguishable from a failure; cannot attest a warm prefix, so affinity would be a guess. Tool-calling requests: not until measured.
```

and health listed it under `placement.foreign_nodes_added_since_start` (`raw/g-health.txt`,
`raw/final-allowed-health.txt`).

## Gates and ablations on the same head

`raw/gates/`: `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean with only
`clippy::unnecessary_cast` allowed (one pre-existing hit in `src/metal.rs` under the node's Homebrew
rustc 1.94; CI pins 1.95); `cargo test --all-targets --no-fail-fast` 3277 passed, 55 ignored, 2
failed — the two `fabric::http` TLS tests that fail on this dual-stack node without the change too.
Every CI frontend smoke passed; `smoke:math-browser` failed once on a font-fetch timing check and
passed on a re-run, and `smoke:workspace-visual` needs the preview server CI starts first and
passes when run that way.

`raw/ablations/`: 17 single-edit sabotages, each with its catcher named beforehand; every named test
failed and every file was restored and verified by SHA-256. The table is in AGENT.md §5.2.

## Not established here

- **Cold-load latency.** `COLD_LOAD_COST` (4) is a stated guess; nothing here times a cold load.
- **A real Ollama queue-full.** The refusal cases used a stand-in; what Ollama 0.33.2 itself
  answers when its queue is full was not provoked.
- **The live contrast with Camelid's own queue-full being re-placed** is covered offline
  (`a_camelid_queue_full_is_still_re_placed_under_mixed_mode`), not in this run.
- **A tool-result follow-up** carrying `role: "tool"` messages without a `tools` key is not detected:
  seeing it would mean reading the messages.
- **Alias lines** are read at startup; editing them needs a restart.
