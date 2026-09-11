# Configuration Guide

Last updated: 2026-07-28

This guide documents Camelid's current local configuration reality without pretending every workflow is fully automated.

## Toolchain expectations

### Rust / Cargo

Camelid currently requires Rust/Cargo 1.87+.

On hosts where `/usr/bin/cargo` is older than the required toolchain, prefer one of these paths:

```bash
source "$HOME/.cargo/env"
```

or:

```bash
scripts/with-rustup-cargo.sh build --release --bin camelid
scripts/with-rustup-cargo.sh test --all-targets --all-features
```

### Node / npm

The frontend expects a working Node.js + npm install. Use `npm ci` when you want a reproducible install from the committed lockfile.

## Backend runtime defaults

The common local backend bind address in repo docs is:

```text
127.0.0.1:8181
```

Typical start command:

```bash
target/release/camelid serve --addr 127.0.0.1:8181
```

Appliance-style start command:

```bash
target/release/camelid serve --model /path/to/model.gguf
```

That startup path loads the model immediately and applies the default `auto` execution profile for the current host. Use `CAMELID_PROFILE=safe|auto|experimental|debug` when you need to change planner behavior; keep lower-level experiment env vars as developer overrides rather than the primary user workflow.

### Prompt-prefix cache: partial hits on the Metal lane

A partial prompt-prefix-cache hit resumes a cached session at a non-zero KV position, and
the batched Metal prefill only builds a cache from empty — so the divergent suffix would
fall to the CPU dense forward. Measured on an M4 / 16 GiB with
Llama-3.2-3B-Instruct-Q4_K_M and a ~500-token prompt, taking that hit cost 20.79 s against
1.33 s for a cold miss, and returned different tokens than a cold prefill of the same
prompt. Camelid therefore declines a partial resume when the batched Metal prefill would
otherwise have applied, and prefills the whole prompt on the GPU instead. Exact hits, and
every non-Metal session, are unaffected.

`CAMELID_METAL_PREFIX_PARTIAL_RESUME=1` restores the old unconditional resume. It exists
so the measurement can be reproduced against a single binary
(`qa/evidence-bundles/metal-partial-prefix-hit-20260909/`), not as a tuning knob.

### Trading prefill speed for exactness on K-quant models (macOS/Metal)

`CAMELID_METAL_KQUANT_ATTN_MM=1` admits a K-quant model's attention to the
attention-as-matmul prefill. It is **off by default, deliberately** — this is a choice
about output, not a feature waiting on qualification, so it is documented here rather than
left to be discovered.

Measured on an M4 / 16 GiB with Llama-3.2-3B-Instruct-Q4_K_M and a 2351-token prompt
(hardware GPU-busy per stage): prefill attention **5519 ms → 381 ms**, total prefill
**9511 ms → 4372 ms** — a **2.18x** cut in time to first token. Every non-attention stage
matches within 1 ms, so the flag moves attention and nothing else.

The cost is that attention-as-matmul stages K/Q scores as half, and on a K-quant model's
flatter logits that can move greedy output. Across five prompt shapes at greedy/128
tokens, run twice: four token-identical, one divergent (prose at 595 prompt tokens, first
differing at generated character 238, reproducing byte-for-byte). Divergence is occasional
and content-dependent, and deterministic rather than flaky.

Set it when time to first token on long prompts matters more than matching the exact lane
token for token; leave it unset when it does not. It additionally requires
`CAMELID_METAL_KQUANT_MM` (already on by default), and does nothing on its own.
Receipts: `qa/evidence-bundles/metal-kquant-attn-mm-20260909/`.

### Trading prefill for KV memory on Q8_0 models (macOS/Metal)

A Q8_0 model keeps an F32 resident KV cache by default. `CAMELID_METAL_KV_DTYPE=f16`
switches it to a half primary, which halves the KV footprint.

Measured on an M4 / 16 GiB with Llama-3.2-3B-Instruct-Q8_0: **597 MB saved** at ~4200
positions (6554 MB → 5957 MB physical footprint), against **~11% slower prefill** on a
2900-token prompt (4.53 s → 5.03 s). Decode is unchanged (29.9 vs 30.0 tok/s) and
generated text was identical on every prompt compared.

The saving scales with context and layer count, so it matters most where memory is the
binding constraint — a long-context 8B on a 16 GiB machine — and least on short prompts,
where there is little KV to halve and the prefill cost still applies.

This is off by default. Prior to the `all_q8` admission in `use_attn_mm` it was a much
worse deal (2.24x prefill, because an F16 primary silently lost the attention-as-matmul
lane); the numbers above are on a tree that has it. Receipts:
`qa/evidence-bundles/metal-q8-f16-kv-primary-20260909/`.

## Production HTTP policy

Anonymous loopback serving remains the default. A non-loopback address has to answer two separate
questions, because they are two separate risks and answering one does not answer the other:

| | anonymous | authenticated |
|---|---|---|
| **cleartext** | refused: needs both acknowledgements | refused: needs `--tls-cert`/`--tls-key` or `--allow-cleartext-remote` |
| **TLS** | refused: needs `--api-key`, `--api-key-file`, or `--allow-unauthenticated-remote` | serves |

Loopback is unaffected and needs neither. This is the same policy `camelid fabric serve` applies,
for the same reason: a key travels on every request, so over cleartext the credential that is meant
to protect a routable bind is itself given away, along with every prompt and completion.

The recommended production shape is therefore TLS plus a key. Prefer a key file because a literal
command-line key can be visible to other local users:

```bash
target/release/camelid serve \
  --addr 0.0.0.0:8443 \
  --api-key-file ./camelid-api.key \
  --tls-cert ./server-cert-chain \
  --tls-key ./server-private-key \
  --cors-origin https://chat.example
```

If TLS is terminated in front of Camelid — a reverse proxy, a service mesh, an SSH tunnel — then the
hop Camelid itself serves is still cleartext, and it has no way to know what is in front of it. Say
so explicitly:

```bash
target/release/camelid serve \
  --addr 0.0.0.0:8181 \
  --api-key-file ./camelid-api.key \
  --allow-cleartext-remote
```

API clients can send either `Authorization: Bearer <key>` or `X-API-Key: <key>`. Health and embedded
same-origin UI assets remain public; API routes, including `/metrics`, require the key. CORS is
disabled by default and accepts only explicit `http://` or `https://` origins—wildcard and `null`
origins are refused. The bundled browser UI stores an entered key in that browser and injects it
only for the configured API origin. A fresh browser that reaches an authenticated listener opens
the Settings credential field; a wrong key remains there instead of reporting the server offline.

### Optional GitHub quota credential for Web Auto

`CAMELID_WEB_GITHUB_TOKEN` optionally supplies a GitHub token to the bounded
`/api/web/research` preflight. Camelid sends it only on its own repository
metadata/tree/README requests whose current hop is exactly HTTPS
`api.github.com`; it is never forwarded to arbitrary linked API URLs, GitHub
raw content, search providers, redirects outside those managed request shapes,
or the model. Private
repositories remain outside public-web research even when the token can see
them. Fetched public responses are held only in a short, bounded in-memory
TTL/ETag cache; generated answers are never cached by this feature.

### Trusted-LAN browser Chat

Create the LAN credential on the laptop first:

```bash
camelid lan-key
```

The command creates a 256-bit key under the platform app-data directory, or displays the existing
key without changing it. Share the displayed key directly with the phone user and treat it like a
password. Do not put it in a URL, screenshot, issue, or chat message. Run `camelid lan-key --rotate`
only when you intend to invalidate every browser holding the old key.

`--lan-chat-only` exposes the minimum authenticated backend surface used by the embedded Chat UI.
It permits Chat and switching among direct regular `.gguf` files already present in the configured
models directory. The switch request carries an explicit filename; absolute paths, traversal,
prefixed paths, symlinks/reparse points, missing files, and non-GGUF files are rejected before any
model transition. Downloads, deletion, arbitrary-path loading, unload-only operations, runtime
controls, Workspace, Responses/Conversations, metrics, and every unknown protected route remain
refused. Missing or wrong credentials still get `401` before route scope is considered. Bind the
laptop's specific private address rather than every adapter:

```bash
target/release/camelid serve \
  --addr <LAPTOP-LAN-IP>:8181 \
  --model /path/to/model.gguf \
  --api-key-file ./camelid-api.key \
  --lan-chat-only \
  --allow-cleartext-remote \
  --no-open
```

Open `http://<LAPTOP-LAN-IP>:8181` on the phone, then enter the same key under Settings. Both devices
use inference on the laptop; the phone only runs the browser interface. The Chat model selector can
switch to another local GGUF from that host's configured model directory. No CORS flag is needed for
the embedded same-origin UI. Permit the port only on the operating system's private-network firewall
profile.

This mode authenticates but does not encrypt plain HTTP. `--allow-cleartext-remote` is the explicit
acknowledgement required for that direct-LAN shape; it does not protect the credential, prompts, or
responses from anyone on the path. Use it only on a trusted private LAN, or put the listener behind
an encrypted private transport. Ordinary Chat conversations still live in each browser's storage,
so laptop and phone history do not synchronize in this phase.

Both acknowledgements also read from the environment, as `CAMELID_ALLOW_UNAUTHENTICATED_REMOTE` and
`CAMELID_ALLOW_CLEARTEXT_REMOTE`.

### Private cross-network browser Chat with Tailscale

Use this path when the browser and the Camelid host are on different networks. Install Tailscale
1.52 or newer on the host and the phone, sign both devices in, and grant the phone access to the
host in the tailnet policy. Camelid uses Tailscale Serve, not Tailscale Funnel: the generated HTTPS
URL is reachable only by devices permitted through Tailscale.

Keep Camelid on loopback. In the first terminal, provision the key and start the restricted surface:

```bash
camelid lan-key
camelid serve \
  --addr 127.0.0.1:8181 \
  --model /path/to/model.gguf \
  --api-key-file <PATH-PRINTED-BY-LAN-KEY> \
  --lan-chat-only \
  --no-open
```

In a second terminal, publish that verified listener through private tailnet HTTPS:

```bash
camelid remote-chat start
```

The first start may take up to two minutes while Tailscale provisions HTTPS. That longer allowance
applies only to Serve creation; version, status, verification, and stop commands remain bounded to
30 seconds.

The command prints a URL such as `https://<DEVICE>.<TAILNET>.ts.net/`. Open it on the phone and enter
the same Camelid key under Settings. The key is still required even though Tailscale authenticates
the device. The key is never passed to the Tailscale CLI, placed in the URL, or written into the
Serve configuration.

The command fails closed unless all of these are true:

- the backend address is exactly IPv4 loopback (`127.0.0.1`);
- `/v1/health` reports `api_surface=lan_chat_only`, which also proves Camelid resolved an API key;
- Tailscale is connected and reports an online device DNS name;
- HTTPS port 443 is unused or already maps `/` to this exact Camelid listener; and
- no Tailscale Funnel mapping is enabled on HTTPS port 443.

After creating a mapping, Camelid reads Tailscale's Serve configuration back and removes the new
mapping if the expected private proxy is not present. It never replaces another service already on
port 443. No CORS flag is needed: the embedded UI and API remain same-origin at the Tailscale URL,
while Tailscale terminates HTTPS and forwards only to loopback.

Inspect or remove the mapping with:

```bash
camelid remote-chat status
camelid remote-chat status --json
camelid remote-chat stop
```

`start` is idempotent for the exact Camelid mapping. `stop` removes only that root mapping and leaves
unrelated Tailscale Serve configuration alone. The `--bg` mapping survives a Tailscale restart, but
the Camelid server must also be running for the URL to answer. If Tailscale is installed outside its
standard location, pass `--tailscale-bin <ABSOLUTE_PATH>` or set `CAMELID_TAILSCALE_BIN`.

This is private cross-network access, not an anonymous public link. A recipient needs both Tailscale
access to the host and the Camelid API key. Funnel and direct router port forwarding are outside this
workflow. Browser-local Chat histories still do not synchronize between devices.

Direct TLS is optional and requires a PEM certificate chain and private key together:

```bash
target/release/camelid serve \
  --addr 0.0.0.0:8443 \
  --api-key-file ./camelid-api.key \
  --tls-cert ./server-cert-chain \
  --tls-key ./server-private-key
```

Resource ceilings are resolved once at startup. Their CLI names and environment aliases are:

| Limit | Default | Environment |
|---|---:|---|
| `--max-request-body-bytes` | 16 MiB | `CAMELID_MAX_REQUEST_BODY_BYTES` |
| `--max-prompt-tokens` | 131,072 | `CAMELID_MAX_PROMPT_TOKENS` |
| `--max-generation-tokens` | 8,192 | `CAMELID_MAX_GENERATION_TOKENS` |
| `--max-download-bytes` | 64 GiB | `CAMELID_MAX_DOWNLOAD_BYTES` |

`--lan-chat-only` also has the environment alias `CAMELID_LAN_CHAT_ONLY=1` and always requires
`CAMELID_API_KEY` or `CAMELID_API_KEY_FILE`. The command line refuses it alongside
`--allow-unauthenticated-remote`. Setting both through the environment is not refused there, but it
changes nothing: this mode has no anonymous form, so the listener still will not start without a key.

`GET /metrics` exposes bounded-name Prometheus counters and gauges for HTTP/generation latency,
prompt/decode tokens, prompt and weight cache outcomes, engine queue/slot progress, process RSS,
and CUDA VRAM. It contains no model-path, prompt, API-key, or per-user labels.

## Fabric proxy policy

`camelid fabric serve` puts one HTTP address in front of several independent nodes. It defaults to
`127.0.0.1:8282`. A non-loopback address has to answer two separate questions, because they are two
separate risks, and answering one does not answer the other:

| | anonymous | authenticated |
|---|---|---|
| **cleartext** | refused: needs both acknowledgements | refused: needs `--tls-cert`/`--tls-key` or `--allow-cleartext-remote` |
| **TLS** | refused: needs `--api-key`, `--client-keys`, or `--allow-unauthenticated-remote` | serves |

Loopback is unaffected and needs neither.

```bash
target/release/camelid fabric serve \
  --node a=host-a --node b=host-b \
  --addr 127.0.0.1:8282
```

### Letting a browser read the proxy

The web UI is served from the engine's address (`camelid serve`, `:8181` by default) or from a dev
server (`:5173`), never from the proxy's, so a browser withholds every answer the proxy gives it —
the fabric view and the Compare screen included — unless the proxy names that origin:

```bash
target/release/camelid fabric serve \
  --node a=host-a --node b=host-b \
  --cors-origin http://127.0.0.1:8181 --cors-origin http://127.0.0.1:5173
```

`--cors-origin` is repeatable and validated exactly as the engine validates its own: a wildcard,
`null`, or an origin carrying a path or a query is refused at startup, before anything is bound. With
no origin named — the default — the proxy sends no CORS header at all and answers `OPTIONS` exactly
as before — a preflight to `POST /v1/fabric/compare` then gets `405`, and the browser never sends the
request. A named origin's preflight is answered by the proxy itself; the request that follows still
needs the client key, and a refusal of it carries the allow-origin header so the page can say why. The allowed
request headers are `content-type`, `authorization` and `x-api-key`. The flag deliberately does not
read `CAMELID_CORS_ORIGINS`: that variable configures the engine, and one value must not open both.

### Which engine a node runs

A node is written `LABEL=[ENGINE://]HOST[:PORT]`. The engine is optional and defaults to `camelid`,
so every specification written before the fabric could read more than one still means exactly what
it did.

| Engine | Scheme | Default port | Placed on |
|---|---|---:|---|
| Camelid | `camelid://` (or omitted) | 8181 | yes |
| Ollama | `ollama://` | 11434 | **no — read and reported only** |
| LM Studio | `lmstudio://` | 1234 | **no — read and reported only** |

```bash
target/release/camelid fabric status \
  --node win=127.0.0.1:8181 \
  --node studio=ollama://workstation.local \
  --node desk=lmstudio://192.0.2.13
```

The engine is **declared, never detected**. Probing an address and inferring the engine from what
answered would make a typo look like a different product, and would present this fabric's bearer
token to whatever was listening. An unrecognised scheme is refused at parse time with the list of
engines the build knows.

A foreign node is reported — every model it holds, which one is loaded where that is knowable, and
whether it answered — but the fabric will not send work to it. Placement ranks nodes on reported
load, re-places a typed queue-full refusal on a sibling, and honours session affinity; neither
foreign API answers those three, so routing to one would mean ranking a node whose load was never
measured, relaying a refusal that might have been retried, and claiming a warm prefix nobody
attested. Such a node therefore reports no load at all rather than a zero, `fabric status` prints
`?` in that column, and the health payload omits the load fields entirely. Models only such a node
holds are deliberately absent from `GET /v1/models`: advertising a model the fabric would then
refuse to route is worse than not advertising it.

LM Studio requires 0.3.6 or newer, which is where `GET /api/v0/models` — the only endpoint this
fabric reads — was introduced. That listing is also the only thing it reads, so a node is described
in one request. LM Studio publishes no version endpoint, so its version is reported as unknown
rather than guessed from a runtime name that describes an inference backend instead of the
application. A model it has downloaded but not loaded is still listed as something it holds, and
`active_model_id` is filled in only when exactly one model is loaded; if any model reports a state
this build does not recognise, residency becomes unknown rather than being narrowed to the states it
did understand.

### What an engine can be asked, and how we know

Each node carries a `capabilities` block in `GET /v1/health`, and each entry carries the provenance
of its own answer:

| Provenance | Meaning |
|---|---|
| `measured` | Exercised against **this exact engine version** and recorded |
| `declared` | Follows from the documented API surface — the endpoint that would carry it does not exist |
| `not_probed` | Nobody has checked; `supported` is `null` |

`not_probed` is never rendered as a no. A capability nobody measured is unknown, and a build that
answered "unsupported" there would be inventing a result. A measurement is bound to the version
string the node reported, so a neighbouring version inherits nothing from it — this is why an Ollama
node reporting `0.33.3` shows `not_probed` for tool calls even though `0.33.1` was measured.

`placement_blockers` lists, in the operator's words, the capabilities whose absence is why a node is
not placed on. It is derived from the same capability entries that appear in the matrix, so the
reason shown can never drift from the verdict it explains.

Every answer carries `x-camelid-fabric-engine` alongside `x-camelid-fabric-node`, in every routing
mode, so a client never has to know which engines a fabric places on to find out which one served it.

### Placing work on an engine this fabric only reads

Off by default: `fabric route` and `fabric run` place on Camelid nodes only, and a healthy foreign
node is reported and refused. `--allow-mixed-engines` turns that off, and it is worth being precise
about what it accepts, because none of it goes away by being asked for.

A foreign engine publishes no load, cannot distinguish a full queue from a failure, and cannot
attest that a session's prefix is still warm. Mixed placement accepts those three, each with a
stated consequence:

| Consequence | What the fabric does |
|---|---|
| No load to rank on | The node is charged a fixed cost rather than treated as idle. Charged nothing it would outrank every node that honestly reported work and win **every** placement, concentrating load exactly where the fabric can see least. The cost is a deliberate guess, which is why this is opt-in. |
| Tools | A request carrying tools never lands on a backend nobody has measured. The capability table answers `not_probed` for a foreign engine until somebody measures that exact version, and a vendor's documentation is not a measurement. |
| Affinity | A sticky request for a node that cannot attest warmth is **refused** — `400 affinity_unsupported` through the proxy — not quietly served there. Honouring affinity nothing attests is a slower random choice wearing the word "affinity", and the caller would never learn the difference. Retry without a sticky node to be placed normally. |

```bash
target/release/camelid fabric route \
  --node win=127.0.0.1:8181 \
  --node studio=ollama://127.0.0.1:11434 \
  --allow-mixed-engines --model llama-3.2-1b-instruct
```

Two things the flag never changes. A node must still be **ready**: mixed placement relaxes which
engines are placed on, not whether a down or still-loading node can take work, so a fabric with
nothing ready refuses exactly as it would without the flag — retryably, never as a permanent
`model_not_found`. And the fabric's bearer (`--bearer` or `CAMELID_API_KEY`) is only ever sent to a
**Camelid** node: it is a Camelid API key, and a foreign engine that takes work under this flag
never receives it, on any path.

Every eligibility rule above is written against **capabilities, not engine names**, so a backend
added later is a new row rather than a new argument — and placement still contains no engine name
anywhere.

### Comparing two nodes

`fabric compare` asks two named nodes the same question and reports what differed. It is
**measurement, not placement**: both nodes are named rather than chosen, nothing fails over, and a
foreign engine may be compared even though the fabric will not route to it — reading a node was
never the thing we refused.

```bash
target/release/camelid fabric compare \
  --node win=127.0.0.1:8181 \
  --node studio=ollama://127.0.0.1:11434 \
  --left win --right studio \
  --model llama-3.2-1b-instruct \
  --prompt 'What is 7 plus 5?'
```

The same comparison is available to the WebUI as `POST /v1/fabric/compare` on the proxy. Unlike
`/v1/health` that route *causes work* — it makes two machines generate — so it is behind the client
key, the prompt is capped, and repetitions and token counts are clamped.

**It never says which answer is correct.** It reports difference, provenance and reproducibility.
Four rules decide what it is allowed to conclude:

| Rule | Consequence |
|---|---|
| Each side is run more than once | A difference between two nodes means nothing until each node has been shown to agree with itself. A side that does not repeat its own answer makes the comparison `not_attributable`, **and the diff is withheld** — rendering one would invite exactly the reading the verdict refused. |
| Model identity is checked first | Two nodes serving different models can differ for the most boring reason there is. That is `different_models`, never divergence. Each side records the id it was asked for (`model`) and the id its own answers named (`reported_model`, `null` when they named none); a side that answered as a model other than the one it was asked for makes the comparison `different_models` whatever the operator asserted. An equal id on two *different* engines is listed as an uncontrolled `model identity`, because each engine resolves a name by its own rules and an equal name is not equal weights. |
| Asked is not applied | LM Studio's documented completion API has no seed parameter, so a comparison against it is unseeded however the flag was set. The plan and the per-side reality are reported separately, and the uncontrolled parameter is named. |
| Correctness is not awarded | No verdict, reason or rendered line ranks the two sides. |

`--repeat 1` is permitted and yields no attributable verdict, by design: one run per side never
tested self-consistency, which is a different thing from having tested it and found none.
`--repeat` takes 1 to 5 and `--temperature` 0 to 2, and anything else is refused rather than
clamped, so the plan a receipt records is the plan that ran. The proxy clamps repetitions to the same
five and refuses a temperature outside that range with 400.

Each generation is bounded by `--timeout-s` on the CLI and by `--forward-timeout-s` on the proxy's
`POST /v1/fabric/compare`; the proxy's `--timeout-ms` bounds only the status probe and the template
read. A comparison that could not run answers **400** when the request was at fault — an unknown
label, the same node twice, a model the node does not hold — and **502** when a node was: not
serving, or failing to generate.

The diff keeps line terminators. Every line carries `eol` — `lf`, `crlf`, or `none` for a final line
with no newline — beside its `text`, which never contains the terminator, so two answers that differ
only in a trailing newline or in CRLF against LF show a changed line rather than an unchanged one
under a `divergent` verdict. The terminal draws them as `⏎`, `␍⏎` and `(no newline at end)`.

### Telling the fabric that two names are the same model

Engines do not agree on what to call the same weights. Ollama suffixes `:latest`, LM Studio does
not, Camelid uses its catalog id — and none of them publishes a digest this fabric could compare
(Ollama's `/api/tags` carries a *manifest* digest, not the GGUF's). Exact-match identity therefore
refuses a cross-engine comparison even when the two nodes really are serving the same file.

This build will not guess. Stripping a `:latest` suffix to make two ids match is an inference, and
it would be wrong the first time somebody has two genuinely different builds under similar names.
Instead an operator declares it, and the claim travels with every result that rests on it.

Declare it once, in the nodes file:

```
studio=ollama://127.0.0.1:11434
desk=lmstudio://127.0.0.1:1234

# what each of them calls the same weights
alias llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest
alias llama-3.2-1b-instruct=desk:llama-3.2-1b-instruct
```

Then one id works everywhere:

```bash
target/release/camelid fabric compare --nodes-file fabric.nodes \
  --left studio --right desk \
  --model llama-3.2-1b-instruct \
  --prompt 'What is 7 plus 5?'
```

`--model-alias CANONICAL=LABEL:LOCAL` takes the same declaration on the command line, repeatably,
for a one-off. `--left-model` / `--right-model` still override a single side and win over anything
declared, being the more specific statement of the same claim.

Details that matter:

| | |
|---|---|
| `LOCAL` is everything after the **first** colon | A model id may contain colons — `llama-3.2-1b-instruct:latest` is the case this exists for. A node label may not. |
| No alias means no change | Resolution falls through to the id as given, so an existing fabric behaves exactly as it did and only the names that genuinely differ need declaring. |
| One node, one name per model | A second declaration for the same pair is refused rather than silently overwriting the first. |
| Aliases are read **once, at startup** | Nodes hot-reload; aliases do not. An alias is a claim about what weights *are*, and one that changed underneath a running comparison would make its receipt unreproducible. |

Whenever a declaration is used and the two sides end up asking for different names, the result
carries `model_identity: asserted_by_operator`, lists `model identity` among the uncontrolled
variables, and prints `MODEL IDENTITY ASSERTED, NOT VERIFIED`. Nothing here checks that the weights
match, and the output never pretends otherwise.

Where the engine exposes it, the chat template each backend applied is captured too — usually the
actual explanation for a divergence:

| Engine | Template source |
|---|---|
| Camelid | `GET /props` → `chat_template` |
| Ollama | `POST /api/show` → `template` |
| LM Studio | **none** — its documented API exposes no prompt template, reported as such rather than as an empty one |

An exposed proxy therefore looks like this:

```bash
target/release/camelid fabric serve \
  --node a=host-a --node b=host-b \
  --addr 0.0.0.0:8282 \
  --api-key-file ./camelid-proxy.key \
  --tls-cert ./proxy-cert-chain \
  --tls-key ./proxy-private-key
```

The certificate is read while the arguments are still being resolved, so one that is missing or is
not a PEM pair stops the proxy before it binds or announces anything. `--allow-cleartext-remote`
(or `CAMELID_ALLOW_CLEARTEXT_REMOTE`) serves without one, and should only be used behind something
that terminates TLS itself.

Three limits are deliberate and worth knowing before you deploy it:

- `--api-key` (or `--api-key-file`, or `--client-keys` below) is the key the proxy requires *from its
  clients*. It deliberately reads no environment variable: on this command `CAMELID_API_KEY` already
  names the token sent to nodes, and one value must not silently configure both directions. Without
  one of them, `--allow-unauthenticated-remote` is the only way to bind a routable address, and it
  should only be used behind something that does authenticate.
- `--bearer` (or `CAMELID_API_KEY`) is the token the proxy presents *to its nodes*, the same as
  `fabric status|route|run`. A node started with `--api-key` needs it: `/v1/health` is exempt from
  that node's auth, so without a token the node observes as ready and then answers every forwarded
  request with 401.
- Client-facing TLS (`--tls-cert`/`--tls-key`) and node-facing TLS (`--node-tls-ca`) are independent.
  Configuring one does not protect the other hop.

### Encrypting the node hop

Every Fabric command applies the same node transport policy to health probes, buffered requests,
streaming requests, failover attempts, and nodes added later through `--nodes-file`.

| Node transport | Required option | Reachable nodes |
|---|---|---|
| CA-pinned TLS | `--node-tls-ca <PATH>` | any address whose certificate SAN matches the configured host/IP |
| tunnel/local cleartext | none | loopback resolutions only |
| direct cleartext | `--allow-cleartext-node-transport` | any resolution, explicitly unencrypted |

Without extra flags, cleartext node transport is restricted after DNS resolution to loopback
addresses. That preserves local nodes and encrypted tunnels while refusing a hostname that resolves
only to another machine. Direct cleartext requires the explicit
`--allow-cleartext-node-transport` acknowledgement (or
`CAMELID_ALLOW_CLEARTEXT_NODE_TRANSPORT=1`). This is separate from
`--allow-cleartext-remote`, which controls the client-facing proxy listener.

Named nodes use the host operating system's resolver, preserving its hosts-file, split-DNS, VPN, and
multicast-discovery behavior. Resolution runs in a fixed-size process-wide pool and shares the
command's deadline, so a stuck resolver cannot hold the calling request indefinitely or grow one
thread per request. IP literals bypass resolution and remain usable even while DNS is unavailable.

For direct connections between machines, issue each node a certificate from one private CA. The
certificate SAN must match the exact host or IP literal in `LABEL=HOST[:PORT]`; a DNS name is not
accepted merely because it resolves to the certificate's IP. Start each node with the normal engine
TLS and API-key options, then pin that CA at the Fabric:

```bash
# On each node (use that node's own certificate and key):
camelid serve \
  --addr <NODE-HOST>:8181 \
  --model /path/to/model.gguf \
  --api-key-file ./node-api.key \
  --tls-cert ./node-cert-chain \
  --tls-key ./node-private-key \
  --no-open

# On the proxy host:
export CAMELID_API_KEY="$(cat ./node-api.key)"
camelid fabric serve \
  --node a=node-a.example:8181 --node b=node-b.example:8181 \
  --node-tls-ca ./fabric-node-ca \
  --addr 127.0.0.1:8282
```

The CA bundle is loaded before the first probe. Missing, empty, malformed, or unusable bundles stop
the command. TLS verifies the node certificate and host/IP SAN before the HTTP request head is sent,
so a failed handshake sends neither the bearer nor prompt bytes. The existing bearer authenticates
the Fabric to the node; mutual TLS is not a second required client-identity system.

A tunnel remains a supported alternative. Bind each node to loopback, forward a distinct local port
to it with SSH, Tailscale, or another authenticated encrypted transport, and name the local tunnel
endpoints (for example `a=127.0.0.1:18181`). No cleartext acknowledgement is needed because the
Fabric sees only loopback; the tunnel owns encryption and remote identity.

One Fabric uses one transport posture for all nodes: either one pinned CA or guarded cleartext. This
prevents a node from silently downgrading while a node file reloads. CA rotation or changing posture
requires restarting the Fabric; adding or removing nodes does not.

### Placement modes

`--mode throughput` is the default and preserves the established policy: choose the smallest
`max(node-reported in-flight jobs, requests this proxy has reserved)`, breaking ties by node label.
`--mode affinity` prefers the node named by `x-camelid-fabric-sticky` and falls back to that same
least-load rule if the node is unavailable or no longer serves the requested model. A sticky header
requests affinity whatever default mode the proxy was started with.

`--mode completion-time` is opt-in. It learns an exponentially weighted service time from clean,
successful requests completed during this proxy process, then minimizes:

```text
(observed-or-reserved load + 1) * learned service time
```

The estimate is scoped to the node's label, address, backend and version; the model; the API route;
streaming versus buffered delivery; and coarse power-of-two buckets for request bytes and requested
output tokens. That prevents a node which happened to receive a longer prompt or generation from
being labelled intrinsically slow. Only a completion placed while the node's observed-or-reserved
load was zero is sampled. A busy completion includes unknown queueing from other workload classes,
so dividing its wall time by total in-flight work would invent a service time the proxy did not
observe. No prompt or response content is retained.

Every eligible node needs five successful completions in the same workload class before an estimate
may decide placement. Until then the proxy uses least-load, rotating equal-load cold candidates by
least-recent selection so sequential traffic samples all of them. The response header says
`x-camelid-fabric-reason: LeastLoaded` while cold and `EstimatedCompletion` once learned timing made
the decision.

Only a clean, queue-free success is a speed sample. A node-attributable 5xx, unreadable answer,
transport failure, or truncated stream invalidates that node's estimate for the workload
immediately. A queue-full refusal does not: it says the load bound worked, not that service became
slow. Client cancellation does not either, because it says nothing about the node. A stream is
sampled only at clean EOF, and not if the bounded relay channel filled behind a slow client. A
request carrying a sticky header is not sampled either: affinity, rather than this policy, chose its
node.

Estimates older than five minutes become cold and have to be sampled again. They are memory-only,
bounded to 1,024 node/workload entries, and disappear when the proxy restarts. `fabric route` and
`fabric run` reject `completion-time`: as one-shot commands they have no resident history and could
only pretend to make a learned decision.

This mode changes placement, not node capacity or admission. A sole eligible node still receives the
request, and a genuinely full node still owns its typed 503. Treat performance improvement as a
measurement question: use a paired, interleaved fabric campaign before making a deployment claim.

### Naming clients, and cutting one off

`--api-key` gives every client the same secret, so the access log can only say that *someone*
authenticated, and withdrawing access from one client means changing the key for all of them and
restarting. `--client-keys <PATH>` replaces it with a set of named clients:

```json
{"clients": [
  {"name": "laptop", "key": "..."},
  {"name": "ci",     "key": "..."}
]}
```

It conflicts with `--api-key` and `--api-key-file`, because two different answers to "who may call"
must not be configurable at once. A set answers the *authentication* question in the table above
exactly as a single key does, and like a single key it answers only that one: an exposed proxy still
needs a certificate or `--allow-cleartext-remote` as well.

Each key is validated by exactly the rules `--api-key` applies, and a presented credential is read
from the same two headers (`Authorization: Bearer` or `X-API-Key`) and compared the same way. A set
is refused at startup — rather than started open — if it is unreadable, is not valid JSON, lists no
clients, names a client twice, or gives two clients the same key.

The name, never the key, is written to every access-log line that client causes, as `client_name`.

Deleting an entry revokes that client without a restart and without disturbing anyone else. The file
is re-read at most once a second, and only when its size or modification time has changed, so this
costs nothing per request. If a re-read fails — which is what the moment of an atomic replace looks
like — the previous set stays in force, so an ordinary edit cannot become an outage.

That fallback has a cost worth stating plainly: **a revocation written into a file that does not
parse, or deleting the key file altogether, does not revoke anybody.** The previously loaded set goes
on being served until the file is readable again or the proxy is restarted — and a restart refuses to
come up at all while the file is unusable. So the proxy prints to stderr when a re-read starts
failing, and again when it recovers, rather than relying on `RUST_LOG` being set:

```
fabric: could not reload client keys: <why>. The previous set of 2 clients is still in force, so a
revocation written here has NOT taken effect.
```

### Adding and removing a machine while it runs

`--node` is read once at startup, so changing the fabric means restarting the proxy — which drops
whatever everyone else has in flight. `--nodes-file <PATH>` puts the set in a file instead and
re-reads it as it changes:

```
# the machines I place on
desktop=192.0.2.10:8181
laptop=192.0.2.11:8181
```

One `LABEL=HOST[:PORT]` per line, exactly the syntax `--node` takes, parsed by the same code. Blank
lines and whole-line `#` comments are ignored, so a machine can be taken out for an hour by
commenting it rather than deleting what you wrote. It conflicts with `--node`, because "which
machines am I placing on" must have one answer.

Adding a line makes that machine placeable; removing one stops it being placed on. Neither restarts
the proxy, and **neither disturbs a request already running on the machine you removed** — that
request is answered by the node it was placed on, and only new placements see the new set.

The file is re-read at most once a second, and only when its size or modification time has changed.
A change is only acted on if the set actually differs, so saving the file without editing it costs
nothing.

Two refusals are deliberate. A file that cannot be read, is not valid, or **names no nodes at all**
is refused at startup rather than started on an empty fabric, which would answer every request with
503 while looking as though it had started. And if a *re-read* fails the same way, the previous set
stays in force — a file is usually replaced by writing a new one and renaming it over the old, so a
read landing mid-swap sees a partial or empty file, and emptying the fabric on that would turn an
ordinary edit into a total outage.

That fallback has a cost worth stating plainly: **a change written into a file that does not parse,
or deleting the node file altogether, does not change anything.** The previously loaded set goes on
being placed on until the file is readable again or the proxy is restarted — and a restart refuses to
come up at all while the file is unusable. A machine you meant to take out is still taking requests.
So the proxy prints to stderr when a re-read starts failing, and again when it recovers, rather than
relying on `RUST_LOG` being set:

```
fabric: could not reload the node file: <why>. The previous set of 2 machines is still being
placed on, so a change written here has NOT taken effect.
```

It is printed once each way, not once per re-read, so a file left broken does not emit a line a
second for as long as it stays that way.

The proxy re-probes its nodes at most once per `--observation-max-age-ms` (500 by default) rather
than once per request. Inside that window its view can be wrong, so a request placed on a node that
has gone since is placed again on another node serving the same model, up to `--max-forward-attempts`
(2 by default) nodes in all. A request is only ever sent twice when the first node was never reached
and so cannot have started it: a node that accepted the request and then failed, or that failed
part-way through a stream, ends the request with 502 rather than risking a second generation. Set
`--max-forward-attempts 1` to fail on the first node instead. Every answer carries
`x-camelid-fabric-node`, `x-camelid-fabric-reason` and `x-camelid-fabric-attempts`, so a client can
tell a first-choice placement from a failover.

A client that hangs up takes its request's work with it. The proxy notices the connection going and
hangs up on the node in turn, which that node reads as its own client leaving — it stops generating
within one decode step. So a request nobody is waiting for gives back the node's generation slot
instead of holding it for the rest of `--forward-timeout-s`, which matters because a node runs one
generation at a time. It is not instantaneous: the check happens between socket operations, so it
takes effect within about 100 ms while the proxy is waiting on a node. A probe round already under
way is not interrupted, but it is bounded by `--timeout-ms` and costs a node a health read rather
than its generation slot.

Request bodies are bounded at the same 16 MiB default the node itself uses. A streaming request is
relayed as it arrives. `--forward-timeout-s` bounds the wait for the first response head and each
later silent gap; every event resets the latter, so it is not a cap on total stream duration.
`fabric run`, which returns one complete answer, refuses `stream: true` with 400 instead.

### What the proxy serves

It places the engine's stateless inference routes, and answers discovery for the whole fabric:

| Route | Behaviour |
| --- | --- |
| `POST /v1/chat/completions` | Placed on a node serving the model named in the body. |
| `POST /v1/completions` | Same. |
| `POST /v1/embeddings` | Same. |
| `POST /v1/rerank`, `POST /v1/reranking` | Same. |
| `GET /v1/models` | The union of what every ready node is serving. |
| `GET /v1/models/{model}` | 200 exactly when a request naming that model would be placed. |

A route is placed only if any node serving the named model could answer it, because the client cannot
tell which node it reached and must not need to. That rules out the Responses and Conversations APIs:
they keep their items in a SQLite store on the node that served the request, so a follow-up landing on
another node would find nothing. Those routes are refused with **501** and a reason rather than
half-supported — send them to a node directly, where they work as documented. Any other route is a
**404** naming what is served, so a client can tell a wrong route from a wrong model. Both refusals
are behind the client key, if one is configured: an unauthenticated caller gets a 401 and learns
nothing about the fabric.

`stream: true` is read from the request rather than assumed from the route, so a client that sets it
on a route with nothing to stream still gets that route's complete answer instead of an empty stream.

## Mixtral diagnostic gate

The checked Mixtral 8×7B Q8 row has one-token evidence, but its longer continuation parity gate is
still open. Camelid therefore rejects `max_tokens > 1` for Mixtral by default instead of allowing a
very slow file-backed request to look hung. Controlled diagnostics must explicitly set
`CAMELID_MIXTRAL_LONG_GENERATION=1`.

MoE expert storage defaults to `CAMELID_MOE_EXPERT_STORAGE=file_backed`, preserving the existing
low-RAM path. `resident_q8` keeps the quantized expert blocks in RAM and selects experts through
zero-copy shared ranges. It is opt-in and fails before allocation unless live available RAM can
hold all owned expert tensors, peak decode/load scratch for the largest expert descriptor, and
Camelid's host headroom floor:

```bash
CAMELID_MIXTRAL_LONG_GENERATION=1 \
CAMELID_MOE_EXPERT_STORAGE=resident_q8 \
target/release/camelid serve --model /path/to/Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf
```

On PowerShell, set the two variables with `$env:NAME = "value"` before starting the server. Use
`scripts/diagnose-mixtral-watchdog.ps1` to capture health, slot progress, metrics, response, and
generated-index-9 expert diagnostics under `qa/local-artifacts/`.

## Frontend API base override

The frontend defaults to:

```text
http://127.0.0.1:8181
```

Override it for local dev/build with:

```bash
VITE_CAMELID_API_BASE=http://127.0.0.1:8181 npm run dev
```

You can also edit the API base in the UI while testing.

## Model-path guidance

Repo examples often use paths such as:

```text
models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf
```

These are example local paths, not a guarantee that the repo fetches or manages model files for you.

Recommended practice:

- keep local GGUFs outside version control
- use stable local paths during validation so commands and artifacts stay reproducible
- avoid documenting private absolute paths in public artifacts or docs

## Environment and local-shell assumptions

Current public docs assume:

- `cargo` resolves to a Rust 1.87+ toolchain
- `node` and `npm` are available for frontend work
- `llama-server` is in `PATH` only when you are running parity comparisons

Backend runtime knobs used during performance work:

- `CAMELID_GPU_TEMP_SAMPLING` controls the CUDA-resident Gumbel-max path for plain temperature sampling. It defaults to enabled after seeded device/reference and streaming validation, avoiding a full-vocabulary device-to-host copy and CPU sort on each sampled token. Set it to `0`, `false`, `off`, or `no` to force the CPU sampling fallback for diagnosis.
- `CAMELID_CUDA_RESIDENT_PREFILL_BATCHED` overrides the resident CUDA prefill policy. Q8_0 uses batched prefill by default; Q4_K/Q6_K keep the sustained-throughput winner (serial prefill) by default on the Windows/WDDM reference host. Set it to `1`, `true`, or `on` to exercise the parity-checked Q4_K/Q6_K batched kernels, or `0`, `false`, or `off` to force serial prefill for any quant lane.
- `CAMELID_CUDA_PREFIX_CONTINUATION` controls whether a CUDA-resident prefill reuses the KV rows a previous prefill already built for the same leading tokens. Default: enabled. Every turn of a conversation re-sends the whole history, and a KV row is a pure function of the token prefix that produced it, so rows the engine already holds for an identical prefix are exactly the rows the new prompt needs; the prefill starts after them instead of rebuilding from position 0. This is bit-identical, not an approximation, and it never leaves the GPU — unlike the prompt-prefix cache, which is bypassed on this lane because it reseeds GPU KV from f16-rounded host history. The reuse claim is dropped whenever anything else writes the cache (a reseed from host history, a tree compaction, a layer's cache being dropped, an SSM state reset), so those paths pay one cold prefill rather than continuing from rows the engine can no longer vouch for. Set it to `0`, `false`, or `off` to force a full prefill on every request, which is how the saving is A/B'd against the same binary. `CAMELID_RESIDENT_TRACE` prints how many positions each request reused.
- `CAMELID_CUDA_EAGER_KV_MIRROR` restores the eager GPU→host KV mirror that used to run after every CUDA-resident prefill. Default: off, meaning the CPU-side KV cache is mirrored back lazily by `ensure_cpu_kv_materialized` at the moment a CPU reader actually needs the history. The eager copy scales with the whole context rather than with the new tokens (`n_layers × n_kv × positions × head_dim × 2` elements, plus a host-side expansion), so once prefix continuation reduced prefill to the newly appended tokens it became the dominant cost of a follow-up turn — 62–74% of the reported prefill time on the RTX 3060 Laptop reference. Every consumer of the CPU KV history materializes on demand (the three CPU forward readers and `rollback_to_position`), so on the common path — GPU prefill, GPU decode, no fallback — the copy is never paid. Set it to `1`, `true`, or `on` to force the eager copy for A/B measurement or as an escape hatch. Output is identical either way; only when the work happens changes.
- `CAMELID_CUDA_KQUANT_BATCH_TOKENS` selects the requested Q4_K/Q6_K CUDA prefill tile size from `1` through `4` when batched K-quant prefill is explicitly enabled. Default: `2`; the runtime clamps it to the model dimensions and portable shared-memory budget. This remains a diagnostic tuning knob until a target GPU shows a sustained gain.
- `CAMELID_CUDA_PREFILL_BATCH_TOKENS` requests how many prompt tokens the CUDA-resident batched prefill processes per chunk. Unset, prefill uses the same chunk the batched layer stack uses for speculative verify, so the shipped path is unchanged. Any requested value is clamped to the largest chunk this model's batched GEMMs can stage inside the portable 46 KiB shared-memory budget, so it cannot produce a launch the driver refuses. This is a diagnostic tuning knob; chunk size is a separate lever from the flash attention kernel below and must be measured separately.
- `CAMELID_FLASH_PREFILL` enables the fused tiled flash prefill attention kernel for CUDA-resident prompt ingestion. Default: off, and **measurement says leave it off** on Ampere. Set it to `1` (or any value other than `0`, `false`, or `off`) to enable it. Opt-in only, prefill only; the default path and speculative verify keep the bit-identity contract and never take it.

  Measured on an RTX 3060 Laptop (sm_86) with Llama-3.2-3B-Instruct-Q8_0, one binary, ABBA-ordered with a thermal gate: prefill is **1.11x / 1.17x / 1.22x SLOWER** at 1424 / 3025 / 6024 prompt tokens — the penalty grows with context, which is the opposite of the point. The kernel was developed and reported on sm_89, and the register-vs-occupancy tradeoff behind this is architecture-sensitive, so it may behave differently there; but no committed evidence isolates this flag on any device. (The "20.95x" and "TTFT 2,140 ms → 70.8 ms" figures in `improvements.md` are GPU-resident vs CPU, not flash-on vs flash-off.)

  The online-softmax reassociation is applied **per layer**, so it is not token-parity in general, despite earlier wording here that said it was. Same host, greedy, three distinct prompts: token-identical 3/3 at 1429 tokens, but only **1/3 at 6029 tokens**. Divergence is deterministic, not flaky. Treat it as an approximation whose error grows with context, not as a parity-preserving path. Receipts: `qa/evidence-bundles/cuda-flash-prefill-ab-20260910/`.
- `CAMELID_PREFILL_CHUNK_TOKENS` controls how many non-final prompt tokens the backend processes per chunk in the chunked prefill path. Default: `256`, matching the current long-prefill performance lane while keeping the global lazy Q8 file cache disabled outside explicit/scoped reuse. Set it to `1` to force the older sequential prefill path while debugging; invalid/zero values fall back to the default. This is a runtime/performance knob only; it is not support evidence for any model row by itself; the separate published source/runtime-head PASS bundle and synchronized docs/API/frontend updates are what close exact Llama 3 8B checked 1024/2048 packs; the knob itself is not evidence for today's checkout.
- `CAMELID_PREFILL_LAYER_MAJOR` controls the long-context prefill schedule that processes all prefill chunks one layer at a time, reusing file-backed Q8_0 weights across chunks before moving to the next layer. By default it is enabled only when lazy Q8_0 file-backed weights are present. Set it to `0`, `false`, `off`, or `disabled` to force the older chunk-major schedule while debugging.
- `CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS` controls the per-layer prompt chunk size only for the layer-major schedule. Default: `512`, unless `CAMELID_PREFILL_CHUNK_TOKENS` is explicitly set, in which case the shared chunk setting is reused for comparability. It also accepts `all`, `full`, `prompt`, or `unbounded` for one diagnostic full-prompt prefill chunk. This is a runtime/performance knob only and does not promote any 8B 1024/2048 support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES` controls the layer-major-only scoped Q8_0 raw-byte reuse window when lazy file-backed Q8_0 weights are present and `CAMELID_Q8_0_FILE_CACHE_BYTES` is unset. Default: `268435456` (256 MiB) only for multi-chunk layer-major prefill, where file-backed Q8_0 weights can be reused across chunks; single-chunk prefill skips the default scoped cache unless this scoped knob is set explicitly. Set it to `0` to disable the scoped layer-major cache, or set the global cache knob explicitly to take over all Q8 file-reader cache sizing. This is a bounded RSS/read-reuse tuning knob only and does not promote any 8B support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION` enables optional structured per-layer/per-prefill-chunk attribution for the layer-major schedule inside forward-memory timings. This is diagnostic instrumentation for memory/Q8 read attribution, not support evidence or a promotion signal by itself.
- Q8 byte-count knobs accept plain bytes or binary suffixes (`KiB`/`MiB`/`GiB`, also `K`/`M`/`G`; underscores and spaces are ignored). This covers `CAMELID_Q8_0_FILE_CACHE_BYTES`, `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES`, `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES`, `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES`, and `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` without changing their numeric defaults.
- `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES` controls the target Q8_0 row-read chunk size for borrowed/file-backed row readers. Default: `33554432` (32 MiB). This is a read-pattern/performance knob only.
- `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES` caps reusable f32 output scratch for multi-row lazy-Q8 file-backed matmuls. Default: `67108864` (64 MiB). This is an RSS/read-reuse tuning knob only.
- `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` caps how much per-thread Q8 file-reader scratch capacity is retained after oversized row, scale, quantized-input, and output chunks. Default: `67108864` (64 MiB). This is an RSS headroom knob only; it does not promote 8B 1024/2048 support by itself.
- `CAMELID_KV_CACHE_GROW_TOKENS` controls KV-cache allocation growth for model-sized contexts. Default: `256` positions when context length is at least 512; tiny diagnostic/test contexts keep exact one-position growth. This reduces repeated realloc/copy churn during decode and is a runtime performance knob only.
- `CAMELID_METAL_Q8` / `--metal-q8` enables the macOS Metal Q8_0 encoded file-backed row-dot path. It falls back to CPU when unavailable and is not support evidence by itself.
- `CAMELID_PROFILE` selects the execution-planning profile: `safe` keeps only conservative known-good paths, `auto` keeps default-off experiment lanes disabled, `experimental` allows evidence-lane experiments with a warning, and `debug` favors diagnostics over performance claims.
- On the Ubuntu x86_64 dense Llama Q8_0 evidence lane, the appliance planner keeps x86 Q8 experiment flags off by default. Manual developer overrides remain evidence-lane only and must not be treated as support-contract, portability, accelerator-backend, or broader model-family evidence. Current reference truth for this lane is `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/README.md`.
- `CAMELID_X86_Q8_REPACK=on` is a default-off Ubuntu x86_64 developer experiment that loads selected dense Llama Q8_0 linears into backend-owned packed runtime storage instead of retaining a duplicate row-major packed sidecar. The current x86 slice covers the dense attention projection family (`blk.*.attn_{q,k,v}.weight`, `blk.*.attn_output.weight`), dense FFN gate/up/down rows, and `output.weight`; leave it unset for the safe fallback.
- `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER=on`, `CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER=on`, and `CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER=on` are default-off Ubuntu x86_64 developer experiments that let one-row dense attention Q/K/V and attention-output projections consume backend-owned packed Q8_0 runtime storage directly. They fall back unless the runtime plan, tensor type, shape, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense attention Q/K/V packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.attn_{q,k,v}.weight`, and falls back unless all three projections, the runtime plan, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense attention-output packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.attn_output.weight`, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row decode output projection consume backend-owned packed `output.weight` storage directly; it falls back unless the tensor, shape, and packed-row guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row `output.weight` packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `output.weight`, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly. Current evidence for this exact flag is local parity/gate coverage only; no Ubuntu timing/profiling validation is recorded for that local slice, and it must not be treated as Ubuntu throughput, support, portability, or default-on evidence.
- `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row dense FFN gate/up activation consume backend-owned packed `blk.*.ffn_{gate,up}.weight` storage directly with one input quantization; it falls back unless both tensors, shapes, and packed-row guards match exactly. `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION=on` is a narrower default-off follow-on that fuses the gate/up activation write for that same decode route after the same guards pass. `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT=on` is a further default-off follow-on that evaluates paired gate/up packed-row dot products inside the fused decode activation route without widening the route guards.
- `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense FFN gate/up packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.ffn_{gate,up}.weight`, quantizes the shared activation rows once for both projections, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row dense FFN-down projection consume backend-owned packed `blk.*.ffn_down.weight` storage directly; the execution planner clears stale values unless a future slice explicitly owns and validates that consumer gate for the run.
- `CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL=on` is the role-specific, default-off Ubuntu x86_64 developer experiment for the current dense FFN-down multi-row packed-runtime matmul slice. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `ffn_down`, and falls back unless the runtime plan, tensor type, dimensions, row grouping, and packed interleave guards match exactly. `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL=on` remains a compatibility alias for the same FFN-down slice.
- `CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL=on`, `CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED=on`, and `CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2=on` are default-off Ubuntu x86_64 developer experiments for FFN-down rows4 GEMM4 work over backend-owned `PackedRows4` runtime storage. Treat the AVX2 flag as code/route-gate evidence only until a fresh canonical Ubuntu parity plus same-host timing/profiling bundle retains it; do not treat any of these flags as support, portability, production-throughput, RSS, or default-on evidence by themselves.
- Explicit x86 disables remain available as developer overrides. The execution planner respects `CAMELID_X86_Q8_REPACK=off` and `CAMELID_X86_Q8_KERNEL=off|disabled` by failing closed to the safe CPU path, and it manages the x86 decode-consumer/matmul/GEMM4 flags so stale owner experiments are cleared unless explicitly selected in a developer run.
- `CAMELID_METAL_Q8_RETAINED` enables the retained-Q8 all-Metal kernel path for focused kernel experiments. Current local 3B profiling showed all-Metal retained Q8 is slower than the retained-Q8 CPU path, so normal macOS serving keeps this off unless explicitly enabled.
- `CAMELID_HYBRID_Q8_RETAINED` controls the retained-Q8 CPU+Metal split path for single-row decode projections. It defaults to off because same-host Apple Silicon sweeps showed the Metal suffix scheduler was slower and used more RSS than the paired CPU Q8 path on the measured 3B short-decode gate. Set it to `1`, `true`, `on`, or `enabled` to opt into the experiment; set it to `0`, `false`, `off`, `disabled`, or `cpu` to force CPU-only. When enabled, it launches a Metal command buffer for a suffix of output rows while CPU threads compute the rest, then merges the output. Tune the GPU slice with `CAMELID_HYBRID_Q8_GPU_PERCENT` (default `10`, capped below 100) or `CAMELID_HYBRID_Q8_GPU_ROWS`.

If a command depends on more than that, document the requirement in the same PR.

## Maintainer-only/private workflows

The following are intentionally not public contributor requirements:

- SSH-based validation-lane access
- private host aliases or machine-specific setup
- unpublished remote worktree conventions
- local absolute paths from a maintainer workstation

Public docs may mention that some promotion-grade reruns happen on an approved Ubuntu validation lane, but they should not expose private operator details.

When summarizing Ubuntu validation status, distinguish host access from evidence status. Do not report negative host-access status in public docs; keep probe commands, host details, and failure output in private operator notes. If remote validation was not attempted, say that plainly and keep the claim scoped to the missing evidence, such as "no Ubuntu timing/profiling validation is recorded."

## Documentation rule of thumb

When adding a new variable, path convention, or host assumption:

1. document the public/local requirement here if contributors need it
2. keep private operator details out of public docs
3. avoid claiming a workflow is turnkey unless the repo actually makes it turnkey
