# A cancelled request stops while it is still reading its prompt (#803)

When a client gave up on a request (Stop in the chat, a closed tab, an
aborted API call), Camelid stopped the work "within one step". Reading the
whole prompt was a single step, though. So a request cancelled while its
prompt was being read kept the model busy until the whole prompt was read,
and the next request waited behind it. On a slow lane that was minutes.

With this change the prompt read itself checks for cancellation, and the
next request starts within about a second (or one layer of a slow CPU model).

## What changed

**Dense lane** (Llama, Mistral, Qwen3 and the other `LlamaInferenceSession`
models):

- The session carries the request's cancel check
  (`set_prefill_interrupt`). The API installs it after any prompt-cache
  swap, on the non-streaming path and both streaming paths. A cached copy of
  a session never carries it, because `Clone` drops it.
- The CPU prompt read checks it between layers (layer-major prefill), between
  chunks and between the layers inside a chunk (chunked prefill), and between
  tokens (token-by-token prefill).
- The serial CUDA-resident prefill (the default for Q4_K/Q6_K rows) now runs
  in 128-token segments. Each segment is the same per-token forwards in the
  same order, and the check runs between segments.
- A cancelled read returns `BackendError::Cancelled`, which the API turns
  into `generation_cancelled`, or `generation_timeout` when it was the
  request's deadline that fired. Before, it would have been
  `generation_step_failed`. A cancelled CUDA prefill does not fall back to
  reading the prompt on the CPU.
- A batched GPU prefill (one command buffer) is only checked before it
  starts.

**Runnable lane** (Ornith / qwen35 and the other runnable models):

- Non-streaming chat now has the disconnect guard that streaming chat
  already had. Before, a non-streaming request had no way to learn its
  client was gone.
- qwen35 on the CPU checks before each layer of the prompt read. During
  generation the check rides on the existing per-token stop hook.
- qwen35 on CUDA checks every 64 prompt tokens and after each decode chunk
  (both after a sync, so the check reflects real GPU progress). The host-fed
  CUDA loop checks before its prompt read and after every token.
- A cancelled CUDA run is not retried on the CPU.
- A cancelled run is an error, never a short reply.

**Engine queue.** A job whose client left while it waited in the queue now
returns before grammar setup and the prompt-cache lookup. Main already
skipped such a job before reading its prompt (the decode loop checks the
cancel token before its first step), and the queue check below measured the
same on both builds. This is not a behaviour change anyone will notice.

## How it was measured

One machine with 4 vCPUs, 15.6 GB of RAM and an NVIDIA L4, nothing else
running. Release builds of upstream main (`fa24c19c`) and of this branch,
each served with `harness/serve.sh` and driven by `harness/run-matrix.sh`.

- **Cancel** (`harness/cancel-bench.mjs`): time a tiny chat request ("Say
  hi.", 4 tokens) on an idle server. Then send a long prompt, abort the client
  part-way through reading it, and time the same tiny request again. Run once
  non-streaming and once streaming.
- **Stress** (`harness/stress.mjs`): abort a long prompt at a pseudo-random
  time (seed 7), then time a tiny request; repeat 2 to 8 times, alternating
  streaming and non-streaming. Each round's prompt starts with a different
  `Request <n>:` so the prompt cache can never skip a read. Afterwards, round
  0's prompt (the one cancelled first) is run to completion and its reply
  compared with the same prompt run on a fresh server that never saw a
  cancel. A half-read prompt reused through the prompt cache would show up
  here.
- **Queue** (`harness/queue.mjs`, dense lane only, since the runnable lane
  has no engine queue): A reads a long prompt; B, a different long prompt,
  is sent and aborted while still queued behind A; then C, a tiny request,
  is timed.

The long prompts are filler sentences ("The river carried the valley towns
in year 1500. ...") plus "In one sentence, what is this text about?". The
prompt sizes below are the ones the server reported.

### The next request after a cancel

The two numbers per cell are the non-streaming and the streaming cancel.

| Model | Lane | Prompt tokens | Tiny reply, idle (main / PR) | Tiny reply after a cancel, main | Tiny reply after a cancel, this PR |
|---|---|---|---|---|---|
| Llama 3.2 3B Q4_K_M | GPU (default) | 5127 | 0.1 s / 0.1 s | 80.1 s / 80.1 s | 1.0 s / 1.0 s |
| Llama 3.2 3B Q4_K_M | CPU (`--gpu off`) | 1940 | 0.7 s / 0.8 s | 570.8 s / 568.8 s | 0.8 s / 0.8 s |
| Ornith 1.0 9B Q8_0 | CPU (default) | 323 | 12.0 s / 14.1 s | 142.2 s / 183.8 s | 15.0 s / 15.0 s |
| Ornith 1.0 9B Q8_0 | CUDA (`CAMELID_QWEN35_CUDA=1`) | 5929 | 0.7 s / 0.7 s | 220.2 s / 220.0 s | 2.2 s / 2.3 s |

Ornith on the CPU reads its prompt a layer at a time, and one layer of a
323-token prompt takes a few seconds on this machine. So after a cancel the
tiny request waits for the layer in progress, then runs at its idle speed.

### Stress

| Lane | Build | Random cancels | Slowest tiny reply after one | Tiny reply text unchanged | Same-prompt reply equals uncancelled run |
|---|---|---|---|---|---|
| Llama 3.2 3B GPU | main | 4 | 80.7 s | yes | yes |
| Llama 3.2 3B GPU | this PR | 8 | 2.0 s | yes | yes |
| Llama 3.2 3B CPU | main | 2 | 563.6 s | yes | yes |
| Llama 3.2 3B CPU | this PR | 8 | 2.2 s | yes | yes |
| Ornith 9B CPU | main | 2 | 200.7 s | yes | yes |
| Ornith 9B CPU | this PR (earlier build, see below) | 6 | 20.5 s | yes | yes |
| Ornith 9B CUDA | main | 2 | 220.1 s | yes | yes |
| Ornith 9B CUDA | this PR | 8 | 2.9 s | yes | yes |

Main got fewer rounds only because each of its rounds takes minutes. Every
server answered `/v1/health` as ready after its stress run.

### Queue

| Lane | Build | C waited | C finished after A by |
|---|---|---|---|
| Llama 3.2 3B GPU | main | 84.8 s | 0.1 s |
| Llama 3.2 3B GPU | this PR | 84.7 s | 0.1 s |
| Llama 3.2 3B CPU | main | 565.2 s | 3.7 s |
| Llama 3.2 3B CPU | this PR | 578.2 s | 3.9 s |

C finishes right after A on both builds, so B never read its prompt on
either.

`harness/summarize.py` prints these tables from `data/api/`.

## In the chat

`harness/live803.mjs` drives the real chat UI (the `frontend/` dev server
from this branch, which does not change the frontend) in Chrome:

1. Send the long message.
2. Press Stop while the model is still reading it.
3. Start a **New chat** and ask "Say hi in three words."
4. Time that reply.

It records every chat request, and checks that the short question went out
alone (`short_request_roles: ["user"]`).

| Model, lane | Short reply after Stop, main | Short reply after Stop, this PR |
|---|---|---|
| Ornith 1.0 9B, CPU | 178.2 s (first token 129.3 s) | 18.8 s (first token 15.8 s) |
| Llama 3.2 3B, GPU | 77.2 s (first token 77.2 s) | 0.3 s (first token 330 ms) |

The timings are the browser's: the question was sent, until its response was
fully received. The first-token figures are the chat's own diagnostics, shown
in the screenshots.

Screenshots, per run (`screenshots/<run>-N-*.png`):

1. reading the long message;
2. stopped;
3. 20 s after asking the short question;
4. the reply.

On main, the 20-second shot still shows "Generating response"; with this
change the reply is already there.

Two things seen along the way, neither caused by this change:

- **Asking again in the same chat** resends the stopped message as history,
  so that request must read the long prompt anyway (a first run of the
  script did this: 337 prompt tokens for the short question). That is why
  the script opens a new chat.
- **In the Ornith run on main** (and in an earlier attempt of it that is not
  kept here), while the short question waited, the chat view switched back
  to the stopped conversation. The script reopens the short question's chat
  from the sidebar before each screenshot (`data/live/ornith-before.json`
  records it). It was not seen in the other three runs. Why the view
  switches was not investigated.

## Tests

New unit tests, run by `cargo test`:

- `api::tests::prefill_interrupt_stops_inside_the_prompt_read`: a cancel
  stops the read after the first check inside it.
- `api::tests::prefill_interrupt_stops_between_layers_of_a_chunk`: a cancel
  stops the read inside a chunk.
- `api::tests::prefill_interrupt_that_never_fires_changes_nothing`: same
  token, logits and KV cache as with no check installed.
- `api::tests::prefill_interrupt_is_per_request_state`: a clone carries no
  check; `take_for_step` keeps it.
- `api::tests::prompt_without_prefill_ignores_the_prefill_interrupt`.
- `api::tests::armed_prefill_follows_the_request_cancel_token`.
- `api::tests::prefill_stopped_by_the_deadline_reports_a_timeout`.
- `api::tests::queued_job_whose_client_left_is_skipped`.
- `runnable::model::qwen35_cuda_fallback_tests::cancelled_cuda_run_is_not_retried_on_cpu`.

Ignored tests that need a model file:

- `runnable::model::gpu_ssm_layer_tests::qwen35_cpu_cancel_stops_prompt_read_and_decode`
  (`CAMELID_ORNITH_GGUF`): the real Ornith stops before layer 1 of the
  prompt read; no token is produced after a cancel; and an installed check
  that never fires gives the same tokens.
- `inference::tests::serial_resident_prefill_stops_between_segments_when_cancelled`
  (`CAMELID_3B_GGUF`, CUDA): the serial CUDA prefill stops before its second
  segment, and the next request on the same warm engine gets the same token
  and logits as a freshly built engine.

## What this does not cover

- A batched GPU prefill (one command buffer, Metal or CUDA) is checked only
  before it starts.
- The qwen35 Metal lane gets the per-token check during generation, not
  during its batched prompt read. No Mac was used here; that path compiled
  but did not run.
- Runnable-lane image prompts (`generate_vision_greedy`) are not wired to the
  disconnect guard.
- gemma4 and diffusion lanes are unchanged; gemma4 already had a cancellable
  prompt read.
- Once a request's deadline (`CAMELID_GENERATION_TIMEOUT_MS`) passes during
  the prompt read, the read now stops there too, and the request reports
  `generation_timeout` as before, just sooner.

## Builds

- Every "this PR" row is the final build of the branch, except Ornith CPU
  stress. That stress ran on an earlier build of the branch: the later
  commits only added the per-layer check inside chunked dense prefill, the
  "no CPU retry after a cancelled CUDA run" guard (not reached on the CPU
  lane), a test and formatting. The final build's Ornith CPU cancel and
  reference checks are in `data/api/ornith-9b-cpu-fix.jsonl`.
- The Llama GPU main run used the same scripts before they switched to
  `harness/http-json.mjs`. The switch changed only how requests are sent,
  so that a non-streaming request can wait past `fetch`'s 300 s header
  timeout.

## Files

- `harness/`: the scripts above.
- `data/api/<model>-<lane>-<build>.jsonl`: one JSON line per check.
- `data/live/<run>.json`: the live runs: every event with its time, the
  short question's request roles and the reply.
- `screenshots/`: 16 screenshots, 4 per live run.
- `SHA256SUMS`.
