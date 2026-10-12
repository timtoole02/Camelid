# Chat answers only from excerpts about what was asked (#800)

When a chat searches documents, the four passages closest to the question go
to the model with the question. Some of them are about something merely
similar: the same plant's other clarifier, the same observatory's other
mirror. The chat's old instruction ("Refer to the following retrieved document
excerpts to answer the prompt") did not say to leave those out, and a small
model answered with them too, as if they were about what was asked.

The new instruction, in `frontend/src/views/ChatWorkspace.jsx`:

> Answer the question from the document excerpts below. Use an excerpt only if
> it is about exactly what the question asks: one about something similar -
> another item, part, person, date or place - does not answer it, so leave it
> out. Cite each fact inline as [1], [2], matching the excerpt it came from;
> every answer cites at least one excerpt.

Each excerpt is now labelled `[N] (from file):` instead of
`[Citation N from file]:`, so the label is the `[N]` the reply is asked to
cite by. The `--- DOCUMENT CONTEXT ---` block and the citation pills are
unchanged.

## How it was measured

All of this ran on one CPU machine (4 vCPU), with Llama 3.2 3B Instruct
Q4_K_M at temperature 0.

- **The library** is the 550 PDFs from the #804 run of the 500-PDF gate
  harness (#798): 500 arXiv papers and 50 planted PDFs, in 114,083 passages.
  It sits in one collection, "F2a 500-PDF gate".
- **The questions.** Each planted PDF holds a fact, for example "The north
  clarifier of the Kessring water treatment plant was relined in 2015 at a
  cost of 208,400 euros." Another planted PDF holds its decoy: the same
  sentence about the neighbouring part (south clarifier, secondary mirror,
  west turntable), with a different amount. There are 50 questions, each
  asking for a fact's amount.
- **The split.** The wording was chosen on the 25 even-numbered questions
  (dev). The 25 odd-numbered questions (test) were run only to report the
  result.
- **The harness.** `harness/eval_rag_prompt.py` takes the chat's steps for
  each question. It searches the collection with `top_k: 4` (what the chat
  sends), builds the prompt, and asks the model. It sends the prompt as the
  only message.

Three wordings were tried, all in the harness:

- `current`: what the chat sends today.
- `exact`: the new instruction without "every answer cites at least one
  excerpt".
- `exact2`: the wording shipped here.

**Sorting the answers.** An answer counts as "right" when it states the
fact's amount. It counts as "cited" when one of its `[N]` markers points at
an excerpt that holds that amount.

An answer that also states the decoy's amount is split in two by
`harness/classify.py`:

- **labelled:** the reply names the decoy's part word, the one word in the
  decoy sentence that is not in the fact sentence ("south", "secondary",
  "west").
- **misattributed:** the decoy's part word is missing, so the reply offers
  the decoy's amount as an answer to the question.

## Results

From `data/prompt-ab/classification.json`. The decoy reached the model's
excerpts for 24 of 25 dev questions and 23 of 25 test questions.

| Split | Wording | Right | Cited | Any marker | Decoy stated | Misattributed | Labelled |
|---|---|---|---|---|---|---|---|
| dev | current | 25/25 | 25 | 25 | 17 | 17 | 0 |
| dev | exact | 24/25 | 23 | 24 | 6 | 5 | 1 |
| dev | **exact2** | 25/25 | 25 | 25 | 7 | **3** | 4 |
| test | current | 25/25 | 25 | 25 | 10 | 10 | 0 |
| test | exact | 25/25 | 23 | 23 | 3 | 3 | 0 |
| test | **exact2** | 25/25 | 25 | 25 | 4 | **2** | 2 |

On the held-out questions, the wrong amount offered as an answer went from
10 of 25 to 2. Every answer is still right and still cites the excerpt the
amount came from.

`exact` stopped two answers from citing anything. The last clause of `exact2`
brings citations back to 25 of 25.

**What is left.** Two test answers (13 and 19) still state the decoy's
amount without saying it belongs to another part. Answer 13 calls it "likely
an error". Answer 19 lists three amounts. The wording does not get a 3B model
past this, and this change does not claim it does.

## Live, in the chat

`harness/live-check.mjs` drives the real web UI in Chrome against the same
server and library. For each question it opens a new chat, puts the
collection in it, asks, and records the request the UI sent and the reply.
`data/live/*.json` shows that each request carried only that one user
message.

- **before** is the UI embedded in the running server (build
  `v0.7.8-33-g29798f1d-dirty`). Its prompt is the one on `main`; the captured
  request reads "Refer to the following retrieved document excerpts…".
  - That build predates the "4 passages used" chip on the sent message, so
    the chip does not show in these screenshots.
- **after** is this branch's `frontend/`, served by `vite` with `/api` and
  `/v1` proxied to the same server.

| Question | Before | After |
|---|---|---|
| 21: primary mirror, Ivarsdal observatory (fact 335,300; decoy 937,700) | lists 335,300 [1] and 937,700 [2] as two costs ([screenshot](screenshots/before-q21.png)) | 335,300 [1] only ([screenshot](screenshots/after-q21.png)) |
| 13: north clarifier, Wyncroft plant (fact 487,200; decoy 707,600) | lists 487,200 [1] and 707,600 [2] ([screenshot](screenshots/before-q13.png)) | 487,200 [1]; says [2]'s 707,600 is "likely an error" ([screenshot](screenshots/after-q13.png)) |

Question 21 is one that `current` got wrong in the harness. Question 13 is
one of the two that `exact2` still gets wrong.

The live prompts are byte-identical to what the harness builds for these
questions, with `current` before and `exact2` after. Both UIs sent the same
request settings: `temperature: 0`, `max_tokens: 8192`, streaming
(`data/live/*.json`, under `params`).

The replies match the harness's. The harness asks without streaming and with
`max_tokens: 160`, and records the raw text. In the live check the UI renders
that same text: the after-replies begin with the harness's reply word for
word. The before-replies have the same words and amounts, but the model wrote
the markers as "(Citation 1 from planted-21.pdf)", which the chat renders as
the [1] pill (#798).

## Files

- `harness/eval_rag_prompt.py`: the A/B harness.
  - Usage: `python3 eval_rag_prompt.py <port> <corpus dir> <state.json> <out json> <dev|test> <wording>...`
  - The corpus and `state.json` come from the 500-PDF gate harness in #798.
- `harness/classify.py`: sorts the answers that state the decoy, and writes
  `classification.json`.
  - Usage: `python3 classify.py <planted-facts.json> <result json>...`
- `harness/live-check.mjs`: the live check.
  - Usage: `node live-check.mjs <ui origin> <before|after> <out dir>`, run
    from `frontend/`.
- `data/prompt-ab/`:
  - `dev-1` and `test-1` hold `current` and `exact`;
  - `dev-2` and `test-2` hold `exact2`;
  - each holds every question, reply and check.
- `data/live/`: the live check's requests and replies.
- `screenshots/`: the live check's screenshots.
