#!/usr/bin/env node
/* Continue-a-truncated-reply smoke.

   Two halves, both pure (no browser, no engine):

   1. lib/chatContinuation.js — which replies may be continued, how the new
      text is joined onto the old (the whitespace and repeat cases a model
      actually produces), and how usage accumulates.
   2. The rendered MessageTurn — that the button appears for exactly the
      finish_reason="length" case, that it is absent when the caller does not
      supply a handler, and that a continued reply discloses that it was
      continued. Rendered as React elements; there is no innerHTML path. */
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const {
    CONTINUATION_INSTRUCTION,
    canContinueMessage,
    continuationCountOf,
    joinContinuation,
    mergeContinuedUsage,
  } = await server.ssrLoadModule('/src/lib/chatContinuation.js')

  /* ---- which replies may be continued ---------------------------------- */
  const truncated = { role: 'assistant', finish_reason: 'length', content: 'half an answer' }
  assert.equal(canContinueMessage(truncated), true, 'a length-truncated reply with text is continuable')
  assert.equal(
    canContinueMessage({ ...truncated, finish_reason: 'stop' }),
    false,
    'a reply that stopped on its own is complete — continuing it would append to a finished answer',
  )
  assert.equal(
    canContinueMessage({ ...truncated, streaming: true }),
    false,
    'a reply still streaming must not be continuable — the request is not finished',
  )
  assert.equal(
    canContinueMessage({ ...truncated, content: '   ' }),
    false,
    'there is nothing to continue from when the budget went entirely to hidden channel tokens',
  )
  assert.equal(canContinueMessage({ ...truncated, role: 'user' }), false, 'user turns are not continuable')
  assert.equal(canContinueMessage(null), false, 'a missing message is not continuable')

  /* ---- joining the continuation onto the text it resumes ---------------- */
  assert.equal(
    joinContinuation('the imple', 'mentation is done'),
    'the implementation is done',
    'a mid-word resume must not have a space inserted into it',
  )
  assert.equal(
    joinContinuation('First line.\n', '  Second line.'),
    'First line.\nSecond line.',
    'when the kept text already ends in whitespace the continuation must not double it',
  )
  assert.equal(
    joinContinuation('ends with a word', ' and continues'),
    'ends with a word and continues',
    'a continuation that leads with its own space keeps it when the prefix has none',
  )
  const restated = 'Step one is to load the model into memory before anything else happens.'
  assert.equal(
    joinContinuation(`Intro. ${restated}`, `${restated} Step two is to tokenize.`),
    `Intro. ${restated} Step two is to tokenize.`,
    'a model that resumes by restating its last sentence must not have it duplicated',
  )
  assert.equal(
    joinContinuation('...and then', ' and then some more'),
    '...and then and then some more',
    'a short coincidental overlap is real text and must be preserved',
  )
  assert.equal(joinContinuation('', 'all of it'), 'all of it', 'an empty prefix returns the continuation')
  assert.equal(joinContinuation('kept', ''), 'kept', 'an empty continuation returns the kept text')

  /* ---- usage accumulates across continuations --------------------------- */
  const merged = mergeContinuedUsage(
    { prompt_tokens: 100, completion_tokens: 512, total_tokens: 612 },
    { prompt_tokens: 640, completion_tokens: 300 },
  )
  assert.equal(merged.completion_tokens, 812, 'output covers the whole reply the reader can see')
  assert.equal(merged.prompt_tokens, 640, 'prompt stays the last request — a summed prompt describes no real request')
  assert.equal(merged.total_tokens, 1452, 'total is consistent with the two fields beside it')
  assert.equal(
    mergeContinuedUsage(undefined, { prompt_tokens: 10, completion_tokens: 5 }).completion_tokens,
    5,
    'a first continuation over a message with no recorded usage counts only what it generated',
  )

  assert.equal(continuationCountOf({ continuation_count: 3 }), 3)
  assert.equal(continuationCountOf({}), 0, 'a reply that was never continued reports zero, not NaN')
  assert.match(CONTINUATION_INSTRUCTION, /Do not repeat any text/, 'the instruction must forbid restating')

  /* ---- the button itself ------------------------------------------------ */
  const { MessageTurn } = await server.ssrLoadModule('/src/components/chat/MessageTurn.jsx')
  const renderTurn = (message, props = {}) =>
    renderToStaticMarkup(React.createElement(MessageTurn, { message, ...props }))

  const lengthStopped = {
    id: 'm1',
    role: 'assistant',
    content: 'A partial answer that ran out of room',
    finish_reason: 'length',
    usage: { prompt_tokens: 10, completion_tokens: 512 },
  }

  const withHandler = renderTurn(lengthStopped, { onContinue: () => {} })
  assert.match(withHandler, /cxturn__warning-action/, 'a length-truncated reply offers Continue')
  assert.match(withHandler, />Continue</, 'the Continue button is labelled Continue')
  assert.match(withHandler, /Stopped at the response budget/, 'the reason for the button is stated next to it')

  const withoutHandler = renderTurn(lengthStopped)
  assert.doesNotMatch(
    withoutHandler,
    /cxturn__warning-action/,
    'no Continue button when the caller cannot honour it (busy, gated, or not the last reply)',
  )
  assert.match(withoutHandler, /Stopped at the response budget/, 'the warning still explains what happened')

  const finished = renderTurn({ ...lengthStopped, finish_reason: 'stop' }, { onContinue: () => {} })
  assert.doesNotMatch(finished, /cxturn__warning-action/, 'a completed reply never offers Continue')

  const continuedOnce = renderTurn({ ...lengthStopped, finish_reason: 'stop', continuation_count: 1 })
  assert.match(continuedOnce, /continued/, 'a resumed reply discloses that it was resumed')
  assert.doesNotMatch(continuedOnce, /continued ×/, 'one continuation reads "continued", not "continued ×1"')

  const continuedTwice = renderTurn({ ...lengthStopped, finish_reason: 'stop', continuation_count: 2 })
  assert.match(continuedTwice, /continued ×2/, 'repeated continuations are counted')

  const neverContinued = renderTurn({ ...lengthStopped, finish_reason: 'stop' })
  assert.doesNotMatch(neverContinued, /continued/, 'an ordinary reply carries no continuation marker')

  console.log('chat continuation smoke passed')
} finally {
  await server.close()
}
