#!/usr/bin/env node
import assert from 'node:assert/strict'

import { extractSseEvents, readChatCompletionJsonPayload, readStreamingChatCompletion } from '../src/lib/chatCompletionStream.js'
import { authoritativeOutputRate, describeMtp12WidthSchedule, readTargetVerifiedMtp12 } from '../src/lib/nativeGenerationMetrics.js'

const partial = 'data: {"choices":[{"delta":{"content":"hel"}}]}\r\n\r\ndata: {"choices":[{"delta":{"content":"lo"}}]}'
const firstPass = extractSseEvents(partial)
assert.equal(firstPass.events.length, 1, 'complete SSE events should flush while partial backend chunks stay buffered')
assert.match(firstPass.remainder, /"lo"/, 'partial SSE data should remain buffered until the blank-line event boundary arrives')
const secondPass = extractSseEvents(`${firstPass.remainder}\n\ndata: [DONE]\n\n`)
assert.equal(secondPass.events.length, 2, 'the remaining partial SSE event should flush after its boundary arrives')

const jsonPayload = readChatCompletionJsonPayload({
  choices: [{ message: { content: 'json reply' }, finish_reason: 'stop' }],
  usage: { completion_tokens: 2 },
})
assert.equal(jsonPayload.content, 'json reply', 'non-streaming JSON fallback should preserve assistant content')
assert.equal(jsonPayload.completionTokens, 2, 'JSON usage should remain exact when the backend provides it')

function streamFromChunks(chunks) {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

const fallbackEvents = []
const fallbackDeltas = []
const fallback = await readStreamingChatCompletion(new Response(JSON.stringify({
  choices: [{ message: { content: 'json fallback reply' }, finish_reason: 'stop' }],
  usage: { completion_tokens: 3 },
}), {
  status: 200,
  headers: { 'content-type': 'application/json' },
}), (delta, fullContent, metrics) => {
  fallbackDeltas.push({ delta, fullContent, firstByteMs: metrics.firstByteMs, firstContentMs: metrics.firstContentMs })
}, {
  onStreamEvent(event) {
    fallbackEvents.push(event.type)
  },
})
assert.equal(fallback.content, 'json fallback reply', 'JSON fallback should preserve assistant content through the streaming reader')
assert.equal(fallback.completionTokens, 3, 'JSON fallback should preserve exact backend completion-token usage')
assert.equal(fallback.firstByteMs, 0, 'JSON fallback should expose response-header progress so the UI can stay visibly active')
assert.ok(fallback.firstContentMs >= 0, 'JSON fallback should expose first-content timing once the body is parsed')
assert.deepEqual(fallbackEvents, ['json_fallback'], 'JSON fallback should notify callers before the final assistant content is available')
assert.deepEqual(fallbackDeltas.map((item) => item.fullContent), ['json fallback reply'], 'JSON fallback should still deliver one visible content update')

const response = new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":"```js\\nconst"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":" answer = 42"}}]}\n',
  '\n',
  'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":7,"total_tokens":10},"camelid":{"mtp12":{"lossless_target_verified":true,"decode_tokens_per_second":51.494,"decode_us":135937,"decode_output_tokens":7,"configured_verify_width":16,"accepted_drafts":6,"drafted":7,"selector":"CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16","width_schedule":{"selector":"CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16","selector_value":"1","enabled":true,"active_for_configured_width":true,"warmup_verify_width":8,"padded_tail_policy":"remaining_10_through_15_reserves_bonus_anchor; logical_remaining_minus_1; physical_w16; repeat_anchor_padding","policy":"one_shot_w8_then_w16; partial_acceptance_does_not_fallback; padded_w16_tail"}}}}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
})

const deltas = []
const streamEvents = []
const streamed = await readStreamingChatCompletion(response, (delta, fullContent, metrics) => {
  deltas.push({ delta, fullContent, completionTokens: metrics.completionTokens })
}, {
  onStreamEvent(event) {
    streamEvents.push(event.type)
  },
})

assert.equal(streamed.content, '```js\nconst answer = 42', 'stream parser should preserve incomplete fenced code content safely for live rendering')
assert.equal(streamed.finishReason, 'stop', 'stream parser should preserve finish_reason from the terminal chunk')
assert.deepEqual(deltas.map((item) => item.fullContent), ['```js\nconst', '```js\nconst answer = 42'], 'stream deltas should update visible content before backend completion')
assert.deepEqual(deltas.map((item) => item.completionTokens), [1, 2], 'stream metrics should advance while generation is active')
assert.equal(streamed.completionTokens, 7, 'stream parser should preserve exact backend completion-token usage from the terminal chunk')
assert.deepEqual(streamed.usage, { prompt_tokens: 3, completion_tokens: 7, total_tokens: 10 }, 'stream parser should preserve exact backend usage evidence instead of replacing it with estimates')
assert.equal(streamed.camelid?.mtp12?.decode_tokens_per_second, 51.494, 'terminal SSE should preserve native MTP12 diagnostics')
assert.equal(readTargetVerifiedMtp12(streamed.camelid)?.configured_verify_width, 16, 'target-verified MTP12 diagnostics should normalize finite native fields')
assert.deepEqual(
  describeMtp12WidthSchedule(streamed.camelid.mtp12),
  { widths: 'W8 bootstrap → W16 verify', policy: 'remaining 10 through 15 reserves bonus anchor; logical remaining minus 1; physical W16; repeat anchor padding' },
  'the UI should derive readable widths from the backend provenance object instead of assuming an invented array',
)
assert.equal(authoritativeOutputRate({ camelid: streamed.camelid, tokens_out_per_sec: 2 }), 51.494, 'native target-verified rate should outrank the browser estimate')
assert.equal(authoritativeOutputRate({ camelid: { mtp12: { lossless_target_verified: false, decode_tokens_per_second: 99 } }, tokens_out_per_sec: 2 }), 2, 'unverified native claims must not replace the browser estimate')
assert.ok(streamEvents.includes('bytes'), 'stream parser should expose first-byte progress before content')
assert.ok(streamEvents.includes('role'), 'stream parser should expose role-only chunks while waiting for first content token')
assert.ok(streamEvents.includes('usage'), 'stream parser should expose backend usage chunks before finalizing the assistant row')
assert.deepEqual(streamEvents.slice(-3), ['finish', 'usage', 'done'], 'terminal stream evidence should preserve finish, usage, then [DONE] ordering')

const reasoningEvents = []
const reasoningOnly = await readStreamingChatCompletion(new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
  'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
  'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
  'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
}), () => {}, {
  onStreamEvent(event) {
    reasoningEvents.push(event.type)
  },
})
assert.equal(reasoningOnly.content, '', 'reasoning-only streams must not leak hidden reasoning into assistant content')
assert.ok(reasoningEvents.includes('reasoning'), 'reasoning-only streams should expose forward progress to diagnostics and smoke checks')
assert.deepEqual(reasoningEvents.slice(-3), ['finish', 'usage', 'done'], 'reasoning-only terminal evidence should preserve finish, usage, then [DONE] ordering')

const multilinePayload = await readStreamingChatCompletion(new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"content":"multi"}}],\n',
  'data: "usage":{"completion_tokens":4}}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
}), () => {})
assert.equal(multilinePayload.content, 'multi', 'SSE parser should join multi-line data payloads before parsing JSON')
assert.equal(multilinePayload.completionTokens, 4, 'SSE parser should preserve usage from joined multi-line data payloads')

const batchedPayloadDeltas = []
const batchedPayload = await readStreamingChatCompletion(new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"content":"batch"}}]}\n',
  'data: {"choices":[{"delta":{"content":"ed"}}]}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
}), (_delta, fullContent) => {
  batchedPayloadDeltas.push(fullContent)
})
assert.equal(batchedPayload.content, 'batched', 'SSE parser should keep accepting backend batches with several JSON payloads in one event')
assert.deepEqual(batchedPayloadDeltas, ['batch', 'batched'], 'batched payloads should still stream each visible update')

const segmentedDeltas = []
const segmentedEvents = []
const segmentedPayload = await readStreamingChatCompletion(new Response(streamFromChunks([
  'data: {"choices":[{"delta":{"content":"## First"}}]}\n\n',
  'data: {"choices":[{"delta":{"camelid_segment":{"index":0,"token_ids_exact":true,"requested_tokens":1,"verified_tokens":1,"decode_output_tokens":1,"decode_us":20000,"render_tokens_per_second":50,"boundary":"\\n\\n"}}}]}\n\n',
  'data: {"choices":[{"delta":{"content":"## Second"}}]}\n\n',
  'data: {"choices":[{"delta":{"camelid_segment":{"index":1,"token_ids_exact":true,"requested_tokens":1,"verified_tokens":1,"decode_output_tokens":1,"decode_us":19608,"render_tokens_per_second":51,"boundary":""}}}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n',
  'data: [DONE]\n\n',
]), {
  status: 200,
  headers: { 'content-type': 'text/event-stream' },
}), (delta, fullContent, metrics) => {
  segmentedDeltas.push({ delta, fullContent, completionTokens: metrics.completionTokens })
}, {
  onStreamEvent(event) {
    if (event.type === 'segment') segmentedEvents.push(event.segment)
  },
})
assert.equal(segmentedPayload.content, '## First\n\n## Second', 'UI-authored segment boundaries should produce clean Markdown exactly once')
assert.deepEqual(segmentedDeltas.map((item) => item.completionTokens), [1, 1, 2], 'presentation boundaries must not inflate exact model-token counts')
assert.equal(segmentedEvents.length, 2, 'every backend segment progress item should surface to the live UI')
assert.equal(segmentedEvents[0].render_tokens_per_second, 50, 'the completed section native rate should remain available during the next prefill gap')
assert.equal(segmentedEvents[1].boundary, '', 'the final progress item should not append a trailing separator')

const partialBeforeError = []
const errorEvents = []
await assert.rejects(
  () => readStreamingChatCompletion(new Response(streamFromChunks([
    'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
    'event: error\n',
    'data: {"error":{"code":"generation_step_failed","message":"backend failed after headers"}}\n\n',
    'data: [DONE]\n\n',
  ]), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  }), (_delta, fullContent) => {
    partialBeforeError.push(fullContent)
  }, {
    onStreamEvent(event) {
      errorEvents.push(event.type)
    },
  }),
  (error) => {
    assert.equal(error.message, 'backend failed after headers')
    assert.equal(error.code, 'generation_step_failed')
    assert.deepEqual(error.payload, { error: { code: 'generation_step_failed', message: 'backend failed after headers' } })
    return true
  },
  'SSE error events sent after streaming headers should reject instead of becoming an empty assistant reply',
)
assert.deepEqual(partialBeforeError, ['partial'], 'stream parser should expose visible partial content before a later SSE error')
assert.ok(errorEvents.includes('error'), 'stream parser should surface structured SSE error events to callers')

console.log('Streaming parser smoke passed')
