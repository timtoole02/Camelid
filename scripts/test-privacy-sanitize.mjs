#!/usr/bin/env node
import assert from 'node:assert/strict'

import {
  sanitizeMessages,
  sanitizePath,
  sanitizeSseEvent,
  sanitizeText,
  sanitizeTimeoutTrace,
  sanitizeUrl,
} from './lib/privacy-sanitize.mjs'

const text = sanitizeText('secret prompt')
assert.equal(text.char_count, 13)
assert.equal(text.sha256.length, 64)
assert.notEqual(text.sha256, 'secret prompt')

const messages = sanitizeMessages([
  { role: 'system', content: 'private rule' },
  { role: 'user', content: 'private question' },
])
assert.deepEqual(messages.roles, ['system', 'user'])
assert.deepEqual(messages.content_char_counts, [12, 16])
assert.equal(messages.content_sha256.length, 2)
assert.equal(messages.combined_sha256.length, 64)

assert.deepEqual(sanitizeUrl('http://private-host.local:8181/v1/chat/completions'), {
  protocol: 'http',
  host: 'redacted',
  port_present: true,
  path: '/v1/chat/completions',
})
const sanitizedPath = sanitizePath('/private/models/Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf')
assert.equal(sanitizedPath.basename, 'Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf')
assert.equal(sanitizedPath.sha256.length, 64)
assert.equal(JSON.stringify(sanitizedPath).includes('/private/models'), false)

const contentEvent = sanitizeSseEvent({
  elapsed_ms: 25,
  event: 'message',
  data: '{"choices":[{"index":0,"delta":{"content":"private token text"},"finish_reason":null}]}',
  parsed: {
    object: 'chat.completion.chunk',
    choices: [{
      index: 0,
      delta: { content: 'private token text' },
      finish_reason: null,
    }],
  },
})
assert.equal(contentEvent.data_kind, 'json')
assert.equal(contentEvent.object, 'chat.completion.chunk')
assert.equal(contentEvent.content.char_count, 18)
assert.equal(contentEvent.content.sha256.length, 64)
assert.equal(JSON.stringify(contentEvent).includes('private token text'), false)

const timeoutTrace = { timeout_ms: 7, elapsed_ms: 11, generated_tokens: 3, timeout_env: 'CAMELID_GENERATION_TIMEOUT_MS' }
assert.deepEqual(sanitizeTimeoutTrace(timeoutTrace), timeoutTrace)
const timeoutEvent = sanitizeSseEvent({
  elapsed_ms: 11,
  event: 'error',
  data: '{"error":{"code":"generation_timeout","message":"private host failed","timeout_trace":{"timeout_ms":7,"elapsed_ms":11,"generated_tokens":3,"timeout_env":"CAMELID_GENERATION_TIMEOUT_MS"}}}',
  parsed: {
    error: {
      code: 'generation_timeout',
      message: 'private host failed',
      timeout_trace: timeoutTrace,
    },
  },
})
assert.equal(timeoutEvent.error.code, 'generation_timeout')
assert.equal(timeoutEvent.error.message.char_count, 19)
assert.deepEqual(timeoutEvent.error.timeout_trace, timeoutTrace)
assert.equal(JSON.stringify(timeoutEvent).includes('private host failed'), false)

assert.equal(sanitizeSseEvent({ elapsed_ms: 30, event: 'message', data: '[DONE]', parsed: null }).data_kind, 'done')

console.log('privacy-sanitize self-test passed')
