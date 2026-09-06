#!/usr/bin/env node
import assert from 'node:assert/strict'

import camelidBenchmarkProviderExtension, {
  normalizeCamelidOverflowMessage,
} from './adapters/pi-camelid-provider.mjs'

const overflow = {
  role: 'assistant',
  provider: 'camelid-benchmark',
  stopReason: 'error',
  errorMessage: '413 prompt encoded to 360012 tokens, above the server ceiling of 131072',
}
const normalized = normalizeCamelidOverflowMessage(overflow)
assert.equal(
  normalized.errorMessage,
  'context_length_exceeded: 413 prompt encoded to 360012 tokens, above the server ceiling of 131072',
)
assert.equal(normalizeCamelidOverflowMessage(normalized), null)
assert.equal(normalizeCamelidOverflowMessage({ ...overflow, provider: 'other-provider' }), null)
assert.equal(normalizeCamelidOverflowMessage({ ...overflow, provider: 'other-provider' }, 'camelid-benchmark').errorMessage.startsWith('context_length_exceeded:'), true)
assert.equal(normalizeCamelidOverflowMessage({ ...overflow, errorMessage: 'rate limit exceeded' }), null)
assert.equal(normalizeCamelidOverflowMessage({ ...overflow, stopReason: 'stop' }), null)
assert.match(
  normalizeCamelidOverflowMessage({ ...overflow, errorMessage: 'prompt_token_limit_exceeded' }).errorMessage,
  /^context_length_exceeded:/,
)

let handler = null
camelidBenchmarkProviderExtension({
  on(event, callback) {
    assert.equal(event, 'message_end')
    handler = callback
  },
})
assert.equal(typeof handler, 'function')
assert.equal(handler({ message: overflow }, { model: null }).message.errorMessage, normalized.errorMessage)
assert.equal(handler({ message: { ...overflow, provider: 'other-provider' } }, { model: { provider: 'other-provider' } }), undefined)

console.log('benchmark Phase 4 Pi provider extension: PASS')