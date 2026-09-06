#!/usr/bin/env node
/* Send-time compaction must never lose the conversation's meaning.
 *
 * Compaction exists so a long chat stays sendable instead of hitting the
 * backend's `context_length_exceeded`. It trims the request payload only --
 * the stored transcript is untouched -- and it does so by ELISION, never by
 * asking a model to summarise. That mirrors `src/chat/agent.rs::compact`
 * (D-DROVER-1, "the safety spine").
 *
 * Every check below is a rule from that contract. The one worth naming: no
 * synthetic marker message may appear in the payload. Injecting a
 * mid-conversation system turn would change how a row's chat template renders,
 * and template fidelity is not negotiable here. The elision is reported to the
 * caller for the UI instead. `payload_contains_only_original_messages` pins
 * that by object identity, so a future "helpful" marker fails here.
 *
 * Pure module test -- no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import {
  compactForSend,
  shouldAutoCompact,
  applySendCompaction,
  KEEP_RECENT_MESSAGES,
  AUTO_COMPACT_THRESHOLD_PERCENT,
} from '../src/lib/conversationCompaction.js'

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

/* A realistic long chat: one system turn, then alternating exchanges. */
function conversation(exchanges) {
  const messages = [{ role: 'system', content: 'you are helpful', id: 'sys' }]
  for (let i = 0; i < exchanges; i += 1) {
    messages.push({ role: 'user', content: `question ${i}`, id: `u${i}` })
    messages.push({ role: 'assistant', content: `answer ${i}`.repeat(200), id: `a${i}` })
  }
  return messages
}

console.log('the safety spine')

check('every system message survives, wherever it sits', () => {
  const messages = conversation(20)
  messages.splice(10, 0, { role: 'system', content: 'mid-chat rule', id: 'sys2' })
  const out = compactForSend(messages).messages
  const systems = out.filter((m) => m.role === 'system').map((m) => m.id)
  assert.deepEqual(systems, ['sys', 'sys2'])
})

check('every user message survives', () => {
  const messages = conversation(20)
  const before = messages.filter((m) => m.role === 'user').map((m) => m.id)
  const after = compactForSend(messages).messages.filter((m) => m.role === 'user').map((m) => m.id)
  assert.deepEqual(after, before, 'dropping an earlier question while its answer survives inverts the chat')
})

check('the current question is always the last user turn', () => {
  const messages = conversation(30)
  const out = compactForSend(messages).messages
  const lastUser = [...out].reverse().find((m) => m.role === 'user')
  assert.equal(lastUser.id, 'u29')
})

check('the last KEEP_RECENT messages survive verbatim', () => {
  const messages = conversation(20)
  const tail = messages.slice(-KEEP_RECENT_MESSAGES).map((m) => m.id)
  const out = compactForSend(messages).messages.map((m) => m.id)
  assert.deepEqual(out.slice(-KEEP_RECENT_MESSAGES), tail)
})

check('order is preserved', () => {
  const messages = conversation(20)
  const out = compactForSend(messages).messages
  const positions = out.map((m) => messages.indexOf(m))
  const sorted = [...positions].sort((a, b) => a - b)
  assert.deepEqual(positions, sorted)
})

console.log('template fidelity')

check('payload_contains_only_original_messages -- no marker is injected', () => {
  const messages = conversation(20)
  const out = compactForSend(messages).messages
  for (const message of out) {
    assert.ok(
      messages.includes(message),
      `payload gained a synthetic message (${JSON.stringify(message).slice(0, 80)}); a mid-chat marker changes chat-template rendering`,
    )
  }
})

check('the input array is never mutated', () => {
  const messages = conversation(20)
  const snapshot = messages.map((m) => m.id)
  compactForSend(messages)
  assert.deepEqual(messages.map((m) => m.id), snapshot)
})

console.log('doing nothing is a valid answer')

check('a short chat elides nothing and returns null', () => {
  assert.equal(compactForSend(conversation(2)), null)
  assert.equal(compactForSend([]), null)
  assert.equal(compactForSend(null), null)
})

check('a chat of only protected turns returns null', () => {
  const messages = [
    { role: 'system', content: 'a' },
    { role: 'user', content: 'b' },
    { role: 'user', content: 'c' },
  ]
  assert.equal(compactForSend(messages), null, 'nothing elidable means no payload change')
})

check('applySendCompaction returns the SAME array when it does not fire', () => {
  const messages = conversation(20)
  const off = applySendCompaction(messages, { enabled: false, filledPercent: 99 })
  assert.equal(off.messages, messages, 'an untriggered send must be byte-identical')
  assert.equal(off.compacted, false)
  const under = applySendCompaction(messages, { enabled: true, filledPercent: 10 })
  assert.equal(under.messages, messages)
})

console.log(`automatic threshold (${AUTO_COMPACT_THRESHOLD_PERCENT}%)`)

check('the threshold is inclusive and does not fire below it', () => {
  const at = (percent) => shouldAutoCompact({ enabled: true, filledPercent: percent })
  assert.equal(at(79.9), false)
  assert.equal(at(AUTO_COMPACT_THRESHOLD_PERCENT), true)
  assert.equal(at(80.1), true)
  assert.equal(at(100), true)
})

check('disabled means disabled, at any fill', () => {
  assert.equal(shouldAutoCompact({ enabled: false, filledPercent: 100 }), false)
})

check('a missing or broken fill reading never triggers a trim', () => {
  assert.equal(shouldAutoCompact({ enabled: true, filledPercent: NaN }), false)
  assert.equal(shouldAutoCompact({ enabled: true, filledPercent: undefined }), false)
  assert.equal(shouldAutoCompact({ enabled: true }), false)
})

check('a manual compact fires regardless of fill', () => {
  const messages = conversation(20)
  const forced = applySendCompaction(messages, { enabled: false, forced: true, filledPercent: 1 })
  assert.equal(forced.compacted, true)
  assert.ok(forced.elidedCount > 0)
  assert.ok(forced.messages.length < messages.length)
})

console.log('it actually saves something')

check('a long chat drops its middle answers', () => {
  const messages = conversation(30)
  const result = compactForSend(messages)
  assert.ok(result.elidedCount >= 20, `expected a real saving, elided ${result.elidedCount}`)
  const elidedIds = messages
    .filter((m) => !result.messages.includes(m))
    .map((m) => m.id)
  assert.ok(elidedIds.every((id) => id.startsWith('a')), 'only assistant turns may be elided')
})

console.log(`\ncontext-compaction smoke: ${checks} checks passed`)
