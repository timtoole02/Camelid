#!/usr/bin/env node
/* Regenerated-reply siblings smoke.
 *
 * The contract that matters is the mirroring one: the active variant's fields
 * are ALSO the message's own fields, so every existing reader (renderer,
 * exporter, request builder, context estimator, telemetry footer) keeps
 * working without knowing variants exist. If a switch ever leaves the message
 * showing one alternative's text beside another's token counts, that is the
 * bug this file is here to catch.
 */
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
    activeVariantIndexOf,
    canBranchMessage,
    hasVariants,
    normalizeMessageVariants,
    snapshotVariant,
    variantCountOf,
    variantsOf,
    withActiveVariant,
    withActiveVariantRemoved,
    withVariantAppended,
  } = await server.ssrLoadModule('/src/lib/messageVariants.js')

  const reply = (content, extra = {}) => ({
    id: 'm1',
    role: 'assistant',
    content,
    finish_reason: 'stop',
    usage: { prompt_tokens: 10, completion_tokens: content.length, total_tokens: 10 + content.length },
    ...extra,
  })

  /* ---- migration: a pre-variants message IS its own first variant -------- */
  const legacy = reply('the only answer')
  assert.equal(variantCountOf(legacy), 1, 'a stored reply with no variants counts as one')
  assert.equal(hasVariants(legacy), false, 'and does not claim to have alternatives')
  assert.equal(activeVariantIndexOf(legacy), 0, 'its active index is its only index')
  assert.equal(variantsOf(legacy)[0].content, 'the only answer', 'migration happens on read, with no rewrite of stored data')

  /* ---- a variant is the message minus its identity ----------------------- */
  const snapshot = snapshotVariant(reply('x', { camelid_receipt: { seal: 'abc' }, variants: ['ignored'] }))
  assert.equal(snapshot.id, undefined, 'identity fields never enter a variant')
  assert.equal(snapshot.role, undefined)
  assert.equal(snapshot.variants, undefined, 'variants do not nest inside variants')
  assert.deepEqual(snapshot.camelid_receipt, { seal: 'abc' }, 'everything else rides along — defining a variant by subtraction means new reply fields carry automatically')

  /* ---- appending ---------------------------------------------------------- */
  const two = withVariantAppended(legacy, reply('a second answer'))
  assert.equal(variantCountOf(two), 2, 'regenerating adds a sibling rather than replacing')
  assert.equal(activeVariantIndexOf(two), 1, 'and selects the new one')
  assert.equal(two.content, 'a second answer', 'the message mirrors the active variant')
  assert.equal(two.id, 'm1', 'the message keeps its identity across a re-roll')
  assert.equal(two.variants[0].content, 'the only answer', 'the original is still there')

  const three = withVariantAppended(two, reply('a third answer'))
  assert.equal(variantCountOf(three), 3, 'a re-roll of a re-roll keeps the whole set rather than collapsing to two')
  assert.deepEqual(
    three.variants.map((v) => v.content),
    ['the only answer', 'a second answer', 'a third answer'],
    'siblings stay in generation order',
  )

  /* ---- switching mirrors EVERY field, not just the text ------------------ */
  const back = withActiveVariant(three, 0)
  assert.equal(back.content, 'the only answer', 'switching shows the chosen text')
  assert.equal(
    back.usage.completion_tokens,
    'the only answer'.length,
    'and its OWN token counts — a footer describing a different answer is the bug this guards',
  )
  assert.equal(back.active_variant, 0)
  assert.equal(variantCountOf(back), 3, 'switching never loses siblings')
  assert.equal(withActiveVariant(three, 99).active_variant, 2, 'an out-of-range index is clamped, not crashed')
  assert.equal(withActiveVariant(three, -5).active_variant, 0, 'including below zero')

  /* ---- discarding -------------------------------------------------------- */
  const discarded = withActiveVariantRemoved(withActiveVariant(three, 1))
  assert.equal(variantCountOf(discarded), 2, 'discarding removes exactly one')
  assert.deepEqual(discarded.variants.map((v) => v.content), ['the only answer', 'a third answer'])
  assert.equal(discarded.content, 'a third answer', 'and selects a neighbour')
  const lastOne = withActiveVariantRemoved(legacy)
  assert.equal(variantCountOf(lastOne), 1, 'the last remaining version cannot be discarded')
  assert.equal(lastOne.content, 'the only answer', 'a reply with no content has nothing to render')

  /* ---- eligibility -------------------------------------------------------- */
  assert.equal(canBranchMessage(reply('done')), true)
  assert.equal(canBranchMessage({ ...reply('x'), streaming: true }), false, 'a streaming reply is not finished')
  assert.equal(canBranchMessage({ ...reply('x'), role: 'user' }), false, 'user turns use Edit & resend, which changes the question instead of re-asking it')
  assert.equal(canBranchMessage(reply('   ')), false, 'an empty reply has nothing to re-roll')
  assert.equal(canBranchMessage(null), false)

  /* ---- storage normalization --------------------------------------------- */
  const desynced = { id: 'm1', role: 'assistant', content: 'STALE', usage: { completion_tokens: 999 }, variants: [{ content: 'first' }, { content: 'second' }], active_variant: 1 }
  const fixed = normalizeMessageVariants(desynced)
  assert.equal(fixed.content, 'second', 'a hand-edited or restored transcript is re-synced to its active variant')
  assert.equal(fixed.usage, undefined, 'stale mirrored fields the active variant does not carry are dropped, not left behind')
  const singleton = normalizeMessageVariants({ id: 'm1', role: 'assistant', content: 'x', variants: [{ content: 'x' }], active_variant: 0 })
  assert.equal(singleton.variants, undefined, 'a single-element variant list carries no alternatives and is not stored')
  assert.equal(singleton.content, 'x', 'while the reply itself is untouched')
  const userTurn = { id: 'u1', role: 'user', content: 'hi' }
  assert.deepEqual(normalizeMessageVariants(userTurn), userTurn, 'user turns pass through unchanged')

  /* ---- storage round trip ------------------------------------------------- */
  const { normalizeStoredConversations } = await server.ssrLoadModule('/src/lib/conversationStorage.js')
  const [restored] = normalizeStoredConversations([{ id: 'c1', messages: [desynced] }])
  assert.equal(restored.messages[0].content, 'second', 'the same re-sync happens on every conversation load')

  /* ---- the control ------------------------------------------------------- */
  const { MessageTurn } = await server.ssrLoadModule('/src/components/chat/MessageTurn.jsx')
  const renderTurn = (message, props = {}) => renderToStaticMarkup(React.createElement(MessageTurn, { message, ...props }))

  const plain = renderTurn(reply('just one'))
  assert.doesNotMatch(plain, /cxturn__variants/, 'a reply with one version shows no navigation — an ordinary thread is visually unchanged')

  const branched = renderTurn(three, { onSelectVariant: () => {}, onDiscardVariant: () => {} })
  assert.match(branched, /cxturn__variants/, 'a re-rolled reply shows navigation')
  assert.match(branched, /3\/3/, 'labelled with position and total')
  assert.match(branched, /aria-label="Reply 3 of 3"/, 'and announced to a screen reader')
  assert.match(branched, /Previous version of this reply/, 'with a way back to the earlier answer')
  assert.match(branched, /Discard this version/, 'and a way to drop the one shown')

  const noDiscard = renderTurn(three, { onSelectVariant: () => {} })
  assert.doesNotMatch(noDiscard, /Discard this version/, 'discard is withheld while a turn is in flight')
  assert.match(noDiscard, /cxturn__variants/, 'but navigation stays usable — switching costs nothing')

  /* Regenerate's promise differs by position, so its label must too. */
  const keeps = renderTurn(reply('x'), { onRegenerate: () => {} })
  assert.match(keeps, /keeps this one alongside it/, 'on the last reply, Regenerate says the answer is kept')
  const replaces = renderTurn(reply('x'), { onRegenerate: () => {}, regenerateReplacesThread: true })
  assert.match(replaces, /every turn after it/, 'mid-thread it warns that later turns go with it')

  console.log('message variants smoke passed')
} finally {
  await server.close()
}
