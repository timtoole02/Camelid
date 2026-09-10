#!/usr/bin/env node
/* Conversation organization smoke: pins, archive, tags, bulk export, import.
 *
 * The import half carries most of the weight. It reads a file the user chose,
 * so the file is DATA -- never a source of shape. Nothing is spread from it.
 * These assertions are what stop a hand-written or hostile file from putting
 * a filesystem path, a `__proto__` key, an id that overwrites existing data,
 * or an unbounded message array into stored state.
 */
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

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
  const org = await server.ssrLoadModule('/src/lib/conversationOrganization.js')
  const { parseImportedConversations } = await server.ssrLoadModule('/src/lib/conversationImport.js')
  const { exportableConversations } = await server.ssrLoadModule('/src/lib/conversationExport.js')

  const chat = (id, title, extra = {}) => ({
    id,
    title,
    updated_at: extra.updated_at || '2026-09-01T10:00:00.000Z',
    created_at: '2026-09-01T09:00:00.000Z',
    messages: [
      { id: `${id}-u`, role: 'user', content: `question about ${title}` },
      { id: `${id}-a`, role: 'assistant', content: `answer about ${title}` },
    ],
    ...extra,
  })

  /* ---- tags normalize so a filter row cannot hold two identical chips ---- */
  assert.equal(org.normalizeTag('  Rust  '), 'rust', 'tags are trimmed and lowercased')
  assert.equal(org.normalizeTag('a   b'), 'a b', 'inner whitespace collapses')
  assert.equal(org.normalizeTag('x'.repeat(50)).length, org.MAX_TAG_LENGTH, 'tags are bounded')
  assert.equal(org.normalizeTag('   '), '', 'a blank tag is no tag')
  assert.deepEqual(
    org.tagsOf({ tags: ['Rust', 'rust', ' RUST ', 'cuda'] }),
    ['rust', 'cuda'],
    'case variants of one tag collapse to one',
  )

  const tagged = org.withTagAdded(chat('c1', 'one'), 'CUDA')
  assert.deepEqual(org.tagsOf(tagged), ['cuda'])
  assert.deepEqual(org.tagsOf(org.withTagAdded(tagged, 'cuda')), ['cuda'], 'adding a tag twice is a no-op')
  let saturated = chat('c1', 'one')
  for (let i = 0; i < org.MAX_TAGS_PER_CONVERSATION + 4; i += 1) saturated = org.withTagAdded(saturated, `t${i}`)
  assert.equal(org.tagsOf(saturated).length, org.MAX_TAGS_PER_CONVERSATION, 'tags per conversation are bounded')
  assert.equal(org.withTagRemoved(tagged, 'cuda').tags, undefined, 'removing the last tag drops the field rather than storing []')

  /* ---- pin and archive are mutually exclusive --------------------------- */
  const pinnedThenArchived = org.withArchived(org.withPinned(chat('c1', 'one'), true), true)
  assert.equal(org.isArchived(pinnedThenArchived), true, 'archiving a pinned thread archives it')
  assert.equal(org.isPinned(pinnedThenArchived), false, 'and drops the pin — the two contradict')
  const archivedThenPinned = org.withPinned(org.withArchived(chat('c1', 'one'), true), true)
  assert.equal(org.isPinned(archivedThenPinned), true, 'pinning an archived thread pins it')
  assert.equal(org.isArchived(archivedThenPinned), false, 'and brings it back into the list')
  assert.equal(org.withPinned(chat('c1', 'one'), false).pinned, undefined, 'unpinning drops the field, it does not store false')

  /* ---- ordering and filtering -------------------------------------------- */
  const list = [
    chat('old', 'oldest', { updated_at: '2026-08-01T10:00:00.000Z' }),
    chat('new', 'newest', { updated_at: '2026-09-05T10:00:00.000Z' }),
    chat('pin', 'pinned one', { updated_at: '2026-07-01T10:00:00.000Z', pinned: true }),
    chat('arc', 'archived one', { updated_at: '2026-09-06T10:00:00.000Z', archived: true }),
    chat('tag', 'tagged one', { updated_at: '2026-08-15T10:00:00.000Z', tags: ['cuda'] }),
  ]

  const ids = (result) => result.map((c) => c.id)
  assert.deepEqual(
    ids(org.organizeConversations(list, {})),
    ['pin', 'new', 'tag', 'old'],
    'pinned sorts first even when it is the OLDEST, then recency; archived is hidden',
  )
  assert.deepEqual(
    ids(org.organizeConversations(list, { includeArchived: true })),
    ['pin', 'arc', 'new', 'tag', 'old'],
    'showing archived puts it back in recency order',
  )
  assert.deepEqual(ids(org.organizeConversations(list, { tags: ['cuda'] })), ['tag'], 'a tag filter narrows to that tag')
  assert.deepEqual(
    ids(org.organizeConversations(list, { search: 'cuda' })),
    ['tag'],
    'the ordinary search box matches tags too, so tags need no separate search',
  )
  assert.deepEqual(ids(org.organizeConversations(list, { search: 'answer about newest' })), ['new'], 'search still matches message bodies')
  assert.deepEqual(ids(org.organizeConversations(list, { search: 'NEWEST' })), ['new'], 'search is case-insensitive')

  /* An archived thread that matches an explicit tag filter is still shown:
     filtering to a tag and getting nothing back reads as data loss. */
  const archivedTagged = [chat('at', 'archived tagged', { archived: true, tags: ['cuda'] })]
  assert.deepEqual(
    ids(org.organizeConversations(archivedTagged, { tags: ['cuda'] })),
    ['at'],
    'an explicit tag filter reaches archived threads',
  )
  assert.deepEqual(ids(org.organizeConversations(archivedTagged, {})), [], 'but they stay hidden without one')

  assert.deepEqual(org.allTags(list).map((t) => t.tag), ['cuda'], 'tag inventory covers the list')
  assert.equal(org.allTags(list)[0].count, 1, 'with usage counts for the filter row')
  assert.equal(org.archivedCount(list), 1)
  assert.deepEqual(org.organizeConversations(null, {}), [], 'a missing list is an empty list, not a crash')

  /* ---- storage normalization --------------------------------------------- */
  const messy = org.normalizeConversationOrganization({ id: 'x', pinned: false, archived: false, tags: ['A', 'a', ''] })
  assert.equal(messy.pinned, undefined, 'falsey organization fields are dropped rather than stored forever')
  assert.equal(messy.archived, undefined)
  assert.deepEqual(messy.tags, ['a'], 'tags are re-normalized on load')
  const contradictory = org.normalizeConversationOrganization({ id: 'x', pinned: true, archived: true })
  assert.equal(contradictory.archived, undefined, 'a file claiming both pinned and archived is resolved, not obeyed')

  /* ---- bulk export ------------------------------------------------------- */
  const bundle = exportableConversations([chat('c1', 'one', { tags: ['cuda'], pinned: true, model_path: '/home/someone/models/secret.gguf' })])
  assert.equal(bundle.format, 'camelid.conversations/v1')
  assert.equal(bundle.conversation_count, 1)
  assert.deepEqual(bundle.conversations[0].tags, ['cuda'], 'tags travel so an import lands organized')
  assert.equal(bundle.conversations[0].pinned, undefined, 'pinned describes THIS machine’s list, not the conversation')
  assert.equal(
    JSON.stringify(bundle).includes('secret.gguf'),
    false,
    'the per-conversation field whitelist still governs bulk export — local paths never leave',
  )

  /* ---- import: the untrusted half ---------------------------------------- */
  const roundTrip = parseImportedConversations(JSON.stringify(bundle))
  assert.equal(roundTrip.error, null, 'a bundle this app produced imports cleanly')
  assert.equal(roundTrip.conversations.length, 1, 'round trip preserves the conversation')
  assert.equal(roundTrip.conversations[0].messages.length, 2, 'and its messages')
  assert.deepEqual(roundTrip.conversations[0].tags, ['cuda'], 'and its tags')
  assert.notEqual(roundTrip.conversations[0].id, 'c1', 'ids are ALWAYS regenerated — an import must never overwrite a conversation already here')

  const single = parseImportedConversations(JSON.stringify({ format: 'camelid.conversation/v1', title: 'solo', messages: [{ role: 'user', content: 'hi' }] }))
  assert.equal(single.conversations.length, 1, 'a single-conversation export is accepted too')
  const bare = parseImportedConversations(JSON.stringify([{ title: 'bare', messages: [{ role: 'user', content: 'hi' }] }]))
  assert.equal(bare.conversations.length, 1, 'so is a bare array')

  assert.match(parseImportedConversations('not json').error, /not valid JSON/, 'a malformed file reports rather than throws')
  assert.match(parseImportedConversations('{"nope":1}').error, /does not look like/, 'an unrelated JSON file is rejected')
  assert.match(parseImportedConversations('[]').error, /No conversations/, 'an empty list is reported, not silently accepted')

  const hostile = parseImportedConversations(JSON.stringify([{
    title: 'x'.repeat(400),
    id: 'conversation-existing',
    pinned: true,
    archived: true,
    model_path: '/home/someone/.ssh/id_rsa',
    __proto__: { polluted: true },
    tags: Array.from({ length: 40 }, (_, i) => `tag${i}`),
    messages: [
      { role: 'user', content: 'ok' },
      { role: 'root', content: 'privileged' },
      { role: 'assistant', content: '' },
      { role: 'assistant', content: 'fine', usage: { prompt_tokens: -5, completion_tokens: 3 }, usage_source: 'backend' },
      'not even an object',
    ],
  }]))
  const [imported] = hostile.conversations
  assert.equal(imported.title.length, 120, 'an over-long title is truncated, not stored whole')
  assert.notEqual(imported.id, 'conversation-existing', 'a chosen id cannot target an existing conversation')
  assert.equal(imported.pinned, undefined, 'a file cannot pin itself to the top of the list')
  assert.equal(imported.archived, undefined)
  assert.equal(imported.model_path, undefined, 'a field the exporter never writes cannot arrive by being present in the file')
  assert.equal(Object.prototype.hasOwnProperty.call(imported, 'polluted'), false, 'no prototype pollution reaches stored state')
  assert.equal(imported.tags.length, org.MAX_TAGS_PER_CONVERSATION, 'imported tags are bounded like any other')
  assert.equal(imported.messages.length, 2, 'unknown roles, empty content and non-objects are dropped')
  assert.deepEqual(imported.messages.map((m) => m.content), ['ok', 'fine'])
  assert.equal(imported.messages[1].usage.prompt_tokens, 0, 'negative token counts are clamped')
  assert.equal(
    imported.messages[1].usage_source,
    'client_estimate',
    'imported counts describe someone else’s run and can never claim to be this backend’s reported usage',
  )

  const noneReadable = parseImportedConversations(JSON.stringify([{ title: 'empty', messages: [{ role: 'user', content: '   ' }] }]))
  assert.match(noneReadable.error, /readable messages/, 'a conversation with nothing readable is reported')

  const many = parseImportedConversations(JSON.stringify(
    Array.from({ length: 600 }, (_, i) => ({ title: `c${i}`, messages: [{ role: 'user', content: 'hi' }] })),
  ))
  assert.equal(many.conversations.length, 500, 'an oversized import is bounded rather than accepted whole')

  console.log('conversation organization smoke passed')
} finally {
  await server.close()
}
