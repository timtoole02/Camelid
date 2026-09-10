/* Pinning, archiving and tagging for the conversation list.

   The sidebar shows six recent conversations and sorts strictly by recency,
   so the thread you keep coming back to falls off the list the moment six
   other chats happen. Search finds it again only if you remember a word from
   it. Pinning is the fix for that; archiving is the fix for the opposite
   problem, a list full of threads you are done with.

   Tags rather than folders. A folder is a tag that only allows one, and it
   costs a tree the reader has to build and maintain before it helps. Tags
   also fall out of the existing search box for free -- a tag is matched by
   the same query that matches a title.

   Every field here is optional. A conversation written before this existed
   has no pinned/archived/tags and behaves exactly as it did. */

export const MAX_TAG_LENGTH = 24
export const MAX_TAGS_PER_CONVERSATION = 8

/* Tags are compared and stored lowercased so "Rust" and "rust" are one tag
   rather than two that look identical in a filter row. */
export function normalizeTag(value) {
  return String(value || '')
    .trim()
    .replace(/\s+/g, ' ')
    .slice(0, MAX_TAG_LENGTH)
    .toLowerCase()
}

export function tagsOf(conversation) {
  const raw = Array.isArray(conversation?.tags) ? conversation.tags : []
  const seen = new Set()
  const tags = []
  for (const entry of raw) {
    const tag = normalizeTag(entry)
    if (!tag || seen.has(tag)) continue
    seen.add(tag)
    tags.push(tag)
  }
  return tags
}

export function isPinned(conversation) {
  return Boolean(conversation?.pinned)
}

export function isArchived(conversation) {
  return Boolean(conversation?.archived)
}

export function withPinned(conversation, pinned) {
  const next = { ...conversation }
  if (pinned) {
    next.pinned = true
    // Pinning something you had archived is a contradiction; treat the pin as
    // the more recent intent and bring it back into the list.
    delete next.archived
  } else {
    delete next.pinned
  }
  return next
}

export function withArchived(conversation, archived) {
  const next = { ...conversation }
  if (archived) {
    next.archived = true
    delete next.pinned
  } else {
    delete next.archived
  }
  return next
}

export function withTagAdded(conversation, tag) {
  const normalized = normalizeTag(tag)
  if (!normalized) return conversation
  const tags = tagsOf(conversation)
  if (tags.includes(normalized)) return conversation
  if (tags.length >= MAX_TAGS_PER_CONVERSATION) return conversation
  return { ...conversation, tags: [...tags, normalized] }
}

export function withTagRemoved(conversation, tag) {
  const normalized = normalizeTag(tag)
  const tags = tagsOf(conversation).filter((entry) => entry !== normalized)
  const next = { ...conversation }
  if (tags.length) next.tags = tags
  else delete next.tags
  return next
}

/* Every tag in use, most-used first then alphabetical, so the filter row is
   stable between renders and does not reorder as you click through it. */
export function allTags(conversations) {
  const counts = new Map()
  for (const conversation of conversations || []) {
    for (const tag of tagsOf(conversation)) {
      counts.set(tag, (counts.get(tag) || 0) + 1)
    }
  }
  return [...counts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => (b.count - a.count) || a.tag.localeCompare(b.tag))
}

function matchesSearch(conversation, query) {
  if (!query) return true
  const q = query.toLowerCase()
  if (String(conversation?.title || '').toLowerCase().includes(q)) return true
  if (tagsOf(conversation).some((tag) => tag.includes(q))) return true
  return (conversation?.messages || []).some((message) => (
    String(message?.content || '').toLowerCase().includes(q)
  ))
}

const timeOf = (conversation) => {
  const parsed = Date.parse(conversation?.updated_at || conversation?.created_at || '')
  return Number.isFinite(parsed) ? parsed : 0
}

/* One place decides what the list contains and in what order, so the sidebar,
   the history page and the tag counts can never disagree about it.

   Archived threads are excluded unless asked for OR unless a tag filter names
   them: filtering to a tag and getting nothing back, when the matching thread
   is merely archived, reads as data loss. */
export function organizeConversations(conversations, options = {}) {
  const {
    search = '',
    tags: tagFilter = [],
    includeArchived = false,
  } = options
  const query = String(search || '').trim().toLowerCase()
  const wanted = (Array.isArray(tagFilter) ? tagFilter : []).map(normalizeTag).filter(Boolean)
  const list = Array.isArray(conversations) ? conversations : []

  const matched = list.filter((conversation) => {
    if (!matchesSearch(conversation, query)) return false
    if (wanted.length) {
      const owned = tagsOf(conversation)
      if (!wanted.every((tag) => owned.includes(tag))) return false
      return true
    }
    return includeArchived || !isArchived(conversation)
  })

  return matched.slice().sort((a, b) => {
    const pinDelta = Number(isPinned(b)) - Number(isPinned(a))
    if (pinDelta) return pinDelta
    return timeOf(b) - timeOf(a)
  })
}

export function archivedCount(conversations) {
  return (conversations || []).filter(isArchived).length
}

/* Storage normalization: drop empty/absent organization fields rather than
   storing `pinned: false` and `tags: []` on every conversation forever, and
   re-normalize tags that arrived from an import or a hand-edited file. */
export function normalizeConversationOrganization(conversation) {
  if (!conversation || typeof conversation !== 'object') return conversation
  const next = { ...conversation }
  if (next.pinned) next.pinned = true
  else delete next.pinned
  if (next.archived) next.archived = true
  else delete next.archived
  const tags = tagsOf(next).slice(0, MAX_TAGS_PER_CONVERSATION)
  if (tags.length) next.tags = tags
  else delete next.tags
  // A pinned thread is never also archived, whatever the file said.
  if (next.pinned && next.archived) delete next.archived
  return next
}
