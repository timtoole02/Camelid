/* Conversation import.

   Export existed; import did not, which meant a transcript could leave this
   machine but never arrive on another one. That is the actual gap: the data
   was already portable, it just had nowhere to land.

   This reads a file the user chose, so the file is DATA, never instructions
   and never a source of shape. Nothing is spread from it. Every field is
   copied by name, coerced to the type this app expects, and bounded. A field
   the exporter never writes cannot arrive by being present in the JSON --
   which is what keeps a hand-written file from injecting `pinned`, a
   filesystem path, an enormous message array, or a `__proto__` key into
   stored state.

   Ids are always regenerated. Reusing an id from the file would let an
   import silently overwrite a conversation already on this machine, and an
   import that destroys existing data is worse than no import. */

import { normalizeConversationOrganization, normalizeTag, MAX_TAGS_PER_CONVERSATION } from './conversationOrganization.js'

export const IMPORT_FORMATS = ['camelid.conversation/v1', 'camelid.conversations/v1']

const MAX_CONVERSATIONS_PER_IMPORT = 500
const MAX_MESSAGES_PER_CONVERSATION = 5000
const MAX_CONTENT_CHARS = 200_000
const ROLES = ['user', 'assistant', 'system']

const str = (value, max) => {
  if (typeof value !== 'string') return ''
  return max ? value.slice(0, max) : value
}

const isoOrNull = (value) => {
  if (typeof value !== 'string') return null
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : null
}

const finiteOrNull = (value) => {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function importUsage(raw) {
  if (!raw || typeof raw !== 'object') return null
  const prompt = finiteOrNull(raw.prompt_tokens)
  const completion = finiteOrNull(raw.completion_tokens)
  if (prompt === null && completion === null) return null
  return {
    prompt_tokens: Math.max(0, prompt ?? 0),
    completion_tokens: Math.max(0, completion ?? 0),
    total_tokens: Math.max(0, (prompt ?? 0) + (completion ?? 0)),
  }
}

function importMessage(raw, index) {
  if (!raw || typeof raw !== 'object') return null
  const role = ROLES.includes(raw.role) ? raw.role : null
  if (!role) return null
  const content = str(raw.content, MAX_CONTENT_CHARS)
  if (!content.trim()) return null
  const usage = importUsage(raw.usage)
  const message = {
    id: `imported-message-${index}-${Math.random().toString(36).slice(2, 10)}`,
    role,
    content,
    created_at: isoOrNull(raw.created_at) || new Date().toISOString(),
  }
  const modelId = str(raw.model_id, 200)
  if (modelId) message.model_id = modelId
  if (usage) {
    message.usage = usage
    /* Imported counts describe a run on someone else's machine. Labelling
       them as this backend's reported usage would let another engine's
       numbers masquerade as Camelid telemetry, so they are always an
       estimate here whatever the file claimed. */
    message.usage_source = 'client_estimate'
  }
  if (['stop', 'length', 'error', 'interrupted'].includes(raw.finish_reason)) {
    message.finish_reason = raw.finish_reason
  }
  return message
}

function importConversation(raw, index) {
  if (!raw || typeof raw !== 'object') return null
  const rawMessages = Array.isArray(raw.messages) ? raw.messages.slice(0, MAX_MESSAGES_PER_CONVERSATION) : []
  const messages = rawMessages.map(importMessage).filter(Boolean)
  if (!messages.length) return null
  const created = isoOrNull(raw.created_at) || new Date().toISOString()
  const tags = (Array.isArray(raw.tags) ? raw.tags : [])
    .map(normalizeTag)
    .filter(Boolean)
    .slice(0, MAX_TAGS_PER_CONVERSATION)
  const conversation = {
    id: `conversation-imported-${Date.now()}-${index}-${Math.random().toString(36).slice(2, 10)}`,
    title: str(raw.title, 120).trim() || 'Imported conversation',
    created_at: created,
    updated_at: isoOrNull(raw.updated_at) || created,
    messages,
  }
  const modelId = str(raw.model_id, 200)
  if (modelId) conversation.model_id = modelId
  if (tags.length) conversation.tags = tags
  return normalizeConversationOrganization(conversation)
}

/* Accepts a single-conversation export, a bulk export, or a bare array.
   Returns { conversations, skipped, error } -- never throws, because the
   caller is a file picker and a malformed file is an ordinary outcome. */
export function parseImportedConversations(text) {
  let parsed
  try {
    parsed = JSON.parse(String(text || ''))
  } catch {
    return { conversations: [], skipped: 0, error: 'That file is not valid JSON.' }
  }

  let rawList
  if (Array.isArray(parsed)) rawList = parsed
  else if (Array.isArray(parsed?.conversations)) rawList = parsed.conversations
  else if (parsed && typeof parsed === 'object' && Array.isArray(parsed.messages)) rawList = [parsed]
  else {
    return { conversations: [], skipped: 0, error: 'That file does not look like a Camelid conversation export.' }
  }

  const bounded = rawList.slice(0, MAX_CONVERSATIONS_PER_IMPORT)
  const conversations = bounded.map(importConversation).filter(Boolean)
  const skipped = rawList.length - conversations.length
  if (!conversations.length) {
    return { conversations: [], skipped, error: 'No conversations in that file had any readable messages.' }
  }
  return { conversations, skipped, error: null }
}
