import { normalizeMessageVariants } from './messageVariants.js'

export function cleanLegacyDemoCapCopy(value) {
  if (typeof value !== 'string') return value
  const stripped = value
    .replace(/\s*\(demo cap\)/gi, '')
    .replace(/\s*·\s*raw\s+16-token-cap\s+local\s+run;\s*inspect\s+before\s+trusting\s+polish/gi, ' · raw local run')
    .replace(/\s*Longer-generation\s+polish\s+still\s+needs\s+separate\s+validation\.?/gi, '')
    .replace(/\s*Longer\s+generation\s+is\s+not\s+polished\s+yet\.?/gi, '')
  // Only tidy up whitespace when a legacy phrase was actually removed; content
  // with no legacy phrases is returned untouched so multi-line code and
  // markdown keep their newlines and indentation intact. Even when cleaning,
  // collapse only runs of horizontal whitespace (spaces/tabs) — never newlines.
  if (stripped === value) return value
  return stripped.replace(/[^\S\n]{2,}/g, ' ').trim()
}

export function normalizeStoredMessage(message, { clearStaleStreaming = false } = {}) {
  if (!message || typeof message !== 'object') return message
  const { demo_token_cap: _demoTokenCap, ...stored } = message
  /* Keep the sibling list and the mirrored top-level fields in step. A
     transcript restored from an export, hand-edited, or written by a build
     that predates variants must never render one alternative's text beside
     another's token counts. */
  const rest = normalizeMessageVariants(stored)
  const content = cleanLegacyDemoCapCopy(rest.content)
  if (clearStaleStreaming && rest.streaming) {
    return {
      ...rest,
      content: content && content !== '…' ? content : '(generation interrupted)',
      finish_reason: rest.finish_reason || 'interrupted',
      streaming: false,
      streaming_phase: null,
    }
  }
  return {
    ...rest,
    content,
  }
}

export function normalizeStoredConversations(records, options = {}) {
  return (Array.isArray(records) ? records : []).map((conversation) => ({
    ...conversation,
    messages: Array.isArray(conversation?.messages)
      ? conversation.messages.map((message) => normalizeStoredMessage(message, options))
      : [],
  }))
}
