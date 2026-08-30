/* Single-conversation export (Phase 2). Field WHITELIST, not blacklist: only
   the fields named here can ever reach an export, so local filesystem paths
   (model_path etc.) are excluded by construction (I7). Token counts and
   timings ride along explicitly labeled as client/backend telemetry — they are
   operational data, not support evidence (I4). */

const MESSAGE_EXPORT_FIELDS = [
  'id', 'role', 'content', 'created_at', 'model_id', 'model_name',
  'finish_reason', 'usage', 'usage_source', 'elapsed_ms',
  'first_content_ms', 'tokens_out_per_sec', 'support_row',
  'web_research',
]

const SUPPORT_ROW_FIELDS = ['id', 'status', 'supported']
const WEB_RESEARCH_FIELDS = ['reason', 'query', 'sources', 'warnings']
const WEB_SOURCE_FIELDS = ['title', 'url']

function pick(source, fields) {
  const out = {}
  for (const field of fields) {
    if (source?.[field] !== undefined && source?.[field] !== null) out[field] = source[field]
  }
  return out
}

function safeWebUrl(value) {
  try {
    const url = new URL(String(value || ''))
    return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null
  } catch {
    return null
  }
}

function markdownLabel(value) {
  return String(value || '').replace(/([\\\[\]`*_<>])/g, '\\$1').replace(/[\r\n]+/g, ' ').trim()
}

export function exportableConversation(conversation) {
  return {
    format: 'camelid.conversation/v1',
    exported_at: new Date().toISOString(),
    telemetry_note: 'Timing/token fields are operational telemetry (client-measured unless usage_source=backend). They are not compatibility or support evidence.',
    id: conversation?.id || null,
    title: conversation?.title || 'Conversation',
    created_at: conversation?.created_at || null,
    updated_at: conversation?.updated_at || null,
    model_id: conversation?.model_id || null,
    messages: (conversation?.messages || []).map((message) => {
      const picked = pick(message, MESSAGE_EXPORT_FIELDS)
      if (picked.support_row) picked.support_row = pick(picked.support_row, SUPPORT_ROW_FIELDS)
      if (picked.web_research) {
        picked.web_research = pick(picked.web_research, WEB_RESEARCH_FIELDS)
        picked.web_research.sources = (picked.web_research.sources || [])
          .map((source) => pick(source, WEB_SOURCE_FIELDS))
          .map((source) => ({ ...source, url: safeWebUrl(source.url) }))
          .filter((source) => source.url)
        picked.web_research.warnings = (picked.web_research.warnings || []).map(String).filter(Boolean)
      }
      return picked
    }),
  }
}

export function conversationToJson(conversation) {
  return JSON.stringify(exportableConversation(conversation), null, 2)
}

export function conversationToMarkdown(conversation) {
  const data = exportableConversation(conversation)
  const lines = [`# ${data.title}`, '']
  if (data.model_id) lines.push(`Model: \`${data.model_id}\`  `)
  if (data.created_at) lines.push(`Started: ${data.created_at}  `)
  lines.push('', '---', '')
  for (const message of data.messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue
    lines.push(`## ${message.role === 'user' ? 'You' : 'Camelid'}`, '')
    lines.push(message.content || '', '')
    if (message.role === 'assistant') {
      const meta = []
      if (message.model_id) meta.push(`model \`${message.model_id}\``)
      if (message.support_row?.id) meta.push(`row \`${message.support_row.id}\` (${message.support_row.status})`)
      if (message.usage?.completion_tokens !== undefined) {
        meta.push(`${message.usage.prompt_tokens}→${message.usage.completion_tokens} tokens${message.usage_source === 'backend' ? '' : ' (client estimate)'}`)
      }
      if (meta.length) lines.push(`> ${meta.join(' · ')} — telemetry, not support evidence`, '')
      if (message.web_research?.sources?.length || message.web_research?.warnings?.length) {
        lines.push('### Web sources', '')
        message.web_research.sources.forEach((source) => lines.push(`- [${markdownLabel(source.title || source.url)}](<${source.url}>)`))
        message.web_research.warnings.forEach((warning) => lines.push(`> Web research warning: ${markdownLabel(warning)}`))
        lines.push('')
      }
    }
  }
  return lines.join('\n')
}

export function downloadConversation(conversation, format = 'markdown') {
  const isJson = format === 'json'
  const content = isJson ? conversationToJson(conversation) : conversationToMarkdown(conversation)
  const blob = new Blob([content], { type: isJson ? 'application/json' : 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  const safeTitle = String(conversation?.title || 'conversation').replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48) || 'conversation'
  anchor.href = url
  anchor.download = `${safeTitle}.${isJson ? 'json' : 'md'}`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
