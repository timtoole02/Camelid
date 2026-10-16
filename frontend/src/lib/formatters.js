export function formatRate(value) {
  if (value === null || value === undefined) return '—'
  return `${Number(value).toFixed(1)} tok/s`
}

export function formatDurationMs(value) {
  const duration = Number(value)
  if (!Number.isFinite(duration) || duration <= 0) return '0 ms'
  if (duration < 1) return `${Math.round(duration * 1000)} μs`
  if (duration < 1000) return `${duration.toFixed(duration < 10 ? 1 : 0)} ms`
  return `${(duration / 1000).toFixed(1)} s`
}

export function formatDate(value) {
  if (!value) return ''
  return new Date(value).toLocaleString()
}

export function formatSidebarDate(value) {
  if (!value) return ''
  const date = new Date(value)
  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()
  if (sameDay) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  }
  return date.toLocaleDateString([], { month: 'numeric', day: 'numeric' })
}

export function formatHistoryDate(value) {
  if (!value) return ''
  const date = new Date(value)
  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()
  if (sameDay) {
    return `Today, ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/* Decimal units, matching the labels. This divided by 1024 while printing KB/MB/GB,
   so every size in the UI was a binary magnitude wearing an SI name -- and the CLI,
   which uses `/1e9`, disagreed with the UI about the same file:

     Llama-3.2-3B-Instruct-Q8_0.gguf   CLI "3.4 GB"   UI "3.2 GB"

   Two numbers for one file is worse than either convention being wrong, and the
   catalog literals, the Hub's own listing and `camelid pull` are all decimal, so
   the UI was the outlier. Dividing by 1000 makes the labels honest and the two
   surfaces agree. Sizes shown in the UI go up slightly; nothing downloads more. */
export function formatBytes(value) {
  if (value === null || value === undefined) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = Number(value)
  let unit = 0
  while (size >= 1000 && unit < units.length - 1) {
    size /= 1000
    unit += 1
  }
  return `${size.toFixed(size >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

export function formatCompactNumber(value) {
  if (value === null || value === undefined) return '0'
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value))
}

/* Card previews are plain text, so markdown syntax the model emitted (fences,
   emphasis, link brackets) must not leak through as literal characters. Only
   the markers are removed — the words and code inside are kept. */
function stripMarkdown(value) {
  return String(value)
    .replace(/```[\w+-]*\n?/g, ' ')            // fence openers (with language) and closers
    .replace(/~~~[\w+-]*\n?/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')  // images -> alt text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')   // links -> link text
    .replace(/`([^`]+)`/g, '$1')               // inline code
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')        // headings
    .replace(/^\s{0,3}>\s?/gm, '')             // blockquotes
    .replace(/^\s{0,3}(?:[-*+]|\d+\.)\s+/gm, '') // list markers
    .replace(/^\s{0,3}(?:[-*_]\s*){3,}$/gm, ' ') // horizontal rules
    .replace(/(\*\*|__)(.*?)\1/g, '$2')        // bold
    .replace(/(^|\W)[*_]([^*_\n]+)[*_](?=\W|$)/g, '$1$2') // italic
    .replace(/~~(.*?)~~/g, '$1')               // strikethrough
}

export function formatPreview(value, maxLength = 120) {
  if (!value) return 'No messages yet'
  const normalized = stripMarkdown(value).replace(/\s+/g, ' ').trim()
  if (!normalized) return 'No messages yet'
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 1).trimEnd()}…`
}

/* Conversations record whichever model name was in play at the time: some hold
   a display name ("Llama 3.2 1B Instruct"), older ones a raw catalog id
   ("tinyllama_tinyllama-1.1b-chat-v1.0-q8_0"). Render both the same way. */
const MODEL_FAMILY_NAMES = new Map([
  ['tinyllama', 'TinyLlama'], ['llama', 'Llama'], ['llama3', 'Llama 3'], ['qwen', 'Qwen'], ['qwen2', 'Qwen2'],
  ['qwen3', 'Qwen3'], ['gemma', 'Gemma'], ['gemma3', 'Gemma 3'], ['gemma4', 'Gemma 4'],
  ['phi3', 'Phi-3'], ['mistral', 'Mistral'], ['deepseek', 'DeepSeek'], ['bonsai', 'Bonsai'],
  ['nomic', 'Nomic'], ['ornith', 'Ornith'], ['cohere', 'Cohere'],
])

export function formatModelLabel(value) {
  const raw = String(value || '').trim()
  if (!raw) return 'No model recorded'
  // Already a display name (or a HF repo path) — leave it alone.
  if (/\s/.test(raw)) return raw
  const tail = raw.split('/').pop().replace(/\.gguf$/i, '')
  // The generation number is separate from the parameter count (2 4B, not 2.4B).
  const bonsai2 = tail.match(/^ternary[-_]bonsai[-_]2[-_](2|4|9|27)b[-_]ptq1[-_]0$/i)
  if (bonsai2) return `Bonsai 2 ${bonsai2[1]}B PTQ1_0`
  const tokens = tail.split(/[_-]+/).filter(Boolean)
  const merged = []
  for (const token of tokens) {
    const previous = merged[merged.length - 1]
    // Drop a duplicated family prefix ("tinyllama_tinyllama-...").
    if (previous && previous.toLowerCase() === token.toLowerCase()) continue
    // Rejoin split quant labels, repeatedly: ["q4","k","m"] -> "Q4_K_M".
    if (previous && /^i?q\d+(?:_[a-z0-9]+)*$/i.test(previous) && /^(?:\d+|[a-z]{1,2})$/i.test(token)) {
      merged[merged.length - 1] = `${previous.toUpperCase()}_${token.toUpperCase()}`
      continue
    }
    // Rejoin split versions: ["v0","3"] -> "v0.3".
    if (previous && /^v\d+$/i.test(previous) && /^\d+$/.test(token)) {
      merged[merged.length - 1] = `${previous.toLowerCase()}.${token}`
      continue
    }
    // Rejoin split parameter counts: ["0","6b"] -> "0.6B".
    if (previous && /^\d+$/.test(previous) && /^\d+[bm]$/i.test(token)) {
      merged[merged.length - 1] = `${previous}.${token.toUpperCase()}`
      continue
    }
    merged.push(token)
  }
  return merged
    .map((token) => {
      const lower = token.toLowerCase()
      if (MODEL_FAMILY_NAMES.has(lower)) return MODEL_FAMILY_NAMES.get(lower)
      // Catalog ids fuse family and version: "llama32" -> "Llama 3.2".
      const versioned = lower.match(/^([a-z]+)(\d)(\d)$/)
      if (versioned && MODEL_FAMILY_NAMES.has(versioned[1])) {
        return `${MODEL_FAMILY_NAMES.get(versioned[1])} ${versioned[2]}.${versioned[3]}`
      }
      if (/_/.test(token)) return token                       // already-normalized quant
      if (/^i?q\d/i.test(token)) return token.toUpperCase()   // q8, iq4xs
      if (/^\d+(?:\.\d+)?[bm]$/i.test(token)) return token.toUpperCase() // 1.1b -> 1.1B
      if (/^a\d+b$/i.test(token)) return token.toUpperCase()  // a4b (active params) -> A4B
      if (/^v\d/i.test(token)) return token.toLowerCase()     // v1.0
      return token.charAt(0).toUpperCase() + token.slice(1)
    })
    .join(' ')
}

export function clampText(value, maxLength = 72) {
  if (!value) return ''
  const normalized = String(value).replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength - 1)}…`
}
