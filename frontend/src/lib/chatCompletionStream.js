function defaultEstimateTokenCount(value) {
  const text = String(value || '').trim()
  if (!text) return 0
  const wordPieces = text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
  return Math.max(1, Math.round(Math.max(wordPieces.length, text.length / 4)))
}

export function extractSseEvents(buffer) {
  const normalized = String(buffer || '').replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  return {
    events: parts.slice(0, -1),
    remainder: parts.at(-1) || '',
  }
}

function readSseDataLines(eventText) {
  return String(eventText || '')
    .replace(/\r\n/g, '\n')
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
}

function readSsePayloads(eventText, { errorEvent = false } = {}) {
  const dataLines = readSseDataLines(eventText)
  if (dataLines.length <= 1) return dataLines

  // SSE allows one logical payload to be split across several `data:` lines.
  // Prefer the spec-compliant joined payload when it parses, but keep accepting
  // backend batches that place several complete JSON payloads in one event.
  const joinedPayload = dataLines.join('\n')
  try {
    JSON.parse(joinedPayload)
    return [joinedPayload]
  } catch {
    return errorEvent ? [joinedPayload] : dataLines
  }
}

function isSseErrorEvent(eventText) {
  return String(eventText || '')
    .replace(/\r\n/g, '\n')
    .split('\n')
    .some((line) => line.trim().toLowerCase() === 'event: error')
}

function makeStreamPayloadError(payload, fallback = 'Streaming response failed.') {
  const message = payload?.error?.message || payload?.message || fallback
  const error = new Error(message)
  error.payload = payload
  error.code = payload?.error?.code || payload?.code || ''
  return error
}

export function readChatCompletionJsonPayload(payload, { estimateTokenCount = defaultEstimateTokenCount } = {}) {
  const choice = payload?.choices?.[0]
  const content = choice?.message?.content ?? choice?.text ?? ''
  return {
    content,
    finishReason: choice?.finish_reason ?? null,
    completionTokens: payload?.usage?.completion_tokens ?? estimateTokenCount(content),
    firstContentMs: null,
    usage: payload?.usage || null,
  }
}

export async function readStreamingChatCompletion(response, onDelta, { estimateTokenCount = defaultEstimateTokenCount, onStreamEvent = null } = {}) {
  if (!response.ok) {
    let detail = null
    try {
      detail = await response.json()
    } catch {
      // Fall through to generic response status below.
    }
    const message = detail?.error?.message || detail?.message || `Request failed with HTTP ${response.status}`
    const error = new Error(message)
    error.payload = detail
    throw error
  }

  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    const responseStartedAt = performance.now()
    onStreamEvent?.({ type: 'json_fallback', elapsedMs: 0, firstByteMs: 0, firstEventMs: null, firstContentMs: null })
    const payload = await response.json()
    const parsed = readChatCompletionJsonPayload(payload, { estimateTokenCount })
    const elapsedMs = performance.now() - responseStartedAt
    if (parsed.content) onDelta(parsed.content, parsed.content, { completionTokens: parsed.completionTokens, elapsedMs, firstByteMs: 0, firstEventMs: null, firstContentMs: elapsedMs })
    return { ...parsed, camelid: payload?.camelid || null, camelidReceipt: payload?.camelid_receipt || null, firstByteMs: 0, firstEventMs: null, firstContentMs: parsed.content ? elapsedMs : null }
  }

  const reader = response.body?.getReader()
  if (!reader) return { content: '', finishReason: null, completionTokens: 0, firstContentMs: null, firstByteMs: null, firstEventMs: null, usage: null, camelid: null, camelidReceipt: null }
  const decoder = new TextDecoder()
  let buffer = ''
  let content = ''
  let finishReason = null
  let completionTokens = 0
  let usage = null
  let camelid = null
  const streamStartedAt = performance.now()
  let firstByteMs = null
  let firstEventMs = null
  let firstContentMs = null

  const streamMetrics = () => ({
    completionTokens,
    elapsedMs: performance.now() - streamStartedAt,
    firstByteMs,
    firstEventMs,
    firstContentMs,
  })

  const consumeEvent = (eventText) => {
    const dataLines = readSseDataLines(eventText)
    const errorEvent = isSseErrorEvent(eventText)
    const payloads = readSsePayloads(eventText, { errorEvent })
    if (dataLines.length && firstEventMs === null) firstEventMs = performance.now() - streamStartedAt
    for (const data of payloads) {
      if (!data) continue
      if (data === '[DONE]') {
        onStreamEvent?.({ type: 'done', ...streamMetrics() })
        continue
      }
      let chunk = null
      try {
        chunk = JSON.parse(data)
      } catch {
        if (errorEvent) throw makeStreamPayloadError(null, data || 'Streaming response failed.')
        continue
      }
      if (chunk?.error || errorEvent) {
        onStreamEvent?.({ type: 'error', error: chunk?.error || chunk, ...streamMetrics() })
        throw makeStreamPayloadError(chunk)
      }
      const choice = chunk?.choices?.[0]
      const role = choice?.delta?.role ?? null
      const delta = choice?.delta?.content ?? choice?.text ?? ''
      const segment = choice?.delta?.camelid_segment
      // Reasoning deltas are GENERATED TOKENS even though they are not visible
      // content. A thinking model can spend an entire reply in `reasoning_content`
      // (LFM2's generation prompt opens a <think> block), so counting only
      // `content` made the tok/s readout report a fraction of the real rate —
      // or zero. Counted here, never appended to `content`.
      const reasoningDelta = choice?.delta?.reasoning_content ?? ''
      if (role && !delta && !reasoningDelta) onStreamEvent?.({ type: 'role', role, ...streamMetrics() })
      if (reasoningDelta && !delta) {
        completionTokens += 1
        onStreamEvent?.({ type: 'reasoning', delta: reasoningDelta, ...streamMetrics() })
      }
      if (delta) {
        completionTokens += 1
        if (firstContentMs === null) firstContentMs = performance.now() - streamStartedAt
        content += delta
        const metrics = streamMetrics()
        onStreamEvent?.({ type: 'content', delta, ...metrics })
        onDelta(delta, content, metrics)
      }
      if (segment && typeof segment === 'object') {
        const boundary = typeof segment.boundary === 'string' ? segment.boundary : ''
        onStreamEvent?.({ type: 'segment', segment, ...streamMetrics() })
        // Segment separators are presentation-only Markdown. They are emitted
        // separately from model content so they never inflate the exact token
        // count or target-verification receipt.
        if (boundary) {
          content += boundary
          onDelta(boundary, content, streamMetrics())
        }
      }
      if (choice?.finish_reason) {
        finishReason = choice.finish_reason
        onStreamEvent?.({ type: 'finish', finishReason, ...streamMetrics() })
      }
      if (chunk?.usage && typeof chunk.usage === 'object') {
        usage = chunk.usage
        if (Number.isFinite(Number(usage.completion_tokens))) completionTokens = Number(usage.completion_tokens)
        onStreamEvent?.({ type: 'usage', usage, ...streamMetrics() })
      }
      if (chunk?.camelid) {
        camelid = chunk.camelid
      }
    }
  }

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    if (firstByteMs === null) {
      firstByteMs = performance.now() - streamStartedAt
      onStreamEvent?.({ type: 'bytes', bytes: value.byteLength, ...streamMetrics() })
    }
    buffer += decoder.decode(value, { stream: true })
    const { events, remainder } = extractSseEvents(buffer)
    events.forEach(consumeEvent)
    buffer = remainder
  }
  buffer += decoder.decode()
  if (buffer.trim()) consumeEvent(buffer.replace(/\r\n/g, '\n'))
  // Receipts only attach to non-streaming responses (the JSON fallback above).
  return { content, finishReason, completionTokens, firstContentMs, firstByteMs, firstEventMs, usage, camelid, camelidReceipt: null }
}
