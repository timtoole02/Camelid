import crypto from 'node:crypto'
import path from 'node:path'

export function sha256(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex')
}

export function sanitizeText(value) {
  const text = String(value ?? '')
  return {
    char_count: text.length,
    sha256: sha256(text),
  }
}

export function sanitizeMessages(messages) {
  const normalized = Array.isArray(messages) ? messages : []
  return {
    message_count: normalized.length,
    roles: normalized.map(message => String(message?.role ?? '')),
    content_char_counts: normalized.map(message => String(message?.content ?? '').length),
    content_sha256: normalized.map(message => sha256(message?.content ?? '')),
    combined_sha256: sha256(JSON.stringify(normalized)),
  }
}

export function sanitizePath(value) {
  if (!value) return null
  return {
    basename: path.basename(String(value)),
    sha256: sha256(String(value)),
  }
}

export function sanitizeUrl(value) {
  if (!value) return null
  try {
    const url = new URL(String(value))
    return {
      protocol: url.protocol.replace(/:$/, ''),
      host: 'redacted',
      port_present: Boolean(url.port),
      path: url.pathname || '/',
    }
  } catch {
    return {
      raw: 'redacted',
      sha256: sha256(String(value)),
    }
  }
}

export function sanitizeSseEvent(event) {
  const parsed = event?.parsed
  const choice = parsed?.choices?.[0]
  const delta = choice?.delta
  const content = typeof delta?.content === 'string'
    ? delta.content
    : typeof choice?.text === 'string'
      ? choice.text
      : null
  const error = parsed?.error
  return {
    elapsed_ms: event?.elapsed_ms ?? null,
    event: event?.event ?? 'message',
    data_kind: event?.data === '[DONE]' ? 'done' : parsed ? 'json' : 'non_json',
    object: typeof parsed?.object === 'string' ? parsed.object : null,
    choice_index: Number.isInteger(choice?.index) ? choice.index : null,
    role: typeof delta?.role === 'string' ? delta.role : null,
    finish_reason: choice?.finish_reason ?? null,
    content: content === null ? null : sanitizeText(content),
    error: error ? sanitizeError(error) : null,
  }
}

function sanitizeError(error) {
  return {
    code: typeof error.code === 'string' ? error.code : null,
    type: typeof error.type === 'string' ? error.type : null,
    param: typeof error.param === 'string' ? error.param : null,
    message: typeof error.message === 'string' ? sanitizeText(error.message) : null,
    timeout_trace: sanitizeTimeoutTrace(error.timeout_trace),
  }
}

export function sanitizeTimeoutTrace(timeoutTrace) {
  if (!timeoutTrace || typeof timeoutTrace !== 'object') return null
  return {
    timeout_ms: integerOrNull(timeoutTrace.timeout_ms),
    elapsed_ms: integerOrNull(timeoutTrace.elapsed_ms),
    generated_tokens: integerOrNull(timeoutTrace.generated_tokens),
    timeout_env: typeof timeoutTrace.timeout_env === 'string' ? timeoutTrace.timeout_env : null,
  }
}

function integerOrNull(value) {
  return Number.isInteger(value) ? value : null
}
