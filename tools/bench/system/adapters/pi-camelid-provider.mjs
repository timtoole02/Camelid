const PROVIDER_ID = 'camelid-benchmark'
const CONTEXT_OVERFLOW = /(?:prompt[_ ]token[_ ]limit[_ ]exceeded|prompt encoded to [\d,]+ tokens, above the server ceiling of [\d,]+)/i

export function normalizeCamelidOverflowMessage(message, activeProvider = null) {
  if (message?.role !== 'assistant' || message.stopReason !== 'error') return null
  if (message.provider !== PROVIDER_ID && activeProvider !== PROVIDER_ID) return null
  const errorMessage = message.errorMessage ?? ''
  if (errorMessage.includes('context_length_exceeded') || !CONTEXT_OVERFLOW.test(errorMessage)) return null
  return {
    ...message,
    errorMessage: `context_length_exceeded: ${errorMessage}`,
  }
}

export default function camelidBenchmarkProviderExtension(pi) {
  pi.on('message_end', (event, context) => {
    const normalized = normalizeCamelidOverflowMessage(event.message, context.model?.provider)
    return normalized === null ? undefined : { message: normalized }
  })
}