import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { compatibilityHintCopy, compatibilityHintLabel, exactRowSupportLanes, findCompatibilityHint } from '../lib/capabilities'
import { clampText, formatDate, formatRate } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel } from '../lib/modelState'

export const GeminiSparkle = ({ className = '', size = 24 }) => (
  <svg
    className={`gemini-sparkle-icon ${className}`}
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M12 3C12 3 12.3 8.3 15.5 11.5C18.7 14.7 24 15 24 15C24 15 18.7 15.3 15.5 18.5C12.3 21.7 12 27 12 27C12 27 11.7 21.7 8.5 18.5C5.3 15.3 0 15 0 15C0 15 5.3 14.7 8.5 11.5C11.7 8.3 12 3 12 3Z"
      fill="url(#gemini-sparkle-grad)"
    />
    <defs>
      <linearGradient id="gemini-sparkle-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#4285f4" />
        <stop offset="35%" stopColor="#9b51e0" />
        <stop offset="70%" stopColor="#e289f2" />
        <stop offset="100%" stopColor="#fa9085" />
      </linearGradient>
    </defs>
  </svg>
)


const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const cleanLegacyDemoCapCopy = (value) => {
  if (typeof value !== 'string') return value
  return value
    .replace(/\s*\(demo cap\)/gi, '')
    .replace(/\s*·\s*raw\s+16-token-cap\s+local\s+run;\s*inspect\s+before\s+trusting\s+polish/gi, ' · raw local run')
    .replace(/\s*Longer-generation\s+polish\s+still\s+needs\s+separate\s+validation\.?/gi, '')
    .replace(/\s*Longer\s+generation\s+is\s+not\s+polished\s+yet\.?/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function getChatCapabilityLaneCopy(selectedChatGate, capabilities) {
  if (!selectedChatGate.contractSupported || !selectedChatGate.hint?.target) {
    return {
      label: 'Exact row unavailable',
      copy: 'Capability lanes stay hidden until the selected model has an exact supported /api/capabilities row and matching quant evidence.',
    }
  }

  const lanes = exactRowSupportLanes(selectedChatGate.hint.target, capabilities?.api_features || [])
  const template = lanes.find((lane) => lane.key === 'template')
  const context = lanes.find((lane) => lane.key === 'context')
  const throughput = lanes.find((lane) => lane.key === 'throughput')
  return {
    label: `${template?.ready ? 'Template ready' : 'Template gated'} · ${context?.ready ? 'Context ready' : 'Context gated'} · ${throughput?.ready ? 'Throughput ready' : 'Throughput not promoted'}`,
    copy: 'Row-scoped /api/capabilities evidence; it does not widen model-native context, production-throughput, portability, neighboring-row, or broad-family support.',
  }
}

function readinessTone({ ready = false, blocked = false, offline = false, waiting = false } = {}) {
  if (ready) return 'ready'
  if (offline || blocked) return 'blocked'
  if (waiting) return 'waiting'
  return 'idle'
}

const normalizeCodeLanguage = (value) => {
  const language = String(value || '').trim().replace(/[^a-zA-Z0-9_+#.-].*$/, '')
  if (!language) return 'Code'
  if (language.toLowerCase() === 'js') return 'JavaScript'
  if (language.toLowerCase() === 'ts') return 'TypeScript'
  if (language.toLowerCase() === 'html') return 'HTML'
  if (language.toLowerCase() === 'css') return 'CSS'
  return language.toUpperCase()
}

const copyText = async (text) => {
  try {
    await navigator.clipboard?.writeText(text)
  } catch {
    // Clipboard access can be denied outside secure browser contexts; rendering still works.
  }
}

const renderInlineMarkdown = (text, keyPrefix) => {
  const parts = String(text || '').split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean)
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={key} className="inline-code">{part.slice(1, -1)}</code>
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }
    return <span key={key}>{part}</span>
  })
}

const normalizeProseForReading = (text) => String(text || '')
  .replace(/\r\n/g, '\n')
  .replace(/\s+(Page\s+\d+\b)/gi, '\n\n$1')
  .replace(/\s+(References?\s*:)/gi, '\n\n$1')
  .replace(/\s+(Works\s+Cited\s*:)/gi, '\n\n$1')
  .replace(/\s+([•*-]\s+["“])/g, '\n$1')

const splitLongParagraph = (value) => {
  const text = String(value || '').trim()
  if (text.length <= 520) return text ? [text] : []
  const sentences = text.match(/[^.!?]+[.!?]+["”']?|[^.!?]+$/g) || [text]
  const paragraphs = []
  let current = ''

  sentences.forEach((sentence) => {
    const next = `${current}${current ? ' ' : ''}${sentence.trim()}`.trim()
    if (current && (next.length > 620 || current.split(/[.!?]+/).filter(Boolean).length >= 4)) {
      paragraphs.push(current)
      current = sentence.trim()
    } else {
      current = next
    }
  })
  if (current) paragraphs.push(current)
  return paragraphs
}

const pushParagraphBlocks = (blocks, value, keyPrefix) => {
  splitLongParagraph(value).forEach((paragraph) => {
    blocks.push(<p key={`${keyPrefix}-p-${blocks.length}`}>{renderInlineMarkdown(paragraph, `${keyPrefix}-p-${blocks.length}`)}</p>)
  })
}

const renderMarkdownText = (text, keyPrefix) => {
  const lines = normalizeProseForReading(text).split('\n')
  const blocks = []
  let paragraph = []
  let list = []

  const flushParagraph = () => {
    if (paragraph.length) {
      const value = paragraph.join(' ').trim()
      if (value) {
        pushParagraphBlocks(blocks, value, keyPrefix)
      }
      paragraph = []
    }
  }
  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={`${keyPrefix}-ul-${blocks.length}`}>
          {list.map((item, index) => (
            <li key={`${keyPrefix}-li-${blocks.length}-${index}`}>{renderInlineMarkdown(item, `${keyPrefix}-li-${index}`)}</li>
          ))}
        </ul>,
      )
      list = []
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      return
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      flushParagraph()
      flushList()
      const Tag = heading[1].length === 1 ? 'h2' : 'h3'
      blocks.push(<Tag key={`${keyPrefix}-h-${blocks.length}`}>{renderInlineMarkdown(heading[2], `${keyPrefix}-h-${blocks.length}`)}</Tag>)
      return
    }
    const pageHeading = line.match(/^(Page\s+\d+)\b[:\s.-]*(.*)$/i)
    if (pageHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-page-${blocks.length}`}>{pageHeading[1]}</h3>)
      if (pageHeading[2]) {
        pushParagraphBlocks(blocks, pageHeading[2], keyPrefix)
      }
      return
    }
    const referencesHeading = line.match(/^(References?|Works\s+Cited)\s*:?(.*)$/i)
    if (referencesHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-ref-${blocks.length}`}>{referencesHeading[1]}</h3>)
      if (referencesHeading[2]) {
        pushParagraphBlocks(blocks, referencesHeading[2].replace(/^\s*[:*-]\s*/, ''), keyPrefix)
      }
      return
    }
    const listItem = line.match(/^[-*]\s+(.+)$/)
    if (listItem) {
      flushParagraph()
      list.push(listItem[1])
      return
    }
    flushList()
    paragraph.push(line)
  })
  flushParagraph()
  flushList()
  return blocks
}

const syntaxClassForToken = (token, language) => {
  const lowerLanguage = String(language || '').toLowerCase()
  if (/^\s+$/.test(token)) return ''
  if (/^\/\//.test(token) || /^\/\*/.test(token) || /^<!--/.test(token)) return 'comment'
  if (/^['"`]/.test(token)) return 'string'
  if (/^\d/.test(token)) return 'number'
  if (lowerLanguage.includes('html') && /^<\/?[\w-]+/.test(token)) return 'tag'
  if (lowerLanguage.includes('html') && /^[\w:-]+(?==)/.test(token)) return 'attr'
  if (/^(const|let|var|function|return|if|else|for|while|class|new|true|false|null|undefined|import|export|from|async|await|document|window)$/.test(token)) return 'keyword'
  if (lowerLanguage.includes('css') && /^[\w-]+(?=\s*:)/.test(token)) return 'attr'
  return ''
}

const renderHighlightedCode = (code, language, keyPrefix) => {
  const lowerLanguage = String(language || '').toLowerCase()
  const pattern = lowerLanguage.includes('html')
    ? /(<!--[\s\S]*?-->|<\/?[\w-]+|\/?>|[\w:-]+(?==)|"(?:\\.|[^"])*"|'(?:\\.|[^'])*')/g
    : lowerLanguage.includes('css')
      ? /(\/\*[\s\S]*?\*\/|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|#[\da-fA-F]{3,8}|\b\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw)?\b|[\w-]+(?=\s*:))/g
      : /(\/\/.*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|`(?:\\.|[^`])*`|\b(?:const|let|var|function|return|if|else|for|while|class|new|true|false|null|undefined|import|export|from|async|await|document|window)\b|\b\d+(?:\.\d+)?\b)/g
  const nodes = []
  let cursor = 0
  let match = pattern.exec(code)
  while (match) {
    if (match.index > cursor) nodes.push(code.slice(cursor, match.index))
    const token = match[0]
    const tokenClass = syntaxClassForToken(token, lowerLanguage)
    nodes.push(tokenClass
      ? <span key={`${keyPrefix}-${nodes.length}`} className={`syntax-token ${tokenClass}`}>{token}</span>
      : token)
    cursor = match.index + token.length
    match = pattern.exec(code)
  }
  if (cursor < code.length) nodes.push(code.slice(cursor))
  return nodes
}

const splitFenceInfo = (value) => {
  const trimmed = String(value || '').trim()
  if (!trimmed) return { language: 'Code', firstCodeLine: '' }
  const [, rawLanguage = '', firstCodeLine = ''] = trimmed.match(/^([a-zA-Z0-9_+#.-]+)?\s*([\s\S]*)$/) || []
  return {
    language: normalizeCodeLanguage(rawLanguage),
    firstCodeLine: firstCodeLine.trimStart(),
  }
}

const CODE_CARD_STREAMING_LABEL = 'Still generating — code block incomplete'

function CodeBlockCard({ language, code, keyPrefix, stillGenerating }) {
  const preRef = useRef(null)
  const autoFollowCodeRef = useRef(true)

  useEffect(() => {
    if (!stillGenerating) return undefined
    autoFollowCodeRef.current = true
    const pre = preRef.current
    if (!pre) return undefined
    const updateAutoFollow = () => {
      const distanceFromBottom = pre.scrollHeight - (pre.scrollTop + pre.clientHeight)
      autoFollowCodeRef.current = distanceFromBottom < 80
    }
    pre.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => pre.removeEventListener('scroll', updateAutoFollow)
  }, [stillGenerating])

  useLayoutEffect(() => {
    if (!stillGenerating || !autoFollowCodeRef.current) return
    const pre = preRef.current
    if (pre) pre.scrollTop = pre.scrollHeight
  }, [code, stillGenerating])

  return (
    <figure
      className={`message-code-card ${stillGenerating ? 'is-generating' : ''}`}
      aria-busy={stillGenerating ? 'true' : undefined}
      data-code-streaming-state={stillGenerating ? 'open' : undefined}
    >
      <figcaption>
        <span className="message-code-card-title">{language}</span>
        {stillGenerating && <span className="message-code-card-status" aria-live="polite" data-live-status="active">{CODE_CARD_STREAMING_LABEL}</span>}
        <button type="button" onClick={() => copyText(code)} aria-label={`Copy ${language} code`}>Copy</button>
      </figcaption>
      <pre ref={preRef}><code>{renderHighlightedCode(code, language, keyPrefix)}</code></pre>
    </figure>
  )
}

const pushCodeBlock = (blocks, language, code, keyPrefix, { incomplete = false, streaming = false } = {}) => {
  const trimmedCode = String(code || '').replace(/^\n+|\n+$/g, '')
  const stillGenerating = Boolean(incomplete && streaming)
  blocks.push(
    <CodeBlockCard
      key={`code-${blocks.length}`}
      language={language}
      code={trimmedCode}
      keyPrefix={keyPrefix}
      stillGenerating={stillGenerating}
    />,
  )
}

const hasOpenCodeFence = (content) => {
  const matches = String(content || '').match(/```/g)
  return Boolean(matches && matches.length % 2 === 1)
}

const PREPARING_STREAMING_LABEL = 'Preparing local response'
const FIRST_TOKEN_STREAMING_LABEL = 'Backend is generating'
const LONG_FIRST_TOKEN_STREAMING_LABEL = 'Local response is taking a while'
const ACTIVE_STREAMING_LABEL = 'Streaming response'
const OPEN_CODE_STREAMING_LABEL = 'Streaming code response'

const DEMO_PROMPTS = [
  'Summarize this implementation plan and call out the risks',
  'Draft a concise release note from these changes',
  'Turn this checklist into a prioritized next-step plan',
  'Review this response and tighten it into a shorter final answer',
]

const FOLLOW_UP_PROMPTS = [
  'Continue with the exact next steps.',
  'Tighten that into a shorter final answer.',
  'Turn this into a checklist I can execute.',
]

function formatCountLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`
}

const streamingStatusLabel = (phase, elapsedSeconds, isOpenCode = false) => {
  if (phase === 'preparing') return PREPARING_STREAMING_LABEL
  if (phase === 'streaming') return isOpenCode ? OPEN_CODE_STREAMING_LABEL : ACTIVE_STREAMING_LABEL
  if (elapsedSeconds >= 20) return LONG_FIRST_TOKEN_STREAMING_LABEL
  return FIRST_TOKEN_STREAMING_LABEL
}

function StreamingLoader({ elapsedSeconds, label = ACTIVE_STREAMING_LABEL, compact = false }) {
  return (
    <div className={`streaming-loader ${compact ? 'streaming-loader-compact' : ''}`} role="status" aria-live="polite" aria-label={`${label}. ${elapsedSeconds} seconds elapsed.`}>
      <div className="streaming-loader-track" aria-hidden="true">
        <span className="streaming-loader-dot streaming-loader-dot-1" />
        <span className="streaming-loader-dot streaming-loader-dot-2" />
        <span className="streaming-loader-dot streaming-loader-dot-3" />
      </div>
    </div>
  )
}

function LiveGenerationBadge({ elapsedSeconds, label = ACTIVE_STREAMING_LABEL }) {
  return (
    <div className="message-live-generation-badge" role="status" aria-live="polite" data-live-status="active">
      <span className="message-live-dot" aria-hidden="true" />
      <span>{label}</span>
      <span>{elapsedSeconds}s</span>
    </div>
  )
}

function ChatSurfaceNotice({ state, title, copy, actionLabel, onAction }) {
  if (!title && !copy) return null
  return (
    <div className={`chat-surface-notice is-${state}`} role="status" aria-live="polite">
      <span className="chat-surface-notice-dot" aria-hidden="true" />
      <div>
        {title && <strong>{title}</strong>}
        {copy && <p>{copy}</p>}
      </div>
      {actionLabel && onAction && (
        <button type="button" className="ghost-button ghost-button-quiet" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  )
}

function AssistantMarkdownInner({ content, streaming = false }) {
  const normalized = String(content || '').replace(/\r\n/g, '\n')
  const blocks = []
  let cursor = 0
  let fenceStart = normalized.indexOf('```', cursor)

  while (fenceStart !== -1) {
    const before = normalized.slice(cursor, fenceStart)
    blocks.push(...renderMarkdownText(before, `md-${blocks.length}`))

    const infoStart = fenceStart + 3
    const nextLine = normalized.indexOf('\n', infoStart)
    const infoEnd = nextLine === -1 ? normalized.length : nextLine
    const { language, firstCodeLine } = splitFenceInfo(normalized.slice(infoStart, infoEnd))
    const codeStart = nextLine === -1 ? infoEnd : nextLine + 1
    const fenceEnd = normalized.indexOf('```', codeStart)
    const incompleteFence = fenceEnd === -1
    const codeEnd = fenceEnd === -1 ? normalized.length : fenceEnd
    const codeBody = normalized.slice(codeStart, codeEnd)
    const code = firstCodeLine ? `${firstCodeLine}${codeBody ? `\n${codeBody}` : ''}` : codeBody

    pushCodeBlock(blocks, language, code, `code-${blocks.length}`, { incomplete: incompleteFence, streaming })
    cursor = fenceEnd === -1 ? normalized.length : fenceEnd + 3
    fenceStart = normalized.indexOf('```', cursor)
  }
  blocks.push(...renderMarkdownText(normalized.slice(cursor), `md-${blocks.length}`))

  return <div className="message-markdown">{blocks.length ? blocks : <p>{content}</p>}</div>
}

const AssistantMarkdown = memo(AssistantMarkdownInner)

function DeveloperDiagnosticsBlock({ message }) {
  const [isOpen, setIsOpen] = useState(false)

  if (!message.camelid && !message.tokens_out_per_sec && !message.first_content_ms) return null

  const metrics = message.camelid?.timings_ms || {}
  const layers = metrics.layers || []
  const maxLayerTime = layers.reduce((max, layer) => Math.max(max, layer.total || 0), 0.0001)

  const formatMs = (val) => {
    const num = Number(val)
    if (!Number.isFinite(num) || num <= 0) return '0 ms'
    if (num < 1000) return `${num.toFixed(0)} μs`
    return `${(num / 1000).toFixed(1)} ms`
  }

  const ttfb = message.first_byte_ms !== null && message.first_byte_ms !== undefined
    ? `${(Number(message.first_byte_ms) / 1000).toFixed(2)}s`
    : null
  const ttft = message.first_content_ms !== null && message.first_content_ms !== undefined
    ? `${(Number(message.first_content_ms) / 1000).toFixed(2)}s`
    : null
  const decodeRate = formatRate(message.tokens_out_per_sec)

  const tokenizeTime = metrics.tokenize ? formatMs(metrics.tokenize) : null
  const weightLoadTime = metrics.weight_load ? formatMs(metrics.weight_load) : null
  const totalGenTime = metrics.generate ? formatMs(metrics.generate) : null

  return (
    <div className="developer-diagnostics-container">
      <button
        type="button"
        className={`developer-diagnostics-trigger ${isOpen ? 'is-open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="trigger-icon">📊</span>
        <span>Developer Diagnostics</span>
        {decodeRate && <span className="trigger-badge">{decodeRate}</span>}
      </button>

      {isOpen && (
        <div className="developer-diagnostics-panel animate-slide-down">
          <div className="diagnostics-grid-summary">
            {ttft && (
              <div className="summary-card">
                <span className="card-label">Time to First Token (TTFT)</span>
                <strong className="card-value">{ttft}</strong>
              </div>
            )}
            {decodeRate && (
              <div className="summary-card">
                <span className="card-label">Decode Speed</span>
                <strong className="card-value">{decodeRate}</strong>
              </div>
            )}
            {totalGenTime && (
              <div className="summary-card">
                <span className="card-label">Generation Time</span>
                <strong className="card-value">{totalGenTime}</strong>
              </div>
            )}
            {weightLoadTime && (
              <div className="summary-card">
                <span className="card-label">Weight Load (VM Map)</span>
                <strong className="card-value">{weightLoadTime}</strong>
              </div>
            )}
          </div>

          {layers.length > 0 && (
            <div className="layer-breakdown-section">
              <h4>Layer Latency Breakdown</h4>
              <p className="section-meta">Active transformer computation spent in Attention vs. Feed-Forward networks across {layers.length} layers.</p>

              <div className="layer-bars-container">
                {layers.map((layer) => {
                  const attnTime = (layer.attention_q || 0) + (layer.attention_k || 0) + (layer.attention_v || 0) + (layer.attention_context || 0) + (layer.attention_output || 0)
                  const ffnTime = (layer.ffn_gate || 0) + (layer.ffn_up || 0) + (layer.ffn_down || 0)
                  const otherTime = Math.max(0, (layer.total || 0) - (attnTime + ffnTime))

                  const attnPercent = layer.total > 0 ? (attnTime / layer.total) * 100 : 0
                  const ffnPercent = layer.total > 0 ? (ffnTime / layer.total) * 100 : 0
                  const otherPercent = layer.total > 0 ? (otherTime / layer.total) * 100 : 0

                  const totalPercent = Math.max(2, (layer.total / maxLayerTime) * 100)

                  return (
                    <div key={layer.layer_index} className="layer-bar-row">
                      <div className="layer-label">
                        <span>L{layer.layer_index}</span>
                        <small>{formatMs(layer.total)}</small>
                      </div>
                      <div className="layer-bar-track">
                        <div
                          className="layer-bar-fill"
                          style={{ width: `${totalPercent}%` }}
                        >
                          {attnPercent > 0 && (
                            <div
                              className="segment-attn"
                              style={{ width: `${attnPercent}%` }}
                              title={`Attention: ${formatMs(attnTime)}`}
                            />
                          )}
                          {ffnPercent > 0 && (
                            <div
                              className="segment-ffn"
                              style={{ width: `${ffnPercent}%` }}
                              title={`Feed-Forward: ${formatMs(ffnTime)}`}
                            />
                          )}
                          {otherPercent > 0 && (
                            <div
                              className="segment-other"
                              style={{ width: `${otherPercent}%` }}
                              title={`Residual / Overhead: ${formatMs(otherTime)}`}
                            />
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              <div className="layer-legend">
                <span className="legend-item"><span className="legend-dot dot-attn" /> Attention</span>
                <span className="legend-item"><span className="legend-dot dot-ffn" /> Feed-Forward</span>
                <span className="legend-item"><span className="legend-dot dot-other" /> Residual / Norm</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const isInterruptedPlaceholderMessage = (message) => {
  if (message?.role !== 'assistant') return false
  const content = String(message?.content || '').trim().toLowerCase()
  return content === '(generation interrupted)' || content === '(generation stopped)'
}

const ChatMessageRow = memo(function ChatMessageRow({ message, generationElapsedSeconds, priorUserPrompt, onReusePrompt }) {
  const [copied, setCopied] = useState(false)
  const copiedResetRef = useRef(null)
  const messageContent = cleanLegacyDemoCapCopy(message.content)
  const assistantStreaming = message.role === 'assistant' && Boolean(message.streaming)
  const isOpenStreamingCode = assistantStreaming && hasOpenCodeFence(messageContent)
  const streamingPhase = message.streaming_phase || (messageContent ? 'streaming' : 'generating')
  const liveStatusLabel = streamingStatusLabel(streamingPhase, generationElapsedSeconds, isOpenStreamingCode)
  const hasTokenMetrics = false
  const showStreamingStatus = assistantStreaming && !messageContent
  const showLiveGenerationBadge = assistantStreaming && Boolean(messageContent)
  const showLengthWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'length'
  const showErrorWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'error'
  const showInterruptedWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'interrupted'
  const showReusePromptAction = Boolean(priorUserPrompt) && (showErrorWarning || showInterruptedWarning)
  const showMessageActions = message.role === 'assistant' && Boolean(String(messageContent || '').trim())

  useEffect(() => () => {
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
  }, [])

  const handleCopyMessage = async () => {
    await copyText(messageContent)
    setCopied(true)
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
    copiedResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <article
      className={`message-row message-row-assistant ${message.role} ${assistantStreaming ? 'is-streaming' : ''}`}
      aria-busy={assistantStreaming ? 'true' : undefined}
      data-streaming-state={assistantStreaming ? 'active' : undefined}
      data-streaming-code-state={isOpenStreamingCode ? 'open' : undefined}
    >
      <div className="message-row-inner">
        {message.role === 'assistant' && (
          <div className="gemini-assistant-avatar-wrapper" aria-hidden="true">
            <GeminiSparkle size={22} className="gemini-assistant-avatar-sparkle" />
          </div>
        )}
        <div className={`message-bubble message-bubble-assistant ${message.role}`}>
          {showStreamingStatus && <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} compact />}
          {message.role === 'assistant'
            ? messageContent || !assistantStreaming
              ? <AssistantMarkdown content={messageContent} streaming={assistantStreaming} />
              : null
            : <p>{messageContent}</p>}
          {showLiveGenerationBadge && <LiveGenerationBadge elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} />}
          {showMessageActions && (
            <div className="message-actions" aria-label="Message actions">
              <button type="button" className="message-action-button" onClick={handleCopyMessage}>
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          )}
        {showLengthWarning && (
          <div className="message-finish-warning" role="status">
            Stopped before completing. Ask “continue” for a complete file.
          </div>
        )}
        {showErrorWarning && (
          <div className="message-finish-warning message-finish-warning-error" role="status">
            Generation stopped before Camelid returned a complete reply.
          </div>
        )}
        {showInterruptedWarning && (
          <div className="message-finish-warning message-finish-warning-interrupted" role="status">
            Generation was interrupted before the reply finished.
          </div>
        )}
        {showReusePromptAction && (
          <div className="message-recovery-actions" aria-label="Recovery actions">
            <button type="button" className="message-action-button" onClick={() => onReusePrompt?.(priorUserPrompt)}>
              Use prompt again
            </button>
          </div>
        )}
        <DeveloperDiagnosticsBlock message={message} />
      </div>
    </div>
  </article>
  )
})

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  models,
  runtime,
  capabilities,
  pendingConversation,
  composer,
  setComposer,
  saveToMemory,
  sendMessage,
  stopGeneration,
  sending,
  stoppingGeneration = false,
  selectedModelRunnable,
  setTab,
  showNewChatLanding = null,
  demoMode = false,
}) {
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const chatBottomRef = useRef(null)
  const composerRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
  const composerReadinessId = 'camelid-chat-readiness-note'
  const rawVisibleMessages = useMemo(
    () => (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message)),
    [selectedConversation?.messages],
  )
  const hasStreamingAssistant = rawVisibleMessages.some((message) => message.role === 'assistant' && message.streaming)
  const hasStreamingAssistantContent = rawVisibleMessages.some((message) => message.role === 'assistant' && message.streaming && String(message.content || '').trim())
  const generationActive = Boolean(sending || hasStreamingAssistant)
  const visibleMessages = useMemo(() => {
    if (!generationActive) return rawVisibleMessages
    return rawVisibleMessages.filter((message, index, messages) => {
      const isTrailingInterruptedPlaceholder = index === messages.length - 1 && isInterruptedPlaceholderMessage(message)
      return !isTrailingInterruptedPlaceholder
    })
  }, [generationActive, rawVisibleMessages])
  const pendingPrompt = (pendingConversation?.content || (sending ? composer.trim() : '')).trim()
  const pendingPromptAlreadyVisible = Boolean(
    pendingPrompt && [...visibleMessages].reverse().some((message) => message.role === 'user' && message.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const lastVisibleMessage = visibleMessages.at(-1)
  const lastVisibleMessageIsUser = lastVisibleMessage?.role === 'user'
  const awaitingAssistant = Boolean(generationActive && !hasStreamingAssistantContent && !hasStreamingAssistant && (pendingPrompt || lastVisibleMessageIsUser || sending))
  const streamingScrollSignature = useMemo(() => (
    visibleMessages.map((message) => `${message.id}:${message.streaming ? 'streaming' : 'done'}:${String(message.content || '').length}`).join('|')
    + `|awaiting:${awaitingAssistant ? '1' : '0'}|active:${generationActive ? '1' : '0'}`
  ), [awaitingAssistant, generationActive, visibleMessages])
  const isFreshThread = selectedConversation
    ? (visibleMessages.length === 0 && !pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
    : (!pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)

  useEffect(() => {
    if (!generationActive) {
      setGenerationElapsedSeconds(0)
      return undefined
    }
    setGenerationElapsedSeconds(0)
    const startedAt = Date.now()
    const interval = window.setInterval(() => {
      setGenerationElapsedSeconds(Math.max(1, Math.floor((Date.now() - startedAt) / 1000)))
    }, 1000)
    return () => window.clearInterval(interval)
  }, [generationActive])

  useEffect(() => {
    if (!generationActive) return undefined
    autoFollowGenerationRef.current = true
    const updateAutoFollow = () => {
      const doc = document.documentElement
      const distanceFromBottom = doc.scrollHeight - (window.scrollY + window.innerHeight)
      autoFollowGenerationRef.current = distanceFromBottom < 260
    }
    window.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => window.removeEventListener('scroll', updateAutoFollow)
  }, [generationActive, selectedConversation?.id])

  useLayoutEffect(() => {
    if (!generationActive || !autoFollowGenerationRef.current) return undefined
    const frame = window.requestAnimationFrame(() => {
      chatBottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [generationActive, streamingScrollSignature])

  useLayoutEffect(() => {
    const input = composerRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`
  }, [composer, isFreshThread, selectedConversation?.id])

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Escape' && generationActive) {
      event.preventDefault()
      stopGeneration?.()
      return
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) {
        await sendMessage()
      }
    }
  }

  const rawConversationTitle = selectedConversation?.title?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const conversationLabel = clampText(hasCustomConversationTitle ? rawConversationTitle : 'Untitled chat', 30)
  const lastUpdated = selectedConversation?.updated_at ? formatDate(selectedConversation.updated_at) : null
  const modelPickerTitle = selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what Camelid should use for this chat.'
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const apiUnavailable = runtime?.status === 'offline'
  const selectedRuntimeReady = selectedChatGate.runtimeReady
  const selectedModelCapabilitySupported = selectedChatGate.contractSupported
  const supportBlocked = selectedRuntimeReady && !selectedModelCapabilitySupported
  const selectedRuntimeMatchesLoadedModel = Boolean(selectedChatGate.runtimeLoaded)
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityLabel = selectedModel
    ? compatibilityHintLabel(selectedCompatibilityHint, 'No matching COMPATIBILITY.md row')
    : 'No model selected'
  const selectedCompatibilityCopy = selectedModel
    ? compatibilityHintCopy(selectedCompatibilityHint)
    : 'Choose a model before inferring any support boundary. Camelid will not promote filenames or saved paths into compatibility claims.'
  const selectedModelMeta = selectedModelRunnable
    ? 'Ready to send'
    : apiUnavailable
      ? 'Draft offline'
      : supportBlocked
        ? 'Support gated'
        : selectedModel
          ? 'Draft unlocked'
          : 'Choose a model'
  const canSubmit = Boolean(composer.trim()) && selectedModelRunnable && !generationActive
  const capabilityLaneStatus = getChatCapabilityLaneCopy(selectedChatGate, capabilities)
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const messageCount = visibleMessages.length
  const userMessageCount = visibleMessages.filter((message) => message.role === 'user').length + (pendingUserPrompt ? 1 : 0)
  const assistantMessageCount = visibleMessages.filter((message) => message.role === 'assistant').length
  const runtimeStatusLabel = apiUnavailable
    ? 'API unavailable'
    : selectedModelRunnable
    ? 'Local chat ready'
    : selectedRuntimeReady
      ? 'Runtime ready, support gated'
      : runtime?.loaded_now
        ? 'Loaded, not generation-ready'
        : 'No generation-ready model'
  const runtimeStatusCopy = apiUnavailable
    ? 'Camelid did not respond. Start the server or check the API base before loading a model.'
    : selectedModelRunnable
    ? `${selectedModelName} is loaded now and generation_ready=true.`
    : selectedRuntimeReady
      ? 'The runtime is ready; Camelid still needs an exact supported row before chat unlocks.'
      : runtime?.loaded_now
        ? 'Wait for generation_ready=true before sending prompts.'
        : 'Load a local GGUF from Library to start the readiness check.'
  const supportStatusLabel = selectedModelCapabilitySupported
    ? selectedCompatibilityLabel
    : apiUnavailable
      ? 'Contract unavailable'
    : selectedModel
      ? selectedCompatibilityLabel
      : 'Choose model first'
  const supportStatusCopy = selectedModelCapabilitySupported
    ? `${selectedCompatibilityLabel}. COMPATIBILITY.md and /api/capabilities agree for this model and quant.`
    : apiUnavailable
      ? 'The /api/capabilities contract could not be read while the API is unavailable.'
    : selectedModel
      ? selectedCompatibilityCopy
      : 'Camelid does not infer broad support from filenames, families, or saved paths.'
  const readinessFinePrint = selectedModelRunnable
    ? 'Ready for this loaded exact row.'
    : apiUnavailable
      ? 'Drafts stay editable while the Camelid API reconnects.'
    : selectedModel
      ? 'Chat unlocks only after loaded_now=true, generation_ready=true, and an exact supported compatibility row all match.'
      : 'Choose a model, then Camelid will show what still needs to pass before send unlocks.'
  const selectedModelIssue = selectedModel?.load_error || selectedModel?.install_error || ''
  const readinessActionTab = apiUnavailable ? 'api' : 'library'
  const readinessActionLabel = apiUnavailable ? 'Open API' : 'Open Models'
  const selectedModelReadinessCopy = selectedModelRunnable
    ? 'Selected model is ready for Camelid chat.'
    : apiUnavailable
      ? 'The API is offline, so readiness cannot be checked yet.'
    : selectedModelIssue
      ? selectedModelIssue
    : supportBlocked
      ? 'This row is loaded, but chat stays locked until the exact support contract matches.'
    : selectedRuntimeMatchesLoadedModel
      ? 'This model is loaded and still warming up. Send unlocks once generation readiness turns on.'
      : selectedModel
        ? 'Keep drafting here while Camelid prepares this model.'
        : 'Choose a model before starting a Camelid chat.'
  const selectedModelGateSummary = selectedModel
    ? selectedModelRunnable
      ? 'Selected model is ready for Camelid chat.'
      : selectedModelIssue
        ? selectedModelIssue
        : selectedModelReadinessCopy
    : 'Choose a model before starting a Camelid chat.'
  const emptyHeroEyebrow = 'Camelid'
  const promptHintCopy = selectedModelRunnable
    ? 'Enter sends · Shift+Enter adds a line break'
    : apiUnavailable
      ? 'Draft now · send unlocks after the API reconnects'
      : supportBlocked
        ? 'Send unlocks after exact-row readiness passes'
      : selectedModel
        ? 'Draft now · send unlocks after readiness passes'
        : 'Choose a model to unlock sending'
  const readinessState = selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const readinessLabel = selectedModelRunnable
    ? 'Ready'
    : apiUnavailable
      ? 'API unavailable'
    : supportBlocked
      ? 'Choose a supported model.'
      : selectedModel
        ? 'Waiting on readiness'
        : 'Choose a model to begin'
  const productHeroTitle = "Hi Tim, let's get into it"
  const productHeroSummary = selectedModelRunnable
    ? 'A clean local assistant surface with the current runtime state kept visible.'
    : apiUnavailable
      ? 'Keep writing here. Send unlocks again once the local API responds.'
      : supportBlocked
        ? 'The runtime is up, but chat still needs an exact supported row before send unlocks.'
        : selectedModel
          ? 'Your draft is ready now. Send unlocks as soon as this model is ready.'
          : 'Pick a local GGUF model first. Camelid will show the readiness path here.'
  const surfaceNoticeTitle = selectedModelRunnable
    ? ''
    : apiUnavailable
      ? 'Camelid API is unavailable'
      : supportBlocked
        ? 'Exact support row required'
        : selectedModel
          ? 'Runtime readiness pending'
          : 'Choose a model'
  const surfaceNoticeCopy = selectedModelRunnable
    ? ''
    : apiUnavailable
      ? 'The chat UI is ready, and drafting stays available, but the local API must respond before prompts can be sent.'
      : supportBlocked
        ? 'This runtime is up. Keep drafting, but Camelid still requires an exact supported row for the selected model and quant.'
        : selectedModel
          ? selectedModelGateSummary
          : 'Add or select a local GGUF model from Models, then load it into the Camelid runtime.'
  const runtimeTone = readinessTone({
    ready: selectedModelRunnable,
    offline: apiUnavailable,
    waiting: Boolean(runtime?.loaded_now || selectedModel),
  })
  const supportTone = readinessTone({
    ready: selectedModelCapabilitySupported && !apiUnavailable,
    offline: apiUnavailable,
    blocked: supportBlocked,
    waiting: Boolean(selectedModel),
  })
  const capabilityTone = readinessTone({
    ready: selectedChatGate.contractSupported && !apiUnavailable,
    offline: apiUnavailable,
    blocked: supportBlocked,
    waiting: Boolean(selectedModel),
  })
  const currentConversationSummary = hasCustomConversationTitle
    ? `${conversationLabel}${lastUpdated ? ` · ${lastUpdated}` : ''}`
    : lastUpdated || 'Fresh chat'
  const availableModelCount = models.length
  const readyModelCount = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked).length
  const modelInventoryLabel = availableModelCount
    ? `${readyModelCount}/${availableModelCount} ready`
    : 'No models added'
  const composerDraftUnlocked = Boolean(selectedModel || apiUnavailable)
  const composerPlaceholder = selectedModelRunnable
    ? 'Message Camelid…'
    : apiUnavailable
      ? 'Draft a prompt while the Camelid API comes back'
      : composerDraftUnlocked
      ? 'Draft a prompt while Camelid finishes getting ready'
      : isFreshThread
          ? 'Load a model first'
          : 'Choose a ready model first'
  const composerSendLabel = generationActive ? `Generating ${generationElapsedSeconds}s…` : 'Send'
  const composerStopLabel = stoppingGeneration ? 'Stopping…' : 'Stop'
  const secondaryActionLabel = selectedModelRunnable ? 'Save to memory' : readinessActionLabel
  const secondaryAction = selectedModelRunnable ? saveToMemory : () => setTab(readinessActionTab)
  const secondaryActionDisabled = selectedModelRunnable ? generationActive : false
  const composerDisabled = !composerDraftUnlocked
  const selectionSummaryTone = selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : selectedModelIssue ? 'blocked' : supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const selectionSummaryLabel = selectedModelRunnable
    ? 'Ready now'
    : apiUnavailable
      ? 'API unavailable'
    : selectedModelIssue
      ? 'Needs attention'
    : supportBlocked
      ? 'Support gated'
    : selectedModel
      ? 'Waiting on readiness'
      : 'Choose a model'
  const selectionSummaryCopy = selectedModelRunnable
    ? `${selectedModelName} is loaded now with generation_ready=true and the current exact-row contract unlocked.`
    : apiUnavailable
      ? 'The frontend is available, but the Camelid API must respond before model readiness can be checked.'
    : selectedModelIssue
      ? selectedModelIssue
    : supportBlocked
      ? 'This selected row is loaded, but send stays locked until the exact supported row matches.'
    : selectedModel
      ? 'Drafting stays unlocked. Camelid will unlock send as soon as this selected row is loaded, generation-ready, and supported.'
      : 'Pick a local model first, then Camelid will keep the runtime and support boundary visible here.'
  const sendDisabledReason = selectedModelRunnable
    ? ''
    : generationActive
      ? 'Wait for the current reply to finish or stop it before sending again.'
    : apiUnavailable
      ? 'Send unlocks after the Camelid API reconnects.'
    : selectedModel
      ? 'Send unlocks when Camelid marks this model ready and supported.'
      : 'Choose a model before sending.'
  const draftStatusLabel = generationActive
    ? 'Drafting stays available while Camelid replies.'
    : selectedModelRunnable
      ? 'Draft and send are both available.'
      : apiUnavailable
        ? 'Drafts stay local until the API reconnects.'
        : selectedModel
          ? 'Drafting is unlocked. Send unlocks after readiness passes.'
          : 'Choose a model to unlock drafting and send.'
  const composerHintCopy = canSubmit ? promptHintCopy : sendDisabledReason || promptHintCopy

  const handleDemoPrompt = (prompt) => {
    if (!composerDraftUnlocked) return
    setComposer(prompt)
  }

  useEffect(() => {
    if (generationActive || !composerDraftUnlocked) return
    const input = composerRef.current
    if (!input) return
    const activeElement = document.activeElement
    if (activeElement && activeElement !== document.body && activeElement !== input) return
    const frame = window.requestAnimationFrame(() => input.focus())
    return () => window.cancelAnimationFrame(frame)
  }, [composerDraftUnlocked, generationActive, isFreshThread, selectedConversation?.id])

  const composerStatusItems = [
    { label: 'Model', value: selectedModelName },
    { label: 'Chat', value: selectedModelRunnable ? 'Ready' : readinessLabel },
    { label: 'Draft', value: draftStatusLabel },
  ]

  const conversationSnapshotItems = [
    { label: 'Messages', value: formatCountLabel(messageCount, 'message') },
    { label: 'Prompts', value: formatCountLabel(userMessageCount, 'prompt') },
    { label: 'Replies', value: formatCountLabel(assistantMessageCount, 'reply') },
  ]
  const readinessCardItems = [
    { label: 'Runtime', value: runtimeStatusLabel, copy: runtimeStatusCopy, tone: runtimeTone },
    { label: 'Support', value: supportStatusLabel, copy: supportStatusCopy, tone: supportTone },
    { label: 'Selected model', value: selectedModelName, copy: selectionSummaryCopy, tone: selectionSummaryTone },
  ]

  const renderReadinessPills = (extraClass = '', ariaLabel = 'Chat readiness and support boundary') => (
    <div className={`chat-readiness-pill-row chat-readiness-strip-live ${extraClass} is-${readinessState}`} aria-label={ariaLabel} aria-live="polite">
      <div className={`chat-readiness-pill is-${runtimeTone}`} title={runtimeStatusCopy}>
        <span>Runtime</span>
        <strong>{runtimeStatusLabel}</strong>
      </div>
      <div className={`chat-readiness-pill is-${supportTone}`} title={supportStatusCopy}>
        <span>Support</span>
        <strong>{supportStatusLabel}</strong>
      </div>
      <div className={`chat-readiness-pill chat-readiness-pill-wide is-${capabilityTone}`} title={capabilityLaneStatus.copy}>
        <span>Capabilities</span>
        <strong>{capabilityLaneStatus.label}</strong>
      </div>
    </div>
  )

  const renderComposerModelSummary = (extraClass = '') => (
    <div className={`composer-model-summary is-${selectionSummaryTone} ${extraClass}`.trim()} aria-live="polite">
      <span>{selectionSummaryLabel}</span>
      <strong>{selectedModelName}</strong>
      <p>{selectionSummaryCopy}</p>
    </div>
  )

  const renderModelPicker = () => {
    if (!models.length) {
      return (
        <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>
          Add model
        </button>
      )
    }

    const modelOptionLabel = (model) => {
      const gate = getChatGateState(capabilities, model, runtime)
      if (gate.chatUnlocked) return `${model.name} · Ready`
      if (apiUnavailable) return `${model.name} · API unavailable`
      if (gate.runtimeReady) return `${model.name} · Support gated`
      if (gate.runtimeLoaded) return `${model.name} · Loading`
      return `${model.name} · Not loaded`
    }

    const runnableModels = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked)
    const waitingModels = models.filter((model) => !getChatGateState(capabilities, model, runtime).chatUnlocked)
    const selectedPickerModelId = models.some((model) => model.id === selectedModel?.id) ? selectedModel.id : ''

    return (
      <label className={`composer-model-picker is-${readinessState}`} title={modelPickerTitle}>
        <span className="composer-tool-label">Model</span>
        <span className="composer-model-caption">{modelInventoryLabel}</span>
        <select
          className="composer-model-select"
          aria-label="Choose model for chat"
          value={selectedPickerModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={generationActive}
        >
          {!selectedModel && <option value="">Choose model</option>}
          {runnableModels.length > 0 && (
            <optgroup label="Ready">
              {runnableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {modelOptionLabel(model)}
                </option>
              ))}
            </optgroup>
          )}
          {waitingModels.length > 0 && (
            <optgroup label="Needs readiness">
              {waitingModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {modelOptionLabel(model)}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </label>
    )
  }

  const primaryEmptyActionLabel = apiUnavailable
    ? 'Open API'
    : selectedModel
      ? 'Open Models'
      : 'Choose model'
  const primaryEmptyAction = () => setTab(apiUnavailable ? 'api' : 'library')
  const showPromptStarters = selectedModelRunnable || composerDraftUnlocked
  const heroFactItems = [
    {
      label: 'Selected model',
      value: selectedModelName,
      copy: selectionSummaryCopy,
      tone: selectionSummaryTone,
      wide: true,
    },
    {
      label: 'Current gate',
      value: selectedModelRunnable ? 'Ready to chat' : readinessLabel,
      copy: runtimeStatusCopy,
      tone: runtimeTone,
    },
    {
      label: 'Support boundary',
      value: selectedModelCapabilitySupported ? 'Exact row unlocked' : 'Exact row required',
      copy: supportStatusCopy,
      tone: supportTone,
    },
    {
      label: 'Draft',
      value: generationActive ? 'Keep writing while it replies' : selectedModelRunnable ? 'Send now' : supportBlocked ? 'Locked by support row' : selectedModel ? 'Ready when the model is' : 'Choose model first',
      copy: draftStatusLabel,
      tone: 'idle',
    },
  ]


  return (
    <section className={`chat-layout chat-layout-assistant chat-layout-modern view-stack ${isFreshThread ? 'chat-layout-empty' : ''}`}>
      {!demoMode && selectedConversation && (
        <div className="mobile-conversation-bar" aria-label="Conversation navigation">
          <button className="ghost-button mobile-conversation-trigger" onClick={() => setTab('history')}>
            <span>Conversations</span>
            <strong title={rawConversationTitle || 'Untitled chat'}>{conversationLabel}</strong>
          </button>
          <div className="mobile-conversation-status">
            {lastUpdated ? `Updated ${lastUpdated}` : 'Current thread'}
          </div>
        </div>
      )}

      <div className={`chat-canvas chat-canvas-modern ${isFreshThread ? 'chat-canvas-empty' : ''}`}>
        {isFreshThread ? (
          <div className="chat-empty-shell chat-empty-shell-assistant chat-empty-shell-modern">
            <div className="chat-empty-clean-glow" />
            <div className={`chat-empty-stage chat-empty-stage-clean chat-empty-stage-product is-${readinessState}`}>
              <div className="chat-stage-grid">
                <div className="chat-stage-main">
                  {/* Premium Gemini-style Hero Banner */}
                  <div className="chat-empty-hero chat-empty-hero-assistant chat-empty-hero-clean">
                    <div className="gemini-sparkle-greeting-wrapper">
                      <GeminiSparkle size={56} className="gemini-sparkle-greeting-icon" />
                    </div>
                    <h2 className="aurora-text-gradient">{productHeroTitle}</h2>
                    {productHeroSummary && <p className="hero-summary">{productHeroSummary}</p>}
                  </div>

                  {/* Suggestion Cards */}
                  {showPromptStarters && (
                    <div className="gemini-suggestion-section" aria-label="Prompt starters">
                      <div className="gemini-suggestion-grid">
                        {DEMO_PROMPTS.map((prompt, idx) => {
                          const icons = [
                            <svg key="0" className="gemini-card-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>,
                            <svg key="1" className="gemini-card-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>,
                            <svg key="2" className="gemini-card-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>,
                            <svg key="3" className="gemini-card-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>
                          ];
                          return (
                            <button
                              key={prompt}
                              type="button"
                              className="gemini-suggestion-card"
                              onClick={() => handleDemoPrompt(prompt)}
                              disabled={!composerDraftUnlocked}
                            >
                              <span className="suggestion-text">{prompt}</span>
                              <div className="suggestion-icon-wrapper">
                                {icons[idx] || icons[0]}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Compact Status Indicator Ribbon */}
                  <div className="gemini-readiness-row">
                    <span className="readiness-label">Camelid state:</span>
                    <span className={`readiness-status-badge is-${readinessState}`}>
                      <span className="status-dot"></span>
                      {selectedModelRunnable ? 'Ready for local chat' : readinessLabel}
                    </span>
                    <span className="readiness-model-name">({selectedModelName})</span>
                    {!selectedModelRunnable && (
                      <button type="button" className="readiness-action-btn" onClick={() => setTab(apiUnavailable ? 'api' : 'library')}>
                        {apiUnavailable ? 'API Dashboard' : 'Manage Models'}
                      </button>
                    )}
                  </div>
                </div>

                <div className="chat-stage-side">
                  <div className={`composer composer-assistant composer-assistant-stage composer-assistant-stage-clean composer-assistant-product composer-assistant-stage-modern is-${readinessState}`}>
                    <div className="composer-status-bar" aria-label="Composer status">
                      {composerStatusItems.map((item) => (
                        <div key={item.label} className="composer-status-chip">
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                        </div>
                      ))}
                    </div>
                    <textarea ref={composerRef} className="composer-input composer-input-assistant composer-input-assistant-stage" aria-label="Message Camelid" aria-describedby={composerReadinessId} value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={2} placeholder={composerPlaceholder} disabled={composerDisabled} />
                    <div className="composer-assistant-footer composer-assistant-footer-stage composer-assistant-footer-stage-clean">
                      <div className="composer-assistant-tools composer-assistant-tools-stage composer-assistant-tools-stage-clean">
                        {renderModelPicker()}
                        <button className="ghost-button ghost-button-quiet" onClick={secondaryAction} disabled={secondaryActionDisabled}>{secondaryActionLabel}</button>
                      </div>
                      <div className="composer-assistant-actions composer-assistant-actions-stage">
                        {generationActive && (
                          <button
                            className="ghost-button composer-stop-button"
                            aria-label="Stop Camelid generation"
                            onClick={stopGeneration}
                            disabled={stoppingGeneration}
                          >
                            {composerStopLabel}
                          </button>
                        )}
                        <button className="primary-button composer-send-button" aria-label="Send message to Camelid" title={!canSubmit ? sendDisabledReason : 'Send message to Camelid'} onClick={sendMessage} disabled={!canSubmit}>{composerSendLabel}</button>
                      </div>
                    </div>
                    {renderComposerModelSummary('composer-model-summary-stage')}
                    <p id={composerReadinessId} className={`composer-assistant-readiness-note is-${readinessState}`}>{readinessFinePrint}</p>
                    <p className={`composer-assistant-hint is-${canSubmit ? 'ready' : readinessState}`}>{composerHintCopy}</p>
                    {!selectedModelRunnable && <p className="composer-assistant-readiness-detail">{selectedModelGateSummary}</p>}
                  </div>
                </div>
              </div>
            </div>
          </div>

        ) : (
          <div className="chat-thread-shell">
            {!demoMode && (
              <>
                <div className={`chat-thread-header is-${readinessState}`} aria-label="Current Camelid chat status">
                  <div className="chat-thread-header-main">
                    <div className="chat-thread-header-copy">
                      <span className="chat-thread-header-eyebrow">Camelid chat</span>
                      <strong>{currentConversationSummary}</strong>
                      <p>{selectedModelRunnable ? 'Local assistant responses stay grounded in the current runtime and exact supported row.' : selectedModelGateSummary}</p>
                    </div>
                    <div className="chat-thread-header-badges chat-thread-header-badges-compact" aria-label="Conversation snapshot">
                      <div className={`chat-thread-header-badge chat-thread-header-badge-wide is-${selectionSummaryTone}`}>
                        <span>Selected model</span>
                        <strong>{selectedModelName}</strong>
                        <small>{selectedModelMeta}</small>
                      </div>
                      {conversationSnapshotItems.map((item) => (
                        <div key={item.label} className="chat-thread-header-badge">
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="chat-thread-toolbar" aria-label="Chat controls">
                    <div className="chat-thread-toolbar-main">
                      {renderModelPicker()}
                      <div className={`chat-thread-toolbar-status is-${readinessState}`}>
                        <span>State</span>
                        <strong>{selectedModelRunnable ? 'Ready to reply' : readinessLabel}</strong>
                      </div>
                    </div>
                    <div className="chat-thread-toolbar-actions">
                      <button className="ghost-button ghost-button-quiet" onClick={() => showNewChatLanding?.()} disabled={!showNewChatLanding}>
                        New chat
                      </button>
                      <button className="ghost-button ghost-button-quiet" onClick={secondaryAction} disabled={secondaryActionDisabled}>{secondaryActionLabel}</button>
                    </div>
                  </div>
                </div>

                {renderReadinessPills('chat-readiness-strip-live', 'Live chat exact-row readiness')}
              </>
            )}

            {!selectedModelRunnable && (
              <ChatSurfaceNotice
                state={readinessState}
                title={surfaceNoticeTitle}
                copy={surfaceNoticeCopy}
                actionLabel={readinessActionLabel}
                onAction={() => setTab(readinessActionTab)}
              />
            )}

            <div className="chat-thread chat-thread-assistant">
              {visibleMessages.length === 0 && !awaitingAssistant && <div className="empty-state empty-state-chat">Pick a ready model, then send the first message when you’re ready.</div>}
              {visibleMessages.length > 0 && !generationActive && selectedModelRunnable && (
                <div className="chat-follow-up-strip" aria-label="Follow-up prompts">
                  {FOLLOW_UP_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" className="demo-prompt-chip" onClick={() => handleDemoPrompt(prompt)}>
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
              {visibleMessages.map((message, index) => {
                const priorUserPrompt = message.role === 'assistant'
                  ? [...visibleMessages.slice(0, index)].reverse().find((item) => item.role === 'user')?.content
                  : null
                return (
                  <ChatMessageRow
                    key={message.id}
                    message={message}
                    generationElapsedSeconds={generationElapsedSeconds}
                    priorUserPrompt={priorUserPrompt}
                    onReusePrompt={setComposer}
                  />
                )
              })}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="message-row message-row-assistant user pending">
                      <div className="message-bubble message-bubble-assistant user pending">
                        <p>{pendingUserPrompt}</p>
                      </div>
                    </article>
                  )}
                  <article className="message-row message-row-assistant assistant pending is-streaming" aria-busy="true" data-streaming-state="active">
                    <div className="message-bubble message-bubble-assistant assistant pending">
                      <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={PREPARING_STREAMING_LABEL} />
                    </div>
                  </article>
                </>
              )}
              <div className="chat-thread-stream-anchor" ref={chatBottomRef} aria-hidden="true" />
            </div>
          </div>
        )}
      </div>

      {!isFreshThread && (
        <div className={`composer composer-assistant composer-assistant-floating composer-assistant-floating-modern is-${readinessState}`}>
          <div className="composer-status-bar composer-status-bar-floating" aria-label="Composer status">
            {composerStatusItems.map((item) => (
              <div key={item.label} className="composer-status-chip">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <textarea ref={composerRef} className="composer-input composer-input-assistant" aria-label="Message Camelid" aria-describedby={composerReadinessId} value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={3} placeholder={composerPlaceholder} disabled={composerDisabled} />
          <div className="composer-assistant-footer">
            <div className="composer-assistant-tools">
              {renderModelPicker()}
              {!demoMode && <span className="composer-meta-pill">{selectedModelMeta}</span>}
              {!demoMode && <button className="ghost-button subtle-action" onClick={secondaryAction} disabled={secondaryActionDisabled}>{secondaryActionLabel}</button>}
            </div>
            <div className="composer-assistant-actions">
              {generationActive && (
                <button
                  className="ghost-button composer-stop-button"
                  aria-label="Stop Camelid generation"
                  onClick={stopGeneration}
                  disabled={stoppingGeneration}
                >
                  {composerStopLabel}
                </button>
              )}
              <button className="primary-button composer-send-button" aria-label="Send message to Camelid" title={!canSubmit ? sendDisabledReason : 'Send message to Camelid'} onClick={sendMessage} disabled={!canSubmit}>{composerSendLabel}</button>
            </div>
          </div>
          {renderComposerModelSummary('composer-model-summary-floating')}
          <p id={composerReadinessId} className={`composer-assistant-readiness-note composer-assistant-readiness-note-floating is-${readinessState}`}>{readinessFinePrint}</p>
          <p className={`composer-assistant-hint composer-assistant-hint-floating is-${canSubmit ? 'ready' : readinessState}`}>{composerHintCopy}</p>
          {!selectedModelRunnable && <p className="composer-assistant-readiness-detail composer-assistant-readiness-detail-floating">{selectedModelGateSummary}</p>}
        </div>
      )}
    </section>
  )
}
