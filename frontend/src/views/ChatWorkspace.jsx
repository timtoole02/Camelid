import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { compatibilityHintCopy, compatibilityHintLabel, exactRowSupportLanes, findCompatibilityHint, isCompatibilitySupportedForModel } from '../lib/capabilities'
import { clampText, formatDate, formatRate } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel } from '../lib/modelState'

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
  'Build a tiny arcade maze game in one self-contained HTML file',
  'Create a glassy launch page with animated CSS and one working button',
  'Write a compact Python snake game using tkinter',
]

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

const isInterruptedPlaceholderMessage = (message) => {
  if (message?.role !== 'assistant') return false
  const content = String(message?.content || '').trim().toLowerCase()
  return content === '(generation interrupted)' || content === '(generation stopped)'
}

const ChatMessageRow = memo(function ChatMessageRow({ message, generationElapsedSeconds, priorUserPrompt }) {
  const messageContent = cleanLegacyDemoCapCopy(message.content)
  const assistantStreaming = message.role === 'assistant' && Boolean(message.streaming)
  const isOpenStreamingCode = assistantStreaming && hasOpenCodeFence(messageContent)
  const streamingPhase = message.streaming_phase || (messageContent ? 'streaming' : 'generating')
  const liveStatusLabel = streamingStatusLabel(streamingPhase, generationElapsedSeconds, isOpenStreamingCode)
  const hasTokenMetrics = false
  const showStreamingStatus = assistantStreaming && !messageContent
  const showLiveGenerationBadge = assistantStreaming && Boolean(messageContent)
  const showLengthWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'length'

  return (
    <article
      className={`message-row message-row-gemini ${message.role} ${assistantStreaming ? 'is-streaming' : ''}`}
      aria-busy={assistantStreaming ? 'true' : undefined}
      data-streaming-state={assistantStreaming ? 'active' : undefined}
      data-streaming-code-state={isOpenStreamingCode ? 'open' : undefined}
    >
      <div className={`message-bubble message-bubble-gemini ${message.role}`}>
        {showStreamingStatus && <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} compact />}
        {message.role === 'assistant'
          ? messageContent || !assistantStreaming
            ? <AssistantMarkdown content={messageContent} streaming={assistantStreaming} />
            : null
          : <p>{messageContent}</p>}
        {showLiveGenerationBadge && <LiveGenerationBadge elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} />}
        {showLengthWarning && (
          <div className="message-finish-warning" role="status">
            Stopped before completing. Ask “continue” for a complete file.
          </div>
        )}
        {hasTokenMetrics && (
          <div className="message-token-metrics" aria-label="Generation speed">
            {message.first_byte_ms !== null && message.first_byte_ms !== undefined && <span>TTFB {(Number(message.first_byte_ms) / 1000).toFixed(2)}s</span>}
            {message.first_event_ms !== null && message.first_event_ms !== undefined && <span>First event {(Number(message.first_event_ms) / 1000).toFixed(2)}s</span>}
            {message.first_content_ms !== null && message.first_content_ms !== undefined && <span>TTFT {(Number(message.first_content_ms) / 1000).toFixed(2)}s</span>}
            <span>In {formatRate(message.tokens_in_per_sec)}</span>
            <span>Decode {formatRate(message.tokens_out_per_sec)}</span>
          </div>
        )}
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
  sending,
  selectedModelRunnable,
  setTab,
  demoMode = false,
}) {
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const chatBottomRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
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

  const isFreshThread = selectedConversation
    ? (visibleMessages.length === 0 && !pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
    : (!pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
  const handleComposerKeyDown = async (event) => {
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
  const runnableModels = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked)
  const hasRunnableChoices = runnableModels.length > 0
  const modelPickerTitle = selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what Camelid should use for this chat.'
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedRuntimeReady = selectedChatGate.runtimeReady
  const selectedModelCapabilitySupported = selectedChatGate.contractSupported || isCompatibilitySupportedForModel(capabilities, selectedModel)
  const supportBlocked = selectedRuntimeReady && !selectedModelCapabilitySupported
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityLabel = selectedModel
    ? compatibilityHintLabel(selectedCompatibilityHint, 'No matching COMPATIBILITY.md row')
    : 'No model selected'
  const selectedCompatibilityCopy = selectedModel
    ? compatibilityHintCopy(selectedCompatibilityHint)
    : 'Choose a model before inferring any support boundary. Camelid will not promote filenames or saved paths into compatibility claims.'
  const selectedModelMeta = supportBlocked
    ? 'Load a supported model to chat'
    : !selectedModelRunnable
      ? describeModelState(selectedModel)
      : selectedChatGate.runtimeLoaded
      ? 'Ready'
      : 'Ready to chat'
  const canSubmit = Boolean(composer.trim()) && selectedModelRunnable && !generationActive
  const capabilityLaneStatus = getChatCapabilityLaneCopy(selectedChatGate, capabilities)
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const runtimeStatusLabel = selectedModelRunnable
    ? 'Local chat ready'
    : selectedRuntimeReady
      ? 'Runtime ready, support gated'
      : runtime?.loaded_now
        ? 'Loaded, not generation-ready'
        : 'No generation-ready model'
  const runtimeStatusCopy = selectedModelRunnable
    ? `${selectedModelName} is loaded now and generation_ready=true.`
    : selectedRuntimeReady
      ? 'The runtime is ready; Camelid still needs an exact supported row before chat unlocks.'
      : runtime?.loaded_now
        ? 'Wait for generation_ready=true before sending prompts.'
        : 'Load a local GGUF from Library to start the readiness check.'
  const supportStatusLabel = selectedModelCapabilitySupported
    ? selectedCompatibilityLabel
    : selectedModel
      ? selectedCompatibilityLabel
      : 'Choose model first'
  const supportStatusCopy = selectedModelCapabilitySupported
    ? `${selectedCompatibilityLabel}. COMPATIBILITY.md and /api/capabilities agree for this model and quant.`
    : selectedModel
      ? selectedCompatibilityCopy
      : 'Camelid does not infer broad support from filenames, families, or saved paths.'
  const readinessFinePrint = selectedModelRunnable
    ? 'Ready for this loaded exact row. Broader scope details stay in /api/capabilities instead of the chat composer.'
    : 'Chat unlocks only after loaded_now=true, generation_ready=true, and an exact supported compatibility row all match.'
  const emptyHeroEyebrow = 'Camelid'
  const readinessState = selectedModelRunnable ? 'ready' : supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const readinessLabel = selectedModelRunnable
    ? 'Ready'
    : supportBlocked
      ? 'Choose a supported model'
      : selectedModel
        ? 'Loading model'
        : 'Choose a model to begin'
  const productHeroTitle = selectedModelRunnable
    ? 'How can I help?'
    : supportBlocked
      ? 'Choose a supported model.'
      : 'Load a model to begin.'
  const productHeroSummary = selectedModelRunnable
    ? 'Pick a polished demo prompt or ask anything — Camelid is running the selected exact row locally.'
    : supportBlocked
      ? 'Camelid keeps demos honest: chat unlocks only for exact supported model rows.'
      : 'Load a generation-ready GGUF model, then run a clean local demo without cloud handoffs.'
  const handleDemoPrompt = (prompt) => {
    if (generationActive || !selectedModelRunnable) return
    setComposer(prompt)
  }

  const renderReadinessPills = (extraClass = '', ariaLabel = 'Chat readiness and support boundary') => (
    <div className={`chat-readiness-pill-row chat-readiness-strip-live ${extraClass} is-${readinessState}`} aria-label={ariaLabel}>
      <div className="chat-readiness-pill" title={runtimeStatusCopy}>
        <span>Runtime</span>
        <strong>{runtimeStatusLabel}</strong>
      </div>
      <div className="chat-readiness-pill" title={supportStatusCopy}>
        <span>Support</span>
        <strong>{supportStatusLabel}</strong>
      </div>
      <div className="chat-readiness-pill chat-readiness-pill-wide" title={capabilityLaneStatus.copy}>
        <span>Capabilities</span>
        <strong>{capabilityLaneStatus.label}</strong>
      </div>
    </div>
  )

  const renderModelPicker = () => {
    if (!hasRunnableChoices) {
      return (
        <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>
          Choose model
        </button>
      )
    }

    return (
      <label className="composer-model-picker" title={modelPickerTitle}>
        <span className="composer-tool-label">Model</span>
        <select
          className="composer-model-select"
          aria-label="Choose model for chat"
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={generationActive}
        >
          {runnableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
    )
  }


  return (
    <section className={`chat-layout chat-layout-gemini view-stack ${isFreshThread ? 'chat-layout-empty' : ''}`}>
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

      <div className={`chat-canvas ${isFreshThread ? 'chat-canvas-empty' : ''}`}>
        {isFreshThread ? (
          <div className="chat-empty-shell chat-empty-shell-gemini">
            <div className={`chat-empty-stage chat-empty-stage-clean chat-empty-stage-product is-${readinessState}`}>
              <div className="chat-empty-hero chat-empty-hero-gemini chat-empty-hero-clean">
                <p className="chat-empty-greeting">{emptyHeroEyebrow}</p>
                <h2>{productHeroTitle}</h2>
                {productHeroSummary && <p className="hero-summary">{productHeroSummary}</p>}
              </div>

              {!demoMode && (
                renderReadinessPills()
              )}

              {selectedModelRunnable && (
                <div className="demo-prompt-panel" aria-label="Polished demo prompts">
                  <span>Demo starters</span>
                  <div className="demo-prompt-strip">
                    {DEMO_PROMPTS.map((prompt) => (
                      <button key={prompt} type="button" className="demo-prompt-chip" onClick={() => handleDemoPrompt(prompt)} disabled={generationActive}>
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="composer composer-gemini composer-gemini-stage composer-gemini-stage-clean composer-gemini-product">
                <textarea className="composer-input composer-input-gemini composer-input-gemini-stage" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={2} placeholder={selectedModelRunnable ? 'Message Camelid…' : 'Load a model first'} disabled={generationActive || !selectedModelRunnable} />
                <div className="composer-gemini-footer composer-gemini-footer-stage composer-gemini-footer-stage-clean">
                  <div className="composer-gemini-tools composer-gemini-tools-stage composer-gemini-tools-stage-clean">
                    {renderModelPicker()}
                    {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>Open Library</button>}
                  </div>
                  <div className="composer-gemini-actions composer-gemini-actions-stage">
                    <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{generationActive ? `Generating ${generationElapsedSeconds}s…` : 'Send'}</button>
                  </div>
                </div>
                <p className={`composer-gemini-readiness-note is-${readinessState}`}>{readinessFinePrint}</p>
              </div>
            </div>
          </div>
        ) : (
          <>
            {!demoMode && (
              <>
                <div className={`chat-session-strip is-${readinessState}`} aria-label="Current Camelid chat status">
                  <span className="chat-session-dot" aria-hidden="true" />
                  <strong>{selectedModelName}</strong>
                  <small>{selectedModelRunnable ? 'Ready when you are' : readinessLabel}</small>
                </div>

                {renderReadinessPills('chat-readiness-strip-live', 'Live chat exact-row readiness')}
              </>
            )}

            {!selectedModelRunnable && (
              <div className="setup-card setup-card-inline setup-card-gemini">
                <div>
                  <p className="panel-kicker">Before you chat</p>
                  <h2>{supportBlocked ? 'Support contract needs an exact row' : 'Choose a runnable model'}</h2>
                  <p className="hero-summary">{supportBlocked ? `${selectedCompatibilityLabel}. ${selectedCompatibilityCopy}` : describeModelState(selectedModel)}</p>
                </div>
                <div className="composer-actions single-action-row">
                  <button className="primary-button" onClick={() => setTab('library')}>Open Library</button>
                </div>
              </div>
            )}

            <div className="chat-thread chat-thread-gemini">
              {visibleMessages.length === 0 && !awaitingAssistant && <div className="empty-state empty-state-chat">Pick a ready model, then send the first message when you’re ready.</div>}
              {visibleMessages.map((message, index) => {
                const priorUserPrompt = message.role === 'assistant'
                  ? [...visibleMessages.slice(0, index)].reverse().find((item) => item.role === 'user')?.content
                  : null
                return <ChatMessageRow key={message.id} message={message} generationElapsedSeconds={generationElapsedSeconds} priorUserPrompt={priorUserPrompt} />
              })}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="message-row message-row-gemini user pending">
                      <div className="message-bubble message-bubble-gemini user pending">
                        <p>{pendingUserPrompt}</p>
                      </div>
                    </article>
                  )}
                  <article className="message-row message-row-gemini assistant pending is-streaming" aria-busy="true" data-streaming-state="active">
                    <div className="message-bubble message-bubble-gemini assistant pending">
                      <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={PREPARING_STREAMING_LABEL} />
                    </div>
                  </article>
                </>
              )}
              <div className="chat-thread-stream-anchor" ref={chatBottomRef} aria-hidden="true" />
            </div>
          </>
        )}
      </div>

      {!isFreshThread && (
        <div className="composer composer-gemini composer-gemini-floating">
          <textarea className="composer-input composer-input-gemini" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={3} placeholder={selectedModelRunnable ? 'Ask Camelid' : 'Load a model first'} disabled={generationActive || !selectedModelRunnable} />
          <div className="composer-gemini-footer">
            <div className="composer-gemini-tools">
              {renderModelPicker()}
              {!demoMode && <span className="composer-meta-pill">{selectedModelMeta}</span>}
              {!demoMode && selectedModelRunnable && <button className="ghost-button subtle-action" onClick={saveToMemory} disabled={generationActive}>Save to memory</button>}
            </div>
            <div className="composer-gemini-actions">
              {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button" onClick={() => setTab('library')}>Open Library</button>}
              <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{generationActive ? `Generating ${generationElapsedSeconds}s…` : 'Send'}</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
