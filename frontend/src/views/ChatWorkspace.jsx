import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { getChatGateState } from '../lib/chatGate'
import { displayQuantLabel, exactArtifactFilenameForRow } from '../lib/capabilities'
import { formatModelLabel } from '../lib/formatters'
import { isEmbeddingOnlyModel, isGenerationCapableModel } from '../lib/modelCapabilities.js'
import { applyGemma4GhostChatTokenCap, getConfiguredMaxTokens, isBitNetB158ChatModel, modelContextLength, validateSendBudget } from '../lib/responseLimits'
import { CamelidMark } from '../components/ui/CamelidMark'
import { Avatar } from '../components/ui/Avatar'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { IconSend, IconStop, IconMemory, IconReceipt, IconThinking, IconBolt, IconChart, IconChat, IconChevronDown, IconEdit, IconImage, IconInfo, IconClose, IconSearch } from '../components/ui/icons'
import { Tooltip } from '../components/ui/Tooltip'
import { MessageTurn } from '../components/chat/MessageTurn'
import { ChatControls } from '../components/chat/ChatControls'
import { PREPARING_STREAMING_LABEL, StreamingLoader } from '../components/chat/render/StreamingIndicator'
import { classifyWebResearchNeed } from '../lib/webResearch.js'

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const isInterruptedPlaceholderMessage = (message) => {
  if (message?.role !== 'assistant') return false
  const content = String(message?.content || '').trim().toLowerCase()
  return content === '(generation interrupted)' || content === '(generation stopped)'
}

const SUGGESTIONS = [
  { title: 'Summarize this plan', body: 'Summarize this implementation plan and call out the risks', Icon: IconChart },
  { title: 'Draft a release note', body: 'Draft a concise release note from these changes', Icon: IconEdit },
  { title: 'Prioritize next steps', body: 'Turn this checklist into a prioritized next-step plan', Icon: IconBolt },
  { title: 'Tighten this answer', body: 'Review this response and tighten it into a shorter final answer', Icon: IconChat },
]

const FOLLOW_UP_PROMPTS = [
  'Continue with the exact next steps.',
  'Tighten that into a shorter final answer.',
  'Turn this into a checklist I can execute.',
]

const MAX_VISION_UPLOAD_BYTES = 3 * 1024 * 1024
const MAX_VISION_EDGE = 1600

/* Day separators: a calendar-day key plus a short label ("Today", "Yesterday",
   "Tue, Aug 4") rendered between turns whenever the day changes. */
const dayKeyOf = (value) => {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`
}

const formatDayLabel = (value) => {
  const date = new Date(value)
  const now = new Date()
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
  if (dayKeyOf(date) === dayKeyOf(now)) return 'Today'
  if (dayKeyOf(date) === dayKeyOf(yesterday)) return 'Yesterday'
  const sameYear = date.getFullYear() === now.getFullYear()
  return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', ...(sameYear ? {} : { year: 'numeric' }) })
}

const readAsDataUrl = (blob) => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(String(reader.result || ''))
  reader.onerror = () => reject(reader.error || new Error('Could not read the image.'))
  reader.readAsDataURL(blob)
})

const loadBrowserImage = (file) => new Promise((resolve, reject) => {
  const url = URL.createObjectURL(file)
  const image = new Image()
  image.onload = () => {
    URL.revokeObjectURL(url)
    resolve(image)
  }
  image.onerror = () => {
    URL.revokeObjectURL(url)
    reject(new Error('The selected file is not a readable PNG or JPEG image.'))
  }
  image.src = url
})

const canvasBlob = (canvas, quality) => new Promise((resolve) => {
  canvas.toBlob(resolve, 'image/jpeg', quality)
})

const resizeComposerInput = (input) => {
  if (!input) return
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 220)}px`
}

async function prepareVisionAttachment(file) {
  if (!['image/png', 'image/jpeg'].includes(file.type)) {
    throw new Error('Choose a PNG or JPEG image.')
  }
  const image = await loadBrowserImage(file)
  let blob = file
  let type = file.type
  if (file.size > MAX_VISION_UPLOAD_BYTES || Math.max(image.naturalWidth, image.naturalHeight) > MAX_VISION_EDGE) {
    const scale = Math.min(1, MAX_VISION_EDGE / Math.max(image.naturalWidth, image.naturalHeight))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale))
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale))
    const context = canvas.getContext('2d')
    context.fillStyle = '#fff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    blob = await canvasBlob(canvas, 0.9)
    if (blob?.size > MAX_VISION_UPLOAD_BYTES) blob = await canvasBlob(canvas, 0.72)
    if (!blob) throw new Error('Could not prepare the selected image.')
    type = 'image/jpeg'
  }
  if (blob.size > MAX_VISION_UPLOAD_BYTES) {
    throw new Error('The prepared image is still too large. Choose an image under 3 MB.')
  }
  return {
    name: file.name,
    type,
    size: blob.size,
    width: image.naturalWidth,
    height: image.naturalHeight,
    data_url: await readAsDataUrl(blob),
  }
}

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  activateModel = null,
  loadingModelId = null,
  models,
  runtime,
  capabilities,
  pendingConversation,
  composer,
  setComposer,
  saveToMemory,
  sendMessage,
  resendFromMessage = null,
  stopGeneration,
  sending,
  receiptMode = false,
  setReceiptMode = null,
  thinkingMode = false,
  setThinkingMode = null,
  webResearchEnabled = true,
  setWebResearchEnabled = null,
  webResearchStatus = { phase: 'idle', sourceCount: 0 },
  stoppingGeneration = false,
  selectedModelRunnable,
  selectedModelExperimental = false,
  setTab,
  showNewChatLanding = null,
  firstRunActive = false,
  demoMode = false,
}) {
  // Derive readiness from the shared gate here as well as in the dashboard hook.
  // This keeps the rendered surface coherent in the first frame after a runtime
  // transition, before parent props finish refreshing.
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const supportedChatReady = selectedChatGate.chatUnlocked
  const verifiedChatReady = selectedChatGate.chatMode === 'verified'
  const varianceChatReady = selectedChatGate.chatMode === 'variance'
  const unverifiedChatReady = selectedChatGate.chatMode === 'experimental'
  const canChat = supportedChatReady || verifiedChatReady || varianceChatReady || unverifiedChatReady
  const nonSupportedChatReady = !supportedChatReady && (selectedModelExperimental || verifiedChatReady || varianceChatReady || unverifiedChatReady)
  const visionReady = canChat && Boolean(runtime?.vision_ready)
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const [showControls, setShowControls] = useState(false)
  const [showAllMessages, setShowAllMessages] = useState(false)
  const [userScrolledAway, setUserScrolledAway] = useState(false)
  const [composerImage, setComposerImage] = useState(null)
  const [imageError, setImageError] = useState('')
  const chatBottomRef = useRef(null)
  const composerRef = useRef(null)
  const imageInputRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
  const composerReadinessId = 'camelid-chat-readiness-note'

  const rawVisibleMessages = useMemo(
    () => (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message)),
    [selectedConversation?.messages],
  )
  const visibleWebResearchStatus = !webResearchStatus?.conversationId
    || webResearchStatus.conversationId === selectedConversation?.id
    ? webResearchStatus
    : { phase: 'idle', sourceCount: 0, conversationId: null }
  const hasStreamingAssistant = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming)
  const hasStreamingAssistantContent = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming && String(m.content || '').trim())
  const requestActive = Boolean(sending)
  // Sending is process-global (only one local-model request may run), while
  // loaders, stop controls, and auto-follow belong only to the conversation
  // that owns the pending/streaming turn.
  const generationActive = Boolean(pendingConversation || hasStreamingAssistant)
  const visibleMessages = useMemo(() => {
    if (!generationActive) return rawVisibleMessages
    return rawVisibleMessages.filter((message, index, messages) => {
      const isTrailingInterruptedPlaceholder = index === messages.length - 1 && isInterruptedPlaceholderMessage(message)
      return !isTrailingInterruptedPlaceholder
    })
  }, [generationActive, rawVisibleMessages])
  const pendingPrompt = String(pendingConversation?.content || '').trim()
  const pendingPromptAlreadyVisible = Boolean(
    pendingPrompt && [...visibleMessages].reverse().some((m) => m.role === 'user' && m.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const lastVisibleMessage = visibleMessages.at(-1)
  const lastVisibleMessageIsUser = lastVisibleMessage?.role === 'user'
  const awaitingAssistant = Boolean(generationActive && !hasStreamingAssistantContent && !hasStreamingAssistant && (pendingPrompt || lastVisibleMessageIsUser))
  const streamingScrollSignature = useMemo(() => (
    visibleMessages.map((m) => `${m.id}:${m.streaming ? 'streaming' : 'done'}:${String(m.content || '').length}`).join('|')
    + `|awaiting:${awaitingAssistant ? '1' : '0'}|active:${generationActive ? '1' : '0'}`
  ), [awaitingAssistant, generationActive, visibleMessages])
  const isFreshThread = selectedConversation
    ? (visibleMessages.length === 0 && !pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
    : (!pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)

  // ----- Gate / readiness derivations (shared exact-row chat gate) -----
  const selectedEmbeddingOnly = selectedChatGate.embeddingOnly
  const selectedEmbeddingReady = selectedChatGate.embeddingReady
  const selectedBitNetChatModel = isBitNetB158ChatModel(selectedModel, runtime, selectedModelId)
  const apiUnavailable = runtime?.status === 'offline'
  const selectedRuntimeReady = selectedChatGate.runtimeReady
  const selectedModelCapabilitySupported = selectedChatGate.contractSupported

  useEffect(() => {
    if (selectedBitNetChatModel && thinkingMode && setThinkingMode) setThinkingMode(false)
  }, [selectedBitNetChatModel, setThinkingMode, thinkingMode])
  const supportBlocked = selectedRuntimeReady && !selectedModelCapabilitySupported
  /* The two blocked states a reader can actually act on, each named concretely.
     "Pick a verified model" alone leaves someone who is one file away from
     working guessing at which file that is. */
  const blockedSpecifics = (() => {
    const hint = selectedChatGate.hint
    if (hint?.kind === 'artifact_mismatch') {
      const filename = exactArtifactFilenameForRow(hint.target)
      return filename ? `it requires the exact ${filename} artifact` : null
    }
    if (hint?.kind === 'quant_mismatch') {
      // observedQuant is a match key ("Q40"), not something to show a reader.
      const verified = displayQuantLabel(hint.target?.quantization)
      const observed = displayQuantLabel(selectedModel?.quant || hint.observedQuant)
      if (verified && observed) return `this build is ${observed} and the verified build is ${verified}`
      if (verified) return `only the ${verified} build is verified`
    }
    return null
  })()
  const selectedRuntimeMatchesLoadedModel = Boolean(selectedChatGate.runtimeLoaded)
  const selectedRuntimeLoadedButNotReady = Boolean(
    selectedRuntimeMatchesLoadedModel && !selectedChatGate.runtimeGenerationReady,
  )
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const selectedModelIssue = selectedModel?.load_error || selectedModel?.install_error || ''

  /* One-line composer status: dot + a single short sentence. The longer detail
     (send gate, reply cap, local-inference note) folds into the tooltip below. */
  const webResearchPlan = useMemo(() => classifyWebResearchNeed(composer), [composer])
  const webResearchWillUsePublicWeb = webResearchEnabled && webResearchPlan.needed && canChat
  const statusLine = visibleWebResearchStatus?.phase === 'researching'
    ? 'Reading relevant web sources before Camelid answers…'
    : webResearchWillUsePublicWeb
      ? 'Web Auto will send linked URLs or a search query to the public web.'
    : apiUnavailable
    ? 'Not connected — start the local server to chat.'
    : selectedEmbeddingOnly
      ? selectedEmbeddingReady
        ? `${selectedModelName} is ready for embeddings and reranking, not Chat.`
        : `${selectedModelName} is an embedding model — load it from Models.`
      : supportedChatReady
        ? `${selectedModelName} is loaded and ready.`
      : verifiedChatReady
        ? `${selectedModelName} is loaded and verified for its checked envelope.`
      : varianceChatReady
        ? `${selectedModelName} is loaded and ready; reference output can vary.`
      : unverifiedChatReady
        ? `${selectedModelName} is ready — replies are not verified.`
      : selectedModelIssue
        ? selectedModelIssue
        : selectedRuntimeLoadedButNotReady
          ? `${selectedModelName} is loaded, but this build cannot run it for Chat.`
        : supportBlocked
          ? `${selectedModelName} isn't verified for chat yet.`
          : selectedRuntimeMatchesLoadedModel
            ? `${selectedModelName} is warming up — send unlocks shortly.`
            : selectedModel
              ? `${selectedModelName} is getting ready — you can draft now.`
              // The activation card above owns the instruction during first run;
              // this line just points at it instead of restating it.
              : firstRunActive
                ? 'Send unlocks as soon as the model above finishes setting up.'
                : models.length
                  ? 'No model loaded — choose one above to chat.'
                  : 'No model loaded — add one above to chat.'

  const productHeroTitle = canChat ? 'How can I help?' : "Hi there, let's get into it"
  const productHeroSummary = supportedChatReady
    ? 'Local chat is ready. Ask anything — responses stay grounded in the loaded model.'
    : verifiedChatReady
      ? 'Verified local chat is ready. Extended-context support is still limited.'
    : varianceChatReady
      ? 'Local chat is ready. This exact model runs normally, with disclosed reference-output variance.'
    : unverifiedChatReady
      ? 'Unverified local chat is ready. Replies are clearly marked.'
    : apiUnavailable
      ? 'Keep writing here. Send unlocks again once the local API responds.'
      : selectedEmbeddingOnly
        ? selectedEmbeddingReady
          ? 'This model is loaded for embeddings and reranking. Choose a generation model to chat.'
          : 'This model creates embeddings for search and reranking. Load it from Models, or choose a generation model to chat.'
        : selectedRuntimeLoadedButNotReady
          ? 'This model is loaded but not runnable for Chat in this build. Choose another model to continue.'
        : supportBlocked
          /* When the blocker is a near miss — wrong file, or the right model at an
             unverified quantization — naming it is far more actionable than "pick a
             verified model": they are usually one download away, not one decision
             away. */
          ? (blockedSpecifics
            ? `This model isn't verified for chat yet: ${blockedSpecifics}. Pick a verified model to unlock send.`
            : "This model isn't verified for chat yet. Pick a verified model to unlock send.")
        : selectedModel
          ? 'Your draft is ready now. Send unlocks as soon as this model is ready.'
          // The activation card above already names the one thing to do; repeating
          // "pick a model" here would offer a second, vaguer instruction.
          : firstRunActive
            ? 'Camelid answers with a model running on this machine. Set one up above and this becomes a chat.'
            : 'Pick a local GGUF model first. Camelid will show the readiness path here.'

  const readinessState = canChat ? 'ready' : apiUnavailable ? 'offline' : selectedEmbeddingOnly ? 'blocked' : selectedRuntimeLoadedButNotReady || supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const statusTone = visibleWebResearchStatus?.phase === 'researching'
    ? 'ready'
    : webResearchWillUsePublicWeb
      ? 'warn'
    : supportedChatReady || verifiedChatReady ? 'ready' : varianceChatReady || unverifiedChatReady ? 'warn' : apiUnavailable ? 'offline' : selectedEmbeddingReady ? 'ready' : selectedEmbeddingOnly ? 'neutral' : supportBlocked ? 'warn' : runtime?.loaded_now ? 'warn' : 'neutral'

  const canSubmit = Boolean(composer.trim()) && canChat && !requestActive
  const sendDisabledReason = requestActive
    ? 'Wait for the current reply to finish before sending again.'
    : canChat
      ? ''
      : apiUnavailable
        ? 'Sending unlocks once the connection is back.'
        : selectedEmbeddingOnly
          ? 'Choose a generation model to send this chat.'
          : supportBlocked
          ? 'Choose a verified model to send.'
          : selectedRuntimeLoadedButNotReady
            ? 'Choose a runnable model to send.'
          : selectedModel
            ? 'Sending unlocks once this model is ready.'
            : 'Choose a model before sending.'

  const composerDraftUnlocked = Boolean(selectedModel || apiUnavailable)
  const composerDisabled = !composerDraftUnlocked
  const composerPlaceholder = canChat
    ? 'Message Camelid…'
    : apiUnavailable
      ? 'Draft a prompt while the Camelid API comes back'
      : selectedEmbeddingOnly
        ? 'Choose a generation model to send a chat'
        : selectedRuntimeLoadedButNotReady
          ? 'Choose a runnable model; this loaded model is blocked'
        : composerDraftUnlocked
        ? 'Draft a prompt while Camelid finishes getting ready'
        : firstRunActive
          ? 'Set up the model above, then chat here'
          : isFreshThread
            ? 'Load a model first'
            : 'Choose a ready model first'
  const composerStopLabel = stoppingGeneration
    ? 'Stopping…'
    : visibleWebResearchStatus?.phase === 'researching'
      ? 'Stop research'
      : 'Stop'
  const composerStopAriaLabel = visibleWebResearchStatus?.phase === 'researching'
    ? 'Stop web research'
    : 'Stop Camelid generation'
  const awaitingAssistantLabel = visibleWebResearchStatus?.phase === 'researching'
    ? 'Reading relevant web sources…'
    : PREPARING_STREAMING_LABEL
  const secondaryActionLabel = canChat ? 'Save to memory' : (apiUnavailable ? 'Open API' : 'Open Models')
  const secondaryAction = canChat ? saveToMemory : () => setTab(apiUnavailable ? 'api' : 'library')
  const secondaryActionDisabled = canChat ? requestActive : false

  // ----- Effects -----
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
    if (!visionReady) {
      setComposerImage(null)
      setImageError('')
    }
  }, [visionReady, selectedModelId])

  useEffect(() => {
    if (!generationActive) return undefined
    autoFollowGenerationRef.current = true
    setUserScrolledAway(false)
    /* Auto-follow is released by the user's GESTURE, not by how far they got.
       Keying it to a distance threshold meant a small trackpad scroll left the
       viewport inside the band, so the next token re-anchored to the bottom and
       the page appeared to fight the scroll. Any upward wheel/touch/key intent
       releases it immediately; it re-engages only once they return to the
       bottom, so a released stream never silently yanks back. */
    const setFollow = (follow) => {
      if (follow !== autoFollowGenerationRef.current) setUserScrolledAway(!follow)
      autoFollowGenerationRef.current = follow
    }
    const releaseOnUpwardIntent = (event) => {
      if (event.type === 'wheel' && event.deltaY >= 0) return
      if (event.type === 'keydown' && !['ArrowUp', 'PageUp', 'Home'].includes(event.key)) return
      setFollow(false)
    }
    const updateAutoFollow = () => {
      const el = document.querySelector('.cxchat__scroll')
      if (!el) return
      // Re-engage only at the very bottom; never re-engage mid-scroll.
      const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight)
      if (distanceFromBottom <= 24) setFollow(true)
    }
    const el = document.querySelector('.cxchat__scroll')
    el?.addEventListener('scroll', updateAutoFollow, { passive: true })
    el?.addEventListener('wheel', releaseOnUpwardIntent, { passive: true })
    el?.addEventListener('touchmove', releaseOnUpwardIntent, { passive: true })
    el?.addEventListener('keydown', releaseOnUpwardIntent)
    return () => {
      el?.removeEventListener('scroll', updateAutoFollow)
      el?.removeEventListener('wheel', releaseOnUpwardIntent)
      el?.removeEventListener('touchmove', releaseOnUpwardIntent)
      el?.removeEventListener('keydown', releaseOnUpwardIntent)
    }
  }, [generationActive, selectedConversation?.id])

  useLayoutEffect(() => {
    if (!generationActive || !autoFollowGenerationRef.current) return undefined
    const frame = window.requestAnimationFrame(() => {
      chatBottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [generationActive, streamingScrollSignature])

  useLayoutEffect(() => {
    const resize = () => resizeComposerInput(composerRef.current)
    window.addEventListener('resize', resize)
    window.visualViewport?.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      window.visualViewport?.removeEventListener('resize', resize)
    }
  }, [])

  useLayoutEffect(() => {
    resizeComposerInput(composerRef.current)
  }, [composer, isFreshThread, selectedConversation?.id])

  useEffect(() => {
    if (generationActive || !composerDraftUnlocked) return
    const input = composerRef.current
    if (!input) return
    const activeElement = document.activeElement
    if (activeElement && activeElement !== document.body && activeElement !== input) return
    const frame = window.requestAnimationFrame(() => input.focus())
    return () => window.cancelAnimationFrame(frame)
  }, [composerDraftUnlocked, generationActive, isFreshThread, selectedConversation?.id])

  const handleSendMessage = async () => {
    const image = composerImage
    setComposerImage(null)
    setImageError('')
    await sendMessage({ overrideImage: image })
  }

  const handleVisionFile = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setImageError('')
    try {
      setComposerImage(await prepareVisionAttachment(file))
      composerRef.current?.focus()
    } catch (error) {
      setComposerImage(null)
      setImageError(error?.message || 'Could not attach the image.')
    }
  }

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Escape' && generationActive) {
      event.preventDefault()
      stopGeneration?.()
      return
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) await handleSendMessage()
    }
  }

  /* Focus the composer with the caret at the end, so Enter sends instead of
     re-triggering the clicked suggestion (matches the vision-attach flow). */
  const handleSuggestion = (prompt) => {
    if (!composerDraftUnlocked) return
    setComposer(prompt)
    window.requestAnimationFrame(() => {
      const input = composerRef.current
      if (!input) return
      input.focus()
      const end = input.value.length
      input.setSelectionRange(end, end)
    })
  }

  // ----- Model picker -----
  const modelCanChat = (model) => ['supported', 'verified', 'variance', 'experimental'].includes(getChatGateState(capabilities, model, runtime).chatMode)
  const chatModels = models.filter((model) => isGenerationCapableModel(model, runtime))
  const embeddingModels = models.filter((model) => isEmbeddingOnlyModel(model, runtime))
  const runnableModels = chatModels.filter(modelCanChat)
  const waitingModels = chatModels.filter((model) => !modelCanChat(model))
  const selectedPickerModelId = chatModels.some((model) => model.id === selectedModel?.id) ? selectedModel.id : ''
  /* The picker sat next to a top bar and message footer that both render clean
     names, while it showed the raw GGUF filename — one model wearing two names
     in the same view. formatModelLabel passes display names through untouched. */
  const modelOptionLabel = (model) => {
    const gate = getChatGateState(capabilities, model, runtime)
    const name = formatModelLabel(model.name)
    if (gate.embeddingOnly) return `${name} · Embedding only`
    if (gate.chatUnlocked) return `${name} · Ready`
    if (gate.chatMode === 'verified') return `${name} · Verified ready`
    if (gate.chatMode === 'variance') return `${name} · Runnable ready`
    if (gate.chatMode === 'experimental') return `${name} · Unverified ready`
    if (apiUnavailable) return `${name} · Not connected`
    if (gate.runtimeReady) return `${name} · Not verified`
    if (gate.runtimeLoaded) return `${name} · Not runnable`
    return `${name} · Not loaded`
  }

  /* Send-time budget check: the response limit is an upper bound the backend
     clamps to the context's remaining room, so an overshoot is a non-blocking
     notice — only a prompt that fills the whole context is a hard error. Prompt
     size is a client estimate, labeled as such. */
  const estimatedPromptTokens = useMemo(() => {
    const history = visibleMessages.map((m) => String(m.content || '')).join(' ')
    const text = `${history} ${composer}`
    const pieces = text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
    return Math.max(1, Math.round(Math.max(pieces.length, text.length / 4)))
  }, [visibleMessages, composer])
  const configuredMaxTokens = getConfiguredMaxTokens(selectedModelId)
  const effectiveMaxTokens = applyGemma4GhostChatTokenCap(
    configuredMaxTokens,
    runtime?.gemma4_serve_lane,
  )
  const ghostBudgetCapped = effectiveMaxTokens < configuredMaxTokens
  const sendBudget = validateSendBudget({
    promptTokens: estimatedPromptTokens,
    maxTokens: effectiveMaxTokens,
    contextLength: runtime?.active_context_length || modelContextLength(selectedModel),
  })

  /* Folded fine print: everything that used to stack under the composer now
     lives in the status line's tooltip. Error and budget notices still render
     their own line while active. */
  const statusDetail = [
    canChat ? 'Enter sends. Shift+Enter starts a new line.' : sendDisabledReason,
    webResearchEnabled
      ? 'Web Auto reads explicit links and searches only when needed. Triggered URLs or a prompt-derived search query leave this device for the public web.'
      : 'Web research is off; no source lookup runs for the next message.',
    ghostBudgetCapped ? `Replies from this model are capped at ${effectiveMaxTokens.toLocaleString()} tokens to keep memory usage stable.` : '',
    'Camelid runs the loaded model locally. Verify important output.',
  ].filter(Boolean).join(' ')

  const renderComposer = () => (
    <div className={`cxcomposer is-${readinessState}`}>
      {showControls && (
        <ChatControls
          capabilities={capabilities}
          modelId={selectedModelId}
          onClose={() => setShowControls(false)}
        />
      )}
      <div className="cxcomposer__box">
        {composerImage && (
          <div className="cxcomposer__image" role="status">
            <img src={composerImage.data_url} alt={`Attached ${composerImage.name}`} />
            <div className="cxcomposer__image-copy">
              <strong>{composerImage.name}</strong>
              <span>{Math.round(composerImage.size / 1024)} KB · ready for Prism vision</span>
            </div>
            <button
              type="button"
              className="cxcomposer__image-remove"
              aria-label="Remove attached image"
              onClick={() => setComposerImage(null)}
            >
              <IconClose size={16} />
            </button>
          </div>
        )}
        <textarea
          ref={composerRef}
          className="cxcomposer__input"
          aria-label="Message Camelid"
          aria-describedby={composerReadinessId}
          value={composer}
          onChange={(e) => setComposer(e.target.value)}
          onKeyDown={handleComposerKeyDown}
          rows={1}
          placeholder={composerPlaceholder}
          disabled={composerDisabled}
        />
        <div className="cxcomposer__toolbar">
          <div className="cxcomposer__tools">
            {models.length ? (
              <label className="cxcomposer__model" title="Choose what Camelid should use for this chat.">
                <span className="sr-only">Choose model for chat</span>
                <select
                  className="cxcomposer__model-select"
                  aria-label="Choose model for chat"
                  value={selectedPickerModelId}
                  onChange={(e) => {
                    const id = e.target.value
                    if (!id) return
                    // Actually switch: load the chosen model into the runtime (which
                    // also sets it as selected). Falls back to selection-only if the
                    // loader wasn't provided.
                    if (activateModel) activateModel(id)
                    else setSelectedModelId(id)
                  }}
                  disabled={requestActive || Boolean(loadingModelId)}
                >
                  {!selectedPickerModelId && <option value="">Choose chat model</option>}
                  {runnableModels.length > 0 && (
                    <optgroup label="Ready">
                      {runnableModels.map((model) => <option key={model.id} value={model.id}>{modelOptionLabel(model)}</option>)}
                    </optgroup>
                  )}
                  {waitingModels.length > 0 && (
                    <optgroup label="Needs readiness">
                      {waitingModels.map((model) => <option key={model.id} value={model.id}>{modelOptionLabel(model)}</option>)}
                    </optgroup>
                  )}
                  {embeddingModels.length > 0 && (
                    <optgroup label="Embedding only">
                      {embeddingModels.map((model) => (
                        <option key={model.id} value={`embedding:${model.id}`} disabled>{modelOptionLabel(model)}</option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </label>
            ) : (
              <button type="button" className="cxcomposer__tool" onClick={() => setTab('library')}>Add a model</button>
            )}
            {visionReady && (
              <>
                <input
                  ref={imageInputRef}
                  className="sr-only"
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleVisionFile}
                  tabIndex={-1}
                />
                <button
                  type="button"
                  className={`cxcomposer__tool cxcomposer__tool--collapsible ${composerImage ? 'is-on' : ''}`}
                  onClick={() => imageInputRef.current?.click()}
                  disabled={requestActive}
                  aria-label="Attach image"
                  title="Attach one PNG or JPEG for the loaded Prism vision model"
                >
                  <IconImage size={16} /> <span className="cxcomposer__tool-label">{composerImage ? 'Image ready' : 'Image'}</span>
                </button>
              </>
            )}
            {!demoMode && setWebResearchEnabled && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${webResearchEnabled ? 'is-on' : ''}`}
                title={webResearchEnabled
                  ? 'Web Auto is on: linked URLs or a prompt-derived query may be sent to the public web when research is needed'
                  : 'Web research is off: the next message will make no web lookup'}
                aria-label={webResearchEnabled ? 'Turn off automatic web research' : 'Turn on automatic web research'}
                aria-pressed={webResearchEnabled}
                onClick={() => setWebResearchEnabled(!webResearchEnabled)}
                disabled={requestActive}
              >
                <IconSearch size={16} />
                <span className="cxcomposer__tool-label">
                  {visibleWebResearchStatus?.phase === 'researching' ? 'Reading web…' : webResearchEnabled ? 'Web auto' : 'Web off'}
                </span>
              </button>
            )}
            {!demoMode && setReceiptMode && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${receiptMode ? 'is-on' : ''}`}
                title="Attach a verification receipt to the next reply"
                aria-label="Verification receipt"
                aria-pressed={receiptMode}
                onClick={() => setReceiptMode(!receiptMode)}
              >
                <IconReceipt size={16} /> <span className="cxcomposer__tool-label">{receiptMode ? 'Receipt on' : 'Receipt'}</span>
              </button>
            )}
            {!demoMode && setThinkingMode && !selectedBitNetChatModel && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${thinkingMode ? 'is-on' : ''}`}
                title="Show the model's reasoning before the final answer (experimental)"
                aria-label="Thinking mode"
                aria-pressed={thinkingMode}
                onClick={() => setThinkingMode(!thinkingMode)}
              >
                <IconThinking size={16} /> <span className="cxcomposer__tool-label">{thinkingMode ? 'Thinking on (experimental)' : 'Thinking'}</span>
              </button>
            )}
            {!demoMode && (
              <button
                type="button"
                className="cxcomposer__tool cxcomposer__tool--collapsible"
                onClick={secondaryAction}
                disabled={secondaryActionDisabled}
                aria-label={secondaryActionLabel}
                title={secondaryActionLabel}
              >
                <IconMemory size={16} /> <span className="cxcomposer__tool-label">{secondaryActionLabel}</span>
              </button>
            )}
            {!demoMode && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${showControls ? 'is-on' : ''}`}
                aria-expanded={showControls}
                aria-label="Generation controls"
                onClick={() => setShowControls((value) => !value)}
                title="System prompt and generation settings"
              >
                <IconBolt size={16} /> <span className="cxcomposer__tool-label">Controls</span>
              </button>
            )}
          </div>
          <div className="cxcomposer__actions">
            {generationActive && (
              <button type="button" className="cxcomposer__stop" aria-label={composerStopAriaLabel} onClick={stopGeneration} disabled={stoppingGeneration}>
                <IconStop size={16} /> {composerStopLabel}
              </button>
            )}
            <button
              type="button"
              className="cxcomposer__send"
              aria-label="Send message"
              data-send-ready={canSubmit ? 'true' : 'false'}
              title={sendBudget.level === 'error' ? sendBudget.message : !canSubmit ? sendDisabledReason : 'Send message to Camelid'}
              onClick={handleSendMessage}
              disabled={!canSubmit || sendBudget.level === 'error'}
            >
              <IconSend size={20} />
            </button>
          </div>
        </div>
      </div>

      {imageError && <p className="cxcomposer__image-error" role="alert">{imageError}</p>}

      {sendBudget.level === 'error' && (
        <p className="cxcomposer__budget-error" role="alert">
          <IconClose size={14} /> {sendBudget.message}
        </p>
      )}
      {sendBudget.level === 'notice' && (
        <p className="cxcomposer__budget-notice">
          <IconInfo size={14} /> {sendBudget.message}
        </p>
      )}
      {/* The live region wraps only the one-line status; the longer detail sits
         in an accessible Tooltip trigger beside it instead of a native title. */}
      <div id={composerReadinessId} className={`cxcomposer__status is-${statusTone}`}>
        <span className="cxcomposer__status-line" role="status" aria-live="polite">
          <StatusDot tone={statusTone} pulse={supportedChatReady || verifiedChatReady || varianceChatReady} />
          <span className="cxcomposer__status-text">{statusLine}</span>
        </span>
        {statusDetail && (
          <Tooltip content={statusDetail} placement="top">
            <button type="button" className="cxcomposer__status-info" aria-label="Chat status details">
              <IconInfo size={14} />
            </button>
          </Tooltip>
        )}
      </div>
    </div>
  )

  return (
    <section className={`cxchat is-${readinessState} ${userScrolledAway ? 'is-user-scrolled' : ''} ${isFreshThread ? 'cxchat--empty' : ''}`} data-view="chat">
      <div className="cxchat__scroll">
        <div className="cxchat__column">
          {verifiedChatReady && (
            <div className="cxchat__experimental-banner" role="note">
              <EvidenceChip state="runnable" asText>Verified</EvidenceChip>
              <span>
                This exact row passed load, deterministic output comparison, and guarded app/API
                checks. Extended-context and broader portability support remain limited.
              </span>
            </div>
          )}
          {nonSupportedChatReady && varianceChatReady && (
            <div className="cxchat__experimental-banner" role="note">
              <EvidenceChip state="runnable" asText>Runnable</EvidenceChip>
              <span>
                This exact model loads and generates normally. Some deterministic token IDs differ
                from the pinned reference, so it is runnable but not labeled Verified or Supported.
              </span>
            </div>
          )}
          {nonSupportedChatReady && unverifiedChatReady && (
            <div className="cxchat__experimental-banner" role="note">
              <EvidenceChip state="unsupported" asText>Unverified</EvidenceChip>
              <span>
                Replies from this model are <strong>not verified</strong>.{' '}
                {blockedSpecifics
                  ? `It can chat, but ${blockedSpecifics}; every reply below is marked unverified.`
                  : 'It can chat, but its output has not been checked against a reference — every reply below is marked unverified.'}
              </span>
            </div>
          )}
          {isFreshThread ? (
            <div className="cxchat__empty">
              <div className="cxchat-hero">
                <CamelidMark size={52} className="cxchat-hero__mark" />
                <h2 className="cxchat-hero__title">{productHeroTitle}</h2>
                <p className="cxchat-hero__summary">{productHeroSummary}</p>
              </div>
              {composerDraftUnlocked && (
                <div className="cxchat__suggestions" aria-label="Prompt starters">
                  {SUGGESTIONS.map(({ title, body, Icon }) => (
                    <button key={body} type="button" className="cxchat__suggestion" onClick={() => handleSuggestion(body)} disabled={!composerDraftUnlocked}>
                      <span className="cxchat__suggestion-text">{body}</span>
                      <span className="cxchat__suggestion-icon"><Icon size={18} /></span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="cxchat__thread">
              {/* Long-thread windowing (Phase 7): render the latest 60 turns;
                  earlier turns mount on demand. Keeps streaming smooth without
                  a virtualization dependency. */}
              {!showAllMessages && visibleMessages.length > 60 && (
                <button type="button" className="cxchat__show-earlier" onClick={() => setShowAllMessages(true)}>
                  Show {visibleMessages.length - 60} earlier messages
                </button>
              )}
              {(showAllMessages ? visibleMessages : visibleMessages.slice(-60)).map((message) => {
                const index = visibleMessages.indexOf(message)
                const priorUserMessage = message.role === 'assistant'
                  ? [...visibleMessages.slice(0, index)].reverse().find((item) => item.role === 'user')
                  : null
                const priorUserPrompt = priorUserMessage?.content || null
                const canResend = Boolean(resendFromMessage) && !requestActive && canChat
                const priorMessage = index > 0 ? visibleMessages[index - 1] : null
                const dayKey = dayKeyOf(message.created_at)
                const priorDayKey = priorMessage ? dayKeyOf(priorMessage.created_at) : null
                const showDaySeparator = Boolean(dayKey && priorDayKey && dayKey !== priorDayKey)
                return (
                  <Fragment key={message.id}>
                    {showDaySeparator && (
                      <div className="cxchat__day-sep" role="separator">
                        <span>{formatDayLabel(message.created_at)}</span>
                      </div>
                    )}
                    <MessageTurn
                      message={message}
                      generationElapsedSeconds={generationElapsedSeconds}
                      priorUserPrompt={priorUserPrompt}
                      onReusePrompt={setComposer}
                      onRegenerate={canResend && priorUserMessage ? () => resendFromMessage(priorUserMessage.id) : null}
                      onEditResend={canResend && message.role === 'user' ? (messageId, content) => resendFromMessage(messageId, content) : null}
                    />
                  </Fragment>
                )
              })}
              {generationActive && (
                <button
                  type="button"
                  className="cxchat__jump-latest"
                  data-autofollow-affordance
                  onClick={() => { autoFollowGenerationRef.current = true; setUserScrolledAway(false); chatBottomRef.current?.scrollIntoView({ block: 'end' }) }}
                >
                  <IconChevronDown size={12} /> Jump to latest
                </button>
              )}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="cxturn cxturn--user"><div className="cxturn__user-chip"><p>{pendingUserPrompt}</p></div></article>
                  )}
                  <article className="cxturn cxturn--assistant is-streaming" aria-busy="true" data-streaming-state="active">
                    <div className="cxturn__avatar"><Avatar size={30} state="awaiting" /></div>
                    <div className="cxturn__body"><StreamingLoader elapsedSeconds={generationElapsedSeconds} label={awaitingAssistantLabel} /></div>
                  </article>
                </>
              )}
              {/* Follow-up prompts sit under the latest reply — they act on it. */}
              {visibleMessages.length > 0 && !generationActive && canChat && (
                <div className="cxchat__followups" aria-label="Follow-up prompts">
                  {FOLLOW_UP_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" className="cxchat__followup" onClick={() => handleSuggestion(prompt)}>{prompt}</button>
                  ))}
                </div>
              )}
              <div className="cxchat__anchor" ref={chatBottomRef} aria-hidden="true" />
            </div>
          )}
        </div>
      </div>

      <div className="cxchat__dock">
        <div className="cxchat__column">
          {renderComposer()}
        </div>
      </div>
    </section>
  )
}
