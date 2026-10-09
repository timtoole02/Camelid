import { ToolActivityCard } from '../components/mcp/ToolActivityCard'
import { toolActivityGroups } from '../lib/toolActivity.js'
import { apiFetch } from '../lib/apiRequest.js'
import { ComposerMenu } from '../components/chat/ComposerMenu'
import { ConversationContext } from '../components/context/ContextEditors'
import { contextCollectionRefs, contextSourceMessages, chatHistoryForRequest, normalizeChatContext, withCollection, withoutCollection } from '../lib/projectContext.js'
import { DOCUMENT_ACCEPT, documentsCoverage, ingestLibraryFile } from '../lib/knowledgeCollections.js'
import { useKnowledgeCollections } from '../hooks/useKnowledgeCollections.js'
import { KnowledgeLibrary } from '../components/knowledge/KnowledgeLibrary'
import { ConversationFiles, ConversationFilesTray } from '../components/outputs/ConversationFiles'
import { conversationFiles } from '../lib/conversationFiles.js'
import { OutputPanelContext, OutputMessageContext, ToolOutputGallery } from '../components/outputs/OutputActions.jsx'
import { ConnectedTools, McpRunPanel } from '../components/mcp/ConnectedTools'
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { getChatGateState } from '../lib/chatGate'
import { getRuntimeRequestModelId } from '../lib/modelState'
import { detectRepeatedCall, normalizeToolCalls } from '../lib/toolCalling'
import { displayQuantLabel, exactArtifactFilenameForRow } from '../lib/capabilities'
import { formatModelLabel } from '../lib/formatters'
import { isEmbeddingOnlyModel, isGenerationCapableModel } from '../lib/modelCapabilities.js'
import { applyGemma4GhostChatTokenCap, getConfiguredMaxTokens, isBitNetB158ChatModel, modelContextLength, validateSendBudget, verifiedContextBound } from '../lib/responseLimits'
import { CamelidMark } from '../components/ui/CamelidMark'
import { Avatar } from '../components/ui/Avatar'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { IconSend, IconStop, IconMemory, IconReceipt, IconThinking, IconBolt, IconChart, IconChat, IconChevronDown, IconEdit, IconImage, IconInfo, IconClose, IconSearch, IconFile, IconCollection } from '../components/ui/icons'
import { Tooltip } from '../components/ui/Tooltip'
import { MessageTurn } from '../components/chat/MessageTurn'
import { DocumentViewer } from '../components/chat/DocumentViewer'
import { VoiceInput } from '../components/chat/VoiceInput'
import { ChatControls } from '../components/chat/ChatControls'
import { ContextMeter } from '../components/chat/ContextMeter'
import { composeContextBudget } from '../lib/contextBudget.js'
import { canContinueMessage } from '../lib/chatContinuation.js'
import { canBranchMessage } from '../lib/messageVariants.js'
import {
  AUTO_COMPACT_THRESHOLD_PERCENT,
  applySendCompaction,
  compactForSend,
  getAutoCompactEnabled,
  setAutoCompactEnabled,
  getCompactionOverride,
  setCompactionOverride,
} from '../lib/conversationCompaction.js'
import { PREPARING_STREAMING_LABEL, StreamingLoader } from '../components/chat/render/StreamingIndicator'
import { classifyWebResearchNeed, estimateWebResearchChatTokens } from '../lib/webResearch.js'
import {
  ATTACHED_DOCUMENTS_STORAGE_KEY,
  normalizeAttachedDocuments,
  readAttachedDocuments,
  writeAttachedDocuments,
} from '../lib/documentAttachments.js'
import { isGemma4Mtp12TargetVerifiedVideoOptedIn, shouldUseGemma4Mtp12TargetVerifiedRender } from '../lib/targetVerifiedRender.js'
import { isGemma4Mtp12SegmentedVideoOptedIn, readGemma4Mtp12PreparedSegments } from '../lib/segmentedWebResearchSynthesis.js'

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
const INDEX_STATUS_POLL_MS = 1500
const INDEX_PROGRESS = {
  running: { label: 'indexing', title: 'Indexing for search by meaning. Keyword search works meanwhile.' },
  waiting: { label: 'waiting to index', title: 'Waiting to be indexed for search by meaning. Keyword search works meanwhile.' },
  stopped: { label: 'indexing stopped', title: 'Indexing for search by meaning stopped. Keyword search still works.' },
}
// A whole-library search finds a passage outside what is attached only once it is indexed.
const LIBRARY_INDEX_TITLES = {
  running: 'Passages are found only once they are indexed for search by meaning.',
  waiting: 'Waiting to be indexed for search by meaning. Passages are found only once they are indexed.',
  stopped: 'Indexing for search by meaning stopped. Passages not yet indexed are not found.',
}

/** How background indexing stands while chunks are `pending`, or null when there is nothing left to index. */
function indexProgress(semantic, pending) {
  if (!semantic?.available || !pending) return null
  if (semantic.indexing) return 'running'
  return semantic.error ? 'stopped' : 'waiting'
}

const RETRIEVAL_NOTES = {
  hybrid: 'Found by keyword and by meaning',
  semantic: 'Found by meaning',
  keyword: 'Found by keyword',
  attached: 'Included because nothing else matched',
}

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
  // Track the dimensions alongside the bytes: the resize branch below replaces
  // the blob, and reporting the source dimensions for the resized bytes would
  // describe an image that was never sent.
  let width = image.naturalWidth
  let height = image.naturalHeight
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
    // Both canvasBlob calls encode this same canvas, so these describe the
    // bytes actually sent under the 0.9 and the 0.72 retry path alike.
    width = canvas.width
    height = canvas.height
  }
  if (blob.size > MAX_VISION_UPLOAD_BYTES) {
    throw new Error('The prepared image is still too large. Choose an image under 3 MB.')
  }
  return {
    name: file.name,
    type,
    size: blob.size,
    width,
    height,
    data_url: await readAsDataUrl(blob),
  }
}

export default function ChatWorkspace({
  projects = [], chatContext = {}, updateChatContext = null, contextSources = [], globalPrompt, updateGlobalPrompt,
  mcp = null, mcpSelectedKeys = [], replaceMcpTools = null, mcpActivity = null, mcpApproval = null, decideMcpApproval = null,
  selectedConversation,
  apiBase,
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
  continueFromMessage = null,
  regenerateAsVariant = null,
  selectMessageVariant = null,
  discardMessageVariant = null,
  stopGeneration,
  sending,
  receiptMode = false,
  setReceiptMode = null,
  inspectMode = false,
  setInspectMode = null,
  tokenInspections = {},
  inspectionSupported = false,
  structuredMode = 'off',
  setStructuredMode = null,
  structuredSchema = '',
  setStructuredSchema = null,
  structuredGrammar = '',
  setStructuredGrammar = null,
  structuredRecords = {},
  structuredSupported = false,
  structuredReadiness = { ready: false, reason: null },
  toolsEnabled = false,
  setToolsEnabled = null,
  toolsText = '',
  setToolsText = null,
  toolCapability = { capable: false, reason: null },
  toolsReadiness = { ready: false, reason: null },
  toolCallSignatures = {},
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
  const [filesOpen, setFilesOpen] = useState(false)
  const [selectedFileId, setSelectedFileId] = useState(null)
  const files = useMemo(() => conversationFiles(selectedConversation?.messages), [selectedConversation?.messages])
  useEffect(() => { setFilesOpen(false); setSelectedFileId(null) }, [selectedConversation?.id])
  const openOutput = (output, messageId) => {
    const found = files.find(file => file.messageId === messageId && file.mime === output.mime && file.name === output.name && file.text === output.text)
    if (!found) return false
    setSelectedFileId(found.id)
    setFilesOpen(true)
    return true
  }
  const [showControls, setShowControls] = useState(false)
  const [voiceBusy, setVoiceBusy] = useState(false)
  const [showAllMessages, setShowAllMessages] = useState(false)
  const [userScrolledAway, setUserScrolledAway] = useState(false)
  const [composerImage, setComposerImage] = useState(null)
  const [imageError, setImageError] = useState('')
  const [attachedDocuments, setAttachedDocumentsState] = useState(readAttachedDocuments)
  const [indexStatus, setIndexStatus] = useState(null)
  const [documentIngesting, setDocumentIngesting] = useState(false)
  const [documentError, setDocumentError] = useState('')
  const [activeCitation, setActiveCitation] = useState(null)
  const [citationView, setCitationView] = useState(null)
  const [viewerDocument, setViewerDocument] = useState(null)
  const openDocument = useCallback((doc) => setViewerDocument(doc), [])
  // Collections are a full-API feature; wait for health to say which surface this is.
  const knowledgeEnabled = !demoMode && Boolean(runtime) && runtime.api_surface !== 'lan_chat_only'
  const knowledge = useKnowledgeCollections(knowledgeEnabled, apiBase)
  const [library, setLibrary] = useState(null)
  const [collectionError, setCollectionError] = useState('')
  const collectionRefs = knowledgeEnabled ? contextCollectionRefs(chatContext, projects) : []
  const collectionsById = new Map((knowledge.collections || []).map((collection) => [collection.id, collection]))
  const searchedCollectionIds = new Set(collectionRefs.map((ref) => ref.id))
  const searchLibrary = knowledgeEnabled && normalizeChatContext(chatContext).search_library
  // Throws while a reply is generating; callers show the message.
  const setLibrarySearched = (searched) => updateChatContext({ ...normalizeChatContext(chatContext), search_library: searched })
  const openLibrary = (collectionId = null) => {
    knowledge.refresh()
    setLibrary({ collectionId })
  }
  // Throws while a reply is generating; callers show the message.
  const setCollectionSearched = (collectionId, searched) => updateChatContext(searched
    ? withCollection(chatContext, projects, collectionId)
    : withoutCollection(chatContext, projects, collectionId))
  const closeDocumentViewer = useCallback(() => setViewerDocument(null), [])
  const chatBottomRef = useRef(null)
  const composerRef = useRef(null)
  const imageInputRef = useRef(null)
  const docInputRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
  const composerReadinessId = 'camelid-chat-readiness-note'

  const setAttachedDocuments = (valueOrUpdater) => {
    setAttachedDocumentsState((current) => {
      const value = typeof valueOrUpdater === 'function'
        ? valueOrUpdater(current)
        : valueOrUpdater
      return writeAttachedDocuments(value)
    })
  }

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
  const connectedToolsAvailable = Boolean(mcp && !demoMode && runtime?.api_surface !== 'lan_chat_only')
  // Sending is process-global (only one local-model request may run), while
  // loaders, stop controls, and auto-follow belong only to the conversation
  // that owns the pending/streaming turn.
  const generationActive = Boolean(pendingConversation || hasStreamingAssistant)
  const followActive = generationActive || Boolean(mcpActivity && mcpActivity.phase !== 'idle')
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
  const loadProgress = runtime?.model_load_progress?.[0]
  const statusLine = loadProgress
    ? `Checking ${loadProgress.filename}: ${Math.floor(100 * loadProgress.bytes_read / Math.max(1, loadProgress.total_bytes))}% read.`
    : visibleWebResearchStatus?.phase === 'researching'
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
          ? runtime?.generation_readiness_reason || `${selectedModelName} is loaded, but Chat is unavailable. Check Models for details.`
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
          ? runtime?.generation_readiness_reason || 'This model is loaded, but Chat is unavailable. Check Models for details.'
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

  const canSubmit = Boolean(composer.trim()) && canChat && !requestActive && !voiceBusy
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

  // Document ids contain no file contents or local paths. Persist this small
  // association so an attached RAG source survives a reload, app navigation,
  // or a second browser tab. Reconcile it against the server when possible so
  // a removed document cannot leave a permanently stale pill.
  useEffect(() => {
    let cancelled = false
    if (attachedDocuments.length) {
      apiFetch('/api/documents', {}, apiBase)
        .then((response) => (response.ok ? response.json() : null))
        .then((documents) => {
          if (cancelled || !Array.isArray(documents)) return
          const availableIds = new Set(documents.map((document) => document.id))
          setAttachedDocumentsState((current) => {
            const next = current.filter((document) => availableIds.has(document.doc_id))
            return next.length === current.length ? current : writeAttachedDocuments(next)
          })
        })
        .catch(() => {})
    }
    const handleStorage = (event) => {
      if (event.key === ATTACHED_DOCUMENTS_STORAGE_KEY) {
        setAttachedDocumentsState(readAttachedDocuments())
      }
    }
    window.addEventListener('storage', handleStorage)
    return () => {
      cancelled = true
      window.removeEventListener('storage', handleStorage)
    }
    // Reconcile against each selected backend; later edits already come from
    // successful ingest/remove actions in this component.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase])

  // Semantic indexing runs in the background after an upload. Poll only while
  // an attached document is still being indexed; keyword search needs none of it.
  const watchedDocIdsKey = [...new Set([
    ...attachedDocuments.map((doc) => doc.doc_id),
    ...collectionRefs.flatMap((ref) => collectionsById.get(ref.id)?.doc_ids || []),
  ])].join('\n')
  useEffect(() => {
    if (!watchedDocIdsKey && !searchLibrary) {
      setIndexStatus(null)
      return undefined
    }
    const ids = watchedDocIdsKey ? watchedDocIdsKey.split('\n') : []
    setIndexStatus(null)
    let cancelled = false
    let timer = null
    const poll = async () => {
      try {
        const res = await apiFetch('/api/documents/index-status', {}, apiBase)
        const status = res.ok ? await res.json() : null
        if (cancelled || !status?.semantic) return
        const byId = Object.fromEntries((status.documents || []).map((doc) => [doc.id, doc]))
        setIndexStatus({ semantic: status.semantic, byId })
        // A stopped indexer restarts only for a new upload or a restart, so polling it changes nothing.
        const watched = searchLibrary ? Object.keys(byId) : ids
        if (['running', 'waiting'].includes(indexProgress(status.semantic, documentsCoverage(watched, byId).pending))) {
          timer = window.setTimeout(poll, INDEX_STATUS_POLL_MS)
        }
      } catch {
        // Status is informational; search keeps working without it.
      }
    }
    poll()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [watchedDocIdsKey, searchLibrary, apiBase])

  useEffect(() => {
    if (!followActive) return undefined
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
  }, [followActive, selectedConversation?.id])

  useLayoutEffect(() => {
    if (!followActive || !autoFollowGenerationRef.current) return undefined
    const frame = window.requestAnimationFrame(() => {
      chatBottomRef.current?.scrollIntoView({ block: 'end', behavior: 'auto' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [followActive, streamingScrollSignature, mcpActivity?.phase, mcpApproval?.id])

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

  useEffect(() => {
    const handleCitationClick = (e) => {
      const cite = e.detail?.citation
      if (cite) {
        setActiveCitation(cite)
      }
    }
    window.addEventListener('camelid-citation-click', handleCitationClick)
    return () => window.removeEventListener('camelid-citation-click', handleCitationClick)
  }, [])

  // A citation is shown only after the server re-derives it from the stored
  // source and every hash still matches. Failures are refused, never rendered.
  useEffect(() => {
    if (!activeCitation) {
      setCitationView(null)
      return undefined
    }

    const docId = activeCitation.doc_id
    const chunkIndex = activeCitation.chunk_index
    if (!docId || chunkIndex === null || chunkIndex === undefined) {
      setCitationView({
        status: 'refused',
        code: 'citation_unverifiable',
        message: 'This citation carries no source binding, so it cannot be verified.',
      })
      return undefined
    }

    let cancelled = false
    setCitationView({ status: 'verifying' })

    const verify = async () => {
      try {
        const res = await apiFetch('/api/documents/citation/resolve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            doc_id: docId,
            chunk_index: chunkIndex,
            chunk_sha256: activeCitation.chunk_sha256 || null,
            doc_sha256: activeCitation.doc_sha256 || null,
          }),
        }, apiBase)
        const payload = await res.json().catch(() => null)
        if (cancelled) return
        if (res.ok && payload) {
          setCitationView({ status: 'verified', data: payload })
        } else {
          setCitationView({
            status: 'refused',
            code: payload?.error?.code || 'citation_refused',
            message:
              payload?.error?.message ||
              'This citation could not be verified against its source.',
          })
        }
      } catch {
        if (cancelled) return
        setCitationView({
          status: 'refused',
          code: 'citation_unreachable',
          message: 'The source could not be reached to verify this citation.',
        })
      }
    }

    verify()
    return () => {
      cancelled = true
    }
  }, [activeCitation, apiBase])

  const handleDocumentFiles = async (files) => {
    if (!files || !files.length) return
    setDocumentError('')
    setDocumentIngesting(true)
    for (const file of Array.from(files)) {
      try {
        const doc = await ingestLibraryFile(file, [], file.name, apiBase)
        setAttachedDocuments((prev) => [
          ...prev.filter((item) => item.doc_id !== doc.doc_id && item.filename !== doc.filename),
          doc,
        ])
      } catch (err) {
        console.error('Failed to ingest document:', file.name, err)
        setDocumentError(err?.message || `Could not index ${file.name}.`)
      }
    }
    setDocumentIngesting(false)
  }

  const handleSendMessage = async () => {
    if (voiceBusy) return
    const image = composerImage
    setComposerImage(null)
    setImageError('')

    let contentToSend = composer
    let requestCitations = []
    setCollectionError('')
    // Re-read collections so a collection deleted elsewhere is skipped, not a failed search.
    let searchedCollections = []
    if (collectionRefs.length > 0) {
      const current = await knowledge.refresh()
      if (current) {
        const live = new Map(current.map((collection) => [collection.id, collection]))
        searchedCollections = collectionRefs.map((ref) => live.get(ref.id)).filter(Boolean)
      } else {
        setCollectionError('Collections could not be read, so this message was sent without them.')
      }
    }
    const pinnedSources = attachedDocuments.length > 0 || searchedCollections.length > 0
    let librarySearched = false
    if (pinnedSources || searchLibrary) {
      const searchBody = {
        query: composer,
        ...(attachedDocuments.length > 0 ? { doc_ids: attachedDocuments.map((d) => d.doc_id) } : {}),
        ...(searchedCollections.length > 0 ? { collection_ids: searchedCollections.map((c) => c.id) } : {}),
        top_k: 4,
      }
      const search = (body) => apiFetch('/api/documents/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }, apiBase)
      try {
        let res = null
        if (searchLibrary) {
          // A request that never answers is a failure too, so attached sources are still searched.
          res = await search({ ...searchBody, library: true }).catch((err) => {
            console.error('Whole-library search error:', err)
            return null
          })
          librarySearched = Boolean(res?.ok)
          if (!res?.ok) {
            const failure = res ? await res.json().catch(() => null) : null
            setCollectionError(`${failure?.error?.message || 'Whole-library search failed.'} This message was sent without searching the whole library.`)
            res = null
          }
        }
        if (!res && pinnedSources) {
          res = await search(searchBody)
          if (!res.ok && searchedCollections.length > 0) {
            const failure = await res.json().catch(() => null)
            setCollectionError(failure?.error?.message || 'Document search failed, so this message was sent without document context.')
          }
        }
        if (res?.ok) {
          const data = await res.json()
          if (data.results && data.results.length > 0) {
            const citations = data.results
            requestCitations = citations

            const contextText = citations
              .map((c, idx) => `[Citation ${idx + 1} from ${c.filename}]:\n${c.excerpt}`)
              .join('\n\n')

            contentToSend = `Refer to the following retrieved document excerpts to answer the prompt. Cite your sources inline using [1], [2], etc.\n\n--- DOCUMENT CONTEXT ---\n${contextText}\n--- END CONTEXT ---\n\nUser Question: ${composer}`
          }
        }
      } catch (err) {
        console.error('Document search error:', err)
      }
    }

    await sendMessage({
      overrideImage: image,
      requestContent: contentToSend !== composer ? contentToSend : null,
      citations: requestCitations,
      documents: attachedDocuments.map((doc) => ({
        ...doc,
        passages: requestCitations.filter((citation) => citation.doc_id === doc.doc_id).length,
      })),
      collections: searchedCollections.map((collection) => ({
        id: collection.id,
        name: collection.name,
        passages: requestCitations.filter((citation) => collection.doc_ids.includes(citation.doc_id)).length,
      })),
      // Counts only what the library added beyond the attached documents and collections.
      library: librarySearched ? {
        passages: requestCitations.filter((citation) => !attachedDocuments.some((doc) => doc.doc_id === citation.doc_id)
          && !searchedCollections.some((collection) => collection.doc_ids.includes(citation.doc_id))).length,
      } : null,
    })
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
  const previewMessages = [...contextSourceMessages(contextSources), ...chatHistoryForRequest([
    ...visibleMessages.filter(message => !message.streaming),
    ...(composer.trim() ? [{ id: 'context-preview-draft', role: 'user', content: composer.trim(), ...(composerImage ? { image: composerImage } : {}) }] : []),
  ])]
  const estimatePrompt = messages => estimateWebResearchChatTokens(messages, { visionTokenAllowance: runtime?.vision_token_allowance })
  const untrimmedPromptTokens = estimatePrompt(previewMessages)
  const configuredMaxTokens = getConfiguredMaxTokens(selectedModelId)
  const effectiveMaxTokens = applyGemma4GhostChatTokenCap(
    configuredMaxTokens,
    runtime?.gemma4_serve_lane,
  )
  const ghostBudgetCapped = effectiveMaxTokens < configuredMaxTokens
  const activeContextLength = runtime?.active_context_length || modelContextLength(selectedModel)
  /* The meter reads the same three numbers the budget check does, so the chip
     and the notice under the composer can never disagree. The verified bound is
     drawn as a marker rather than a limit: past it the row is still served, it
     simply has no committed evidence pack. */
  const verifiedBound = verifiedContextBound(capabilities, selectedModel)
  const executionLane = runtime?.execution_plan?.selected_backend || ''

  /* Compaction preview. The panel must describe what the NEXT send will do, so
     it runs the same pure trim the send path runs, over the same preference
     store -- there is no second copy of the rule to drift. */
  const conversationId = selectedConversation?.id || ''
  const [autoCompact, setAutoCompactState] = useState(() => getAutoCompactEnabled())
  const [compactionOverride, setCompactionOverrideState] = useState(null)
  useEffect(() => {
    setCompactionOverrideState(getCompactionOverride(conversationId))
  }, [conversationId])

  const contextBudget = composeContextBudget({
    contextLength: activeContextLength,
    promptTokens: untrimmedPromptTokens,
    reservedTokens: effectiveMaxTokens,
    verifiedBound,
    warnAtPercent: AUTO_COMPACT_THRESHOLD_PERCENT,
  })
  const compactionPreview = applySendCompaction(previewMessages, {
    enabled: compactionOverride === 'off' ? false : autoCompact,
    forced: compactionOverride === 'force',
    filledPercent: contextBudget?.filledPercent ?? 0,
  })
  const estimatedPromptTokens = estimatePrompt(compactionPreview.messages)
  const systemTokens = estimatePrompt(compactionPreview.messages.filter(message => message.role === 'system'))
  const elidedTokenEstimate = Math.max(0, untrimmedPromptTokens - estimatedPromptTokens)

  const rawSendBudget = validateSendBudget({
    promptTokens: estimatedPromptTokens,
    maxTokens: effectiveMaxTokens,
    contextLength: activeContextLength,
  })
  const segmentedVideoComposerBypass = isGemma4Mtp12SegmentedVideoOptedIn()
    && Boolean(readGemma4Mtp12PreparedSegments())
    && shouldUseGemma4Mtp12TargetVerifiedRender({
      runtime,
      requestModelId: runtime?.active_model_id,
      compatibilityRowId: selectedChatGate.hint?.target?.id,
      research: { sources: [{}, {}] },
      receiptMode,
      videoRigOptIn: isGemma4Mtp12TargetVerifiedVideoOptedIn(),
    })
  // The private segmented lane verifies six independently bounded prompts;
  // the long product brief itself is Web Auto input, not a 512-position model
  // prompt. Keep the ordinary composer fail-closed everywhere else.
  const sendBudget = segmentedVideoComposerBypass && rawSendBudget.level === 'error'
    ? { ...rawSendBudget, level: 'ok', message: null }
    : rawSendBudget

  const handleToggleAutoCompact = (next) => {
    setAutoCompactEnabled(next)
    setAutoCompactState(next)
    /* Changing the preference clears a per-chat override, otherwise the
       checkbox would appear to do nothing in this conversation. */
    setCompactionOverride(conversationId, null)
    setCompactionOverrideState(null)
  }
  const handleCompactNow = () => {
    setCompactionOverride(conversationId, 'force')
    setCompactionOverrideState('force')
  }
  const handleSendEverything = () => {
    setCompactionOverride(conversationId, 'off')
    setCompactionOverrideState('off')
  }

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

  const renderConversationContext = compact => (updateChatContext && <ConversationContext compact={compact} key={selectedConversation?.id || 'draft'} context={chatContext} projects={projects} sources={contextSources} globalPrompt={globalPrompt || ''} onSave={updateChatContext} onManageProjects={() => setTab('projects')} busy={sending}
    collections={knowledgeEnabled && !knowledge.error ? knowledge.collections : undefined} />)

  const toolActivity = useMemo(() => toolActivityGroups(visibleMessages), [visibleMessages])
  const hasInlineActivity = Boolean(mcpActivity?.messageId && toolActivity.groups.has(mcpActivity.messageId))

  const renderComposer = () => (
    <div className={`cxcomposer is-${readinessState}`}>
      {renderConversationContext(false)}
      {showControls && (
        <ChatControls
          capabilities={capabilities}
          globalPrompt={globalPrompt} onGlobalPromptChange={updateGlobalPrompt} busy={sending}
          modelId={getRuntimeRequestModelId(selectedModel, runtime, selectedModelId)}
          onClose={() => setShowControls(false)}
        />
      )}
      {!connectedToolsAvailable && toolCapability.capable && toolsEnabled && setToolsText && !mcpSelectedKeys.length && (
        <div className="tooldef">
          <textarea
            className="tooldef__field"
            aria-label="Tool definitions"
            spellCheck={false}
            value={toolsText}
            onChange={(event) => setToolsText(event.target.value)}
          />
          <p className={`tooldef__status ${toolsReadiness.ready ? '' : 'is-invalid'}`}>
            {toolsReadiness.ready
              ? 'Offered to the model on the next turn. Camelid does not execute tool calls — a request comes back for you to answer.'
              : toolsReadiness.reason}
          </p>
        </div>
      )}
      <div
        className="cxcomposer__box"
        onDragOver={(e) => {
          e.preventDefault()
          e.stopPropagation()
        }}
        onDrop={(e) => {
          e.preventDefault()
          e.stopPropagation()
          if (e.dataTransfer.files?.length) {
            handleDocumentFiles(e.dataTransfer.files)
          }
        }}
      >
        {(attachedDocuments.length > 0 || documentIngesting || collectionRefs.length > 0 || searchLibrary) && (
          <div className="cxcomposer__docs">
            {searchLibrary && (() => {
              const libraryIds = indexStatus ? Object.keys(indexStatus.byId) : null
              const coverage = libraryIds ? documentsCoverage(libraryIds, indexStatus.byId) : null
              const progress = indexProgress(indexStatus?.semantic, coverage?.pending)
              return (
                <div className="cxcomposer__doc-pill cxcomposer__doc-pill--library">
                  <button
                    type="button"
                    className="cxcomposer__doc-remove"
                    aria-label="Stop searching the whole library"
                    title="Stop searching the whole library in this chat"
                    disabled={requestActive || !updateChatContext}
                    onClick={() => {
                      try { setLibrarySearched(false) } catch (failure) { setCollectionError(failure.message) }
                    }}
                  >
                    <IconClose size={14} />
                  </button>
                  <button type="button" className="cxcomposer__doc-open" title="Every document in the library is searched; only passages close in meaning to your message are used. Open the knowledge library." onClick={() => openLibrary()}>
                    <IconSearch size={14} />
                    <span className="cxcomposer__doc-name">Whole library</span>
                    {progress ? (
                      <span
                        className={`cxcomposer__doc-chunks cxcomposer__doc-chunks--${progress === 'stopped' ? 'stopped' : 'indexing'}`}
                        title={LIBRARY_INDEX_TITLES[progress]}
                      >
                        {INDEX_PROGRESS[progress].label} {coverage.done}/{coverage.indexable}
                      </span>
                    ) : libraryIds && (
                      <span className="cxcomposer__doc-chunks">{libraryIds.length} {libraryIds.length === 1 ? 'doc' : 'docs'}</span>
                    )}
                  </button>
                </div>
              )
            })()}
            {knowledge.collections !== null && collectionRefs.map((ref) => {
              const collection = collectionsById.get(ref.id)
              const label = collection?.name || 'Collection unavailable'
              const coverage = collection ? documentsCoverage(collection.doc_ids, indexStatus?.byId) : null
              const progress = indexProgress(indexStatus?.semantic, coverage?.pending)
              return (
                <div key={ref.id} className={`cxcomposer__doc-pill cxcomposer__doc-pill--collection${collection ? '' : ' is-unavailable'}`}>
                  <button
                    type="button"
                    className="cxcomposer__doc-remove"
                    aria-label={`Stop searching ${label}`}
                    title={ref.from === 'project' ? 'Stop searching this project collection in this chat' : 'Stop searching this collection in this chat'}
                    disabled={requestActive || !updateChatContext}
                    onClick={() => {
                      try { setCollectionSearched(ref.id, false) } catch (failure) { setCollectionError(failure.message) }
                    }}
                  >
                    <IconClose size={14} />
                  </button>
                  <button type="button" className="cxcomposer__doc-open" disabled={!collection} title={collection ? `Open ${collection.name} in the knowledge library` : 'This collection was deleted. Remove it from this chat.'} onClick={() => openLibrary(ref.id)}>
                    <IconCollection size={14} />
                    <span className="cxcomposer__doc-name">{label}</span>
                    {collection && (progress ? (
                      <span
                        className={`cxcomposer__doc-chunks cxcomposer__doc-chunks--${progress === 'stopped' ? 'stopped' : 'indexing'}`}
                        title={INDEX_PROGRESS[progress].title}
                      >
                        {INDEX_PROGRESS[progress].label} {coverage.done}/{coverage.indexable}
                      </span>
                    ) : (
                      <span className="cxcomposer__doc-chunks">{collection.doc_ids.length} {collection.doc_ids.length === 1 ? 'doc' : 'docs'}{ref.from === 'project' ? ' · project' : ''}</span>
                    ))}
                  </button>
                </div>
              )
            })}
            {attachedDocuments.map((doc) => {
              const coverage = indexStatus?.byId[doc.doc_id]
              const progress = indexProgress(indexStatus?.semantic, documentsCoverage([doc.doc_id], indexStatus?.byId).pending)
              return (
                <div key={doc.doc_id} className="cxcomposer__doc-pill">
                  <button
                    type="button"
                    className="cxcomposer__doc-remove"
                    aria-label={`Remove ${doc.filename}`}
                    title="Remove attachment"
                    onClick={() => {
                      setAttachedDocuments((prev) => prev.filter((d) => d.doc_id !== doc.doc_id))
                      composerRef.current?.focus()
                    }}
                  >
                    <IconClose size={14} />
                  </button>
                  <button type="button" className="cxcomposer__doc-open" title={`Open ${doc.filename}`} onClick={() => openDocument(doc)}>
                    <IconFile size={14} />
                    <span className="cxcomposer__doc-name">{doc.filename}</span>
                    {progress ? (
                      <span
                        className={`cxcomposer__doc-chunks cxcomposer__doc-chunks--${progress === 'stopped' ? 'stopped' : 'indexing'}`}
                        title={INDEX_PROGRESS[progress].title}
                      >
                        {INDEX_PROGRESS[progress].label} {coverage.indexed_chunks}/{coverage.indexable_chunks}
                      </span>
                    ) : (
                      <span className="cxcomposer__doc-chunks">{doc.chunk_count} chunks</span>
                    )}
                  </button>
                </div>
              )
            })}
            {documentIngesting && <span className="cxcomposer__doc-status">Indexing document…</span>}
            {searchLibrary && indexStatus?.semantic && !indexStatus.semantic.available && (
              <p className="cxcomposer__semantic-note cxcomposer__semantic-note--library" role="status">
                Whole-library search needs search by meaning. {indexStatus.semantic.message}
                {indexStatus.semantic.reason === 'encoder_not_installed' && (
                  <button type="button" onClick={() => setTab('library')}>Open Models</button>
                )}
              </p>
            )}
            {(searchLibrary || watchedDocIdsKey) && indexProgress(indexStatus?.semantic, documentsCoverage(searchLibrary ? Object.keys(indexStatus?.byId || {}) : watchedDocIdsKey.split('\n'), indexStatus?.byId).pending) === 'stopped' && (
              <p className="cxcomposer__semantic-note cxcomposer__semantic-note--stopped" role="status">
                <span>Indexing for search by meaning stopped:</span>{' '}
                <span className="cxcomposer__semantic-error">{indexStatus.semantic.error}</span>{' '}
                <span>
                  {searchLibrary
                    ? 'Until it runs again, when a document is added or Camelid restarts, passages not yet indexed are found only by keyword in attached documents and collections, and not at all elsewhere in the library.'
                    : 'Passages not yet indexed are found by keyword only until it runs again, when a document is added or Camelid restarts.'}
                </span>
              </p>
            )}
            {(attachedDocuments.length > 0 || collectionRefs.length > 0) && ['encoder_not_installed', 'encoder_mismatch', 'encoder_load_failed'].includes(indexStatus?.semantic?.reason) && (
              <p className="cxcomposer__semantic-note" role="status">
                Keyword search only. {indexStatus.semantic.message}
                {indexStatus.semantic.reason === 'encoder_not_installed' && (
                  <button type="button" onClick={() => setTab('library')}>Open Models</button>
                )}
              </p>
            )}
          </div>
        )}
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
        <div className="cxcomposer__option-chips" aria-label="Active message options">
          {[
            [thinkingMode && !selectedBitNetChatModel, 'Thinking', () => setThinkingMode?.(false)],
            [webResearchEnabled, 'Web auto', () => setWebResearchEnabled?.(false)],
            [receiptMode, 'Receipt', () => setReceiptMode?.(false)],
            [structuredSupported && structuredMode !== 'off', structuredMode === 'grammar' ? 'Grammar' : 'JSON output', () => setStructuredMode?.('off')],
            [inspectionSupported && inspectMode, 'Token probabilities', () => setInspectMode?.(false)],
          ].filter(([active]) => active).map(([, label, remove]) => <button key={label} type="button" disabled={requestActive} aria-label={'Remove ' + label} onClick={remove}>{label}<IconClose size={11} /></button>)}
        </div>
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
            {connectedToolsAvailable && <ConnectedTools key={'mcp-' + (selectedConversation?.id || 'draft')} connections={mcp.connections} selectedKeys={mcpSelectedKeys}
              onSelectionChange={replaceMcpTools} onManage={() => setTab('connections')} disabled={requestActive} capability={toolCapability}
              connectionBusy={mcp.busy} error={mcp.error} onRetry={() => mcp.refresh()}
              onConnect={id => mcp.mutate('/connections/' + id + '/connect', { method: 'POST' })}
              manualEnabled={toolsEnabled} onManualEnabledChange={setToolsEnabled} manualText={toolsText} onManualTextChange={setToolsText}
              manualReadiness={toolsReadiness} structuredMode={structuredMode} />}
            {renderConversationContext(true)}
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
              </>
            )}
            <input
              ref={docInputRef}
              className="sr-only"
              type="file"
              accept={DOCUMENT_ACCEPT}
              multiple
              onChange={(e) => {
                handleDocumentFiles(e.target.files)
                e.target.value = ''
              }}
              tabIndex={-1}
            />
            <ComposerMenu label="Attach" description={visionReady ? "Attach one PNG or JPEG for the loaded Prism vision model, or add documents for retrieval" : "Attach documents for local retrieval"} icon={<IconFile size={16} />} disabled={requestActive}>
              {close => <div className="composer-menu__attachments">
            <button
              type="button"
              className={`cxcomposer__tool cxcomposer__tool--collapsible ${attachedDocuments.length > 0 ? 'is-on' : ''}`}
              onClick={() => { close(); docInputRef.current?.click() }}
              disabled={requestActive || documentIngesting}
              aria-label="Attach documents for RAG"
              title="Drag & drop or attach PDF, Word, HTML, Markdown, text or source code documents for local RAG"
            >
              <IconFile size={16} />{' '}
              <span className="cxcomposer__tool-label">
                {documentIngesting ? 'Indexing…' : attachedDocuments.length > 0 ? `Docs (${attachedDocuments.length})` : 'Documents'}
              </span>
            </button>
                {knowledgeEnabled && (
                  <button
                    type="button"
                    className={`cxcomposer__tool cxcomposer__tool--collapsible ${collectionRefs.length > 0 ? 'is-on' : ''}`}
                    onClick={() => { close(); openLibrary() }}
                    aria-label="Knowledge collections"
                    title="Search a collection of library documents in this chat"
                  >
                    <IconCollection size={16} />{' '}
                    <span className="cxcomposer__tool-label">{collectionRefs.length > 0 ? `Collections (${collectionRefs.length})` : 'Collections'}</span>
                  </button>
                )}
                {knowledgeEnabled && (
                  <button
                    type="button"
                    className={`cxcomposer__tool cxcomposer__tool--collapsible ${searchLibrary ? 'is-on' : ''}`}
                    onClick={() => {
                      close()
                      try { setLibrarySearched(!searchLibrary) } catch (failure) { setCollectionError(failure.message) }
                    }}
                    disabled={!updateChatContext}
                    aria-label="Search the whole library"
                    aria-pressed={searchLibrary}
                    title="Search every document in the library, using only passages close in meaning to your message"
                  >
                    <IconSearch size={16} />{' '}
                    <span className="cxcomposer__tool-label">Whole library</span>
                  </button>
                )}
                {visionReady && <>                <button
                  type="button"
                  className={`cxcomposer__tool cxcomposer__tool--collapsible ${composerImage ? 'is-on' : ''}`}
                  onClick={() => { close(); imageInputRef.current?.click() }}
                  disabled={requestActive}
                  aria-label="Attach image"
                  title="Attach one PNG or JPEG for the loaded Prism vision model"
                >
                  <IconImage size={16} /> <span className="cxcomposer__tool-label">{composerImage ? 'Image ready' : 'Image'}</span>
                </button>
</>}
                <p>Documents: PDF, Word, Markdown, text, CSV or JSON. Collections search a group of library documents. Whole library searches every document and uses only passages close in meaning to your message. Images are available with a vision model.</p>
              </div>}
            </ComposerMenu>
            {/* Guarded on BOTH halves of the gate: the engine must advertise the
                protocol and the LOADED MODEL must carry a tool receipt. The second
                half is STRICTER than the engine — POST /v1/chat/completions gates
                on the chat template and never reads tool_capable — so the copy
                says Camelid declines, not that the engine refuses. */}
            {!connectedToolsAvailable && !demoMode && setToolsEnabled && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${toolsEnabled && toolCapability.capable ? 'is-on' : ''}`}
                title={toolCapability.capable
                  ? 'Offer tools to the model on the next turn'
                  : toolCapability.reason || 'This model is not tool-capable.'}
                aria-label={toolCapability.capable ? 'Tools' : 'Tools — unavailable for this model'}
                aria-pressed={toolCapability.capable ? toolsEnabled : undefined}
                disabled={!toolCapability.capable}
                onClick={() => setToolsEnabled(!toolsEnabled)}
              >
                <IconBolt size={16} />
                <span className="cxcomposer__tool-label">
                  {!toolCapability.capable ? 'Tools unavailable' : toolsEnabled ? 'Tools on' : 'Tools'}
                </span>
              </button>
            )}
            {!demoMode && <ComposerMenu label="Options" icon={<IconBolt size={16} />}>
              {close => <>
                <div className="composer-menu__choices">
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
                disabled={requestActive}
                onClick={() => setReceiptMode(!receiptMode)}
              >
                <IconReceipt size={16} /> <span className="cxcomposer__tool-label">{receiptMode ? 'Receipt on' : 'Receipt'}</span>
              </button>
            )}
            {/* Constrained decoding is a pre-send choice: the engine refuses a
                constraint on a streaming request, and its streaming decoder never
                builds a grammar state at all, so the turn must be composed
                non-streaming before it is sent. Guarded when the contract does not
                advertise it, rather than live-with-a-disclaimer. */}
            {!demoMode && setStructuredMode && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${structuredMode !== 'off' && structuredSupported ? 'is-on' : ''}`}
                title={structuredSupported
                  ? 'Constrain the next reply to a JSON schema or grammar (sends it without streaming)'
                  : 'This engine does not advertise constrained decoding.'}
                aria-label={structuredSupported ? 'Structured output' : 'Structured output — unavailable on this engine'}
                aria-pressed={structuredSupported ? structuredMode !== 'off' : undefined}
                disabled={requestActive || !structuredSupported}
                onClick={() => setStructuredMode(structuredMode === 'off' ? 'json_schema' : 'off')}
              >
                <IconFile size={16} />
                <span className="cxcomposer__tool-label">
                  {!structuredSupported ? 'Schema unavailable' : structuredMode === 'off' ? 'Schema' : 'Schema on'}
                </span>
              </button>
            )}
            {/* Token inspection is a pre-send choice because the scores are
                CAPTURED during the reply's own decode. Inspecting afterwards would
                mean decoding a second time, and those numbers would describe that
                other generation — on a sampled row, a different reply entirely. */}
            {/* Guarded, not hidden, when the contract does not advertise
                inspection: a live-looking toggle that records nothing is the
                caveated-live surface I3 rules out, and hiding it entirely would
                leave no explanation for why the feature is absent. The accessible
                name contains the visible label so voice control can address it. */}
            {!demoMode && setInspectMode && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${inspectMode && inspectionSupported ? 'is-on' : ''}`}
                title={inspectionSupported
                  ? "Record the model's per-token scores for the next reply (sends it without streaming)"
                  : 'This engine does not advertise per-token probability reporting, so the next reply cannot record it.'}
                aria-label={inspectionSupported ? 'Tokens — record per-token probabilities' : 'Tokens — unavailable on this engine'}
                aria-pressed={inspectionSupported ? inspectMode : undefined}
                disabled={requestActive || !inspectionSupported}
                onClick={() => setInspectMode(!inspectMode)}
              >
                <IconChart size={16} />
                <span className="cxcomposer__tool-label">
                  {inspectionSupported ? (inspectMode ? 'Tokens on' : 'Tokens') : 'Tokens unavailable'}
                </span>
              </button>
            )}
            {!demoMode && setThinkingMode && !selectedBitNetChatModel && (
              <button
                type="button"
                className={`cxcomposer__tool cxcomposer__tool--collapsible ${thinkingMode ? 'is-on' : ''}`}
                title="Show the model's reasoning before the final answer (experimental)"
                aria-label="Thinking mode"
                aria-pressed={thinkingMode}
                disabled={requestActive}
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
                onClick={() => { close(); setShowControls((value) => !value) }}
                title="System prompt and generation settings"
              >
                <IconBolt size={16} /> <span className="cxcomposer__tool-label">Controls</span>
              </button>
            )}
                  <button type="button" className="cxcomposer__tool composer-menu__files" onClick={() => { close(); setFilesOpen(true) }}><IconFile size={16} />Conversation files ({files.length})</button>
                </div>
                <fieldset className="composer-menu__format" disabled={requestActive}>
      {structuredSupported && structuredMode !== 'off' && setStructuredMode && (
        <div className="structout-editor">
          <div className="structout-editor__modes" role="group" aria-label="Constraint form">
            {[
              ['json_schema', 'JSON schema'],
              ['json_object', 'Any JSON'],
              ['grammar', 'Grammar'],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={`structout-editor__mode ${structuredMode === value ? 'is-on' : ''}`}
                aria-pressed={structuredMode === value}
                onClick={() => setStructuredMode(value)}
              >
                {label}
              </button>
            ))}
          </div>
          {structuredMode === 'json_schema' && (
            <textarea
              className="structout-editor__field"
              aria-label="JSON schema"
              spellCheck={false}
              value={structuredSchema}
              onChange={(event) => setStructuredSchema?.(event.target.value)}
            />
          )}
          {structuredMode === 'grammar' && (
            <textarea
              className="structout-editor__field"
              aria-label="Grammar"
              spellCheck={false}
              value={structuredGrammar}
              onChange={(event) => setStructuredGrammar?.(event.target.value)}
            />
          )}
          <p className={`structout-editor__status ${structuredReadiness.ready ? '' : 'is-invalid'}`}>
            {structuredReadiness.ready
              ? 'The next reply is constrained to this. Turn on Tokens as well to see whether the constraint actually diverted the decode.'
              : structuredReadiness.reason}
          </p>
        </div>
      )}
                </fieldset>
              </>}
            </ComposerMenu>}
          </div>
          <div className="cxcomposer__actions">
            {!demoMode && <VoiceInput
              key={`${selectedConversation?.id || "new"}:${apiBase || ""}`}
              apiBase={apiBase}
              disabled={requestActive || apiUnavailable}
              onBusyChange={setVoiceBusy}
              onTranscript={(text) => {
                setComposer((draft) => draft ? `${draft}${/\s$/.test(draft) ? "" : " "}${text}` : text)
                composerRef.current?.focus()
              }}
            />}
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
      {documentError && <p className="cxcomposer__image-error" role="alert">{documentError}</p>}
      {collectionError && <p className="cxcomposer__image-error" role="alert">{collectionError}</p>}

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
        <ContextMeter
          contextLength={activeContextLength}
          promptTokens={estimatedPromptTokens}
          systemTokens={systemTokens}
          reservedTokens={effectiveMaxTokens}
          verifiedBound={verifiedBound}
          executionLane={executionLane}
          autoCompact={autoCompact}
          onToggleAutoCompact={handleToggleAutoCompact}
          onCompactNow={compactionPreview.compacted ? null : handleCompactNow}
          canCompact={compactForSend(previewMessages) !== null}
          compaction={compactionPreview.compacted
            ? {
              active: true,
              elidedCount: compactionPreview.elidedCount,
              freedTokens: elidedTokenEstimate,
            }
            : null}
          onSendEverything={compactionPreview.compacted ? handleSendEverything : null}
        />

      </div>
    </div>
  )

  return (
    <OutputPanelContext.Provider value={openOutput}>
    <div className={'chat-workspace-layout' + (filesOpen ? ' has-files' : '')}>
    <section className={`cxchat is-${readinessState} ${userScrolledAway ? 'is-user-scrolled' : ''} ${isFreshThread ? 'cxchat--empty' : ''}`} data-view="chat">
      <div className="cxchat__utility"><button type="button" aria-label="Conversation files" aria-expanded={filesOpen} onClick={() => setFilesOpen(!filesOpen)}><IconFile size={15} />Files{files.length > 0 && <span>{files.length}</span>}</button></div>
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
                /* Continue is offered on the LAST reply only. Resuming a reply
                   from the middle of a thread would have to discard every turn
                   after it, which is what Edit & resend already does and says. */
                const isLastMessage = index === visibleMessages.length - 1
                const canContinue = Boolean(continueFromMessage)
                  && !requestActive
                  && canChat
                  && isLastMessage
                  && canContinueMessage(message)
                /* Re-rolling the LAST reply keeps the old one as a sibling --
                   nothing after it can go stale, because nothing is after it.
                   Mid-thread it stays the old resend, which does discard the
                   turns below and now says so. */
                const canBranchHere = Boolean(regenerateAsVariant)
                  && !requestActive
                  && canChat
                  && isLastMessage
                  && canBranchMessage(message)
                const priorMessage = index > 0 ? visibleMessages[index - 1] : null
                const dayKey = dayKeyOf(message.created_at)
                const priorDayKey = priorMessage ? dayKeyOf(priorMessage.created_at) : null
                const showDaySeparator = Boolean(dayKey && priorDayKey && dayKey !== priorDayKey)
                if (message.role === 'tool' && toolActivity.pairedResults.has(message.id)) return null
                if (message.role === 'tool') return <OutputMessageContext.Provider key={message.id} value={message.id}><details className="mcp-result" key={message.id}><summary>{message.mcp?.connection ? `${message.mcp.connection} · ` : ''}{message.mcp?.tool || 'Tool result'} · {message.mcp?.status || 'received'}{message.mcp?.is_error ? ' · error' : ''}</summary><ToolOutputGallery content={message.content} /><pre>{message.content}</pre></details></OutputMessageContext.Provider>
                return (
                  <Fragment key={message.id}>
                    {showDaySeparator && (
                      <div className="cxchat__day-sep" role="separator">
                        <span>{formatDayLabel(message.created_at)}</span>
                      </div>
                    )}
                    {(!message.mcp_managed || !message.tool_calls?.length || Boolean(message.content?.trim())) && <OutputMessageContext.Provider value={message.id}><MessageTurn
                      message={message}
                      hideManagedToolCalls={Boolean(toolActivity.groups.has(message.id))}
                      generationElapsedSeconds={generationElapsedSeconds}
                      priorUserPrompt={priorUserPrompt}
                      onReusePrompt={setComposer}
                      onRegenerate={canBranchHere
                        ? () => regenerateAsVariant(message.id)
                        : (canResend && priorUserMessage ? () => resendFromMessage(priorUserMessage.id) : null)}
                      regenerateReplacesThread={!canBranchHere}
                      onSelectVariant={selectMessageVariant ? (index) => selectMessageVariant(message.id, index) : null}
                      onDiscardVariant={discardMessageVariant && !requestActive ? () => discardMessageVariant(message.id) : null}
                      onEditResend={canResend && message.role === 'user' ? (messageId, content) => resendFromMessage(messageId, content) : null}
                      onOpenDocument={openDocument}
                      onContinue={canContinue ? () => continueFromMessage(message.id) : null}
                      tokenInspection={tokenInspections?.[message.id] || null}
                      structuredRecord={structuredRecords?.[message.id] || null}
                      toolCallRepeat={message.tool_calls
                        ? detectRepeatedCall(
                            (toolCallSignatures?.[selectedConversation?.id] || []).slice(0, -1 * (message.tool_calls.length || 1)),
                            normalizeToolCalls(message.tool_calls) || [],
                          )
                        : null}
                    /></OutputMessageContext.Provider>}
                    {toolActivity.groups.get(message.id)?.map(({ call, result }, callIndex) => <ToolActivityCard key={callIndex} call={call} result={result}
                      live={mcpActivity?.calls?.[JSON.stringify([message.id, call.id])]}
                      queued={mcpActivity?.messageId === message.id && mcpActivity.phase !== 'idle'}
                      approval={mcpApproval} onDecision={decideMcpApproval} onStop={stopGeneration} />)}

                  </Fragment>
                )
              })}
              {followActive && (
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
              <McpRunPanel activity={hasInlineActivity ? null : mcpActivity} approval={mcpApproval} onDecision={decideMcpApproval} onStop={stopGeneration} />
              {/* Follow-up prompts sit under the latest reply — they act on it. */}
              {visibleMessages.length > 0 && !requestActive && canChat && (
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
          {files.length > 0 && <ConversationFilesTray files={files} onOpen={(id) => { setSelectedFileId(id); setFilesOpen(true) }} />}
          {renderComposer()}
        </div>
      </div>

      {activeCitation && (
        <div className="citation-modal-overlay" onClick={() => setActiveCitation(null)}>
          <div className="citation-modal" onClick={(e) => e.stopPropagation()}>
            <div className="citation-modal__header">
              <div className="citation-modal__title">
                <IconFile size={16} />
                <span>{activeCitation.filename || 'Source Document'}</span>
              </div>
              {citationView?.status === 'verified' ? (
                <span className="citation-modal__badge citation-modal__badge--verified">
                  Verified &middot; bytes {citationView.data.byte_start}&ndash;{citationView.data.byte_end}
                </span>
              ) : citationView?.status === 'refused' ? (
                <span className="citation-modal__badge citation-modal__badge--refused">Refused</span>
              ) : (
                <span className="citation-modal__badge">Verifying&hellip;</span>
              )}
            </div>
            {citationView?.status === 'verified' ? (
              <div className="citation-modal__body">
                <span className="citation-modal__context">{citationView.data.before}</span>
                <mark className="citation-modal__span">{citationView.data.span}</mark>
                <span className="citation-modal__context">{citationView.data.after}</span>
              </div>
            ) : citationView?.status === 'refused' ? (
              <div className="citation-modal__refusal">
                <p className="citation-modal__refusal-title">Citation refused</p>
                <p className="citation-modal__refusal-message">{citationView.message}</p>
                <p className="citation-modal__refusal-code">{citationView.code}</p>
              </div>
            ) : (
              <div className="citation-modal__body citation-modal__body--pending">
                Verifying this passage against its source&hellip;
              </div>
            )}
            <div className="citation-modal__footer">
              {RETRIEVAL_NOTES[activeCitation.retrieval] && (
                <span className="citation-modal__found">{RETRIEVAL_NOTES[activeCitation.retrieval]}</span>
              )}
              <button type="button" className="button" onClick={() => setActiveCitation(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      {viewerDocument && <DocumentViewer key={viewerDocument.doc_id} document={viewerDocument} onClose={closeDocumentViewer} apiBase={apiBase} />}
      {library && (
        <KnowledgeLibrary
          apiBase={apiBase}
          collections={knowledge.collections}
          refresh={knowledge.refresh}
          initialCollectionId={library.collectionId}
          searchedIds={updateChatContext ? searchedCollectionIds : null}
          onToggleSearch={updateChatContext ? setCollectionSearched : null}
          onClose={() => setLibrary(null)}
          busy={requestActive}
        />
      )}
    </section>
    {filesOpen && <ConversationFiles key={selectedConversation?.id || 'draft'} files={files} selectedId={selectedFileId} onSelect={setSelectedFileId} onClose={() => setFilesOpen(false)} conversationId={selectedConversation?.id || 'draft'} />}
    </div>
    </OutputPanelContext.Provider>
  )
}
