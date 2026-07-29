import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { getConfiguredMaxTokens, modelContextLength, validateSendBudget } from '../lib/responseLimits'
import { CamelidMark } from '../components/ui/CamelidMark'
import { Avatar } from '../components/ui/Avatar'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { IconSend, IconStop, IconMemory, IconReceipt, IconThinking, IconBolt, IconChart, IconChat, IconEdit } from '../components/ui/icons'
import { MessageTurn } from '../components/chat/MessageTurn'
import { ChatControls } from '../components/chat/ChatControls'
import { PREPARING_STREAMING_LABEL, StreamingLoader } from '../components/chat/render/StreamingIndicator'

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const isInterruptedPlaceholderMessage = (message) => {
  if (message?.role !== 'assistant') return false
  const content = String(message?.content || '').trim().toLowerCase()
  return content === '(generation interrupted)' || content === '(generation stopped)'
}

function readinessTone({ ready = false, blocked = false, offline = false, waiting = false } = {}) {
  if (ready) return 'ready'
  if (offline || blocked) return 'blocked'
  if (waiting) return 'waiting'
  return 'idle'
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
  stoppingGeneration = false,
  selectedModelRunnable,
  selectedModelExperimental = false,
  setTab,
  showNewChatLanding = null,
  firstRunActive = false,
  demoMode = false,
}) {
  // Chat is allowed on the supported lane (full gate) OR the weaker experimental
  // lane (implemented-but-unsupported). The supported-specific copy below stays
  // keyed on `selectedModelRunnable`; the experimental lane gets its own banner and
  // never borrows the supported badge.
  const canChat = selectedModelRunnable || selectedModelExperimental
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const [showControls, setShowControls] = useState(false)
  const [showAllMessages, setShowAllMessages] = useState(false)
  const [userScrolledAway, setUserScrolledAway] = useState(false)
  const chatBottomRef = useRef(null)
  const composerRef = useRef(null)
  const autoFollowGenerationRef = useRef(true)
  const composerReadinessId = 'camelid-chat-readiness-note'

  const rawVisibleMessages = useMemo(
    () => (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message)),
    [selectedConversation?.messages],
  )
  const hasStreamingAssistant = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming)
  const hasStreamingAssistantContent = rawVisibleMessages.some((m) => m.role === 'assistant' && m.streaming && String(m.content || '').trim())
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
    pendingPrompt && [...visibleMessages].reverse().some((m) => m.role === 'user' && m.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const lastVisibleMessage = visibleMessages.at(-1)
  const lastVisibleMessageIsUser = lastVisibleMessage?.role === 'user'
  const awaitingAssistant = Boolean(generationActive && !hasStreamingAssistantContent && !hasStreamingAssistant && (pendingPrompt || lastVisibleMessageIsUser || sending))
  const streamingScrollSignature = useMemo(() => (
    visibleMessages.map((m) => `${m.id}:${m.streaming ? 'streaming' : 'done'}:${String(m.content || '').length}`).join('|')
    + `|awaiting:${awaitingAssistant ? '1' : '0'}|active:${generationActive ? '1' : '0'}`
  ), [awaitingAssistant, generationActive, visibleMessages])
  const isFreshThread = selectedConversation
    ? (visibleMessages.length === 0 && !pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)
    : (!pendingPrompt && !awaitingAssistant && !hasStreamingAssistant)

  // ----- Gate / readiness derivations (shared exact-row chat gate) -----
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
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const selectedModelIssue = selectedModel?.load_error || selectedModel?.install_error || ''

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
          : 'Load a local GGUF from Models to start the readiness check.'
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
    ? `${selectedCompatibilityLabel}. Ready for this loaded exact row.`
    : apiUnavailable
      ? 'Drafts stay editable while the Camelid API reconnects.'
      : selectedModel
        ? 'Chat unlocks only after loaded_now=true, generation_ready=true, and an exact supported compatibility row all match.'
        // The activation card above owns the instruction during first run; repeating
        // it here is the third copy of the same sentence on one screen.
        : firstRunActive
          ? 'Send unlocks as soon as the model above finishes setting up.'
          : 'Choose a model, then Camelid will show what still needs to pass before send unlocks.'
  const selectedModelReadinessCopy = selectedModelRunnable
    ? 'Selected model is ready for Camelid chat.'
    : apiUnavailable
      ? 'The API is offline, so readiness cannot be checked yet.'
      : selectedModelIssue
        ? selectedModelIssue
        : supportBlocked
          ? selectedCompatibilityCopy
          : selectedRuntimeMatchesLoadedModel
            ? 'This model is loaded and still warming up. Send unlocks once generation readiness turns on.'
            : selectedModel
              ? 'Keep drafting here while Camelid prepares this model.'
              : 'Choose a model before starting a Camelid chat.'
  const selectedModelGateSummary = selectedModel
    ? selectedModelRunnable
      ? 'Selected model is ready for Camelid chat.'
      : selectedModelIssue || selectedModelReadinessCopy
    : 'Choose a model before starting a Camelid chat.'

  const productHeroTitle = selectedModelRunnable ? 'How can I help?' : "Hi there, let's get into it"
  const productHeroSummary = selectedModelRunnable
    ? 'Local chat is ready. Ask anything — responses stay grounded in the loaded model.'
    : apiUnavailable
      ? 'Keep writing here. Send unlocks again once the local API responds.'
      : supportBlocked
        ? 'The runtime is up, but chat still needs an exact supported row before send unlocks.'
        : selectedModel
          ? 'Your draft is ready now. Send unlocks as soon as this model is ready.'
          // The activation card above already names the one thing to do; repeating
          // "pick a model" here would offer a second, vaguer instruction.
          : firstRunActive
            ? 'Camelid answers with a model running on this machine. Set one up above and this becomes a chat.'
            : 'Pick a local GGUF model first. Camelid will show the readiness path here.'

  const readinessState = selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const runtimeTone = readinessTone({ ready: selectedModelRunnable, offline: apiUnavailable, waiting: Boolean(runtime?.loaded_now || selectedModel) })
  const statusTone = selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : supportBlocked ? 'warn' : runtime?.loaded_now ? 'warn' : 'neutral'

  const selectionSummaryCopy = selectedModelRunnable
    ? `${selectedModelName} is loaded now and generation_ready=true. The current exact-row contract is unlocked.`
    : apiUnavailable
      ? 'The frontend is available, but the Camelid API must respond before model readiness can be checked.'
      : selectedModelIssue
        ? selectedModelIssue
        : supportBlocked
          ? selectedCompatibilityCopy
          : selectedModel
            ? 'Drafting stays unlocked. Camelid will unlock send as soon as this selected row is loaded, generation-ready, and supported.'
            : 'Pick a local model first, then Camelid will keep the runtime and support boundary visible here.'

  const canSubmit = Boolean(composer.trim()) && canChat && !generationActive
  const sendDisabledReason = canChat
    ? ''
    : generationActive
      ? 'Wait for the current reply to finish or stop it before sending again.'
      : apiUnavailable
        ? 'Send unlocks after the Camelid API reconnects.'
        : supportBlocked
          ? 'Choose a supported model.'
          : selectedModel
            ? 'Send unlocks when Camelid marks this model ready and supported.'
            : 'Choose a model before sending.'
  const promptHintCopy = canChat
    ? 'Enter sends · Shift+Enter for a new line'
    : apiUnavailable
      ? 'Draft now · send unlocks after the API reconnects'
      : supportBlocked
        ? 'Send unlocks after exact-row readiness passes'
        : selectedModel
          ? 'Draft now · send unlocks after readiness passes'
          : 'Choose a model to unlock sending'
  /* During first run the activation card is the instruction. The send button keeps its
     own tooltip, but this line would be the fourth restatement on one screen. */
  const composerHintCopy = canSubmit
    ? promptHintCopy
    : firstRunActive && !selectedModel
      ? ''
      : sendDisabledReason || promptHintCopy

  const composerDraftUnlocked = Boolean(selectedModel || apiUnavailable)
  const composerDisabled = !composerDraftUnlocked
  const composerPlaceholder = canChat
    ? 'Message Camelid…'
    : apiUnavailable
      ? 'Draft a prompt while the Camelid API comes back'
      : composerDraftUnlocked
        ? 'Draft a prompt while Camelid finishes getting ready'
        : firstRunActive
          ? 'Set up the model above, then chat here'
          : isFreshThread
            ? 'Load a model first'
            : 'Choose a ready model first'
  const composerStopLabel = stoppingGeneration ? 'Stopping…' : 'Stop'
  const secondaryActionLabel = selectedModelRunnable ? 'Save to memory' : (apiUnavailable ? 'Open API' : 'Open Models')
  const secondaryAction = selectedModelRunnable ? saveToMemory : () => setTab(apiUnavailable ? 'api' : 'library')
  const secondaryActionDisabled = selectedModelRunnable ? generationActive : false

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
    if (!generationActive) return undefined
    autoFollowGenerationRef.current = true
    setUserScrolledAway(false)
    const updateAutoFollow = () => {
      const el = document.querySelector('.cxchat__scroll')
      if (!el) return
      const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight)
      const follow = distanceFromBottom < 260
      if (follow !== autoFollowGenerationRef.current) setUserScrolledAway(!follow)
      autoFollowGenerationRef.current = follow
    }
    const el = document.querySelector('.cxchat__scroll')
    el?.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => el?.removeEventListener('scroll', updateAutoFollow)
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

  useEffect(() => {
    if (generationActive || !composerDraftUnlocked) return
    const input = composerRef.current
    if (!input) return
    const activeElement = document.activeElement
    if (activeElement && activeElement !== document.body && activeElement !== input) return
    const frame = window.requestAnimationFrame(() => input.focus())
    return () => window.cancelAnimationFrame(frame)
  }, [composerDraftUnlocked, generationActive, isFreshThread, selectedConversation?.id])

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Escape' && generationActive) {
      event.preventDefault()
      stopGeneration?.()
      return
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) await sendMessage()
    }
  }

  const handleSuggestion = (prompt) => {
    if (!composerDraftUnlocked) return
    setComposer(prompt)
  }

  // ----- Model picker -----
  const runnableModels = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked)
  const waitingModels = models.filter((model) => !getChatGateState(capabilities, model, runtime).chatUnlocked)
  const selectedPickerModelId = models.some((model) => model.id === selectedModel?.id) ? selectedModel.id : ''
  const modelOptionLabel = (model) => {
    const gate = getChatGateState(capabilities, model, runtime)
    if (gate.chatUnlocked) return `${model.name} · Ready`
    if (apiUnavailable) return `${model.name} · API unavailable`
    if (gate.runtimeReady) return `${model.name} · Support gated`
    if (gate.runtimeLoaded) return `${model.name} · Loading`
    return `${model.name} · Not loaded`
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
  const sendBudget = validateSendBudget({
    promptTokens: estimatedPromptTokens,
    maxTokens: getConfiguredMaxTokens(selectedModelId),
    contextLength: modelContextLength(selectedModel),
  })

  const detailCopy = selectedModelRunnable ? selectionSummaryCopy : (supportBlocked || selectedModelIssue ? selectedCompatibilityCopy : readinessFinePrint)

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
              <label className="cxcomposer__model" title={selectedModel ? supportStatusCopy : 'Choose what Camelid should use for this chat.'}>
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
                  disabled={generationActive || Boolean(loadingModelId)}
                >
                  {!selectedModel && <option value="">Choose model</option>}
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
                </select>
              </label>
            ) : (
              <button type="button" className="cxcomposer__tool" onClick={() => setTab('library')}>Add a model</button>
            )}
            {!demoMode && setReceiptMode && (
              <button
                type="button"
                className={`cxcomposer__tool ${receiptMode ? 'is-on' : ''}`}
                title="Attach a verifiable parity receipt to the next reply (non-streaming). A receipt records one request only; it is not a support claim."
                aria-pressed={receiptMode}
                onClick={() => setReceiptMode(!receiptMode)}
              >
                <IconReceipt size={16} /> {receiptMode ? 'Receipt on' : 'Receipt'}
              </button>
            )}
            {!demoMode && setThinkingMode && (
              <button
                type="button"
                className={`cxcomposer__tool ${thinkingMode ? 'is-on' : ''}`}
                title="Thinking mode (experimental — not parity-locked). The model emits its own <think>…</think> reasoning. Only the leading reasoning trace is evidenced against llama.cpp; the parity-locked exact-row mode stays thinking-disabled."
                aria-pressed={thinkingMode}
                onClick={() => setThinkingMode(!thinkingMode)}
              >
                <IconThinking size={16} /> {thinkingMode ? 'Thinking on (experimental)' : 'Thinking'}
              </button>
            )}
            {!demoMode && (
              <button type="button" className="cxcomposer__tool" onClick={secondaryAction} disabled={secondaryActionDisabled}>
                <IconMemory size={16} /> {secondaryActionLabel}
              </button>
            )}
            {!demoMode && (
              <button
                type="button"
                className={`cxcomposer__tool ${showControls ? 'is-on' : ''}`}
                aria-expanded={showControls}
                onClick={() => setShowControls((value) => !value)}
                title="System prompt and contract-gated sampling controls"
              >
                <IconBolt size={16} /> Controls
              </button>
            )}
          </div>
          <div className="cxcomposer__actions">
            {generationActive && (
              <button type="button" className="cxcomposer__stop" aria-label="Stop Camelid generation" onClick={stopGeneration} disabled={stoppingGeneration}>
                <IconStop size={16} /> {composerStopLabel}
              </button>
            )}
            <button
              type="button"
              className="cxcomposer__send"
              aria-label="Send message"
              data-send-ready={canSubmit ? 'true' : 'false'}
              title={!canSubmit ? sendDisabledReason : 'Send message to Camelid'}
              onClick={sendMessage}
              disabled={!canSubmit || sendBudget.level === 'error'}
            >
              <IconSend size={20} />
            </button>
          </div>
        </div>
      </div>

      {sendBudget.level === 'error' && (
        <p className="cxcomposer__budget-error" role="alert">
          <span aria-hidden="true">✕</span> {sendBudget.message}
        </p>
      )}
      {sendBudget.level === 'notice' && (
        <p className="cxcomposer__budget-notice">
          <span aria-hidden="true">ⓘ</span> {sendBudget.message}
        </p>
      )}
      <div className={`cxcomposer__status is-${statusTone}`} role="status" aria-live="polite" title={`${runtimeStatusCopy} ${supportStatusCopy} ${readinessFinePrint}`}>
        <StatusDot tone={statusTone} pulse={selectedModelRunnable} />
        <strong className="cxcomposer__status-label">{runtimeStatusLabel}</strong>
        <span className="cxcomposer__status-sep" aria-hidden="true">·</span>
        <span className="cxcomposer__status-model">{selectedModelName}</span>
        {selectedModel && (
          <>
            <span className="cxcomposer__status-sep" aria-hidden="true">·</span>
            <EvidenceChip
              status={selectedChatGate.hint?.target?.status || ''}
              state={selectedChatGate.contractSupported ? 'supported' : selectedChatGate.hint?.target?.status ? null : 'unsupported'}
              label={supportStatusLabel}
              source={{ rowId: selectedChatGate.hint?.target?.id, note: selectedChatGate.copy }}
              size="sm"
              className="cxcomposer__status-row"
            />
          </>
        )}
      </div>
      <p id={composerReadinessId} className="cxcomposer__detail">{detailCopy}</p>
      {composerHintCopy ? <p className="cxcomposer__hint">{composerHintCopy}</p> : null}
    </div>
  )

  return (
    <section className={`cxchat is-${readinessState} ${userScrolledAway ? 'is-user-scrolled' : ''} ${isFreshThread ? 'cxchat--empty' : ''}`} data-view="chat">
      <div className="cxchat__scroll">
        <div className="cxchat__column">
          {selectedModelExperimental && !selectedModelRunnable && (
            <div className="cxchat__experimental-banner" role="note">
              <EvidenceChip state="unsupported" asText>Experimental</EvidenceChip>
              <span>
                Output is <strong>unverified and has no parity guarantee</strong>. This model's
                architecture is implemented, but it is not a supported row — every reply below is
                marked experimental.
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
              {visibleMessages.length > 0 && !generationActive && selectedModelRunnable && (
                <div className="cxchat__followups" aria-label="Follow-up prompts">
                  {FOLLOW_UP_PROMPTS.map((prompt) => (
                    <button key={prompt} type="button" className="cxchat__followup" onClick={() => handleSuggestion(prompt)}>{prompt}</button>
                  ))}
                </div>
              )}
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
                const canResend = Boolean(resendFromMessage) && !generationActive && canChat
                return (
                  <MessageTurn
                    key={message.id}
                    message={message}
                    generationElapsedSeconds={generationElapsedSeconds}
                    priorUserPrompt={priorUserPrompt}
                    onReusePrompt={setComposer}
                    onRegenerate={canResend && priorUserMessage ? () => resendFromMessage(priorUserMessage.id) : null}
                    onEditResend={canResend && message.role === 'user' ? (messageId, content) => resendFromMessage(messageId, content) : null}
                  />
                )
              })}
              {generationActive && (
                <button
                  type="button"
                  className="cxchat__jump-latest"
                  data-autofollow-affordance
                  onClick={() => { autoFollowGenerationRef.current = true; setUserScrolledAway(false); chatBottomRef.current?.scrollIntoView({ block: 'end' }) }}
                >
                  ↓ jump to latest
                </button>
              )}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="cxturn cxturn--user"><div className="cxturn__user-chip"><p>{pendingUserPrompt}</p></div></article>
                  )}
                  <article className="cxturn cxturn--assistant is-streaming" aria-busy="true" data-streaming-state="active">
                    <div className="cxturn__avatar"><Avatar size={30} state="awaiting" /></div>
                    <div className="cxturn__body"><StreamingLoader elapsedSeconds={generationElapsedSeconds} label={PREPARING_STREAMING_LABEL} /></div>
                  </article>
                </>
              )}
              <div className="cxchat__anchor" ref={chatBottomRef} aria-hidden="true" />
            </div>
          )}
        </div>
      </div>

      <div className="cxchat__dock">
        <div className="cxchat__column">
          {renderComposer()}
          <p className="cxchat__disclaimer">Camelid runs the loaded model locally. Verify important output.</p>
        </div>
      </div>
    </section>
  )
}
