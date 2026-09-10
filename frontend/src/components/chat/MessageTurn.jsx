import { memo, useEffect, useRef, useState } from 'react'
import { Avatar } from '../ui/Avatar'
import { EvidenceChip } from '../ui/EvidenceChip'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconSearch, IconExternal, IconPlay, IconTrash } from '../ui/icons'
import { AssistantMarkdown, copyText, hasOpenCodeFence } from '../../lib/markdown'
import { capabilityStatusLabel } from '../../lib/capabilities'
import { continuationCountOf } from '../../lib/chatContinuation'
import { activeVariantIndexOf, variantCountOf } from '../../lib/messageVariants'
import { formatModelLabel } from '../../lib/formatters'
import { cleanLegacyDemoCapCopy } from '../../lib/conversationStorage'
import {
  LiveGenerationBadge,
  StreamingLoader,
  streamingStatusLabel,
} from './render/StreamingIndicator'
import { ParityReceiptCard } from './render/ParityReceipt'
import { DeveloperDiagnosticsBlock } from './render/Diagnostics'
import { TokenInspectorCard } from './render/TokenInspector'
import { StructuredOutputCard } from './render/StructuredOutput'
import { ToolCallsCard } from './render/ToolCalls'

const formatMs = (value) => {
  const ms = Number(value)
  if (!Number.isFinite(ms) || ms <= 0) return null
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

const formatRate = (value) => {
  const rate = Number(value)
  if (!Number.isFinite(rate) || rate <= 0) return null
  return `${rate >= 10 ? Math.round(rate) : rate.toFixed(1)} tok/s`
}

const formatTimeOfDay = (value) => {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const formatFullTimestamp = (value) => {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}

const safeWebSourceUrl = (value) => {
  try {
    const parsed = new URL(String(value || ''))
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.toString() : null
  } catch {
    return null
  }
}

function WebResearchSources({ research }) {
  if (!research) return null
  const sources = (Array.isArray(research.sources) ? research.sources : [])
    .map((source) => ({ ...source, safeUrl: safeWebSourceUrl(source?.url) }))
    .filter((source) => source.safeUrl)
  const warnings = (Array.isArray(research.warnings) ? research.warnings : [])
    .map(String)
    .filter(Boolean)
  if (!sources.length && !warnings.length) return null

  if (!sources.length) {
    return (
      <div className="cxturn__web-warning" role="status">
        <IconSearch size={15} />
        <span>
          Web research was unavailable for this reply. Camelid answered without web sources.
          {warnings.map((warning, index) => <small key={`${warning}:${index}`}>{warning}</small>)}
        </span>
      </div>
    )
  }

  return (
    <details className="cxturn__web-sources">
      <summary>
        <IconSearch size={15} />
        <span>Web sources</span>
        <span className="cxturn__web-count">{sources.length}</span>
        {warnings.length > 0 && <span className="cxturn__web-partial">Partial</span>}
      </summary>
      <ol>
        {sources.map((source, index) => (
          <li key={`${source.safeUrl}:${index}`}>
            <a href={source.safeUrl} target="_blank" rel="noopener noreferrer">
              <span>{source.title || source.safeUrl}</span>
              <IconExternal size={13} />
            </a>
          </li>
        ))}
      </ol>
      {warnings.length > 0 && (
        <div className="cxturn__web-source-warning" role="status">
          {warnings.map((warning, index) => <div key={`${warning}:${index}`}>{warning}</div>)}
        </div>
      )}
    </details>
  )
}

/* Sibling navigation for a re-rolled reply.

   Placed at the START of the actions row, before Copy and Regenerate: it is
   the control that tells the reader the other answers still exist, and it is
   useless if they have to discover it after pressing the button that used to
   destroy them. Hidden entirely at one variant, so an ordinary reply is
   visually unchanged. */
function VariantNav({ index, count, onSelect, onDiscard }) {
  if (count <= 1) return null
  const goto = (next) => onSelect?.((next + count) % count)
  return (
    <span className="cxturn__variants" role="group" aria-label={`Reply ${index + 1} of ${count}`}>
      <button
        type="button"
        className="cxturn__variant-step"
        onClick={() => goto(index - 1)}
        aria-label="Previous version of this reply"
        title="Previous version of this reply"
      >
        ‹
      </button>
      <span className="cxturn__variant-count" aria-live="polite">{index + 1}/{count}</span>
      <button
        type="button"
        className="cxturn__variant-step"
        onClick={() => goto(index + 1)}
        aria-label="Next version of this reply"
        title="Next version of this reply"
      >
        ›
      </button>
      {onDiscard && (
        <button
          type="button"
          className="cxturn__variant-step cxturn__variant-discard"
          onClick={() => onDiscard()}
          aria-label="Discard this version"
          title="Discard the version shown; the others are kept"
        >
          <IconTrash size={13} />
        </button>
      )}
    </span>
  )
}

/* Per-message metadata footer. Token counts are labeled by source (backend
   usage vs client estimate); TTFT and tok/s are always client-measured and say
   so — operational telemetry, never support evidence (I4). The Evidence Chip
   cites the contract row that was active when this reply was generated. */
function MessageMetaFooter({ message }) {
  const usage = message.usage
  const ttft = formatMs(message.first_content_ms)
  const rate = formatRate(message.tokens_out_per_sec)
  const duration = formatMs(message.elapsed_ms)
  const usageLabel = message.usage_source === 'backend' ? 'tokens' : 'tokens est.'
  const sentAt = formatTimeOfDay(message.created_at)
  /* A continued reply is more than one request. Disclose that rather than
     letting one set of timings quietly describe only its last segment. */
  const continuedTimes = continuationCountOf(message)
  if (!usage && !ttft && !rate && !message.model_id && !sentAt) return null
  return (
    <footer className="cxturn__meta" aria-label="Generation details (client-measured telemetry)">
      {message.model_id && (
        <span className="cxturn__meta-item cxturn__meta-model" title={message.model_id}>
          {formatModelLabel(message.model_id)}
        </span>
      )}
      {message.support_row && !message.experimental_lane && (
        <EvidenceChip
          status={message.support_row.status}
          state={message.support_row.supported ? 'supported' : null}
          /* Under a reply the status is reassurance, not the record: show the
             plain verdict and keep the row id and raw status in the popover. */
          label={capabilityStatusLabel(message.support_row.status)}
          source={{
            rowId: message.support_row.id,
            detail: `Status ${message.support_row.status} — the row active when this reply was generated.`,
          }}
          size="sm"
        />
      )}
      {message.experimental_lane && (
        <EvidenceChip
          state="unsupported"
          asText
          source={{ detail: 'This model type runs, but this exact build has not been validated against the reference implementation.' }}
          size="sm"
        >
          Unverified model
        </EvidenceChip>
      )}
      {usage && Number.isFinite(Number(usage.prompt_tokens)) && (
        <span className="cxturn__meta-item cxturn__meta-usage" title={message.usage_source === 'backend' ? 'Token counts reported by the backend' : 'Live token counts estimated client-side until the backend reports final usage'}>
          {usageLabel} <strong>in {usage.prompt_tokens}</strong><span aria-hidden="true"> · </span><strong>out {usage.completion_tokens ?? 0}</strong>
        </span>
      )}
      {/* "TTFT" is the term of art; under a reply it just needs to say what it
          measures. The abbreviation stays in the tooltip for anyone comparing. */}
      {ttft && <span className="cxturn__meta-item" title="Time to first content (TTFT), measured in this browser">first token {ttft}</span>}
      {rate && <span className="cxturn__meta-item" title="Decode rate, measured in this browser">{rate}</span>}
      {duration && <span className="cxturn__meta-item" title="Total request duration, measured in this browser">{duration}</span>}
      {continuedTimes > 0 && (
        <span
          className="cxturn__meta-item"
          title={`Resumed ${continuedTimes === 1 ? 'once' : `${continuedTimes} times`} after hitting the response budget. Token counts cover the whole reply; the timings here cover the last segment only.`}
        >
          continued{continuedTimes > 1 ? ` ×${continuedTimes}` : ''}
        </span>
      )}
      {sentAt && <time className="cxturn__meta-item" dateTime={message.created_at} title={formatFullTimestamp(message.created_at)}>{sentAt}</time>}
      <span className="cxturn__meta-item cxturn__meta-note">client-measured</span>
    </footer>
  )
}

/* User rows: copy + inline edit-and-resend. Editing truncates the thread at
   this message and resends through the normal gate-checked send path. Copy is
   always available — only "Edit & resend" is gated on resend being possible. */
function UserTurn({ message, messageContent, onEditResend }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(messageContent)
  const [copied, setCopied] = useState(false)
  const copiedResetRef = useRef(null)
  const sentAt = formatTimeOfDay(message.created_at)
  const submitEdit = () => {
    const next = draft.trim()
    setEditing(false)
    if (next && next !== messageContent) onEditResend?.(message.id, next)
  }

  useEffect(() => () => {
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
  }, [])

  const handleCopy = async () => {
    if (!(await copyText(messageContent))) return
    setCopied(true)
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
    copiedResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }
  return (
    <article className="cxturn cxturn--user">
      <div className="cxturn__user-wrapper">
        <div className="cxturn__user-chip">
          {message.image?.data_url && (
            <img
              className="cxturn__user-image"
              src={message.image.data_url}
              alt={message.image.name ? `Uploaded ${message.image.name}` : 'Uploaded image'}
            />
          )}
          {editing ? (
            <div className="cxturn__edit">
              <textarea
                className="cxturn__edit-input"
                value={draft}
                rows={Math.min(8, Math.max(2, draft.split('\n').length))}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    submitEdit()
                  }
                  if (event.key === 'Escape') {
                    event.stopPropagation()
                    setEditing(false)
                    setDraft(messageContent)
                  }
                }}
                aria-label="Edit message and resend"
                autoFocus
              />
              <div className="cxturn__edit-actions">
                <button type="button" className="cxturn__action" onClick={submitEdit}>Resend</button>
                <button type="button" className="cxturn__action" onClick={() => { setEditing(false); setDraft(messageContent) }}>Cancel</button>
              </div>
            </div>
          ) : (
            <p>{messageContent}</p>
          )}
        </div>
        {!editing && (
          <div className="cxturn__actions cxturn__actions--user" aria-label="Message actions">
            {sentAt && (
              <time className="cxturn__action-time" dateTime={message.created_at} title={formatFullTimestamp(message.created_at)}>{sentAt}</time>
            )}
            <button
              type="button"
              className={`cxturn__action cxturn__action--icon ${copied ? 'is-copied' : ''}`}
              onClick={handleCopy}
              title={copied ? 'Copied' : 'Copy'}
              aria-label={copied ? 'Copied' : 'Copy message'}
            >
              {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
            </button>
            {onEditResend && (
              <button
                type="button"
                className="cxturn__action cxturn__action--icon"
                onClick={() => { setDraft(messageContent); setEditing(true) }}
                title="Edit message"
                aria-label="Edit message"
              >
                <IconEdit size={16} />
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  )
}

export const MessageTurn = memo(function MessageTurn({ message, generationElapsedSeconds, priorUserPrompt, onReusePrompt, onRegenerate, onEditResend, onContinue, onSelectVariant, onDiscardVariant, regenerateReplacesThread = false, tokenInspection = null, structuredRecord = null, toolCallRepeat = null }) {
  const [copied, setCopied] = useState(false)
  const copiedResetRef = useRef(null)
  const messageContent = cleanLegacyDemoCapCopy(message.content)
  const isUser = message.role === 'user'
  const assistantStreaming = message.role === 'assistant' && Boolean(message.streaming)
  const isOpenStreamingCode = assistantStreaming && hasOpenCodeFence(messageContent)
  const streamingPhase = message.streaming_phase || (messageContent ? 'streaming' : 'generating')
  const liveStatusLabel = streamingStatusLabel(streamingPhase, generationElapsedSeconds, isOpenStreamingCode)
  const showStreamingStatus = assistantStreaming && !messageContent
  const showLiveGenerationBadge = assistantStreaming && Boolean(messageContent)
  const noVisibleResponse = message.role === 'assistant' && !assistantStreaming && !String(messageContent || '').trim()
  const hiddenTokenLimit = noVisibleResponse
    && message.finish_reason === 'length'
    && Number(message.usage?.completion_tokens || 0) > 0
  const showLengthWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'length' && !hiddenTokenLimit
  const showErrorWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'error'
  const showInterruptedWarning = message.role === 'assistant' && !assistantStreaming && message.finish_reason === 'interrupted'
  const showReusePromptAction = Boolean(priorUserPrompt) && (showErrorWarning || showInterruptedWarning)
  const showMessageActions = message.role === 'assistant' && Boolean(String(messageContent || '').trim())
  const variantCount = variantCountOf(message)
  const variantIndex = activeVariantIndexOf(message)
  /* Navigation stays usable while another turn streams: switching is a local
     edit with no request behind it. */
  const showVariantNav = message.role === 'assistant' && !assistantStreaming && variantCount > 1

  useEffect(() => () => {
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
  }, [])

  /* Only confirm "Copied" when the text actually reached the clipboard —
     copyText returns false when clipboard access is unavailable or denied. */
  const handleCopyMessage = async () => {
    if (!(await copyText(messageContent))) return
    setCopied(true)
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
    copiedResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  if (isUser) {
    return (
      <UserTurn
        message={message}
        messageContent={messageContent}
        onEditResend={onEditResend}
      />
    )
  }

  return (
    <article
      className={`cxturn cxturn--assistant ${assistantStreaming ? 'is-streaming' : ''}`}
      aria-busy={assistantStreaming ? 'true' : undefined}
      data-streaming-state={assistantStreaming ? 'active' : undefined}
      data-streaming-code-state={isOpenStreamingCode ? 'open' : undefined}
    >
      <div className="cxturn__avatar">
        <Avatar
          size={30}
          state={assistantStreaming ? (messageContent ? 'streaming' : 'awaiting') : 'idle'}
          pulse={assistantStreaming ? String(messageContent || '').length : 0}
        />
      </div>
      <div className="cxturn__body">
        {showStreamingStatus && <StreamingLoader elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} compact />}
        {(messageContent || !assistantStreaming) && (
          <AssistantMarkdown
            content={messageContent}
            streaming={assistantStreaming}
            citations={message.citations}
          />
        )}
        <WebResearchSources research={message.web_research} />
        {showLiveGenerationBadge && <LiveGenerationBadge elapsedSeconds={generationElapsedSeconds} label={liveStatusLabel} tokensPerSec={message.tokens_out_per_sec} />}

        {noVisibleResponse && (
          <div className="cxturn__warning" role="status">
            {hiddenTokenLimit
              ? 'No visible text was produced before the token limit. Gemma used the budget on hidden channel tokens; increase Settings → Chat → Response length.'
              : '(empty response)'}
          </div>
        )}

        {showLengthWarning && (
          <div className="cxturn__warning" role="status">
            <span>Stopped at the response budget, not at the end of the answer.</span>
            {onContinue && (
              /* Resumes this same reply in place. Regenerate, one row down,
                 throws the text away and starts over — keep the two verbs
                 visibly different so neither is clicked for the other. */
              <button
                type="button"
                className="cxturn__warning-action"
                onClick={() => onContinue()}
                title="Ask the model to pick up exactly where it stopped and add to this reply"
              >
                <IconPlay size={13} /> <span>Continue</span>
              </button>
            )}
          </div>
        )}
        {showErrorWarning && (
          <div className="cxturn__warning cxturn__warning--error" role="status">Generation stopped before Camelid returned a complete reply.</div>
        )}
        {showInterruptedWarning && (
          <div className="cxturn__warning cxturn__warning--interrupted" role="status">Generation was interrupted before the reply finished.</div>
        )}

        {(showMessageActions || showReusePromptAction || showVariantNav) && (
          <div className="cxturn__actions" aria-label="Message actions">
            {showVariantNav && (
              <VariantNav
                index={variantIndex}
                count={variantCount}
                onSelect={onSelectVariant}
                onDiscard={onDiscardVariant}
              />
            )}
            {showMessageActions && (
              <button
                type="button"
                className={`cxturn__action cxturn__action--icon ${copied ? 'is-copied' : ''}`}
                onClick={handleCopyMessage}
                title={copied ? 'Copied' : 'Copy response'}
                aria-label={copied ? 'Copied' : 'Copy response'}
              >
                {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
              </button>
            )}
            {showMessageActions && onRegenerate && (
              /* Two different promises behind one icon, so the label has to
                 carry the difference: on the last reply the answer on screen
                 is KEPT as a sibling; mid-thread it is replaced along with
                 every turn after it. */
              <button
                type="button"
                className="cxturn__action cxturn__action--icon"
                onClick={() => onRegenerate()}
                title={regenerateReplacesThread
                  ? 'Regenerate — replaces this reply and every turn after it'
                  : 'Regenerate — writes another answer and keeps this one alongside it'}
                aria-label={regenerateReplacesThread
                  ? 'Regenerate response, replacing this reply and every turn after it'
                  : 'Regenerate response, keeping this one as another version'}
              >
                <IconRefresh size={16} />
              </button>
            )}
            {showReusePromptAction && (
              <button type="button" className="cxturn__action" onClick={() => onReusePrompt?.(priorUserPrompt)} title="Use prompt again">
                <IconRefresh size={14} /> <span>Use prompt again</span>
              </button>
            )}
          </div>
        )}

        {/* Rendered during streaming too: tokens_out_per_sec is live-patched per
           frame (backed token-for-token by window.__tpsTrace), so the footer
           doubles as the live tok/s readout while decoding. Absent fields stay
           hidden until the stream completes; the footer itself reserves the
           layout space the old placeholder div held. */}
        {message.role === 'assistant' && <MessageMetaFooter message={message} />}

        {message.role === 'assistant' && !assistantStreaming && message.camelid_receipt && (
          <ParityReceiptCard receipt={message.camelid_receipt} />
        )}
        {message.role === 'assistant' && !assistantStreaming && structuredRecord && (
          <StructuredOutputCard record={structuredRecord} />
        )}
        {message.role === 'assistant' && !assistantStreaming && tokenInspection && (
          <TokenInspectorCard
            inspection={tokenInspection.logprobs}
            absence={tokenInspection.absence}
          />
        )}
        {message.role === 'assistant' && !assistantStreaming && message.tool_calls && (
          <ToolCallsCard toolCalls={message.tool_calls} repeated={toolCallRepeat} replyContent={messageContent} />
        )}
        <DeveloperDiagnosticsBlock message={message} />
      </div>
    </article>
  )
})

export default MessageTurn
