import { useEffect, useMemo, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel, quantLabelFromGgufFileType } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { resolveLoadedModelDisplayName } from '../lib/loadedModelDisplay'
import { isEmbeddingOnlyModel, isGenerationCapableModel, matchResidentItemsToLocalRecords, modelCapabilityFields } from '../lib/modelCapabilities.js'
import { loadLocalModelForChat, modelFilenameFromPath } from '../lib/modelActivation.js'
import { readStreamingChatCompletion } from '../lib/chatCompletionStream'
import { readExactTargetVerifiedRender, readTargetVerifiedMtp12 } from '../lib/nativeGenerationMetrics'
import { readExactTargetVerifiedSegmentedRender } from '../lib/nativeGenerationMetrics'
import { NEW_CHAT_SENTINEL, resolveSelectedConversation, shouldCreateConversationForSend } from '../lib/chatState'
import { normalizeStoredConversations } from '../lib/conversationStorage.js'
import { appStorage } from '../lib/appStorage.js'
import { composeContextBudget } from '../lib/contextBudget.js'
import {
  AUTO_COMPACT_THRESHOLD_PERCENT,
  applySendCompaction,
  resolveCompactionIntent,
} from '../lib/conversationCompaction.js'
import { getRuntimeRequestModelId, isExternalModel, modelRuntimeIdMatches } from '../lib/modelState'
import { contractSamplingOverrides } from '../lib/samplingContract'
import { inspectionAbsenceReason, inspectionForcesNonStreaming, inspectionRequestFields, normalizeInspection, readInspectionContract } from '../lib/tokenInspection'
import { STRUCTURED_MODES, DEFAULT_SCHEMA, DEFAULT_GRAMMAR, readStructuredOutputContract, structuredOutputForcesNonStreaming, structuredOutputRequestFields, structuredOutputReadiness } from '../lib/structuredOutput'
import { DEFAULT_TOOLS, detectRepeatedCall, normalizeToolCalls, readModelToolCapability, readToolContract, toolCallSignature, toolReadiness, toolRequestFields } from '../lib/toolCalling'
import { executionRuntimeFields } from '../lib/executionPlan'
import {
  createPacerState,
  paceDrain,
  paceFirstVisiblePrefix,
  paceHasPendingText,
  paceStep,
} from '../lib/streamPacing'
import {
  classifyWebResearchNeed,
  deriveFittedWebResearchReplyBudget,
  deriveWebResearchPromptBudget,
  effectiveGenerationTokenLimit,
  estimateWebResearchChatTokens,
  fitWebResearchContext,
  persistWebResearchEnabled,
  readWebResearchEnabled,
  requestWebResearch,
  webResearchMetadata,
} from '../lib/webResearch.js'
import {
  applyBitNetFreshChatTokenCap,
  applyGemma4ChatTokenFloor,
  applyGemma4GhostChatTokenCap,
  getConfiguredMaxTokens as getModelMaxTokens,
  hasExplicitMaxTokensSetting,
  isBitNetB158ChatModel,
  modelContextLength,
} from '../lib/responseLimits'
import { beginRequest, emitFirstContent, emitProgress, getTelemetrySnapshot, recordChatGeneration, recordHealthPoll } from '../lib/telemetryLog'
import { isGemma4Mtp12TargetVerifiedVideoOptedIn, shouldUseGemma4Mtp12TargetVerifiedRender } from '../lib/targetVerifiedRender.js'
import { isGemma4Mtp12SegmentedVideoOptedIn, readGemma4Mtp12PreparedSegments } from '../lib/segmentedWebResearchSynthesis.js'

const TAB_STORAGE_KEY = 'camelid.activeTab'
const SELECTED_CONVERSATION_STORAGE_KEY = 'camelid.selectedConversationId'
const SELECTED_MODEL_STORAGE_KEY = 'camelid.selectedModelId'
const LOCAL_MODELS_STORAGE_KEY = 'camelid.localModels'
const CONVERSATIONS_STORAGE_KEY = 'camelid.conversations'
const MEMORIES_STORAGE_KEY = 'camelid.memories'
const API_BASE_STORAGE_KEY = 'camelid.apiBase'
const VALID_TABS = new Set(['chat', 'workspace', 'library', 'downloads', 'api', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'compatibility', 'telemetry', 'arena', 'observatory'])
// Where the UI looks for the camelid API by default:
//   1. an explicit VITE_CAMELID_API_BASE override always wins;
//   2. otherwise use the page origin. Production is served by Camelid directly;
//      Vite development proxies API routes to the local backend.
function defaultApiBase() {
  if (import.meta.env?.VITE_CAMELID_API_BASE) return import.meta.env.VITE_CAMELID_API_BASE
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin
  }
  return 'http://127.0.0.1:8181'
}
const DEFAULT_API_BASE = defaultApiBase()

function getInitialTab() {
  if (typeof window === 'undefined') return 'chat'
  const saved = appStorage.getItem(TAB_STORAGE_KEY)
  return saved && VALID_TABS.has(saved) ? saved : 'chat'
}

function getInitialConversationId() {
  if (typeof window === 'undefined') return null
  return appStorage.getItem(SELECTED_CONVERSATION_STORAGE_KEY) || null
}

function getInitialModelId() {
  if (typeof window === 'undefined') return ''
  return appStorage.getItem(SELECTED_MODEL_STORAGE_KEY) || ''
}

function getApiBase() {
  if (typeof window === 'undefined') return DEFAULT_API_BASE
  return appStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE
}

function normalizeApiBase(value) {
  return (value || DEFAULT_API_BASE).trim().replace(/\/$/, '')
}

function readJsonStorage(key, fallback) {
  if (typeof window === 'undefined') return fallback
  try {
    const saved = appStorage.getItem(key)
    return saved ? JSON.parse(saved) : fallback
  } catch {
    appStorage.removeItem(key)
    return fallback
  }
}

function writeJsonStorage(key, value) {
  if (typeof window === 'undefined') return
  try {
    appStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Image attachments can exhaust a browser's small localStorage quota.
    // Keep the live in-memory conversation and request working; persistence
    // is best-effort and must never crash or cancel the current generation.
  }
}

function normalizeSortText(value) {
  return (value || '').toString().trim().toLowerCase()
}

function compareModelsByName(left, right) {
  return normalizeSortText(left.name).localeCompare(normalizeSortText(right.name), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.id).localeCompare(normalizeSortText(right.id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function getModelPath(model) {
  return typeof model?.path === 'string' ? model.path : ''
}

export { resolveLoadedModelDisplayName }

function getLoadedModelFileType(model) {
  const metadata = model?.gguf?.metadata || {}
  return metadata?.general?.file_type ?? metadata?.['general.file_type'] ?? null
}

function getLoadedModelQuantLabel(model) {
  const fileType = getLoadedModelFileType(model)
  if (fileType === null || fileType === undefined) return null
  return quantLabelFromGgufFileType(fileType) || `file_type ${fileType}`
}

function estimateTokenCount(value) {
  const text = String(value || '').trim()
  if (!text) return 0
  const wordPieces = text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
  return Math.max(1, Math.round(Math.max(wordPieces.length, text.length / 4)))
}

const CODE_FIRST_SYSTEM_PROMPT = 'begin immediately with complete runnable code. No intro. Output one self-contained file unless the user asks otherwise. For Python, start exactly with ```python, include imports, and close the fence after the complete script. For Python games, prefer tkinter from the standard library over pygame, keep it compact, and include a complete runnable event loop. For HTML output ONE self-contained file. Never use external files or script src. Include inline <style> and inline <script> with working click/game logic before </body>. Start exactly with ```html then <!doctype html> and close the fence after </html>.'
const MAX_TOKENS_STORAGE_KEY = 'camelid.maxTokens'
const DEFAULT_CHAT_MAX_TOKENS = 8192

function getConfiguredMaxTokens() {
  if (typeof window === 'undefined') return DEFAULT_CHAT_MAX_TOKENS
  const value = Number.parseInt(appStorage.getItem(MAX_TOKENS_STORAGE_KEY) || '', 10)
  return Number.isFinite(value) && value >= 256 ? value : DEFAULT_CHAT_MAX_TOKENS
}

export function looksLikeCodePrompt(value) {
  const text = String(value || '').toLowerCase()
  // Planning language is authoritative even when the prompt names a language
  // and begins with "write". Otherwise "Write a Python implementation plan"
  // takes the language fast path before this guard can protect it. A direct
  // request for code remains code-y when "architecture" merely names what the
  // requested implementation follows.
  const directCodeArtifact = /\b(code|source code|runnable|single file|self-contained file)\b/.test(text)
    && /\b(build|create|generate|implement|make|output|provide|write)\b/.test(text)
  const planningDeliverable = /\b(task list|task-list|checklist|implementation plan|roadmap|methodology|multi-step plan|requirements?|architecture|phases?)\b/.test(text)
  if (planningDeliverable && !directCodeArtifact) return false
  const explicitRunnableRequest = directCodeArtifact || (
    /\b(html|css|javascript|python)\b/.test(text)
      && /\b(generate|output|write)\b/.test(text)
  )
  if (explicitRunnableRequest) return true
  return /\b(code|build|create|implement|write|make)\b/.test(text)
    && /\b(html|html5|css|javascript|js|python|py|pygame|game|pacman|pacmac|tetris|app|component|page|website)\b/.test(text)
}

export function activeRuntimeContextFit(messages, {
  activeContextLength = null,
  maxPromptTokens = null,
  estimateTokenCount = estimateWebResearchChatTokens,
  safetyMargin = null,
} = {}) {
  const parsedContextLength = Math.floor(Number(activeContextLength))
  const contextLength = Number.isFinite(parsedContextLength) && parsedContextLength > 0
    ? parsedContextLength
    : null
  const parsedPromptLimit = Math.floor(Number(maxPromptTokens))
  const promptLimit = Number.isFinite(parsedPromptLimit) && parsedPromptLimit > 0
    ? parsedPromptLimit
    : null
  if ((!contextLength && !promptLimit) || typeof estimateTokenCount !== 'function') {
    return {
      status: 'unknown',
      unfit: false,
      contextLength,
      promptLimit,
      promptTokens: null,
      safetyMargin: null,
      replyRoom: null,
      message: '',
    }
  }
  const estimated = Number(estimateTokenCount(messages))
  if (!Number.isFinite(estimated) || estimated < 0) {
    return {
      status: 'unknown',
      unfit: false,
      contextLength,
      promptLimit,
      promptTokens: null,
      safetyMargin: null,
      replyRoom: null,
      message: '',
    }
  }
  const promptTokens = Math.ceil(estimated)
  const suppliedMargin = Number(safetyMargin)
  const margin = contextLength
    ? safetyMargin !== null && safetyMargin !== undefined
      && Number.isFinite(suppliedMargin) && suppliedMargin >= 0
      ? Math.ceil(suppliedMargin)
      : Math.max(16, Math.ceil(Math.sqrt(contextLength)))
    : null
  const replyRoom = contextLength ? contextLength - promptTokens - margin : null
  const promptLimitExceeded = Boolean(promptLimit && promptTokens > promptLimit)
  const contextUnfit = Boolean(contextLength && replyRoom < 1)
  const unfit = promptLimitExceeded || contextUnfit
  return {
    status: unfit ? 'unfit' : 'fit',
    unfit,
    contextLength,
    promptLimit,
    promptTokens,
    safetyMargin: margin,
    replyRoom: replyRoom === null ? null : Math.max(0, replyRoom),
    message: contextUnfit
      ? `This conversation (~${promptTokens.toLocaleString()} tokens, estimated) fills the active ${contextLength.toLocaleString()}-token runtime context, leaving no safe room for a reply. Shorten the prompt or history, start a new chat, or load the model with a larger context.`
      : promptLimitExceeded
        ? `This conversation (~${promptTokens.toLocaleString()} tokens, estimated) exceeds the server's ${promptLimit.toLocaleString()}-token prompt limit. Shorten the prompt or history, start a new chat, or raise the server prompt limit.`
        : '',
  }
}

const SYSTEM_PROMPT_STORAGE_KEY = 'camelid.systemPrompt'

function getConfiguredSystemPrompt() {
  if (typeof window === 'undefined') return ''
  return String(appStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY) || '').trim()
}

function applyLocalChatPolicy(messages) {
  const lastUser = [...(messages || [])].reverse().find((message) => message.role === 'user')
  const systemMessages = []
  // User-configured system prompt (Generation controls drawer) leads; the
  // code-first policy prompt appends behind it when the prompt looks code-y.
  const configuredPrompt = getConfiguredSystemPrompt()
  if (configuredPrompt) systemMessages.push({ role: 'system', content: configuredPrompt })
  if (looksLikeCodePrompt(lastUser?.content)) systemMessages.push({ role: 'system', content: CODE_FIRST_SYSTEM_PROMPT })
  if (!systemMessages.length) return messages
  return [...systemMessages, ...messages]
}

function localChatMaxTokens(history, modelId = '') {
  // Configurable in Settings → Chat (per-model since Phase 9, with the legacy
  // global key as fallback). Defaults generously so long answers and full
  // programs aren't truncated (the old 800/2048 caps cut off larger code).
  return getModelMaxTokens(modelId)
}

function tokensPerSecond(tokens, elapsedMs) {
  const tokenCount = Number(tokens)
  const duration = Number(elapsedMs)
  if (!Number.isFinite(tokenCount) || !Number.isFinite(duration) || tokenCount <= 0 || duration <= 0) return null
  return tokenCount / (duration / 1000)
}

function isLoadedModelGenerationReady(model) {
  return Boolean(model?.llama_config && model?.llama_tensors && model?.tokenizer?.status === 'available')
}

function fallbackModelName(id, modelPath) {
  if (id) return id
  const fileName = modelPath?.split('/').filter(Boolean).pop() || ''
  return fileName.replace(/\.gguf$/i, '') || 'Local GGUF model'
}

function optionalString(value) {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  return trimmed || null
}

function optionalBoolean(value) {
  return typeof value === 'boolean' ? value : null
}

function normalizeEngineName(value) {
  const engine = optionalString(value)?.toLowerCase()
  if (!engine || engine === 'backendinference' || engine === 'backend inference') return 'camelid'
  return engine
}

function normalizeLocalModelStatus(status) {
  // Preserve the in-flight states too — coercing them to 'registered' made a freshly
  // started download look instantly "registered"/downloaded and killed the progress
  // bar (the `status === 'downloading'` UI branch could never fire).
  return status === 'ready' || status === 'registered' || status === 'failed'
    || status === 'downloading' || status === 'canceling'
    ? status
    : 'registered'
}

function normalizeLocalModelRecord(record) {
  if (!record || typeof record !== 'object') return null
  const modelPath = String(record.model_path || record.path || '').trim()
  const id = String(record.id || record.runtime_model_name || fallbackModelName('', modelPath)).trim()
  if (!id || !modelPath) return null
  const capabilityFields = modelCapabilityFields(record)
  return {
    id,
    name: String(record.name || fallbackModelName(id, modelPath)).trim(),
    provider_kind: 'local',
    status: normalizeLocalModelStatus(record.status),
    model_path: modelPath,
    runtime_model_name: String(record.runtime_model_name || id).trim(),
    source: record.source || 'Local GGUF file',
    engine: normalizeEngineName(record.engine),
    quant: record.quant || null,
    architecture: optionalString(record.architecture),
    model_family: optionalString(record.model_family),
    chat_capable: optionalBoolean(record.chat_capable),
    embedding_capable: optionalBoolean(record.embedding_capable) ?? capabilityFields.embedding_capable,
    generation_capable: optionalBoolean(record.generation_capable) ?? capabilityFields.generation_capable,
    task_kind: capabilityFields.task_kind,
    task_tags: Array.isArray(record.task_tags) ? record.task_tags.map(String) : [],
    size_gb: record.size_gb || null,
    api_base: record.api_base || null,
    api_key_configured: false,
    install_error: optionalString(record.install_error),
    load_error: optionalString(record.load_error),
    last_load_attempt_at: optionalString(record.last_load_attempt_at),
    last_loaded_at: optionalString(record.last_loaded_at),
    // Download-progress fields must survive normalization so the progress bar can
    // render a live percentage while status === 'downloading'.
    bytes_downloaded: Number(record.bytes_downloaded) || 0,
    total_bytes: Number(record.total_bytes) || 0,
    progress: Number(record.progress) || 0,
    loaded_now: false,
    generation_ready: false,
    camelid: {
      active: false,
      loaded_now: false,
      generation_ready: false,
      tokenizer_status: null,
      tokenizer_model: null,
      tensor_ready: false,
      config_ready: false,
    },
    updated_at: record.updated_at || nowIso(),
  }
}

function upsertLocalModelRecord(records, record) {
  const normalized = normalizeLocalModelRecord(record)
  if (!normalized) return records
  return [normalized, ...records.filter((item) => item.id !== normalized.id)].sort(compareModelsByName)
}

// A persisted local record is "stale" when it claims a GGUF that should live in
// models/ but the backend's authoritative directory scan (/api/models/local) no
// longer lists it — e.g. the file was deleted, or a catalog download was recorded
// but never actually landed on disk. Such records must not linger and keep
// presenting a model as downloaded/loadable. Records that are safe and must never
// be dropped: hosted/external API models, in-flight downloads (by status OR by an
// active backend download for that id, since the status field can be coerced), and
// user-registered GGUFs that live outside models/ (presence can't be verified from
// the models/ scan).
function isStaleLocalModelRecord(model, presentFilenames, activeDownloadIds) {
  if (!model || isExternalModel(model)) return false
  if (model.status === 'downloading' || model.status === 'canceling') return false
  if (activeDownloadIds.has(model.id)) return false
  const path = String(model.model_path || '')
  if (!/^models[\\/]/.test(path)) return false
  const filename = path.split(/[\\/]/).filter(Boolean).pop() || ''
  return Boolean(filename) && !presentFilenames.has(filename)
}

function modelReadinessFromCurrent(currentModel, active, generationReady) {
  return {
    active,
    loaded_now: active,
    generation_ready: generationReady,
    tokenizer_status: active ? currentModel?.tokenizer?.status || null : null,
    tokenizer_model: active ? currentModel?.tokenizer?.model || null : null,
    tensor_ready: active ? Boolean(currentModel?.llama_tensors) : false,
    config_ready: active ? Boolean(currentModel?.llama_config) : false,
  }
}

function localRecordMatchesBackendId(record, backendModelId) {
  if (!record || !backendModelId) return false
  return backendModelId === record.id || backendModelId === record.runtime_model_name
}

/* The backend's own exact-artifact lane verdict for a model, looked up by the GGUF
   filename it resolves to.

   `/api/models/local` reports `lane_class`, which `classify_model_lane()` computes
   from BOTH the real header architecture AND `filename_is_supported_exact_row()`.
   The Models page has trusted it since lib/modelLanes.js, for a documented reason:
   a compatibility row id concatenates the model's `general.finetune` token that the
   release filename may omit (`qwen3_0_6b_instruct_q8_0` vs `Qwen3-0.6B-Q8_0.gguf`),
   so filename-based identity matching demotes genuinely supported rows.

   Carrying it here is what lets the chat gate reach the same verdict. Without it the
   SAME FILE read as "Local chat ready" when the engine auto-loaded it at startup
   (the id comes from GGUF metadata and matches the row) but "Runtime ready, support
   gated" plus an "unverified, no parity guarantee" banner when the app loaded it
   (the id is the filename, which does not) — one file, two contradictory claims,
   decided by nothing more than which code path issued the load. */
function localFactsByFilename(localList) {
  const byFilename = new Map()
  for (const entry of localList?.models || []) {
    if (entry?.filename) byFilename.set(entry.filename, entry)
  }
  return byFilename
}

function modelFilename(model) {
  return String(model?.model_path || model?.hf_filename || model?.id || '')
    .split(/[\\/]/)
    .filter(Boolean)
    .pop() || ''
}

function modelMatchesHealthActive(model, health) {
  return modelRuntimeIdMatches(model, { active_model_id: health?.active_model_id })
}

function modelFromLocalRecord(record, health, currentModel, apiBase) {
  const active = modelMatchesHealthActive(record, health)
  const generationReady = active && Boolean(health?.generation_ready)
  const quantLabel = active ? getLoadedModelQuantLabel(currentModel) : record.quant
  const modelPath = active ? getModelPath(currentModel) || record.model_path : record.model_path
  return {
    ...record,
    name: resolveLoadedModelDisplayName({ fallbackName: record.name, modelPath, quantLabel }),
    status: generationReady ? 'ready' : record.status,
    model_path: modelPath,
    api_base: apiBase,
    install_error: active ? null : record.install_error,
    load_error: active ? null : record.load_error,
    loaded_now: active,
    generation_ready: generationReady,
    camelid: modelReadinessFromCurrent(currentModel, active, generationReady),
  }
}

function modelFromBackend(item, health, currentModel, localRecord, apiBase) {
  const runtimeModelName = item.id
  const id = localRecord?.id || item.id
  const active = localRecordMatchesBackendId(localRecord, health?.active_model_id) || health?.active_model_id === item.id
  const generationReady = active && Boolean(health?.generation_ready)
  const tokenizer = active ? currentModel?.tokenizer : null
  const quantLabel = active ? getLoadedModelQuantLabel(currentModel) : null
  const modelPath = active ? getModelPath(currentModel) || localRecord?.model_path || '' : localRecord?.model_path || ''
  const fallbackName = localRecord?.name || item.name || item.id
  const capabilitySource = {
    ...item,
    ...(localRecord || {}),
    model_family: active ? health?.model_family : localRecord?.model_family,
    unsupported_runtime: active ? currentModel?.unsupported_runtime : localRecord?.unsupported_runtime,
  }
  const capabilityFields = modelCapabilityFields(capabilitySource, {
    active_model_id: health?.active_model_id,
    model_family: health?.model_family,
    current_model: currentModel,
  })
  const embeddingOnly = capabilityFields.task_kind === 'embedding'

  return {
    id,
    name: resolveLoadedModelDisplayName({ fallbackName, modelPath, quantLabel }),
    /* descriptive model-shape metadata from /v1/models (n_ctx_train etc.) —
       display/limits only, never a support signal (I2) */
    meta: item.meta || localRecord?.meta || null,
    provider_kind: 'local',
    status: generationReady || embeddingOnly ? 'ready' : localRecord?.status || 'registered',
    model_path: modelPath,
    runtime_model_name: runtimeModelName,
    source: localRecord?.source || 'Camelid local runtime',
    engine: 'camelid',
    quant: quantLabel || localRecord?.quant || null,
    architecture: localRecord?.architecture || item?.meta?.architecture || null,
    chat_capable: optionalBoolean(localRecord?.chat_capable),
    embedding_capable: capabilityFields.embedding_capable,
    generation_capable: capabilityFields.generation_capable,
    task_kind: capabilityFields.task_kind,
    task_tags: localRecord?.task_tags || [],
    unsupported_runtime: active ? currentModel?.unsupported_runtime || null : null,
    size_gb: localRecord?.size_gb || null,
    api_base: apiBase,
    api_key_configured: false,
    install_error: active ? null : localRecord?.install_error || null,
    load_error: active ? null : localRecord?.load_error || null,
    last_load_attempt_at: localRecord?.last_load_attempt_at || null,
    last_loaded_at: localRecord?.last_loaded_at || null,
    // Presence in /v1/models means resident, including a non-active embedding
    // sidecar. `camelid.active` remains the separate Chat routing identity.
    loaded_now: true,
    generation_ready: generationReady,
    camelid: modelReadinessFromCurrent(currentModel, active, generationReady),
  }
}

export function mergeModelLists({ modelItems, health, currentModel, localModels, apiBase, localFacts }) {
  const localRecords = localModels.map(normalizeLocalModelRecord).filter(Boolean)
  const activeFilename = modelFilename({ model_path: getModelPath(currentModel) })
  const residentMatches = matchResidentItemsToLocalRecords({
    items: modelItems,
    records: localRecords,
    activeModelId: health?.active_model_id,
    activeFilename,
  })
  const byId = new Map()
  localRecords.forEach((record) => {
    byId.set(record.id, modelFromLocalRecord(record, health, currentModel, apiBase))
  })
  modelItems.forEach((item, index) => {
    const localRecord = residentMatches[index]
    const mergedModel = modelFromBackend(item, health, currentModel, localRecord, apiBase)
    byId.set(mergedModel.id, mergedModel)
  })
  // Stamp the backend's authoritative per-file capability + lane facts onto
  // whichever saved/runtime identity resolves to that file.
  const factsByFilename = localFacts || new Map()
  const runtimeProjection = {
    active_model_id: health?.active_model_id,
    model_family: health?.model_family,
    current_model: currentModel,
  }
  return [...byId.values()]
    .map((model) => {
      const facts = factsByFilename.get(modelFilename(model))
      const enriched = facts
        ? {
            ...model,
            architecture: facts.architecture || model.architecture || null,
            chat_capable: optionalBoolean(facts.chat_capable),
            embedding_capable: optionalBoolean(facts.embedding_capable),
            generation_capable: optionalBoolean(facts.generation_capable),
            task_tags: Array.isArray(facts.task_tags) ? facts.task_tags : model.task_tags || [],
            lane_class: facts.lane_class || model.lane_class,
          }
        : model
      return { ...enriched, ...modelCapabilityFields(enriched, runtimeProjection) }
    })
    .sort(compareModelsByName)
}

function nowIso() {
  return new Date().toISOString()
}

function makeId(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return `${prefix}-${crypto.randomUUID()}`
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function getErrorMessage(error, fallback = 'Request failed.') {
  if (!error) return fallback
  if (typeof error === 'string') return error
  return error?.body?.error?.message || error?.error?.message || error?.message || fallback
}

function getBackendErrorCode(error) {
  return error?.body?.error?.code || error?.payload?.error?.code || error?.error?.code || error?.code || ''
}

function isTypedUnsupportedBackendError(code, message) {
  const normalized = `${code} ${message}`.toLowerCase()
  return normalized.includes('unsupported')
    || normalized.includes('not_supported')
    || normalized.includes('cpu_weight_materialization_exceeds_budget')
    || normalized.includes('exceeds_budget')
}

function getGuardrailErrorMessage(error, fallback = 'Request failed.') {
  const message = getErrorMessage(error, fallback)
  const code = getBackendErrorCode(error)
  if (!isTypedUnsupportedBackendError(code, message)) return message
  const codeCopy = code ? ` (${code})` : ''
  /* This is the message a refused action shows. It is the worst possible place
     for contract vocabulary: the reader just had something fail and needs to
     know what to do, not which files the claim came from. The error code is
     kept — it is the one part worth quoting in a bug report. */
  return `Camelid can't do this with the current model${codeCopy}: ${message}. This combination isn't verified yet — see the Compatibility page for what is.`
}

async function fetchJson(pathOrUrl, options = {}) {
  const response = await fetch(pathOrUrl, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
  })
  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  if (!response.ok) {
    const error = new Error(typeof body === 'string' ? body : getErrorMessage(body, response.statusText))
    error.status = response.status
    error.body = body
    throw error
  }
  return body
}

function makeDashboard({ health, models, currentModel, capabilities, conversations, memories, apiBase }) {
  return {
    app: 'camelid',
    api_base: apiBase,
    health,
    capabilities,
    conversations,
    memories,
    models,
    runtime: {
      engine: normalizeEngineName(health?.engine),
      api_surface: health?.api_surface || 'full',
      // Which build is answering. `runtime` is an explicit projection of health, not a
      // pass-through, so a field absent here is invisible to every view no matter what
      // /v1/health serializes.
      version: health?.version || null,
      build: health?.build || null,
      loaded_now: Boolean(health?.loaded_now ?? health?.active_model_id),
      active_model_id: health?.active_model_id || null,
      generation_ready: Boolean(health?.generation_ready),
      active_context_length: Number(health?.active_context_length) || null,
      max_prompt_tokens: Number(health?.max_prompt_tokens) || null,
      max_generation_tokens: Number(health?.max_generation_tokens) || null,
      model_family: optionalString(health?.model_family),
      vision_ready: Boolean(health?.vision_ready),
      // Optional future/runtime hint. Older servers omit it, in which case the
      // bounded estimator default matches Camelid's current image-token ceiling.
      vision_token_allowance: Number(health?.vision_token_allowance) || null,
      q8_runtime: health?.q8_runtime || null,
      // Required for lane-scoped support truth. All Gemma 4 serve variants use
      // backend="gemma4-runtime"; this discriminator plus projected Ghost
      // component/marker health identifies the supported Windows CUDA lane.
      gemma4_serve_lane: optionalString(health?.gemma4_serve_lane),
      ...executionRuntimeFields(health),
      status: health?.ok ? 'online' : 'offline',
      // Absolute path of the running binary, disclosed by loopback servers only.
      // Captured while the engine is UP so the offline banner can still name a
      // command that actually runs after it goes down.
      executable: health?.executable || null,
      // Address the engine is really bound to. A restart command without this
      // falls back to the default port, which either collides with whatever
      // owns it or comes up somewhere this tab is not looking.
      listen_addr: health?.listen_addr || null,
      api_base: apiBase,
      current_model: currentModel || null,
    },
    stats: {
      conversation_count: conversations.length,
      memory_count: memories.length,
      model_count: models.length,
    },
  }
}

export function useDashboardData({ showNotice, clearNotice }) {
  const [dashboard, setDashboard] = useState(null)
  const [authRequired, setAuthRequired] = useState(false)
  const [apiBase, setApiBaseState] = useState(getApiBase)
  const [tab, setTab] = useState(getInitialTab)
  const [selectedConversationId, setSelectedConversationIdState] = useState(getInitialConversationId)
  const [selectedModelId, setSelectedModelId] = useState(getInitialModelId)
  const [search, setSearch] = useState('')
  const [memorySearch, setMemorySearch] = useState('')
  const [composer, setComposer] = useState('')
  const [newChatTitle, setNewChatTitle] = useState('')
  const [sending, setSending] = useState(false)
  const [webResearchEnabled, setWebResearchEnabledState] = useState(readWebResearchEnabled)
  const [webResearchStatus, setWebResearchStatus] = useState({ phase: 'idle', sourceCount: 0, conversationId: null })
  // Opt-in parity receipts: sends the next message non-streaming with
  // camelid_receipt:true so the response carries a verifiable receipt.
  const [receiptMode, setReceiptMode] = useState(false)
  /* Opt-in token inspection: sends the next message non-streaming with
     logprobs:true so the reply carries the model's per-token scores. Captured
     DURING the decode that produces the visible reply — never reconstructed by a
     second generation, which would describe a different generation.

     Deliberately held OUTSIDE the conversation: a 120-token reply carries roughly
     414 bytes per token at depth 5 and 1.4 KB per token at depth 20 — two orders
     of magnitude more than the reply text. Persisting that would march a
     conversation into the localStorage quota, and persistConversations swallows
     the quota error, so the failure would silently stop saving the WHOLE
     conversation rather than just this record. Session-scoped by design; the
     panel says so and offers a download. */
  const [inspectMode, setInspectMode] = useState(false)
  /* Constrained decoding is a per-turn choice like inspection, and for the same
     reason: the engine refuses a constraint on a streaming request, so the turn
     has to be composed non-streaming before it is sent. */
  const [structuredMode, setStructuredMode] = useState(STRUCTURED_MODES.OFF)
  const [structuredSchema, setStructuredSchema] = useState(DEFAULT_SCHEMA)
  const [structuredGrammar, setStructuredGrammar] = useState(DEFAULT_GRAMMAR)
  const [structuredRecords, setStructuredRecords] = useState({})
  const [tokenInspections, setTokenInspections] = useState({})
  /* Tool definitions are a per-session editing surface, not a persisted setting:
     they are a developer probe against the loaded model, and a stale definition
     silently shaping a later conversation would be worse than retyping one. */
  const [toolsEnabled, setToolsEnabled] = useState(false)
  const [toolsText, setToolsText] = useState(DEFAULT_TOOLS)
  /* Signatures of every call already requested in this conversation, so a model
     that ignores a tool result and re-asks can be named rather than looping
     invisibly. Verified live: Llama 3.2 3B does exactly this. */
  const [toolCallSignatures, setToolCallSignatures] = useState({})
  // Opt-in thinking mode (experimental — NOT parity-locked): sends
  // camelid_enable_thinking:true so the model emits its own <think>…</think>
  // reasoning. Default OFF so chat stays on the parity-locked thinking-DISABLED
  // rendering; only the leading reasoning trace is evidenced vs llama.cpp.
  const [thinkingMode, setThinkingMode] = useState(false)
  const [stoppingGeneration, setStoppingGeneration] = useState(false)
  const [loadingModelId, setLoadingModelId] = useState('')
  const [pendingChat, setPendingChat] = useState(null)
  const [registerForm, setRegisterForm] = useState({ id: '', name: '', model_path: '', runtime_model_name: '' })
  const [externalForm, setExternalForm] = useState({ id: '', name: '', source: 'Hosted API', api_base: 'https://api.example/v1', api_key: '', model_name: '' })
  const [localModels, setLocalModels] = useState(() => readJsonStorage(LOCAL_MODELS_STORAGE_KEY, []).map(normalizeLocalModelRecord).filter(Boolean))
  const [localConversations, setLocalConversations] = useState(() => normalizeStoredConversations(readJsonStorage(CONVERSATIONS_STORAGE_KEY, []), { clearStaleStreaming: true }))
  const [localMemories, setLocalMemories] = useState(() => readJsonStorage(MEMORIES_STORAGE_KEY, []))

  const localModelsRef = useRef(localModels)
  const localConversationsRef = useRef(localConversations)
  const localMemoriesRef = useRef(localMemories)
  const selectedConversationIdRef = useRef(selectedConversationId)
  const activeChatRequestRef = useRef(null)

  useEffect(() => {
    localModelsRef.current = localModels
  }, [localModels])

  useEffect(() => {
    localConversationsRef.current = localConversations
  }, [localConversations])

  useEffect(() => {
    localMemoriesRef.current = localMemories
  }, [localMemories])

  useEffect(() => {
    selectedConversationIdRef.current = selectedConversationId
  }, [selectedConversationId])

  const setSelectedConversationId = (valueOrUpdater) => {
    const next = typeof valueOrUpdater === 'function'
      ? valueOrUpdater(selectedConversationIdRef.current)
      : valueOrUpdater
    selectedConversationIdRef.current = next
    setSelectedConversationIdState(next)
    return next
  }

  const setWebResearchEnabled = (enabled) => {
    const next = Boolean(enabled)
    setWebResearchEnabledState(next)
    persistWebResearchEnabled(next)
    return next
  }

  const normalizedApiBase = normalizeApiBase(apiBase)
  const updateConversationsState = (updater) => {
    setLocalConversations((current) => {
      const next = normalizeStoredConversations(typeof updater === 'function' ? updater(current) : updater)
      localConversationsRef.current = next
      return next
    })
  }

  const persistConversations = (updater) => {
    setLocalConversations((current) => {
      const next = normalizeStoredConversations(typeof updater === 'function' ? updater(current) : updater)
      localConversationsRef.current = next
      writeJsonStorage(CONVERSATIONS_STORAGE_KEY, next)
      return next
    })
  }

  const persistMemories = (updater) => {
    setLocalMemories((current) => {
      const next = typeof updater === 'function' ? updater(current) : updater
      localMemoriesRef.current = next
      writeJsonStorage(MEMORIES_STORAGE_KEY, next)
      return next
    })
  }

  const persistLocalModels = (updater) => {
    const nextModels = (typeof updater === 'function' ? updater(localModelsRef.current) : updater)
      .map(normalizeLocalModelRecord)
      .filter(Boolean)
      .sort(compareModelsByName)
    localModelsRef.current = nextModels
    writeJsonStorage(LOCAL_MODELS_STORAGE_KEY, nextModels)
    setLocalModels(nextModels)
    return nextModels
  }

  const loadDashboard = async ({ silent = false, localModelsOverride = null } = {}) => {
    let observedHealth = null
    try {
      const currentLocalModels = localModelsOverride || localModelsRef.current
      const currentLocalConversations = localConversationsRef.current
      const currentLocalMemories = localMemoriesRef.current
      const healthStartedAt = performance.now()
      const health = await fetchJson(`${normalizedApiBase}/v1/health`).then((result) => {
        recordHealthPoll({ ok: true, latencyMs: performance.now() - healthStartedAt })
        observedHealth = result
        return result
      }, (error) => {
        recordHealthPoll({ ok: false, latencyMs: performance.now() - healthStartedAt })
        throw error
      })
      const [modelList, capabilities, downloads, localList] = await Promise.all([
        fetchJson(`${normalizedApiBase}/v1/models`),
        fetchJson(`${normalizedApiBase}/api/capabilities`).catch(() => null),
        fetchJson(`${normalizedApiBase}/api/models/catalog/downloads`).catch(() => []),
        // Authoritative on-disk presence; null (not []) on failure so a transient
        // error can't be read as "nothing present" and wrongly drop records.
        fetchJson(`${normalizedApiBase}/api/models/local`).catch(() => null),
      ])

      let modelsUpdated = false
      const updatedLocalModels = currentLocalModels.map((model) => {
        if (model.status === 'downloading') {
          const dl = downloads.find((d) => d.id === model.id)
          if (dl) {
            const progress = dl.total_bytes > 0 ? Math.round((dl.bytes_downloaded / dl.total_bytes) * 100) : 0
            if (model.bytes_downloaded !== dl.bytes_downloaded || model.status !== dl.status) {
              modelsUpdated = true
              let newStatus = 'downloading'
              let installError = null
              if (dl.status === 'completed') {
                newStatus = 'registered'
              } else if (dl.status === 'failed') {
                newStatus = 'failed'
                installError = 'Download failed'
              }
              return {
                ...model,
                status: newStatus,
                bytes_downloaded: dl.bytes_downloaded,
                total_bytes: dl.total_bytes,
                progress: newStatus === 'registered' ? 100 : progress,
                install_error: installError,
                updated_at: nowIso(),
              }
            }
          } else {
            // The download is not in the backend's active list. Do NOT assume it
            // completed — a model is only marked ready when the backend reports
            // status 'completed' (handled above). Leaving it 'downloading' avoids a
            // premature "Downloaded"/loadable state when the list momentarily omits
            // it (e.g. right after starting, or after a server restart).
            return model
          }
        }
        return model
      })

      // Reconcile persisted local records against the backend's authoritative
      // models/ scan: drop catalog/download records whose GGUF is no longer on disk
      // so a stale localStorage entry can't keep presenting a model as downloaded or
      // loadable. Only runs when the scan succeeded (localList is non-null).
      let reconciledLocalModels = updatedLocalModels
      let droppedStale = false
      if (localList && Array.isArray(localList.models)) {
        const presentFilenames = new Set(localList.models.map((m) => m.filename))
        const activeDownloadIds = new Set((downloads || []).map((d) => d.id))
        const kept = updatedLocalModels.filter(
          (model) => !isStaleLocalModelRecord(model, presentFilenames, activeDownloadIds),
        )
        if (kept.length !== updatedLocalModels.length) {
          reconciledLocalModels = kept
          droppedStale = true
        }

        // Add on-disk GGUFs the saved list does not track yet — e.g. files placed
        // directly into models/ instead of downloaded through the catalog. Without
        // this they appear in the backend scan (and the lane view) but cannot be
        // loaded from the main UI ("Choose a saved local model before loading it").
        // Matched by filename against existing records so catalog downloads are not
        // duplicated.
        const trackedFilenames = new Set(
          reconciledLocalModels
            .map((m) =>
              String(m.model_path || m.hf_filename || '')
                .split(/[\\/]/)
                .filter(Boolean)
                .pop(),
            )
            .filter(Boolean),
        )
        const additions = []
        for (const entry of localList.models) {
          if (!entry?.filename || trackedFilenames.has(entry.filename)) continue
          const rec = normalizeLocalModelRecord({
            id: entry.filename,
            name: entry.filename,
            model_path: `models/${entry.filename}`,
            status: 'registered',
            quant: entry.quantization,
            architecture: entry.architecture,
            chat_capable: entry.chat_capable,
            embedding_capable: entry.embedding_capable,
            generation_capable: entry.generation_capable,
            task_tags: entry.task_tags,
          })
          if (rec) additions.push(rec)
        }
        if (additions.length) {
          reconciledLocalModels = [...additions, ...reconciledLocalModels]
          droppedStale = true
        }
      }

      let activeLocalModels = currentLocalModels
      if (modelsUpdated || droppedStale) {
        activeLocalModels = persistLocalModels(reconciledLocalModels)
      }

      const currentModel = health?.active_model_id
        ? await fetchJson(`${normalizedApiBase}/api/models/current`).catch(() => null)
        : null
      const modelItems = Array.isArray(modelList?.data) ? modelList.data : []
      const nextModels = mergeModelLists({
        modelItems,
        health,
        currentModel,
        localModels: activeLocalModels,
        apiBase: normalizedApiBase,
        localFacts: localFactsByFilename(localList),
      })
      const nextDashboard = makeDashboard({
        health,
        models: nextModels,
        currentModel,
        capabilities,
        conversations: currentLocalConversations,
        memories: currentLocalMemories,
        apiBase: normalizedApiBase,
      })
      setAuthRequired(false)
      setDashboard(nextDashboard)
      if (!silent) clearNotice()
      setSelectedConversationId((current) => {
        if (current === NEW_CHAT_SENTINEL) return current
        if (!currentLocalConversations.length) return null
        if (current && currentLocalConversations.some((conversation) => conversation.id === current)) return current
        return currentLocalConversations[0]?.id || null
      })
      setSelectedModelId((current) => {
        if (!nextModels.length) return ''
        const currentModel = current ? nextModels.find((model) => model.id === current) : null
        const activeModel = health?.active_model_id ? nextModels.find((model) => modelRuntimeIdMatches(model, { active_model_id: health.active_model_id })) : null
        const activeModelChatGate = activeModel ? getChatGateState(capabilities, activeModel, nextDashboard.runtime) : null
        const currentModelChatGate = currentModel ? getChatGateState(capabilities, currentModel, nextDashboard.runtime) : null
        const chatUnlockedModel = nextModels.find((model) => getChatGateState(capabilities, model, nextDashboard.runtime).chatUnlocked) || null
        const firstChatModel = nextModels.find((model) => isGenerationCapableModel(model, nextDashboard.runtime)) || null

        // The chat API can only use the backend's active model. If a previous browser
        // selection points at an inactive saved model, snap back to the runtime model
        // instead of leaving the composer looking ready for the wrong row.
        if (activeModelChatGate?.chatUnlocked && current !== activeModel.id) return activeModel.id
        if (currentModelChatGate?.chatUnlocked) return current
        if (activeModel && !activeModelChatGate?.embeddingOnly) return activeModel.id
        if (currentModel && !currentModelChatGate?.embeddingOnly) return current
        return chatUnlockedModel?.id || firstChatModel?.id || ''
      })
    } catch (error) {
      const requiresAuth = error?.status === 401
      setAuthRequired(requiresAuth)
      const fallbackHealth = observedHealth || { ok: false, engine: 'camelid', generation_ready: false, active_model_id: null }
      const fallbackDashboard = makeDashboard({
        health: fallbackHealth,
        models: mergeModelLists({
          modelItems: [],
          health: fallbackHealth,
          currentModel: null,
          localModels: localModelsOverride || localModelsRef.current,
          apiBase: normalizedApiBase,
        }),
        currentModel: null,
        capabilities: null,
        conversations: localConversationsRef.current,
        memories: localMemoriesRef.current,
        apiBase: normalizedApiBase,
      })
      setDashboard(fallbackDashboard)
      if (!silent) {
        showNotice(
          requiresAuth
            ? 'Camelid is reachable, but this browser needs the server API key.'
            : `Could not reach Camelid at ${normalizedApiBase}: ${getErrorMessage(error)}`,
          requiresAuth ? 'info' : 'error',
        )
      }
    }
  }

  useEffect(() => {
    loadDashboard()
    /* Self-scheduling refresh with backoff: 2.5s while the backend answers,
       doubling to a 20s ceiling on consecutive failures, reset on success.
       Reachability history comes from these real polls (telemetryLog). */
    let cancelled = false
    let timer = null
    let delay = 2500
    const tick = async () => {
      if (cancelled) return
      await loadDashboard({ silent: true })
      const last = getTelemetrySnapshot().health.at(-1)
      delay = last?.ok === false ? Math.min(delay * 2, 20000) : 2500
      if (!cancelled) timer = setTimeout(tick, delay)
    }
    timer = setTimeout(tick, delay)
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [normalizedApiBase])

  useEffect(() => {
    if (typeof window === 'undefined' || !VALID_TABS.has(tab)) return
    appStorage.setItem(TAB_STORAGE_KEY, tab)
  }, [tab])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedConversationId) appStorage.removeItem(SELECTED_CONVERSATION_STORAGE_KEY)
    else appStorage.setItem(SELECTED_CONVERSATION_STORAGE_KEY, selectedConversationId)
  }, [selectedConversationId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedModelId) appStorage.removeItem(SELECTED_MODEL_STORAGE_KEY)
    else appStorage.setItem(SELECTED_MODEL_STORAGE_KEY, selectedModelId)
  }, [selectedModelId])

  const conversations = localConversations.length ? localConversations : dashboard?.conversations || []
  const memories = localMemories.length ? localMemories : dashboard?.memories || []
  const models = dashboard?.models || []
  const runtime = dashboard?.runtime

  const selectedConversation = useMemo(
    () => resolveSelectedConversation(conversations, selectedConversationId),
    [conversations, selectedConversationId],
  )

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId)
      || models.find((model) => isGenerationCapableModel(model, runtime))
      || models[0],
    [models, runtime, selectedModelId],
  )
  const selectedModelChatGate = getChatGateState(dashboard?.capabilities, selectedModel, runtime)
  /* Whether this engine's contract permits token inspection at all. Surfaced so
     the composer control can render GUARDED rather than appearing live and
     silently recording nothing — a toggle that reads "on" while contributing no
     request fields is exactly the caveated-live surface I3 forbids. */
  const inspectionSupported = readInspectionContract(dashboard?.capabilities).nonStreamingSupported
  const structuredContract = readStructuredOutputContract(dashboard?.capabilities)
  const structuredSupported = structuredContract.nonStreamingSupported
  const structuredReadiness = structuredOutputReadiness({
    enabled: structuredMode !== STRUCTURED_MODES.OFF,
    mode: structuredMode,
    contract: structuredContract,
    schemaText: structuredSchema,
    grammarText: structuredGrammar,
  })
  const toolContract = readToolContract(dashboard?.capabilities)
  const toolCapability = readModelToolCapability(dashboard?.capabilities, selectedModel, runtime)
  const toolsReadiness = toolReadiness({ enabled: toolsEnabled, contract: toolContract, capability: toolCapability, toolsText })
  const selectedModelRunnable = selectedModelChatGate.chatUnlocked
  // Experimental lane: loaded + generation-ready implemented model that is NOT a
  // supported row. Enables a weaker chat affordance; never the supported badge.
  const selectedModelExperimental = selectedModelChatGate.experimentalUnlocked
  const pendingConversation = pendingChat?.conversationId
    && (selectedConversation?.id === pendingChat.conversationId || selectedConversationId === pendingChat.conversationId)
    ? pendingChat
    : null

  const filteredConversations = useMemo(() => {
    if (!search.trim()) return conversations
    const q = search.toLowerCase()
    return conversations.filter((conversation) =>
      conversation.title.toLowerCase().includes(q)
      || conversation.messages.some((message) => message.content.toLowerCase().includes(q)),
    )
  }, [conversations, search])

  const filteredMemories = useMemo(() => {
    if (!memorySearch.trim()) return memories
    const q = memorySearch.toLowerCase()
    return memories.filter((memory) =>
      memory.title.toLowerCase().includes(q)
      || memory.body.toLowerCase().includes(q)
      || memory.scope.toLowerCase().includes(q),
    )
  }, [memories, memorySearch])

  const latestAssistantMessage = useMemo(
    () => [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant'),
    [selectedConversation],
  )

  const createConversationRecord = async ({ manualTitle = '', silent = false } = {}) => {
    const conversation = {
      id: makeId('conversation'),
      title: manualTitle || 'New conversation',
      model_id: selectedModelId || models[0]?.id || null,
      messages: manualTitle ? [] : [{ id: makeId('message'), role: 'assistant', content: 'Conversation created. Load a Camelid model and send a prompt when ready.', created_at: nowIso() }],
      created_at: nowIso(),
      updated_at: nowIso(),
    }
    persistConversations((current) => [conversation, ...current])
    setSelectedConversationId(conversation.id)
    setTab('chat')
    setNewChatTitle('')
    if (!silent) showNotice(manualTitle ? 'Conversation created locally.' : 'Conversation created locally.', 'success')
    return conversation
  }

  const createConversation = async () => {
    try {
      await createConversationRecord({ manualTitle: newChatTitle.trim() })
    } catch (error) {
      showNotice(error.message || 'Could not create the conversation.', 'error')
    }
  }

  const ensureConversation = async () => (
    shouldCreateConversationForSend(selectedConversation, selectedConversationId)
      ? createConversationRecord({ silent: true })
      : selectedConversation
  )

  /* Regenerate / edit-and-resend: truncate the thread at the given user
     message and resend through the normal gate-checked sendMessage path. */
  const resendFromMessage = async (messageId, editedContent = null) => {
    const conversation = selectedConversation
    const index = (conversation?.messages || []).findIndex((message) => message.id === messageId)
    if (index === -1) return
    const target = conversation.messages[index]
    if (target.role !== 'user') return
    const content = String(editedContent ?? target.content ?? '').trim()
    if (!content) return
    await sendMessage({
      overrideContent: content,
      overrideImage: target.image || null,
      truncateFromMessageId: messageId,
    })
  }

  const stopGeneration = () => {
    if (!activeChatRequestRef.current || stoppingGeneration) return false
    setStoppingGeneration(true)
    activeChatRequestRef.current.abort()
    return true
  }

  /* options.overrideContent: replace the composer draft in both transcript and
     request (regenerate / edit-and-resend). options.requestContent: replace only
     the current request payload while preserving what the user typed in the
     transcript. options.truncateFromMessageId drops that message and everything
     after it first. The gate checks below are identical for every path. */
  const sendMessage = async (options = {}) => {
    const {
      overrideContent = null,
      overrideImage = null,
      requestContent = null,
      citations = [],
      truncateFromMessageId = null,
    } = options
    const draftContent = overrideContent ?? composer
    if (!draftContent.trim()) return
    // Supported rows chat through the full gate. Implemented-but-unsupported rows
    // chat through the weaker EXPERIMENTAL lane (every turn marked unverified). Only
    // a model that is not generation-ready at all is fully blocked.
    if (!selectedModelRunnable && !selectedModelExperimental) {
      showNotice('The selected model isn’t ready to generate yet.', 'error')
      return
    }

    const messageContent = draftContent.trim()
    const requestMessageContent = String(requestContent ?? messageContent).trim()
    if (!requestMessageContent) return
    setSending(true)
    let activeConversationId = null
    let assistantId = null
    let chatLifecycleId = null
    let webResearchMs = null
    /* Hoisted so the finally below can always halt the display-pacing loop,
       including on abort and error paths. */
    let stopPacing = () => {}
    let pendingAssistantPatch = null
    let pendingAssistantFrame = null

    try {
      const conversation = await ensureConversation()
      activeConversationId = conversation.id
      // Fresh chats start from the __new__ sentinel. Select the real conversation immediately
      // so the main thread renders the same streaming message object as the sidebar preview.
      setSelectedConversationId(conversation.id)
      const userMessage = {
        id: makeId('message'),
        role: 'user',
        content: messageContent,
        ...(overrideImage ? { image: overrideImage } : {}),
        model_id: selectedModelId,
        created_at: nowIso(),
      }
      const truncateIndex = truncateFromMessageId
        ? (conversation.messages || []).findIndex((message) => message.id === truncateFromMessageId)
        : -1
      const baseMessages = truncateIndex >= 0
        ? (conversation.messages || []).slice(0, truncateIndex)
        : (conversation.messages || [])
      const history = [...baseMessages, userMessage]
        // A token budget can end entirely inside a model-hidden channel. Keep
        // that diagnostic turn in the transcript, but never feed an empty
        // assistant message back into the next model prompt.
        .filter((message) => {
          if (message.role === 'user') return true
          if (message.role !== 'assistant') return false
          const content = String(message.content || '').trim()
          return content && content !== '(empty response)'
        })
        .filter((message) => !message.content.startsWith('Conversation created.'))
      // The current Prism vision lanes accept one image. Retain every attachment in
      // the local transcript, but send only the most recent one so follow-ups
      // keep image context and attaching a replacement does not form an
      // unsupported multi-image request.
      let activeImageIndex = -1
      history.forEach((message, index) => {
        if (message.image?.data_url) activeImageIndex = index
      })
      const requestHistory = history.map(({ id, role, content, image }, index) => {
        const payloadContent = id === userMessage.id ? requestMessageContent : content
        return {
          role,
          content: index === activeImageIndex
            ? [
                { type: 'image_url', image_url: { url: image.data_url } },
                { type: 'text', text: payloadContent },
              ]
            : payloadContent,
        }
      })
      let requestMessages = applyLocalChatPolicy(requestHistory)

      /* Send-time compaction. Trims only this payload -- the stored transcript
         is untouched -- so a wrong call costs the user nothing. Reads the same
         preference store and runs the same pure trim the composer's meter
         previews with, so the panel cannot advertise a trim that does not
         happen here. */
      const compactionReserve = applyGemma4GhostChatTokenCap(
        getModelMaxTokens(selectedModelId),
        runtime?.gemma4_serve_lane,
      )
      const compactionBudget = composeContextBudget({
        contextLength: runtime?.active_context_length || modelContextLength(selectedModel),
        promptTokens: requestMessages.reduce(
          (sum, message) => sum + estimateTokenCount(
            typeof message?.content === 'string'
              ? message.content
              : JSON.stringify(message?.content ?? ''),
          ),
          0,
        ),
        reservedTokens: compactionReserve,
        warnAtPercent: AUTO_COMPACT_THRESHOLD_PERCENT,
      })
      const compactionIntent = resolveCompactionIntent(selectedConversationIdRef.current)
      const sendCompaction = applySendCompaction(requestMessages, {
        enabled: compactionIntent.enabled,
        forced: compactionIntent.forced,
        filledPercent: compactionBudget?.filledPercent ?? 0,
      })
      requestMessages = sendCompaction.messages

      const sendGate = getChatGateState(dashboard?.capabilities, selectedModel, runtime)
      const requestModelId = getRuntimeRequestModelId(selectedModel, runtime, selectedModelId)
      const preparedVideoArtifact = isGemma4Mtp12SegmentedVideoOptedIn()
        ? readGemma4Mtp12PreparedSegments()
        : null
      const segmentedVideoRigRequested = Boolean(preparedVideoArtifact) && shouldUseGemma4Mtp12TargetVerifiedRender({
        runtime,
        requestModelId,
        compatibilityRowId: sendGate.hint?.target?.id,
        research: { sources: [{}, {}] },
        receiptMode,
        videoRigOptIn: isGemma4Mtp12TargetVerifiedVideoOptedIn(),
      })

      const estimateResearchPromptTokens = (candidateMessages) => estimateWebResearchChatTokens(
        candidateMessages,
        { visionTokenAllowance: runtime?.vision_token_allowance },
      )
      const baseContextFit = activeRuntimeContextFit(requestMessages, {
        activeContextLength: runtime?.active_context_length,
        maxPromptTokens: runtime?.max_prompt_tokens,
        estimateTokenCount: estimateResearchPromptTokens,
      })
      if (!segmentedVideoRigRequested && baseContextFit.unfit) {
        showNotice(baseContextFit.message, 'error')
        return
      }

      setPendingChat({ conversationId: conversation.id, content: messageContent, modelId: selectedModelId })
      if (overrideContent === null) setComposer('')

      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? {
              ...item,
              model_id: selectedModelId,
              messages: truncateIndex >= 0 ? [...baseMessages, userMessage] : [...(item.messages || []), userMessage],
              updated_at: nowIso(),
            }
          : item
      )))

      // Gemma 4 26B is not advertised as function-tool-capable by Camelid, so
      // Web UI research is a deterministic preflight: resolve linked/current
      // sources first, then give the ordinary chat request a leading, untrusted
      // evidence message. No tools/tool_choice payload is sent to the model.
      const bitNetB158Chat = isBitNetB158ChatModel(selectedModel, runtime, requestModelId)
      const responseLimitModelIds = [...new Set([
        requestModelId,
        selectedModel?.id,
        selectedModelId,
      ].filter(Boolean))]
      const explicitResponseLimitModelId = responseLimitModelIds.find(hasExplicitMaxTokensSetting) || ''
      const responseLimitModelId = explicitResponseLimitModelId || requestModelId
      const requestedMaxTokens = applyGemma4GhostChatTokenCap(
        applyGemma4ChatTokenFloor(
          applyBitNetFreshChatTokenCap(
            localChatMaxTokens(history, responseLimitModelId),
            {
              bitNetB158: bitNetB158Chat,
              hasExplicitSetting: Boolean(explicitResponseLimitModelId),
            },
          ),
          sendGate.hint?.target?.family,
        ),
        runtime?.gemma4_serve_lane,
      )
      const admittedRequestMaxTokens = effectiveGenerationTokenLimit(
        requestedMaxTokens,
        runtime?.max_generation_tokens,
      ) || requestedMaxTokens
      let requestMaxTokens = admittedRequestMaxTokens
      const requestController = new AbortController()
      activeChatRequestRef.current = requestController
      const researchPlan = classifyWebResearchNeed(messageContent)
      let researchResult = null
      let researchFailure = ''
      // The client planner avoids a network round trip for definite local-only
      // prompts. Once called, the backend remains authoritative about whether
      // research triggered and which evidence is safe to return.
      if (webResearchEnabled && researchPlan.needed) {
        setWebResearchStatus({ phase: 'researching', sourceCount: 0, conversationId: conversation.id })
        const researchStartedAt = performance.now()
        try {
          researchResult = await requestWebResearch(normalizedApiBase, messageContent, {
            signal: requestController.signal,
            plan: researchPlan,
          })
          const researchElapsedMs = performance.now() - researchStartedAt
          webResearchMs = researchResult?.triggered ? researchElapsedMs : null
          if (segmentedVideoRigRequested) {
            // The exact prompt and live source response remain visible and
            // auditable, but the model inputs are the separately prepared,
            // hash-gated <=512-position sections. Do not reject this mode by
            // trying to fit the monolithic prompt/evidence into a 512 runtime.
            requestMessages = [{
              role: 'user',
              content: 'Prepared Web research multi-pass synthesis. Exact user request and live sources are attached to the visible turn; model execution uses the hash-gated bounded section messages.',
            }]
            requestMaxTokens = preparedVideoArtifact.total_tokens
            setWebResearchStatus({
              phase: researchResult?.status === 'failed' ? 'failed' : 'complete',
              sourceCount: Array.isArray(researchResult?.sources) ? researchResult.sources.length : 0,
              conversationId: conversation.id,
            })
          } else {
          const configuredContext = runtime?.active_context_length || modelContextLength(selectedModel)
          const researchBudget = deriveWebResearchPromptBudget({
            contextLength: configuredContext,
            serverMaxPromptTokens: runtime?.max_prompt_tokens,
            serverMaxGenerationTokens: runtime?.max_generation_tokens,
            requestedMaxTokens: requestMaxTokens,
            messages: requestMessages,
            research: researchResult,
            estimateTokenCount: estimateResearchPromptTokens,
            queryText: messageContent,
          })
          const fittedResearch = fitWebResearchContext(requestMessages, researchResult, {
            maxPromptTokens: researchBudget.maxPromptTokens,
            estimateTokenCount: estimateResearchPromptTokens,
            queryText: messageContent,
          })
          researchResult = fittedResearch.research
          requestMessages = fittedResearch.messages
          const fittedReplyBudget = deriveFittedWebResearchReplyBudget({
            contextLength: configuredContext,
            serverMaxGenerationTokens: runtime?.max_generation_tokens,
            requestedMaxTokens: admittedRequestMaxTokens,
            messages: requestMessages,
            estimateTokenCount: estimateResearchPromptTokens,
            safetyMargin: researchBudget.safetyMargin,
          })
          const fittedContextFit = activeRuntimeContextFit(requestMessages, {
            activeContextLength: runtime?.active_context_length,
            maxPromptTokens: runtime?.max_prompt_tokens,
            estimateTokenCount: estimateResearchPromptTokens,
            safetyMargin: researchBudget.safetyMargin,
          })
          if (fittedContextFit.unfit
            || (Number.isFinite(fittedReplyBudget.replyReserve) && fittedReplyBudget.replyReserve <= 0)) {
            setPendingChat(null)
            showNotice(
              fittedContextFit.message
                || 'The fetched evidence leaves no safe room for a reply in the active runtime context. Shorten this chat or load the model with a larger context.',
              'error',
            )
            return
          }
          if (Number.isFinite(fittedReplyBudget.replyReserve)) {
            requestMaxTokens = Math.floor(fittedReplyBudget.replyReserve)
          }
          setWebResearchStatus({
            phase: researchResult?.status === 'failed' ? 'failed' : 'complete',
            sourceCount: Array.isArray(researchResult?.sources) ? researchResult.sources.length : 0,
            conversationId: conversation.id,
          })
          }
        } catch (error) {
          const researchElapsedMs = performance.now() - researchStartedAt
          if (error?.name === 'AbortError') throw error
          if (researchPlan.needed) {
            webResearchMs = researchElapsedMs
            researchFailure = error?.message || 'Web research was unavailable.'
            researchResult = {
              triggered: true,
              reason: researchPlan.reason,
              query: researchPlan.query,
              sources: [],
              warnings: [],
            }
            showNotice('Web research was unavailable. Camelid will answer without web sources.', 'info')
          }
          setWebResearchStatus({ phase: 'failed', sourceCount: 0, conversationId: conversation.id })
        }
      }
      const researchAtSend = webResearchMetadata(researchResult, researchFailure)
      const promptTokenEstimate = estimateResearchPromptTokens(requestMessages)
      const targetVerifiedRender = shouldUseGemma4Mtp12TargetVerifiedRender({
        runtime,
        requestModelId,
        compatibilityRowId: sendGate.hint?.target?.id,
        // Use the fitted source groups, not flattened display metadata where
        // two chunks from one repository could look like two sources.
        research: researchResult,
        receiptMode,
        videoRigOptIn: isGemma4Mtp12TargetVerifiedVideoOptedIn(),
      })
      const segmentedTargetVerifiedRender = targetVerifiedRender
        && isGemma4Mtp12SegmentedVideoOptedIn()
      const preparedSegmentedSynthesis = segmentedTargetVerifiedRender
        ? preparedVideoArtifact
        : null
      if (segmentedTargetVerifiedRender && !preparedSegmentedSynthesis) {
        throw new Error('Prepared Web research synthesis artifact is missing or failed its exact schema/source gate')
      }

      const requestStartedAt = performance.now()
      // Fresh per-token decode trace for this generation (auditable backing for
      // the live tok/s readout; read out after the stream completes).
      if (typeof window !== 'undefined') window.__tpsTrace = []
      const lifecycleId = beginRequest({ kind: 'chat', endpoint: '/v1/chat/completions', modelId: requestModelId })
      chatLifecycleId = lifecycleId
      let firstContentEmitted = false
      let firstTokenAt = null
      let decodeStartTokens = 0
      let latestNativeSegmentRate = null
      let lastProgressAt = 0
      // The private, prepared research lane completes independently verified
      // sections. Hold two completed sections as a reservoir, then reveal only
      // received bytes at a frame-timed cadence. The 240 chars/s cadence is
      // separately gated against exact verified output tokens by the capture
      // rig; ordinary chats retain the low-lag pacer.
      const smoothSegmentedPacing = segmentedTargetVerifiedRender
      const pacer = createPacerState(smoothSegmentedPacing
        ? { steadyCharsPerSecond: 240 }
        : undefined)
      /* The pacer is driven by its own animation frame loop rather than by token
         arrival, so the text keeps flowing smoothly between bursty SSE chunks
         and advances once per display refresh (120Hz where the panel supports
         it). Arrival still records the real stream for the lag bound; only the
         DISPLAY is paced, and metrics never read from it. */
      let latestReceivedContent = ''
      let lastPacedContent = ''
      let pacingFrame = null
      let segmentedPacingReady = !smoothSegmentedPacing
      let completedVerifiedSegments = 0
      let pacingSettle = null
      let streamTransportComplete = false
      stopPacing = () => {
        if (pacingFrame !== null && typeof window !== 'undefined') window.cancelAnimationFrame(pacingFrame)
        pacingFrame = null
      }
      const pacingTick = () => {
        pacingFrame = null
        const fullContent = latestReceivedContent
        const paced = paceStep(pacer, fullContent, performance.now())
        if (paced !== lastPacedContent) {
          lastPacedContent = paced
          markAssistantStreamState({ content: paced })
        }
        if (paceHasPendingText(paced, fullContent)) {
          startPacing()
        } else {
          // If an unexpectedly long verifier prefill exhausts the received-text
          // reservoir, keep the stock response row communicative instead of
          // looking frozen. The next real content delta restores "Streaming
          // response"; successful paced takes normally never enter this state.
          if (smoothSegmentedPacing && !streamTransportComplete && lastPacedContent) {
            markAssistantStreamState({ streaming_phase: 'thinking' })
          }
          if (pacingSettle) {
            const { resolve, timeout } = pacingSettle
            pacingSettle = null
            window.clearTimeout(timeout)
            resolve()
          }
        }
      }
      const startPacing = () => {
        if (pacingFrame === null && typeof window !== 'undefined') pacingFrame = window.requestAnimationFrame(pacingTick)
      }
      const waitForPacingToSettle = () => {
        if (!paceHasPendingText(lastPacedContent, latestReceivedContent)) return Promise.resolve()
        return new Promise((resolve, reject) => {
          const timeout = window.setTimeout(() => {
            pacingSettle = null
            reject(new Error('Smooth received-text reservoir did not settle before its safety deadline'))
          }, 12_000)
          pacingSettle = { resolve, timeout }
          startPacing()
        })
      }
      assistantId = makeId('message')
      /* Snapshot of the support claim that was active when this send left the
         composer: row id + status only (never paths) so the message footer can
         cite the exact contract row that gated this generation. */
      const supportRowAtSend = sendGate.chatMode !== 'experimental' && sendGate.hint?.target
        ? { id: sendGate.hint.target.id, status: sendGate.hint.target.status, supported: sendGate.contractSupported }
        : null
      // An experimental artifact may resemble a verified compatibility row,
      // but it must not render that other artifact's "Verified" chip beside
      // its own unverified verdict. Exact verified-but-limited rows still keep
      // their contract status in support_row.
      const experimentalLaneAtSend = sendGate.chatMode === 'experimental'
      const assistantMessageBase = {
        id: assistantId,
        role: 'assistant',
        content: '',
        model_id: selectedModelId,
        model_name: selectedModel?.name || selectedModelId,
        support_row: supportRowAtSend,
        experimental_lane: experimentalLaneAtSend,
        created_at: nowIso(),
        tokens_in_per_sec: null,
        tokens_out_per_sec: null,
        generated_token_ids: [],
        timings_ms: null,
        // The prompt is fixed when the request starts, so show its estimated
        // token count immediately. Output usage advances with the stream and
        // is replaced by backend-reported totals when they arrive.
        usage: {
          prompt_tokens: promptTokenEstimate,
          completion_tokens: 0,
          total_tokens: promptTokenEstimate,
        },
        usage_source: 'client_estimate',
        streaming: true,
        streaming_phase: targetVerifiedRender ? 'generating' : 'preparing',
        synthesis_mode: segmentedTargetVerifiedRender ? 'prepared_web_research_multi_pass_lossless' : null,
        first_byte_ms: null,
        first_event_ms: null,
        first_content_ms: null,
        web_research_ms: webResearchMs,
        ...(Array.isArray(citations) && citations.length ? { citations } : {}),
        ...(researchAtSend ? { web_research: researchAtSend } : {}),
      }
      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? {
              ...item,
              title: item.title === 'New conversation' ? messageContent.slice(0, 64) : item.title,
              messages: [...(item.messages || []), assistantMessageBase],
              updated_at: nowIso(),
            }
          : item
      )))
      setPendingChat(null)

      // The generic BitNet runnable is a greedy lane. Its experimental status
      // must not cause the browser to advertise Prism's sampling controls that
      // this model does not use.
      const useExperimentalSampling = sendGate.chatMode === 'experimental' && !bitNetB158Chat
      /* Token inspection rides on THIS request rather than a later replay, so the
         scores describe the decode the reader is about to see. The contract is
         resolved once per send and reused for both the stream flag and the
         request fields. */
      const inspectionContract = readInspectionContract(dashboard?.capabilities)
      const inspecting = !targetVerifiedRender
        && inspectionForcesNonStreaming({ enabled: inspectMode, contract: inspectionContract })
      /* A constraint and stream:true is a hard 400, and the streaming decode job
         never builds a grammar state — that route refusal is the only thing
         standing between a streamed constrained request and silently
         unconstrained output. Derived from one predicate with the request fields
         so the two cannot drift. */
      const sendStructuredContract = readStructuredOutputContract(dashboard?.capabilities)
      const constraining = !targetVerifiedRender
        && structuredOutputForcesNonStreaming({ enabled: true, mode: structuredMode, contract: sendStructuredContract })
        && structuredOutputReadiness({ enabled: true, mode: structuredMode, contract: sendStructuredContract, schemaText: structuredSchema, grammarText: structuredGrammar }).ready
      const baseRequestBody = {
        model: requestModelId,
        messages: requestMessages,
        // Supported rows stay greedy (temperature 0) — their behavior is parity-
        // locked. Experimental rows have no parity contract and small models loop
        // badly under greedy decoding, so they sample for usable output. BitNet's
        // runnable lane is explicitly greedy even while its row is experimental.
        temperature: useExperimentalSampling ? 0.7 : 0,
        ...(useExperimentalSampling ? { top_p: 0.95, top_k: 20, min_p: 0 } : {}),
        max_tokens: requestMaxTokens,
        ...contractSamplingOverrides(dashboard?.capabilities?.api_features, requestModelId),
        ...(thinkingMode && !bitNetB158Chat ? { camelid_enable_thinking: true } : {}),
      }

      let targetVerifiedDraftTokenIds = []
      let targetVerifiedSegments = []
      let targetVerifiedPlannerMs = null
      let targetVerifiedPlannerCamelid = null
      if (targetVerifiedRender) {
        const plannerStartedAt = performance.now()
        const segmentPlans = [{ messages: requestMessages, maxTokens: requestMaxTokens }]
        if (segmentedTargetVerifiedRender) {
          targetVerifiedSegments = preparedSegmentedSynthesis.segments
        }
        if (!segmentedTargetVerifiedRender) {
        for (let segmentIndex = 0; segmentIndex < segmentPlans.length; segmentIndex += 1) {
          const segmentPlan = segmentPlans[segmentIndex]
          const plannerResponse = await fetch(`${normalizedApiBase}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: requestController.signal,
            // Every draft is freshly generated for this turn. In segmented
            // mode each compact, grounded section stays inside the backend's
            // 512-position qualification envelope.
            body: JSON.stringify({
              ...baseRequestBody,
              messages: segmentPlan.messages,
              max_tokens: segmentPlan.maxTokens,
              stream: false,
            }),
          })
          let plannerPayload = null
          try {
            plannerPayload = await plannerResponse.json()
          } catch {
            // The typed error below remains useful if a proxy returned non-JSON.
          }
          if (!plannerResponse.ok) {
            throw new Error(plannerPayload?.error?.message || `Gemma 4 section ${segmentIndex + 1} planning failed with HTTP ${plannerResponse.status}`)
          }
          if (String(plannerPayload?.model || '') !== String(requestModelId)) {
            throw new Error(`Gemma 4 planning resolved model ${plannerPayload?.model || '(missing)'}, expected ${requestModelId}`)
          }
          if (plannerPayload?.choices?.[0]?.finish_reason !== 'stop') {
            throw new Error(`Gemma 4 section ${segmentIndex + 1} did not finish naturally (${plannerPayload?.choices?.[0]?.finish_reason || 'missing finish reason'})`)
          }
          const tokenIds = plannerPayload?.camelid?.generated_token_ids || []
          if (!Array.isArray(tokenIds)
            || tokenIds.length === 0
            || tokenIds.length !== Number(plannerPayload?.usage?.completion_tokens)
            || tokenIds.some((token) => !Number.isInteger(token) || token < 0)) {
            throw new Error(`Gemma 4 section ${segmentIndex + 1} did not return a complete authoritative token-id draft`)
          }
          targetVerifiedDraftTokenIds = tokenIds
          targetVerifiedPlannerCamelid = plannerPayload?.camelid
            ? {
                mtp12: plannerPayload.camelid.mtp12 || null,
                timings_ms: plannerPayload.camelid.timings_ms || null,
              }
            : null
        }
        }
        targetVerifiedPlannerMs = performance.now() - plannerStartedAt
        if (segmentedTargetVerifiedRender) targetVerifiedPlannerMs = null
        updateConversationsState((current) => current.map((item) => (
          item.id === conversation.id
            ? {
                ...item,
                messages: (item.messages || []).map((message) => (
                  message.id === assistantId
                    ? {
                        ...message,
                        streaming_phase: 'generating',
                        planner_ms: targetVerifiedPlannerMs,
                        prepared_segment_count: segmentedTargetVerifiedRender ? targetVerifiedSegments.length : null,
                      }
                    : message
                )),
                updated_at: nowIso(),
              }
            : item
        )))
      }
      const response = await fetch(`${normalizedApiBase}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: requestController.signal,
        body: JSON.stringify({
          ...baseRequestBody,
          // Receipts and token inspection only attach to non-streaming
          // responses; the JSON fallback in readStreamingChatCompletion handles
          // that shape. The engine returns a typed 400 for logprobs with
          // stream:true, so `inspecting` MUST also clear the stream flag — the
          // two conditions are derived from one source in tokenInspection.js
          // rather than restated, so they cannot drift apart.
          stream: targetVerifiedRender || !(receiptMode || inspecting || constraining),
          // Ask for the authoritative token count in the final stream chunk.
          // Without it the client can only ESTIMATE from visible content, which
          // undercounts badly on a thinking model: LFM2 emits its reasoning as
          // `reasoning_content`, so a reply that is mostly reasoning looked like
          // almost no tokens and the tok/s readout reported a fraction of the
          // real rate.
          ...(targetVerifiedRender || !(receiptMode || inspecting || constraining) ? { stream_options: { include_usage: true } } : {}),
          ...(!targetVerifiedRender && receiptMode ? { camelid_receipt: true } : {}),
          /* Contributes nothing unless the contract permits inspection on this
             engine, so a guarded row simply never reaches the wire. */
          ...(targetVerifiedRender ? {} : inspectionRequestFields({ enabled: inspectMode, contract: inspectionContract })),
          ...(targetVerifiedRender ? {} : structuredOutputRequestFields({ enabled: true, mode: structuredMode, contract: sendStructuredContract, schemaText: structuredSchema, grammarText: structuredGrammar })),
          /* Tool calling is supported on BOTH the streaming and non-streaming
             paths, so unlike receipts and constrained decoding it does not force
             the turn off the stream. Contributes nothing unless the engine row
             AND the loaded model both carry the capability. */
          ...(targetVerifiedRender ? {} : toolRequestFields({ enabled: toolsEnabled, contract: toolContract, capability: toolCapability, toolsText })),
          ...(targetVerifiedRender ? {
            // The private verifier contract is exact: its output allowance is
            // the complete fresh draft, not the planner's larger upper bound.
            max_tokens: segmentedTargetVerifiedRender
              ? targetVerifiedSegments.reduce((sum, segment) => sum + segment.token_ids.length, 0)
              : targetVerifiedDraftTokenIds.length,
            ...(segmentedTargetVerifiedRender
              ? {
                  n: 1,
                  camelid_target_verified_render_segments: targetVerifiedSegments,
                  camelid_expected_gguf_sha256: '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b',
                }
              : { camelid_target_verified_render_draft_token_ids: targetVerifiedDraftTokenIds }),
          } : {}),
        }),
      })
      const responseIsStreaming = response.ok && !response.headers.get('content-type')?.includes('application/json')
      const applyAssistantStreamPatch = (patch) => {
        updateConversationsState((current) => current.map((item) => (
          item.id === conversation.id
            ? {
                ...item,
                messages: (item.messages || []).map((message) => (
                  message.id === assistantId ? { ...message, ...patch } : message
                )),
                updated_at: nowIso(),
              }
            : item
        )))
      }
      const flushAssistantStreamPatch = () => {
        pendingAssistantFrame = null
        if (!pendingAssistantPatch) return
        const patch = pendingAssistantPatch
        pendingAssistantPatch = null
        applyAssistantStreamPatch(patch)
      }
      const markAssistantStreamState = (patch, { immediate = false } = {}) => {
        if (immediate) {
          pendingAssistantPatch = null
          if (pendingAssistantFrame !== null && typeof window !== 'undefined') {
            window.cancelAnimationFrame(pendingAssistantFrame)
            pendingAssistantFrame = null
          }
          applyAssistantStreamPatch(patch)
          return
        }
        pendingAssistantPatch = { ...(pendingAssistantPatch || {}), ...patch }
        if (pendingAssistantFrame === null && typeof window !== 'undefined') {
          pendingAssistantFrame = window.requestAnimationFrame(flushAssistantStreamPatch)
        }
      }
      if (responseIsStreaming) {
        markAssistantStreamState({ streaming_phase: 'generating' }, { immediate: true })
      }
      const streamed = await readStreamingChatCompletion(response, (_delta, fullContent, metrics) => {
        const now = performance.now()
        const liveElapsedMs = now - requestStartedAt
        /* Decode rate uses the REAL token count (one SSE content delta = one
           generated token in Camelid) only after first content. Web research
           and model TTFT are recorded separately and never dilute tok/s. */
        const realTokens = Number(metrics?.completionTokens) || 0
        const firstVisibleContent = !firstContentEmitted && Boolean(fullContent)
        if (firstVisibleContent) {
          firstContentEmitted = true
          emitFirstContent(lifecycleId, liveElapsedMs)
        }
        // The decode window opens at the first GENERATED token, never at request
        // start, so public-web research and model TTFT cannot dilute this rate.
        const decodedTokens = firstTokenAt === null ? 0 : Math.max(0, realTokens - decodeStartTokens)
        const decodeElapsedMs = firstTokenAt === null ? 0 : now - firstTokenAt
        // This live value is a browser-observed delivery rate computed from
        // real SSE token arrivals. The backend's native target-verifier clock
        // replaces it after each verified segment and in terminal diagnostics.
        const liveTps = responseIsStreaming && decodedTokens >= 4 && decodeElapsedMs >= 200
          ? tokensPerSecond(decodedTokens, decodeElapsedMs)
          : null
        if (typeof window !== 'undefined' && realTokens > 0) {
          if (!Array.isArray(window.__tpsTrace)) window.__tpsTrace = []
          window.__tpsTrace.push({ i: realTokens, t_ms: Math.round(decodeElapsedMs * 10) / 10, tps: liveTps != null ? Math.round(liveTps * 100) / 100 : null, delta: _delta })
        }
        if (performance.now() - lastProgressAt > 100) {
          lastProgressAt = performance.now()
          emitProgress(lifecycleId, { tokens: realTokens, tokensPerSec: liveTps })
        }
        /* Record what truly arrived; the pacing loop above owns the display. */
        latestReceivedContent = fullContent
        if (!segmentedPacingReady) {
          // Do not expose text until two sections have completed and can bridge
          // the final verifier prefill. The buffer contains model output already
          // received in this turn; no prepared response text is read.
          markAssistantStreamState({
            streaming_phase: completedVerifiedSegments > 0 ? 'thinking' : 'generating',
            tokens_in_per_sec: null,
            tokens_out_per_sec: null,
            usage: {
              prompt_tokens: promptTokenEstimate,
              completion_tokens: realTokens,
              total_tokens: promptTokenEstimate + realTokens,
            },
            usage_source: 'client_estimate',
          })
          return
        }
        // Browsers suspend requestAnimationFrame in a hidden tab. Commit the
        // first small text prefix synchronously so a healthy decode cannot sit
        // at "out 0". The rest of a large first network chunk remains paced;
        // later deltas keep frame-batched updates rather than rendering once
        // per generated token.
        const firstRenderedContent = !lastPacedContent && Boolean(fullContent)
        const displayedContent = firstRenderedContent
          ? paceFirstVisiblePrefix(pacer, fullContent, now)
          : paceStep(pacer, fullContent, performance.now()) || '…'
        const contentChanged = displayedContent !== lastPacedContent
        if (contentChanged) lastPacedContent = displayedContent
        markAssistantStreamState({
          ...(contentChanged ? { content: displayedContent } : {}),
          streaming_phase: 'streaming',
          tokens_in_per_sec: null,
          // The segmented lane exposes its backend-native completed-section
          // clock in the single stock streaming badge. Keep the footer rate
          // empty until terminal aggregate diagnostics so two clocks cannot
          // appear simultaneously or disagree on screen.
          tokens_out_per_sec: segmentedTargetVerifiedRender ? null : liveTps,
          streaming_native_segment_rate: latestNativeSegmentRate,
          usage: {
            prompt_tokens: promptTokenEstimate,
            completion_tokens: realTokens,
            total_tokens: promptTokenEstimate + realTokens,
          },
          usage_source: 'client_estimate',
        }, { immediate: firstVisibleContent })
        if (paceHasPendingText(displayedContent, fullContent)) startPacing()
      }, {
        estimateTokenCount,
        onStreamEvent(event) {
          if (event.type === 'segment') {
            completedVerifiedSegments += 1
            const segmentRate = Number(event.segment?.render_tokens_per_second)
            latestNativeSegmentRate = Number.isFinite(segmentRate) && segmentRate > 0 ? segmentRate : null
            // Start a new browser-arrival window for the next independently
            // verified section. The completed section's backend-native rate is
            // surfaced alongside it and is never confused with this clock.
            if (!segmentedPacingReady && completedVerifiedSegments >= 2) {
              segmentedPacingReady = true
              const initialContent = paceFirstVisiblePrefix(
                pacer,
                latestReceivedContent,
                performance.now(),
              )
              lastPacedContent = initialContent
              markAssistantStreamState({
                content: initialContent,
                streaming_phase: 'streaming',
                streaming_native_segment_rate: latestNativeSegmentRate,
                streaming_segment_index: Number(event.segment?.index) + 1,
              }, { immediate: true })
              if (paceHasPendingText(initialContent, latestReceivedContent)) startPacing()
            } else if (!segmentedPacingReady) {
              markAssistantStreamState({
                streaming_phase: 'thinking',
                streaming_native_segment_rate: latestNativeSegmentRate,
                streaming_segment_index: Number(event.segment?.index) + 1,
              }, { immediate: true })
            } else {
              markAssistantStreamState({
                streaming_native_segment_rate: latestNativeSegmentRate,
                streaming_segment_index: Number(event.segment?.index) + 1,
              }, { immediate: true })
            }
          }
          if (event.type === 'reasoning' || event.type === 'content') {
            if (firstTokenAt === null) {
              firstTokenAt = performance.now()
              decodeStartTokens = Number(event.completionTokens) || 0
            }
          }
          if (event.type === 'bytes' || event.type === 'role' || event.type === 'json_fallback') {
            markAssistantStreamState({
              streaming_phase: 'generating',
              first_byte_ms: event.firstByteMs ?? null,
              first_event_ms: event.firstEventMs ?? null,
            }, { immediate: true })
          }
          if (event.type === 'usage' && event.usage) {
            markAssistantStreamState({
              usage: event.usage,
              usage_source: 'backend',
            })
          }
        },
      })
      streamTransportComplete = true
      const streamCompletedAt = performance.now()
      if (smoothSegmentedPacing) {
        latestReceivedContent = streamed.content || ''
        await waitForPacingToSettle()
      }
      stopPacing()
      flushAssistantStreamPatch()
      const targetVerifiedRenderDiagnostics = readExactTargetVerifiedRender(streamed.camelid)
      const targetVerifiedSegmentedDiagnostics = readExactTargetVerifiedSegmentedRender(streamed.camelid)
      if (segmentedTargetVerifiedRender && !targetVerifiedSegmentedDiagnostics) {
        throw new Error('Gemma 4 segmented target verification did not reproduce every prepared section exactly')
      }
      if (targetVerifiedRender && !segmentedTargetVerifiedRender && !targetVerifiedRenderDiagnostics) {
        throw new Error('Gemma 4 target verification did not reproduce the fresh planning draft exactly')
      }
      const elapsedMs = streamCompletedAt - requestStartedAt
      const modelTtftMs = firstTokenAt === null ? null : firstTokenAt - requestStartedAt
      const decodeElapsedMs = firstTokenAt === null ? null : Math.max(0, streamCompletedAt - firstTokenAt)
      const completionTokenCount = streamed.completionTokens || estimateTokenCount(streamed.content)
      const decodedTokenCount = Math.max(0, completionTokenCount - decodeStartTokens)
      // Native facts replace browser-arrival estimates after the backend has
      // completed and attached the lossless verification record. A two-pass
      // turn is fail-closed above, so it never falls through to client timing.
      const nativeMtp12 = readTargetVerifiedMtp12(streamed.camelid)
        || readTargetVerifiedMtp12(targetVerifiedPlannerCamelid)
      /* Inspection lands in session state keyed by message id, NOT on the message
         object — see the state declaration for why persisting it is unsafe.
         Absence is recorded too: a lane that answers 200 without the key must be
         reported as an unmeasured position, never as a flat distribution. */
      if (inspecting) {
        const absence = inspectionAbsenceReason({
          requested: true,
          responded: true,
          hasLogprobs: Boolean(streamed.logprobs),
          streamed: responseIsStreaming,
        })
        setTokenInspections((current) => ({
          ...current,
          [assistantId]: { logprobs: streamed.logprobs || null, absence },
        }))
      }
      /* The strongest evidence a constrained reply can carry is a position whose
         emitted token was not the highest-scoring one — the returned scores are
         unmasked, so that is the mask visibly diverting the decode. It is only
         available when token inspection was ALSO requested, which is why the
         composer says so rather than the card inventing a weaker claim. */
      if (constraining) {
        const record = normalizeInspection(streamed.logprobs || null)
        setStructuredRecords((current) => ({
          ...current,
          [assistantId]: {
            content: streamed.content || '',
            mode: structuredMode,
            schemaText: structuredSchema,
            divertedPositions: record ? record.stats.offTopCount : null,
            greedy: !useExperimentalSampling,
          },
        }))
      }
      /* Record what was asked so a later identical request can be named. Keyed
         by conversation: a repeat only means something within one thread. */
      if (streamed.toolCalls && streamed.toolCalls.length) {
        const signatures = (normalizeToolCalls(streamed.toolCalls) || []).map(toolCallSignature).filter(Boolean)
        if (signatures.length) {
          setToolCallSignatures((current) => ({
            ...current,
            [conversation.id]: [...(current[conversation.id] || []), ...signatures],
          }))
        }
      }
      const assistantMessage = {
        ...assistantMessageBase,
        content: paceDrain(pacer, streamed.content || ''),
        tokens_in_per_sec: tokensPerSecond(promptTokenEstimate, modelTtftMs),
        tokens_out_per_sec: targetVerifiedRender
          ? (targetVerifiedSegmentedDiagnostics || targetVerifiedRenderDiagnostics).render_tokens_per_second
          : nativeMtp12?.decode_tokens_per_second
            ?? (responseIsStreaming ? tokensPerSecond(decodedTokenCount, decodeElapsedMs) : null),
        finish_reason: streamed.finishReason,
        elapsed_ms: elapsedMs,
        usage: streamed.usage || {
          prompt_tokens: promptTokenEstimate,
          completion_tokens: streamed.completionTokens || estimateTokenCount(streamed.content),
          total_tokens: promptTokenEstimate + (streamed.completionTokens || estimateTokenCount(streamed.content)),
        },
        /* Footer labeling: backend-reported usage vs client estimate (I4). */
        usage_source: streamed.usage ? 'backend' : 'client_estimate',
        camelid: streamed.camelid || null,
        planner_camelid: targetVerifiedPlannerCamelid,
        camelid_receipt: streamed.camelidReceipt || null,
        /* Persisted on the message: unlike per-token logprobs these are small, and
           a turn that ended in a tool request is meaningless without them. */
        tool_calls: streamed.toolCalls || null,
        planner_ms: targetVerifiedPlannerMs,
        target_verified_render: targetVerifiedRender,
        segmented_target_verified_render: segmentedTargetVerifiedRender,
        streaming: false,
        streaming_phase: null,
        first_byte_ms: streamed.firstByteMs ?? null,
        first_event_ms: streamed.firstEventMs ?? null,
        first_content_ms: modelTtftMs,
      }
      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? {
              ...item,
              messages: (item.messages || []).map((message) => (
                message.id === assistantId ? assistantMessage : message
              )),
              updated_at: nowIso(),
            }
          : item
      )))
      recordChatGeneration({
        lifecycleId,
        modelId: requestModelId,
        durationMs: elapsedMs,
        ttftMs: modelTtftMs,
        webResearchMs,
        promptTokens: assistantMessage.usage?.prompt_tokens,
        completionTokens: assistantMessage.usage?.completion_tokens,
        tokensPerSec: assistantMessage.tokens_out_per_sec,
        usageSource: assistantMessage.usage_source,
        outcome: streamed.finishReason === 'error' ? 'error' : 'ok',
        promptText: messageContent,
      })
    } catch (error) {
      const requestWasAborted = error?.name === 'AbortError'
      if (chatLifecycleId) {
        recordChatGeneration({
          lifecycleId: chatLifecycleId,
          modelId: getRuntimeRequestModelId(selectedModel, runtime, selectedModelId),
          durationMs: null,
          ttftMs: null,
          webResearchMs,
          outcome: requestWasAborted ? 'interrupted' : 'error',
          promptText: messageContent,
        })
      }
      const pendingPatchAtFailure = pendingAssistantPatch
      if (pendingAssistantFrame !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(pendingAssistantFrame)
        pendingAssistantFrame = null
      }
      pendingAssistantPatch = null
      if (activeConversationId && assistantId) {
        persistConversations((current) => current.map((item) => (
          item.id === activeConversationId
            ? {
                ...item,
                messages: (item.messages || []).map((message) => (
                  message.id === assistantId
                    ? (() => {
                        const patchedMessage = { ...message, ...(pendingPatchAtFailure || {}) }
                        return {
                          ...patchedMessage,
                          content: patchedMessage.content && patchedMessage.content !== '…' ? patchedMessage.content : '(generation stopped)',
                          finish_reason: requestWasAborted ? 'interrupted' : 'error',
                          streaming: false,
                          streaming_phase: null,
                        }
                      })()
                    : message
                )),
                updated_at: nowIso(),
              }
            : item
        )))
      }
      setPendingChat(null)
      if (requestWasAborted) {
        showNotice('Generation stopped.', 'info')
      } else {
        const errorMessage = getGuardrailErrorMessage(error, 'Local inference failed.')
        showNotice(errorMessage, 'error')
      }
    } finally {
      stopPacing()
      activeChatRequestRef.current = null
      setWebResearchStatus({ phase: 'idle', sourceCount: 0, conversationId: null })
      setStoppingGeneration(false)
      setSending(false)
      await loadDashboard({ silent: true })
    }
  }

  const renameConversation = async (id, nextTitle) => {
    const trimmedTitle = nextTitle.trim()
    if (!trimmedTitle) {
      showNotice('Conversation title cannot be empty.', 'error')
      return false
    }
    persistConversations((current) => current.map((conversation) => conversation.id === id ? { ...conversation, title: trimmedTitle, updated_at: nowIso() } : conversation))
    showNotice('Conversation title updated.', 'success')
    return true
  }

  const deleteConversation = async (id) => {
    persistConversations((current) => current.filter((conversation) => conversation.id !== id))
    if (selectedConversationId === id) setSelectedConversationId(null)
    showNotice('Conversation deleted locally.', 'success')
    return true
  }

  /* Settings → "Delete all conversations". Local-only data; memories and
     models are untouched. */
  const deleteAllConversations = async () => {
    persistConversations(() => [])
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    showNotice('All conversations deleted from this browser.', 'success')
    return true
  }

  const showNewChatLanding = () => {
    setTab('chat')
    setSelectedConversationId(NEW_CHAT_SENTINEL)
    setComposer('')
    setPendingChat(null)
  }

  const createMemory = async ({ title, body, scope = 'General' }) => {
    const memory = { id: makeId('memory'), title, body, scope, created_at: nowIso(), updated_at: nowIso() }
    persistMemories((current) => [memory, ...current])
    setTab('memory')
    showNotice('Memory saved in browser storage for this Camelid UI session.', 'success')
    return true
  }

  const updateMemory = async (id, changes, { successMessage = 'Memory updated.' } = {}) => {
    persistMemories((current) => current.map((memory) => memory.id === id ? { ...memory, ...changes, updated_at: nowIso() } : memory))
    if (successMessage) showNotice(successMessage, 'success')
    return true
  }

  const deleteMemory = async (id, { successMessage = 'Memory deleted.' } = {}) => {
    persistMemories((current) => current.filter((memory) => memory.id !== id))
    if (successMessage) showNotice(successMessage, 'success')
    return true
  }

  const saveToMemory = async () => {
    const latestAssistant = [...(selectedConversation?.messages || [])].reverse().find((message) => message.role === 'assistant')
    if (!latestAssistant) {
      showNotice('There is no assistant reply to save yet.', 'error')
      return
    }
    await createMemory({ title: `Saved from ${selectedConversation?.title?.trim() || 'Current chat'}`, body: latestAssistant.content, scope: 'Conversation' })
  }

  const installModel = async (id) => {
    const catalog = [
      {
        catalog_id: "llama32_1b_instruct_q8_0",
        name: "Llama 3.2 1B Instruct Q8_0",
        repo_id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
        filename: "Llama-3.2-1B-Instruct-Q8_0.gguf",
        size_bytes: 1346203104,
        quant: "Q8_0",
      },
      {
        catalog_id: "llama32_3b_instruct_q8_0",
        name: "Llama 3.2 3B Instruct Q8_0",
        repo_id: "unsloth/Llama-3.2-3B-Instruct-GGUF",
        filename: "Llama-3.2-3B-Instruct-Q8_0.gguf",
        size_bytes: 3422709216,
        quant: "Q8_0",
      },
      {
        catalog_id: "tinyllama_1_1b_chat_q8_0",
        name: "TinyLlama 1.1B Chat Q8_0",
        repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        size_bytes: 1169007424,
        quant: "Q8_0",
      },
      {
        catalog_id: "llama3_8b_instruct_q8_0",
        name: "Llama 3 8B Instruct Q8_0",
        repo_id: "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF",
        filename: "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        size_bytes: 8540846592,
        quant: "Q8_0",
      },
      {
        catalog_id: "gemma4_e4b_it_q8_0",
        name: "Gemma 4 E4B-It Q8_0",
        repo_id: "unsloth/gemma-4-E4B-it-GGUF",
        filename: "gemma-4-E4B-it-Q8_0.gguf",
        size_bytes: 8192951456,
        quant: "Q8_0",
      },
      {
        catalog_id: "gemma4_e2b_it_q8_0",
        name: "Gemma 4 E2B-It Q8_0",
        repo_id: "unsloth/gemma-4-E2B-it-GGUF",
        filename: "gemma-4-E2B-it-Q8_0.gguf",
        size_bytes: 5048350848,
        quant: "Q8_0",
      }
    ]

    const item = catalog.find((x) => x.catalog_id === id)
    if (item) {
      return installCatalogModel(item)
    } else {
      showNotice('Unknown model catalog item.', 'error')
      return false
    }
  }

  const installCatalogModel = async (item) => {
    try {
      showNotice(`Starting download for ${item.name}…`, 'info')
      await fetchJson(`${normalizedApiBase}/api/models/catalog/install`, {
        method: 'POST',
        body: JSON.stringify({
          catalog_id: item.catalog_id,
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
        }),
      })

      persistLocalModels((current) => {
        const record = {
          id: item.catalog_id,
          name: item.name,
          model_path: `models/${item.filename}`,
          status: 'downloading',
          bytes_downloaded: 0,
          total_bytes: item.size_bytes,
          progress: 0,
          hf_repo: item.repo_id,
          hf_filename: item.filename,
          quant: item.quant,
          created_at: nowIso(),
          updated_at: nowIso(),
        }
        return upsertLocalModelRecord(current, record)
      })

      showNotice(`Download started for ${item.name}!`, 'success')
      return true
    } catch (error) {
      showNotice(getErrorMessage(error, 'Could not start catalog download.'), 'error')
      return false
    }
  }

  const cancelModelDownload = async (id) => {
    try {
      showNotice('Canceling download…', 'info')
      await fetchJson(`${normalizedApiBase}/api/models/catalog/cancel`, {
        method: 'POST',
        body: JSON.stringify({ id }),
      })

      persistLocalModels((current) => {
        return current.map((model) => {
          if (model.id === id) {
            return {
              ...model,
              status: 'failed',
              install_error: 'Download canceled by user.',
            }
          }
          return model
        })
      })

      showNotice('Download canceled.', 'success')
      return true
    } catch (error) {
      showNotice(getErrorMessage(error, 'Could not cancel download.'), 'error')
      return false
    }
  }

  const activateModel = async (id) => {
    const model = models.find((item) => item.id === id) || localModels.find((item) => item.id === id)

    if (!model) {
      showNotice('Choose a saved local model before loading it.', 'error')
      return
    }
    if (isEmbeddingOnlyModel(model, runtime)) {
      showNotice(
        `${model.name || id} is an embedding-only model. Load it from Models for embeddings or reranking; it cannot replace the Chat model.`,
        'info',
      )
      return false
    }
    if (!isGenerationCapableModel(model, runtime)) {
      showNotice(`${model.name || id} is a companion model asset, not a standalone Chat model.`, 'info')
      return false
    }
    setSelectedModelId(id)
    if (isExternalModel(model)) {
      showNotice('Hosted API chat routing is planned but not wired yet. Keep using local GGUF loading for now.', 'info')
      return
    }
    if (!model.model_path) {
      showNotice('This model needs a local GGUF path before Camelid can load it.', 'error')
      return
    }
    if (modelRuntimeIdMatches(model, runtime) && runtime?.generation_ready) {
      showNotice('That model is already loaded and ready.', 'success')
      return
    }

    setLoadingModelId(id)
    showNotice(`Loading ${model.name || id} into Camelid…`, 'info')
    try {
      // replace: swap models rather than stacking a second resident copy, so the
      // fit preflight judges this model against a host that has released the last one.
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({
          id,
          path: model.model_path,
          filename: modelFilenameFromPath(model.model_path),
          replace: true,
        }),
      })
      const loadedId = loaded?.id || id
      const loadedPath = getModelPath(loaded) || model.model_path
      const ready = isLoadedModelGenerationReady(loaded)
      const fileType = getLoadedModelFileType(loaded)
      const quantLabel = getLoadedModelQuantLabel(loaded) || (fileType !== null && fileType !== undefined ? `file_type ${fileType}` : model.quant)
      const loadedRecord = {
        ...model,
        id: loadedId,
        model_path: loadedPath,
        status: ready ? 'ready' : 'registered',
        quant: quantLabel,
        install_error: null,
        load_error: null,
        last_load_attempt_at: nowIso(),
        last_loaded_at: nowIso(),
        updated_at: nowIso(),
      }
      const supportedByContract = isCompatibilitySupportedForModel(dashboard?.capabilities, loadedRecord)
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, loadedRecord))
      setSelectedModelId(loadedId)
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(
        ready
          ? supportedByContract
            ? 'Model loaded and verified — you can start chatting.'
            : 'Model loaded and running, but this build isn’t verified, so chat stays locked. The Compatibility page lists the verified builds.'
          : 'Model loaded, but it isn’t ready to generate yet. Give it a moment, or check the Models page for details.',
        ready && supportedByContract ? 'success' : 'info',
      )
    } catch (error) {
      const message = getGuardrailErrorMessage(error, 'Could not load that local GGUF into Camelid.')
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, {
        ...model,
        id,
        status: 'registered',
        install_error: null,
        load_error: message,
        last_load_attempt_at: nowIso(),
        updated_at: nowIso(),
      }))
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(message, 'error')
    } finally {
      setLoadingModelId((current) => current === id ? '' : current)
    }
  }

  const unloadCurrentModel = async () => {
    const activeModelId = runtime?.active_model_id
    if (!activeModelId) {
      showNotice('No model is loaded in Camelid right now.', 'info')
      return false
    }

    setLoadingModelId(activeModelId)
    showNotice(`Unloading ${activeModelId} from Camelid…`, 'info')
    try {
      await fetchJson(`${normalizedApiBase}/api/models/unload`, { method: 'POST' })
      await loadDashboard({ silent: true })
      showNotice('Camelid unloaded the current model. Local saved paths are unchanged.', 'success')
      return true
    } catch (error) {
      showNotice(getErrorMessage(error, 'Could not unload the current model.'), 'error')
      return false
    } finally {
      setLoadingModelId((current) => current === activeModelId ? '' : current)
    }
  }

  const connectExternalModel = async () => {
    showNotice('Hosted-provider setup is intentionally disabled until Camelid wires API routing.', 'info')
  }

  const registerModel = async () => {
    const name = registerForm.name.trim()
    const modelPath = registerForm.model_path.trim()
    const derivedId = registerForm.id.trim() || registerForm.runtime_model_name.trim() || name || modelPath.split('/').pop()?.replace(/\.gguf$/i, '') || ''
    if (!modelPath || !derivedId) {
      showNotice('Add a local GGUF path and model name before loading it into Camelid.', 'error')
      return
    }
    setLoadingModelId(derivedId)
    showNotice(`Loading ${name || derivedId} from the local GGUF path…`, 'info')
    try {
      const filename = modelFilenameFromPath(modelPath)
      const candidate = {
        id: derivedId,
        name: name || derivedId,
        model_path: modelPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || derivedId,
      }
      const loaded = await loadLocalModelForChat({
        apiBase: normalizedApiBase,
        filename,
        path: modelPath,
        modelId: derivedId,
        model: candidate,
      })
      if (!loaded.ok) throw new Error(loaded.message)
      const embeddingOnly = Boolean(loaded.embedding)
      const loadedId = loaded.id || derivedId
      const loadedRecord = {
        ...candidate,
        id: loadedId,
        name: name || loadedId,
        runtime_model_name: registerForm.runtime_model_name.trim() || loadedId,
        status: 'ready',
        embedding_capable: embeddingOnly,
        generation_capable: !embeddingOnly,
        task_kind: embeddingOnly ? 'embedding' : 'generation',
        install_error: null,
        load_error: null,
        last_load_attempt_at: nowIso(),
        last_loaded_at: nowIso(),
        updated_at: nowIso(),
      }
      const supportedByContract = isCompatibilitySupportedForModel(dashboard?.capabilities, loadedRecord)
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, loadedRecord))
      if (!embeddingOnly) setSelectedModelId(loadedId)
      setRegisterForm({ id: '', name: '', model_path: '', runtime_model_name: '' })
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(
        embeddingOnly
          ? 'Embedding model loaded as a sidecar. The current Chat model was left active.'
          : supportedByContract
            ? 'Model saved, loaded, and verified — you can start chatting.'
            : 'Model saved and running, but this build isn’t verified, so chat stays locked. The Compatibility page lists the verified builds.',
        embeddingOnly || supportedByContract ? 'success' : 'info',
      )
    } catch (error) {
      const message = getGuardrailErrorMessage(error, 'Could not load that local GGUF.')
      const nextLocalModels = persistLocalModels((current) => upsertLocalModelRecord(current, {
        id: derivedId,
        name: name || derivedId,
        model_path: modelPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || derivedId,
        status: 'registered',
        install_error: null,
        load_error: message,
        last_load_attempt_at: nowIso(),
        updated_at: nowIso(),
      }))
      setSelectedModelId(derivedId)
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(message, 'error')
    } finally {
      setLoadingModelId((current) => current === derivedId ? '' : current)
    }
  }

  return {
    dashboard,
    authRequired,
    tab,
    setTab,
    selectedConversationId,
    setSelectedConversationId,
    selectedModelId,
    setSelectedModelId,
    search,
    setSearch,
    memorySearch,
    setMemorySearch,
    composer,
    setComposer,
    newChatTitle,
    setNewChatTitle,
    sending,
    webResearchEnabled,
    setWebResearchEnabled,
    webResearchStatus,
    receiptMode,
    setReceiptMode,
    inspectMode,
    setInspectMode,
    tokenInspections,
    inspectionSupported,
    structuredMode,
    setStructuredMode,
    structuredSchema,
    setStructuredSchema,
    structuredGrammar,
    setStructuredGrammar,
    structuredRecords,
    structuredSupported,
    structuredReadiness,
    toolsEnabled,
    setToolsEnabled,
    toolsText,
    setToolsText,
    toolContract,
    toolCapability,
    toolsReadiness,
    toolCallSignatures,
    thinkingMode,
    setThinkingMode,
    loadingModelId,
    registerForm,
    setRegisterForm,
    externalForm,
    setExternalForm,
    conversations,
    memories,
    models,
    runtime,
    selectedConversation,
    selectedModel,
    selectedModelRunnable,
    selectedModelExperimental,
    filteredConversations,
    filteredMemories,
    latestAssistantMessage,
    pendingConversation,
    createConversation,
    showNewChatLanding,
    sendMessage,
    resendFromMessage,
    stopGeneration,
    saveToMemory,
    createMemory,
    updateMemory,
    deleteMemory,
    renameConversation,
    deleteConversation,
    deleteAllConversations,
    installModel,
    installCatalogModel,
    cancelModelDownload,
    activateModel,
    unloadCurrentModel,
    registerModel,
    connectExternalModel,
    loadDashboard,
    stoppingGeneration,
    apiBase,
    setApiBase: (value) => {
      const next = normalizeApiBase(value)
      setApiBaseState(next)
      if (typeof window !== 'undefined') appStorage.setItem(API_BASE_STORAGE_KEY, next)
    },
  }
}
