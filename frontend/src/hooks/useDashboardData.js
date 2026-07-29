import { useEffect, useMemo, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel, quantLabelFromGgufFileType } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { resolveLoadedModelDisplayName } from '../lib/loadedModelDisplay'
import { readStreamingChatCompletion } from '../lib/chatCompletionStream'
import { NEW_CHAT_SENTINEL, resolveSelectedConversation, shouldCreateConversationForSend } from '../lib/chatState'
import { normalizeStoredConversations } from '../lib/conversationStorage.js'
import { getRuntimeRequestModelId, isExternalModel, modelRuntimeIdMatches } from '../lib/modelState'
import { contractSamplingOverrides } from '../lib/samplingContract'
import { executionRuntimeFields } from '../lib/executionPlan'
import { createPacerState, paceDrain, paceStep } from '../lib/streamPacing'
import { getConfiguredMaxTokens as getModelMaxTokens } from '../lib/responseLimits'
import { beginRequest, emitFirstContent, emitProgress, getTelemetrySnapshot, recordChatGeneration, recordHealthPoll } from '../lib/telemetryLog'

const TAB_STORAGE_KEY = 'camelid.activeTab'
const SELECTED_CONVERSATION_STORAGE_KEY = 'camelid.selectedConversationId'
const SELECTED_MODEL_STORAGE_KEY = 'camelid.selectedModelId'
const LOCAL_MODELS_STORAGE_KEY = 'camelid.localModels'
const CONVERSATIONS_STORAGE_KEY = 'camelid.conversations'
const MEMORIES_STORAGE_KEY = 'camelid.memories'
const API_BASE_STORAGE_KEY = 'camelid.apiBase'
const VALID_TABS = new Set(['chat', 'workspace', 'library', 'downloads', 'api', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'compatibility', 'telemetry'])
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
  const saved = window.localStorage.getItem(TAB_STORAGE_KEY)
  return saved && VALID_TABS.has(saved) ? saved : 'chat'
}

function getInitialConversationId() {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(SELECTED_CONVERSATION_STORAGE_KEY) || null
}

function getInitialModelId() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY) || ''
}

function getApiBase() {
  if (typeof window === 'undefined') return DEFAULT_API_BASE
  return window.localStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE
}

function normalizeApiBase(value) {
  return (value || DEFAULT_API_BASE).trim().replace(/\/$/, '')
}

function readJsonStorage(key, fallback) {
  if (typeof window === 'undefined') return fallback
  try {
    const saved = window.localStorage.getItem(key)
    return saved ? JSON.parse(saved) : fallback
  } catch {
    window.localStorage.removeItem(key)
    return fallback
  }
}

function writeJsonStorage(key, value) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(key, JSON.stringify(value))
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

function estimateChatTokenCount(messages) {
  return (messages || []).reduce((total, message) => (
    total + estimateTokenCount(message?.role) + estimateTokenCount(message?.content) + 3
  ), 0)
}

const CODE_FIRST_SYSTEM_PROMPT = 'begin immediately with complete runnable code. No intro. Output one self-contained file unless the user asks otherwise. For Python, start exactly with ```python, include imports, and close the fence after the complete script. For Python games, prefer tkinter from the standard library over pygame, keep it compact, and include a complete runnable event loop. For HTML output ONE self-contained file. Never use external files or script src. Include inline <style> and inline <script> with working click/game logic before </body>. Start exactly with ```html then <!doctype html> and close the fence after </html>.'
const MAX_TOKENS_STORAGE_KEY = 'camelid.maxTokens'
const DEFAULT_CHAT_MAX_TOKENS = 8192

function getConfiguredMaxTokens() {
  if (typeof window === 'undefined') return DEFAULT_CHAT_MAX_TOKENS
  const value = Number.parseInt(window.localStorage.getItem(MAX_TOKENS_STORAGE_KEY) || '', 10)
  return Number.isFinite(value) && value >= 256 ? value : DEFAULT_CHAT_MAX_TOKENS
}

function looksLikeCodePrompt(value) {
  const text = String(value || '').toLowerCase()
  return /\b(code|build|create|implement|write|make)\b/.test(text)
    && /\b(html|html5|css|javascript|js|python|py|pygame|game|pacman|pacmac|tetris|app|component|page|website)\b/.test(text)
}

const SYSTEM_PROMPT_STORAGE_KEY = 'camelid.systemPrompt'

function getConfiguredSystemPrompt() {
  if (typeof window === 'undefined') return ''
  return String(window.localStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY) || '').trim()
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
function laneClassByFilename(localList) {
  const byFilename = new Map()
  for (const entry of localList?.models || []) {
    if (entry?.filename && entry?.lane_class) byFilename.set(entry.filename, entry.lane_class)
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

  return {
    id,
    name: resolveLoadedModelDisplayName({ fallbackName, modelPath, quantLabel }),
    /* descriptive model-shape metadata from /v1/models (n_ctx_train etc.) —
       display/limits only, never a support signal (I2) */
    meta: item.meta || localRecord?.meta || null,
    provider_kind: 'local',
    status: generationReady ? 'ready' : localRecord?.status || 'registered',
    model_path: modelPath,
    runtime_model_name: runtimeModelName,
    source: localRecord?.source || 'Camelid local runtime',
    engine: 'camelid',
    quant: quantLabel || localRecord?.quant || null,
    size_gb: localRecord?.size_gb || null,
    api_base: apiBase,
    api_key_configured: false,
    install_error: active ? null : localRecord?.install_error || null,
    load_error: active ? null : localRecord?.load_error || null,
    last_load_attempt_at: localRecord?.last_load_attempt_at || null,
    last_loaded_at: localRecord?.last_loaded_at || null,
    loaded_now: active,
    generation_ready: generationReady,
    camelid: modelReadinessFromCurrent(currentModel, active, generationReady),
  }
}

function mergeModelLists({ modelItems, health, currentModel, localModels, apiBase, laneClasses }) {
  const localRecords = localModels.map(normalizeLocalModelRecord).filter(Boolean)
  const byId = new Map()
  localRecords.forEach((record) => {
    byId.set(record.id, modelFromLocalRecord(record, health, currentModel, apiBase))
  })
  modelItems.forEach((item) => {
    const localRecord = localRecords.find((record) => localRecordMatchesBackendId(record, item.id)) || null
    const mergedModel = modelFromBackend(item, health, currentModel, localRecord, apiBase)
    byId.set(mergedModel.id, mergedModel)
  })
  // Stamp the backend's lane verdict onto whichever record resolves to that file.
  const lanes = laneClasses || new Map()
  return [...byId.values()]
    .map((model) => {
      const laneClass = lanes.get(modelFilename(model))
      return laneClass ? { ...model, lane_class: laneClass } : model
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
  return `Camelid refused this with a typed guardrail${codeCopy}: ${message}. This is not chat-ready support yet; check /api/capabilities and COMPATIBILITY.md before retrying.`
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
      loaded_now: Boolean(health?.loaded_now ?? health?.active_model_id),
      active_model_id: health?.active_model_id || null,
      generation_ready: Boolean(health?.generation_ready),
      q8_runtime: health?.q8_runtime || null,
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
  const [apiBase, setApiBaseState] = useState(getApiBase)
  const [tab, setTab] = useState(getInitialTab)
  const [selectedConversationId, setSelectedConversationIdState] = useState(getInitialConversationId)
  const [selectedModelId, setSelectedModelId] = useState(getInitialModelId)
  const [search, setSearch] = useState('')
  const [memorySearch, setMemorySearch] = useState('')
  const [composer, setComposer] = useState('')
  const [newChatTitle, setNewChatTitle] = useState('')
  const [sending, setSending] = useState(false)
  // Opt-in parity receipts: sends the next message non-streaming with
  // camelid_receipt:true so the response carries a verifiable receipt.
  const [receiptMode, setReceiptMode] = useState(false)
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
    try {
      const currentLocalModels = localModelsOverride || localModelsRef.current
      const currentLocalConversations = localConversationsRef.current
      const currentLocalMemories = localMemoriesRef.current
      const healthStartedAt = performance.now()
      const [health, modelList, capabilities, downloads, localList] = await Promise.all([
        fetchJson(`${normalizedApiBase}/v1/health`).then((result) => {
          recordHealthPoll({ ok: true, latencyMs: performance.now() - healthStartedAt })
          return result
        }, (error) => {
          recordHealthPoll({ ok: false, latencyMs: performance.now() - healthStartedAt })
          throw error
        }),
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
        laneClasses: laneClassByFilename(localList),
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

        // The chat API can only use the backend's active model. If a previous browser
        // selection points at an inactive saved model, snap back to the runtime model
        // instead of leaving the composer looking ready for the wrong row.
        if (activeModelChatGate?.chatUnlocked && current !== activeModel.id) return activeModel.id
        if (currentModelChatGate?.chatUnlocked) return current
        if (activeModel) return activeModel.id
        if (currentModel) return current
        return chatUnlockedModel?.id || nextModels[0]?.id || ''
      })
    } catch (error) {
      const fallbackDashboard = makeDashboard({
        health: { ok: false, engine: 'camelid', generation_ready: false, active_model_id: null },
        models: mergeModelLists({
          modelItems: [],
          health: { ok: false, engine: 'camelid', generation_ready: false, active_model_id: null },
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
      if (!silent) showNotice(`Could not reach Camelid at ${normalizedApiBase}: ${getErrorMessage(error)}`, 'error')
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
    window.localStorage.setItem(TAB_STORAGE_KEY, tab)
  }, [tab])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedConversationId) window.localStorage.removeItem(SELECTED_CONVERSATION_STORAGE_KEY)
    else window.localStorage.setItem(SELECTED_CONVERSATION_STORAGE_KEY, selectedConversationId)
  }, [selectedConversationId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!selectedModelId) window.localStorage.removeItem(SELECTED_MODEL_STORAGE_KEY)
    else window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, selectedModelId)
  }, [selectedModelId])

  const conversations = localConversations.length ? localConversations : dashboard?.conversations || []
  const memories = localMemories.length ? localMemories : dashboard?.memories || []
  const models = dashboard?.models || []
  const runtime = dashboard?.runtime

  const selectedConversation = useMemo(
    () => resolveSelectedConversation(conversations, selectedConversationId),
    [conversations, selectedConversationId],
  )

  const selectedModel = useMemo(() => models.find((model) => model.id === selectedModelId) || models[0], [models, selectedModelId])
  const selectedModelChatGate = getChatGateState(dashboard?.capabilities, selectedModel, runtime)
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
    await sendMessage({ overrideContent: content, truncateFromMessageId: messageId })
  }

  const stopGeneration = () => {
    if (!activeChatRequestRef.current || stoppingGeneration) return false
    setStoppingGeneration(true)
    activeChatRequestRef.current.abort()
    return true
  }

  /* options.overrideContent: send this text instead of the composer draft
     (regenerate / edit-and-resend). options.truncateFromMessageId: drop that
     message and everything after it first, so the resent turn replaces the
     old tail instead of duplicating it. The gate checks below are identical
     for every path — resends cannot bypass the fail-closed chat gate. */
  const sendMessage = async (options = {}) => {
    const { overrideContent = null, truncateFromMessageId = null } = options
    const draftContent = overrideContent ?? composer
    if (!draftContent.trim()) return
    // Supported rows chat through the full gate. Implemented-but-unsupported rows
    // chat through the weaker EXPERIMENTAL lane (every turn marked unverified). Only
    // a model that is not generation-ready at all is fully blocked.
    if (!selectedModelRunnable && !selectedModelExperimental) {
      showNotice('Camelid is not generation-ready for the selected model yet.', 'error')
      return
    }

    const messageContent = draftContent.trim()
    setSending(true)
    let activeConversationId = null
    let assistantId = null
    let chatLifecycleId = null
    let pendingAssistantPatch = null
    let pendingAssistantFrame = null

    try {
      const conversation = await ensureConversation()
      activeConversationId = conversation.id
      // Fresh chats start from the __new__ sentinel. Select the real conversation immediately
      // so the main thread renders the same streaming message object as the sidebar preview.
      setSelectedConversationId(conversation.id)
      const userMessage = { id: makeId('message'), role: 'user', content: messageContent, model_id: selectedModelId, created_at: nowIso() }
      setPendingChat({ conversationId: conversation.id, content: messageContent, modelId: selectedModelId })
      if (overrideContent === null) setComposer('')

      const truncateIndex = truncateFromMessageId
        ? (conversation.messages || []).findIndex((message) => message.id === truncateFromMessageId)
        : -1
      const baseMessages = truncateIndex >= 0
        ? (conversation.messages || []).slice(0, truncateIndex)
        : (conversation.messages || [])
      const history = [...baseMessages, userMessage]
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .filter((message) => !message.content.startsWith('Conversation created.'))
        .map(({ role, content }) => ({ role, content }))
      const requestMessages = applyLocalChatPolicy(history)
      const promptTokenEstimate = estimateChatTokenCount(requestMessages)

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

      const requestStartedAt = performance.now()
      // Fresh per-token decode trace for this generation (auditable backing for
      // the live tok/s readout; read out after the stream completes).
      if (typeof window !== 'undefined') window.__tpsTrace = []
      const lifecycleId = beginRequest({ kind: 'chat', endpoint: '/v1/chat/completions', modelId: getRuntimeRequestModelId(selectedModel, runtime, selectedModelId) })
      chatLifecycleId = lifecycleId
      let firstContentEmitted = false
      let lastProgressAt = 0
      const pacer = createPacerState()
      assistantId = makeId('message')
      /* Snapshot of the support claim that was active when this send left the
         composer: row id + status only (never paths) so the message footer can
         cite the exact contract row that gated this generation. */
      const sendGate = getChatGateState(dashboard?.capabilities, selectedModel, runtime)
      const supportRowAtSend = sendGate.hint?.target
        ? { id: sendGate.hint.target.id, status: sendGate.hint.target.status, supported: sendGate.contractSupported }
        : null
      // Mark this turn experimental when it ran on the weaker (implemented-but-not-
      // supported) lane, so the footer can flag it as unverified with no parity claim.
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
        usage: null,
        streaming: true,
        streaming_phase: 'preparing',
        first_byte_ms: null,
        first_event_ms: null,
        first_content_ms: null,
      }
      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? { ...item, title: item.title === 'New conversation' ? messageContent.slice(0, 64) : item.title, messages: [...(item.messages || []), assistantMessageBase], updated_at: nowIso() }
          : item
      )))
      setPendingChat(null)

      const requestModelId = getRuntimeRequestModelId(selectedModel, runtime, selectedModelId)
      const requestController = new AbortController()
      activeChatRequestRef.current = requestController
      const response = await fetch(`${normalizedApiBase}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: requestController.signal,
        body: JSON.stringify({
          model: requestModelId,
          messages: requestMessages,
          // Supported rows stay greedy (temperature 0) — their behavior is parity-
          // locked. Experimental rows have no parity contract and small models loop
          // badly under greedy decoding, so they sample for usable output.
          temperature: selectedModelExperimental ? 0.7 : 0,
          ...(selectedModelExperimental ? { top_p: 0.9 } : {}),
          max_tokens: localChatMaxTokens(history, requestModelId),
          /* Empty today: a sampling override is sent only when /api/capabilities
             advertises a supported row for that exact parameter. */
          ...contractSamplingOverrides(dashboard?.capabilities?.api_features, requestModelId),
          // Opt-in thinking mode (experimental — not parity-locked). Only sent
          // when the user turns it on; silence keeps the thinking-DISABLED
          // parity-locked rendering.
          ...(thinkingMode ? { camelid_enable_thinking: true } : {}),
          // Receipts only attach to non-streaming responses; the JSON
          // fallback in readStreamingChatCompletion handles that shape.
          stream: !receiptMode,
          ...(receiptMode ? { camelid_receipt: true } : {}),
        }),
      })
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
      if (response.ok && !response.headers.get('content-type')?.includes('application/json')) {
        markAssistantStreamState({ streaming_phase: 'generating' }, { immediate: true })
      }
      const streamed = await readStreamingChatCompletion(response, (_delta, fullContent, metrics) => {
        const liveElapsedMs = performance.now() - requestStartedAt
        /* Live decode rate uses the REAL token count (one SSE content delta =
           one generated token in Camelid) over the decode window, not a
           word-piece estimate — so the on-screen tok/s is auditable and is
           backed token-for-token by window.__tpsTrace below. */
        const realTokens = Number(metrics?.completionTokens) || 0
        const decodeMs = Number(metrics?.elapsedMs) || liveElapsedMs
        const liveTps = tokensPerSecond(realTokens, decodeMs)
        if (typeof window !== 'undefined' && realTokens > 0) {
          if (!Array.isArray(window.__tpsTrace)) window.__tpsTrace = []
          window.__tpsTrace.push({ i: realTokens, t_ms: Math.round(decodeMs * 10) / 10, tps: liveTps != null ? Math.round(liveTps * 100) / 100 : null, delta: _delta })
        }
        if (!firstContentEmitted && fullContent) {
          firstContentEmitted = true
          emitFirstContent(lifecycleId, liveElapsedMs)
        }
        if (performance.now() - lastProgressAt > 100) {
          lastProgressAt = performance.now()
          emitProgress(lifecycleId, { tokens: realTokens, tokensPerSec: liveTps })
        }
        markAssistantStreamState({
          /* paced display (lag-bounded ≤150ms, drains on end; metrics use the
             real stream — see lib/streamPacing.js) */
          content: paceStep(pacer, fullContent, performance.now()) || '…',
          streaming_phase: 'streaming',
          tokens_in_per_sec: null,
          tokens_out_per_sec: liveTps,
        })
      }, {
        estimateTokenCount,
        onStreamEvent(event) {
          if (event.type === 'bytes' || event.type === 'role' || event.type === 'json_fallback') {
            markAssistantStreamState({
              streaming_phase: 'generating',
              first_byte_ms: event.firstByteMs ?? null,
              first_event_ms: event.firstEventMs ?? null,
            }, { immediate: true })
          }
        },
      })
      flushAssistantStreamPatch()
      const elapsedMs = performance.now() - requestStartedAt
      const assistantMessage = {
        ...assistantMessageBase,
        content: paceDrain(pacer, streamed.content || '(empty response)'),
        tokens_in_per_sec: tokensPerSecond(promptTokenEstimate, streamed.firstContentMs),
        tokens_out_per_sec: tokensPerSecond(streamed.completionTokens || estimateTokenCount(streamed.content), elapsedMs),
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
        camelid_receipt: streamed.camelidReceipt || null,
        streaming: false,
        streaming_phase: null,
        first_byte_ms: streamed.firstByteMs ?? null,
        first_event_ms: streamed.firstEventMs ?? null,
        first_content_ms: streamed.firstContentMs ?? null,
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
      setSelectedConversationId(conversation.id)
      recordChatGeneration({
        lifecycleId,
        modelId: requestModelId,
        durationMs: elapsedMs,
        ttftMs: streamed.firstContentMs ?? null,
        promptTokens: assistantMessage.usage?.prompt_tokens,
        completionTokens: assistantMessage.usage?.completion_tokens,
        tokensPerSec: assistantMessage.tokens_out_per_sec,
        usageSource: assistantMessage.usage_source,
        outcome: streamed.finishReason === 'error' ? 'error' : 'ok',
        promptText: messageContent,
      })
    } catch (error) {
      const requestWasAborted = error?.name === 'AbortError'
      recordChatGeneration({
        lifecycleId: chatLifecycleId,
        modelId: getRuntimeRequestModelId(selectedModel, runtime, selectedModelId),
        durationMs: null,
        ttftMs: null,
        outcome: requestWasAborted ? 'interrupted' : 'error',
        promptText: messageContent,
      })
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
      activeChatRequestRef.current = null
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
    setSelectedModelId(id)

    if (!model) {
      showNotice('Choose a saved local model before loading it.', 'error')
      return
    }
    if (isExternalModel(model)) {
      showNotice('Hosted API chat routing is planned but not wired yet. Keep using local GGUF loading for now.', 'info')
      return
    }
    if (!model.model_path) {
      showNotice('This model needs a local GGUF path before Camelid can load it.', 'error')
      return
    }
    if (modelRuntimeIdMatches(model, runtime) && runtime?.generation_ready) {
      showNotice('That model is already loaded and generation-ready.', 'success')
      return
    }

    setLoadingModelId(id)
    showNotice(`Loading ${model.name || id} into Camelid…`, 'info')
    try {
      // replace: swap models rather than stacking a second resident copy, so the
      // fit preflight judges this model against a host that has released the last one.
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({ id, path: model.model_path, replace: true }),
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
            ? 'Model loaded. Camelid reports generation-ready and the /api/capabilities support contract matches this model/quant.'
            : 'Model loaded and generation-ready, but chat stays guarded until /api/capabilities has an exact supported COMPATIBILITY.md row for this model/quant.'
          : 'Model loaded, but Camelid does not report generation-ready yet. Check tokenizer/config/tensor readiness.',
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
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({ id: derivedId, path: modelPath, replace: true }),
      })
      const loadedId = loaded?.id || derivedId
      const loadedPath = getModelPath(loaded) || modelPath
      const ready = isLoadedModelGenerationReady(loaded)
      const fileType = getLoadedModelFileType(loaded)
      const quantLabel = getLoadedModelQuantLabel(loaded) || (fileType !== null && fileType !== undefined ? `file_type ${fileType}` : null)
      const loadedRecord = {
        id: loadedId,
        name: name || loadedId,
        model_path: loadedPath,
        runtime_model_name: registerForm.runtime_model_name.trim() || loadedId,
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
      setRegisterForm({ id: '', name: '', model_path: '', runtime_model_name: '' })
      await loadDashboard({ silent: true, localModelsOverride: nextLocalModels })
      showNotice(
        ready
          ? supportedByContract
            ? 'Local model saved, loaded, generation-ready, and matched to a supported /api/capabilities row.'
            : 'Local model saved and generation-ready, but chat stays guarded until COMPATIBILITY.md and /api/capabilities explicitly support this model/quant.'
          : 'Local model saved and loaded, but Camelid still reports generation is not ready.',
        ready && supportedByContract ? 'success' : 'info',
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
    receiptMode,
    setReceiptMode,
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
      if (typeof window !== 'undefined') window.localStorage.setItem(API_BASE_STORAGE_KEY, next)
    },
  }
}
