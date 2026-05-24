import { useEffect, useMemo, useRef, useState } from 'react'
import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel, quantLabelFromGgufFileType } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { resolveLoadedModelDisplayName } from '../lib/loadedModelDisplay'
import { readStreamingChatCompletion } from '../lib/chatCompletionStream'
import { NEW_CHAT_SENTINEL, resolveSelectedConversation, shouldCreateConversationForSend } from '../lib/chatState'
import { normalizeStoredConversations } from '../lib/conversationStorage.js'
import { getRuntimeRequestModelId, isExternalModel, modelRuntimeIdMatches } from '../lib/modelState'

const TAB_STORAGE_KEY = 'camelid.activeTab'
const SELECTED_CONVERSATION_STORAGE_KEY = 'camelid.selectedConversationId'
const SELECTED_MODEL_STORAGE_KEY = 'camelid.selectedModelId'
const LOCAL_MODELS_STORAGE_KEY = 'camelid.localModels'
const CONVERSATIONS_STORAGE_KEY = 'camelid.conversations'
const MEMORIES_STORAGE_KEY = 'camelid.memories'
const API_BASE_STORAGE_KEY = 'camelid.apiBase'
const VALID_TABS = new Set(['chat', 'library', 'api', 'analytics', 'history', 'memory', 'system'])
const DEFAULT_API_BASE = import.meta.env?.VITE_CAMELID_API_BASE || 'http://127.0.0.1:8181'

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
const LOCAL_CHAT_DEMO_MAX_TOKENS = 800
const LOCAL_CHAT_CODE_MAX_TOKENS = 2048

function looksLikeCodePrompt(value) {
  const text = String(value || '').toLowerCase()
  return /\b(code|build|create|implement|write|make)\b/.test(text)
    && /\b(html|html5|css|javascript|js|python|py|pygame|game|pacman|pacmac|tetris|app|component|page|website)\b/.test(text)
}

function applyLocalChatPolicy(messages) {
  const lastUser = [...(messages || [])].reverse().find((message) => message.role === 'user')
  if (!looksLikeCodePrompt(lastUser?.content)) return messages
  return [
    { role: 'system', content: CODE_FIRST_SYSTEM_PROMPT },
    ...messages,
  ]
}

function localChatMaxTokens(messages) {
  const lastUser = [...(messages || [])].reverse().find((message) => message.role === 'user')
  return looksLikeCodePrompt(lastUser?.content) ? LOCAL_CHAT_CODE_MAX_TOKENS : LOCAL_CHAT_DEMO_MAX_TOKENS
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
  return status === 'ready' || status === 'registered' || status === 'failed' ? status : 'registered'
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

function mergeModelLists({ modelItems, health, currentModel, localModels, apiBase }) {
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
  return [...byId.values()].sort(compareModelsByName)
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
      status: health?.ok ? 'online' : 'offline',
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
      const [health, modelList, capabilities] = await Promise.all([
        fetchJson(`${normalizedApiBase}/v1/health`),
        fetchJson(`${normalizedApiBase}/v1/models`),
        fetchJson(`${normalizedApiBase}/api/capabilities`).catch(() => null),
      ])
      const currentModel = health?.active_model_id
        ? await fetchJson(`${normalizedApiBase}/api/models/current`).catch(() => null)
        : null
      const modelItems = Array.isArray(modelList?.data) ? modelList.data : []
      const nextModels = mergeModelLists({
        modelItems,
        health,
        currentModel,
        localModels: currentLocalModels,
        apiBase: normalizedApiBase,
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
    const interval = setInterval(() => loadDashboard({ silent: true }), 2500)
    return () => clearInterval(interval)
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
  const selectedModelRuntimeReady = selectedModelChatGate.runtimeReady
  const selectedModelCapabilitySupported = selectedModelChatGate.contractSupported
  const selectedModelRunnable = selectedModelChatGate.chatUnlocked
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

  const sendMessage = async () => {
    if (!composer.trim()) return
    if (!selectedModelRunnable) {
      if (selectedModelRuntimeReady && !selectedModelCapabilitySupported) {
        const hint = findCompatibilityHint(dashboard?.capabilities, selectedModel)
        showNotice(`${compatibilityHintLabel(hint, 'No matching COMPATIBILITY.md row')}: ${compatibilityHintCopy(hint)} Chat is blocked until /api/capabilities marks this exact model/quant as supported.`, 'error')
      } else {
        showNotice('Camelid is not generation-ready for the selected model yet.', 'error')
      }
      return
    }

    const messageContent = composer.trim()
    setSending(true)
    let activeConversationId = null
    let assistantId = null
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
      setComposer('')

      const history = [...(conversation.messages || []), userMessage]
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .filter((message) => !message.content.startsWith('Conversation created.'))
        .map(({ role, content }) => ({ role, content }))
      const requestMessages = applyLocalChatPolicy(history)
      const promptTokenEstimate = estimateChatTokenCount(requestMessages)

      persistConversations((current) => current.map((item) => (
        item.id === conversation.id
          ? { ...item, model_id: selectedModelId, messages: [...(item.messages || []), userMessage], updated_at: nowIso() }
          : item
      )))

      const requestStartedAt = performance.now()
      assistantId = makeId('message')
      const assistantMessageBase = {
        id: assistantId,
        role: 'assistant',
        content: '',
        model_id: selectedModelId,
        model_name: selectedModel?.name || selectedModelId,
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
      const response = await fetch(`${normalizedApiBase}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: requestModelId, messages: requestMessages, temperature: 0, max_tokens: localChatMaxTokens(history), stream: true }),
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
      const streamed = await readStreamingChatCompletion(response, (_delta, fullContent) => {
        const liveElapsedMs = performance.now() - requestStartedAt
        const liveCompletionTokens = estimateTokenCount(fullContent)
        markAssistantStreamState({
          content: fullContent || '…',
          streaming_phase: 'streaming',
          tokens_in_per_sec: null,
          tokens_out_per_sec: tokensPerSecond(liveCompletionTokens, liveElapsedMs),
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
        content: streamed.content || '(empty response)',
        tokens_in_per_sec: tokensPerSecond(promptTokenEstimate, streamed.firstContentMs),
        tokens_out_per_sec: tokensPerSecond(streamed.completionTokens || estimateTokenCount(streamed.content), elapsedMs),
        finish_reason: streamed.finishReason,
        elapsed_ms: elapsedMs,
        usage: streamed.usage || {
          prompt_tokens: promptTokenEstimate,
          completion_tokens: streamed.completionTokens || estimateTokenCount(streamed.content),
          total_tokens: promptTokenEstimate + (streamed.completionTokens || estimateTokenCount(streamed.content)),
        },
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
    } catch (error) {
      const pendingPatchAtFailure = pendingAssistantPatch
      if (pendingAssistantFrame !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(pendingAssistantFrame)
        pendingAssistantFrame = null
      }
      pendingAssistantPatch = null
      const errorMessage = getGuardrailErrorMessage(error, 'Local inference failed.')
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
                          finish_reason: 'error',
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
      showNotice(errorMessage, 'error')
    } finally {
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

  const installModel = async () => {
    showNotice('Model catalog downloads are not wired in Camelid yet. Camelid currently loads local GGUF paths.', 'error')
  }

  const installCatalogModel = async () => {
    showNotice('Hugging Face catalog install is not wired to Camelid yet.', 'error')
    return false
  }

  const cancelModelDownload = async () => {
    showNotice('No Camelid download is running from this UI.', 'info')
    return false
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
      const loaded = await fetchJson(`${normalizedApiBase}/api/models/load`, {
        method: 'POST',
        body: JSON.stringify({ id, path: model.model_path }),
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
        body: JSON.stringify({ id: derivedId, path: modelPath }),
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
    filteredConversations,
    filteredMemories,
    latestAssistantMessage,
    pendingConversation,
    createConversation,
    showNewChatLanding,
    sendMessage,
    saveToMemory,
    createMemory,
    updateMemory,
    deleteMemory,
    renameConversation,
    deleteConversation,
    installModel,
    installCatalogModel,
    cancelModelDownload,
    activateModel,
    unloadCurrentModel,
    registerModel,
    connectExternalModel,
    loadDashboard,
    apiBase,
    setApiBase: (value) => {
      const next = normalizeApiBase(value)
      setApiBaseState(next)
      if (typeof window !== 'undefined') window.localStorage.setItem(API_BASE_STORAGE_KEY, next)
    },
  }
}
