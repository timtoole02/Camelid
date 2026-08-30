/* Response-length limits (Phase 9). Pure helpers; every number traces to a
   real source: model context from /v1/models meta.n_ctx_train (descriptive
   metadata, not a support claim — I2), the verified bound from validated
   bounded-context packs on the exact /api/capabilities row. Memory and
   KV-cost inputs do not exist on the backend yet
   (frontend/design-evidence/BACKEND_ASKS.md #3) — the
   UI renders those indicators ABSENT, never estimated client-side. */

import { findCompatibilityHint, isExactCompatibilityHint } from './capabilities.js'
import { appStorage } from './appStorage.js'

export const MAX_RESPONSE_TOKENS = 1000000
export const MIN_RESPONSE_TOKENS = 1
export const GEMMA4_MIN_CHAT_TOKENS = 8
export const GEMMA4_GHOST_WEBUI_CONTEXT_TOKENS = 4096
// Ghost's default common-core Metal cache holds 4,096 positions. Keeping the
// WebUI reply allowance at 512 leaves a conservative 3,584-position envelope
// for ordinary prompts/history instead of letting the global 8,192 default
// force CPU common execution before position zero.
export const GEMMA4_GHOST_WEBUI_MAX_TOKENS = 512
// BitNet-b1.58-2B-4T is useful interactively at a short, bounded first-turn
// budget. A valid per-model setting remains authoritative; this ceiling only
// replaces Camelid's much larger legacy/global default for a fresh BitNet setup.
export const BITNET_B1_58_DEFAULT_CHAT_MAX_TOKENS = 128
export const DETENTS = [256, 1000, 4000, 16000, 64000, 256000, 1000000]

const BITNET_B1_58_IDENTITIES = new Set([
  'bitnet_b1_58_2b_4t_i2_s',
  'bitnet_b1_58_2b_4t',
  'bitnet2b',
])
const BITNET_B1_58_FILENAME = 'ggml-model-i2_s.gguf'

function normalizedModelIdentity(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function filenameFromIdentity(value) {
  return String(value || '').split(/[\\/]/).pop()?.toLowerCase() || ''
}

function ggufArchitecture(record) {
  const metadata = record?.gguf?.metadata || record?.metadata || {}
  return record?.architecture
    ?? record?.gguf?.architecture
    ?? metadata?.general?.architecture
    ?? metadata?.['general.architecture']
    ?? null
}

/* Identify only the causal Microsoft 2B row. The two BitNet embedding models
   deliberately do not match: their inspected architectures are qwen3/gemma3,
   their filenames differ, and none of their ids are in the exact identity set. */
export function isBitNetB158ChatModel(model, runtime, requestModelId = '') {
  const currentModel = runtime?.current_model || null
  const architectures = [ggufArchitecture(model), ggufArchitecture(currentModel)]
  if (architectures.some((value) => normalizedModelIdentity(value) === 'bitnet_b1_58')) return true

  const identities = [
    requestModelId,
    runtime?.active_model_id,
    model?.id,
    model?.catalog_id,
    model?.runtime_model_name,
    model?.name,
    model?.model_path,
    model?.hf_filename,
    currentModel?.id,
    currentModel?.name,
    currentModel?.path,
    currentModel?.model_path,
  ]
  return identities.some((value) => (
    BITNET_B1_58_IDENTITIES.has(normalizedModelIdentity(value))
    || filenameFromIdentity(value) === BITNET_B1_58_FILENAME
  ))
}

export function applyBitNetFreshChatTokenCap(value, {
  bitNetB158 = false,
  hasExplicitSetting = false,
} = {}) {
  const configured = Number.isFinite(Number(value))
    ? Math.max(MIN_RESPONSE_TOKENS, Math.round(Number(value)))
    : MIN_RESPONSE_TOKENS
  return bitNetB158 && !hasExplicitSetting
    ? Math.min(configured, BITNET_B1_58_DEFAULT_CHAT_MAX_TOKENS)
    : configured
}

export function applyGemma4ChatTokenFloor(value, compatibilityFamily = '') {
  const configured = Number.isFinite(Number(value)) ? Math.round(Number(value)) : MIN_RESPONSE_TOKENS
  return String(compatibilityFamily).startsWith('gemma4_')
    ? Math.max(configured, GEMMA4_MIN_CHAT_TOKENS)
    : configured
}

export function gemma4ChatTokenFloorForModel(capabilities, model) {
  const hint = findCompatibilityHint(capabilities, model)
  return isExactCompatibilityHint(hint) && String(hint?.target?.family || '').startsWith('gemma4_')
    ? GEMMA4_MIN_CHAT_TOKENS
    : null
}

export function gemma4GhostChatTokenCap(serveLane = '') {
  const lane = String(serveLane || '').trim().toLowerCase().replace(/-/g, '_')
  return lane === 'ghost_moe' ? GEMMA4_GHOST_WEBUI_MAX_TOKENS : null
}

export function applyGemma4GhostChatTokenCap(value, serveLane = '') {
  const configured = Number.isFinite(Number(value))
    ? Math.max(MIN_RESPONSE_TOKENS, Math.round(Number(value)))
    : MIN_RESPONSE_TOKENS
  const cap = gemma4GhostChatTokenCap(serveLane)
  return cap === null ? configured : Math.min(configured, cap)
}

export function modelContextLength(model) {
  const value = Number(model?.meta?.n_ctx_train)
  return Number.isFinite(value) && value > 0 ? value : null
}

/* Highest bounded-context window whose pack status is validated on the exact
   matched row. Family/quant resemblance never produces a bound. */
export function verifiedContextBound(capabilities, model) {
  const hint = findCompatibilityHint(capabilities, model)
  if (!isExactCompatibilityHint(hint) || !hint.target) return null
  const row = hint.target
  let bound = null
  for (const key of Object.keys(row)) {
    const match = key.match(/^bounded_context_(\d+)_pack$/)
    if (match && String(row[key]).startsWith('validated')) {
      const window = Number(row[`bounded_context_${match[1]}_window`] ?? match[1])
      if (Number.isFinite(window)) bound = Math.max(bound ?? 0, window)
    }
  }
  return bound
}

/* Log-scale slider mapping: position 0..1 over [MIN, MAX]. */
const LOG_MIN = Math.log(MIN_RESPONSE_TOKENS)
const LOG_MAX = Math.log(MAX_RESPONSE_TOKENS)

export function tokensToSlider(value) {
  const clamped = Math.min(Math.max(value, MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
  return (Math.log(clamped) - LOG_MIN) / (LOG_MAX - LOG_MIN)
}

export function sliderToTokens(position) {
  const value = Math.round(Math.exp(LOG_MIN + (LOG_MAX - LOG_MIN) * Math.min(Math.max(position, 0), 1)))
  // light detent snap: within 2% of track distance
  for (const detent of DETENTS) {
    if (Math.abs(tokensToSlider(detent) - position) < 0.012) return detent
  }
  return value
}

/* Validation states, priority-ordered. A response limit above the model context
   is a non-blocking caution now — the backend auto-limits (clamps) it to the room
   left after the prompt, it does not reject. amber =
   allowed but beyond the verified row's tested context; slate stays for
   support states elsewhere. */
export function validateResponseLength({ value, contextLength = null, verifiedBound = null, modelName = 'the loaded model' }) {
  if (contextLength !== null && value > contextLength) {
    return {
      level: 'caution',
      code: 'over_model_context',
      message: `Exceeds ${modelName}’s ${contextLength.toLocaleString()}-token context — the backend auto-limits each response to the room left after the prompt, so replies may be shorter than this. Load a longer-context model for full-length replies.`,
    }
  }
  if (verifiedBound !== null && value > verifiedBound) {
    return {
      level: 'caution',
      code: 'over_verified_bound',
      message: `Beyond the verified row’s tested ${verifiedBound.toLocaleString()}-token context — allowed, untested. Evidence covers the checked packs only.`,
    }
  }
  return { level: 'ok', code: 'ok', message: '' }
}

/* Send-time budget check. The response limit is an UPPER BOUND: the backend
   clamps it to the room left in the context window, so exceeding it is a
   non-blocking notice, not an error. The only hard failure is a prompt that
   already fills the whole context (no room to generate), which the backend
   rejects with context_length_exceeded. Prompt size is a client estimate. */
export function validateSendBudget({ promptTokens, maxTokens, contextLength }) {
  if (contextLength === null || !Number.isFinite(promptTokens)) return { level: 'ok' }
  if (promptTokens >= contextLength) {
    return {
      level: 'error',
      code: 'prompt_fills_context',
      message: `This prompt (~${promptTokens.toLocaleString()} tokens, estimated) fills the model’s ${contextLength.toLocaleString()}-token context, leaving no room for a reply. Shorten the prompt or load a longer-context model.`,
    }
  }
  if (promptTokens + maxTokens > contextLength) {
    const room = contextLength - promptTokens
    return {
      level: 'notice',
      code: 'response_auto_limited',
      message: `Response will be auto-limited to ~${room.toLocaleString()} tokens to fit the ${contextLength.toLocaleString()}-token context.`,
    }
  }
  return { level: 'ok' }
}

const MAX_TOKENS_KEY = 'camelid.maxTokens'

export function hasExplicitMaxTokensSetting(modelId = '') {
  if (typeof window === 'undefined' || !modelId) return false
  const value = Number.parseInt(appStorage.getItem(`${MAX_TOKENS_KEY}.${modelId}`) || '', 10)
  return Number.isFinite(value) && value >= MIN_RESPONSE_TOKENS
}

export function getConfiguredMaxTokens(modelId = '') {
  if (typeof window === 'undefined') return 8192
  const perModel = modelId ? Number.parseInt(appStorage.getItem(`${MAX_TOKENS_KEY}.${modelId}`) || '', 10) : NaN
  if (Number.isFinite(perModel) && perModel >= MIN_RESPONSE_TOKENS) return perModel
  const legacy = Number.parseInt(appStorage.getItem(MAX_TOKENS_KEY) || '', 10)
  return Number.isFinite(legacy) && legacy >= 256 ? legacy : 8192
}

export function setConfiguredMaxTokens(modelId, value) {
  if (typeof window === 'undefined') return
  const clamped = Math.min(Math.max(Math.round(value), MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
  if (modelId) appStorage.setItem(`${MAX_TOKENS_KEY}.${modelId}`, String(clamped))
  else appStorage.setItem(MAX_TOKENS_KEY, String(clamped))
}
