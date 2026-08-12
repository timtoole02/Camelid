#!/usr/bin/env node
import { mkdir, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'

import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel, quantLabelFromGgufFileType } from '../src/lib/capabilities.js'
import { getChatGateState } from '../src/lib/chatGate.js'
import { readStreamingChatCompletion } from '../src/lib/chatCompletionStream.js'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i]
  if (arg.startsWith('--')) {
    const [key, inline] = arg.slice(2).split('=', 2)
    const value = inline ?? (process.argv[i + 1]?.startsWith('--') ? 'true' : process.argv[++i] ?? 'true')
    args.set(key, value)
  }
}

const apiBase = (args.get('api') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const frontendUrl = (args.get('frontend') || process.env.CAMELID_FRONTEND_URL || 'http://127.0.0.1:4175').replace(/\/$/, '')
const loadTiny = args.has('load-tiny') || process.env.CAMELID_SMOKE_LOAD_TINY === '1'
const rawModelPath = args.get('model') || process.env.CAMELID_SMOKE_MODEL
const modelPath = rawModelPath ? resolve(rawModelPath) : undefined
const modelId = args.get('model-id') || process.env.CAMELID_SMOKE_MODEL_ID || (modelPath ? 'smoke-model' : 'tiny-generation')
const requireGeneration = args.has('require-generation') || process.env.CAMELID_SMOKE_REQUIRE_GENERATION === '1'
const allowGuardedChat = args.has('allow-guarded-chat') || process.env.CAMELID_SMOKE_ALLOW_GUARDED_CHAT === '1'
const replaceLoadedModel = args.has('replace-loaded-model') || process.env.CAMELID_SMOKE_REPLACE_LOADED_MODEL === '1'
const chatRepeats = Number.parseInt(args.get('chat-repeats') || process.env.CAMELID_SMOKE_CHAT_REPEATS || '1', 10)
const streamMaxTokens = Number.parseInt(args.get('stream-max-tokens') || process.env.CAMELID_SMOKE_STREAM_MAX_TOKENS || '24', 10)
const chatMessage = args.get('message') ?? process.env.CAMELID_SMOKE_MESSAGE ?? 'hello'
const expectCompatibilityRow = args.get('expect-compatibility-row') || process.env.CAMELID_SMOKE_EXPECT_COMPATIBILITY_ROW || ''
const expectCompatibilityStatus = args.get('expect-compatibility-status') || process.env.CAMELID_SMOKE_EXPECT_COMPATIBILITY_STATUS || ''
const expectContractSupported = parseOptionalBoolean(args.get('expect-contract-supported') || process.env.CAMELID_SMOKE_EXPECT_CONTRACT_SUPPORTED, 'expect-contract-supported')
const expectWebUiChat = args.get('expect-webui-chat') || process.env.CAMELID_SMOKE_EXPECT_WEBUI_CHAT || ''
const expectLocalLaneClass = args.get('expect-local-lane-class') || process.env.CAMELID_SMOKE_EXPECT_LOCAL_LANE_CLASS || (expectContractSupported === true ? 'supported' : '')
const expectGgufSha256 = normalizeSha256(args.get('expect-gguf-sha256') || process.env.CAMELID_SMOKE_EXPECT_GGUF_SHA256 || '')
const requireLocalModel = args.has('require-local-model') || process.env.CAMELID_SMOKE_REQUIRE_LOCAL_MODEL === '1' || Boolean(expectLocalLaneClass || expectGgufSha256)

if (!Number.isInteger(chatRepeats) || chatRepeats < 1) {
  throw new Error(`--chat-repeats must be a positive integer, got ${args.get('chat-repeats')}`)
}
if (!Number.isInteger(streamMaxTokens) || streamMaxTokens < 1) {
  throw new Error(`--stream-max-tokens must be a positive integer, got ${args.get('stream-max-tokens')}`)
}
if (expectLocalLaneClass && !['supported', 'runnable_with_variance', 'experimental_implemented', 'unsupported'].includes(expectLocalLaneClass)) {
  throw new Error(`--expect-local-lane-class must be one of supported, runnable_with_variance, experimental_implemented, unsupported; got ${expectLocalLaneClass}`)
}

if (loadTiny && modelPath) {
  throw new Error('Use either --load-tiny or --model <path>, not both')
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      'content-type': 'application/json',
      ...(options.headers || {}),
    },
  })
  const text = await response.text()
  const body = text ? JSON.parse(text) : null
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${body?.error?.message || text}`)
  }
  return body
}

async function timed(label, fn) {
  const started = performance.now()
  const result = await fn()
  const elapsedMs = performance.now() - started
  console.log(`  ${label}_ms=${elapsedMs.toFixed(0)}`)
  return { result, elapsedMs }
}

function parseOptionalBoolean(value, name) {
  if (value === undefined || value === null || value === '') return null
  const normalized = value.toString().trim().toLowerCase()
  if (['1', 'true', 'yes'].includes(normalized)) return true
  if (['0', 'false', 'no'].includes(normalized)) return false
  throw new Error(`--${name} must be true or false, got ${value}`)
}

function assertExpected(label, actual, expected) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

function normalizeSha256(value) {
  if (!value) return ''
  const normalized = String(value).trim().toLowerCase()
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`--expect-gguf-sha256 must be 64 hexadecimal characters, got ${value}`)
  }
  return normalized
}

function normalizedPath(value) {
  if (!value) return null
  const normalized = resolve(String(value))
  return process.platform === 'win32' ? normalized.toLowerCase() : normalized
}

async function resolveExactLocalModel({ localModels, current, required }) {
  const entries = Array.isArray(localModels?.models) ? localModels.models : []
  const expectedFilename = modelPath
    ? basename(modelPath)
    : current?.lane?.gguf_filename || (current?.path ? basename(current.path) : '')
  const matches = expectedFilename ? entries.filter(entry => entry?.filename === expectedFilename) : []
  const local = matches.length === 1 ? matches[0] : null

  if (!required) return local
  if (!modelPath) throw new Error('--require-local-model needs --model <path> so exact path and size can be asserted')

  const modelStats = await stat(modelPath)
  assertExpected('current model id', current?.id ?? null, modelId)
  assertExpected('current lane model id', current?.lane?.model_id ?? null, modelId)
  assertExpected('current model path', normalizedPath(current?.path), normalizedPath(modelPath))
  assertExpected('current lane filename', current?.lane?.gguf_filename ?? null, expectedFilename)
  assertExpected('/api/models/local exact filename match count', matches.length, 1)
  assertExpected('/api/models/local exact path', normalizedPath(localModels?.models_dir && local ? join(localModels.models_dir, local.filename) : null), normalizedPath(modelPath))
  assertExpected('/api/models/local exact file size', local?.size_bytes ?? null, modelStats.size)

  const actualSha256 = String(current?.lane?.gguf_sha256 || '').toLowerCase()
  if (!/^[a-f0-9]{64}$/.test(actualSha256)) {
    throw new Error(`current lane GGUF SHA-256 is missing or malformed: ${JSON.stringify(current?.lane?.gguf_sha256 ?? null)}`)
  }
  if (expectGgufSha256) assertExpected('current lane GGUF SHA-256', actualSha256, expectGgufSha256)
  if (expectLocalLaneClass) assertExpected('/api/models/local lane_class', local?.lane_class ?? null, expectLocalLaneClass)

  console.log(`✓ exact local identity filename=${expectedFilename}; size_bytes=${modelStats.size}; lane_class=${local.lane_class}; gguf_sha256=${actualSha256}`)
  return local
}

function assertTerminalStreamEvidence(streamed, streamEventDetails, maxTokens) {
  const eventTypes = streamEventDetails.map(event => event.type)
  const finishIndexes = eventTypes.flatMap((type, index) => type === 'finish' ? [index] : [])
  const usageIndexes = eventTypes.flatMap((type, index) => type === 'usage' ? [index] : [])
  const doneIndexes = eventTypes.flatMap((type, index) => type === 'done' ? [index] : [])
  if (finishIndexes.length !== 1) throw new Error(`chat stream must contain exactly one terminal finish event; got ${finishIndexes.length}`)
  if (usageIndexes.length !== 1) throw new Error(`chat stream must contain exactly one include_usage event; got ${usageIndexes.length}`)
  if (doneIndexes.length !== 1) throw new Error(`chat stream must contain exactly one [DONE] event; got ${doneIndexes.length}`)
  const [finishIndex] = finishIndexes
  const [usageIndex] = usageIndexes
  const [doneIndex] = doneIndexes
  if (!(finishIndex < usageIndex && usageIndex < doneIndex)) {
    throw new Error(`chat stream terminal order must be finish, usage, [DONE]; got ${eventTypes.join(',')}`)
  }
  if (usageIndex !== doneIndex - 1 || doneIndex !== eventTypes.length - 1) {
    throw new Error(`chat stream usage must be terminal immediately before [DONE], with no trailing events; got ${eventTypes.join(',')}`)
  }
  if (eventTypes.includes('error')) throw new Error('chat stream reported an error event before termination')
  if (typeof streamed.finishReason !== 'string' || !streamed.finishReason) {
    throw new Error(`chat stream did not preserve a terminal finish_reason: ${JSON.stringify(streamed.finishReason)}`)
  }
  const usage = streamed.usage
  for (const field of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
    if (!Number.isInteger(usage?.[field]) || usage[field] < 0) {
      throw new Error(`chat stream terminal usage.${field} must be a non-negative integer; got ${JSON.stringify(usage?.[field])}`)
    }
  }
  if (usage.total_tokens !== usage.prompt_tokens + usage.completion_tokens) {
    throw new Error(`chat stream terminal usage total mismatch: ${JSON.stringify(usage)}`)
  }
  if (usage.completion_tokens < 1 || usage.completion_tokens > maxTokens) {
    throw new Error(`chat stream completion_tokens must be between 1 and requested max_tokens=${maxTokens}; got ${usage.completion_tokens}`)
  }
}

async function fetchStreamingChatCompletion(url, body) {
  const streamEventDetails = []
  const streamedSnapshots = []
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  const contentType = response.headers.get('content-type') || ''
  if (response.ok && !contentType.includes('text/event-stream')) {
    throw new Error(`chat stream did not return text/event-stream; got ${contentType || 'no content-type'}`)
  }
  const streamed = await readStreamingChatCompletion(response, (_delta, fullContent, metrics) => {
    streamedSnapshots.push({ fullContent, firstContentMs: metrics.firstContentMs })
  }, {
    onStreamEvent(event) {
      streamEventDetails.push(event)
    },
  })
  const streamEvents = streamEventDetails.map(event => event.type)
  if (!streamEvents.includes('bytes')) throw new Error('chat stream did not expose first-byte progress')
  const hasContent = streamEvents.includes('content')
  const hasReasoning = streamEvents.includes('reasoning')
  if (!hasContent && !hasReasoning) throw new Error('chat stream completed without any content or reasoning delta')
  if (hasContent && !streamedSnapshots.length) throw new Error('chat stream did not publish visible content before final completion')
  assertTerminalStreamEvidence(streamed, streamEventDetails, body.max_tokens)
  return { streamed, streamEvents, streamEventDetails, streamedSnapshots }
}

function pushU32(bytes, value) {
  const b = Buffer.alloc(4)
  b.writeUInt32LE(value)
  bytes.push(b)
}
function pushI32(bytes, value) {
  const b = Buffer.alloc(4)
  b.writeInt32LE(value)
  bytes.push(b)
}
function pushU64(bytes, value) {
  const b = Buffer.alloc(8)
  b.writeBigUInt64LE(BigInt(value))
  bytes.push(b)
}
function pushI64(bytes, value) {
  const b = Buffer.alloc(8)
  b.writeBigInt64LE(BigInt(value))
  bytes.push(b)
}
function pushF32(bytes, value) {
  const b = Buffer.alloc(4)
  b.writeFloatLE(value)
  bytes.push(b)
}
function pushString(bytes, value) {
  const b = Buffer.from(value, 'utf8')
  pushU64(bytes, b.length)
  bytes.push(b)
}
function pushKvString(bytes, key, value) {
  pushString(bytes, key); pushI32(bytes, 8); pushString(bytes, value)
}
function pushKvU32(bytes, key, value) {
  pushString(bytes, key); pushI32(bytes, 4); pushU32(bytes, value)
}
function pushKvF32(bytes, key, value) {
  pushString(bytes, key); pushI32(bytes, 6); pushF32(bytes, value)
}
function pushKvBool(bytes, key, value) {
  pushString(bytes, key); pushI32(bytes, 7); bytes.push(Buffer.from([value ? 1 : 0]))
}
function pushKvArrayStrings(bytes, key, values) {
  pushString(bytes, key); pushI32(bytes, 9); pushI32(bytes, 8); pushU64(bytes, values.length)
  for (const value of values) pushString(bytes, value)
}
function pushKvArrayF32(bytes, key, values) {
  pushString(bytes, key); pushI32(bytes, 9); pushI32(bytes, 6); pushU64(bytes, values.length)
  for (const value of values) pushF32(bytes, value)
}
function pushKvArrayI32(bytes, key, values) {
  pushString(bytes, key); pushI32(bytes, 9); pushI32(bytes, 5); pushU64(bytes, values.length)
  for (const value of values) pushI32(bytes, value)
}

function writeTinyGenerationGgufBytes() {
  const tensors = [
    ['token_embd.weight', [4, 4]],
    ['output_norm.weight', [4]],
    ['output.weight', [4, 4]],
    ['blk.0.attn_norm.weight', [4]],
    ['blk.0.attn_q.weight', [4, 4]],
    ['blk.0.attn_k.weight', [4, 2]],
    ['blk.0.attn_v.weight', [4, 2]],
    ['blk.0.attn_output.weight', [4, 4]],
    ['blk.0.ffn_norm.weight', [4]],
    ['blk.0.ffn_gate.weight', [4, 6]],
    ['blk.0.ffn_up.weight', [4, 6]],
    ['blk.0.ffn_down.weight', [6, 4]],
  ]
  const tokens = ['<unk>', '<s>', '</s>', '▁hello']
  const scores = [0.0, 0.0, 0.0, 10.0]
  const tokenTypes = [2, 3, 3, 1]
  const bytes = [Buffer.from('GGUF')]
  pushU32(bytes, 3)
  pushI64(bytes, tensors.length)
  pushI64(bytes, 21)
  pushKvString(bytes, 'general.architecture', 'llama')
  pushKvU32(bytes, 'general.file_type', 0)
  pushKvU32(bytes, 'llama.context_length', 64)
  pushKvU32(bytes, 'llama.embedding_length', 4)
  pushKvU32(bytes, 'llama.block_count', 1)
  pushKvU32(bytes, 'llama.feed_forward_length', 6)
  pushKvU32(bytes, 'llama.attention.head_count', 2)
  pushKvU32(bytes, 'llama.attention.head_count_kv', 1)
  pushKvU32(bytes, 'llama.rope.dimension_count', 2)
  pushKvF32(bytes, 'llama.rope.freq_base', 10000.0)
  pushKvF32(bytes, 'llama.attention.layer_norm_rms_epsilon', 1e-6)
  pushKvU32(bytes, 'llama.vocab_size', 4)
  pushKvString(bytes, 'tokenizer.ggml.model', 'llama')
  pushKvArrayStrings(bytes, 'tokenizer.ggml.tokens', tokens)
  pushKvArrayF32(bytes, 'tokenizer.ggml.scores', scores)
  pushKvArrayI32(bytes, 'tokenizer.ggml.token_type', tokenTypes)
  pushKvU32(bytes, 'tokenizer.ggml.bos_token_id', 1)
  pushKvU32(bytes, 'tokenizer.ggml.eos_token_id', 2)
  pushKvBool(bytes, 'tokenizer.ggml.add_bos_token', true)
  pushKvBool(bytes, 'tokenizer.ggml.add_eos_token', false)
  pushKvBool(bytes, 'tokenizer.ggml.add_space_prefix', true)

  let relativeOffset = 0
  for (const [name, dims] of tensors) {
    pushString(bytes, name)
    pushU32(bytes, dims.length)
    for (const dim of dims) pushI64(bytes, dim)
    pushI32(bytes, 0)
    pushU64(bytes, relativeOffset)
    relativeOffset += dims.reduce((product, dim) => product * dim, 1) * 4
    while (relativeOffset % 32 !== 0) relativeOffset += 1
  }
  let buffer = Buffer.concat(bytes)
  while (buffer.length % 32 !== 0) buffer = Buffer.concat([buffer, Buffer.from([0])])
  return Buffer.concat([buffer, Buffer.alloc(relativeOffset)])
}

console.log(`Smoke target API: ${apiBase}`)
console.log(`Smoke target frontend: ${frontendUrl}`)

const { result: frontendResponse } = await timed('frontend_http_200', () => fetch(`${frontendUrl}/`))
if (!frontendResponse.ok) throw new Error(`frontend unavailable: ${frontendResponse.status} ${frontendResponse.statusText}`)
console.log('✓ frontend returned HTTP 200')

if (loadTiny || modelPath) {
  let pathToLoad = modelPath
  if (loadTiny) {
    const dir = join(tmpdir(), 'camelid-smoke')
    await mkdir(dir, { recursive: true })
    pathToLoad = join(dir, 'tiny-generation.gguf')
    await writeFile(pathToLoad, writeTinyGenerationGgufBytes())
  }

  const { result: loaded } = await timed('model_load', () => fetchJson(`${apiBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify({ path: pathToLoad, id: modelId, ...(replaceLoadedModel ? { replace: true } : {}) }),
  }))
  const tensorState = loaded.llama_tensors ? 'dense tensors bound' : 'dense tensors not bound'
  const tokenizerState = loaded.tokenizer?.status || 'unknown tokenizer state'
  const chatTemplate = loaded.tokenizer?.chat_template
  console.log(`✓ loaded ${loadTiny ? 'tiny smoke' : 'requested'} GGUF: ${pathToLoad}`)
  console.log(`  model_id=${loaded.id || modelId}; tokenizer=${tokenizerState}; ${tensorState}`)
  if (chatTemplate) {
    console.log(`  chat_template=${chatTemplate.detected_format}; source=${chatTemplate.source}; length=${chatTemplate.length}`)
  }
}

const { result: health } = await timed('health', () => fetchJson(`${apiBase}/v1/health`))
console.log(`✓ health ok; generation_ready=${health.generation_ready}; active_model_id=${health.active_model_id || 'none'}`)

const { result: models } = await timed('models', () => fetchJson(`${apiBase}/v1/models`))
const modelIds = Array.isArray(models.data) ? models.data.map((model) => model.id) : []
console.log(`✓ /v1/models returned ${modelIds.length} model(s): ${modelIds.join(', ') || 'none'}`)

const { result: activeModelDetails } = health.active_model_id
  ? await timed('current_model', () => fetchJson(`${apiBase}/api/models/current`).catch(() => null))
  : { result: null }

const { result: localModels } = await timed('local_models', () => fetchJson(`${apiBase}/api/models/local`))
const activeLocalModel = await resolveExactLocalModel({
  localModels,
  current: activeModelDetails,
  required: requireLocalModel,
})

const { result: capabilities } = await timed('capabilities', () => fetchJson(`${apiBase}/api/capabilities`))
const supportGate = capabilities?.support_contract?.current_gate || 'none'
const compatibilityRows = Array.isArray(capabilities?.model_compatibility) ? capabilities.model_compatibility.length : 0
const supportedFamilyRows = Array.isArray(capabilities?.supported_model_families) ? capabilities.supported_model_families.length : 0
const plannedFamilyRows = Array.isArray(capabilities?.planned_model_families) ? capabilities.planned_model_families.length : 0
const apiFeatures = Array.isArray(capabilities?.api_features) ? capabilities.api_features : []
const guardedApiFeatures = apiFeatures.filter((feature) => !['supported', 'validated', 'measured'].includes(feature.status) && !feature.status?.startsWith('supported_'))
console.log(`✓ /api/capabilities returned support gate: ${supportGate}; compatibility_rows=${compatibilityRows}; supported_model_families=${supportedFamilyRows}; planned_model_families=${plannedFamilyRows}; guarded_api_features=${guardedApiFeatures.length}`)

const activeModelFileType = activeModelDetails?.gguf?.metadata?.general?.file_type ?? activeModelDetails?.gguf?.metadata?.['general.file_type'] ?? null
const activeModelQuant = activeModelFileType === null || activeModelFileType === undefined ? null : quantLabelFromGgufFileType(activeModelFileType) || `file_type ${activeModelFileType}`
const activeModel = health.active_model_id ? {
  ...(activeLocalModel || {}),
  id: health.active_model_id,
  name: health.active_model_id,
  runtime_model_name: health.active_model_id,
  model_path: activeModelDetails?.path || '',
  quant: activeLocalModel?.quantization || activeModelQuant,
  provider_kind: 'local',
  status: health.generation_ready ? 'ready' : 'registered',
  loaded_now: Boolean(health.loaded_now ?? health.active_model_id),
  generation_ready: Boolean(health.generation_ready),
} : null
const activeCompatibilityHint = activeModel ? findCompatibilityHint(capabilities, activeModel) : null
// Mirror chatGate's fail-closed identity precedence: once /api/models/local has
// supplied a lane verdict, display-name matching may not override it.
const activeSupportedByIdentity = activeModel
  ? activeModel.lane_class
    ? activeModel.lane_class === 'supported'
    : isCompatibilitySupportedForModel(capabilities, activeModel)
  : false
if (activeModel) {
  console.log(`✓ WebUI chat gate model=${activeModel.id}; quant=${activeModel.quant || 'unknown'}; backend_lane=${activeModel.lane_class || 'not reported'}; ${compatibilityHintLabel(activeCompatibilityHint, 'no exact support row')}; artifact_supported=${activeSupportedByIdentity}`)
  console.log(`  WebUI contract note=${compatibilityHintCopy(activeCompatibilityHint)}`)
}

if (expectCompatibilityRow) {
  assertExpected('compatibility row', activeCompatibilityHint?.target?.id || null, expectCompatibilityRow)
}
if (expectCompatibilityStatus) {
  assertExpected('compatibility status', activeCompatibilityHint?.target?.status || null, expectCompatibilityStatus)
}
if (requireGeneration && !health.generation_ready) {
  throw new Error('generation_ready=false after smoke setup; omit --require-generation to allow metadata/UI guardrail smoke runs')
}

const activeModelListed = Boolean(activeModel && modelIds.includes(activeModel.id))
const activeChatGate = activeModel ? getChatGateState(capabilities, activeModel, { active_model_id: health.active_model_id, loaded_now: health.loaded_now ?? Boolean(health.active_model_id), generation_ready: Boolean(health.generation_ready) }) : null
const webuiChatState = !activeModelListed || !activeChatGate
  ? 'blocked'
  : activeChatGate.chatMode === 'supported'
    ? 'enabled'
    : activeChatGate.chatMode === 'verified'
      ? 'verified'
    : activeChatGate.chatMode === 'variance'
      ? 'variance'
    : activeChatGate.chatMode === 'experimental'
      ? 'experimental'
      : 'blocked'
const webuiChatEnabled = ['enabled', 'verified', 'variance', 'experimental'].includes(webuiChatState)
const qaChatBypass = Boolean(allowGuardedChat && health.generation_ready && activeModel && activeModelListed && !webuiChatEnabled)

if (expectContractSupported !== null) {
  assertExpected('contract supported', Boolean(activeChatGate?.contractSupported), expectContractSupported)
}

if (expectWebUiChat) {
  if (!['enabled', 'verified', 'variance', 'experimental', 'blocked'].includes(expectWebUiChat)) {
    throw new Error(`--expect-webui-chat must be one of enabled, verified, variance, experimental, blocked; got ${expectWebUiChat}`)
  }
  assertExpected('WebUI chat state', webuiChatState, expectWebUiChat)
}

if (qaChatBypass) {
  console.log('ℹ allow-guarded-chat enabled: WebUI remains blocked, but the smoke harness will run one backend QA chat outside the exact supported /api/capabilities row')
}

if (webuiChatEnabled || qaChatBypass) {
  const { result: streamingChat, elapsedMs: streamingChatMs } = await timed('chat_completion_stream', () => fetchStreamingChatCompletion(`${apiBase}/v1/chat/completions`, {
    model: health.active_model_id || modelIds[0],
    messages: [{ role: 'user', content: chatMessage }],
    // Thinking rows can spend their first tokens in a hidden reasoning channel.
    // Keep the default bounded, but let exact-row qualification raise it enough
    // to require an eventual visible content delta without fabricating a failure.
    max_tokens: streamMaxTokens,
    stream: true,
    stream_options: { include_usage: true },
    temperature: 0,
  }))
  const reasoningUpdates = streamingChat.streamEvents.filter(event => event === 'reasoning').length
  console.log(`✓ streaming chat published ${streamingChat.streamedSnapshots.length} visible and ${reasoningUpdates} reasoning update(s) before final completion in ${(streamingChatMs / 1000).toFixed(2)}s: ${JSON.stringify(streamingChat.streamed.content)}`)
  console.log(`  stream_events=${streamingChat.streamEvents.join(',')}`)
  console.log(`  terminal_finish_reason=${streamingChat.streamed.finishReason}; terminal_usage=${JSON.stringify(streamingChat.streamed.usage)}`)

  const chatTimings = []
  for (let idx = 0; idx < chatRepeats; idx += 1) {
    const repeatLabel = chatRepeats === 1 ? 'chat_completion' : `chat_completion_${idx + 1}`
    const { result: chat, elapsedMs: chatMs } = await timed(repeatLabel, () => fetchJson(`${apiBase}/v1/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({
        model: health.active_model_id || modelIds[0],
        messages: [{ role: 'user', content: chatMessage }],
        max_tokens: 1,
        stream: false,
        temperature: 0,
      }),
    }))
    const text = chat?.choices?.[0]?.message?.content
    if (typeof text !== 'string') throw new Error('chat response did not include choices[0].message.content')
    const diagnostics = chat?.camelid
    if (diagnostics?.prompt_token_ids) {
      console.log(`  backend_prompt_token_ids=${JSON.stringify(diagnostics.prompt_token_ids)}`)
    }
    if (diagnostics?.generated_token_ids) {
      console.log(`  backend_generated_token_ids=${JSON.stringify(diagnostics.generated_token_ids)}`)
    }
    const timings = diagnostics?.timings_ms
    if (timings) {
      console.log(`  backend_tokenize_ms=${timings.tokenize}`)
      console.log(`  backend_weight_load_ms=${timings.weight_load}`)
      console.log(`  backend_weight_cache_hit=${timings.weight_cache_hit}`)
      console.log(`  backend_session_create_ms=${timings.session_create}`)
      console.log(`  backend_generate_ms=${timings.generate}`)
      if (timings.generation) {
        console.log(`  backend_forward_total_ms=${timings.generation.forward_total}`)
        console.log(`  backend_layers_total_ms=${timings.generation.layers_total}`)
        console.log(`  backend_logits_ms=${timings.generation.logits}`)
        console.log(`  backend_sample_ms=${timings.generation.sample}`)
      }
      if (Array.isArray(timings.layers) && timings.layers.length > 0) {
        const sum = (field) => timings.layers.reduce((total, layer) => total + Number(layer[field] || 0), 0)
        const ffnGateMs = sum('ffn_gate')
        const ffnUpMs = sum('ffn_up')
        const ffnActivationMs = sum('ffn_activation')
        const ffnDownMs = sum('ffn_down')
        console.log(`  backend_layer_attention_proj_ms=${sum('attention_q') + sum('attention_k') + sum('attention_v') + sum('attention_output')}`)
        console.log(`  backend_layer_attention_context_ms=${sum('attention_context')}`)
        console.log(`  backend_layer_ffn_gate_ms=${ffnGateMs}`)
        console.log(`  backend_layer_ffn_up_ms=${ffnUpMs}`)
        console.log(`  backend_layer_ffn_activation_ms=${ffnActivationMs}`)
        console.log(`  backend_layer_ffn_down_ms=${ffnDownMs}`)
        console.log(`  backend_layer_ffn_ms=${ffnGateMs + ffnUpMs + ffnActivationMs + ffnDownMs}`)
      }
      chatTimings.push({ chatMs, timings })
    }
    console.log(`✓ chat completion ${idx + 1}/${chatRepeats} returned in ${(chatMs / 1000).toFixed(2)}s: ${JSON.stringify(text)}`)
  }
  if (chatTimings.length > 1) {
    const warm = chatTimings.slice(1)
    const avg = (values) => values.reduce((sum, value) => sum + value, 0) / values.length
    console.log(`  warm_chat_avg_ms=${avg(warm.map(({ chatMs }) => chatMs)).toFixed(0)}`)
    console.log(`  warm_backend_generate_avg_ms=${avg(warm.map(({ timings }) => Number(timings.generate))).toFixed(0)}`)
    if (warm.every(({ timings }) => timings.generation)) {
      console.log(`  warm_backend_forward_avg_ms=${avg(warm.map(({ timings }) => Number(timings.generation.forward_total))).toFixed(0)}`)
      console.log(`  warm_backend_layers_avg_ms=${avg(warm.map(({ timings }) => Number(timings.generation.layers_total))).toFixed(0)}`)
      console.log(`  warm_backend_logits_avg_ms=${avg(warm.map(({ timings }) => Number(timings.generation.logits))).toFixed(0)}`)
    }
  }
} else {
  const reason = !health.generation_ready
    ? 'generation is not ready'
    : activeChatGate?.chatMode === 'blocked'
      ? 'the active model is blocked by the WebUI chat gate'
      : 'no active model is listed by /v1/models'
  console.log(`ℹ chat completion skipped because ${reason}; frontend should keep chat disabled${allowGuardedChat ? ' until a QA-only guarded chat rerun is requested' : ''}`)
}
