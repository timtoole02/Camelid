#!/usr/bin/env node

import { createHash, randomBytes } from 'node:crypto'
import { execFile, spawn } from 'node:child_process'
import { createReadStream } from 'node:fs'
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  statfs,
  writeFile,
} from 'node:fs/promises'
import { freemem } from 'node:os'
import { createServer } from 'node:net'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { setTimeout as sleep } from 'node:timers/promises'
import { promisify, types as utilTypes } from 'node:util'
import { canonicalGitTextBytes } from './lib/canonical-git-text.mjs'

// Only row-neutral, already hardened lifecycle primitives are reused. The
// parity identity, requests, response normalization, receipt, validator, and
// CLI remain exact-closed and independently owned by this file.
import {
  acquireWindowsArtifactReadLock,
  assertAutoSelectRootsEmpty,
  canonicalJson,
  classifySmolLM3LoadSmokeError,
  createResourceGuard,
  inspectProvenance,
  sealReceipt,
  SmolLM3LoadSmokeError,
  startCamelidProcess,
  terminateSpawnedChild,
} from './hf-qualification-smollm3-load-smoke.mjs'

const execFileAsync = promisify(execFile)

const RECEIPT_SCHEMA = 'camelid.model-qualification.chat-parity-preparation/v1'
const ROW_ID = 'smollm3_3b_q8_0'
const BINARY_PROFILE = 'release-fat-lto'
const CAMELID_RELEASE_VERSION = '0.6.1'
const CAMELID_ADDR = '127.0.0.1:8297'
const LLAMA_ADDR = '127.0.0.1:8299'
const CAMELID_ORIGIN = `http://${CAMELID_ADDR}`
const LLAMA_ORIGIN = `http://${LLAMA_ADDR}`
const DIAGNOSTICS_STATUS = 'operator-requested RSS timings enabled; performance claims disabled'
const DATE_PLACEHOLDER = '{{CURRENT_DATE_DD_MONTH_YYYY}}'
const DATE_PATTERN = /Today Date: (\d{2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\n/g

const EXACT_ROW = Object.freeze({
  id: ROW_ID,
  family: 'smollm3',
  architecture: 'smollm3',
  quantization: 'Q8_0',
  target_tier: 'experimental_exact_row',
  disposition: 'hold',
  source: Object.freeze({
    repo: 'ggml-org/SmolLM3-3B-GGUF',
    file: 'SmolLM3-Q8_0.gguf',
    revision: '4965cb60b150737b68a0408c36aeefb65078f894',
    size_bytes: 3_275_574_624,
    sha256: '8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248',
    license: 'apache-2.0',
  }),
})

const TEMPLATE_IDENTITY = Object.freeze({
  utf8_bytes: 5_493,
  sha256: 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e',
})

const GROUNDING_FILES = Object.freeze({
  source_lock: Object.freeze({
    path: 'qa/model-qualification/smollm3-3b-q8-source-lock.json',
    sha256: '24ef388389050e55f32698a5095da386efe7d8316f85552d64cd39e2e4dcfaf9',
  }),
  header_receipt: Object.freeze({
    path: 'qa/model-qualification/smollm3-3b-q8-header-inspection.json',
    sha256: '6faa8cee4a70b5821f485e2debc8a9263d02ebbf5c346138221e9f78a46c9dae',
  }),
  tokenizer_receipt: Object.freeze({
    path: 'qa/model-qualification/smollm3-3b-q8-header-tokenizer-parity.json',
    sha256: '4e3f3b74346b4b005a462ab888a50d9bacdd01cc5a20d05e57682d70eff9afe4',
  }),
  shape_pack: Object.freeze({
    path: 'qa/prompt-packs/smollm3-chat-template-shapes-v1.json',
    sha256: 'd46794448b7c2585d0aa83dfd7bb17d4904c2dcbc048ade1ef68cd3863166de6',
  }),
  runtime_envelope: Object.freeze({
    path: 'qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json',
    sha256: 'c99b9752db771529ff447a05993a6f1119095c15b53fca3bf3491468c2c8590e',
  }),
})

const RENDERER_GIT_BLOB_SHA1 = 'ecfac8933fd4d550c8e45f2822e7560e5f0a69ff'
const SHAPE_CASE_ID = 'default_think_single_user_generation_prompt'
const NORMALIZED_PROMPT_UTF8_BYTES = 1_392
const NORMALIZED_PROMPT_SHA256 = '7619416ae94ba9a00378d976bfa944f5ba726747f9b67ba4e862d9a7fe20e4f1'

const LLAMA_PIN = Object.freeze({
  project: 'llama.cpp',
  build: 9_632,
  revision: 'acd79d603cb2e1c84c0886137b80f1ad649b6857',
  reported_revision: 'acd79d603',
  platform: 'win32-x64',
  executable: 'llama-server.exe',
  executable_size_bytes: 9_216,
  executable_sha256: '6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae',
  server_impl_executable: 'llama-server-impl.dll',
  server_impl_size_bytes: 9_072_128,
  server_impl_sha256: 'df4fd737e380ba2cd39654af9233da22e2cdcbbf43ac376009cbec47aaaec750',
  package_file_count: 51,
  package_manifest_bytes: 4_682,
  package_manifest_sha256: 'd70bbe8beb7848396d0993ee533062c200350fd9961e2b92c799b24f94a33e93',
  archive: 'llama-b9632-bin-win-cpu-x64.zip',
  archive_size_bytes: 16_899_258,
  archive_sha256: 'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd',
})

const SAFE_CAMELID_ENV = Object.freeze({
  CAMELID_PROFILE: 'safe',
  CAMELID_LAZY_Q8_0_LINEAR: '1',
  CAMELID_X86_Q8_REPACK: 'off',
  CAMELID_Q8_0_FILE_CACHE_BYTES: '0',
  CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES: '0',
  CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES: '1073741824',
  CAMELID_MAX_KV_CACHE_BYTES: '268435456',
  CAMELID_KV_POOL_BUDGET_BYTES: '268435456',
  CAMELID_FORWARD_RSS_TIMINGS: 'on',
  CAMELID_GENERATION_TIMEOUT_MS: '1800000',
  CAMELID_QUEUE_DEPTH: '1',
  CAMELID_CONTINUOUS_BATCH_SLOTS: '1',
  CAMELID_NO_REMOTE_DIMS: '1',
  CAMELID_PREFIX_CACHE_MIN_TOKENS: '1024',
})

const SAFE_LLAMA_ENV = Object.freeze({ CUDA_VISIBLE_DEVICES: '-1' })

const WINDOWS_CHILD_ENV_ALLOWLIST = Object.freeze([
  'COMSPEC',
  'PATH',
  'PATHEXT',
  'SYSTEMDRIVE',
  'SYSTEMROOT',
  'TEMP',
  'TMP',
  'WINDIR',
])

const LOCAL_FILE_LIMITS = Object.freeze({
  binary_bytes: 512 * 1024 ** 2,
  grounding_bytes: 1024 * 1024,
  oracle_file_bytes: 256 * 1024 ** 2,
  oracle_package_bytes: 1024 * 1024 ** 2,
})

const GROUNDING_FILE_SIZE_BYTES = Object.freeze({
  source_lock: 514,
  header_receipt: 2_531,
  tokenizer_receipt: 10_847,
  shape_pack: 14_968,
  runtime_envelope: 5_578,
})

const LIMITS = Object.freeze({
  startup_timeout_ms: 60_000,
  load_timeout_ms: 20 * 60_000,
  generation_timeout_ms: 31 * 60_000,
  ordinary_request_timeout_ms: 30_000,
  monitor_interval_ms: 1_000,
  low_memory_abort_bytes: 1 * 1024 ** 3,
  camelid_child_working_set_abort_bytes: 2 * 1024 ** 3,
  llama_child_working_set_abort_bytes: 4 * 1024 ** 3,
  consecutive_abort_samples: 2,
  preflight_disk_bytes: 4 * 1024 ** 3,
  preflight_physical_bytes: 6 * 1024 ** 3,
  max_response_bytes: 16 * 1024 ** 2,
  max_prompt_tokens: 1_024,
  generated_tokens: 4,
})

const CHAT_REQUEST = Object.freeze({
  model: ROW_ID,
  messages: Object.freeze([Object.freeze({ role: 'user', content: 'Hello, please help me.' })]),
  max_tokens: 4,
  temperature: 0,
  top_k: 1,
  seed: 0,
  stream: false,
})

const APPLY_TEMPLATE_REQUEST = Object.freeze({ messages: CHAT_REQUEST.messages })
const TOKENIZE_FLAGS = Object.freeze({ add_special: false, parse_special: true })
const LLAMA_COMPLETION_SETTINGS = Object.freeze({
  n_predict: 4,
  temperature: 0,
  top_k: 1,
  seed: 0,
  cache_prompt: false,
  samplers: Object.freeze(['top_k']),
  return_tokens: true,
  stream: false,
})
const EOG_TOKEN_IDS = Object.freeze([128_012])

const STEP_CONTRACT = Object.freeze([
  Object.freeze(['camelid_baseline_health', 'camelid', 'GET', '/v1/health']),
  Object.freeze(['camelid_baseline_gpu', 'camelid', 'GET', '/api/runtime/gpu']),
  Object.freeze(['camelid_load', 'camelid', 'POST', '/models/load']),
  Object.freeze(['camelid_verify_identity', 'camelid', 'GET', '/api/models/verify']),
  Object.freeze(['camelid_loaded_health', 'camelid', 'GET', '/v1/health']),
  Object.freeze(['camelid_props', 'camelid', 'GET', '/props']),
  Object.freeze(['camelid_apply_template_before', 'camelid', 'POST', '/apply-template']),
  Object.freeze(['camelid_tokenize_prompt', 'camelid', 'POST', '/tokenize']),
  Object.freeze(['camelid_chat_first_forward', 'camelid', 'POST', '/v1/chat/completions']),
  Object.freeze(['camelid_detokenize_generated', 'camelid', 'POST', '/detokenize']),
  Object.freeze(['camelid_apply_template_after', 'camelid', 'POST', '/apply-template']),
  Object.freeze(['camelid_final_health', 'camelid', 'GET', '/v1/health']),
  Object.freeze(['camelid_final_gpu', 'camelid', 'GET', '/api/runtime/gpu']),
  Object.freeze(['llama_health', 'llama_cpp', 'GET', '/health']),
  Object.freeze(['llama_tokenize_prompt', 'llama_cpp', 'POST', '/tokenize']),
  Object.freeze(['llama_completion_first_forward', 'llama_cpp', 'POST', '/completion']),
  Object.freeze(['llama_detokenize_generated', 'llama_cpp', 'POST', '/detokenize']),
  Object.freeze(['llama_final_health', 'llama_cpp', 'GET', '/health']),
])

const DOES_NOT_PROVE = Object.freeze([
  'the roster parity matrix at token counts 1, 5, and 50',
  'the blocked load-smoke gate',
  'the blocked full template gate',
  'api_webui or SSE qualification',
  'context-512 qualification',
  'support or promotion',
  'performance or throughput',
  'GPU execution or GPU parity',
  'adjacent sizes, variants, or quantizations',
  'more than one final-user default-thinking text case',
  'system, tools, no-think, multimodal, streaming, or non-text chat shapes',
])

const ERROR_CONTRACTS = Object.freeze({
  chat_parity_options_invalid: ['blocked', 'the SmolLM3 chat-parity invocation is incomplete or unsafe'],
  chat_parity_platform_invalid: ['blocked', 'the exact preparation requires Windows x86_64'],
  chat_parity_grounding_invalid: ['fail', 'a pinned SmolLM3 grounding identity changed or is internally inconsistent'],
  chat_parity_artifact_unavailable: ['blocked', 'the ignored exact SmolLM3 artifact is unavailable'],
  chat_parity_artifact_identity_mismatch: ['fail', 'the artifact does not match the exact SmolLM3 source lock'],
  chat_parity_artifact_not_ignored: ['blocked', 'the full artifact is not contained under an ignored path'],
  chat_parity_artifact_lock_failed: ['blocked', 'the exact artifact could not be protected by a Windows read-share lock'],
  chat_parity_artifact_lock_lost: ['fail', 'the Windows artifact lock exited before both engines and posthash completed'],
  chat_parity_artifact_lock_release_failed: ['blocked', 'the Windows artifact lock could not be released and observed cleanly'],
  chat_parity_source_dirty: ['blocked', 'tracked source and the Camelid binary must be one clean exact build'],
  chat_parity_source_changed: ['blocked', 'source, binary, groundings, selectors, or oracle package changed during preparation'],
  chat_parity_oracle_unavailable: ['blocked', 'the pinned llama.cpp b9632 CPU package is unavailable'],
  chat_parity_oracle_identity_mismatch: ['fail', 'the llama.cpp binary, package manifest, archive, or version does not match the pin'],
  chat_parity_auto_select_invalid: ['blocked', 'an isolated no-model Camelid auto-selection root is unsafe'],
  chat_parity_port_in_use: ['blocked', 'an isolated loopback qualification port is already in use'],
  chat_parity_engine_present: ['blocked', 'a pre-existing Camelid or llama-server process would violate solo-engine isolation'],
  chat_parity_resources_low: ['blocked', 'preflight disk or physical memory is below the fixed safety budget'],
  chat_parity_process_start_failed: ['blocked', 'an exact engine child could not start'],
  chat_parity_process_exited: ['fail', 'an exact engine child exited before its bounded phase completed'],
  chat_parity_startup_timeout: ['blocked', 'an engine did not become healthy within the startup budget'],
  chat_parity_http_failed: ['blocked', 'an isolated loopback request failed, timed out, or exceeded its response bound'],
  chat_parity_camelid_contract_invalid: ['fail', 'Camelid did not meet the exact no-model, load, template, tokenizer, or first-chat contract'],
  chat_parity_llama_contract_invalid: ['fail', 'llama.cpp did not meet the exact tokenize, completion, or detokenize contract'],
  chat_parity_resource_abort: ['blocked', 'an engine crossed a fixed memory-safety abort threshold'],
  chat_parity_resource_telemetry_unavailable: ['blocked', 'required child resource telemetry became unavailable'],
  chat_parity_warmup_detected: ['fail', 'startup generation warm-up violated the first-forward claim'],
  chat_parity_overlap_detected: ['fail', 'Camelid and llama.cpp engine lifetimes overlapped'],
  chat_parity_termination_failed: ['blocked', 'an exact spawned child could not be terminated and drained'],
  chat_parity_receipt_invalid: ['fail', 'the sealed parity-preparation receipt failed its durable contract'],
  chat_parity_output_failed: ['blocked', 'the sealed receipt could not be written atomically'],
})

const ERROR_CODES = new WeakMap()

class SmolLM3ChatParityError extends Error {
  constructor(code) {
    const canonical = Object.hasOwn(ERROR_CONTRACTS, code) ? code : 'chat_parity_http_failed'
    super(ERROR_CONTRACTS[canonical][1])
    this.name = 'SmolLM3ChatParityError'
    this.status = ERROR_CONTRACTS[canonical][0]
    ERROR_CODES.set(this, canonical)
    Object.defineProperty(this, 'code', {
      configurable: false,
      enumerable: true,
      get: () => ERROR_CODES.get(this),
    })
  }
}

function parityError(code) { return new SmolLM3ChatParityError(code) }

function bridgeSmolError(error) {
  if (!(error instanceof SmolLM3LoadSmokeError)) return null
  const code = classifySmolLM3LoadSmokeError(error).error_code
  return ({
    load_smoke_resource_abort: 'chat_parity_resource_abort',
    load_smoke_resource_telemetry_unavailable: 'chat_parity_resource_telemetry_unavailable',
    load_smoke_process_start_failed: 'chat_parity_process_start_failed',
    load_smoke_process_exited: 'chat_parity_process_exited',
    load_smoke_termination_failed: 'chat_parity_termination_failed',
    load_smoke_artifact_lock_failed: 'chat_parity_artifact_lock_failed',
    load_smoke_artifact_lock_lost: 'chat_parity_artifact_lock_lost',
    load_smoke_artifact_lock_release_failed: 'chat_parity_artifact_lock_release_failed',
    load_smoke_source_dirty: 'chat_parity_source_dirty',
    load_smoke_binary_stale: 'chat_parity_source_dirty',
    load_smoke_auto_select_root_invalid: 'chat_parity_auto_select_invalid',
    load_smoke_auto_select_candidate_present: 'chat_parity_auto_select_invalid',
  })[code] || null
}

function classifySmolLM3ChatParityError(error) {
  let code = null
  if (error instanceof SmolLM3ChatParityError) code = ERROR_CODES.get(error)
  code ||= bridgeSmolError(error)
  if (!Object.hasOwn(ERROR_CONTRACTS, code)) code = 'chat_parity_http_failed'
  return { status: ERROR_CONTRACTS[code][0], error_code: code, reason: ERROR_CONTRACTS[code][1] }
}

function expect(condition, code) { if (!condition) throw parityError(code) }
function sha256(value) { return createHash('sha256').update(value).digest('hex') }
function sameJson(left, right) { return canonicalJson(left) === canonicalJson(right) }
function finiteNumber(value) { return typeof value === 'number' && Number.isFinite(value) }
function nonNegativeInteger(value) { return Number.isSafeInteger(value) && value >= 0 }
function positiveInteger(value) { return Number.isSafeInteger(value) && value > 0 }
function tokenArraySha256(tokens) { return sha256(Buffer.from(JSON.stringify(tokens), 'utf8')) }

function buildWindowsChildEnv(inherited = process.env) {
  const clean = {}
  for (const [key, value] of Object.entries(inherited)) {
    const canonicalKey = key.toUpperCase()
    if (!WINDOWS_CHILD_ENV_ALLOWLIST.includes(canonicalKey)) continue
    expect(typeof value === 'string' && !Object.hasOwn(clean, canonicalKey),
      'chat_parity_options_invalid')
    clean[canonicalKey] = value
  }
  return Object.freeze(clean)
}

function buildChildEnv(inherited = process.env) {
  const clean = buildWindowsChildEnv(inherited)
  return Object.freeze({ ...clean, ...SAFE_CAMELID_ENV })
}

function buildLlamaEnv(inherited = process.env) {
  const clean = buildWindowsChildEnv(inherited)
  return Object.freeze({ ...clean, ...SAFE_LLAMA_ENV })
}

function hardenWindowsChildDeps(deps = {}) {
  const inheritedEnv = Object.freeze(Object.fromEntries(
    Object.entries(deps.inheritedEnv ?? process.env)
      .filter(([, value]) => typeof value === 'string'),
  ))
  const execFileImpl = deps.execFileImpl || execFileAsync
  const spawnImpl = deps.spawnImpl || spawn
  return {
    ...deps,
    inheritedEnv,
    execFileImpl: (file, args, options = {}) => execFileImpl(file, args, {
      ...options,
      env: buildWindowsChildEnv(inheritedEnv),
    }),
    spawnImpl: (file, args, options = {}) => spawnImpl(file, args, {
      ...options,
      env: options.env || buildWindowsChildEnv(inheritedEnv),
    }),
  }
}

function childEnvSubset(env, prefix) {
  return Object.fromEntries(Object.entries(env)
    .filter(([key]) => key.toUpperCase().startsWith(prefix))
    .sort(([left], [right]) => left.localeCompare(right)))
}

function describeChildEnvironment(env, modelOverrides) {
  const inheritedKeys = Object.keys(env)
    .filter((key) => WINDOWS_CHILD_ENV_ALLOWLIST.includes(key))
    .sort()
  const inheritedEnvironment = Object.fromEntries(
    inheritedKeys.map((key) => [key, env[key]]),
  )
  const effectiveKeys = Object.keys(env).sort()
  const expectedKeys = [...inheritedKeys, ...Object.keys(modelOverrides)].sort()
  expect(sameJson(effectiveKeys, expectedKeys)
    && sameJson(Object.fromEntries(Object.entries(env)
      .filter(([key]) => Object.hasOwn(modelOverrides, key))), modelOverrides),
  'chat_parity_options_invalid')
  return {
    schema: 'camelid.windows-child-environment/v1',
    model_overrides: structuredClone(modelOverrides),
    inherited_os_allowlist: [...WINDOWS_CHILD_ENV_ALLOWLIST],
    inherited_os_keys_present: inheritedKeys,
    inherited_os_values_redacted: true,
    inherited_os_environment_sha256: sha256(Buffer.from(
      `camelid.windows-child-environment/v1\0${canonicalJson(inheritedEnvironment)}`,
      'utf8',
    )),
    effective_keys: effectiveKeys,
    unlisted_keys_present: false,
  }
}

function buildCamelidServeArgs(modelsDir) {
  expect(typeof modelsDir === 'string' && isAbsolute(modelsDir), 'chat_parity_options_invalid')
  return [
    'serve', '--addr', CAMELID_ADDR, '--models-dir', modelsDir, '--threads', '4',
    '--gpu', 'off', '--deterministic', '--kv-quant', 'f16', '--no-open',
    '--max-prompt-tokens', String(LIMITS.max_prompt_tokens),
    '--max-generation-tokens', String(LIMITS.generated_tokens),
  ]
}

function buildLlamaServeArgs(artifact) {
  expect(typeof artifact === 'string' && isAbsolute(artifact), 'chat_parity_options_invalid')
  return [
    '--host', '127.0.0.1', '--port', '8299', '-m', artifact,
    '-ngl', '0', '-c', '512', '-b', '512', '-ub', '512', '-t', '4',
    '-ctk', 'f16', '-ctv', 'f16', '-fa', 'off', '--no-repack', '--no-warmup',
    '-np', '1', '--no-cont-batching',
  ]
}

function receiptCamelidCommand() {
  const placeholder = resolve('<empty-models-dir>')
  return ['<camelid>', ...buildCamelidServeArgs(placeholder)
    .map((value) => value === placeholder ? '<empty-models-dir>' : value)]
}

function receiptLlamaCommand() {
  const placeholder = resolve('<artifact>')
  return ['<llama-server>', ...buildLlamaServeArgs(placeholder)
    .map((value) => value === placeholder ? '<artifact>' : value)]
}

async function sha256File(path, {
  createReadStreamImpl = createReadStream,
  expectedSize = null,
  maxBytes = Number.MAX_SAFE_INTEGER,
} = {}) {
  const digest = createHash('sha256')
  let observedBytes = 0
  await new Promise((resolvePromise, rejectPromise) => {
    const input = createReadStreamImpl(path)
    let settled = false
    const rejectOnce = (error) => {
      if (settled) return
      settled = true
      input.destroy?.()
      rejectPromise(error)
    }
    input.on('data', (chunk) => {
      if (settled) return
      observedBytes += chunk.length
      if (observedBytes > maxBytes
        || (expectedSize !== null && observedBytes > expectedSize)) {
        rejectOnce(new Error('local file crossed its byte bound while hashing'))
        return
      }
      digest.update(chunk)
    })
    input.once('end', () => {
      if (settled) return
      if (expectedSize !== null && observedBytes !== expectedSize) {
        rejectOnce(new Error('local file size changed while hashing'))
        return
      }
      settled = true
      resolvePromise()
    })
    input.once('error', rejectOnce)
  })
  return digest.digest('hex')
}

async function inspectRegularFileMetadata(path, {
  lstatImpl = lstat,
  statImpl = stat,
  expectedSize = null,
  maxBytes = Number.MAX_SAFE_INTEGER,
  unavailableCode,
  mismatchCode,
} = {}) {
  let linkStats
  let fileStats
  try { linkStats = await lstatImpl(path) }
  catch { throw parityError(unavailableCode) }
  expect(linkStats?.isFile?.() === true && linkStats?.isSymbolicLink?.() === false,
    mismatchCode)
  try { fileStats = await statImpl(path) }
  catch { throw parityError(unavailableCode) }
  expect(fileStats?.isFile?.() === true
    && Number.isSafeInteger(fileStats.size)
    && fileStats.size >= 0
    && fileStats.size <= maxBytes
    && (expectedSize === null || fileStats.size === expectedSize),
  mismatchCode)
  return fileStats
}

async function inspectExactArtifactIdentity(path, {
  lstatImpl = lstat,
  statImpl = stat,
  sha256FileImpl = (candidate, options) => sha256File(candidate, options),
} = {}) {
  const fileStats = await inspectRegularFileMetadata(path, {
    lstatImpl,
    statImpl,
    expectedSize: EXACT_ROW.source.size_bytes,
    maxBytes: EXACT_ROW.source.size_bytes,
    unavailableCode: 'chat_parity_artifact_unavailable',
    mismatchCode: 'chat_parity_artifact_identity_mismatch',
  })
  let digest
  try {
    digest = await sha256FileImpl(path, {
      expectedSize: EXACT_ROW.source.size_bytes,
      maxBytes: EXACT_ROW.source.size_bytes,
    })
  } catch {
    throw parityError('chat_parity_artifact_unavailable')
  }
  expect(digest === EXACT_ROW.source.sha256, 'chat_parity_artifact_identity_mismatch')
  return { size_bytes: fileStats.size, sha256: digest }
}

async function inspectLlamaPackage(server, {
  platformInfo = () => ({ platform: process.platform, arch: process.arch }),
  lstatImpl = lstat,
  statImpl = stat,
  sha256FileImpl = (candidate, options) => sha256File(candidate, options),
  readdirImpl = readdir,
  execFileImpl = execFileAsync,
  inheritedEnv = process.env,
} = {}) {
  const platform = platformInfo()
  expect(platform.platform === 'win32' && platform.arch === 'x64',
    'chat_parity_oracle_identity_mismatch')
  expect(basename(server) === LLAMA_PIN.executable, 'chat_parity_oracle_identity_mismatch')
  const binDir = dirname(server)
  let entries
  try { entries = await readdirImpl(binDir, { withFileTypes: true }) }
  catch { throw parityError('chat_parity_oracle_unavailable') }
  expect(Array.isArray(entries) && entries.every((entry) => entry?.isFile?.()),
    'chat_parity_oracle_identity_mismatch')
  const names = entries.map((entry) => String(entry.name)).sort()
  expect(names.length === LLAMA_PIN.package_file_count
    && names.every((name) => /^[a-z0-9._+-]+$/.test(name)),
  'chat_parity_oracle_identity_mismatch')
  const lines = []
  const identities = new Map()
  let aggregateBytes = 0
  for (const name of names) {
    const expectedSize = name === LLAMA_PIN.executable
      ? LLAMA_PIN.executable_size_bytes
      : name === LLAMA_PIN.server_impl_executable
        ? LLAMA_PIN.server_impl_size_bytes : null
    const filePath = join(binDir, name)
    const fileStats = await inspectRegularFileMetadata(filePath, {
      lstatImpl,
      statImpl,
      expectedSize,
      maxBytes: LOCAL_FILE_LIMITS.oracle_file_bytes,
      unavailableCode: 'chat_parity_oracle_unavailable',
      mismatchCode: 'chat_parity_oracle_identity_mismatch',
    })
    aggregateBytes += fileStats.size
    expect(aggregateBytes <= LOCAL_FILE_LIMITS.oracle_package_bytes,
      'chat_parity_oracle_identity_mismatch')
    let digest
    try {
      digest = await sha256FileImpl(filePath, {
        expectedSize: fileStats.size,
        maxBytes: LOCAL_FILE_LIMITS.oracle_file_bytes,
      })
    } catch { throw parityError('chat_parity_oracle_unavailable') }
    const identity = { size_bytes: fileStats.size, sha256: digest }
    identities.set(name, identity)
    lines.push(`${name}\t${identity.size_bytes}\t${identity.sha256}\n`)
  }
  const manifest = Buffer.from(lines.join(''), 'utf8')
  const archivePath = join(dirname(binDir), LLAMA_PIN.archive)
  const archiveStats = await inspectRegularFileMetadata(archivePath, {
    lstatImpl,
    statImpl,
    expectedSize: LLAMA_PIN.archive_size_bytes,
    maxBytes: LLAMA_PIN.archive_size_bytes,
    unavailableCode: 'chat_parity_oracle_unavailable',
    mismatchCode: 'chat_parity_oracle_identity_mismatch',
  })
  aggregateBytes += archiveStats.size
  expect(aggregateBytes <= LOCAL_FILE_LIMITS.oracle_package_bytes,
    'chat_parity_oracle_identity_mismatch')
  let archiveSha256
  try {
    archiveSha256 = await sha256FileImpl(archivePath, {
      expectedSize: LLAMA_PIN.archive_size_bytes,
      maxBytes: LLAMA_PIN.archive_size_bytes,
    })
  } catch { throw parityError('chat_parity_oracle_unavailable') }
  let versionOutput
  try {
    const result = await execFileImpl(server, ['--version'], {
      timeout: 10_000, maxBuffer: 1024 * 1024, windowsHide: true,
      env: buildWindowsChildEnv(inheritedEnv),
    })
    versionOutput = `${result.stdout || ''}\n${result.stderr || ''}`
  } catch { throw parityError('chat_parity_oracle_unavailable') }
  const version = /version:\s*(\d+)\s*\(([0-9a-f]{9})\)/i.exec(versionOutput)
  const target = /built with [^\r\n]+ for Windows x86_64/i.test(versionOutput)
  const executable = identities.get(LLAMA_PIN.executable)
  const serverImpl = identities.get(LLAMA_PIN.server_impl_executable)
  expect(version && Number(version[1]) === LLAMA_PIN.build
    && version[2].toLowerCase() === LLAMA_PIN.reported_revision && target
    && executable?.size_bytes === LLAMA_PIN.executable_size_bytes
    && executable?.sha256 === LLAMA_PIN.executable_sha256
    && serverImpl?.size_bytes === LLAMA_PIN.server_impl_size_bytes
    && serverImpl?.sha256 === LLAMA_PIN.server_impl_sha256
    && manifest.length === LLAMA_PIN.package_manifest_bytes
    && sha256(manifest) === LLAMA_PIN.package_manifest_sha256
    && archiveStats.size === LLAMA_PIN.archive_size_bytes
    && archiveSha256 === LLAMA_PIN.archive_sha256,
  'chat_parity_oracle_identity_mismatch')
  return {
    ...structuredClone(LLAMA_PIN),
    version_verified: true,
    executable_path_redacted: true,
    package_path_redacted: true,
  }
}

async function inspectGroundings(root, {
  readFileImpl = readFile,
  lstatImpl = lstat,
  statImpl = stat,
  sha256FileImpl = (candidate, options) => sha256File(candidate, options),
  execFileImpl = execFileAsync,
  inheritedEnv = process.env,
} = {}) {
  const documents = {}
  for (const [name, identity] of Object.entries(GROUNDING_FILES)) {
    const path = resolve(root, identity.path)
    const expectedSize = GROUNDING_FILE_SIZE_BYTES[name]
    await inspectRegularFileMetadata(path, {
      lstatImpl,
      statImpl,
      maxBytes: LOCAL_FILE_LIMITS.grounding_bytes,
      unavailableCode: 'chat_parity_grounding_invalid',
      mismatchCode: 'chat_parity_grounding_invalid',
    })
    let streamedDigest
    let bytes
    try {
      streamedDigest = await sha256FileImpl(path, {
        maxBytes: LOCAL_FILE_LIMITS.grounding_bytes,
      })
      bytes = await readFileImpl(path)
    }
    catch { throw parityError('chat_parity_grounding_invalid') }
    const canonicalBytes = Buffer.isBuffer(bytes) ? canonicalGitTextBytes(bytes) : null
    expect(Buffer.isBuffer(bytes) && bytes.length <= LOCAL_FILE_LIMITS.grounding_bytes
      && streamedDigest === sha256(bytes) && canonicalBytes.length === expectedSize
      && sha256(canonicalBytes) === identity.sha256,
    'chat_parity_grounding_invalid')
    try { documents[name] = JSON.parse(canonicalBytes) }
    catch { throw parityError('chat_parity_grounding_invalid') }
  }
  const source = documents.source_lock
  const shape = documents.shape_pack
  const runtime = documents.runtime_envelope
  const tokenizer = documents.tokenizer_receipt
  const shapeCase = shape?.cases?.find((entry) => entry?.id === SHAPE_CASE_ID)
  expect(source?.row_id === ROW_ID && sameJson({
    repo: source?.repo,
    file: source?.file,
    revision: source?.revision,
    size_bytes: source?.size_bytes,
    sha256: source?.sha256,
    license: source?.license,
  }, EXACT_ROW.source),
    'chat_parity_grounding_invalid')
  expect(shape?.row_id === ROW_ID && sameJson(shape?.source, EXACT_ROW.source)
    && shape?.source_template?.utf8_bytes === TEMPLATE_IDENTITY.utf8_bytes
    && shape?.source_template?.sha256 === TEMPLATE_IDENTITY.sha256
    && shapeCase?.enable_thinking === true && shapeCase?.add_generation_prompt === true
    && sameJson(shapeCase?.messages, CHAT_REQUEST.messages)
    && Buffer.byteLength(shapeCase?.normalized_prompt || '', 'utf8') === NORMALIZED_PROMPT_UTF8_BYTES
    && sha256(Buffer.from(shapeCase.normalized_prompt || '', 'utf8')) === NORMALIZED_PROMPT_SHA256
    && shapeCase?.normalized_prompt_utf8_bytes === NORMALIZED_PROMPT_UTF8_BYTES
    && shapeCase?.normalized_prompt_sha256 === NORMALIZED_PROMPT_SHA256
    && shapeCase?.oracle_exact_match_after_date_normalization === true,
  'chat_parity_grounding_invalid')
  expect(runtime?.schema === 'camelid.smollm3-runtime-envelope/v1'
    && runtime?.fixture_id === 'smollm3-default-thinking-runtime-envelope-v1'
    && runtime?.row_id === ROW_ID && runtime?.status === 'partial_renderer_qualified_template_gate_blocked'
    && sameJson(runtime?.source, EXACT_ROW.source)
    && runtime?.template?.utf8_bytes === TEMPLATE_IDENTITY.utf8_bytes
    && runtime?.template?.sha256 === TEMPLATE_IDENTITY.sha256
    && runtime?.qualified_envelope?.content === 'text_only'
    && runtime?.qualified_envelope?.thinking?.includes('omitted_defaults_true')
    && runtime?.qualified_envelope?.add_generation_prompt === true
    && runtime?.gate_decision?.template_gate === 'blocked'
    && runtime?.gate_decision?.support_claim === false
    && runtime?.gate_decision?.disposition === 'hold',
  'chat_parity_grounding_invalid')
  expect(tokenizer?.row_id === ROW_ID
    && tokenizer?.tokenizer_metadata?.eos_token_id === EOG_TOKEN_IDS[0]
    && tokenizer?.tokenizer_metadata?.chat_control_token_ids?.im_end === EOG_TOKEN_IDS[0],
  'chat_parity_grounding_invalid')
  let rendererBlob
  try {
    const { stdout } = await execFileImpl('git', ['-C', root, 'hash-object', '--', 'src/api/mod.rs'], {
      timeout: 10_000, windowsHide: true, env: buildWindowsChildEnv(inheritedEnv),
    })
    rendererBlob = String(stdout).trim().toLowerCase()
  } catch { throw parityError('chat_parity_grounding_invalid') }
  expect(rendererBlob === RENDERER_GIT_BLOB_SHA1, 'chat_parity_grounding_invalid')
  return {
    identities: structuredClone(GROUNDING_FILES),
    renderer_git_blob_sha1: rendererBlob,
    template: structuredClone(TEMPLATE_IDENTITY),
    shape_case: {
      id: SHAPE_CASE_ID,
      normalized_prompt_utf8_bytes: NORMALIZED_PROMPT_UTF8_BYTES,
      normalized_prompt_sha256: NORMALIZED_PROMPT_SHA256,
      date_placeholder: DATE_PLACEHOLDER,
    },
    expected_normalized_prompt: shapeCase.normalized_prompt,
  }
}

function pathInside(parent, candidate) {
  const rel = relative(resolve(parent), resolve(candidate))
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

async function gitPathIgnored(root, path, {
  execFileImpl = execFileAsync,
  inheritedEnv = process.env,
} = {}) {
  try {
    await execFileImpl('git', ['-C', root, 'check-ignore', '--quiet', '--', path], {
      timeout: 10_000,
      env: buildWindowsChildEnv(inheritedEnv),
    })
    return true
  } catch { return false }
}

async function diskFreeBytes(path, { statfsImpl = statfs } = {}) {
  const stats = await statfsImpl(path, { bigint: true })
  return Number(stats.bavail * stats.bsize)
}

async function assertPortFree({ host, port }, { createServerImpl = createServer } = {}) {
  await new Promise((resolvePromise, rejectPromise) => {
    const server = createServerImpl()
    server.unref?.()
    server.once('error', () => rejectPromise(parityError('chat_parity_port_in_use')))
    server.listen({ host, port, exclusive: true }, () => server.close((error) => (
      error ? rejectPromise(parityError('chat_parity_port_in_use')) : resolvePromise()
    )))
  })
}

async function engineProcessesPresent({
  execFileImpl = execFileAsync,
  inheritedEnv = process.env,
} = {}) {
  try {
    const { stdout } = await execFileImpl('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-Command',
      "@(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'camelid*' -or $_.ProcessName -eq 'llama-server' }).Count",
    ], {
      timeout: 10_000,
      windowsHide: true,
      env: buildWindowsChildEnv(inheritedEnv),
    })
    const count = Number(String(stdout).trim())
    expect(nonNegativeInteger(count), 'chat_parity_engine_present')
    return count > 0
  } catch (error) {
    if (error instanceof SmolLM3ChatParityError) throw error
    throw parityError('chat_parity_engine_present')
  }
}

function autoRootsFromOptions(options) {
  return [
    join(dirname(options.binary), 'models'), dirname(options.binary),
    join(options.cwd, 'models'), options.cwd, options.modelsDir,
  ]
}

async function runPreflight(options, deps = {}) {
  deps = hardenWindowsChildDeps(deps)
  const platform = deps.platformInfo?.() || { platform: process.platform, arch: process.arch }
  expect(platform.platform === 'win32' && platform.arch === 'x64', 'chat_parity_platform_invalid')
  for (const value of [options.root, options.binary, options.artifact, options.cwd,
    options.modelsDir, options.llamaServer]) {
    expect(typeof value === 'string' && isAbsolute(value), 'chat_parity_options_invalid')
  }
  expect(options.binaryProfile === BINARY_PROFILE
    && options.cwd !== options.modelsDir
    && dirname(options.binary) !== options.cwd
    && dirname(options.binary) !== options.modelsDir,
  'chat_parity_options_invalid')
  expect(autoRootsFromOptions(options).every((candidate) => !pathInside(candidate, options.artifact)),
    'chat_parity_options_invalid')
  const fileStats = await inspectRegularFileMetadata(options.artifact, {
    lstatImpl: deps.lstatImpl || lstat,
    statImpl: deps.statImpl || stat,
    expectedSize: EXACT_ROW.source.size_bytes,
    maxBytes: EXACT_ROW.source.size_bytes,
    unavailableCode: 'chat_parity_artifact_unavailable',
    mismatchCode: 'chat_parity_artifact_identity_mismatch',
  })
  await inspectRegularFileMetadata(options.binary, {
    lstatImpl: deps.lstatImpl || lstat,
    statImpl: deps.statImpl || stat,
    maxBytes: LOCAL_FILE_LIMITS.binary_bytes,
    unavailableCode: 'chat_parity_source_dirty',
    mismatchCode: 'chat_parity_source_dirty',
  })
  const ignored = deps.checkIgnoredImpl
    ? await deps.checkIgnoredImpl(options.root, options.artifact)
    : await gitPathIgnored(options.root, options.artifact, deps)
  expect(ignored === true, 'chat_parity_artifact_not_ignored')
  let provenance
  try {
    provenance = deps.inspectProvenanceImpl
      ? await deps.inspectProvenanceImpl(options)
      : await inspectProvenance(options, deps)
  } catch (error) {
    const bridged = bridgeSmolError(error)
    throw parityError(bridged || 'chat_parity_source_dirty')
  }
  expect(/^[0-9a-f]{40}$/.test(provenance?.runtime_head || '')
    && provenance?.tracked_files_clean === true && provenance?.untracked_files_excluded === true
    && provenance?.binary_profile === BINARY_PROFILE
    && /^[0-9a-f]{64}$/.test(provenance?.binary_sha256 || '')
    && provenance?.binary_version === `camelid ${provenance?.source_describe}`,
  'chat_parity_source_dirty')
  let autoSelectRoots
  try {
    autoSelectRoots = deps.assertAutoSelectRootsEmptyImpl
      ? await deps.assertAutoSelectRootsEmptyImpl(options)
      : await assertAutoSelectRootsEmpty(options, deps)
  } catch { throw parityError('chat_parity_auto_select_invalid') }
  const portFree = deps.assertPortFreeImpl || ((address) => assertPortFree(address, deps))
  await portFree({ host: '127.0.0.1', port: 8297 })
  await portFree({ host: '127.0.0.1', port: 8299 })
  const present = deps.engineProcessesPresentImpl
    ? await deps.engineProcessesPresentImpl()
    : await engineProcessesPresent(deps)
  expect(present === false, 'chat_parity_engine_present')
  const availablePhysicalBytes = deps.freePhysicalBytesImpl
    ? await deps.freePhysicalBytesImpl()
    : freemem()
  const availableDiskBytes = deps.diskFreeBytesImpl
    ? await deps.diskFreeBytesImpl(options.cwd)
    : await diskFreeBytes(options.cwd, deps)
  expect(availablePhysicalBytes >= LIMITS.preflight_physical_bytes
    && availableDiskBytes >= LIMITS.preflight_disk_bytes,
  'chat_parity_resources_low')
  const groundings = deps.inspectGroundingsImpl
    ? await deps.inspectGroundingsImpl(options.root)
    : await inspectGroundings(options.root, deps)
  const llama = deps.inspectLlamaPackageImpl
    ? await deps.inspectLlamaPackageImpl(options.llamaServer)
    : await inspectLlamaPackage(options.llamaServer, deps)
  return {
    platform: 'windows-x86_64',
    provenance,
    artifact: { size_bytes: fileStats.size, expected_sha256: EXACT_ROW.source.sha256,
      hash_recomputed: false, ignored: true, path_redacted: true },
    auto_select_roots: autoSelectRoots,
    ports_unbound: [CAMELID_ADDR, LLAMA_ADDR],
    preexisting_engine_processes_absent: true,
    available_physical_bytes: availablePhysicalBytes,
    available_disk_bytes: availableDiskBytes,
    groundings,
    llama,
  }
}

async function runPostflight(options, preflight, deps = {}) {
  deps = hardenWindowsChildDeps(deps)
  await inspectRegularFileMetadata(options.binary, {
    lstatImpl: deps.lstatImpl || lstat,
    statImpl: deps.statImpl || stat,
    maxBytes: LOCAL_FILE_LIMITS.binary_bytes,
    unavailableCode: 'chat_parity_source_changed',
    mismatchCode: 'chat_parity_source_changed',
  })
  let provenance
  try {
    provenance = deps.inspectProvenanceImpl
      ? await deps.inspectProvenanceImpl(options)
      : await inspectProvenance(options, deps)
  } catch { throw parityError('chat_parity_source_changed') }
  let roots
  try {
    roots = deps.assertAutoSelectRootsEmptyImpl
      ? await deps.assertAutoSelectRootsEmptyImpl(options)
      : await assertAutoSelectRootsEmpty(options, deps)
  } catch { throw parityError('chat_parity_source_changed') }
  const groundings = deps.inspectGroundingsImpl
    ? await deps.inspectGroundingsImpl(options.root)
    : await inspectGroundings(options.root, deps)
  const llama = deps.inspectLlamaPackageImpl
    ? await deps.inspectLlamaPackageImpl(options.llamaServer)
    : await inspectLlamaPackage(options.llamaServer, deps)
  expect(sameJson(provenance, preflight.provenance)
    && sameJson(roots, preflight.auto_select_roots)
    && sameJson(groundings, preflight.groundings)
    && sameJson(llama, preflight.llama),
  'chat_parity_source_changed')
  return { provenance, auto_select_roots: roots, groundings, llama }
}

function boundedTimeout(ms, value) {
  let timer
  const promise = new Promise((resolvePromise) => {
    timer = setTimeout(() => resolvePromise(value), ms)
    timer.unref?.()
  })
  return { promise, cancel: () => clearTimeout(timer) }
}

async function readResponseTextBounded(response, controller) {
  const declared = Number(response?.headers?.get?.('content-length'))
  if (Number.isFinite(declared) && declared > LIMITS.max_response_bytes) {
    const error = parityError('chat_parity_http_failed')
    controller.abort(error)
    try { await response?.body?.cancel?.(error) } catch { /* bounded failure */ }
    throw error
  }
  if (response?.body && typeof response.body.getReader === 'function') {
    const reader = response.body.getReader()
    const chunks = []
    let bytes = 0
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        expect(ArrayBuffer.isView(value), 'chat_parity_http_failed')
        bytes += value.byteLength
        if (bytes > LIMITS.max_response_bytes) {
          const error = parityError('chat_parity_http_failed')
          controller.abort(error)
          try { await reader.cancel(error) } catch { /* bounded failure */ }
          throw error
        }
        chunks.push(Buffer.from(value.buffer, value.byteOffset, value.byteLength))
      }
      return Buffer.concat(chunks, bytes).toString('utf8')
    } finally {
      try { reader.releaseLock?.() } catch { /* already cancelled */ }
    }
  }
  const text = await response.text()
  expect(Buffer.byteLength(text) <= LIMITS.max_response_bytes, 'chat_parity_http_failed')
  return text
}

async function httpJson({ origin, allowedEndpoints, method, endpoint, body, timeoutMs, signal,
  fetchImpl = fetch }) {
  expect(allowedEndpoints.has(endpoint), 'chat_parity_http_failed')
  if (signal?.aborted) throw signal.reason || parityError('chat_parity_http_failed')
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason || parityError('chat_parity_http_failed'))
  signal?.addEventListener('abort', abort, { once: true })
  if (signal?.aborted) abort()
  const timeout = setTimeout(() => controller.abort(parityError('chat_parity_http_failed')), timeoutMs)
  try {
    const response = await fetchImpl(`${origin}${endpoint}`, {
      method,
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await readResponseTextBounded(response, controller)
    let parsed
    try { parsed = JSON.parse(text) } catch { throw parityError('chat_parity_http_failed') }
    return { status: response.status, body: parsed }
  } catch (error) {
    if (signal?.aborted) {
      const bridged = bridgeSmolError(signal.reason)
      if (bridged) throw parityError(bridged)
      if (signal.reason instanceof SmolLM3ChatParityError) throw signal.reason
    }
    if (error instanceof SmolLM3ChatParityError) throw error
    throw parityError('chat_parity_http_failed')
  } finally {
    clearTimeout(timeout)
    signal?.removeEventListener('abort', abort)
  }
}

function normalizeHealth(body, { loaded, final = false, build }) {
  const code = 'chat_parity_camelid_contract_invalid'
  expect(body?.ok === true && body?.engine === 'camelid'
    && body?.version === CAMELID_RELEASE_VERSION && body?.build === build
    && body?.loaded_now === loaded && body?.generation_ready === loaded
    && body?.vision_ready === false && body?.active_model_id === (loaded ? ROW_ID : null)
    && body?.backend === (loaded ? 'llama' : 'none')
    && body?.model_family === (loaded ? 'llama-family' : null)
    && body?.engine_queue_depth === 0 && body?.engine_queued_tasks === 0
    && body?.engine_active_task_id === null && body?.engine_active_generated_tokens === 0
    && body?.continuous_batch_slots === 1 && body?.listen_addr === CAMELID_ADDR
    && body?.q8_runtime?.policy === 'forced_lazy_file_backed_q8'
    && body?.q8_runtime?.lazy_q8_linear === true
    && body?.q8_runtime?.retain_q8_blocks === false
    && body?.q8_runtime?.file_cache_bytes === 0,
  code)
  if (loaded) {
    const plan = body.execution_plan
    expect(plan?.profile === 'safe' && plan?.operating_system === 'windows'
      && plan?.architecture === 'x86_64' && plan?.model_family === 'smollm3'
      && plan?.quant_type === 'Q8_0' && plan?.exact_model_row === EXACT_ROW.source.file
      && plan?.support_level === 'unknown_or_unvalidated'
      && plan?.selected_backend === 'cpu_reference'
      && plan?.selected_q8_path === 'safe_dense_or_q8_cpu'
      && plan?.diagnostics_status === DIAGNOSTICS_STATUS
      && plan?.cuda_resident_active === false,
    code)
  } else expect(body?.execution_plan === null, code)
  if (final) expect(body?.engine_active_elapsed_seconds === 0 && body?.engine_stalled_seconds === 0,
    code)
  return {
    loaded_now: loaded,
    generation_ready: loaded,
    active_model_id: body.active_model_id,
    version: CAMELID_RELEASE_VERSION,
    build,
    backend: body.backend,
    selected_backend: loaded ? body.execution_plan.selected_backend : null,
    cuda_resident_active: loaded ? false : null,
    queue_idle: true,
    q8_policy: 'forced_lazy_file_backed_q8',
    listen_addr: CAMELID_ADDR,
  }
}

function normalizeGpu(body) {
  expect(body && typeof body.available === 'boolean' && body.enabled === false
    && body.run_count === 0 && typeof body.backend === 'string',
  'chat_parity_camelid_contract_invalid')
  return { available: body.available, enabled: false, backend_redacted: true,
    run_count: 0, device_redacted: true }
}

function normalizeLoad(body) {
  expect(body?.data?.id === ROW_ID && body.data.path === null
    && body.data.status?.value === 'loaded' && body.data.camelid?.generation_ready === true
    && body.data.camelid?.model_path_redacted === true
    && body.camelid?.model_path_redacted === true
    && body.camelid?.compatibility === 'partial_llama_server_models_load_local_path'
    && body.camelid?.scope === 'single_local_model_load_alias',
  'chat_parity_camelid_contract_invalid')
  return { id: ROW_ID, status: 'loaded', generation_ready: true,
    model_path_redacted: true, request_path_redacted: true }
}

function normalizeVerify(body) {
  expect(body?.model_id === ROW_ID && body?.gguf_sha256 === EXACT_ROW.source.sha256
    && body?.eligible === false && body?.profile_id === null && body?.report === null,
  'chat_parity_camelid_contract_invalid')
  return { model_id: ROW_ID, gguf_sha256: EXACT_ROW.source.sha256,
    eligible: false, profile_id: null, report: null }
}

function normalizeProps(body) {
  expect(body?.model_path === null && body?.model_id === ROW_ID
    && body?.camelid?.generation_ready === true && body?.camelid?.model_path_redacted === true
    && body?.modalities?.vision === false && body?.total_slots === 1
    && body?.default_generation_settings?.is_processing === false
    && body?.default_generation_settings?.next_token?.has_next_token === true
    && typeof body?.chat_template === 'string'
    && Buffer.byteLength(body.chat_template, 'utf8') === TEMPLATE_IDENTITY.utf8_bytes
    && sha256(Buffer.from(body.chat_template, 'utf8')) === TEMPLATE_IDENTITY.sha256
    && body?.chat_template_caps?.detected_format === 'smollm3_exact_default_thinking_text_qualified'
    && body?.chat_template_caps?.render_prompt_envelope?.content === 'text_only'
    && body?.chat_template_caps?.render_prompt_envelope?.add_generation_prompt === true,
  'chat_parity_camelid_contract_invalid')
  return { model_id: ROW_ID, generation_ready: true, template: structuredClone(TEMPLATE_IDENTITY),
    template_redacted: true, default_thinking_text_envelope: true }
}

function normalizeRenderedPrompt(body, expectedNormalizedPrompt) {
  expect(body && typeof body.prompt === 'string' && Buffer.byteLength(body.prompt, 'utf8') > 0,
    'chat_parity_camelid_contract_invalid')
  const occurrences = [...body.prompt.matchAll(DATE_PATTERN)]
  expect(occurrences.length === 1, 'chat_parity_camelid_contract_invalid')
  const normalized = body.prompt.replace(DATE_PATTERN, `Today Date: ${DATE_PLACEHOLDER}\n`)
  expect(normalized === expectedNormalizedPrompt
    && Buffer.byteLength(normalized, 'utf8') === NORMALIZED_PROMPT_UTF8_BYTES
    && sha256(Buffer.from(normalized, 'utf8')) === NORMALIZED_PROMPT_SHA256,
  'chat_parity_camelid_contract_invalid')
  return {
    evidence: {
      actual_prompt_utf8_bytes: Buffer.byteLength(body.prompt, 'utf8'),
      actual_prompt_sha256: sha256(Buffer.from(body.prompt, 'utf8')),
      normalized_prompt_utf8_bytes: NORMALIZED_PROMPT_UTF8_BYTES,
      normalized_prompt_sha256: NORMALIZED_PROMPT_SHA256,
      dynamic_date_occurrences: 1,
      dynamic_date_redacted: true,
      default_thinking: true,
      add_generation_prompt: true,
    },
    prompt: body.prompt,
  }
}

function normalizeTokenize(body, engine) {
  expect(body && Array.isArray(body.tokens) && body.tokens.length > 0
    && body.tokens.every(nonNegativeInteger),
  engine === 'camelid' ? 'chat_parity_camelid_contract_invalid' : 'chat_parity_llama_contract_invalid')
  return {
    evidence: {
      token_count: body.tokens.length,
      token_ids_sha256: tokenArraySha256(body.tokens),
      add_special: false,
      parse_special: true,
      prompt_text_redacted: true,
    },
    tokens: [...body.tokens],
  }
}

function normalizeCamelidChat(body) {
  const code = 'chat_parity_camelid_contract_invalid'
  expect(body?.model === ROW_ID && body?.lane === 'experimental'
    && Array.isArray(body?.choices) && body.choices.length === 1
    && body?.choices[0]?.finish_reason === 'length'
    && typeof body?.choices[0]?.message?.content === 'string'
    && body?.usage?.completion_tokens === LIMITS.generated_tokens
    && positiveInteger(body?.usage?.prompt_tokens)
    && body?.usage?.total_tokens === body.usage.prompt_tokens + LIMITS.generated_tokens
    && !Object.hasOwn(body, 'camelid_receipt'),
  code)
  const diagnostics = body.camelid
  expect(Array.isArray(diagnostics?.prompt_token_ids) && diagnostics.prompt_token_ids.length > 0
    && diagnostics.prompt_token_ids.every(nonNegativeInteger)
    && diagnostics.prompt_token_ids.length === body.usage.prompt_tokens
    && Array.isArray(diagnostics?.generated_token_ids)
    && diagnostics.generated_token_ids.length === LIMITS.generated_tokens
    && diagnostics.generated_token_ids.every(nonNegativeInteger)
    && diagnostics.generated_token_ids.every((id) => !EOG_TOKEN_IDS.includes(id))
    && diagnostics?.timings_ms?.weight_cache_hit === false
    && diagnostics?.timings_ms?.prompt_cache_hit === false
    && positiveInteger(diagnostics?.timings_ms?.weight_load)
    && diagnostics?.timings_ms?.prompt_evaluation?.first_token_evaluated === true,
  code)
  const firstTopLogits = diagnostics.top_logits
  expect(Array.isArray(firstTopLogits) && firstTopLogits.length > 0
    && firstTopLogits.every((entry) => nonNegativeInteger(entry?.token_id)
      && finiteNumber(entry?.logit) && finiteNumber(entry?.probability)
      && Number.isSafeInteger(entry?.rank) && entry.rank >= 1 && entry.selected === false)
    && firstTopLogits.filter((entry) => entry.rank === 1).length === 1
    && firstTopLogits.find((entry) => entry.rank === 1).token_id
      === diagnostics.generated_token_ids[0]
    && (diagnostics.step_top_logits === undefined
      || (Array.isArray(diagnostics.step_top_logits) && diagnostics.step_top_logits.length === 0)),
  code)
  const bubble = body.choices[0].message.content
  return {
    evidence: {
      prompt_token_ids: [...diagnostics.prompt_token_ids],
      generated_token_ids: [...diagnostics.generated_token_ids],
      bubble: { redacted: true, utf8_bytes: Buffer.byteLength(bubble, 'utf8'),
        sha256: sha256(Buffer.from(bubble, 'utf8')) },
      usage: { prompt_tokens: body.usage.prompt_tokens,
        completion_tokens: LIMITS.generated_tokens, total_tokens: body.usage.total_tokens },
      finish_reason: 'length',
      first_token_top_logit_verified: true,
      per_step_logit_diagnostics_absent_or_empty: true,
      first_forward: { weight_load_observed: true, weight_cache_hit: false,
        prompt_cache_hit: false, first_token_evaluated: true },
      generated_ids_non_eog: true,
      camelid_receipt_present: false,
    },
    promptTokens: [...diagnostics.prompt_token_ids],
    generatedTokens: [...diagnostics.generated_token_ids],
    bubble,
  }
}

function normalizeDetokenize(body, engine) {
  const code = engine === 'camelid'
    ? 'chat_parity_camelid_contract_invalid'
    : 'chat_parity_llama_contract_invalid'
  expect(body && typeof body.content === 'string', code)
  const evidence = { generated_token_count: LIMITS.generated_tokens,
    content: { redacted: true, utf8_bytes: Buffer.byteLength(body.content, 'utf8'),
      sha256: sha256(Buffer.from(body.content, 'utf8')) } }
  if (engine === 'camelid') evidence.trimmed_content = redactedText(body.content.trim())
  return { evidence, content: body.content }
}

function normalizeLlamaHealth(body) {
  expect(body?.status === 'ok', 'chat_parity_llama_contract_invalid')
  return { status: 'ok', model_loaded: true, address: LLAMA_ADDR }
}

function normalizeLlamaCompletion(body) {
  const code = 'chat_parity_llama_contract_invalid'
  expect(body && Array.isArray(body.tokens) && body.tokens.length === LIMITS.generated_tokens
    && body.tokens.every(nonNegativeInteger) && body.tokens.every((id) => !EOG_TOKEN_IDS.includes(id))
    && typeof body.content === 'string'
    && (body.tokens_predicted === undefined || body.tokens_predicted === LIMITS.generated_tokens),
  code)
  if (body.completion_probabilities !== undefined) {
    expect(Array.isArray(body.completion_probabilities)
      && body.completion_probabilities.length === LIMITS.generated_tokens
      && body.completion_probabilities.every((step, index) => {
        const probabilities = step?.probs
        return step?.id === body.tokens[index] && Array.isArray(probabilities)
          && probabilities.length > 0
          && probabilities.every((entry) => nonNegativeInteger(entry?.tok_str === undefined
            ? entry?.id : entry?.id) && finiteNumber(entry?.prob))
      }), code)
  }
  return {
    evidence: {
      generated_token_ids: [...body.tokens],
      content: { redacted: true, utf8_bytes: Buffer.byteLength(body.content, 'utf8'),
        sha256: sha256(Buffer.from(body.content, 'utf8')) },
      generated_token_count: LIMITS.generated_tokens,
      generated_ids_non_eog: true,
      deterministic_top_k_only: true,
      first_forward: true,
    },
    generatedTokens: [...body.tokens],
    content: body.content,
  }
}

function requestBodyForStep(name, context) {
  switch (name) {
    case 'camelid_load': return { path: context.artifact, id: ROW_ID }
    case 'camelid_apply_template_before':
    case 'camelid_apply_template_after': return structuredClone(APPLY_TEMPLATE_REQUEST)
    case 'camelid_tokenize_prompt': return { content: context.frozenPrompt,
      ...structuredClone(TOKENIZE_FLAGS) }
    case 'camelid_chat_first_forward': return structuredClone(CHAT_REQUEST)
    case 'camelid_detokenize_generated': return { tokens: [...context.camelidGeneratedTokens] }
    case 'llama_tokenize_prompt': return { content: context.frozenPrompt,
      ...structuredClone(TOKENIZE_FLAGS) }
    case 'llama_completion_first_forward': return { prompt: [...context.commonPromptTokens],
      ...structuredClone(LLAMA_COMPLETION_SETTINGS) }
    case 'llama_detokenize_generated': return { tokens: [...context.llamaGeneratedTokens] }
    default: return undefined
  }
}

function timeoutForStep(name) {
  if (name === 'camelid_load') return LIMITS.load_timeout_ms
  if (name.includes('chat_first_forward') || name.includes('completion_first_forward')) {
    return LIMITS.generation_timeout_ms
  }
  return LIMITS.ordinary_request_timeout_ms
}

async function raceRequest({ request, handle, artifactLock, guard }) {
  guard.throwIfAborted()
  artifactLock.assertHeld()
  const outcome = await Promise.race([
    Promise.resolve().then(request).then((response) => ({ response }), (error) => ({ requestError: error })),
    handle.exited.then((status) => ({ exited: status })),
    artifactLock.exited.then((status) => ({ lockExited: status })),
  ])
  if (outcome.lockExited) throw parityError('chat_parity_artifact_lock_lost')
  if (outcome.exited) throw parityError('chat_parity_process_exited')
  if (outcome.requestError) {
    if (artifactLock.isExited()) throw parityError('chat_parity_artifact_lock_lost')
    if (handle.isExited()) throw parityError('chat_parity_process_exited')
    throw outcome.requestError
  }
  guard.throwIfAborted()
  artifactLock.assertHeld()
  if (handle.isExited()) throw parityError('chat_parity_process_exited')
  return outcome.response
}

async function waitForHealth({ request, handle, artifactLock, guard, nowMs, sleepImpl }) {
  const deadline = nowMs() + LIMITS.startup_timeout_ms
  while (nowMs() < deadline) {
    try {
      const response = await raceRequest({ request, handle, artifactLock, guard })
      if (response?.status === 200) return response
    } catch (error) {
      if (error instanceof SmolLM3ChatParityError
        && error.code !== 'chat_parity_http_failed') throw error
    }
    await sleepImpl(100)
  }
  throw parityError('chat_parity_startup_timeout')
}

function assertHandle(handle) {
  expect(handle && positiveInteger(handle.pid) && handle.exited && handle.closed
    && typeof handle.kill === 'function' && typeof handle.isExited === 'function'
    && typeof handle.isClosed === 'function' && typeof handle.logMarkers === 'function',
  'chat_parity_process_start_failed')
  return handle
}

function makeGuard(handle, engine, deps) {
  if (deps.createResourceGuardImpl) return deps.createResourceGuardImpl(handle, engine)
  const limit = engine === 'camelid'
    ? LIMITS.camelid_child_working_set_abort_bytes
    : LIMITS.llama_child_working_set_abort_bytes
  return createResourceGuard(handle, {
    sampleImpl: deps.sampleResourceImpl
      ? (pid) => deps.sampleResourceImpl(pid, engine)
      : undefined,
    sleepImpl: deps.sleepImpl || sleep,
    limits: {
      monitor_interval_ms: LIMITS.monitor_interval_ms,
      low_memory_abort_bytes: LIMITS.low_memory_abort_bytes,
      child_working_set_abort_bytes: limit,
      consecutive_abort_samples: LIMITS.consecutive_abort_samples,
    },
  })
}

async function stopEngine({ engine, handle, guard, lifecycle, deps }) {
  let resourceSummary
  let resourceError = null
  let terminationError = null
  if (guard) {
    try {
      const stopped = await guard.stop()
      expect(stopped?.observed === true, 'chat_parity_resource_telemetry_unavailable')
      resourceSummary = guard.summary()
      try { guard.throwIfAborted() } catch (error) {
        throw parityError(bridgeSmolError(error) || 'chat_parity_resource_telemetry_unavailable')
      }
    } catch (error) {
      resourceError = error instanceof SmolLM3ChatParityError
        ? error : parityError('chat_parity_resource_telemetry_unavailable')
    }
  } else {
    resourceError = parityError('chat_parity_resource_telemetry_unavailable')
  }
  // Resource-monitor setup or shutdown failure never skips child cleanup.
  // The exact handle is still terminated and its stdio `close` boundary must
  // be observed before the artifact lock can be released.
  try {
    const wasExited = handle.isExited()
    const terminated = deps.terminateChildImpl
      ? await deps.terminateChildImpl(handle, engine)
      : await terminateSpawnedChild(handle, { sleepImpl: deps.sleepImpl || sleep })
    expect(terminated?.observed === true && handle.isExited() === true
      && handle.isClosed() === true,
      'chat_parity_termination_failed')
    if (wasExited || terminated?.already_exited === true
      || terminated?.termination_requested !== true) {
      terminationError = parityError('chat_parity_process_exited')
    }
    const markers = handle.logMarkers()
    expect(markers?.warming_up_seen === false
      && markers?.generation_warmup_complete_seen === false
      && markers?.raw_output_persisted === false,
    'chat_parity_warmup_detected')
    lifecycle.activeEngine = null
    lifecycle.events.push(`${engine}_closed`)
    lifecycle.markers[engine] = markers
    const portFree = deps.assertPortFreeImpl || ((address) => assertPortFree(address, deps))
    await portFree(engine === 'camelid'
      ? { host: '127.0.0.1', port: 8297 }
      : { host: '127.0.0.1', port: 8299 })
    lifecycle.portsFreeAfterStop[engine] = true
  } catch (error) {
    const bridged = bridgeSmolError(error)
    terminationError = error instanceof SmolLM3ChatParityError
      ? error : parityError(bridged || 'chat_parity_termination_failed')
  }
  if (terminationError) throw terminationError
  if (resourceError) throw resourceError
  expect(resourceSummary?.samples > 0
    && nonNegativeInteger(resourceSummary?.minimum_available_physical_bytes)
    && positiveInteger(resourceSummary?.peak_child_working_set_bytes)
    && resourceSummary?.thresholds_tripped === false,
  'chat_parity_resource_telemetry_unavailable')
  return resourceSummary
}

async function startEngine({ engine, binary, args, cwd, env, lifecycle, deps }) {
  expect(lifecycle.activeEngine === null, 'chat_parity_overlap_detected')
  let handle
  try {
    handle = await (deps.startProcessImpl
      ? deps.startProcessImpl({ engine, binary, args, cwd, env })
      : startCamelidProcess({ binary, args, cwd, env }, deps))
  } catch { throw parityError('chat_parity_process_start_failed') }
  try { assertHandle(handle) } catch {
    if (!handle || typeof handle.kill !== 'function' || !handle.exited || !handle.closed
      || typeof handle.isExited !== 'function' || typeof handle.isClosed !== 'function') {
      throw parityError('chat_parity_termination_failed')
    }
    try {
      const terminated = deps.terminateChildImpl
        ? await deps.terminateChildImpl(handle, engine)
        : await terminateSpawnedChild(handle, { sleepImpl: deps.sleepImpl || sleep })
      expect(terminated?.observed === true && handle.isExited() === true
        && handle.isClosed() === true,
        'chat_parity_termination_failed')
    } catch { throw parityError('chat_parity_termination_failed') }
    throw parityError('chat_parity_process_start_failed')
  }
  lifecycle.activeEngine = engine
  lifecycle.maxConcurrentEngines = Math.max(lifecycle.maxConcurrentEngines, 1)
  lifecycle.events.push(`${engine}_started`)
  return handle
}

async function runCamelidPhase({ options, preflight, artifactLock, lifecycle, steps, deps }) {
  const env = buildChildEnv(deps.inheritedEnv || process.env)
  expect(sameJson(childEnvSubset(env, 'CAMELID_'), SAFE_CAMELID_ENV),
    'chat_parity_options_invalid')
  const environment = describeChildEnvironment(env, SAFE_CAMELID_ENV)
  const args = buildCamelidServeArgs(options.modelsDir)
  expect(!args.includes('--model'), 'chat_parity_options_invalid')
  const handle = await startEngine({ engine: 'camelid', binary: options.binary, args,
    cwd: options.cwd, env, lifecycle, deps })
  let guard
  let primaryError
  let resources
  const context = { artifact: options.artifact }
  const nowMs = deps.nowMsImpl || Date.now
  const sleepImpl = deps.sleepImpl || sleep
  const requestImpl = deps.httpJsonImpl || ((request) => httpJson({ ...request,
    fetchImpl: deps.fetchImpl }))
  const allowedEndpoints = new Set(STEP_CONTRACT.filter(([, engine]) => engine === 'camelid')
    .map(([, , , endpoint]) => endpoint))
  try {
    guard = await makeGuard(handle, 'camelid', deps)
    expect(guard?.signal && typeof guard.throwIfAborted === 'function'
      && typeof guard.stop === 'function' && typeof guard.summary === 'function',
    'chat_parity_resource_telemetry_unavailable')
    for (let index = 0; index < 13; index += 1) {
      const [name, engine, method, endpoint] = STEP_CONTRACT[index]
      const body = requestBodyForStep(name, context)
      const started = nowMs()
      const request = () => requestImpl({ origin: CAMELID_ORIGIN, allowedEndpoints,
        method, endpoint, body, timeoutMs: index === 0 ? 2_000 : timeoutForStep(name),
        signal: guard.signal })
      const response = index === 0
        ? await waitForHealth({ request, handle, artifactLock, guard, nowMs, sleepImpl })
        : await raceRequest({ request, handle, artifactLock, guard })
      expect(response?.status === 200 && response.body && typeof response.body === 'object',
        'chat_parity_http_failed')
      let normalized
      switch (name) {
        case 'camelid_baseline_health':
          normalized = { evidence: normalizeHealth(response.body,
            { loaded: false, build: preflight.provenance.source_describe }) }
          break
        case 'camelid_baseline_gpu':
        case 'camelid_final_gpu': normalized = { evidence: normalizeGpu(response.body) }; break
        case 'camelid_load': normalized = { evidence: normalizeLoad(response.body) }; break
        case 'camelid_verify_identity': normalized = { evidence: normalizeVerify(response.body) }; break
        case 'camelid_loaded_health':
          normalized = { evidence: normalizeHealth(response.body,
            { loaded: true, build: preflight.provenance.source_describe }) }
          break
        case 'camelid_final_health':
          normalized = { evidence: normalizeHealth(response.body,
            { loaded: true, final: true, build: preflight.provenance.source_describe }) }
          break
        case 'camelid_props': normalized = { evidence: normalizeProps(response.body) }; break
        case 'camelid_apply_template_before': {
          normalized = normalizeRenderedPrompt(response.body,
            preflight.groundings.expected_normalized_prompt)
          context.frozenPrompt = normalized.prompt
          break
        }
        case 'camelid_tokenize_prompt': {
          normalized = normalizeTokenize(response.body, 'camelid')
          context.commonPromptTokens = normalized.tokens
          break
        }
        case 'camelid_chat_first_forward': {
          normalized = normalizeCamelidChat(response.body)
          expect(sameJson(normalized.promptTokens, context.commonPromptTokens),
            'chat_parity_camelid_contract_invalid')
          context.camelidGeneratedTokens = normalized.generatedTokens
          context.camelidBubble = normalized.bubble
          break
        }
        case 'camelid_detokenize_generated': {
          normalized = normalizeDetokenize(response.body, 'camelid')
          expect(context.camelidBubble === normalized.content.trim(),
            'chat_parity_camelid_contract_invalid')
          normalized.evidence.bubble_equals_detokenized_trim = true
          context.camelidDetokenized = normalized.content
          break
        }
        case 'camelid_apply_template_after': {
          normalized = normalizeRenderedPrompt(response.body,
            preflight.groundings.expected_normalized_prompt)
          expect(normalized.prompt === context.frozenPrompt,
            'chat_parity_camelid_contract_invalid')
          normalized.evidence.identical_to_frozen_pre_forward_prompt = true
          break
        }
        default: throw parityError('chat_parity_camelid_contract_invalid')
      }
      steps.push({ ordinal: index + 1, name, engine, method, endpoint,
        http_status: 200, elapsed_ms: Math.max(0, Math.round(nowMs() - started)),
        evidence: normalized.evidence })
      artifactLock.assertHeld()
    }
    const markers = handle.logMarkers()
    expect(markers?.warming_up_seen === false
      && markers?.generation_warmup_complete_seen === false,
    'chat_parity_warmup_detected')
  } catch (error) {
    try { guard?.throwIfAborted() } catch (guardError) {
      error = parityError(bridgeSmolError(guardError) || 'chat_parity_resource_telemetry_unavailable')
    }
    primaryError = error instanceof SmolLM3ChatParityError
      ? error : parityError(bridgeSmolError(error) || 'chat_parity_http_failed')
  } finally {
    try { resources = await stopEngine({ engine: 'camelid', handle, guard, lifecycle, deps }) }
    catch (error) { primaryError = error }
  }
  if (primaryError) throw primaryError
  return { context, resources, environment }
}

async function runLlamaPhase({ options, camelid, artifactLock, lifecycle, steps, deps }) {
  expect(lifecycle.activeEngine === null
    && lifecycle.events.at(-1) === 'camelid_closed'
    && lifecycle.portsFreeAfterStop.camelid === true,
  'chat_parity_overlap_detected')
  const portFree = deps.assertPortFreeImpl || ((address) => assertPortFree(address, deps))
  await portFree({ host: '127.0.0.1', port: 8297 })
  await portFree({ host: '127.0.0.1', port: 8299 })
  const env = buildLlamaEnv(deps.inheritedEnv || process.env)
  expect(sameJson(childEnvSubset(env, 'CUDA_'), SAFE_LLAMA_ENV),
    'chat_parity_options_invalid')
  const environment = describeChildEnvironment(env, SAFE_LLAMA_ENV)
  const args = buildLlamaServeArgs(options.artifact)
  const handle = await startEngine({ engine: 'llama_cpp', binary: options.llamaServer, args,
    cwd: dirname(options.llamaServer), env, lifecycle, deps })
  let guard
  let primaryError
  let resources
  const context = {
    artifact: options.artifact,
    frozenPrompt: camelid.context.frozenPrompt,
    commonPromptTokens: camelid.context.commonPromptTokens,
  }
  const nowMs = deps.nowMsImpl || Date.now
  const sleepImpl = deps.sleepImpl || sleep
  const requestImpl = deps.httpJsonImpl || ((request) => httpJson({ ...request,
    fetchImpl: deps.fetchImpl }))
  const allowedEndpoints = new Set(STEP_CONTRACT.filter(([, engine]) => engine === 'llama_cpp')
    .map(([, , , endpoint]) => endpoint))
  try {
    guard = await makeGuard(handle, 'llama_cpp', deps)
    expect(guard?.signal && typeof guard.throwIfAborted === 'function'
      && typeof guard.stop === 'function' && typeof guard.summary === 'function',
    'chat_parity_resource_telemetry_unavailable')
    for (let index = 13; index < STEP_CONTRACT.length; index += 1) {
      const [name, engine, method, endpoint] = STEP_CONTRACT[index]
      const body = requestBodyForStep(name, context)
      const started = nowMs()
      const request = () => requestImpl({ origin: LLAMA_ORIGIN, allowedEndpoints,
        method, endpoint, body, timeoutMs: index === 13 ? 2_000 : timeoutForStep(name),
        signal: guard.signal })
      const response = index === 13
        ? await waitForHealth({ request, handle, artifactLock, guard, nowMs, sleepImpl })
        : await raceRequest({ request, handle, artifactLock, guard })
      expect(response?.status === 200 && response.body && typeof response.body === 'object',
        'chat_parity_http_failed')
      let normalized
      switch (name) {
        case 'llama_health':
        case 'llama_final_health': normalized = { evidence: normalizeLlamaHealth(response.body) }; break
        case 'llama_tokenize_prompt': {
          normalized = normalizeTokenize(response.body, 'llama_cpp')
          expect(sameJson(normalized.tokens, context.commonPromptTokens),
            'chat_parity_llama_contract_invalid')
          normalized.evidence.identical_to_camelid_prompt_token_ids = true
          break
        }
        case 'llama_completion_first_forward': {
          normalized = normalizeLlamaCompletion(response.body)
          context.llamaGeneratedTokens = normalized.generatedTokens
          context.llamaContent = normalized.content
          break
        }
        case 'llama_detokenize_generated': {
          normalized = normalizeDetokenize(response.body, 'llama_cpp')
          expect(context.llamaContent === normalized.content,
            'chat_parity_llama_contract_invalid')
          normalized.evidence.completion_content_equals_detokenized = true
          context.llamaDetokenized = normalized.content
          break
        }
        default: throw parityError('chat_parity_llama_contract_invalid')
      }
      steps.push({ ordinal: index + 1, name, engine, method, endpoint,
        http_status: 200, elapsed_ms: Math.max(0, Math.round(nowMs() - started)),
        evidence: normalized.evidence })
      artifactLock.assertHeld()
    }
    const markers = handle.logMarkers()
    expect(markers?.warming_up_seen === false
      && markers?.generation_warmup_complete_seen === false,
    'chat_parity_warmup_detected')
  } catch (error) {
    try { guard?.throwIfAborted() } catch (guardError) {
      error = parityError(bridgeSmolError(guardError) || 'chat_parity_resource_telemetry_unavailable')
    }
    primaryError = error instanceof SmolLM3ChatParityError
      ? error : parityError(bridgeSmolError(error) || 'chat_parity_http_failed')
  } finally {
    try { resources = await stopEngine({ engine: 'llama_cpp', handle, guard, lifecycle, deps }) }
    catch (error) { primaryError = error }
  }
  if (primaryError) throw primaryError
  return { context, resources, environment }
}

function redactedText(text) {
  return { redacted: true, utf8_bytes: Buffer.byteLength(text, 'utf8'),
    sha256: sha256(Buffer.from(text, 'utf8')) }
}

function buildReceipt({ preflight, artifact, lifecycle, steps, camelid, llama, createdUtc }) {
  const tokenMatch = sameJson(camelid.context.camelidGeneratedTokens,
    llama.context.llamaGeneratedTokens)
  const detokenizedMatch = camelid.context.camelidDetokenized === llama.context.llamaDetokenized
  const exactMatch = tokenMatch && detokenizedMatch
  const firstDivergentTokenIndex = tokenMatch ? -1
    : camelid.context.camelidGeneratedTokens.findIndex(
      (token, index) => token !== llama.context.llamaGeneratedTokens[index])
  const body = {
    schema: RECEIPT_SCHEMA,
    created_utc: createdUtc,
    gate: 'parity_preparation',
    row: structuredClone(EXACT_ROW),
    grounding: {
      files: structuredClone(GROUNDING_FILES),
      renderer_git_blob_sha1: RENDERER_GIT_BLOB_SHA1,
      template: structuredClone(TEMPLATE_IDENTITY),
      shape_case: structuredClone(preflight.groundings.shape_case),
      runtime_envelope: 'smollm3-default-thinking-runtime-envelope-v1',
    },
    provenance: {
      runtime_head: preflight.provenance.runtime_head,
      source_describe: preflight.provenance.source_describe,
      tracked_files_clean: true,
      untracked_files_excluded: true,
      camelid_binary: {
        profile: BINARY_PROFILE,
        sha256: preflight.provenance.binary_sha256,
        version: preflight.provenance.binary_version,
        health_build: preflight.provenance.source_describe,
        path_redacted: true,
      },
      llama_cpp: preflight.llama,
      artifact,
      platform: 'windows-x86_64',
      paths_redacted: true,
      hostname_redacted: true,
    },
    isolation: {
      addresses: { camelid: CAMELID_ADDR, llama_cpp: LLAMA_ADDR },
      engine_order: ['camelid', 'llama_cpp'],
      max_concurrent_engine_children: lifecycle.maxConcurrentEngines,
      lifecycle_events: [...lifecycle.events],
      preexisting_engine_processes_absent: true,
      both_ports_unbound_before_start: true,
      ports_unbound_after_each_stop: { ...lifecycle.portsFreeAfterStop },
      camelid_no_startup_model: true,
      camelid_chat_first_and_only_forward: true,
      llama_completion_first_and_only_forward: true,
      camelid_closed_and_stdio_drained_before_llama_start: true,
      llama_closed_and_stdio_drained_before_posthash: true,
      child_handle_only_termination: true,
      termination_observed: { camelid: true, llama_cpp: true },
      startup_warmup_markers: {
        camelid: { warming_up_seen: false, generation_warmup_complete_seen: false,
          raw_output_persisted: false },
        llama_cpp: { warming_up_seen: false, generation_warmup_complete_seen: false,
          raw_output_persisted: false },
      },
    },
    runtime_contract: {
      camelid: {
        command: receiptCamelidCommand(),
        environment: structuredClone(camelid.environment),
        request: { ...structuredClone(CHAT_REQUEST), camelid_enable_thinking_omitted: true,
          camelid_receipt_requested: false },
      },
      llama_cpp: {
        command: receiptLlamaCommand(),
        environment: structuredClone(llama.environment),
        tokenize: { content: 'exact_frozen_camelid_rendered_prompt',
          ...structuredClone(TOKENIZE_FLAGS) },
        completion: { prompt: 'identical_cross_engine_prompt_token_ids',
          ...structuredClone(LLAMA_COMPLETION_SETTINGS) },
      },
      prompt: {
        rendered_text_redacted: true,
        actual_utf8_bytes: Buffer.byteLength(camelid.context.frozenPrompt, 'utf8'),
        actual_sha256: sha256(Buffer.from(camelid.context.frozenPrompt, 'utf8')),
        normalized_utf8_bytes: NORMALIZED_PROMPT_UTF8_BYTES,
        normalized_sha256: NORMALIZED_PROMPT_SHA256,
        token_count: camelid.context.commonPromptTokens.length,
        token_ids_sha256: tokenArraySha256(camelid.context.commonPromptTokens),
        token_ids_persisted_in_camelid_step: true,
      },
      limits: structuredClone(LIMITS),
    },
    steps,
    comparison: {
      prompt_render_stable_across_camelid_forward: true,
      prompt_token_ids_cross_engine_exact: true,
      generated_token_ids: {
        camelid: [...camelid.context.camelidGeneratedTokens],
        llama_cpp: [...llama.context.llamaGeneratedTokens],
        exact_match: tokenMatch,
        first_divergent_index: firstDivergentTokenIndex,
        all_four_non_eog: camelid.context.camelidGeneratedTokens.every(
          (id) => !EOG_TOKEN_IDS.includes(id))
          && llama.context.llamaGeneratedTokens.every((id) => !EOG_TOKEN_IDS.includes(id)),
      },
      canonical_detokenized_text: {
        camelid: redactedText(camelid.context.camelidDetokenized),
        llama_cpp: redactedText(llama.context.llamaDetokenized),
        exact_utf8_match: detokenizedMatch,
        camelid_bubble_equals_detokenized_trim: true,
        llama_content_equals_detokenized: true,
      },
      exact_token_and_text_match: exactMatch,
    },
    resource_observations: {
      preflight_available_physical_bytes: preflight.available_physical_bytes,
      preflight_available_disk_bytes: preflight.available_disk_bytes,
      camelid: camelid.resources,
      llama_cpp: llama.resources,
      thresholds_tripped: false,
    },
    gate_decision: {
      parity_preparation: exactMatch ? 'pass' : 'mismatch',
      bounded_evidence_publishable: exactMatch,
      roster_parity_gate: 'blocked_unchanged',
      template_gate: 'blocked_unchanged',
      api_webui_gate: 'pending_unchanged',
      context_gate: 'pending_unchanged',
      load_smoke_gate: 'blocked_unchanged',
      support_claim: false,
      disposition: 'hold',
      target_tier: 'experimental_exact_row',
      authorized_roster_scope: [],
      other_gates_unchanged: true,
    },
    does_not_prove: [...DOES_NOT_PROVE],
  }
  return sealReceipt(body)
}

function exactKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && sameJson(Object.keys(value).sort(), [...keys].sort())
}

// Reject hostile object graphs before any receipt property is read. Node's
// native Proxy detector does not execute traps, preserving validator totality.
function structuralErrors(value) {
  const errors = []
  const seen = new WeakSet()
  const stack = [{ node: value, path: '$', depth: 0 }]
  let visited = 0
  while (stack.length) {
    const { node, path, depth } = stack.pop()
    if (++visited > 50_000) { errors.push('structural scan exceeded its bounded node budget'); break }
    if (node === null || ['string', 'number', 'boolean'].includes(typeof node)) continue
    if (typeof node !== 'object') { errors.push(`${path} contains a non-JSON value`); continue }
    if (utilTypes.isProxy(node)) { errors.push(`${path} is a Proxy and cannot be inspected safely`); continue }
    if (depth >= 128) { errors.push(`${path} exceeds the bounded structural depth`); continue }
    if (seen.has(node)) { errors.push(`${path} contains a cycle or repeated object reference`); continue }
    seen.add(node)
    let descriptors
    let symbols
    let prototype
    try {
      descriptors = Object.getOwnPropertyDescriptors(node)
      symbols = Object.getOwnPropertySymbols(node)
      prototype = Object.getPrototypeOf(node)
    } catch { errors.push(`${path} could not be inspected safely`); continue }
    if (symbols.length) errors.push(`${path} contains symbol-keyed data`)
    const array = Array.isArray(node)
    if (array ? prototype !== Array.prototype : prototype !== Object.prototype && prototype !== null) {
      errors.push(`${path} has an unexpected prototype`)
    }
    const length = array ? descriptors.length?.value : null
    if (array) {
      const keys = Object.keys(descriptors).filter((key) => key !== 'length')
      const indexes = keys.filter((key) => /^(?:0|[1-9][0-9]*)$/.test(key) && Number(key) < length)
      if (!nonNegativeInteger(length) || length > 50_000 || keys.length !== indexes.length
        || indexes.length !== length) errors.push(`${path} contains a sparse or invalid array`)
    }
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (array && key === 'length') continue
      if (!Object.hasOwn(descriptor, 'value')) { errors.push(`${path}.${key} uses an accessor`); continue }
      if (!descriptor.enumerable) errors.push(`${path}.${key} is non-enumerable`)
      stack.push({ node: descriptor.value, path: `${path}.${key}`, depth: depth + 1 })
    }
  }
  return [...new Set(errors)]
}

function privacyErrors(value) {
  const errors = []
  const bannedKeys = new Set(['hostname', 'pid', 'process_id', 'artifact_path', 'binary_path',
    'executable_path', 'raw_log', 'raw_logs', 'authorization', 'cookie', 'password', 'secret',
    'token'])
  const allowedRoutes = new Set(STEP_CONTRACT.map(([, , , endpoint]) => endpoint))
  const allowedAbsolute = new Set([CAMELID_ADDR, LLAMA_ADDR])
  const seen = new WeakSet()
  const stack = [{ node: value, path: '$', depth: 0 }]
  let visited = 0
  while (stack.length) {
    const { node, path, depth } = stack.pop()
    if (++visited > 50_000) { errors.push('privacy scan exceeded its bounded node budget'); break }
    if (typeof node === 'string') {
      if (/[A-Za-z]:[\\/]/.test(node) || /\\\\[^\\]/.test(node) || /\bfile:\/\//i.test(node)) {
        errors.push(`${path} contains an absolute local path`)
      }
      if (/\bhf_[A-Za-z0-9]{8,}\b/.test(node)
        || /\b(?:gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})\b/.test(node)
        || /\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+/i.test(node)
        || /(?:^|[?&#;,\s])(?:access[_-]?token|auth[_-]?token|authorization|client[_-]?secret|credential|api[_-]?key|password|private[_-]?key|refresh[_-]?token|secret|token)\s*[:=]\s*[^\s&#;,]+/i.test(node)) {
        errors.push(`${path} contains credential-like data`)
      }
      if (node.startsWith('/') && !allowedRoutes.has(node)) errors.push(`${path} contains an unexpected absolute path`)
      if (/^(?:https?:\/\/|[^\s]+\.(?:local|lan|internal))$/i.test(node)
        && !allowedAbsolute.has(node)) errors.push(`${path} contains a hostname or URL`)
      if (node.length > 4_096) errors.push(`${path} contains oversized raw text`)
      continue
    }
    if (typeof node === 'number') { if (!Number.isFinite(node)) errors.push(`${path} contains a non-finite number`); continue }
    if (node === null || typeof node === 'boolean') continue
    if (typeof node !== 'object' || depth >= 128 || seen.has(node)) {
      errors.push(`${path} could not be inspected as bounded JSON`); continue
    }
    seen.add(node)
    const descriptors = Object.getOwnPropertyDescriptors(node)
    for (const [key, descriptor] of Object.entries(descriptors)) {
      if (Array.isArray(node) && key === 'length') continue
      if (!Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) continue
      if (bannedKeys.has(key.toLowerCase())) errors.push(`${path}.${key} uses a forbidden key`)
      stack.push({ node: descriptor.value, path: `${path}.${key}`, depth: depth + 1 })
    }
  }
  return errors
}

function validateReceiptUnsafe(receipt) {
  const unsafe = structuralErrors(receipt)
  if (unsafe.length) return unsafe
  const errors = []
  const check = (condition, message) => { if (!condition) errors.push(message) }
  const close = (value, keys, path) => check(exactKeys(value, keys), `${path} keys must be exact`)
  close(receipt, ['schema', 'receipt_id', 'created_utc', 'gate', 'row', 'grounding', 'provenance',
    'isolation', 'runtime_contract', 'steps', 'comparison', 'resource_observations',
    'gate_decision', 'does_not_prove'], 'receipt')
  check(receipt?.schema === RECEIPT_SCHEMA && receipt?.gate === 'parity_preparation'
    && sameJson(receipt?.row, EXACT_ROW), 'schema, gate, and row must remain exact')
  check(/^[0-9a-f]{64}$/.test(receipt?.receipt_id || ''), 'receipt_id must be lowercase SHA-256')
  if (receipt && typeof receipt === 'object') {
    const { receipt_id: _id, ...body } = receipt
    check(receipt.receipt_id === sha256(Buffer.from(canonicalJson(body), 'utf8')),
      'receipt_id must seal the canonical body')
  }
  check(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(receipt?.created_utc || '')
    && new Date(receipt.created_utc).toISOString() === receipt.created_utc,
  'created_utc must be canonical UTC')
  close(receipt?.grounding, ['files', 'renderer_git_blob_sha1', 'template', 'shape_case',
    'runtime_envelope'], 'grounding')
  check(sameJson(receipt?.grounding?.files, GROUNDING_FILES)
    && receipt?.grounding?.renderer_git_blob_sha1 === RENDERER_GIT_BLOB_SHA1
    && sameJson(receipt?.grounding?.template, TEMPLATE_IDENTITY)
    && sameJson(receipt?.grounding?.shape_case, {
      id: SHAPE_CASE_ID,
      normalized_prompt_utf8_bytes: NORMALIZED_PROMPT_UTF8_BYTES,
      normalized_prompt_sha256: NORMALIZED_PROMPT_SHA256,
      date_placeholder: DATE_PLACEHOLDER,
    })
    && receipt?.grounding?.runtime_envelope === 'smollm3-default-thinking-runtime-envelope-v1',
  'grounding identities must remain exact')
  close(receipt?.provenance, ['runtime_head', 'source_describe', 'tracked_files_clean',
    'untracked_files_excluded', 'camelid_binary', 'llama_cpp', 'artifact', 'platform',
    'paths_redacted', 'hostname_redacted'], 'provenance')
  close(receipt?.provenance?.camelid_binary, ['profile', 'sha256', 'version', 'health_build',
    'path_redacted'], 'provenance.camelid_binary')
  const sourceMatch = /^(?:([0-9a-f]{7,40})|.*-g([0-9a-f]{7,40}))$/i
    .exec(receipt?.provenance?.source_describe || '')
  const sourceAbbreviation = sourceMatch?.[1] || sourceMatch?.[2] || ''
  check(/^[0-9a-f]{40}$/.test(receipt?.provenance?.runtime_head || '')
    && typeof receipt?.provenance?.source_describe === 'string'
    && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(receipt.provenance.source_describe)
    && !/-dirty/i.test(receipt.provenance.source_describe)
    && sourceAbbreviation.length >= 7
    && receipt.provenance.runtime_head.startsWith(sourceAbbreviation.toLowerCase())
    && receipt?.provenance?.tracked_files_clean === true
    && receipt?.provenance?.untracked_files_excluded === true
    && receipt?.provenance?.camelid_binary?.profile === BINARY_PROFILE
    && /^[0-9a-f]{64}$/.test(receipt?.provenance?.camelid_binary?.sha256 || '')
    && receipt?.provenance?.camelid_binary?.version === `camelid ${receipt?.provenance?.source_describe}`
    && receipt?.provenance?.camelid_binary?.health_build === receipt?.provenance?.source_describe
    && receipt?.provenance?.camelid_binary?.path_redacted === true
    && sameJson(receipt?.provenance?.llama_cpp, { ...structuredClone(LLAMA_PIN),
      version_verified: true, executable_path_redacted: true, package_path_redacted: true })
    && receipt?.provenance?.platform === 'windows-x86_64'
    && receipt?.provenance?.paths_redacted === true && receipt?.provenance?.hostname_redacted === true,
  'engine provenance must remain exact and private')
  const artifact = receipt?.provenance?.artifact
  close(artifact, ['size_bytes', 'sha256', 'verified_after_lock_acquisition',
    'verified_after_both_engines', 'mutation_guard', 'path_redacted'], 'provenance.artifact')
  close(artifact?.mutation_guard, ['mechanism', 'read_access', 'write_access', 'delete_access',
    'rename_access', 'symbolic_links_rejected', 'acquired_before_prehash',
    'held_through_camelid', 'held_through_llama_cpp', 'held_through_posthash',
    'released_token_observed', 'helper_exit_code', 'artifact_path_in_helper_argv'],
  'provenance.artifact.mutation_guard')
  check(artifact?.size_bytes === EXACT_ROW.source.size_bytes && artifact?.sha256 === EXACT_ROW.source.sha256
    && artifact?.verified_after_lock_acquisition === true
    && artifact?.verified_after_both_engines === true && artifact?.path_redacted === true
    && sameJson(artifact?.mutation_guard, {
      mechanism: 'windows_file_stream_share_read', read_access: 'allowed', write_access: 'denied',
      delete_access: 'denied', rename_access: 'denied', symbolic_links_rejected: true,
      acquired_before_prehash: true, held_through_camelid: true, held_through_llama_cpp: true,
      held_through_posthash: true, released_token_observed: true, helper_exit_code: 0,
      artifact_path_in_helper_argv: false,
    }), 'artifact lock and hashes must remain exact')
  close(receipt?.isolation, ['addresses', 'engine_order', 'max_concurrent_engine_children',
    'lifecycle_events', 'preexisting_engine_processes_absent', 'both_ports_unbound_before_start',
    'ports_unbound_after_each_stop', 'camelid_no_startup_model',
    'camelid_chat_first_and_only_forward', 'llama_completion_first_and_only_forward',
    'camelid_closed_and_stdio_drained_before_llama_start',
    'llama_closed_and_stdio_drained_before_posthash', 'child_handle_only_termination',
    'termination_observed', 'startup_warmup_markers'], 'isolation')
  close(receipt?.isolation?.startup_warmup_markers, ['camelid', 'llama_cpp'],
    'isolation.startup_warmup_markers')
  check(sameJson(receipt?.isolation?.addresses, { camelid: CAMELID_ADDR, llama_cpp: LLAMA_ADDR })
    && sameJson(receipt?.isolation?.engine_order, ['camelid', 'llama_cpp'])
    && receipt?.isolation?.max_concurrent_engine_children === 1
    && sameJson(receipt?.isolation?.lifecycle_events,
      ['camelid_started', 'camelid_closed', 'llama_cpp_started', 'llama_cpp_closed'])
    && receipt?.isolation?.preexisting_engine_processes_absent === true
    && receipt?.isolation?.both_ports_unbound_before_start === true
    && sameJson(receipt?.isolation?.ports_unbound_after_each_stop,
      { camelid: true, llama_cpp: true })
    && receipt?.isolation?.camelid_no_startup_model === true
    && receipt?.isolation?.camelid_chat_first_and_only_forward === true
    && receipt?.isolation?.llama_completion_first_and_only_forward === true
    && receipt?.isolation?.camelid_closed_and_stdio_drained_before_llama_start === true
    && receipt?.isolation?.llama_closed_and_stdio_drained_before_posthash === true
    && receipt?.isolation?.child_handle_only_termination === true
    && sameJson(receipt?.isolation?.termination_observed, { camelid: true, llama_cpp: true })
    && ['camelid', 'llama_cpp'].every((engine) => sameJson(
      receipt?.isolation?.startup_warmup_markers?.[engine], {
        warming_up_seen: false, generation_warmup_complete_seen: false,
        raw_output_persisted: false,
      })), 'engine lifetimes and first-forward isolation must remain exact')
  close(receipt?.runtime_contract, ['camelid', 'llama_cpp', 'prompt', 'limits'], 'runtime_contract')
  close(receipt?.runtime_contract?.camelid, ['command', 'environment', 'request'],
    'runtime_contract.camelid')
  close(receipt?.runtime_contract?.llama_cpp, ['command', 'environment', 'tokenize', 'completion'],
    'runtime_contract.llama_cpp')
  const validateEnvironment = (environment, modelOverrides, path) => {
    close(environment, ['schema', 'model_overrides', 'inherited_os_allowlist',
      'inherited_os_keys_present', 'inherited_os_values_redacted',
      'inherited_os_environment_sha256', 'effective_keys', 'unlisted_keys_present'], path)
    const inheritedKeys = Array.isArray(environment?.inherited_os_keys_present)
      ? environment.inherited_os_keys_present : []
    const effectiveKeys = Array.isArray(environment?.effective_keys)
      ? environment.effective_keys : []
    const expectedKeys = [...inheritedKeys, ...Object.keys(modelOverrides)].sort()
    check(environment?.schema === 'camelid.windows-child-environment/v1'
      && sameJson(environment?.model_overrides, modelOverrides)
      && sameJson(environment?.inherited_os_allowlist, WINDOWS_CHILD_ENV_ALLOWLIST)
      && inheritedKeys.every((key) => typeof key === 'string'
        && WINDOWS_CHILD_ENV_ALLOWLIST.includes(key))
      && new Set(inheritedKeys).size === inheritedKeys.length
      && sameJson(inheritedKeys, [...inheritedKeys].sort())
      && environment?.inherited_os_values_redacted === true
      && /^[0-9a-f]{64}$/.test(environment?.inherited_os_environment_sha256 || '')
      && effectiveKeys.every((key) => typeof key === 'string')
      && sameJson(effectiveKeys, expectedKeys)
      && environment?.unlisted_keys_present === false,
    `${path} must seal the exact privacy-safe child environment contract`)
  }
  validateEnvironment(receipt?.runtime_contract?.camelid?.environment, SAFE_CAMELID_ENV,
    'runtime_contract.camelid.environment')
  validateEnvironment(receipt?.runtime_contract?.llama_cpp?.environment, SAFE_LLAMA_ENV,
    'runtime_contract.llama_cpp.environment')
  check(sameJson(
    receipt?.runtime_contract?.camelid?.environment?.inherited_os_keys_present,
    receipt?.runtime_contract?.llama_cpp?.environment?.inherited_os_keys_present,
  ) && receipt?.runtime_contract?.camelid?.environment?.inherited_os_environment_sha256
    === receipt?.runtime_contract?.llama_cpp?.environment?.inherited_os_environment_sha256,
  'both engines must seal the same inherited OS environment')
  check(sameJson(receipt?.runtime_contract?.camelid?.command, receiptCamelidCommand())
    && !receipt?.runtime_contract?.camelid?.command?.includes('--model')
    && sameJson(receipt?.runtime_contract?.camelid?.request, {
      ...structuredClone(CHAT_REQUEST), camelid_enable_thinking_omitted: true,
      camelid_receipt_requested: false,
    })
    && sameJson(receipt?.runtime_contract?.llama_cpp?.command, receiptLlamaCommand())
    && sameJson(receipt?.runtime_contract?.llama_cpp?.tokenize, {
      content: 'exact_frozen_camelid_rendered_prompt', ...structuredClone(TOKENIZE_FLAGS),
    })
    && sameJson(receipt?.runtime_contract?.llama_cpp?.completion, {
      prompt: 'identical_cross_engine_prompt_token_ids',
      ...structuredClone(LLAMA_COMPLETION_SETTINGS),
    })
    && sameJson(receipt?.runtime_contract?.limits, LIMITS),
  'runtime commands, environments, requests, and limits must remain exact')
  const prompt = receipt?.runtime_contract?.prompt
  close(prompt, ['rendered_text_redacted', 'actual_utf8_bytes', 'actual_sha256',
    'normalized_utf8_bytes', 'normalized_sha256', 'token_count', 'token_ids_sha256',
    'token_ids_persisted_in_camelid_step'], 'runtime_contract.prompt')
  check(prompt?.rendered_text_redacted === true && positiveInteger(prompt?.actual_utf8_bytes)
    && /^[0-9a-f]{64}$/.test(prompt?.actual_sha256 || '')
    && prompt?.normalized_utf8_bytes === NORMALIZED_PROMPT_UTF8_BYTES
    && prompt?.normalized_sha256 === NORMALIZED_PROMPT_SHA256
    && positiveInteger(prompt?.token_count) && /^[0-9a-f]{64}$/.test(prompt?.token_ids_sha256 || '')
    && prompt?.token_ids_persisted_in_camelid_step === true,
  'prompt identity must be redacted and exact')
  check(Array.isArray(receipt?.steps) && receipt.steps.length === STEP_CONTRACT.length,
    'step count must remain exact')
  if (Array.isArray(receipt?.steps)) {
    STEP_CONTRACT.forEach(([name, engine, method, endpoint], index) => {
      const step = receipt.steps[index]
      close(step, ['ordinal', 'name', 'engine', 'method', 'endpoint', 'http_status',
        'elapsed_ms', 'evidence'], `steps.${index}`)
      check(step?.ordinal === index + 1 && step?.name === name && step?.engine === engine
        && step?.method === method && step?.endpoint === endpoint && step?.http_status === 200
        && nonNegativeInteger(step?.elapsed_ms), `step ${index + 1} order must remain exact`)
    })
    const byName = Object.fromEntries(receipt.steps.map((step) => [step.name, step.evidence]))
    for (const name of ['camelid_baseline_health', 'camelid_loaded_health',
      'camelid_final_health']) {
      close(byName[name], ['loaded_now', 'generation_ready', 'active_model_id', 'build',
        'version', 'backend', 'selected_backend', 'cuda_resident_active', 'queue_idle', 'q8_policy',
        'listen_addr'], `steps.${name}.evidence`)
      const loaded = name !== 'camelid_baseline_health'
      check(byName[name]?.loaded_now === loaded && byName[name]?.generation_ready === loaded
        && byName[name]?.active_model_id === (loaded ? ROW_ID : null)
        && byName[name]?.version === CAMELID_RELEASE_VERSION
        && byName[name]?.build === receipt?.provenance?.source_describe
        && byName[name]?.backend === (loaded ? 'llama' : 'none')
        && byName[name]?.selected_backend === (loaded ? 'cpu_reference' : null)
        && byName[name]?.cuda_resident_active === (loaded ? false : null)
        && byName[name]?.queue_idle === true
        && byName[name]?.q8_policy === 'forced_lazy_file_backed_q8'
        && byName[name]?.listen_addr === CAMELID_ADDR,
      `${name} must remain the exact idle ${loaded ? 'loaded' : 'unloaded'} health state`)
    }
    for (const name of ['camelid_baseline_gpu', 'camelid_final_gpu']) {
      close(byName[name], ['available', 'enabled', 'backend_redacted', 'run_count',
        'device_redacted'], `steps.${name}.evidence`)
      check(typeof byName[name]?.available === 'boolean' && byName[name]?.enabled === false
        && byName[name]?.backend_redacted === true && byName[name]?.run_count === 0
        && byName[name]?.device_redacted === true,
      `${name} must remain disabled and fully redacted`)
    }
    close(byName.camelid_load, ['id', 'status', 'generation_ready', 'model_path_redacted',
      'request_path_redacted'], 'steps.camelid_load.evidence')
    check(sameJson(byName.camelid_load, { id: ROW_ID, status: 'loaded', generation_ready: true,
      model_path_redacted: true, request_path_redacted: true }),
    'Camelid load evidence must remain exact and path-redacted')
    close(byName.camelid_verify_identity, ['model_id', 'gguf_sha256', 'eligible',
      'profile_id', 'report'], 'steps.camelid_verify_identity.evidence')
    check(sameJson(byName.camelid_verify_identity, { model_id: ROW_ID,
      gguf_sha256: EXACT_ROW.source.sha256, eligible: false, profile_id: null, report: null }),
    'Camelid verification must bind the exact unsupported row')
    close(byName.camelid_props, ['model_id', 'generation_ready', 'template',
      'template_redacted', 'default_thinking_text_envelope'], 'steps.camelid_props.evidence')
    close(byName.camelid_props?.template, ['utf8_bytes', 'sha256'],
      'steps.camelid_props.evidence.template')
    check(byName.camelid_props?.model_id === ROW_ID
      && byName.camelid_props?.generation_ready === true
      && sameJson(byName.camelid_props?.template, TEMPLATE_IDENTITY)
      && byName.camelid_props?.template_redacted === true
      && byName.camelid_props?.default_thinking_text_envelope === true,
    'Camelid props must bind the exact template and bounded envelope')
    for (const name of ['camelid_apply_template_before', 'camelid_apply_template_after']) {
      const expectedKeys = ['actual_prompt_utf8_bytes', 'actual_prompt_sha256',
        'normalized_prompt_utf8_bytes', 'normalized_prompt_sha256',
        'dynamic_date_occurrences', 'dynamic_date_redacted', 'default_thinking',
        'add_generation_prompt']
      if (name.endsWith('_after')) expectedKeys.push('identical_to_frozen_pre_forward_prompt')
      close(byName[name], expectedKeys, `steps.${name}.evidence`)
      check(positiveInteger(byName[name]?.actual_prompt_utf8_bytes)
        && /^[0-9a-f]{64}$/.test(byName[name]?.actual_prompt_sha256 || '')
        && byName[name]?.normalized_prompt_utf8_bytes === NORMALIZED_PROMPT_UTF8_BYTES
        && byName[name]?.normalized_prompt_sha256 === NORMALIZED_PROMPT_SHA256
        && byName[name]?.dynamic_date_occurrences === 1
        && byName[name]?.dynamic_date_redacted === true
        && byName[name]?.default_thinking === true
        && byName[name]?.add_generation_prompt === true
        && (!name.endsWith('_after')
          || byName[name]?.identical_to_frozen_pre_forward_prompt === true),
      `${name} must remain the exact date-redacted default-thinking prompt`)
    }
    close(byName.camelid_tokenize_prompt, ['token_count', 'token_ids_sha256',
      'add_special', 'parse_special', 'prompt_text_redacted'],
    'steps.camelid_tokenize_prompt.evidence')
    close(byName.llama_tokenize_prompt, ['token_count', 'token_ids_sha256',
      'add_special', 'parse_special', 'prompt_text_redacted',
      'identical_to_camelid_prompt_token_ids'], 'steps.llama_tokenize_prompt.evidence')
    for (const name of ['camelid_tokenize_prompt', 'llama_tokenize_prompt']) {
      check(positiveInteger(byName[name]?.token_count)
        && /^[0-9a-f]{64}$/.test(byName[name]?.token_ids_sha256 || '')
        && byName[name]?.add_special === false && byName[name]?.parse_special === true
        && byName[name]?.prompt_text_redacted === true,
      `${name} must use the exact special-token parsing flags`)
    }
    const chat = byName.camelid_chat_first_forward
    close(chat, ['prompt_token_ids', 'generated_token_ids', 'bubble', 'usage',
      'finish_reason', 'first_token_top_logit_verified',
      'per_step_logit_diagnostics_absent_or_empty', 'first_forward',
      'generated_ids_non_eog', 'camelid_receipt_present'],
    'steps.camelid_chat_first_forward.evidence')
    close(chat?.bubble, ['redacted', 'utf8_bytes', 'sha256'],
      'steps.camelid_chat_first_forward.evidence.bubble')
    close(chat?.usage, ['prompt_tokens', 'completion_tokens', 'total_tokens'],
      'steps.camelid_chat_first_forward.evidence.usage')
    close(chat?.first_forward, ['weight_load_observed', 'weight_cache_hit',
      'prompt_cache_hit', 'first_token_evaluated'],
    'steps.camelid_chat_first_forward.evidence.first_forward')
    check(Array.isArray(chat?.prompt_token_ids) && chat.prompt_token_ids.length === chat?.usage?.prompt_tokens
      && chat.prompt_token_ids.every(nonNegativeInteger)
      && tokenArraySha256(chat.prompt_token_ids) === prompt?.token_ids_sha256
      && chat.prompt_token_ids.length === prompt?.token_count
      && Array.isArray(chat?.generated_token_ids)
      && chat.generated_token_ids.length === LIMITS.generated_tokens
      && chat.generated_token_ids.every((id) => nonNegativeInteger(id) && !EOG_TOKEN_IDS.includes(id))
      && chat?.usage?.completion_tokens === LIMITS.generated_tokens
      && chat?.usage?.total_tokens === chat?.usage?.prompt_tokens + LIMITS.generated_tokens
      && chat?.finish_reason === 'length' && chat?.first_token_top_logit_verified === true
      && chat?.per_step_logit_diagnostics_absent_or_empty === true
      && chat?.first_forward?.weight_load_observed === true
      && chat?.first_forward?.weight_cache_hit === false
      && chat?.first_forward?.prompt_cache_hit === false
      && chat?.first_forward?.first_token_evaluated === true
      && chat?.bubble?.redacted === true && nonNegativeInteger(chat?.bubble?.utf8_bytes)
      && /^[0-9a-f]{64}$/.test(chat?.bubble?.sha256 || '')
      && chat?.generated_ids_non_eog === true && chat?.camelid_receipt_present === false,
    'Camelid chat prompt and generation must remain the exact four-token first forward')
    check(byName.camelid_tokenize_prompt?.token_count === prompt?.token_count
      && byName.camelid_tokenize_prompt?.token_ids_sha256 === prompt?.token_ids_sha256
      && byName.llama_tokenize_prompt?.token_count === prompt?.token_count
      && byName.llama_tokenize_prompt?.token_ids_sha256 === prompt?.token_ids_sha256
      && byName.llama_tokenize_prompt?.identical_to_camelid_prompt_token_ids === true,
    'both tokenizers must bind the same frozen prompt IDs')
    check(byName.camelid_apply_template_before?.actual_prompt_sha256 === prompt?.actual_sha256
      && byName.camelid_apply_template_after?.actual_prompt_sha256 === prompt?.actual_sha256
      && byName.camelid_apply_template_before?.actual_prompt_utf8_bytes
        === prompt?.actual_utf8_bytes
      && byName.camelid_apply_template_after?.actual_prompt_utf8_bytes
        === prompt?.actual_utf8_bytes
      && byName.camelid_apply_template_after?.identical_to_frozen_pre_forward_prompt === true,
    'Camelid rendered prompt must remain stable across the only forward')
    close(byName.camelid_detokenize_generated, ['generated_token_count', 'content',
      'trimmed_content', 'bubble_equals_detokenized_trim'],
    'steps.camelid_detokenize_generated.evidence')
    close(byName.llama_detokenize_generated, ['generated_token_count', 'content',
      'completion_content_equals_detokenized'], 'steps.llama_detokenize_generated.evidence')
    for (const name of ['camelid_detokenize_generated', 'llama_detokenize_generated']) {
      close(byName[name]?.content, ['redacted', 'utf8_bytes', 'sha256'],
        `steps.${name}.evidence.content`)
      check(byName[name]?.generated_token_count === LIMITS.generated_tokens
        && byName[name]?.content?.redacted === true
        && nonNegativeInteger(byName[name]?.content?.utf8_bytes)
        && /^[0-9a-f]{64}$/.test(byName[name]?.content?.sha256 || ''),
      `${name} must retain only the redacted canonical byte surface`)
    }
    close(byName.camelid_detokenize_generated?.trimmed_content,
      ['redacted', 'utf8_bytes', 'sha256'],
      'steps.camelid_detokenize_generated.evidence.trimmed_content')
    check(sameJson(byName.camelid_detokenize_generated?.trimmed_content, chat?.bubble),
      'Camelid bubble must remain exactly bound to trimmed canonical detokenization')
    check(byName.camelid_detokenize_generated?.bubble_equals_detokenized_trim === true
      && byName.llama_detokenize_generated?.completion_content_equals_detokenized === true,
    'engine response text must remain bound to canonical detokenization')
    for (const name of ['llama_health', 'llama_final_health']) {
      close(byName[name], ['status', 'model_loaded', 'address'], `steps.${name}.evidence`)
      check(sameJson(byName[name], { status: 'ok', model_loaded: true, address: LLAMA_ADDR }),
        `${name} must remain the exact loopback loaded state`)
    }
    const llamaCompletion = byName.llama_completion_first_forward
    close(llamaCompletion, ['generated_token_ids', 'content', 'generated_token_count',
      'generated_ids_non_eog', 'deterministic_top_k_only', 'first_forward'],
    'steps.llama_completion_first_forward.evidence')
    close(llamaCompletion?.content, ['redacted', 'utf8_bytes', 'sha256'],
      'steps.llama_completion_first_forward.evidence.content')
    check(Array.isArray(llamaCompletion?.generated_token_ids)
      && llamaCompletion.generated_token_ids.length === LIMITS.generated_tokens
      && llamaCompletion.generated_token_ids.every((id) => nonNegativeInteger(id)
        && !EOG_TOKEN_IDS.includes(id))
      && llamaCompletion?.generated_token_count === LIMITS.generated_tokens
      && llamaCompletion?.generated_ids_non_eog === true
      && llamaCompletion?.deterministic_top_k_only === true
      && llamaCompletion?.first_forward === true
      && llamaCompletion?.content?.redacted === true
      && nonNegativeInteger(llamaCompletion?.content?.utf8_bytes)
      && /^[0-9a-f]{64}$/.test(llamaCompletion?.content?.sha256 || ''),
    'llama.cpp completion must remain the exact deterministic four-token first forward')
    check(sameJson(llamaCompletion?.content, byName.llama_detokenize_generated?.content),
      'llama.cpp completion content must remain exactly bound to canonical detokenization')
  }
  const comparison = receipt?.comparison
  close(comparison, ['prompt_render_stable_across_camelid_forward',
    'prompt_token_ids_cross_engine_exact', 'generated_token_ids',
    'canonical_detokenized_text', 'exact_token_and_text_match'], 'comparison')
  const ids = comparison?.generated_token_ids
  close(ids, ['camelid', 'llama_cpp', 'exact_match', 'first_divergent_index',
    'all_four_non_eog'], 'comparison.generated_token_ids')
  const computedTokenMatch = sameJson(ids?.camelid, ids?.llama_cpp)
  const computedDivergence = computedTokenMatch ? -1
    : ids?.camelid?.findIndex((token, index) => token !== ids?.llama_cpp?.[index])
  const texts = comparison?.canonical_detokenized_text
  close(texts, ['camelid', 'llama_cpp', 'exact_utf8_match',
    'camelid_bubble_equals_detokenized_trim', 'llama_content_equals_detokenized'],
  'comparison.canonical_detokenized_text')
  const computedTextMatch = texts?.camelid?.utf8_bytes === texts?.llama_cpp?.utf8_bytes
    && texts?.camelid?.sha256 === texts?.llama_cpp?.sha256
  check(comparison?.prompt_render_stable_across_camelid_forward === true
    && comparison?.prompt_token_ids_cross_engine_exact === true
    && Array.isArray(ids?.camelid) && ids.camelid.length === LIMITS.generated_tokens
    && Array.isArray(ids?.llama_cpp) && ids.llama_cpp.length === LIMITS.generated_tokens
    && sameJson(ids?.camelid,
      receipt?.steps?.find((step) => step?.name === 'camelid_chat_first_forward')
        ?.evidence?.generated_token_ids)
    && sameJson(ids?.llama_cpp,
      receipt?.steps?.find((step) => step?.name === 'llama_completion_first_forward')
        ?.evidence?.generated_token_ids)
    && ids?.exact_match === computedTokenMatch && ids?.first_divergent_index === computedDivergence
    && ids?.all_four_non_eog === [...(ids?.camelid || []), ...(ids?.llama_cpp || [])]
      .every((id) => nonNegativeInteger(id) && !EOG_TOKEN_IDS.includes(id))
    && texts?.camelid?.redacted === true && texts?.llama_cpp?.redacted === true
    && /^[0-9a-f]{64}$/.test(texts?.camelid?.sha256 || '')
    && /^[0-9a-f]{64}$/.test(texts?.llama_cpp?.sha256 || '')
    && sameJson(texts?.camelid,
      receipt?.steps?.find((step) => step?.name === 'camelid_detokenize_generated')
        ?.evidence?.content)
    && sameJson(texts?.llama_cpp,
      receipt?.steps?.find((step) => step?.name === 'llama_detokenize_generated')
        ?.evidence?.content)
    && texts?.exact_utf8_match === computedTextMatch
    && texts?.camelid_bubble_equals_detokenized_trim === true
    && texts?.llama_content_equals_detokenized === true
    && comparison?.exact_token_and_text_match === (computedTokenMatch && computedTextMatch),
  'comparison must be internally exact and derived')
  close(receipt?.resource_observations, ['preflight_available_physical_bytes',
    'preflight_available_disk_bytes', 'camelid', 'llama_cpp', 'thresholds_tripped'],
  'resource_observations')
  check(receipt?.resource_observations?.preflight_available_physical_bytes
      >= LIMITS.preflight_physical_bytes
    && receipt?.resource_observations?.preflight_available_disk_bytes >= LIMITS.preflight_disk_bytes
    && receipt?.resource_observations?.thresholds_tripped === false
    && ['camelid', 'llama_cpp'].every((engine) => {
      const resource = receipt?.resource_observations?.[engine]
      return exactKeys(resource, ['samples', 'minimum_available_physical_bytes',
        'peak_child_working_set_bytes', 'thresholds_tripped'])
        && positiveInteger(resource.samples)
        && nonNegativeInteger(resource.minimum_available_physical_bytes)
        && positiveInteger(resource.peak_child_working_set_bytes)
        && resource.thresholds_tripped === false
    }), 'both resource guards must report bounded observations')
  const decision = receipt?.gate_decision
  close(decision, ['parity_preparation', 'bounded_evidence_publishable', 'roster_parity_gate',
    'template_gate', 'api_webui_gate', 'context_gate', 'load_smoke_gate', 'support_claim',
    'disposition', 'target_tier', 'authorized_roster_scope', 'other_gates_unchanged'],
  'gate_decision')
  const exact = comparison?.exact_token_and_text_match === true
  check(decision?.parity_preparation === (exact ? 'pass' : 'mismatch')
    && decision?.bounded_evidence_publishable === exact
    && decision?.roster_parity_gate === 'blocked_unchanged'
    && decision?.template_gate === 'blocked_unchanged'
    && decision?.api_webui_gate === 'pending_unchanged'
    && decision?.context_gate === 'pending_unchanged'
    && decision?.load_smoke_gate === 'blocked_unchanged'
    && decision?.support_claim === false && decision?.disposition === 'hold'
    && decision?.target_tier === 'experimental_exact_row'
    && sameJson(decision?.authorized_roster_scope, [])
    && decision?.other_gates_unchanged === true,
  'decision must remain preparation-only under HOLD with no roster authorization')
  check(sameJson(receipt?.does_not_prove, DOES_NOT_PROVE), 'scope exclusions must remain exact')
  errors.push(...privacyErrors(receipt))
  return [...new Set(errors)]
}

function validateChatParityReceipt(receipt) {
  try { return validateReceiptUnsafe(receipt) }
  catch { return ['receipt validation could not safely inspect malformed input'] }
}

function assertValidChatParityReceipt(receipt) {
  if (validateChatParityReceipt(receipt).length) throw parityError('chat_parity_receipt_invalid')
  return receipt
}

async function runSmolLM3ChatParity(rawOptions, deps = {}) {
  deps = hardenWindowsChildDeps(deps)
  expect(rawOptions?.binary && rawOptions?.artifact && rawOptions?.cwd
    && rawOptions?.modelsDir && rawOptions?.llamaServer,
  'chat_parity_options_invalid')
  const options = {
    root: resolve(rawOptions.root || '.'),
    binary: resolve(rawOptions.binary),
    artifact: resolve(rawOptions.artifact),
    cwd: resolve(rawOptions.cwd),
    modelsDir: resolve(rawOptions.modelsDir),
    llamaServer: resolve(rawOptions.llamaServer),
    binaryProfile: rawOptions.binaryProfile || BINARY_PROFILE,
  }
  const preflight = deps.preflightImpl
    ? await deps.preflightImpl(options)
    : await runPreflight(options, deps)
  expect(preflight?.platform === 'windows-x86_64'
    && preflight?.preexisting_engine_processes_absent === true,
  'chat_parity_options_invalid')
  let artifactLock
  try {
    artifactLock = deps.acquireArtifactLockImpl
      ? await deps.acquireArtifactLockImpl(options.artifact)
      : await acquireWindowsArtifactReadLock(options.artifact, deps)
  } catch (error) {
    throw parityError(bridgeSmolError(error) || 'chat_parity_artifact_lock_failed')
  }
  const lockShapeValid = artifactLock?.acquired === true && artifactLock.exited
    && artifactLock.closed && typeof artifactLock.isExited === 'function'
    && typeof artifactLock.exitStatus === 'function'
    && typeof artifactLock.assertHeld === 'function' && typeof artifactLock.release === 'function'
  if (!lockShapeValid) {
    if (artifactLock?.acquired === true) {
      if (typeof artifactLock.release !== 'function') {
        throw parityError('chat_parity_artifact_lock_release_failed')
      }
      try {
        const cleanup = await artifactLock.release()
        expect(cleanup?.observed === true && cleanup?.released_token_observed === true
          && cleanup?.exit_code === 0,
        'chat_parity_artifact_lock_release_failed')
      } catch { throw parityError('chat_parity_artifact_lock_release_failed') }
    }
    throw parityError('chat_parity_artifact_lock_failed')
  }
  const whileLocked = async (operation) => {
    try { artifactLock.assertHeld() } catch { throw parityError('chat_parity_artifact_lock_lost') }
    const outcome = await Promise.race([
      Promise.resolve().then(operation).then((value) => ({ value }), (error) => ({ error })),
      artifactLock.exited.then((status) => ({ lockExited: status })),
    ])
    if (outcome.lockExited) throw parityError('chat_parity_artifact_lock_lost')
    if (outcome.error) throw outcome.error
    try { artifactLock.assertHeld() } catch { throw parityError('chat_parity_artifact_lock_lost') }
    return outcome.value
  }
  const lifecycle = { activeEngine: null, maxConcurrentEngines: 0, events: [], markers: {},
    portsFreeAfterStop: { camelid: false, llama_cpp: false } }
  const steps = []
  let prehash
  let posthash
  let postflight
  let camelid
  let llama
  let evidenceError
  try {
    prehash = await whileLocked(() => (deps.inspectArtifactIdentityImpl
      ? deps.inspectArtifactIdentityImpl(options.artifact, 'prehash')
      : inspectExactArtifactIdentity(options.artifact, deps)))
    expect(prehash?.size_bytes === EXACT_ROW.source.size_bytes
      && prehash?.sha256 === EXACT_ROW.source.sha256,
    'chat_parity_artifact_identity_mismatch')
    camelid = await runCamelidPhase({ options, preflight, artifactLock, lifecycle, steps, deps })
    llama = await runLlamaPhase({ options, camelid, artifactLock, lifecycle, steps, deps })
    expect(lifecycle.activeEngine === null && lifecycle.maxConcurrentEngines === 1
      && sameJson(lifecycle.events,
        ['camelid_started', 'camelid_closed', 'llama_cpp_started', 'llama_cpp_closed']),
    'chat_parity_overlap_detected')
    posthash = await whileLocked(() => (deps.inspectArtifactIdentityImpl
      ? deps.inspectArtifactIdentityImpl(options.artifact, 'posthash')
      : inspectExactArtifactIdentity(options.artifact, deps)))
    expect(sameJson(prehash, posthash), 'chat_parity_artifact_identity_mismatch')
    postflight = await whileLocked(() => (deps.postflightImpl
      ? deps.postflightImpl(options, preflight)
      : runPostflight(options, preflight, deps)))
    expect(postflight && sameJson(postflight.provenance, preflight.provenance)
      && sameJson(postflight.auto_select_roots, preflight.auto_select_roots)
      && sameJson(postflight.groundings, preflight.groundings)
      && sameJson(postflight.llama, preflight.llama),
    'chat_parity_source_changed')
  } catch (error) {
    evidenceError = error instanceof SmolLM3ChatParityError
      ? error : parityError(bridgeSmolError(error) || 'chat_parity_http_failed')
  }
  let release
  if (artifactLock.isExited()) {
    try {
      const timeout = boundedTimeout(2_000, null)
      const observation = await Promise.race([
        Promise.all([artifactLock.exited, artifactLock.closed]),
        timeout.promise,
      ])
      timeout.cancel()
      expect(Array.isArray(observation) && sameJson(observation[0], artifactLock.exitStatus()),
        'chat_parity_artifact_lock_release_failed')
    } catch { throw parityError('chat_parity_artifact_lock_release_failed') }
    if (!['chat_parity_resource_telemetry_unavailable', 'chat_parity_termination_failed']
      .includes(evidenceError?.code)) evidenceError = parityError('chat_parity_artifact_lock_lost')
  } else {
    try {
      release = await artifactLock.release()
      expect(release?.observed === true && release?.released_token_observed === true
        && release?.exit_code === 0,
      'chat_parity_artifact_lock_release_failed')
    } catch { throw parityError('chat_parity_artifact_lock_release_failed') }
  }
  if (evidenceError) throw evidenceError
  const artifact = {
    size_bytes: posthash.size_bytes,
    sha256: posthash.sha256,
    verified_after_lock_acquisition: true,
    verified_after_both_engines: true,
    mutation_guard: {
      mechanism: 'windows_file_stream_share_read', read_access: 'allowed', write_access: 'denied',
      delete_access: 'denied', rename_access: 'denied', symbolic_links_rejected: true,
      acquired_before_prehash: true, held_through_camelid: true, held_through_llama_cpp: true,
      held_through_posthash: true, released_token_observed: true, helper_exit_code: 0,
      artifact_path_in_helper_argv: false,
    },
    path_redacted: true,
  }
  const receipt = buildReceipt({ preflight, artifact, lifecycle, steps, camelid, llama,
    createdUtc: (deps.nowIsoImpl || (() => new Date().toISOString()))() })
  return assertValidChatParityReceipt(receipt)
}

async function writeReceiptAtomic(path, receipt, {
  mkdirImpl = mkdir, writeFileImpl = writeFile, renameImpl = rename, rmImpl = rm,
} = {}) {
  assertValidChatParityReceipt(receipt)
  const target = resolve(path)
  const temporary = `${target}.tmp-${randomBytes(8).toString('hex')}`
  try {
    await mkdirImpl(dirname(target), { recursive: true })
    await writeFileImpl(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' })
    await renameImpl(temporary, target)
  } catch {
    try { await rmImpl(temporary, { force: true }) } catch { /* fail closed */ }
    throw parityError('chat_parity_output_failed')
  }
}

const CLI_VALUE_OPTIONS = new Set([
  'root', 'binary', 'artifact', 'cwd', 'models-dir', 'llama-server', 'binary-profile', 'out',
])
const CLI_BOOLEAN_OPTIONS = new Set(['help'])

function parseArgs(argv) {
  const parsed = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    expect(typeof token === 'string' && token.startsWith('--'), 'chat_parity_options_invalid')
    const equals = token.indexOf('=')
    const name = token.slice(2, equals === -1 ? undefined : equals)
    expect(name && (CLI_VALUE_OPTIONS.has(name) || CLI_BOOLEAN_OPTIONS.has(name))
      && !parsed.has(name), 'chat_parity_options_invalid')
    if (CLI_BOOLEAN_OPTIONS.has(name)) {
      expect(equals === -1, 'chat_parity_options_invalid')
      parsed.set(name, true)
      continue
    }
    const value = equals === -1 ? argv[++index] : token.slice(equals + 1)
    expect(typeof value === 'string' && value.length > 0 && !value.startsWith('--'),
      'chat_parity_options_invalid')
    parsed.set(name, value)
  }
  if (parsed.has('help')) expect(parsed.size === 1, 'chat_parity_options_invalid')
  return parsed
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv)
  if (args.has('help')) {
    process.stdout.write('Usage: node scripts/hf-qualification-smollm3-chat-parity.mjs \\\n'
      + '  --binary <camelid.exe> --artifact <SmolLM3-Q8_0.gguf> \\\n'
      + '  --cwd <isolated-dir> --models-dir <empty-dir> --llama-server <llama-server.exe> [options]\n'
      + 'Options:\n'
      + '  --root <path>         Clean tracked Camelid source root (default: .)\n'
      + '  --binary-profile <id> Provenance label (default: release-fat-lto)\n'
      + '  --out <path>          Atomically write the privacy-safe sealed receipt\n')
    return
  }
  for (const required of ['binary', 'artifact', 'cwd', 'models-dir', 'llama-server']) {
    expect(args.has(required), 'chat_parity_options_invalid')
  }
  const root = resolve(args.get('root') || '.')
  const receipt = await runSmolLM3ChatParity({
    root,
    binary: resolve(root, args.get('binary')),
    artifact: resolve(root, args.get('artifact')),
    cwd: resolve(root, args.get('cwd')),
    modelsDir: resolve(root, args.get('models-dir')),
    llamaServer: resolve(root, args.get('llama-server')),
    binaryProfile: args.get('binary-profile') || BINARY_PROFILE,
  })
  if (args.has('out')) await writeReceiptAtomic(resolve(root, args.get('out')), receipt)
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`)
  if (!receipt.comparison.exact_token_and_text_match) process.exitCode = 2
}

export {
  APPLY_TEMPLATE_REQUEST,
  BINARY_PROFILE,
  CAMELID_ADDR,
  CAMELID_RELEASE_VERSION,
  CHAT_REQUEST,
  DOES_NOT_PROVE,
  EOG_TOKEN_IDS,
  EXACT_ROW,
  GROUNDING_FILES,
  LIMITS,
  LOCAL_FILE_LIMITS,
  LLAMA_ADDR,
  LLAMA_COMPLETION_SETTINGS,
  LLAMA_PIN,
  RECEIPT_SCHEMA,
  ROW_ID,
  SAFE_CAMELID_ENV,
  SAFE_LLAMA_ENV,
  STEP_CONTRACT,
  TEMPLATE_IDENTITY,
  TOKENIZE_FLAGS,
  WINDOWS_CHILD_ENV_ALLOWLIST,
  SmolLM3ChatParityError,
  assertValidChatParityReceipt,
  buildCamelidServeArgs,
  buildChildEnv,
  buildLlamaEnv,
  buildLlamaServeArgs,
  buildWindowsChildEnv,
  buildReceipt,
  classifySmolLM3ChatParityError,
  describeChildEnvironment,
  httpJson,
  inspectExactArtifactIdentity,
  inspectGroundings,
  inspectLlamaPackage,
  normalizeCamelidChat,
  normalizeDetokenize,
  normalizeRenderedPrompt,
  normalizeTokenize,
  parseArgs,
  receiptCamelidCommand,
  receiptLlamaCommand,
  runPreflight,
  runSmolLM3ChatParity,
  sha256File,
  validateChatParityReceipt,
  writeReceiptAtomic,
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    const failure = classifySmolLM3ChatParityError(error)
    console.error(`${failure.error_code}: ${failure.reason}`)
    process.exit(1)
  })
}
