#!/usr/bin/env node

import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import {
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
  buildCamelidServeArgs,
  buildChildEnv,
  buildLlamaEnv,
  buildLlamaServeArgs,
  buildWindowsChildEnv,
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
  runSmolLM3ChatParity,
  validateChatParityReceipt,
  writeReceiptAtomic,
} from './hf-qualification-smollm3-chat-parity.mjs'
import { SmolLM3LoadSmokeError } from './hf-qualification-smollm3-load-smoke.mjs'

const root = resolve('.')
const binary = resolve('qualification-bin', 'camelid.exe')
const artifact = resolve('qualification-artifacts', EXACT_ROW.source.file)
const cwd = resolve('qualification-run', 'work')
const modelsDir = resolve('qualification-run', 'empty-models')
const llamaServer = resolve('qualification-reference', 'bin', 'llama-server.exe')
const sourceHead = 'a'.repeat(40)
const sourceDescribe = 'v0.6.1-1-gaaaaaaaa'
const binarySha256 = 'b'.repeat(64)
const createdUtc = '2026-08-10T20:00:00.000Z'
const promptTokens = [128011, 9125, 198, 9906, 128012, 128011, 78191, 198]
const generatedTokens = [128002, 220, 1234, 5678]
const camelidBubble = '<think> One two'
const detokenized = '<think> One two '

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]))
  }
  return value
}

function reseal(receipt) {
  const { receipt_id: _old, ...body } = receipt
  const receiptId = createHash('sha256').update(JSON.stringify(canonical(body))).digest('hex')
  return { schema: body.schema, receipt_id: receiptId,
    ...Object.fromEntries(Object.entries(body).filter(([key]) => key !== 'schema')) }
}

function clone(value) { return structuredClone(value) }

async function expectCode(promise, code) {
  await assert.rejects(promise, (error) => {
    const classified = classifySmolLM3ChatParityError(error)
    assert.equal(classified.error_code, code)
    return true
  })
}

assert.equal(RECEIPT_SCHEMA, 'camelid.model-qualification.chat-parity-preparation/v1')
assert.equal(ROW_ID, 'smollm3_3b_q8_0')
assert.equal(BINARY_PROFILE, 'release-fat-lto')
assert.equal(CAMELID_ADDR, '127.0.0.1:8297')
assert.equal(CAMELID_RELEASE_VERSION, '0.6.1')
assert.equal(LLAMA_ADDR, '127.0.0.1:8299')
assert.deepEqual(EXACT_ROW.source, {
  repo: 'ggml-org/SmolLM3-3B-GGUF',
  file: 'SmolLM3-Q8_0.gguf',
  revision: '4965cb60b150737b68a0408c36aeefb65078f894',
  size_bytes: 3_275_574_624,
  sha256: '8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248',
  license: 'apache-2.0',
})
assert.deepEqual(TEMPLATE_IDENTITY, {
  utf8_bytes: 5_493,
  sha256: 'b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e',
})
assert.equal(GROUNDING_FILES.shape_pack.sha256,
  'd46794448b7c2585d0aa83dfd7bb17d4904c2dcbc048ade1ef68cd3863166de6')
assert.equal(GROUNDING_FILES.runtime_envelope.sha256,
  'c99b9752db771529ff447a05993a6f1119095c15b53fca3bf3491468c2c8590e')
assert.equal(LLAMA_PIN.executable_sha256,
  '6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae')
assert.equal(LLAMA_PIN.server_impl_sha256,
  'df4fd737e380ba2cd39654af9233da22e2cdcbbf43ac376009cbec47aaaec750')
assert.equal(LLAMA_PIN.package_manifest_sha256,
  'd70bbe8beb7848396d0993ee533062c200350fd9961e2b92c799b24f94a33e93')
assert.equal(LLAMA_PIN.archive_sha256,
  'b835d5c5155dd2a5ed748a0351debf2ede0dc9f808757e0429f8700a11832dcd')
assert.deepEqual(EOG_TOKEN_IDS, [128012])
assert.deepEqual(CHAT_REQUEST, {
  model: ROW_ID,
  messages: [{ role: 'user', content: 'Hello, please help me.' }],
  max_tokens: 4,
  temperature: 0,
  top_k: 1,
  seed: 0,
  stream: false,
})
for (const field of ['tools', 'stop', 'logprobs', 'camelid_receipt', 'camelid_enable_thinking',
  'camelid_logit_token_ids']) assert.equal(Object.hasOwn(CHAT_REQUEST, field), false)
assert.deepEqual(APPLY_TEMPLATE_REQUEST, { messages: CHAT_REQUEST.messages })
assert.deepEqual(TOKENIZE_FLAGS, { add_special: false, parse_special: true })
assert.deepEqual(LLAMA_COMPLETION_SETTINGS, {
  n_predict: 4,
  temperature: 0,
  top_k: 1,
  seed: 0,
  cache_prompt: false,
  samplers: ['top_k'],
  return_tokens: true,
  stream: false,
})
assert.equal(LIMITS.generated_tokens, 4)
assert.equal(LIMITS.llama_child_working_set_abort_bytes, 4 * 1024 ** 3)
assert.ok(LIMITS.preflight_physical_bytes
  >= LIMITS.llama_child_working_set_abort_bytes + 2 * 1024 ** 3)
assert.ok(LIMITS.llama_child_working_set_abort_bytes
  > LIMITS.camelid_child_working_set_abort_bytes)
assert.ok(DOES_NOT_PROVE.some((claim) => claim.includes('1, 5, and 50')))
assert.ok(DOES_NOT_PROVE.some((claim) => claim.includes('blocked load-smoke gate')))
assert.ok(DOES_NOT_PROVE.some((claim) => claim.includes('support or promotion')))

const camelidArgs = buildCamelidServeArgs(modelsDir)
assert.deepEqual(camelidArgs, [
  'serve', '--addr', CAMELID_ADDR, '--models-dir', modelsDir, '--threads', '4',
  '--gpu', 'off', '--deterministic', '--kv-quant', 'f16', '--no-open',
  '--max-prompt-tokens', '1024', '--max-generation-tokens', '4',
])
assert.equal(camelidArgs.includes('--model'), false)
const llamaArgs = buildLlamaServeArgs(artifact)
assert.deepEqual(llamaArgs, [
  '--host', '127.0.0.1', '--port', '8299', '-m', artifact,
  '-ngl', '0', '-c', '512', '-b', '512', '-ub', '512', '-t', '4',
  '-ctk', 'f16', '-ctv', 'f16', '-fa', 'off', '--no-repack', '--no-warmup',
  '-np', '1', '--no-cont-batching',
])
assert.deepEqual(receiptCamelidCommand(), [
  '<camelid>', 'serve', '--addr', CAMELID_ADDR, '--models-dir', '<empty-models-dir>',
  '--threads', '4', '--gpu', 'off', '--deterministic', '--kv-quant', 'f16',
  '--no-open', '--max-prompt-tokens', '1024', '--max-generation-tokens', '4',
])
assert.deepEqual(receiptLlamaCommand(), [
  '<llama-server>', '--host', '127.0.0.1', '--port', '8299', '-m', '<artifact>',
  '-ngl', '0', '-c', '512', '-b', '512', '-ub', '512', '-t', '4',
  '-ctk', 'f16', '-ctv', 'f16', '-fa', 'off', '--no-repack', '--no-warmup',
  '-np', '1', '--no-cont-batching',
])

const hostileInheritedEnv = {
  Path: 'kept',
  PATHEXT: '.EXE',
  SYSTEMROOT: 'C:\\Windows',
  HF_TOKEN: 'hf_private',
  GH_TOKEN: 'ghp_private',
  AWS_ACCESS_KEY_ID: 'private',
  AWS_SECRET_ACCESS_KEY: 'private',
  CAMELID_PROFILE: 'fast',
  CAMELID_SECRET: 'removed',
  CUDA_VISIBLE_DEVICES: '7',
  LLAMA_ARG: 'unsafe',
}
assert.deepEqual(WINDOWS_CHILD_ENV_ALLOWLIST, [
  'COMSPEC', 'PATH', 'PATHEXT', 'SYSTEMDRIVE', 'SYSTEMROOT', 'TEMP', 'TMP', 'WINDIR',
])
assert.deepEqual(buildWindowsChildEnv(hostileInheritedEnv), {
  PATH: 'kept',
  PATHEXT: '.EXE',
  SYSTEMROOT: 'C:\\Windows',
})
for (const inherited of [
  { Path: 'first', PATH: 'second' },
  { PATH: 7 },
]) {
  assert.throws(() => buildWindowsChildEnv(inherited),
    (error) => error?.code === 'chat_parity_options_invalid')
}
const childEnv = buildChildEnv(hostileInheritedEnv)
assert.equal(childEnv.PATH, 'kept')
assert.equal(childEnv.CAMELID_SECRET, undefined)
for (const [key, value] of Object.entries(SAFE_CAMELID_ENV)) assert.equal(childEnv[key], value)
const llamaEnv = buildLlamaEnv(hostileInheritedEnv)
assert.equal(llamaEnv.PATH, 'kept')
assert.equal(llamaEnv.LLAMA_ARG, undefined)
assert.equal(llamaEnv.CUDA_VISIBLE_DEVICES, '-1')
assert.deepEqual(SAFE_LLAMA_ENV, { CUDA_VISIBLE_DEVICES: '-1' })
const camelidEnvironmentContract = describeChildEnvironment(childEnv, SAFE_CAMELID_ENV)
const llamaEnvironmentContract = describeChildEnvironment(llamaEnv, SAFE_LLAMA_ENV)
for (const [contract, env, overrides] of [
  [camelidEnvironmentContract, childEnv, SAFE_CAMELID_ENV],
  [llamaEnvironmentContract, llamaEnv, SAFE_LLAMA_ENV],
]) {
  assert.equal(contract.schema, 'camelid.windows-child-environment/v1')
  assert.deepEqual(contract.model_overrides, overrides)
  assert.deepEqual(contract.inherited_os_allowlist, WINDOWS_CHILD_ENV_ALLOWLIST)
  assert.deepEqual(contract.inherited_os_keys_present, ['PATH', 'PATHEXT', 'SYSTEMROOT'])
  assert.equal(contract.inherited_os_values_redacted, true)
  assert.match(contract.inherited_os_environment_sha256, /^[0-9a-f]{64}$/)
  assert.deepEqual(contract.effective_keys, Object.keys(env).sort())
  assert.equal(contract.unlisted_keys_present, false)
  assert.doesNotMatch(JSON.stringify(contract), /kept|C:\\Windows/)
}
assert.equal(camelidEnvironmentContract.inherited_os_environment_sha256,
  llamaEnvironmentContract.inherited_os_environment_sha256)
for (const secret of ['HF_TOKEN', 'GH_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']) {
  assert.equal(childEnv[secret], undefined)
  assert.equal(llamaEnv[secret], undefined)
}

const regularStats = (size) => ({
  size,
  isFile: () => true,
  isSymbolicLink: () => false,
})
let rejectedArtifactStatCalls = 0
let rejectedArtifactHashCalls = 0
await expectCode(inspectExactArtifactIdentity('symlink.gguf', {
  lstatImpl: async () => ({
    size: EXACT_ROW.source.size_bytes,
    isFile: () => false,
    isSymbolicLink: () => true,
  }),
  statImpl: async () => { rejectedArtifactStatCalls += 1; return regularStats(EXACT_ROW.source.size_bytes) },
  sha256FileImpl: async () => { rejectedArtifactHashCalls += 1; return EXACT_ROW.source.sha256 },
}), 'chat_parity_artifact_identity_mismatch')
assert.equal(rejectedArtifactStatCalls, 0, 'a symlink must fail before target stat')
assert.equal(rejectedArtifactHashCalls, 0, 'a symlink must fail before hashing')
await expectCode(inspectExactArtifactIdentity('wrong-size.gguf', {
  lstatImpl: async () => regularStats(EXACT_ROW.source.size_bytes - 1),
  statImpl: async () => regularStats(EXACT_ROW.source.size_bytes - 1),
  sha256FileImpl: async () => { rejectedArtifactHashCalls += 1; return EXACT_ROW.source.sha256 },
}), 'chat_parity_artifact_identity_mismatch')
assert.equal(rejectedArtifactHashCalls, 0, 'a wrong pinned size must fail before hashing')

const fakeOracleNames = [
  LLAMA_PIN.executable,
  LLAMA_PIN.server_impl_executable,
  ...Array.from({ length: LLAMA_PIN.package_file_count - 2 }, (_, index) => `bounded-${index}.dll`),
]
let oversizedOracleHashCalls = 0
await expectCode(inspectLlamaPackage(llamaServer, {
  platformInfo: () => ({ platform: 'win32', arch: 'x64' }),
  readdirImpl: async () => fakeOracleNames.map((name) => ({ name, isFile: () => true })),
  lstatImpl: async () => regularStats(1),
  statImpl: async (path) => regularStats(path.endsWith('bounded-0.dll')
    ? LOCAL_FILE_LIMITS.oracle_file_bytes + 1
    : path.endsWith(LLAMA_PIN.executable) ? LLAMA_PIN.executable_size_bytes
      : path.endsWith(LLAMA_PIN.server_impl_executable) ? LLAMA_PIN.server_impl_size_bytes : 1),
  sha256FileImpl: async () => { oversizedOracleHashCalls += 1; return '0'.repeat(64) },
  execFileImpl: async () => { throw new Error('must not execute') },
}), 'chat_parity_oracle_identity_mismatch')
assert.equal(oversizedOracleHashCalls, 0,
  'an oversized package member must fail before any package hash begins')
let aggregateOracleHashCalls = 0
await expectCode(inspectLlamaPackage(llamaServer, {
  platformInfo: () => ({ platform: 'win32', arch: 'x64' }),
  readdirImpl: async () => fakeOracleNames.map((name) => ({ name, isFile: () => true })),
  lstatImpl: async () => regularStats(1),
  statImpl: async (path) => regularStats(path.endsWith(LLAMA_PIN.executable)
    ? LLAMA_PIN.executable_size_bytes
    : path.endsWith(LLAMA_PIN.server_impl_executable) ? LLAMA_PIN.server_impl_size_bytes
      : 32 * 1024 ** 2),
  sha256FileImpl: async () => { aggregateOracleHashCalls += 1; return '0'.repeat(64) },
  execFileImpl: async () => { throw new Error('must not execute') },
}), 'chat_parity_oracle_identity_mismatch')
assert.ok(aggregateOracleHashCalls > 0 && aggregateOracleHashCalls < fakeOracleNames.length,
  'the aggregate package cap must stop hashing at the first member that crosses the bound')

assert.deepEqual(parseArgs(['--help']), new Map([['help', true]]))
assert.deepEqual(parseArgs(['--binary=x', '--artifact', 'a', '--cwd', 'c',
  '--models-dir=m', '--llama-server', 'l']), new Map([
  ['binary', 'x'], ['artifact', 'a'], ['cwd', 'c'], ['models-dir', 'm'],
  ['llama-server', 'l'],
]))
for (const invalid of [
  ['positional'], ['--unknown=x'], ['--binary'], ['--binary', '--artifact=x'],
  ['--binary=x', '--binary=y'], ['--help', '--binary=x'], ['--help=true'],
]) {
  assert.throws(() => parseArgs(invalid), SmolLM3ChatParityError)
}

const privateParityError = new SmolLM3ChatParityError('chat_parity_resource_abort')
assert.equal(privateParityError.code, 'chat_parity_resource_abort')
assert.throws(() => { privateParityError.code = 'chat_parity_http_failed' }, TypeError)
assert.equal(classifySmolLM3ChatParityError(privateParityError).error_code,
  'chat_parity_resource_abort')
const forgedError = new Error('forged')
forgedError.code = 'chat_parity_resource_abort'
assert.equal(classifySmolLM3ChatParityError(forgedError).error_code,
  'chat_parity_http_failed')
const sharedSmolError = new SmolLM3LoadSmokeError('load_smoke_resource_abort')
sharedSmolError.code = 'load_smoke_http_failed'
assert.equal(classifySmolLM3ChatParityError(sharedSmolError).error_code,
  'chat_parity_resource_abort')

const committedGrounding = await inspectGroundings(root)
assert.equal(committedGrounding.renderer_git_blob_sha1,
  'ecfac8933fd4d550c8e45f2822e7560e5f0a69ff')
assert.equal(committedGrounding.shape_case.normalized_prompt_sha256,
  '7619416ae94ba9a00378d976bfa944f5ba726747f9b67ba4e862d9a7fe20e4f1')
const shapePack = JSON.parse(await readFile(resolve(GROUNDING_FILES.shape_pack.path), 'utf8'))
const shapeCase = shapePack.cases.find((entry) => entry.id
  === 'default_think_single_user_generation_prompt')
const renderedPrompt = shapeCase.normalized_prompt.replace(
  '{{CURRENT_DATE_DD_MONTH_YYYY}}', '10 August 2026')
const rendered = normalizeRenderedPrompt({ prompt: renderedPrompt }, shapeCase.normalized_prompt)
assert.equal(rendered.prompt, renderedPrompt)
assert.equal(rendered.evidence.dynamic_date_occurrences, 1)
assert.equal(rendered.evidence.normalized_prompt_sha256, shapeCase.normalized_prompt_sha256)
assert.throws(() => normalizeRenderedPrompt({ prompt: renderedPrompt.replace(
  '10 August 2026', 'August 10, 2026') }, shapeCase.normalized_prompt), SmolLM3ChatParityError)
assert.throws(() => normalizeRenderedPrompt({ prompt: `${renderedPrompt}Today Date: 10 August 2026\n` },
  shapeCase.normalized_prompt), SmolLM3ChatParityError)
assert.deepEqual(normalizeTokenize({ tokens: promptTokens }, 'camelid').tokens, promptTokens)
assert.throws(() => normalizeTokenize({ tokens: [1, -1] }, 'llama_cpp'), SmolLM3ChatParityError)
const normalizedCamelidDetokenize = normalizeDetokenize({ content: detokenized }, 'camelid')
assert.equal(normalizedCamelidDetokenize.content, detokenized)
assert.deepEqual(normalizedCamelidDetokenize.evidence.trimmed_content,
  { redacted: true, utf8_bytes: Buffer.byteLength(detokenized.trim(), 'utf8'),
    sha256: createHash('sha256').update(detokenized.trim()).digest('hex') })

function health(loaded, final = false, version = CAMELID_RELEASE_VERSION,
  build = sourceDescribe) {
  const body = {
    ok: true,
    engine: 'camelid',
    version,
    build,
    loaded_now: loaded,
    generation_ready: loaded,
    vision_ready: false,
    active_model_id: loaded ? ROW_ID : null,
    backend: loaded ? 'llama' : 'none',
    model_family: loaded ? 'llama-family' : null,
    engine_queue_depth: 0,
    engine_queued_tasks: 0,
    engine_active_task_id: null,
    engine_active_generated_tokens: 0,
    continuous_batch_slots: 1,
    executable: 'private-but-normalizer-redacts.exe',
    listen_addr: CAMELID_ADDR,
    q8_runtime: {
      policy: 'forced_lazy_file_backed_q8', lazy_q8_linear: true,
      retain_q8_blocks: false, file_cache_bytes: 0,
    },
    execution_plan: loaded ? {
      profile: 'safe', operating_system: 'windows', architecture: 'x86_64',
      model_family: 'smollm3', quant_type: 'Q8_0', exact_model_row: EXACT_ROW.source.file,
      support_level: 'unknown_or_unvalidated', selected_backend: 'cpu_reference',
      selected_q8_path: 'safe_dense_or_q8_cpu',
      diagnostics_status: 'operator-requested RSS timings enabled; performance claims disabled',
      cuda_resident_active: false,
    } : null,
  }
  if (final) {
    body.engine_active_elapsed_seconds = 0
    body.engine_stalled_seconds = 0
  }
  return body
}

const gpu = { available: false, enabled: false, backend: 'none', run_count: 0 }
const load = {
  data: { id: ROW_ID, path: null, status: { value: 'loaded' },
    camelid: { generation_ready: true, model_path_redacted: true } },
  camelid: { model_path_redacted: true,
    compatibility: 'partial_llama_server_models_load_local_path',
    scope: 'single_local_model_load_alias' },
}
const verify = { model_id: ROW_ID, gguf_sha256: EXACT_ROW.source.sha256,
  eligible: false, profile_id: null, report: null }
const props = {
  model_path: null,
  model_id: ROW_ID,
  camelid: { generation_ready: true, model_path_redacted: true },
  modalities: { vision: false },
  total_slots: 1,
  default_generation_settings: { is_processing: false, next_token: { has_next_token: true } },
  chat_template: shapePack.source_template.text,
  chat_template_caps: {
    detected_format: 'smollm3_exact_default_thinking_text_qualified',
    render_prompt_envelope: { content: 'text_only', add_generation_prompt: true },
  },
}

function chatBody(tokens = generatedTokens, bubble = camelidBubble) {
  return {
    model: ROW_ID,
    lane: 'experimental',
    choices: [{ index: 0, message: { role: 'assistant', content: bubble },
      finish_reason: 'length' }],
    usage: { prompt_tokens: promptTokens.length, completion_tokens: 4,
      total_tokens: promptTokens.length + 4 },
    camelid: {
      prompt_token_ids: [...promptTokens],
      generated_token_ids: [...tokens],
      top_logits: [{ token_id: tokens[0], logit: 9, probability: 1, rank: 1,
        selected: false }],
      step_top_logits: [],
      timings_ms: {
        weight_load: 123,
        weight_cache_hit: false,
        prompt_cache_hit: false,
        prompt_evaluation: { first_token_evaluated: true },
      },
    },
  }
}

const normalizedChat = normalizeCamelidChat(chatBody())
assert.deepEqual(normalizedChat.promptTokens, promptTokens)
assert.deepEqual(normalizedChat.generatedTokens, generatedTokens)
assert.equal(normalizedChat.evidence.first_forward.weight_cache_hit, false)
const omittedStepLogits = chatBody()
delete omittedStepLogits.camelid.step_top_logits
assert.deepEqual(normalizeCamelidChat(omittedStepLogits).generatedTokens, generatedTokens)
for (const mutate of [
  (body) => { body.usage.completion_tokens = 1 },
  (body) => { body.camelid.prompt_token_ids.pop() },
  (body) => { body.camelid.generated_token_ids[3] = EOG_TOKEN_IDS[0] },
  (body) => { body.camelid.timings_ms.weight_cache_hit = true },
  (body) => { body.camelid.top_logits[0].token_id += 1 },
  (body) => { body.camelid.step_top_logits.push(body.camelid.top_logits) },
  (body) => { body.choices[0].finish_reason = 'stop' },
]) {
  const body = chatBody()
  mutate(body)
  assert.throws(() => normalizeCamelidChat(body), SmolLM3ChatParityError)
}

function fakePreflight() {
  return {
    platform: 'windows-x86_64',
    provenance: {
      runtime_head: sourceHead,
      source_describe: sourceDescribe,
      tracked_files_clean: true,
      untracked_files_excluded: true,
      binary_profile: BINARY_PROFILE,
      binary_sha256: binarySha256,
      binary_version: `camelid ${sourceDescribe}`,
    },
    artifact: { size_bytes: EXACT_ROW.source.size_bytes,
      expected_sha256: EXACT_ROW.source.sha256, hash_recomputed: false,
      ignored: true, path_redacted: true },
    auto_select_roots: [
      'configured_models_dir', 'executable_models_dir', 'executable_dir', 'cwd_models_dir', 'cwd',
    ].map((kind) => ({ kind, exists: false, path_redacted: true,
      gguf_candidates: 0, default_preference_present: false })),
    ports_unbound: [CAMELID_ADDR, LLAMA_ADDR],
    preexisting_engine_processes_absent: true,
    available_physical_bytes: LIMITS.preflight_physical_bytes + 1,
    available_disk_bytes: LIMITS.preflight_disk_bytes + 1,
    groundings: committedGrounding,
    llama: { ...clone(LLAMA_PIN), version_verified: true,
      executable_path_redacted: true, package_path_redacted: true },
  }
}

function fakeLock({ deferLossClose = false } = {}) {
  let held = true
  let exitStatus = null
  let resolveExit
  let resolveClose
  let lossCloseReleased = false
  const exited = new Promise((resolvePromise) => { resolveExit = resolvePromise })
  const closed = new Promise((resolvePromise) => { resolveClose = resolvePromise })
  return {
    acquired: true,
    exited,
    closed,
    isExited: () => !held,
    exitStatus: () => exitStatus,
    assertHeld() { if (!held) throw new Error('lock lost') },
    async release() {
      assert.equal(held, true)
      held = false
      exitStatus = { error: false, code: 0, signal: null }
      resolveExit(exitStatus)
      resolveClose(true)
      return { observed: true, released_token_observed: true, exit_code: 0 }
    },
    lose() {
      held = false
      exitStatus = { error: false, code: 7, signal: null }
      resolveExit(exitStatus)
      if (!deferLossClose) {
        lossCloseReleased = true
        resolveClose({ code: 7, signal: null })
      }
    },
    closeLost() {
      lossCloseReleased = true
      resolveClose({ code: 7, signal: null })
    },
    lossCloseReleased: () => lossCloseReleased,
  }
}

function fakeHandle(pid) {
  let exitedStatus = null
  let closedStatus = null
  let resolveExit
  let resolveClose
  const exited = new Promise((resolvePromise) => { resolveExit = resolvePromise })
  const closed = new Promise((resolvePromise) => { resolveClose = resolvePromise })
  return {
    pid,
    exited,
    closed,
    kill: () => true,
    isExited: () => exitedStatus !== null,
    isClosed: () => closedStatus !== null,
    exitStatus: () => exitedStatus,
    closeStatus: () => closedStatus,
    logMarkers: () => ({ warming_up_seen: false,
      generation_warmup_complete_seen: false, raw_output_persisted: false,
      output_captured_only_for_markers: true, observed_output_bytes: 12 }),
    terminate() {
      exitedStatus = { error: false, code: 0, signal: 'SIGTERM' }
      closedStatus = { code: 0, signal: 'SIGTERM' }
      resolveExit(exitedStatus)
      resolveClose(closedStatus)
    },
    closeOnly() {
      closedStatus = { code: 0, signal: null }
      resolveClose(closedStatus)
    },
    exitEarly() {
      exitedStatus = { error: false, code: 7, signal: null }
      closedStatus = { code: 7, signal: null }
      resolveExit(exitedStatus)
      resolveClose(closedStatus)
    },
  }
}

function fakeGuard(engine, abortAt = null) {
  const controller = new AbortController()
  let checks = 0
  return {
    signal: controller.signal,
    throwIfAborted() {
      checks += 1
      if (abortAt !== null && checks >= abortAt) {
        throw new SmolLM3LoadSmokeError('load_smoke_resource_abort')
      }
    },
    async stop() { return { observed: true } },
    summary() {
      return { samples: 2, minimum_available_physical_bytes: 7 * 1024 ** 3,
        peak_child_working_set_bytes: engine === 'camelid' ? 800_000_000 : 3_500_000_000,
        thresholds_tripped: false }
    },
  }
}

function responses({ llamaTokens = generatedTokens, llamaText = detokenized,
  camelidVersion = CAMELID_RELEASE_VERSION, camelidBuild = sourceDescribe } = {}) {
  return [
    health(false, false, camelidVersion, camelidBuild), gpu, load, verify,
    health(true, false, camelidVersion, camelidBuild), props,
    { prompt: renderedPrompt },
    { tokens: promptTokens },
    chatBody(),
    { content: detokenized },
    { prompt: renderedPrompt },
    health(true, true, camelidVersion, camelidBuild), gpu,
    { status: 'ok' },
    { tokens: promptTokens },
    { tokens: llamaTokens, content: llamaText, tokens_predicted: 4 },
    { content: llamaText },
    { status: 'ok' },
  ]
}

function syntheticDeps(options = {}) {
  const starts = []
  const terminated = []
  const calls = []
  const portChecks = []
  const handles = []
  const queue = responses(options)
  const lock = fakeLock({ deferLossClose: options.deferLockLossClose === true })
  const preflight = fakePreflight()
  const deps = {
    inheritedEnv: hostileInheritedEnv,
    preflightImpl: async () => preflight,
    acquireArtifactLockImpl: async () => lock,
    inspectArtifactIdentityImpl: async (_path, phase) => {
      calls.push({ kind: phase })
      return { size_bytes: EXACT_ROW.source.size_bytes, sha256: EXACT_ROW.source.sha256 }
    },
    startProcessImpl({ engine, binary: executable, args, cwd: childCwd, env }) {
      assert.equal(handles.some((handle) => !handle.isClosed()), false,
        'engine children must never overlap')
      const handle = fakeHandle(100 + handles.length)
      if (options.malformedHandleEngine === engine) delete handle.logMarkers
      handles.push(handle)
      starts.push({ engine, executable, args, childCwd, env })
      return handle
    },
    createResourceGuardImpl: (handle, engine) => {
      if (options.guardSetupFailureEngine === engine) throw new Error('guard setup failed')
      return fakeGuard(engine, options.resourceAbortEngine === engine ? 1 : null)
    },
    terminateChildImpl: async (handle, engine) => {
      terminated.push(engine)
      if (options.terminationFailureEngine === engine) {
        return { observed: false, already_exited: false, termination_requested: false }
      }
      if (options.lyingTerminationEngine === engine) {
        handle.closeOnly()
        return { observed: true, already_exited: false, termination_requested: true }
      }
      handle.terminate()
      return { observed: true, already_exited: false, termination_requested: true,
        status: handle.exitStatus() }
    },
    assertPortFreeImpl: async (address) => { portChecks.push(address) },
    httpJsonImpl: async (request) => {
      const index = calls.filter((call) => call.kind === 'http').length
      calls.push({ kind: 'http', engine: request.origin.includes('8297') ? 'camelid' : 'llama_cpp',
        method: request.method, endpoint: request.endpoint, body: clone(request.body) })
      if (options.exitAtHttpIndex === index) handles.at(-1).exitEarly()
      if (options.lockLossAtHttpIndex === index) lock.lose()
      if (options.httpFailureAt === index) throw new Error('synthetic HTTP failure')
      const body = queue.shift()
      assert.notEqual(body, undefined, `unexpected request ${request.endpoint}`)
      return { status: 200, body }
    },
    postflightImpl: async () => ({
      provenance: preflight.provenance,
      auto_select_roots: preflight.auto_select_roots,
      groundings: preflight.groundings,
      llama: preflight.llama,
    }),
    nowMsImpl: (() => { let value = 0; return () => value++ })(),
    nowIsoImpl: () => createdUtc,
    sleepImpl: async () => {},
    yieldImpl: async () => {},
  }
  return { deps, starts, terminated, calls, portChecks, handles, lock }
}

const runOptions = { root, binary, artifact, cwd, modelsDir, llamaServer, binaryProfile: BINARY_PROFILE }
const happySynthetic = syntheticDeps()
const receipt = await runSmolLM3ChatParity(runOptions, happySynthetic.deps)
assert.deepEqual(validateChatParityReceipt(receipt), [])
assert.equal(receipt.schema, RECEIPT_SCHEMA)
assert.equal(receipt.gate, 'parity_preparation')
assert.equal(receipt.comparison.exact_token_and_text_match, true)
assert.equal(receipt.gate_decision.parity_preparation, 'pass')
assert.equal(receipt.gate_decision.bounded_evidence_publishable, true)
assert.equal(receipt.gate_decision.roster_parity_gate, 'blocked_unchanged')
assert.equal(receipt.gate_decision.load_smoke_gate, 'blocked_unchanged')
assert.deepEqual(receipt.gate_decision.authorized_roster_scope, [])
assert.equal(receipt.gate_decision.support_claim, false)
assert.deepEqual(receipt.runtime_contract.camelid.environment,
  describeChildEnvironment(happySynthetic.starts[0].env, SAFE_CAMELID_ENV))
assert.deepEqual(receipt.runtime_contract.llama_cpp.environment,
  describeChildEnvironment(happySynthetic.starts[1].env, SAFE_LLAMA_ENV))
for (const name of ['camelid_baseline_health', 'camelid_loaded_health',
  'camelid_final_health']) {
  const evidence = receipt.steps.find((step) => step.name === name).evidence
  assert.equal(evidence.version, CAMELID_RELEASE_VERSION)
  assert.equal(evidence.build, sourceDescribe)
}
assert.deepEqual(receipt.isolation.lifecycle_events,
  ['camelid_started', 'camelid_closed', 'llama_cpp_started', 'llama_cpp_closed'])
assert.equal(receipt.isolation.max_concurrent_engine_children, 1)
assert.deepEqual(happySynthetic.starts.map((entry) => entry.engine), ['camelid', 'llama_cpp'])
for (const start of happySynthetic.starts) {
  assert.equal(Object.isFrozen(start.env), true, `${start.engine} environment must be immutable`)
  for (const secret of ['HF_TOKEN', 'GH_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']) {
    assert.equal(start.env[secret], undefined, `${start.engine} must not inherit ${secret}`)
  }
}
assert.deepEqual(happySynthetic.terminated, ['camelid', 'llama_cpp'])
assert.equal(happySynthetic.calls.filter((call) => call.kind === 'http').length,
  STEP_CONTRACT.length)
assert.deepEqual(happySynthetic.calls.filter((call) => call.kind === 'http')
  .map(({ engine, method, endpoint }) => [engine, method, endpoint]),
STEP_CONTRACT.map(([, engine, method, endpoint]) => [engine, method, endpoint]))
const httpCalls = happySynthetic.calls.filter((call) => call.kind === 'http')
assert.deepEqual(httpCalls.find((call) => call.endpoint === '/v1/chat/completions').body,
  CHAT_REQUEST)
assert.deepEqual(httpCalls.filter((call) => call.endpoint === '/tokenize').map((call) => call.body), [
  { content: renderedPrompt, add_special: false, parse_special: true },
  { content: renderedPrompt, add_special: false, parse_special: true },
])
assert.deepEqual(httpCalls.find((call) => call.endpoint === '/completion').body, {
  prompt: promptTokens,
  ...LLAMA_COMPLETION_SETTINGS,
})
assert.deepEqual(httpCalls.filter((call) => call.endpoint === '/detokenize').map((call) => call.body), [
  { tokens: generatedTokens }, { tokens: generatedTokens },
])
assert.equal(httpCalls.filter((call) => call.endpoint === '/v1/chat/completions').length, 1)
assert.equal(httpCalls.filter((call) => call.endpoint === '/completion').length, 1)
assert.deepEqual(happySynthetic.calls.filter((call) => call.kind !== 'http').map((call) => call.kind),
  ['prehash', 'posthash'])

for (const [label, options] of [
  ['composite version', { camelidVersion: `camelid ${sourceDescribe}` }],
  ['wrong release scalar', { camelidVersion: '0.6.0' }],
  ['wrong build identity', { camelidBuild: `${sourceDescribe}-drift` }],
]) {
  const invalidHealth = syntheticDeps(options)
  await expectCode(runSmolLM3ChatParity(runOptions, invalidHealth.deps),
    'chat_parity_camelid_contract_invalid')
  assert.deepEqual(invalidHealth.starts.map((entry) => entry.engine), ['camelid'], label)
  assert.deepEqual(invalidHealth.terminated, ['camelid'], label)
  assert.deepEqual(invalidHealth.calls.filter((call) => call.kind === 'http')
    .map((call) => call.endpoint), ['/v1/health'], label)
}

const mismatchSynthetic = syntheticDeps({
  llamaTokens: [128002, 220, 9999, 5678],
  llamaText: '<think> Different ',
})
const mismatch = await runSmolLM3ChatParity(runOptions, mismatchSynthetic.deps)
assert.deepEqual(validateChatParityReceipt(mismatch), [])
assert.equal(mismatch.comparison.generated_token_ids.exact_match, false)
assert.equal(mismatch.comparison.generated_token_ids.first_divergent_index, 2)
assert.equal(mismatch.comparison.canonical_detokenized_text.exact_utf8_match, false)
assert.equal(mismatch.comparison.exact_token_and_text_match, false)
assert.equal(mismatch.gate_decision.parity_preparation, 'mismatch')
assert.equal(mismatch.gate_decision.bounded_evidence_publishable, false)
assert.equal(mismatch.gate_decision.roster_parity_gate, 'blocked_unchanged')

const terminationFailure = syntheticDeps({ terminationFailureEngine: 'camelid' })
await expectCode(runSmolLM3ChatParity(runOptions, terminationFailure.deps),
  'chat_parity_termination_failed')
assert.deepEqual(terminationFailure.starts.map((entry) => entry.engine), ['camelid'])
assert.equal(terminationFailure.calls.some((call) => call.engine === 'llama_cpp'), false)

const lyingTermination = syntheticDeps({ lyingTerminationEngine: 'camelid' })
await expectCode(runSmolLM3ChatParity(runOptions, lyingTermination.deps),
  'chat_parity_termination_failed')
assert.deepEqual(lyingTermination.starts.map((entry) => entry.engine), ['camelid'])
assert.equal(lyingTermination.handles[0].isExited(), false)
assert.equal(lyingTermination.handles[0].isClosed(), true)
assert.equal(lyingTermination.calls.some((call) => call.engine === 'llama_cpp'), false,
  'a close-only terminator must never authorize the second engine')

const httpFailure = syntheticDeps({ httpFailureAt: 8 })
await expectCode(runSmolLM3ChatParity(runOptions, httpFailure.deps), 'chat_parity_http_failed')
assert.deepEqual(httpFailure.terminated, ['camelid'])
assert.deepEqual(httpFailure.starts.map((entry) => entry.engine), ['camelid'])

const resourceAbort = syntheticDeps({ resourceAbortEngine: 'camelid' })
await expectCode(runSmolLM3ChatParity(runOptions, resourceAbort.deps),
  'chat_parity_resource_abort')
assert.deepEqual(resourceAbort.starts.map((entry) => entry.engine), ['camelid'])

const guardSetupFailure = syntheticDeps({ guardSetupFailureEngine: 'camelid' })
await expectCode(runSmolLM3ChatParity(runOptions, guardSetupFailure.deps),
  'chat_parity_resource_telemetry_unavailable')
assert.deepEqual(guardSetupFailure.terminated, ['camelid'],
  'guard setup failure must still terminate and drain the spawned child')
assert.equal(guardSetupFailure.handles[0].isClosed(), true)

const malformedHandle = syntheticDeps({ malformedHandleEngine: 'camelid' })
await expectCode(runSmolLM3ChatParity(runOptions, malformedHandle.deps),
  'chat_parity_process_start_failed')
assert.deepEqual(malformedHandle.terminated, ['camelid'],
  'a returned malformed child handle must still be terminated and drained')
assert.equal(malformedHandle.handles[0].isClosed(), true)
assert.equal(malformedHandle.lock.isExited(), true,
  'artifact lock must release only after malformed-child cleanup')

const malformedLockRun = syntheticDeps()
const malformedLock = fakeLock()
delete malformedLock.closed
let malformedLockReleases = 0
const releaseMalformedLock = malformedLock.release
malformedLock.release = async () => {
  malformedLockReleases += 1
  return releaseMalformedLock.call(malformedLock)
}
malformedLockRun.deps.acquireArtifactLockImpl = async () => malformedLock
await expectCode(runSmolLM3ChatParity(runOptions, malformedLockRun.deps),
  'chat_parity_artifact_lock_failed')
assert.equal(malformedLockReleases, 1,
  'an acquired malformed artifact-lock handle must be released')

const earlyExit = syntheticDeps({ exitAtHttpIndex: 8 })
await expectCode(runSmolLM3ChatParity(runOptions, earlyExit.deps),
  'chat_parity_process_exited')
assert.deepEqual(earlyExit.starts.map((entry) => entry.engine), ['camelid'])

const lockLoss = syntheticDeps({ lockLossAtHttpIndex: 8, deferLockLossClose: true })
const lockLossRun = runSmolLM3ChatParity(runOptions, lockLoss.deps)
setTimeout(() => lockLoss.lock.closeLost(), 10)
await expectCode(lockLossRun,
  'chat_parity_artifact_lock_lost')
assert.deepEqual(lockLoss.terminated, ['camelid'])
assert.equal(lockLoss.lock.lossCloseReleased(), true,
  'unexpected lock exit must await the helper close/drain boundary')
assert.deepEqual(lockLoss.lock.exitStatus(), { error: false, code: 7, signal: null })

const tamperCases = [
  ['seal', (value) => { value.created_utc = '2026-08-10T20:00:01.000Z' }, 'receipt_id'],
  ['unknown top field', (value) => { value.extra = true }, 'keys must be exact'],
  ['unknown evidence field', (value) => { value.steps[8].evidence.extra = true },
    'keys must be exact'],
  ['schema', (value) => { value.schema = 'other' }, 'schema'],
  ['row source', (value) => { value.row.source.sha256 = '0'.repeat(64) }, 'row'],
  ['shape pack', (value) => { value.grounding.files.shape_pack.sha256 = '0'.repeat(64) },
    'grounding'],
  ['renderer blob', (value) => { value.grounding.renderer_git_blob_sha1 = '0'.repeat(40) },
    'grounding'],
  ['llama executable', (value) => { value.provenance.llama_cpp.executable_sha256 = '0'.repeat(64) },
    'provenance'],
  ['llama archive', (value) => { value.provenance.llama_cpp.archive_sha256 = '0'.repeat(64) },
    'provenance'],
  ['artifact', (value) => { value.provenance.artifact.sha256 = '0'.repeat(64) },
    'artifact'],
  ['lock scope', (value) => { value.provenance.artifact.mutation_guard.held_through_llama_cpp = false },
    'artifact'],
  ['engine overlap', (value) => { value.isolation.max_concurrent_engine_children = 2 },
    'lifetimes'],
  ['engine order', (value) => { value.isolation.lifecycle_events.reverse() }, 'lifetimes'],
  ['warmup', (value) => { value.isolation.startup_warmup_markers.llama_cpp.warming_up_seen = true },
    'lifetimes'],
  ['health release version', (value) => { value.steps[0].evidence.version = '0.6.0' },
    'health'],
  ['unknown warmup container field', (value) => {
    value.isolation.startup_warmup_markers.extra = true
  }, 'keys must be exact'],
  ['chat not first', (value) => { value.isolation.camelid_chat_first_and_only_forward = false },
    'lifetimes'],
  ['unknown Camelid contract field', (value) => { value.runtime_contract.camelid.extra = true },
    'keys must be exact'],
  ['unknown llama contract field', (value) => { value.runtime_contract.llama_cpp.extra = true },
    'keys must be exact'],
  ['cross-engine inherited environment', (value) => {
    value.runtime_contract.llama_cpp.environment.inherited_os_environment_sha256 = '0'.repeat(64)
  }, 'same inherited OS environment'],
  ['llama inherited environment key', (value) => {
    value.runtime_contract.llama_cpp.environment.inherited_os_keys_present.push('WINDIR')
  }, 'environment contract'],
  ['unlisted environment key flag', (value) => {
    value.runtime_contract.camelid.environment.unlisted_keys_present = true
  }, 'environment contract'],
  ['llama environment override', (value) => {
    value.runtime_contract.llama_cpp.environment.model_overrides.CUDA_VISIBLE_DEVICES = '0'
  }, 'environment contract'],
  ['llama request depth', (value) => { value.runtime_contract.llama_cpp.completion.n_predict = 1 },
    'runtime'],
  ['prompt byte count', (value) => { value.runtime_contract.prompt.actual_utf8_bytes += 1 },
    'prompt'],
  ['apply-before byte count', (value) => { value.steps[6].evidence.actual_prompt_utf8_bytes += 1 },
    'rendered prompt'],
  ['apply-after byte count', (value) => { value.steps[10].evidence.actual_prompt_utf8_bytes += 1 },
    'rendered prompt'],
  ['prompt hash', (value) => { value.runtime_contract.prompt.token_ids_sha256 = '0'.repeat(64) },
    'prompt'],
  ['prompt IDs', (value) => { value.steps[8].evidence.prompt_token_ids[0] += 1 },
    'prompt'],
  ['Camelid EOG', (value) => { value.steps[8].evidence.generated_token_ids[3] = EOG_TOKEN_IDS[0] },
    'first forward'],
  ['llama EOG', (value) => { value.steps[15].evidence.generated_token_ids[3] = EOG_TOKEN_IDS[0] },
    'four-token'],
  ['Camelid bubble surface', (value) => { value.steps[8].evidence.bubble.utf8_bytes += 1 },
    'bubble'],
  ['Camelid trimmed detokenize surface', (value) => {
    value.steps[9].evidence.trimmed_content.sha256 = '0'.repeat(64)
  }, 'bubble'],
  ['llama completion content bytes', (value) => { value.steps[15].evidence.content.utf8_bytes += 1 },
    'completion content'],
  ['llama completion content hash', (value) => {
    value.steps[15].evidence.content.sha256 = '0'.repeat(64)
  }, 'completion content'],
  ['token match lie', (value) => { value.comparison.generated_token_ids.exact_match = false },
    'comparison'],
  ['text match lie', (value) => { value.comparison.canonical_detokenized_text.exact_utf8_match = false },
    'comparison'],
  ['preparation result lie', (value) => { value.gate_decision.parity_preparation = 'mismatch' },
    'decision'],
  ['roster promotion', (value) => { value.gate_decision.roster_parity_gate = 'pass' },
    'decision'],
  ['scope expansion', (value) => { value.gate_decision.authorized_roster_scope = ['gates.parity'] },
    'decision'],
  ['support', (value) => { value.gate_decision.support_claim = true }, 'decision'],
  ['disposition', (value) => { value.gate_decision.disposition = 'promotion_candidate' },
    'decision'],
  ['resource threshold', (value) => { value.resource_observations.thresholds_tripped = true },
    'resource'],
  ['1/5/50 claim removed', (value) => { value.does_not_prove.shift() }, 'scope'],
  ['absolute path', (value) => {
    value.provenance.camelid_binary.profile = ['C:', 'Users', 'private', 'x'].join('\\')
  },
    'absolute local path'],
  ['credential', (value) => {
    value.provenance.camelid_binary.profile = ['ghp', '1234567890abcdefghijklmnop'].join('_')
  },
    'credential'],
  ['hostname', (value) => { value.provenance.camelid_binary.profile = 'private.internal' },
    'hostname'],
]
for (const [name, mutate, pattern] of tamperCases) {
  const value = clone(receipt)
  mutate(value)
  const candidate = name === 'seal' ? value : reseal(value)
  const errors = validateChatParityReceipt(candidate)
  assert.ok(errors.length > 0, `${name} tamper should fail`)
  assert.match(errors.join(' | '), new RegExp(pattern, 'i'), `${name} tamper diagnostic`)
}

for (const primitive of [null, undefined, true, false, 0, 1, '', 'receipt', Symbol('x'), 1n]) {
  assert.doesNotThrow(() => validateChatParityReceipt(primitive))
  assert.ok(validateChatParityReceipt(primitive).length > 0)
}
const cyclic = clone(receipt)
cyclic.loop = cyclic
assert.doesNotThrow(() => validateChatParityReceipt(cyclic))
assert.match(validateChatParityReceipt(cyclic).join(' '), /cycle|repeated/i)
const sparse = []
sparse[2] = 'x'
const sparseReceipt = clone(receipt)
sparseReceipt.does_not_prove = sparse
assert.doesNotThrow(() => validateChatParityReceipt(sparseReceipt))
assert.match(validateChatParityReceipt(sparseReceipt).join(' '), /sparse|invalid array/i)
let getterCalls = 0
const getterReceipt = clone(receipt)
Object.defineProperty(getterReceipt, 'schema', {
  enumerable: true,
  get() { getterCalls += 1; return RECEIPT_SCHEMA },
})
assert.doesNotThrow(() => validateChatParityReceipt(getterReceipt))
assert.equal(getterCalls, 0)
assert.match(validateChatParityReceipt(getterReceipt).join(' '), /accessor/i)
let transparentTraps = 0
const transparentProxy = new Proxy({ value: true }, {
  get(target, key, receiver) { transparentTraps += 1; return Reflect.get(target, key, receiver) },
  ownKeys(target) { transparentTraps += 1; return Reflect.ownKeys(target) },
})
const proxyReceipt = clone(receipt)
proxyReceipt.grounding.shape_case = transparentProxy
assert.doesNotThrow(() => validateChatParityReceipt(proxyReceipt))
assert.equal(transparentTraps, 0)
assert.match(validateChatParityReceipt(proxyReceipt).join(' '), /Proxy/i)
let throwingTraps = 0
const throwingProxy = new Proxy({}, {
  ownKeys() { throwingTraps += 1; throw new Error('trap') },
  get() { throwingTraps += 1; throw new Error('trap') },
})
assert.doesNotThrow(() => validateChatParityReceipt(throwingProxy))
assert.equal(throwingTraps, 0)
assert.match(validateChatParityReceipt(throwingProxy).join(' '), /Proxy/i)

const allowedEndpoints = new Set(['/health'])
await assert.rejects(httpJson({ origin: 'http://127.0.0.1:1', allowedEndpoints,
  method: 'GET', endpoint: '/private', timeoutMs: 1, fetchImpl: async () => ({}) }),
SmolLM3ChatParityError)
const parsedHttp = await httpJson({
  origin: 'http://127.0.0.1:1', allowedEndpoints, method: 'GET', endpoint: '/health',
  timeoutMs: 100,
  fetchImpl: async (url, init) => {
    assert.equal(url, 'http://127.0.0.1:1/health')
    assert.equal(init.method, 'GET')
    return { status: 200, text: async () => '{"status":"ok"}' }
  },
})
assert.deepEqual(parsedHttp, { status: 200, body: { status: 'ok' } })

let streamReadIndex = 0
const streamChunks = [Buffer.from('{"status":'), Buffer.from('"ok"}')]
const streamedHttp = await httpJson({
  origin: 'http://127.0.0.1:1', allowedEndpoints, method: 'GET', endpoint: '/health',
  timeoutMs: 100,
  fetchImpl: async () => ({
    status: 200,
    body: { getReader: () => ({
      read: async () => streamReadIndex < streamChunks.length
        ? { done: false, value: streamChunks[streamReadIndex++] }
        : { done: true },
      releaseLock() {},
    }) },
  }),
})
assert.deepEqual(streamedHttp, { status: 200, body: { status: 'ok' } })

let overflowReads = 0
let overflowCancelled = 0
const overflowChunks = [Buffer.alloc(8 * 1024 ** 2), Buffer.alloc(8 * 1024 ** 2 + 1)]
await expectCode(httpJson({
  origin: 'http://127.0.0.1:1', allowedEndpoints, method: 'GET', endpoint: '/health',
  timeoutMs: 100,
  fetchImpl: async () => ({
    status: 200,
    body: { getReader: () => ({
      read: async () => overflowReads < overflowChunks.length
        ? { done: false, value: overflowChunks[overflowReads++] }
        : { done: true },
      async cancel() { overflowCancelled += 1 },
      releaseLock() {},
    }) },
  }),
}), 'chat_parity_http_failed')
assert.equal(overflowReads, 2)
assert.equal(overflowCancelled, 1,
  'stream reader must cancel immediately when the 16 MiB response bound is crossed')

let written
let renamed
await writeReceiptAtomic(resolve('out', 'receipt.json'), receipt, {
  mkdirImpl: async () => {},
  writeFileImpl: async (path, bytes, options) => { written = { path, bytes, options } },
  renameImpl: async (from, to) => { renamed = { from, to } },
  rmImpl: async () => { throw new Error('unexpected cleanup') },
})
assert.match(written.path, /receipt\.json\.tmp-[0-9a-f]{16}$/)
assert.equal(written.options.flag, 'wx')
assert.deepEqual(JSON.parse(written.bytes), receipt)
assert.equal(renamed.from, written.path)
assert.ok(renamed.to.endsWith('receipt.json'))
let removed
await expectCode(writeReceiptAtomic(resolve('out', 'receipt.json'), receipt, {
  mkdirImpl: async () => {},
  writeFileImpl: async () => { throw new Error('disk full') },
  renameImpl: async () => {},
  rmImpl: async (path, options) => { removed = { path, options } },
}), 'chat_parity_output_failed')
assert.match(removed.path, /receipt\.json\.tmp-[0-9a-f]{16}$/)
assert.deepEqual(removed.options, { force: true })

console.log('SmolLM3 bounded chat-parity foundation tests passed')
