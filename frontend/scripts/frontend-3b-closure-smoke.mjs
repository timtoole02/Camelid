#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  LLAMA32_3B_ACCEPTANCE_AVAILABILITY,
  LLAMA32_3B_ACCEPTANCE_GATING_NOTE,
  LLAMA32_3B_ACCEPTANCE_SUMMARY,
  LLAMA32_3B_ACCEPTANCE_TARGET,
} from '../src/lib/acceptanceTargets.js'
import {
  compatibilityHintCopy,
  compatibilityHintLabel,
  compatibilityHintMatchesExactTarget,
  exactRowSupportLanes,
  findCompatibilityHint,
  isCompatibilitySupportedForModel,
  rowSupportBoundaryCopy,
  rowSupportNextStepCopy,
} from '../src/lib/capabilities.js'
import { getChatGateState } from '../src/lib/chatGate.js'
import {
  getRuntimeRequestModelId,
  isRunnableInCurrentRuntime,
  modelRuntimeIdMatches,
} from '../src/lib/modelState.js'

const llama32ThreeBTarget = {
  id: 'llama32_3b_instruct_q8_0',
  family: 'llama_bpe_decoder',
  quantization: 'Q8_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_smoke_only',
  full_support_status: 'blocked_pending_normalized_full_support',
  full_support_blockers: 'model-native/larger context beyond checked packs, broader arbitrary/Jinja templates beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable repeated current-head bundles remain missing',
  frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id',
  chat_template_renderer: 'metadata_jinja_supported_for_exact_row',
  chat_template_shape_pack: 'validated_bounded_pack',
  performance_measured: 'bounded_unique_chat_perf_rss_validated',
  bounded_context_512_pack: 'validated_first_pack',
  bounded_context_1024_pack: 'validated_second_pack',
  bounded_context_2048_pack: 'validated_third_pack',
  latest_checked_bucket: 'llama3-context-2048-smoke-v1',
  latest_checked_result: 'pass',
  latest_checked_output: 'CMLD-204',
  evidence: '3B exact-row canonical Ubuntu API/WebUI refresh, load, completion, chat, frontend smoke, compact parity, broader prompt-pack, first 512-context, second 1024-context, third 2048-context, and metadata-Jinja row-template evidence',
  next_step: 'preserve exact-row smoke plus checked 512/1024/2048 context support while normalizing model-native/larger context, broader arbitrary/Jinja template behavior beyond row-scoped metadata-Jinja/template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable full-support bundle evidence before any broader/full-support claim',
}

const capabilities = {
  support_contract: {
    current_gate: 'Current exact-row support: Llama 3.2 3B Instruct Q8_0 is smoke-supported for local chat only when the exact row, Q8_0 quant, loaded_now=true, generation_ready=true, and active_model_id match; no model-native/larger context beyond checked packs, production throughput, portability, neighboring-row, or broad-family support is implied.',
  },
  api_features: [],
  model_compatibility: [
    llama32ThreeBTarget,
    {
      id: 'llama32_1b_instruct_q8_0',
      family: 'llama_bpe_decoder',
      quantization: 'Q8_0',
      status: 'supported_exact_row_smoke',
      frontend_readiness_gate: 'green only for the exact 1B row',
      evidence: '1B row evidence fixture',
    },
    {
      id: 'llama3_8b_instruct_q8_0',
      family: 'llama_bpe_decoder',
      quantization: 'Q8_0',
      status: 'supported_exact_row_smoke',
      frontend_readiness_gate: 'green only for the exact 8B row',
      evidence: '8B row evidence fixture',
    },
  ],
}

const runtime = {
  active_model_id: 'scalar_default_rerun',
  loaded_now: true,
  generation_ready: true,
}

const exactThreeBModel = {
  id: 'scalar_default_rerun',
  name: 'scalar_default_rerun',
  runtime_model_name: 'scalar_default_rerun',
  provider_kind: 'local',
  status: 'ready',
  model_path: '/models/Llama-3.2-3B-Instruct-Q8_0.gguf',
  quant: 'file_type 7',
  loaded_now: true,
  generation_ready: true,
}

const exactHint = findCompatibilityHint(capabilities, exactThreeBModel)
assert.equal(exactHint.target.id, 'llama32_3b_instruct_q8_0', '3B closure must resolve backend run labels through the exact GGUF path plus Q8_0 file_type evidence')
assert.equal(exactHint.exact, true, '3B closure must be an exact compatibility hint, not a family fallback')
assert.equal(compatibilityHintLabel(exactHint), 'llama32_3b_instruct_q8_0: supported exact row smoke')
assert.match(compatibilityHintCopy(exactHint), /runtime generation still requires loaded_now=true and generation_ready=true/)
assert.equal(isCompatibilitySupportedForModel(capabilities, exactThreeBModel), true, 'supported 3B rows require an exact row plus Q8_0 evidence')
const quantMismatchHint = findCompatibilityHint(capabilities, { ...exactThreeBModel, quant: 'Q4_K_M' })
assert.equal(compatibilityHintLabel(quantMismatchHint), 'llama32_3b_instruct_q8_0: quant mismatch', '3B exact-row surfaces must name quant mismatch instead of falling back to another supported row')
assert.equal(compatibilityHintMatchesExactTarget(capabilities, exactThreeBModel, llama32ThreeBTarget), true, 'ModelsView exact-row matching must accept the canonical 3B row')
assert.equal(modelRuntimeIdMatches(exactThreeBModel, runtime), true, '3B backend active_model_id must match the selected runtime row')
assert.equal(isRunnableInCurrentRuntime(exactThreeBModel, runtime), true, '3B runtime readiness must require the active backend row and generation_ready=true')
assert.equal(getRuntimeRequestModelId(exactThreeBModel, runtime, 'fallback'), 'scalar_default_rerun', 'API/chat requests should use the loaded backend model id for alias-safe 3B sends')

const exactGate = getChatGateState(capabilities, exactThreeBModel, runtime)
assert.deepEqual(
  [exactGate.runtimeLoaded, exactGate.runtimeGenerationReady, exactGate.runtimeReady, exactGate.contractSupported, exactGate.chatUnlocked],
  [true, true, true, true, true],
  '3B WebUI chat unlock is retained only when loaded_now, generation_ready, active_model_id, and exact supported row all pass',
)

for (const [label, model, runtimeOverride] of [
  ['loaded_now=false', exactThreeBModel, { ...runtime, loaded_now: false }],
  ['generation_ready=false', exactThreeBModel, { ...runtime, generation_ready: false }],
  ['active_model_id mismatch', exactThreeBModel, { ...runtime, active_model_id: 'other-model' }],
  ['Q4 quant mismatch', { ...exactThreeBModel, quant: 'Q4_K_M' }, runtime],
  ['base/non-instruct model', { ...exactThreeBModel, name: 'Llama 3.2 3B Base Q8_0', model_path: '/models/Llama-3.2-3B-Q8_0.gguf' }, runtime],
]) {
  assert.equal(
    getChatGateState(capabilities, model, runtimeOverride).chatUnlocked,
    false,
    `3B WebUI chat must fail closed for ${label}`,
  )
}

const unsupportedCapabilities = {
  ...capabilities,
  model_compatibility: capabilities.model_compatibility.map((row) => row.id === 'llama32_3b_instruct_q8_0' ? { ...row, status: 'active_validation_unsupported' } : row),
}
const unsupportedGate = getChatGateState(unsupportedCapabilities, exactThreeBModel, runtime)
assert.equal(unsupportedGate.runtimeReady, true, 'runtime readiness remains visible when the row is unsupported')
assert.equal(unsupportedGate.contractSupported, false, 'unsupported 3B row status must not become support')
assert.equal(unsupportedGate.chatUnlocked, false, 'runtime-green 3B still stays blocked if /api/capabilities does not promote the exact row')

const noThreeBRowCapabilities = {
  ...capabilities,
  model_compatibility: capabilities.model_compatibility.filter((row) => row.id !== 'llama32_3b_instruct_q8_0'),
}
assert.equal(findCompatibilityHint(noThreeBRowCapabilities, exactThreeBModel), null, '3B must not inherit 1B/8B support when the exact 3B row is absent')
assert.equal(getChatGateState(noThreeBRowCapabilities, exactThreeBModel, runtime).chatUnlocked, false, '3B WebUI chat must stay blocked without the exact compatibility row')

const lanes = exactRowSupportLanes(llama32ThreeBTarget, capabilities.api_features)
assert.deepEqual(lanes.map((lane) => [lane.key, lane.ready]), [['template', true], ['throughput', false]], '3B template/Jinja readiness is row-green while production throughput remains unpromoted')
assert.doesNotMatch(rowSupportBoundaryCopy(llama32ThreeBTarget, capabilities.api_features), /arbitrary|Jinja/i, '3B boundary copy should not repeat resolved row-scoped metadata-Jinja caveats')
assert.match(rowSupportBoundaryCopy(llama32ThreeBTarget, capabilities.api_features), /production|throughput/i, '3B boundary copy must keep production-throughput caveats visible')
assert.doesNotMatch(rowSupportNextStepCopy(llama32ThreeBTarget, capabilities.api_features), /arbitrary|Jinja/i, '3B next-step copy should not repeat resolved template/Jinja caveats')
assert.match(rowSupportNextStepCopy(llama32ThreeBTarget, capabilities.api_features), /production|throughput/i, '3B next-step copy must keep production-throughput caveats visible')

assert.equal(LLAMA32_3B_ACCEPTANCE_TARGET.id, 'llama-3.2-3b-instruct-q8')
assert.match(LLAMA32_3B_ACCEPTANCE_TARGET.model_path, /Llama-3\.2-3B-Instruct-Q8_0\.gguf$/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /canonical Ubuntu API\/WebUI support-gate refresh/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /production throughput remains unpromoted/)
assert.match(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /does not currently show the exact 3B row/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /loaded_now=true and generation_ready=true/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /exact supported Llama 3\.2 3B Q8_0 compatibility row/)

const hookSource = readFileSync(new URL('../src/hooks/useDashboardData.js', import.meta.url), 'utf8')
const chatSource = readFileSync(new URL('../src/views/ChatWorkspace.jsx', import.meta.url), 'utf8')
const modelsSource = readFileSync(new URL('../src/views/ModelsView.jsx', import.meta.url), 'utf8')
const apiSource = readFileSync(new URL('../src/views/ApiView.jsx', import.meta.url), 'utf8')
const topBarSource = readFileSync(new URL('../src/components/TopBar.jsx', import.meta.url), 'utf8')

assert.match(hookSource, /selectedModelChatGate\s*=\s*getChatGateState\(dashboard\?\.capabilities, selectedModel, runtime\)/, 'dashboard selectedModelRunnable must be derived from the shared exact-row chat gate')
assert.match(hookSource, /selectedModelRunnable\s*=\s*selectedModelChatGate\.chatUnlocked/, 'dashboard must pass chatUnlocked, not runtime readiness alone, into the composer')
assert.match(hookSource, /LLAMA32_3B_ACCEPTANCE_FILENAME[\s\S]*normalizeQuantLabel\(quantLabel\) === 'Q8_0'/, 'backend 3B display aliasing must stay exact-filename plus Q8_0 gated')
assert.match(chatSource, /runnableModels\s*=\s*models\.filter\(\(model\) => getChatGateState\(capabilities, model, runtime\)\.chatUnlocked\)/, 'chat model picker must list only exact-row unlocked models')
assert.match(chatSource, /canSubmit\s*=\s*Boolean\(composer\.trim\(\)\) && selectedModelRunnable && !generationActive/, 'composer send button must be blocked unless the exact-row chat gate unlocked')
assert.match(chatSource, /runtimeStatusCopy[\s\S]*loaded now and generation_ready=true/, 'chat readiness copy must name the runtime readiness requirement')
assert.match(chatSource, /supportStatusCopy[\s\S]*COMPATIBILITY\.md and \/api\/capabilities agree/, 'chat readiness copy must name the support-contract requirement')
assert.match(chatSource, /chat-readiness-strip-live[\s\S]*runtimeStatusLabel[\s\S]*supportStatusLabel[\s\S]*capabilityLaneStatus\.label/, 'non-empty live 3B chat must keep runtime, exact-row support, and row-scoped capability readiness visible after messages exist')
assert.match(chatSource, /getChatCapabilityLaneCopy\(selectedChatGate, capabilities\)/, 'live 3B chat must derive capability lane copy from the shared exact-row chat gate')
assert.match(chatSource, /Row-scoped \/api\/capabilities evidence; it does not widen model-native context/, 'live 3B capability copy must not widen support beyond the exact row')
assert.match(chatSource, /LiveGenerationBadge/, 'live 3B chat must keep an active streaming badge after first content arrives')
assert.match(chatSource, /StreamingLoader/, 'live 3B chat must keep an accessible pre-token loader')
assert.match(modelsSource, /matchesLlama32ThreeBTarget\(model, capabilities\)/, 'ModelsView 3B acceptance target must hide only on exact target match')
assert.match(modelsSource, /Fill import form with exact path/, 'ModelsView must provide the exact 3B import path affordance when the row is absent locally')
assert.match(modelsSource, /Chat unlockable/, 'ModelsView must expose the retained exact-row chat-unlock state')
assert.match(modelsSource, /matchedChatGate\s*=\s*matchedModel \? getChatGateState\(capabilities, matchedModel, runtime\) : null/, 'ModelsView retained 3B row cards must use the shared chat gate for loaded_now and generation_ready checks')
assert.match(apiSource, /Selected exact-row evidence/, 'API view must surface selected 3B exact-row evidence')
assert.match(apiSource, /selectedExactRowReady/, 'API view endpoint readiness must use selected exact-row readiness, not broad family evidence')
assert.match(apiSource, /selectedCompatibilityTarget\.frontend_readiness_gate/, 'API view must render the 3B frontend readiness gate from /api/capabilities')
assert.match(topBarSource, /exactHintDetail\(activeChatGate\.hint\) \|\| exactHintDetail\(selectedChatGate\.hint\)/, 'TopBar support contract detail must prioritize the active/selected exact 3B hint label, including quant-mismatch and quant-missing blockers')
assert.match(topBarSource, /exactTargetFromHint\(activeChatGate\.hint\)[\s\S]*exactTargetFromHint\(selectedChatGate\.hint\)[\s\S]*getCurrentCompatibilityTarget/, 'TopBar support contract detail must fall back to the first current gate row only after active/selected exact-row hints')

console.log('✓ frontend 3B closure smoke passed')
