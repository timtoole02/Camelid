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
  frontendSupportContractCopy,
  isCompatibilitySupportedForModel,
  rowSupportBoundaryCopy,
  rowSupportNextStepCopy,
} from '../src/lib/capabilities.js'
import { getChatGateState } from '../src/lib/chatGate.js'
import { resolveLoadedModelDisplayName } from '../src/lib/loadedModelDisplay.js'
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
  full_support_blockers: 'the May-era typed lanes (template shapes, perf/RSS, API/WebUI smoke, broader prompt-pack parity) were measured on the prior upload (sha256 b5607b50...) and await re-anchoring to the canonical file; plus model-native/larger context beyond the anchored 512-8192 ladder, broader arbitrary/Jinja templates beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable repeated current-head bundles remain missing',
  frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id',
  chat_template_renderer: 'metadata_jinja_supported_for_exact_row',
  chat_template_shape_pack: 'validated_bounded_pack',
  performance_measured: 'bounded_unique_chat_perf_rss_validated',
  bounded_context_512_pack: 'validated_anchored_raw_decode_ladder',
  bounded_context_512_pack_id: 'llama32-3b-anchored-raw-ladder-v1',
  bounded_context_1024_pack: 'validated_anchored_raw_decode_ladder',
  bounded_context_1024_pack_id: 'llama32-3b-anchored-raw-ladder-v1',
  bounded_context_2048_pack: 'validated_anchored_raw_decode_ladder',
  bounded_context_2048_pack_id: 'llama32-3b-anchored-raw-ladder-v1',
  bounded_context_4096_pack: 'validated_anchored_raw_decode_ladder',
  bounded_context_4096_pack_id: 'llama32-3b-anchored-raw-ladder-v1',
  bounded_context_8192_pack: 'validated_anchored_raw_decode_ladder',
  bounded_context_8192_pack_id: 'llama32-3b-anchored-raw-ladder-v1',
  latest_checked_bucket: 'llama32-3b-anchored-raw-ladder-v1',
  latest_checked_result: 'pass',
  latest_checked_output: '50/50 greedy tokens identical on all five buckets',
  evidence: 'the exact tracked Llama-3.2-3B-Instruct-Q8_0 GGUF (sha256 f34112a1..., the file pinned by the June capability-matrix receipts) has a fresh anchored raw-decode context ladder — 512/1024/2048/4096/8192 buckets (408/885/1893/3978/8049 llama3 tokens), token-AND-text identical to pinned llama.cpp acd79d603 at 50 greedy tokens, fully GPU-resident, at qa/evidence-bundles/llama32-3b-context-512-8192-anchored-20260710T2119-head-6527a770/manifest.json — plus June 2026 capability-matrix receipts on the same file. Earlier evidence (canonical Ubuntu API/WebUI refresh at qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json, compact/broader prompt-pack parity, the May 512/1024/2048 template-context packs, template-shape coverage, and bounded unique-chat perf/RSS) was produced against a PRIOR upload of this model name (sha256 b5607b50...) and is retained as historical evidence for that file; it is disclosed, not inherited. Camelid supports exact-row smoke for this row only, not broader/full support',
  next_step: 'preserve exact-row smoke plus the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder while re-anchoring the remaining May-era evidence families (API/WebUI refresh, template shapes, perf/RSS) to the canonical file and normalizing model-native/larger context, broader arbitrary/Jinja template behavior beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable full-support bundle evidence before any broader/full-support claim',
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
const catalogThreeBHint = findCompatibilityHint(capabilities, null, {
  name: 'Llama 3.2 3B Instruct Q8_0',
  repo_id: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
  filename: 'Llama-3.2-3B-Instruct-Q8_0.gguf',
  quant: 'Q8_0',
})
assert.equal(catalogThreeBHint.target.id, 'llama32_3b_instruct_q8_0', '3B catalog cards must resolve the exact supported row from catalog filename + Q8_0 evidence')
assert.equal(catalogThreeBHint.exact, true, '3B catalog cards must not render family-level support from catalog metadata')
const catalogThreeBWrongArtifactHint = findCompatibilityHint(capabilities, null, {
  name: 'Llama 3.2 3B Instruct Q8_0',
  repo_id: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
  filename: 'Llama-3.2-3B-Instruct-Q8_0-neighbor.gguf',
  quant: 'Q8_0',
})
assert.equal(compatibilityHintLabel(catalogThreeBWrongArtifactHint), 'llama32_3b_instruct_q8_0: exact GGUF not verified', '3B catalog cards must not turn an exact title plus a neighboring GGUF filename into support')
assert.equal(isCompatibilitySupportedForModel(capabilities, null, {
  name: 'Llama 3.2 3B Instruct Q8_0',
  repo_id: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
  filename: 'Llama-3.2-3B-Instruct-Q8_0-neighbor.gguf',
  quant: 'Q8_0',
}), false, '3B catalog support must fail closed when the exact GGUF filename is missing')
assert.equal(isCompatibilitySupportedForModel(capabilities, exactThreeBModel), true, 'supported 3B rows require an exact row plus Q8_0 evidence')
const quantMismatchHint = findCompatibilityHint(capabilities, { ...exactThreeBModel, quant: 'Q4_K_M' })
assert.equal(compatibilityHintLabel(quantMismatchHint), 'llama32_3b_instruct_q8_0: quant mismatch', '3B exact-row surfaces must name quant mismatch instead of falling back to another supported row')
const spoofedThreeBRowIdWrongArtifact = {
  ...exactThreeBModel,
  id: 'llama32_3b_instruct_q8_0',
  name: 'llama32_3b_instruct_q8_0',
  runtime_model_name: 'llama32_3b_instruct_q8_0',
  model_path: '/models/not-Llama-3.2-3B-Instruct-Q8_0.gguf',
  quant: 'Q8_0',
}
const spoofedThreeBHint = findCompatibilityHint(capabilities, spoofedThreeBRowIdWrongArtifact)
assert.equal(compatibilityHintLabel(spoofedThreeBHint), 'llama32_3b_instruct_q8_0: exact GGUF not verified', '3B exact-row support must not unlock from a saved row id without the exact GGUF filename')
assert.equal(isCompatibilitySupportedForModel(capabilities, spoofedThreeBRowIdWrongArtifact), false, '3B row-id spoofing with a neighboring GGUF path must fail closed')
assert.equal(
  getChatGateState(capabilities, spoofedThreeBRowIdWrongArtifact, { ...runtime, active_model_id: 'llama32_3b_instruct_q8_0' }).chatUnlocked,
  false,
  '3B WebUI chat must stay blocked when the active runtime row id is spoofed but artifact identity does not match',
)
const spoofedThreeBNameWrongArtifact = {
  ...exactThreeBModel,
  id: 'local-wrong-artifact',
  name: 'Llama 3.2 3B Instruct Q8_0',
  runtime_model_name: 'local-wrong-artifact',
  model_path: '/models/Llama-3.2-3B-Instruct-Q8_0-neighbor.gguf',
  quant: 'Q8_0',
}
assert.equal(compatibilityHintLabel(findCompatibilityHint(capabilities, spoofedThreeBNameWrongArtifact)), 'llama32_3b_instruct_q8_0: exact GGUF not verified', '3B model-size labels still need the exact GGUF artifact identity')
const spoofedThreeBSourceWrongArtifact = {
  ...spoofedThreeBNameWrongArtifact,
  id: 'local-wrong-artifact-with-source',
  runtime_model_name: 'local-wrong-artifact-with-source',
  source: 'bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf',
}
assert.equal(compatibilityHintLabel(findCompatibilityHint(capabilities, spoofedThreeBSourceWrongArtifact)), 'llama32_3b_instruct_q8_0: exact GGUF not verified', '3B source metadata must not override the actual local GGUF filename used by runtime gating')
assert.equal(isCompatibilitySupportedForModel(capabilities, spoofedThreeBSourceWrongArtifact), false, '3B support must fail closed when source metadata names the exact GGUF but model_path names a neighboring artifact')
assert.equal(compatibilityHintMatchesExactTarget(capabilities, exactThreeBModel, llama32ThreeBTarget), true, 'ModelsView exact-row matching must accept the canonical 3B row')
assert.equal(modelRuntimeIdMatches(exactThreeBModel, runtime), true, '3B backend active_model_id must match the selected runtime row')
assert.equal(isRunnableInCurrentRuntime(exactThreeBModel, runtime), true, '3B runtime readiness must require the active backend row and generation_ready=true')
assert.equal(getRuntimeRequestModelId(exactThreeBModel, runtime, 'fallback'), 'scalar_default_rerun', 'API/chat requests should use the loaded backend model id for alias-safe 3B sends')
assert.equal(
  resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: exactThreeBModel.model_path, quantLabel: 'Q8_0' }),
  LLAMA32_3B_ACCEPTANCE_TARGET.name,
  'loaded backend aliases should render as the canonical 3B row only when the exact GGUF filename and decoded Q8_0 file_type evidence match',
)
assert.equal(
  resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: exactThreeBModel.model_path, quantLabel: 'file_type 7' }),
  LLAMA32_3B_ACCEPTANCE_TARGET.name,
  'loaded backend aliases should also accept direct GGUF file_type 7 quant evidence for the exact 3B Q8_0 row',
)
assert.equal(
  resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: exactThreeBModel.model_path, quantLabel: 'general.file_type: 7' }),
  LLAMA32_3B_ACCEPTANCE_TARGET.name,
  'loaded backend aliases should accept metadata-shaped GGUF general.file_type: 7 quant evidence for the exact 3B Q8_0 row',
)
assert.equal(
  isCompatibilitySupportedForModel(capabilities, { ...exactThreeBModel, quant: 'general.file_type=7' }),
  true,
  '3B exact-row support should accept metadata-shaped GGUF general.file_type=7 quant evidence without weakening the artifact gate',
)
assert.equal(
  resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: exactThreeBModel.model_path, quantLabel: 'Q4_K_M' }),
  'scalar_default_rerun',
  'loaded backend aliases must not render as the 3B supported row when quant evidence is not Q8_0',
)
assert.equal(
  resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: '/models/llama32-3b-instruct-q8-neighbor.gguf', quantLabel: 'Q8_0' }),
  'scalar_default_rerun',
  'loaded backend aliases must not render as the 3B supported row from a loose neighboring filename plus Q8_0 label',
)

const exactGate = getChatGateState(capabilities, exactThreeBModel, runtime)
assert.deepEqual(
  [exactGate.runtimeLoaded, exactGate.runtimeGenerationReady, exactGate.runtimeReady, exactGate.contractSupported, exactGate.chatUnlocked],
  [true, true, true, true, true],
  '3B WebUI chat unlock is retained only when loaded_now, generation_ready, active_model_id, and exact supported row all pass',
)

const missingCapabilitiesGate = getChatGateState(null, exactThreeBModel, runtime)
assert.equal(missingCapabilitiesGate.runtimeReady, true, '3B runtime readiness must remain visible when /api/capabilities is unavailable')
assert.equal(missingCapabilitiesGate.contractSupported, false, '3B support must fail closed when /api/capabilities is unavailable')
assert.equal(missingCapabilitiesGate.chatUnlocked, false, '3B WebUI chat must not unlock from runtime health alone without the exact capabilities row')
assert.equal(missingCapabilitiesGate.label, 'No matching COMPATIBILITY.md row', '3B support-gated copy should name the missing exact row when capabilities are absent')

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
assert.deepEqual(lanes.map((lane) => [lane.key, lane.ready]), [['template', true], ['context', true], ['throughput', false]], '3B template/Jinja and checked-context readiness are row-green while production throughput remains unpromoted')
const contextLane = lanes.find((lane) => lane.key === 'context')
assert.match(contextLane.copy, /512 context validated anchored raw decode ladder, 1024 context validated anchored raw decode ladder, 2048 context validated anchored raw decode ladder, 4096 context validated anchored raw decode ladder, 8192 context validated anchored raw decode ladder/, '3B checked-context copy must name the anchored raw-decode ladder on all five checked buckets, including the newly promoted 4096/8192')
assert.match(contextLane.copy, /latest checked llama32-3b-anchored-raw-ladder-v1/, '3B checked-context copy must cite the anchored-ladder pack id as the latest checked bucket')
assert.match(contextLane.copy, /does not promote model-native\/larger context beyond the checked packs/, '3B checked-context copy must keep the larger-context boundary visible')
const genericThroughputCapabilities = {
  ...capabilities,
  api_features: [
    {
      id: 'production_throughput_api',
      status: 'supported_current_gate',
      notes: 'generic API throughput reporting exists',
    },
  ],
}
const lanesWithGenericThroughput = exactRowSupportLanes(llama32ThreeBTarget, genericThroughputCapabilities.api_features)
const throughputLaneWithGenericFeature = lanesWithGenericThroughput.find((lane) => lane.key === 'throughput')
assert.equal(throughputLaneWithGenericFeature.ready, false, '3B production-throughput readiness must stay row-owned and not inherit a generic API feature')
assert.match(throughputLaneWithGenericFeature.copy, /generic API feature supported current gate does not widen row support/, '3B throughput copy must explain why generic API features do not promote exact-row throughput')
assert.doesNotMatch(rowSupportBoundaryCopy(llama32ThreeBTarget, capabilities.api_features), /arbitrary|Jinja/i, '3B boundary copy should not repeat resolved row-scoped metadata-Jinja caveats')
assert.match(rowSupportBoundaryCopy(llama32ThreeBTarget, capabilities.api_features), /production|throughput/i, '3B boundary copy must keep production-throughput caveats visible')
assert.doesNotMatch(rowSupportNextStepCopy(llama32ThreeBTarget, capabilities.api_features), /arbitrary|Jinja/i, '3B next-step copy should not repeat resolved template/Jinja caveats')
assert.match(rowSupportNextStepCopy(llama32ThreeBTarget, capabilities.api_features), /production|throughput/i, '3B next-step copy must keep production-throughput caveats visible')
assert.doesNotMatch(frontendSupportContractCopy(capabilities), /arbitrary|Jinja/i, '3B support-contract copy should remove resolved row-scoped template/Jinja caveats when every supported exact row has row evidence')
assert.match(frontendSupportContractCopy(capabilities), /production|throughput/i, '3B support-contract copy must keep production-throughput unpromoted until the exact row reports production-throughput evidence')

assert.equal(LLAMA32_3B_ACCEPTANCE_TARGET.id, 'llama-3.2-3b-instruct-q8')
assert.match(LLAMA32_3B_ACCEPTANCE_TARGET.model_path, /Llama-3\.2-3B-Instruct-Q8_0\.gguf$/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /canonical Ubuntu API\/WebUI support-gate refresh/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /production throughput remains unpromoted/)
assert.match(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /does not currently show the exact 3B row/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /loaded_now=true and generation_ready=true/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /exact supported Llama 3\.2 3B Q8_0 compatibility row/)

const hookSource = readFileSync(new URL('../src/hooks/useDashboardData.js', import.meta.url), 'utf8')
const loadedModelDisplaySource = readFileSync(new URL('../src/lib/loadedModelDisplay.js', import.meta.url), 'utf8')
const chatSource = readFileSync(new URL('../src/views/ChatWorkspace.jsx', import.meta.url), 'utf8')
const modelsSource = readFileSync(new URL('../src/views/ModelsView.jsx', import.meta.url), 'utf8')
const apiSource = readFileSync(new URL('../src/views/ApiView.jsx', import.meta.url), 'utf8')
const systemSource = readFileSync(new URL('../src/views/SystemView.jsx', import.meta.url), 'utf8')
const topBarSource = readFileSync(new URL('../src/components/TopBar.jsx', import.meta.url), 'utf8')

assert.match(hookSource, /selectedModelChatGate\s*=\s*getChatGateState\(dashboard\?\.capabilities, selectedModel, runtime\)/, 'dashboard selectedModelRunnable must be derived from the shared exact-row chat gate')
assert.match(hookSource, /selectedModelRunnable\s*=\s*selectedModelChatGate\.chatUnlocked/, 'dashboard must pass chatUnlocked, not runtime readiness alone, into the composer')
assert.match(hookSource, /activeModelChatGate\s*=\s*activeModel \? getChatGateState\(capabilities, activeModel, nextDashboard\.runtime\) : null/, 'dashboard active-model selection must use the shared exact-row chat gate')
assert.match(hookSource, /chatUnlockedModel\s*=\s*nextModels\.find\(\(model\) => getChatGateState\(capabilities, model, nextDashboard\.runtime\)\.chatUnlocked\)/, 'dashboard model fallback must prefer exact-row chat-unlocked models instead of runtime-only readiness')
assert.doesNotMatch(hookSource, /activeModelRunnable|currentModelRunnable|nextModels\.find\(\(model\) => isRunnableModel\(model\)\)/, 'dashboard selection must not use runtime-only model readiness as chat readiness')
assert.match(hookSource, /const quantLabel = active \? getLoadedModelQuantLabel\(currentModel\) : record\.quant[\s\S]*const modelPath = active \? getModelPath\(currentModel\) \|\| record\.model_path : record\.model_path[\s\S]*name: resolveLoadedModelDisplayName\(\{ fallbackName: record\.name, modelPath, quantLabel \}\)/, 'dashboard local-record merge must preserve /api/models/current 3B GGUF filename plus decoded Q8_0 file_type evidence before canonical display/gating')
assert.match(hookSource, /const quantLabel = active \? getLoadedModelQuantLabel\(currentModel\) : null[\s\S]*const modelPath = active \? getModelPath\(currentModel\) \|\| localRecord\?\.model_path \|\| '' : localRecord\?\.model_path \|\| ''[\s\S]*name: resolveLoadedModelDisplayName\(\{ fallbackName, modelPath, quantLabel \}\)/, 'dashboard backend-row merge must preserve exact 3B loaded-model path and quant metadata even when /v1/models exposes a run-label id')
assert.match(loadedModelDisplaySource, /ggufFileTypeValueFromLabel[\s\S]*quantLabelFromGgufFileType[\s\S]*LLAMA32_3B_ACCEPTANCE_FILENAME[\s\S]*normalizeQuantLabel\(quantLabel\) === 'Q8_0'/, 'backend 3B display aliasing must stay exact-filename plus decoded Q8_0/file_type 7 gated')
assert.match(hookSource, /resolveLoadedModelDisplayName/, 'dashboard model merge must use the shared exact-filename plus Q8_0 loaded-model display gate')
assert.match(chatSource, /runnableModels\s*=\s*models\.filter\(\(model\) => getChatGateState\(capabilities, model, runtime\)\.chatUnlocked\)/, 'chat model picker must list only exact-row unlocked models')
assert.match(chatSource, /canSubmit\s*=\s*Boolean\(composer\.trim\(\)\) && canChat && !generationActive/, 'composer send button must be blocked unless the chat gate (supported exact row or experimental lane) unlocked')
assert.match(chatSource, /runtimeStatusCopy[\s\S]*loaded now and generation_ready=true/, 'chat readiness copy must name the runtime readiness requirement')
assert.match(chatSource, /supportStatusCopy[\s\S]*COMPATIBILITY\.md and \/api\/capabilities agree/, 'chat readiness copy must name the support-contract requirement')
assert.match(chatSource, /selectedRuntimeReady\s*=\s*selectedChatGate\.runtimeReady/, 'live 3B chat readiness must use the shared exact-row gate instead of stale browser runtime fields')
assert.match(chatSource, /selectedModelCapabilitySupported\s*=\s*selectedChatGate\.contractSupported/, 'live 3B support readiness must use the shared exact-row gate contract state')
assert.doesNotMatch(chatSource, /selectedChatGate\.runtimeReady\s*\|\|\s*isRunnableInCurrentRuntime/, 'live 3B chat must not bypass runtime loaded_now through the older runnable helper')
assert.doesNotMatch(chatSource, /isCompatibilitySupportedForModel\(capabilities, selectedModel\)/, 'live 3B support readiness must not re-check support outside the shared chat gate')
assert.match(chatSource, /supportStatusLabel\s*=\s*selectedModelCapabilitySupported[\s\S]*\?\s*selectedCompatibilityLabel/, 'supported live chat readiness must name the exact /api/capabilities row instead of a generic green label')
assert.match(chatSource, /supportStatusCopy\s*=\s*selectedModelCapabilitySupported[\s\S]*`\$\{selectedCompatibilityLabel\}\. COMPATIBILITY\.md and \/api\/capabilities agree/, 'supported live chat readiness copy must preserve the exact row id after the gate turns green')
// Redesign (2026-06): the six overlapping readiness surfaces were consolidated into one
// status line in the docked composer. Runtime + exact-row support readiness still render
// together (honesty preserved); row-scoped capability-lane copy now lives in the System/API
// views (still asserted there). Behavioral intent — never show ready when gated — is unchanged.
assert.match(chatSource, /cxcomposer__status[\s\S]*runtimeStatusLabel[\s\S]*supportStatusLabel/, 'redesigned chat must keep runtime + exact-row support readiness in the consolidated status surface')
assert.match(chatSource, /selectedCompatibilityLabel/, 'redesigned chat must surface the exact-row compatibility label, not a generic green badge')
assert.match(chatSource, /StreamingLoader/, 'live 3B chat must keep an accessible pre-token loader')
assert.equal((chatSource.match(/aria-label="Message Camelid"/g) || []).length, 1, 'redesigned chat docks a single shared composer textarea')
// Redesign (2026-07, D14): the Models page was rebuilt as five derived zones. The 3B
// acceptance panel, tracked-row cards, and legacy catalog cards were deleted; exact-row
// honesty now flows through lib/modelLanes (laneOf → shared capability matcher, including
// the exact-artifact and quant gates) and the inspect-first fail-closed load flow.
// Row-scoped capability-lane copy stays asserted on the System/API views below.
const modelLanesSource = readFileSync(new URL('../src/lib/modelLanes.js', import.meta.url), 'utf8')
const catalogSource = readFileSync(new URL('../src/components/models/CatalogLaneBrowse.jsx', import.meta.url), 'utf8')
const modelActivationSource = readFileSync(new URL('../src/lib/modelActivation.js', import.meta.url), 'utf8')
assert.match(modelLanesSource, /isCompatibilitySupportedForModel\(capabilities, matchModel\(entry\)\)/, 'Models lane derivation must ask the shared contract matcher — the supported gate stays the contract voice')
assert.match(modelsSource, /bucketByLane\(spine\.local\.models, capabilities\)/, 'ModelsView section membership must be derived from the live scan + capabilities at render time')
assert.doesNotMatch(modelsSource, /SUPPORTED_MODELS/, 'ModelsView must not place models from a hand-authored array')
/* Inspect-first fail-closed loading now lives in lib/modelActivation.js, shared by
   the Models page and the first-run activation card; the invariant is unchanged and
   is asserted where it is defined. Both callers must route through it. */
assert.match(modelActivationSource, /api\/models\/inspect[\s\S]*blocker[\s\S]*return[\s\S]*api\/models\/load/, 'the shared load flow must inspect first and stop on typed blockers before any load attempt')
assert.match(modelsSource, /loadLocalModelForChat\(/, 'ModelsView must load through the shared activation protocol')
assert.doesNotMatch(modelsSource, /api\/models\/load/, 'ModelsView must not hand-roll a second load path')
assert.doesNotMatch(modelsSource, /runtimeReady\s*=\s*isRunnableModel\(model\)/, 'ModelsView must not label runtime readiness from stale model-only readiness')
assert.match(catalogSource, /isCompatibilitySupportedForModel\(capabilities, null, item\)/, 'catalog rows must resolve their landing lane through the shared capability matcher')
assert.match(catalogSource, /if \(item\.group === 'experimental'\) return 'not_anchored'/, 'live Hugging Face rows must never anchor a lane or imply support')
assert.match(apiSource, /Selected exact-row evidence/, 'API view must surface selected 3B exact-row evidence')
assert.match(apiSource, /selectedChatGate\s*=\s*getChatGateState\(capabilities, selectedModel, runtime\)/, 'API view must use the shared exact-row chat gate for 3B endpoint readiness')
assert.match(apiSource, /selectedExactRowReady\s*=\s*selectedChatGate\.chatUnlocked/, 'API view must not reimplement 3B endpoint readiness separately from Chat/System')
assert.match(apiSource, /selectedExactRowReady/, 'API view endpoint readiness must use selected exact-row readiness, not broad family evidence')
assert.match(apiSource, /selectedCompatibilityTarget\.frontend_readiness_gate/, 'API view must render the 3B frontend readiness gate from /api/capabilities')
assert.match(systemSource, /selectedChatGate\s*=\s*getChatGateState\(capabilities, selectedModel, runtime\)/, 'System view must use the shared exact-row chat gate for 3B readiness surfaces')
assert.match(systemSource, /selectedCompatibilityHint\s*=\s*selectedChatGate\.hint \|\| findCompatibilityHint\(capabilities, selectedModel\)/, 'System selected exact-row evidence must stay anchored to the shared chat gate compatibility hint')
assert.match(systemSource, /selectedExactRowReady\s*=\s*selectedChatGate\.chatUnlocked/, 'System view must not promote /v1 chat readiness from generation_ready alone')
assert.match(systemSource, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'System curl copy must stay blocked until 3B exact-row support and runtime readiness both match')
assert.match(systemSource, /Endpoint\/chat gate:/, 'System selected 3B evidence must show the retained endpoint/chat readiness gate')
assert.match(systemSource, /rowSupportNextStepCopy\(target, apiFeatures\)/, 'System compatibility rows must render filtered exact-row next-step copy instead of raw blocker text')
// Redesign (2026-06): the TopBar support-contract strip was removed; the slim TopBar now shows
// the conversation title + a compact model status chip. It must still derive that chip from the
// shared exact-row chat gate and resolve the active model through runtime_model_name aliases
// (so the chip never mislabels the loaded model). The exact-row hint detail itself now lives in
// the chat composer status line and the System/API views (asserted there).
assert.match(topBarSource, /getChatGateState\(capabilities, selectedModel, runtime\)/, 'TopBar must derive its model status from the shared exact-row chat gate, not a separate support reimplementation')
assert.match(topBarSource, /modelRuntimeIdMatches/, 'TopBar must resolve the active model through runtime_model_name aliases')

console.log('✓ frontend 3B closure smoke passed')
