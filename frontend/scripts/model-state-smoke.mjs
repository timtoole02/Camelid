#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  LLAMA32_3B_ACCEPTANCE_AVAILABILITY,
  LLAMA32_3B_ACCEPTANCE_GATING_NOTE,
  LLAMA32_3B_ACCEPTANCE_SUMMARY,
} from '../src/lib/acceptanceTargets.js'

import {
  capabilityStatusTone,
  compatibilityHintCopy,
  compatibilityHintLabel,
  compatibilityHintMatchesExactTarget,
  exactRowSupportLanes,
  findCompatibilityHint,
  formatCapabilityStatus,
  frontendSupportContractCopy,
  getCurrentCompatibilityTarget,
  getTrackedCompatibilityTargets,
  guardedCapabilityCopy,
  isCompatibilitySupportedForModel,
  isExactCompatibilityHint,
  isGuardedCapabilityStatus,
  isSupportedCapabilityStatus,
  quantLabelFromGgufFileType,
  rowSupportBoundaryCopy,
  rowSupportNextStepCopy,
  summarizeCapabilityItems,
} from '../src/lib/capabilities.js'

import {
  canLoadIntoRuntime,
  describeModelState,
  getModelStatusLabel,
  getRuntimeRequestModelId,
  hasLocalModelPath,
  isExternalModel,
  isHostedRoutingAvailable,
  isModelGenerationReady,
  isModelLoadedNow,
  isRunnableInCurrentRuntime,
  isRunnableModel,
  modelRuntimeIdMatches,
} from '../src/lib/modelState.js'

import { getChatGateState } from '../src/lib/chatGate.js'
import { isEmbeddingOnlyModel, isGenerationCapableModel, modelTaskKind } from '../src/lib/modelCapabilities.js'
import { formatDurationMs } from '../src/lib/formatters.js'
import { SUPPORTED_MODELS } from '../src/lib/supportedModels.js'

const localLoadedReady = {
  id: 'tiny-generation',
  name: 'Tiny generation',
  provider_kind: 'local',
  status: 'ready',
  model_path: '/tmp/tiny-generation.gguf',
  loaded_now: true,
  generation_ready: true,
}

assert.equal(isExternalModel(localLoadedReady), false)
assert.equal(hasLocalModelPath(localLoadedReady), true)
assert.equal(isModelLoadedNow(localLoadedReady), true)
assert.equal(isModelGenerationReady(localLoadedReady), true)
assert.equal(isRunnableModel(localLoadedReady), true)
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'tiny-generation', generation_ready: true }), true)
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'other-model', generation_ready: true }), false, 'a local model is not runnable for chat if a different model is active in Camelid')
assert.equal(isRunnableInCurrentRuntime(localLoadedReady, { active_model_id: 'tiny-generation', generation_ready: false }), false, 'loaded_now alone is not enough without runtime generation_ready=true')
const localReadyWithRuntimeName = { ...localLoadedReady, id: 'browser-alias', runtime_model_name: 'backend-runtime-id' }
assert.equal(modelRuntimeIdMatches(localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id' }), true, 'API/support readiness should accept the backend runtime model id when it differs from the browser alias')
assert.equal(getRuntimeRequestModelId(localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id' }, 'browser-alias'), 'backend-runtime-id', 'chat/API curl requests should use the loaded backend id when the selected browser row is an alias')
assert.equal(getRuntimeRequestModelId(localReadyWithRuntimeName, { active_model_id: 'other-runtime-id' }, 'browser-alias'), 'backend-runtime-id', 'inactive alias rows should still prefer their runtime_model_name over a browser-only id')
assert.equal(isRunnableInCurrentRuntime(localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id', generation_ready: true }), true, 'runtime-name matches keep chat/API gating tied to the exact loaded backend row')
assert.equal(getChatGateState({ model_compatibility: [] }, localReadyWithRuntimeName, { active_model_id: 'backend-runtime-id', loaded_now: true, generation_ready: true }).runtimeReady, true, 'chat gate runtime readiness should use the same runtime id matcher as the API view')
assert.equal(getModelStatusLabel(localLoadedReady), 'Loaded + generation-ready')
assert.match(describeModelState(localLoadedReady), /generation_ready=true/)

const bitnetEmbedding = {
  ...localLoadedReady,
  id: 'bitnet-embeddings-270m',
  name: 'Microsoft BitNet Embedding 270M',
  model_path: '/tmp/bitnet-embeddings-270m-bf16-i2_s.gguf',
  embedding_capable: true,
  generation_capable: false,
  generation_ready: false,
}
const embeddingRuntime = {
  active_model_id: 'bitnet-embeddings-270m',
  loaded_now: true,
  // A stale/incorrect true must never promote an embedding-only row into Chat.
  generation_ready: true,
  model_family: 'embedding',
}
assert.equal(isEmbeddingOnlyModel(bitnetEmbedding, embeddingRuntime), true)
assert.equal(modelTaskKind(bitnetEmbedding, embeddingRuntime), 'embedding')
assert.equal(isRunnableModel(bitnetEmbedding), false)
assert.equal(isRunnableInCurrentRuntime(bitnetEmbedding, embeddingRuntime), false)
assert.equal(getChatGateState({ model_compatibility: [] }, bitnetEmbedding, embeddingRuntime).embeddingOnly, true)
assert.equal(getChatGateState({ model_compatibility: [] }, bitnetEmbedding, embeddingRuntime).chatUnlocked, false)
assert.equal(getModelStatusLabel(bitnetEmbedding), 'Ready for embeddings')
assert.match(describeModelState(bitnetEmbedding), /embeddings and reranking/i)

const causalBaseModel = {
  ...localLoadedReady,
  id: 'bitnet-causal',
  chat_capable: false,
  embedding_capable: false,
  generation_capable: true,
}
assert.equal(isEmbeddingOnlyModel(causalBaseModel), false, 'chat_capable=false must not demote a causal base model to embedding-only')
assert.equal(isGenerationCapableModel(causalBaseModel), true)
const companionProjector = {
  id: 'vision-projector',
  architecture: 'clip',
  embedding_capable: false,
  generation_capable: false,
}
assert.equal(isGenerationCapableModel(companionProjector), false)
assert.equal(modelTaskKind(companionProjector), 'companion')
assert.equal(formatDurationMs(0.42), '420 μs')
assert.equal(formatDurationMs(18.7), '19 ms')
assert.equal(formatDurationMs(328.92), '329 ms')
assert.equal(formatDurationMs(19762.21), '19.8 s')

const nestedLoadedReady = {
  ...localLoadedReady,
  loaded_now: false,
  generation_ready: false,
  camelid: { loaded_now: true, generation_ready: true },
}
assert.equal(isModelLoadedNow(nestedLoadedReady), true)
assert.equal(isModelGenerationReady(nestedLoadedReady), true)
assert.equal(isRunnableModel(nestedLoadedReady), true, 'nested backend readiness should unlock chat when the local GGUF path is present')

const localSavedPath = {
  ...localLoadedReady,
  status: 'registered',
  loaded_now: false,
  generation_ready: false,
  camelid: { loaded_now: false, generation_ready: false },
}
assert.equal(canLoadIntoRuntime(localSavedPath), true)
assert.equal(isRunnableModel(localSavedPath), false)
assert.equal(getModelStatusLabel(localSavedPath), 'Local path saved')
assert.match(describeModelState(localSavedPath), /Use Load now/)

const localLoadedNotReady = {
  ...localLoadedReady,
  loaded_now: true,
  generation_ready: false,
  camelid: { loaded_now: true, generation_ready: false },
}
assert.equal(isRunnableModel(localLoadedNotReady), false)
assert.equal(getModelStatusLabel(localLoadedNotReady), 'Loaded, not generation-ready')
assert.match(describeModelState(localLoadedNotReady), /generation_ready=false/)
assert.match(describeModelState(localLoadedNotReady), /materialization budget/)

const staleReadyRecord = {
  ...localLoadedReady,
  loaded_now: false,
  camelid: { loaded_now: false, generation_ready: true },
}
assert.equal(isRunnableModel(staleReadyRecord), false, 'a stale saved record is not runnable unless it is loaded now')
assert.equal(isRunnableInCurrentRuntime(staleReadyRecord, { active_model_id: 'tiny-generation', generation_ready: true }), false)

const hostedPlanned = {
  id: 'openai-gpt-4o-mini',
  name: 'OpenAI GPT-4o mini',
  provider_kind: 'external',
  status: 'ready',
  api_base: 'https://api.openai.com/v1',
  runtime_model_name: 'gpt-4o-mini',
  api_key_configured: true,
}
assert.equal(isExternalModel(hostedPlanned), true)
assert.equal(isHostedRoutingAvailable(hostedPlanned), false)
assert.equal(isRunnableModel(hostedPlanned), false)
assert.equal(canLoadIntoRuntime(hostedPlanned), false)
assert.equal(getModelStatusLabel(hostedPlanned), 'API routing planned')
assert.match(describeModelState(hostedPlanned), /not wired yet/)

const hostedReady = { ...hostedPlanned, hosted_routing_ready: true }
assert.equal(isHostedRoutingAvailable(hostedReady), true)
assert.equal(isRunnableModel(hostedReady), true)
assert.equal(getModelStatusLabel(hostedReady), 'API routing ready')

assert.equal(formatCapabilityStatus('planned_phase_11_12'), 'planned phase 11 12')
assert.equal(quantLabelFromGgufFileType(7), 'Q8_0')
assert.equal(quantLabelFromGgufFileType('15'), 'Q4_K_M')
assert.equal(quantLabelFromGgufFileType(32), 'BF16')
assert.equal(quantLabelFromGgufFileType('unknown'), null)
assert.equal(isSupportedCapabilityStatus('supported_current_gate'), true)
assert.equal(isSupportedCapabilityStatus('validated'), false, 'validated evidence must not be treated as a support status')
assert.equal(isSupportedCapabilityStatus('measured'), false, 'measurement evidence must not be treated as a support status')
assert.equal(isGuardedCapabilityStatus('future'), true)
assert.equal(capabilityStatusTone('blocked_until_tensor_load_and_parity'), 'warm')
assert.equal(capabilityStatusTone('groundwork_backend_evidence_only'), 'warm')
assert.equal(capabilityStatusTone('blocked_unsupported_bringup'), 'warm')
assert.equal(capabilityStatusTone('validated_second_pack'), 'ready')
assert.equal(capabilityStatusTone('validated_bounded_pack_not_promoted'), 'warm')
assert.equal(capabilityStatusTone('fail-closed_until_promotion'), 'warm')
assert.equal(capabilityStatusTone('supported_exact_row_smoke'), 'ready')
assert.match(summarizeCapabilityItems([{ id: 'Q8_0', status: 'supported_current_gate' }]), /Q8_0: supported current gate/)
assert.match(guardedCapabilityCopy({ notes: 'Multi-choice is not implemented yet' }, 'API controls'), /API controls should stay disabled.*typed backend refusals.*not silently drop/)
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }, { id: 'tiny', status: 'supported_current_gate' }] }).id, 'tiny')
assert.equal(getCurrentCompatibilityTarget({ model_compatibility: [{ id: 'planned', status: 'planned' }] }), null, 'a planned/evidence row must not become the current supported gate fallback')

const capabilityFixture = {
  planned_model_families: [
    { id: 'larger_llama_instruct', status: 'planned', notes: 'progressively larger LLaMA-family instruct models' },
  ],
  model_compatibility: [
    { id: 'tinyllama_1_1b_chat_q8_0', family: 'llama_spm_decoder', quantization: 'Q8_0', status: 'supported_current_gate', support_scope: 'current_full_gate_exact_row', full_support_status: 'current_gate_refresh_under_stricter_bar', full_support_blockers: 'do not widen beyond TinyLlama 1.1B Chat Q8_0 without repeated current-head API/WebUI/parity/RSS/context evidence under the stricter bar; arbitrary/Jinja template behavior and production throughput remain outside this exact current gate unless separately validated', frontend_readiness_gate: 'green only when this exact Q8_0 row is loaded_now=true, generation_ready=true, and selected by active_model_id', chat_template_renderer: 'tinyllama-marker', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'measured', bounded_context_512_pack: 'validated_bounded_pack', bounded_context_1024_pack: 'not_promoted', bounded_context_2048_pack: 'not_promoted', latest_checked_bucket: 'direct_chat_smoke', latest_checked_result: 'pass', latest_checked_output: 'Certainly! Here', evidence: 'TinyLlama Q8_0 evidence', next_step: 'extend to larger contexts and additional LLaMA-family/quant targets before broadening support claims' },
    { id: 'llama_spm_q4_k_q5_k', family: 'llama_spm_decoder', quantization: 'Q4_K_M/Q5_K_M', status: 'planned_phase_10', next_step: 'implement K-quant support' },
    { id: 'llama32_1b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_smoke_only', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond checked packs, broader arbitrary templates beyond the supported metadata-Jinja Llama 3.2 1B row template, production throughput, portability, and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', chat_template_renderer: 'metadata_jinja_supported_for_exact_row', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'bounded_unique_chat_perf_rss_validated', bounded_context_512_pack: 'validated_bounded_pack', bounded_context_1024_pack: 'validated_second_pack', bounded_context_2048_pack: 'validated_third_pack', bounded_context_4096_pack: 'validated_fourth_pack', bounded_context_8192_pack: 'validated_fifth_pack', latest_checked_bucket: 'llama3-context-8192-smoke-v1', latest_checked_result: 'pass', latest_checked_output: 'CMLD-819', evidence: '1B exact-row load, completion, chat, frontend smoke, metadata-Jinja row-template parity, checked 512/1024/2048/4096/8192 context evidence, and bounded unique-chat perf/RSS evidence', next_step: 'preserve exact-row smoke plus checked 512/1024/2048/4096/8192 context support while normalizing model-native/larger context beyond checked packs, broader arbitrary-template behavior beyond the supported 1B metadata-Jinja row template, production throughput, portability, and durable full-support bundle evidence before any broader/full-support claim' },
    { id: 'llama32_3b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_smoke_only', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'the May-era typed lanes (template shapes, perf/RSS, API/WebUI smoke, broader prompt-pack parity) were measured on the prior upload (sha256 b5607b50...) and await re-anchoring to the canonical file; plus model-native/larger context beyond the anchored 512-8192 ladder, broader arbitrary/Jinja templates beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', chat_template_renderer: 'metadata_jinja_supported_for_exact_row', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'bounded_unique_chat_perf_rss_validated', bounded_context_512_pack: 'validated_anchored_raw_decode_ladder', bounded_context_512_pack_id: 'llama32-3b-anchored-raw-ladder-v1', bounded_context_1024_pack: 'validated_anchored_raw_decode_ladder', bounded_context_1024_pack_id: 'llama32-3b-anchored-raw-ladder-v1', bounded_context_2048_pack: 'validated_anchored_raw_decode_ladder', bounded_context_2048_pack_id: 'llama32-3b-anchored-raw-ladder-v1', bounded_context_4096_pack: 'validated_anchored_raw_decode_ladder', bounded_context_4096_pack_id: 'llama32-3b-anchored-raw-ladder-v1', bounded_context_8192_pack: 'validated_anchored_raw_decode_ladder', bounded_context_8192_pack_id: 'llama32-3b-anchored-raw-ladder-v1', latest_checked_bucket: 'llama32-3b-anchored-raw-ladder-v1', latest_checked_result: 'pass', latest_checked_output: '50/50 greedy tokens identical on all five buckets', evidence: '3B exact-row anchored raw-decode 512/1024/2048/4096/8192 context ladder on the canonical GGUF (sha256 f34112a1..., the June capability-receipt file), token+text identical to pinned llama.cpp acd79d603 at 50 greedy tokens, plus load, completion, chat, frontend smoke, and metadata-Jinja row-template evidence; the May-era canonical Ubuntu API/WebUI refresh, compact/broader prompt-pack parity, template-context packs, and perf/RSS evidence was measured on the prior upload (sha256 b5607b50...) and is retained as historical evidence for that file — disclosed, not inherited', next_step: 'preserve exact-row smoke plus the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder while re-anchoring the remaining May-era evidence families (API/WebUI refresh, template shapes, perf/RSS) to the canonical file and normalizing model-native/larger context, broader arbitrary/Jinja template behavior beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable full-support bundle evidence before any broader/full-support claim' },
    { id: 'llama3_8b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_smoke_only', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond the checked 512/1024/2048 packs, arbitrary templates, throughput, portability, repeated current-head evidence, and durable normalized full-support bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', chat_template_renderer: 'compact', chat_template_shape_pack: 'validated_compact_pack', performance_measured: 'bounded_ubuntu_backend_memory_gate_plus_lazy_q8_hotpath_costs', bounded_context_512_pack: 'validated_first_pack', bounded_context_1024_pack: 'validated_second_pack', bounded_context_2048_pack: 'validated_third_pack', latest_checked_bucket: 'llama3-context-2048-smoke-v1', latest_checked_result: 'pass', latest_checked_output: 'CMLD-204', evidence: '8B exact-row API/frontend smoke plus compact 50-token, broader 50-token, checked 512/1024/2048-context packs, compact template-shapes pack evidence, bounded memory/hot-path measurements, and current-head 1024/2048 PASS evidence. No model-native/larger context or broader/full support is implied.', next_step: 'preserve exact-row smoke plus checked 512/1024/2048 context support while collecting model-native/larger-context proof, broader/full-support, production-throughput, portability, arbitrary-template evidence, and repeated current-head evidence before any wider 8B claim' },
    { id: 'mistral_7b_instruct_v0_3_q8_0', family: 'mistral', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_smoke_only', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'model-native/larger context beyond checked packs, broader arbitrary/Jinja templates beyond the row-scoped renderer and template-shape evidence, production throughput beyond bounded perf/RSS evidence, portability, and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id', performance_measured: 'bounded_unique_chat_perf_rss_validated', evidence: 'Mistral v0.3 exact-row smoke (promoted post-v0.1.0, head d7b1699): metadata/tokenizer/template validated, tensors load, API completion+chat smoke plus broader 50-token API smoke, tokenizer/template/1-token/bounded/broader-50-token and GPU-vs-CPU greedy parity pass, bounded unique-chat perf/RSS validated; full support still blocked pending normalized evidence' },
    { id: 'mixtral_8x7b_instruct_v0_1_q8_0', family: 'mixtral_moe', quantization: 'Q8_0', status: 'active_validation_partial_runtime', support_scope: 'exact_row_bounded_moe_runtime_only', full_support_status: 'blocked_later_generation_divergence', full_support_blockers: 'later short-prompt generation still diverges from llama.cpp; API/WebUI readiness, long-context evidence, production throughput, portability, and durable broad prompt coverage are missing', frontend_readiness_gate: 'fail-closed for broad readiness: exact row may be described only as bounded one-token backend runtime evidence until later-generation parity and API/WebUI gates close', evidence: 'Mixtral bounded one-token backend MoE runtime evidence only; later-generation divergence keeps frontend/API/WebUI support blocked' },
    { id: 'qwen25_7b_instruct_q8_0', family: 'qwen2', quantization: 'Q8_0', status: 'planned_unsupported', support_scope: 'future_exact_row_planning_only', full_support_status: 'not_applicable_until_runtime_support', full_support_blockers: 'qwen2 runtime, tokenizer/pre-tokenizer fixtures, ChatML parity, bounded load/readiness, API/WebUI, RSS/timing, context, and durable bundle evidence are missing', evidence: 'Qwen 2.5 planning row only; no support evidence exists' },
    { id: 'gemma2_9b_it_q8_0', family: 'gemma2', quantization: 'Q8_0', status: 'active_validation_api_webui_pass_pending_context', support_scope: 'phase2_exact_row_validation_only', full_support_status: 'blocked_pending_context_performance_and_portability', full_support_blockers: 'short deterministic parity and guarded API/WebUI smoke pass; the exact-row bounded 512-context receipt is still required before support promotion', tensors_load: 'validated_real_weight_forward', generation_runs: 'validated_deterministic_greedy', parity_audited: 'pass_exact_greedy_token_ids', frontend_load_path_verified: 'validated_guarded_api_webui_smoke', evidence: 'qa/model-qualification/phase2-runtime/gemma2_9b_it_q8_0.json' },
    // Real shipped /api/capabilities row, copied field-for-field from the
    // generated ledger (ledger/camelid-ledger.json) rather than paraphrased.
    // The id IS the normalized GGUF filename, which is what lets the exact
    // identity matcher resolve the local file and unlock chat; the historical
    // `gemma3_1b_it_q8_0` spelling joined nothing and demoted the row. Keeping
    // the real strings here is what makes this smoke able to fail the way
    // production failed during the lane-gate fixture-drift incident.
    { id: 'gemma_3_1b_it_q8_0', family: 'gemma3_windowed_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', support_scope: 'exact_row_gpu_resident_windowed_chat_smoke_only_metal_or_cuda', full_support_status: 'blocked_pending_normalized_full_support', full_support_blockers: 'context above 2,403 prompt tokens (the file\'s native 32,768 is UNMEASURED, as is everything between ~2.4k and 32k); token-exact raw /v1/completions at depth 50 (three disclosed near-tie flips, one of them a 0.4471-nat stable-oracle near-tie — the clean raw-decode claim stops at DEPTH 5, while the chat lane is clean at 1/5/50); the repo\'s bounded-context ladder packs (512/1024/2048/4096/8192) were NOT run for this row — the windowed pack is a different artifact; throughput/performance (NOT claimed and NOT measured for release on this lane — a separate measurement phase owes it); the RUNNABLE CPU BRIDGE lane, which serves this row wherever the Metal resident lane cannot run and implements no window mask, so on THAT lane context above the 512-token sliding window remains mathematically wrong by construction (demonstrated: 1.667-nat disagreement with the pinned oracle at the divergence position of a 606-token prompt); tool capability (no gemma3 tools branch and no certified grammar on any lane — tools fail closed with a typed 422); multi-turn, streaming, speculative decode (declined for windowed archs) and the prompt-prefix cache (bypassed for windowed archs); neighbouring gemma3 sizes/quants (a non-Q8_0 gemma3 file is declined by resident admission and falls back to the runnable bridge); frontend load-path promotion; portability; and durable repeated current-head bundles remain missing', frontend_readiness_gate: 'green only when this exact gemma3 Q8_0 row (gemma-3-1b-it-Q8_0.gguf, sha256 b205840c...) is loaded_now=true, generation_ready=true, matching active_model_id, and served on EITHER GPU-resident windowed lane: Metal (metal_resident_q8_runtime) or CUDA (cuda_resident_windowed_runtime). Off both lanes the runnable CPU bridge serves the row and only the sub-512 envelope is green', chat_template_renderer: 'gemma3_marker_native_byte_locked_by_shapes_pack', chat_template_shape_pack: 'validated_in_src_pack_lock_test', performance_measured: 'not_claimed_resident_lane_throughput_is_a_separate_unshipped_measurement_phase', tested_context: 'gemma3_chat_greedy_1_5_50_at_606_1205_and_2403_prompt_tokens_plus_50_generated_all_above_the_512_token_sliding_window_metal_resident_lane_and_cuda_resident_windowed_lane_independently', bounded_context_512_pack: 'not_promoted', bounded_context_1024_pack: 'not_promoted', bounded_context_2048_pack: 'not_promoted', latest_checked_bucket: 'metal_gpu_resident_windowed_chat_parity_606_1205_2403_prompt_tokens', latest_checked_result: 'pass', latest_checked_output: 'qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-20260730-head-6eaf9053/README.md', evidence: 'exact row gemma-3-1b-it-Q8_0.gguf (sha256 b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5, 1,069,306,368 bytes, ggml-org/gemma-3-1b-it-GGUF upstream-verified exact, license gemma): 183 Q8_0 + 157 F32 tensors, tied embedding. LANE: on a Metal host the row is served by DEFAULT on the GPU-resident Q8_0 lane (selected_backend metal_resident_q8_runtime, decode_path q8_0_metal_resident_decode, prefill_path q8_0_metal_resident_prefill), the first gemma3 forward in-tree carrying the 5:1 local/global schedule and the 512-token sliding-window mask; the gemma3 marker renderer (byte-locked against the pinned oracle\'s /apply-template output by qa/prompt-packs/gemma3-chat-template-shapes-v1.json and an in-src pack-lock test) is shared with the dense lane, and decode stops on EOG. ORACLE: pinned llama.cpp acd79d603 (llama-server build 9632, CPU backend, -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096, binary sha256 prefix 382096b1), two-phase throughout — the two engines were never resident together. SUB-512 CHAT: committed 5-prompt gate pack (qa/prompt-packs/gemma3-chat-gate-pack-v1.json) at greedy depths 1/5/50 — 15/15 generation legs token-AND-text IDENTICAL, zero flips, with cross-engine prompt tokenization identical 5/5. ABOVE THE WINDOW (the claim this lane exists for): committed pack qa/prompt-packs/gemma3-windowed-context-pack-v1.json, three prompts rendering to 606 / 1205 / 2403 tokens (1.18x / 2.35x / 4.69x the file\'s own gemma3.attention.sliding_window=512), each answerable only from its first sentence — 3/3 prompt tokenization identical and 9/9 generation legs at 1/5/50 token-AND-text IDENTICAL, ZERO flips. That the window is REAL and not decorative is shown by the contrast leg: on the same 606-token prompt and the same oracle capture, the runnable CPU bridge (no window mask) diverges at generated index 2 and never resynchronises, and re-fed the identical prefix the oracle ranks the runnable lane\'s token 1.667 nats behind its own — four times the largest disclosed near-tie in the bundle, so not a near-tie. That leg is ONE prompt at ONE depth: a demonstration, not a runnable-lane receipt. DETERMINISM: two fresh camelid serve processes with a full stop/start between them produce byte-identical receipt files (6 chat legs incl. the 2403-token prompt plus 5 raw-completion legs carrying actual emitted ids; sha256 prefix 632992c6). RAW /v1/completions is a SEPARATE harness over its own 4-prompt set (not the 5-prompt chat pack above), committed as-is with all_pass=false and DISCLOSED rather than hidden: depths 1 and 5 are clean 4/4, and at depth 50 three legs flip — camelid\'s token is the oracle\'s rank 1 on two of them when the oracle decodes continuously rather than from a re-fed prefix, while the third (\'Once upon a time,\') has a rank-1-stable oracle and camelid\'s token at 0.4471 nat, ABOVE both the Ornith 0.33-nat line and the frozen runnable bundle\'s 0.3416; a separate harness artifact (SPM whitespace re-encode) is identified and excluded from that count rather than tuned away. Bundle qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-20260730-head-6eaf9053 (README + manifest + SHA256SUMS, 20 artifacts). The frozen runnable-lane bundle qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb stands as history for that other lane and is NOT re-adjudicated here. NOT claimed: any throughput or speed number on this lane, any comparison against another engine\'s speed, token-exact raw completions at depth 50, any context above 2,403 prompt tokens, the bounded-context ladder packs, multi-turn, streaming, tools (fail closed, typed 422), speculative decode or the prompt-prefix cache (both fail closed for windowed archs), neighbouring gemma3 sizes/quants, or the gemma4 family.', next_step: 'measure and publish the resident lane\'s throughput as its own receipt before any perf wording appears on this row; extend the windowed pack past 2,403 prompt tokens toward the file\'s native 32,768 before widening tested_context; adjudicate the three depth-50 raw-completion near-ties with a chat-shaped raw pack or an oracle-side reduction-order study before any raw token-exactness claim past depth 5; run the bounded-context ladder packs; a runnable-lane sliding-window mask (or removing that fallback) before any >512 claim travels to non-Metal hosts; agent-eval battery before any tool_capable claim' },
  ],
}
const boundedOnlySupportFixture = {
  support_contract: {
    current_gate: 'Current exact-row support: These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary/Jinja template behavior, production throughput, portability, neighboring-row, or broad-family support is implied.',
  },
  api_features: [],
  model_compatibility: [
    { id: 'supported_one', status: 'supported_exact_row_smoke', chat_template_renderer: 'metadata_jinja_supported_for_exact_row', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'bounded_unique_chat_perf_rss_validated', full_support_blockers: 'model-native/larger context, arbitrary/Jinja templates, production throughput, portability' },
    { id: 'supported_two', status: 'supported_current_gate', chat_template_renderer: 'tinyllama-marker', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'measured', full_support_blockers: 'arbitrary-template behavior and production throughput remain outside this gate unless separately validated' },
    { id: 'supported_three', status: 'supported_exact_row_smoke', chat_template_renderer: 'compact', chat_template_shape_pack: 'validated_compact_pack', performance_measured: 'bounded_ubuntu_backend_memory_gate_plus_lazy_q8_hotpath_costs', full_support_blockers: 'arbitrary templates, throughput, portability' },
  ],
}
const boundedOnlySupportLanes = exactRowSupportLanes(boundedOnlySupportFixture.model_compatibility[0], boundedOnlySupportFixture.api_features)
assert.deepEqual(boundedOnlySupportLanes.map((lane) => [lane.key, lane.ready]), [['template', true], ['context', false], ['throughput', false]], 'bounded perf/RSS evidence should not promote checked-context or production-throughput readiness for current supported rows')
assert.doesNotMatch(rowSupportBoundaryCopy(boundedOnlySupportFixture.model_compatibility[0], boundedOnlySupportFixture.api_features), /arbitrary|Jinja/i, 'resolved exact-row template/Jinja blockers should not remain in exact-row boundary copy')
assert.match(rowSupportBoundaryCopy(boundedOnlySupportFixture.model_compatibility[0], boundedOnlySupportFixture.api_features), /production|throughput/i, 'production-throughput blockers must remain until explicit production-throughput evidence is advertised')
assert.match(frontendSupportContractCopy(boundedOnlySupportFixture), /production|throughput/i, 'support contract copy must keep production-throughput caveats when only bounded perf/RSS evidence exists')
const unsupportedEvidenceFixture = {
  support_contract: boundedOnlySupportFixture.support_contract,
  api_features: [],
  model_compatibility: [
    { id: 'validation_only', status: 'active_validation_unsupported', chat_template_renderer: 'metadata_jinja_supported_for_exact_row', chat_template_shape_pack: 'validated_bounded_pack', performance_measured: 'bounded_unique_chat_perf_rss_validated', full_support_blockers: 'API/WebUI readiness, arbitrary/Jinja templates, production throughput, portability' },
  ],
}
const unsupportedEvidenceLanes = exactRowSupportLanes(unsupportedEvidenceFixture.model_compatibility[0], unsupportedEvidenceFixture.api_features)
assert.deepEqual(unsupportedEvidenceLanes.map((lane) => [lane.key, lane.ready]), [['template', false], ['context', false], ['throughput', false]], 'validation-only rows must not turn template/Jinja, checked-context, or throughput lanes green until the row itself is supported')
assert.match(rowSupportBoundaryCopy(unsupportedEvidenceFixture.model_compatibility[0], unsupportedEvidenceFixture.api_features), /arbitrary|Jinja|production|throughput/i, 'unsupported rows should keep template/Jinja and throughput blockers visible')
const promotedSupportFixture = {
  support_contract: boundedOnlySupportFixture.support_contract,
  api_features: [{ id: 'production_throughput', status: 'supported_exact_row_evidence', notes: 'explicit production-throughput lane' }],
  model_compatibility: [
    { id: 'supported_one', status: 'supported_exact_row_smoke', chat_template_renderer: 'arbitrary_jinja_supported', performance_measured: 'production_throughput_validated', full_support_blockers: 'model-native/larger context, arbitrary/Jinja templates, production throughput, portability' },
  ],
}
const promotedSupportLanes = exactRowSupportLanes(promotedSupportFixture.model_compatibility[0], promotedSupportFixture.api_features)
assert.deepEqual(promotedSupportLanes.map((lane) => [lane.key, lane.ready]), [['template', true], ['context', false], ['throughput', true]], 'explicit broad template and production-throughput evidence may clear those readiness lanes without inventing checked-context evidence')
assert.doesNotMatch(rowSupportBoundaryCopy(promotedSupportFixture.model_compatibility[0], promotedSupportFixture.api_features), /arbitrary|Jinja|production|throughput/i, 'resolved template/Jinja and production-throughput blockers should not remain in exact-row boundary copy')
// Redesign (2026-07, D14): the Models page was rebuilt as five derived zones. Hardcoded
// per-row evidence badges (e.g. the 8B pin-badge copy) were deleted with the tracked-row
// panel; row evidence renders from live /api/capabilities in the Compatibility ledger.
// The page-level invariant is now that membership is DERIVED, never hand-authored.
const modelsViewSource = readFileSync(new URL('../src/views/ModelsView.jsx', import.meta.url), 'utf8')
assert.doesNotMatch(
  modelsViewSource,
  /pin-badge/,
  'ModelsView must not hardcode per-row evidence badges; evidence must come from live /api/capabilities surfaces',
)
assert.match(
  modelsViewSource,
  /bucketByLane\(spine\.local\.models, capabilities\)/,
  'ModelsView section membership must be derived from the live /api/models/local scan + support contract at render time',
)

const genericExactRowFixture = {
  model_compatibility: [
    {
      id: 'custom_exact_row_q8_0',
      family: 'custom_decoder',
      quantization: 'Q8_0',
      status: 'supported_exact_row_smoke',
      frontend_readiness_gate: 'green only for this exact custom row id plus runtime readiness',
      evidence: 'custom row id fixture evidence',
    },
  ],
}
const genericExactRowHint = findCompatibilityHint(genericExactRowFixture, { id: 'custom-exact-row-q8-0', name: 'Custom exact row Q8_0', quant: 'Q8_0' })
assert.equal(genericExactRowHint.target.id, 'custom_exact_row_q8_0', 'generic backend compatibility row ids should be visible as exact evidence without adding a family-specific matcher')
assert.equal(isExactCompatibilityHint(genericExactRowHint), true, 'generic row-id matches should stay exact-row scoped')
assert.equal(isCompatibilitySupportedForModel(genericExactRowFixture, { id: 'custom-exact-row-q8-0', quant: 'Q8_0' }), true, 'supported generic exact rows should unlock only through the exact row id and quant evidence')
assert.equal(isCompatibilitySupportedForModel(genericExactRowFixture, { id: 'custom-exact-row-q4-0', quant: 'Q4_0' }), false, 'generic exact rows should not unlock neighboring quantized filenames')

// The family column mirrors the real contract, which is NOT uniform: the 4B/8B
// GGUFs declare `general.architecture=qwen3` and only the 27B pair declares
// `qwen35`. The Supported lane is derived from id/filename/quant, so family is
// incidental here — it is kept accurate so the fixture cannot teach the wrong shape.
const prismCatalogRows = [
  ['bonsai_4b_q1_0', 'Bonsai-4B-Q1_0.gguf', 'Q1_0', 'qwen3_bonsai_gpu'],
  ['ternary_bonsai_4b_q2_0', 'Ternary-Bonsai-4B-Q2_0.gguf', 'Q2_0', 'qwen3_bonsai_gpu'],
  ['ternary_bonsai_4b_pq2_0', 'Ternary-Bonsai-4B-PQ2_0.gguf', 'PQ2_0', 'qwen3_bonsai_gpu'],
  ['bonsai_8b_q1_0', 'Bonsai-8B-Q1_0.gguf', 'Q1_0', 'qwen3_bonsai_gpu'],
  ['ternary_bonsai_8b_q2_0', 'Ternary-Bonsai-8B-Q2_0.gguf', 'Q2_0', 'qwen3_bonsai_gpu'],
  ['bonsai_27b_q1_0', 'Bonsai-27B-Q1_0.gguf', 'Q1_0', 'qwen35_bonsai_gpu_vision'],
  ['ternary_bonsai_27b_q2_0', 'Ternary-Bonsai-27B-Q2_0.gguf', 'Q2_0', 'qwen35_bonsai_gpu_vision'],
]
const prismCapabilityFixture = {
  model_compatibility: prismCatalogRows.map(([id, filename, quantization, family]) => ({
    id,
    family,
    quantization,
    status: 'supported_exact_row_smoke',
    frontend_readiness_gate: `green only for ${filename} on a checked Metal or Windows CUDA lane`,
    evidence: 'paired Metal and Windows CUDA exact-artifact receipts',
  })),
}
for (const [id, filename, quant] of prismCatalogRows) {
  const catalogItem = SUPPORTED_MODELS.find((item) => item.catalog_id === id)
  assert.ok(catalogItem, `${id} must be visible in the frontend curated catalog decoration`)
  assert.equal(catalogItem.filename, filename)
  assert.equal(catalogItem.quant, quant)
  assert.equal(
    isCompatibilitySupportedForModel(prismCapabilityFixture, null, catalogItem),
    true,
    `${filename} must derive the Models-page Supported lane from its exact capability row`,
  )
}

const lfmCatalogItem = SUPPORTED_MODELS.find((item) => item.catalog_id === 'lfm2_5_2_6b_q8_0')
assert.ok(lfmCatalogItem, 'LFM2.5 2.6B Q8_0 must be visible in the frontend curated catalog decoration')
assert.equal(lfmCatalogItem.filename, 'LFM2.5-2.6B-Q8_0.gguf')
assert.equal(lfmCatalogItem.size_bytes, 2874779456)
assert.equal(lfmCatalogItem.quant, 'Q8_0')
assert.equal(
  isCompatibilitySupportedForModel(prismCapabilityFixture, null, {
    catalog_id: 'ternary_bonsai_8b_pq2_0',
    filename: 'Ternary-Bonsai-8B-PQ2_0.gguf',
    quant: 'PQ2_0',
  }),
  false,
  'neighboring Bonsai artifacts must remain experimental until their own exact support row is certified',
)
assert.equal(quantLabelFromGgufFileType(40), 'Q1_0')
assert.equal(quantLabelFromGgufFileType(41), 'Q2_0')
assert.equal(quantLabelFromGgufFileType(142), null, 'tensor type ids must not be mistaken for general.file_type ids')

const trackedTargets = getTrackedCompatibilityTargets(capabilityFixture)
assert.deepEqual(
  trackedTargets.map((target) => target.id),
  ['tinyllama_1_1b_chat_q8_0', 'llama32_1b_instruct_q8_0', 'llama32_3b_instruct_q8_0', 'llama3_8b_instruct_q8_0'],
  'tracked full-support hardening rows should stay pinned to the exact TinyLlama/1B/3B/8B ids in /api/capabilities order',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, exactRowSupportLanes(target, capabilityFixture.api_features).map((lane) => [lane.key, lane.ready])]),
  [
    ['tinyllama_1_1b_chat_q8_0', [['template', true], ['context', true], ['throughput', false]]],
    ['llama32_1b_instruct_q8_0', [['template', true], ['context', true], ['throughput', false]]],
    ['llama32_3b_instruct_q8_0', [['template', true], ['context', true], ['throughput', false]]],
    ['llama3_8b_instruct_q8_0', [['template', true], ['context', true], ['throughput', false]]],
  ],
  'current supported rows should expose green template and checked-context lanes while keeping production-throughput unpromoted without explicit /api/capabilities evidence',
)
for (const target of trackedTargets) {
  assert.doesNotMatch(
    rowSupportBoundaryCopy(target, capabilityFixture.api_features),
    /arbitrary|Jinja/i,
    `${target.id} remaining support boundary should not repeat resolved template/Jinja caveats`,
  )
  if (/production|throughput/i.test(target.full_support_blockers || '')) {
    assert.match(
      rowSupportBoundaryCopy(target, capabilityFixture.api_features),
      /production|throughput/i,
      `${target.id} remaining support boundary should keep production-throughput caveats until explicit evidence exists`,
    )
  }
  assert.doesNotMatch(
    rowSupportNextStepCopy(target, capabilityFixture.api_features),
    /arbitrary|Jinja/i,
    `${target.id} next-step copy should not repeat resolved template/Jinja caveats`,
  )
  if (/production|throughput/i.test(target.next_step || '')) {
    assert.match(
      rowSupportNextStepCopy(target, capabilityFixture.api_features),
      /production|throughput/i,
      `${target.id} next-step copy should keep production-throughput caveats until explicit evidence exists`,
    )
  }
}
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.full_support_status, Boolean(target.full_support_blockers), Boolean(target.frontend_readiness_gate)]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'current_gate_refresh_under_stricter_bar', true, true],
    ['llama32_1b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
    ['llama32_3b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
    ['llama3_8b_instruct_q8_0', 'blocked_pending_normalized_full_support', true, true],
  ],
  'all current rows should carry an explicit stricter full-support bar and fail-closed frontend readiness gate',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_512_pack || 'not_promoted']),
  [
    ['tinyllama_1_1b_chat_q8_0', 'validated_bounded_pack'],
    ['llama32_1b_instruct_q8_0', 'validated_bounded_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_anchored_raw_decode_ladder'],
    ['llama3_8b_instruct_q8_0', 'validated_first_pack'],
  ],
  'frontend tracked rows should preserve the API 512-context boundary for exact TinyLlama/1B/3B/8B checked packs',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_1024_pack]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'not_promoted'],
    ['llama32_1b_instruct_q8_0', 'validated_second_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_anchored_raw_decode_ladder'],
    ['llama3_8b_instruct_q8_0', 'validated_second_pack'],
  ],
  'frontend tracked rows should preserve the API 1024-context boundary: TinyLlama not promoted; exact 1B/3B/8B promoted only for their checked bounded packs',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_2048_pack]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'not_promoted'],
    ['llama32_1b_instruct_q8_0', 'validated_third_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_anchored_raw_decode_ladder'],
    ['llama3_8b_instruct_q8_0', 'validated_third_pack'],
  ],
  'frontend tracked rows should preserve the API 2048-context boundary: TinyLlama not promoted; exact 1B/3B/8B promoted only for their checked bounded packs',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.bounded_context_4096_pack || 'not_promoted', target.bounded_context_8192_pack || 'not_promoted']),
  [
    ['tinyllama_1_1b_chat_q8_0', 'not_promoted', 'not_promoted'],
    ['llama32_1b_instruct_q8_0', 'validated_fourth_pack', 'validated_fifth_pack'],
    ['llama32_3b_instruct_q8_0', 'validated_anchored_raw_decode_ladder', 'validated_anchored_raw_decode_ladder'],
    ['llama3_8b_instruct_q8_0', 'not_promoted', 'not_promoted'],
  ],
  'frontend tracked rows should preserve the API 4096/8192-context boundary: only the exact 1B and 3B rows are promoted for their checked packs (3B via the anchored raw-decode ladder); TinyLlama/8B stay not promoted',
)
assert.deepEqual(
  trackedTargets.map((target) => [target.id, target.latest_checked_bucket, target.latest_checked_result, target.latest_checked_output]),
  [
    ['tinyllama_1_1b_chat_q8_0', 'direct_chat_smoke', 'pass', 'Certainly! Here'],
    ['llama32_1b_instruct_q8_0', 'llama3-context-8192-smoke-v1', 'pass', 'CMLD-819'],
    ['llama32_3b_instruct_q8_0', 'llama32-3b-anchored-raw-ladder-v1', 'pass', '50/50 greedy tokens identical on all five buckets'],
    ['llama3_8b_instruct_q8_0', 'llama3-context-2048-smoke-v1', 'pass', 'CMLD-204'],
  ],
  'frontend tracked rows should surface the API latest bounded checks without implying broad/full support or model-native/larger-context support',
)

const tinyQ8Hint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' })
assert.equal(tinyQ8Hint.target.id, 'tinyllama_1_1b_chat_q8_0')
assert.equal(compatibilityHintLabel(tinyQ8Hint), 'tinyllama_1_1b_chat_q8_0: supported current gate')
assert.equal(isExactCompatibilityHint(tinyQ8Hint), true, 'TinyLlama support should come from its exact row, not a broad family row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q8_0' }), true)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'file_type 7' }), true, 'GGUF file_type labels should map to exact quant rows')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'general.file_type: 7' }), true, 'GGUF metadata-shaped file_type labels should map to exact quant rows')
const tinyNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat' })
assert.equal(tinyNoQuantHint.kind, 'quant_missing', 'TinyLlama current gate still needs exact Q8_0 evidence before chat unlocks')
assert.equal(compatibilityHintLabel(tinyNoQuantHint), 'tinyllama_1_1b_chat_q8_0: quant not verified')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'TinyLlama 1.1B Chat' }), false, 'chat should not unlock from a family/name match without quant evidence')
const tinyKQuantHint = findCompatibilityHint(capabilityFixture, { name: 'TinyLlama 1.1B Chat', quant: 'Q4_K_M' })
assert.equal(tinyKQuantHint.kind, 'family', 'TinyLlama K-quant should be shown as a guarded family row, not exact-row evidence')
assert.equal(tinyKQuantHint.target.id, 'llama_spm_q4_k_q5_k', 'TinyLlama family names must not inherit Q8 support for a K-quant entry')
assert.equal(compatibilityHintLabel(tinyKQuantHint), 'llama_spm_q4_k_q5_k: planned phase 10')
assert.equal(isExactCompatibilityHint(tinyKQuantHint), false)
assert.match(compatibilityHintCopy(tinyKQuantHint), /not chat-ready support|concrete exact compatibility row/)
const llama3Q4Hint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q4_K_M' })
assert.equal(llama3Q4Hint.kind, 'quant_mismatch')
assert.match(compatibilityHintCopy(llama3Q4Hint), /Do not inherit the supported gate|wait for an exact COMPATIBILITY\.md row/)
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'Q8_0' }), false, '8B rows must not unlock from name/quant without exact artifact evidence')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Meta Llama 3 8B Instruct', quant: 'file_type 7' }), false, 'GGUF file_type evidence still needs exact 8B artifact identity before chat unlocks')
const llama32OneBExactArtifactModel = {
  name: 'Llama 3.2 1B Instruct Q8_0',
  model_path: '<ubuntu-model-path>/Llama-3.2-1B-Instruct-Q8_0.gguf',
  quant: 'Q8_0',
}
const llama32OneBHint = findCompatibilityHint(capabilityFixture, llama32OneBExactArtifactModel)
assert.equal(llama32OneBHint.target.id, 'llama32_1b_instruct_q8_0', 'Llama 3.2 1B must match its exact promoted row')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, llama32OneBExactArtifactModel), true, 'exact promoted 1B rows are supported only with exact size/instruct/quant/artifact evidence')
assert.deepEqual(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-1b', ...llama32OneBExactArtifactModel }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true }),
  {
    hint: llama32OneBHint,
    taskKind: 'unknown',
    embeddingOnly: false,
    embeddingReady: false,
    generationCapable: true,
    runtimeReady: true,
    runtimeLoaded: true,
    runtimeGenerationReady: true,
    contractSupported: true,
    chatUnlocked: true,
    experimentalUnlocked: false,
    chatMode: 'supported',
    // `label` is the human layer shown in the UI; the raw row id and its
    // evidence stay in `hint`, which this same assertion pins above.
    label: 'Verified',
    copy: compatibilityHintCopy(llama32OneBHint),
  },
  'Llama 3.2 1B runtime-green exact rows should unlock supported WebUI chat without broad family claims',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: false, generation_ready: true }).chatUnlocked,
  false,
  'exact supported rows still require runtime loaded_now=true before chat unlocks',
)
const llama32OneBNameOnlyHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32OneBNameOnlyHint.kind, 'artifact_mismatch', 'Llama 3.2 1B exact-size matches must not become compatibility matches without exact artifact evidence')
assert.equal(compatibilityHintLabel(llama32OneBNameOnlyHint), 'llama32_1b_instruct_q8_0: exact GGUF not verified')
assert.match(compatibilityHintCopy(llama32OneBNameOnlyHint), /requires the exact Llama-3\.2-1B-Instruct-Q8_0\.gguf artifact/)
const promotedOneBFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'supported_current_gate' } : row),
}
assert.equal(isCompatibilitySupportedForModel(promotedOneBFixture, { name: 'Llama 3.2 1B Instruct' }), false, 'exact-size Llama rows still need exact artifact evidence even after promotion')
const llama32ThreeBExactArtifactModel = {
  name: 'Llama 3.2 3B Instruct Q8_0',
  model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf',
  quant: 'Q8_0',
}
const llama32ThreeBHint = findCompatibilityHint(capabilityFixture, llama32ThreeBExactArtifactModel)
assert.equal(llama32ThreeBHint.target.id, 'llama32_3b_instruct_q8_0', 'Llama 3.2 3B must match its exact row rather than inheriting the 8B row')
assert.equal(compatibilityHintLabel(llama32ThreeBHint), 'llama32_3b_instruct_q8_0: supported exact row smoke')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, llama32ThreeBExactArtifactModel), true, 'exact promoted 3B rows are supported only with exact size/instruct/quant/artifact evidence')
assert.equal(compatibilityHintMatchesExactTarget(capabilityFixture, llama32ThreeBExactArtifactModel, { id: 'llama32_3b_instruct_q8_0' }), true, '3B Q8_0 exact-row helpers should match only the promoted row')
const llama32ThreeBNameOnlyHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(compatibilityHintLabel(llama32ThreeBNameOnlyHint), 'llama32_3b_instruct_q8_0: exact GGUF not verified', '3B name-only rows must not unlock without the exact GGUF filename')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' }), false, '3B name-size-quant evidence alone is not enough without exact artifact identity')
assert.equal(compatibilityHintMatchesExactTarget(capabilityFixture, { name: 'Llama 3.2 3B Instruct Q4_K_M', quant: 'Q4_K_M' }, { id: 'llama32_3b_instruct_q8_0' }), false, '3B non-Q8 entries must not satisfy exact-row frontend card/readiness matching')
assert.equal(compatibilityHintMatchesExactTarget(capabilityFixture, { name: 'Llama 3.2 3B Base Q8_0', quant: 'Q8_0' }, { id: 'llama32_3b_instruct_q8_0' }), false, '3B base/non-instruct entries must not satisfy the exact instruct support row')
const llama32ThreeBQ4PathModel = { name: 'Llama 3.2 3B Instruct', id: 'llama32_3b_instruct_q8_0', model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q4_0.gguf' }
const llama32ThreeBQ4PathHint = findCompatibilityHint(capabilityFixture, llama32ThreeBQ4PathModel)
assert.equal(llama32ThreeBQ4PathHint.kind, 'quant_mismatch', 'a canonical 3B row id must not override neighboring-quant evidence from the loaded GGUF path')
assert.equal(llama32ThreeBQ4PathHint.observedQuant, 'Q40', '3B exact-row mismatch should carry the path-derived neighboring quant key')
assert.match(compatibilityHintCopy(llama32ThreeBQ4PathHint), /appears to be Q4_0/, '3B exact-row mismatch copy should display the loaded artifact quant in GGUF-style form')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, llama32ThreeBQ4PathModel), false, '3B exact-row support requires the loaded artifact quant to match Q8_0, not just the browser row id')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, ...llama32ThreeBQ4PathModel }, { active_model_id: 'llama32_3b_instruct_q8_0', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'runtime-green 3B rows must fail closed when the loaded GGUF path is a neighboring quant despite the canonical row id',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-3b', name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-3b', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'Llama 3.2 3B name-only rows should not unlock supported WebUI chat even when runtime-green',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama32-3b', ...llama32ThreeBExactArtifactModel }, { active_model_id: 'llama32-3b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'Llama 3.2 3B exact artifact rows should unlock supported WebUI chat when runtime-green',
)
const liveScalarThreeBModel = {
  ...localLoadedReady,
  id: 'scalar_default_rerun',
  name: 'scalar_default_rerun',
  runtime_model_name: 'scalar_default_rerun',
  model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf',
  quant: 'file_type 7',
}
const liveScalarThreeBGate = getChatGateState(capabilityFixture, liveScalarThreeBModel, { active_model_id: 'scalar_default_rerun', loaded_now: true, generation_ready: true })
assert.equal(liveScalarThreeBGate.hint.target.id, 'llama32_3b_instruct_q8_0', 'canonical Ubuntu 3B runtime ids like scalar_default_rerun should resolve to the exact 3B row from GGUF path + file_type 7 evidence')
assert.equal(liveScalarThreeBGate.runtimeReady, true, 'canonical Ubuntu 3B runtime health should remain visible when active_model_id is a backend run label')
assert.equal(liveScalarThreeBGate.chatUnlocked, true, 'canonical Ubuntu 3B backend run labels should unlock only when path, Q8_0 file_type, active_model_id, loaded_now, and generation_ready are all green')
assert.equal(
  getChatGateState(capabilityFixture, { ...liveScalarThreeBModel, quant: 'general.file_type=7' }, { active_model_id: 'scalar_default_rerun', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'canonical Ubuntu 3B backend run labels should also unlock with metadata-shaped GGUF general.file_type=7 quant evidence',
)
const liveNamedThreeBModel = {
  ...liveScalarThreeBModel,
  id: 'Llama 3.2 3B Instruct',
  name: 'Llama 3.2 3B Instruct',
  runtime_model_name: 'Llama 3.2 3B Instruct',
  quant: 'Q8_0',
}
const liveNamedThreeBGate = getChatGateState(capabilityFixture, liveNamedThreeBModel, { active_model_id: 'Llama 3.2 3B Instruct', loaded_now: true, generation_ready: true })
assert.equal(liveNamedThreeBGate.hint.target.id, 'llama32_3b_instruct_q8_0', 'canonical Ubuntu 3B active_model_id copy without Q8_0 in the runtime name should still resolve through the exact loaded GGUF path plus Q8_0 metadata')
assert.equal(liveNamedThreeBGate.chatUnlocked, true, 'canonical Ubuntu 3B human-readable active_model_id should unlock WebUI chat only when the exact loaded path, Q8_0 metadata, and runtime readiness are all green')
const liveMisleadingTinyThreeBModel = {
  ...liveScalarThreeBModel,
  id: 'tinyllama-q8',
  name: 'Llama 3.2 3B Instruct Q8_0',
  runtime_model_name: 'tinyllama-q8',
}
const liveMisleadingTinyThreeBGate = getChatGateState(capabilityFixture, liveMisleadingTinyThreeBModel, { active_model_id: 'tinyllama-q8', loaded_now: true, generation_ready: true })
assert.equal(liveMisleadingTinyThreeBGate.hint.target.id, 'llama32_3b_instruct_q8_0', 'canonical Ubuntu 3B runtime labels that still say tinyllama-q8 must resolve through the exact 3B GGUF path/name + Q8_0 metadata, not the TinyLlama row')
assert.equal(liveMisleadingTinyThreeBGate.chatUnlocked, true, 'misleading backend runtime ids should not block or mislabel exact 3B WebUI support when the loaded GGUF path, quant, and runtime readiness are green')
assert.equal(
  getChatGateState(capabilityFixture, { ...liveScalarThreeBModel, quant: 'Q4_K_M' }, { active_model_id: 'scalar_default_rerun', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'canonical Ubuntu 3B runtime labels must still fail closed when explicit quant evidence disagrees with the supported Q8_0 row',
)
const llama3EightBExactArtifactModel = {
  name: 'Meta Llama 3 8B Instruct Q8_0',
  model_path: '<ubuntu-model-path>/Meta-Llama-3-8B-Instruct.Q8_0.gguf',
  quant: 'Q8_0',
}
const llama3EightBHint = findCompatibilityHint(capabilityFixture, llama3EightBExactArtifactModel)
assert.equal(llama3EightBHint.target.id, 'llama3_8b_instruct_q8_0', 'Llama 3 8B must match its exact supported row')
assert.match(compatibilityHintCopy(llama3EightBHint), /checked 512\/1024\/2048-context packs, compact template-shapes pack evidence, bounded memory\/hot-path measurements, and current-head 1024\/2048 PASS evidence/)
assert.match(compatibilityHintCopy(llama3EightBHint), /No model-native\/larger context or broader\/full support is implied/)
const llama3HyphenEightBHint = findCompatibilityHint(capabilityFixture, { model_path: '<ubuntu-model-path>/Meta-Llama-3-8B-Instruct.Q8_0.gguf', quant: 'Q8_0' })
assert.equal(llama3HyphenEightBHint.target.id, 'llama3_8b_instruct_q8_0', 'Llama-3-8B filenames should match the exact Llama 3 8B row')
const llama3EightBNameOnlyHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama3EightBNameOnlyHint.kind, 'artifact_mismatch', 'Llama 3 8B must not unlock from a size/instruct/quant match without exact artifact evidence')
assert.equal(compatibilityHintLabel(llama3EightBNameOnlyHint), 'llama3_8b_instruct_q8_0: exact GGUF not verified')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'llama3-8b', ...llama3EightBExactArtifactModel }, { active_model_id: 'llama3-8b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'Llama 3 8B exact artifact rows should unlock supported WebUI chat when runtime-green',
)
const llama31EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3.1 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama31EightBHint, null, 'Llama 3.1 8B must not inherit the Llama 3 8B row')
const llama33EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3.3 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama33EightBHint, null, 'Llama 3.3 8B must not inherit the Llama 3 8B row')
const llama32NoSizeHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32NoSizeHint, null, 'Llama 3.2 names without exact 1B/3B size must not inherit a tracked row or family readiness hint')
const llama32EightBHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 8B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama32EightBHint, null, 'Llama 3.2 8B must not inherit the Llama 3 8B row or a family readiness hint')
const llama3OneBHint = findCompatibilityHint(capabilityFixture, { name: 'Meta Llama 3 1B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(llama3OneBHint, null, 'Llama 3 1B must not inherit the Llama 3.2 1B row or a family readiness hint')
const llama32OneBBaseHint = findCompatibilityHint(capabilityFixture, { name: 'Llama 3.2 1B Base Q8_0', quant: 'Q8_0' })
assert.equal(llama32OneBBaseHint, null, 'Llama 3.2 1B non-instruct names must not inherit the exact Instruct row')
const noExactThreeBHint = findCompatibilityHint({ ...capabilityFixture, model_compatibility: capabilityFixture.model_compatibility.filter((row) => row.id !== 'llama32_3b_instruct_q8_0') }, { name: 'Llama 3.2 3B Instruct Q8_0', quant: 'Q8_0' })
assert.equal(noExactThreeBHint, null, 'Llama 3.2 3B must not show family readiness when no exact compatibility row exists')
assert.match(compatibilityHintCopy(noExactThreeBHint), /No exact COMPATIBILITY\.md row matched/)
const evidenceOnly1BFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'groundwork_backend_evidence_only' } : row),
}
const evidenceOnly1BGate = getChatGateState(evidenceOnly1BFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true })
assert.equal(evidenceOnly1BGate.runtimeReady, true, 'runtime readiness should be visible even for evidence-only rows')
assert.equal(evidenceOnly1BGate.contractSupported, false, 'evidence-only rows are not exact supported rows')
assert.equal(evidenceOnly1BGate.chatUnlocked, false, 'WebUI chat must remain blocked unless runtime readiness and an exact supported compatibility row both pass')
const validatedOnly1BFixture = {
  ...capabilityFixture,
  model_compatibility: capabilityFixture.model_compatibility.map((row) => row.id === 'llama32_1b_instruct_q8_0' ? { ...row, status: 'validated' } : row),
}
const validatedOnly1BGate = getChatGateState(validatedOnly1BFixture, { ...localLoadedReady, id: 'llama32-1b', name: 'Llama 3.2 1B Instruct Q8_0', quant: 'Q8_0' }, { active_model_id: 'llama32-1b', loaded_now: true, generation_ready: true })
assert.equal(validatedOnly1BGate.contractSupported, false, 'validated rows are evidence boundaries only, not support statuses')
assert.equal(validatedOnly1BGate.chatUnlocked, false, 'WebUI chat must not unlock from a generic validated row status')
const mistralExactHint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' })
assert.equal(mistralExactHint.kind, 'compatibility', 'the future Mistral lane should identify only the exact v0.3 7B Instruct Q8_0 row')
assert.equal(mistralExactHint.target.id, 'mistral_7b_instruct_v0_3_q8_0')
assert.equal(mistralExactHint.target.status, 'supported_exact_row_smoke', 'Mistral exact-row matching advertises the promoted supported_exact_row_smoke contract status (matches src/api/mod.rs)')
assert.equal(mistralExactHint.target.full_support_status, 'blocked_pending_normalized_full_support', 'Mistral exact-row matching advertises full support still blocked pending normalized evidence')
assert.match(mistralExactHint.target.full_support_blockers, /model-native\/larger context|production throughput|portability|durable repeated current-head bundles/i, 'Mistral exact-row matching must carry its remaining full-support blocking evidence list')
assert.doesNotMatch(mistralExactHint.target.full_support_blockers, /source\/SHA\/license|1-token generation parity .*not complete/i, 'Mistral exact-row matching must not mark already-green row-specific evidence as missing')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' }), true, 'the promoted Mistral exact row (supported_exact_row_smoke) is contract-supported for the exact v0.3 7B Q8_0 file')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mistral-v03', name: 'Mistral-7B-Instruct-v0.3 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mistral-v03', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'runtime-green Mistral v0.3 unlocks supported chat now that /api/capabilities records the exact row as supported_exact_row_smoke (behavior already live in production since head d7b1699)',
)
const mistralNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.3' })
assert.equal(mistralNoQuantHint.kind, 'quant_missing', 'Mistral exact-row support must still require quant evidence')
const mistralV02Hint = findCompatibilityHint(capabilityFixture, { name: 'Mistral-7B-Instruct-v0.2 Q8_0', quant: 'Q8_0' })
assert.equal(mistralV02Hint.kind, 'family', 'Mistral v0.2 must not inherit the v0.3 exact-row lane')
assert.match(compatibilityHintCopy(mistralV02Hint), /not chat-ready support|not support/i)
const mixtralHint = findCompatibilityHint(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' })
assert.equal(mixtralHint.kind, 'compatibility', 'Mixtral should match only its exact active-validation row, not a Mistral exact-row match')
assert.equal(mixtralHint.target.id, 'mixtral_8x7b_instruct_v0_1_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }), false)
const mixtralNoQuantHint = findCompatibilityHint(capabilityFixture, { name: 'Mixtral-8x7B-Instruct-v0.1' })
assert.equal(mixtralNoQuantHint.kind, 'quant_missing', 'Mixtral exact-row support must still require quant evidence')
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mixtral-v01', name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mixtral-v01', loaded_now: true, generation_ready: true }).chatUnlocked,
  false,
  'runtime-green Mixtral v0.1 exact Q8_0 row stays blocked while /api/capabilities keeps it active-validation unsupported',
)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'mixtral-v01', name: 'Mixtral-8x7B-Instruct-v0.1 Q8_0', quant: 'Q8_0' }, { active_model_id: 'mixtral-v01', loaded_now: true, generation_ready: false }).chatUnlocked,
  false,
  'Mixtral v0.1 exact Q8_0 row remains blocked when runtime generation_ready is false',
)
const qwenHint = findCompatibilityHint(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q8_0', quant: 'Q8_0' })
assert.equal(qwenHint.kind, 'compatibility', 'Qwen should match only its exact future planning row')
assert.equal(qwenHint.target.id, 'qwen25_7b_instruct_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q8_0', quant: 'Q8_0' }), false)
const qwenQ4Hint = findCompatibilityHint(capabilityFixture, { name: 'Qwen2.5-7B-Instruct-Q4_K_M', quant: 'Q4_K_M' })
assert.equal(qwenQ4Hint.kind, 'quant_mismatch', 'Qwen planning rows must not absorb different quantizations')
const gemmaHint = findCompatibilityHint(capabilityFixture, { name: 'gemma-2-9b-it-Q8_0', model_path: 'gemma-2-9b-it-q8_0.gguf', quant: 'Q8_0' })
assert.equal(gemmaHint.kind, 'compatibility', 'Gemma should match only its exact qualified row')
assert.equal(gemmaHint.target.id, 'gemma2_9b_it_q8_0')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'gemma-2-9b-it-Q8_0', quant: 'Q8_0' }), false)
const gemma2VerifiedGate = getChatGateState(
  capabilityFixture,
  { ...localLoadedReady, id: 'gemma2-9b', name: 'gemma-2-9b-it-Q8_0', model_path: 'gemma-2-9b-it-q8_0.gguf', quant: 'Q8_0' },
  { active_model_id: 'gemma2-9b', loaded_now: true, generation_ready: true },
)
assert.equal(gemma2VerifiedGate.chatUnlocked, false, 'verified-runnable is narrower than Supported')
assert.equal(gemma2VerifiedGate.chatMode, 'verified', 'Gemma 2 must preserve its passed exact-row qualification in chat')
assert.equal(gemma2VerifiedGate.experimentalUnlocked, true, 'verified-runnable Gemma 2 remains usable')

const gemma3VarianceCapabilities = {
  ...capabilityFixture,
  model_compatibility: [
    ...capabilityFixture.model_compatibility,
    {
      id: 'gemma3_4b_it_q8_0',
      family: 'gemma3',
      quantization: 'Q8_0',
      status: 'runnable_exact_row_numerical_variance',
      tensors_load: 'validated_real_weight_forward',
      generation_runs: 'validated_deterministic_greedy',
      parity_audited: 'failed_exact_greedy_token_ids',
    },
  ],
}
const gemma3VarianceGate = getChatGateState(
  gemma3VarianceCapabilities,
  { ...localLoadedReady, id: 'gemma3-4b', name: 'gemma-3-4b-it-Q8_0', model_path: 'gemma-3-4b-it-Q8_0.gguf', quant: 'Q8_0', lane_class: 'runnable_with_variance' },
  { active_model_id: 'gemma3-4b', loaded_now: true, generation_ready: true },
)
assert.equal(gemma3VarianceGate.contractSupported, false, 'numerical variance does not borrow Supported')
assert.equal(gemma3VarianceGate.chatMode, 'variance', 'a qualified numerical-variance row gets its own runnable chat mode')
assert.equal(gemma3VarianceGate.label, 'Runnable (reference differs)')

// gemma3 1B Q8_0: the promoted row, and the counter-case to the gemma2 planning
// row directly above. The exact-row id is the normalized GGUF filename, so the
// identity matcher must resolve it from the local file's name alone and chat
// must unlock — the failure this pins is the one that shipped once already,
// where a supported row was demoted because the frontend could not join it.
const gemma3Hint = findCompatibilityHint(capabilityFixture, { name: 'gemma-3-1b-it-Q8_0', quant: 'Q8_0' })
assert.equal(gemma3Hint.kind, 'compatibility', 'the promoted gemma3 row must resolve as an exact compatibility match')
assert.equal(gemma3Hint.target.id, 'gemma_3_1b_it_q8_0')
assert.equal(gemma3Hint.exact, true, 'gemma3 must match by exact row id, not by family name')
assert.equal(isCompatibilitySupportedForModel(capabilityFixture, { name: 'gemma-3-1b-it-Q8_0', quant: 'Q8_0' }), true)
assert.equal(
  getChatGateState(capabilityFixture, { ...localLoadedReady, id: 'gemma3-1b', name: 'gemma-3-1b-it-Q8_0', quant: 'Q8_0' }, { active_model_id: 'gemma3-1b', loaded_now: true, generation_ready: true }).chatUnlocked,
  true,
  'the promoted gemma3 Q8_0 row must unlock chat when the runtime is loaded and generation-ready',
)
// The row is pinned to Q8_0: resident admission declines every other quant and
// serve falls back to the runnable CPU bridge, which has no window mask. A
// K-quant gemma3 file must never inherit this row's support.
assert.equal(
  isCompatibilitySupportedForModel(capabilityFixture, { name: 'gemma-3-1b-it-Q4_K_M', quant: 'Q4_K_M' }),
  false,
  'a non-Q8_0 gemma3 file must not inherit the Q8_0 row',
)
// The row advertises NO throughput evidence. Production-throughput readiness
// must stay guarded, and the row boundary copy must keep saying so.
const gemma3Row = capabilityFixture.model_compatibility.find((row) => row.id === 'gemma_3_1b_it_q8_0')
assert.equal(
  exactRowSupportLanes(gemma3Row, []).find((lane) => lane.key === 'throughput').ready,
  false,
  'a row that declines to claim throughput must not advertise production-throughput readiness',
)
assert.match(rowSupportBoundaryCopy(gemma3Row, []), /throughput/i)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /smoke-supported for local chat/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /\/api\/models\/load, \/v1\/completions, \/v1\/chat\/completions, frontend smoke, compact parity/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /current canonical GGUF \(sha256 f34112a1\.\.\., the file the June capability receipts pin\)/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /anchored by the checked 512\/1024\/2048\/4096\/8192 raw-decode context ladder/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /measured on a prior upload \(sha256 b5607b50\.\.\.\) and are retained as historical evidence/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /production throughput remains unpromoted/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /does not promote neighboring Llama sizes/)
assert.match(LLAMA32_3B_ACCEPTANCE_SUMMARY, /model-native\/larger contexts beyond the checked ladder/)
assert.match(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /does not currently show the exact 3B row/)
assert.doesNotMatch(LLAMA32_3B_ACCEPTANCE_AVAILABILITY, /not present locally yet/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /loaded_now=true and generation_ready=true/)
assert.match(LLAMA32_3B_ACCEPTANCE_GATING_NOTE, /exact supported Llama 3\.2 3B Q8_0 compatibility row/)

/* A family fallback must never name a row of a different model size. Several Qwen3
   sizes are certified under one family, and the fallback used to take the FIRST row
   whose id merely contains "qwen" — so a 4B file inherited the 0.6B row's id, status
   and evidence copy purely from array order. */
const multiSizeQwenFixture = {
  model_compatibility: [
    { id: 'qwen3_0_6b_instruct_q8_0', family: 'qwen3', quantization: 'Q8_0', status: 'supported_exact_row_smoke' },
    { id: 'qwen3_4b_instruct_q8_0', family: 'qwen3', quantization: 'Q8_0', status: 'supported_exact_row_smoke' },
    { id: 'qwen3_8b_instruct_q8_0', family: 'qwen3', quantization: 'Q8_0', status: 'supported_exact_row_smoke' },
  ],
}
const qwen4bFamilyHint = findCompatibilityHint(multiSizeQwenFixture, {
  id: 'Qwen3-4B-Q8_0.gguf',
  name: 'Qwen3-4B-Q8_0.gguf',
  model_path: 'Qwen3-4B-Q8_0.gguf',
  quant: 'Q8_0',
})
assert.equal(qwen4bFamilyHint.kind, 'family', 'a row id carrying a finetune token the filename omits stays a non-exact family hint')
assert.equal(
  qwen4bFamilyHint.target.id,
  'qwen3_4b_instruct_q8_0',
  'the family fallback must select the row matching the subject size, not the first same-family row in array order',
)
assert.equal(
  isCompatibilitySupportedForModel(multiSizeQwenFixture, { id: 'Qwen3-4B-Q8_0.gguf', quant: 'Q8_0' }),
  false,
  'a size-matched family hint is still advisory and must never unlock chat on its own',
)

/* When the subject states a size that no row covers, decline the row-specific hint
   rather than naming a confidently wrong neighbour. */
const qwen32bFamilyHint = findCompatibilityHint(multiSizeQwenFixture, {
  id: 'Qwen3-32B-Q8_0.gguf',
  name: 'Qwen3-32B-Q8_0.gguf',
  model_path: 'Qwen3-32B-Q8_0.gguf',
  quant: 'Q8_0',
})
assert.equal(qwen32bFamilyHint, null, 'an uncertified model size must not inherit a different size row as evidence')

console.log('✓ model-state smoke passed')
