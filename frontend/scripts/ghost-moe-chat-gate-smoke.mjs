#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { getChatGateState } from '../src/lib/chatGate.js'

const MINI2_RUNTIME_PROFILE = 'mini2-h71r-h58-h60-h62-1408-ctx1024-mtp15-adaptive-v1'

const gemma426bRow = {
  id: 'gemma4_26b_a4b_it_q4_0',
  family: 'gemma4_a4b_moe_decoder',
  quantization: 'Q4_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_distributed_or_apple_m4_full_metal_ghost_moe_smoke_only',
  evidence: 'distributed serve receipt',
}

const residentRow = {
  id: 'gemma4_e4b_it_q8_0',
  family: 'gemma4_decoder',
  quantization: 'Q8_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_resident_serve_smoke_only',
  evidence: 'resident serve receipt',
}

// Keep the resident row first: generic Gemma family fallback ordering must not
// make the 26B lane inherit or display the wrong row.
const capabilities = { model_compatibility: [residentRow, gemma426bRow] }
const model26b = {
  id: 'gemma-4-26B_q4_0-it.gguf',
  runtime_model_name: 'gemma-4-26B_q4_0-it.gguf',
  catalog_id: gemma426bRow.id,
  name: 'Gemma 4 26B A4B Q4_0',
  quant: 'Q4_0',
  lane_class: 'supported',
  provider_kind: 'local',
  model_path: 'models/gemma-4-26B_q4_0-it.gguf',
  status: 'ready',
  loaded_now: true,
  generation_ready: true,
}
const baseRuntime = {
  status: 'online',
  loaded_now: true,
  generation_ready: true,
  active_model_id: model26b.runtime_model_name,
  backend: 'gemma4-runtime',
}

for (const gemma4_serve_lane of ['ghost_moe', 'local', 'cuda', undefined, 'future_lane']) {
  const gate = getChatGateState(
    capabilities,
    model26b,
    { ...baseRuntime, gemma4_serve_lane },
  )
  assert.equal(gate.contractSupported, false, `${gemma4_serve_lane || 'absent'} must not inherit distributed support`)
  assert.equal(gate.chatUnlocked, false)
  assert.equal(gate.experimentalUnlocked, true)
  assert.equal(gate.chatMode, 'experimental')
  assert.equal(gate.hint?.target?.status, 'experimental_runtime_lane')
  assert.match(gate.label, /unverified/i)
  assert.match(gate.copy, /not been verified|unverified/i)
}

const catalogCudaGhostRuntime = {
  ...baseRuntime,
  gemma4_serve_lane: 'ghost_moe',
  gemma4_ghost_catalog_managed: true,
  gemma4_ghost_backend: 'cuda',
  gemma4_ghost_common_gpu_active: true,
  gemma4_ghost_experts_gpu_active: true,
  gemma4_ghost_head_gpu_active: true,
  execution_plan: {
    operating_system: 'windows',
    architecture: 'x86_64',
    support_level: 'supported_exact_row_smoke',
    selected_backend: 'gemma4_ghost_moe_cuda_runtime',
    prefill_path: 'gemma4_ghost_moe_cuda_prefill',
    decode_path: 'gemma4_ghost_moe_cuda_decode',
  },
}
const catalogCudaGhost = getChatGateState(capabilities, model26b, catalogCudaGhostRuntime)
assert.equal(catalogCudaGhost.contractSupported, false, 'v1 catalog markers cannot verify current CUDA artifact bytes')
assert.equal(catalogCudaGhost.chatUnlocked, false)
assert.equal(catalogCudaGhost.experimentalUnlocked, true)
assert.equal(catalogCudaGhost.chatMode, 'experimental')

function catalogCudaRuntimeWith({ plan, ...patch }) {
  return {
    ...catalogCudaGhostRuntime,
    ...patch,
    execution_plan: plan === null
      ? null
      : { ...catalogCudaGhostRuntime.execution_plan, ...(plan || {}) },
  }
}

for (const [label, runtime] of [
  ['catalog marker absent', catalogCudaRuntimeWith({ gemma4_ghost_catalog_managed: false })],
  ['wrong Ghost backend', catalogCudaRuntimeWith({ gemma4_ghost_backend: 'metal' })],
  ['common CUDA inactive', catalogCudaRuntimeWith({ gemma4_ghost_common_gpu_active: false })],
  ['experts CUDA inactive', catalogCudaRuntimeWith({ gemma4_ghost_experts_gpu_active: false })],
  ['head CUDA inactive', catalogCudaRuntimeWith({ gemma4_ghost_head_gpu_active: false })],
  ['missing CUDA execution plan', catalogCudaRuntimeWith({ plan: null })],
  ['wrong CUDA operating system', catalogCudaRuntimeWith({ plan: { operating_system: 'linux' } })],
  ['wrong CUDA architecture', catalogCudaRuntimeWith({ plan: { architecture: 'aarch64' } })],
  ['unverified CUDA execution plan', catalogCudaRuntimeWith({ plan: { support_level: 'unknown_or_unvalidated' } })],
  ['wrong CUDA selected backend', catalogCudaRuntimeWith({ plan: { selected_backend: 'gemma4_cpu_runtime' } })],
  ['wrong CUDA prefill path', catalogCudaRuntimeWith({ plan: { prefill_path: 'gemma4_cpu_prefill' } })],
  ['wrong CUDA decode path', catalogCudaRuntimeWith({ plan: { decode_path: 'gemma4_cpu_decode' } })],
]) {
  const gate = getChatGateState(capabilities, model26b, runtime)
  assert.equal(gate.contractSupported, false, `${label} must keep the catalog CUDA Ghost shape experimental`)
  assert.equal(gate.chatMode, 'experimental')
}

const appleM4MetalGhostRuntime = {
  ...baseRuntime,
  gemma4_serve_lane: 'ghost_moe',
  gemma4_ghost_catalog_managed: false,
  gemma4_ghost_backend: 'metal',
  gemma4_ghost_execution_mode: 'full_common_metal',
  gemma4_ghost_common_metal_active: true,
  gemma4_ghost_experts_metal_active: true,
  gemma4_ghost_head_metal_active: true,
  gemma4_ghost_common_gpu_active: true,
  gemma4_ghost_experts_gpu_active: true,
  gemma4_ghost_head_gpu_active: true,
  gemma4_mtp_assistant_loaded: true,
  gemma4_mtp_full_q4_active: true,
  gemma4_ghost_exact_expert_policy_active: true,
  gemma4_ghost_common_metal_context_capacity: 1024,
  gemma4_ghost_runtime_profile: MINI2_RUNTIME_PROFILE,
  execution_plan: {
    operating_system: 'macos',
    architecture: 'aarch64',
    cpu_model: 'Apple M4',
    support_level: 'supported_exact_row_smoke',
    selected_backend: 'gemma4_ghost_moe_metal_runtime',
    prefill_path: 'gemma4_ghost_moe_metal_prefill',
    decode_path: 'gemma4_ghost_moe_metal_speculative_decode',
  },
}
const appleM4MetalGhost = getChatGateState(capabilities, model26b, appleM4MetalGhostRuntime)
assert.equal(appleM4MetalGhost.contractSupported, true, 'the complete Apple M4 full-Metal/MTP Ghost shape is supported')
assert.equal(appleM4MetalGhost.chatUnlocked, true)
assert.equal(appleM4MetalGhost.experimentalUnlocked, false)
assert.equal(appleM4MetalGhost.chatMode, 'supported')
assert.equal(appleM4MetalGhost.label, 'Verified')
assert.equal(appleM4MetalGhost.hint?.target?.status, 'supported_exact_row_smoke')

function appleM4RuntimeWith({ plan, ...patch }) {
  return {
    ...appleM4MetalGhostRuntime,
    ...patch,
    execution_plan: plan === null
      ? null
      : { ...appleM4MetalGhostRuntime.execution_plan, ...(plan || {}) },
  }
}

for (const [label, runtime] of [
  ['wrong lane', appleM4RuntimeWith({ gemma4_serve_lane: 'local' })],
  ['wrong runtime backend', appleM4RuntimeWith({ backend: 'runnable-runtime' })],
  ['missing execution plan', appleM4RuntimeWith({ plan: null })],
  ['wrong operating system', appleM4RuntimeWith({ plan: { operating_system: 'windows' } })],
  ['wrong architecture', appleM4RuntimeWith({ plan: { architecture: 'x86_64' } })],
  ['wrong Apple host', appleM4RuntimeWith({ plan: { cpu_model: 'Apple M3' } })],
  ['unverified execution plan', appleM4RuntimeWith({ plan: { support_level: 'unknown_or_unvalidated' } })],
  ['wrong selected backend', appleM4RuntimeWith({ plan: { selected_backend: 'gemma4_cpu_runtime' } })],
  ['wrong prefill path', appleM4RuntimeWith({ plan: { prefill_path: 'gemma4_cpu_prefill' } })],
  ['wrong decode path', appleM4RuntimeWith({ plan: { decode_path: 'gemma4_cpu_decode' } })],
  ['wrong Ghost backend', appleM4RuntimeWith({ gemma4_ghost_backend: 'cuda' })],
  ['partial execution mode', appleM4RuntimeWith({ gemma4_ghost_execution_mode: 'hybrid_metal' })],
  ['common Metal inactive', appleM4RuntimeWith({ gemma4_ghost_common_metal_active: false })],
  ['experts Metal inactive', appleM4RuntimeWith({ gemma4_ghost_experts_metal_active: false })],
  ['head Metal inactive', appleM4RuntimeWith({ gemma4_ghost_head_metal_active: false })],
  ['common GPU inactive', appleM4RuntimeWith({ gemma4_ghost_common_gpu_active: false })],
  ['experts GPU inactive', appleM4RuntimeWith({ gemma4_ghost_experts_gpu_active: false })],
  ['head GPU inactive', appleM4RuntimeWith({ gemma4_ghost_head_gpu_active: false })],
  ['MTP assistant absent', appleM4RuntimeWith({ gemma4_mtp_assistant_loaded: false })],
  ['MTP assistant not full Q4', appleM4RuntimeWith({ gemma4_mtp_full_q4_active: false })],
  ['exact expert policy absent', appleM4RuntimeWith({ gemma4_ghost_exact_expert_policy_active: false })],
  ['Metal context absent', appleM4RuntimeWith({ gemma4_ghost_common_metal_context_capacity: null })],
  ['Metal context undersized', appleM4RuntimeWith({ gemma4_ghost_common_metal_context_capacity: 512 })],
  ['Metal context malformed', appleM4RuntimeWith({ gemma4_ghost_common_metal_context_capacity: '1024' })],
  ['Mini2 runtime profile absent', appleM4RuntimeWith({ gemma4_ghost_runtime_profile: null })],
  ['Mini2 runtime profile wrong', appleM4RuntimeWith({ gemma4_ghost_runtime_profile: 'mini2-future-profile' })],
]) {
  const gate = getChatGateState(capabilities, model26b, runtime)
  assert.equal(gate.contractSupported, false, `${label} must keep the Metal Ghost shape experimental`)
  assert.equal(gate.chatMode, 'experimental')
  assert.equal(gate.hint?.target?.status, 'experimental_runtime_lane')
}

// The active hot pair lives outside the catalog scan, so the dashboard model has
// no lane_class. Real Q4_0 metadata plus the complete live M4 attestation and
// hash/host-reconciled execution plan is the narrowly scoped fallback that fixes
// that exact direct-load case.
const hotModel = {
  ...model26b,
  id: '26B_dequant_it_hf',
  runtime_model_name: '26B_dequant_it_hf',
  catalog_id: undefined,
  name: 'gemma-4-26B_q4_0-it.hot',
  model_path: '/models/gemma-4-26B_q4_0-it.hot.gguf',
  lane_class: undefined,
}
const hotRuntime = { ...appleM4MetalGhostRuntime, active_model_id: hotModel.runtime_model_name }
const hotGate = getChatGateState(capabilities, hotModel, hotRuntime)
assert.equal(hotGate.contractSupported, true, 'the evidenced direct-load hot pair must not be mislabeled unverified')
assert.equal(hotGate.chatUnlocked, true)
assert.equal(hotGate.chatMode, 'supported')
assert.equal(hotGate.label, 'Verified')

const explicitExperimentalHotGate = getChatGateState(
  capabilities,
  { ...hotModel, lane_class: 'experimental_implemented' },
  hotRuntime,
)
assert.equal(explicitExperimentalHotGate.contractSupported, false, 'live evidence must not override an explicit backend artifact rejection')
assert.equal(explicitExperimentalHotGate.chatMode, 'experimental')

const wrongQuantHotGate = getChatGateState(
  capabilities,
  {
    ...hotModel,
    name: 'gemma-4-26B-q8_0.hot',
    model_path: '/models/gemma-4-26B-q8_0.hot.gguf',
    quant: 'Q8_0',
  },
  hotRuntime,
)
assert.equal(wrongQuantHotGate.contractSupported, false, 'the M4 runtime shape must not promote a neighboring quant')
assert.equal(wrongQuantHotGate.chatMode, 'experimental')

const distributed = getChatGateState(
  capabilities,
  model26b,
  { ...baseRuntime, gemma4_serve_lane: 'distributed' },
)
assert.equal(distributed.contractSupported, true, 'the evidenced distributed lane stays supported')
assert.equal(distributed.chatUnlocked, true)
assert.equal(distributed.experimentalUnlocked, false)
assert.equal(distributed.chatMode, 'supported')
assert.equal(distributed.hint?.target?.status, 'supported_exact_row_smoke')

const residentModel = {
  ...model26b,
  id: 'gemma-4-E4B-it-Q8_0.gguf',
  runtime_model_name: 'gemma-4-E4B-it-Q8_0.gguf',
  catalog_id: residentRow.id,
  name: 'Gemma 4 E4B Q8_0',
  quant: 'Q8_0',
  model_path: 'models/gemma-4-E4B-it-Q8_0.gguf',
}
const resident = getChatGateState(
  capabilities,
  residentModel,
  {
    ...baseRuntime,
    active_model_id: residentModel.runtime_model_name,
    gemma4_serve_lane: 'local',
  },
)
assert.equal(resident.contractSupported, true, 'a genuinely supported resident row is unchanged')
assert.equal(resident.chatMode, 'supported')
assert.equal(resident.chatUnlocked, true)

const dashboardSource = readFileSync(
  new URL('../src/hooks/useDashboardData.js', import.meta.url),
  'utf8',
)
assert.match(
  dashboardSource,
  /gemma4_serve_lane:\s*optionalString\(health\?\.gemma4_serve_lane\)/,
  'health lane must survive the dashboard projection used by every chat gate',
)

const executionPlanSource = readFileSync(
  new URL('../src/lib/executionPlan.js', import.meta.url),
  'utf8',
)
assert.match(
  executionPlanSource,
  /gemma4_ghost_catalog_managed:\s*optionalBoolean\(health\?\.gemma4_ghost_catalog_managed\)/,
  'catalog-managed Ghost truth must survive the health projection used by chat gating',
)
assert.match(
  executionPlanSource,
  /gemma4_mtp_assistant_loaded:\s*optionalBoolean\(health\?\.gemma4_mtp_assistant_loaded\)/,
  'MTP assistant truth must survive the health projection used by chat gating',
)
assert.match(
  executionPlanSource,
  /gemma4_mtp_full_q4_active:\s*optionalBoolean\(health\?\.gemma4_mtp_full_q4_active\)/,
  'full-Q4 MTP truth must survive the health projection used by chat gating',
)
assert.match(
  executionPlanSource,
  /gemma4_ghost_exact_expert_policy_active:\s*optionalBoolean\(health\?\.gemma4_ghost_exact_expert_policy_active\)/,
  'exact expert-policy truth must survive the health projection used by chat gating',
)
assert.match(
  executionPlanSource,
  /gemma4_ghost_common_metal_context_capacity:\s*optionalPositiveInteger\(health\?\.gemma4_ghost_common_metal_context_capacity\)/,
  'the live Metal context capacity must survive the health projection used by send budgeting',
)
assert.match(
  executionPlanSource,
  /gemma4_ghost_runtime_profile:\s*health\?\.gemma4_ghost_runtime_profile === GEMMA4_MINI2_WEBUI_PROFILE_ID/,
  'only the exact Mini2 runtime-profile receipt must survive the health projection',
)

console.log('ghost-moe chat gate smoke: ok')
