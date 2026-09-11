#!/usr/bin/env node
import assert from 'node:assert/strict'

import { findCompatibilityHint } from '../src/lib/capabilities.js'
import { getChatGateState } from '../src/lib/chatGate.js'

const olderQ8Row = {
  id: 'gemma4_12b_it_q8_0',
  family: 'gemma4_dense_decoder',
  quantization: 'Q8_0',
  status: 'supported_exact_row_smoke',
  evidence: 'older distributed Q8 row',
}
const mtp12Row = {
  id: 'gemma4_12b_it_qat_q4_0_mtp12',
  family: 'gemma4_dense_decoder_mtp12',
  quantization: 'Q4_0',
  status: 'supported_exact_row_smoke',
  support_scope: 'exact_row_single_node_macos_metal_lossless_mtp12_performance_smoke_only',
  evidence: 'exact target plus exact MTP12 assistant receipt',
}
// Put the tempting older row first. Exact QAT artifact identity must win over
// generic Gemma 12B family/size ordering.
const capabilities = { model_compatibility: [olderQ8Row, mtp12Row] }
const exactMtp12Model = {
  id: 'gemma-4-12b-it-qat-q4_0.gguf',
  runtime_model_name: 'gemma-4-12b-it-qat-q4_0.gguf',
  name: 'Gemma 4 12B It QAT Q4_0',
  model_path: '/models/gemma-4-12b-it-qat-q4_0.gguf',
  quant: 'Q4_0',
  lane_class: 'supported', // backend SHA + host + exact-row verdict
  provider_kind: 'local',
  status: 'ready',
  loaded_now: true,
  generation_ready: true,
}
const baseRuntime = {
  status: 'online',
  loaded_now: true,
  generation_ready: true,
  active_model_id: exactMtp12Model.runtime_model_name,
  backend: 'gemma4-runtime',
}

const exactHint = findCompatibilityHint(capabilities, exactMtp12Model)
assert.equal(exactHint?.kind, 'compatibility')
assert.equal(exactHint?.exact, true)
assert.equal(
  exactHint?.target?.id,
  mtp12Row.id,
  'the exact non-catalog QAT filename must resolve the MTP12 row, never the older 12B Q8 row',
)
assert.equal(
  findCompatibilityHint(capabilities, { ...exactMtp12Model, catalog_id: olderQ8Row.id })?.target?.id,
  mtp12Row.id,
  'the exact QAT filename must outrank a stale saved 12B Q8 catalog id',
)

const supported = getChatGateState(capabilities, exactMtp12Model, {
  ...baseRuntime,
  gemma4_serve_lane: 'mtp12_metal',
})
assert.equal(supported.contractSupported, true)
assert.equal(supported.chatUnlocked, true)
assert.equal(supported.chatMode, 'supported')
assert.equal(supported.hint?.target?.id, mtp12Row.id)

for (const gemma4_serve_lane of ['local', 'distributed', 'cuda', 'ghost_moe', undefined, 'future_lane']) {
  const gate = getChatGateState(capabilities, exactMtp12Model, {
    ...baseRuntime,
    gemma4_serve_lane,
  })
  assert.equal(gate.contractSupported, false, `${gemma4_serve_lane || 'absent'} must not inherit MTP12 support`)
  assert.equal(gate.chatUnlocked, false)
  assert.equal(gate.chatMode, 'experimental')
  assert.equal(gate.hint?.target?.id, mtp12Row.id)
  assert.equal(gate.hint?.target?.status, 'experimental_runtime_lane')
}

const wrongBackend = getChatGateState(capabilities, exactMtp12Model, {
  ...baseRuntime,
  backend: 'runnable-runtime',
  gemma4_serve_lane: 'mtp12_metal',
})
assert.equal(wrongBackend.contractSupported, false, 'the MTP12 lane name on another backend must fail closed')
assert.equal(wrongBackend.chatMode, 'experimental')

for (const lane_class of [undefined, 'experimental', 'runnable_with_variance']) {
  const gate = getChatGateState(
    capabilities,
    { ...exactMtp12Model, lane_class },
    { ...baseRuntime, gemma4_serve_lane: 'mtp12_metal' },
  )
  assert.equal(gate.contractSupported, false, `backend artifact verdict ${lane_class || 'absent'} must not satisfy the hash-pinned row`)
  assert.equal(gate.chatMode, lane_class === 'runnable_with_variance' ? 'variance' : 'experimental')
  assert.equal(gate.hint?.target?.id, mtp12Row.id, 'Evidence Chip may identify the row while still failing its runtime/artifact gate')
}

const neighboringArtifact = {
  ...exactMtp12Model,
  id: 'gemma-4-12b-it-qat-q4_0-neighbor.gguf',
  runtime_model_name: 'gemma-4-12b-it-qat-q4_0-neighbor.gguf',
  model_path: '/models/gemma-4-12b-it-qat-q4_0-neighbor.gguf',
  lane_class: 'experimental',
}
const neighborHint = findCompatibilityHint(capabilities, neighboringArtifact)
assert.notEqual(neighborHint?.target?.id, mtp12Row.id, 'same display name and quant cannot forge the exact MTP12 Evidence Chip')

const olderQ8Model = {
  ...exactMtp12Model,
  id: 'gemma-4-12b-it-Q8_0.gguf',
  runtime_model_name: 'gemma-4-12b-it-Q8_0.gguf',
  model_path: '/models/gemma-4-12b-it-Q8_0.gguf',
  name: 'Gemma 4 12B It Q8_0',
  quant: 'Q8_0',
}
assert.equal(
  findCompatibilityHint(capabilities, olderQ8Model)?.target?.id,
  olderQ8Row.id,
  'the existing Q8 artifact remains on its own exact row',
)

console.log('Gemma 4 12B MTP12 chat gate smoke: ok')
