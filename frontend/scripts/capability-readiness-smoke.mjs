/* Smoke for capabilityReadiness.js: the readiness taxonomy must come from the
   row's own contract fields, never from family-name matching, and multimodal
   input must always classify as unsupported. */

import {
  classifyCapabilityRow,
  classifyInputModality,
  isExactRowSupported,
  readinessLabel,
  READINESS,
} from '../src/lib/capabilityReadiness.js'

let failures = 0
function check(name, actual, expected) {
  const ok = actual === expected
  if (!ok) {
    failures += 1
    console.error(`FAIL ${name}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`)
  } else {
    console.log(`ok ${name}`)
  }
}

check(
  'supported exact row',
  classifyCapabilityRow({ id: 'gemma4_e2b_it_q8_0', status: 'supported_exact_row_smoke' }),
  READINESS.SUPPORTED_EXACT_ROW,
)
check(
  'active validation',
  classifyCapabilityRow({ id: 'mixtral_8x7b_instruct_v0_1_q8_0', status: 'active_validation_partial_runtime' }),
  READINESS.ACTIVE_VALIDATION,
)
check(
  'runnable exact row with reference variance',
  classifyCapabilityRow({ id: 'gemma3_4b_it_q8_0', status: 'runnable_exact_row_numerical_variance' }),
  READINESS.RUNNABLE_VARIANCE,
)
// Real shipped row (gemma3 -> Metal, then -> CUDA): a GPU-resident windowed
// lane is the default serve lane for this row on BOTH GPU backends, and the row
// carries NO throughput claim on either.
// `performance_measured` must therefore never read as measured evidence — a row
// that declines to claim perf must not be classified as GPU-experimental or
// promoted to a throughput-ready lane. Field values copied from the shipped
// /api/capabilities contract, not paraphrased.
check(
  'supported exact row: gemma3 1B Q8_0 on a GPU-resident windowed lane',
  classifyCapabilityRow({
    id: 'gemma_3_1b_it_q8_0',
    status: 'supported_exact_row_smoke',
    performance_measured: 'not_claimed_resident_lane_throughput_is_a_separate_unshipped_measurement_phase',
  }),
  READINESS.SUPPORTED_EXACT_ROW,
)
check(
  'planned candidate',
  classifyCapabilityRow({ id: 'gemma2_9b', status: 'planned_exact_row_candidate' }),
  READINESS.PLANNED,
)
check(
  'unsupported quantization',
  classifyCapabilityRow({ id: 'llama_spm_q4_0_q5_0', status: 'planned_phase_10', tensors_load: 'unsupported_typed_error' }),
  READINESS.UNSUPPORTED_QUANTIZATION,
)
check(
  'gpu experimental stays distinct from green',
  classifyCapabilityRow({ id: 'x', status: 'supported_exact_row_smoke', performance_measured: 'gpu_experimental_parity_pending' }),
  READINESS.GPU_EXPERIMENTAL,
)
check('image input fails closed', classifyInputModality('image'), READINESS.UNSUPPORTED_MULTIMODAL)
check('audio input fails closed', classifyInputModality('audio'), READINESS.UNSUPPORTED_MULTIMODAL)
check('video input fails closed', classifyInputModality('video'), READINESS.UNSUPPORTED_MULTIMODAL)
check('text input is allowed', classifyInputModality('text'), READINESS.SUPPORTED_EXACT_ROW)

const capabilities = {
  model_compatibility: [
    { id: 'gemma4_e4b_it_q8_0', status: 'supported_exact_row_smoke' },
    { id: 'gemma4_e2b_it_q8_0', status: 'supported_exact_row_smoke' },
    { id: 'gemma_3_1b_it_q8_0', status: 'supported_exact_row_smoke' },
  ],
}
check('exact row id supported', isExactRowSupported(capabilities, 'gemma4_e2b_it_q8_0'), true)
// The shipped gemma3 row id, verbatim. The near-miss spelling below is the one
// that actually shipped in the download catalog for a while and joined nothing.
check('exact row id supported: gemma3', isExactRowSupported(capabilities, 'gemma_3_1b_it_q8_0'), true)
check(
  'the historical gemma3 id spelling must NOT resolve',
  isExactRowSupported(capabilities, 'gemma3_1b_it_q8_0'),
  false,
)
check(
  'family-name prefix must NOT count as supported',
  isExactRowSupported(capabilities, 'gemma4_12b_it_q8_0'),
  false,
)
check('family string never matches', isExactRowSupported(capabilities, 'gemma4'), false)
check('gemma3 family string never matches', isExactRowSupported(capabilities, 'gemma3'), false)
check('label text', readinessLabel(READINESS.UNSUPPORTED_MULTIMODAL), 'Unsupported: multimodal input')

if (failures > 0) {
  console.error(`${failures} capability-readiness checks failed`)
  process.exit(1)
}
console.log('capability-readiness smoke passed')
