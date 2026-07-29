/* Smoke for lib/modelLanes.js: lane membership must be derived from
   /api/models/local lane facts + the /api/capabilities contract, with the
   supported gate speaking only through isCompatibilitySupportedForModel —
   including its exact-artifact gate. Covers all four lanes and precedence. */

import { laneOf, bucketByLane, matchModel } from '../src/lib/modelLanes.js'

let failures = 0
function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected)
  if (!ok) {
    failures += 1
    console.error(`FAIL ${name}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`)
  } else {
    console.log(`ok ${name}`)
  }
}

const CAPABILITIES = {
  model_compatibility: [
    {
      id: 'llama32_3b_instruct_q8_0',
      family: 'llama_bpe_decoder',
      quantization: 'Q8_0',
      status: 'supported_exact_row_smoke',
    },
    // Real /api/capabilities row id: it carries the model's `general.finetune`
    // token ("Instruct"), which the canonical release filename
    // (Qwen3-4B-Q8_0.gguf) does NOT. The fixture must mirror the shipped row id,
    // or this file silently tests an identity match that production never makes.
    {
      id: 'qwen3_4b_instruct_q8_0',
      family: 'qwen3',
      quantization: 'Q8_0',
      status: 'supported_exact_row_smoke',
    },
    {
      id: 'gemma2_9b_it_q8_0',
      family: 'gemma2_decoder',
      quantization: 'Q8_0',
      status: 'tracked_not_supported',
    },
  ],
}

const entry = (filename, extra = {}) => ({
  filename,
  quantization: (filename.match(/\b(Q\d(?:_K_[MS]|_\d)?|BF16|F16|F32)\b/i) || [])[1] || '',
  runnable_receipt_present: false,
  admitted: false,
  oracle_qualified: false,
  ...extra,
})

// Lane 1: exact supported row (llama BPE identity + exact artifact filename).
check(
  'supported: exact row + exact artifact',
  laneOf(entry('Llama-3.2-3B-Instruct-Q8_0.gguf'), CAPABILITIES),
  'supported',
)

// The shipped row id carries a `general.finetune` token the canonical release
// filename omits (qwen3_4b_instruct_q8_0 vs Qwen3-4B-Q8_0.gguf), so the
// identity match cannot resolve it from the filename alone. The backend's
// exact-artifact verdict must still place it in the supported lane — otherwise
// a loadable, generation-ready model is demoted to Experimental and the UI
// offers no "Use for chat".
check(
  'supported: backend exact-artifact verdict when the row id carries a finetune token',
  laneOf(entry('Qwen3-4B-Q8_0.gguf', { lane_class: 'supported' }), CAPABILITIES),
  'supported',
)

// The backend verdict only ever promotes on an exact `supported`. An
// implemented-but-uncertified artifact keeps its evidence-derived lane.
check(
  'compatible: backend experimental verdict never promotes',
  laneOf(
    entry('Qwen3-4B-Q4_K_M.gguf', {
      lane_class: 'experimental_implemented',
      runnable_receipt_present: true,
    }),
    CAPABILITIES,
  ),
  'compatible',
)

// The artifact gate must hold: same model row, wrong GGUF filename punctuation
// (not the certified artifact) may not inherit the supported lane.
check(
  'not_anchored: near-miss artifact never inherits supported',
  laneOf(entry('Llama-3.2-3B-Instruct.Q8_0.gguf'), CAPABILITIES),
  'not_anchored',
)

// A matching row whose status is not supported_* must not promote.
check(
  'not_anchored: tracked (unsupported) row never promotes',
  laneOf(entry('gemma-2-9b-it-Q8_0.gguf'), CAPABILITIES),
  'not_anchored',
)

// Lane 2: runnable receipt present → compatible.
check(
  'compatible: runnable receipt present',
  laneOf(entry('ornith-1.0-9b-Q8_0.gguf', { runnable_receipt_present: true }), CAPABILITIES),
  'compatible',
)

// Lane 3: admitted + oracle-qualified, not yet smoked → eligible.
check(
  'eligible: admitted + oracle_qualified',
  laneOf(entry('phi-3-mini-F32.gguf', { admitted: true, oracle_qualified: true }), CAPABILITIES),
  'eligible',
)

// admitted alone (not oracle-qualified) is NOT eligible.
check(
  'not_anchored: admitted without oracle_qualified',
  laneOf(entry('phi-3-mini-Q8_0.gguf', { admitted: true }), CAPABILITIES),
  'not_anchored',
)

// Lane 4: no facts at all.
check('not_anchored: no evidence', laneOf(entry('mystery-model-Q4_K_M.gguf'), CAPABILITIES), 'not_anchored')

// Precedence: supported wins over a runnable receipt.
check(
  'precedence: supported beats compatible',
  laneOf(entry('Llama-3.2-3B-Instruct-Q8_0.gguf', { runnable_receipt_present: true }), CAPABILITIES),
  'supported',
)

// No capabilities at all → nothing may claim supported.
check(
  'no contract: supported artifact stays un-promoted',
  laneOf(entry('Llama-3.2-3B-Instruct-Q8_0.gguf'), null),
  'not_anchored',
)

// bucketByLane groups every model exactly once, in lane order.
const models = [
  entry('Llama-3.2-3B-Instruct-Q8_0.gguf'),
  entry('ornith-1.0-9b-Q8_0.gguf', { runnable_receipt_present: true }),
  entry('phi-3-mini-F32.gguf', { admitted: true, oracle_qualified: true }),
  entry('mystery-model-Q4_K_M.gguf'),
]
const buckets = bucketByLane(models, CAPABILITIES)
check(
  'bucketByLane: one model per lane',
  {
    supported: buckets.supported.map((m) => m.filename),
    compatible: buckets.compatible.map((m) => m.filename),
    eligible: buckets.eligible.map((m) => m.filename),
    not_anchored: buckets.not_anchored.map((m) => m.filename),
  },
  {
    supported: ['Llama-3.2-3B-Instruct-Q8_0.gguf'],
    compatible: ['ornith-1.0-9b-Q8_0.gguf'],
    eligible: ['phi-3-mini-F32.gguf'],
    not_anchored: ['mystery-model-Q4_K_M.gguf'],
  },
)

// matchModel keeps the contract matcher's expected shape.
check(
  'matchModel shape',
  matchModel(entry('a.gguf')),
  { id: 'a.gguf', name: 'a.gguf', model_path: 'a.gguf', hf_filename: 'a.gguf', quant: '' },
)

if (failures) {
  console.error(`\n${failures} failure(s)`)
  process.exit(1)
}
console.log('\nmodel-lanes smoke: all checks passed')
