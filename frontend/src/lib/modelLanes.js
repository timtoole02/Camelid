import {
  isCompatibilitySupportedForModel,
  isCompatibilityNumericalVarianceRunnableForModel,
  isCompatibilityVerifiedRunnableForModel,
} from './capabilities.js'

/* Derived lane membership for local models — extracted verbatim from
   LocalLaneSections so every consumer computes lanes the same way. Membership is
   computed from /api/models/local lane facts + the /api/capabilities contract —
   never a hand-authored array. */

export const LANES = ['supported', 'compatible', 'eligible', 'not_anchored']

/* A model object shaped for the existing contract matcher (it reads id/name/
   model_path/quant). The supported gate stays the contract's voice — we only ask it. */
export function matchModel(entry) {
  return {
    id: entry.filename,
    name: entry.filename,
    model_path: entry.filename,
    hf_filename: entry.filename,
    quant: entry.quantization,
  }
}

/* The backend's own exact-artifact verdict for this local file. `lane_class`
   comes from /api/models/local, where classify_model_lane() requires BOTH an
   implemented `general.architecture` (parsed from the real header, never a
   filename guess) AND filename_is_supported_exact_row() — an exact filename
   equality against the curated catalog whose row id must also be in
   supported_compatibility_row_ids(). That conjunction is strictly stronger
   evidence than the frontend's subject-string matching, and it is the same
   support contract resolved where the curated catalog actually lives.

   It is needed because a compatibility row id concatenates the model's
   `general.finetune` token, which the canonical release filename may omit:
   `qwen3_4b_instruct_q8_0` vs `Qwen3-4B-Q8_0.gguf`. The local-lane record
   carries only the filename, so the identity match cannot see "instruct" and
   the file falls to a non-exact family match — demoting a genuinely supported,
   loadable model into Experimental, where no "Use for chat" is offered. */
function backendMarksSupported(entry) {
  return entry?.lane_class === 'supported'
}

export function laneOf(entry, capabilities) {
  if (backendMarksSupported(entry)) return 'supported'
  if (entry?.lane_class === 'runnable_with_variance') return 'compatible'
  // An explicit backend verdict is authoritative. In particular, a hash-pinned
  // filename (Prism, the non-catalog allowlist, and the curated rows carrying a
  // recorded digest) remains experimental until the server has verified its
  // loaded digest, and stays experimental if those bytes are not the certified
  // ones; the looser frontend name/quant matcher must not promote it back.
  if (!entry?.lane_class && isCompatibilitySupportedForModel(capabilities, matchModel(entry))) return 'supported'
  // An exact row that passed load + deterministic parity + guarded API/WebUI is
  // already verified runnable even while extended-context evidence keeps it out
  // of the full Supported lane. Do not collapse it back into an unverified bucket.
  if (isCompatibilityVerifiedRunnableForModel(capabilities, matchModel(entry))) return 'compatible'
  if (isCompatibilityNumericalVarianceRunnableForModel(capabilities, matchModel(entry))) return 'compatible'
  if (entry.runnable_receipt_present) return 'compatible'
  if (entry.admitted && entry.oracle_qualified) return 'eligible'
  return 'not_anchored'
}

export function bucketByLane(models = [], capabilities) {
  const buckets = { supported: [], compatible: [], eligible: [], not_anchored: [] }
  for (const m of models) buckets[laneOf(m, capabilities)].push(m)
  return buckets
}
