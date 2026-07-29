import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel } from './capabilities.js'
import { isRunnableInCurrentRuntime, modelRuntimeIdMatches } from './modelState.js'

/* The backend's exact-artifact verdict for this model's GGUF, from
   `/api/models/local`. `classify_model_lane()` requires BOTH an implemented
   `general.architecture` parsed from the real header AND
   `filename_is_supported_exact_row()` — an exact filename equality against the
   curated catalog whose row id must also be `supported_*`. That conjunction is
   strictly stronger evidence than matching a display name against a row id, and
   lib/modelLanes.js has treated it as authoritative for the Models page since the
   same problem was found there.

   It is required here for the same reason: a compatibility row id concatenates the
   model's `general.finetune` token that the release filename may omit
   (`qwen3_0_6b_instruct_q8_0` vs `Qwen3-0.6B-Q8_0.gguf`). Whether the identity match
   could see that token depended on which path issued the load — the engine's startup
   auto-load names a model from GGUF metadata, `POST /api/models/load` names it
   whatever id the caller sent — so one file produced two contradictory claims
   ("Local chat ready" vs "unverified, no parity guarantee") on the same machine.

   It only ever ADDS the contract's own verdict for an exact artifact; it cannot
   unlock chat by itself, because `chatUnlocked` still requires runtime readiness. */
function backendMarksSupportedRow(model) {
  return model?.lane_class === 'supported'
}

export function getChatGateState(capabilities, model, runtime) {
  const runtimeLoaded = Boolean(runtime?.loaded_now && modelRuntimeIdMatches(model, runtime))
  const runtimeGenerationReady = Boolean(runtime?.generation_ready && modelRuntimeIdMatches(model, runtime))
  const runtimeReady = Boolean(isRunnableInCurrentRuntime(model, runtime) && runtimeLoaded && runtimeGenerationReady)
  const hint = findCompatibilityHint(capabilities, model)
  const contractSupported = backendMarksSupportedRow(model)
    || isCompatibilitySupportedForModel(capabilities, model)
  const chatUnlocked = Boolean(runtimeReady && contractSupported)
  // Experimental lane: the model loaded and is generation-ready (so its architecture
  // is implemented — generation_ready is false for unimplemented archs) but it is NOT
  // a supported contract row. A separate, weaker affordance from the supported gate:
  // chat is allowed but every turn is marked unverified with no parity claim.
  const experimentalUnlocked = Boolean(runtimeReady && !contractSupported)
  const chatMode = contractSupported ? 'supported' : experimentalUnlocked ? 'experimental' : 'blocked'

  return {
    hint,
    runtimeReady,
    runtimeLoaded,
    runtimeGenerationReady,
    contractSupported,
    chatUnlocked,
    experimentalUnlocked,
    chatMode,
    label: compatibilityHintLabel(hint, 'No matching COMPATIBILITY.md row'),
    copy: compatibilityHintCopy(hint),
  }
}
