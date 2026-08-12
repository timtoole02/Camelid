import {
  compatibilityHintCopy,
  findCompatibilityHint,
  isCompatibilityNumericalVarianceRunnableForModel,
  isCompatibilitySupportedForModel,
  isCompatibilityVerifiedRunnableForModel,
} from './capabilities.js'
import { isEmbeddingOnlyModel, modelTaskKind } from './modelCapabilities.js'
import { isModelLoadedNow, isRunnableInCurrentRuntime, modelRuntimeIdMatches } from './modelState.js'

/* The backend's exact-artifact verdict for this model's GGUF, from
   `/api/models/local`. `classify_model_lane()` requires BOTH an implemented
   `general.architecture` parsed from the real header AND
   `filename_is_supported_exact_row()` — an exact filename equality against the
   curated catalog whose row id must also be `supported_*` — plus, for the
   hash-pinned artifacts (Prism, the non-catalog allowlist, and the curated
   rows carrying a recorded digest), the certified sha256 of the bytes
   actually loaded. That conjunction is
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

const GEMMA4_26B_LANE_SCOPED_ROW = 'gemma4_26b_a4b_it_q4_0'
const GEMMA4_SERVE_LANES = new Set(['ghost_moe', 'local', 'distributed', 'cuda'])
const LFM2_26B_LANE_SCOPED_ROW = 'lfm2_5_2_6b_q8_0'

function normalizedGemma4ServeLane(runtime) {
  const lane = String(runtime?.gemma4_serve_lane || '').trim().toLowerCase().replace(/-/g, '_')
  return GEMMA4_SERVE_LANES.has(lane) ? lane : null
}

function isGemma426bLaneScopedRow(model, hint) {
  if (hint?.kind === 'compatibility' && hint?.exact === true
    && hint?.target?.id === GEMMA4_26B_LANE_SCOPED_ROW) return true
  const identities = [
    model?.catalog_id,
    model?.compatibility_id,
    model?.runtime_model_name,
    model?.id,
    model?.name,
    model?.model_path,
    model?.hf_filename,
  ]
  return identities.some((identity) => {
    const normalized = String(identity || '').toLowerCase().replace(/[^a-z0-9]+/g, '_')
    return normalized === GEMMA4_26B_LANE_SCOPED_ROW
      || (/gemma_?4_26b(?:_|$)/.test(normalized) && normalized.includes('q4_0'))
  })
}

function normalizedIdentity(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
}

function isLfm226bLaneScopedRow(model, hint) {
  if (hint?.kind === 'compatibility' && hint?.exact === true
    && hint?.target?.id === LFM2_26B_LANE_SCOPED_ROW) return true
  const identities = [
    model?.catalog_id,
    model?.compatibility_id,
    model?.runtime_model_name,
    model?.id,
    model?.name,
    model?.model_path,
    model?.hf_filename,
  ]
  return identities.some((identity) => {
    const normalized = normalizedIdentity(identity)
    return normalized === LFM2_26B_LANE_SCOPED_ROW
      || normalized.endsWith('_lfm2_5_2_6b_q8_0_gguf')
      || normalized === 'lfm2_5_2_6b_q8_0_gguf'
  })
}

function supportedLfm2Runtime(runtime) {
  const plan = runtime?.execution_plan
  if (runtime?.backend !== 'runnable-runtime' || !plan) return false
  if (normalizedIdentity(plan.model_family) !== 'lfm2'
    || normalizedIdentity(plan.exact_model_row) !== 'lfm2_5_2_6b_q8_0_gguf'
    || plan.quant_type !== 'Q8_0'
    || plan.support_level !== 'supported_exact_row_smoke') return false

  if (plan.operating_system === 'windows' && plan.architecture === 'x86_64') {
    return plan.selected_backend === 'cpu_reference'
      && plan.prefill_path === 'safe_cpu_prefill'
      && plan.decode_path === 'safe_cpu_decode'
  }

  return plan.operating_system === 'macos'
    && plan.architecture === 'aarch64'
    && plan.cpu_model === 'Apple M4'
    && plan.selected_backend === 'metal_resident_lfm2_runtime'
    && plan.prefill_path === 'lfm2_metal_resident_prefill'
    && plan.decode_path === 'lfm2_metal_resident_decode'
}

function supportedCatalogGhostCuda(runtime, runtimeLane) {
  return runtimeLane === 'ghost_moe'
    && runtime?.backend === 'gemma4-runtime'
    && runtime?.gemma4_ghost_catalog_managed === true
    && runtime?.gemma4_ghost_backend === 'cuda'
    && runtime?.gemma4_ghost_common_gpu_active === true
    && runtime?.gemma4_ghost_experts_gpu_active === true
    && runtime?.gemma4_ghost_head_gpu_active === true
}

function runtimeLaneHint(hint, laneScopedRow, runtimeLane) {
  const reason = runtimeLane === 'ghost_moe'
    ? 'This model is running in a configuration that has not been verified, so replies are marked unverified.'
    : 'No verified run configuration was detected for this model, so replies are marked unverified.'
  const target = hint?.target || (laneScopedRow
    ? {
        id: GEMMA4_26B_LANE_SCOPED_ROW,
        family: 'gemma4_a4b_moe_decoder',
        quantization: 'Q4_0',
      }
    : null)
  if (!target) return hint
  return {
    ...(hint || { kind: 'runtime_lane_mismatch', exact: false }),
    kind: 'runtime_lane_mismatch',
    confidence: 'this run configuration is not covered by verification',
    target: {
      ...target,
      status: 'experimental_runtime_lane',
      evidence: reason,
      next_step: reason,
    },
  }
}

export function getChatGateState(capabilities, model, runtime) {
  const embeddingOnly = isEmbeddingOnlyModel(model, runtime)
  const embeddingReady = Boolean(embeddingOnly && isModelLoadedNow(model))
  const taskKind = modelTaskKind(model, runtime)
  const runtimeLoaded = Boolean(runtime?.loaded_now && modelRuntimeIdMatches(model, runtime))
  const runtimeGenerationReady = Boolean(runtime?.generation_ready && modelRuntimeIdMatches(model, runtime))
  const runtimeReady = Boolean(!embeddingOnly && isRunnableInCurrentRuntime(model, runtime) && runtimeLoaded && runtimeGenerationReady)
  const discoveredHint = findCompatibilityHint(capabilities, model)
  const runtimeLane = normalizedGemma4ServeLane(runtime)
  const gemmaLaneScopedRow = isGemma426bLaneScopedRow(model, discoveredHint)
  const lfm2LaneScopedRow = isLfm226bLaneScopedRow(model, discoveredHint)
  const scopedRowId = gemmaLaneScopedRow
    ? GEMMA4_26B_LANE_SCOPED_ROW
    : lfm2LaneScopedRow
      ? LFM2_26B_LANE_SCOPED_ROW
      : null
  const scopedTarget = scopedRowId
    ? capabilities?.model_compatibility?.find((row) => row?.id === scopedRowId)
    : null
  const contractHint = scopedTarget
    ? {
        kind: 'compatibility',
        target: scopedTarget,
        confidence: `exact ${gemmaLaneScopedRow ? 'Gemma 4 26B' : 'LFM2.5 2.6B'} artifact identity`,
        exact: true,
      }
    : discoveredHint
  // Support evidence is lane-scoped. Gemma 4 26B is green only for distributed
  // serve or the durable catalog-managed Windows CUDA Ghost pair with every GPU
  // component live. LFM2 is green only for the two receipted runnable lanes,
  // with an execution plan whose host scope and live path are both verified.
  const gemmaRuntimeLaneEligible = !gemmaLaneScopedRow
    || runtimeLane === 'distributed'
    || supportedCatalogGhostCuda(runtime, runtimeLane)
  const lfm2RuntimeLaneEligible = !lfm2LaneScopedRow || supportedLfm2Runtime(runtime)
  const runtimeLaneEligible = gemmaRuntimeLaneEligible && lfm2RuntimeLaneEligible
  // Once /api/models/local supplies a lane verdict, it owns artifact identity.
  // Falling back to name matching after an explicit experimental verdict would
  // let a same-named, wrong-hash file inherit the supported contract. Runtime
  // lane scope is an additional gate: a supported exact row cannot promote an
  // ad-hoc or partially accelerated Ghost run.
  const hasBackendLaneVerdict = Boolean(model?.lane_class)
  const artifactSupported = hasBackendLaneVerdict
    ? backendMarksSupportedRow(model)
    : !lfm2LaneScopedRow && isCompatibilitySupportedForModel(capabilities, model)
  const contractSupported = Boolean(!embeddingOnly && runtimeLaneEligible && artifactSupported)
  const exactRowVerifiedRunnable = Boolean(
    !embeddingOnly
    && runtimeLaneEligible
    && isCompatibilityVerifiedRunnableForModel(capabilities, model),
  )
  const exactRowNumericalVariance = Boolean(
    !embeddingOnly
    && runtimeLaneEligible
    && (
      model?.lane_class === 'runnable_with_variance'
      || isCompatibilityNumericalVarianceRunnableForModel(capabilities, model)
    )
  )
  const hint = runtimeLaneEligible
    ? contractHint
    : runtimeLaneHint(contractHint, gemmaLaneScopedRow, runtimeLane)
  const chatUnlocked = Boolean(runtimeReady && contractSupported)
  const verifiedUnlocked = Boolean(runtimeReady && !contractSupported && exactRowVerifiedRunnable)
  const varianceUnlocked = Boolean(runtimeReady && !contractSupported && exactRowNumericalVariance)
  // Experimental lane: the model loaded and is generation-ready (so its architecture
  // is implemented — generation_ready is false for unimplemented archs) but it is NOT
  // a supported contract row. A separate, weaker affordance from the supported gate:
  // chat is allowed but every turn is marked unverified with no parity claim.
  const experimentalUnlocked = Boolean(runtimeReady && !contractSupported)
  const chatMode = contractSupported
    ? 'supported'
    : verifiedUnlocked
      ? 'verified'
      : varianceUnlocked
        ? 'variance'
      : experimentalUnlocked
        ? 'experimental'
        : 'blocked'

  return {
    hint,
    taskKind,
    embeddingOnly,
    embeddingReady,
    generationCapable: !embeddingOnly && model?.generation_capable !== false,
    runtimeReady,
    runtimeLoaded,
    runtimeGenerationReady,
    contractSupported,
    chatUnlocked,
    experimentalUnlocked,
    chatMode,
    // Human label layer: plain product language for every surface that shows
    // the gate. Raw row ids stay in `hint` for technical views and popovers.
    label: !model
      ? 'No model loaded'
      : embeddingOnly
        ? 'Embedding only'
      : contractSupported
        ? 'Verified'
        : verifiedUnlocked
          ? 'Verified (limited)'
        : varianceUnlocked
          ? 'Runnable (reference differs)'
        : experimentalUnlocked
          ? 'Runnable (unverified)'
          : 'Not verified',
    copy: compatibilityHintCopy(hint),
  }
}
