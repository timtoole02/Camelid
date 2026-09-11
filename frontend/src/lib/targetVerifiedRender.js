export const GEMMA4_MTP12_EXACT_ROW_ID = 'gemma4_12b_it_qat_q4_0_mtp12'
export const GEMMA4_MTP12_TARGET_VERIFIED_VIDEO_OPT_IN_KEY = 'camelid.video.gemma4Mtp12TargetVerifiedRender'

export function isGemma4Mtp12TargetVerifiedVideoOptedIn() {
  if (typeof window === 'undefined') return false
  try {
    return window.localStorage.getItem(GEMMA4_MTP12_TARGET_VERIFIED_VIDEO_OPT_IN_KEY) === '1'
  } catch {
    return false
  }
}

export function shouldUseGemma4Mtp12TargetVerifiedRender({
  runtime,
  requestModelId,
  compatibilityRowId,
  research,
  receiptMode,
  videoRigOptIn,
}) {
  if (receiptMode || videoRigOptIn !== true) return false

  const lane = String(runtime?.gemma4_serve_lane || '').toLowerCase().replace(/-/g, '_')
  const requestedId = String(requestModelId || '')
  const activeId = String(runtime?.active_model_id || '')
  const exactMtp12Runtime = runtime?.backend === 'gemma4-runtime'
    && lane === 'mtp12_metal'
    && compatibilityRowId === GEMMA4_MTP12_EXACT_ROW_ID
    && requestedId !== ''
    && requestedId === activeId
  if (!exactMtp12Runtime) return false

  // This two-pass presentation is reserved for an explicitly opted-in video
  // rig and a successfully grounded Web Auto turn. Ordinary MTP12 research
  // chats remain on the normal one-pass path. The first pass still performs a
  // fresh model generation; only its authoritative token IDs are passed to the
  // visible target verifier.
  const sources = Array.isArray(research?.sources) ? research.sources : []
  return sources.length >= 2
}
