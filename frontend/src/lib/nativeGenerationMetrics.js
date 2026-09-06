export function readTargetVerifiedMtp12(camelid) {
  const mtp12 = camelid?.mtp12
  if (!mtp12 || mtp12.lossless_target_verified !== true) return null

  const decodeTokensPerSecond = Number(mtp12.decode_tokens_per_second)
  if (!Number.isFinite(decodeTokensPerSecond) || decodeTokensPerSecond <= 0) return null

  return {
    ...mtp12,
    decode_tokens_per_second: decodeTokensPerSecond,
    decode_us: Number.isFinite(Number(mtp12.decode_us)) ? Number(mtp12.decode_us) : null,
    decode_output_tokens: Number.isFinite(Number(mtp12.decode_output_tokens))
      ? Number(mtp12.decode_output_tokens)
      : null,
    configured_verify_width: Number.isFinite(Number(mtp12.configured_verify_width))
      ? Number(mtp12.configured_verify_width)
      : null,
    accepted_drafts: Number.isFinite(Number(mtp12.accepted_drafts))
      ? Number(mtp12.accepted_drafts)
      : null,
    drafted: Number.isFinite(Number(mtp12.drafted)) ? Number(mtp12.drafted) : null,
  }
}

export function readExactTargetVerifiedRender(camelid) {
  const render = camelid?.target_verified_render
  if (!render || render.token_ids_exact !== true) return null

  const renderTokensPerSecond = Number(render.render_tokens_per_second)
  const verifiedTokens = Number(render.verified_tokens)
  if (!Number.isFinite(renderTokensPerSecond) || renderTokensPerSecond <= 0) return null
  if (!Number.isInteger(verifiedTokens) || verifiedTokens <= 0) return null

  return {
    ...render,
    render_tokens_per_second: renderTokensPerSecond,
    verified_tokens: verifiedTokens,
  }
}

export function readExactTargetVerifiedSegmentedRender(camelid) {
  const render = camelid?.target_verified_segmented_render
  if (!render
    || render.mode !== 'prepared_web_research_segmented_target_verify'
    || render.segments_exact !== true
    || render.qualification_envelope_max_positions !== 512
    || render.target_model_sha256 !== MTP12_TARGET_SHA256
    || render.assistant_model_sha256 !== MTP12_ASSISTANT_SHA256) return null

  const segmentCount = Number(render.segment_count)
  const totalPromptTokens = Number(render.total_prompt_tokens)
  const requestedTokens = Number(render.requested_tokens)
  const verifiedTokens = Number(render.verified_tokens)
  const decodeOutputTokens = Number(render.decode_output_tokens)
  const decodeUs = Number(render.decode_us)
  const renderTokensPerSecond = Number(render.render_tokens_per_second)
  const segments = Array.isArray(render.segments) ? render.segments : []
  if (!Number.isInteger(segmentCount) || segmentCount < 2 || segmentCount > 8 || segments.length !== segmentCount) return null
  if (!Number.isInteger(totalPromptTokens) || totalPromptTokens <= 0) return null
  if (!Number.isInteger(requestedTokens) || requestedTokens <= 0 || requestedTokens !== verifiedTokens) return null
  if (!Number.isInteger(verifiedTokens) || verifiedTokens <= 0) return null
  if (!Number.isInteger(decodeOutputTokens) || decodeOutputTokens <= 0 || !Number.isFinite(decodeUs) || decodeUs <= 0) return null
  if (!Number.isFinite(renderTokensPerSecond) || renderTokensPerSecond <= 0) return null
  if (segments.some((segment, index) => segment?.index !== index
    || segment?.token_ids_exact !== true
    || !Number.isInteger(Number(segment?.prompt_tokens))
    || Number(segment?.prompt_tokens) <= 0
    || !Number.isInteger(Number(segment?.requested_tokens))
    || Number(segment?.requested_tokens) <= 0
    || Number(segment?.prompt_tokens) + Number(segment?.requested_tokens) + 16 > 512
    || Number(segment?.verified_tokens) !== Number(segment?.requested_tokens))) return null
  const segmentRequestedTokens = segments.reduce((sum, segment) => sum + Number(segment.requested_tokens), 0)
  const segmentPromptTokens = segments.reduce((sum, segment) => sum + Number(segment.prompt_tokens), 0)
  const segmentDecodeOutputTokens = segments.reduce((sum, segment) => sum + Number(segment.decode_output_tokens), 0)
  const segmentDecodeUs = segments.reduce((sum, segment) => sum + Number(segment.decode_us), 0)
  const aggregateRate = segmentDecodeOutputTokens / segmentDecodeUs * 1_000_000
  if (segmentRequestedTokens !== verifiedTokens
    || segmentPromptTokens !== totalPromptTokens
    || segmentDecodeOutputTokens !== decodeOutputTokens
    || segmentDecodeUs !== decodeUs
    || !Number.isFinite(aggregateRate)
    || Math.abs(aggregateRate - renderTokensPerSecond) > 0.000_001) return null

  return {
    ...render,
    segment_count: segmentCount,
    total_prompt_tokens: totalPromptTokens,
    requested_tokens: requestedTokens,
    verified_tokens: verifiedTokens,
    decode_output_tokens: decodeOutputTokens,
    decode_us: decodeUs,
    render_tokens_per_second: renderTokensPerSecond,
    segments,
  }
}

const MTP12_TARGET_SHA256 = '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b'
const MTP12_ASSISTANT_SHA256 = '67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6'
const MTP12_QUALIFICATION = Object.freeze({
  workload: 'short_context_lossless_mtp_qualification',
  selector: 'CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16',
  target_sha256: MTP12_TARGET_SHA256,
  assistant_sha256: MTP12_ASSISTANT_SHA256,
  prompt_tokens: 14,
  output_tokens: 96,
  max_positions: 512,
  primary_decode_tokens_per_second: 51.493947835,
  confirmation_decode_tokens_per_second: 51.304677961,
  mean_decode_tokens_per_second: 51.399,
})

export function readMtp12NativeReceiptQualification(camelid) {
  const qualification = camelid?.mtp12?.native_receipt_qualification
  if (!qualification || typeof qualification !== 'object') return null
  for (const [field, expected] of Object.entries(MTP12_QUALIFICATION)) {
    if (qualification[field] !== expected) return null
  }

  return qualification
}

export function authoritativeOutputRate(message) {
  return readExactTargetVerifiedSegmentedRender(message?.camelid)?.render_tokens_per_second
    ?? readExactTargetVerifiedRender(message?.camelid)?.render_tokens_per_second
    ?? readTargetVerifiedMtp12(message?.camelid)?.decode_tokens_per_second
    ?? message?.tokens_out_per_sec
    ?? null
}

function positiveInteger(value) {
  const number = Number(value)
  return Number.isInteger(number) && number > 0 ? number : null
}

function humanizePolicy(value) {
  const text = String(value || '').trim()
  if (!text) return null
  return text
    .replace(/logical[_-]?w(\d+)/gi, 'logical W$1')
    .replace(/physical[_-]?w(\d+)/gi, 'physical W$1')
    .replace(/[_-]+/g, ' ')
    .replace(/\bpad16\b/gi, 'W16 padding')
    .replace(/\s+/g, ' ')
    .trim()
}

export function describeMtp12WidthSchedule(mtp12) {
  const configuredWidth = positiveInteger(mtp12?.configured_verify_width)
  const provenance = mtp12?.width_schedule

  // Compatibility for already-saved development conversations. The released
  // backend contract is the provenance object handled below.
  if (Array.isArray(provenance)) {
    const widths = provenance.map(positiveInteger).filter(Boolean)
    return widths.length ? { widths: widths.map((width) => `W${width}`).join(' → '), policy: null } : null
  }

  if (!provenance || typeof provenance !== 'object') {
    return configuredWidth ? { widths: `W${configuredWidth} fixed`, policy: null } : null
  }

  const warmupWidth = positiveInteger(provenance.warmup_verify_width)
  const scheduleActive = provenance.enabled === true
    && provenance.active_for_configured_width === true
    && warmupWidth
    && configuredWidth
    && warmupWidth !== configuredWidth
  const widths = scheduleActive
    ? `W${warmupWidth} bootstrap → W${configuredWidth} verify`
    : configuredWidth
      ? `W${configuredWidth} fixed`
      : warmupWidth
        ? `W${warmupWidth} bootstrap`
        : null
  const policy = humanizePolicy(provenance.padded_tail_policy || provenance.policy)
  return widths || policy ? { widths, policy } : null
}
