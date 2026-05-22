export function formatCapabilityStatus(value) {
  return (value || '').toString().replace(/_/g, ' ')
}

const GGUF_FILE_TYPE_QUANT_LABELS = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  7: 'Q8_0',
  8: 'Q5_0',
  9: 'Q5_1',
  10: 'Q2_K',
  11: 'Q3_K_S',
  12: 'Q3_K_M',
  13: 'Q3_K_L',
  14: 'Q4_K_S',
  15: 'Q4_K_M',
  16: 'Q5_K_S',
  17: 'Q5_K_M',
  18: 'Q6_K',
  19: 'IQ2_XXS',
  20: 'IQ2_XS',
  21: 'Q2_K_S',
  22: 'IQ3_XS',
  23: 'IQ3_XXS',
  24: 'IQ1_S',
  25: 'IQ4_NL',
  26: 'IQ3_S',
  27: 'IQ3_M',
  28: 'IQ2_S',
  29: 'IQ2_M',
  30: 'IQ4_XS',
  31: 'IQ1_M',
  32: 'BF16',
  36: 'TQ1_0',
  37: 'TQ2_0',
  38: 'MXFP4_MOE',
  39: 'NVFP4',
  40: 'Q1_0',
}

export function quantLabelFromGgufFileType(fileType) {
  const value = Number(fileType)
  if (!Number.isInteger(value)) return null
  return GGUF_FILE_TYPE_QUANT_LABELS[value] || null
}

function normalizeCapabilityKey(value) {
  return (value || '').toString().trim().toUpperCase().replace(/[^A-Z0-9]+/g, '')
}

function splitCapabilityKeys(value) {
  return (value || '').toString().split('/').map(normalizeCapabilityKey).filter(Boolean)
}

function extractQuantKey(model, catalogItem, subject) {
  const explicitLabel = model?.quant || catalogItem?.quant
  const explicitFileType = explicitLabel?.toString().match(/\bfile[_\s-]*type\s*(\d+)\b/i)?.[1]
  const explicit = normalizeCapabilityKey(explicitFileType ? quantLabelFromGgufFileType(explicitFileType) : explicitLabel)
  if (explicit) return explicit

  const artifactText = [model?.model_path, model?.path, model?.hf_filename, catalogItem?.filename].filter(Boolean).join(' ')
  const artifactMatch = artifactText.match(/\b(q\d(?:_k_[ms]|_\d)|bf16|f16|f32)\b/i)
  const artifactQuant = normalizeCapabilityKey(artifactMatch?.[1])
  if (artifactQuant) return artifactQuant

  const text = subject || ''
  const match = text.match(/\b(q\d(?:_k_[ms]|_\d)|bf16|f16|f32)\b/i)
  return normalizeCapabilityKey(match?.[1])
}

function targetMatchesQuant(target, quantKey) {
  if (!quantKey) return true
  return splitCapabilityKeys(target?.quantization).includes(quantKey)
}

function findCompatibilityRowForQuant(rows, family, quantKey) {
  if (!quantKey) return null
  return rows.find((row) => row.family === family && targetMatchesQuant(row, quantKey)) || null
}

const EXACT_LLAMA_PROMOTION_ROWS = [
  { id: 'llama32_1b_instruct_q8_0', versionKey: '3.2', sizeKey: '1B', requiresInstruct: true },
  { id: 'llama32_3b_instruct_q8_0', versionKey: '3.2', sizeKey: '3B', requiresInstruct: true },
  { id: 'llama3_8b_instruct_q8_0', versionKey: '3', sizeKey: '8B', requiresInstruct: true },
]

const EXACT_ARTIFACT_GATED_ROWS = {
  llama32_3b_instruct_q8_0: 'Llama-3.2-3B-Instruct-Q8_0.gguf',
}

function pathBasename(value) {
  return String(value || '').split(/[\\/]/).filter(Boolean).pop() || ''
}

function exactArtifactFilenameForRow(row) {
  return EXACT_ARTIFACT_GATED_ROWS[row?.id] || null
}

function hasExactArtifactIdentity(row, model, catalogItem) {
  const filename = exactArtifactFilenameForRow(row)
  if (!filename) return true
  const observedFilenames = [
    model?.model_path,
    model?.path,
    model?.hf_filename,
    catalogItem?.filename,
  ].map(pathBasename).filter(Boolean)
  return observedFilenames.some((observed) => observed.toLowerCase() === filename.toLowerCase())
}

function exactArtifactMissingHint(target) {
  return {
    kind: 'artifact_mismatch',
    target,
    confidence: 'exact row id or model-size label without required GGUF filename evidence',
    exact: false,
  }
}

function applyExactArtifactGate(hint, model, catalogItem) {
  if (!hint?.target) return hint
  if (hint.kind === 'quant_mismatch') return hint
  if (!hasExactArtifactIdentity(hint.target, model, catalogItem)) return exactArtifactMissingHint(hint.target)
  return hint
}

function detectLlamaBpeTarget(subject) {
  if (!/llama[\s._-]*3|meta[\s._-]*llama[\s._-]*3/.test(subject)) return null
  const sizeMatch = subject.match(/(?:^|[^a-z0-9])([138])\s*b(?:[^a-z0-9]|$)/i)
  const minorVersionMatch = subject.match(/llama[\s._-]*3[._]\s*(\d+)\b/) || subject.match(/\bllama3(\d+)\b/)
  const versionKey = minorVersionMatch ? (minorVersionMatch[1] === '2' ? '3.2' : null) : '3'
  if (!versionKey) return null
  return {
    family: 'llama_bpe_decoder',
    sizeKey: sizeMatch ? `${sizeMatch[1]}B` : null,
    versionKey,
    instruct: /(?:^|[^a-z0-9])instruct(?:[^a-z0-9]|$)/i.test(subject),
  }
}

function rowMatchesModelSizeAndVersion(row, identity) {
  const exactRow = EXACT_LLAMA_PROMOTION_ROWS.find((target) => (
    target.id === row?.id
    && target.versionKey === identity.versionKey
    && target.sizeKey === identity.sizeKey
  ))
  if (!exactRow) return false
  if (exactRow.requiresInstruct && !identity.instruct) return false
  return true
}

function findLlamaBpeCompatibilityHint(rows, plannedFamilies, quantKey, identity) {
  const familyRows = rows.filter((row) => row.family === identity.family)
  const exactTarget = familyRows.find((row) => rowMatchesModelSizeAndVersion(row, identity)) || null
  if (exactTarget && quantKey && targetMatchesQuant(exactTarget, quantKey)) {
    return { kind: 'compatibility', target: exactTarget, confidence: 'exact model-size + quant match', exact: true }
  }
  if (exactTarget && !quantKey) {
    return { kind: 'quant_missing', target: exactTarget, confidence: 'exact model-size match without quant evidence', exact: true }
  }
  if (exactTarget) {
    return { kind: 'quant_mismatch', target: exactTarget, observedQuant: quantKey, confidence: 'exact model-size match with different quant', exact: true }
  }

  return null
}

function quantAwareCompatibilityHint(target, quantKey, confidence, { exact = false } = {}) {
  if (!target) return null
  if (quantKey && targetMatchesQuant(target, quantKey)) return { kind: 'compatibility', target, confidence, exact }
  if (!quantKey) return { kind: 'quant_missing', target, confidence: `${confidence} without quant evidence`, exact }
  return { kind: 'quant_mismatch', target, observedQuant: quantKey, confidence: `${confidence} with different quant`, exact }
}

function futureExactRowHint(rows, subject, quantKey) {
  const matchers = [
    {
      confidence: 'Mistral exact row + quant match',
      predicate: (row) => row.id === 'mistral_7b_instruct_v0_3_q8_0',
      subjectMatches: () => subject.includes('mistral') && /(?:^|[^a-z0-9])7\s*b(?:[^a-z0-9]|$)/i.test(subject) && subject.includes('instruct') && /v?0[._-]?3/.test(subject),
    },
    {
      confidence: 'Mixtral exact row + quant match',
      predicate: (row) => row.id === 'mixtral_8x7b_instruct_v0_1_q8_0',
      subjectMatches: () => subject.includes('mixtral') && /8\s*x\s*7\s*b/.test(subject) && subject.includes('instruct') && /v?0[._-]?1/.test(subject),
    },
    {
      confidence: 'Qwen exact row + quant match',
      predicate: (row) => row.id === 'qwen25_7b_instruct_q8_0',
      subjectMatches: () => subject.includes('qwen') && /(qwen2[._-]?5|qwen25)/.test(subject) && /(?:^|[^a-z0-9])7\s*b(?:[^a-z0-9]|$)/i.test(subject) && subject.includes('instruct'),
    },
    {
      confidence: 'Gemma exact row + quant match',
      predicate: (row) => row.id === 'gemma2_9b_it_q8_0',
      subjectMatches: () => subject.includes('gemma') && /gemma[\s._-]*2/.test(subject) && /(?:^|[^a-z0-9])9\s*b(?:[^a-z0-9]|$)/i.test(subject) && /(?:^|[^a-z0-9])(?:it|instruct)(?:[^a-z0-9]|$)/i.test(subject),
    },
  ]

  const matcher = matchers.find((item) => item.subjectMatches())
  if (!matcher) return null
  return quantAwareCompatibilityHint(rows.find(matcher.predicate) || null, quantKey, matcher.confidence, { exact: true })
}

export function isSupportedCapabilityStatus(status = '') {
  const value = status.toLowerCase()
  return value === 'supported' || value.startsWith('supported_')
}

function isReadyEvidenceStatus(status = '') {
  const value = status.toLowerCase()
  if (value.includes('not_promoted') || value.includes('fail_closed') || value.includes('fail-closed')) return false
  return value === 'validated' || value.startsWith('validated_') || value === 'measured' || value.startsWith('measured_') || value === 'pass' || value.startsWith('pass_')
}

function statusContainsSupportedEvidence(value = '') {
  const status = String(value || '').toLowerCase()
  if (!status || status.includes('not_') || status.includes('unsupported') || status.includes('blocked') || status.includes('missing') || status.includes('planned')) return false
  return isSupportedCapabilityStatus(status) || isReadyEvidenceStatus(status) || status.includes('supported') || status.includes('validated') || status.includes('measured') || status.includes('pass')
}

function findSupportedFeature(features = [], pattern) {
  return features.find((feature) => pattern.test(String(feature?.id || '')) && isSupportedCapabilityStatus(feature?.status || '')) || null
}

function hasExactRowTemplateReadiness(target) {
  const renderer = String(target?.chat_template_renderer || '')
  const shapePack = String(target?.chat_template_shape_pack || '')
  return isSupportedCapabilityStatus(target?.status || '')
    && (statusContainsSupportedEvidence(renderer) || statusContainsSupportedEvidence(shapePack))
}

function hasExactRowProductionThroughputReadiness(target) {
  const performance = String(target?.performance_measured || '').toLowerCase()
  if (!isSupportedCapabilityStatus(target?.status || '') || !performance) return false
  if (performance.includes('not_') || performance.includes('unsupported') || performance.includes('blocked') || performance.includes('missing') || performance.includes('planned') || performance.includes('fail_closed') || performance.includes('fail-closed')) return false
  return performance.includes('production_throughput') || performance.includes('production-throughput')
}

function hasExactRowBoundedPerformanceEvidence(target) {
  const performance = String(target?.performance_measured || '').toLowerCase()
  if (!isSupportedCapabilityStatus(target?.status || '') || !performance) return false
  if (performance.includes('not_') || performance.includes('unsupported') || performance.includes('blocked') || performance.includes('missing') || performance.includes('planned') || performance.includes('fail_closed') || performance.includes('fail-closed')) return false
  return statusContainsSupportedEvidence(performance)
    || performance.includes('perf')
    || performance.includes('rss')
    || performance.includes('timing')
    || performance.includes('cost')
    || performance.includes('hotpath')
}

const CHECKED_CONTEXT_PACK_FIELDS = [
  ['512', 'bounded_context_512_pack'],
  ['1024', 'bounded_context_1024_pack'],
  ['2048', 'bounded_context_2048_pack'],
  ['4096', 'bounded_context_4096_pack'],
  ['8192', 'bounded_context_8192_pack'],
]

function checkedContextPacks(target) {
  return CHECKED_CONTEXT_PACK_FIELDS
    .map(([label, field]) => ({ label, status: target?.[field] || '' }))
    .filter((pack) => pack.status)
}

function readyCheckedContextPacks(target) {
  if (!isSupportedCapabilityStatus(target?.status || '')) return []
  return checkedContextPacks(target).filter((pack) => statusContainsSupportedEvidence(pack.status))
}

export function describeTemplateReadiness(target) {
  if (!target) {
    return { key: 'template', label: 'Template/Jinja', status: 'No exact row selected', tone: '', ready: false, copy: 'Choose an exact compatibility row before treating template behavior as supported.' }
  }

  const renderer = target.chat_template_renderer || ''
  const shapePack = target.chat_template_shape_pack || ''
  const exactRowReady = hasExactRowTemplateReadiness(target)
  const metadataJinjaReady = exactRowReady && /metadata[_-]?jinja/i.test(renderer)
  const label = exactRowReady
    ? metadataJinjaReady
      ? 'Metadata-Jinja template ready for this exact row'
      : 'Template rendering ready for this exact row'
    : 'Template support not promoted'

  return {
    key: 'template',
    label,
    status: renderer || shapePack || 'not advertised',
    tone: exactRowReady ? 'ready' : 'warm',
    ready: exactRowReady,
    copy: exactRowReady
      ? metadataJinjaReady
        ? `Template/Jinja readiness is green for this supported exact row: metadata-Jinja renderer ${formatCapabilityStatus(renderer)} with ${formatCapabilityStatus(shapePack || 'row evidence')} coverage.`
        : `Template readiness is green for this supported exact row: renderer ${formatCapabilityStatus(renderer || 'not advertised')} with ${formatCapabilityStatus(shapePack || 'row evidence')} coverage.`
      : 'Template/Jinja evidence is not promoted for this row; keep readiness guarded until /api/capabilities reports supported row evidence.',
  }
}

export function describeCheckedContextReadiness(target) {
  if (!target) {
    return { key: 'context', label: 'Checked context', status: 'No exact row selected', tone: '', ready: false, copy: 'Choose an exact compatibility row before treating context behavior as supported.' }
  }

  const packs = checkedContextPacks(target)
  const readyPacks = readyCheckedContextPacks(target)
  const ready = readyPacks.length > 0
  const latest = [target.latest_checked_bucket, target.latest_checked_result].filter(Boolean).map(formatCapabilityStatus).join(' · ')
  const status = readyPacks.length
    ? readyPacks.map((pack) => `${pack.label}: ${formatCapabilityStatus(pack.status)}`).join(' · ')
    : packs.length
      ? packs.map((pack) => `${pack.label}: ${formatCapabilityStatus(pack.status)}`).join(' · ')
      : latest || 'not advertised'

  return {
    key: 'context',
    label: ready ? 'Checked context packs ready for this exact row' : 'Checked context packs not promoted',
    status,
    tone: ready ? 'ready' : 'warm',
    ready,
    copy: ready
      ? `Checked context readiness is green for this supported exact row: ${readyPacks.map((pack) => `${pack.label} context ${formatCapabilityStatus(pack.status)}`).join(', ')}${latest ? `; latest checked ${latest}` : ''}. This does not promote model-native/larger context beyond the checked packs.`
      : 'Checked context evidence is not promoted for this row; keep context claims guarded until /api/capabilities reports supported exact-row context packs.',
  }
}

export function describeThroughputReadiness(target, apiFeatures = []) {
  if (!target) {
    return { key: 'throughput', label: 'Throughput', status: 'No exact row selected', tone: '', ready: false, copy: 'Choose an exact compatibility row before treating throughput behavior as supported.' }
  }

  const feature = findSupportedFeature(apiFeatures, /(?:^|[_-])production[_-]?throughput(?:$|[_-])/i)
  const performance = target.performance_measured || ''
  const rowThroughputReady = hasExactRowProductionThroughputReadiness(target)
  const boundedPerformanceReady = hasExactRowBoundedPerformanceEvidence(target)
  const ready = rowThroughputReady
  const label = ready ? 'Production throughput ready for this exact row' : 'Production throughput not promoted'

  return {
    key: 'throughput',
    label,
    status: performance || feature?.status || 'not advertised',
    tone: ready ? 'ready' : boundedPerformanceReady ? 'warm' : 'warm',
    ready,
    copy: rowThroughputReady
        ? `Production-throughput readiness is green for this supported exact row from ${formatCapabilityStatus(performance)} evidence reported by /api/capabilities.`
        : boundedPerformanceReady
          ? `Bounded row-scoped performance/RSS evidence is present as ${formatCapabilityStatus(performance)}, but production throughput is still not promoted for this exact row${feature ? `; generic API feature ${formatCapabilityStatus(feature.status)} does not widen row support.` : '.'}`
          : `Production throughput evidence is not promoted for this row; keep readiness guarded until /api/capabilities reports explicit production-throughput support on the exact row${feature ? `, not just generic API feature ${formatCapabilityStatus(feature.status)}.` : '.'}`,
  }
}

export function exactRowSupportLanes(target, apiFeatures = []) {
  return [describeTemplateReadiness(target), describeCheckedContextReadiness(target), describeThroughputReadiness(target, apiFeatures)]
}

function resolvedLaneState(target, apiFeatures = []) {
  const lanes = exactRowSupportLanes(target, apiFeatures)
  return {
    templateReady: lanes.some((lane) => lane.key === 'template' && lane.ready),
    throughputReady: lanes.some((lane) => lane.key === 'throughput' && lane.ready),
  }
}

function removeResolvedTemplateCaveat(part) {
  return String(part || '')
    .replace(/\bbroader arbitrary\/Jinja templates? beyond[^,.;]*(?:,?\s*and\s*)?/gi, '')
    .replace(/\bbroader arbitrary\/Jinja template behavior beyond[^,.;]*(?:,?\s*and\s*)?/gi, '')
    .replace(/\bbroader arbitrary templates? beyond[^,.;]*(?:,?\s*and\s*)?/gi, '')
    .replace(/\bbroader arbitrary[- ]template behavior beyond[^,.;]*(?:,?\s*and\s*)?/gi, '')
    .replace(/\barbitrary\/Jinja[- ]?templates?(?:\s+behavior)?\s+and\s+/gi, '')
    .replace(/\barbitrary\/Jinja[- ]?templates?(?:\s+behavior)?$/gi, '')
    .replace(/\barbitrary Jinja[- ]?templates?(?:\s+behavior)?\s+and\s+/gi, '')
    .replace(/\barbitrary Jinja[- ]?templates?(?:\s+behavior)?$/gi, '')
    .replace(/\barbitrary[- ]templates?(?:\s+(?:behavior|evidence))?\s+and\s+/gi, '')
    .replace(/\barbitrary[- ]templates?(?:\s+(?:behavior|evidence))?$/gi, '')
    .replace(/^\s*(?:and|or)\s+/i, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function removeResolvedThroughputCaveat(part) {
  return String(part || '')
    .replace(/\bproduction[- ]throughput(?:\s+(?:behavior|support|evidence|readiness))?(?:\s+remain outside[^,.;]*)?/gi, '')
    .replace(/\bthroughput(?:\s+(?:behavior|support|evidence|readiness))?(?:\s+remain outside[^,.;]*)?/gi, '')
    .replace(/^\s*(?:and|or)\s+/i, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function filterResolvedSupportCaveats(copy, target, apiFeatures = []) {
  const text = String(copy || '').trim()
  if (!text) return ''
  const { templateReady, throughputReady } = resolvedLaneState(target, apiFeatures)
  return text
    .split(/;\s*|,\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => templateReady ? removeResolvedTemplateCaveat(part) : part)
    .map((part) => throughputReady ? removeResolvedThroughputCaveat(part) : part)
    .filter(Boolean)
    .join('; ')
}

export function rowSupportBoundaryCopy(target, apiFeatures = []) {
  if (!target) return 'No exact row selected.'
  const blockers = String(target.full_support_blockers || '').trim()
  if (!blockers) return 'No remaining full-support boundary is advertised for this exact row.'
  const remaining = filterResolvedSupportCaveats(blockers, target, apiFeatures)

  return remaining.length ? remaining : 'Template/Jinja and production-throughput lanes are green for this exact row; remaining readiness follows runtime loaded_now/generation_ready and any other /api/capabilities fields.'
}

export function rowSupportNextStepCopy(target, apiFeatures = []) {
  if (!target) return 'No exact row selected.'
  const nextStep = String(target.next_step || '').trim()
  if (!nextStep) return 'No next support step is advertised for this exact row.'
  const remaining = filterResolvedSupportCaveats(nextStep, target, apiFeatures)
  return remaining.length ? remaining : 'Template/Jinja and production-throughput are already represented by green row-scoped readiness lanes for this exact row; continue to require runtime loaded_now/generation_ready and any other advertised evidence before widening support.'
}

function supportedRowsHaveGreenLane(capabilities, laneKey) {
  const rows = capabilities?.model_compatibility || []
  const apiFeatures = capabilities?.api_features || []
  const supportedRows = rows.filter((target) => isSupportedCapabilityStatus(target.status))
  return Boolean(supportedRows.length && supportedRows.every((target) => {
    const lanes = exactRowSupportLanes(target, apiFeatures)
    return lanes.some((lane) => lane.key === laneKey && lane.ready)
  }))
}

export function supportedRowsHaveGreenTemplateAndThroughput(capabilities) {
  return supportedRowsHaveGreenLane(capabilities, 'template') && supportedRowsHaveGreenLane(capabilities, 'throughput')
}

function stripResolvedCurrentGateCaveats(copy, { templateReady = false, throughputReady = false } = {}) {
  let text = String(copy || '')
  if (templateReady) {
    text = text
      .replace(/,?\s*broader arbitrary[- ]template behavior beyond[^,.;]*/gi, '')
      .replace(/,?\s*broader arbitrary templates beyond[^,.;]*/gi, '')
      .replace(/,?\s*arbitrary GGUF\/Jinja templates?(?:\s+behavior)?/gi, '')
      .replace(/,?\s*arbitrary\/Jinja[- ]?templates?(?:\s+behavior)?/gi, '')
      .replace(/,?\s*arbitrary Jinja[- ]?templates?(?:\s+behavior)?/gi, '')
      .replace(/,?\s*arbitrary[- ]templates?(?:\s+behavior)?(?:\s+remain outside[^,.;]*)?/gi, '')
  }
  if (throughputReady) {
    text = text
      .replace(/,?\s*production[- ]throughput(?:\s+(?:behavior|support|evidence|readiness))?(?:\s+remain outside[^,.;]*)?/gi, '')
      .replace(/,?\s*throughput(?:\s+(?:behavior|support|evidence|readiness))?(?:\s+remain outside[^,.;]*)?/gi, '')
  }
  return text
    .replace(/no model-native\/larger context beyond the checked packs,\s*or portability/gi, 'no model-native/larger context beyond the checked packs or portability')
    .replace(/,\s*,/g, ',')
    .replace(/,\s*or\s*,/g, ',')
    .replace(/,\s*and\s*,/g, ',')
    .replace(/\s+(?:and|or)\s+(?=[,.;])/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function frontendSupportContractCopy(capabilities) {
  const currentGate = capabilities?.support_contract?.current_gate || ''
  if (!currentGate) return 'Capabilities unavailable'
  const templateReady = supportedRowsHaveGreenLane(capabilities, 'template')
  const throughputReady = supportedRowsHaveGreenLane(capabilities, 'throughput')
  if (!templateReady && !throughputReady) return currentGate

  return stripResolvedCurrentGateCaveats(currentGate, { templateReady, throughputReady })
}

export function isGuardedCapabilityStatus(status = '') {
  return !isSupportedCapabilityStatus(status)
}

export function capabilityStatusTone(status = '') {
  const value = status.toLowerCase()
  if (isSupportedCapabilityStatus(value)) return 'ready'
  if (isReadyEvidenceStatus(value)) return 'ready'
  if (
    value.includes('planned')
    || value.includes('partial')
    || value.includes('pending')
    || value.includes('guarded')
    || value.includes('groundwork')
    || value.includes('evidence')
    || value.includes('blocked')
    || value.includes('unsupported')
    || value.includes('not_promoted')
    || value.includes('future')
    || value.includes('fail_closed')
    || value.includes('fail-closed')
    || value.includes('not_started')
  ) return 'warm'
  return ''
}

export function summarizeCapabilityItems(items = [], fallback = 'Not advertised by this backend yet.') {
  if (!items.length) return fallback
  return items.map((item) => `${item.id}: ${formatCapabilityStatus(item.status)}`).join(' · ')
}

export function guardedCapabilityCopy(item, subject = 'UI controls') {
  const notes = item?.notes ? `${item.notes}. ` : ''
  return `${notes}${subject} should stay disabled, labeled planned/unsupported, or require an explicit verification path until /api/capabilities reports this row as supported; callers should expect typed backend refusals and surface them directly, not silently drop parameters, downgrade behavior, or infer broad compatibility.`
}

const STANDARD_PROVIDER_TOKEN = 'open' + 'ai'
const STANDARD_PROVIDER_PREFIX_PATTERN = new RegExp(`^${STANDARD_PROVIDER_TOKEN}[\\s._-]+`, 'i')
const STANDARD_PROVIDER_COMPAT_PATTERN = new RegExp(`\\b${STANDARD_PROVIDER_TOKEN}[\\s._-]*compatible\\b`, 'gi')
const STANDARD_PROVIDER_NAME_PATTERN = new RegExp(`\\b${STANDARD_PROVIDER_TOKEN}\\b`, 'gi')
const HOSTED_BRAND_PATTERNS = [
  new RegExp(`\\b${'chat' + 'gpt'}\\b`, 'gi'),
  new RegExp(`\\b${'clau' + 'de'}\\b`, 'gi'),
  new RegExp(`\\b${'gem' + 'ini'}\\b`, 'gi'),
]

export function displayCapabilityId(value = '') {
  return String(value || '')
    .replace(STANDARD_PROVIDER_PREFIX_PATTERN, '')
    .replace(STANDARD_PROVIDER_COMPAT_PATTERN, 'standard compatible')
    .replace(STANDARD_PROVIDER_NAME_PATTERN, 'standard')
    .replace(/[_.-]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function displayCapabilityCopy(value = '') {
  return HOSTED_BRAND_PATTERNS.reduce(
    (copy, pattern) => copy.replace(pattern, 'hosted model'),
    String(value || '')
      .replace(STANDARD_PROVIDER_COMPAT_PATTERN, 'standard-compatible')
      .replace(STANDARD_PROVIDER_NAME_PATTERN, 'standard'),
  )
}

export const TRACKED_FULL_SUPPORT_ROW_IDS = [
  'tinyllama_1_1b_chat_q8_0',
  'llama32_1b_instruct_q8_0',
  'llama32_3b_instruct_q8_0',
  'llama3_8b_instruct_q8_0',
]

export const TRACKED_LLAMA_PROMOTION_ROW_IDS = TRACKED_FULL_SUPPORT_ROW_IDS

export function getCurrentCompatibilityTarget(capabilities) {
  const targets = capabilities?.model_compatibility || []
  return targets.find((target) => target.status === 'supported_current_gate') || null
}

export function getTrackedCompatibilityTargets(capabilities, ids = TRACKED_FULL_SUPPORT_ROW_IDS) {
  const targets = capabilities?.model_compatibility || []
  return ids.map((id) => targets.find((target) => target.id === id) || null).filter(Boolean)
}

function getModelCapabilityFields(model, catalogItem) {
  return [
    model?.id,
    model?.runtime_model_name,
    model?.name,
    model?.hf_repo,
    model?.hf_filename,
    model?.model_path,
    catalogItem?.name,
    catalogItem?.repo_id,
    catalogItem?.filename,
  ].filter(Boolean).map((value) => String(value))
}

function getModelCapabilitySubject(model, catalogItem) {
  return getModelCapabilityFields(model, catalogItem).join(' ').toLowerCase()
}

function normalizeExactRowIdentity(value) {
  return String(value || '').trim().toLowerCase().replace(/\.gguf$/i, '').replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
}

function findExactCompatibilityRowByIdentity(rows, model, catalogItem) {
  const identities = new Set(getModelCapabilityFields(model, catalogItem).map(normalizeExactRowIdentity).filter(Boolean))
  if (!identities.size) return null
  return rows.find((row) => row?.id && identities.has(normalizeExactRowIdentity(row.id))) || null
}

export function findCompatibilityHint(capabilities, model, catalogItem) {
  const subject = getModelCapabilitySubject(model, catalogItem)
  if (!subject) return null
  const rows = capabilities?.model_compatibility || []
  const plannedFamilies = capabilities?.planned_model_families || []
  const quantKey = extractQuantKey(model, catalogItem, subject)

  const findRow = (predicate) => rows.find(predicate) || null
  const findFamily = (predicate) => plannedFamilies.find(predicate) || null
  const exactIdentityTarget = findExactCompatibilityRowByIdentity(rows, model, catalogItem)
  if (exactIdentityTarget) {
    return applyExactArtifactGate(
      quantAwareCompatibilityHint(exactIdentityTarget, quantKey, 'exact /api/capabilities row id match', { exact: true }),
      model,
      catalogItem,
    )
  }

  const llamaBpeIdentity = detectLlamaBpeTarget(subject)
  if (llamaBpeIdentity) {
    const hint = findLlamaBpeCompatibilityHint(rows, plannedFamilies, quantKey, llamaBpeIdentity)
    if (hint) {
      const normalizedHint = hint.kind === 'quant_mismatch' ? { ...hint, observedQuant: model?.quant || catalogItem?.quant || quantKey } : hint
      return applyExactArtifactGate(normalizedHint, model, catalogItem)
    }
  }

  if (subject.includes('tinyllama')) {
    const target = findRow((row) => row.id.includes('tinyllama'))
    if (target && quantKey && targetMatchesQuant(target, quantKey)) return { kind: 'compatibility', target, confidence: 'exact TinyLlama row + quant match', exact: true }
    if (target && !quantKey) return { kind: 'quant_missing', target, confidence: 'TinyLlama exact-row match without quant evidence', exact: true }
    const quantSpecificTarget = findCompatibilityRowForQuant(rows, 'llama_spm_decoder', quantKey)
    if (quantSpecificTarget) return { kind: 'family', target: quantSpecificTarget, observedQuant: model?.quant || catalogItem?.quant || quantKey, confidence: 'family + quant match without exact TinyLlama row' }
    if (target) return { kind: 'quant_mismatch', target, observedQuant: model?.quant || catalogItem?.quant || quantKey, confidence: 'name/path match with different quant', exact: true }
  }

  if (subject.includes('mistral')) {
    const hint = futureExactRowHint(rows, subject, quantKey)
    if (hint) return hint
    const target = findRow((row) => row.family === 'mistral' || row.id.includes('mistral'))
    if (target) return { kind: 'family', target, confidence: 'Mistral family name match without exact row match' }
    const family = findFamily((item) => item.id.includes('mistral'))
    if (family) return { kind: 'family', target: family, confidence: 'family name match' }
  }

  if (subject.includes('mixtral')) {
    const hint = futureExactRowHint(rows, subject, quantKey)
    if (hint) return hint
    const target = findRow((row) => row.family === 'mixtral_moe' || row.family === 'mixtral' || row.id.includes('mixtral'))
    if (target) return { kind: 'family', target, confidence: 'Mixtral family name match without exact row match' }
    const family = findFamily((item) => item.id.includes('mixtral'))
    if (family) return { kind: 'family', target: family, confidence: 'family name match' }
  }

  if (subject.includes('qwen')) {
    const hint = futureExactRowHint(rows, subject, quantKey)
    if (hint) return hint
    const target = findRow((row) => row.family === 'qwen_decoder' || row.family === 'qwen2' || row.id.includes('qwen'))
    if (target) return { kind: 'family', target, confidence: 'Qwen family name match without exact row match' }
    const family = findFamily((item) => item.id.includes('qwen'))
    if (family) return { kind: 'family', target: family, confidence: 'family name match' }
  }

  if (subject.includes('gemma')) {
    const hint = futureExactRowHint(rows, subject, quantKey)
    if (hint) return hint
    const target = findRow((row) => row.family === 'gemma2_decoder' || row.family === 'gemma2' || row.id.includes('gemma'))
    if (target) return { kind: 'family', target, confidence: 'Gemma family name match without exact row match' }
    const family = findFamily((item) => item.id.includes('gemma'))
    if (family) return { kind: 'family', target: family, confidence: 'family name match' }
  }

  const futureFamily = findFamily((item) => item.id.includes('phi_falcon_mamba') && /(phi|falcon|mamba)/.test(subject))
  if (futureFamily) return { kind: 'family', target: futureFamily, confidence: 'future family name match' }

  return null
}

export function compatibilityHintLabel(hint, fallback = 'No matching compatibility row') {
  if (!hint) return fallback
  if (hint.kind === 'artifact_mismatch') return `${hint.target.id}: exact GGUF not verified`
  if (hint.kind === 'quant_missing') return `${hint.target.id}: quant not verified`
  if (hint.kind === 'quant_mismatch') return `${hint.target.id}: quant mismatch`
  return `${hint.target.id}: ${formatCapabilityStatus(hint.target.status)}`
}

function displayObservedQuant(value) {
  const text = String(value || '')
  const qDigit = text.match(/^Q(\d)(\d)$/i)
  if (qDigit) return `Q${qDigit[1]}_${qDigit[2]}`
  const qK = text.match(/^Q(\d)K([MSL])$/i)
  if (qK) return `Q${qK[1]}_K_${qK[2].toUpperCase()}`
  return text
}

export function compatibilityHintCopy(hint) {
  if (!hint) return 'No exact COMPATIBILITY.md row matched this model name/path, so the UI will not infer model-family support; load results and typed backend errors remain the source of truth.'
  if (hint.kind === 'family') {
    const boundary = hint.target?.notes || hint.target?.next_step || `${hint.target?.id || 'This family row'} is ${formatCapabilityStatus(hint.target?.status || 'not_supported')}`
    return `${boundary}. This is only a ${hint.confidence}; it is not chat-ready support until a concrete exact compatibility row is validated.`
  }
  if (hint.kind === 'artifact_mismatch') return `${hint.target.id} requires the exact ${exactArtifactFilenameForRow(hint.target) || 'row GGUF'} artifact before the frontend may treat the support contract as matched. Do not unlock chat from a saved row id, model-size label, or neighboring GGUF filename alone; wait for exact artifact evidence plus loaded_now=true and generation_ready=true.`
  if (hint.kind === 'quant_missing') return `${hint.target.id} is the right model-size row, but this local record does not expose a quant label yet. Do not unlock chat from a size/name match alone; wait for GGUF quant evidence from the loaded model metadata plus generation_ready=true.`
  if (hint.kind === 'quant_mismatch') return `${hint.target.id} is scoped to ${hint.target.quantization}, but this entry appears to be ${displayObservedQuant(hint.observedQuant) || 'a different quantization'}. Do not inherit the supported gate from a same-family row; wait for an exact COMPATIBILITY.md row plus generation_ready=true.`
  return `${hint.target.family} · ${hint.target.quantization} · ${hint.target.evidence || hint.target.next_step}. Match source: ${hint.confidence}; runtime generation still requires loaded_now=true and generation_ready=true.`
}

export function isExactCompatibilityHint(hint) {
  return Boolean(hint?.kind === 'compatibility' && hint.exact === true)
}

export function compatibilityHintMatchesExactTarget(capabilities, model, target, catalogItem) {
  const hint = findCompatibilityHint(capabilities, model, catalogItem)
  return Boolean(
    isExactCompatibilityHint(hint)
    && target?.id
    && hint.target?.id === target.id,
  )
}

export function isCompatibilitySupportedForModel(capabilities, model, catalogItem) {
  const hint = findCompatibilityHint(capabilities, model, catalogItem)
  return Boolean(
    isExactCompatibilityHint(hint)
    && isSupportedCapabilityStatus(hint.target?.status),
  )
}
