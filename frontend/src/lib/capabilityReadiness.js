/* Readiness labels derived from an /api/capabilities model_compatibility row.
   The label is computed from the row's own status/scope fields — NEVER from the
   model family name. A family-name match must never produce a green state; only
   the exact row's recorded evidence can. */

export const READINESS = {
  SUPPORTED_EXACT_ROW: 'supported-exact-row',
  RUNNABLE_VARIANCE: 'runnable-variance',
  ACTIVE_VALIDATION: 'active-validation',
  CPU_ONLY: 'cpu-only',
  GPU_EXPERIMENTAL: 'gpu-experimental',
  UNSUPPORTED_MULTIMODAL: 'unsupported-multimodal',
  UNSUPPORTED_QUANTIZATION: 'unsupported-quantization',
  PLANNED: 'planned',
  BLOCKED: 'blocked',
  UNKNOWN: 'unknown',
}

export const READINESS_LABELS = {
  [READINESS.SUPPORTED_EXACT_ROW]: 'Supported exact row',
  [READINESS.RUNNABLE_VARIANCE]: 'Runnable with reference variance',
  [READINESS.ACTIVE_VALIDATION]: 'Active validation',
  [READINESS.CPU_ONLY]: 'CPU only',
  [READINESS.GPU_EXPERIMENTAL]: 'GPU experimental',
  [READINESS.UNSUPPORTED_MULTIMODAL]: 'Unsupported: multimodal input',
  [READINESS.UNSUPPORTED_QUANTIZATION]: 'Unsupported: quantization',
  [READINESS.PLANNED]: 'Planned exact-row candidate',
  [READINESS.BLOCKED]: 'Blocked',
  [READINESS.UNKNOWN]: 'No support contract',
}

/* Classify one /api/capabilities model_compatibility row. */
export function classifyCapabilityRow(row) {
  if (!row || typeof row !== 'object') return READINESS.UNKNOWN
  const status = (row.status || '').toString()
  const generation = (row.generation_runs || '').toString()
  const tensors = (row.tensors_load || '').toString()

  if (tensors.includes('unsupported_typed_error') || status.includes('unsupported_quant')) {
    return READINESS.UNSUPPORTED_QUANTIZATION
  }
  if (status.startsWith('supported')) {
    // A supported row may still be CPU-only or carry an experimental GPU lane;
    // the capability row records that in performance/generation fields.
    const perf = (row.performance_measured || '').toString()
    if (perf.includes('gpu_experimental')) return READINESS.GPU_EXPERIMENTAL
    if (perf.includes('cpu_only') || generation.includes('cpu_only')) return READINESS.CPU_ONLY
    return READINESS.SUPPORTED_EXACT_ROW
  }
  if (status === 'runnable_exact_row_numerical_variance') return READINESS.RUNNABLE_VARIANCE
  if (status.startsWith('active_validation')) return READINESS.ACTIVE_VALIDATION
  if (status.startsWith('planned')) return READINESS.PLANNED
  if (status.startsWith('blocked') || (row.full_support_status || '').toString().startsWith('blocked')) {
    return READINESS.BLOCKED
  }
  return READINESS.UNKNOWN
}

/* Multimodal requests are fail-closed for every Camelid row: classify a chat
   attachment kind against the text-only contract. */
export function classifyInputModality(kind) {
  if (kind === 'text') return READINESS.SUPPORTED_EXACT_ROW
  return READINESS.UNSUPPORTED_MULTIMODAL
}

/* True only when the EXACT row id appears in the capabilities contract with a
   supported* status. Family-name or prefix matches never count. */
export function isExactRowSupported(capabilities, modelId) {
  const rows = capabilities?.model_compatibility
  if (!Array.isArray(rows) || !modelId) return false
  const row = rows.find((r) => r?.id === modelId)
  if (!row) return false
  return (row.status || '').toString().startsWith('supported')
}

export function readinessLabel(state) {
  return READINESS_LABELS[state] || READINESS_LABELS[READINESS.UNKNOWN]
}
