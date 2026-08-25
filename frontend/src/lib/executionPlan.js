function displayPlanValue(value = '') {
  return String(value).trim().replace(/_/g, ' ')
}

const CPU_BACKENDS = new Set([
  'cpu_reference',
  'cpu_q8_runtime_repack',
  'cpu_kquant_block_dot',
])

const CUDA_BACKENDS = new Set([
  'cuda_resident_q8_runtime',
  'cuda_resident_q8_runtime_runnable_unvalidated',
  'cuda_resident_kquant_runtime',
])

const METAL_BACKENDS = new Set([
  'metal_resident_q8_runtime',
])

const SPECIALIZED_BACKENDS = new Set([
  'gemma4-runtime',
  'runnable-runtime',
  'diffusion-gemma-runtime',
])

const GHOST_EXECUTION_MODES = new Set([
  'full_common_metal',
  'hybrid_metal',
  'full_common_cuda',
  'hybrid_cuda',
  'cpu_storage',
])

const GHOST_BACKENDS = new Set(['cpu', 'metal', 'cuda'])

export const GEMMA4_MINI2_WEBUI_PROFILE_ID = 'mini2-h71r-h58-h60-h62-1408-ctx1024-mtp15-adaptive-v2'

function optionalBoolean(value) {
  return typeof value === 'boolean' ? value : null
}

function optionalPositiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0 ? value : null
}

function ghostComponentSummary({ accelerator, common, experts, head }) {
  const gpu = accelerator === 'cuda' ? 'CUDA' : 'Metal'
  return `Common core: ${common ? gpu : 'CPU'}. Persistent Q4_0 expert slots: ${experts ? gpu : 'CPU/storage'}. Q6_K tied head: ${head ? gpu : 'CPU'}. Routed expert records page from the local SSD.`
}

export function executionRuntimeFields(health) {
  const executionMode = String(health?.gemma4_ghost_execution_mode || '')
  const ghostBackend = String(health?.gemma4_ghost_backend || '')
  return {
    execution_plan: health?.execution_plan || null,
    backend: health?.backend || 'none',
    gemma4_ghost_execution_mode: GHOST_EXECUTION_MODES.has(executionMode) ? executionMode : null,
    gemma4_ghost_common_metal_active: optionalBoolean(health?.gemma4_ghost_common_metal_active),
    gemma4_ghost_experts_metal_active: optionalBoolean(health?.gemma4_ghost_experts_metal_active),
    gemma4_ghost_head_metal_active: optionalBoolean(health?.gemma4_ghost_head_metal_active),
    gemma4_mtp_assistant_loaded: optionalBoolean(health?.gemma4_mtp_assistant_loaded),
    gemma4_mtp_full_q4_active: optionalBoolean(health?.gemma4_mtp_full_q4_active),
    gemma4_ghost_exact_expert_policy_active: optionalBoolean(health?.gemma4_ghost_exact_expert_policy_active),
    gemma4_ghost_common_metal_context_capacity: optionalPositiveInteger(health?.gemma4_ghost_common_metal_context_capacity),
    gemma4_ghost_runtime_profile: health?.gemma4_ghost_runtime_profile === GEMMA4_MINI2_WEBUI_PROFILE_ID
      ? GEMMA4_MINI2_WEBUI_PROFILE_ID
      : null,
    gemma4_ghost_backend: GHOST_BACKENDS.has(ghostBackend) ? ghostBackend : null,
    gemma4_ghost_catalog_managed: optionalBoolean(health?.gemma4_ghost_catalog_managed),
    gemma4_ghost_common_gpu_active: optionalBoolean(health?.gemma4_ghost_common_gpu_active),
    gemma4_ghost_experts_gpu_active: optionalBoolean(health?.gemma4_ghost_experts_gpu_active),
    gemma4_ghost_head_gpu_active: optionalBoolean(health?.gemma4_ghost_head_gpu_active),
  }
}

export function describeExecutionPlan(runtime) {
  if (runtime?.status === 'offline') {
    return {
      state: 'offline',
      device: 'Unavailable',
      backend: 'Backend offline',
      summary: 'Execution details are unavailable while the Camelid backend is offline.',
    }
  }

  if (!runtime?.loaded_now) {
    return {
      state: 'idle',
      device: 'No model loaded',
      backend: 'No active plan',
      summary: 'No model is loaded, so Camelid has no active execution plan.',
    }
  }

  if (!runtime?.generation_ready) {
    return {
      state: 'pending',
      device: 'Not active',
      backend: 'Model not generation-ready',
      summary: 'A model is loaded, but Camelid is not generation-ready, so no execution claim is shown.',
    }
  }

  if (runtime?.backend === 'gemma4-runtime' && runtime?.gemma4_serve_lane === 'ghost_moe') {
    const reportedBackend = GHOST_BACKENDS.has(runtime?.gemma4_ghost_backend)
      ? runtime.gemma4_ghost_backend
      : null
    // Prefer the backend-neutral contract. Older engines expose only the
    // Metal-specific fields, which remain a supported compatibility fallback.
    const accelerator = reportedBackend === 'cuda'
      ? 'cuda'
      : reportedBackend === 'metal'
        ? 'metal'
        : reportedBackend === 'cpu'
          ? null
          : 'metal'
    const neutral = reportedBackend !== null
    const common = neutral
      ? runtime?.gemma4_ghost_common_gpu_active === true
      : runtime?.gemma4_ghost_common_metal_active === true
    const experts = neutral
      ? runtime?.gemma4_ghost_experts_gpu_active === true
      : runtime?.gemma4_ghost_experts_metal_active === true
    const head = neutral
      ? runtime?.gemma4_ghost_head_gpu_active === true
      : runtime?.gemma4_ghost_head_metal_active === true
    const componentMode = accelerator && common
      ? `full_common_${accelerator}`
      : accelerator && (experts || head)
        ? `hybrid_${accelerator}`
        : 'cpu_storage'
    // The explicit mode is accepted only when its component booleans agree.
    // This prevents a malformed/stale string from manufacturing a GPU claim.
    const reportedMode = GHOST_EXECUTION_MODES.has(runtime?.gemma4_ghost_execution_mode)
      ? runtime.gemma4_ghost_execution_mode
      : null
    const mode = reportedMode === componentMode ? reportedMode : componentMode
    const summary = ghostComponentSummary({ accelerator, common, experts, head })
    if (mode === 'full_common_metal' || mode === 'hybrid_metal') {
      return {
        state: 'metal',
        device: `${mode === 'full_common_metal' ? 'Full-common' : 'Hybrid'} Metal + local SSD`,
        backend: 'Ghost-MoE',
        summary,
      }
    }
    if (mode === 'full_common_cuda' || mode === 'hybrid_cuda') {
      return {
        state: 'cuda',
        device: `${mode === 'full_common_cuda' ? 'Full-common' : 'Hybrid'} CUDA + local SSD`,
        backend: 'Ghost-MoE',
        summary,
      }
    }
    return {
      state: 'cpu',
      device: 'CPU + local SSD',
      backend: 'Ghost-MoE',
      summary,
    }
  }

  if (SPECIALIZED_BACKENDS.has(runtime?.backend)) {
    const backend = displayPlanValue(runtime.backend)
    return {
      state: 'specialized',
      device: 'Runtime-specific',
      backend,
      summary: `Camelid reports the active model is served by ${backend}; the generic load-time execution plan is not used for a device claim.`,
    }
  }

  if (runtime?.backend !== 'llama') {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: displayPlanValue(runtime?.backend) || 'Backend unavailable',
      summary: 'Camelid did not report a recognized serving backend for the loaded model.',
    }
  }

  const plan = runtime?.execution_plan || null
  if (!plan) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: 'Plan unavailable',
      summary: 'A model is loaded, but Camelid did not return an active execution plan.',
    }
  }

  const selectedBackend = String(plan.selected_backend || '')
  if (!selectedBackend) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: 'Plan unavailable',
      summary: 'Camelid returned an execution plan without a selected backend.',
    }
  }

  const cudaSelected = CUDA_BACKENDS.has(selectedBackend)
  const metalActive = METAL_BACKENDS.has(selectedBackend)
  const cpuSelected = CPU_BACKENDS.has(selectedBackend)
  const cudaConsistent = cudaSelected && plan.cuda_resident_active === true
  const contradictoryCuda = cudaSelected !== (plan.cuda_resident_active === true)
  if ((!cpuSelected && !cudaSelected && !metalActive) || contradictoryCuda) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: displayPlanValue(selectedBackend),
      summary: 'Camelid returned a load-time execution plan that this UI cannot classify safely.',
    }
  }

  const device = cudaConsistent ? 'CUDA GPU' : metalActive ? 'Metal GPU' : 'CPU'
  const backend = displayPlanValue(selectedBackend)

  return {
    state: cudaConsistent ? 'cuda' : metalActive ? 'metal' : 'cpu',
    device,
    backend,
    summary: `At model load, Camelid selected ${device} using ${backend}. Runtime controls may change the effective path afterward.`,
  }
}
