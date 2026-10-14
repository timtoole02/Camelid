#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  describeExecutionPlan,
  executionRuntimeFields,
} from '../src/lib/executionPlan.js'

const baseGhostRuntime = {
  status: 'online',
  loaded_now: true,
  generation_ready: true,
  backend: 'gemma4-runtime',
  gemma4_serve_lane: 'ghost_moe',
}

assert.deepEqual(
  describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_execution_mode: 'full_common_metal',
    gemma4_ghost_common_metal_active: true,
    gemma4_ghost_experts_metal_active: true,
    gemma4_ghost_head_metal_active: true,
  }),
  {
    state: 'metal',
    device: 'Full-common Metal + local SSD',
    backend: 'Ghost-MoE',
    summary: 'Common core: Metal. Persistent Q4_0 expert slots: Metal. Q6_K tied head: Metal. Routed expert records page from the local SSD.',
  },
  'a live full-common report should identify every Metal component',
)

assert.deepEqual(
  describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_backend: 'cuda',
    gemma4_ghost_execution_mode: 'hybrid_cuda',
    gemma4_ghost_common_gpu_active: false,
    gemma4_ghost_experts_gpu_active: true,
    gemma4_ghost_head_gpu_active: true,
  }),
  {
    state: 'cuda',
    device: 'Hybrid CUDA + local SSD',
    backend: 'Ghost-MoE',
    summary: 'Common core: CPU. Persistent Q4_0 expert slots: CUDA. Q6_K tied head: CUDA. Routed expert records page from the local SSD.',
  },
  'a Windows Ghost runtime should report its live CUDA expert and head components',
)

assert.deepEqual(
  describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_execution_mode: 'hybrid_metal',
    gemma4_ghost_common_metal_active: false,
    gemma4_ghost_experts_metal_active: true,
    gemma4_ghost_head_metal_active: true,
  }),
  {
    state: 'metal',
    device: 'Hybrid Metal + local SSD',
    backend: 'Ghost-MoE',
    summary: 'Common core: CPU. Persistent Q4_0 expert slots: Metal. Q6_K tied head: Metal. Routed expert records page from the local SSD.',
  },
  'CPU common plus live persistent experts/head must be labeled as hybrid Metal',
)

for (const active of [false, null, undefined, 'true']) {
  const plan = describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_execution_mode: 'cpu_storage',
    gemma4_ghost_common_metal_active: active,
    gemma4_ghost_experts_metal_active: false,
    gemma4_ghost_head_metal_active: false,
  })
  assert.equal(plan.state, 'cpu')
  assert.equal(plan.device, 'CPU + local SSD')
}

assert.deepEqual(
  executionRuntimeFields({
    backend: 'gemma4-runtime',
    gemma4_ghost_execution_mode: 'hybrid_metal',
    gemma4_ghost_common_metal_active: false,
    gemma4_ghost_experts_metal_active: true,
    gemma4_ghost_head_metal_active: false,
    gemma4_ghost_backend: 'metal',
    gemma4_ghost_common_gpu_active: false,
    gemma4_ghost_experts_gpu_active: true,
    gemma4_ghost_head_gpu_active: false,
  }),
  {
    execution_plan: null,
    backend: 'gemma4-runtime',
    gemma4_ghost_catalog_managed: null,
    gemma4_ghost_execution_mode: 'hybrid_metal',
    gemma4_ghost_common_metal_active: false,
    gemma4_ghost_experts_metal_active: true,
    gemma4_ghost_head_metal_active: false,
    gemma4_ghost_backend: 'metal',
    gemma4_ghost_common_gpu_active: false,
    gemma4_ghost_experts_gpu_active: true,
    gemma4_ghost_head_gpu_active: false,
  },
  'dashboard projection must preserve the exact hybrid component report',
)
assert.equal(
  executionRuntimeFields({ gemma4_ghost_common_metal_active: 'true' })
    .gemma4_ghost_common_metal_active,
  null,
  'malformed health values must fail closed instead of producing a GPU claim',
)
assert.equal(
  executionRuntimeFields({ gemma4_ghost_execution_mode: 'future_metal' })
    .gemma4_ghost_execution_mode,
  null,
  'unknown execution modes must fail closed',
)
assert.equal(
  executionRuntimeFields({ gemma4_ghost_backend: 'directml' }).gemma4_ghost_backend,
  null,
  'unknown Ghost backends must fail closed',
)
assert.equal(
  describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_execution_mode: 'hybrid_metal',
    gemma4_ghost_common_metal_active: false,
    gemma4_ghost_experts_metal_active: false,
    gemma4_ghost_head_metal_active: false,
  }).state,
  'cpu',
  'a mode string without matching live component booleans must not manufacture a Metal claim',
)
assert.equal(
  describeExecutionPlan({
    ...baseGhostRuntime,
    gemma4_ghost_backend: 'cuda',
    gemma4_ghost_execution_mode: 'full_common_cuda',
    gemma4_ghost_common_gpu_active: 'true',
    gemma4_ghost_experts_gpu_active: false,
    gemma4_ghost_head_gpu_active: false,
  }).state,
  'cpu',
  'malformed neutral component fields must not manufacture a CUDA claim',
)

const dashboardSource = readFileSync(
  new URL('../src/hooks/useDashboardData.js', import.meta.url),
  'utf8',
)
assert.match(
  dashboardSource,
  /\.\.\.executionRuntimeFields\(health\)/,
  'the health-only execution projection must reach the dashboard runtime snapshot',
)

console.log('ghost-moe execution plan smoke: ok')
