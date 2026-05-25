#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

const script = 'scripts/bench-llama3-same-host.mjs'
const tmp = await mkdtemp(join(tmpdir(), 'camelid-bench-plan-'))

try {
  const planPath = join(tmp, 'plan.json')
  const planRun = spawnSync(process.execPath, [
    script,
    '--print-plan',
    '--model', '/tmp/Camelid Test/Llama-3.2-1B-Instruct-Q8_0.gguf',
    '--model-id', 'llama32-1b-q8-plan',
    '--row-id', 'llama32_1b_instruct_q8_0',
    '--max-tokens', '8',
    '--warmup', '0',
    '--repeats', '2',
    '--threads', '4',
    '--expected-camelid-q8-policy', 'lazy_q8_linear_default_or_auto_retain',
    '--out', planPath,
  ], { encoding: 'utf8' })

  assert.equal(planRun.status, 0, planRun.stderr)
  assert.match(planRun.stdout, /harness_command=node scripts\/bench-llama3-same-host\.mjs/)
  assert.match(planRun.stdout, /claim_boundary=.*1B.*Mixtral.*separate row-specific evidence/s)

  const plan = JSON.parse(await readFile(planPath, 'utf8'))
  assert.equal(plan.schema, 'camelid.same_host_llama3_benchmark_plan.v1')
  assert.equal(plan.model.row_id, 'llama32_1b_instruct_q8_0')
  assert.equal(plan.method.max_tokens, 8)
  assert.equal(plan.method.warmup, 0)
  assert.equal(plan.method.repeats, 2)
  assert.equal(plan.method.threads, 4)
  assert.equal(plan.method.expected_marker, 'CMLD-BENCH')
  assert.equal(plan.method.expected_camelid_q8_policy, 'lazy_q8_linear_default_or_auto_retain')
  assert.equal(plan.method.require_marker, false)
  assert.equal(plan.method.unique_prompt, false)
  assert.equal(plan.method.evidence_context.model_artifact.sha256, 'not_computed_in_plan_mode')
  assert.ok(plan.method.evidence_context.host_class.cpu_count >= 1)
  assert.equal(plan.method.resource_snapshots.pre_start.label, 'pre_start')
  assert.match(plan.commands.harness, /--row-id llama32_1b_instruct_q8_0/)
  assert.match(plan.commands.harness, /'\/tmp\/Camelid Test\/Llama-3\.2-1B-Instruct-Q8_0\.gguf'/)
  assert.match(plan.commands.harness, /--expected-camelid-q8-policy lazy_q8_linear_default_or_auto_retain/)
  assert.match(plan.commands.llama_server, /--no-warmup/)
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('not tokenizer-ground-truth tokens')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('marker_presence')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('camelid_backend_generate_ms')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('camelid_q8_runtime_policy')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('FFN-down decode')))
  assert.match(plan.outputs.guardrail, /--expected-camelid-q8-policy/)
  assert.match(plan.claim_boundary, /does not widen support/)
  assert.match(plan.claim_boundary, /production-throughput/)
  assert.match(plan.claim_boundary, /Mixtral claims/)

  const uniquePlanPath = join(tmp, 'unique-plan.json')
  const uniquePlanRun = spawnSync(process.execPath, [
    script,
    '--print-plan',
    '--model', '/tmp/Camelid Test/Llama-3.2-1B-Instruct-Q8_0.gguf',
    '--model-id', 'llama32-1b-q8-plan',
    '--row-id', 'llama32_1b_instruct_q8_0',
    '--max-tokens', '8',
    '--warmup', '0',
    '--repeats', '2',
    '--unique-prompt',
    '--require-marker',
    '--expected-marker', 'CMLD-UNIQUE',
    '--out', uniquePlanPath,
  ], { encoding: 'utf8' })

  assert.equal(uniquePlanRun.status, 0, uniquePlanRun.stderr)
  const uniquePlan = JSON.parse(await readFile(uniquePlanPath, 'utf8'))
  assert.equal(uniquePlan.method.unique_prompt, true)
  assert.equal(uniquePlan.method.require_marker, true)
  assert.equal(uniquePlan.method.expected_marker, 'CMLD-UNIQUE')
  assert.match(uniquePlan.commands.harness, /--unique-prompt/)
  assert.match(uniquePlan.commands.harness, /--require-marker --expected-marker CMLD-UNIQUE/)

  const truncatedMarkerRun = spawnSync(process.execPath, [
    script,
    '--print-plan',
    '--model', '/tmp/Camelid Test/Llama-3.2-1B-Instruct-Q8_0.gguf',
    '--model-id', 'llama32-1b-q8-plan',
    '--row-id', 'llama32_1b_instruct_q8_0',
    '--max-tokens', '4',
    '--require-marker',
  ], { encoding: 'utf8' })

  assert.notEqual(truncatedMarkerRun.status, 0)
  assert.match(truncatedMarkerRun.stderr, /--require-marker needs --max-tokens >= 8/)

  const helpRun = spawnSync(process.execPath, [script, '--help'], { encoding: 'utf8' })
  assert.equal(helpRun.status, 0, helpRun.stderr)
  assert.match(helpRun.stdout, /--print-plan/)
  assert.match(helpRun.stdout, /--unique-prompt/)
  assert.match(helpRun.stdout, /--expected-camelid-q8-policy/)
  assert.match(helpRun.stdout, /CAMELID_STREAM_TIMING_DIAGNOSTICS=on/)
  assert.match(helpRun.stdout, /JSON report schema: camelid\.same_host_llama3_benchmark\.v1/)
  assert.match(helpRun.stdout, /does not promote production throughput/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}
