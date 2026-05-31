#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawn, spawnSync } from 'node:child_process'
import { createServer } from 'node:http'

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
  assert.equal(plan.method.require_marker, false)
  assert.equal(plan.method.unique_prompt, false)
  assert.equal(plan.method.evidence_context.model_artifact.sha256, 'not_computed_in_plan_mode')
  assert.ok(plan.method.evidence_context.host_class.cpu_count >= 1)
  assert.equal(plan.method.resource_snapshots.pre_start.label, 'pre_start')
  assert.match(plan.commands.harness, /--row-id llama32_1b_instruct_q8_0/)
  assert.match(plan.commands.harness, /'\/tmp\/Camelid Test\/Llama-3\.2-1B-Instruct-Q8_0\.gguf'/)
  assert.match(plan.commands.llama_server, /--no-warmup/)
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('not tokenizer-ground-truth tokens')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('marker_presence')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('camelid_backend_generate_ms')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('FFN-down decode')))
  assert.match(plan.outputs.guardrail, /--require-marker/)
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

  const scrubbedPlanPath = join(tmp, 'scrubbed-plan.json')
  const scrubbedPlanRun = spawnSync(process.execPath, [
    script,
    '--print-plan',
    '--model', '/tmp/Camelid Test/private-models/Llama-3.2-3B-Instruct-Q8_0.gguf',
    '--backend-bin', '/tmp/Camelid Test/private-build/camelid',
    '--llama-server', '/tmp/Camelid Test/private-reference/llama-server',
    '--model-id', 'llama32-3b-q8-plan',
    '--row-id', 'llama32_3b_instruct_q8_0',
    '--max-tokens', '8',
    '--warmup', '0',
    '--repeats', '1',
    '--out', scrubbedPlanPath,
    '--scrub-local-paths',
  ], { encoding: 'utf8' })

  assert.equal(scrubbedPlanRun.status, 0, scrubbedPlanRun.stderr)
  assert.doesNotMatch(scrubbedPlanRun.stdout, /\/tmp\/Camelid Test/)
  assert.match(scrubbedPlanRun.stdout, /--scrub-local-paths/)
  const scrubbedPlan = JSON.parse(await readFile(scrubbedPlanPath, 'utf8'))
  const scrubbedPlanText = JSON.stringify(scrubbedPlan)
  assert.doesNotMatch(scrubbedPlanText, /\/tmp\/Camelid Test/)
  assert.doesNotMatch(scrubbedPlanText, /\/Users\/|\/home\//)
  assert.equal(scrubbedPlan.model.model_path, '<redacted-model>/Llama-3.2-3B-Instruct-Q8_0.gguf')
  assert.equal(scrubbedPlan.model.model_path_redacted, true)
  assert.equal(scrubbedPlan.method.evidence_context.model_artifact.path_redacted, true)
  assert.equal(scrubbedPlan.method.resource_snapshots.pre_start.storage.path_redacted, true)
  assert.match(scrubbedPlan.commands.harness, /--model '<redacted-model>\/Llama-3\.2-3B-Instruct-Q8_0\.gguf'/)
  assert.match(scrubbedPlan.commands.camelid_serve, /^camelid serve --addr/)
  assert.match(scrubbedPlan.commands.llama_server, /^llama-server /)
  assert.match(scrubbedPlan.method.evidence_context.privacy_note, /redacted for public-safe evidence/)

  const camelidServer = await startFakeCamelidServer()
  const llamaServer = await startFakeLlamaServer()
  try {
    const scrubbedReportPath = join(tmp, 'scrubbed-report.json')
    const reportRun = await spawnNode([
      script,
      '--model', '/tmp/Camelid Test/private-models/Llama-3.2-3B-Instruct-Q8_0.gguf',
      '--backend-bin', '/tmp/Camelid Test/private-build/camelid',
      '--llama-server', '/tmp/Camelid Test/private-reference/llama-server',
      '--backend', camelidServer.url,
      '--llama-url', llamaServer.url,
      '--model-id', 'llama32-3b-q8-report',
      '--row-id', 'llama32_3b_instruct_q8_0',
      '--max-tokens', '8',
      '--warmup', '0',
      '--repeats', '1',
      '--start-backend=false',
      '--start-llama-server=false',
      '--require-marker',
      '--out', scrubbedReportPath,
      '--scrub-local-paths',
    ])

    assert.equal(reportRun.status, 0, reportRun.stderr)
    const scrubbedReport = JSON.parse(await readFile(scrubbedReportPath, 'utf8'))
    const scrubbedReportText = JSON.stringify(scrubbedReport)
    assert.equal(scrubbedReport.schema, 'camelid.same_host_llama3_benchmark.v1')
    assert.equal(scrubbedReport.guardrails.passed, true)
    assert.equal(scrubbedReport.llama_cpp.binary, '<redacted-binary>/llama-server')
    assert.equal(scrubbedReport.llama_cpp.binary_path_redacted, true)
    assert.doesNotMatch(scrubbedReportText, /\/tmp\/Camelid Test/)
    assert.doesNotMatch(scrubbedReportText, /\/Users\/|\/home\//)
  } finally {
    await Promise.all([camelidServer.close(), llamaServer.close()])
  }

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
  assert.match(helpRun.stdout, /--scrub-local-paths/)
  assert.match(helpRun.stdout, /CAMELID_STREAM_TIMING_DIAGNOSTICS=on/)
  assert.match(helpRun.stdout, /JSON report schema: camelid\.same_host_llama3_benchmark\.v1/)
  assert.match(helpRun.stdout, /does not promote production throughput/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}

function spawnNode(argv) {
  return new Promise((resolvePromise) => {
    const child = spawn(process.execPath, argv, {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('close', (status) => resolvePromise({ status, stdout, stderr }))
  })
}

async function startFakeCamelidServer() {
  return startServer((request, response) => {
    if (request.url === '/v1/health') return json(response, { ok: true })
    if (request.url === '/api/models/load') return json(response, { ok: true })
    if (request.url === '/v1/chat/completions') {
      return sse(response, [
        { choices: [{ delta: { content: 'CMLD-' } }] },
        {
          choices: [{ delta: { content: 'BENCH' } }],
          camelid: {
            stream_timing_diagnostics: {
              timings_ms: { generate: 12, first_content: 3 },
              q8_schedule: { i8mm_single_projection_calls: 2 },
            },
          },
        },
      ])
    }
    response.writeHead(404).end()
  })
}

async function startFakeLlamaServer() {
  return startServer((request, response) => {
    if (request.url === '/health') return json(response, { status: 'ok' })
    if (request.url === '/completion') return sse(response, [
      { content: 'CMLD-' },
      { content: 'BENCH' },
    ])
    response.writeHead(404).end()
  })
}

function startServer(handler) {
  const server = createServer(handler)
  return new Promise((resolvePromise, rejectPromise) => {
    server.once('error', rejectPromise)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      resolvePromise({
        url: `http://127.0.0.1:${address.port}`,
        close: () => new Promise((resolveClose) => server.close(resolveClose)),
      })
    })
  })
}

function json(response, body) {
  response.writeHead(200, { 'content-type': 'application/json' })
  response.end(`${JSON.stringify(body)}\n`)
}

function sse(response, payloads) {
  response.writeHead(200, { 'content-type': 'text/event-stream' })
  for (const payload of payloads) response.write(`data: ${JSON.stringify(payload)}\n\n`)
  response.end('data: [DONE]\n\n')
}
