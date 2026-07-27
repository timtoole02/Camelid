#!/usr/bin/env node
/* Inference Observatory smoke: guards the truthfulness contract and the
   store's event reduction.

   1. The telemetry store reduces a realistic real-event sequence correctly
      (lifecycle, prefill, decode, KV, sampler, receipt, workers, errors).
   2. The view renders the honest empty states (Phase 6.1 Flow Bench: the
      no-traffic copy + the never-fake-motion promise; stillness is honest).
   3. No renderer module spawns activity outside onEvent — visual activity
      may only originate from real backend events. */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')
let failures = 0
const check = (name, fn) => {
  try {
    fn()
    console.log(`ok   ${name}`)
  } catch (err) {
    failures += 1
    console.error(`FAIL ${name}\n     ${err.message}`)
  }
}

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const { createInferenceTelemetryStore, CONNECTION } = await server.ssrLoadModule('/src/lib/inferenceTelemetry.js')
  const { default: InferenceObservatoryView } = await server.ssrLoadModule('/src/views/InferenceObservatoryView.jsx')

  // --- 1. store reduction over a realistic captured sequence -------------
  const store = createInferenceTelemetryStore()
  const sequence = [
    { event: 'hello', schema: 'camelid.telemetry/v1' },
    { event: 'inference_started', request_id: 'r1', model_id: 'Llama 3.2 3B Instruct', backend: 'metal_resident_q8_runtime', quantization: 'Q8_0', architecture: 'llama', prompt_tokens: 16, max_tokens: 24, context_length: 131072, temperature: 0, stream: true },
    { event: 'prefill_started', prefill_tokens: 15, path: 'auto', layers_total: 28 },
    { event: 'prefill_progress', tokens_done: 15, tokens_total: 15 },
    { event: 'decode_started', context_position: 15 },
    { event: 'sampler_step', chosen_token_id: 16, mode: 'greedy', candidates: [{ token_id: 16, prob: 0.56 }, { token_id: 8586, prob: 0.36 }] },
    { event: 'token_decoded', token_id: 16, context_position: 16, layers_total: 28 },
    { event: 'kv_cache_updated', position: 16, capacity: 131072, approx_bytes: 3670016 },
    { event: 'token_decoded', token_id: 271, context_position: 17, layers_total: 28 },
    { event: 'worker_node_active', node: '169.254.156.89:9300', detail: 'decode @ position 17' },
    { event: 'worker_node_idle', node: '169.254.156.89:9300' },
    { event: 'receipt_written', receipt_id: 'abc123def456', reproducible: true, gguf_sha256: 'f00d'.repeat(16) },
    { event: 'inference_finished', status: 'ok', finish_reason: 'length', completion_tokens: 24, total_ms: 1065, ttft_ms: 211, decode_tps: 26.9 },
  ]
  sequence.forEach((evt) => store.ingest(evt))

  check('store: run identity captured from inference_started', () => {
    const run = store.getLastRun()
    assert.equal(run.modelId, 'Llama 3.2 3B Instruct')
    assert.equal(run.backend, 'metal_resident_q8_runtime')
    assert.equal(run.quantization, 'Q8_0')
    assert.equal(run.promptTokens, 16)
    assert.equal(run.contextLength, 131072)
  })
  check('store: prefill/decode/kv/sampler state reduced', () => {
    const run = store.getLastRun()
    assert.equal(run.prefill.done, 15)
    assert.equal(run.decode.tokens, 2)
    assert.deepEqual(run.generatedTokenIds, [16, 271])
    assert.equal(run.kv.position, 17)
    assert.equal(run.kv.approxBytes, 3670016)
    assert.equal(run.lastSampler.chosenTokenId, 16)
    assert.equal(run.layersTotal, 28)
  })
  check('store: finish + receipt + workers recorded', () => {
    const run = store.getLastRun()
    assert.equal(run.finish.status, 'ok')
    assert.equal(run.finish.ttftMs, 211)
    assert.equal(run.receipt.reproducible, true)
    assert.equal(run.active, false)
    const worker = store.getWorkers().get('169.254.156.89:9300')
    assert.equal(worker.status, 'idle')
  })
  check('store: inference_error surfaces and marks the run', () => {
    const s2 = createInferenceTelemetryStore()
    s2.ingest({ event: 'inference_started', request_id: 'r2', model_id: 'm', backend: 'b', quantization: 'q', architecture: 'a', prompt_tokens: 1, max_tokens: 8, context_length: 64, temperature: 0, stream: false })
    s2.ingest({ event: 'inference_error', code: 'generation_step_failed', message: 'boom' })
    assert.equal(s2.getRun().phase, 'error')
    assert.equal(s2.getRun().errors.length, 1)
  })
  check('store: joining mid-run activates from run-scoped events', () => {
    const s4 = createInferenceTelemetryStore()
    // Tab opened while a generation is already in flight: no inference_started seen.
    s4.ingest({ event: 'token_decoded', token_id: 42, context_position: 9, layers_total: 28, request_id: 'r9', model_id: 'mid-run-model' })
    assert.equal(s4.getRun().active, true)
    assert.equal(s4.getRun().modelId, 'mid-run-model')
    s4.ingest({ event: 'inference_finished', status: 'ok', finish_reason: 'stop', completion_tokens: 3, total_ms: 100, request_id: 'r9' })
    assert.equal(s4.getRun().active, false)
    // Trailing throttled event of the closed run must not reactivate it.
    s4.ingest({ event: 'kv_cache_updated', position: 10, capacity: 64, request_id: 'r9' })
    assert.equal(s4.getRun().active, false)
  })
  check('store: idle store reports no run and drains nothing', () => {
    const s3 = createInferenceTelemetryStore()
    assert.equal(s3.getRun().active, false)
    assert.deepEqual(s3.drainEvents(), [])
    assert.equal(s3.getConnection(), CONNECTION.CONNECTING)
  })

  // --- 2. honest empty states render --------------------------------------
  check('view: honest no-traffic state renders with no requests', () => {
    const html = renderToStaticMarkup(React.createElement(InferenceObservatoryView, { apiBase: 'http://127.0.0.1:1' }))
    assert.ok(html.includes('No session traffic yet'), 'honest empty copy missing')
    assert.ok(html.includes('The Neural Field'), 'default renderer title missing')
    assert.ok(html.includes('neuralfield'), 'default renderer stage missing')
    assert.ok(html.includes('operational telemetry'), 'not-evidence affordance missing')
  })
  check('view: never-fake-motion promise exists in source', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/views/InferenceObservatoryView.jsx'), 'utf8')
    assert.ok(source.includes('never fake motion'))
    assert.ok(source.includes('the bench stays still until a real request runs'))
  })

  // --- 3. renderers only animate from real events --------------------------
  check('renderers: particle spawns happen only in onEvent handlers', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/lib/observatory/tokenParticles.js'), 'utf8')
    const drawBody = source.slice(source.indexOf('draw(ctx, frame)'))
    assert.ok(!/spawn/i.test(drawBody), 'draw() must not spawn particles; only onEvent may')
  })
  check('renderers: no module fabricates telemetry', () => {
    for (const file of ['tokenParticles.js', 'layerVisualizer.js', 'kvCacheTrail.js', 'samplerBloom.js', 'clusterConstellation.js']) {
      const source = readFileSync(resolve(frontendRoot, `src/lib/observatory/${file}`), 'utf8')
      assert.ok(!/(mock|simulate|fake|demoEvent)/i.test(source), `${file} contains a simulation marker`)
    }
    const storeSource = readFileSync(resolve(frontendRoot, 'src/lib/inferenceTelemetry.js'), 'utf8')
    assert.ok(!/setInterval\([^)]*ingest/.test(storeSource), 'store must not self-feed events')
  })
} finally {
  await server.close()
}

if (failures > 0) {
  console.error(`\n${failures} observatory smoke check(s) failed`)
  process.exit(1)
}
console.log('\nobservatory smoke: all checks passed')
