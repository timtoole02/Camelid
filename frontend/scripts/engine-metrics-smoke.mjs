#!/usr/bin/env node
/* Engine metrics and active-configuration readout.
 *
 * Three classes of defect here would ship a confident wrong number rather than a
 * visible break:
 *
 *   1. A LIFETIME AVERAGE PRESENTED AS A CURRENT RATE. Every counter the engine
 *      exposes is cumulative since process start, so any ratio derived from two of
 *      them describes the whole run. A server that was fast for an hour and has
 *      been slow for a minute still reports the hour. The copy must say so.
 *
 *   2. ZERO ATTEMPTS RENDERED AS A ZERO RATE. A cache that has never been
 *      consulted has an UNKNOWN hit rate, not a 0% one — reporting 0% accuses a
 *      cache of failing when nothing ever asked it.
 *
 *   3. A NON-METRICS 200 READ AS AN IDLE ENGINE. A proxy that answers /metrics
 *      with its own HTML returns 200, and a parser that shrugs at that reports an
 *      engine with nothing to say instead of a routing fault. This actually
 *      happened during development: the dev server proxied only /api and /v1.
 *
 * The fixture is the real exposition captured from a live engine.
 *
 * SSR component test — no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

/* Captured verbatim from a live engine (trimmed to the families under test). */
const LIVE_METRICS = `# HELP camelid_http_requests_total HTTP requests completed.
# TYPE camelid_http_requests_total counter
camelid_http_requests_total 4
# HELP camelid_http_failures_total HTTP responses with a 4xx or 5xx status.
# TYPE camelid_http_failures_total counter
camelid_http_failures_total 0
# TYPE camelid_generation_requests_total counter
camelid_generation_requests_total 2
# TYPE camelid_generation_failures_total counter
camelid_generation_failures_total 0
# TYPE camelid_prompt_tokens_total counter
camelid_prompt_tokens_total 22
# TYPE camelid_decode_tokens_total counter
camelid_decode_tokens_total 9
# TYPE camelid_prompt_evaluation_duration_seconds_sum counter
camelid_prompt_evaluation_duration_seconds_sum 1.332763
# TYPE camelid_decode_duration_seconds_sum counter
camelid_decode_duration_seconds_sum 0.051702
# TYPE camelid_prompt_cache_hits_total counter
camelid_prompt_cache_hits_total 0
# TYPE camelid_prompt_cache_misses_total counter
camelid_prompt_cache_misses_total 2
# TYPE camelid_weight_cache_hits_total counter
camelid_weight_cache_hits_total 1
# TYPE camelid_weight_cache_misses_total counter
camelid_weight_cache_misses_total 1
# TYPE camelid_engine_queue_depth gauge
camelid_engine_queue_depth 0
# TYPE camelid_process_resident_memory_bytes gauge
camelid_process_resident_memory_bytes 1977520128
# TYPE camelid_cuda_vram_total_bytes gauge
camelid_cuda_vram_total_bytes 6441926656
# TYPE camelid_cuda_vram_free_bytes gauge
camelid_cuda_vram_free_bytes 469762048
`

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const mod = await server.ssrLoadModule('/src/lib/runtimeMetrics.js')
  const { EngineMetricsPanel } = await server.ssrLoadModule('/src/components/analytics/EngineMetricsPanel.jsx')
  const {
    parsePrometheusText,
    summarizeMetrics,
    readActiveLane,
    LANE_RISK,
    formatShare,
    formatBytes,
  } = mod

  console.log('engine metrics — parsing')

  check('the live exposition parses into named values', () => {
    const { values, types } = parsePrometheusText(LIVE_METRICS)
    assert.equal(values.get('camelid_prompt_tokens_total'), 22)
    assert.equal(values.get('camelid_cuda_vram_free_bytes'), 469762048)
    assert.equal(types.get('camelid_engine_queue_depth'), 'gauge')
  })

  check('a labelled series is ignored rather than guessed at', () => {
    // The engine emits no labels. If a future build does, summing across them
    // would invent a number it never reported.
    const { values } = parsePrometheusText('# TYPE x counter\nx{a="b"} 5\nx 7\n')
    assert.equal(values.get('x'), 7)
    assert.equal(values.size, 1)
  })

  check('non-metrics text yields nothing rather than a partial reading', () => {
    assert.equal(summarizeMetrics('<!doctype html><html><body>hi</body></html>'), null)
    assert.equal(summarizeMetrics(''), null)
  })

  console.log('engine metrics — derived figures are lifetime averages')

  const summary = summarizeMetrics(LIVE_METRICS)

  check('the prefill share is computed from server-reported forward time', () => {
    // 1.332763 / (1.332763 + 0.051702) — the number a browser cannot measure.
    assert.ok(Math.abs(summary.lifetime.prefillShare - 0.96266) < 0.001)
    assert.equal(formatShare(summary.lifetime.prefillShare), '96.3%')
  })

  check('a cache consulted and missed is 0%, and one never consulted is unknown', () => {
    // Consulted twice, hit zero times: 0% is the correct reading.
    assert.equal(summary.lifetime.promptCacheHitRate, 0)
    assert.equal(summary.lifetime.promptCacheAttempts, 2)
    // Never consulted: unknown, NOT zero.
    const idle = summarizeMetrics('# TYPE camelid_prompt_cache_hits_total counter\ncamelid_prompt_cache_hits_total 0\n# TYPE camelid_prompt_cache_misses_total counter\ncamelid_prompt_cache_misses_total 0\n')
    assert.equal(idle.lifetime.promptCacheHitRate, null, 'zero attempts must not read as a zero hit rate')
    assert.equal(formatShare(null), '—')
  })

  check('an absent metric is null, never zero', () => {
    const sparse = summarizeMetrics('# TYPE camelid_engine_queue_depth gauge\ncamelid_engine_queue_depth 3\n')
    assert.equal(sparse.live.queueDepth, 3)
    assert.equal(sparse.counters.promptTokens, null, 'a metric this build does not emit is unknown')
    assert.equal(sparse.lifetime.prefillShare, null)
    assert.ok(sparse.missing.length > 0, 'and the absence is reported')
  })

  check('VRAM in use is derived, and only when both ends are reported', () => {
    assert.equal(summary.memory.vramUsedBytes, 6441926656 - 469762048)
    assert.equal(formatBytes(summary.memory.vramUsedBytes), '5.6 GB')
    const partial = summarizeMetrics('# TYPE camelid_cuda_vram_total_bytes gauge\ncamelid_cuda_vram_total_bytes 100\n')
    assert.equal(partial.memory.vramUsedBytes, null)
  })

  console.log('engine metrics — the active lane')

  const cudaRuntime = {
    execution_plan: {
      selected_backend: 'cuda_resident_q8_runtime',
      decode_path: 'q8_0_cuda_resident_decode',
      prefill_path: 'q8_0_cuda_resident_prefill',
      cuda_resident_active: true,
      thread_count: 16,
      support_level: 'supported_exact_row_smoke',
      reasons: ['profile=auto default'],
    },
    q8_runtime: { policy: 'resident_q8_default' },
  }

  check('an evidenced backend is reported as such', () => {
    const lane = readActiveLane(cudaRuntime)
    assert.equal(lane.risk, LANE_RISK.EVIDENCED)
    assert.ok(lane.rows.length >= 5)
  })

  check('an unrecognised backend is not accused of being wrong', () => {
    const lane = readActiveLane({ execution_plan: { selected_backend: 'some_future_lane' } })
    assert.equal(lane.risk, LANE_RISK.UNEVIDENCED)
  })

  check('no execution plan claims nothing either way', () => {
    assert.equal(readActiveLane({}).risk, LANE_RISK.UNKNOWN)
    assert.equal(readActiveLane({}).present, false)
  })

  check('exactly one lever is described as changeable while running', () => {
    const lane = readActiveLane(cudaRuntime)
    const live = lane.rows.filter((row) => row.changeable === 'gpu_toggle')
    assert.equal(live.length, 1, 'only the GPU setting reaches a running engine')
    assert.equal(live[0].key, 'cuda_resident_active')
    assert.ok(lane.rows.filter((row) => row.changeable === 'restart').length >= 4)
  })

  console.log('engine metrics — rendered copy')

  const render = (props) => renderToStaticMarkup(React.createElement(EngineMetricsPanel, {
    apiBase: '', runtime: cudaRuntime, initialMetricsText: LIVE_METRICS, ...props,
  }))

  check('derived figures are labelled as lifetime, never as current', () => {
    const html = render({})
    assert.match(html, /96\.3%/, 'the prefill share must render')
    assert.match(html, /cumulative since the engine started/i)
    assert.match(html, /Lifetime average|whole run|Not a current rate/i)
  })

  check('the panel states why it offers no controls', () => {
    const html = render({})
    assert.match(html, /Restart required/)
    assert.match(html, /latched for the life of the process/i,
      'the absence of switches must be explained, not merely implied')
  })

  check('an unevidenced lane is not styled or worded as an error', () => {
    const html = renderToStaticMarkup(React.createElement(EngineMetricsPanel, {
      apiBase: '', initialMetricsText: LIVE_METRICS,
      runtime: { execution_plan: { selected_backend: 'some_future_lane' } },
    }))
    /* Scoped to the lane block: the metrics section legitimately has a "Failed"
       counter, and matching that would be a false positive about the lane's tone. */
    const lane = html.match(/<div class="engmetrics__lane[\s\S]*?<\/div>/)
    assert.ok(lane, 'the lane block must render')
    assert.doesNotMatch(lane[0], /\bfail(ed|ure|s)\b/i, 'an unevidenced lane is not a failure')
    assert.doesNotMatch(lane[0], /\berror\b|\bunsafe\b|\bdanger/i, 'and is not styled as an alarm')
    assert.match(lane[0], /does not mean the output is wrong/i, 'it must say plainly what it is not claiming')
  })

  check('nothing renders a metric the engine did not report', () => {
    const html = renderToStaticMarkup(React.createElement(EngineMetricsPanel, {
      apiBase: '', runtime: cudaRuntime,
      initialMetricsText: '# TYPE camelid_engine_queue_depth gauge\ncamelid_engine_queue_depth 0\n',
    }))
    // Every unreported figure is an em-dash, not a zero.
    assert.match(html, /—/)
    assert.match(html, /not reported by this engine build/i)
  })

  console.log(`\n${checks} checks passed`)
} finally {
  await server.close()
}
