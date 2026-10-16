#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import { exportableConversation, exportableConversations } from '../src/lib/conversationExport.js'
import { normalizeStoredMessage } from '../src/lib/conversationStorage.js'
import { withVariantAppended, withActiveVariant } from '../src/lib/messageVariants.js'
import { authoritativeOutputRate, requestOutputRate } from '../src/lib/nativeGenerationMetrics.js'
import { exportTelemetryJson, getTelemetrySnapshot, rateSourceLabel, recordChatGeneration, summarizeTelemetry } from '../src/lib/telemetryLog.js'

const requestMessage = {
  id: 'request-rate', role: 'assistant', content: 'Paris.',
  tokens_out_per_sec: requestOutputRate(7, 1500),
  output_rate_source: 'client_request', elapsed_ms: 1500,
  usage: { prompt_tokens: 24, completion_tokens: 7, total_tokens: 31 },
  usage_source: 'backend',
}
const restored = normalizeStoredMessage(JSON.parse(JSON.stringify(requestMessage)))
assert.equal(restored.output_rate_source, 'client_request')
assert.equal(restored.tokens_out_per_sec, 7 / 1.5)
const nativeMessage = {
  ...requestMessage, output_rate_source: 'backend_native', tokens_out_per_sec: 50,
  camelid: { mtp12: { lossless_target_verified: true, decode_tokens_per_second: 50 } },
}
const variants = normalizeStoredMessage(JSON.parse(JSON.stringify(withVariantAppended(requestMessage, nativeMessage))))
assert.equal(variants.output_rate_source, 'backend_native')
assert.equal(withActiveVariant(variants, 0).output_rate_source, 'client_request')
assert.equal(authoritativeOutputRate({ ...nativeMessage, tokens_out_per_sec: 999 }), 50)

const chat = { id: 'clock-export', messages: [requestMessage, nativeMessage] }
const single = exportableConversation(chat)
const bulk = exportableConversations([chat])
for (const exported of [single, bulk.conversations[0]]) {
  assert.equal(exported.messages[0].output_rate_source, 'client_request')
  assert.equal(exported.messages[0].usage_source, 'backend', 'backend token counts must not change the request clock source')
  assert.equal(exported.messages[1].output_rate_source, 'backend_native')
  assert.match(exported.telemetry_note, /client_request includes prefill/)
  assert.match(exported.telemetry_note, /Token counts use usage_source/)
  assert.match(exported.telemetry_note, /Output rates use output_rate_source/)
}
assert.equal(bulk.telemetry_note, single.telemetry_note)

for (const [rateSource, tokensPerSec] of [['client_request', 4], ['client_request', 6], ['backend_native', 50], [undefined, 12]]) {
  recordChatGeneration({ modelId: 'same-model', tokensPerSec, rateSource, usageSource: 'backend' })
}
const { requests } = getTelemetrySnapshot()
const summary = summarizeTelemetry(requests)
assert.deepEqual(summary.outputRates.map(({ source, median }) => [source, median]), [
  ['client_request', 5], ['backend_native', 50], ['unspecified', 12],
])
assert.equal(summary.medianTokensPerSec, null, 'different clocks must not be combined into one median')
assert.equal(rateSourceLabel(undefined), 'unspecified clock', 'historical values must not acquire an invented source')
const records = JSON.parse(exportTelemetryJson()).requests
assert.equal(records[0].rateSource, 'client_request')
assert.equal(records[2].rateSource, 'backend_native')
assert.equal(records[3].rateSource, 'unspecified')

const hook = await readFile(new URL('../src/hooks/useDashboardData.js', import.meta.url), 'utf8')
assert.match(hook, /rateSource: assistantMessage\.output_rate_source/, 'the chat completion must hand its clock source to session telemetry')
const observatory = await readFile(new URL('../src/views/InferenceObservatoryView.jsx', import.meta.url), 'utf8')
assert.match(observatory, /summary\.outputRates\.map/, 'Observatory must keep the same separated clocks as Telemetry')

const vite = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'silent' })
try {
  // Vite's module instance owns a separate telemetry store from Node's imports.
  const telemetry = await vite.ssrLoadModule('/src/lib/telemetryLog.js')
  for (const record of requests) telemetry.recordChatGeneration(record)
  const { default: TelemetryView } = await vite.ssrLoadModule('/src/views/TelemetryView.jsx')
  const html = renderToStaticMarkup(React.createElement(TelemetryView))
  assert.match(html, /request incl\. prefill/)
  assert.match(html, /backend decode/)
  assert.match(html, /unspecified clock/)
  assert.match(html, /4\.0 tok\/s · request incl\. prefill/, 'the request log must label each rate by its clock')
  assert.doesNotMatch(html, /Decode median|NaN|Infinity/)
} finally {
  await vite.close()
}
console.log('Request rate provenance smoke passed')
