#!/usr/bin/env node
/* Retrieval reranking — contract, response normalization, and score honesty.
 *
 * Two defects would ship a misleading surface rather than a broken one:
 *
 *   1. SCORES READ AS PROBABILITIES. `relevance_score` is a cosine similarity.
 *      Observed live against this engine, an obviously irrelevant document still
 *      scored 0.373 while the best match scored 0.638 — so a percentage rendering
 *      would tell a reader that an unrelated document is "37% relevant". The
 *      useful signal is order, gaps, and movement. Asserted here as a rendering
 *      rule, not left to reviewer memory.
 *
 *   2. SCORES MISMATCHED TO TEXTS. The engine returns results by `index` into the
 *      submitted array and OMITS `document` unless asked. Reading response order
 *      as document order, or assuming `document` is present, silently pairs the
 *      wrong score with the wrong text — and the output still looks plausible.
 *
 * Fixtures reproduce the live response shape captured from this engine.
 *
 * SSR component test — no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
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

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const mod = await server.ssrLoadModule('/src/lib/embeddingRerank.js')
  const { RerankPlayground } = await server.ssrLoadModule('/src/components/models/RerankPlayground.jsx')
  const {
    MAX_RERANK_DOCUMENTS,
    readRerankContract,
    resolveEncoder,
    splitDocuments,
    rerankReadiness,
    rerankRequestBody,
    normalizeRerank,
    formatSimilarity,
    formatMovement,
  } = mod

  const live = {
    api_features: [{ id: 'embedding_similarity_reranking', status: 'supported_exact_model_row', notes: 'Bi-encoder reranking on the exact Nomic row.' }],
    api_conformance: [{ id: 'embedding_similarity_reranking', status: 'supported_exact_model_row', supported_modes: ['string_documents', 'top_n'] }],
  }
  const DOCS = [
    'Bake the sourdough at 230C for 35 minutes on a stone.',
    'Thermal throttling happens when the CPU exceeds its junction temperature.',
    'The Treaty of Westphalia was signed in 1648.',
    'Elevate the chassis for airflow and avoid using it on a duvet.',
  ]
  /* The live shape: results ordered by score, addressed by `index`, `document`
     absent because return_documents was false. */
  const LIVE_PAYLOAD = {
    id: 'rerank-1',
    model: 'nomic-embed-text-v1.5',
    results: [
      { index: 1, relevance_score: 0.6381 },
      { index: 3, relevance_score: 0.5894 },
      { index: 0, relevance_score: 0.4988 },
      { index: 2, relevance_score: 0.3729 },
    ],
    usage: { prompt_tokens: 126, total_tokens: 126 },
  }

  console.log('rerank — contract and encoder resolution')

  check('a live contract is supported; an absent row fails closed', () => {
    assert.equal(readRerankContract(live).supported, true)
    assert.equal(readRerankContract({}).present, false)
    assert.equal(readRerankContract({}).supported, false)
  })

  check('resemblance is not evidence — a near-miss row id does not count', () => {
    assert.equal(readRerankContract({ api_conformance: [{ id: 'embedding_similarity_reranking_v2', status: 'supported' }] }).present, false)
  })

  check('an encoder is resolved from the served models, and its support scope is separate', () => {
    const found = resolveEncoder([{ id: 'nomic-embed-text-v1.5' }, { id: 'Llama 3.2 1B Instruct' }], [])
    assert.equal(found.ready, true)
    assert.equal(found.encoderId, 'nomic-embed-text-v1.5')
    assert.equal(found.isSupportedRow, true, 'the pinned artifact carries the exact-row claim')
    const none = resolveEncoder([{ id: 'Llama 3.2 1B Instruct' }], [])
    assert.equal(none.ready, false)
    assert.equal(none.encoderId, null)
  })

  check('a chat-only engine is guarded with copy that does not imply eviction', () => {
    const readiness = rerankReadiness({
      contract: readRerankContract(live),
      encoder: resolveEncoder([{ id: 'Llama 3.2 1B Instruct' }], []),
      query: 'q',
      documents: DOCS,
    })
    assert.equal(readiness.ready, false)
    assert.match(readiness.reason, /sidecar|does not replace/i,
      'the copy must say the encoder loads alongside the chat model, not instead of it')
  })

  console.log('rerank — request composition')

  check('the document cap is enforced before the wire', () => {
    const tooMany = Array.from({ length: MAX_RERANK_DOCUMENTS + 1 }, (_, i) => `doc ${i}`)
    const readiness = rerankReadiness({ contract: readRerankContract(live), encoder: { ready: true, encoderId: 'e' }, query: 'q', documents: tooMany })
    assert.equal(readiness.ready, false)
    assert.match(readiness.reason, new RegExp(String(MAX_RERANK_DOCUMENTS)))
  })

  check('fewer than two documents is refused — ranking one thing is meaningless', () => {
    const readiness = rerankReadiness({ contract: readRerankContract(live), encoder: { ready: true, encoderId: 'e' }, query: 'q', documents: ['only one'] })
    assert.equal(readiness.ready, false)
  })

  check('blank lines never become empty documents', () => {
    assert.deepEqual(splitDocuments('a\n\n  \nb\n'), ['a', 'b'])
  })

  check('the body asks for indices, not documents', () => {
    const body = rerankRequestBody({ encoderId: 'e', query: '  q  ', documents: DOCS })
    assert.equal(body.query, 'q')
    assert.equal(body.return_documents, false, 'text comes from the caller list, so it need not be echoed')
    assert.equal(body.top_n, DOCS.length)
  })

  console.log('rerank — normalization cannot mismatch a score to a text')

  check('results are joined back by index, not by response order', () => {
    const model = normalizeRerank(LIVE_PAYLOAD, DOCS)
    assert.equal(model.ranked.length, 4)
    // Rank 1 is index 1 — the thermal doc — NOT DOCS[0].
    assert.equal(model.ranked[0].index, 1)
    assert.equal(model.ranked[0].text, DOCS[1])
    assert.match(model.ranked[0].text, /Thermal throttling/)
    // Reading response order as document order would have paired 0.6381 with the
    // sourdough line.
    assert.notEqual(model.ranked[0].text, DOCS[0])
  })

  check('movement is computed against the submitted order', () => {
    const model = normalizeRerank(LIVE_PAYLOAD, DOCS)
    assert.equal(model.ranked[0].movedBy, 1, 'index 1 rose to position 0')
    assert.equal(model.ranked[2].movedBy, -2, 'index 0 fell to position 2')
    assert.equal(model.stats.moved, 4)
    assert.equal(formatMovement(0), 'unchanged')
    assert.equal(formatMovement(2), 'up 2')
    assert.equal(formatMovement(-3), 'down 3')
  })

  check('an out-of-range index is dropped rather than paired with undefined text', () => {
    const model = normalizeRerank({ results: [{ index: 99, relevance_score: 0.5 }, { index: 0, relevance_score: 0.4 }] }, DOCS)
    assert.equal(model.ranked.length, 1)
    assert.equal(model.ranked[0].index, 0)
  })

  check('an empty or malformed payload yields nothing rather than a broken table', () => {
    assert.equal(normalizeRerank(null, DOCS), null)
    assert.equal(normalizeRerank({ results: [] }, DOCS), null)
  })

  console.log('rerank — a similarity is not a probability')

  check('scores render as bare similarities, never as percentages', () => {
    assert.equal(formatSimilarity(0.6381), '0.638')
    assert.equal(formatSimilarity(0.3729), '0.373')
    assert.equal(formatSimilarity(null), '—')
    // Three decimals because neighbouring scores often differ in the third place.
    assert.doesNotMatch(formatSimilarity(0.5), /%/)
  })

  check('the spread is reported, because one score alone says little', () => {
    const model = normalizeRerank(LIVE_PAYLOAD, DOCS)
    assert.ok(Math.abs(model.stats.spread - (0.6381 - 0.3729)) < 1e-6)
  })

  check('no rendered copy converts a score to a percentage or calls it a probability', () => {
    // Rendered WITH results: the score-interpretation copy lives in that state, so
    // an empty-form render would let this assertion pass against markup that never
    // showed a score at all.
    const html = renderToStaticMarkup(React.createElement(RerankPlayground, {
      apiBase: '', capabilities: live, initialResult: normalizeRerank(LIVE_PAYLOAD, DOCS),
    }))
    assert.match(html, /0.638/, 'the ranked table must actually render')
    assert.doesNotMatch(html, /\d%/, 'a similarity must never be rendered as a percentage')
    assert.match(html, /not probabilities/i, 'the surface must say what the score is not')
    assert.match(html, /alongside/i, 'and that the encoder does not replace the chat model')
  })

  check('the source never converts a score to a percentage', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/lib/embeddingRerank.js'), 'utf8')
    assert.doesNotMatch(source, /\*\s*100|toFixed\(\d\)\s*\+\s*'%'/, 'no score-to-percent conversion may exist')
  })

  console.log(`\n${checks} checks passed`)
} finally {
  await server.close()
}
