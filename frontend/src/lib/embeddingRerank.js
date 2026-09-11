/* Retrieval reranking contract lane.

   The engine ships a bi-encoder that scores a query against a set of documents
   by cosine similarity in embedding space. That is a different capability from
   the chat model's, served by a separate CPU encoder runtime, and it is what the
   existing "Chat with Documents" flow does NOT use — that retrieval is purely
   lexical (SQLite full-text), so it can only find documents that share words with
   the query.

   TWO THINGS THIS MODULE EXISTS TO GET RIGHT.

   1. A RELEVANCE SCORE IS NOT A PROBABILITY. `relevance_score` is a cosine
      similarity. Observed live, an obviously irrelevant document still scored
      0.37 against an unrelated query, while the best match scored 0.64. Rendering
      those as "37% relevant" and "64% relevant" would be badly wrong: the
      absolute value carries almost nothing, the useful signal is the ORDER and
      the GAPS between neighbours. Nothing here converts a score to a percentage,
      and the formatter keeps it as a bare similarity.

   2. THE ENCODER IS A SIDECAR, NOT A REPLACEMENT. The engine holds loaded models
      in a map: an encoder loads with `replace:false, set_active:false` and the
      chat model stays resident and generation-ready alongside it (verified live —
      both appear in /v1/models, active_model_id unchanged). So this surface must
      never imply the user is trading their chat model away to use it. */

import { isSupportedCapabilityStatus, displayCapabilityCopy } from './capabilities.js'

const ROW_ID = 'embedding_similarity_reranking'

/* The engine caps a rerank call at this many documents. Mirrored so the UI can
   refuse before the wire rather than surfacing a typed 400. */
export const MAX_RERANK_DOCUMENTS = 255

function findRow(rows, id) {
  return (rows || []).find((row) => row?.id === id) || null
}

/* Resolve the contract by EXACT row id, preferring the conformance projection —
   it carries the machine-readable modes and ships no `notes`, so it cannot smuggle
   a vendor name into rendered copy. */
export function readRerankContract(capabilities) {
  const conformance = findRow(capabilities?.api_conformance, ROW_ID)
  const feature = findRow(capabilities?.api_features, ROW_ID)
  const row = conformance || feature
  if (!row) {
    return { present: false, supported: false, rowId: ROW_ID, status: null, note: null }
  }
  return {
    present: true,
    supported: isSupportedCapabilityStatus(String(row.status || '')),
    rowId: ROW_ID,
    status: row.status || null,
    note: feature?.notes ? displayCapabilityCopy(feature.notes) : null,
  }
}

/* Find a resident encoder among the served models.

   The row is `supported_exact_model_row`, so support is scoped to one artifact —
   but the engine decides embedding-capability from GGUF architecture alone, which
   means other encoders will also answer. Both facts are reported so the surface
   can run against what is loaded while being explicit about which one carries the
   support claim. */
export function resolveEncoder(models, localRecords) {
  const served = (models || []).map((model) => (typeof model === 'string' ? model : model?.id)).filter(Boolean)
  const embeddingCapable = new Set(
    (localRecords || [])
      .filter((record) => record?.embedding_capable)
      .map((record) => record?.model_id || record?.id || record?.filename)
      .filter(Boolean),
  )
  const byName = served.find((id) => /nomic-embed-text-v1\.5/i.test(id))
  const byCapability = served.find((id) => embeddingCapable.has(id))
  const encoderId = byName || byCapability || null
  return {
    encoderId,
    ready: Boolean(encoderId),
    /* Only the pinned Nomic artifact carries the exact-row support claim. */
    isSupportedRow: Boolean(byName),
  }
}

export function splitDocuments(text) {
  return String(text || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
}

export function rerankReadiness({ contract, encoder, query, documents }) {
  if (!contract?.present) return { ready: false, reason: 'This engine does not advertise reranking.' }
  if (!contract.supported) return { ready: false, reason: 'The reranking capability row is not marked supported on this engine.' }
  if (!encoder?.ready) {
    return {
      ready: false,
      reason: 'No encoder is loaded. An encoder loads alongside the chat model as a sidecar — it does not replace it.',
    }
  }
  if (!String(query || '').trim()) return { ready: false, reason: 'Enter a query.' }
  if (!documents || documents.length < 2) return { ready: false, reason: 'Enter at least two documents, one per line.' }
  if (documents.length > MAX_RERANK_DOCUMENTS) {
    return { ready: false, reason: `The engine accepts at most ${MAX_RERANK_DOCUMENTS} documents; this is ${documents.length}.` }
  }
  return { ready: true, reason: null }
}

export function rerankRequestBody({ encoderId, query, documents, topN }) {
  return {
    model: encoderId,
    query: String(query || '').trim(),
    documents,
    top_n: Math.min(documents.length, Math.max(1, topN || documents.length)),
    return_documents: false,
  }
}

/* Normalize a response back onto the SUBMITTED documents.

   The engine returns results by `index` into the request array, and `document` is
   omitted unless asked for — so the text must come from the caller's own list.
   Reading the response order as the document order, or assuming `document` is
   present, would silently mismatch scores to texts. */
export function normalizeRerank(payload, documents) {
  const results = Array.isArray(payload?.results) ? payload.results : []
  if (!results.length) return null
  const ranked = results
    .map((result, position) => {
      const index = Number(result?.index)
      const text = Number.isInteger(index) ? documents[index] : undefined
      if (text === undefined) return null
      const score = Number(result?.relevance_score)
      return {
        index,
        text,
        score: Number.isFinite(score) ? score : null,
        rank: position + 1,
        /* How far this document moved from the order it was submitted in. That
           movement is the whole point of a rerank, and it is the one number here
           that means something absolute. */
        movedBy: index - position,
      }
    })
    .filter(Boolean)
  if (!ranked.length) return null
  const scores = ranked.map((entry) => entry.score).filter((value) => value !== null)
  return {
    ranked,
    stats: {
      count: ranked.length,
      moved: ranked.filter((entry) => entry.movedBy !== 0).length,
      topScore: scores.length ? Math.max(...scores) : null,
      bottomScore: scores.length ? Math.min(...scores) : null,
      /* The spread is what tells a reader whether the ranking is decisive or
         nearly a tie — far more informative than any single score. */
      spread: scores.length ? Math.max(...scores) - Math.min(...scores) : null,
    },
    usage: payload?.usage || null,
    model: payload?.model || null,
  }
}

/* Similarity, never a percentage. Three decimals because the useful comparison is
   between neighbouring scores, and the gaps are often in the third place. */
export function formatSimilarity(score) {
  if (score === null || score === undefined || !Number.isFinite(score)) return '—'
  return score.toFixed(3)
}

export function formatMovement(movedBy) {
  if (movedBy === 0) return 'unchanged'
  return movedBy > 0 ? `up ${movedBy}` : `down ${Math.abs(movedBy)}`
}
