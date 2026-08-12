#!/usr/bin/env node
// extract-capabilities-to-ledger.mjs — CAIRN Phase 2, ONE-TIME bootstrap.
//
// Parses the static `CapabilitiesResponse { ... }` literal that
// capabilities_response_with_plan() builds in src/api/mod.rs plus the typed
// `API_CONFORMANCE_CASES` registry in src/api/contract.rs, then emits
// ledger/camelid-ledger.json (camelid.ledger/v1). The capability structs derive
// plain serde `Serialize` with no renames, and every parsed string field is a
// single-line literal with no escaped quotes, so the values are byte-identical
// to what /api/capabilities serves — no build, no server, no model load (this
// box is memory-constrained; see the bench-safety rules).
//
// This is a bootstrap tool: once the Phase 3 generator exists it runs the OTHER
// way (ledger -> capabilities), and this script is retired. The output is
// validated by scripts/check-ledger-schema.mjs.
import { readFile, writeFile, mkdir, access } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { execSync } from 'node:child_process'

const ROOT = resolve('.')
const OUT = join(ROOT, 'ledger', 'camelid-ledger.json')

// --- extract the balanced CapabilitiesResponse { ... } block ---------------
function balancedBlock(src, openIdx, open = '{', close = '}') {
  let depth = 0, inStr = false
  for (let i = openIdx; i < src.length; i++) {
    const c = src[i]
    if (inStr) { if (c === '"') inStr = false; continue }
    if (c === '"') { inStr = true; continue }
    if (c === open) depth++
    else if (c === close) { depth--; if (depth === 0) return src.slice(openIdx, i + 1) }
  }
  throw new Error('unbalanced block')
}

// --- tokenizer for the Rust-literal subset ---------------------------------
function tokenize(src) {
  const toks = []
  let i = 0
  while (i < src.length) {
    const c = src[i]
    if (c === ' ' || c === '\n' || c === '\r' || c === '\t') { i++; continue }
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue }
    if (c === '"') { // no escaped quotes in this block (verified)
      let j = i + 1, s = ''
      while (j < src.length && src[j] !== '"') { s += src[j]; j++ }
      toks.push({ t: 'str', v: s }); i = j + 1; continue
    }
    if (c >= '0' && c <= '9') { let j = i; while (j < src.length && src[j] >= '0' && src[j] <= '9') j++; toks.push({ t: 'num', v: Number(src.slice(i, j)) }); i = j; continue }
    if (/[A-Za-z_]/.test(c)) { let j = i; while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++; toks.push({ t: 'ident', v: src.slice(i, j) }); i = j; continue }
    if ('{}[](),:!&'.includes(c)) { toks.push({ t: c }); i++; continue }
    i++ // skip anything else (`.`, etc.)
  }
  return toks
}

// --- recursive-descent parser ----------------------------------------------
function parse(toks) {
  let p = 0
  const peek = () => toks[p]
  function value() {
    const t = toks[p]
    if (t.t === 'str') { p++; return t.v }
    if (t.t === 'num') { p++; return t.v }
    if (t.t === '&') { p++; return value() }
    if (t.t === '[') return array()
    if (t.t === '{') return object()
    if (t.t === 'ident') {
      if (t.v === 'true' || t.v === 'false') { p++; return t.v === 'true' }
      if (t.v === 'vec') { p++; if (peek()?.t === '!') p++; return array() }
      if (toks[p + 1]?.t === '{') return object() // TypeName { ... }
      const path = [t.v]
      p++
      while (peek()?.t === ':' && toks[p + 1]?.t === ':' && toks[p + 2]?.t === 'ident') {
        p += 2
        path.push(toks[p].v)
        p++
      }
      return { __ident: path.join('::') } // bare/path identifier (field-init shorthand target)
    }
    p++; return null
  }
  function array() {
    if (peek().t !== '[') throw new Error('expected [')
    p++; const arr = []
    while (peek() && peek().t !== ']') { arr.push(value()); if (peek()?.t === ',') p++ }
    p++; return arr
  }
  function object() {
    if (peek()?.t === 'ident') p++ // skip TypeName prefix
    if (peek().t !== '{') throw new Error('expected {')
    p++; const obj = {}
    while (peek() && peek().t !== '}') {
      const key = peek().v; p++
      if (peek().t === ',' || peek().t === '}') { obj[key] = { __shorthand: key }; if (peek().t === ',') p++; continue }
      if (peek().t === ':') p++
      obj[key] = value()
      if (peek()?.t === ',') p++
    }
    p++; return obj
  }
  return value()
}

// --- helpers ---------------------------------------------------------------
const GGUF_RE = /([A-Za-z0-9][A-Za-z0-9._-]*\.gguf)/
const SHA_RE = /\b([a-f0-9]{64})\b/
const RECEIPT_RE = /qa\/evidence-bundles\/[A-Za-z0-9._/-]+?\/(?:manifest\.json|SHA256SUMS)/g

async function exists(p) { try { await access(p); return true } catch { return false } }

// --- hash-pinned artifact identities ---------------------------------------
// Some supported rows are pinned to EXACT BYTES, not just an exact filename:
// the paired Prism evidence bundles, every NON_CATALOG_SUPPORTED_ARTIFACTS
// entry (in-house requants and side-loads the operator cannot re-download to
// compare), and the curated rows carrying a recorded digest. src/api/mod.rs
// enforces those digests at classification time, so they are contract facts and
// belong in identity.sha256 — not scraped out of prose, where the ornith rows
// only ever carried an 8-hex display prefix.
//
// Read as a flat top-level const: slice from the declaration to the closing
// `];`, drop `//` comments (they contain quotes), then take the quoted strings
// in order. A tuple-arity change fails loudly rather than emitting a wrong pin.
function rustConstTuples(src, name, arity) {
  const decl = new RegExp(`const\\s+${name}\\s*:[^=]*=\\s*&\\[`).exec(src)
  if (!decl) throw new Error(`${name} not found in src/api/mod.rs`)
  const start = decl.index + decl[0].length
  const end = src.indexOf('\n];', start)
  if (end < 0) throw new Error(`${name} literal is not a flat top-level const array`)
  const body = src.slice(start, end).replace(/\/\/[^\n]*/g, '')
  if (body.includes('\\')) throw new Error(`${name} contains a string escape the extractor cannot represent`)
  const strings = [...body.matchAll(/"([^"]*)"/g)].map((m) => m[1])
  if (strings.length === 0 || strings.length % arity !== 0) {
    throw new Error(`${name} did not parse as ${arity}-tuples (${strings.length} strings) — update scripts/extract-capabilities-to-ledger.mjs to match the new shape`)
  }
  const tuples = []
  for (let i = 0; i < strings.length; i += arity) tuples.push(strings.slice(i, i + arity))
  return tuples
}

// catalog_id -> filename, from the curated_catalog() literal. Lets a curated pin
// reach its ledger row by ID, which prose-scraping cannot do: some rows never
// name their own .gguf in prose (llama32_3b_instruct_q8_0 cites only a digest).
function curatedFilenamesByCatalogId(src) {
  const start = src.indexOf('pub fn curated_catalog()')
  if (start < 0) throw new Error('curated_catalog() not found in src/api/mod.rs')
  const body = src.slice(start, src.indexOf('\n}\n', start))
  const byCatalogId = new Map()
  for (const item of body.split('CatalogItem {').slice(1)) {
    const id = /catalog_id:\s*"([^"]+)"/.exec(item)
    const filename = /filename:\s*"([^"]+)"/.exec(item)
    if (id && filename) byCatalogId.set(id[1], filename[1])
  }
  if (byCatalogId.size === 0) throw new Error('curated_catalog() parsed to zero (catalog_id, filename) pairs')
  return byCatalogId
}

// -> { byRowId: Map<row_id, {filename, sha256}>, byFilename: Map<filename, sha256> }
function extractHashPinnedArtifacts(src) {
  const byRowId = new Map()
  const byFilename = new Map()
  for (const [filename, rowId, sha256] of rustConstTuples(src, 'NON_CATALOG_SUPPORTED_ARTIFACTS', 3)) {
    byRowId.set(rowId, { filename, sha256 })
    byFilename.set(filename, sha256)
  }
  for (const table of ['PRISM_SUPPORTED_ARTIFACT_SHA256', 'CURATED_SUPPORTED_ARTIFACT_SHA256']) {
    for (const [filename, sha256] of rustConstTuples(src, table, 2)) {
      const prior = byFilename.get(filename)
      if (prior && prior !== sha256) throw new Error(`${filename} is pinned to two different digests across the artifact tables`)
      byFilename.set(filename, sha256)
    }
  }
  // Catalog-backed pins join by row id too, so a row whose prose never names its
  // artifact still carries the enforced digest.
  const catalogFilenames = curatedFilenamesByCatalogId(src)
  for (const [catalogId, filename] of catalogFilenames) {
    const sha256 = byFilename.get(filename)
    if (sha256 && !byRowId.has(catalogId)) byRowId.set(catalogId, { filename, sha256 })
  }
  for (const [filename, sha256] of byFilename) {
    if (!SHA_RE.test(sha256)) throw new Error(`hash-pinned artifact ${filename} has a malformed sha256`)
  }
  return { byRowId, byFilename }
}

// --- api_features: projected from the typed registry in contract.rs --------
// balancedBlock and tokenize assume the Rust-literal subset: no escape
// sequences and no comments inside the extracted block. Violations would
// corrupt or truncate the parse SILENTLY (the drift check re-derives through
// this same parser, so CI would stay green on a wrong ledger) — fail loudly
// here instead so the contract.rs edit that introduced them gets fixed.
function assertLiteralSubset(block) {
  let inStr = false
  for (let i = 0; i < block.length; i++) {
    const c = block[i]
    if (c === '\\') {
      throw new Error(
        'API_CONFORMANCE_CASES contains a backslash escape, which the ledger extractor cannot represent — rewrite the string without escapes or extend scripts/extract-capabilities-to-ledger.mjs'
      )
    }
    if (inStr) {
      if (c === '"') inStr = false
      continue
    }
    if (c === '"') {
      inStr = true
      continue
    }
    if (c === '/' && block[i + 1] === '/') {
      throw new Error(
        'API_CONFORMANCE_CASES contains a // comment inside the literal, which the block extractor cannot skip — move the comment outside the const or extend scripts/extract-capabilities-to-ledger.mjs'
      )
    }
  }
}

async function extractApiFeatureContract(root) {
  const src = await readFile(join(root, 'src', 'api', 'contract.rs'), 'utf8')

  // Status strings come from SupportStatus::as_str() itself rather than a
  // hand-maintained mirror, so a variant OR string rename flows through; only
  // reshaping as_str() (or a variant missing from it) fails loudly below.
  const statuses = new Map()
  for (const m of src.matchAll(/Self::([A-Za-z0-9_]+) => "([^"]+)"/g)) statuses.set(m[1], m[2])
  if (statuses.size === 0) {
    throw new Error('SupportStatus::as_str() match arms not found in src/api/contract.rs')
  }

  const marker = 'pub(super) const API_CONFORMANCE_CASES'
  const markerIndex = src.indexOf(marker)
  if (markerIndex < 0) throw new Error('API_CONFORMANCE_CASES registry not found')
  const assignmentIndex = src.indexOf('=', markerIndex)
  const referenceIndex = src.indexOf('&[', assignmentIndex)
  if (referenceIndex < 0) throw new Error('API_CONFORMANCE_CASES array not found')
  const block = balancedBlock(src, referenceIndex + 1, '[', ']')
  assertLiteralSubset(block)
  const cases = parse(tokenize(block))
  // Truncation backstop: every case the source declares must have parsed.
  const declared = (src.slice(assignmentIndex).match(/ApiConformanceCase \{/g) || []).length
  if (!Array.isArray(cases) || declared === 0 || cases.length !== declared) {
    throw new Error(
      `API_CONFORMANCE_CASES parsed to ${Array.isArray(cases) ? cases.length : 'a non-array'} case(s) but the source declares ${declared}`
    )
  }
  return cases.map((entry, index) => {
    if (!entry || typeof entry !== 'object') {
      throw new Error(`API_CONFORMANCE_CASES[${index}] is not an object`)
    }
    const statusPath = entry.status?.__ident
    const statusName = typeof statusPath === 'string' ? statusPath.split('::').at(-1) : null
    const status = statuses.get(statusName)
    if (typeof entry.id !== 'string' || typeof entry.notes !== 'string' || !status) {
      throw new Error(`API_CONFORMANCE_CASES[${index}] is missing a supported id/status/notes field: ${JSON.stringify(entry)}`)
    }
    return { id: entry.id, status, notes: entry.notes }
  })
}

function phase2CompatibilityContract({ id, family, quantization, loadPass, parityPass, templatePass, evidence }) {
  const runnableWithVariance = loadPass && !parityPass && templatePass
  const status = !loadPass
    ? 'active_validation_blocked_load'
    : !templatePass
      ? 'active_validation_blocked_template'
      : !parityPass
        ? 'runnable_exact_row_numerical_variance'
        : 'active_validation_api_webui_pass_pending_context'
  const blocker = !loadPass
    ? 'the exact artifact does not yet complete tensor binding/load; parity, API/WebUI, context, performance, and portability remain blocked'
    : !parityPass
      ? 'the exact artifact loads and generates, but deterministic greedy token IDs differ from the pinned llama.cpp oracle. Runtime use is allowed with a numerical-variance warning; Verified/Supported promotion, tools, and checked context remain held'
      : !templatePass
        ? 'raw deterministic parity passes, but the public chat-template envelope is intentionally bounded and has not earned API/WebUI or context promotion'
        : 'short deterministic parity and guarded API/WebUI smoke pass; the exact-row bounded 512-context receipt is still required before support promotion'
  return {
    id,
    family,
    quantization,
    status,
    tool_capable: false,
    support_scope: 'phase2_exact_row_validation_only',
    full_support_status: 'blocked_pending_context_performance_and_portability',
    full_support_blockers: blocker,
    metadata_parses: 'validated_exact_artifact',
    tokenizer_works: 'validated_against_pinned_llama_cpp_b9632',
    tensors_load: loadPass ? 'validated_real_weight_forward' : 'failed_exact_artifact_tensor_binding',
    generation_runs: loadPass ? 'validated_deterministic_greedy' : 'blocked_by_load_failure',
    parity_audited: parityPass ? 'pass_exact_greedy_token_ids' : loadPass ? 'failed_exact_greedy_token_ids' : 'blocked_by_load_failure',
    performance_measured: 'not_promoted',
    frontend_load_path_verified: parityPass
      ? 'validated_guarded_api_webui_smoke'
      : runnableWithVariance
        ? 'runnable_normal_inspect_and_load_path'
        : 'fail_closed_phase2_validation',
    frontend_readiness_gate: loadPass && parityPass && templatePass
      ? 'verified-runnable UI is green for this exact row after parity and guarded API/WebUI pass; Supported remains fail-closed until the bounded 512-context gate passes'
      : runnableWithVariance
        ? 'amber runnable UI is allowed for this exact hash-pinned row after the normal inspect/load path reports loaded_now=true and generation_ready=true; label it Runnable, disclose numerical variance, and do not label it Verified or Supported'
        : 'fail-closed; this exact row must clear its recorded load, parity, or template blocker before guarded API/WebUI qualification',
    tested_context: 'short_prompt_oracle_pack_only',
    chat_template_renderer: templatePass ? 'validated_exact_row_shape_pack' : 'bounded_default_envelope_only',
    chat_template_shape_pack: templatePass ? 'pass' : 'blocked_partial_envelope',
    chat_template_shape_pack_id: 'phase2-roster-template-evidence',
    bounded_context_512_pack: 'not_started',
    bounded_context_512_pack_id: 'phase2-context-512-v1',
    bounded_context_window: 512,
    bounded_context_1024_pack: 'not_promoted',
    bounded_context_1024_pack_id: 'not_selected',
    bounded_context_1024_window: 1024,
    bounded_context_2048_pack: 'not_promoted',
    bounded_context_2048_pack_id: 'not_selected',
    bounded_context_2048_window: 2048,
    bounded_context_4096_pack: 'not_promoted',
    bounded_context_4096_pack_id: 'not_selected',
    bounded_context_4096_window: 4096,
    bounded_context_8192_pack: 'not_promoted',
    bounded_context_8192_pack_id: 'not_selected',
    bounded_context_8192_window: 8192,
    latest_checked_bucket: parityPass
      ? 'phase2_guarded_api_webui_smoke'
      : runnableWithVariance
        ? 'phase2_short_greedy_numerical_variance'
        : 'phase2_short_greedy_parity',
    latest_checked_result: status,
    latest_checked_output: evidence,
    evidence,
    next_step: runnableWithVariance
      ? 'keep the exact row usable with an amber numerical-variance warning; capture explicit API/WebUI load and bounded 512-context receipts, and require a documented parity/tolerance decision before Verified or Supported promotion'
      : 'close the recorded blocker, then capture exact-row API/WebUI and bounded 512-context evidence before support promotion',
  }
}

function extractPhase2CompatibilityRows(src) {
  const marker = 'fn phase2_model_compatibility_targets()'
  const start = src.indexOf(marker)
  if (start < 0) return []
  const block = balancedBlock(src, src.indexOf('{', start))
  const rows = []
  const callRe = /phase2_model_compatibility_target\s*\(/g
  for (const match of block.matchAll(callRe)) {
    const open = match.index + match[0].lastIndexOf('(')
    const call = balancedBlock(block, open, '(', ')')
    const values = [...call.matchAll(/"([^"]*)"|\b(true|false)\b/g)].map((token) => (
      token[1] === undefined ? token[2] === 'true' : token[1]
    ))
    if (values.length !== 7) {
      throw new Error(`phase2_model_compatibility_target call parsed ${values.length} arguments instead of 7`)
    }
    const [id, family, quantization, loadPass, parityPass, templatePass, evidence] = values
    rows.push(phase2CompatibilityContract({ id, family, quantization, loadPass, parityPass, templatePass, evidence }))
  }
  if (rows.length === 0) throw new Error('phase2_model_compatibility_targets() parsed to zero rows')
  return rows
}

function extractModelCompatibilityRows(src, capabilitiesBlock, parsedCapabilities) {
  if (Array.isArray(parsedCapabilities.model_compatibility)) return parsedCapabilities.model_compatibility

  const fieldStart = capabilitiesBlock.indexOf('model_compatibility:')
  const runtimeStart = capabilitiesBlock.indexOf('runtime_projects:', fieldStart)
  const composedField = capabilitiesBlock.slice(fieldStart, runtimeStart)
  const baseMarker = 'let mut rows = vec!['
  const baseStart = composedField.indexOf(baseMarker)
  if (baseStart < 0) throw new Error('composed model_compatibility is missing its base vec literal')
  const baseOpen = composedField.indexOf('[', baseStart)
  const baseRows = parse(tokenize(balancedBlock(composedField, baseOpen, '[', ']')))
  if (!Array.isArray(baseRows)) throw new Error('composed model_compatibility base did not parse to an array')

  const phase2 = extractPhase2CompatibilityRows(src)
  const phase2Ids = new Set(phase2.map((row) => row.id))
  return baseRows.filter((row) => !phase2Ids.has(row.id)).concat(phase2)
}

export async function buildLedger(root = ROOT) {
  const src = await readFile(join(root, 'src', 'api', 'mod.rs'), 'utf8')
  const marker = 'CapabilitiesResponse {'
  const fnPos = src.indexOf('fn capabilities_response_with_plan')
  // The fn signature ends `-> CapabilitiesResponse {` (the fn body brace); the
  // struct literal `CapabilitiesResponse {` is the NEXT occurrence (the return
  // expression) — that is the one we parse.
  const sigIdx = src.indexOf(marker, fnPos)
  const idx = src.indexOf(marker, sigIdx + marker.length)
  if (idx < 0) throw new Error('CapabilitiesResponse struct literal not found')
  const block = balancedBlock(src, src.indexOf('{', idx))
  const cr = parse(tokenize(block))

  // execution_plan is the function param (None) -> null in the static contract
  cr.execution_plan = null

  const rows = extractModelCompatibilityRows(src, block, cr)
  const apiFeatures = await extractApiFeatureContract(root)

  const pinned = extractHashPinnedArtifacts(src)

  const receiptWarnings = []
  const model_rows = []
  for (const contract of rows) {
    const prose = [contract.evidence, contract.frontend_readiness_gate, contract.full_support_blockers, contract.tested_context].join(' ')
    const identity = { id: contract.id, family: contract.family, quantization: contract.quantization }
    const gguf = GGUF_RE.exec(prose)
    if (gguf) identity.gguf_filename = gguf[1]
    const sha = SHA_RE.exec(prose)
    if (sha) identity.sha256 = sha[1]
    // A code-enforced pin outranks the prose scrape: it is the digest the server
    // actually checks before granting this row, and the prose may carry only a
    // display prefix (or a receipt hash that happens to appear first).
    const pin = pinned.byRowId.get(contract.id)
      || (identity.gguf_filename && pinned.byFilename.has(identity.gguf_filename)
        ? { filename: identity.gguf_filename, sha256: pinned.byFilename.get(identity.gguf_filename) }
        : null)
    if (pin) {
      identity.gguf_filename = pin.filename
      identity.sha256 = pin.sha256
    }

    const receipts = []
    const seen = new Set()
    for (const m of prose.matchAll(RECEIPT_RE)) {
      const path = m[0]
      if (seen.has(path)) continue
      seen.add(path)
      if (await exists(join(root, path))) receipts.push({ path })
      else receiptWarnings.push(`${contract.id}: receipt ${path} does not resolve on disk (omitted)`)
    }
    const row = { identity, contract }
    if (receipts.length) row.receipts = receipts
    model_rows.push(row)
  }

  const capabilities = {
    engine: cr.engine,
    gguf_metadata: cr.gguf_metadata,
    tensor_loading: cr.tensor_loading,
    tokenization: cr.tokenization,
    inference: cr.inference,
    streaming: cr.streaming,
    model_downloads: cr.model_downloads,
    hf_catalog_install: cr.hf_catalog_install,
    execution_plan: null,
    support_contract: cr.support_contract,
    supported_quantization: cr.supported_quantization,
    planned_quantization: cr.planned_quantization,
    supported_model_families: cr.supported_model_families,
    planned_model_families: cr.planned_model_families,
    api_features: apiFeatures,
    notes: cr.notes,
  }

  let head = 'unknown'
  try { head = execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim() } catch {}

  const ledger = {
    ledger_version: 'camelid.ledger/v1',
    provenance: {
      source_head: head,
      note: 'Derived from the static CapabilitiesResponse literal in src/api/mod.rs and API_CONFORMANCE_CASES in src/api/contract.rs by scripts/extract-capabilities-to-ledger.mjs. Contract fields are byte-faithful (plain serde Serialize, no renames). Per CAIRN Amendment 1, the CODE is the contract source of truth; this ledger is its derived canonical form, and scripts/check-ledger-drift.mjs re-derives from code and fails CI if code and ledger disagree (provenance excluded).',
    },
    capabilities,
    model_rows,
  }

  return { ledger, receiptWarnings, fieldCount: Object.keys(rows[0]).length }
}

async function main() {
  const { ledger, receiptWarnings, fieldCount } = await buildLedger(ROOT)
  await mkdir(join(ROOT, 'ledger'), { recursive: true })
  await writeFile(OUT, JSON.stringify(ledger, null, 2) + '\n')
  const c = ledger.capabilities
  console.log(`extracted ${ledger.model_rows.length} model row(s); each contract has ${fieldCount} fields`)
  console.log(`envelope: ${c.supported_quantization.length} supported_quant, ${c.planned_quantization.length} planned_quant, ${c.supported_model_families.length} supported_fam, ${c.planned_model_families.length} planned_fam, ${c.api_features.length} api_features, ${c.notes.length} notes`)
  console.log(`receipts attached: ${ledger.model_rows.reduce((n, r) => n + (r.receipts?.length || 0), 0)} (resolved on disk)`)
  if (receiptWarnings.length) { console.log('receipt notes:'); receiptWarnings.forEach((w) => console.log('  - ' + w)) }
  console.log(`wrote ${OUT}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((e) => { console.error(e); process.exit(1) })
}
