#!/usr/bin/env node
// check-ledger-drift.mjs — CAIRN Phase 3/4 (drift-check model; CAIRN Amendment 1).
//
// The CODE (the ModelCompatibilityTarget table in src/api/mod.rs) is the contract
// source of truth. ledger/camelid-ledger.json is its derived canonical form. This
// check makes drift a build failure:
//
//   A. Freshness — re-derive the ledger from the code and fail if the committed
//      ledger disagrees (provenance excluded). This is what makes "the ledger" ==
//      "the code": a contract change with no `extract-capabilities-to-ledger.mjs`
//      re-run is caught here.
//   B. Surface non-contradiction — the public supported-models tables (README and
//      COMPATIBILITY) may not claim support the contract denies. A doc row that
//      maps to a ledger row whose status is not `supported*` is a hard failure.
//      Rows that don't map are LOGGED, never silently skipped, and never fail
//      (no false positives — only a confirmed contradiction goes red). This is the
//      class of drift that produced the Mistral fixture bug fixed under Amendment 1.
//
// Zero dependencies (pure node); reuses buildLedger() from the extractor.
import { readFile, access } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { buildLedger } from './extract-capabilities-to-ledger.mjs'

const ROOT = resolve(process.argv[2] === '--root' ? process.argv[3] : '.')
const failures = []
const fail = (m) => failures.push(m)
const info = (m) => console.log('  ' + m)

// --- helpers ---------------------------------------------------------------
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '')
// clean a display label: drop parentheticals, a trailing .gguf, and markdown.
const clean = (s) => String(s).replace(/\([^)]*\)/g, ' ').replace(/\.gguf\b/gi, ' ').replace(/[*`]/g, ' ')
// filler tokens present in some ids but omitted in display labels (and vice versa)
const FILLERS = /\b(instruct|chat|it|qat|community|hybrid|arch|moe)\b/gi
// split id/filename separators to spaces first so \b filler-word boundaries fire
const fillerKey = (s) => norm(clean(s).replace(/[_-]/g, ' ').replace(FILLERS, ' '))
const stripMd = (s) => String(s).replace(/[*`]/g, '').trim()
const isSupported = (st) => typeof st === 'string' && (st === 'supported' || st.startsWith('supported_'))
async function exists(p) { try { await access(p); return true } catch { return false } }

function canon(v) {
  if (Array.isArray(v)) return v.map(canon)
  if (v && typeof v === 'object') { const o = {}; for (const k of Object.keys(v).sort()) o[k] = canon(v[k]); return o }
  return v
}
function firstDiff(a, b, path = '') {
  const ta = Array.isArray(a) ? 'array' : a === null ? 'null' : typeof a
  const tb = Array.isArray(b) ? 'array' : b === null ? 'null' : typeof b
  if (ta !== tb) return `${path} (type ${ta} vs ${tb})`
  if (ta === 'array') {
    if (a.length !== b.length) return `${path}.length (${a.length} vs ${b.length})`
    for (let i = 0; i < a.length; i++) { const d = firstDiff(a[i], b[i], `${path}[${i}]`); if (d) return d }
    return null
  }
  if (ta === 'object') {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!(k in a)) return `${path}.${k} (missing in fresh)`
      if (!(k in b)) return `${path}.${k} (missing in committed)`
      const d = firstDiff(a[k], b[k], `${path}.${k}`); if (d) return d
    }
    return null
  }
  return a === b ? null : `${path} (${JSON.stringify(a)} vs ${JSON.stringify(b)})`
}

// parse the first markdown table whose header line matches headerRe; return rows
// as arrays of trimmed cell strings.
function parseTable(md, headerRe) {
  const lines = md.split('\n')
  const h = lines.findIndex((l) => headerRe.test(l))
  if (h < 0) return null
  const rows = []
  for (let i = h + 2; i < lines.length && lines[i].trim().startsWith('|'); i++) {
    const cells = lines[i].split('|').slice(1, -1).map((c) => c.trim())
    if (cells.length) rows.push(cells)
  }
  return rows
}

// --- Check A: freshness (ledger == code contract) --------------------------
async function checkFreshness(committed) {
  const { ledger: fresh } = await buildLedger(ROOT)
  const strip = (l) => { const { provenance, ...rest } = l; return rest }
  const d = firstDiff(canon(strip(fresh)), canon(strip(committed)))
  if (d) fail(`ledger is STALE vs the code contract at ${d} — regenerate with: node scripts/extract-capabilities-to-ledger.mjs`)
  else info('freshness ok: ledger/camelid-ledger.json matches the code contract (provenance excluded)')
}

// --- Check B: supported-table non-contradiction ----------------------------
async function checkSupportedTables(committed) {
  const index = new Map()
  const put = (k, row) => { if (k && !index.has(k)) index.set(k, row) }
  for (const row of committed.model_rows) {
    for (const src of [row.identity.id, row.identity.gguf_filename]) {
      if (!src) continue
      put(norm(clean(src)), row)
      put(fillerKey(src), row)
    }
  }
  const lookup = (...labels) => {
    for (const l of labels) { const r = index.get(norm(clean(l))) || index.get(fillerKey(l)); if (r) return r }
    return null
  }

  let mapped = 0, total = 0
  const unmapped = []
  const consider = (surface, label, quant, claimIsSupported) => {
    if (!claimIsSupported) return
    total++
    const row = lookup(`${label} ${quant || ''}`, label)
    if (!row) { unmapped.push(`${surface}: "${stripMd(label)}"${quant ? ` (${stripMd(quant)})` : ''}`); return }
    mapped++
    if (!isSupported(row.contract.status)) {
      fail(`${surface} presents "${stripMd(label)}" as supported, but the code contract status for ${row.contract.id} is "${row.contract.status}"`)
    }
  }

  // README supported-models table (every data row is a support claim)
  const readme = await readFile(join(ROOT, 'README.md'), 'utf8')
  const rt = parseTable(readme, /^\|\s*Model row\s*\|\s*Quant\s*\|/)
  if (!rt) fail('README.md supported-models table (| Model row | Quant | ...) not found')
  else for (const c of rt) consider('README supported-models', c[0], c[1], true)

  // COMPATIBILITY at-a-glance (support claim = "Public claim" says supported)
  const compat = await readFile(join(ROOT, 'COMPATIBILITY.md'), 'utf8')
  const ct = parseTable(compat, /^\|\s*Exact row\s*\|\s*Public claim\s*\|/)
  if (!ct) fail('COMPATIBILITY.md at-a-glance table (| Exact row | Public claim | ...) not found')
  else for (const c of ct) {
    const claim = c[1] || ''
    const supportedClaim = /supported/i.test(claim) && !/unsupported|not a supported/i.test(claim)
    consider('COMPATIBILITY at-a-glance', c[0], null, supportedClaim)
  }

  info(`supported-table non-contradiction: ${mapped}/${total} support-claim rows mapped to a ledger row and consistent`)
  if (unmapped.length) {
    info(`unmapped support-claim rows (not checkable by exact key — logged, not failed): ${unmapped.length}`)
    unmapped.forEach((u) => info('  · ' + u))
  }
}

// --- Check C: frontend catalog consistency ---------------------------------
// The frontend download catalog keys entries by the exact contract id, so this
// is an id-exact membership check: a catalog entry with no contract row is drift.
async function checkFrontendCatalog(committed) {
  const contractIds = new Set(committed.model_rows.map((r) => r.contract.id))
  const p = join(ROOT, 'frontend', 'src', 'lib', 'supportedModels.js')
  if (!(await exists(p))) { info('frontend catalog: supportedModels.js not present, skipped'); return }
  const catalogIds = [...(await readFile(p, 'utf8')).matchAll(/catalog_id:\s*'([a-z0-9_]+)'/g)].map((m) => m[1])
  if (!catalogIds.length) { fail('no catalog_id entries parsed from frontend/src/lib/supportedModels.js'); return }
  let ok = 0
  for (const cid of catalogIds) {
    if (contractIds.has(cid)) ok++
    else fail(`frontend catalog lists catalog_id "${cid}" with no matching /api/capabilities contract row`)
  }
  info(`frontend catalog: ${ok}/${catalogIds.length} catalog_id(s) resolve to a contract row`)
}

// --- Check D: sha256 cross-surface agreement -------------------------------
// For each ledger-anchored (gguf_filename -> sha256), any surface line that names
// that file AND states a full sha must state the ledger's sha. A full sha that is
// some OTHER known file's sha (co-occurring on the same line) is ignored, so this
// only fires on a genuinely wrong/stale hash — no false positives.
const SHA256 = /\b[a-f0-9]{64}\b/g
function shaFindings(rel, text, canonicalByFile, knownShas) {
  const findings = []
  let verified = 0
  text.split('\n').forEach((line, i) => {
    const shas = line.match(SHA256)
    if (!shas) return
    for (const [fname, canon] of canonicalByFile) {
      if (!line.includes(fname)) continue
      for (const sha of shas) {
        if (sha === canon.sha) verified++
        else if (!knownShas.has(sha)) findings.push(`${rel}:${i + 1} states sha256 ${sha.slice(0, 12)}… for ${fname}, but the ledger records ${canon.sha.slice(0, 12)}… (row ${canon.id})`)
      }
    }
  })
  return { verified, findings }
}
async function checkSha256(committed) {
  const canonicalByFile = new Map()
  for (const r of committed.model_rows) {
    if (r.identity.sha256 && r.identity.gguf_filename) canonicalByFile.set(r.identity.gguf_filename, { sha: r.identity.sha256, id: r.contract.id })
  }
  if (!canonicalByFile.size) { info('sha256 agreement: no ledger-anchored full sha256 to check'); return }
  const knownShas = new Set([...canonicalByFile.values()].map((v) => v.sha))
  let verified = 0
  for (const rel of ['README.md', 'COMPATIBILITY.md', join('docs', 'reference', 'STATUS.md'), join('src', 'api', 'mod.rs')]) {
    const p = join(ROOT, rel)
    if (!(await exists(p))) continue
    const { verified: v, findings } = shaFindings(rel, await readFile(p, 'utf8'), canonicalByFile, knownShas)
    verified += v
    findings.forEach(fail)
  }
  info(`sha256 cross-surface agreement: ${verified} correct filename+sha co-occurrence(s) across surfaces, ${canonicalByFile.size} anchored file(s)`)
}

// --- Check E: durable evidence anchors have a single home ------------------
// CAIRN Phase 5 made docs/reference/STATUS.md the canonical anchor index and collapsed the
// COMPATIBILITY.md section to a pointer. Keep it that way: the anchor bundle
// list must not be re-duplicated into COMPATIBILITY.md.
async function checkAnchorsSingleHome() {
  const p = join(ROOT, 'COMPATIBILITY.md')
  if (!(await exists(p))) return
  const md = (await readFile(p, 'utf8')).split('\n')
  const s = md.findIndex((l) => l.trim() === '## Durable evidence anchors')
  if (s < 0) { info('anchors single-home: COMPATIBILITY.md has no anchors section'); return }
  let e = md.length
  for (let i = s + 1; i < md.length; i++) if (/^## /.test(md[i])) { e = i; break }
  const count = (md.slice(s, e).join('\n').match(/qa\/evidence-bundles/g) || []).length
  if (count > 0) fail(`COMPATIBILITY.md "Durable evidence anchors" re-lists ${count} evidence bundle(s); the canonical index is docs/reference/STATUS.md — keep it a pointer (CAIRN Phase 5)`)
  else info('anchors single-home: COMPATIBILITY.md anchors is a pointer; the index lives only in docs/reference/STATUS.md')
}

// ---------------------------------------------------------------------------
async function main() {
  const ledgerPath = join(ROOT, 'ledger', 'camelid-ledger.json')
  if (!(await exists(ledgerPath))) { console.error(`no ledger at ${ledgerPath}`); process.exit(1) }
  const committed = JSON.parse(await readFile(ledgerPath, 'utf8'))

  await checkFreshness(committed)
  await checkSupportedTables(committed)
  await checkFrontendCatalog(committed)
  await checkSha256(committed)
  await checkAnchorsSingleHome()

  if (failures.length) {
    console.error(`\nledger drift check FAILED (${failures.length}):`)
    for (const f of failures) console.error(`  - ${f}`)
    process.exit(1)
  }
  console.log('\nledger drift check passed: ledger == code contract, and no surface contradicts it')
}

export { firstDiff, canon, norm, fillerKey, parseTable, isSupported, shaFindings }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((e) => { console.error(e); process.exit(1) })
}
