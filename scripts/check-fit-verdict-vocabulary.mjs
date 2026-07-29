#!/usr/bin/env node
// check-fit-verdict-vocabulary.mjs — keeps the WebUI's fit vocabulary tied to the code.
//
// `FitVerdict` in src/fit.rs is serialized straight onto /api/models/catalog and
// /api/models/catalog/fit, and the Models tab branches on those exact strings. The
// JS side re-declares them as literals, so a renamed or added variant silently
// stops matching: no compiler sees it, and a frontend smoke that asserts the
// frontend's own literals against each other passes either way. That is the
// lane-gate class of bug — the UI re-deciding something the backend already said,
// against a vocabulary that has drifted.
//
// Rust is the source of truth. Three declarations there define the whole partition:
//   as_str()          — the complete vocabulary
//   is_positive_fit() — the verdicts that mean "can run somehow"
//   refuses_load()    — the verdicts a pre-load guard refuses on
// This asserts the JS sets agree EXACTLY (not merely as a superset — an extra JS
// entry is drift too, e.g. a verdict deleted in Rust but still special-cased in the
// UI), and that every verdict string is actually handled somewhere in the lib.
//
// Usage: node scripts/check-fit-verdict-vocabulary.mjs
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const ROOT = resolve('.')
const FIT_RS = resolve(ROOT, 'src/fit.rs')
const BROWSE_JS = resolve(ROOT, 'frontend/src/lib/catalogBrowse.js')

let failures = 0
function fail(msg) {
  console.error(`  FAIL ${msg}`)
  failures++
}

/** Body of `fn <name>` up to the closing brace at its own indentation.
 *
 * The indent must come from the START of the declaring line, not from just before
 * the `fn` token: `    pub fn as_str(` would otherwise yield a one-space indent, no
 * `\n }` would ever match, and the "body" would run to end-of-file and pick up
 * later functions' match arms (`cli_label`'s "fits" was exactly that). */
export function fnBody(src, name) {
  const decl = new RegExp(`^([ \\t]*)(?:pub(?:\\([^)]*\\))?[ \\t]+)?fn ${name}\\(`, 'm')
  const m = src.match(decl)
  if (!m) return null
  const rest = src.slice(m.index)
  const end = rest.indexOf(`\n${m[1]}}`)
  return end === -1 ? rest : rest.slice(0, end)
}

/** variant -> wire string, from as_str()'s match arms. */
export function variantStrings(src) {
  const body = fnBody(src, 'as_str')
  if (!body) return new Map()
  const map = new Map()
  for (const m of body.matchAll(/FitVerdict::(\w+)\s*=>\s*"([a-z0-9_]+)"/g)) {
    map.set(m[1], m[2])
  }
  return map
}

/** Variants named inside `matches!(self, ...)` in is_positive_fit(). */
export function positiveVariants(src) {
  const body = fnBody(src, 'is_positive_fit')
  if (!body) return null
  return new Set([...body.matchAll(/FitVerdict::(\w+)/g)].map((m) => m[1]))
}

/** Variants on the `=> true` arm of refuses_load(). */
export function refusingVariants(src) {
  const body = fnBody(src, 'refuses_load')
  if (!body) return null
  // Take the arm text preceding `=> true`; `|`-joined patterns may span lines.
  const arm = body.match(/((?:\s*FitVerdict::\w+\s*\|?)+)=>\s*true/)
  if (!arm) return null
  return new Set([...arm[1].matchAll(/FitVerdict::(\w+)/g)].map((m) => m[1]))
}

/** A `new Set([...])` literal's string members, by const name. */
export function jsSet(src, constName) {
  const m = src.match(new RegExp(`const\\s+${constName}\\s*=\\s*new Set\\(\\[([^\\]]*)\\]`))
  if (!m) return null
  return new Set([...m[1].matchAll(/['"]([a-z0-9_]+)['"]/g)].map((x) => x[1]))
}

function compare(label, expected, actual) {
  if (!actual) {
    fail(`${label}: could not parse the JS set (has it been renamed or reshaped?)`)
    return
  }
  const missing = [...expected].filter((v) => !actual.has(v)).sort()
  const extra = [...actual].filter((v) => !expected.has(v)).sort()
  if (missing.length) fail(`${label} is missing verdict(s) the code emits: ${missing.join(', ')}`)
  if (extra.length) fail(`${label} has verdict(s) the code no longer emits: ${extra.join(', ')}`)
  if (!missing.length && !extra.length) {
    console.log(`  ${label} matches src/fit.rs exactly (${[...expected].sort().join(', ')})`)
  }
}

async function main() {
  const rust = await readFile(FIT_RS, 'utf8')
  const js = await readFile(BROWSE_JS, 'utf8')

  const strings = variantStrings(rust)
  if (!strings.size) {
    fail('no FitVerdict::X => "y" arms found in src/fit.rs as_str() (check cannot run)')
    process.exit(1)
  }
  console.log(`  vocabulary: ${strings.size} verdict(s) declared by FitVerdict::as_str()`)

  const toWire = (variants, what) => {
    if (!variants) {
      fail(`could not parse ${what} in src/fit.rs (check cannot run)`)
      return null
    }
    const wire = new Set()
    for (const v of variants) {
      const s = strings.get(v)
      if (!s) fail(`${what} names FitVerdict::${v}, which has no as_str() arm`)
      else wire.add(s)
    }
    return wire
  }

  const positive = toWire(positiveVariants(rust), 'is_positive_fit()')
  const refusing = toWire(refusingVariants(rust), 'refuses_load()')

  if (positive) compare('POSITIVE_FITS', positive, jsSet(js, 'POSITIVE_FITS'))
  if (refusing) compare('REFUSING_FITS', refusing, jsSet(js, 'REFUSING_FITS'))

  // Every verdict must be mentioned somewhere in the lib, so a new variant cannot
  // fall through the label/detail switches into a blank UI string.
  const unhandled = [...strings.values()].filter((s) => !js.includes(`'${s}'`)).sort()
  if (unhandled.length) {
    fail(`frontend/src/lib/catalogBrowse.js never mentions: ${unhandled.join(', ')}`)
  } else {
    console.log(`  all ${strings.size} verdict(s) are referenced in catalogBrowse.js`)
  }

  if (failures) {
    console.error(`\nfit-verdict vocabulary check FAILED (${failures} problem(s))`)
    process.exit(1)
  }
  console.log('\nfit-verdict vocabulary check passed: WebUI vocabulary == src/fit.rs')
}

// Importable for scripts/test-check-fit-verdict-vocabulary.mjs; runs when invoked.
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err)
    process.exit(1)
  })
}
