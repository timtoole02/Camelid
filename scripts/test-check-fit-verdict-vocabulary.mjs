#!/usr/bin/env node
// Self-test for check-fit-verdict-vocabulary.mjs (run in CI by the
// validation-scripts job's test-*.mjs glob). A drift guard that cannot fail is
// worse than no guard — it reads as coverage. So this asserts the parsers both
// agree with the real src/fit.rs AND reject each way the two sides can diverge.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  fnBody,
  variantStrings,
  positiveVariants,
  refusingVariants,
  jsSet,
} from './check-fit-verdict-vocabulary.mjs'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const realRust = await readFile(join(repoRoot, 'src', 'fit.rs'), 'utf8')
const realJs = await readFile(join(repoRoot, 'frontend', 'src', 'lib', 'catalogBrowse.js'), 'utf8')

// --- baseline: the real sources parse, and agree -----------------------------
const strings = variantStrings(realRust)
assert.ok(strings.size >= 6, `expected >=6 verdicts, got ${strings.size}`)
assert.equal(strings.get('FitsResident'), 'fits_resident')
assert.equal(strings.get('InsufficientFreeMemory'), 'insufficient_free_memory')
assert.equal(strings.get('Unknown'), 'unknown')

const wire = (vs) => new Set([...vs].map((v) => strings.get(v)))
assert.deepEqual(
  wire(positiveVariants(realRust)),
  jsSet(realJs, 'POSITIVE_FITS'),
  'real POSITIVE_FITS must match is_positive_fit()',
)
assert.deepEqual(
  wire(refusingVariants(realRust)),
  jsSet(realJs, 'REFUSING_FITS'),
  'real REFUSING_FITS must match refuses_load()',
)

// --- regression guard for the indent bug ------------------------------------
// fnBody once derived indentation from just before the `fn` token, so `pub fn`
// yielded a 1-space indent, no closing brace matched, and the body ran to EOF —
// swallowing later functions. That made cli_label's `"fits"` look like as_str's
// wire string for FitsResident. Assert the body stops at its own function.
const asStrBody = fnBody(realRust, 'as_str')
assert.ok(asStrBody, 'as_str() body must be found')
assert.ok(!asStrBody.includes('cli_label'), 'as_str() body must not run past its own closing brace')
assert.ok(!/=>\s*"fits"/.test(asStrBody), 'as_str() body must not include cli_label arms')

// --- synthetic drift: each divergence must be detectable --------------------
const rustFor = (arms, positive, refusing) => `
impl FitVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
${arms.map(([v, s]) => `            FitVerdict::${v} => "${s}",`).join('\n')}
        }
    }

    pub fn is_positive_fit(self) -> bool {
        matches!(
            self,
            ${positive.map((v) => `FitVerdict::${v}`).join(' | ')}
        )
    }

    pub fn refuses_load(self) -> bool {
        match self {
            ${refusing.map((v) => `FitVerdict::${v}`).join(' | ')} => true,
            _ => false,
        }
    }

    pub fn cli_label(self) -> &'static str {
        match self {
            FitVerdict::FitsResident => "fits",
        }
    }
}
`

// A RENAMED wire string is visible: as_str changes, the JS literal does not.
const renamed = rustFor(
  [['FitsResident', 'fits_vram_resident'], ['WontFit', 'wont_fit']],
  ['FitsResident'],
  ['WontFit'],
)
assert.equal(variantStrings(renamed).get('FitsResident'), 'fits_vram_resident')
assert.ok(
  !jsSet(realJs, 'POSITIVE_FITS').has('fits_vram_resident'),
  'a renamed verdict must not already be in the JS set (else the guard proves nothing)',
)

// An ADDED verdict that Rust calls positive but JS never lists.
const added = rustFor(
  [['FitsResident', 'fits_resident'], ['FitsPartly', 'fits_partly'], ['WontFit', 'wont_fit']],
  ['FitsResident', 'FitsPartly'],
  ['WontFit'],
)
const addedWire = new Set([...positiveVariants(added)].map((v) => variantStrings(added).get(v)))
assert.ok(addedWire.has('fits_partly'))
assert.ok(!jsSet(realJs, 'POSITIVE_FITS').has('fits_partly'), 'added verdict must be missing from JS')

// A verdict REMOVED from Rust but still special-cased in JS (extra-side drift).
const shrunk = rustFor([['FitsResident', 'fits_resident'], ['WontFit', 'wont_fit']], ['FitsResident'], ['WontFit'])
const shrunkRefusing = new Set([...refusingVariants(shrunk)].map((v) => variantStrings(shrunk).get(v)))
assert.deepEqual(shrunkRefusing, new Set(['wont_fit']))
assert.ok(
  jsSet(realJs, 'REFUSING_FITS').has('insufficient_free_memory'),
  'JS still lists insufficient_free_memory, so a Rust-side removal would be caught as extra',
)

// refuses_load()'s `|`-joined arm may span lines; the parser must gather all of it.
const multiline = `
impl FitVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            FitVerdict::WontFit => "wont_fit",
            FitVerdict::InsufficientFreeMemory => "insufficient_free_memory",
            FitVerdict::Unknown => "unknown",
        }
    }

    pub fn refuses_load(self) -> bool {
        match self {
            FitVerdict::WontFit
            | FitVerdict::InsufficientFreeMemory => true,
            FitVerdict::Unknown => false,
        }
    }
}
`
assert.deepEqual(
  refusingVariants(multiline),
  new Set(['WontFit', 'InsufficientFreeMemory']),
  'multi-line `|` arms must all be collected',
)

// A reshaped/renamed JS set is reported rather than silently passing.
assert.equal(jsSet('const SOMETHING_ELSE = new Set([])', 'POSITIVE_FITS'), null)

console.log('check-fit-verdict-vocabulary self-test: ok')
