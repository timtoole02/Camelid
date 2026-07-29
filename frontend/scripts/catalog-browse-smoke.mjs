/* Unit smoke for the catalog browse logic (frontend/src/lib/catalogBrowse.js).

   Covers the three behaviours the module exists for: collapsing a repo's quant
   permutations into one model, ordering/annotating those quants, and never turning
   an unknown fit into a claim. No DOM, no network. */

import assert from 'node:assert/strict'
import {
  compareByQuality,
  defaultFileIndex,
  fitDetail,
  fitIsRecheckable,
  fitIsSettled,
  fitLabel,
  groupHfFilesByRepo,
  isPositiveFit,
  isRefusingFit,
  partitionByArchSupport,
  partitionCuratedByFit,
  quantAdvice,
  repoOwner,
  repoTitle,
} from '../src/lib/catalogBrowse.js'

const GB = 1024 * 1024 * 1024

function hfFile({ repo = 'unsloth/Phi-4-mini-instruct-GGUF', quant, size, fit = 'unknown', arch = 'phi3', archSupport = 'implemented' }) {
  return {
    catalog_id: `hf::${repo}::${quant}`,
    group: 'experimental',
    repo_id: repo,
    filename: `Phi-4-mini-instruct-${quant}.gguf`,
    quant,
    size_bytes: size,
    fit,
    fit_confidence: fit === 'unknown' ? 'unknown' : 'exact',
    architecture: arch,
    arch_support: archSupport,
    downloads: 634146,
    likes: 701,
  }
}

/* --- fit vocabulary ------------------------------------------------------- */
{
  assert.equal(isPositiveFit('cpu_only_ok'), true)
  assert.equal(isPositiveFit('fits_with_offload'), true)
  assert.equal(isPositiveFit('unknown'), false)

  // The whole point of the new verdict: refused, but NOT "too big".
  assert.equal(isRefusingFit('insufficient_free_memory'), true)
  assert.equal(isRefusingFit('wont_fit'), true)
  assert.equal(isRefusingFit('unknown'), false, 'unknown is the absence of a claim, not a negative')

  assert.equal(fitLabel('insufficient_free_memory'), 'Not enough free memory right now')
  assert.equal(fitLabel('wont_fit'), 'Too big for this machine')
  assert.equal(fitLabel('unknown'), null, 'an unknown verdict must render nothing, not a guess')
  assert.equal(fitLabel(undefined), null)

  assert.match(fitDetail('insufficient_free_memory'), /Close some applications/)
  assert.match(fitDetail('wont_fit'), /smaller model/)
  assert.equal(fitDetail('cpu_only_ok'), null)
  // A busy machine must never be described as an undersized one.
  assert.doesNotMatch(fitDetail('insufficient_free_memory'), /too big|smaller model/i)

  // The remedy must name the action that actually works. The catalog listing is
  // built from a startup memory snapshot, so "reload the page" would be false
  // advice: only the live re-check can observe freed memory.
  assert.match(fitDetail('insufficient_free_memory'), /Re-check/)
  assert.doesNotMatch(fitDetail('insufficient_free_memory'), /reload|refresh/i)

  // Only a transient shortage is worth re-checking. A model bigger than the whole
  // machine stays too big however much is freed.
  assert.equal(fitIsRecheckable('insufficient_free_memory'), true)
  assert.equal(fitIsRecheckable('wont_fit'), false)
  assert.equal(fitIsRecheckable('cpu_only_ok'), false)
  assert.equal(fitIsRecheckable('unknown'), false)
  assert.equal(fitIsRecheckable(undefined), false)
}

/* --- unchecked vs. settled ------------------------------------------------ */
{
  // All three arrive as `fit: 'unknown'`. Rendering them the same put the check
  // button in a loop: press it, get `unknown` back, see the same button again.
  assert.equal(
    fitIsSettled({ fit: 'unknown', fit_confidence: 'unknown' }),
    false,
    'never measured -> offer a check',
  )
  assert.equal(
    fitIsSettled({ fit: 'unknown', fit_confidence: 'exact' }),
    true,
    'measured, advisor still abstains -> retire the button',
  )
  assert.equal(fitIsSettled({ fit: 'unknown', fit_confidence: 'approx' }), true)
  assert.equal(fitIsSettled({ fit: 'cpu_only_ok', fit_confidence: 'exact' }), true)

  // The server's explicit verdict wins over the confidence heuristic in BOTH
  // directions. This is the case the heuristic alone got wrong: a settled negative
  // comes back with `unknown` confidence, and looked identical to "not checked".
  assert.equal(
    fitIsSettled({ fit: 'unknown', fit_confidence: 'unknown', fit_checked: true }),
    true,
    'checked and definitively unresolvable -> settled',
  )
  assert.equal(
    fitIsSettled({ fit: 'unknown', fit_confidence: 'exact', fit_checked: false }),
    false,
    'a failed attempt stays retryable',
  )

  assert.equal(fitIsSettled({}), false)
  assert.equal(fitIsSettled(undefined), false)
}

/* --- quant advice --------------------------------------------------------- */
{
  assert.equal(quantAdvice('Q4_K_M').tier, 'balanced')
  assert.equal(quantAdvice('q4_k_m').tier, 'balanced', 'tokens are case-insensitive')
  assert.equal(quantAdvice('Q8_0').tier, 'high')
  assert.equal(quantAdvice('Q2_K').tier, 'severe')
  assert.equal(quantAdvice('IQ1_S').tier, 'extreme')
  assert.equal(quantAdvice('F16').tier, 'full')

  // Ordering is total across families and within a family.
  assert.ok(quantAdvice('Q8_0').rank > quantAdvice('Q6_K').rank)
  assert.ok(quantAdvice('Q6_K').rank > quantAdvice('Q4_K_M').rank)
  assert.ok(quantAdvice('Q4_K_M').rank > quantAdvice('Q4_K_S').rank)
  assert.ok(quantAdvice('Q4_K_S').rank > quantAdvice('Q3_K_L').rank)
  assert.ok(quantAdvice('Q3_K_L').rank > quantAdvice('Q2_K').rank)
  assert.ok(quantAdvice('Q2_K').rank > quantAdvice('IQ1_S').rank)

  // An unrecognized token gets no advice rather than invented advice.
  const mystery = quantAdvice('Q9_WAT')
  assert.equal(mystery.tier, 'unknown')
  assert.equal(mystery.note, null)
  assert.equal(mystery.rank, 0)
  assert.equal(quantAdvice('').tier, 'unknown')
}

/* --- repo naming ---------------------------------------------------------- */
{
  assert.equal(repoTitle('unsloth/Phi-4-mini-instruct-GGUF'), 'Phi-4-mini-instruct')
  assert.equal(repoTitle('Qwen/Qwen3-4B-GGUF'), 'Qwen3-4B')
  assert.equal(repoTitle('someone/plain-model'), 'plain-model')
  assert.equal(repoTitle('bare'), 'bare')
  assert.equal(repoTitle(''), '')
  // A repo whose name IS "GGUF" must not collapse to an empty title.
  assert.equal(repoTitle('someone/GGUF'), 'GGUF')

  assert.equal(repoOwner('unsloth/Phi-4-mini-instruct-GGUF'), 'unsloth')
  assert.equal(repoOwner('bare'), '')
}

/* --- grouping ------------------------------------------------------------- */
{
  // The measured shape of a real search: two repos, many quants each, interleaved.
  const items = [
    hfFile({ quant: 'Q2_K', size: 1.6 * GB }),
    hfFile({ quant: 'Q8_0', size: 4.1 * GB }),
    hfFile({ repo: 'unsloth/Qwen3-4B-GGUF', quant: 'Q4_K_M', size: 2.5 * GB, arch: 'qwen3' }),
    hfFile({ quant: 'Q4_K_M', size: 2.5 * GB }),
    hfFile({ repo: 'unsloth/Qwen3-4B-GGUF', quant: 'Q2_K', size: 1.7 * GB, arch: 'qwen3' }),
  ]
  const groups = groupHfFilesByRepo(items)

  assert.equal(groups.length, 2, '5 files from 2 repos collapse to 2 cards')
  assert.deepEqual(
    groups.map((g) => g.repoId),
    ['unsloth/Phi-4-mini-instruct-GGUF', 'unsloth/Qwen3-4B-GGUF'],
    'repo order follows first appearance, which is the Hub relevance order',
  )
  assert.equal(groups[0].title, 'Phi-4-mini-instruct')
  assert.equal(groups[0].owner, 'unsloth')
  assert.equal(groups[0].architecture, 'phi3')
  assert.equal(groups[0].likes, 701)
  assert.deepEqual(
    groups[0].files.map((f) => f.quant),
    ['Q8_0', 'Q4_K_M', 'Q2_K'],
    'files within a card are ordered best quality first',
  )
  assert.equal(groupHfFilesByRepo([]).length, 0)
}

/* --- quality ordering ties ------------------------------------------------ */
{
  const a = hfFile({ quant: 'Q4_K_M', size: 3 * GB })
  const b = { ...hfFile({ quant: 'Q4_K_M', size: 2 * GB }), filename: 'other-Q4_K_M.gguf' }
  assert.ok(compareByQuality(a, b) < 0, 'same quant: the larger file sorts first')
  // Total and stable: comparing either direction agrees.
  assert.ok(compareByQuality(b, a) > 0)
}

/* --- default quantization ------------------------------------------------- */
{
  // 1. A proven fit wins, and it is the BEST proven one, not the first listed.
  const withFits = [
    hfFile({ quant: 'Q8_0', size: 4.1 * GB, fit: 'wont_fit' }),
    hfFile({ quant: 'Q4_K_M', size: 2.5 * GB, fit: 'cpu_only_ok' }),
    hfFile({ quant: 'Q2_K', size: 1.6 * GB, fit: 'cpu_only_ok' }),
  ]
  assert.equal(defaultFileIndex(withFits), 1, 'best quality that actually fits')

  // 2. Nothing proven (the common case: HF rows have no dims yet) -> balanced tier,
  //    never the biggest file we merely failed to rule out.
  const allUnknown = [
    hfFile({ quant: 'F16', size: 8 * GB }),
    hfFile({ quant: 'Q8_0', size: 4.1 * GB }),
    hfFile({ quant: 'Q4_K_M', size: 2.5 * GB }),
    hfFile({ quant: 'Q2_K', size: 1.6 * GB }),
  ]
  assert.equal(allUnknown[defaultFileIndex(allUnknown)].quant, 'Q4_K_M')

  // 3. No balanced quant on offer -> the smallest option we cannot rule out.
  const noBalanced = [hfFile({ quant: 'Q8_0', size: 4.1 * GB }), hfFile({ quant: 'Q6_K', size: 3.2 * GB })]
  assert.equal(noBalanced[defaultFileIndex(noBalanced)].quant, 'Q6_K')

  // 4. Everything refused -> still offer the smallest rather than nothing.
  const allTooBig = [
    hfFile({ quant: 'Q8_0', size: 40 * GB, fit: 'wont_fit' }),
    hfFile({ quant: 'Q4_K_M', size: 24 * GB, fit: 'insufficient_free_memory' }),
  ]
  assert.equal(allTooBig[defaultFileIndex(allTooBig)].quant, 'Q4_K_M')

  assert.equal(defaultFileIndex([]), -1)
}

/* --- architecture partition ----------------------------------------------- */
{
  const groups = groupHfFilesByRepo([
    hfFile({ repo: 'a/llama-GGUF', quant: 'Q4_K_M', size: GB, archSupport: 'implemented' }),
    hfFile({ repo: 'b/mamba-GGUF', quant: 'Q4_K_M', size: GB, archSupport: 'not_implemented' }),
    hfFile({ repo: 'c/mystery-GGUF', quant: 'Q4_K_M', size: GB, archSupport: 'unknown' }),
  ])
  const { loadable, unimplemented } = partitionByArchSupport(groups)
  assert.deepEqual(loadable.map((g) => g.repoId), ['a/llama-GGUF', 'c/mystery-GGUF'])
  assert.deepEqual(unimplemented.map((g) => g.repoId), ['b/mamba-GGUF'])
  // "We could not tell" must never be filtered out as if it were a known negative.
  assert.ok(loadable.some((g) => g.archSupport === 'unknown'))
}

/* --- curated landing partition -------------------------------------------- */
{
  const curated = [
    { catalog_id: 'a', fit: 'cpu_only_ok' },
    { catalog_id: 'b', fit: 'wont_fit' },
    { catalog_id: 'c', fit: 'insufficient_free_memory' },
    { catalog_id: 'd', fit: 'unknown' },
  ]
  const { runnable, blocked } = partitionCuratedByFit(curated)
  assert.deepEqual(runnable.map((i) => i.catalog_id), ['a', 'd'], 'unprobed hosts keep their catalog')
  assert.deepEqual(blocked.map((i) => i.catalog_id), ['b', 'c'])
}

console.log('catalog browse smoke: ok')
