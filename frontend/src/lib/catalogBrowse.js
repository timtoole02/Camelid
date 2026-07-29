/* Pure presentation logic for the "Get models" catalog browse.

   It lives outside the component so the three rules that actually decide what the
   user sees are unit-testable and have exactly one definition each:

   1. A Hugging Face *repo* is one model; its many `.gguf` files are quantizations
      OF that model. The browse API returns files, because a file is what you
      download — but rendering one card per file is what made the list feel
      endless (15 repos routinely expand to 250+ rows, e.g. `Q2_K`, `Q2_K_L`,
      `IQ3_M`, `IQ4_XS`, ... of the same weights). Grouping happens here.
   2. Quantization is a quality/size tradeoff the user is currently given no help
      with. The ordering and the plain-language note come from one table.
   3. Fit is never guessed. Every helper here distinguishes "does not fit",
      "does not fit right now", and "we do not know" — collapsing them is what made
      the advisory read as "nothing works on this machine". */

/* --- fit ------------------------------------------------------------------ */

/* Verdicts that mean the model can run somehow on this host. */
const POSITIVE_FITS = new Set(['fits_resident', 'fits_with_offload', 'cpu_only_ok'])

/* Verdicts that mean the model cannot be loaded as things stand. `unknown` is
   deliberately absent: it is the absence of a claim, not a negative one. */
const REFUSING_FITS = new Set(['wont_fit', 'insufficient_free_memory'])

export function isPositiveFit(fit) {
  return POSITIVE_FITS.has(fit)
}

export function isRefusingFit(fit) {
  return REFUSING_FITS.has(fit)
}

/* Capacity advisory for THIS host (fit axis, NOT a support claim). `unknown` and
   anything unrecognized render nothing rather than a guess. */
export function fitLabel(fit) {
  switch (fit) {
    case 'fits_resident':
      return 'Fits your machine'
    case 'fits_with_offload':
      return 'Fits (GPU + RAM offload)'
    case 'cpu_only_ok':
      return 'Fits (CPU)'
    case 'insufficient_free_memory':
      return 'Not enough free memory right now'
    case 'wont_fit':
      return 'Too big for this machine'
    default:
      return null
  }
}

/* Whether this row's capacity question is settled — i.e. asking again cannot teach
   us anything new.

   `fit: 'unknown'` has THREE causes and they need different UI. Either nothing has
   been resolved yet (offer a check); or the footprint WAS resolved and the advisor
   still abstained (`fit_confidence: 'exact'`/`'approx'` — happens when host memory
   cannot be probed, or when the GPU has room while free system memory is too low to
   stage the weights); or a check ran and did not settle (offline, rate-limited —
   `fit_checked: false`, worth retrying).

   Treating the middle case as "not checked yet" left the check button in a loop:
   press it, get `unknown` back, see the same button. The backend reports `checked`
   precisely so this is decidable client-side. */
export function fitIsSettled(item) {
  if (item?.fit_checked === true) return true
  if (item?.fit_checked === false) return false
  return item?.fit_confidence === 'exact' || item?.fit_confidence === 'approx'
}

/* Why the row is refused or unresolved, and what the user can actually do about
   it. A machine that is merely busy must never be described as too small.

   The wording is tied to a real mechanism: a refusal caused by memory pressure is
   re-checkable, because the check endpoint probes memory live. Telling the user to
   free memory would be empty advice otherwise — the catalog listing reads a
   startup snapshot, so a page reload alone changes nothing. */
export function fitDetail(fit) {
  switch (fit) {
    case 'insufficient_free_memory':
      return 'This machine is big enough — the memory is in use. Close some applications, then use Re-check.'
    case 'wont_fit':
      return 'Freeing memory will not help; pick a smaller model or a smaller quantization.'
    default:
      return null
  }
}

/* Whether re-checking this row could ever change its answer.

   Only memory pressure is transient. A model larger than the whole machine stays
   too large however much is freed, so offering a re-check there would be busywork
   dressed as a remedy. */
export function fitIsRecheckable(fit) {
  return fit === 'insufficient_free_memory'
}

/* --- quantization --------------------------------------------------------- */

/* Ordered from highest to lowest fidelity. Rank is ordinal only — it encodes the
   established ordering of the GGUF quant families (wider integer types keep more
   of the original weights; within a family `_L` > `_M` > `_S`) and is never
   presented as a measured quality score.

   `note` is what the user needs to decide, in their words, not ours. */
const QUANT_TABLE = [
  ['F32', 'full', 'Unquantized — largest file, no quantization loss'],
  ['BF16', 'full', 'Unquantized — largest file, no quantization loss'],
  ['F16', 'full', 'Unquantized — largest file, no quantization loss'],
  ['Q8_K', 'high', 'Near-lossless — big, closest to the original weights'],
  ['Q8_0', 'high', 'Near-lossless — big, closest to the original weights'],
  ['Q6_K', 'high', 'Very close to the original weights'],
  ['Q5_K_M', 'good', 'High quality'],
  ['Q5_K_S', 'good', 'High quality'],
  ['Q5_K', 'good', 'High quality'],
  ['Q5_1', 'good', 'High quality'],
  ['Q5_0', 'good', 'High quality'],
  ['Q4_K_M', 'balanced', 'Balanced — the usual default'],
  ['Q4_K_S', 'balanced', 'Balanced, slightly smaller than Q4_K_M'],
  ['Q4_K', 'balanced', 'Balanced'],
  ['Q4_1', 'balanced', 'Balanced'],
  ['Q4_0', 'balanced', 'Balanced'],
  ['IQ4_NL', 'balanced', 'Balanced (imatrix) — needs an imatrix-aware runtime'],
  ['IQ4_XS', 'balanced', 'Balanced (imatrix) — needs an imatrix-aware runtime'],
  ['Q3_K_L', 'reduced', 'Smaller, noticeably lower quality'],
  ['Q3_K_M', 'reduced', 'Smaller, noticeably lower quality'],
  ['Q3_K_S', 'reduced', 'Smaller, noticeably lower quality'],
  ['Q3_K', 'reduced', 'Smaller, noticeably lower quality'],
  ['IQ3_M', 'reduced', 'Smaller, noticeably lower quality (imatrix)'],
  ['IQ3_S', 'reduced', 'Smaller, noticeably lower quality (imatrix)'],
  ['IQ3_XS', 'reduced', 'Smaller, noticeably lower quality (imatrix)'],
  ['IQ3_XXS', 'reduced', 'Smaller, noticeably lower quality (imatrix)'],
  ['Q2_K_S', 'severe', 'Heavy quality loss — a last resort for tight memory'],
  ['Q2_K', 'severe', 'Heavy quality loss — a last resort for tight memory'],
  ['IQ2_M', 'severe', 'Heavy quality loss (imatrix)'],
  ['IQ2_S', 'severe', 'Heavy quality loss (imatrix)'],
  ['IQ2_XS', 'severe', 'Heavy quality loss (imatrix)'],
  ['IQ2_XXS', 'severe', 'Heavy quality loss (imatrix)'],
  ['IQ1_M', 'extreme', 'Extreme compression — output is often incoherent'],
  ['IQ1_S', 'extreme', 'Extreme compression — output is often incoherent'],
]

const QUANT_BY_TOKEN = new Map(
  QUANT_TABLE.map(([token, tier, note], index) => [
    token,
    { token, tier, note, rank: QUANT_TABLE.length - index },
  ]),
)

/* What we can say about a quantization token. An unrecognized token gets a rank of
   0 and no note: the browse layer guesses quants from filenames, so "we do not
   recognize this" must not be dressed up as advice. */
export function quantAdvice(quant) {
  return (
    QUANT_BY_TOKEN.get(String(quant || '').toUpperCase()) || {
      token: quant || '',
      tier: 'unknown',
      note: null,
      rank: 0,
    }
  )
}

/* Highest fidelity first; ties broken by size then filename so the order is total
   and stable across renders (two files can share a quant token when a repo ships
   variants the filename guesser cannot tell apart). */
export function compareByQuality(a, b) {
  const byRank = quantAdvice(b.quant).rank - quantAdvice(a.quant).rank
  if (byRank !== 0) return byRank
  const bySize = (b.size_bytes || 0) - (a.size_bytes || 0)
  if (bySize !== 0) return bySize
  return String(a.filename).localeCompare(String(b.filename))
}

/* --- repo grouping -------------------------------------------------------- */

/* `owner/Model-Name-GGUF` → `Model-Name`. The `-GGUF` suffix is a packaging
   convention, not part of the model's name, and repeating the owner in the title
   wastes the widest line in the card. */
export function repoTitle(repoId) {
  const withoutOwner = String(repoId || '').split('/').pop() || ''
  return withoutOwner.replace(/[-_. ]?GGUF$/i, '') || withoutOwner
}

export function repoOwner(repoId) {
  const parts = String(repoId || '').split('/')
  return parts.length > 1 ? parts[0] : ''
}

/* Collapse flat file rows into one entry per repo, preserving the order the repos
   first appeared in.

   That order is the ranking: the Hub returns repos by search relevance and the
   server preserves it, so re-sorting here (by downloads, say) would discard
   relevance for a number that is identical across every file of a repo and
   therefore cannot discriminate within one. */
export function groupHfFilesByRepo(items) {
  const order = []
  const byRepo = new Map()
  for (const item of items) {
    const key = item.repo_id
    if (!byRepo.has(key)) {
      byRepo.set(key, [])
      order.push(key)
    }
    byRepo.get(key).push(item)
  }
  return order.map((repoId) => {
    const files = byRepo.get(repoId).slice().sort(compareByQuality)
    return {
      repoId,
      owner: repoOwner(repoId),
      title: repoTitle(repoId),
      files,
      // Repo-level facts: identical on every file, so read them off the first.
      downloads: files[0]?.downloads || 0,
      likes: files[0]?.likes || 0,
      architecture: files[0]?.architecture || '',
      archSupport: files[0]?.arch_support || 'unknown',
    }
  })
}

/* Which quantization to preselect, as an index into a quality-sorted `files`.

   Order of preference:
   1. The highest-fidelity file with a PROVEN fit — the best thing this machine can
      actually run.
   2. Nothing proven: the balanced tier, which is what a person picks when they do
      not know their memory budget. Defaulting to the largest file we merely failed
      to rule out would preselect a 24 GB download on an unknown host.
   3. Everything is refused: the smallest file, so the card still shows the closest
      thing to viable. */
export function defaultFileIndex(files) {
  if (!files.length) return -1
  const proven = files.findIndex((file) => isPositiveFit(file.fit))
  if (proven >= 0) return proven
  const candidates = files.filter((file) => !isRefusingFit(file.fit))
  if (!candidates.length) return files.length - 1
  const balanced = candidates.find((file) => quantAdvice(file.quant).tier === 'balanced')
  return files.indexOf(balanced || candidates[candidates.length - 1])
}

/* Split repo groups by whether Camelid implements the architecture at all.

   `unknown` groups with the runnable ones on purpose: the architecture came from a
   filename guess, so "we could not tell" must not hide a model that would load.
   Only a recognized-and-unimplemented architecture is filtered out. */
export function partitionByArchSupport(groups) {
  return {
    loadable: groups.filter((group) => group.archSupport !== 'not_implemented'),
    unimplemented: groups.filter((group) => group.archSupport === 'not_implemented'),
  }
}

/* Split curated rows into what this machine can run and what it cannot, for the
   no-search landing state. Rows with an `unknown` verdict count as runnable: an
   unprobed host must not have its whole catalog folded away. */
export function partitionCuratedByFit(items) {
  return {
    runnable: items.filter((item) => !isRefusingFit(item.fit)),
    blocked: items.filter((item) => isRefusingFit(item.fit)),
  }
}
