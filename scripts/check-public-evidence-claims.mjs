#!/usr/bin/env node
import { access, readdir, readFile } from 'node:fs/promises'
import { join, relative, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const rootDir = resolve(args.get('root') || 'qa/evidence-bundles')
const failures = []
const privacyScannedPaths = new Set()
let checkedBundles = 0
let checkedSummaries = 0

const manifestPaths = await findManifestPaths(rootDir)
for (const manifestPath of manifestPaths) {
  checkedBundles += 1
  await validateBundle(manifestPath)
}
await validateMixtralBlockerReconciliation(rootDir, manifestPaths)
await validateMistralSupportPromotionLedger(rootDir, manifestPaths)

if (failures.length > 0) {
  console.error(`public evidence claim check failed with ${failures.length} finding(s):`)
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`public evidence claim check passed: ${checkedBundles} manifest(s), ${checkedSummaries} summary file(s)`)

async function validateMistralSupportPromotionLedger(root, manifestPaths) {
  if ((relative(process.cwd(), root).replace(/\\/g, '/') || '.') !== 'qa/evidence-bundles') return

  const manifests = []
  for (const manifestPath of manifestPaths) manifests.push({ path: manifestPath, json: await readJson(manifestPath) })
  const promotion = manifests.find(({ json }) => {
    return json?.row === 'Mistral-7B-Instruct-v0.3.Q8_0.gguf' && json?.validation_kind === 'mistral_support_promotion_api_webui_smoke'
  })
  if (!promotion) {
    fail('qa/evidence-bundles', 'Mistral support ledger requires the support-promotion API/WebUI smoke manifest')
    return
  }

  validateMistralSupportPromotionManifest(evidenceBundlePath(promotion.path), promotion.json)
  await validateMistralSupportSurfaceText()
}

function validateMistralSupportPromotionManifest(bundleRel, manifest) {
  if (manifest.passed !== true) fail(bundleRel, 'Mistral support-promotion manifest must record passed=true')
  if (manifest.smoke_exit !== 0) fail(bundleRel, 'Mistral support-promotion manifest must record smoke_exit=0')
  if (!String(manifest.support_claim || '').includes('supported_exact_row_smoke')) {
    fail(bundleRel, 'Mistral support-promotion manifest support_claim must include supported_exact_row_smoke')
  }
  const contract = manifest.expected_contract || {}
  if (contract.compatibility_row !== 'mistral_7b_instruct_v0_3_q8_0') fail(bundleRel, 'Mistral support-promotion compatibility_row must match the exact row id')
  if (contract.compatibility_status !== 'supported_exact_row_smoke') fail(bundleRel, 'Mistral support-promotion compatibility_status must be supported_exact_row_smoke')
  if (contract.contract_supported !== true) fail(bundleRel, 'Mistral support-promotion contract_supported must be true')
  if (!/^enabled/i.test(String(contract.webui_chat || ''))) fail(bundleRel, 'Mistral support-promotion webui_chat must be enabled for the exact row')
  if (manifest.observed?.capabilities_row_status !== 'supported_exact_row_smoke') fail(bundleRel, 'Mistral support-promotion observed capabilities row status must be supported_exact_row_smoke')
  if (!String(manifest.observed?.execution_plan_support_level || '').includes('supported_exact_row_smoke')) {
    fail(bundleRel, 'Mistral support-promotion execution plan support level must include supported_exact_row_smoke')
  }
}

async function validateMistralSupportSurfaceText() {
  const required = new Map([
    ['README.md', ['Mistral 7B Instruct v0.3 | Q8_0', 'Exact-row smoke + bounded context 512']],
    ['docs/reference/STATUS.md', ['Mistral 7B Instruct v0.3 Q8_0 is promoted to supported exact-row smoke', 'mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json']],
    ['ROADMAP.md', ['Mistral 7B Instruct v0.3 Q8_0 is supported as exact-row smoke', 'support-promotion API/WebUI smoke bundle']],
    ['COMPATIBILITY.md', ['Mistral-7B-Instruct-v0.3.Q8_0.gguf is supported exact-row smoke', 'mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json']],
    ['docs/VALIDATION_MATRIX.md', ['Mistral 7B Instruct v0.3 Q8_0 is supported exact-row smoke only', 'mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json']],
    ['docs/release/CORRECTNESS_v0.1.md', ['Camelid v0.1 has exact-row smoke correctness proof for the exact Mistral-7B-Instruct-v0.3 Q8_0 row', 'mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/manifest.json']],
    ['qa/evidence-bundles/README.md', ['mistral-7b-v0.3-q8-support-promotion-20260605T090914Z-head-d7b1699/', 'No blocker for exact-row smoke support']],
    ['src/api/mod.rs', ['id: "mistral_7b_instruct_v0_3_q8_0"', 'status: "supported_exact_row_smoke"']],
  ])
  const stalePatterns = [
    /Mistral(?:-|\s)7B[^\n]{0,260}(?:remains active exact-row validation only|remains evidence-only active validation|active validation only|not supported yet|support is not promoted|support remains fail-closed|row fail-closed)/i,
    /Mistral(?:-|\s)7B[^\n]{0,260}active_validation_unsupported/i,
    /Mistral(?:-|\s)7B[^\n]{0,260}fail-closed API\/WebUI\/RSS evidence/i,
  ]

  for (const [path, snippets] of required) {
    const text = await readText(path)
    for (const snippet of snippets) {
      if (!text.includes(snippet)) fail(path, `Mistral support ledger text must include ${JSON.stringify(snippet)}`)
    }
    for (const pattern of stalePatterns) {
      if (pattern.test(text)) fail(path, `Mistral support ledger text still matches stale fail-closed wording ${pattern}`)
    }
  }
}

async function validateBundle(manifestPath) {
  const bundleDir = manifestPath.slice(0, -'/manifest.json'.length)
  const bundleRel = relative(process.cwd(), bundleDir) || '.'
  const manifest = await readJson(manifestPath)
  validatePublicJsonNoLocalPaths(manifestPath, manifest)

  if (!manifest || typeof manifest !== 'object') {
    fail(bundleRel, 'manifest.json is not a JSON object')
    return
  }
  const schema = typeof manifest.schema === 'string' ? manifest.schema : ''

  const summaryPath = join(bundleDir, 'summary.json')
  const summaryExists = await exists(summaryPath)
  if (summaryExists) {
    checkedSummaries += 1
    const summary = await readJson(summaryPath)
    validatePublicJsonNoLocalPaths(summaryPath, summary)
    validateSummaryAgreement(bundleRel, manifest, summary)
  }

  if (schema === 'camelid.four_row_context_512_public_evidence.v1') {
    if (!summaryExists) fail(bundleRel, 'four-row context-512 bundle must include summary.json')
    await validateFourRowContext512(bundleRel, manifest, summaryExists ? await readJson(summaryPath) : null)
  }

  if (schema === 'camelid.llama3_8b_context_1024_2048_current_head_public_evidence.v1') {
    if (!summaryExists) fail(bundleRel, 'Llama 3 8B context-1024/2048 bundle must include summary.json')
    validateLlama3_8bContext1024And2048(bundleRel, manifest)
  }

  if (schema === 'camelid.mistral_7b_context_512_1024_2048_current_head_public_evidence.v1') {
    if (!summaryExists) fail(bundleRel, 'Mistral 7B context-512/1024/2048 bundle must include summary.json')
    validateMistral7bContext51210242048(bundleRel, manifest)
  }

  if (schema === 'camelid.backend_local_current_head_exact_row_guardrail.v1') {
    validateBackendLocalCurrentHeadExactRowGuardrail(bundleRel, manifest)
  }

  const singleRowContext = singleRowContextSchema(schema)
  if (singleRowContext) validateSingleRowContextBundle(bundleRel, manifest, singleRowContext)

  const legacyPublicContext = legacyPublicContextSchema(manifest)
  if (legacyPublicContext) validateLegacyPublicContextBundle(bundleRel, manifest, legacyPublicContext)
}

async function validateMixtralBlockerReconciliation(root, manifestPaths) {
  const summaryPaths = await findSummaryPaths(root)
  const manifests = []
  for (const manifestPath of manifestPaths) manifests.push({ path: manifestPath, json: await readJson(manifestPath) })
  const summaries = []
  for (const summaryPath of summaryPaths) {
    const json = await readJson(summaryPath)
    validatePublicJsonNoLocalPaths(summaryPath, json)
    summaries.push({ path: summaryPath, json })
  }

  const promotionClaimPaths = new Set()
  for (const { path, json } of manifests) {
    if (json?.schema !== 'camelid.mixtral_exact_row_manifest.v1') continue
    promotionClaimPaths.add(evidenceBundlePath(path))
    for (const dependency of mixtralPromotionDependencyBundles(json, root)) promotionClaimPaths.add(dependency)
  }
  for (const { path, json } of summaries) {
    if (json?.schema !== 'camelid.mixtral_exact_row_promotion_sync.v1' && json?.support_label !== 'supported_exact_row_smoke') continue
    promotionClaimPaths.add(evidenceBundlePath(path))
    for (const dependency of mixtralPromotionDependencyBundles(json, root)) promotionClaimPaths.add(dependency)
  }
  const blockerPaths = summaries
    .filter(({ json }) => mixtralSummaryRecordsBlocker(json))
    .map(({ path }) => path)
  if (promotionClaimPaths.size === 0 || blockerPaths.length === 0) return

  const reconciliation = manifests.find(({ json }) => json?.schema === 'camelid.mixtral_blocker_reconciliation.v1')
  if (!reconciliation) {
    fail(relative(process.cwd(), root) || '.', 'Mixtral promotion claims conflict with later long-output blocker evidence; add a camelid.mixtral_blocker_reconciliation.v1 manifest before public claim checks can pass')
    return
  }
  validateMixtralReconciliationManifest(relative(process.cwd(), reconciliation.path), reconciliation.json, [...promotionClaimPaths], blockerPaths)
}

function mixtralPromotionDependencyBundles(json, root) {
  const bundles = []
  for (const value of arrayStrings(json?.required_prior_gates)) bundles.push(normalizeEvidenceBundleRef(value, root))
  for (const value of arrayStrings(json?.artifact_bundles)) bundles.push(normalizeEvidenceBundleRef(value, root))
  if (Array.isArray(json?.promotion_gate_artifacts)) {
    for (const artifact of json.promotion_gate_artifacts) {
      if (typeof artifact?.bundle === 'string') bundles.push(normalizeEvidenceBundleRef(artifact.bundle, root))
    }
  }
  return bundles.filter(Boolean)
}

function arrayStrings(value) {
  return Array.isArray(value) ? value.filter((item) => typeof item === 'string') : []
}

function normalizeEvidenceBundleRef(value, root) {
  if (value.startsWith('qa/evidence-bundles/')) return value
  if (value.startsWith('/')) return relative(process.cwd(), value)
  const rootRel = relative(process.cwd(), root) || '.'
  if (value.startsWith(`${rootRel}/`) || value === rootRel) return value
  return `${rootRel}/${value}`
}

function mixtralSummaryRecordsBlocker(summary) {
  if (!summary || typeof summary !== 'object' || Array.isArray(summary)) return false
  if (summary.schema === 'camelid.mixtral_gate9a_long_output.v1') {
    return summary.all_token_parity === false || summary.all_completed === false || /diverg/i.test(String(summary.stop_reason || ''))
  }
  if (summary.schema === 'camelid.mixtral_longgen_continuation.summary.v1') {
    return summary.status === 'partial_failure' || /hung|diverg/i.test(String(summary.runner_stop_reason || ''))
  }
  if (summary.schema === 'camelid.mixtral_backend_hang_guard.v1') {
    return /blocked|hang|partial/i.test(String(summary.status || summary.result || ''))
  }
  return false
}

function validateMixtralReconciliationManifest(bundleRel, manifest, promotionClaimPaths, blockerPaths) {
  const boundary = String(manifest.support_boundary || manifest.claim_boundary || '')
  if (manifest.current_status !== 'active_validation_partial_runtime') {
    fail(bundleRel, 'Mixtral blocker reconciliation current_status must be active_validation_partial_runtime')
  }
  if (manifest.support_label !== 'unsupported_bounded_one_token_runtime_only') {
    fail(bundleRel, 'Mixtral blocker reconciliation support_label must be unsupported_bounded_one_token_runtime_only')
  }
  if (!/one-token/i.test(boundary) || !/unsupported|blocked/i.test(boundary) || !/no broad Mixtral/i.test(boundary)) {
    fail(bundleRel, 'Mixtral blocker reconciliation support_boundary must say one-token evidence is unsupported/blocked and no broad Mixtral support is claimed')
  }
  const supersedes = new Set(Array.isArray(manifest.supersedes) ? manifest.supersedes : [])
  const blockers = new Set(Array.isArray(manifest.blocker_evidence) ? manifest.blocker_evidence : [])
  for (const path of promotionClaimPaths) {
    const bundlePath = evidenceBundlePath(path)
    if (!supersedes.has(bundlePath)) fail(bundleRel, `Mixtral blocker reconciliation supersedes must include ${bundlePath}`)
  }
  for (const path of blockerPaths) {
    const bundlePath = evidenceBundlePath(path)
    if (!blockers.has(bundlePath)) fail(bundleRel, `Mixtral blocker reconciliation blocker_evidence must include ${bundlePath}`)
  }
}

function validateBackendLocalCurrentHeadExactRowGuardrail(bundleRel, manifest) {
  if (manifest.local_only !== true) fail(bundleRel, 'backend local guardrail local_only must be true')
  const noRemoteValidationUsed = manifest.no_remote_validation_used === true || (manifest.remote_ssh_attempted === false && manifest.remote_validation_host_assumed_available === false)
  if (!noRemoteValidationUsed) fail(bundleRel, 'backend local guardrail no_remote_validation_used must be true')
  if (manifest.passed !== true && manifest.exit_code !== 0) fail(bundleRel, 'backend local guardrail passed must be true')
  const head = String(manifest.head || manifest.git?.head || '')
  if (!/^[0-9a-f]{40}$/.test(head)) {
    fail(bundleRel, 'backend local guardrail head must be a full 40-character git sha')
  }
  const headShort = String(manifest.head_short || manifest.git?.head_short || '')
  if (!/^[0-9a-f]{7,12}$/.test(headShort)) {
    fail(bundleRel, 'backend local guardrail head_short must be a 7-12 character git sha prefix')
  } else if (head && !head.startsWith(headShort)) {
    fail(bundleRel, 'backend local guardrail head_short must prefix head')
  }
  const requiredCommands = ['cargo-test-capabilities', 'public-evidence-claims', 'public-scrub', 'git-diff-check']
  if (Array.isArray(manifest.commands)) {
    const commandNames = new Set(manifest.commands.map((command) => String(command?.name || '').replaceAll('_', '-')))
    const legacyCommandNames = new Map([
      ['cargo-test-capabilities', 'cargo-capabilities-all-targets'],
      ['public-evidence-claims', 'public-evidence-claims'],
      ['public-scrub', 'public-scrub'],
      ['git-diff-check', 'git-diff-check'],
    ])
    for (const command of requiredCommands) {
      if (!commandNames.has(legacyCommandNames.get(command))) fail(bundleRel, `backend local guardrail command ${command} must be recorded`)
    }
  } else {
    const commands = manifest.commands && typeof manifest.commands === 'object' ? manifest.commands : null
    for (const command of requiredCommands) {
      if (commands?.[command] !== 0) fail(bundleRel, `backend local guardrail command ${command} must exit 0`)
    }
  }
  const boundary = typeof manifest.support_boundary === 'string' ? manifest.support_boundary : JSON.stringify(manifest.support_boundary || {})
  if (!/Mixtral/i.test(boundary) || !/unsupported|blocked|blocker/i.test(boundary) || !/Llama/i.test(boundary) || !/exact-row bounded/i.test(boundary) || !/no broad/i.test(boundary)) {
    fail(bundleRel, 'backend local guardrail support_boundary must preserve Mixtral blocked, Llama exact-row bounded, and no broad/full-support promotion truth')
  }
}

function evidenceBundlePath(path) {
  // Normalize to forward slashes so comparisons match the `/`-style paths stored
  // in evidence manifests on every platform. Node's `relative` emits the OS
  // separator (backslashes on Windows), which otherwise both breaks the
  // `/manifest.json` trim below and mismatches the manifests' stored refs —
  // producing false failures on Windows while Ubuntu CI stays green.
  const rel = relative(process.cwd(), path).replace(/\\/g, '/')
  const marker = '/manifest.json'
  if (rel.endsWith(marker)) return rel.slice(0, -marker.length)
  const summaryMarker = '/summary.json'
  if (rel.endsWith(summaryMarker)) return rel.slice(0, -summaryMarker.length)
  return rel
}

function validateSummaryAgreement(bundleRel, manifest, summary) {
  if (!summary || typeof summary !== 'object') {
    fail(bundleRel, 'summary.json is not a JSON object')
    return
  }
  if (summary.source_manifest !== undefined && summary.source_manifest !== 'manifest.json') {
    fail(bundleRel, `summary.json source_manifest must be manifest.json, got ${JSON.stringify(summary.source_manifest)}`)
  }
  if (typeof manifest.passed === 'boolean' && typeof summary.passed === 'boolean' && manifest.passed !== summary.passed) {
    fail(bundleRel, `manifest passed=${manifest.passed} disagrees with summary passed=${summary.passed}`)
  }
  validateCurrentHeadAgreement(bundleRel, manifest, summary)
  if (typeof manifest.claim_boundary === 'string' && typeof summary.claim_boundary === 'string' && manifest.claim_boundary !== summary.claim_boundary) {
    fail(bundleRel, 'manifest and summary claim_boundary differ')
  }
  if (Array.isArray(manifest.rows) && Array.isArray(summary.rows)) {
    if (manifest.rows.length !== summary.rows.length) {
      fail(bundleRel, `manifest has ${manifest.rows.length} row(s) but summary has ${summary.rows.length}`)
      return
    }
    for (let index = 0; index < manifest.rows.length; index += 1) {
      const manifestRow = manifest.rows[index]
      const summaryRow = summary.rows[index]
      const rowId = manifestRow.row_id || `row-${index}`
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'row_id')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'context_window')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'reference_prompt_token_count')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'max_tokens')
      compareRowField(bundleRel, rowId, manifestRow, summaryRow, 'max_resident_set_kib')
      compareQ8FileReadStats(bundleRel, rowId, manifestRow, summaryRow)
      compareQ8FileReadStats(bundleRel, rowId, manifestRow, summaryRow, 'prefill_q8_file_reads')
      compareQ8FileReadStats(bundleRel, rowId, manifestRow, summaryRow, 'first_token_q8_file_reads')
      compareQ8FileReadStats(bundleRel, rowId, manifestRow, summaryRow, 'generation_q8_file_reads')
      if (typeof summaryRow.passed === 'boolean') {
        const manifestPassed = rowPassed(manifestRow)
        if (manifestPassed !== summaryRow.passed) fail(bundleRel, `${rowId} summary passed=${summaryRow.passed} disagrees with manifest row checks`)
      }
    }
  }
}

function rowPassed(row) {
  if (typeof row?.passed === 'boolean') return row.passed
  if (
    typeof row?.prompt_tokens_match === 'boolean' ||
    typeof row?.generated_tokens_match === 'boolean' ||
    typeof row?.generated_text_match === 'boolean'
  ) {
    return row.prompt_tokens_match === true && row.generated_tokens_match === true && row.generated_text_match === true
  }
  if (
    typeof row?.prompt_tokens_all_match === 'boolean' ||
    typeof row?.generated_tokens_all_match === 'boolean' ||
    typeof row?.generated_text_all_match === 'boolean'
  ) {
    return row.prompt_tokens_all_match === true && row.generated_tokens_all_match === true && row.generated_text_all_match === true
  }
  return undefined
}

async function validateFourRowContext512(bundleRel, manifest, summary) {
  const expectedRows = [
    'tinyllama_1_1b_chat_q8_0',
    'llama32_1b_instruct_q8_0',
    'llama32_3b_instruct_q8_0',
    'llama3_8b_instruct_q8_0',
  ]

  if (manifest.passed !== true) fail(bundleRel, 'four-row context-512 manifest must be passed=true')
  if (manifest.checkout_clean !== true) fail(bundleRel, 'four-row context-512 manifest must record checkout_clean=true')
  if (manifest.pack?.target_context_window !== 512) fail(bundleRel, 'four-row context-512 manifest pack target_context_window must be 512')
  if (manifest.pack?.max_tokens !== 5) fail(bundleRel, 'four-row context-512 manifest pack max_tokens must be 5')
  if (manifest.pack?.source_prompt_pack !== 'qa/prompt-packs/llama3-context-512-smoke.json') {
    fail(bundleRel, 'four-row context-512 manifest source_prompt_pack must stay on the checked llama3 context pack')
  }
  if (!boundaryIsNarrow(manifest.claim_boundary)) fail(bundleRel, 'four-row context-512 claim_boundary must explicitly avoid broader/full-family promotion')

  validateChecksObject(bundleRel, 'summary.checks', summary?.checks, [
    'checkout_clean',
    'prompt_tokens_all_match',
    'generated_tokens_all_match',
    'generated_text_all_match',
    'all_rows_have_bounded_rss',
  ])

  if (!Array.isArray(manifest.rows)) {
    fail(bundleRel, 'four-row context-512 manifest rows must be an array')
    return
  }
  const rowIds = manifest.rows.map((row) => row.row_id)
  if (JSON.stringify(rowIds) !== JSON.stringify(expectedRows)) {
    fail(bundleRel, `four-row context-512 row order changed: ${JSON.stringify(rowIds)}`)
  }
  for (const row of manifest.rows) validateContext512Row(bundleRel, row)
}

function validateContext512Row(bundleRel, row) {
  validateContextRow(bundleRel, row, {
    contextWindow: 512,
    maxTokens: 5,
    minPromptTokens: 1,
  })
}

function validateLlama3_8bContext1024And2048(bundleRel, manifest) {
  if (manifest.passed !== true) fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must be passed=true')
  if (manifest.checkout_clean !== true) fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must record checkout_clean=true')
  if (!boundaryIsNarrow(manifest.claim_boundary)) {
    fail(bundleRel, 'Llama 3 8B context-1024/2048 claim_boundary must explicitly avoid broader/full-family promotion')
  }
  const expectedPackIds = ['llama3-context-1024-smoke-v1', 'llama3-context-2048-smoke-v1']
  const expectedSourcePacks = ['qa/prompt-packs/llama3-context-1024-smoke.json', 'qa/prompt-packs/llama3-context-2048-smoke.json']
  if (manifest.pack !== undefined) {
    if (manifest.pack?.max_tokens !== 5) fail(bundleRel, 'Llama 3 8B context-1024/2048 pack max_tokens must be 5')
    if (JSON.stringify(manifest.pack?.ids) !== JSON.stringify(expectedPackIds)) {
      fail(bundleRel, `Llama 3 8B context-1024/2048 pack ids changed: ${JSON.stringify(manifest.pack?.ids)}`)
    }
    if (JSON.stringify(manifest.pack?.source_prompt_packs) !== JSON.stringify(expectedSourcePacks)) {
      fail(bundleRel, `Llama 3 8B context-1024/2048 source_prompt_packs changed: ${JSON.stringify(manifest.pack?.source_prompt_packs)}`)
    }
  }
  if (!Array.isArray(manifest.rows) || manifest.rows.length !== 2) {
    fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must include exactly two rows')
    return
  }
  validateContextRow(bundleRel, manifest.rows[0], {
    rowId: 'llama3_8b_instruct_q8_0',
    contextWindow: 1024,
    maxTokens: 5,
    minPromptTokens: 513,
    promptId: 'roughly-1024-token-recall',
  })
  validateContextText(bundleRel, 'llama3_8b_instruct_q8_0 context-1024', manifest.rows[0], 'CMLD-102')
  validateContextRow(bundleRel, manifest.rows[1], {
    rowId: 'llama3_8b_instruct_q8_0',
    contextWindow: 2048,
    maxTokens: 5,
    minPromptTokens: 1025,
    promptId: 'roughly-2048-token-recall',
  })
  validateContextText(bundleRel, 'llama3_8b_instruct_q8_0 context-2048', manifest.rows[1], 'CMLD-204')
}

function validateMistral7bContext51210242048(bundleRel, manifest) {
  if (manifest.passed !== true) fail(bundleRel, 'Mistral 7B context-512/1024/2048 manifest must be passed=true')
  if (!boundaryIsNarrow(manifest.claim_boundary)) {
    fail(bundleRel, 'Mistral 7B context-512/1024/2048 claim_boundary must explicitly avoid broader/full-family promotion')
  }
  validateChecksObject(bundleRel, 'manifest.checks', manifest.checks, [
    'prompt_tokens_all_match',
    'generated_tokens_all_match',
    'generated_text_all_match',
    'api_health_loaded_generation_ready',
    'v1_completions_passed',
    'v1_chat_completions_passed',
    'privacy_scrub_required',
  ])
  const expectedPackIds = ['mistral-context-512-smoke-v1', 'mistral-context-1024-smoke-v1', 'mistral-context-2048-smoke-v1']
  const expectedSourcePacks = ['qa/prompt-packs/mistral-context-512-smoke.json', 'qa/prompt-packs/mistral-context-1024-smoke.json', 'qa/prompt-packs/mistral-context-2048-smoke.json']
  if (manifest.pack !== undefined) {
    if (manifest.pack?.max_tokens !== 5) fail(bundleRel, 'Mistral 7B context-512/1024/2048 pack max_tokens must be 5')
    if (JSON.stringify(manifest.pack?.ids) !== JSON.stringify(expectedPackIds)) {
      fail(bundleRel, `Mistral 7B context-512/1024/2048 pack ids changed: ${JSON.stringify(manifest.pack?.ids)}`)
    }
    if (JSON.stringify(manifest.pack?.source_prompt_packs) !== JSON.stringify(expectedSourcePacks)) {
      fail(bundleRel, `Mistral 7B context-512/1024/2048 source_prompt_packs changed: ${JSON.stringify(manifest.pack?.source_prompt_packs)}`)
    }
  }
  if (!Array.isArray(manifest.rows) || manifest.rows.length !== 3) {
    fail(bundleRel, 'Mistral 7B context-512/1024/2048 manifest must include exactly three rows')
    return
  }
  const expectedByWindow = new Map([
    [512, { promptId: 'mistral-roughly-512-token-recall', minPromptTokens: 257 }],
    [1024, { promptId: 'mistral-roughly-1024-token-recall', minPromptTokens: 513 }],
    [2048, { promptId: 'mistral-roughly-2048-token-recall', minPromptTokens: 1025 }],
  ])
  const seen = new Set()
  for (const row of manifest.rows) {
    const expected = expectedByWindow.get(row?.context_window)
    if (!expected) {
      fail(bundleRel, `unexpected Mistral context_window ${JSON.stringify(row?.context_window)}`)
      continue
    }
    seen.add(row.context_window)
    validateMistralContextRow(bundleRel, row, expected)
  }
  for (const window of expectedByWindow.keys()) {
    if (!seen.has(window)) fail(bundleRel, `missing Mistral context_window ${window}`)
  }
}

function validateMistralContextRow(bundleRel, row, expected) {
  const rowId = row?.row_id || '<missing row_id>'
  if (rowId !== 'mistral_7b_instruct_v0_3_q8_0') fail(bundleRel, `row_id must be mistral_7b_instruct_v0_3_q8_0, got ${rowId}`)
  if (row.max_tokens !== 5) fail(bundleRel, `${rowId} max_tokens must be 5`)
  if (row.prompt_id !== expected.promptId) fail(bundleRel, `${rowId} prompt_id must be ${expected.promptId}`)
  if (row.generated_text !== ' The repeat marker is "') fail(bundleRel, `${rowId} generated_text must stay ${JSON.stringify(' The repeat marker is "')}, got ${JSON.stringify(row.generated_text)}`)
  if (row.llama_text !== ' The repeat marker is "') fail(bundleRel, `${rowId} llama_text must stay ${JSON.stringify(' The repeat marker is "')}, got ${JSON.stringify(row.llama_text)}`)
  if (row.passed !== undefined && row.passed !== true) fail(bundleRel, `${rowId} passed must be true`)
  if (row.prompt_tokens_match !== true) fail(bundleRel, `${rowId} prompt_tokens_match must be true`)
  if (row.generated_tokens_match !== true) fail(bundleRel, `${rowId} generated_tokens_match must be true`)
  if (row.generated_text_match !== true) fail(bundleRel, `${rowId} generated_text_match must be true`)
  if (row.first_generated_token_diff_index !== -1) fail(bundleRel, `${rowId} first_generated_token_diff_index must be -1`)
  if (!Number.isInteger(row.reference_prompt_token_count) || row.reference_prompt_token_count < expected.minPromptTokens) {
    fail(bundleRel, `${rowId} reference_prompt_token_count must be at least ${expected.minPromptTokens}`)
  }
  if (JSON.stringify(row.backend_generated_tokens) !== JSON.stringify([1183, 14518, 19612, 1117, 1113])) {
    fail(bundleRel, `${rowId} backend_generated_tokens must stay [1183,14518,19612,1117,1113]`)
  }
  if (JSON.stringify(row.llama_generated_tokens) !== JSON.stringify([1183, 14518, 19612, 1117, 1113])) {
    fail(bundleRel, `${rowId} llama_generated_tokens must stay [1183,14518,19612,1117,1113]`)
  }
  if (typeof row.model_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(row.model_sha256)) fail(bundleRel, `${rowId} model_sha256 must be a 64-character lowercase sha256`)
  if (typeof row.raw_artifact !== 'string' || row.raw_artifact.startsWith('/') || row.raw_artifact.includes('..')) fail(bundleRel, `${rowId} raw_artifact must be a safe relative path`)
}

function validateContextText(bundleRel, label, row, expected) {
  const generatedText = row.generated_text ?? row.backend_text
  const llamaText = row.llama_text ?? generatedText
  if (generatedText !== expected) fail(bundleRel, `${label} generated text must stay ${JSON.stringify(expected)}, got ${JSON.stringify(generatedText)}`)
  if (llamaText !== expected) fail(bundleRel, `${label} llama text must stay ${JSON.stringify(expected)}, got ${JSON.stringify(llamaText)}`)
}

function validateSingleRowContextBundle(bundleRel, manifest, expected) {
  if (manifest.passed !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must be passed=true`)
  if (manifest.checkout_clean !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must record checkout_clean=true`)
  if (manifest.pack?.target_context_window !== expected.contextWindow) {
    fail(bundleRel, `${expected.rowId} pack target_context_window must be ${expected.contextWindow}`)
  }
  if (manifest.pack?.max_tokens !== expected.maxTokens) fail(bundleRel, `${expected.rowId} pack max_tokens must be ${expected.maxTokens}`)
  if (manifest.pack?.source_prompt_pack !== expected.sourcePromptPack) {
    fail(bundleRel, `${expected.rowId} source_prompt_pack must be ${expected.sourcePromptPack}`)
  }
  if (manifest.pack?.prompt_count !== 1) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} pack prompt_count must be 1`)
  if (!boundaryIsNarrow(manifest.claim_boundary)) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} claim_boundary must explicitly avoid broader/full-family promotion`)
  }
  if (!Array.isArray(manifest.rows) || manifest.rows.length !== 1) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} manifest must include exactly one row`)
    return
  }
  validateContextRow(bundleRel, manifest.rows[0], expected)
}

function validateContextRow(bundleRel, row, expected) {
  const rowId = row?.row_id || '<missing row_id>'
  if (expected.rowId && rowId !== expected.rowId) fail(bundleRel, `row_id must be ${expected.rowId}, got ${rowId}`)
  if (row.context_window !== expected.contextWindow) fail(bundleRel, `${rowId} context_window must be ${expected.contextWindow}`)
  if (row.max_tokens !== expected.maxTokens) fail(bundleRel, `${rowId} max_tokens must be ${expected.maxTokens}`)
  if (expected.promptId && row.prompt_id !== expected.promptId) fail(bundleRel, `${rowId} prompt_id must be ${expected.promptId}`)
  if (expected.generatedText && row.generated_text !== expected.generatedText) {
    fail(bundleRel, `${rowId} generated_text must stay ${JSON.stringify(expected.generatedText)}`)
  }
  if (row.passed !== undefined && row.passed !== true) fail(bundleRel, `${rowId} passed must be true`)
  if (row.prompt_tokens_match !== true) fail(bundleRel, `${rowId} prompt_tokens_match must be true`)
  if (row.generated_tokens_match !== true) fail(bundleRel, `${rowId} generated_tokens_match must be true`)
  if (row.generated_text_match !== true) fail(bundleRel, `${rowId} generated_text_match must be true`)
  if (row.first_generated_token_diff_index !== -1) fail(bundleRel, `${rowId} first_generated_token_diff_index must be -1`)
  if (!Number.isInteger(row.reference_prompt_token_count) || row.reference_prompt_token_count < expected.minPromptTokens) {
    fail(bundleRel, `${rowId} reference_prompt_token_count must be at least ${expected.minPromptTokens}`)
  }
  if (!Number.isInteger(row.max_resident_set_kib) || row.max_resident_set_kib <= 0) fail(bundleRel, `${rowId} max_resident_set_kib must be positive`)
  if (typeof row.model_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(row.model_sha256)) fail(bundleRel, `${rowId} model_sha256 must be a 64-character lowercase sha256`)
  if (typeof row.raw_artifact !== 'string' || row.raw_artifact.startsWith('/') || row.raw_artifact.includes('..')) fail(bundleRel, `${rowId} raw_artifact must be a safe relative path`)
}

function singleRowContextSchema(schema) {
  const schemas = {
    'camelid.llama32_1b_context_1024_public_evidence.v1': {
      rowId: 'llama32_1b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama32_3b_context_1024_public_evidence.v1': {
      rowId: 'llama32_3b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama32_3b_context_2048_public_evidence.v1': {
      rowId: 'llama32_3b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-2048-smoke.json',
      promptId: 'roughly-2048-token-recall',
      generatedText: 'CMLD-204',
    },
    'camelid.llama32_1b_context_8192_current_head_public_evidence.v1': {
      rowId: 'llama32_1b_instruct_q8_0',
      contextWindow: 8192,
      maxTokens: 5,
      minPromptTokens: 4097,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-8192-smoke.json',
      promptId: 'roughly-8192-token-recall',
      generatedText: 'CMLD-819',
    },
    'camelid.llama3_8b_context_1024_public_evidence.v1': {
      rowId: 'llama3_8b_instruct_q8_0',
      contextWindow: 1024,
      maxTokens: 5,
      minPromptTokens: 513,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-1024-smoke.json',
      promptId: 'roughly-1024-token-recall',
      generatedText: 'CMLD-102',
    },
    'camelid.llama3_8b_context_2048_public_evidence.v1': {
      rowId: 'llama3_8b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      sourcePromptPack: 'qa/prompt-packs/llama3-context-2048-smoke.json',
      promptId: 'roughly-2048-token-recall',
      generatedText: 'CMLD-204',
    },
  }
  return schemas[schema]
}

function legacyPublicContextSchema(manifest) {
  if (manifest.schema !== 'camelid.public-evidence-bundle.v1') return undefined
  if (
    manifest.model_row === 'llama32_1b_instruct_q8_0' &&
    manifest.pack_id === 'llama3-context-2048-smoke-v1' &&
    manifest.target_context_window === 2048
  ) {
    return {
      rowId: 'llama32_1b_instruct_q8_0',
      contextWindow: 2048,
      maxTokens: 5,
      minPromptTokens: 1025,
      generatedText: 'CMLD-204',
      generatedTokens: [34, 2735, 35, 12, 7854],
    }
  }
  return undefined
}

function validateLegacyPublicContextBundle(bundleRel, manifest, expected) {
  if (manifest.model_row !== expected.rowId) fail(bundleRel, `model_row must be ${expected.rowId}, got ${manifest.model_row}`)
  if (manifest.target_context_window !== expected.contextWindow) fail(bundleRel, `${expected.rowId} target_context_window must be ${expected.contextWindow}`)
  if (manifest.max_tokens !== expected.maxTokens) fail(bundleRel, `${expected.rowId} max_tokens must be ${expected.maxTokens}`)
  if (!Number.isInteger(manifest.reference_prompt_token_count) || manifest.reference_prompt_token_count < expected.minPromptTokens) {
    fail(bundleRel, `${expected.rowId} reference_prompt_token_count must be at least ${expected.minPromptTokens}`)
  }
  if (typeof manifest.boundary !== 'string' || !boundaryIsNarrow(manifest.boundary)) {
    fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} boundary must explicitly avoid broader/full-family promotion`)
  }
  const result = manifest.result || {}
  if (result.passed !== true) fail(bundleRel, `${expected.rowId} context-${expected.contextWindow} result.passed must be true`)
  if (result.prompt_tokens_all_match !== true) fail(bundleRel, `${expected.rowId} prompt_tokens_all_match must be true`)
  if (result.generated_tokens_all_match !== true) fail(bundleRel, `${expected.rowId} generated_tokens_all_match must be true`)
  if (result.generated_text_all_match !== true) fail(bundleRel, `${expected.rowId} generated_text_all_match must be true`)
  compareExactJson(bundleRel, `${expected.rowId} backend_generated_tokens`, result.backend_generated_tokens, expected.generatedTokens)
  compareExactJson(bundleRel, `${expected.rowId} reference_generated_tokens`, result.reference_generated_tokens, expected.generatedTokens)
  if (result.backend_text !== expected.generatedText) fail(bundleRel, `${expected.rowId} backend_text must stay ${JSON.stringify(expected.generatedText)}`)
  if (result.reference_text !== expected.generatedText) fail(bundleRel, `${expected.rowId} reference_text must stay ${JSON.stringify(expected.generatedText)}`)
  if (!Array.isArray(manifest.primary_artifacts) || manifest.primary_artifacts.length === 0) {
    fail(bundleRel, `${expected.rowId} primary_artifacts must list the raw pack artifacts`)
  } else {
    for (const artifact of manifest.primary_artifacts) {
      if (typeof artifact !== 'string' || artifact.startsWith('/') || artifact.includes('..')) {
        fail(bundleRel, `${expected.rowId} primary_artifacts must be safe relative paths`)
      }
    }
  }
}

function compareExactJson(bundleRel, label, actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(bundleRel, `${label} must stay ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

function validateChecksObject(bundleRel, label, checks, requiredKeys) {
  if (!checks || typeof checks !== 'object') {
    fail(bundleRel, `${label} must be present`)
    return
  }
  for (const key of requiredKeys) {
    if (checks[key] !== true) fail(bundleRel, `${label}.${key} must be true`)
  }
}

function boundaryIsNarrow(boundary) {
  if (typeof boundary !== 'string') return false
  return /does not promote/i.test(boundary) && (/full Llama-family support/i.test(boundary) || /Mistral-family support/i.test(boundary) || /full support/i.test(boundary))
}

function compareRowField(bundleRel, rowId, manifestRow, summaryRow, field) {
  if (summaryRow[field] === undefined) return
  if (manifestRow[field] !== summaryRow[field]) {
    fail(bundleRel, `${rowId} ${field} mismatch: manifest=${JSON.stringify(manifestRow[field])} summary=${JSON.stringify(summaryRow[field])}`)
  }
}

function validateCurrentHeadAgreement(bundleRel, manifest, summary) {
  const manifestHead = firstString(manifest.git_head, manifest.source_runtime_head, manifest.source_head, manifest.head)
  const summaryHead = firstString(summary.git_head, summary.source_runtime_head, summary.source_head, summary.head)
  const isCurrentHeadBundle = manifest.schema === 'camelid.llama3_8b_context_1024_2048_current_head_public_evidence.v1'
  if (isCurrentHeadBundle) {
    if (!manifestHead || !/^[a-f0-9]{12,40}$/.test(manifestHead)) {
      fail(bundleRel, 'Llama 3 8B context-1024/2048 manifest must record a source/runtime head')
    }
    if (!summaryHead || !/^[a-f0-9]{12,40}$/.test(summaryHead)) {
      fail(bundleRel, 'Llama 3 8B context-1024/2048 summary must record the same source/runtime head as manifest')
    }
  }
  if (manifestHead && summaryHead && manifestHead !== summaryHead) {
    fail(bundleRel, `summary head mismatch: manifest=${JSON.stringify(manifestHead)} summary=${JSON.stringify(summaryHead)}`)
  }
}

function firstString(...values) {
  return values.find((value) => typeof value === 'string' && value.length > 0)
}

function compareQ8FileReadStats(bundleRel, rowId, manifestRow, summaryRow, field = 'q8_file_reads') {
  const manifestStats = manifestRow[field]
  const summaryStats = summaryRow[field]
  if (manifestStats === undefined && summaryStats === undefined) return
  if (manifestStats == null && summaryStats == null) return
  if (!isPlainObject(manifestStats) || !isPlainObject(summaryStats)) {
    fail(bundleRel, `${rowId} ${field} must be present in both manifest and summary when recorded`)
    return
  }
  const keys = [
    'read_calls',
    'read_bytes',
    'cache_hits',
    'cache_hit_bytes',
    'cache_misses',
    'cache_miss_bytes',
    'cache_inserts',
    'cache_insert_bytes',
    'cache_evictions',
    'cache_evicted_bytes',
    'cache_merges',
    'cache_merged_bytes',
    'cache_entries',
    'cache_bytes',
    'cache_capacity_bytes',
  ]
  for (const key of keys) {
    if (manifestStats[key] !== summaryStats[key]) {
      fail(bundleRel, `${rowId} ${field}.${key} mismatch: manifest=${JSON.stringify(manifestStats[key])} summary=${JSON.stringify(summaryStats[key])}`)
    }
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

async function findManifestPaths(root) {
  const paths = []
  await walk(root)
  return paths.sort()

  async function walk(currentDir) {
    const entries = await readdir(currentDir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(currentDir, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
      } else if (entry.isFile() && entry.name === 'manifest.json') {
        paths.push(fullPath)
      }
    }
  }
}

async function findSummaryPaths(root) {
  const paths = []
  await walk(root)
  return paths.sort()

  async function walk(currentDir) {
    const entries = await readdir(currentDir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(currentDir, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath)
      } else if (entry.isFile() && entry.name === 'summary.json') {
        paths.push(fullPath)
      }
    }
  }
}

function validatePublicJsonNoLocalPaths(path, value) {
  if (privacyScannedPaths.has(path)) return
  privacyScannedPaths.add(path)
  const rel = relative(process.cwd(), path)
  for (const finding of findLocalPathStrings(value)) {
    fail(rel, `${finding.location} must not expose local/private path ${JSON.stringify(finding.value)}`)
  }
}

function findLocalPathStrings(value, location = '$') {
  if (typeof value === 'string') {
    return localPathPattern().test(value) ? [{ location, value }] : []
  }
  if (!value || typeof value !== 'object') return []
  if (Array.isArray(value)) return value.flatMap((item, index) => findLocalPathStrings(item, `${location}[${index}]`))
  return Object.entries(value).flatMap(([key, item]) => findLocalPathStrings(item, `${location}.${key}`))
}

function localPathPattern() {
  return /(?:^|[\s"'])(?:(?:file:\/\/(?:localhost)?)?(?:\/Users\/[^\s"']+|\/home\/[^\s"']+|\/private\/(?:tmp|var)\/[^\s"']+|\/var\/folders\/[^\s"']+|\/tmp\/[^\s"']+|\/Volumes\/[^\s"']+)|(?:file:\/\/\/)?[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s"']+)/i
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (error) {
    fail(relative(process.cwd(), path), `failed to read JSON: ${error.message}`)
    return null
  }
}

async function readText(path) {
  try {
    return await readFile(path, 'utf8')
  } catch (error) {
    fail(path, `failed to read text: ${error.message}`)
    return ''
  }
}

async function exists(path) {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

function fail(bundleRel, message) {
  failures.push(`${bundleRel}: ${message}`)
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    if (inline !== undefined) {
      parsed.set(key, inline)
      continue
    }
    if (!next || next.startsWith('--')) {
      parsed.set(key, 'true')
      continue
    }
    parsed.set(key, next)
    i += 1
  }
  return parsed
}
