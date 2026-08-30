#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { createServer as createViteServer } from 'vite'

import {
  applyWebResearchContext,
  boundWebResearchResult,
  canEnableNativeModelTools,
  classifyWebResearchNeed,
  deriveFittedWebResearchReplyBudget,
  deriveWebResearchPromptBudget,
  effectiveGenerationTokenLimit,
  estimateWebResearchChatTokens,
  extractPromptUrls,
  fitWebResearchContext,
  persistWebResearchEnabled,
  readWebResearchEnabled,
  requestWebResearch,
  webResearchMetadata,
} from '../src/lib/webResearch.js'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const helperServer = await createViteServer({
  root: resolve(scriptDir, '..'),
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})
let looksLikeCodePrompt
let activeRuntimeContextFit
try {
  ({ looksLikeCodePrompt, activeRuntimeContextFit } = await helperServer.ssrLoadModule('/src/hooks/useDashboardData.js'))
} finally {
  await helperServer.close()
}

const streamingDashboardHookSource = await readFile(
  resolve(scriptDir, '../src/hooks/useDashboardData.js'),
  'utf8',
)
assert.match(
  streamingDashboardHookSource,
  /const firstVisibleContent = !firstContentEmitted && Boolean\(fullContent\)[\s\S]*paceFirstVisiblePrefix\(pacer, fullContent, now\)[\s\S]*\{ immediate: firstVisibleContent \}/,
  'the first visible stream delta must bypass animation-frame throttling while later deltas remain batched',
)

assert.equal(
  looksLikeCodePrompt('Write a Python implementation plan with architecture and phases.'),
  false,
  'planning language must win before the runnable Python fast path',
)
assert.equal(
  looksLikeCodePrompt('Write complete runnable Python code for this parser.'),
  true,
  'an actual runnable Python request must still receive code-first policy',
)
const overfullActiveContext = activeRuntimeContextFit(
  [{ role: 'user', content: 'overfull history' }],
  { activeContextLength: 4_096, estimateTokenCount: () => 4_096 },
)
assert.equal(overfullActiveContext.status, 'unfit')
assert.equal(overfullActiveContext.unfit, true, 'a rendered prompt filling a 4K active context must be explicitly unfit')
assert.equal(overfullActiveContext.replyRoom, 0)
assert.match(overfullActiveContext.message, /active 4,096-token runtime context/)
const normalActiveContext = activeRuntimeContextFit(
  [{ role: 'user', content: 'normal prompt' }],
  { activeContextLength: 4_096, estimateTokenCount: () => 256 },
)
assert.equal(normalActiveContext.status, 'fit')
assert.ok(normalActiveContext.replyRoom > 0, 'a normal prompt must retain reply room in the active context')
const serverPromptLimitOverflow = activeRuntimeContextFit(
  [{ role: 'user', content: 'prompt above the server admission ceiling' }],
  {
    activeContextLength: 8_192,
    maxPromptTokens: 4_096,
    estimateTokenCount: () => 5_000,
  },
)
assert.equal(serverPromptLimitOverflow.status, 'unfit')
assert.equal(serverPromptLimitOverflow.promptLimit, 4_096)
assert.match(serverPromptLimitOverflow.message, /server's 4,096-token prompt limit/)
assert.equal(
  activeRuntimeContextFit([{ role: 'user', content: 'unknown runtime' }]).status,
  'unknown',
  'an older runtime with no active-context field must preserve the safe backend-checked path',
)

const classifierFixture = JSON.parse(await readFile(
  resolve(scriptDir, '../../tests/fixtures/websearch/classifier_cases.json'),
  'utf8',
))

for (const testCase of classifierFixture) {
  const plan = classifyWebResearchNeed(testCase.prompt)
  assert.equal(plan.needed, testCase.needed, `shared classifier needed mismatch: ${testCase.prompt}`)
  assert.equal(Boolean(plan.query), testCase.has_query, `shared classifier query mismatch: ${testCase.prompt}`)
  assert.equal(plan.urls.length, testCase.url_count, `shared classifier URL mismatch: ${testCase.prompt}`)
}

const acceptancePrompt = `I want to update an app that I once developed. Build a multi-step Xcode task list.
1. connect to a bluetooth based scale for food measurement ([https://github.com/bburky/smartchef-web-bluetooth/](https://github.com/bburky/smartchef-web-bluetooth/) and [https://github.com/PanamaHitek/SmartScale)2](https://github.com/PanamaHitek/SmartScale\\)2). allow meal assembly mode.`

assert.deepEqual(extractPromptUrls(acceptancePrompt), [
  'https://github.com/bburky/smartchef-web-bluetooth/',
  'https://github.com/PanamaHitek/SmartScale',
], 'the acceptance prompt should recover and deduplicate both canonical GitHub repositories')

const linkedPlan = classifyWebResearchNeed(acceptancePrompt)
assert.equal(linkedPlan.needed, true)
assert.equal(linkedPlan.reason, 'linked_urls')
assert.equal(linkedPlan.urls.length, 2)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: true, modelArchitecture: 'gemma4', certifiedGemma4Tools: true }), false)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: false, modelArchitecture: 'qwen3', certifiedGemma4Tools: true }), false)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: false, modelArchitecture: 'gemma4', certifiedGemma4Tools: true }), true)

assert.deepEqual(extractPromptUrls('Read https://github.com/acme/widgets/tree/feature/web-research'), [
  'https://github.com/acme/widgets/tree/feature/web-research',
], 'GitHub branch refs and paths must survive browser URL normalization')
assert.deepEqual(extractPromptUrls('Read https://github.com/acme/widgets/tree/release)2'), [
  'https://github.com/acme/widgets/tree/release)2',
], 'a legal GitHub branch ref ending in )digits must not be paste-repaired')
assert.deepEqual(extractPromptUrls('Read https://github.com/acme/widgets/blob/main/spec)'), [
  'https://github.com/acme/widgets/blob/main/spec)',
], 'a legal GitHub filename ending in ) must not be mistaken for prose punctuation')
assert.deepEqual(extractPromptUrls('Read (https://github.com/acme/widgets/blob/main/spec))'), [
  'https://github.com/acme/widgets/blob/main/spec)',
], 'only the closing parenthesis proven by an adjacent wrapper may be removed')
assert.deepEqual(extractPromptUrls('Read http://[2606:4700:4700::1111]/dns and http://[2001:db8::1].'), [
  'http://[2606:4700:4700::1111]/dns',
  'http://[2001:db8::1]/',
], 'bracketed IPv6 URLs must reach backend validation intact')
assert.equal(
  classifyWebResearchNeed('Inspect https://[not-a-valid-host').needed,
  true,
  'an explicit HTTP(S) scheme must conservatively invoke the authoritative backend parser',
)
const mixedPlan = classifyWebResearchNeed('Read https://example.com/spec and also search the web for current alternatives.')
assert.equal(mixedPlan.reason, 'linked_urls_and_search')
assert.deepEqual(mixedPlan.urls, ['https://example.com/spec'])
assert.match(mixedPlan.query, /search the web/)
const lookUpSupplementalPlan = classifyWebResearchNeed('Read https://example.com/spec and look up competing libraries too.')
assert.equal(lookUpSupplementalPlan.reason, 'linked_urls_and_search')
assert.deepEqual(lookUpSupplementalPlan.urls, ['https://example.com/spec'])
assert.match(lookUpSupplementalPlan.query, /look up competing libraries/)
const linkedReadOnlyPlan = classifyWebResearchNeed('Read the linked docs https://example.com/spec')
assert.equal(linkedReadOnlyPlan.reason, 'linked_urls')
assert.equal(linkedReadOnlyPlan.query, null, 'reading an explicit link must not create a second search query')

for (const ordinary of [
  'Rewrite this paragraph more concisely.',
  'Turn this checklist into ordered steps.',
  'Today I went to the store.',
  'Use the current fitness setup from my draft.',
]) {
  assert.equal(classifyWebResearchNeed(ordinary).needed, false, `ordinary local prompt should not browse: ${ordinary}`)
}

const currentPlan = classifyWebResearchNeed('Search the web for the current Xcode release and cite sources.')
assert.equal(currentPlan.needed, true)
assert.equal(currentPlan.reason, 'explicit_search')
for (const publicWebCue of [
  'Search online for Xcode release notes.',
  'Use the internet to compare these libraries.',
  'Browse online for a current answer.',
  'What is the latest Xcode release?',
  'Show the most recent Xcode information.',
]) {
  assert.equal(classifyWebResearchNeed(publicWebCue).needed, true, `public-web egress must be disclosed before send: ${publicWebCue}`)
}
for (const explicitOptOut of [
  'Do not use the web; answer from the text I supplied.',
  'Don\'t consult the web; compare the current versions from my notes.',
  'Without accessing the internet, rewrite this paragraph.',
  'No web search: summarize this locally.',
  'Never read this URL https://example.com/private; rewrite the sentence only.',
  'Don’t follow the links https://example.com/private; use only my description.',
  'Do not use online sources; read https://example.com/private.',
  'Without using outside sources, compare the current versions in this draft.',
  'Don\'t consult external sites; compare the current versions in this draft.',
  'Stay offline and summarize https://example.com/private.',
  'Keep this task offline; find the latest release at https://example.com/private.',
  'Do not use the network; search online for the current price.',
  'No network access; search the web only if you think it would help.',
  'No external sources; compare the current versions from my notes.',
  'Answer with no outside sources; compare the latest releases from my notes.',
  'Do not connect to the internet; compare the current versions from my notes.',
  'Answer without external sources; compare the latest releases from my notes.',
]) {
  const plan = classifyWebResearchNeed(explicitOptOut)
  assert.equal(plan.needed, false, `an unambiguous Web Auto veto must win: ${explicitOptOut}`)
  assert.deepEqual(plan.urls, [])
  assert.equal(plan.query, null)
}
const linksOnlyVeto = classifyWebResearchNeed('Do not access https://example.com/private; search the web for public alternatives instead.')
assert.equal(linksOnlyVeto.needed, true, 'a link-specific veto must not suppress separately requested supplemental search')
assert.deepEqual(linksOnlyVeto.urls, [])
assert.match(linksOnlyVeto.query, /search the web/)
assert.equal(
  classifyWebResearchNeed('Do not use the web page marketing copy; search the web for independent reviews.').needed,
  true,
  'a veto of one page\'s content must not be mistaken for a global privacy veto',
)
const pageOnlyVeto = classifyWebResearchNeed('Do not use the web page at https://example.com/old; search the web for alternatives.')
assert.equal(pageOnlyVeto.needed, true)
assert.deepEqual(pageOnlyVeto.urls, [], 'a page-specific veto must suppress only that linked page')
assert.ok(pageOnlyVeto.query, 'a page-specific veto must preserve the supplemental search request')
const descriptiveNetworkState = classifyWebResearchNeed('Design the state shown when there is no network access; search the web for current browser APIs.')
assert.equal(descriptiveNetworkState.needed, true, 'describing a no-network UI state is not a privacy directive')
assert.ok(descriptiveNetworkState.query)
for (const descriptivePrivacyCopy of [
  'Design a banner that says no web access; search the web for current patterns.',
  'Write UI copy that says do not connect to the internet; search the web for current patterns.',
  'Explain the label no internet access; search online for current recovery guidance.',
  'Review the copy no external sources; search the web for current alternatives.',
]) {
  const plan = classifyWebResearchNeed(descriptivePrivacyCopy)
  assert.equal(plan.needed, true, `descriptive privacy copy must not become a user directive: ${descriptivePrivacyCopy}`)
  assert.ok(plan.query)
}
for (const pluralCurrentCue of [
  'Compare the current versions of React and Vue.',
  'Summarize the latest releases from Apple.',
]) {
  const plan = classifyWebResearchNeed(pluralCurrentCue)
  assert.equal(plan.needed, true, `plural current/latest wording must trigger research: ${pluralCurrentCue}`)
  assert.ok(plan.query)
}

const originalMessages = [{ role: 'user', content: acceptancePrompt }]
const enriched = applyWebResearchContext(originalMessages, {
  triggered: true,
  reason: 'linked_urls',
  sources: [
    { title: 'SmartChef Web Bluetooth', url: 'https://github.com/bburky/smartchef-web-bluetooth', excerpt: 'UNIQUE_GATT_MARKER' },
    { title: 'SmartScale', url: 'https://github.com/PanamaHitek/SmartScale', excerpt: 'UNIQUE_ADVERTISEMENT_MARKER' },
  ],
})
assert.equal(enriched[0].role, 'system', 'research evidence should be a leading system message')
assert.match(enriched[0].content, /UNTRUSTED EXTERNAL DATA/)
assert.match(enriched[0].content, /UNIQUE_GATT_MARKER/)
assert.match(enriched[0].content, /UNIQUE_ADVERTISEMENT_MARKER/)
assert.match(enriched[0].content, /https:\/\/github\.com\/bburky\/smartchef-web-bluetooth/)
assert.deepEqual(enriched.at(-1), originalMessages[0], 'the original user prompt must remain unchanged')

const bounded = applyWebResearchContext(originalMessages, {
  sources: [
    { title: 'Large A', url: 'https://example.com/a', excerpt: `${'A'.repeat(7_000)}TAIL_A` },
    { title: 'Large B', url: 'https://example.com/b', excerpt: `${'B'.repeat(7_000)}TAIL_B` },
  ],
})
assert.doesNotMatch(bounded[0].content, /TAIL_A|TAIL_B/, 'source excerpts must stay within the Gemma Web UI prompt envelope')
assert.ok(bounded[0].content.length < 12_000, 'combined web evidence must have a hard total character bound')

const balanced = boundWebResearchResult({
  sources: [
    { title: 'Repo one', url: 'https://example.com/one', excerpt: 'A'.repeat(7_000) },
    { title: 'Repo two', url: 'https://example.com/two', excerpt: `${'B'.repeat(3_900)}SECOND_REPO_FORMULA` },
  ],
})
assert.equal(balanced.sources[0].excerpt.length, 4_000)
assert.match(balanced.sources[1].excerpt, /SECOND_REPO_FORMULA/, 'a long first repository must not starve the second source of evidence')

const chunked = boundWebResearchResult({
  sources: [
    {
      title: 'Chunked repository',
      url: 'https://github.com/acme/widgets/tree/feature/web-research',
      chunks: [
        {
          path: 'docs/overview.md',
          url: 'https://github.com/acme/widgets/blob/feature/web-research/docs/overview.md',
          text: 'generic prose '.repeat(300),
        },
        {
          path: 'src/protocol.rs',
          url: 'https://github.com/acme/widgets/blob/feature/web-research/src/protocol.rs',
          text: `${'setup '.repeat(300)}QUERY_PROTOCOL_MARKER decoder`,
        },
      ],
    },
  ],
}, { maxExcerptChars: 900, queryText: 'protocol decoder' })
assert.equal(chunked.sources[0].url, 'https://github.com/acme/widgets/tree/feature/web-research')
assert.equal(chunked.sources[0].chunks[0].path, 'src/protocol.rs')
assert.equal(chunked.sources[0].chunks[0].url, 'https://github.com/acme/widgets/blob/feature/web-research/src/protocol.rs')
assert.match(chunked.sources[0].chunks[0].text, /QUERY_PROTOCOL_MARKER/, 'query-aware chunks must retain a late relevant section')

const minimumPerSource = boundWebResearchResult({
  sources: [1, 2, 3].map((id) => ({ title: `Source ${id}`, url: `https://example.com/${id}`, excerpt: String(id).repeat(2_000) })),
}, { maxExcerptChars: 768 })
assert.equal(minimumPerSource.sources.length, 3)
assert.ok(minimumPerSource.sources.every((source) => source.chunks[0].text.length >= 256), 'every retained source should receive the named minimum when room permits')
const insufficientMinimum = boundWebResearchResult({
  sources: [1, 2, 3].map((id) => ({ title: `Source ${id}`, url: `https://example.com/${id}`, excerpt: String(id).repeat(2_000) })),
}, { maxExcerptChars: 500 })
assert.equal(insufficientMinimum.sources.length, 1, 'insufficient room should omit whole trailing sources instead of listing empty evidence')
assert.match(insufficientMinimum.warnings.at(-1), /1 of 3/)
const belowMinimum = boundWebResearchResult({
  sources: [{ title: 'Too tight', url: 'https://example.com/tight', excerpt: 'x'.repeat(2_000) }],
}, { maxExcerptChars: 255 })
assert.equal(belowMinimum.sources.length, 0, 'a source must not be listed when it cannot receive the named minimum evidence space')

const relevantAmongDistractors = boundWebResearchResult({
  sources: [{
    title: 'Many chunks',
    url: 'https://example.com/many',
    chunks: [
      ...Array.from({ length: 8 }, (_, index) => ({ path: `docs/${index}.md`, text: 'distractor '.repeat(100) })),
      { path: 'src/decoder.rs', text: `${'setup '.repeat(100)}HIGH_VALUE_DECODER` },
    ],
  }],
}, { maxExcerptChars: 256, queryText: 'decoder' })
assert.match(relevantAmongDistractors.sources[0].chunks[0].text, /HIGH_VALUE_DECODER/, 'the best query chunk must retain meaningful space even with many distractors')

const fitted = fitWebResearchContext(originalMessages, {
  triggered: true,
  sources: [
    {
      title: `A title with JSON escapes ${'\\"'.repeat(300)}`,
      url: 'https://example.com/a-long-source-url',
      excerpt: `${'\\"'.repeat(5_000)}CRITICAL_TAIL`,
    },
  ],
}, {
  maxPromptTokens: 700,
  estimateTokenCount: (messages) => messages.reduce((total, message) => total + String(message.content || '').length, 0),
})
assert.ok(fitted.messages.reduce((total, message) => total + String(message.content || '').length, 0) <= 700, 'the complete injected system message must fit, including JSON escaping and metadata')
assert.ok(fitted.research.warnings.length > 0 || fitted.research.sources.length > 0)
const unboundedFit = fitWebResearchContext(originalMessages, {
  triggered: true,
  sources: [{ title: 'Usable', url: 'https://example.com/usable', excerpt: 'EVIDENCE_SURVIVES_WITHOUT_MODEL_CONTEXT_METADATA' }],
}, { maxPromptTokens: null, estimateTokenCount: () => Number.MAX_SAFE_INTEGER })
assert.match(unboundedFit.messages[0].content, /EVIDENCE_SURVIVES_WITHOUT_MODEL_CONTEXT_METADATA/, 'missing context metadata must not be treated as a zero-token prompt budget')

const tinyImagePrompt = [{
  role: 'user',
  content: [
    { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AA==' } },
    { type: 'text', text: 'Describe this image.' },
  ],
}]
const hugeImagePrompt = [{
  role: 'user',
  content: [
    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${'A'.repeat(1_000_000)}` } },
    { type: 'text', text: 'Describe this image.' },
  ],
}]
assert.equal(
  estimateWebResearchChatTokens(tinyImagePrompt, { visionTokenAllowance: 384 }),
  estimateWebResearchChatTokens(hugeImagePrompt, { visionTokenAllowance: 384 }),
  'base64 transport size must never be counted as prompt text',
)
assert.equal(
  estimateWebResearchChatTokens(hugeImagePrompt, { visionTokenAllowance: 768 })
    - estimateWebResearchChatTokens(hugeImagePrompt, { visionTokenAllowance: 384 }),
  384,
  'the bounded runtime vision-token allowance must control image context cost',
)
const cjkPrompt = [{ role: 'user', content: '最新版本と外部資料を比較してください。' }]
assert.ok(
  estimateWebResearchChatTokens(cjkPrompt) >= new TextEncoder().encode(cjkPrompt[0].content).length,
  'CJK input must not be estimated with an unsafe Latin-text character ratio',
)
const escapedExternalPrompt = applyWebResearchContext([{ role: 'user', content: 'Compare the evidence.' }], {
  triggered: true,
  sources: [{
    title: `Escaped ${'\\"'.repeat(40)}`,
    url: 'https://example.com/external',
    excerpt: `外部データ ${'\\"'.repeat(200)}`,
  }],
})
assert.ok(
  estimateWebResearchChatTokens(escapedExternalPrompt) >= new TextEncoder().encode(escapedExternalPrompt[0].content).length,
  'rendered untrusted JSON, escaping, and source text must be conservatively budgeted',
)

const estimatedTokens = (messages) => messages.reduce((total, message) => total + Math.ceil(String(message.content || '').length / 4) + 4, 3)
const budgetHistory = [
  { role: 'system', content: 'Configured system policy '.repeat(20) },
  { role: 'user', content: 'Earlier history '.repeat(80) },
  { role: 'assistant', content: 'Earlier response '.repeat(80) },
  { role: 'user', content: acceptancePrompt },
]
const runtimeBudget = deriveWebResearchPromptBudget({
  contextLength: 4_096,
  serverMaxPromptTokens: 4_096,
  serverMaxGenerationTokens: 8_192,
  requestedMaxTokens: 1_536,
  messages: budgetHistory,
  research: {
    triggered: true,
    sources: [{ title: 'Runtime evidence', url: 'https://example.com/runtime', excerpt: 'runtime evidence '.repeat(800) }],
  },
  estimateTokenCount: estimatedTokens,
  queryText: 'runtime evidence',
})
assert.ok(runtimeBudget.replyReserve > 0 && runtimeBudget.replyReserve <= 1_536, 'joint budgeting must retain useful reply room within the requested limit')
assert.ok(runtimeBudget.evidenceReserve > 0, 'joint budgeting must retain actual fetched evidence on a 4K context')
assert.equal(effectiveGenerationTokenLimit(9_000, 4_096), 4_096, 'the same server generation ceiling used for budgeting must cap the outgoing request')
const runtimeFit = fitWebResearchContext(budgetHistory, {
  triggered: true,
  sources: [{ title: `Escaped ${'\\"'.repeat(80)}`, url: 'https://example.com/runtime', excerpt: `${'evidence\\"'.repeat(1_500)}TAIL` }],
}, {
  maxPromptTokens: runtimeBudget.maxPromptTokens,
  estimateTokenCount: estimatedTokens,
  queryText: 'runtime evidence',
})
assert.ok(
  estimatedTokens(runtimeFit.messages) + runtimeBudget.replyReserve + runtimeBudget.safetyMargin <= runtimeBudget.contextLength,
  'history, policy, evidence JSON/template overhead, safety, and reply reserve must fit the active runtime context',
)
const droppedEvidenceHistory = [{ role: 'user', content: 'H'.repeat(3_800) }]
const renderedLength = (messages) => messages.reduce((total, message) => total + String(message?.content || '').length, 0)
const droppedEvidenceBudget = deriveWebResearchPromptBudget({
  contextLength: 4_096,
  serverMaxPromptTokens: 4_096,
  serverMaxGenerationTokens: 8_192,
  requestedMaxTokens: 8_192,
  messages: droppedEvidenceHistory,
  research: {
    triggered: true,
    sources: [{ title: 'Cannot fit minimum', url: 'https://example.com/tight', excerpt: 'evidence '.repeat(500) }],
  },
  estimateTokenCount: renderedLength,
})
const droppedEvidenceFit = fitWebResearchContext(droppedEvidenceHistory, {
  triggered: true,
  sources: [{ title: 'Cannot fit minimum', url: 'https://example.com/tight', excerpt: 'evidence '.repeat(500) }],
}, {
  maxPromptTokens: droppedEvidenceBudget.maxPromptTokens,
  estimateTokenCount: renderedLength,
})
assert.equal(droppedEvidenceFit.research.sources.length, 0, 'the controlled tight prompt should drop evidence that cannot receive its minimum space')
const reclaimedReply = deriveFittedWebResearchReplyBudget({
  contextLength: 4_096,
  serverMaxGenerationTokens: 8_192,
  requestedMaxTokens: 8_192,
  messages: droppedEvidenceFit.messages,
  estimateTokenCount: renderedLength,
  safetyMargin: droppedEvidenceBudget.safetyMargin,
})
assert.ok(
  deriveFittedWebResearchReplyBudget({
    contextLength: 4_096,
    requestedMaxTokens: 8_192,
    messages: droppedEvidenceFit.messages,
    estimateTokenCount: renderedLength,
  }).safetyMargin >= 16,
  'post-fit recomputation must retain a default safety margin when none is supplied',
)
assert.ok(
  reclaimedReply.replyReserve > droppedEvidenceBudget.replyReserve,
  'reply max_tokens must reclaim context when actual fitting drops provisional evidence',
)
assert.ok(
  reclaimedReply.promptTokens + reclaimedReply.replyReserve + reclaimedReply.safetyMargin <= reclaimedReply.contextLength,
  'the recomputed fitted prompt and reply must still fit jointly',
)
const largerRuntimeBudget = deriveWebResearchPromptBudget({
  contextLength: 32_768,
  requestedMaxTokens: 3_072,
  messages: budgetHistory,
  research: {
    triggered: true,
    sources: [{ title: 'Runtime evidence', url: 'https://example.com/runtime', excerpt: 'runtime evidence '.repeat(800) }],
  },
  estimateTokenCount: estimatedTokens,
  queryText: 'runtime evidence',
})
assert.ok(largerRuntimeBudget.replyReserve > runtimeBudget.replyReserve, 'reply room must grow with the active runtime context once the actual evidence fits')

const defaultLimitResearch = {
  triggered: true,
  sources: [{ title: 'Large evidence', url: 'https://example.com/large', excerpt: 'grounded evidence '.repeat(1_000) }],
}
const defaultLimitBudgets = [4_096, 8_192].map((contextLength) => deriveWebResearchPromptBudget({
  contextLength,
  serverMaxPromptTokens: contextLength,
  serverMaxGenerationTokens: 8_192,
  requestedMaxTokens: 8_192,
  messages: originalMessages,
  research: defaultLimitResearch,
  estimateTokenCount: estimatedTokens,
  queryText: 'grounded evidence',
}))
for (const budget of defaultLimitBudgets) {
  assert.ok(budget.evidenceReserve > 0, `${budget.contextLength}-token context must not starve fetched evidence under the default 8192 reply setting`)
  assert.ok(budget.replyReserve > 0, `${budget.contextLength}-token context must retain reply room`)
  assert.ok(
    budget.maxPromptTokens + budget.replyReserve + budget.safetyMargin <= budget.contextLength,
    `${budget.contextLength}-token joint budget must fit the active context`,
  )
}
assert.ok(defaultLimitBudgets[1].replyReserve > defaultLimitBudgets[0].replyReserve, 'reply room should grow with an 8K context instead of using a fixed answer cap')

const noSources = applyWebResearchContext(originalMessages, { triggered: true, sources: [] })
assert.equal(noSources, originalMessages, 'failed research must not inject stale or fabricated evidence')

const metadata = webResearchMetadata({
  triggered: true,
  reason: 'linked_urls',
  sources: [
    { title: 'Safe', url: 'https://example.com/source', excerpt: 'usable evidence' },
    { title: 'Unsafe', url: 'javascript:alert(1)', excerpt: 'must not display' },
  ],
  warnings: ['one linked page was unavailable'],
})
assert.deepEqual(metadata.sources, [{ title: 'Safe', url: 'https://example.com/source' }])
assert.deepEqual(metadata.warnings, ['one linked page was unavailable'])
assert.doesNotMatch(JSON.stringify(metadata), /usable evidence|excerpt|chunks|path/, 'persisted source metadata must never contain fetched text or chunk paths')
const fittedMetadata = webResearchMetadata(insufficientMinimum)
assert.equal(fittedMetadata.sources.length, insufficientMinimum.sources.length, 'persisted provenance must list only sources retained by prompt fitting')
const emptyEvidenceMetadata = webResearchMetadata({
  triggered: true,
  sources: [{ title: 'Not actually read', url: 'https://example.com/empty', excerpt: '' }],
  warnings: [],
})
assert.deepEqual(emptyEvidenceMetadata.sources, [], 'a URL with no injected excerpt must never be displayed as read provenance')
assert.match(emptyEvidenceMetadata.warnings[0], /no usable text excerpts/)
const exactChunkMetadata = webResearchMetadata(chunked)
assert.deepEqual(exactChunkMetadata.sources, [
  {
    title: 'Chunked repository — src/protocol.rs',
    url: 'https://github.com/acme/widgets/blob/feature/web-research/src/protocol.rs',
  },
  {
    title: 'Chunked repository — docs/overview.md',
    url: 'https://github.com/acme/widgets/blob/feature/web-research/docs/overview.md',
  },
], 'display provenance should prefer exact included chunk URLs over a repository root')
assert.doesNotMatch(JSON.stringify(exactChunkMetadata), /QUERY_PROTOCOL_MARKER|generic prose|text|chunks/, 'exact provenance must still persist no fetched text')
const fairMetadata = webResearchMetadata({
  triggered: true,
  sources: [
    {
      title: 'Many chunks',
      url: 'https://example.com/many',
      chunks: Array.from({ length: 6 }, (_, index) => ({
        path: `many-${index}.md`,
        url: `https://example.com/many/${index}`,
        text: `evidence ${index}`,
      })),
    },
    {
      title: 'Second source',
      url: 'https://example.org/second',
      chunks: [{ path: 'only.md', url: 'https://example.org/second/only', text: 'second evidence' }],
    },
  ],
  warnings: Array.from({ length: 6 }, (_, index) => `warning ${index + 1}`),
})
assert.ok(
  fairMetadata.sources.some((source) => source.url === 'https://example.org/second/only'),
  'bounded metadata must retain at least one exact provenance URL from every fitted source before extras',
)
assert.deepEqual(
  fairMetadata.warnings,
  Array.from({ length: 6 }, (_, index) => `warning ${index + 1}`),
  'every bounded backend warning must remain visible instead of being silently capped',
)

assert.equal(readWebResearchEnabled(), true, 'Web Auto should default on so the pasted acceptance prompt works without setup')

const stored = new Map()
globalThis.window = {
  localStorage: {
    getItem: (key) => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
    removeItem: (key) => stored.delete(key),
  },
}
persistWebResearchEnabled(false)
assert.equal(readWebResearchEnabled(), false, 'turning Web off must persist the zero-research choice')
persistWebResearchEnabled(true)
assert.equal(readWebResearchEnabled(), true, 'turning Web back on must persist Auto mode')
delete globalThis.window

const nativeFetch = globalThis.fetch
try {
  globalThis.fetch = async () => new Response('<html>proxy error</html>', { status: 200, headers: { 'content-type': 'text/html' } })
  await assert.rejects(
    () => requestWebResearch('http://127.0.0.1:8181', 'search the web'),
    /invalid response/,
    'a malformed 2xx response must not silently masquerade as a skipped lookup',
  )

  let requested = null
  globalThis.fetch = async (url, init) => {
    requested = { url, init }
    return new Response(JSON.stringify({ status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  }
  const skipped = await requestWebResearch('http://127.0.0.1:8181/', 'Rewrite this sentence.')
  assert.equal(skipped.triggered, false)
  assert.equal(requested, null, 'a definite local-only prompt must not call /api/web/research')

  globalThis.fetch = async (_url, init) => new Promise((resolve, reject) => {
    init.signal.addEventListener('abort', () => {
      const error = new Error('aborted')
      error.name = 'AbortError'
      reject(error)
    }, { once: true })
  })
  await assert.rejects(
    () => requestWebResearch('http://127.0.0.1:8181', 'Search the web', { timeoutMs: 5 }),
    /timed out/,
    'a stalled research helper must degrade back to local chat instead of hanging the composer',
  )

  let parentAbortReachedFetch = false
  globalThis.fetch = async (_url, init) => new Promise((resolve, reject) => {
    init.signal.addEventListener('abort', () => {
      parentAbortReachedFetch = true
      const error = new Error('aborted by caller')
      error.name = 'AbortError'
      reject(error)
    }, { once: true })
  })
  const parentController = new AbortController()
  const parentCancelled = requestWebResearch(
    'http://127.0.0.1:8181',
    'Search the web',
    { signal: parentController.signal, timeoutMs: 1_000 },
  )
  parentController.abort()
  await assert.rejects(parentCancelled, { name: 'AbortError' })
  assert.equal(parentAbortReachedFetch, true, 'a parent chat cancellation must abort the in-flight research request')
} finally {
  globalThis.fetch = nativeFetch
}

const dashboardHookSource = await readFile(resolve(scriptDir, '../src/hooks/useDashboardData.js'), 'utf8')
const chatWorkspaceSource = await readFile(resolve(scriptDir, '../src/views/ChatWorkspace.jsx'), 'utf8')
const messageTurnSource = await readFile(resolve(scriptDir, '../src/components/chat/MessageTurn.jsx'), 'utf8')
assert.match(dashboardHookSource, /if \(webResearchEnabled\s*&&\s*researchPlan\.needed\) \{[\s\S]*requestWebResearch/, 'definite local-only prompts must skip the research endpoint')
assert.match(dashboardHookSource, /fitWebResearchContext[\s\S]*deriveFittedWebResearchReplyBudget[\s\S]*fittedReplyBudget\.replyReserve <= 0[\s\S]*requestMaxTokens = Math\.floor\(fittedReplyBudget\.replyReserve\)/, 'the outgoing answer limit must be recomputed from the fitted prompt and zero room must block')
assert.doesNotMatch(dashboardHookSource, /Math\.max\(1, Math\.floor\(fittedReplyBudget\.replyReserve\)\)/, 'known context overflow must never be coerced into a max_tokens=1 request')
assert.match(dashboardHookSource, /activeContextLength:\s*runtime\?\.active_context_length/, 'send-time fitting must use the active runtime context rather than training metadata')
assert.match(dashboardHookSource, /visionTokenAllowance:\s*runtime\?\.vision_token_allowance/, 'vision context must use a runtime allowance instead of base64 transport length')
assert.doesNotMatch(dashboardHookSource, /\btools\s*:/, 'Gemma Web research must not add unsupported function tools to chat requests')
assert.doesNotMatch(dashboardHookSource, /\btool_choice\s*:/, 'Web Auto must remain separate from native model tool selection')
assert.doesNotMatch(dashboardHookSource, /\bcamelid_tools\s*:/, 'Web Auto must not enter a camelid_tools loop')
assert.match(dashboardHookSource, /web_research_ms: webResearchMs/, 'public-web latency must be persisted separately from model TTFT and decode rate')
assert.match(dashboardHookSource, /now - firstTokenAt/, 'decode tok\/s must start at the first generated token instead of including research or TTFT')
assert.match(dashboardHookSource, /responseIsStreaming \? tokensPerSecond\(decodedTokenCount, decodeElapsedMs\) : null/, 'non-streamed or replay-shaped answers must not report browser-invented decode throughput')
assert.match(chatWorkspaceSource, /Web Auto will send linked URLs or a search query to the public web/, 'a triggering draft must disclose public-web egress before send, including on touch devices')
assert.match(messageTurnSource, /warnings\.map/, 'failed or partial research should retain the backend\'s actionable warnings in the reply')

console.log('Web research smoke passed')
