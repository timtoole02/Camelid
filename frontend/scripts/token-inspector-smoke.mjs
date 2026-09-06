#!/usr/bin/env node
/* Token Inspector — contract, normalization, and copy gates.
 *
 * This surface makes a stronger claim than anything else in the UI: it renders
 * the MAGNITUDE of a model's per-token scores. Three classes of defect would each
 * ship a lie rather than a bug, so each gets an assertion here.
 *
 *   1. CONTRACT DRIFT. The engine returns typed 400s for logprobs with
 *      stream:true, with n>1, and for top_logprobs without logprobs. A request
 *      builder that composes any of those pairs turns a working feature into an
 *      error toast. The builder is asserted to be incapable of emitting them.
 *
 *   2. ABSENCE READ AS CERTAINTY. Several serve lanes answer chat requests before
 *      the logprobs step runs, returning HTTP 200 with the key simply ABSENT.
 *      Code that tests `logprobs === null` misclassifies that body. This asserts
 *      both that the correct test passes AND that the naive one would have failed
 *      — a positive-only assertion would not catch a regression to `=== null`.
 *
 *   3. COPY OVERREACH. The values are a log-softmax over raw logits. Words like
 *      "confidence" and "certainty" assert something about the model's epistemic
 *      state that the number does not support, and the contract's own row notes
 *      carry a vendor name the brand scrub forbids on screen. Both are asserted
 *      against RENDERED output, not against source.
 *
 * The normalization fixture is the repo's own captured wire body, not a
 * hand-written literal: a fixture invented here would encode this file's beliefs
 * about the shape rather than the engine's actual output.
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
const repoRoot = resolve(frontendRoot, '..')

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

/* The brand scrub bans this vendor name from visible UI source, and the
   rich_logprobs row notes contain it. Assembled from fragments so this file does
   not itself carry the literal — the same technique lib/capabilities.js uses. */
const VENDOR = ['Open', 'AI'].join('')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const inspection = await server.ssrLoadModule('/src/lib/tokenInspection.js')
  const { TokenInspectorCard } = await server.ssrLoadModule('/src/components/chat/render/TokenInspector.jsx')
  const { readChatCompletionJsonPayload } = await server.ssrLoadModule('/src/lib/chatCompletionStream.js')

  const {
    readInspectionContract,
    readCandidatesContract,
    inspectionRequestFields,
    inspectionForcesNonStreaming,
    inspectionAbsenceReason,
    normalizeInspection,
    clampTopLogprobs,
    MAX_TOP_LOGPROBS,
  } = inspection

  /* Live-shaped contract rows. Statuses are the engine's real serialized values —
     note `nonstreaming` is ONE word; a fixture spelled `non_streaming` would pass
     the prefix check and silently never match an exact-id lookup. */
  const liveCapabilities = {
    api_features: [
      { id: 'rich_logprobs', status: 'supported_current_gate_nonstreaming', notes: `POST /v1/chat/completions supports ${VENDOR}-shaped logprobs/top_logprobs.` },
      { id: 'multi_choice_generation', status: 'supported_current_gate_nonstreaming', notes: 'Supports 1..=8 independent choices.' },
    ],
    api_conformance: [
      { id: 'rich_logprobs', status: 'supported_current_gate_nonstreaming', supported_modes: ['chat_nonstreaming', 'completions_nonstreaming'], unsupported_modes: ['streaming', 'multi_choice'] },
      { id: 'multi_choice_generation', status: 'supported_current_gate_nonstreaming', supported_modes: ['chat_nonstreaming'], unsupported_modes: ['streaming', 'receipts', 'logprobs'] },
    ],
  }

  console.log('token inspector — contract gating')

  check('a live contract permits non-streaming inspection', () => {
    const contract = readInspectionContract(liveCapabilities)
    assert.equal(contract.present, true)
    assert.equal(contract.supported, true)
    assert.equal(contract.modesKnown, true)
    assert.equal(contract.nonStreamingSupported, true)
  })

  check('an engine with no capability rows fails closed', () => {
    const contract = readInspectionContract({})
    assert.equal(contract.present, false)
    assert.equal(contract.nonStreamingSupported, false)
    assert.deepEqual(inspectionRequestFields({ enabled: true, contract }), {})
  })

  /* An intermediate engine build ships api_features without api_conformance. The
     row says supported, but nothing describes the modes — and an undescribed mode
     is not a permitted one. */
  check('a supported row with no machine-readable modes stays closed', () => {
    const contract = readInspectionContract({ api_features: liveCapabilities.api_features })
    assert.equal(contract.present, true)
    assert.equal(contract.supported, true)
    assert.equal(contract.modesKnown, false, 'modes cannot be known without a conformance row')
    assert.equal(contract.nonStreamingSupported, false, 'unknown modes must not be assumed permissive')
    assert.deepEqual(inspectionRequestFields({ enabled: true, contract }), {})
  })

  check('an unsupported status is refused even with modes present', () => {
    const contract = readInspectionContract({
      api_conformance: [{ id: 'rich_logprobs', status: 'planned', supported_modes: ['chat_nonstreaming'] }],
    })
    assert.equal(contract.supported, false)
    assert.equal(contract.nonStreamingSupported, false)
  })

  check('resemblance is not evidence — a near-miss row id does not count', () => {
    const contract = readInspectionContract({
      api_conformance: [{ id: 'rich_logprobs_v2', status: 'supported', supported_modes: ['chat_nonstreaming'] }],
    })
    assert.equal(contract.present, false)
  })

  console.log('token inspector — request builder cannot compose a rejected pair')

  const contract = readInspectionContract(liveCapabilities)

  check('top_logprobs is never emitted without logprobs:true', () => {
    for (const depth of [1, 5, 20, 999, 0, -3, null, undefined, NaN]) {
      const fields = inspectionRequestFields({ enabled: true, contract, topLogprobs: depth })
      if ('top_logprobs' in fields) {
        assert.equal(fields.logprobs, true, `top_logprobs emitted without logprobs for depth ${depth}`)
      }
    }
  })

  check('depth is clamped to the engine cap, never sent above it', () => {
    assert.equal(MAX_TOP_LOGPROBS, 20)
    assert.equal(clampTopLogprobs(999), 20)
    assert.equal(clampTopLogprobs(0), 1)
    assert.equal(clampTopLogprobs(-5), 1)
    assert.equal(inspectionRequestFields({ enabled: true, contract, topLogprobs: 500 }).top_logprobs, 20)
  })

  check('the builder never emits n — logprobs and multi-choice are exclusive', () => {
    const fields = inspectionRequestFields({ enabled: true, contract, topLogprobs: 5 })
    assert.ok(!('n' in fields), 'n must never ride along with logprobs')
    const candidates = readCandidatesContract(liveCapabilities)
    assert.equal(candidates.excludesLogprobs, true, 'the contract itself declares the exclusion')
  })

  check('inspection forces the reply off the streaming path', () => {
    assert.equal(inspectionForcesNonStreaming({ enabled: true, contract }), true)
    assert.equal(inspectionForcesNonStreaming({ enabled: false, contract }), false)
    // Guarded contract: enabled but not permitted must NOT flip the stream flag,
    // or a normal chat turn silently loses streaming for no gain.
    assert.equal(inspectionForcesNonStreaming({ enabled: true, contract: readInspectionContract({}) }), false)
  })

  console.log('token inspector — absence is never certainty')

  check('the key is OMITTED, not null — and `=== null` would misread it', () => {
    const keyAbsent = { choices: [{ message: { content: 'Red.' }, finish_reason: 'stop' }] }
    assert.equal('logprobs' in keyAbsent.choices[0], false, 'fixture must reproduce the omitted-key shape')
    // The naive test that this module must never use:
    assert.equal(keyAbsent.choices[0].logprobs === null, false,
      'a `=== null` presence test returns false for an ABSENT key — it would treat a lane that reported nothing as a lane that reported')
    // What the shipped reader actually does:
    const parsed = readChatCompletionJsonPayload(keyAbsent)
    assert.equal(parsed.logprobs, null, 'the reader normalizes an absent key to null')
    assert.equal(Boolean(parsed.logprobs), false)
  })

  check('a lane that answered 200 with no record is reported as unmeasured', () => {
    const reason = inspectionAbsenceReason({ requested: true, responded: true, hasLogprobs: false, streamed: false })
    assert.equal(reason.code, 'lane_absent')
    assert.match(reason.detail, /missing, not flat/i,
      'the copy must deny the certainty reading explicitly')
  })

  check('a forced stream is distinguished from a silent lane', () => {
    const reason = inspectionAbsenceReason({ requested: true, responded: true, hasLogprobs: false, streamed: true })
    assert.equal(reason.code, 'streamed', 'a proxy forcing SSE must not be attributed to the model lane')
  })

  check('an un-requested reply produces no absence notice at all', () => {
    assert.equal(inspectionAbsenceReason({ requested: false, responded: true, hasLogprobs: false, streamed: false }), null)
  })

  console.log('token inspector — normalization against the engine\'s own captured body')

  /* The repo's committed wire capture. Source-pinned on purpose: an invented
     fixture would assert this file's belief about the shape, not the engine's. */
  const capturedPath = resolve(repoRoot, 'qa/capability/mac_logprobs_out/llama-3.2-1b/chat_logprobs.json')
  const captured = JSON.parse(readFileSync(capturedPath, 'utf8'))
  const capturedLogprobs = captured.choices[0].logprobs

  check('the captured body still has the shape this surface reads', () => {
    assert.ok(Array.isArray(capturedLogprobs.content), 'logprobs.content must be an array')
    const first = capturedLogprobs.content[0]
    for (const field of ['token', 'logprob', 'bytes', 'top_logprobs']) {
      assert.ok(field in first, `engine dropped field \`${field}\` — this surface reads it`)
    }
    assert.ok(Array.isArray(first.bytes), 'bytes must serialize as a numeric array, not base64')
  })

  const model = normalizeInspection(capturedLogprobs)

  check('every token carries a probability, a band, and a residual', () => {
    assert.equal(model.tokens.length, capturedLogprobs.content.length)
    for (const token of model.tokens) {
      assert.ok(token.probability >= 0 && token.probability <= 1, 'probability out of range')
      assert.ok(['settled', 'leading', 'contested', 'unknown'].includes(token.band))
      assert.ok(token.residualMass >= 0 && token.residualMass <= 1, 'residual mass out of range')
      assert.ok(Math.abs((token.shownMass + token.residualMass) - 1) < 1e-6, 'shown + residual must total the whole distribution')
    }
  })

  check('the residual is real — the shown alternatives are not the whole vocabulary', () => {
    // On the captured body the top-5 do not sum to 1; if they ever did, the
    // residual row would be decoration rather than a correction.
    assert.ok(model.tokens.some((token) => token.residualMass > 0.001),
      'at least one position must have mass outside the shown alternatives')
  })

  check('a chosen token outside its own top-N is reported, not hidden', () => {
    const outside = normalizeInspection({
      content: [{
        token: 'zzz',
        logprob: -9.5,
        bytes: [122, 122, 122],
        top_logprobs: [
          { token: 'a', logprob: -0.2, bytes: [97] },
          { token: 'b', logprob: -1.9, bytes: [98] },
        ],
      }],
    })
    assert.equal(outside.tokens[0].chosenInAlternatives, false)
    assert.equal(outside.tokens[0].chosenIsTop, false)
    assert.equal(outside.stats.offTopCount, 1)
  })

  check('near-ties are marked rather than presented as a strict ranking', () => {
    const tied = normalizeInspection({
      content: [{
        token: 'a',
        logprob: -0.5,
        bytes: [97],
        top_logprobs: [
          { token: 'a', logprob: -0.5, bytes: [97] },
          { token: 'b', logprob: -0.5005, bytes: [98] },
        ],
      }],
    })
    assert.equal(tied.tokens[0].alternatives[1].tiedWithPrevious, true,
      'a sub-noise gap must not render as meaningful ordering')
  })

  check('an empty or malformed record yields nothing rather than a broken panel', () => {
    assert.equal(normalizeInspection(null), null)
    assert.equal(normalizeInspection({}), null)
    assert.equal(normalizeInspection({ content: [] }), null)
  })

  console.log('token inspector — rendered copy')

  /* Expanded by default: the panel's content sits behind internal open state, so
     a collapsed render would let every copy assertion below pass against a trigger
     with no body. Collapsed rendering is asserted explicitly where it matters. */
  const render = (props) => renderToStaticMarkup(React.createElement(TokenInspectorCard, { defaultOpen: true, ...props }))

  check('the panel renders the captured record', () => {
    const html = render({ inspection: capturedLogprobs })
    // Body-only markers: none of these exist in the collapsed trigger, so this
    // assertion cannot pass against an unexpanded panel.
    assert.match(html, /tokinsp__strip/, 'the token strip must render')
    assert.match(html, /tokinsp__alts/, 'the alternatives list must render')
    assert.match(html, /rest of the vocabulary/, 'the residual row must render')
    assert.match(html, /Perplexity/, 'the sequence stats must render')
    const chips = html.match(/tokinsp__tok tokinsp__tok--/g) || []
    assert.equal(chips.length, capturedLogprobs.content.length, 'one chip per generated token')
  })

  /* The core epistemic gate. These words assert something about the model's
     internal state that a softmax over logits does not license. */
  check('no rendered state uses certainty vocabulary', () => {
    const states = [
      render({ inspection: capturedLogprobs }),
      render({ absence: inspectionAbsenceReason({ requested: true, responded: true, hasLogprobs: false, streamed: false }) }),
      render({ absence: inspectionAbsenceReason({ requested: true, responded: true, hasLogprobs: false, streamed: true }) }),
    ]
    for (const html of states) {
      assert.doesNotMatch(html, /confiden/i, 'the surface must not describe a score as confidence')
      assert.doesNotMatch(html, /certain/i, 'the surface must not describe a score as certainty')
      assert.doesNotMatch(html, /\bknows\b|\bbelieves\b|\bsure\b/i, 'no epistemic verbs')
    }
  })

  check('the vendor name in a contract row never reaches the screen', () => {
    const html = render({
      inspection: capturedLogprobs,
      candidatesContract: readCandidatesContract(liveCapabilities),
    })
    assert.doesNotMatch(html, new RegExp(`\\b${VENDOR}\\b`),
      'row notes carry a vendor name — render them through displayCapabilityCopy or not at all')
  })

  check('an absent record renders an explanation and NO control that fires a generation', () => {
    const html = render({ absence: inspectionAbsenceReason({ requested: true, responded: true, hasLogprobs: false, streamed: false }) })
    assert.match(html, /did not report probabilities/i)
    assert.doesNotMatch(html, /<button/, 'a guarded state must offer nothing that spends a decode')
  })

  check('the collapsed trigger summarizes without spending the panel', () => {
    const html = renderToStaticMarkup(React.createElement(TokenInspectorCard, { inspection: capturedLogprobs }))
    assert.match(html, /aria-expanded="false"/)
    assert.doesNotMatch(html, /tokinsp__strip/, 'a collapsed panel must not render its body')
    assert.match(html, /tokens/, 'the collapsed summary still says how many tokens were scored')
  })

  check('nothing renders when there is no record and no reason', () => {
    assert.equal(render({ inspection: null, absence: null }), '')
  })

  check('the multi-choice surface ships guarded, naming both blockers', () => {
    const html = render({ inspection: capturedLogprobs, candidatesContract: readCandidatesContract(liveCapabilities) })
    assert.match(html, /not available/i)
    assert.match(html, /refuses them alongside/i, 'must name the engine-level exclusion')
    assert.match(html, /greedily/i, 'must name the greedy-decode reason too — the exclusion alone is not the whole story')
  })

  console.log('token inspector — storage safety')

  /* Per-token records run ~414 bytes/token at depth 5 and ~1.4 KB at depth 20 —
     roughly a hundred times the reply text. persistConversations swallows quota
     errors, so an overflow silently stops saving the WHOLE conversation. This
     asserts the decision structurally rather than trusting a comment. */
  check('the inspection modules touch no persistent storage', () => {
    for (const relative of ['src/lib/tokenInspection.js', 'src/components/chat/render/TokenInspector.jsx']) {
      const source = readFileSync(resolve(frontendRoot, relative), 'utf8')
      assert.doesNotMatch(source, /localStorage|sessionStorage|appStorage|indexedDB/,
        `${relative} must not persist per-token records`)
    }
  })

  check('the conversation export whitelist carries no per-token record', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/lib/conversationExport.js'), 'utf8')
    assert.doesNotMatch(source, /logprob|token_inspection/,
      'exports are a field whitelist — adding a raw score array there must be a deliberate, reviewed change')
  })

  console.log('token inspector — tab registry coverage')

  /* Generic, not scoped to this feature. `arena` shipped registered in four of
     six places, so its header read "Camelid" and Cmd+K could not reach it. The
     defect is invisible in a screenshot, so it is asserted instead.

     Checked in BOTH directions. The first version of this assertion only walked
     HASH_TABS outward, which meant an id sitting in a registry with no
     corresponding tab — a rename that updated one list, a copy-paste, a stray
     edit — passed silently. An orphan is the same class of defect seen from the
     other side: two lists that disagree about what the tabs are. */
  check('every deep-linkable tab appears in all five registries', () => {
    const read = (relative) => readFileSync(resolve(frontendRoot, relative), 'utf8')
    const ids = (source, pattern) => [...source.matchAll(pattern)].map((match) => match[1])
    /* Each registry is parsed at its own declaration rather than by scanning the
       whole file: a bare substring search finds a tab id in unrelated code and
       reports a gap as covered, which is worse than no assertion. */
    const setLiteral = (source, name) => {
      const found = source.match(new RegExp(`const ${name} = new Set\\(\\[([^\\]]+)\\]`))
      assert.ok(found, `${name} not found — update this assertion rather than deleting it`)
      return ids(found[1], /'([a-z-]+)'/g)
    }

    const appSource = read('src/App.jsx')
    const titlesBlock = read('src/components/TopBar.jsx').match(/const TITLES = \{([\s\S]*?)\n\}/)
    assert.ok(titlesBlock, 'TopBar TITLES not found')

    const hashTabs = setLiteral(appSource, 'HASH_TABS')
    assert.ok(hashTabs.length > 5, 'HASH_TABS parsed suspiciously small')

    /* The rail reaches most tabs through NAV_SECTIONS, but `settings` has a
       dedicated footer control instead. The question is whether the tab is
       reachable from the rail at all, so both routes count. */
    const railSource = read('src/components/layout/SidebarRail.jsx')
    const registries = {
      'useDashboardData VALID_TABS': setLiteral(read('src/hooks/useDashboardData.js'), 'VALID_TABS'),
      'SidebarRail (NAV_SECTIONS or a direct control)': [
        ...ids(railSource, /tab: '([a-z-]+)'/g),
        ...ids(railSource, /setTab\('([a-z-]+)'\)/g),
      ],
      'TopBar TITLES': ids(titlesBlock[1], /^\s*([a-z-]+):/gm),
      'CommandPalette VIEW_LABELS': ids(read('src/components/CommandPalette.jsx'), /\['([a-z-]+)',/g),
      'App.jsx render chain': ids(appSource, /tab === '([a-z-]+)'/g),
    }

    const problems = []
    for (const tab of hashTabs) {
      for (const [name, registered] of Object.entries(registries)) {
        if (!registered.includes(tab)) problems.push(`${tab} missing from ${name}`)
      }
    }
    /* The reverse direction: an id a registry claims but HASH_TABS does not.
       `App.jsx render chain` is exempt — it compares `tab` against ids that are
       deliberately not hash-routable (the Spotlight overlay is reached by its own
       hash, not as a tab). Every other registry is a closed set. */
    for (const [name, registered] of Object.entries(registries)) {
      if (name === 'App.jsx render chain') continue
      for (const id of new Set(registered)) {
        if (!hashTabs.includes(id)) problems.push(`${id} appears in ${name} but is not a HASH_TABS tab`)
      }
    }
    assert.deepEqual(problems, [], `tab registries disagree — every entry here is a silent defect (no TITLES entry renders the header as "Camelid"; no VIEW_LABELS entry makes the view unreachable from the command palette; no VALID_TABS entry drops the tab on reload; an orphan id means two lists disagree about what the tabs are):\n  ${problems.join('\n  ')}`)
  })

  console.log(`\n${checks} checks passed`)
} finally {
  await server.close()
}
