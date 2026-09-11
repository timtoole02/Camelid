#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { readExactTargetVerifiedRender, readExactTargetVerifiedSegmentedRender, authoritativeOutputRate, readMtp12NativeReceiptQualification } from '../src/lib/nativeGenerationMetrics.js'
import { buildGemma4Mtp12ResearchSegments } from '../src/lib/segmentedWebResearchSynthesis.js'
import {
  GEMMA4_MTP12_EXACT_ROW_ID,
  GEMMA4_MTP12_TARGET_VERIFIED_VIDEO_OPT_IN_KEY,
  shouldUseGemma4Mtp12TargetVerifiedRender,
} from '../src/lib/targetVerifiedRender.js'

const runtime = {
  backend: 'gemma4-runtime',
  gemma4_serve_lane: 'mtp12_metal',
  active_model_id: 'gemma-4-12b-it-qat-q4_0.gguf',
}
const grounded = {
  sources: [
    { title: 'SmartChef', url: 'https://github.com/bburky/smartchef-web-bluetooth/' },
    { title: 'SmartScale', url: 'https://github.com/PanamaHitek/SmartScale' },
  ],
}
const enabled = {
  runtime,
  requestModelId: runtime.active_model_id,
  compatibilityRowId: GEMMA4_MTP12_EXACT_ROW_ID,
  research: grounded,
  receiptMode: false,
  videoRigOptIn: true,
}
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender(enabled), true)
assert.equal(
  shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, videoRigOptIn: false }),
  false,
  'ordinary MTP12 Web Auto chats must remain one-pass unless the video rig explicitly opts in',
)
assert.equal(GEMMA4_MTP12_TARGET_VERIFIED_VIDEO_OPT_IN_KEY, 'camelid.video.gemma4Mtp12TargetVerifiedRender')
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, receiptMode: true }), false)
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, research: { sources: grounded.sources.slice(0, 1) } }), false)
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, compatibilityRowId: 'gemma4_12b_it_q8_0' }), false)
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, runtime: { ...runtime, gemma4_serve_lane: 'ghost_moe' } }), false)
assert.equal(shouldUseGemma4Mtp12TargetVerifiedRender({ ...enabled, requestModelId: 'neighbor.gguf' }), false)

const exactRender = {
  target_verified_render: {
    token_ids_exact: true,
    verified_tokens: 700,
    render_tokens_per_second: 52.125,
  },
}
assert.equal(readExactTargetVerifiedRender(exactRender)?.verified_tokens, 700)
assert.equal(authoritativeOutputRate({ camelid: exactRender, tokens_out_per_sec: 999 }), 52.125)
assert.equal(readExactTargetVerifiedRender({ target_verified_render: { ...exactRender.target_verified_render, token_ids_exact: false } }), null)
assert.equal(readExactTargetVerifiedRender({ target_verified_render: { ...exactRender.target_verified_render, render_tokens_per_second: 0 } }), null)

const segmentedRender = {
  target_verified_segmented_render: {
    mode: 'prepared_web_research_segmented_target_verify',
    segment_count: 2,
    segments_exact: true,
    total_prompt_tokens: 400,
    requested_tokens: 200,
    verified_tokens: 200,
    decode_output_tokens: 198,
    decode_us: 4_000_000,
    render_tokens_per_second: 49.5,
    qualification_envelope_max_positions: 512,
    target_model_sha256: '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b',
    assistant_model_sha256: '67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6',
    segments: [0, 1].map((index) => ({
      index,
      prompt_tokens: 200,
      requested_tokens: 100,
      verified_tokens: 100,
      token_ids_exact: true,
      decode_output_tokens: 99,
      decode_us: 2_000_000,
      render_tokens_per_second: 49.5,
    })),
  },
}
assert.equal(readExactTargetVerifiedSegmentedRender(segmentedRender)?.render_tokens_per_second, 49.5)
assert.equal(authoritativeOutputRate({ camelid: segmentedRender, tokens_out_per_sec: 999 }), 49.5)
assert.equal(readExactTargetVerifiedSegmentedRender({ target_verified_segmented_render: { ...segmentedRender.target_verified_segmented_render, segments_exact: false } }), null)

const researchSegments = buildGemma4Mtp12ResearchSegments({
  sources: [
    { title: 'SmartChef', url: 'https://github.com/bburky/smartchef-web-bluetooth', chunks: [{ text: 'Web Bluetooth BLE scale implementation.' }] },
    { title: 'SmartScale', url: 'https://github.com/PanamaHitek/SmartScale', chunks: [{ text: 'Bluetooth scale reader and environment notes.' }] },
  ],
})
assert.equal(researchSegments.length, 6)
assert.ok(researchSegments.every((segment) => segment.maxTokens === 144 && segment.messages[0].content.length < 1600))
assert.equal(buildGemma4Mtp12ResearchSegments({ sources: researchSegments.slice(0, 1) }).length, 0)

const qualification = {
  workload: 'short_context_lossless_mtp_qualification',
  primary_decode_tokens_per_second: 51.493947835,
  confirmation_decode_tokens_per_second: 51.304677961,
  mean_decode_tokens_per_second: 51.399,
  prompt_tokens: 14,
  output_tokens: 96,
  max_positions: 512,
  selector: 'CAMELID_GEMMA4_MTP_W16_ONESHOT_W8_PAD16',
  target_sha256: '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b',
  assistant_sha256: '67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6',
}
assert.equal(readMtp12NativeReceiptQualification({ mtp12: { native_receipt_qualification: qualification } })?.primary_decode_tokens_per_second, 51.493947835)
assert.equal(readMtp12NativeReceiptQualification({ mtp12: { native_receipt_qualification: { ...qualification, output_tokens: 700 } } }), null, 'current-answer scope must not masquerade as the 96-token qualification')
assert.equal(readMtp12NativeReceiptQualification({ mtp12: { native_receipt_qualification: { ...qualification, target_sha256: 'wrong' } } }), null, 'qualification must remain hash-pinned')
assert.equal(readMtp12NativeReceiptQualification({ mtp12: { native_receipt_qualification: { ...qualification, primary_decode_tokens_per_second: 99 } } }), null, 'current-answer measurements must not replace the pinned qualification rate')

const hookSource = await readFile(new URL('../src/hooks/useDashboardData.js', import.meta.url), 'utf8')
assert.match(hookSource, /messages: segmentPlan\.messages,[\s\S]*max_tokens: segmentPlan\.maxTokens,[\s\S]*stream: false/, 'each planner must be a fresh non-stream request over its exact bounded messages')
assert.match(hookSource, /camelid_target_verified_render_draft_token_ids: targetVerifiedDraftTokenIds/, 'visible request must pass authoritative draft token IDs')
assert.match(hookSource, /: targetVerifiedDraftTokenIds\.length/, 'monolithic visible render bound must equal the authoritative draft length')
assert.match(hookSource, /targetVerifiedSegments\.reduce\(\(sum, segment\) => sum \+ segment\.token_ids\.length, 0\)/, 'segmented visible render bound must equal the sum of authoritative section drafts')
assert.match(hookSource, /\(targetVerifiedSegmentedDiagnostics \|\| targetVerifiedRenderDiagnostics\)\.render_tokens_per_second/, 'two-pass final rate must use the backend target-verification clock')
assert.match(hookSource, /const liveTps = responseIsStreaming && decodedTokens >= 4 && decodeElapsedMs >= 200\s*\n\s*\? tokensPerSecond\(decodedTokens, decodeElapsedMs\)/, 'two-pass live UI must expose the real browser-observed stream delivery rate after a stable measurement window')
assert.match(hookSource, /videoRigOptIn: isGemma4Mtp12TargetVerifiedVideoOptedIn\(\)/, 'the frontend gate must read the narrow video-only localStorage opt-in')
assert.match(hookSource, /camelid_target_verified_render_segments: targetVerifiedSegments/)
assert.match(hookSource, /camelid_expected_gguf_sha256: '93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b'/)
assert.match(hookSource, /prepared_web_research_multi_pass_lossless/)

const diagnosticsSource = await readFile(new URL('../src/components/chat/render/Diagnostics.jsx', import.meta.url), 'utf8')
assert.match(diagnosticsSource, /<span className="card-label">Decode Rate<\/span>/)
assert.match(diagnosticsSource, /data-native-rate=\{message\.tokens_out_per_sec/)
assert.doesNotMatch(diagnosticsSource, /Fresh Gemma planning pass|This researched answer|Qualified short-context peak|Fresh Plan Native Decode Rate|Multi-pass researched answer|aggregate native verifier clock|Target-verified lossless MTP/)

const streamingIndicatorSource = await readFile(new URL('../src/components/chat/render/StreamingIndicator.jsx', import.meta.url), 'utf8')
assert.match(streamingIndicatorSource, /data-rate-source=\{hasNativeRate \? 'backend-native' : 'browser'\}/, 'the one stock rate chip must remain auditable without a visible source label')
assert.match(streamingIndicatorSource, /waitForNativeRate/)
assert.doesNotMatch(streamingIndicatorSource, /message-live-tps__source|message-live-tps--native-segment|Streaming multi-pass synthesis|native segment/)

const messageTurnSource = await readFile(new URL('../src/components/chat/MessageTurn.jsx', import.meta.url), 'utf8')
assert.match(messageTurnSource, /<p>\{messageContent\}<\/p>/)
assert.match(messageTurnSource, /waitForNativeRate=\{message\.synthesis_mode/)
assert.match(messageTurnSource, /message\.target_verified_render \? null : formatRate/, 'native aggregate must not be mislabeled as the browser footer rate')
assert.doesNotMatch(messageTurnSource, /messageContent\.length > 900|exact characters|planner_segment_progress|cxturn__long-prompt|Xcode nutrition app research/)

assert.doesNotMatch(streamingIndicatorSource, /Gemma 4 is planning the response|target-verifying|multi-pass/)
assert.doesNotMatch(streamingIndicatorSource, /writing the implementation/)

const apiSource = await readFile(new URL('../../src/api/mod.rs', import.meta.url), 'utf8')
assert.match(
  apiSource,
  /gemma4_streaming_finish_reason\(\s*completion_tokens,\s*max_tokens,\s*target_verified_render,?\s*\)/s,
  'stream completion must distinguish exact-draft render completion from ordinary length exhaustion',
)
assert.match(apiSource, /if target_verified_render \{\s*"stop"/s)

console.log('Gemma 4 12B MTP12 target-verified render smoke: ok')
