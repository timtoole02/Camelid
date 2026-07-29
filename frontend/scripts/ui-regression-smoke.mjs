#!/usr/bin/env node
/* UI regression smoke — re-baselined in Phase 2 pre-work against the shipped
   Phase 1 reality (Evidence Chip, dark-first tokens, post-redesign chat stack).

   Heritage: every assertion from the pre-rebaseline script was either ported
   (verbatim where the source still matches, re-pointed where the code moved
   to MessageTurn.jsx / lib/markdown.jsx / chat.css) or explicitly retired —
   the retirement list with reasons lives in the re-baseline commit message.
   From that commit onward this smoke is part of the standing I6 gate set. */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  NEW_CHAT_SENTINEL,
  resolveSelectedConversation,
  shouldCreateConversationForSend,
} from '../src/lib/chatState.js'
import { normalizeStoredConversations } from '../src/lib/conversationStorage.js'
import { conversationToJson, conversationToMarkdown } from '../src/lib/conversationExport.js'
import { canonicalStatementLabel, splitCanonicalStatement } from '../src/lib/canonicalStatement.js'
import { describeExecutionPlan, executionRuntimeFields } from '../src/lib/executionPlan.js'

/* ---- Behavioral asserts: conversation selection + stored-stream recovery ---- */
const oldChat = { id: 'old-chat', title: 'Old chat', messages: [{ role: 'user', content: 'old prompt' }] }
const newerChat = { id: 'newer-chat', title: 'Newer chat', messages: [{ role: 'user', content: 'newer prompt' }] }
const conversations = [newerChat, oldChat]

const healthExecutionPlan = { selected_backend: 'cpu_reference' }
assert.deepEqual(executionRuntimeFields({ execution_plan: healthExecutionPlan, backend: 'llama' }), {
  execution_plan: healthExecutionPlan,
  backend: 'llama',
}, 'dashboard normalization should preserve health execution plan identity and serving backend')
assert.deepEqual(executionRuntimeFields(null), { execution_plan: null, backend: 'none' }, 'missing health should fail closed to no plan and no backend')

assert.deepEqual(describeExecutionPlan({ status: 'offline' }), {
  state: 'offline', device: 'Unavailable', backend: 'Backend offline', summary: 'Execution details are unavailable while the Camelid backend is offline.',
}, 'offline runtime must not infer an execution device')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: false }).state, 'idle', 'online runtime without a model should report no active plan')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama' }).state, 'unknown', 'generation-ready runtime without plan data should stay neutral')
assert.deepEqual(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'cpu_q8_runtime_repack' } }), {
  state: 'cpu', device: 'CPU', backend: 'cpu q8 runtime repack', summary: 'At model load, Camelid selected CPU using cpu q8 runtime repack. Runtime controls may change the effective path afterward.',
}, 'CPU plan copy should come from selected_backend')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'cuda_resident_q8_runtime', cuda_resident_active: true } }).device, 'CUDA GPU', 'consistent CUDA load plan should select GPU copy')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'cuda_resident_q8_runtime', cuda_resident_active: false } }).state, 'unknown', 'contradictory CUDA plan should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'cpu_reference', cuda_resident_active: true } }).state, 'unknown', 'contradictory CPU plan should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'metal_resident_q8_runtime' } }).device, 'Metal GPU', 'Metal load plan should select GPU copy')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'future_accelerator' } }).state, 'unknown', 'unknown future backends should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'cuda_resident_future', cuda_resident_active: true } }).state, 'unknown', 'unknown CUDA-prefixed backends should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama', execution_plan: { selected_backend: 'metal_resident_future' } }).state, 'unknown', 'unknown Metal-prefixed backends should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: false, backend: 'llama', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'pending', 'non-ready models should not produce execution claims')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'gemma4-runtime', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'specialized', 'specialized serving runtimes should not inherit generic plan device claims')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'runnable-runtime', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'specialized', 'runnable serving runtime should not inherit generic plan device claims')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'diffusion-gemma-runtime', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'specialized', 'DiffusionGemma serving runtime should not inherit generic plan device claims')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'none', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'unknown', 'ready payload with no serving backend should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'future-runtime', execution_plan: { selected_backend: 'cpu_reference' } }).state, 'unknown', 'unknown serving backends should fail neutral')
assert.equal(describeExecutionPlan({ status: 'online', loaded_now: true, generation_ready: true, backend: 'llama' }, { execution_plan: { selected_backend: 'cpu_reference' } }).state, 'unknown', 'capabilities must not resurrect a plan missing from health')

const canonicalStatementSample = 'Current exact-row support: Alpha (detail; retained); Beta evidence. These are exact bounded lanes only.'
const canonicalStatementParts = splitCanonicalStatement(canonicalStatementSample)
assert.deepEqual(canonicalStatementParts, [
  'Current exact-row support: Alpha (detail; retained);',
  'Beta evidence.',
  'These are exact bounded lanes only.',
], 'canonical statements should split only at top-level evidence boundaries')
assert.equal(canonicalStatementParts.join(' '), canonicalStatementSample, 'structured canonical statements must preserve the complete source text')
assert.equal(canonicalStatementLabel(canonicalStatementParts[0], 0), 'Current gate', 'the opening contract claim should retain its current-gate identity')
assert.equal(canonicalStatementLabel(canonicalStatementParts[2], 2), 'Contract statement', 'prose-derived blocks should stay neutral unless the backend explicitly names their semantic type')

assert.equal(resolveSelectedConversation(conversations, NEW_CHAT_SENTINEL), null, 'new-chat sentinel must render an empty landing, not the newest old chat')
assert.equal(resolveSelectedConversation(conversations, null), newerChat, 'null selection should recover to the newest available chat so the main pane does not blank during streaming')
assert.equal(resolveSelectedConversation(conversations, 'missing-chat'), newerChat, 'missing selection should recover to the newest available chat so streaming stays attached to a visible thread')
assert.equal(resolveSelectedConversation(conversations, 'old-chat'), oldChat, 'explicit old-chat selection should still open that chat')
assert.equal(shouldCreateConversationForSend(null, NEW_CHAT_SENTINEL), true, 'sending from new-chat landing should create a fresh conversation')
assert.equal(shouldCreateConversationForSend(oldChat, NEW_CHAT_SENTINEL), true, 'the sentinel must win even if a stale selectedConversation prop exists')
assert.equal(shouldCreateConversationForSend(oldChat, 'old-chat'), false, 'sending from an explicit existing chat should append to that chat')

const revivedInterruptedChat = normalizeStoredConversations([{ id: 'stale-chat', messages: [{ id: 'stale-assistant', role: 'assistant', content: '', streaming: true, streaming_phase: 'streaming' }] }], { clearStaleStreaming: true })[0]
assert.equal(revivedInterruptedChat.messages[0].streaming, false, 'reloaded interrupted streams should not claim the backend is still generating')
assert.equal(revivedInterruptedChat.messages[0].streaming_phase, null, 'reloaded interrupted streams should clear live generation phase')
assert.equal(revivedInterruptedChat.messages[0].finish_reason, 'interrupted', 'reloaded interrupted streams should be marked as interrupted')
assert.equal(revivedInterruptedChat.messages[0].content, '(generation interrupted)', 'blank reloaded interrupted streams should render safely')
const liveStreamingChat = normalizeStoredConversations([{ id: 'live-chat', messages: [{ id: 'live-assistant', role: 'assistant', content: 'partial', streaming: true, streaming_phase: 'streaming' }] }])[0]
assert.equal(liveStreamingChat.messages[0].streaming, true, 'live in-memory stream normalization should preserve active generation state')

/* ---- Conversation export must be path-free by construction (I7) ---- */
const sneakyConversation = {
  id: 'conv-1',
  title: 'Export test',
  model_id: 'llama32_3b_instruct_q8_0',
  model_path: '/Volumes/Untitled/models/secret.gguf',
  messages: [{
    id: 'm1', role: 'assistant', content: 'hello', model_id: 'llama32_3b_instruct_q8_0',
    model_path: '/Volumes/Untitled/models/secret.gguf',
    camelid: { backend_path: '/private/tmp/x.gguf' },
    support_row: { id: 'llama32_3b_instruct_q8_0', status: 'supported_exact_row_smoke', supported: true, manifest_path: '/Volumes/ExampleHome/qa/manifest.json' },
    usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
    usage_source: 'backend',
  }],
}
for (const exported of [conversationToJson(sneakyConversation), conversationToMarkdown(sneakyConversation)]) {
  assert.doesNotMatch(exported, /model_path|backend_path|manifest_path|\/Volumes\/|\/private\/tmp|\/Users\//, 'exports must never include filesystem paths — whitelisted fields only')
}
assert.match(conversationToJson(sneakyConversation), /telemetry, not support evidence|telemetry_note/, 'exports must carry the telemetry-not-evidence note')
assert.match(conversationToMarkdown(sneakyConversation), /telemetry, not support evidence/, 'markdown exports must label telemetry as not support evidence')

/* ---- Sources ---- */
const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8')
const readmeSource = read('../../README.md')
const chatWorkspaceSource = read('../src/views/ChatWorkspace.jsx')
const messageTurnSource = read('../src/components/chat/MessageTurn.jsx')
const markdownSource = read('../src/lib/markdown.jsx')
const dashboardHookSource = read('../src/hooks/useDashboardData.js')
const executionPlanSource = read('../src/lib/executionPlan.js')
const loadedModelDisplaySource = read('../src/lib/loadedModelDisplay.js')
const apiViewSource = read('../src/views/ApiView.jsx')
const systemViewSource = read('../src/views/SystemView.jsx')
const modelsViewSource = read('../src/views/ModelsView.jsx')
const topBarSource = read('../src/components/TopBar.jsx')
const analyticsViewSource = read('../src/views/AnalyticsView.jsx')
const capabilitiesSource = read('../src/lib/capabilities.js')
const streamParserSource = read('../src/lib/chatCompletionStream.js')
const evidenceChipSource = read('../src/components/ui/EvidenceChip.jsx')
const canonicalStatementSource = read('../src/components/ui/CanonicalStatement.jsx')
const exactRowEvidenceSummarySource = read('../src/components/ui/ExactRowEvidenceSummary.jsx')
const modelInspectorSource = read('../src/components/models/ModelInspector.jsx')
const compatibilityViewSource = read('../src/views/CompatibilityView.jsx')
const apiWorkbenchSource = read('../src/components/api/ApiWorkbench.jsx')
const telemetryViewSource = read('../src/views/TelemetryView.jsx')
const telemetryLogSource = read('../src/lib/telemetryLog.js')
const appSource = read('../src/App.jsx')
const tokenizerPlaygroundSource = read('../src/components/models/TokenizerPlayground.jsx')
const evidenceStatusSource = read('../src/lib/evidenceStatus.js')
const useThemeSource = read('../src/hooks/useTheme.js')
const mainSource = read('../src/main.jsx')
const tokensCss = read('../src/styles/tokens.css')
const evidenceCss = read('../src/styles/evidence.css')
const chatCss = read('../src/styles/chat.css')
const uiCss = read('../src/styles/ui.css')
const statusSheets = ['../src/styles/ui.css', '../src/styles/shell.css', '../src/styles/chat.css', '../src/styles/views.css', '../src/styles/cluster.css', '../src/styles/observatory.css']
  .map((path) => [path, read(path)])

/* ---- README product surface ---- */
assert.match(readmeSource, /docs\/assets\/camelid-readme-chat-surface-dark\.png/, 'README should use the approved dark collapsed-rail chat screenshot')
assert.doesNotMatch(readmeSource, /assets\/camelid-banner\.png/, 'README should not lead with the disliked first banner image')
assert.doesNotMatch(readmeSource, /docs\/assets\/ui-screenshot-v2\.png/, 'README must not regress to the retired light screenshot')

/* ---- Chat workspace ---- */
assert.match(chatWorkspaceSource, /lastVisibleMessageIsUser[\s\S]*awaitingAssistant[\s\S]*generationActive && !hasStreamingAssistantContent/, 'a sent user row should keep showing an awaiting assistant indicator until streamed assistant content is visible')
assert.match(chatWorkspaceSource, /hasStreamingAssistant[\s\S]*generationActive/, 'a persisted streaming row should keep the UI active even if the send call state changes')
assert.match(chatWorkspaceSource, /cxcomposer__status/, 'the composer should keep the consolidated runtime/support status line')
assert.match(chatWorkspaceSource, /<EvidenceChip/, 'the composer support claim should render through the Evidence Chip, not an ad-hoc badge')
assert.match(chatWorkspaceSource, /selectedChatGate\.contractSupported \? 'supported'/, 'the composer Evidence Chip must take its supported state only from the shared chat gate')
/* Ported (Phase 9): the original blunt regex predates the composer's send-time
   budget VALIDATION (which legitimately references the configured cap). The
   intent stands: chat must not render a max-token PICKER — the control lives
   in Settings. */
assert.doesNotMatch(chatWorkspaceSource, /<ResponseLengthControl|Response length<|setConfiguredMaxTokens/, 'Chat UI must not expose the response-length picker — it lives in Settings')

/* ---- Message rendering (moved from pre-redesign ChatWorkspace to MessageTurn/markdown) ---- */
assert.match(messageTurnSource, /aria-busy=\{assistantStreaming \? 'true' : undefined\}/, 'streaming assistant rows should expose row-level busy state while text is incomplete')
assert.match(messageTurnSource, /data-streaming-state=\{assistantStreaming \? 'active' : undefined\}/, 'streaming assistant rows should expose an active state marker for regression coverage')
assert.match(messageTurnSource, /\$\{assistantStreaming \? 'is-streaming' : ''\}/, 'only assistant rows that are actively streaming should receive the animated streaming class')
assert.doesNotMatch(messageTurnSource, /\$\{message\.streaming \? 'is-streaming' : ''\}/, 'raw message.streaming should not keep completed/non-assistant rows visually active')
assert.match(messageTurnSource, /\{message\.role === 'assistant' && <MessageMetaFooter message=\{message\} \/>\}/, 'the assistant meta footer should render during streaming too — it is the live tok/s readout (tokens_out_per_sec is live-patched per frame)')
assert.doesNotMatch(messageTurnSource, /cxturn__meta--reserve/, 'the invisible footer placeholder is gone; the live footer itself holds the layout slot')
assert.doesNotMatch(read('../src/styles/chat.css'), /cxturn__meta--reserve/, 'the reserved-footer spacer css must not outlive the placeholder it styled')
assert.match(messageTurnSource, /streaming=\{assistantStreaming\}/, 'assistant markdown should know when an assistant row is still streaming')
assert.match(markdownSource, /splitFenceInfo/, 'streaming/incomplete fenced code blocks should be parsed as code instead of prose')
assert.match(markdownSource, /pushCodeBlock/, 'code block rendering should stay centralized for complete and incomplete fences')
assert.match(markdownSource, /CODE_CARD_STREAMING_LABEL\s*=\s*'Still generating — code block incomplete'/, 'incomplete streaming code blocks should visibly say the code is still incomplete')
assert.match(markdownSource, /data-code-streaming-state=\{stillGenerating \? 'open' : undefined\}/, 'open streaming code fences should expose an active code state marker')
assert.match(markdownSource, /message-code-card-status[^>]*aria-live="polite"[^>]*data-live-status="active"[^>]*>\{CODE_CARD_STREAMING_LABEL\}</, 'incomplete streaming code blocks should show a live active still-generating badge')
assert.doesNotMatch(markdownSource, /dangerouslySetInnerHTML/, 'model output must never reach the DOM through dangerouslySetInnerHTML')
assert.doesNotMatch(messageTurnSource, /dangerouslySetInnerHTML/, 'message rows must never use dangerouslySetInnerHTML')

/* ---- Dashboard data hook ---- */
assert.match(dashboardHookSource, /Include inline <style> and inline <script>/, 'HTML code prompts should ask for inline CSS and JS, not an unfinished fragment')
assert.match(dashboardHookSource, /max_tokens:\s*localChatMaxTokens\(history, requestModelId\)/, 'local chat sends should choose the per-model token budget (Phase 9)')
assert.match(dashboardHookSource, /getRuntimeRequestModelId\(selectedModel, runtime, selectedModelId\)/, 'chat sends should use the backend active runtime model id when a browser alias is selected')
assert.doesNotMatch(dashboardHookSource, /Camelid streamed the local reply\./, 'successful streams should not show a noisy demo-breaking toast')
assert.match(dashboardHookSource, /readStreamingChatCompletion\(response/, 'dashboard chat send should use the centralized stream parser')
assert.match(dashboardHookSource, /finish_reason:\s*requestWasAborted\s*\?\s*'interrupted'\s*:\s*'error',[\s\S]*streaming:\s*false/, 'failed or interrupted generations should clear streaming state instead of leaving active pellets/status forever')
assert.match(dashboardHookSource, /const conversations = localConversations\.length \? localConversations : dashboard\?\.conversations \|\| \[\]/, 'main chat should resolve selectedConversation from live local conversation state before stale dashboard snapshots')
assert.match(dashboardHookSource, /currentLocalConversations\.some\(\(conversation\) => conversation\.id === current\)/, 'dashboard refresh should validate selected conversation against the same current local conversation snapshot it renders')
assert.match(dashboardHookSource, /const selectedConversationIdRef = useRef\(selectedConversationId\)/, 'conversation selection should keep an immediate ref so background refreshes do not lose the active thread between state commits')
assert.match(dashboardHookSource, /selectedConversationIdRef\.current = next[\s\S]*setSelectedConversationIdState\(next\)/, 'conversation selection updates should write the ref immediately before the async state commit')
assert.match(dashboardHookSource, /activeModelChatGate\?\.chatUnlocked && current !== activeModel\.id/, 'browser-selected model should snap back to the backend active model only through the shared exact-row chat gate')
assert.match(dashboardHookSource, /modelRuntimeIdMatches/, 'dashboard model merge should treat runtime_model_name as an active_model_id alias instead of losing readiness for imported exact rows')
assert.match(dashboardHookSource, /resolveLoadedModelDisplayName/, 'dashboard model merge should rewrite backend-generated active ids to the exact 3B display row only from exact GGUF filename plus Q8_0 metadata')
assert.match(loadedModelDisplaySource, /ggufFileTypeValueFromLabel[\s\S]*quantLabelFromGgufFileType[\s\S]*LLAMA32_3B_ACCEPTANCE_FILENAME[\s\S]*normalizeQuantLabel\(quantLabel\) === 'Q8_0'/, 'the 3B display alias must stay exact-row and decoded Q8_0/file_type 7 gated rather than broad-family')
assert.match(dashboardHookSource, /localRecordMatchesBackendId/, 'dashboard model merge should de-duplicate backend model rows against saved browser records by id or runtime_model_name')
assert.match(dashboardHookSource, /const id = localRecord\?\.id \|\| item\.id/, 'backend model merges should preserve the browser row id while keeping the backend runtime id as runtime_model_name')
assert.match(dashboardHookSource, /const conversation = await ensureConversation\(\)[\s\S]*?setSelectedConversationId\(conversation\.id\)[\s\S]*?fetch\(`\$\{normalizedApiBase\}\/v1\/chat\/completions`/, 'fresh-chat sends must select the real conversation before streaming starts so the main pane updates with sidebar previews')
assert.match(dashboardHookSource, /applyLocalChatPolicy\(history\)/, 'code/html prompts should use the local code-first request policy')
assert.match(dashboardHookSource, /CODE_FIRST_SYSTEM_PROMPT/, 'frontend should keep a code-first system prompt for code/html local chat requests')
assert.match(dashboardHookSource, /begin immediately with complete runnable code/, 'code-first prompt should suppress slow prose preambles before code and ask for complete output')
assert.match(dashboardHookSource, /Start exactly with ```html then <!doctype html>/, 'HTML code prompts should request visible code at the beginning of the stream')
assert.match(dashboardHookSource, /ONE self-contained file/, 'HTML code prompts should ask for one complete file, not separated assets')
assert.match(dashboardHookSource, /For Python, start exactly with ```python/, 'Python code prompts should get a Python-specific complete-script instruction')
assert.match(dashboardHookSource, /prefer tkinter from the standard library over pygame/, 'Python game prompts should prefer compact standard-library demos over sprawling dependency-heavy pygame output')
assert.match(dashboardHookSource, /complete runnable event loop/, 'Python game prompts should ask for runnable game logic, not a sketch')
assert.match(dashboardHookSource, /python\|py\|pygame\|game\|pacman\|pacmac/, 'code-first detection should catch Python game demos and the pacmac typo')
assert.match(dashboardHookSource, /Never use external files or script src/, 'HTML code prompts should prevent unusable external script references in demos')

/* ---- Stream parser ---- */
assert.match(streamParserSource, /function defaultEstimateTokenCount/, 'central stream parser should keep a JSON fallback token estimator')
assert.match(streamParserSource, /function readSseDataLines/, 'central stream parser should isolate SSE data-line handling')
assert.match(streamParserSource, /export function extractSseEvents/, 'stream parser should keep SSE boundary handling centralized')
assert.match(streamParserSource, /replace\(/, 'stream parser should normalize line endings before splitting SSE events')
assert.match(streamParserSource, /split\('\\n\\n'\)/, 'stream parser should split normalized SSE events on blank lines for partial rendering')

/* ---- API view ---- */
assert.match(apiViewSource, /Selected exact-row evidence/, 'API support view should show selected exact-row evidence instead of a broad validated-target claim')
assert.match(apiViewSource, /<CanonicalStatement text=\{supportContractCurrentGate\}/, 'API should render the complete gate through the shared structured canonical statement')
assert.match(apiViewSource, /selectedChatGate\s*=\s*getChatGateState\(capabilities, selectedModel, runtime\)/, 'API endpoint readiness should use the shared exact-row chat gate')
assert.match(apiViewSource, /selectedExactRowReady\s*=\s*selectedChatGate\.chatUnlocked/, 'API endpoint readiness should stay aligned with Chat/System exact-row chat unlocks')
assert.match(apiViewSource, /selectedRuntimeMatches/, 'API endpoint readiness should require active_model_id to match the selected model')
assert.match(apiViewSource, /readinessPillCopy/, 'API endpoint status copy should come from the exact-row readiness gate, not generation_ready alone')
assert.match(apiViewSource, /chatCompletionsCopy/, 'API chat-completions copy should stay gated unless selected exact-row evidence and runtime readiness both match')
assert.match(apiViewSource, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl example should fail closed until exact-row evidence and runtime readiness match')
assert.match(apiViewSource, /selectedCompatibilityTarget\.frontend_readiness_gate/, 'API support view should surface the selected row readiness gate verbatim from /api/capabilities')
assert.match(apiViewSource, /selectedCompatibilityTarget\.support_scope/, 'API support view should surface exact-row support scope instead of inferring a broader claim')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_bucket/, 'API support view should surface exact-row latest checked bucket evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_output/, 'API support view should surface exact-row latest output evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.full_support_status/, 'API support view should show the exact row full-support status boundary')
assert.match(apiViewSource, /exactRowSupportLanes\(selectedCompatibilityTarget, apiFeatures\)/, 'API support view should show template/Jinja, checked-context, and throughput readiness lanes for the selected exact row')
assert.match(apiViewSource, /rowSupportBoundaryCopy\(selectedCompatibilityTarget, apiFeatures\)/, 'API support view should filter resolved template/Jinja and throughput blockers out of the remaining support boundary')
assert.match(apiViewSource, /rowSupportNextStepCopy\(target, apiFeatures\)/, 'API support view should filter resolved template/Jinja and throughput blockers out of row next-step copy')
assert.match(capabilitiesSource, /function frontendSupportContractCopy/, 'frontend support contract copy should filter resolved template/Jinja and throughput caveats for current supported rows')
assert.match(capabilitiesSource, /Production-throughput readiness is green/, 'capability helpers should describe production-throughput as a green exact-row readiness lane when perf evidence is supported')
assert.match(exactRowEvidenceSummarySource, /function groupExactRows[\s\S]*target\?\.id && target\?\.\[field\]/, 'exact-row evidence summaries should group only concrete compatibility rows with the requested field')
assert.match(apiViewSource, /<ExactRowEvidenceSummary targets=\{compatibilityTargets\} field="quantization"/, 'API support view should summarize quant evidence with the shared exact-row renderer')
assert.match(apiViewSource, /<ExactRowEvidenceSummary targets=\{compatibilityTargets\} field="family"/, 'API support view should summarize family evidence with the shared exact-row renderer')
assert.match(apiViewSource, /Exact-row quant evidence/, 'API support view should label quant evidence as exact-row scoped')
assert.match(apiViewSource, /Exact-row family evidence/, 'API support view should label family evidence as exact-row scoped')
assert.match(apiViewSource, /broad quant lists do not unlock chat/, 'API support view should not promote broad quant lists into chat readiness')
assert.match(apiViewSource, /row-scoped family\/quant evidence/, 'API endpoint summary should describe family and quant evidence as row-scoped')
assert.doesNotMatch(apiViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|summarizeCapabilityItems/, 'API support view should not render non-row capability lists as support evidence')
assert.match(apiViewSource, /No exact compatibility row matched this selected model/, 'API selected model contract should fail closed instead of displaying family or saved-path guesses')
assert.match(apiViewSource, /displayCapabilityCopy\(selectedCompatibilityTarget\.evidence\)/, 'API support view should sanitize and display exact-row evidence copy')
assert.match(capabilitiesSource, /function displayCapabilityId/, 'capability ids should be display-normalized before support/API UI rendering')
assert.match(capabilitiesSource, /function displayCapabilityCopy/, 'backend capability copy should be display-normalized before support/API UI rendering')
assert.match(apiViewSource, /displayCapabilityId\(feature\.id\)/, 'API view should not render raw provider-scoped API feature ids')
assert.match(apiViewSource, /getRuntimeRequestModelId\(selectedModel, runtime, '<loaded-model-id>'\)/, 'API curl examples should use the loaded backend model id for alias-selected exact rows')
assert.match(apiViewSource, /<EvidenceChip/, 'API contract rows should render their status claims through the Evidence Chip')

/* ---- System view ---- */
assert.match(systemViewSource, /Selected exact-row evidence/, 'System support view should show selected exact-row evidence instead of broad quant or family capability lists')
assert.match(systemViewSource, /<CanonicalStatement text=\{supportContractCurrentGate\}/, 'System should render the complete gate through the shared structured canonical statement')
assert.match(systemViewSource, /Exact-row quant evidence/, 'System support view should scope quant evidence to compatibility rows')
assert.match(systemViewSource, /Exact-row family evidence/, 'System support view should scope family evidence to compatibility rows')
assert.match(systemViewSource, /<ExactRowEvidenceSummary targets=\{compatibilityTargets\} field="quantization"/, 'System support view should summarize quant evidence with the shared exact-row renderer')
assert.match(systemViewSource, /<ExactRowEvidenceSummary targets=\{compatibilityTargets\} field="family"/, 'System support view should summarize family evidence with the shared exact-row renderer')
assert.doesNotMatch(systemViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|summarizeCapabilityItems/, 'System support view should not render non-row capability lists as support evidence')
assert.match(systemViewSource, /displayCapabilityId\(feature\.id\)/, 'System view should not render raw provider-scoped API feature ids')
assert.match(systemViewSource, /getRuntimeRequestModelId\(selectedModel, runtime, '<loaded-model-id>'\)/, 'System curl examples should use the loaded backend model id for alias-selected exact rows')
assert.match(systemViewSource, /<EvidenceChip/, 'System contract rows should render their status claims through the Evidence Chip')
assert.match(dashboardHookSource, /\.\.\.executionRuntimeFields\(health\)/, 'dashboard runtime state should use the tested health execution-field mapper')
assert.match(systemViewSource, /describeExecutionPlan\(runtime\)/, 'System execution copy should come only from the health-derived runtime snapshot')
assert.doesNotMatch(systemViewSource, /GPU acceleration remains future work|local CPU generation path today/, 'System must not retain static backend execution claims')
assert.match(executionPlanSource, /CUDA_BACKENDS\.has\(selectedBackend\)[\s\S]*METAL_BACKENDS\.has\(selectedBackend\)/, 'execution-plan presenter should classify only explicitly known CUDA and Metal backends')
assert.match(compatibilityViewSource, /<CanonicalStatement text=\{displayCapabilityCopy\(supportContract\.current_gate\)\}/, 'Compatibility should render the complete gate through the shared structured canonical statement')
assert.match(canonicalStatementSource, /View as one canonical paragraph/, 'the structured disclosure should retain one-click access to the unbroken canonical paragraph')
assert.match(compatibilityViewSource, /if \(query \|\| posture !== 'all'\)[\s\S]*setQuery\(''\)[\s\S]*setPosture\('all'\)/, 'ledger deep-links should clear filters before resolving their target row')
assert.match(compatibilityViewSource, /if \(!node\) return undefined[\s\S]*node\.scrollIntoView[\s\S]*onFocusConsumed/, 'ledger deep-links should be consumed only after the target row exists')
assert.match(compatibilityViewSource, /node\.focus\(\{ preventScroll: true \}\)[\s\S]*node\.scrollIntoView/, 'ledger deep-links should transfer keyboard focus before scrolling the destination into view')
assert.match(compatibilityViewSource, /data-row-id=\{row\.id\}[\s\S]*tabIndex=\{-1\}/, 'model evidence rows should be programmatically focusable')
assert.match(compatibilityViewSource, /data-row-id=\{feature\.id\} tabIndex=\{-1\}/, 'API feature evidence rows should be programmatically focusable')

/* ---- Models view ----
   Redesign (2026-07, D14): the page was rebuilt as five derived zones. The tracked-row
   cards, acceptance panel, and per-card evidence blocks are gone; the invariants are now
   that membership is DERIVED (live scan + contract via lib/modelLanes), loads are
   inspect-first fail-closed, and no hand-authored array or localStorage record places a
   model or claims a download. Row-scoped lane copy stays asserted on System/API above. */
const modelLanesSource = read('../src/lib/modelLanes.js')
const laneRowsSource = read('../src/components/models/LaneRows.jsx')
const catalogBrowseSource = read('../src/components/models/CatalogLaneBrowse.jsx')
const downloadsPanelSource = read('../src/components/models/DownloadsPanel.jsx')
const modelActivationSource = read('../src/lib/modelActivation.js')
const firstRunCardSource = read('../src/components/onboarding/FirstRunCard.jsx')
assert.match(modelsViewSource, /bucketByLane\(spine\.local\.models, capabilities\)/, 'Models section membership must be derived from the live scan + contract at render time')
assert.match(modelLanesSource, /isCompatibilitySupportedForModel\(capabilities, matchModel\(entry\)\)/, 'Models lane derivation must ask the shared contract matcher — the supported gate stays the contract voice')
assert.doesNotMatch(modelsViewSource, /SUPPORTED_MODELS/, 'Models view must not place models from a hand-authored array')
assert.doesNotMatch(modelsViewSource, /localStorage\.(get|set|remove)Item/, 'Models view must not read or write localStorage truth')
/* The inspect-first load protocol moved into lib/modelActivation.js when the
   first-run card became a second caller: two hand-written copies of an ordered
   fail-closed sequence is how one of them loses a step. The invariant is unchanged
   and now has one home, so it is asserted there — and both surfaces must route
   through it rather than reach for the load endpoint directly.
   (frontend/scripts/first-run-activation-smoke.mjs proves the ordering by
   execution; this only pins that nobody re-forks the protocol.) */
assert.match(modelActivationSource, /api\/models\/inspect[\s\S]*blocker[\s\S]*return[\s\S]*api\/models\/load/, 'Models loads must inspect first and stop on typed blockers before any load attempt')
assert.match(modelsViewSource, /loadLocalModelForChat\(/, 'Models view must load through the shared activation protocol')
assert.doesNotMatch(modelsViewSource, /api\/models\/load/, 'Models view must not hand-roll a second load path')
assert.match(modelsViewSource, /UnsupportedBlocker/, 'typed fail-closed blockers must render verbatim through UnsupportedBlocker')
assert.doesNotMatch(modelsViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|getQuantCapability|quantCapabilityLabel|quantCapabilityCopy/, 'Models view should not render broad quant/family capability lists as support evidence')
assert.match(laneRowsSource, /<EvidenceChip/, 'Models lane rows must render their status claims through the Evidence Chip')
assert.match(laneRowsSource, /never copper/, 'runnable rows must document that they never take the reserved supported (copper) styling')
assert.match(catalogBrowseSource, /if \(item\.group === 'experimental'\) return 'not_anchored'/, 'live Hugging Face rows must never anchor a lane or imply support')
assert.match(catalogBrowseSource, /Confirm download/, 'catalog downloads must go through an explicit confirmation phase')
assert.match(catalogBrowseSource, /Download and start/, 'curated catalog rows should offer the complete activation workflow')
assert.match(catalogBrowseSource, /settlementInFlightRef\.current/, 'catalog settlement must be single-flight across polling ticks')
assert.match(catalogBrowseSource, /canceledCatalogIds\.has\(item\.catalog_id\)/, 'catalog cancellation must be keyed by catalog identity, not filename')
assert.match(catalogBrowseSource, /aria-label="Search model catalog"/, 'catalog search must have an explicit accessible name')
assert.match(catalogBrowseSource, /downloadAndStart = lane === 'supported' && !refusedByFit/, 'automatic start must be limited to supported rows that are not known to exceed this host')
assert.match(catalogBrowseSource, /refusedByFit[\s\S]*item\.oracle_qualified/, 'rows this host cannot load must not automatically run generic smoke admission')
// The refusal set must stay the FULL one. Testing `fit !== 'wont_fit'` alone was a
// real defect: an `insufficient_free_memory` row would chain into a load that the
// 422 preload guard refuses, since that guard blocks on both negative verdicts.
assert.match(catalogBrowseSource, /const refusedByFit = isRefusingFit\(item\.fit\)/, 'auto-start must gate on every load-refusing verdict, not just wont_fit')
assert.match(modelActivationSource, /if \(!inspectRes\.ok\)[\s\S]*return \{ ok: false, stage: CHECKING, message, code, blocker/, 'automatic activation must fail closed on an HTTP-level inspect failure')
assert.match(modelActivationSource, /activeFilename !== filename[\s\S]*did not confirm/, 'automatic navigation must wait for current-model confirmation')
assert.match(modelActivationSource, /v1\/health[\s\S]*health\.loaded_now[\s\S]*health\.generation_ready[\s\S]*health\.active_model_id !== filename/, 'automatic navigation must wait for live generation readiness and active-model identity')
assert.match(modelsViewSource, /readActiveFilename: async \(\) => modelFilenameFromPath\(\(await spine\.refreshCurrent\(\)\)\?\.path\)/, 'the Models page must answer the identity check from its own current-model refresh')
assert.match(appSource, /modelsVisited[\s\S]*hidden=\{tab !== 'library'\}/, 'Models must remain mounted after first visit so active downloads retain their activation coordinator')
/* First-run activation, same rule for the same reason: an in-flight install must keep
   its watcher when the user navigates, and it must not be unmounted the moment the
   landed file stops making the host look like a fresh install. */
assert.match(appSource, /firstRunActive = firstRun \|\| firstRunCardActive/, 'the first-run card must survive the host no longer looking fresh, for as long as it still owns the flow')
assert.match(appSource, /firstRunActive &&[\s\S]*hidden=\{tab !== 'chat'\}/, 'the first-run card must stay mounted across navigation and for the whole activation')
assert.match(firstRunCardSource, /RETAINED_PHASES = new Set\(\[\.\.\.IN_FLIGHT_PHASES, 'failed'\]\)/, "a failure still owns the flow: its retry is the only route to an artifact that already downloaded")
assert.match(appSource, /isFirstRunHost\(\{ runtime, models \}\)/, 'first-run state must be derived from live runtime state, never from a stored onboarding flag')
assert.doesNotMatch(firstRunCardSource, /localStorage/, 'first-run onboarding must not record itself in localStorage')
assert.match(firstRunCardSource, /recommendFirstRunModel\(catalogItems, capabilities\)/, 'the first-run offer must be derived from the live catalog and the support contract')
assert.match(firstRunCardSource, /settlementInFlightRef/, 'first-run settlement must be single-flight across polling ticks')
assert.match(firstRunCardSource, /catalog\/cancel/, 'a first-run download must stay cancellable')
assert.match(firstRunCardSource, /warmGenerationPath/, 'the first message after activation must not pay the cold engine build')
/* A failure AFTER the artifact landed must retry the check/load, never the download:
   re-installing refetches the whole file and drops the completed record, and the new
   download's rename onto the existing GGUF can fail. */
assert.match(firstRunCardSource, /firstRunRetryAction\(\{ artifactInstalled \}\)/, 'the retry target must be decided by whether the artifact landed')
assert.match(firstRunCardSource, /retryTarget === 'activate' \? retryActivation : startDownload/, 'a landed artifact must retry activation instead of re-downloading')
/* Cancelling is a request, not a fact: 200 stopped a download, 409 means it finished
   first and KEPT its file, 404 can mean the install has not registered yet. */
assert.match(firstRunCardSource, /confirmed = res\.ok/, 'only a successful cancel counts as a confirmed stop')
assert.match(firstRunCardSource, /observeAfterCancel\(\)/, 'the cancel outcome must be decided by re-reading downloads plus the local scan')
assert.match(firstRunCardSource, /firstRunCancelOutcome\(\{ confirmed, \.\.\.observed \}\)/, 'and routed through the shared outcome rule')
assert.doesNotMatch(firstRunCardSource, /finally \{[\s\S]{0,200}fail\('Download canceled/, 'cancellation must never be reported unconditionally from a finally block')
assert.match(modelsViewSource, /loadInFlightRef\.current[\s\S]*(already loading|finish loading, then retry)/, 'model loading must be single-flight across catalog completions')
assert.match(modelsViewSource, /deleteLocalModel\(entry\)/, 'local deletion must submit the scanned entry identity rather than filename alone')
assert.match(modelsViewSource, /Delete model from disk\?/, 'local model deletion must require destructive confirmation')
assert.match(laneRowsSource, /entry\.delete_token/, 'delete controls must require the scan-issued opaque identity token')
assert.match(laneRowsSource, /model-delete-guard/, 'disabled delete controls must reference visible guard copy')
assert.match(downloadsPanelSource, /catalog\/downloads|bytes_downloaded/, 'download progress must come only from the backend downloads poll')

/* ---- Model management (Phase 3) ---- */
assert.match(modelInspectorSource, /not support evidence/, 'the model inspector must label its contents as descriptive, not support evidence')
assert.doesNotMatch(modelInspectorSource, /getChatGateState|isCompatibilitySupportedForModel|findCompatibilityHint/, 'the inspector renders metadata; it must never compute or imply gate state')
assert.match(modelInspectorSource, /items\]|items…/, 'huge GGUF arrays must be summarized, not dumped')
assert.match(tokenizerPlaygroundSource, /does not widen generation support/, 'the tokenizer playground must say its output is not generation-support evidence')
assert.match(tokenizerPlaygroundSource, /tokenizer_encode_decode/, 'the playground chip must cite the exact contract feature row')

/* ---- Observatory lifecycle (Phase 6.1 defect guards) ---- */
const inferenceTelemetryHookSource = read('../src/hooks/useInferenceTelemetry.js')
assert.match(inferenceTelemetryHookSource, /const sharedStore = createInferenceTelemetryStore\(\)/, 'the inference telemetry store must be a shared app-lifetime singleton, not per-mount (DEFECT 1)')
assert.doesNotMatch(inferenceTelemetryHookSource, /useMemo\(\(\) => createInferenceTelemetryStore/, 'per-mount store creation loses every event emitted while the view is unmounted')
assert.doesNotMatch(inferenceTelemetryHookSource, /store\.disconnect\(\)/, 'unmount must not tear down the shared stream — navigation would wipe run state (DEFECT 2)')
assert.match(appSource, /ensureInferenceTelemetryConnected/, 'the app shell must connect the observatory stream at startup, not first view mount (DEFECT 1)')

/* ---- Response-length control (Phase 9) ---- */
const responseLimitsSource = read('../src/lib/responseLimits.js')
const rlcSource = read('../src/components/settings/ResponseLengthControl.jsx')
const { validateResponseLength, validateSendBudget, verifiedContextBound, sliderToTokens, tokensToSlider } = await import('../src/lib/responseLimits.js')
{
  const overCtx = validateResponseLength({ value: 8192, contextLength: 64, verifiedBound: null, modelName: 'fixture' })
  assert.equal(overCtx.level, 'caution', 'above the model context is a non-blocking caution now (backend auto-limits, does not reject)')
  assert.match(overCtx.message, /auto-limit|room left|shorter/, 'over-context copy must explain the auto-limit, not claim rejection')
  const amber = validateResponseLength({ value: 4096, contextLength: 131072, verifiedBound: 2048 })
  assert.equal(amber.level, 'caution', 'value above verified bound but under model max is amber')
  assert.match(amber.message, /allowed, untested/, 'amber copy must state allowed-but-untested')
  assert.equal(validateResponseLength({ value: 1024, contextLength: 131072, verifiedBound: 2048 }).level, 'ok')
  // The response limit is an upper bound the backend clamps to fit, so an
  // overshoot that still leaves prompt room is a non-blocking notice, not a block.
  const clamped = validateSendBudget({ promptTokens: 60, maxTokens: 50, contextLength: 64 })
  assert.equal(clamped.level, 'notice', 'over-limit but room-remaining must be a non-blocking notice (backend clamps)')
  assert.notEqual(clamped.level, 'error', 'send must not be blocked when the response can be auto-limited to fit')
  // A prompt that already fills the whole context leaves no room — the one genuine error.
  const noRoom = validateSendBudget({ promptTokens: 64, maxTokens: 50, contextLength: 64 })
  assert.equal(noRoom.level, 'error', 'a prompt that fills the context has no room to generate and must error')
  // 3B row mirrors the anchored raw-decode ladder (all five buckets validated on the
  // canonical f34112a1 GGUF), so the verified bound now reaches 8192.
  const anchoredThreeBRow = { id: 'llama32_3b_instruct_q8_0', family: 'llama_bpe_decoder', quantization: 'Q8_0', status: 'supported_exact_row_smoke', bounded_context_512_pack: 'validated_anchored_raw_decode_ladder', bounded_context_512_window: 512, bounded_context_1024_pack: 'validated_anchored_raw_decode_ladder', bounded_context_1024_window: 1024, bounded_context_2048_pack: 'validated_anchored_raw_decode_ladder', bounded_context_2048_window: 2048, bounded_context_4096_pack: 'validated_anchored_raw_decode_ladder', bounded_context_4096_window: 4096, bounded_context_8192_pack: 'validated_anchored_raw_decode_ladder', bounded_context_8192_window: 8192 }
  const anchoredThreeBModel = { id: 'llama32_3b_instruct_q8_0', name: 'x', quant: 'Q8_0', model_path: '/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf' }
  const bound = verifiedContextBound({ model_compatibility: [anchoredThreeBRow] }, anchoredThreeBModel)
  assert.equal(bound, 8192, 'verified bound must reach 8192 now that the 3B anchored raw-decode ladder validates all five buckets')
  const heldBound = verifiedContextBound({ model_compatibility: [{ ...anchoredThreeBRow, bounded_context_8192_pack: 'not_promoted' }] }, anchoredThreeBModel)
  assert.equal(heldBound, 4096, 'verified bound is the max VALIDATED pack window, never an unvalidated one')
  assert.ok(Math.abs(tokensToSlider(sliderToTokens(0.5)) - 0.5) < 0.03, 'log slider mapping round-trips')
}
assert.match(rlcSource, /from model metadata, not a support claim/, 'the context marker must disclaim support (I2)')
assert.match(rlcSource, /memory estimate unavailable — backend does not yet report/, 'missing memory data renders an honest absent line, never a fake gauge')
assert.match(rlcSource, /estimated/, 'the future readout contract keeps the estimated label in source')
assert.doesNotMatch(rlcSource, /navigator\.deviceMemory|performance\.memory/, 'no client-side memory guessing')
assert.match(chatWorkspaceSource, /validateSendBudget/, 'the composer must validate prompt+max_tokens against the real context rule at send time')

/* ---- Display pacing honesty bounds (Phase 8B) ---- */
const { createPacerState, paceStep, paceDrain, MAX_LAG_MS } = await import('../src/lib/streamPacing.js')
{
  const state = createPacerState()
  let received = ''
  let now = 0
  // bursty arrival: 40 chars instantly, then nothing, then a 200-char burst
  received = 'x'.repeat(40)
  for (let i = 0; i < 30; i += 1) { now += 16; paceStep(state, received, now) }
  let shown = paceStep(state, received, now += 16)
  assert.equal(shown.length, 40, 'paced display must fully catch up during quiet periods')
  received = 'x'.repeat(240)
  const arrivalAt = now
  let lagViolated = false
  while (shown.length < 240) {
    now += 16
    shown = paceStep(state, received, now)
    if (now - arrivalAt > MAX_LAG_MS && shown.length < 240) lagViolated = true
  }
  assert.equal(lagViolated, false, `paced display must never lag real arrival by more than ${MAX_LAG_MS}ms`)
  const drained = paceDrain(state, received)
  assert.equal(drained, received, 'drain must be instant and byte-identical to the received stream')
}
assert.match(dashboardHookSource, /paceStep\(pacer, fullContent/, 'streaming display must go through the bounded pacer')
assert.match(dashboardHookSource, /paceDrain\(pacer, streamed\.content/, 'final content must drain byte-identical from the real stream')

/* ---- Camelid mark (Phase 8): original glyph, derivative sparkle retired ---- */
const camelidMarkSource = read('../src/components/ui/CamelidMark.jsx')
const avatarSource = read('../src/components/ui/Avatar.jsx')
const faviconSource = read('../public/favicon.svg')
assert.match(topBarSource, /<CamelidMark/, 'the TopBar must render the Camelid mark')
assert.match(chatWorkspaceSource, /CamelidMark|Avatar/, 'chat must render the Camelid mark (directly or via Avatar)')
assert.match(avatarSource, /CamelidMark/, 'the assistant avatar must frame the Camelid mark')
assert.match(camelidMarkSource, /camelid-mark__ear/, 'the mark keeps its animatable ear sub-elements')
assert.match(camelidMarkSource, /reduced-motion/, 'mark states must document the reduced-motion contract')
for (const [path, source] of [['Avatar.jsx', avatarSource], ['TopBar.jsx', topBarSource], ['favicon.svg', faviconSource], ['ChatWorkspace.jsx', chatWorkspaceSource]]) {
  assert.doesNotMatch(source, /[Ss]parkle|camelid-sparkle-grad/, `${path} must not ship the retired sparkle mark`)
}
assert.doesNotMatch(faviconSource, /9b51e0|4285f4|fa9085/, 'favicon must not carry the old gradient stops')

/* ---- Flow Bench (Phase 6.1) ---- */
const flowBenchSource = read('../src/components/observatory/FlowBench.jsx')
const flowBenchEngineSource = read('../src/lib/observatory/flowBench.js')
const observatoryViewSource = read('../src/views/InferenceObservatoryView.jsx')
assert.match(observatoryViewSource, /operational telemetry — not compatibility evidence/, 'the Flow Bench view must carry the telemetry-not-evidence affordance')
assert.match(observatoryViewSource, /flowbench-rail__tiles/, 'the instrument rail tiles must be present')
assert.match(flowBenchSource, /aria-hidden="true"/, 'the sim canvases must be aria-hidden; the rail and log carry the information')
assert.match(flowBenchSource, /reducedMotion/, 'reduced motion must render a static field instead of animation')
assert.match(flowBenchSource, /visibilitychange/, 'the sim must pause on document.hidden')
assert.match(flowBenchSource, /subscribeLifecycle/, 'the sim must consume the shared lifecycle bus — no separate measurement path')
assert.doesNotMatch(flowBenchEngineSource, /--color-verified|--color-evidence/, 'copper and amber are claim colors and are forbidden in the fluid')
assert.doesNotMatch(flowBenchSource, /promptText|messageContent|\.content\b/, 'the sim consumes counts and timings only, never content')
assert.match(telemetryLogSource, /export function beginRequest/, 'request ids must be minted at send time so sim and metrics logs match one-to-one')

/* ---- Command palette + shortcuts (Phase 7) ---- */
const paletteSource = read('../src/components/CommandPalette.jsx')
const frontendReadmeSource = read('../README.md')
assert.match(appSource, /<CommandPalette/, 'the app must mount the command palette')
assert.match(appSource, /<ShortcutsOverlay/, 'the app must mount the shortcuts overlay')
assert.match(appSource, /lazy\(\(\) => import\('\.\/views\//, 'non-chat views must stay route-split')
assert.match(paletteSource, /readiness still gates send/, 'palette model switching must stay gate-honest')
assert.match(paletteSource, /camelid:open-ledger/, 'palette ledger jumps must use the shared deep-link event')
assert.match(frontendReadmeSource, /readiness-gate semantics are \*\*unchanged\*\*/, 'frontend README must state gate semantics are unchanged after the overhaul')

/* ---- Session telemetry (Phase 6) ---- */
assert.match(telemetryViewSource, /operational telemetry — not compatibility evidence/, 'every telemetry surface must carry the not-evidence affordance')
assert.match(telemetryViewSource, /useState\(false\)/, 'prompt reveal must default to redacted')
assert.match(telemetryViewSource, /•••• redacted/, 'redacted prompts must render visibly redacted')
assert.match(telemetryViewSource, /It never seeds or invents data/, 'the empty state must promise no synthetic data')
assert.doesNotMatch(telemetryViewSource, /EvidenceChip[\s\S]{0,300}(ttftMs|tokensPerSec|durationMs|medianT)/, 'perf numbers must never render inside Evidence Chips')
assert.doesNotMatch(telemetryLogSource, /Math\.random|seedData|sampleData|fakeData|demoData/, 'the telemetry store must have no synthetic data path')
assert.match(dashboardHookSource, /recordChatGeneration\(/, 'chat sends must feed the session telemetry store')
assert.match(dashboardHookSource, /recordHealthPoll\(/, 'health polls must feed the reachability history')
assert.match(apiWorkbenchSource, /recordWorkbenchRun\(/, 'workbench try-its must feed the session telemetry store')

/* Behavioral: export is path/content-free by whitelist even for salted records. */
const { recordChatGeneration: telRecord, exportTelemetryJson: telExport } = await import('../src/lib/telemetryLog.js')
telRecord({ modelId: 'salt-model', durationMs: 12, ttftMs: 5, outcome: 'ok', promptText: 'SECRET PROMPT /Volumes/Untitled/models/secret.gguf' })
const telExported = telExport()
assert.doesNotMatch(telExported, /SECRET PROMPT|\/Volumes\/|promptText/, 'telemetry exports must exclude prompt content and paths by whitelist')
assert.match(telExported, /salt-model/, 'telemetry exports keep whitelisted fields')
assert.match(telExported, /Not compatibility or support evidence/, 'telemetry exports must carry the not-evidence note')

/* ---- API workbench (Phase 5) ---- */
assert.match(apiViewSource, /<ApiWorkbench/, 'the API view must mount the workbench')
assert.match(apiViewSource, /chatUnlocked=\{selectedExactRowReady\}/, 'workbench generation gating must come from the shared exact-row chat gate')
assert.match(apiWorkbenchSource, /Requires a loaded supported model/, 'gated generation try-its must say they require a loaded supported model')
assert.match(apiWorkbenchSource, /gated exactly like chat/, 'the guarded copy must tie the workbench gate to the chat gate')
assert.match(apiWorkbenchSource, /operational telemetry — not compatibility evidence/, 'the request inspector must carry the telemetry-not-evidence banner')
assert.match(apiWorkbenchSource, /fail_closed/, 'fail-closed routes must render their typed guarded state')
assert.doesNotMatch(apiWorkbenchSource, /dangerouslySetInnerHTML/, 'inspector output must render as text')
/* lib/apiExamples.js is deliberately NOT in the brand sweep: code samples may
   name the SDK class they instantiate (technical compatibility content); UI
   copy may not. The sweep still covers the workbench component itself. */

/* ---- Compatibility ledger (Phase 4) ---- */
assert.match(compatibilityViewSource, /capabilities\?\.model_compatibility/, 'the ledger must render rows from the live contract only')
assert.match(compatibilityViewSource, /Not claimed/, 'the ledger must render the not-claimed column')
assert.match(compatibilityViewSource, /Resemblance is not evidence/, 'the ledger explainer must state that resemblance is not evidence')
assert.match(compatibilityViewSource, /Promotion path/, 'non-supported rows must show their promotion path from contract next_step copy')
assert.doesNotMatch(compatibilityViewSource, /supported_exact_row_smoke|supported_current_gate|tinyllama_|llama32_|llama3_|mistral|mixtral/i, 'the ledger source must contain zero hardcoded row ids or support statuses — the contract is the only voice')
assert.match(evidenceChipSource, /camelid:open-ledger/, 'Evidence Chips must deep-link to the ledger via the open-ledger event')
assert.match(appSource, /camelid:open-ledger/, 'the app shell must listen for ledger deep-links')
// (D14: the Models page no longer renders per-card ledger links; Evidence Chips carry
// the open-ledger deep link and the ledger stays one click away in the sidebar.)

/* ---- Analytics ---- */
assert.match(analyticsViewSource, /displayCapabilityId\(feature\.id\)/, 'Analytics view should not render raw provider-scoped API feature ids')

/* ---- TopBar (re-baselined to the Evidence Chip gate) ---- */
assert.match(topBarSource, /getChatGateState\(capabilities, selectedModel, runtime\)/, 'TopBar must derive its support claim from the shared chat gate')
assert.match(topBarSource, /<EvidenceChip/, 'TopBar support gate must render through the Evidence Chip')
assert.match(topBarSource, /className="topbar__gate"/, 'TopBar gate block must render on every tab, not only chat')
assert.doesNotMatch(topBarSource, /tab === 'chat' && !demoMode &&[\s\S]*topbar__gate/, 'TopBar gate visibility must not be restricted to the chat tab')
assert.match(topBarSource, /state=\{gate\.contractSupported \? 'supported'/, 'TopBar Evidence Chip supported state must come only from the shared gate contract flag')

/* ---- Evidence Chip system (Phase 1 contract) ---- */
assert.doesNotMatch(evidenceChipSource, /fetch\(|getChatGateState|isCompatibilitySupportedForModel|findCompatibilityHint/, 'EvidenceChip must stay purely presentational — it renders gate state, never computes or fetches it')
assert.match(evidenceStatusSource, /if \(value === 'supported' \|\| value\.startsWith\('supported_'\)\) return 'supported'/, 'only contract supported/supported_* statuses may classify into the copper supported state')
assert.match(evidenceCss, /\.ev-chip--supported\s*\{[^}]*var\(--color-verified\)/s, 'the supported chip state must use the reserved copper tokens')
for (const [path, css] of statusSheets) {
  assert.doesNotMatch(css, /var\(--color-verified\)/, `${path} must not spend the copper supported color on non-claim surfaces`)
}

/* ---- Runnable lane (Phase 7): unfakeable + unconfusable, never copper ---- */
const parityReceiptSource = read('../src/components/chat/render/ParityReceipt.jsx')
assert.match(evidenceStatusSource, /value === 'runnable' \|\| value\.startsWith\('runnable_'\)\) return 'runnable'/, 'runnable status must classify to its own state, never into copper supported')
assert.match(evidenceStatusSource, /EVIDENCE_STATES = \[\s*'supported',\s*'runnable'/, 'runnable must be a first-class evidence state, distinct from supported')
const runnableChipRule = evidenceCss.match(/\.ev-chip--runnable\s*\{[^}]*\}/s)
assert.ok(runnableChipRule, 'evidence.css must define the runnable chip state')
assert.doesNotMatch(runnableChipRule[0], /var\(--color-verified\)/, 'the runnable chip must never spend the reserved copper token')
assert.match(runnableChipRule[0], /var\(--color-evidence\)/, 'the runnable chip must use the amber evidence token (the 🟡 legend state)')
assert.match(parityReceiptSource, /execution_lane === 'runnable'/, 'the receipt card must detect the runnable lane from the receipt schema')
assert.match(parityReceiptSource, /Runnable lane/, 'a runnable receipt must be labelled distinctly from a supported parity receipt')
const runnableCardRule = chatCss.match(/\.parity-receipt-lane-badge\s*\{[^}]*\}/s)
assert.ok(runnableCardRule, 'chat.css must style the runnable lane badge')
assert.doesNotMatch(runnableCardRule[0], /var\(--color-verified\)/, 'the runnable lane badge must never be copper')

/* ---- Tokens, fonts, themes (Phase 1 contract) ---- */
assert.doesNotMatch(tokensCss, /@import url\(['"]?https?:/, 'tokens.css must not import from a CDN — the app renders fully offline')
assert.match(tokensCss, /:root \{\s*\n\s*color-scheme: dark;/, 'dark is the canonical :root palette (dark-first)')
assert.match(tokensCss, /--color-verified:/, 'tokens must define the reserved copper supported color')
assert.match(tokensCss, /--color-evidence:/, 'tokens must define the bounded-evidence amber distinct from copper')
assert.match(mainSource, /@fontsource-variable\/inter/, 'body font must be self-hosted via Fontsource')
assert.match(mainSource, /@fontsource\/ibm-plex-mono/, 'mono font must be self-hosted via Fontsource')
assert.doesNotMatch(mainSource, /fonts\.googleapis|fonts\.gstatic/, 'no third-party font CDN calls')
assert.match(useThemeSource, /return saved && VALID\.has\(saved\) \? saved : 'dark'/, 'theme preference must default to dark')

/* ---- Brand hygiene across visible UI sources ---- */
const visibleUiSources = [
  '../src/views/ChatWorkspace.jsx',
  '../src/views/ApiView.jsx',
  '../src/views/SystemView.jsx',
  '../src/views/ModelsView.jsx',
  '../src/hooks/useDashboardData.js',
  '../src/components/TopBar.jsx',
  '../src/components/chat/MessageTurn.jsx',
  '../src/components/ui/EvidenceChip.jsx',
  '../src/lib/evidenceStatus.js',
  '../src/lib/markdown.jsx',
  '../src/components/models/ModelInspector.jsx',
  '../src/components/models/TokenizerPlayground.jsx',
  '../src/views/CompatibilityView.jsx',
  '../src/components/api/ApiWorkbench.jsx',
  '../src/views/TelemetryView.jsx',
].map((path) => [path, read(path)])
for (const [path, source] of visibleUiSources) {
  assert.doesNotMatch(source, /\b(OpenAI|ChatGPT|Claude|Gemini)\b/, `${path} visible copy should not mention competitor brands`)
}

/* ---- Streaming visuals (current chat.css/ui.css truth) ---- */
assert.match(chatCss, /\.streaming-loader\s*\{[^}]*display:\s*inline-flex/s, 'streaming assistant rows should keep a dedicated loader')
assert.match(chatCss, /\.streaming-loader-dot\s*\{[^}]*border-radius:\s*50%[^}]*animation:\s*camelidDotBounce/s, 'streaming loader dots should animate only while the loader is rendered')
assert.match(chatCss, /\.streaming-loader-compact\s*\{[^}]*padding:\s*0 0 8px/s, 'compact streaming loader should sit above pre-token assistant content without extra copy')
assert.match(chatCss, /\.message-code-card\.is-generating\s*\{/, 'incomplete streaming code cards should have an active visual treatment')
assert.match(chatCss, /\.message-live-generation-badge\s*\{/, 'streaming assistant content should keep a visible active badge while the backend is generating')
assert.match(chatCss, /\.message-live-dot\s*\{[^}]*animation:\s*cxPulse/s, 'live generation badges should visibly pulse only while the badge is rendered')
assert.match(uiCss, /@keyframes cxPulse/, 'the live pulse keyframes must exist')
assert.match(tokensCss, /@keyframes camelidDotBounce/, 'the streaming dot bounce keyframes must exist')

/* ---- Catalog browse logic ----
   Runs the unit smoke for src/lib/catalogBrowse.js (quantization ordering and
   advice, repo grouping, default-quantization rules, the architecture partition,
   and the unchecked/settled/retryable split) inside this job.

   Imported here rather than added to the workflow because that is the frontend
   suite CI already gates; a behavioural module with no gate is a regression
   waiting to happen. The module asserts at import time and throws on failure. */
await import('./catalog-browse-smoke.mjs')

console.log('UI regression smoke passed (re-baselined Phase 2 pre-work)')
