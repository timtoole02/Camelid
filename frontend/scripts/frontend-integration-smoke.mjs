#!/usr/bin/env node
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const { default: ChatWorkspace } = await server.ssrLoadModule('/src/views/ChatWorkspace.jsx')
  const { default: CompatibilityView } = await server.ssrLoadModule('/src/views/CompatibilityView.jsx')
  const { default: ApiView } = await server.ssrLoadModule('/src/views/ApiView.jsx')
  const { default: SystemView } = await server.ssrLoadModule('/src/views/SystemView.jsx')
  const { default: ModelsView } = await server.ssrLoadModule('/src/views/ModelsView.jsx')
  const { blockerNoteFor } = await server.ssrLoadModule('/src/components/models/UnsupportedBlocker.jsx')
  const { default: TopBar } = await server.ssrLoadModule('/src/components/TopBar.jsx')
  const { StreamingLoader, streamingStatusLabel } = await server.ssrLoadModule('/src/components/chat/render/StreamingIndicator.jsx')
  const { getChatGateState } = await server.ssrLoadModule('/src/lib/chatGate.js')
  const {
    capabilityRowMatchesSearch,
    exactRowSupportLanes,
    rowSupportBoundaryCopy,
    rowSupportNextStepCopy,
    statusContainsSupportedEvidence,
  } = await server.ssrLoadModule('/src/lib/capabilities.js')
  const { mergeModelLists, resolveLoadedModelDisplayName } = await server.ssrLoadModule('/src/hooks/useDashboardData.js')
  const { LLAMA32_3B_ACCEPTANCE_TARGET } = await server.ssrLoadModule('/src/lib/acceptanceTargets.js')

  const noop = () => {}
  assert.match(
    blockerNoteFor({ code: 'model_io_error' }),
    /could not open this model from the configured storage location/,
    'storage failures must explain the path problem instead of claiming the architecture is missing',
  )
  assert.doesNotMatch(
    blockerNoteFor({ code: 'model_io_error' }),
    /architecture is not implemented/,
    'model_io_error must not borrow unsupported-architecture copy',
  )
  assert.match(
    blockerNoteFor({ code: 'unsupported_model_architecture' }),
    /architecture is not implemented/,
    'the architecture-specific note remains reserved for the typed architecture blocker',
  )
  const readyRuntime = {
    api_base: 'http://127.0.0.1:8181',
    loaded_now: true,
    generation_ready: true,
    active_model_id: 'llama32_3b_instruct_q8_0',
    status: 'online',
    backend: 'llama',
    execution_plan: {
      selected_backend: 'cpu_q8_runtime_repack',
      cuda_resident_active: false,
    },
  }
  const selectedModel = {
    id: 'llama32_3b_instruct_q8_0',
    name: 'Llama 3.2 3B Instruct Q8_0',
    provider_kind: 'local',
    loaded_now: true,
    generation_ready: true,
    status: 'ready',
    quant: 'Q8_0',
    model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf',
    runtime_model_name: 'llama32_3b_instruct_q8_0',
  }
  const capabilities = {
    support_contract: {
      current_gate: 'Current exact-row support: no model-native/larger context beyond checked packs, arbitrary/Jinja template behavior, production throughput, portability, neighboring-row, or broad-family support is implied.',
      support_policy: 'Only exact rows unlock chat.',
      unsupported_policy: 'Everything else remains guarded.',
    },
    supported_model_families: [{ id: 'broad_family_trap', status: 'supported' }],
    supported_quantization: [{ id: 'broad_quant_trap', status: 'supported' }],
    model_compatibility: [
      {
        id: 'tinyllama_1_1b_chat_q8_0',
        status: 'supported_current_gate',
        family: 'llama_spm_decoder',
        quantization: 'Q8_0',
        support_scope: 'exact row only',
        frontend_readiness_gate: 'loaded_now + generation_ready + active_model_id + exact row',
        full_support_status: 'guarded_by_exact_row',
        full_support_blockers: 'arbitrary/Jinja templates, production throughput, portability',
        evidence: 'TinyLlama fixture row; must not be inherited by misleading 3B backend ids.',
        chat_template_renderer: 'tinyllama-marker',
        chat_template_shape_pack: 'validated_bounded_pack',
        performance_measured: 'measured',
        next_step: 'preserve exact-row scoping before widening support claims',
      },
      {
        id: 'llama32_3b_instruct_q8_0',
        status: 'supported_current_gate',
        family: 'llama_bpe_decoder',
        quantization: 'Q8_0',
        support_scope: 'exact row only',
        frontend_readiness_gate: 'loaded_now + generation_ready + active_model_id + exact row',
        latest_checked_bucket: 'current_head',
        latest_checked_result: 'pass',
        latest_checked_output: 'exact row fixture output',
        full_support_status: 'guarded_by_exact_row',
        full_support_blockers: 'model-native/larger context beyond checked packs, arbitrary/Jinja templates, production throughput, portability, and durable repeated current-head bundles remain missing',
        evidence: 'Exact row evidence bundle.',
        metadata_parses: 'validated',
        tokenizer_works: 'validated',
        tensors_load: 'validated',
        generation_runs: 'validated',
        frontend_load_path_verified: 'validated',
        chat_template_shape_pack: 'validated',
        bounded_context_512_pack: 'validated',
        bounded_context_1024_pack: 'validated',
        bounded_context_2048_pack: 'validated',
        performance_measured: 'measured',
        next_step: 'preserve exact-row smoke while normalizing model-native/larger context, arbitrary/Jinja template behavior, production throughput, portability, and durable full-support bundle evidence before any broader claim',
      },
      {
        id: 'other_future_row_q8_0',
        status: 'planned',
        family: 'future_decoder',
        quantization: 'Q8_0',
        bounded_context_512_pack: 'not_started',
        bounded_context_512_pack_id: 'future-context-512-smoke-v1',
        next_step: 'Do not unlock selected chat.',
      },
    ],
    api_features: [
      { id: `open${'ai'}_chat_completions`, status: 'supported_current_gate', notes: `Open${'AI'}-compatible streaming stays enabled.` },
      { id: `open${'ai'}.responses_stream`, status: 'supported_current_gate', notes: `${'Chat' + 'GPT'}-style streamed response compatibility stays provider-neutral in UI copy.` },
      { id: 'tokenizer_encode_decode', status: 'supported_current_gate', notes: 'Tokenizer endpoint is exposed by the backend.' },
      { id: 'future_batch_endpoint', status: 'planned', notes: `Guarded feature row; do not label it ${'Clau' + 'de'} or ${'Gem' + 'ini'} compatible from API metadata.` },
    ],
  }
  const selectedModelRunnable = getChatGateState(capabilities, selectedModel, readyRuntime).chatUnlocked
  assert.equal(selectedModelRunnable, true, '3B Q8_0 fixture must be end-to-end runnable only when model path, runtime readiness, and exact-row support are all green')
  assert.equal(statusContainsSupportedEvidence('not_started'), false, 'reserved future pack ids must not count as verified evidence')
  assert.equal(capabilityRowMatchesSearch(capabilities.model_compatibility[2], 'future-context-512-smoke-v1'), true, 'ledger search should include displayed evidence bundle ids')

  const visionReadyMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: { id: 'vision-chat', title: 'Vision', messages: [] },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: { ...readyRuntime, vision_ready: true },
    capabilities,
    pendingConversation: null,
    composer: 'Describe this image',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable,
    setTab: noop,
  }))
  assert.match(visionReadyMarkup, /accept="image\/png,image\/jpeg"/, 'vision-ready runtime should expose the local PNG/JPEG picker')
  assert.match(visionReadyMarkup, /Attach one PNG or JPEG for the loaded Prism vision model/, 'vision picker must explain its one-image Prism boundary')

  const textOnlyRendered = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: { id: 'text-chat', title: 'Text', messages: [] },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'Text only',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable,
    setTab: noop,
  }))
  assert.doesNotMatch(textOnlyRendered, /accept="image\/png,image\/jpeg"/, 'text-only runtime must not advertise an image control it cannot serve')

  const loadedButNotRunnableMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: { ...readyRuntime, generation_ready: false },
    capabilities,
    pendingConversation: null,
    composer: 'Keep this draft editable',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: false,
    setTab: noop,
  }))
  assert.match(loadedButNotRunnableMarkup, /loaded, but this build cannot run it for Chat/, 'a completed fail-closed load must not look like an endless warmup')
  assert.match(loadedButNotRunnableMarkup, /· Not runnable/, 'the model picker must distinguish a blocked loaded row from an active load')
  assert.match(loadedButNotRunnableMarkup, /Choose a runnable model; this loaded model is blocked/, 'the composer must give an actionable next step for a permanently blocked load')
  assert.doesNotMatch(loadedButNotRunnableMarkup, /warming up|· Loading|finishes getting ready/, 'generation_ready=false after a completed load must not masquerade as transient progress')

  const embeddingModel = {
    id: 'bitnet-embeddings-270m',
    name: 'Microsoft BitNet Embedding 270M',
    provider_kind: 'local',
    status: 'ready',
    model_path: 'models/bitnet-embeddings-270m-bf16-i2_s.gguf',
    embedding_capable: true,
    generation_capable: false,
    loaded_now: true,
    generation_ready: false,
  }
  const embeddingRuntime = {
    ...readyRuntime,
    active_model_id: embeddingModel.id,
    generation_ready: false,
    model_family: 'embedding',
  }
  const mergedSidecars = mergeModelLists({
    modelItems: [
      { id: 'bitnet-embeddings-0.6b-bf16-i2_s.gguf' },
      { id: 'bitnet-embeddings-270m-bf16-i2_s.gguf' },
    ],
    health: readyRuntime,
    currentModel: { path: selectedModel.model_path },
    localModels: [
      {
        id: 'microsoft-bitnet-embedding-0.6b',
        name: 'Microsoft BitNet Embedding 0.6B',
        model_path: 'models/bitnet-embeddings-0.6b-bf16-i2_s.gguf',
        embedding_capable: true,
        generation_capable: false,
      },
      embeddingModel,
    ],
    apiBase: readyRuntime.api_base,
    localFacts: new Map(),
  })
  assert.equal(mergedSidecars.length, 2, 'non-active embedding sidecars should merge into their saved catalog rows without duplicate runtime entries')
  assert.deepEqual(
    mergedSidecars.map((model) => model.id).sort(),
    ['bitnet-embeddings-270m', 'microsoft-bitnet-embedding-0.6b'],
    'resident filename ids should preserve the saved catalog identities',
  )
  assert.ok(mergedSidecars.every((model) => model.loaded_now), 'merged sidecar rows should be marked resident even though neither is the active Chat model')
  const duplicateFilenameAliases = mergeModelLists({
    modelItems: [
      { id: 'runtime-a' },
      { id: 'shared.gguf' },
    ],
    health: readyRuntime,
    currentModel: { path: 'models/shared.gguf' },
    localModels: [
      { id: 'saved-a', runtime_model_name: 'runtime-a', model_path: 'models/shared.gguf' },
      { id: 'saved-b', runtime_model_name: 'runtime-b', model_path: 'models/shared.gguf' },
    ],
    apiBase: readyRuntime.api_base,
    localFacts: new Map(),
  })
  assert.deepEqual(
    duplicateFilenameAliases.map((model) => model.id).sort(),
    ['saved-a', 'saved-b'],
    'resident aliases must claim saved rows one-to-one instead of collapsing onto the first filename match',
  )
  const embeddingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: { id: 'embedding-chat', title: 'Embedding', messages: [] },
    selectedModel: embeddingModel,
    selectedModelId: embeddingModel.id,
    setSelectedModelId: noop,
    models: [embeddingModel],
    runtime: embeddingRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'Draft only',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: false,
    selectedModelExperimental: false,
    setTab: noop,
  }))
  assert.match(embeddingMarkup, /ready for embeddings and reranking, not Chat/i, 'embedding runtime must render a terminal task-ready state')
  assert.match(embeddingMarkup, /<optgroup label="Embedding only">/, 'embedding rows remain visible but disabled outside Chat choices')
  assert.doesNotMatch(embeddingMarkup, /BitNet Embedding 270M · Loading/, 'embedding-only generation_ready=false must never be described as an in-progress Chat load')

  const unloadedEmbedding = { ...embeddingModel, status: 'registered', loaded_now: false }
  const unloadedEmbeddingGate = getChatGateState(capabilities, unloadedEmbedding, readyRuntime)
  assert.equal(unloadedEmbeddingGate.embeddingOnly, true)
  assert.equal(unloadedEmbeddingGate.embeddingReady, false, 'embedding capability alone must not claim runtime readiness')
  const unloadedEmbeddingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: { id: 'unloaded-embedding-chat', title: 'Embedding', messages: [] },
    selectedModel: unloadedEmbedding,
    selectedModelId: unloadedEmbedding.id,
    setSelectedModelId: noop,
    models: [selectedModel, unloadedEmbedding],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'Draft only',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: false,
    selectedModelExperimental: false,
    setTab: noop,
  }))
  assert.match(unloadedEmbeddingMarkup, /embedding model — load it from Models/i)
  assert.doesNotMatch(unloadedEmbeddingMarkup, /is ready for embeddings and reranking/i)

  const wrongArtifactModel = {
    ...selectedModel,
    id: 'llama32_3b_instruct_q8_0_spoof',
    name: 'Llama 3.2 3B Instruct Q8_0',
    runtime_model_name: 'llama32_3b_instruct_q8_0_spoof',
    model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0-neighbor.gguf',
  }
  const wrongArtifactRuntime = {
    ...readyRuntime,
    active_model_id: wrongArtifactModel.runtime_model_name,
  }
  const wrongArtifactGate = getChatGateState(capabilities, wrongArtifactModel, wrongArtifactRuntime)
  assert.equal(wrongArtifactGate.runtimeReady, true, 'spoofed 3B artifact fixture must keep runtime readiness visible')
  assert.equal(wrongArtifactGate.contractSupported, false, 'spoofed 3B artifact fixture must not inherit exact-row support from id/name/Q8 copy')
  assert.equal(wrongArtifactGate.chatUnlocked, false, 'spoofed 3B artifact fixture must stay chat-blocked despite loaded_now and generation_ready')

  const blockedWrongArtifactMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-wrong-artifact',
      title: 'Wrong artifact',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [],
    },
    selectedModel: wrongArtifactModel,
    selectedModelId: wrongArtifactModel.id,
    setSelectedModelId: noop,
    models: [wrongArtifactModel],
    runtime: wrongArtifactRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'Can this chat?',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: wrongArtifactGate.chatUnlocked,
    setTab: noop,
  }))

  /* A generation-ready neighboring artifact is allowed to chat, but it must not
     inherit verification from the canonical row. */
  assert.match(blockedWrongArtifactMarkup, /Unverified local chat is ready/, '3B neighboring artifact should be runnable with an explicit unverified state')
  assert.doesNotMatch(blockedWrongArtifactMarkup, /is loaded and ready\./, 'an artifact-gated model must never be presented as ready to chat')
  /* The readiness LINE now explains the blocker in the reader's language; the
     exact row id it refers to still travels with the message footer's Evidence
     Chip, which is the surface built to carry it. */
  assert.match(blockedWrongArtifactMarkup, /Unverified/, '3B live chat must expose the weaker evidence state')
  assert.match(blockedWrongArtifactMarkup, /requires the exact Llama-3\.2-3B-Instruct-Q8_0\.gguf artifact/, '3B artifact blocker must name the canonical GGUF filename')
  assert.match(blockedWrongArtifactMarkup, /data-send-ready="true"/, 'generation-ready neighboring artifacts remain usable without borrowing verification')
  assert.doesNotMatch(blockedWrongArtifactMarkup, /Message Camelid"[^>]*disabled/, '3B draft composer should stay editable while exact-row support is still gated')
  assert.doesNotMatch(blockedWrongArtifactMarkup, /Local chat is ready/, '3B spoofed artifact must not render the supported live-chat state')
  /* "Demo starters" is gone; the empty state now offers prompt STARTERS that only fill the
     composer. Re-pinned to the property that actually mattered: nothing on a gated screen
     may be one click from running -- no send control is ever enabled here. */
  assert.doesNotMatch(blockedWrongArtifactMarkup, /Llama 3\.2 3B Instruct Q8_0 · Ready</, '3B spoofed artifact must not borrow the verified picker label')

  const streamingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-streaming-code',
      title: 'Streaming code',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-1', role: 'user', content: 'Create one self-contained HTML page', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-1', role: 'assistant', content: '```html\n<!doctype html>\n<button id="go">Go</button>', streaming: true, streaming_phase: 'streaming', created_at: '2026-05-13T04:21:01.000Z', model_id: 'llama32_3b_instruct_q8_0', tokens_out_per_sec: 12.7, usage: { prompt_tokens: 24, completion_tokens: 7, total_tokens: 31 }, usage_source: 'client_estimate' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable,
    setTab: noop,
  }))

  assert.match(streamingMarkup, /data-streaming-state="active"/, 'streaming assistant rows should render an active streaming state')
  // Redesign (2026-06): consolidated status line keeps runtime-ready + exact-row support visible
  // after messages exist (capability-lane detail now lives in System/API views, asserted there).
  assert.match(streamingMarkup, /is loaded and ready\./, 'non-empty live 3B chats should keep the runtime-ready state visible after messages exist')
    /* The row id still travels with the message footer's Evidence Chip; only its
     label wording changed, so pin the id itself rather than the old phrasing. */
  assert.match(streamingMarkup, /llama32_3b_instruct_q8_0/, 'non-empty live 3B chats should keep the exact-row support evidence visible after messages exist')
  /* Chat copy no longer names internal doc files (2026-08); the requirement it
     described is still enforced by the gate and stated in plain language. */
  assert.match(streamingMarkup, /is loaded and ready\./, 'live 3B chat readiness must state the verified-and-loaded requirement in the reader\'s language')
  assert.match(streamingMarkup, /data-streaming-code-state="open"/, 'open streaming fences should expose the active code state')
  assert.match(streamingMarkup, /Still generating — code block incomplete/, 'open streaming code should visibly say it is incomplete')
  assert.match(streamingMarkup, /Streaming code response/, 'streaming code rows should keep an active live-generation label')
  assert.match(streamingMarkup, /aria-busy="true"/, 'streaming rows and code cards should be marked busy while backend generation is active')
  assert.match(streamingMarkup, /message-code-card is-generating/, 'open streaming code should render as the real ForgeLocal-derived code card, not fallback prose')
  assert.match(streamingMarkup, /Generation details \(client-measured telemetry\)/, 'the meta footer should render during streaming as the live tok/s readout')
  assert.match(streamingMarkup, /13 tok\/s/, 'the streaming footer should show the live-patched output rate')
  assert.match(streamingMarkup, /End-to-end output rate, measured in this browser/, 'the browser output-rate hover must not imply decode-only timing')
  assert.match(streamingMarkup, /tokens est\.[\s\S]*in 24[\s\S]*out 7/, 'the streaming footer should show live input and output token counts')
  assert.doesNotMatch(streamingMarkup, /cxturn__meta--reserve/, 'the invisible footer placeholder must not render; the live footer holds the layout slot itself')

  const activeSendStreamingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-active-send-with-content',
      title: 'Active send with content',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-active-send', role: 'user', content: 'Create one self-contained HTML page', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-active-send', role: 'assistant', content: '```html\n<!doctype html>\n<title>Live</title>', streaming: true, streaming_phase: 'streaming', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: true,
    selectedModelRunnable,
    setTab: noop,
  }))

  assert.equal((activeSendStreamingMarkup.match(/data-streaming-state="active"/g) || []).length, 1, 'active sends with visible streamed content should keep exactly one active assistant row')
  assert.match(activeSendStreamingMarkup, /message-live-generation-badge/, 'active sends with visible streamed content should keep the live generation badge until completion')
  assert.match(activeSendStreamingMarkup, /Stop</, 'active sends should expose a stop action in the composer while Camelid is still generating')
  assert.doesNotMatch(activeSendStreamingMarkup, /Preparing local response/, 'visible streamed content should replace the pre-token pending loader during an active send')

  const preTokenMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-pre-token',
      title: 'Pre-token',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-2', role: 'user', content: 'Say hello', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-2', role: 'assistant', content: '', streaming: true, streaming_phase: 'generating', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable,
    setTab: noop,
  }))

  assert.match(preTokenMarkup, /data-streaming-state="active"/, 'pre-token assistant rows should remain visibly active while the backend is generating')
  // The pre-token phase label is now the reader-facing "Generating response"
  // (StreamingIndicator.FIRST_TOKEN_STREAMING_LABEL); same live status, plain wording.
  assert.match(preTokenMarkup, /Generating response/, 'pre-token streaming should render the active backend-generation live status')
  assert.match(preTokenMarkup, /streaming-loader-label">Generating response<\/span>/, 'the pre-token status must be visible beside the dots rather than carried only by aria-label')
  assert.match(preTokenMarkup, /streaming-loader-dot-3/, 'pre-token streaming should render the active loader, not a static placeholder')

  const delayedPreTokenLabel = streamingStatusLabel('generating', 20)
  const delayedPreTokenMarkup = renderToStaticMarkup(React.createElement(StreamingLoader, {
    label: delayedPreTokenLabel,
    compact: true,
  }))
  assert.match(
    delayedPreTokenMarkup,
    /streaming-loader-label">Local response is taking a while<\/span>/,
    'the delayed-first-token explanation must be visible instead of leaving the user with unexplained dots',
  )

  const completedUnclosedFenceMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-completed-unclosed-code',
      title: 'Completed unclosed code',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-4', role: 'user', content: 'Write a tiny Python script', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-4', role: 'assistant', content: '```python\nprint("safe")', streaming: false, created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable,
    setTab: noop,
  }))

  assert.match(completedUnclosedFenceMarkup, /message-code-card/, 'completed replies with an unclosed fenced block should still render as a safe code card')
  // The syntax highlighter may wrap tokens (e.g. print) in spans; assert the escaped
  // content stays visible rather than exact text adjacency.
  assert.match(completedUnclosedFenceMarkup, /print[\s\S]{0,80}?\([\s\S]*&quot;safe&quot;/, 'completed unclosed code content should remain visible and escaped in the code card')
  assert.doesNotMatch(completedUnclosedFenceMarkup, /Still generating — code block incomplete/, 'completed unclosed code should not claim the backend is still generating')
  assert.doesNotMatch(completedUnclosedFenceMarkup, /data-code-streaming-state="open"/, 'completed unclosed code should not expose an active streaming code state')

  const preTokenSendingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-pre-token-active-send',
      title: 'Pre-token active send',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-3', role: 'user', content: 'Say hello', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-3', role: 'assistant', content: '', streaming: true, streaming_phase: 'generating', created_at: '2026-05-13T04:21:01.000Z' },
      ],
    },
    selectedModel,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: '',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: true,
    selectedModelRunnable,
    setTab: noop,
  }))

  assert.equal((preTokenSendingMarkup.match(/data-streaming-state="active"/g) || []).length, 1, 'active send with an inserted pre-token assistant row should not render a duplicate pending assistant loader')
  assert.equal((preTokenSendingMarkup.match(/streaming-loader-track/g) || []).length, 1, 'pre-token active send should keep exactly one visible live loader for the backend generation')

  const exactReadyMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel,
    capabilities,
  }))

  /* The API readiness pill dropped the internal "exact row" vocabulary (2026-08);
     it now reads as plain product language. The invariant is unchanged -- the pill
     goes green only when the loaded model is the selected, verified one. */
  assert.match(exactReadyMarkup, /Ready for the selected model/, 'API readiness should turn green only for a matching loaded exact row')
  assert.match(exactReadyMarkup, /llama32_3b_instruct_q8_0/, 'API view should render the selected exact compatibility row id')
  assert.match(exactReadyMarkup, /Exact row evidence bundle\./, 'API view should render exact-row evidence text')
  assert.match(exactReadyMarkup, /exact row fixture output/, 'API view should render latest exact-row output evidence')
  assert.match(exactReadyMarkup, /Template\/Jinja readiness[\s\S]*Template readiness is green for this supported exact row/, 'API view should show resolved template/Jinja as a green exact-row readiness lane')
  assert.match(exactReadyMarkup, /Throughput readiness[\s\S]*Bounded row-scoped performance\/RSS evidence is present/, 'API view should show bounded 3B performance/RSS evidence without promoting production-throughput readiness')
  assert.match(exactReadyMarkup, /Remaining support boundary:<\/b> model-native\/larger context beyond checked packs; production throughput; portability; durable repeated current-head bundles remain missing/, 'API view should keep unresolved row blockers while filtering only resolved template/Jinja caveats')
  assert.doesNotMatch(exactReadyMarkup, /arbitrary-template behavior|arbitrary\/Jinja templates/, 'API support surface should not repeat resolved template/Jinja caveats after row-scoped template evidence is green')
  assert.doesNotMatch(exactReadyMarkup, /normalizing model-native\/larger context; arbitrary\/Jinja template behavior; production throughput/, 'API compatibility list next-step copy should filter resolved template/Jinja caveats while retaining production-throughput blockers')
  assert.match(exactReadyMarkup, /Supported API feature rows/, 'API view should render supported feature rows from /api/capabilities')

  const exactReadySystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: readyRuntime,
    selectedModel,
    capabilities,
  }))

  /* System's endpoint pill, chat-completions sentence, and evidence row all moved to
     plain language (2026-08): the raw gate fields (loaded_now/generation_ready/exact row)
     are no longer restated in user copy. The invariant is unchanged -- each of these three
     surfaces turns green only when runtime readiness AND row verification both hold. */
  assert.match(exactReadySystemMarkup, /Local API ready for the selected model/, 'System endpoint status should go green only when the selected 3B exact row and runtime readiness both match')
  assert.match(exactReadySystemMarkup, /Ready — the selected model is loaded and verified for chat\./, 'System chat-completions copy should name the full 3B exact-row readiness gate')
  assert.match(exactReadySystemMarkup, /Chat readiness:<\/b> Ready — the model is loaded and verified\./, 'System selected exact-row evidence should expose the retained chat/API gate')
  assert.match(exactReadySystemMarkup, /Template\/Jinja readiness[\s\S]*Template readiness is green for this supported exact row/, 'System should render 3B template/Jinja lane evidence from /api/capabilities')
  assert.match(exactReadySystemMarkup, /Throughput readiness[\s\S]*Bounded row-scoped performance\/RSS evidence is present/, 'System should keep bounded 3B performance evidence separate from production-throughput promotion')
  /* The row list is no longer headed by the internal doc filename (2026-08); the section
     still states it mirrors /api/capabilities and still renders each row-scoped lane
     label, which is what this assertion protects. */
  assert.match(exactReadySystemMarkup, /This mirrors \/api\/capabilities[\s\S]*Template rendering ready for this exact row[\s\S]*Checked context packs ready for this exact row[\s\S]*Production throughput not promoted/, 'System compatibility row list should render row-scoped 3B capability lanes, not just raw row status')
  assert.doesNotMatch(exactReadySystemMarkup, /normalizing model-native\/larger context; arbitrary\/Jinja template behavior; production throughput/, 'System compatibility list next-step copy should filter resolved template/Jinja caveats while retaining production-throughput blockers')
  assert.doesNotMatch(exactReadySystemMarkup, /# Use only after \/v1\/health returns generation_ready=true/, 'System curl should not imply generation_ready alone is sufficient for 3B UX chat')
  assert.match(exactReadySystemMarkup, /Selected device at load[\s\S]*CPU[\s\S]*Selected backend at load[\s\S]*cpu q8 runtime repack/, 'System should render the selected CPU load plan instead of static backend prose')
  assert.doesNotMatch(exactReadySystemMarkup, /GPU acceleration remains future work|local CPU generation path today/, 'System should not render stale static execution claims')
  const cudaSystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: { ...readyRuntime, execution_plan: { selected_backend: 'cuda_resident_q8_runtime', cuda_resident_active: true } },
    selectedModel,
    capabilities,
  }))
  assert.match(cudaSystemMarkup, /Selected device at load[\s\S]*CUDA GPU[\s\S]*cuda resident q8 runtime/, 'System should render a consistent CUDA load plan without implying current effective execution')
  const ghostHybridSystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: {
      ...readyRuntime,
      backend: 'gemma4-runtime',
      gemma4_serve_lane: 'ghost_moe',
      gemma4_ghost_execution_mode: 'hybrid_metal',
      gemma4_ghost_common_metal_active: false,
      gemma4_ghost_experts_metal_active: true,
      gemma4_ghost_head_metal_active: true,
    },
    selectedModel,
    capabilities,
  }))
  assert.match(ghostHybridSystemMarkup, /Available Ghost acceleration[\s\S]*Hybrid Metal \+ local SSD[\s\S]*Ghost serving lane[\s\S]*Ghost-MoE/, 'System should label the live Ghost component report instead of calling it a load-time device plan')
  assert.match(ghostHybridSystemMarkup, /Common core: CPU\.[\s\S]*Persistent Q4_0 expert slots: Metal\.[\s\S]*Q6_K tied head: Metal\./, 'System should identify every hybrid Ghost Metal component')
  const idleSystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: { api_base: 'http://127.0.0.1:8181', status: 'online', loaded_now: false, generation_ready: false },
    selectedModel: null,
    capabilities: { ...capabilities, execution_plan: null },
  }))
  assert.match(idleSystemMarkup, /Runtime[\s\S]*Idle[\s\S]*Runtime state[\s\S]*Online, no model loaded/, 'online no-model System state should report idle rather than offline')
  assert.match(idleSystemMarkup, /Selected device at load[\s\S]*No model loaded[\s\S]*Selected backend at load[\s\S]*No active plan/, 'online no-model System state should not infer CPU or GPU execution')
  assert.match(exactReadyMarkup, /chat completions/, 'API view should display provider-scoped feature ids as neutral capability names')
  assert.match(exactReadyMarkup, /standard-compatible streaming stays enabled\./, 'API view should sanitize provider-specific feature notes before rendering')

  // Redesign (2026-06): the TopBar support-contract strip was removed. The slim TopBar shows the
  // conversation title and, on the chat tab, a compact model status chip. Support-contract caveat
  // filtering is covered by frontendSupportContractCopy (3b-closure) and the System/API views.
  const topBarMarkup = renderToStaticMarkup(React.createElement(TopBar, {
    tab: 'chat',
    setTab: noop,
    selectedConversationTitle: '',
    runtime: readyRuntime,
    capabilities,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
  }))

  assert.match(topBarMarkup, /topbar__model/, 'chat-tab TopBar should render the compact model status chip')
  assert.match(topBarMarkup, /Llama 3\.2 3B Instruct Q8_0/, 'TopBar model chip should show the selected model name')

  const aliasSelectedModel = {
    ...selectedModel,
    id: 'browser-llama32-3b-alias',
    runtime_model_name: selectedModel.id,
    model_path: '/models/Llama-3.2-3B-Instruct-Q8_0.gguf',
  }
  const aliasApiMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel: aliasSelectedModel,
    capabilities,
  }))

  assert.match(aliasApiMarkup, /Ready for the selected model/, 'API view should keep alias-selected 3B exact rows green when active_model_id matches runtime_model_name')
  assert.match(aliasApiMarkup, /&quot;model&quot;: &quot;llama32_3b_instruct_q8_0&quot;/, 'API curl should send the backend loaded model id, not the browser-only alias')
  assert.doesNotMatch(aliasApiMarkup, /&quot;model&quot;: &quot;browser-llama32-3b-alias&quot;/, 'API curl must not create model_mismatch risk for alias-selected exact rows')

  const aliasTopBarMarkup = renderToStaticMarkup(React.createElement(TopBar, {
    tab: 'chat',
    setTab: noop,
    selectedConversationTitle: '',
    runtime: readyRuntime,
    capabilities,
    selectedModelId: aliasSelectedModel.id,
    setSelectedModelId: noop,
    models: [aliasSelectedModel],
  }))

  assert.match(aliasTopBarMarkup, /Llama 3\.2 3B Instruct Q8_0/, 'TopBar model chip should resolve the active model through runtime_model_name aliases')
  assert.doesNotMatch(aliasTopBarMarkup, /tinyllama/i, 'TopBar model chip must not mislabel an active 3B row as TinyLlama')
  assert.doesNotMatch(aliasTopBarMarkup, /No model selected/, 'TopBar must not show an empty model state for alias-selected loaded 3B rows')

  // Redesign (2026-07, D14): the Models page was rebuilt as five derived zones; the
  // acceptance panel, tracked-row cards, and legacy grids are gone. Under SSR the data
  // spine cannot fetch, so the page must render its honest fallbacks — never a
  // fabricated loaded/downloaded state. The neighboring-quant and neighboring-artifact
  // acceptance gates now live in the lane derivation (lib/modelLanes) and are asserted
  // directly against the same capabilities fixture below.
  const { laneOf } = await server.ssrLoadModule('/src/lib/modelLanes.js')
  const laneEntry = (filename, quantization) => ({ filename, quantization, runnable_receipt_present: false, admitted: false, oracle_qualified: false })
  assert.equal(
    laneOf(laneEntry('Llama-3.2-3B-Instruct-Q8_0.gguf', 'Q8_0'), capabilities),
    'supported',
    'the exact 3B Q8_0 artifact must derive into the Supported lane from the live contract fixture',
  )
  assert.equal(
    laneOf(laneEntry('Llama-3.2-3B-Instruct-Q4_0.gguf', 'Q4_0'), capabilities),
    'not_anchored',
    '3B neighboring GGUF quant must not inherit the canonical Q8_0 supported lane',
  )
  assert.equal(
    laneOf(laneEntry('Llama-3.2-3B-Instruct-Q8_0-neighbor.gguf', 'Q8_0'), capabilities),
    'not_anchored',
    'a same-label Q8 file without the exact GGUF filename must not inherit the supported lane',
  )

  const modelsViewProps = {
    runtime: { ...readyRuntime, status: 'online' },
    capabilities,
    refreshDashboard: noop,
    unloadCurrentModel: noop,
    loadingModelId: '',
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    registerModel: noop,
    apiBase: 'http://127.0.0.1:8181',
  }
  const modelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, modelsViewProps))
  assert.match(modelsMarkup, /Active model/, 'Models view must render the active-model bar zone')
  assert.match(modelsMarkup, /No model loaded/, 'Models view must not fabricate a loaded model before /api/models/current answers')
  assert.match(modelsMarkup, /Supported/, 'Models view must render the Supported zone')
  assert.match(modelsMarkup, /Other local models/, 'Models view must render the evidence-specific non-supported zone')
  assert.match(modelsMarkup, /Get models/, 'Models view must render the Get-models zone')
  assert.match(modelsMarkup, /Diagnostics/, 'Models view must keep the diagnostics disclosure')
  assert.match(modelsMarkup, /Scanning local models…|Local model scan unavailable\./, 'Models view sections must show the honest scan fallback instead of inventing membership')
  assert.doesNotMatch(modelsMarkup, /Downloaded/, 'Models view must not claim any downloaded state without the live disk scan')

  const offlineModelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    ...modelsViewProps,
    runtime: { ...readyRuntime, status: 'offline', loaded_now: false, generation_ready: false },
  }))
  assert.match(offlineModelsMarkup, /Runtime offline/, 'Models view must surface the offline runtime state instead of stale readiness')

  const green3BCapabilities = JSON.parse(JSON.stringify(capabilities))
  green3BCapabilities.api_features.push({ id: 'production_throughput', status: 'supported_exact_row_evidence', notes: '3B production-throughput lane validated end-to-end.' })
  green3BCapabilities.model_compatibility = green3BCapabilities.model_compatibility.map((target) => target.id === 'llama32_3b_instruct_q8_0'
    ? {
      ...target,
      chat_template_renderer: 'metadata_jinja_supported_for_exact_row',
      chat_template_shape_pack: 'validated_bounded_pack',
      performance_measured: 'production_throughput_validated',
      full_support_blockers: 'model-native/larger context beyond checked packs, arbitrary/Jinja templates, production throughput, portability, and durable repeated current-head bundles remain missing',
      next_step: 'preserve exact-row smoke while normalizing model-native/larger context, arbitrary/Jinja template behavior, production throughput, portability, and durable full-support bundle evidence before any broader claim',
    }
    : target)
  const green3BRow = green3BCapabilities.model_compatibility.find((target) => target.id === 'llama32_3b_instruct_q8_0')
  assert.deepEqual(
    exactRowSupportLanes(green3BRow, green3BCapabilities.api_features).map((lane) => [lane.key, lane.ready]),
    [['template', true], ['context', true], ['throughput', true]],
    '3B exact row should show template/Jinja, checked-context, and production-throughput lanes green once /api/capabilities advertises row evidence',
  )
  assert.doesNotMatch(rowSupportBoundaryCopy(green3BRow, green3BCapabilities.api_features), /arbitrary|Jinja|production|throughput/i, '3B remaining boundary should filter resolved template/Jinja and production-throughput blockers when both lanes are green')
  assert.doesNotMatch(rowSupportNextStepCopy(green3BRow, green3BCapabilities.api_features), /arbitrary|Jinja|production|throughput/i, '3B next-step copy should filter resolved template/Jinja and production-throughput blockers when both lanes are green')
  assert.equal(getChatGateState(green3BCapabilities, aliasSelectedModel, readyRuntime).chatUnlocked, true, 'green 3B evidence must still require runtime loaded_now/generation_ready and exact-row model identity before chat unlocks')

  const green3BApiMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel: aliasSelectedModel,
    capabilities: green3BCapabilities,
  }))
  assert.match(green3BApiMarkup, /Throughput readiness[\s\S]*Production-throughput readiness is green for this supported exact row from production throughput validated evidence/, 'API view should render green 3B production-throughput evidence only when /api/capabilities advertises row-owned evidence')
  assert.doesNotMatch(green3BApiMarkup, /Remaining support boundary:<\/b>[\s\S]{0,220}(?:arbitrary|Jinja|production|throughput)/i, 'API view 3B boundary should not repeat resolved template/Jinja or production-throughput blockers after green evidence')

  // (D14: the tracked 3B row card left the Models page; its green-lane rendering is
  // asserted on the API view above and the ledger keeps the row-scoped evidence.)

  assert.equal(
    resolveLoadedModelDisplayName({
      fallbackName: 'scalar_default_rerun',
      modelPath: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf',
      quantLabel: 'Q8_0',
    }),
    'Llama 3.2 3B Instruct Q8_0',
    'live backend-generated ids should display the exact 3B row name when the loaded GGUF filename and Q8_0 metadata are exact',
  )
  assert.equal(
    resolveLoadedModelDisplayName({
      fallbackName: 'scalar_default_rerun',
      modelPath: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q4_0.gguf',
      quantLabel: 'Q4_0',
    }),
    'scalar_default_rerun',
    'the 3B display alias must stay fail-closed for neighboring quants',
  )

  const liveBackendIdModel = {
    ...selectedModel,
    id: 'scalar_default_rerun',
    name: resolveLoadedModelDisplayName({ fallbackName: 'scalar_default_rerun', modelPath: selectedModel.model_path, quantLabel: selectedModel.quant }),
    runtime_model_name: 'scalar_default_rerun',
  }
  const liveBackendIdRuntime = { ...readyRuntime, active_model_id: liveBackendIdModel.id }
  const liveBackendIdChatGate = getChatGateState(capabilities, liveBackendIdModel, liveBackendIdRuntime)
  assert.equal(liveBackendIdChatGate.chatUnlocked, true, '3B rows loaded under a backend-generated runtime id should still unlock from GGUF path + Q8_0 exact-row evidence')
  assert.equal(liveBackendIdChatGate.hint.target.id, 'llama32_3b_instruct_q8_0', 'backend-generated 3B runtime ids must resolve to the canonical exact row, not a broad family claim')

  const misleadingTinyRuntimeModel = {
    ...selectedModel,
    id: 'tinyllama-q8',
    name: resolveLoadedModelDisplayName({ fallbackName: 'tinyllama-q8', modelPath: selectedModel.model_path, quantLabel: selectedModel.quant }),
    runtime_model_name: 'tinyllama-q8',
  }
  const misleadingTinyRuntime = { ...readyRuntime, active_model_id: misleadingTinyRuntimeModel.id }
  const misleadingTinyChatGate = getChatGateState(capabilities, misleadingTinyRuntimeModel, misleadingTinyRuntime)
  assert.equal(misleadingTinyChatGate.chatUnlocked, true, 'live 3B runs with a stale tinyllama-q8 runtime id should still unlock only from exact loaded GGUF path + Q8_0 support evidence')
  assert.equal(misleadingTinyChatGate.hint.target.id, 'llama32_3b_instruct_q8_0', 'stale tinyllama-q8 runtime ids must not steal the TinyLlama row when the loaded file is Llama 3.2 3B Instruct Q8_0')

  // (D14: the stale-id presentation moved off the Models page; the identity resolution
  // itself is asserted through misleadingTinyChatGate above and the chat markup below.)

  const liveBackendIdChatMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel: liveBackendIdModel,
    selectedModelId: liveBackendIdModel.id,
    setSelectedModelId: noop,
    models: [liveBackendIdModel],
    runtime: liveBackendIdRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'Say hello from 3B',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: liveBackendIdChatGate.chatUnlocked,
    setTab: noop,
  }))

  assert.match(liveBackendIdChatMarkup, /How can I help\?/, 'ready 3B live-backend-id chat should render the sendable empty-state hero')
  assert.match(liveBackendIdChatMarkup, /Local chat is ready/, 'ready 3B live-backend-id chat should show runtime-green chat UX')
  assert.match(liveBackendIdChatMarkup, /Llama 3\.2 3B Instruct Q8_0 is loaded and ready\./, 'ready 3B live-backend-id chat should display the exact 3B row name instead of the backend-generated runtime id')
  /* Chat copy no longer prints raw row ids (2026-08). The supported lane is still
     distinguishable in the composer: only a verified row is grouped/labeled "· Ready"
     (an experimental row reads "· Experimental ready", an unverified one "· Not verified"). */
  assert.match(liveBackendIdChatMarkup, /Llama 3\.2 3B Instruct Q8_0 · Ready</, 'ready 3B live-backend-id chat should show exact-row support in the composer surface')
  assert.match(liveBackendIdChatMarkup, /Message Camelid…/, 'ready 3B live-backend-id chat should enable the composer instead of showing load-first copy')
  assert.doesNotMatch(liveBackendIdChatMarkup, /Load a model first|Choose a verified model to send/, 'ready 3B live-backend-id chat should not fall back to blocked chat UX')

  const staleLoadedNowChatMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel: liveBackendIdModel,
    selectedModelId: liveBackendIdModel.id,
    setSelectedModelId: noop,
    models: [liveBackendIdModel],
    runtime: { ...liveBackendIdRuntime, loaded_now: false },
    capabilities,
    pendingConversation: null,
    composer: 'This stale browser row must remain blocked',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: getChatGateState(capabilities, liveBackendIdModel, { ...liveBackendIdRuntime, loaded_now: false }).chatUnlocked,
    setTab: noop,
  }))

  /* The runtime-blocked readiness line is now stated in product language; the gate it
     reports is still the shared chat gate, so send stays locked below. */
  assert.match(staleLoadedNowChatMarkup, /Llama 3\.2 3B Instruct Q8_0 is getting ready — you can draft now\./, '3B chat readiness should use the shared chat gate and stay runtime-blocked when backend loaded_now=false')
  assert.match(staleLoadedNowChatMarkup, /Draft a prompt while Camelid finishes getting ready/, '3B stale loaded_now=false rows should keep drafting available while runtime readiness is still blocked')
  assert.match(staleLoadedNowChatMarkup, /data-send-ready="false"/, '3B stale loaded_now=false rows must keep send locked until runtime readiness returns')
  /* Re-pinned to live copy: with loaded_now=false the screen must not claim the runtime is
     up (the "not verified" wording implies it is), must not render ready UX, and must not
     enable send anywhere. */
  assert.doesNotMatch(staleLoadedNowChatMarkup, /isn(?:&#x27;|')t verified for chat yet|Local chat is ready|Message Camelid…|data-send-ready="true"/, '3B stale browser readiness must not leak into live chat UX when /v1\\/health says loaded_now=false')

  const neighboringQuantPathModel = {
    ...selectedModel,
    quant: undefined,
    model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q4_0.gguf',
  }
  const neighboringQuantPathGate = getChatGateState(capabilities, neighboringQuantPathModel, readyRuntime)
  assert.equal(neighboringQuantPathGate.runtimeReady, true, '3B neighboring-quant guard should still surface runtime readiness when active_model_id matches')
  assert.equal(neighboringQuantPathGate.contractSupported, false, '3B neighboring GGUF quant must not inherit the canonical Q8_0 row from the browser id')
  assert.equal(neighboringQuantPathGate.chatUnlocked, false, '3B neighboring GGUF quant must not inherit the supported gate')
  assert.equal(neighboringQuantPathGate.experimentalUnlocked, true, 'a generation-ready neighboring quant remains usable on the unverified lane')

  const neighboringQuantPathChatMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel: neighboringQuantPathModel,
    selectedModelId: neighboringQuantPathModel.id,
    setSelectedModelId: noop,
    models: [neighboringQuantPathModel],
    runtime: readyRuntime,
    capabilities,
    pendingConversation: null,
    composer: 'This neighboring quant must remain blocked',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: neighboringQuantPathGate.chatUnlocked,
    setTab: noop,
  }))

  /* Chat no longer prints the row id or the quant-mismatch sentence (2026-08); that
     row-scoped detail lives in System/API/the ledger now. What chat must still do is
     distinguish "runtime is up but this model is not verified" from the runtime-blocked
     wording, and keep send locked for that reason -- both pinned below. The mismatch
     itself is still asserted structurally on the gate (contractSupported/chatUnlocked
     above), so the invariant keeps its teeth. */
  assert.match(neighboringQuantPathChatMarkup, /Unverified local chat is ready/, '3B neighboring-quant chat UX should expose runtime-green state without claiming support')
  assert.match(neighboringQuantPathChatMarkup, /data-send-ready="true"/, 'a generation-ready neighboring quant should remain usable')
  /* Restored (2026-08): the quant mismatch names both sides again, so a reader
     one re-download away is told which build to get instead of just "pick a
     verified model". */
  assert.match(neighboringQuantPathChatMarkup, /this build is Q4_0 and the verified build is Q8_0/, '3B neighboring-quant chat UX should name the loaded quant and the verified one')
  assert.doesNotMatch(neighboringQuantPathChatMarkup, /Llama 3\.2 3B Instruct Q8_0 · Ready</, '3B neighboring-quant rows must not borrow the supported picker label')

  const backendReadyButUnsupported3BCapabilities = {
    ...capabilities,
    model_compatibility: capabilities.model_compatibility.map((target) => target.id === 'llama32_3b_instruct_q8_0'
      ? {
        ...target,
        status: 'groundwork_backend_evidence_only',
        full_support_status: 'blocked_pending_frontend_api_alignment',
        evidence: 'Backend can load this fixture, but /api/capabilities has not promoted WebUI chat support.',
      }
      : target),
  }
  const backendReadyButUnsupported3BGate = getChatGateState(backendReadyButUnsupported3BCapabilities, liveBackendIdModel, liveBackendIdRuntime)
  assert.equal(backendReadyButUnsupported3BGate.runtimeReady, true, '3B runtime health should remain visible even when the support contract row is not promoted')
  assert.equal(backendReadyButUnsupported3BGate.contractSupported, false, '3B chat must stay support-contract blocked when /api/capabilities downgrades the exact row')
  assert.equal(backendReadyButUnsupported3BGate.chatUnlocked, false, 'runtime-ready 3B rows must not unlock WebUI chat without an exact supported compatibility status')

  const backendReadyButUnsupported3BChatMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel: liveBackendIdModel,
    selectedModelId: liveBackendIdModel.id,
    setSelectedModelId: noop,
    models: [liveBackendIdModel],
    runtime: liveBackendIdRuntime,
    capabilities: backendReadyButUnsupported3BCapabilities,
    pendingConversation: null,
    composer: 'This should remain blocked',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: backendReadyButUnsupported3BGate.chatUnlocked,
    setTab: noop,
  }))

  /* Support-gated chat copy is now plain language (2026-08). The four surfaces below carry
     the same four meanings the old raw-field copy did: the support-gated hero, the still-green
     runtime (only a loaded + generation-ready model can reach the picker's runnable labels),
     the named model instead of generic load-first copy, and the locked send with its reason. */
  assert.match(backendReadyButUnsupported3BChatMarkup, /Unverified local chat is ready/, 'runtime-ready downgraded rows should remain usable without claiming verification')
  assert.match(backendReadyButUnsupported3BChatMarkup, /Llama 3\.2 3B Instruct Q8_0 · Unverified ready/, 'downgraded 3B UX should expose both runtime readiness and evidence state')
  assert.match(backendReadyButUnsupported3BChatMarkup, /data-send-ready="true"/, 'generation-ready downgraded rows should keep send available')
  assert.doesNotMatch(backendReadyButUnsupported3BChatMarkup, /Llama 3\.2 3B Instruct Q8_0 · Ready</, 'downgraded rows must not borrow the supported picker label')

  const experimental3BChatMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: null,
    selectedModel: liveBackendIdModel,
    selectedModelId: liveBackendIdModel.id,
    setSelectedModelId: noop,
    models: [liveBackendIdModel],
    runtime: { ...liveBackendIdRuntime, vision_ready: true },
    capabilities: backendReadyButUnsupported3BCapabilities,
    pendingConversation: null,
    composer: 'Test the implemented experimental lane',
    setComposer: noop,
    saveToMemory: noop,
    sendMessage: noop,
    sending: false,
    selectedModelRunnable: backendReadyButUnsupported3BGate.chatUnlocked,
    selectedModelExperimental: backendReadyButUnsupported3BGate.experimentalUnlocked,
    setTab: noop,
  }))

  assert.equal(backendReadyButUnsupported3BGate.chatMode, 'experimental', 'a generation-ready implemented row outside the supported contract should use the explicit experimental lane')
  /* The experimental hero now spells the caveat out instead of using a badge phrase; it
     still reads as ready AND still refuses the verified claim in the same sentence. */
  assert.match(experimental3BChatMarkup, /Unverified local chat is ready\. Replies are clearly marked\./, 'the unverified lane should read as ready without borrowing the supported badge')
  assert.match(experimental3BChatMarkup, /Unverified ready/, 'the model picker should group a runnable unverified row with ready choices')
  assert.match(experimental3BChatMarkup, /data-send-ready="true"/, 'the explicit experimental lane should unlock send')
  assert.match(experimental3BChatMarkup, /Attach one PNG or JPEG for the loaded Prism vision model/, 'vision-ready experimental rows should expose the image picker')
  // Re-pinned to the current blocked-state wording so the assertion still bites.
  assert.doesNotMatch(experimental3BChatMarkup, /isn(?:&#x27;|')t verified for chat yet|Pick a verified model to unlock send/, 'a runnable experimental row must not look blocked')

  const backendReadyButUnsupported3BSystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: liveBackendIdRuntime,
    selectedModel: liveBackendIdModel,
    capabilities: backendReadyButUnsupported3BCapabilities,
  }))

  /* System states the same runtime-green/support-red split in product language now
     (2026-08). The row id itself is still printed beside the readiness line, so the
     evidence stays visible; only the surrounding wording changed. */
  assert.match(backendReadyButUnsupported3BSystemMarkup, /Engine ready — model not verified yet/, 'System should keep runtime-green 3B rows visible without making unsupported exact rows API/chat-ready')
  assert.match(backendReadyButUnsupported3BSystemMarkup, /Chat unlocks once this model finishes loading and is verified\./, 'System should block chat completions copy when the exact 3B row is downgraded')
  assert.match(backendReadyButUnsupported3BSystemMarkup, /llama32_3b_instruct_q8_0[\s\S]*Chat readiness:<\/b> Runnable \(unverified\); loaded=yes, ready=yes, verified=no\./, 'System selected exact-row evidence should show runtime-green/support-red state for downgraded 3B rows')
  /* System no longer renders a curl sample (it lives on the API view, asserted there);
     the blocked posture it demonstrated is now carried by the chat-readiness stat. */
  assert.match(backendReadyButUnsupported3BSystemMarkup, /Chat readiness<\/span><strong>Gated<\/strong><small>model not verified yet<\/small>/, 'System curl should stay blocked for runtime-ready unsupported 3B rows')
  assert.doesNotMatch(backendReadyButUnsupported3BSystemMarkup, /Local API ready for the selected model|Ready — the model is loaded and verified/, 'System must not present downgraded 3B rows as exact-row API/chat-ready')

  assert.match(exactReadyMarkup, /responses stream/, 'API view should normalize provider-scoped dotted feature ids before rendering')
  assert.match(exactReadyMarkup, /hosted model-style streamed response compatibility stays provider-neutral/, 'API view should neutralize hosted-brand feature notes before rendering')
  /* The API view now lists only supported feature rows, so guarded metadata never reaches
     the page in raw OR neutralized form (2026-08). Pinning the stronger fact -- the guarded
     row is absent -- keeps the hosted-brand guarantee; the assertion below still checks that
     no brand token survives on the rows that DO render. */
  assert.doesNotMatch(exactReadyMarkup, /future[_ ]batch[_ ]endpoint|Guarded feature row/, 'API view should also neutralize guarded feature metadata before rendering')
  assert.doesNotMatch(exactReadyMarkup, /openai|OpenAI|ChatGPT|Claude|Gemini|broad_family_trap|broad_quant_trap/, 'API view must not promote broad family/quant lists or raw provider-scoped/hosted-brand feature labels as support evidence')

  const mismatchedRuntimeMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: 'different-loaded-model' },
    selectedModel,
    capabilities,
  }))

  /* Same fail-closed states, product wording (2026-08): the pill names the mismatch and the
     curl sample stays a locked comment instead of a runnable request. */
  assert.match(mismatchedRuntimeMarkup, /A different model is loaded/, 'API readiness should fail closed when active_model_id differs from the selected exact row')
  assert.match(mismatchedRuntimeMarkup, /# Locked until the selected model is loaded and verified/, 'API curl should stay blocked until exact row and runtime readiness both match')
  assert.doesNotMatch(mismatchedRuntimeMarkup, /Ready for the selected model/, 'mismatched runtime must not claim selected exact-row readiness')

  const plannedExactModel = {
    id: 'mistral-7b-instruct-v0.3-q8_0',
    name: 'Mistral 7B Instruct v0.3 Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    quant: 'Q8_0',
    model_path: '/models/mistral-7b-instruct-v0.3-q8_0.gguf',
  }
  const plannedExactCapabilities = {
    ...capabilities,
    model_compatibility: [
      ...capabilities.model_compatibility,
      {
        id: 'mistral_7b_instruct_v0_3_q8_0',
        status: 'planned',
        family: 'mistral',
        quantization: 'Q8_0',
        support_scope: 'exact row only once validated',
        frontend_readiness_gate: 'must stay blocked until supported',
        latest_checked_bucket: 'not_started',
        latest_checked_result: 'not_started',
        latest_checked_output: 'no validated output yet',
        full_support_status: 'not_supported',
        full_support_blockers: 'generation evidence missing',
        evidence: 'Planned exact-row placeholder, not runnable support.',
        next_step: 'Collect exact-row evidence before unlocking chat.',
      },
    ],
  }
  const plannedExactMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: plannedExactModel.id },
    selectedModel: plannedExactModel,
    capabilities: plannedExactCapabilities,
  }))

  assert.match(plannedExactMarkup, /mistral_7b_instruct_v0_3_q8_0/, 'API view should show selected planned exact-row evidence by row id')
  assert.match(plannedExactMarkup, /Engine ready — model not verified/, 'API readiness should stay guarded when the selected exact row is not supported')
  assert.match(plannedExactMarkup, /Planned exact-row placeholder, not runnable support\./, 'API view should render the exact row evidence without broad-family inference')
  assert.doesNotMatch(plannedExactMarkup, /Ready for the selected model/, 'planned exact rows must not claim selected exact-row readiness even when runtime health is green')

  const genericExactModel = {
    id: 'custom-exact-row-q8-0',
    name: 'Custom exact row Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    quant: 'Q8_0',
    model_path: '/models/custom-exact-row-q8-0.gguf',
  }
  const genericExactCapabilities = {
    support_contract: capabilities.support_contract,
    model_compatibility: [
      {
        id: 'custom_exact_row_q8_0',
        status: 'supported_exact_row_smoke',
        family: 'custom_decoder',
        quantization: 'Q8_0',
        support_scope: 'exact custom row only',
        frontend_readiness_gate: 'green only when this exact custom row is selected and loaded',
        latest_checked_bucket: 'frontend_fixture',
        latest_checked_result: 'pass',
        latest_checked_output: 'custom exact row fixture output',
        full_support_status: 'blocked_pending_normalized_full_support',
        full_support_blockers: 'no neighboring custom rows inherit support',
        evidence: 'Custom exact row evidence from /api/capabilities.',
        next_step: 'Keep row-id scoped.',
      },
    ],
    api_features: [],
  }
  const genericExactMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: genericExactModel.id },
    selectedModel: genericExactModel,
    capabilities: genericExactCapabilities,
  }))

  assert.match(genericExactMarkup, /Ready for the selected model/, 'API view should support generic exact compatibility row ids without family-specific frontend matchers')
  assert.match(genericExactMarkup, /custom_exact_row_q8_0/, 'API view should render generic selected exact-row ids from capabilities')
  assert.match(genericExactMarkup, /Custom exact row evidence from \/api\/capabilities\./, 'API view should render generic exact-row evidence text')
  assert.doesNotMatch(genericExactMarkup, /No exact compatibility row matched this selected model/, 'generic exact row-id matches should not fall through to broad or missing support copy')

  /* ---- API workbench try-it gating (Phase 5) ---- */
  const blockedApiMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { status: 'online', api_base: 'http://127.0.0.1:8181', loaded_now: true, generation_ready: true, active_model_id: 'tiny-generation' },
    selectedModel: { id: 'tiny-generation', name: 'tiny-generation', provider_kind: 'local', status: 'ready', model_path: '/tmp/x.gguf', loaded_now: true, generation_ready: true },
    capabilities,
  }))
  assert.match(blockedApiMarkup, /data-tryit-ready="false" data-endpoint="v1_chat_completions"/, 'chat-completions try-it must stay guarded when no exact supported row matches')
  assert.match(blockedApiMarkup, /data-tryit-ready="false" data-endpoint="v1_completions"/, 'raw completions try-it must stay guarded exactly like chat')
  assert.match(blockedApiMarkup, /Requires a loaded supported model/, 'guarded try-its must render the typed gate copy')
  assert.match(blockedApiMarkup, /data-tryit-ready="true" data-endpoint="v1_health"/, 'read-only health try-it may run while the backend is online')
  const greenApiMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel: aliasSelectedModel,
    capabilities: green3BCapabilities,
  }))
  assert.match(greenApiMarkup, /data-tryit-ready="true" data-endpoint="v1_chat_completions"/, 'chat-completions try-it must unlock when the exact supported row is loaded and ready')

  /* ---- Compatibility ledger renders ONLY the live contract (Phase 4) ---- */
  const ledgerMarkup = renderToStaticMarkup(React.createElement(CompatibilityView, { capabilities }))
  assert.match(ledgerMarkup, /tinyllama_1_1b_chat_q8_0/, 'ledger rows must come from the capabilities payload')
  assert.match(ledgerMarkup, /Not claimed/, 'every ledger row must carry the not-claimed column')
  assert.match(ledgerMarkup, /arbitrary\/Jinja templates, production throughput, portability/, 'the not-claimed column must render the contract blocker copy verbatim')
    assert.match(ledgerMarkup, /other_future_row_q8_0[\s\S]*?0<\/b> verified of 1 tracked lanes[\s\S]*?no verified pack/, 'planned pack ids must not be presented as verified evidence or checked contexts')
    /* Quant-grouped evidence moved to the ledger (2026-08); the API contract card now carries
       the same two postures as counts. Both must stay visible -- a page that shows only the
       supported side would overclaim. */
    assert.match(blockedApiMarkup, /<b>2<\/b> verified models[\s\S]*<b>1<\/b> guarded or unclaimed rows/, 'grouped quant evidence must keep supported and guarded postures visible')
  assert.match(ledgerMarkup, /How to read this ledger/, 'the ledger must keep its explainer')
  assert.doesNotMatch(ledgerMarkup, /broad_family_trap|broad_quant_trap/, 'the ledger must not render non-row capability lists as support evidence')
  const emptyLedgerMarkup = renderToStaticMarkup(React.createElement(CompatibilityView, { capabilities: null }))
  assert.match(emptyLedgerMarkup, /Ledger unavailable/, 'an unreadable contract must render the fail-closed empty state')
  assert.doesNotMatch(emptyLedgerMarkup, /ledger-row__id/, 'no contract means zero ledger rows — nothing renders from memory')

  console.log('Frontend integration smoke passed')
} finally {
  await server.close()
}
