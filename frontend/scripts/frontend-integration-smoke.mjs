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
  const { default: ApiView } = await server.ssrLoadModule('/src/views/ApiView.jsx')
  const { default: SystemView } = await server.ssrLoadModule('/src/views/SystemView.jsx')
  const { default: ModelsView } = await server.ssrLoadModule('/src/views/ModelsView.jsx')
  const { default: TopBar } = await server.ssrLoadModule('/src/components/TopBar.jsx')
  const { getChatGateState } = await server.ssrLoadModule('/src/lib/chatGate.js')
  const {
    exactRowSupportLanes,
    rowSupportBoundaryCopy,
    rowSupportNextStepCopy,
  } = await server.ssrLoadModule('/src/lib/capabilities.js')
  const { resolveLoadedModelDisplayName } = await server.ssrLoadModule('/src/hooks/useDashboardData.js')
  const { LLAMA32_3B_ACCEPTANCE_TARGET } = await server.ssrLoadModule('/src/lib/acceptanceTargets.js')

  const noop = () => {}
  const readyRuntime = {
    api_base: 'http://127.0.0.1:8181',
    loaded_now: true,
    generation_ready: true,
    active_model_id: 'llama32_3b_instruct_q8_0',
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

  const streamingMarkup = renderToStaticMarkup(React.createElement(ChatWorkspace, {
    selectedConversation: {
      id: 'conversation-streaming-code',
      title: 'Streaming code',
      updated_at: '2026-05-13T04:21:00.000Z',
      messages: [
        { id: 'user-1', role: 'user', content: 'Create one self-contained HTML page', created_at: '2026-05-13T04:21:00.000Z' },
        { id: 'assistant-1', role: 'assistant', content: '```html\n<!doctype html>\n<button id="go">Go</button>', streaming: true, streaming_phase: 'streaming', created_at: '2026-05-13T04:21:01.000Z' },
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
  assert.match(streamingMarkup, /Live chat exact-row readiness[\s\S]*Runtime[\s\S]*Local chat ready[\s\S]*Support[\s\S]*Exact row supported[\s\S]*Capabilities[\s\S]*Template ready · Throughput not promoted/, 'non-empty live 3B chats should keep runtime, exact-row support, and row-scoped capability lanes visible after messages exist')
  assert.match(streamingMarkup, /Row-scoped \/api\/capabilities evidence; it does not widen model-native context/, 'live 3B capability lanes must avoid widening exact-row support into broader claims')
  assert.match(streamingMarkup, /COMPATIBILITY\.md and \/api\/capabilities agree/, 'live 3B chat readiness must name the exact-row support-contract requirement')
  assert.match(streamingMarkup, /data-streaming-code-state="open"/, 'open streaming fences should expose the active code state')
  assert.match(streamingMarkup, /Still generating — code block incomplete/, 'open streaming code should visibly say it is incomplete')
  assert.match(streamingMarkup, /Streaming code response/, 'streaming code rows should keep an active live-generation label')
  assert.match(streamingMarkup, /aria-busy="true"/, 'streaming rows and code cards should be marked busy while backend generation is active')
  assert.match(streamingMarkup, /message-code-card is-generating/, 'open streaming code should render as the real ForgeLocal-derived code card, not fallback prose')

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
  assert.match(preTokenMarkup, /Backend is generating/, 'pre-token streaming should render the active backend-generation live status')
  assert.match(preTokenMarkup, /streaming-loader-dot-3/, 'pre-token streaming should render the active loader, not a static placeholder')

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
  assert.match(completedUnclosedFenceMarkup, /print\([\s\S]*&quot;safe&quot;[\s\S]*\)/, 'completed unclosed code content should remain visible and escaped in the code card')
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

  assert.match(exactReadyMarkup, /Selected exact row ready/, 'API readiness should turn green only for a matching loaded exact row')
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

  assert.match(exactReadySystemMarkup, /Selected exact-row local \/v1 ready/, 'System endpoint status should go green only when the selected 3B exact row and runtime readiness both match')
  assert.match(exactReadySystemMarkup, /Runs now for this selected GGUF because loaded_now=true, generation_ready=true, active_model_id matches, and the exact \/api\/capabilities row is supported\./, 'System chat-completions copy should name the full 3B exact-row readiness gate')
  assert.match(exactReadySystemMarkup, /Endpoint\/chat gate:[\s\S]*Ready: runtime readiness and exact-row support both match\./, 'System selected exact-row evidence should expose the retained chat/API gate')
  assert.match(exactReadySystemMarkup, /Template\/Jinja readiness[\s\S]*Template readiness is green for this supported exact row/, 'System should render 3B template/Jinja lane evidence from /api/capabilities')
  assert.match(exactReadySystemMarkup, /Throughput readiness[\s\S]*Bounded row-scoped performance\/RSS evidence is present/, 'System should keep bounded 3B performance evidence separate from production-throughput promotion')
  assert.doesNotMatch(exactReadySystemMarkup, /# Use only after \/v1\/health returns generation_ready=true/, 'System curl should not imply generation_ready alone is sufficient for 3B UX chat')
  assert.match(exactReadyMarkup, /chat completions/, 'API view should display provider-scoped feature ids as neutral capability names')
  assert.match(exactReadyMarkup, /standard-compatible streaming stays enabled\./, 'API view should sanitize provider-specific feature notes before rendering')

  const topBarMarkup = renderToStaticMarkup(React.createElement(TopBar, {
    tab: 'api',
    setTab: noop,
    selectedConversationTitle: '',
    selectedConversationUpdatedAt: '',
    selectedConversationPreview: '',
    runtime: readyRuntime,
    capabilities,
    selectedModelId: selectedModel.id,
    setSelectedModelId: noop,
    models: [selectedModel],
  }))

  assert.match(topBarMarkup, /Support contract/, 'TopBar should render the support contract status surface')
  assert.doesNotMatch(topBarMarkup, /arbitrary|Jinja/s, 'TopBar support contract label should filter resolved template/Jinja caveats')
  assert.match(topBarMarkup, /production throughput|throughput/s, 'TopBar support contract label should keep production-throughput caveats until explicit production evidence is green')

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

  assert.match(aliasApiMarkup, /Selected exact row ready/, 'API view should keep alias-selected 3B exact rows green when active_model_id matches runtime_model_name')
  assert.match(aliasApiMarkup, /&quot;model&quot;: &quot;llama32_3b_instruct_q8_0&quot;/, 'API curl should send the backend loaded model id, not the browser-only alias')
  assert.doesNotMatch(aliasApiMarkup, /&quot;model&quot;: &quot;browser-llama32-3b-alias&quot;/, 'API curl must not create model_mismatch risk for alias-selected exact rows')

  const aliasTopBarMarkup = renderToStaticMarkup(React.createElement(TopBar, {
    tab: 'api',
    setTab: noop,
    selectedConversationTitle: '',
    selectedConversationUpdatedAt: '',
    selectedConversationPreview: '',
    runtime: readyRuntime,
    capabilities,
    selectedModelId: aliasSelectedModel.id,
    setSelectedModelId: noop,
    models: [aliasSelectedModel],
  }))

  assert.match(aliasTopBarMarkup, /Runtime chat gate[\s\S]*Llama 3\.2 3B Instruct Q8_0/, 'TopBar runtime readiness should resolve the active model through runtime_model_name aliases')
  assert.match(aliasTopBarMarkup, /llama32_3b_instruct_q8_0: supported current gate/, 'TopBar support detail should prioritize the active exact 3B row instead of the first supported row')
  assert.doesNotMatch(aliasTopBarMarkup, /tinyllama_1_1b_chat_q8_0: supported current gate/, 'TopBar support detail must not point at TinyLlama when a 3B exact row is active')
  assert.doesNotMatch(aliasTopBarMarkup, /Nothing loaded now/, 'TopBar must not show an empty runtime state for alias-selected loaded 3B rows')

  const modelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    runtime: readyRuntime,
    capabilities,
    refreshDashboard: noop,
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    externalForm: { id: '', name: '', source: '', api_base: '', api_key: '', model_name: '' },
    setExternalForm: noop,
    registerModel: noop,
    connectExternalModel: noop,
    models: [aliasSelectedModel],
    selectedModelId: aliasSelectedModel.id,
    setSelectedModelId: noop,
    loadingModelId: '',
    activateModel: noop,
    unloadCurrentModel: noop,
    installModel: noop,
    installCatalogModel: noop,
    cancelModelDownload: noop,
  }))

  assert.match(modelsMarkup, /Llama 3\.2 3B Instruct Q8_0/, 'Models view should render the exact 3B row when the browser id differs from the backend runtime id')
  assert.match(modelsMarkup, /chat enabled|The selected model is loaded, generation-ready, and backed by an exact supported \/api\/capabilities row/, 'Models view should treat runtime_model_name active_model_id matches as next-chat ready when exact 3B evidence is supported')
  assert.match(modelsMarkup, /Chat unlockable/, 'Tracked 3B card should turn green only after exact row support and runtime readiness both match')
  assert.match(modelsMarkup, /Loaded exact-row match/, 'Tracked 3B card should mark the alias model as the loaded exact-row match')
  assert.match(modelsMarkup, /3B API\/WebUI smoke passed/, 'Models view should keep 3B end-to-end WebUI evidence visible on the exact row card')
  assert.doesNotMatch(modelsMarkup, /This browser\/runtime list does not currently show the exact 3B row/, 'Alias runtime matches must not fall through to the missing-3B acceptance placeholder')

  const staleRuntimeModelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    runtime: { ...readyRuntime, loaded_now: false },
    capabilities,
    refreshDashboard: noop,
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    externalForm: { id: '', name: '', source: '', api_base: '', api_key: '', model_name: '' },
    setExternalForm: noop,
    registerModel: noop,
    connectExternalModel: noop,
    models: [aliasSelectedModel],
    selectedModelId: aliasSelectedModel.id,
    setSelectedModelId: noop,
    loadingModelId: '',
    activateModel: noop,
    unloadCurrentModel: noop,
    installModel: noop,
    installCatalogModel: noop,
    cancelModelDownload: noop,
  }))

  assert.match(staleRuntimeModelsMarkup, /Runtime still needed/, 'Tracked 3B card must use the shared chat gate and stay blocked when runtime loaded_now=false')
  assert.doesNotMatch(staleRuntimeModelsMarkup, /Chat unlockable/, 'Tracked 3B card must not present stale browser generation_ready state as WebUI chat support')

  const neighboringQuantAcceptanceRecord = {
    ...aliasSelectedModel,
    id: LLAMA32_3B_ACCEPTANCE_TARGET.id,
    name: 'Llama 3.2 3B Instruct Q4_0',
    runtime_model_name: 'llama32_3b_instruct_q4_0',
    quant: 'Q4_0',
    model_path: '/models/Llama-3.2-3B-Instruct-Q4_0.gguf',
  }
  const neighboringQuantAcceptanceMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    runtime: { ...readyRuntime, active_model_id: neighboringQuantAcceptanceRecord.runtime_model_name },
    capabilities,
    refreshDashboard: noop,
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    externalForm: { id: '', name: '', source: '', api_base: '', api_key: '', model_name: '' },
    setExternalForm: noop,
    registerModel: noop,
    connectExternalModel: noop,
    models: [neighboringQuantAcceptanceRecord],
    selectedModelId: neighboringQuantAcceptanceRecord.id,
    setSelectedModelId: noop,
    loadingModelId: '',
    activateModel: noop,
    unloadCurrentModel: noop,
    installModel: noop,
    installCatalogModel: noop,
    cancelModelDownload: noop,
  }))
  assert.match(neighboringQuantAcceptanceMarkup, /This browser\/runtime list does not currently show the exact 3B row/, '3B acceptance placeholder must stay visible when the browser acceptance id is backed by a neighboring quant')
  assert.match(neighboringQuantAcceptanceMarkup, /llama32_3b_instruct_q8_0: quant mismatch/, '3B acceptance hardening should surface the Q8_0 row mismatch instead of treating the browser id as exact evidence')

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
    [['template', true], ['throughput', true]],
    '3B exact row should show both template/Jinja and production-throughput lanes green once /api/capabilities advertises row evidence',
  )
  assert.doesNotMatch(rowSupportBoundaryCopy(green3BRow, green3BCapabilities.api_features), /arbitrary|Jinja|production|throughput/i, '3B remaining boundary should filter resolved template/Jinja and production-throughput blockers when both lanes are green')
  assert.doesNotMatch(rowSupportNextStepCopy(green3BRow, green3BCapabilities.api_features), /arbitrary|Jinja|production|throughput/i, '3B next-step copy should filter resolved template/Jinja and production-throughput blockers when both lanes are green')
  assert.equal(getChatGateState(green3BCapabilities, aliasSelectedModel, readyRuntime).chatUnlocked, true, 'green 3B evidence must still require runtime loaded_now/generation_ready and exact-row model identity before chat unlocks')

  const green3BApiMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: readyRuntime,
    selectedModel: aliasSelectedModel,
    capabilities: green3BCapabilities,
  }))
  assert.match(green3BApiMarkup, /Throughput readiness[\s\S]*Production-throughput support is advertised by \/api\/capabilities as supported exact row evidence/, 'API view should render green 3B production-throughput evidence only when /api/capabilities advertises it')
  assert.doesNotMatch(green3BApiMarkup, /Remaining support boundary:<\/b>[\s\S]{0,220}(?:arbitrary|Jinja|production|throughput)/i, 'API view 3B boundary should not repeat resolved template/Jinja or production-throughput blockers after green evidence')

  const green3BModelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    runtime: readyRuntime,
    capabilities: green3BCapabilities,
    refreshDashboard: noop,
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    externalForm: { id: '', name: '', source: '', api_base: '', api_key: '', model_name: '' },
    setExternalForm: noop,
    registerModel: noop,
    connectExternalModel: noop,
    models: [aliasSelectedModel],
    selectedModelId: aliasSelectedModel.id,
    setSelectedModelId: noop,
    loadingModelId: '',
    activateModel: noop,
    unloadCurrentModel: noop,
    installModel: noop,
    installCatalogModel: noop,
    cancelModelDownload: noop,
  }))
  assert.match(green3BModelsMarkup, /Chat unlockable/, 'Models tracked 3B card should remain chat-unlockable when exact row, runtime readiness, and green evidence all align')
  assert.match(green3BModelsMarkup, /Throughput: Production throughput ready for this exact row/, 'Models tracked 3B card should show production-throughput green only from row/API evidence')
  assert.doesNotMatch(green3BModelsMarkup, /Remaining support boundary:<\/b>[\s\S]{0,220}(?:arbitrary|Jinja|production|throughput)/i, 'Models tracked 3B card should not keep resolved 3B support blockers visible when evidence is green')

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

  const misleadingTinyModelsMarkup = renderToStaticMarkup(React.createElement(ModelsView, {
    runtime: misleadingTinyRuntime,
    capabilities,
    refreshDashboard: noop,
    registerForm: { id: '', name: '', model_path: '', runtime_model_name: '' },
    setRegisterForm: noop,
    externalForm: { id: '', name: '', source: '', api_base: '', api_key: '', model_name: '' },
    setExternalForm: noop,
    registerModel: noop,
    connectExternalModel: noop,
    models: [misleadingTinyRuntimeModel],
    selectedModelId: misleadingTinyRuntimeModel.id,
    setSelectedModelId: noop,
    loadingModelId: '',
    activateModel: noop,
    unloadCurrentModel: noop,
    installModel: noop,
    installCatalogModel: noop,
    cancelModelDownload: noop,
  }))

  assert.match(misleadingTinyModelsMarkup, /llama32_3b_instruct_q8_0: supported current gate/, 'Models view should present the stale-id live run as the exact 3B row, not TinyLlama')
  assert.match(misleadingTinyModelsMarkup, /Loaded exact-row match/, 'Models tracked 3B card should mark stale-id live runs as loaded exact-row matches')
  assert.doesNotMatch(misleadingTinyModelsMarkup, /tinyllama_1_1b_chat_q8_0: supported current gate/, 'stale tinyllama-q8 runtime ids must not make the selected live 3B model inherit TinyLlama support copy')

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
  assert.match(liveBackendIdChatMarkup, /Local chat ready/, 'ready 3B live-backend-id chat should show runtime-green chat UX')
  assert.match(liveBackendIdChatMarkup, /Llama 3\.2 3B Instruct Q8_0 is loaded now and generation_ready=true\./, 'ready 3B live-backend-id chat should display the exact 3B row name instead of the backend-generated runtime id')
  assert.match(liveBackendIdChatMarkup, /Exact row supported/, 'ready 3B live-backend-id chat should show exact-row support in the composer surface')
  assert.match(liveBackendIdChatMarkup, /Message Camelid…/, 'ready 3B live-backend-id chat should enable the composer instead of showing load-first copy')
  assert.doesNotMatch(liveBackendIdChatMarkup, /Load a model first|Choose a supported model/, 'ready 3B live-backend-id chat should not fall back to blocked chat UX')

  const neighboringQuantPathModel = {
    ...selectedModel,
    quant: undefined,
    model_path: '<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q4_0.gguf',
  }
  const neighboringQuantPathGate = getChatGateState(capabilities, neighboringQuantPathModel, readyRuntime)
  assert.equal(neighboringQuantPathGate.runtimeReady, true, '3B neighboring-quant guard should still surface runtime readiness when active_model_id matches')
  assert.equal(neighboringQuantPathGate.contractSupported, false, '3B neighboring GGUF quant must not inherit the canonical Q8_0 row from the browser id')
  assert.equal(neighboringQuantPathGate.chatUnlocked, false, '3B neighboring GGUF quant must keep live chat locked even when runtime loaded_now/generation_ready are green')

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

  assert.match(neighboringQuantPathChatMarkup, /Runtime ready, support gated/, '3B neighboring-quant chat UX should expose runtime-green state without claiming support')
  assert.match(neighboringQuantPathChatMarkup, /llama32_3b_instruct_q8_0: quant mismatch/, '3B neighboring-quant chat UX should name the exact row mismatch instead of showing ready chat')
  assert.match(neighboringQuantPathChatMarkup, /scoped to Q8_0[\s\S]*appears to be Q4_0/, '3B neighboring-quant chat UX should explain that the loaded artifact quant does not match the support contract row')
  assert.doesNotMatch(neighboringQuantPathChatMarkup, /Local chat ready|Message Camelid…|Demo starters/, '3B neighboring-quant rows must not render the live-chat ready UX')

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

  assert.match(backendReadyButUnsupported3BChatMarkup, /Choose a supported model\./, 'runtime-ready 3B rows should render support-gated chat UX when the exact row is downgraded')
  assert.match(backendReadyButUnsupported3BChatMarkup, /Runtime ready, support gated/, 'support-gated 3B UX should still expose that loaded_now and generation_ready are green')
  assert.match(backendReadyButUnsupported3BChatMarkup, /llama32_3b_instruct_q8_0: groundwork backend evidence only/, 'support-gated 3B UX should name the exact unpromoted capabilities row rather than hiding behind generic load-first copy')
  assert.match(backendReadyButUnsupported3BChatMarkup, /Chat unlocks only after loaded_now=true, generation_ready=true, and an exact supported compatibility row all match\./, 'support-gated 3B UX should preserve the exact-row frontend readiness rule')
  assert.match(backendReadyButUnsupported3BChatMarkup, /Load a model first/, 'support-gated 3B composer should stay disabled instead of accepting prompts')
  assert.doesNotMatch(backendReadyButUnsupported3BChatMarkup, /Local chat ready|Message Camelid…|Demo starters/, 'support-gated 3B rows must not render the live-chat ready UX')

  const backendReadyButUnsupported3BSystemMarkup = renderToStaticMarkup(React.createElement(SystemView, {
    runtime: liveBackendIdRuntime,
    selectedModel: liveBackendIdModel,
    capabilities: backendReadyButUnsupported3BCapabilities,
  }))

  assert.match(backendReadyButUnsupported3BSystemMarkup, /Runtime ready, support gated/, 'System should keep runtime-green 3B rows visible without making unsupported exact rows API/chat-ready')
  assert.match(backendReadyButUnsupported3BSystemMarkup, /Blocked for UX chat until loaded_now=true, generation_ready=true, active_model_id matches, and this exact row is supported\./, 'System should block chat completions copy when the exact 3B row is downgraded')
  assert.match(backendReadyButUnsupported3BSystemMarkup, /Endpoint\/chat gate:[\s\S]*llama32_3b_instruct_q8_0: groundwork backend evidence only; loaded_now=true, generation_ready=true, exact row supported=false\./, 'System selected exact-row evidence should show runtime-green/support-red state for downgraded 3B rows')
  assert.match(backendReadyButUnsupported3BSystemMarkup, /# Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'System curl should stay blocked for runtime-ready unsupported 3B rows')
  assert.doesNotMatch(backendReadyButUnsupported3BSystemMarkup, /Selected exact-row local \/v1 ready|Ready: runtime readiness and exact-row support both match/, 'System must not present downgraded 3B rows as exact-row API/chat-ready')

  assert.match(exactReadyMarkup, /responses stream/, 'API view should normalize provider-scoped dotted feature ids before rendering')
  assert.match(exactReadyMarkup, /hosted model-style streamed response compatibility stays provider-neutral/, 'API view should neutralize hosted-brand feature notes before rendering')
  assert.match(exactReadyMarkup, /Guarded feature row; do not label it hosted model or hosted model compatible from API metadata\./, 'API view should also neutralize guarded feature metadata before rendering')
  assert.doesNotMatch(exactReadyMarkup, /openai|OpenAI|ChatGPT|Claude|Gemini|broad_family_trap|broad_quant_trap/, 'API view must not promote broad family/quant lists or raw provider-scoped/hosted-brand feature labels as support evidence')

  const mismatchedRuntimeMarkup = renderToStaticMarkup(React.createElement(ApiView, {
    runtime: { ...readyRuntime, active_model_id: 'different-loaded-model' },
    selectedModel,
    capabilities,
  }))

  assert.match(mismatchedRuntimeMarkup, /Different loaded model is ready/, 'API readiness should fail closed when active_model_id differs from the selected exact row')
  assert.match(mismatchedRuntimeMarkup, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl should stay blocked until exact row and runtime readiness both match')
  assert.doesNotMatch(mismatchedRuntimeMarkup, /Selected exact row ready/, 'mismatched runtime must not claim selected exact-row readiness')

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
  assert.match(plannedExactMarkup, /Generation ready; exact row required/, 'API readiness should stay guarded when the selected exact row is not supported')
  assert.match(plannedExactMarkup, /Planned exact-row placeholder, not runnable support\./, 'API view should render the exact row evidence without broad-family inference')
  assert.doesNotMatch(plannedExactMarkup, /Selected exact row ready/, 'planned exact rows must not claim selected exact-row readiness even when runtime health is green')

  const genericExactModel = {
    id: 'custom-exact-row-q8-0',
    name: 'Custom exact row Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    quant: 'Q8_0',
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

  assert.match(genericExactMarkup, /Selected exact row ready/, 'API view should support generic exact compatibility row ids without family-specific frontend matchers')
  assert.match(genericExactMarkup, /custom_exact_row_q8_0/, 'API view should render generic selected exact-row ids from capabilities')
  assert.match(genericExactMarkup, /Custom exact row evidence from \/api\/capabilities\./, 'API view should render generic exact-row evidence text')
  assert.doesNotMatch(genericExactMarkup, /No selected model exact row matched/, 'generic exact row-id matches should not fall through to broad or missing support copy')

  console.log('Frontend integration smoke passed')
} finally {
  await server.close()
}
