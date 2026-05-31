#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'

const args = parseArgs(process.argv.slice(2))
if (!args.input || !args['json-out']) usage()

const inputPath = args.input
const outputPath = args['json-out']
const root = JSON.parse(fs.readFileSync(inputPath, 'utf8'))
const diagnostics = findCamelidDiagnostics(root)
if (!diagnostics?.dense || typeof diagnostics.dense !== 'object') {
  throw new Error(`${inputPath}: missing camelid.dense diagnostics; rerun with camelid_dense_diagnostics=true`)
}

const dense = diagnostics.dense
const layers = parseLayerList(args.layers ?? 'all', dense.layers?.length ?? 0)
const stages = buildStages(dense, layers)
const knownGoodTokenIds = root.llama_generated_tokens ?? root.known_good_token_ids ?? root.llama_generated_tokens_from_text ?? null
const extracted = {
  schema: 'camelid.forward-trace.v1',
  source: {
    input: inputPath,
    basename: path.basename(inputPath),
    name: root.name ?? null,
    prompt_tokens_match: root.prompt_tokens_match ?? null,
    generated_text_match: root.generated_text_match ?? null,
    selected_token_id: root.selected_token_id ?? diagnostics.generated_token_ids?.[0] ?? null,
    known_good_token_ids: knownGoodTokenIds,
    known_good_token_ids_from_text: root.llama_generated_tokens_from_text ?? null,
    known_good_top_logprobs: compactTopLogprobs(root.llama_top_logprobs),
    backend_text: root.backend_text ?? null,
    known_good_text: root.llama_text ?? null,
  },
  prompt_token_ids: diagnostics.prompt_token_ids ?? null,
  generated_token_ids: diagnostics.generated_token_ids ?? null,
  input_token_ids: dense.input_token_ids ?? null,
  position_ids: dense.position_ids ?? null,
  dense_metadata: diagnostics.dense_metadata ?? null,
  layer_count: dense.layers?.length ?? 0,
  selected_layers: layers,
  stage_count: stages.length,
  stages,
  top_logits: diagnostics.top_logits ?? [],
  output_projection: compactOutputProjection(diagnostics.output_projection ?? []),
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true })
fs.writeFileSync(outputPath, `${JSON.stringify(extracted, null, 2)}\n`)
console.log(`wrote=${outputPath}`)
console.log(`schema=${extracted.schema}`)
console.log(`selected_layers=${layers.join(',')}`)
console.log(`stage_count=${stages.length}`)
console.log(`first_stage=${stages[0]?.path ?? 'none'}`)
console.log(`last_stage=${stages.at(-1)?.path ?? 'none'}`)

function usage() {
  console.error('usage: node scripts/extract-forward-trace.mjs --input <diagnostics.json> --json-out <forward-trace.json> [--layers all|0,2,20]')
  console.error('extracts an ordered Camelid dense forward trace from embedding through per-layer norm/QKV/RoPE/attention/FFN to final norm/logits')
  process.exit(2)
}

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i]
    if (!key.startsWith('--')) usage()
    const name = key.slice(2)
    const value = argv[i + 1]
    if (!value || value.startsWith('--')) usage()
    out[name] = value
    i++
  }
  return out
}

function parseLayerList(value, layerCount) {
  if (value === 'all') return Array.from({ length: layerCount }, (_, index) => index)
  const layers = value.split(',')
    .map(part => part.trim())
    .filter(Boolean)
    .map(part => Number.parseInt(part, 10))
  if (layers.length === 0 || layers.some(layer => !Number.isInteger(layer) || layer < 0)) {
    throw new Error(`invalid --layers value: ${value}`)
  }
  for (const layer of layers) {
    if (layer >= layerCount) throw new Error(`requested layer ${layer}, but diagnostics only contain ${layerCount} layer(s)`)
  }
  return [...new Set(layers)].sort((left, right) => left - right)
}

function findCamelidDiagnostics(value) {
  if (value?.camelid) return value.camelid
  if (value?.backend_chat?.camelid) return value.backend_chat.camelid
  if (value?.backend?.diagnostics?.camelid) return value.backend.diagnostics.camelid
  if (value?.diagnostics?.camelid) return value.diagnostics.camelid
  if (value?.choices?.[0]?.message?.camelid) return value.choices[0].message.camelid
  if (value?.choices?.[0]?.camelid) return value.choices[0].camelid
  return null
}

function buildStages(dense, layers) {
  const stages = []
  const addStats = (pathName, stats, extra = {}) => stages.push(stage(pathName, 'tensor_stats', { ...extra, stats: compactStats(stats) }))
  const addTrace = (pathName, trace, extra = {}) => stages.push(stage(pathName, 'attention_trace', { ...extra, attention_trace: compactAttentionTrace(trace) }))
  const addKvTrace = (pathName, trace, extra = {}) => stages.push(stage(pathName, 'kv_cache_trace', { ...extra, kv_cache_trace: compactKvCacheTrace(trace) }))
  const addReconstruction = (pathName, reconstruction, extra = {}) => stages.push(stage(pathName, 'reconstruction', { ...extra, reconstruction: compactReconstruction(reconstruction) }))

  addStats('embedding', dense.embedding)
  for (const layerIndex of layers) {
    const layer = dense.layers?.[layerIndex]
    if (!layer) throw new Error(`diagnostics are missing layer ${layerIndex}`)
    const layerExtra = { layer_index: layerIndex }
    addStats(`layers.${layerIndex}.attention_input`, layer.residual_flow?.attention_input, layerExtra)
    addStats(`layers.${layerIndex}.attention_norm`, layer.attention_norm, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_norm_reconstruction) })
    addStats(`layers.${layerIndex}.attention_q`, layer.attention_q, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_q_reconstruction) })
    addStats(`layers.${layerIndex}.attention_k`, layer.attention_k, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_k_reconstruction) })
    addStats(`layers.${layerIndex}.attention_v`, layer.attention_v, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_v_reconstruction) })
    addStats(`layers.${layerIndex}.attention_q_rope`, layer.attention_q_rope, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_q_rope_reconstruction) })
    addStats(`layers.${layerIndex}.attention_k_rope`, layer.attention_k_rope, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_k_rope_reconstruction) })
    addKvTrace(`layers.${layerIndex}.kv_cache_trace`, layer.kv_cache_trace, layerExtra)
    addTrace(`layers.${layerIndex}.attention_trace`, layer.attention_trace, layerExtra)
    addStats(`layers.${layerIndex}.attention_context`, layer.attention_context, layerExtra)
    addStats(`layers.${layerIndex}.attention_output`, layer.attention_output, { ...layerExtra, reconstruction: compactReconstruction(layer.attention_output_reconstruction) })
    addStats(`layers.${layerIndex}.attention_residual`, layer.attention_residual, { ...layerExtra, residual_delta: compactReconstruction(layer.residual_flow?.attention_delta) })
    addStats(`layers.${layerIndex}.ffn_input`, layer.residual_flow?.ffn_input, layerExtra)
    addStats(`layers.${layerIndex}.ffn_norm`, layer.ffn_norm, { ...layerExtra, reconstruction: compactReconstruction(layer.ffn_norm_reconstruction) })
    if (layer.moe_router) stages.push(stage(`layers.${layerIndex}.moe_router`, 'moe_router', { ...layerExtra, moe_router: compactMoeRouter(layer.moe_router) }))
    addStats(`layers.${layerIndex}.ffn_gate`, layer.ffn_gate, { ...layerExtra, reconstruction: compactReconstruction(layer.ffn_gate_reconstruction) })
    addStats(`layers.${layerIndex}.ffn_up`, layer.ffn_up, { ...layerExtra, reconstruction: compactReconstruction(layer.ffn_up_reconstruction) })
    addStats(`layers.${layerIndex}.ffn_activation`, layer.ffn_activation, { ...layerExtra, reconstruction: compactReconstruction(layer.ffn_activation_reconstruction) })
    addStats(`layers.${layerIndex}.ffn_output`, layer.ffn_output, { ...layerExtra, reconstruction: compactReconstruction(layer.ffn_down_reconstruction) })
    addStats(`layers.${layerIndex}.ffn_residual`, layer.ffn_residual, { ...layerExtra, residual_delta: compactReconstruction(layer.residual_flow?.ffn_delta) })
  }
  addStats('final_hidden', dense.final_hidden)
  addReconstruction('final_norm', dense.final_norm)
  addStats('output_norm', dense.output_norm)
  addStats('logits', dense.logits)

  return stages.map((entry, order) => ({ order, ...entry }))
}

function stage(pathName, kind, payload) {
  return { path: pathName, kind, ...payload }
}

function compactMoeRouter(router) {
  if (!router) return null
  return {
    expert_count: integerOrNull(router.expert_count),
    expert_used_count: integerOrNull(router.expert_used_count),
    rows: (router.rows ?? []).map(row => ({
      row_index: integerOrNull(row.row_index),
      router_logits: numericArray(row.router_logits),
      selected_experts: (row.selected_experts ?? []).map(expert => ({
        expert_id: integerOrNull(expert.expert_id),
        selected_rank: integerOrNull(expert.selected_rank),
        router_logit: numberOrNull(expert.router_logit),
        router_probability: numberOrNull(expert.router_probability),
        selected_weight: numberOrNull(expert.selected_weight),
      })),
    })),
  }
}

function compactStats(stats) {
  if (!stats) return null
  return {
    shape: stats.checkpoint?.shape ?? stats.shape ?? null,
    len: stats.checkpoint?.len ?? stats.len ?? null,
    min: numberOrNull(stats.min),
    min_index: integerOrNull(stats.min_index),
    max: numberOrNull(stats.max),
    max_index: integerOrNull(stats.max_index),
    mean: numberOrNull(stats.mean),
    rms: numberOrNull(stats.rms),
    max_abs: numberOrNull(stats.max_abs),
    max_abs_index: integerOrNull(stats.max_abs_index),
    first_values: numericArray(stats.checkpoint?.first_values ?? stats.first_values),
    max_abs_window_start: integerOrNull(stats.checkpoint?.max_abs_window_start ?? stats.max_abs_window_start),
    max_abs_window: numericArray(stats.checkpoint?.max_abs_window ?? stats.max_abs_window),
  }
}

function compactReconstruction(value) {
  if (!value || typeof value !== 'object') return null
  const keep = [
    'role',
    'layout',
    'epsilon',
    'hidden_mean_square',
    'hidden_rms',
    'input_mean_square',
    'input_rms',
    'delta_rms',
    'reported_rms',
    'delta_to_input_rms_ratio',
    'delta_input_cosine_similarity',
    'scale',
    'position',
    'effective_position',
    'pairing',
    'direction',
    'frequency_base',
    'attention_head_count',
    'attention_head_count_kv',
    'head_dim',
    'rope_dimension_count',
    'input_width',
    'output_width',
    'activation_order',
    'reported_max_abs_index',
    'reported_max_abs',
    'reported_max_abs_window_start',
    'max_abs_delta_index',
    'max_abs_delta',
  ]
  const out = {}
  for (const key of keep) {
    if (value[key] !== undefined) out[key] = copyScalarOrArray(value[key])
  }
  for (const key of Object.keys(value)) {
    if (key.endsWith('_first_values') || key.endsWith('_window') || key === 'weight_shape') {
      out[key] = copyScalarOrArray(value[key])
    }
  }
  return Object.keys(out).length > 0 ? out : null
}

function compactAttentionTrace(trace) {
  if (!trace) return null
  return {
    scale: numberOrNull(trace.scale),
    position_count: integerOrNull(trace.position_count),
    head_dim: integerOrNull(trace.head_dim),
    heads: (trace.heads ?? []).map(head => ({
      attention_head: integerOrNull(head.attention_head),
      kv_head: integerOrNull(head.kv_head),
      query_first_values: numericArray(head.query_first_values),
      context_first_values: numericArray(head.context_first_values),
      reconstructed_context_first_values: numericArray(head.reconstructed_context_first_values),
      context_reconstruction_max_abs_delta_index: integerOrNull(head.context_reconstruction_max_abs_delta_index),
      context_reconstruction_max_abs_delta: numberOrNull(head.context_reconstruction_max_abs_delta),
      probability_sum: numberOrNull(head.probability_sum),
      probability_entropy: numberOrNull(head.probability_entropy),
      probability_rms: numberOrNull(head.probability_rms),
      max_probability_position: integerOrNull(head.max_probability_position),
      max_probability: numberOrNull(head.max_probability),
      top_probability_positions: (head.top_probability_positions ?? []).map(position => ({
        position: integerOrNull(position.position),
        score: numberOrNull(position.score),
        probability: numberOrNull(position.probability),
        key_first_values: numericArray(position.key_first_values),
        value_first_values: numericArray(position.value_first_values),
      })),
      positions: (head.positions ?? []).map(position => ({
        position: integerOrNull(position.position),
        score: numberOrNull(position.score),
        reconstructed_score: numberOrNull(position.reconstructed_score),
        score_reconstruction_delta: numberOrNull(position.score_reconstruction_delta),
        probability: numberOrNull(position.probability),
        key_first_values: numericArray(position.key_first_values),
        qk_products_first_values: numericArray(position.qk_products_first_values),
        qk_products_max_abs_window_start: integerOrNull(position.qk_products_max_abs_window_start),
        qk_products_max_abs_window: numericArray(position.qk_products_max_abs_window),
        value_first_values: numericArray(position.value_first_values),
      })),
    })),
  }
}

function compactKvCacheTrace(trace) {
  if (!trace) return null
  return {
    layer_index: integerOrNull(trace.layer_index),
    position_count: integerOrNull(trace.position_count),
    kv_head_count: integerOrNull(trace.kv_head_count),
    head_dim: integerOrNull(trace.head_dim),
    key_value_width: integerOrNull(trace.key_value_width),
    key_checksum: numberOrNull(trace.key_checksum),
    value_checksum: numberOrNull(trace.value_checksum),
    key_rms: numberOrNull(trace.key_rms),
    value_rms: numberOrNull(trace.value_rms),
    key_max_abs: numberOrNull(trace.key_max_abs),
    key_max_abs_position: integerOrNull(trace.key_max_abs_position),
    key_max_abs_index: integerOrNull(trace.key_max_abs_index),
    value_max_abs: numberOrNull(trace.value_max_abs),
    value_max_abs_position: integerOrNull(trace.value_max_abs_position),
    value_max_abs_index: integerOrNull(trace.value_max_abs_index),
    sampled_positions: (trace.sampled_positions ?? []).map(position => ({
      position: integerOrNull(position.position),
      key_checksum: numberOrNull(position.key_checksum),
      value_checksum: numberOrNull(position.value_checksum),
      key_rms: numberOrNull(position.key_rms),
      value_rms: numberOrNull(position.value_rms),
      key_max_abs: numberOrNull(position.key_max_abs),
      value_max_abs: numberOrNull(position.value_max_abs),
      key_first_values: numericArray(position.key_first_values),
      value_first_values: numericArray(position.value_first_values),
    })),
  }
}

function compactOutputProjection(rows) {
  return rows.map(row => ({
    token_id: integerOrNull(row.token_id),
    layout: row.layout ?? null,
    reported_logit: numberOrNull(row.reported_logit),
    reconstructed_logit: numberOrNull(row.reconstructed_logit),
    absolute_delta: numberOrNull(row.absolute_delta),
    max_abs_component_index: integerOrNull(row.max_abs_component_index),
    max_abs_component: numberOrNull(row.max_abs_component),
    positive_component_sum: numberOrNull(row.positive_component_sum),
    negative_component_sum: numberOrNull(row.negative_component_sum),
    top_positive_components: compactProjectionComponents(row.top_positive_components),
    top_negative_components: compactProjectionComponents(row.top_negative_components),
  }))
}

function compactTopLogprobs(rows) {
  return (rows ?? []).map(row => ({
    id: integerOrNull(row.id),
    token: typeof row.token === 'string' ? row.token : null,
    logprob: numberOrNull(row.logprob),
  }))
}

function compactProjectionComponents(components) {
  return (components ?? []).map(component => ({
    index: integerOrNull(component.index),
    final_hidden_value: numberOrNull(component.final_hidden_value),
    output_norm_weight_value: numberOrNull(component.output_norm_weight_value),
    output_norm_scale: numberOrNull(component.output_norm_scale),
    reconstructed_output_norm_value: numberOrNull(component.reconstructed_output_norm_value),
    output_norm_value: numberOrNull(component.output_norm_value),
    output_row_value: numberOrNull(component.output_row_value),
    component: numberOrNull(component.component),
  }))
}

function copyScalarOrArray(value) {
  if (Array.isArray(value)) return value.map(copyScalarOrArray)
  if (typeof value === 'number') return numberOrNull(value)
  if (typeof value === 'string' || typeof value === 'boolean' || value === null) return value
  return value
}

function numericArray(value) {
  return Array.isArray(value) ? value.map(numberOrNull) : []
}

function numberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function integerOrNull(value) {
  return Number.isInteger(value) ? value : null
}
