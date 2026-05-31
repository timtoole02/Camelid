#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const scriptDir = dirname(fileURLToPath(import.meta.url))
const extractScript = join(scriptDir, 'extract-forward-trace.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-forward-trace-'))

try {
  const inputPath = join(tempDir, 'capture.json')
  const outputPath = join(tempDir, 'trace.json')
  await writeFile(inputPath, `${JSON.stringify(capture(), null, 2)}\n`)

  const { stdout } = await execFileAsync(process.execPath, [
    extractScript,
    '--input', inputPath,
    '--layers', '1,0,1',
    '--json-out', outputPath,
  ], { cwd: resolve(scriptDir, '..') })

  assert.match(stdout, /schema=camelid\.forward-trace\.v1/)
  assert.match(stdout, /selected_layers=0,1/)
  assert.match(stdout, /first_stage=embedding/)
  assert.match(stdout, /last_stage=logits/)
  assert.match(stdout, /stage_count=45/)

  const trace = JSON.parse(await readFile(outputPath, 'utf8'))
  assert.equal(trace.schema, 'camelid.forward-trace.v1')
  assert.deepEqual(trace.selected_layers, [0, 1])
  assert.equal(trace.stage_count, 45)
  assert.deepEqual(trace.prompt_token_ids, [1, 529, 29989])
  assert.deepEqual(trace.generated_token_ids, [16301])
  assert.deepEqual(trace.input_token_ids, [22557])
  assert.deepEqual(trace.position_ids, [8])
  assert.deepEqual(trace.source.known_good_token_ids, [29907])
  assert.deepEqual(trace.source.known_good_token_ids_from_text, [315])
  assert.deepEqual(trace.source.known_good_top_logprobs, [
    { id: 29907, token: 'C', logprob: -0.03 },
    { id: 315, token: ' C', logprob: -5.25 },
  ])
  assert.equal(trace.dense_metadata.output_is_tied_embedding, false)
  assert.deepEqual(trace.top_logits.map(row => row.token_id), [16301, 315])

  const paths = trace.stages.map(stage => stage.path)
  trace.stages.forEach((stage, index) => {
    assert.equal(stage.order, index)
  })
  assert.deepEqual(paths.slice(0, 8), [
    'embedding',
    'layers.0.attention_input',
    'layers.0.attention_norm',
    'layers.0.attention_q',
    'layers.0.attention_k',
    'layers.0.attention_v',
    'layers.0.attention_q_rope',
    'layers.0.attention_k_rope',
  ])
  assert.deepEqual(paths.slice(-4), ['final_hidden', 'final_norm', 'output_norm', 'logits'])
  assert.equal(paths.indexOf('layers.1.attention_input'), 21)
  assert.ok(paths.indexOf('layers.0.ffn_norm') < paths.indexOf('layers.0.moe_router'))
  assert.ok(paths.indexOf('layers.0.moe_router') < paths.indexOf('layers.0.ffn_gate'))
  assert.ok(paths.indexOf('layers.0.ffn_gate') < paths.indexOf('layers.0.ffn_activation'))
  assert.ok(paths.indexOf('layers.0.attention_q_rope') < paths.indexOf('layers.0.kv_cache_trace'))
  assert.ok(paths.indexOf('layers.0.kv_cache_trace') < paths.indexOf('layers.0.attention_trace'))
  assert.ok(paths.indexOf('layers.0.attention_trace') < paths.indexOf('layers.0.attention_context'))
  assert.ok(paths.indexOf('layers.0.ffn_residual') < paths.indexOf('layers.1.attention_input'))

  const qStage = trace.stages.find(stage => stage.path === 'layers.0.attention_q')
  assert.equal(qStage.kind, 'tensor_stats')
  assert.deepEqual(qStage.stats.shape, [1, 2])
  assert.deepEqual(qStage.stats.first_values, [0.1, -0.2])
  assert.equal(qStage.reconstruction.layout, 'descriptor')
  assert.equal(qStage.reconstruction.max_abs_delta, 0)
  assert.deepEqual(qStage.reconstruction.reported_max_abs_window, [0.1, -0.2])

  const kvStage = trace.stages.find(stage => stage.path === 'layers.0.kv_cache_trace')
  assert.equal(kvStage.kind, 'kv_cache_trace')
  assert.equal(kvStage.kv_cache_trace.position_count, 18)
  assert.equal(kvStage.kv_cache_trace.key_checksum, 12.5)
  assert.equal(kvStage.kv_cache_trace.sampled_positions[0].position, 17)
  assert.deepEqual(kvStage.kv_cache_trace.sampled_positions[0].value_first_values, [0.5, -0.6])

  const traceStage = trace.stages.find(stage => stage.path === 'layers.0.attention_trace')
  assert.equal(traceStage.kind, 'attention_trace')
  assert.equal(traceStage.attention_trace.heads[0].attention_head, 0)
  assert.equal(traceStage.attention_trace.heads[0].probability_entropy, 0.69)
  assert.equal(traceStage.attention_trace.heads[0].top_probability_positions[0].position, 17)
  assert.equal(traceStage.attention_trace.heads[0].top_probability_positions[0].probability, 1)
  assert.equal(traceStage.attention_trace.heads[0].positions[0].reconstructed_score, 0.5)
  assert.deepEqual(traceStage.attention_trace.heads[0].positions[0].qk_products_first_values, [0.05, 0.1])

  const residualStage = trace.stages.find(stage => stage.path === 'layers.0.ffn_residual')
  assert.equal(residualStage.residual_delta.delta_to_input_rms_ratio, 0.25)
  assert.equal(residualStage.residual_delta.delta_input_cosine_similarity, -0.1)

  const moeStage = trace.stages.find(stage => stage.path === 'layers.0.moe_router')
  assert.equal(moeStage.kind, 'moe_router')
  assert.deepEqual(moeStage.moe_router.rows[0].router_logits, [0.5, -0.25, 1.25])
  assert.equal(moeStage.moe_router.rows[0].selected_experts[0].expert_id, 2)
  assert.equal(moeStage.moe_router.rows[0].selected_experts[0].selected_weight, 0.75)

  const finalNorm = trace.stages.find(stage => stage.path === 'final_norm')
  assert.equal(finalNorm.kind, 'reconstruction')
  assert.equal(finalNorm.reconstruction.epsilon, 0.00001)
  assert.equal(finalNorm.reconstruction.scale, 2)

  assert.equal(trace.output_projection[0].token_id, 16301)
  assert.equal(trace.output_projection[0].top_positive_components[0].index, 1315)
  assert.equal(trace.output_projection[1].top_negative_components[0].component, -0.37)

  const oneLayerPath = join(tempDir, 'trace-one-layer.json')
  const { stdout: oneLayerStdout } = await execFileAsync(process.execPath, [
    extractScript,
    '--input', inputPath,
    '--layers', '1',
    '--json-out', oneLayerPath,
  ], { cwd: resolve(scriptDir, '..') })
  assert.match(oneLayerStdout, /selected_layers=1/)
  assert.match(oneLayerStdout, /stage_count=25/)
  const oneLayerTrace = JSON.parse(await readFile(oneLayerPath, 'utf8'))
  assert.deepEqual(oneLayerTrace.selected_layers, [1])
  assert.ok(!oneLayerTrace.stages.some(stage => stage.path.startsWith('layers.0.')))
  assert.equal(oneLayerTrace.stages[1].path, 'layers.1.attention_input')

  await assert.rejects(
    () => execFileAsync(process.execPath, [extractScript, '--input', inputPath, '--layers', '2', '--json-out', join(tempDir, 'bad.json')], { cwd: resolve(scriptDir, '..') }),
    /requested layer 2, but diagnostics only contain 2 layer\(s\)/,
  )

  const sparsePath = join(tempDir, 'sparse-capture.json')
  await writeFile(sparsePath, `${JSON.stringify({ camelid: { prompt_token_ids: [1] } }, null, 2)}\n`)
  await assert.rejects(
    () => execFileAsync(process.execPath, [extractScript, '--input', sparsePath, '--json-out', join(tempDir, 'sparse-trace.json')], { cwd: resolve(scriptDir, '..') }),
    /missing camelid\.dense diagnostics; rerun with camelid_dense_diagnostics=true/,
  )

  console.log('extract-forward-trace self-test passed')
} finally {
  await rm(tempDir, { recursive: true, force: true })
}

function capture() {
  return {
    prompt_tokens_match: true,
    generated_text_match: false,
    backend_text: 'lei',
    llama_text: 'C',
    llama_generated_tokens: [29907],
    llama_generated_tokens_from_text: [315],
    llama_top_logprobs: [
      { id: 29907, token: 'C', logprob: -0.03 },
      { id: 315, token: ' C', logprob: -5.25 },
    ],
    camelid: {
      prompt_token_ids: [1, 529, 29989],
      generated_token_ids: [16301],
      dense_metadata: {
        embedding_length: 2,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        output_is_tied_embedding: false,
      },
      top_logits: [
        { token_id: 16301, text: 'lei', logit: 7.5, rank: 1, selected: true },
        { token_id: 315, text: 'C', logit: -0.1, rank: 12003, selected: false },
      ],
      output_projection: [
        outputProjectionRow({ tokenId: 16301, maxIndex: 1315, positiveComponent: 1.37, negativeComponent: -0.25 }),
        outputProjectionRow({ tokenId: 315, maxIndex: 945, positiveComponent: 0.35, negativeComponent: -0.37 }),
      ],
      dense: {
        input_token_ids: [22557],
        position_ids: [8],
        embedding: stats([0.25, -0.5]),
        layers: [layer(0), layer(1)],
        final_hidden: stats([0.9, -0.7]),
        final_norm: reconstruction({ epsilon: 0.00001, scale: 2 }),
        output_norm: stats([1.8, -1.4]),
        logits: stats([7.5, -0.1]),
      },
    },
  }
}

function layer(layerIndex) {
  return {
    layer_index: layerIndex,
    residual_flow: {
      attention_input: stats([0.25 + layerIndex, -0.5]),
      attention_delta: residualDelta(),
      ffn_input: stats([0.3 + layerIndex, -0.4]),
      ffn_delta: residualDelta(),
    },
    attention_norm: stats([0.2 + layerIndex, -0.1]),
    attention_norm_reconstruction: reconstruction({ epsilon: 0.00001, scale: 1.5 }),
    attention_q: stats([0.1 + layerIndex, -0.2]),
    attention_q_reconstruction: reconstruction({ layout: 'descriptor', role: 'attention_q' }),
    attention_k: stats([0.3 + layerIndex, -0.4]),
    attention_k_reconstruction: reconstruction({ layout: 'descriptor', role: 'attention_k' }),
    attention_v: stats([0.5 + layerIndex, -0.6]),
    attention_v_reconstruction: reconstruction({ layout: 'descriptor', role: 'attention_v' }),
    attention_q_rope: stats([0.11 + layerIndex, -0.19]),
    attention_q_rope_reconstruction: reconstruction({ pairing: 'adjacent_even_odd', position: 17, effective_position: 17 }),
    attention_k_rope: stats([0.31 + layerIndex, -0.39]),
    attention_k_rope_reconstruction: reconstruction({ pairing: 'adjacent_even_odd', position: 17, effective_position: 17 }),
    kv_cache_trace: kvCacheTrace(layerIndex),
    attention_trace: attentionTrace(),
    attention_context: stats([0.5 + layerIndex, -0.6]),
    attention_output: stats([0.07 + layerIndex, -0.08]),
    attention_output_reconstruction: reconstruction({ layout: 'descriptor', role: 'attention_output' }),
    attention_residual: stats([0.32 + layerIndex, -0.58]),
    ffn_norm: stats([0.32 + layerIndex, -0.58]),
    ffn_norm_reconstruction: reconstruction({ epsilon: 0.00001, scale: 1.25 }),
    moe_router: moeRouter(),
    ffn_gate: stats([0.7 + layerIndex, -0.8]),
    ffn_gate_reconstruction: reconstruction({ layout: 'descriptor', role: 'ffn_gate' }),
    ffn_up: stats([0.9 + layerIndex, -1]),
    ffn_up_reconstruction: reconstruction({ layout: 'descriptor', role: 'ffn_up' }),
    ffn_activation: stats([0.47 + layerIndex, 0.31]),
    ffn_activation_reconstruction: reconstruction({ activation_order: 'silu_gate_times_up' }),
    ffn_output: stats([0.12 + layerIndex, -0.14]),
    ffn_down_reconstruction: reconstruction({ layout: 'descriptor', role: 'ffn_down' }),
    ffn_residual: stats([0.44 + layerIndex, -0.72]),
  }
}

function stats(values) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const maxAbs = Math.max(...values.map(value => Math.abs(value)))
  return {
    min,
    min_index: values.indexOf(min),
    max,
    max_index: values.indexOf(max),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length),
    max_abs: maxAbs,
    max_abs_index: values.findIndex(value => Math.abs(value) === maxAbs),
    checkpoint: {
      shape: [1, values.length],
      len: values.length,
      first_values: values,
      max_abs_window_start: 0,
      max_abs_window: values,
    },
  }
}

function reconstruction(extra) {
  return {
    input_width: 2,
    output_width: 2,
    weight_shape: [2, 2],
    reported_max_abs_index: 1,
    reported_max_abs: 0.2,
    reported_max_abs_window_start: 0,
    reported_max_abs_window: [0.1, -0.2],
    reconstructed_reported_max_abs_window: [0.1, -0.2],
    input_first_values: [0.25, -0.5],
    weight_first_values: [1, 0.5],
    reconstructed_first_values: [0.1, -0.2],
    reported_first_values: [0.1, -0.2],
    max_abs_delta_index: 0,
    max_abs_delta: 0,
    ...extra,
  }
}

function residualDelta() {
  return {
    input_rms: 0.5,
    delta_rms: 0.125,
    reported_rms: 0.6,
    delta_to_input_rms_ratio: 0.25,
    delta_input_cosine_similarity: -0.1,
    max_abs_delta_index: 0,
    max_abs_delta: 0,
    input_first_values: [0.25, -0.5],
    delta_first_values: [0.07, -0.08],
    reconstructed_first_values: [0.32, -0.58],
    reported_first_values: [0.32, -0.58],
  }
}

function attentionTrace() {
  return {
    scale: 0.125,
    position_count: 18,
    head_dim: 2,
    heads: [
      {
        attention_head: 0,
        kv_head: 0,
        query_first_values: [0.1, -0.2],
        context_first_values: [0.5, -0.6],
        reconstructed_context_first_values: [0.5, -0.6],
        context_reconstruction_max_abs_delta_index: 0,
        context_reconstruction_max_abs_delta: 0,
        probability_sum: 1,
        probability_entropy: 0.69,
        probability_rms: 0.5,
        max_probability_position: 17,
        max_probability: 1,
        top_probability_positions: [
          {
            position: 17,
            score: 0.5,
            probability: 1,
            key_first_values: [0.3, -0.4],
            value_first_values: [0.5, -0.6],
          },
        ],
        positions: [
          {
            position: 17,
            score: 0.5,
            reconstructed_score: 0.5,
            score_reconstruction_delta: 0,
            probability: 1,
            key_first_values: [0.3, -0.4],
            qk_products_first_values: [0.05, 0.1],
            qk_products_max_abs_window_start: 0,
            qk_products_max_abs_window: [0.05, 0.1],
            value_first_values: [0.5, -0.6],
          },
        ],
      },
    ],
  }
}

function kvCacheTrace(layerIndex) {
  return {
    layer_index: layerIndex,
    position_count: 18,
    kv_head_count: 1,
    head_dim: 2,
    key_value_width: 2,
    key_checksum: 12.5 + layerIndex,
    value_checksum: -3.25 - layerIndex,
    key_rms: 0.35,
    value_rms: 0.42,
    key_max_abs: 0.4,
    key_max_abs_position: 17,
    key_max_abs_index: 1,
    value_max_abs: 0.6,
    value_max_abs_position: 17,
    value_max_abs_index: 1,
    sampled_positions: [
      {
        position: 17,
        key_checksum: 1.1 + layerIndex,
        value_checksum: -0.7 - layerIndex,
        key_rms: 0.35,
        value_rms: 0.42,
        key_max_abs: 0.4,
        value_max_abs: 0.6,
        key_first_values: [0.3 + layerIndex, -0.4],
        value_first_values: [0.5, -0.6],
      },
    ],
  }
}

function moeRouter() {
  return {
    shape: [1, 3],
    rows: [
      {
        row_index: 0,
        router_logits: [0.5, -0.25, 1.25],
        router_probabilities: [0.29, 0.14, 0.57],
        selected_experts: [
          {
            expert_id: 2,
            selected_rank: 0,
            router_logit: 1.25,
            router_probability: 0.57,
            selected_weight: 0.75,
          },
          {
            expert_id: 0,
            selected_rank: 1,
            router_logit: 0.5,
            router_probability: 0.29,
            selected_weight: 0.25,
          },
        ],
      },
    ],
  }
}

function outputProjectionRow({ tokenId, maxIndex, positiveComponent, negativeComponent }) {
  return {
    token_id: tokenId,
    layout: 'descriptor',
    reported_logit: tokenId === 16301 ? 7.5 : -0.1,
    reconstructed_logit: tokenId === 16301 ? 7.5 : -0.1,
    absolute_delta: 0,
    max_abs_component_index: maxIndex,
    max_abs_component: Math.abs(positiveComponent) > Math.abs(negativeComponent) ? positiveComponent : negativeComponent,
    positive_component_sum: positiveComponent + 0.1,
    negative_component_sum: negativeComponent - 0.1,
    top_positive_components: [component({ index: maxIndex, component: positiveComponent })],
    top_negative_components: [component({ index: maxIndex + 1, component: negativeComponent })],
  }
}

function component({ index, component }) {
  return {
    index,
    final_hidden_value: 0.9,
    output_norm_weight_value: 2,
    output_norm_scale: 1.1,
    reconstructed_output_norm_value: 1.98,
    output_norm_value: 1.98,
    output_row_value: component / 1.98,
    component,
  }
}
