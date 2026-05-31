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
const compareScript = join(scriptDir, 'compare-forward-traces.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-forward-trace-compare-'))

try {
  const leftPath = join(tempDir, 'left.json')
  const rightPath = join(tempDir, 'right.json')
  const reportPath = join(tempDir, 'report.json')
  const left = trace()
  const right = clone(left)
  const attentionQ = right.stages.find(stage => stage.path === 'layers.0.attention_q')
  attentionQ.stats.first_values[1] = 0.42
  attentionQ.stats.max_abs_window[1] = 0.42
  attentionQ.reconstruction.reported_first_values[1] = 0.42
  right.stages.find(stage => stage.path === 'output_norm').stats.first_values[0] = 1.1
  right.stages.find(stage => stage.path === 'logits').stats.max = 9.5
  right.stages.find(stage => stage.path === 'logits').stats.first_values[0] = 9.5
  right.top_logits[0].logit = 9.5
  right.output_projection[0].top_positive_components[0].component = 1.5
  right.generated_token_ids = [315]

  await writeFile(leftPath, `${JSON.stringify(left, null, 2)}\n`)
  await writeFile(rightPath, `${JSON.stringify(right, null, 2)}\n`)

  const { stdout } = await execFileAsync(process.execPath, [
    compareScript,
    '--left', leftPath,
    '--right', rightPath,
    '--json-out', reportPath,
    '--top', '5',
  ], { cwd: resolve(scriptDir, '..') })

  assert.match(stdout, /schema=camelid\.forward-trace-comparison\.v1/)
  assert.match(stdout, /input_token_delta=\{"left":\[16301\],"right":\[16301\],"match":true\}/)
  assert.match(stdout, /position_id_delta=\{"left":\[17\],"right":\[17\],"match":true\}/)
  assert.match(stdout, /known_good_token_delta=\{"left":\[29907\],"right":\[29907\],"match":true\}/)
  assert.match(stdout, /stage_paths_match=true/)
  assert.match(stdout, /first_changed_stage=layers\.0\.attention_q/)
  assert.match(stdout, /changed_stage_count=3/)

  const report = JSON.parse(await readFile(reportPath, 'utf8'))
  assert.equal(report.schema, 'camelid.forward-trace-comparison.v1')
  assert.equal(report.prompt_tokens_match, true)
  assert.deepEqual(report.input_token_delta.left, [16301])
  assert.deepEqual(report.input_token_delta.right, [16301])
  assert.equal(report.input_token_delta.match, true)
  assert.deepEqual(report.position_id_delta.left, [17])
  assert.deepEqual(report.position_id_delta.right, [17])
  assert.equal(report.position_id_delta.match, true)
  assert.deepEqual(report.generated_token_delta.left, [16301])
  assert.deepEqual(report.generated_token_delta.right, [315])
  assert.equal(report.generated_token_delta.match, false)
  assert.deepEqual(report.known_good_token_delta.left, [29907])
  assert.deepEqual(report.known_good_token_delta.right, [29907])
  assert.equal(report.known_good_token_delta.match, true)
  assert.equal(report.stage_path_alignment.match, true)
  assert.equal(report.first_changed_stage.path, 'layers.0.attention_q')
  assert.equal(report.first_changed_stage.order, 3)
  assert.equal(report.first_changed_stage.changed_field_count, 3)
  assert.ok(report.first_changed_stage.top_changed_paths.some(change => change.path === 'layers.0.attention_q.stats.first_values[1]'))
  assert.equal(report.largest_stage_deltas[0].path, 'logits')
  assert.equal(report.top_logit_deltas[0].token_id, 16301)
  assert.equal(report.output_projection_token_deltas[0].token_id, 16301)

  const sameReportPath = join(tempDir, 'same-report.json')
  const { stdout: sameStdout } = await execFileAsync(process.execPath, [
    compareScript,
    '--left', leftPath,
    '--right', leftPath,
    '--json-out', sameReportPath,
  ], { cwd: resolve(scriptDir, '..') })
  assert.match(sameStdout, /first_changed_stage=none/)
  assert.match(sameStdout, /changed_stage_count=0/)
  const sameReport = JSON.parse(await readFile(sameReportPath, 'utf8'))
  assert.equal(sameReport.first_changed_stage, null)
  assert.equal(sameReport.changed_stage_count, 0)
  assert.deepEqual(sameReport.largest_stage_deltas, [])

  const tolerantRightPath = join(tempDir, 'tolerant-right.json')
  const tolerantRight = clone(left)
  tolerantRight.stages[0].stats.rms += 0.0000005
  await writeFile(tolerantRightPath, `${JSON.stringify(tolerantRight, null, 2)}\n`)
  const tolerantReportPath = join(tempDir, 'tolerant-report.json')
  await execFileAsync(process.execPath, [
    compareScript,
    '--left', leftPath,
    '--right', tolerantRightPath,
    '--tolerance', '0.000001',
    '--json-out', tolerantReportPath,
  ], { cwd: resolve(scriptDir, '..') })
  const tolerantReport = JSON.parse(await readFile(tolerantReportPath, 'utf8'))
  assert.equal(tolerantReport.changed_stage_count, 0)
  assert.equal(tolerantReport.first_changed_stage, null)

  const mismatchedPath = join(tempDir, 'mismatched.json')
  const mismatched = clone(left)
  mismatched.stages.splice(2, 1)
  mismatched.stages.forEach((stage, index) => { stage.order = index })
  mismatched.stage_count = mismatched.stages.length
  await writeFile(mismatchedPath, `${JSON.stringify(mismatched, null, 2)}\n`)
  const mismatchReportPath = join(tempDir, 'mismatch-report.json')
  const { stdout: mismatchStdout } = await execFileAsync(process.execPath, [
    compareScript,
    '--left', leftPath,
    '--right', mismatchedPath,
    '--json-out', mismatchReportPath,
  ], { cwd: resolve(scriptDir, '..') })
  assert.match(mismatchStdout, /stage_paths_match=false/)
  const mismatchReport = JSON.parse(await readFile(mismatchReportPath, 'utf8'))
  assert.equal(mismatchReport.stage_path_alignment.match, false)
  assert.deepEqual(mismatchReport.stage_path_alignment.first_mismatch, {
    order: 2,
    left: 'layers.0.attention_norm',
    right: 'layers.0.attention_q',
  })
  assert.equal(mismatchReport.first_changed_stage.reason, 'stage_path_mismatch')

  const badCountPath = join(tempDir, 'bad-count.json')
  const badCount = clone(left)
  badCount.stage_count = badCount.stages.length + 1
  await writeFile(badCountPath, `${JSON.stringify(badCount, null, 2)}\n`)
  await assert.rejects(
    () => execFileAsync(process.execPath, [compareScript, '--left', leftPath, '--right', badCountPath], { cwd: resolve(scriptDir, '..') }),
    /stage_count 24 does not match stages length 23/,
  )

  await assert.rejects(
    () => execFileAsync(process.execPath, [compareScript, '--left', leftPath, '--right', join(tempDir, 'missing.json')], { cwd: resolve(scriptDir, '..') }),
    /ENOENT/,
  )

  console.log('compare-forward-traces self-test passed')
} finally {
  await rm(tempDir, { recursive: true, force: true })
}

function trace() {
  const stages = [
    stage(0, 'embedding', stats([0.10, -0.20])),
    stage(1, 'layers.0.attention_input', stats([0.10, -0.20]), { layer_index: 0 }),
    stage(2, 'layers.0.attention_norm', stats([0.20, -0.30]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_norm', epsilon: 0.00001 }) }),
    stage(3, 'layers.0.attention_q', stats([0.30, -0.40]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_q', layout: 'descriptor' }) }),
    stage(4, 'layers.0.attention_k', stats([0.35, -0.45]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_k', layout: 'descriptor' }) }),
    stage(5, 'layers.0.attention_v', stats([0.40, -0.50]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_v', layout: 'descriptor' }) }),
    stage(6, 'layers.0.attention_q_rope', stats([0.31, -0.39]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_q_rope', pairing: 'adjacent_even_odd' }) }),
    stage(7, 'layers.0.attention_k_rope', stats([0.36, -0.44]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_k_rope', pairing: 'adjacent_even_odd' }) }),
    {
      order: 8,
      path: 'layers.0.attention_trace',
      kind: 'attention_trace',
      layer_index: 0,
      attention_trace: {
        scale: 0.125,
        position_count: 18,
        head_dim: 64,
        heads: [
          {
            attention_head: 0,
            kv_head: 0,
            query_first_values: [0.31, -0.39],
            context_first_values: [0.11, 0.12],
            reconstructed_context_first_values: [0.11, 0.12],
            context_reconstruction_max_abs_delta: 0,
            probability_sum: 1,
            probability_entropy: 0.7,
            probability_rms: 0.25,
            positions: [
              {
                position: 17,
                score: 0.5,
                reconstructed_score: 0.5,
                score_reconstruction_delta: 0,
                probability: 1,
                key_first_values: [0.36, -0.44],
                qk_products_first_values: [0.1116, 0.1716],
                value_first_values: [0.40, -0.50],
              },
            ],
          },
        ],
      },
    },
    stage(9, 'layers.0.attention_context', stats([0.11, 0.12]), { layer_index: 0 }),
    stage(10, 'layers.0.attention_output', stats([0.13, -0.14]), { layer_index: 0, reconstruction: reconstruction({ role: 'attention_output', layout: 'descriptor' }) }),
    stage(11, 'layers.0.attention_residual', stats([0.23, -0.34]), { layer_index: 0, residual_delta: residualDelta() }),
    stage(12, 'layers.0.ffn_input', stats([0.23, -0.34]), { layer_index: 0 }),
    stage(13, 'layers.0.ffn_norm', stats([0.24, -0.35]), { layer_index: 0, reconstruction: reconstruction({ role: 'ffn_norm', epsilon: 0.00001 }) }),
    stage(14, 'layers.0.ffn_gate', stats([0.50, -0.60]), { layer_index: 0, reconstruction: reconstruction({ role: 'ffn_gate', layout: 'descriptor' }) }),
    stage(15, 'layers.0.ffn_up', stats([0.70, -0.80]), { layer_index: 0, reconstruction: reconstruction({ role: 'ffn_up', layout: 'descriptor' }) }),
    stage(16, 'layers.0.ffn_activation', stats([0.19, 0.31]), { layer_index: 0, reconstruction: reconstruction({ role: 'ffn_activation', activation_order: 'silu_gate_times_up' }) }),
    stage(17, 'layers.0.ffn_output', stats([0.21, -0.22]), { layer_index: 0, reconstruction: reconstruction({ role: 'ffn_down', layout: 'descriptor' }) }),
    stage(18, 'layers.0.ffn_residual', stats([0.44, -0.56]), { layer_index: 0, residual_delta: residualDelta() }),
    stage(19, 'final_hidden', stats([0.44, -0.56])),
    { order: 20, path: 'final_norm', kind: 'reconstruction', reconstruction: reconstruction({ role: 'final_norm', epsilon: 0.00001, scale: 1.9 }) },
    stage(21, 'output_norm', stats([0.84, -1.06])),
    stage(22, 'logits', stats([7.5, -0.1])),
  ]
  return {
    schema: 'camelid.forward-trace.v1',
    source: {
      input: 'chat-parity.json',
      prompt_tokens_match: true,
      selected_token_id: 16301,
      known_good_token_ids: [29907],
      known_good_token_ids_from_text: [315],
      backend_text: 'lei',
      known_good_text: 'C',
    },
    prompt_token_ids: [1, 529, 29989, 13],
    input_token_ids: [16301],
    position_ids: [17],
    generated_token_ids: [16301],
    dense_metadata: {
      attention_head_count: 32,
      attention_head_count_kv: 4,
      output_shape: [2048, 32000],
      output_is_tied_embedding: false,
      ffn_gate_up_order: 'gate_up',
    },
    layer_count: 1,
    selected_layers: [0],
    stage_count: stages.length,
    stages,
    top_logits: [
      { token_id: 16301, text: 'lei', logit: 7.5, rank: 1, selected: true },
      { token_id: 315, text: 'C', logit: -0.1, rank: 12003, selected: false },
    ],
    output_projection: [
      {
        token_id: 16301,
        text: 'lei',
        reconstructed_logit: 7.5,
        top_positive_components: [{ index: 1315, component: 1.37 }],
        top_negative_components: [{ index: 1885, component: -0.25 }],
      },
      {
        token_id: 315,
        text: 'C',
        reconstructed_logit: -0.1,
        top_positive_components: [{ index: 945, component: 0.35 }],
        top_negative_components: [{ index: 1885, component: -0.37 }],
      },
    ],
  }
}

function stage(order, path, statsValue, extra = {}) {
  return { order, path, kind: 'tensor_stats', ...extra, stats: statsValue }
}

function stats(values) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const maxAbs = Math.max(...values.map(value => Math.abs(value)))
  const maxAbsIndex = values.findIndex(value => Math.abs(value) === maxAbs)
  return {
    shape: [1, values.length],
    len: values.length,
    min,
    min_index: values.indexOf(min),
    max,
    max_index: values.indexOf(max),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length),
    max_abs: maxAbs,
    max_abs_index: maxAbsIndex,
    first_values: values,
    max_abs_window_start: 0,
    max_abs_window: values,
  }
}

function reconstruction(extra = {}) {
  return {
    input_width: 2,
    output_width: 2,
    weight_shape: [2, 2],
    input_first_values: [0.10, -0.20],
    weight_first_values: [1, 0.5],
    reconstructed_first_values: [0.30, -0.40],
    reported_first_values: [0.30, -0.40],
    reported_max_abs_index: 1,
    reported_max_abs: 0.40,
    reported_max_abs_window_start: 0,
    reported_max_abs_window: [0.30, -0.40],
    max_abs_delta_index: 0,
    max_abs_delta: 0,
    ...extra,
  }
}

function residualDelta() {
  return {
    input_rms: 0.30,
    delta_rms: 0.10,
    reported_rms: 0.40,
    delta_to_input_rms_ratio: 0.33,
    delta_input_cosine_similarity: 0.25,
    input_first_values: [0.10, -0.20],
    delta_first_values: [0.13, -0.14],
    reconstructed_first_values: [0.23, -0.34],
    reported_first_values: [0.23, -0.34],
    max_abs_delta_index: 0,
    max_abs_delta: 0,
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}
