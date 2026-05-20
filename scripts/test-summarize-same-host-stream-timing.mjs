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
const summaryScript = join(scriptDir, 'summarize-same-host-stream-timing.mjs')
const tempDir = await mkdtemp(join(tmpdir(), 'camelid-same-host-stream-summary-'))

try {
  const inputPath = join(tempDir, 'same-host.json')
  const outPath = join(tempDir, 'summary.json')
  await writeFile(inputPath, `${JSON.stringify(fixture(), null, 2)}\n`)

  const { stdout } = await execFileAsync(process.execPath, [
    summaryScript,
    '--input', inputPath,
    '--out', outPath,
  ], { cwd: resolve(scriptDir, '..') })

  assert.match(stdout, /schema=camelid\.same_host_stream_timing_summary\.v1/)
  assert.match(stdout, /runs=3/)
  assert.match(stdout, /client_first_byte_mean_ms=25\.000/)
  assert.match(stdout, /backend_first_content_mean_ms=210\.000/)
  assert.match(stdout, /first_content_minus_first_byte_mean_ms=195\.000/)
  assert.match(stdout, /backend_generate_minus_first_content_mean_ms=590\.000/)
  assert.match(stdout, /prompt_eval_forward_mean_ms=120\.000/)
  assert.match(stdout, /prompt_eval_logits_mean_ms=12\.000/)
  assert.match(stdout, /prompt_eval_sample_mean_ms=1\.500/)
  assert.match(stdout, /first_content_minus_prompt_eval_forward_mean_ms=90\.000/)
  assert.match(stdout, /role_yield_mean_ms=2\.000/)
  assert.match(stdout, /client_first_byte_minus_role_yield_mean_ms=23\.000/)
  assert.match(stdout, /client_first_content_minus_content_yield_mean_ms=7\.000/)
  assert.match(stdout, /client_minus_backend_first_content_mean_ms=10\.000/)
  assert.match(stdout, /q8_total_gemm_mean_ms=126\.000/)
  assert.match(stdout, /q8_fused_gate_up_calls_mean=28\.000/)
  assert.match(stdout, /projection_route_calls_mean=8\.000/)
  assert.match(stdout, /q8_gate_up_decode_consumer_post_route_mean_ms=4\.650/)
  assert.match(stdout, /1\. generation\.ffn_down mean=92\.000ms/)
  assert.match(stdout, /1\. ffn_down gemm_mean=70\.000ms pack_mean=7\.000ms calls_mean=28\.000/)
  assert.match(stdout, /1\. logits\.q8_0_borrowed_packed_rows4 elapsed_mean=18\.000ms calls_mean=8\.000/)
  assert.match(stdout, /top_projection_routes:/)
  assert.match(stdout, /top_projection_layer_routes:/)
  assert.match(stdout, /1\. generation\.layer_5\.ffn_down\.mac_decode_consumer elapsed_mean=12\.000ms calls_mean=4\.000/)
  assert.match(stdout, /projection_route_denials:/)
  assert.match(stdout, /1\. ffn_gate_up\.decode_consumer\.stream_diagnostics_collect_projection_details denials_mean=28\.000/)
  assert.match(stdout, /1\. generation\.L5\.attention_output mean=13\.000ms/)
  assert.match(stdout, /top_stage_route_role_gaps:/)
  assert.match(stdout, /1\. generation\.ffn_down\.mac_decode_consumer role_mean=92\.000ms route_mean=18\.000ms role_minus_route_mean=74\.000ms calls_mean=8\.000 matched=ffn_down/)
  assert.match(stdout, /top_layer_route_role_gaps:/)
  assert.match(stdout, /1\. generation\.L4\.ffn_down\.mac_decode_consumer role_mean=7\.000ms route_mean=6\.000ms role_minus_route_mean=1\.000ms matched=ffn_down/)
  assert.match(stdout, /role_focus_logits_attention:/)
  assert.match(stdout, /1\. attention_output total_mean=41\.000ms/)

  const report = JSON.parse(await readFile(outPath, 'utf8'))
  assert.equal(report.schema, 'camelid.same_host_stream_timing_summary.v1')
  assert.equal(report.aggregate.camelid_client_first_byte_ms.mean, 25)
  assert.equal(report.aggregate.camelid_client_ttft_ms.mean, 220)
  assert.equal(report.aggregate.llama_cpp_ttft_ms.mean, 150)
  assert.equal(report.aggregate.first_content_minus_first_byte_ms.mean, 195)
  assert.equal(report.aggregate.backend_generate_minus_first_byte_ms.mean, 775)
  assert.equal(report.aggregate.backend_generate_minus_first_content_ms.mean, 590)
  assert.equal(report.aggregate.prompt_eval_forward_ms.mean, 120)
  assert.equal(report.aggregate.prompt_eval_logits_ms.mean, 12)
  assert.equal(report.aggregate.prompt_eval_sample_ms.mean, 1.5)
  assert.equal(report.aggregate.first_content_minus_prompt_eval_forward_ms.mean, 90)
  assert.equal(report.aggregate.first_content_minus_prompt_eval_forward_plus_sample_ms.mean, 88.5)
  assert.equal(report.aggregate.role_yield_ms.mean, 2)
  assert.equal(report.aggregate.first_content_yield_ms.mean, 213)
  assert.equal(report.aggregate.final_yield_ms.mean, 805)
  assert.equal(report.aggregate.client_first_byte_minus_role_yield_ms.mean, 23)
  assert.equal(report.aggregate.client_first_content_minus_content_yield_ms.mean, 7)
  assert.equal(report.aggregate.client_done_minus_final_yield_ms.mean, 295)
  assert.equal(report.aggregate.backend_first_content_delta_vs_llama_cpp_ttft_ms.mean, 60)
  assert.equal(report.aggregate.q8_calls.mean, 140)
  assert.equal(report.aggregate.q8_total_gemm_us.mean, 126000)
  assert.equal(report.aggregate.q8_total_pack_us.mean, 12600)
  assert.equal(report.aggregate.q8_fused_gate_up_calls.mean, 28)
  assert.equal(report.aggregate.q8_gate_up_decode_consumer_activation_us.mean, 4200)
  assert.equal(report.aggregate.q8_gate_up_decode_consumer_tensor_us.mean, 450)
  assert.equal(report.q8_gate_up_decode_consumer_overhead.total_ms_mean, 4.65)
  assert.equal(report.aggregate.projection_route_calls.mean, 8)
  assert.equal(report.aggregate.output_projection_calls.mean, 8)
  assert.equal(report.aggregate.prompt_cache_hits, 0)
  assert.equal(report.aggregate.weight_cache_hits, 3)
  assert.equal(report.stages.prefill.ffn_down.mean, 40)
  assert.equal(report.stages.first_token.logits.mean, 5)
  assert.equal(report.ranked_roles[0].stage, 'generation')
  assert.equal(report.ranked_roles[0].role, 'ffn_down')
  assert.equal(report.q8_role_work[0].role, 'ffn_down')
  assert.equal(report.q8_role_work[0].gemm_ms_mean, 70)
  assert.equal(report.q8_role_work[1].role, 'attention_output')
  assert.equal(report.projection_routes[0].key, 'logits.q8_0_borrowed_packed_rows4')
  assert.equal(report.projection_routes[0].elapsed_ms_mean, 18)
  assert.equal(report.projection_routes[0].calls.mean, 8)
  assert.equal(report.projection_routes[0].output_width.mean, 128256)
  assert.ok(report.projection_routes.some((route) => (
    route.key === 'ffn_down.x86_vnni_decode_consumer'
    && route.calls.mean === 3
    && route.output_width.mean === 8192
  )))
  assert.equal(report.output_projection_routes[0].key, 'logits.q8_0_borrowed_packed_rows4')
  assert.equal(report.projection_layer_routes[0].key, 'generation.layer_5.ffn_down.mac_decode_consumer')
  assert.equal(report.projection_layer_routes[0].stage, 'generation')
  assert.equal(report.projection_layer_routes[0].elapsed_ms_mean, 12)
  assert.equal(report.projection_layer_routes[0].layer_index, 5)
  assert.equal(report.output_projection_layer_routes[0].key, 'generation.layer_5.ffn_down.mac_decode_consumer')
  assert.equal(report.projection_route_denials[0].key, 'ffn_gate_up.decode_consumer.stream_diagnostics_collect_projection_details')
  assert.equal(report.projection_route_denials[0].denials.mean, 28)
  assert.equal(report.projection_route_denials[0].output_width.mean, 8192)
  assert.ok(report.projection_route_denials.some((denial) => (
    denial.key === 'ffn_down.x86_vnni_decode_consumer.gate_off'
    && denial.denials.mean === 2
    && denial.output_width.mean === 8192
  )))
  assert.equal(report.layer_role_hotspots[0].stage, 'generation')
  assert.equal(report.layer_role_hotspots[0].layer_index, 5)
  assert.equal(report.layer_role_hotspots[0].role, 'attention_output')
  assert.equal(report.layer_role_hotspots[0].elapsed_ms.mean, 13)
  assert.equal(report.stage_route_role_gaps[0].stage, 'generation')
  assert.equal(report.stage_route_role_gaps[0].projection_role, 'ffn_down')
  assert.equal(report.stage_route_role_gaps[0].route, 'mac_decode_consumer')
  assert.equal(report.stage_route_role_gaps[0].route_elapsed_ms.mean, 18)
  assert.equal(report.stage_route_role_gaps[0].role_elapsed_ms.mean, 92)
  assert.equal(report.stage_route_role_gaps[0].role_minus_route_ms.mean, 74)
  assert.equal(report.stage_route_role_gaps[0].calls.mean, 8)
  assert.deepEqual(report.stage_route_role_gaps[0].matched_roles, ['ffn_down'])
  assert.equal(report.layer_route_role_gaps[0].stage, 'generation')
  assert.equal(report.layer_route_role_gaps[0].layer_index, 4)
  assert.equal(report.layer_route_role_gaps[0].projection_role, 'ffn_down')
  assert.equal(report.layer_route_role_gaps[0].route_elapsed_ms.mean, 6)
  assert.equal(report.layer_route_role_gaps[0].role_elapsed_ms.mean, 7)
  assert.equal(report.layer_route_role_gaps[0].role_minus_route_ms.mean, 1)
  assert.equal(report.layer_route_role_gaps[0].route_stage, 'generation')
  assert.deepEqual(report.layer_route_role_gaps[0].matched_roles, ['ffn_down'])
  assert.equal(report.role_focus[0].role, 'attention_output')
  assert.equal(report.role_focus[0].total_mean_ms, 41)
  assert.equal(report.role_focus.find((row) => row.role === 'logits').total_mean_ms, 20)
  assert.equal(report.role_focus.find((row) => row.role === 'attention_context').stages.generation.mean, 12)
  assert.equal(report.outliers[0].label, 'camelid-measure-3')
  assert.equal(report.outliers[0].top_role_deltas[0].stage, 'prefill')
  assert.equal(report.outliers[0].top_role_deltas[0].role, 'ffn_down')
  assert.equal(report.runs[1].client_first_byte_ms, 25)
  assert.equal(report.runs[1].first_content_minus_first_byte_ms, 195)
  assert.equal(report.runs[1].backend_generate_minus_first_content_ms, 590)
  assert.equal(report.runs[1].prompt_eval_forward_ms, 120)
  assert.equal(report.runs[1].first_content_minus_prompt_eval_forward_ms, 90)
  assert.equal(report.runs[1].role_yield_ms, 2)
  assert.equal(report.runs[1].client_first_byte_minus_role_yield_ms, 23)
  assert.equal(report.runs[1].client_first_content_minus_content_yield_ms, 7)
  assert.equal(report.runs[1].backend_first_content_delta_vs_llama_cpp_ms, 50)
  assert.equal(report.runs[1].q8_total_gemm_us, 126000)
  assert.equal(report.runs[1].q8_fused_gate_up_calls, 28)
  assert.equal(report.runs[1].projection_route_calls, 8)
  assert.equal(report.runs[1].output_projection_calls, 8)
  assert.equal(report.runs[1].projection_routes['logits.q8_0_borrowed_packed_rows4'].elapsed_us, 18000)
  assert.equal(report.runs[1].output_projection_routes['logits.q8_0_borrowed_packed_rows4'].elapsed_us, 18000)
  assert.equal(report.runs[1].projection_layer_routes['generation.layer_5.ffn_down.mac_decode_consumer'].elapsed_us, 12000)
  assert.equal(report.runs[1].output_projection_layer_routes['generation.layer_5.ffn_down.mac_decode_consumer'].elapsed_us, 12000)
  assert.equal(report.runs[1].projection_route_denials['ffn_gate_up.decode_consumer.stream_diagnostics_collect_projection_details'].denials, 28)

  const missingPath = join(tempDir, 'missing.json')
  await writeFile(missingPath, '{}\n')
  let failed = false
  try {
    await execFileAsync(process.execPath, [summaryScript, missingPath], { cwd: resolve(scriptDir, '..') })
  } catch (err) {
    failed = true
    assert.match(err.stderr, /missing same-host camelid\.runs array/)
  }
  assert.equal(failed, true)

  console.log('summarize-same-host-stream-timing self-test passed')
} finally {
  await rm(tempDir, { recursive: true, force: true })
}

function fixture() {
  return {
    model: { row_id: 'llama32_3b_instruct_q8_0_mac', model_id: 'llama32-3b-q8' },
    method: { warmup: 2, repeats: 3, max_tokens: 8, unique_prompt: true, require_marker: true },
    camelid: {
      runs: [
        run('camelid-measure-1', 20, 190, 200, 780, 10),
        run('camelid-measure-2', 25, 210, 220, 800, 10),
        run('camelid-measure-3', 30, 230, 240, 820, 10),
      ],
    },
    llama_cpp: {
      runs: [
        { label: 'llama-measure-1', first_content_ms: 140 },
        { label: 'llama-measure-2', first_content_ms: 160 },
        { label: 'llama-measure-3', first_content_ms: 150 },
      ],
    },
  }
}

function run(label, firstByte, backendFirstContent, clientTtft, generate, residual) {
  const index = Number(label.match(/(\d+)$/)?.[1] ?? 1)
  return {
    label,
    first_byte_ms: firstByte,
    first_content_ms: backendFirstContent + residual,
    total_elapsed_ms: generate + 300,
    backend_first_content_ms: backendFirstContent,
    backend_generate_ms: generate,
    backend_q8_calls: 140,
    backend_timing: {
      q8_schedule: {
        i8mm_single_projection_calls: 140,
        i8mm_fused_gate_up_calls: 28,
        ffn_gate_up_decode_consumer_activation_us: 4200,
        ffn_gate_up_decode_consumer_tensor_us: 450,
        q8_gemm_compute_us: 126000,
        activation_quantize_pack_us: 12600,
        i8mm_single_projection_by_role: {
          attention_q: { calls: 28, pack_us: 1000, gemm_us: 5000, rows: 2000 },
          attention_k: { calls: 28, pack_us: 1100, gemm_us: 6000, rows: 2000 },
          attention_v: { calls: 28, pack_us: 1200, gemm_us: 7000, rows: 2000 },
          attention_output: { calls: 28, pack_us: 3000, gemm_us: 38000, rows: 2000 },
          ffn_down: { calls: 28, pack_us: 7000, gemm_us: 70000, rows: 2000 },
        },
        projection_route_calls: 8,
        projection_routes: {
          'ffn_down.x86_vnni_decode_consumer': {
            role: 'ffn_down',
            route: 'x86_vnni_decode_consumer',
            calls: 3,
            rows: 3,
            input_width: 3072,
            output_width: 8192,
            elapsed_us: 3000,
          },
          'logits.q8_0_borrowed_packed_rows4': {
            role: 'logits',
            route: 'q8_0_borrowed_packed_rows4',
            calls: 8,
            rows: 8,
            input_width: 3072,
            output_width: 128256,
            elapsed_us: 18000,
          },
        },
        output_projection_by_layer_route: {
          'generation.layer_5.ffn_down.mac_decode_consumer': {
            stage: 'generation',
            layer_index: 5,
            role: 'ffn_down',
            route: 'mac_decode_consumer',
            calls: 4,
            rows: 4,
            input_width: 8192,
            output_width: 3072,
            elapsed_us: 12000,
          },
          'generation.layer_4.ffn_down.mac_decode_consumer': {
            stage: 'generation',
            layer_index: 4,
            role: 'ffn_down',
            route: 'mac_decode_consumer',
            calls: 4,
            rows: 4,
            input_width: 8192,
            output_width: 3072,
            elapsed_us: 6000,
          },
        },
        projection_route_denials: {
          'ffn_down.x86_vnni_decode_consumer.gate_off': {
            role: 'ffn_down',
            route: 'x86_vnni_decode_consumer',
            reason: 'gate_off',
            denials: 2,
            rows: 2,
            input_width: 3072,
            output_width: 8192,
          },
          'ffn_gate_up.decode_consumer.stream_diagnostics_collect_projection_details': {
            role: 'ffn_gate_up',
            route: 'decode_consumer',
            reason: 'stream_diagnostics_collect_projection_details',
            denials: 28,
            rows: 28,
            input_width: 3072,
            output_width: 8192,
          },
        },
      },
      timings_ms: {
        prompt_cache_hit: false,
        weight_cache_hit: true,
        first_content: backendFirstContent,
        generate,
        first_content_accounting: {
          prompt_eval_forward_total: 100 + (index * 10),
          prompt_eval_logits: 10 + index,
          prompt_eval_sample: 1.5,
          prompt_eval_forward_plus_sample: 101.5 + (index * 10),
          first_content_minus_prompt_eval_forward: backendFirstContent - (100 + (index * 10)),
          first_content_minus_prompt_eval_forward_plus_sample: backendFirstContent - (101.5 + (index * 10)),
        },
        stream_event_accounting: {
          role_yield: 2,
          generate_start: 4,
          first_content_yield: backendFirstContent + 3,
          final_yield: generate + 5,
          generate_start_minus_role_yield: 2,
          first_content_yield_minus_role_yield: backendFirstContent + 1,
          final_yield_minus_first_content_yield: generate + 2 - backendFirstContent,
        },
        prefill_role_timings: {
          attention_context: 6,
          attention_output: 14,
          ffn_gate: 30,
          ffn_up: 31,
          ffn_down: 36 + (index * 2),
          logits: 0,
        },
        first_token_role_timings: {
          attention_context: 2,
          attention_output: 3,
          ffn_gate: 4,
          ffn_up: 4,
          ffn_down: 8,
          logits: 5,
        },
        generation_role_timings: {
          attention_context: 12,
          attention_output: 24,
          ffn_gate: 60,
          ffn_up: 61,
          ffn_down: 92,
          logits: 15,
        },
        layer_role_hotspots: {
          prefill: [
            { layer_index: 2, role: 'ffn_down', elapsed_ms: 5 },
            { layer_index: 2, role: 'ffn_up', elapsed_ms: 4 },
          ],
          first_token: [
            { layer_index: 3, role: 'attention_context', elapsed_ms: 0.5 },
          ],
          generation: [
            { layer_index: 5, role: 'attention_output', elapsed_ms: 13 },
            { layer_index: 4, role: 'ffn_down', elapsed_ms: 7 },
          ],
        },
      },
    },
  }
}
