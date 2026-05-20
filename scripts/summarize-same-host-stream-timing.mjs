#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const inputPath = args.get('input') || args.get('in') || args.positionals[0]
const outPath = args.get('out')

if (args.has('help') || args.has('h') || !inputPath) {
  console.log(usage())
  process.exit(inputPath ? 0 : 1)
}

const input = JSON.parse(await readFile(inputPath, 'utf8'))
const report = summarizeSameHostStreamTiming(input, inputPath)

const text = humanSummary(report)
console.log(text)
if (outPath) {
  await writeFile(resolve(outPath), `${JSON.stringify(report, null, 2)}\n`)
}

export function summarizeSameHostStreamTiming(input, inputPath = 'same-host.json') {
  const camelidRuns = input?.camelid?.runs
  if (!Array.isArray(camelidRuns) || camelidRuns.length === 0) {
    throw new Error('missing same-host camelid.runs array')
  }
  const llamaRuns = Array.isArray(input?.llama_cpp?.runs) ? input.llama_cpp.runs : []
  const analyzedRuns = camelidRuns.map((run, index) => analyzeCamelidRun(run, index))
  const llamaTtfts = llamaRuns.map((run) => finite(run.first_content_ms ?? run.first_byte_ms)).filter((value) => value !== null)

  const roleKeys = [
    'attention_context',
    'attention_output',
    'ffn_gate',
    'ffn_up',
    'ffn_down',
    'logits',
  ]
  const stageKeys = ['prefill', 'first_token', 'generation']
  const stages = Object.fromEntries(stageKeys.map((stage) => [
    stage,
    Object.fromEntries(roleKeys.map((role) => [role, stats(analyzedRuns.map((run) => run.roles?.[stage]?.[role]))])),
  ]))

  const firstByteStats = stats(analyzedRuns.map((run) => run.client_first_byte_ms))
  const firstContentStats = stats(analyzedRuns.map((run) => run.backend_first_content_ms))
  const ttftStats = stats(analyzedRuns.map((run) => run.client_ttft_ms))
  const generateStats = stats(analyzedRuns.map((run) => run.backend_generate_ms))
  const totalStats = stats(analyzedRuns.map((run) => run.client_total_ms))
  const llamaTtftStats = stats(llamaTtfts)
  const residuals = analyzedRuns.map((run) => run.client_minus_backend_first_content_ms).filter((value) => value !== null)
  const firstContentMinusFirstByte = analyzedRuns.map((run) => run.client_first_content_minus_first_byte_ms).filter((value) => value !== null)
  const backendGenerateMinusFirstByte = analyzedRuns.map((run) => run.backend_generate_minus_first_byte_ms).filter((value) => value !== null)
  const backendGenerateMinusFirstContent = analyzedRuns.map((run) => run.backend_generate_minus_first_content_ms).filter((value) => value !== null)
  const firstContentMinusPromptEvalForward = analyzedRuns.map((run) => run.first_content_minus_prompt_eval_forward_ms).filter((value) => value !== null)
  const firstContentMinusPromptEvalForwardSample = analyzedRuns.map((run) => run.first_content_minus_prompt_eval_forward_plus_sample_ms).filter((value) => value !== null)
  const clientFirstByteMinusRoleYield = analyzedRuns.map((run) => run.client_first_byte_minus_role_yield_ms).filter((value) => value !== null)
  const clientFirstContentMinusContentYield = analyzedRuns.map((run) => run.client_first_content_minus_content_yield_ms).filter((value) => value !== null)
  const clientDoneMinusFinalYield = analyzedRuns.map((run) => run.client_done_minus_final_yield_ms).filter((value) => value !== null)

  const rankedRoles = rankRoles(stages)
  const outliers = rankOutliers(analyzedRuns, firstContentStats, stages)
  const q8RoleWork = summarizeQ8RoleWork(analyzedRuns)
  const gateUpDecodeOverhead = summarizeGateUpDecodeConsumerOverhead(analyzedRuns)
  const projectionRoutes = summarizeProjectionRoutes(analyzedRuns)
  const projectionLayerRoutes = summarizeProjectionLayerRoutes(analyzedRuns)
  const projectionRouteDenials = summarizeProjectionRouteDenials(analyzedRuns)
  const layerRoleHotspots = summarizeLayerRoleHotspots(analyzedRuns)
  const stageRouteRoleGaps = summarizeStageRouteRoleGaps(analyzedRuns)
  const layerRouteRoleGaps = summarizeLayerRouteRoleGaps(analyzedRuns)
  const roleFocus = summarizeRoleFocus(stages, ['logits', 'attention_context', 'attention_output'])
  const runDeltas = analyzedRuns.map((run, index) => {
    const llamaTtft = llamaTtfts[index] ?? null
    return {
      label: run.label,
      client_first_byte_ms: run.client_first_byte_ms,
      client_ttft_ms: run.client_ttft_ms,
      backend_first_content_ms: run.backend_first_content_ms,
      backend_generate_ms: run.backend_generate_ms,
      first_content_minus_first_byte_ms: run.client_first_content_minus_first_byte_ms,
      backend_generate_minus_first_byte_ms: run.backend_generate_minus_first_byte_ms,
      backend_generate_minus_first_content_ms: run.backend_generate_minus_first_content_ms,
      client_minus_backend_first_content_ms: run.client_minus_backend_first_content_ms,
      first_content_minus_prompt_eval_forward_ms: run.first_content_minus_prompt_eval_forward_ms,
      first_content_minus_prompt_eval_forward_plus_sample_ms: run.first_content_minus_prompt_eval_forward_plus_sample_ms,
      prompt_eval_forward_ms: run.prompt_eval_forward_ms,
      prompt_eval_logits_ms: run.prompt_eval_logits_ms,
      prompt_eval_sample_ms: run.prompt_eval_sample_ms,
      role_yield_ms: run.role_yield_ms,
      generate_start_yield_ms: run.generate_start_yield_ms,
      first_content_yield_ms: run.first_content_yield_ms,
      final_yield_ms: run.final_yield_ms,
      client_first_byte_minus_role_yield_ms: run.client_first_byte_minus_role_yield_ms,
      client_first_content_minus_content_yield_ms: run.client_first_content_minus_content_yield_ms,
      client_done_minus_final_yield_ms: run.client_done_minus_final_yield_ms,
      llama_cpp_ttft_ms: llamaTtft,
      backend_first_content_delta_vs_llama_cpp_ms: delta(run.backend_first_content_ms, llamaTtft),
      q8_calls: run.q8_calls,
      q8_total_gemm_us: run.q8_total_gemm_us,
      q8_total_pack_us: run.q8_total_pack_us,
      q8_fused_gate_up_calls: run.q8_fused_gate_up_calls,
      q8_gate_up_decode_consumer_activation_us: run.q8_gate_up_decode_consumer_activation_us,
      q8_gate_up_decode_consumer_tensor_us: run.q8_gate_up_decode_consumer_tensor_us,
      projection_route_calls: run.projection_route_calls,
      projection_routes: run.projection_routes,
      projection_layer_routes: run.projection_layer_routes,
      output_projection_calls: run.projection_route_calls,
      output_projection_routes: run.projection_routes,
      output_projection_layer_routes: run.projection_layer_routes,
      projection_route_denials: run.projection_route_denials,
      prompt_cache_hit: run.prompt_cache_hit,
      weight_cache_hit: run.weight_cache_hit,
    }
  })

  return {
    schema: 'camelid.same_host_stream_timing_summary.v1',
    input: basename(inputPath),
    generated_utc: new Date().toISOString(),
    method: {
      row_id: input?.model?.row_id ?? null,
      model_id: input?.model?.model_id ?? null,
      warmup: input?.method?.warmup ?? null,
      repeats: input?.method?.repeats ?? camelidRuns.length,
      max_tokens: input?.method?.max_tokens ?? null,
      unique_prompt: input?.method?.unique_prompt ?? null,
      require_marker: input?.method?.require_marker ?? null,
    },
    aggregate: {
      camelid_client_first_byte_ms: firstByteStats,
      camelid_client_ttft_ms: ttftStats,
      camelid_backend_first_content_ms: firstContentStats,
      camelid_backend_generate_ms: generateStats,
      camelid_client_total_ms: totalStats,
      llama_cpp_ttft_ms: llamaTtftStats,
      first_content_minus_first_byte_ms: stats(firstContentMinusFirstByte),
      backend_generate_minus_first_byte_ms: stats(backendGenerateMinusFirstByte),
      backend_generate_minus_first_content_ms: stats(backendGenerateMinusFirstContent),
      first_content_minus_prompt_eval_forward_ms: stats(firstContentMinusPromptEvalForward),
      first_content_minus_prompt_eval_forward_plus_sample_ms: stats(firstContentMinusPromptEvalForwardSample),
      prompt_eval_forward_ms: stats(analyzedRuns.map((run) => run.prompt_eval_forward_ms)),
      prompt_eval_logits_ms: stats(analyzedRuns.map((run) => run.prompt_eval_logits_ms)),
      prompt_eval_sample_ms: stats(analyzedRuns.map((run) => run.prompt_eval_sample_ms)),
      role_yield_ms: stats(analyzedRuns.map((run) => run.role_yield_ms)),
      generate_start_yield_ms: stats(analyzedRuns.map((run) => run.generate_start_yield_ms)),
      first_content_yield_ms: stats(analyzedRuns.map((run) => run.first_content_yield_ms)),
      final_yield_ms: stats(analyzedRuns.map((run) => run.final_yield_ms)),
      client_first_byte_minus_role_yield_ms: stats(clientFirstByteMinusRoleYield),
      client_first_content_minus_content_yield_ms: stats(clientFirstContentMinusContentYield),
      client_done_minus_final_yield_ms: stats(clientDoneMinusFinalYield),
      client_minus_backend_first_content_ms: stats(residuals),
      backend_first_content_delta_vs_llama_cpp_ttft_ms: stats(runDeltas.map((run) => run.backend_first_content_delta_vs_llama_cpp_ms)),
      q8_calls: stats(analyzedRuns.map((run) => run.q8_calls)),
      q8_total_gemm_us: stats(analyzedRuns.map((run) => run.q8_total_gemm_us)),
      q8_total_pack_us: stats(analyzedRuns.map((run) => run.q8_total_pack_us)),
      q8_fused_gate_up_calls: stats(analyzedRuns.map((run) => run.q8_fused_gate_up_calls)),
      q8_gate_up_decode_consumer_activation_us: stats(analyzedRuns.map((run) => run.q8_gate_up_decode_consumer_activation_us)),
      q8_gate_up_decode_consumer_tensor_us: stats(analyzedRuns.map((run) => run.q8_gate_up_decode_consumer_tensor_us)),
      projection_route_calls: stats(analyzedRuns.map((run) => run.projection_route_calls)),
      output_projection_calls: stats(analyzedRuns.map((run) => run.projection_route_calls)),
      prompt_cache_hits: analyzedRuns.filter((run) => run.prompt_cache_hit === true).length,
      weight_cache_hits: analyzedRuns.filter((run) => run.weight_cache_hit === true).length,
    },
    stages,
    ranked_roles: rankedRoles,
    q8_role_work: q8RoleWork,
    q8_gate_up_decode_consumer_overhead: gateUpDecodeOverhead,
    projection_routes: projectionRoutes,
    projection_layer_routes: projectionLayerRoutes,
    output_projection_routes: projectionRoutes,
    output_projection_layer_routes: projectionLayerRoutes,
    projection_route_denials: projectionRouteDenials,
    layer_role_hotspots: layerRoleHotspots,
    stage_route_role_gaps: stageRouteRoleGaps,
    layer_route_role_gaps: layerRouteRoleGaps,
    role_focus: roleFocus,
    outliers,
    runs: runDeltas,
  }
}

function analyzeCamelidRun(run, index) {
  const timings = run?.backend_timing?.timings_ms ?? {}
  const q8Schedule = run?.backend_timing?.q8_schedule ?? {}
  const firstContentAccounting = timings.first_content_accounting ?? {}
  const streamEventAccounting = timings.stream_event_accounting ?? {}
  const roleYield = finite(streamEventAccounting.role_yield)
  const firstContentYield = finite(streamEventAccounting.first_content_yield)
  const finalYield = finite(streamEventAccounting.final_yield)
  const roles = {
    prefill: timings.prefill_role_timings ?? {},
    first_token: timings.first_token_role_timings ?? {},
    generation: timings.generation_role_timings ?? {},
  }
  const clientFirstByte = finite(run.first_byte_ms)
  const clientTtft = finite(run.first_content_ms ?? run.ttft_ms)
  const backendFirstContent = finite(run.backend_first_content_ms ?? timings.first_content)
  const backendGenerate = finite(run.backend_generate_ms ?? timings.generate)
  return {
    label: run.label ?? `camelid-run-${index + 1}`,
    client_first_byte_ms: clientFirstByte,
    client_ttft_ms: clientTtft,
    client_total_ms: finite(run.total_elapsed_ms),
    backend_generate_ms: backendGenerate,
    backend_first_content_ms: backendFirstContent,
    client_first_content_minus_first_byte_ms: delta(clientTtft, clientFirstByte),
    backend_generate_minus_first_byte_ms: delta(backendGenerate, clientFirstByte),
    backend_generate_minus_first_content_ms: delta(backendGenerate, backendFirstContent),
    client_minus_backend_first_content_ms: delta(clientTtft, backendFirstContent),
    first_content_minus_prompt_eval_forward_ms: finite(firstContentAccounting.first_content_minus_prompt_eval_forward),
    first_content_minus_prompt_eval_forward_plus_sample_ms: finite(firstContentAccounting.first_content_minus_prompt_eval_forward_plus_sample),
    prompt_eval_forward_ms: finite(firstContentAccounting.prompt_eval_forward_total),
    prompt_eval_logits_ms: finite(firstContentAccounting.prompt_eval_logits),
    prompt_eval_sample_ms: finite(firstContentAccounting.prompt_eval_sample),
    role_yield_ms: roleYield,
    generate_start_yield_ms: finite(streamEventAccounting.generate_start),
    first_content_yield_ms: firstContentYield,
    final_yield_ms: finalYield,
    client_first_byte_minus_role_yield_ms: delta(clientFirstByte, roleYield),
    client_first_content_minus_content_yield_ms: delta(clientTtft, firstContentYield),
    client_done_minus_final_yield_ms: delta(finite(run.total_elapsed_ms), finalYield),
    q8_calls: q8ScheduleCallCount(run.backend_q8_calls, q8Schedule),
    q8_total_gemm_us: finite(run.backend_q8_gemm_compute_us ?? q8Schedule.q8_gemm_compute_us),
    q8_total_pack_us: finite(run.backend_q8_pack_us ?? q8Schedule.activation_quantize_pack_us),
    q8_fused_gate_up_calls: finite(q8Schedule.i8mm_fused_gate_up_calls),
    q8_gate_up_decode_consumer_activation_us: finite(q8Schedule.ffn_gate_up_decode_consumer_activation_us),
    q8_gate_up_decode_consumer_tensor_us: finite(q8Schedule.ffn_gate_up_decode_consumer_tensor_us),
    q8_roles: q8Schedule.i8mm_single_projection_by_role ?? {},
    projection_route_calls: finite(q8Schedule.projection_route_calls ?? q8Schedule.output_projection_calls),
    projection_routes: q8Schedule.projection_routes ?? q8Schedule.output_projection_by_route ?? {},
    projection_layer_routes: q8Schedule.projection_layer_routes ?? q8Schedule.output_projection_by_layer_route ?? {},
    projection_route_denials: q8Schedule.projection_route_denials ?? {},
    layer_hotspots: timings.layer_role_hotspots ?? {},
    prompt_cache_hit: timings.prompt_cache_hit ?? null,
    weight_cache_hit: timings.weight_cache_hit ?? null,
    roles,
  }
}

function rankRoles(stages) {
  const rows = []
  for (const [stage, roles] of Object.entries(stages)) {
    for (const [role, stat] of Object.entries(roles)) {
      if (stat.count > 0) {
        rows.push({ stage, role, mean_ms: stat.mean, p95_ms: stat.p95, max_ms: stat.max })
      }
    }
  }
  rows.sort((left, right) => (right.mean_ms ?? 0) - (left.mean_ms ?? 0))
  return rows
}

function rankOutliers(runs, baselineStats, stages) {
  const mean = baselineStats.mean ?? 0
  return runs
    .map((run) => ({
      label: run.label,
      backend_first_content_ms: run.backend_first_content_ms,
      client_ttft_ms: run.client_ttft_ms,
      over_mean_ms: delta(run.backend_first_content_ms, mean),
      q8_calls: run.q8_calls,
      q8_total_gemm_us: run.q8_total_gemm_us,
      q8_total_pack_us: run.q8_total_pack_us,
      top_role_deltas: topRoleDeltas(run, stages),
      prompt_cache_hit: run.prompt_cache_hit,
    }))
    .filter((run) => run.over_mean_ms !== null)
    .sort((left, right) => right.over_mean_ms - left.over_mean_ms)
    .slice(0, 5)
}

function topRoleDeltas(run, stages) {
  const rows = []
  for (const [stage, roles] of Object.entries(run.roles ?? {})) {
    for (const [role, value] of Object.entries(roles ?? {})) {
      const valueMs = finite(value)
      const meanMs = finite(stages?.[stage]?.[role]?.mean)
      if (valueMs !== null && meanMs !== null) {
        rows.push({
          stage,
          role,
          value_ms: valueMs,
          over_role_mean_ms: delta(valueMs, meanMs),
        })
      }
    }
  }
  return rows
    .filter((row) => row.over_role_mean_ms !== null)
    .sort((left, right) => right.over_role_mean_ms - left.over_role_mean_ms)
    .slice(0, 5)
}

function summarizeQ8RoleWork(runs) {
  const roleNames = new Set()
  for (const run of runs) {
    for (const role of Object.keys(run.q8_roles ?? {})) roleNames.add(role)
  }
  return [...roleNames].sort().map((role) => {
    const samples = runs.map((run) => run.q8_roles?.[role] ?? {})
    const calls = stats(samples.map((sample) => sample.calls))
    const packUs = stats(samples.map((sample) => sample.pack_us))
    const gemmUs = stats(samples.map((sample) => sample.gemm_us))
    const rows = stats(samples.map((sample) => sample.rows))
    return {
      role,
      calls,
      pack_us: packUs,
      gemm_us: gemmUs,
      rows,
      gemm_ms_mean: gemmUs.mean === null ? null : round(gemmUs.mean / 1000),
      pack_ms_mean: packUs.mean === null ? null : round(packUs.mean / 1000),
    }
  }).sort((left, right) => (right.gemm_us.mean ?? 0) - (left.gemm_us.mean ?? 0))
}

function summarizeGateUpDecodeConsumerOverhead(runs) {
  const activationUs = stats(runs.map((run) => run.q8_gate_up_decode_consumer_activation_us))
  const tensorUs = stats(runs.map((run) => run.q8_gate_up_decode_consumer_tensor_us))
  const totalUs = stats(runs.map((run) => {
    const activation = run.q8_gate_up_decode_consumer_activation_us
    const tensor = run.q8_gate_up_decode_consumer_tensor_us
    return activation === null || tensor === null ? null : activation + tensor
  }))
  return {
    activation_us: activationUs,
    tensor_us: tensorUs,
    total_us: totalUs,
    activation_ms_mean: activationUs.mean === null ? null : round(activationUs.mean / 1000),
    tensor_ms_mean: tensorUs.mean === null ? null : round(tensorUs.mean / 1000),
    total_ms_mean: totalUs.mean === null ? null : round(totalUs.mean / 1000),
  }
}

function summarizeProjectionRoutes(runs) {
  const routeNames = new Set()
  for (const run of runs) {
    for (const routeName of Object.keys(run.projection_routes ?? {})) {
      routeNames.add(routeName)
    }
  }
  return [...routeNames].sort().map((routeName) => {
    const samples = runs.map((run) => run.projection_routes?.[routeName] ?? {})
    const first = samples.find((sample) => typeof sample.route === 'string') ?? {}
    const calls = stats(samples.map((sample) => sample.calls))
    const elapsedUs = stats(samples.map((sample) => sample.elapsed_us))
    const rows = stats(samples.map((sample) => sample.rows))
    const inputWidth = stats(samples.map((sample) => sample.input_width))
    const outputWidth = stats(samples.map((sample) => sample.output_width))
    return {
      key: routeName,
      role: first.role ?? null,
      route: first.route ?? routeName,
      calls,
      rows,
      input_width: inputWidth,
      output_width: outputWidth,
      elapsed_us: elapsedUs,
      elapsed_ms_mean: elapsedUs.mean === null ? null : round(elapsedUs.mean / 1000),
    }
  }).sort((left, right) => (right.elapsed_us.mean ?? 0) - (left.elapsed_us.mean ?? 0))
}

function summarizeProjectionLayerRoutes(runs) {
  const routeNames = new Set()
  for (const run of runs) {
    for (const routeName of Object.keys(run.projection_layer_routes ?? {})) {
      routeNames.add(routeName)
    }
  }
  return [...routeNames].sort().map((routeName) => {
    const samples = runs.map((run) => run.projection_layer_routes?.[routeName] ?? {})
    const first = samples.find((sample) => typeof sample.route === 'string') ?? {}
    const calls = stats(samples.map((sample) => sample.calls))
    const elapsedUs = stats(samples.map((sample) => sample.elapsed_us))
    return {
      key: routeName,
      stage: first.stage ?? null,
      layer_index: finite(first.layer_index),
      role: first.role ?? null,
      route: first.route ?? null,
      calls,
      rows: stats(samples.map((sample) => sample.rows)),
      input_width: stats(samples.map((sample) => sample.input_width)),
      output_width: stats(samples.map((sample) => sample.output_width)),
      elapsed_us: elapsedUs,
      elapsed_ms_mean: elapsedUs.mean === null ? null : round(elapsedUs.mean / 1000),
    }
  }).sort((left, right) => {
    const elapsedDelta = (right.elapsed_us.mean ?? 0) - (left.elapsed_us.mean ?? 0)
    if (elapsedDelta !== 0) return elapsedDelta
    return (left.layer_index ?? 0) - (right.layer_index ?? 0) || left.key.localeCompare(right.key)
  })
}

function summarizeProjectionRouteDenials(runs) {
  const keys = new Set()
  for (const run of runs) {
    for (const key of Object.keys(run.projection_route_denials ?? {})) {
      keys.add(key)
    }
  }
  return [...keys].sort().map((key) => {
    const samples = runs.map((run) => run.projection_route_denials?.[key] ?? {})
    const first = samples.find((sample) => typeof sample.route === 'string') ?? {}
    return {
      key,
      role: first.role ?? null,
      route: first.route ?? null,
      reason: first.reason ?? null,
      denials: stats(samples.map((sample) => sample.denials)),
      rows: stats(samples.map((sample) => sample.rows)),
      input_width: stats(samples.map((sample) => sample.input_width)),
      output_width: stats(samples.map((sample) => sample.output_width)),
    }
  }).sort((left, right) => (right.denials.mean ?? 0) - (left.denials.mean ?? 0))
}

function summarizeRoleFocus(stages, roleNames) {
  return roleNames.map((role) => {
    const stageRows = Object.entries(stages).map(([stage, roles]) => ({
      stage,
      elapsed_ms: roles?.[role] ?? stats([]),
    }))
    const totalMean = stageRows.reduce((acc, row) => acc + (row.elapsed_ms.mean ?? 0), 0)
    return {
      role,
      total_mean_ms: round(totalMean),
      stages: Object.fromEntries(stageRows.map((row) => [row.stage, row.elapsed_ms])),
    }
  }).sort((left, right) => (right.total_mean_ms ?? 0) - (left.total_mean_ms ?? 0))
}

function summarizeLayerRoleHotspots(runs) {
  const groups = new Map()
  for (const run of runs) {
    for (const hotspot of layerHotspotEntries(run)) {
      const key = `${hotspot.stage}\u0000${hotspot.layer_index}\u0000${hotspot.role}`
      if (!groups.has(key)) {
        groups.set(key, {
          stage: hotspot.stage,
          layer_index: hotspot.layer_index,
          role: hotspot.role,
          values: [],
        })
      }
      groups.get(key).values.push(hotspot.elapsed_ms)
    }
  }
  return [...groups.values()].map((group) => ({
    stage: group.stage,
    layer_index: group.layer_index,
    role: group.role,
    elapsed_ms: stats(group.values),
  })).sort((left, right) => {
    const meanDelta = (right.elapsed_ms.mean ?? 0) - (left.elapsed_ms.mean ?? 0)
    if (meanDelta !== 0) return meanDelta
    return left.layer_index - right.layer_index || left.role.localeCompare(right.role)
  })
}

function summarizeLayerRouteRoleGaps(runs) {
  const groups = new Map()
  for (const run of runs) {
    const hotspots = layerHotspotEntries(run)
    for (const [routeKey, route] of Object.entries(run.projection_layer_routes ?? {})) {
      const layerIndex = finite(route?.layer_index)
      const routeElapsedUs = finite(route?.elapsed_us)
      const routeRole = typeof route?.role === 'string' ? route.role : null
      const routeStage = typeof route?.stage === 'string' ? route.stage : null
      if (layerIndex === null || routeElapsedUs === null || !routeRole) continue
      const matchedRoles = hotspotRolesForProjectionRole(routeRole)
      const routeElapsedMs = routeElapsedUs / 1000
      for (const stage of ['prefill', 'first_token', 'generation']) {
        if (routeStage && routeStage !== 'unknown' && routeStage !== stage) continue
        const matched = hotspots.filter((entry) => (
          entry.stage === stage
          && entry.layer_index === layerIndex
          && matchedRoles.includes(entry.role)
        ))
        if (matched.length === 0) continue
        const roleElapsedMs = matched.reduce((sum, entry) => sum + entry.elapsed_ms, 0)
        const gapMs = round(roleElapsedMs - routeElapsedMs)
        const key = `${stage}\u0000${layerIndex}\u0000${routeRole}\u0000${routeKey}`
        if (!groups.has(key)) {
          groups.set(key, {
            stage,
            layer_index: layerIndex,
            projection_role: routeRole,
            route_stage: routeStage,
            route: route.route ?? null,
            route_key: routeKey,
            matched_roles: new Set(),
            role_elapsed_values: [],
            route_elapsed_values: [],
            gap_values: [],
            abs_gap_values: [],
          })
        }
        const group = groups.get(key)
        for (const entry of matched) group.matched_roles.add(entry.role)
        group.role_elapsed_values.push(roleElapsedMs)
        group.route_elapsed_values.push(routeElapsedMs)
        group.gap_values.push(gapMs)
        group.abs_gap_values.push(Math.abs(gapMs))
      }
    }
  }
  return [...groups.values()].map((group) => ({
    stage: group.stage,
    layer_index: group.layer_index,
    projection_role: group.projection_role,
    route_stage: group.route_stage,
    route: group.route,
    route_key: group.route_key,
    matched_roles: [...group.matched_roles].sort(),
    role_elapsed_ms: stats(group.role_elapsed_values),
    route_elapsed_ms: stats(group.route_elapsed_values),
    role_minus_route_ms: stats(group.gap_values),
    abs_gap_ms: stats(group.abs_gap_values),
  })).sort((left, right) => {
    const absDelta = (right.abs_gap_ms.mean ?? 0) - (left.abs_gap_ms.mean ?? 0)
    if (absDelta !== 0) return absDelta
    return left.layer_index - right.layer_index || left.route_key.localeCompare(right.route_key)
  })
}

function summarizeStageRouteRoleGaps(runs) {
  const groups = new Map()
  for (const run of runs) {
    const routeGroups = new Map()
    for (const route of Object.values(run.projection_layer_routes ?? {})) {
      const stage = typeof route?.stage === 'string' ? route.stage : null
      const routeRole = typeof route?.role === 'string' ? route.role : null
      const routeName = typeof route?.route === 'string' ? route.route : null
      const routeElapsedUs = finite(route?.elapsed_us)
      const calls = finite(route?.calls)
      const rows = finite(route?.rows)
      if (!stage || stage === 'unknown' || !routeRole || !routeName || routeElapsedUs === null) {
        continue
      }
      const key = `${stage}\u0000${routeRole}\u0000${routeName}`
      if (!routeGroups.has(key)) {
        routeGroups.set(key, {
          stage,
          projection_role: routeRole,
          route: routeName,
          elapsed_us: 0,
          calls: 0,
          rows: 0,
        })
      }
      const group = routeGroups.get(key)
      group.elapsed_us += routeElapsedUs
      if (calls !== null) group.calls += calls
      if (rows !== null) group.rows += rows
    }

    for (const routeGroup of routeGroups.values()) {
      const matchedRoles = hotspotRolesForProjectionRole(routeGroup.projection_role)
      let roleElapsedMs = 0
      const presentRoles = []
      for (const role of matchedRoles) {
        const elapsedMs = finite(run.roles?.[routeGroup.stage]?.[role])
        if (elapsedMs !== null) {
          roleElapsedMs += elapsedMs
          presentRoles.push(role)
        }
      }
      if (presentRoles.length === 0) continue

      const routeElapsedMs = routeGroup.elapsed_us / 1000
      const gapMs = round(roleElapsedMs - routeElapsedMs)
      const key = `${routeGroup.stage}\u0000${routeGroup.projection_role}\u0000${routeGroup.route}`
      if (!groups.has(key)) {
        groups.set(key, {
          stage: routeGroup.stage,
          projection_role: routeGroup.projection_role,
          route: routeGroup.route,
          matched_roles: new Set(),
          role_elapsed_values: [],
          route_elapsed_values: [],
          gap_values: [],
          abs_gap_values: [],
          calls_values: [],
          rows_values: [],
        })
      }
      const group = groups.get(key)
      for (const role of presentRoles) group.matched_roles.add(role)
      group.role_elapsed_values.push(roleElapsedMs)
      group.route_elapsed_values.push(routeElapsedMs)
      group.gap_values.push(gapMs)
      group.abs_gap_values.push(Math.abs(gapMs))
      group.calls_values.push(routeGroup.calls)
      group.rows_values.push(routeGroup.rows)
    }
  }

  return [...groups.values()].map((group) => ({
    stage: group.stage,
    projection_role: group.projection_role,
    route: group.route,
    matched_roles: [...group.matched_roles].sort(),
    role_elapsed_ms: stats(group.role_elapsed_values),
    route_elapsed_ms: stats(group.route_elapsed_values),
    role_minus_route_ms: stats(group.gap_values),
    abs_gap_ms: stats(group.abs_gap_values),
    calls: stats(group.calls_values),
    rows: stats(group.rows_values),
  })).sort((left, right) => {
    const absDelta = (right.abs_gap_ms.mean ?? 0) - (left.abs_gap_ms.mean ?? 0)
    if (absDelta !== 0) return absDelta
    return left.stage.localeCompare(right.stage)
      || left.projection_role.localeCompare(right.projection_role)
      || left.route.localeCompare(right.route)
  })
}

function layerHotspotEntries(run) {
  const rows = []
  for (const [stage, entries] of Object.entries(run.layer_hotspots ?? {})) {
    if (!Array.isArray(entries)) continue
    for (const entry of entries) {
      const layerIndex = finite(entry?.layer_index)
      const elapsedMs = finite(entry?.elapsed_ms)
      const role = typeof entry?.role === 'string' ? entry.role : null
      if (layerIndex === null || elapsedMs === null || !role) continue
      rows.push({ stage, layer_index: layerIndex, role, elapsed_ms: elapsedMs })
    }
  }
  return rows
}

function hotspotRolesForProjectionRole(role) {
  if (role === 'ffn_gate_up') return ['ffn_gate', 'ffn_up']
  return [role]
}

function stats(values) {
  const xs = values.map(finite).filter((value) => value !== null).sort((a, b) => a - b)
  if (xs.length === 0) {
    return { count: 0, min: null, p50: null, mean: null, p95: null, max: null }
  }
  const sum = xs.reduce((acc, value) => acc + value, 0)
  return {
    count: xs.length,
    min: round(xs[0]),
    p50: round(percentile(xs, 0.5)),
    mean: round(sum / xs.length),
    p95: round(percentile(xs, 0.95)),
    max: round(xs[xs.length - 1]),
  }
}

function percentile(sorted, p) {
  if (sorted.length === 1) return sorted[0]
  const index = (sorted.length - 1) * p
  const lower = Math.floor(index)
  const upper = Math.ceil(index)
  if (lower === upper) return sorted[lower]
  const weight = index - lower
  return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

function humanSummary(report) {
  const top = report.ranked_roles.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.stage}.${row.role} mean=${fmt(row.mean_ms)}ms p95=${fmt(row.p95_ms)}ms`)
    .join('\n')
  const outliers = report.outliers.slice(0, 3)
    .map((run) => {
      const deltas = run.top_role_deltas?.slice(0, 3)
        .map((role) => `${role.stage}.${role.role}+${fmt(role.over_role_mean_ms)}ms`)
        .join(', ')
      const suffix = deltas ? ` role_deltas=[${deltas}]` : ''
      return `${run.label}: backend_first_content=${fmt(run.backend_first_content_ms)}ms over_mean=${fmt(run.over_mean_ms)}ms q8_calls=${run.q8_calls}${suffix}`
    })
    .join('\n')
  const gateUpDecodeOverhead = report.q8_gate_up_decode_consumer_overhead ?? {}
  const q8Roles = report.q8_role_work.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.role} gemm_mean=${fmt(row.gemm_ms_mean)}ms pack_mean=${fmt(row.pack_ms_mean)}ms calls_mean=${fmt(row.calls.mean)}`)
    .join('\n')
  const projectionRoutes = report.projection_routes.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.key} elapsed_mean=${fmt(row.elapsed_ms_mean)}ms calls_mean=${fmt(row.calls.mean)} rows_mean=${fmt(row.rows.mean)} width=${fmt(row.input_width.mean)}x${fmt(row.output_width.mean)}`)
    .join('\n')
  const projectionLayerRoutes = report.projection_layer_routes.slice(0, 8)
    .map((row, index) => `${index + 1}. ${row.key} elapsed_mean=${fmt(row.elapsed_ms_mean)}ms calls_mean=${fmt(row.calls.mean)} width=${fmt(row.input_width.mean)}x${fmt(row.output_width.mean)}`)
    .join('\n')
  const projectionDenials = report.projection_route_denials.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.key} denials_mean=${fmt(row.denials.mean)} rows_mean=${fmt(row.rows.mean)} reason=${row.reason ?? 'n/a'}`)
    .join('\n')
  const layerHotspots = report.layer_role_hotspots.slice(0, 6)
    .map((row, index) => `${index + 1}. ${row.stage}.L${row.layer_index}.${row.role} mean=${fmt(row.elapsed_ms.mean)}ms p95=${fmt(row.elapsed_ms.p95)}ms`)
    .join('\n')
  const stageRouteGaps = report.stage_route_role_gaps.slice(0, 8)
    .map((row, index) => `${index + 1}. ${row.stage}.${row.projection_role}.${row.route ?? 'unknown'} role_mean=${fmt(row.role_elapsed_ms.mean)}ms route_mean=${fmt(row.route_elapsed_ms.mean)}ms role_minus_route_mean=${fmt(row.role_minus_route_ms.mean)}ms calls_mean=${fmt(row.calls.mean)} matched=${row.matched_roles.join('+') || 'none'}`)
    .join('\n')
  const roleFocus = report.role_focus
    .map((row, index) => `${index + 1}. ${row.role} total_mean=${fmt(row.total_mean_ms)}ms prefill=${fmt(row.stages.prefill.mean)}ms first_token=${fmt(row.stages.first_token.mean)}ms generation=${fmt(row.stages.generation.mean)}ms`)
    .join('\n')
  const layerRouteGaps = report.layer_route_role_gaps.slice(0, 8)
    .map((row, index) => `${index + 1}. ${row.stage}.L${row.layer_index}.${row.projection_role}.${row.route ?? 'unknown'} role_mean=${fmt(row.role_elapsed_ms.mean)}ms route_mean=${fmt(row.route_elapsed_ms.mean)}ms role_minus_route_mean=${fmt(row.role_minus_route_ms.mean)}ms matched=${row.matched_roles.join('+') || 'none'}`)
    .join('\n')
  return [
    `schema=${report.schema}`,
    `input=${report.input}`,
    `runs=${report.aggregate.camelid_client_ttft_ms.count}`,
    `client_first_byte_mean_ms=${fmt(report.aggregate.camelid_client_first_byte_ms.mean)}`,
    `camelid_ttft_mean_ms=${fmt(report.aggregate.camelid_client_ttft_ms.mean)}`,
    `backend_first_content_mean_ms=${fmt(report.aggregate.camelid_backend_first_content_ms.mean)}`,
    `backend_generate_mean_ms=${fmt(report.aggregate.camelid_backend_generate_ms.mean)}`,
    `first_content_minus_first_byte_mean_ms=${fmt(report.aggregate.first_content_minus_first_byte_ms.mean)}`,
    `backend_generate_minus_first_byte_mean_ms=${fmt(report.aggregate.backend_generate_minus_first_byte_ms.mean)}`,
    `backend_generate_minus_first_content_mean_ms=${fmt(report.aggregate.backend_generate_minus_first_content_ms.mean)}`,
    `prompt_eval_forward_mean_ms=${fmt(report.aggregate.prompt_eval_forward_ms.mean)}`,
    `prompt_eval_logits_mean_ms=${fmt(report.aggregate.prompt_eval_logits_ms.mean)}`,
    `prompt_eval_sample_mean_ms=${fmt(report.aggregate.prompt_eval_sample_ms.mean)}`,
    `first_content_minus_prompt_eval_forward_mean_ms=${fmt(report.aggregate.first_content_minus_prompt_eval_forward_ms.mean)}`,
    `first_content_minus_prompt_eval_forward_plus_sample_mean_ms=${fmt(report.aggregate.first_content_minus_prompt_eval_forward_plus_sample_ms.mean)}`,
    `role_yield_mean_ms=${fmt(report.aggregate.role_yield_ms.mean)}`,
    `first_content_yield_mean_ms=${fmt(report.aggregate.first_content_yield_ms.mean)}`,
    `final_yield_mean_ms=${fmt(report.aggregate.final_yield_ms.mean)}`,
    `client_first_byte_minus_role_yield_mean_ms=${fmt(report.aggregate.client_first_byte_minus_role_yield_ms.mean)}`,
    `client_first_content_minus_content_yield_mean_ms=${fmt(report.aggregate.client_first_content_minus_content_yield_ms.mean)}`,
    `client_done_minus_final_yield_mean_ms=${fmt(report.aggregate.client_done_minus_final_yield_ms.mean)}`,
    `llama_cpp_ttft_mean_ms=${fmt(report.aggregate.llama_cpp_ttft_ms.mean)}`,
    `client_minus_backend_first_content_mean_ms=${fmt(report.aggregate.client_minus_backend_first_content_ms.mean)}`,
    `backend_first_content_delta_vs_llama_cpp_mean_ms=${fmt(report.aggregate.backend_first_content_delta_vs_llama_cpp_ttft_ms.mean)}`,
    `q8_calls_mean=${fmt(report.aggregate.q8_calls.mean)}`,
    `q8_total_gemm_mean_ms=${fmt(report.aggregate.q8_total_gemm_us.mean === null ? null : report.aggregate.q8_total_gemm_us.mean / 1000)}`,
    `q8_total_pack_mean_ms=${fmt(report.aggregate.q8_total_pack_us.mean === null ? null : report.aggregate.q8_total_pack_us.mean / 1000)}`,
    `q8_fused_gate_up_calls_mean=${fmt(report.aggregate.q8_fused_gate_up_calls.mean)}`,
    `q8_gate_up_decode_consumer_activation_mean_ms=${fmt(gateUpDecodeOverhead.activation_ms_mean)}`,
    `q8_gate_up_decode_consumer_tensor_mean_ms=${fmt(gateUpDecodeOverhead.tensor_ms_mean)}`,
    `q8_gate_up_decode_consumer_post_route_mean_ms=${fmt(gateUpDecodeOverhead.total_ms_mean)}`,
    `projection_route_calls_mean=${fmt(report.aggregate.projection_route_calls.mean)}`,
    'top_roles:',
    top,
    'top_q8_gemm_roles:',
    q8Roles,
    'top_projection_routes:',
    projectionRoutes,
    'top_projection_layer_routes:',
    projectionLayerRoutes || '(none)',
    'projection_route_denials:',
    projectionDenials || '(none)',
    'top_layer_role_hotspots:',
    layerHotspots,
    'top_stage_route_role_gaps:',
    stageRouteGaps || '(none)',
    'top_layer_route_role_gaps:',
    layerRouteGaps || '(none)',
    'role_focus_logits_attention:',
    roleFocus,
    'top_backend_first_content_outliers:',
    outliers,
  ].filter(Boolean).join('\n')
}

function parseArgs(argv) {
  const map = new Map()
  map.positionals = []
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) {
      map.positionals.push(arg)
      continue
    }
    const key = arg.slice(2)
    if (key.includes('=')) {
      const [name, ...rest] = key.split('=')
      map.set(name, rest.join('='))
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
      map.set(key, argv[++i])
    } else {
      map.set(key, true)
    }
  }
  return map
}

function finite(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function q8ScheduleCallCount(explicitValue, q8Schedule) {
  const explicit = finite(explicitValue)
  if (explicit !== null) return explicit
  if (!q8Schedule || typeof q8Schedule !== 'object') return null
  const direct = [
    q8Schedule.i8mm_single_projection_calls,
    q8Schedule.i8mm_fused_gate_up_calls,
    q8Schedule.ffn_down_decode_consumer_taken,
    q8Schedule.ffn_down_vnni_decode_taken,
  ]
  let directTotal = 0
  for (const value of direct) {
    const number = finite(value)
    if (number !== null) directTotal += number
  }
  const routes = q8Schedule.projection_routes ?? q8Schedule.output_projection_by_route
  let routeTotal = 0
  if (routes && typeof routes === 'object') {
    for (const route of Object.values(routes)) {
      const calls = finite(route?.calls)
      if (calls !== null) routeTotal += calls
    }
  }
  const total = Math.max(directTotal, routeTotal)
  return total > 0 ? total : null
}

function delta(left, right) {
  left = finite(left)
  right = finite(right)
  return left === null || right === null ? null : round(left - right)
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null
}

function fmt(value) {
  return value === null || value === undefined ? 'null' : Number(value).toFixed(3)
}

function usage() {
  return `Usage: node scripts/summarize-same-host-stream-timing.mjs --input same-host.json [--out summary.json]\n\nSummarizes Camelid same-host streaming diagnostics, first-byte vs backend generate/first-content gaps, backend first-content residuals, role timing hot spots, per-stage and per-layer route-vs-role gaps, per-layer role hot spots, focused logits/attention role buckets, generic Q8 projection routes, and Q8 scheduler work by role.`
}
