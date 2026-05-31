#!/usr/bin/env node

import fs from 'node:fs'

const args = parseArgs(process.argv.slice(2))
if (!args.left || !args.right) usage()

const topN = positiveInteger(args.top ?? '20', '--top')
const tolerance = nonNegativeNumber(args.tolerance ?? '0.000001', '--tolerance')
const left = loadTrace(args.left)
const right = loadTrace(args.right)
const report = compareForwardTraces(left, right, { tolerance, topN })

if (args['json-out']) fs.writeFileSync(args['json-out'], `${JSON.stringify(report, null, 2)}\n`)

console.log(`left=${args.left}`)
console.log(`right=${args.right}`)
console.log(`schema=${report.schema}`)
console.log(`tolerance=${report.tolerance}`)
console.log(`prompt_tokens_match=${report.prompt_tokens_match}`)
console.log(`input_token_delta=${JSON.stringify(report.input_token_delta)}`)
console.log(`position_id_delta=${JSON.stringify(report.position_id_delta)}`)
console.log(`generated_token_delta=${JSON.stringify(report.generated_token_delta)}`)
console.log(`known_good_token_delta=${JSON.stringify(report.known_good_token_delta)}`)
console.log(`stage_paths_match=${report.stage_path_alignment.match}`)
if (report.stage_path_alignment.first_mismatch) {
  console.log(`first_stage_path_mismatch=${JSON.stringify(report.stage_path_alignment.first_mismatch)}`)
}
if (report.first_changed_stage) console.log(`first_changed_stage=${report.first_changed_stage.path}`)
else console.log('first_changed_stage=none')
console.log(`changed_stage_count=${report.changed_stage_count}`)
console.log(`largest_stage_deltas=${JSON.stringify(report.largest_stage_deltas.slice(0, Math.min(8, report.largest_stage_deltas.length)))}`)
console.log(`top_logit_deltas=${JSON.stringify(report.top_logit_deltas.slice(0, Math.min(8, report.top_logit_deltas.length)))}`)
console.log(`output_projection_token_deltas=${JSON.stringify(report.output_projection_token_deltas.slice(0, Math.min(8, report.output_projection_token_deltas.length)))}`)

function usage() {
  console.error('usage: node scripts/compare-forward-traces.mjs --left <forward-trace.json> --right <forward-trace.json> [--tolerance 0.000001] [--top 20] [--json-out <report.json>]')
  console.error('compares ordered camelid.forward-trace.v1 bundles and reports the first changed forward-pass stage from embedding through logits')
  process.exit(2)
}

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (!arg.startsWith('--')) usage()
    const key = arg.slice(2)
    const value = argv[i + 1]
    if (!value || value.startsWith('--')) usage()
    out[key] = value
    i++
  }
  return out
}

function positiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`)
  return parsed
}

function nonNegativeNumber(value, name) {
  const parsed = Number.parseFloat(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be a non-negative number`)
  return parsed
}

function loadTrace(path) {
  const trace = JSON.parse(fs.readFileSync(path, 'utf8'))
  if (trace.schema !== 'camelid.forward-trace.v1') {
    throw new Error(`${path}: expected schema camelid.forward-trace.v1, got ${JSON.stringify(trace.schema)}`)
  }
  if (!Array.isArray(trace.stages)) throw new Error(`${path}: missing stages array`)
  if (trace.stage_count !== undefined && trace.stage_count !== trace.stages.length) {
    throw new Error(`${path}: stage_count ${JSON.stringify(trace.stage_count)} does not match stages length ${trace.stages.length}`)
  }
  for (const [index, stage] of trace.stages.entries()) {
    if (!stage || typeof stage !== 'object') throw new Error(`${path}: stage ${index} is not an object`)
    if (stage.order !== index) throw new Error(`${path}: stage ${index} has non-contiguous order ${JSON.stringify(stage.order)}`)
    if (typeof stage.path !== 'string' || stage.path.length === 0) throw new Error(`${path}: stage ${index} has missing path`)
    if (typeof stage.kind !== 'string' || stage.kind.length === 0) throw new Error(`${path}: stage ${index} has missing kind`)
  }
  return { path, trace }
}

function compareForwardTraces(left, right, { tolerance, topN }) {
  const stageDeltas = compareStages(left.trace.stages, right.trace.stages, tolerance)
  const changedStageDeltas = stageDeltas.filter(delta => delta.changed)
  return {
    schema: 'camelid.forward-trace-comparison.v1',
    tolerance,
    left: summarizeTrace(left),
    right: summarizeTrace(right),
    prompt_tokens_match: arraysEqual(left.trace.prompt_token_ids, right.trace.prompt_token_ids),
    input_token_delta: {
      left: left.trace.input_token_ids ?? [],
      right: right.trace.input_token_ids ?? [],
      match: arraysEqual(left.trace.input_token_ids, right.trace.input_token_ids),
    },
    position_id_delta: {
      left: left.trace.position_ids ?? [],
      right: right.trace.position_ids ?? [],
      match: arraysEqual(left.trace.position_ids, right.trace.position_ids),
    },
    generated_token_delta: {
      left: left.trace.generated_token_ids ?? [],
      right: right.trace.generated_token_ids ?? [],
      match: arraysEqual(left.trace.generated_token_ids, right.trace.generated_token_ids),
    },
    known_good_token_delta: {
      left: left.trace.source?.known_good_token_ids ?? [],
      right: right.trace.source?.known_good_token_ids ?? [],
      match: arraysEqual(left.trace.source?.known_good_token_ids, right.trace.source?.known_good_token_ids),
    },
    selected_layer_delta: {
      left: left.trace.selected_layers ?? [],
      right: right.trace.selected_layers ?? [],
      match: arraysEqual(left.trace.selected_layers, right.trace.selected_layers),
    },
    dense_metadata_deltas: topChanges(compareAny('dense_metadata', left.trace.dense_metadata ?? {}, right.trace.dense_metadata ?? {}, tolerance), topN),
    stage_path_alignment: compareStagePathAlignment(left.trace.stages, right.trace.stages),
    first_changed_stage: changedStageDeltas[0] ?? null,
    largest_stage_deltas: changedStageDeltas.slice().sort(compareDeltasByScore).slice(0, topN),
    stage_delta_count: stageDeltas.length,
    changed_stage_count: changedStageDeltas.length,
    top_logit_deltas: compareRowsByToken('top_logits', left.trace.top_logits ?? [], right.trace.top_logits ?? [], tolerance, topN),
    output_projection_token_deltas: compareRowsByToken('output_projection', left.trace.output_projection ?? [], right.trace.output_projection ?? [], tolerance, topN),
  }
}

function summarizeTrace({ path, trace }) {
  return {
    path,
    source: trace.source ?? null,
    stage_count: trace.stage_count ?? trace.stages.length,
    selected_layers: trace.selected_layers ?? [],
    layer_count: trace.layer_count ?? null,
    first_stage: trace.stages[0]?.path ?? null,
    last_stage: trace.stages.at(-1)?.path ?? null,
  }
}

function compareStagePathAlignment(leftStages, rightStages) {
  const maxLength = Math.max(leftStages.length, rightStages.length)
  for (let index = 0; index < maxLength; index++) {
    const left = leftStages[index]?.path ?? null
    const right = rightStages[index]?.path ?? null
    if (left !== right) {
      return {
        match: false,
        first_mismatch: { order: index, left, right },
        left_stage_count: leftStages.length,
        right_stage_count: rightStages.length,
      }
    }
  }
  return {
    match: true,
    first_mismatch: null,
    left_stage_count: leftStages.length,
    right_stage_count: rightStages.length,
  }
}

function compareStages(leftStages, rightStages, tolerance) {
  const out = []
  const maxLength = Math.max(leftStages.length, rightStages.length)
  for (let index = 0; index < maxLength; index++) {
    const left = leftStages[index]
    const right = rightStages[index]
    if (!left || !right) {
      out.push({
        order: index,
        path: left?.path ?? right?.path ?? `stage.${index}`,
        changed: true,
        score: Number.POSITIVE_INFINITY,
        reason: !left ? 'missing_left_stage' : 'missing_right_stage',
        changed_field_count: 1,
        top_changed_paths: [{ path: `stages.${index}`, kind: !left ? 'missing_left' : 'missing_right', score: Number.POSITIVE_INFINITY }],
      })
      continue
    }
    if (left.path !== right.path) {
      out.push({
        order: index,
        path: left.path,
        right_path: right.path,
        changed: true,
        score: Number.POSITIVE_INFINITY,
        reason: 'stage_path_mismatch',
        changed_field_count: 1,
        top_changed_paths: [{ path: `stages.${index}.path`, kind: 'value', left: left.path, right: right.path, score: Number.POSITIVE_INFINITY }],
      })
      continue
    }
    const changes = compareAny(left.path, comparableStage(left), comparableStage(right), tolerance)
    out.push({
      order: left.order ?? index,
      path: left.path,
      kind: left.kind,
      changed: changes.length > 0,
      score: maxChangeScore(changes),
      changed_field_count: changes.length,
      top_changed_paths: topChanges(changes, 12),
    })
  }
  return out
}

function comparableStage(stage) {
  const { order: _order, path: _path, ...rest } = stage
  return rest
}

function compareRowsByToken(label, leftRows, rightRows, tolerance, topN) {
  const leftMap = tokenRowMap(leftRows)
  const rightMap = tokenRowMap(rightRows)
  const tokenIds = Array.from(new Set([...leftMap.keys(), ...rightMap.keys()]))
    .sort((left, right) => numericTokenSort(left, right))
  return tokenIds
    .map(tokenId => {
      const left = leftMap.get(tokenId)
      const right = rightMap.get(tokenId)
      if (!left || !right) {
        return {
          token_id: tokenId,
          changed: true,
          score: Number.POSITIVE_INFINITY,
          reason: !left ? 'missing_left_token' : 'missing_right_token',
          left: summarizeTokenRow(left),
          right: summarizeTokenRow(right),
          changed_field_count: 1,
          top_changed_paths: [{ path: `${label}.${tokenId}`, kind: !left ? 'missing_left' : 'missing_right', score: Number.POSITIVE_INFINITY }],
        }
      }
      const changes = compareAny(`${label}.${tokenId}`, left, right, tolerance)
      return {
        token_id: tokenId,
        changed: changes.length > 0,
        score: maxChangeScore(changes),
        left: summarizeTokenRow(left),
        right: summarizeTokenRow(right),
        changed_field_count: changes.length,
        top_changed_paths: topChanges(changes, 8),
      }
    })
    .filter(delta => delta.changed)
    .sort(compareDeltasByScore)
    .slice(0, topN)
}

function tokenRowMap(rows) {
  const map = new Map()
  for (const [index, row] of rows.entries()) {
    if (!row || typeof row !== 'object') continue
    const tokenId = row.token_id ?? `index:${index}`
    map.set(tokenId, row)
  }
  return map
}

function numericTokenSort(left, right) {
  const leftNumber = Number(left)
  const rightNumber = Number(right)
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber
  return String(left).localeCompare(String(right))
}

function summarizeTokenRow(row) {
  if (!row) return null
  return {
    token_id: row.token_id ?? null,
    text: row.text ?? null,
    rank: row.rank ?? null,
    logit: numberOrNull(row.logit),
    reconstructed_logit: numberOrNull(row.reconstructed_logit),
    selected: row.selected ?? null,
    layout: row.layout ?? null,
  }
}

function compareAny(path, left, right, tolerance) {
  const out = []
  compareValue(path, left, right, tolerance, out)
  return out
}

function compareValue(path, left, right, tolerance, out) {
  if (typeof left === 'number' || typeof right === 'number') {
    if (typeof left !== 'number' || typeof right !== 'number') {
      out.push({ path, kind: 'type', left: preview(left), right: preview(right), score: Number.POSITIVE_INFINITY })
      return
    }
    const leftFinite = Number.isFinite(left)
    const rightFinite = Number.isFinite(right)
    if (!leftFinite || !rightFinite) {
      if (left !== right) out.push({ path, kind: 'number', left, right, delta: Number.POSITIVE_INFINITY, score: Number.POSITIVE_INFINITY })
      return
    }
    const delta = Math.abs(left - right)
    if (delta > tolerance) out.push({ path, kind: 'number', left, right, delta, score: delta })
    return
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right)) {
      out.push({ path, kind: 'type', left: preview(left), right: preview(right), score: Number.POSITIVE_INFINITY })
      return
    }
    if (left.length !== right.length) {
      out.push({ path: `${path}.length`, kind: 'array_length', left: left.length, right: right.length, score: Number.POSITIVE_INFINITY })
    }
    const length = Math.min(left.length, right.length)
    for (let index = 0; index < length; index++) compareValue(`${path}[${index}]`, left[index], right[index], tolerance, out)
    return
  }

  if (isPlainObject(left) || isPlainObject(right)) {
    if (!isPlainObject(left) || !isPlainObject(right)) {
      out.push({ path, kind: 'type', left: preview(left), right: preview(right), score: Number.POSITIVE_INFINITY })
      return
    }
    const keys = Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).sort()
    for (const key of keys) {
      if (!(key in left)) {
        out.push({ path: `${path}.${key}`, kind: 'missing_left', right: preview(right[key]), score: Number.POSITIVE_INFINITY })
        continue
      }
      if (!(key in right)) {
        out.push({ path: `${path}.${key}`, kind: 'missing_right', left: preview(left[key]), score: Number.POSITIVE_INFINITY })
        continue
      }
      compareValue(`${path}.${key}`, left[key], right[key], tolerance, out)
    }
    return
  }

  if (left !== right) {
    out.push({ path, kind: 'value', left: preview(left), right: preview(right), score: Number.POSITIVE_INFINITY })
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function preview(value) {
  if (Array.isArray(value)) return { type: 'array', length: value.length }
  if (isPlainObject(value)) return { type: 'object', keys: Object.keys(value).slice(0, 8) }
  return value ?? null
}

function topChanges(changes, topN) {
  return changes.slice().sort(compareDeltasByScore).slice(0, topN)
}

function compareDeltasByScore(left, right) {
  const leftInf = !Number.isFinite(left.score)
  const rightInf = !Number.isFinite(right.score)
  if (leftInf !== rightInf) return leftInf ? -1 : 1
  if (right.score !== left.score) return right.score - left.score
  return String(left.path ?? left.token_id ?? '').localeCompare(String(right.path ?? right.token_id ?? ''))
}

function maxChangeScore(changes) {
  if (changes.length === 0) return 0
  if (changes.some(change => !Number.isFinite(change.score))) return Number.POSITIVE_INFINITY
  return Math.max(...changes.map(change => change.score))
}

function arraysEqual(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right)) return left === right
  if (left.length !== right.length) return false
  return left.every((value, index) => value === right[index])
}

function numberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
