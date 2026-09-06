#!/usr/bin/env node
import assert from 'node:assert/strict'

import {
  arenaDefaultModelA,
  arenaModelIsAlreadyReady,
  arenaModelChoices,
  arenaSelectionsAreReady,
  runArenaSequentially,
} from '../src/lib/arenaModels.js'

const runtime = { active_model_id: 'llama-3b', generation_ready: true }
const models = [
  { id: 'hosted', provider_kind: 'external', status: 'ready', model_path: '/models/hosted.gguf' },
  { id: 'embedding', provider_kind: 'local', status: 'ready', model_path: '/models/embed.gguf', embedding_capable: true, generation_capable: false },
  { id: 'catalog-only', provider_kind: 'local', status: 'ready', generation_capable: true },
  { id: 'downloading', provider_kind: 'local', status: 'downloading', model_path: '/models/pending.gguf', generation_capable: true },
  { id: 'tiny', provider_kind: 'local', status: 'registered', model_path: '/models/tiny.gguf', generation_capable: true },
  { id: 'llama-3b', provider_kind: 'local', status: 'ready', model_path: '/models/llama.gguf', generation_capable: true },
]

assert.deepEqual(
  arenaModelChoices(models, runtime).map((model) => model.id),
  ['tiny', 'llama-3b'],
  'Arena choices must be downloaded local generation models, never hosted, embedding, catalog-only, or in-progress rows',
)
assert.equal(arenaDefaultModelA(models, runtime), 'llama-3b', 'the currently loaded model should be Model A by default')
assert.equal(arenaSelectionsAreReady('llama-3b', 'tiny'), true, 'two different models should be comparable')
assert.equal(arenaSelectionsAreReady('tiny', 'tiny'), false, 'the same model on both sides is not a comparison')
assert.equal(arenaSelectionsAreReady('tiny', ''), false, 'both model choices are required')
assert.equal(arenaModelIsAlreadyReady(models.at(-1), runtime), true, 'an active generation-ready model should not be loaded again')
assert.equal(arenaModelIsAlreadyReady({ id: 'saved-id', runtime_model_name: 'runtime-id', model_path: '/models/file.gguf' }, { active_model_id: 'runtime-id', generation_ready: true }), true, 'runtime aliases should match without a reload')
assert.equal(arenaModelIsAlreadyReady({ id: 'saved-id', model_path: '/models/file.gguf' }, { active_model_id: 'file.gguf', generation_ready: true }), true, 'backend filename identities should match without a reload')
assert.equal(arenaModelIsAlreadyReady(models.at(-1), { ...runtime, generation_ready: false }), false, 'a loaded-but-not-ready model must still pass through activation')

const order = []
let concurrent = 0
let maxConcurrent = 0
await runArenaSequentially({
  modelA: { id: 'a' },
  modelB: { id: 'b' },
  runModel: async (model, side) => {
    concurrent += 1
    maxConcurrent = Math.max(maxConcurrent, concurrent)
    order.push(`start-${side}-${model.id}`)
    await Promise.resolve()
    order.push(`finish-${side}-${model.id}`)
    concurrent -= 1
    return model.id
  },
})
assert.equal(maxConcurrent, 1, 'Arena must not race two model loads or generations')
assert.deepEqual(order, ['start-a-a', 'finish-a-a', 'start-b-b', 'finish-b-b'], 'Model B must wait for Model A to finish')

const controller = new AbortController()
const abortedOrder = []
await runArenaSequentially({
  modelA: { id: 'a' },
  modelB: { id: 'b' },
  signal: controller.signal,
  runModel: async (model) => {
    abortedOrder.push(model.id)
    controller.abort()
  },
})
assert.deepEqual(abortedOrder, ['a'], 'stopping Model A must prevent Model B from starting')

console.log('Model Arena smoke passed')
