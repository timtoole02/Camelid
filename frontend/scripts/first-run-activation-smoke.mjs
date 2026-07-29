import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { isCompatibilitySupportedForModel } from '../src/lib/capabilities.js'
import { getChatGateState } from '../src/lib/chatGate.js'
import {
  firstRunCancelOutcome,
  firstRunFailureIsRetryable,
  firstRunRetryAction,
  isFirstRunHost,
  recommendFirstRunModel,
} from '../src/lib/firstRunActivation.js'
import { loadLocalModelForChat, modelFilenameFromPath, warmGenerationPath } from '../src/lib/modelActivation.js'

function response({ ok = true, status = 200, body = {} } = {}) {
  return { ok, status, json: async () => body }
}

/* A capabilities contract shaped like the real one, with the real row ids:
   `qwen3_0_6b_instruct_q8_0` is a `supported_*` exact row whose id carries a token
   ("instruct") that its catalog name and filename do not — the case that used to fall
   out of the exact-row join — `qwen3_1_7b_instruct_q8_0` is a larger supported row, and
   `phi3_mini_4k_instruct_q8_0` is present but NOT supported. */
const capabilities = {
  model_compatibility: [
    { id: 'qwen3_0_6b_instruct_q8_0', family: 'qwen3', quantization: 'Q8_0', status: 'supported_exact_row_smoke', evidence: 'exact row' },
    { id: 'qwen3_1_7b_instruct_q8_0', family: 'qwen3', quantization: 'Q8_0', status: 'supported_exact_row_smoke', evidence: 'exact row' },
    { id: 'nomic_embed_text_v1_5_q8_0', family: 'nomic-bert', quantization: 'Q8_0', status: 'supported_exact_row_embedding', evidence: 'exact embedding row' },
    { id: 'phi3_mini_4k_instruct_q8_0', family: 'phi3', quantization: 'Q8_0', status: 'active_validation_blocked_parity', evidence: 'not promoted' },
  ],
}

const curatedRow = (overrides) => ({
  group: 'curated',
  fit: 'cpu_only_ok',
  quant: 'Q8_0',
  ...overrides,
})

const qwen06b = curatedRow({ catalog_id: 'qwen3_0_6b_instruct_q8_0', name: 'Qwen3 0.6B Q8_0', filename: 'Qwen3-0.6B-Q8_0.gguf', repo_id: 'Qwen/Qwen3-0.6B-GGUF', size_bytes: 639446688 })
const qwen17b = curatedRow({ catalog_id: 'qwen3_1_7b_instruct_q8_0', name: 'Qwen3 1.7B Q8_0', filename: 'Qwen3-1.7B-Q8_0.gguf', repo_id: 'Qwen/Qwen3-1.7B-GGUF', size_bytes: 1834426016 })
const nomic = curatedRow({ catalog_id: 'nomic_embed_text_v1_5_q8_0', name: 'Nomic Embed Text v1.5 Q8_0', filename: 'nomic-embed-text-v1.5.Q8_0.gguf', repo_id: 'nomic-ai/nomic-embed-text-v1.5-GGUF', size_bytes: 146146432, architecture: 'nomic-bert', task_tags: ['embeddings', 'retrieval'] })
const phi3 = curatedRow({ catalog_id: 'phi3_mini_4k_instruct_q8_0', name: 'Phi-3-mini-4k-instruct Q8_0', filename: 'Phi-3-mini-4k-instruct-Q8_0.gguf', repo_id: 'bartowski/Phi-3-mini-4k-instruct-GGUF', size_bytes: 4061221376 })

/* The join that makes the offer possible at all: a curated row's catalog_id IS its
   compatibility row id, and it is the only field that carries the full id. */
assert.equal(
  isCompatibilitySupportedForModel(capabilities, null, qwen06b),
  true,
  'a curated row must resolve to its exact contract row through catalog_id',
)

/* Same join, against the SHIPPED contract rather than a fixture. The ledger is the
   canonical serialization of the capability contract in src/api/mod.rs (re-derived
   and drift-checked in CI), so this fails if the real row id and the real catalog id
   ever stop lining up — the exact way eight pinned supported rows previously rendered
   as merely experimental. */
{
  const ledger = JSON.parse(readFileSync(new URL('../../ledger/camelid-ledger.json', import.meta.url), 'utf8'))
  const shipped = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }
  const row = shipped.model_compatibility.find((entry) => entry.id === 'qwen3_0_6b_instruct_q8_0')
  assert.ok(row, 'the shipped contract must still carry the qwen3 0.6B exact row')
  assert.match(row.status, /^supported/, 'that row is the one the fixture models')
  assert.equal(
    isCompatibilitySupportedForModel(shipped, null, {
      group: 'curated',
      catalog_id: 'qwen3_0_6b_instruct_q8_0',
      name: 'Qwen3 0.6B Q8_0',
      repo_id: 'Qwen/Qwen3-0.6B-GGUF',
      filename: 'Qwen3-0.6B-Q8_0.gguf',
      quant: 'Q8_0',
    }),
    true,
    'the shipped catalog row must resolve as supported against the shipped contract',
  )
}

/* --- first-run detection ------------------------------------------------- */
{
  const online = { status: 'online', loaded_now: false, active_model_id: null }

  assert.equal(isFirstRunHost({ runtime: online, models: [] }), true, 'empty online host is a first run')

  assert.equal(
    isFirstRunHost({ runtime: { ...online, status: 'offline' }, models: [] }),
    false,
    'an offline backend is the backend banner\u2019s problem, not onboarding\u2019s',
  )
  assert.equal(
    isFirstRunHost({ runtime: { ...online, loaded_now: true }, models: [] }),
    false,
    'a loaded model means the user is already through the funnel',
  )
  assert.equal(
    isFirstRunHost({ runtime: { ...online, active_model_id: 'Qwen3-0.6B-Q8_0.gguf' }, models: [] }),
    false,
    'an active model id alone retires the card',
  )
  assert.equal(
    isFirstRunHost({ runtime: online, models: [{ id: 'a', model_path: 'models/Qwen3-0.6B-Q8_0.gguf' }] }),
    false,
    'a GGUF already on disk is not a fresh install',
  )
  assert.equal(
    isFirstRunHost({ runtime: online, models: [{ id: 'hosted', provider_kind: 'external', api_base: 'https://x' }] }),
    true,
    'a record with no local path cannot stand in for a local model',
  )
  assert.equal(isFirstRunHost({}), false, 'no runtime means no claim')
}

/* --- the offer ------------------------------------------------------------ */
{
  const offer = recommendFirstRunModel([qwen17b, phi3, nomic, qwen06b], capabilities)
  assert.equal(offer.kind, 'recommended')
  assert.equal(
    offer.item.catalog_id,
    'qwen3_0_6b_instruct_q8_0',
    'the smallest fitting supported generative row wins; a smaller encoder sidecar cannot replace the first Chat model',
  )

  /* The regression this pins: a smaller row that is NOT a supported contract row
     must never be offered, however cheap its download is. */
  const tempting = curatedRow({ catalog_id: 'tiny_unverified', name: 'Tiny Unverified', filename: 'tiny-unverified-Q8_0.gguf', repo_id: 'someone/tiny', size_bytes: 1 })
  const guarded = recommendFirstRunModel([tempting, qwen06b], capabilities)
  assert.equal(guarded.item.catalog_id, 'qwen3_0_6b_instruct_q8_0', 'an unverified row must not win on size')

  const experimental = { ...qwen06b, group: 'experimental' }
  assert.equal(
    recommendFirstRunModel([experimental], capabilities).kind,
    'no_supported_row',
    'a live Hugging Face row can never anchor the first-run offer',
  )
  assert.equal(
    recommendFirstRunModel([phi3], capabilities).kind,
    'no_supported_row',
    'a curated row whose contract status is not supported_* is not an offer',
  )
}

{
  // Fit is a hard filter: a row the load guard would refuse must not be offered.
  const offer = recommendFirstRunModel(
    [{ ...qwen06b, fit: 'wont_fit' }, { ...qwen17b, fit: 'fits_resident' }],
    capabilities,
  )
  assert.equal(offer.item.catalog_id, 'qwen3_1_7b_instruct_q8_0', 'skip past a row this host cannot load')

  const busy = recommendFirstRunModel([{ ...qwen06b, fit: 'insufficient_free_memory' }], capabilities)
  assert.equal(busy.kind, 'no_fitting_row', 'memory pressure is still a refusal for the one-click path')
  assert.equal(busy.item, null)
  assert.equal(busy.smallest.catalog_id, 'qwen3_0_6b_instruct_q8_0', 'the blocked state can still name what it would have offered')

  const unknown = recommendFirstRunModel([{ ...qwen06b, fit: 'unknown' }], capabilities)
  assert.equal(unknown.kind, 'recommended', 'an unprobed host must not lose its whole catalog')
}

{
  const a = curatedRow({ catalog_id: 'qwen3_1_7b_instruct_q8_0', name: 'Qwen3 1.7B Q8_0', filename: 'Qwen3-1.7B-Q8_0.gguf', size_bytes: 100 })
  const b = curatedRow({ catalog_id: 'qwen3_0_6b_instruct_q8_0', name: 'Qwen3 0.6B Q8_0', filename: 'Qwen3-0.6B-Q8_0.gguf', size_bytes: 100 })
  assert.equal(
    recommendFirstRunModel([a, b], capabilities).item.catalog_id,
    recommendFirstRunModel([b, a], capabilities).item.catalog_id,
    'equal sizes must resolve deterministically, not by catalog order',
  )
  assert.equal(recommendFirstRunModel([], capabilities).kind, 'no_supported_row')
  assert.equal(recommendFirstRunModel(undefined, null).kind, 'no_supported_row')
}

{
  assert.equal(firstRunFailureIsRetryable('model_too_large_for_host'), false, 'a host that is too small stays too small')
  assert.equal(firstRunFailureIsRetryable('unsupported_model_architecture'), false)
  assert.equal(firstRunFailureIsRetryable('host_memory_unavailable'), true, 'memory pressure clears')
  assert.equal(firstRunFailureIsRetryable(''), true, 'an untyped transport failure is worth one more try')
}

/* --- retry must not re-download a file that already landed ------------------
   Activation only runs after the artifact is on disk, so an inspect/load/readiness
   failure leaves a complete GGUF there. Re-installing would refetch 610 MB AND drop
   the completed download record, and the fresh `curl`'s final rename onto the
   existing file can fail -- leaving the user with less than they started with. */
{
  assert.equal(firstRunRetryAction({ artifactInstalled: true }), 'activate')
  assert.equal(firstRunRetryAction({ artifactInstalled: false }), 'download')
  assert.equal(firstRunRetryAction({}), 'download', 'no evidence of a landed file means the download is still owed')
  assert.equal(firstRunRetryAction(), 'download')
}

/* --- cancel is a request, not a fact ---------------------------------------
   The backend answers three ways and only one is a stop: 200 removed a running
   download; 409 means it finished first and KEPT its file (cancel is not delete);
   404 means there is no such download, which during `starting` can simply mean the
   install has not registered yet. Claiming "Nothing was installed" on any of those
   is how the card ends up lying about a 610 MB file sitting on disk. */
{
  // Finished before the cancel took effect: the file is real, so finish the job.
  assert.equal(
    firstRunCancelOutcome({ confirmed: false, stillDownloading: false, artifactInstalled: true }).action,
    'activate',
    'a completed artifact must be activated, never reported as "nothing was installed"',
  )
  assert.equal(
    firstRunCancelOutcome({ confirmed: true, stillDownloading: false, artifactInstalled: true }).action,
    'activate',
    'even a 200 cancel must not discard an artifact that is on disk',
  )

  // The cancel did not take: keep watching rather than freezing the UI on "failed".
  const resumed = firstRunCancelOutcome({ confirmed: false, stillDownloading: true, artifactInstalled: false })
  assert.equal(resumed.action, 'resume')
  assert.match(resumed.message, /still running/)

  // Unreadable probes must never collapse into "no".
  for (const observation of [
    { stillDownloading: null, artifactInstalled: false },
    { stillDownloading: false, artifactInstalled: null },
    { stillDownloading: null, artifactInstalled: null },
  ]) {
    const outcome = firstRunCancelOutcome({ confirmed: false, ...observation })
    assert.equal(outcome.action, 'resume', `unverified observation must not claim a result: ${JSON.stringify(observation)}`)
    assert.match(outcome.message, /could not confirm/)
  }
  assert.equal(
    firstRunCancelOutcome({ confirmed: true, stillDownloading: null, artifactInstalled: null }).action,
    'resume',
    'not even a confirmed 200 may report "nothing installed" without looking',
  )

  // Nothing running and nothing on disk: the only honest "canceled".
  const canceled = firstRunCancelOutcome({ confirmed: true, stillDownloading: false, artifactInstalled: false })
  assert.equal(canceled.action, 'canceled')
  assert.match(canceled.message, /Nothing was installed/)
  const inferred = firstRunCancelOutcome({ confirmed: false, stillDownloading: false, artifactInstalled: false })
  assert.equal(inferred.action, 'canceled')
  assert.match(inferred.message, /no longer running and nothing was installed/)
  assert.doesNotMatch(inferred.message, /^Download canceled/, 'an unconfirmed stop must not be worded as a confirmed one')

  assert.equal(firstRunCancelOutcome().action, 'resume', 'no observations at all is not a cancellation')
}

/* --- the model the card installs must not read as unverified ---------------
   Observed before this was fixed: the card installs a validated row, and the chat
   surface it hands the user then rendered "Experimental — output is unverified and
   has no parity guarantee" over it. The cause is that the exact-row identity match
   reconstructs a row id from the model's display name, and this row's id carries an
   "instruct" token its filename does not. Whether that token was visible depended on
   WHICH PATH issued the load — the engine's startup auto-load names a model from GGUF
   metadata ("Qwen3 0.6B Instruct", matches), `POST /api/models/load` names it whatever
   id the caller sent ("Qwen3-0.6B-Q8_0.gguf", does not) — so one file produced two
   contradictory claims on the same machine. The gate now also accepts the backend's
   own exact-artifact verdict, exactly as the Models page already does. */
{
  const runtime = { status: 'online', loaded_now: true, generation_ready: true, active_model_id: 'Qwen3-0.6B-Q8_0.gguf' }
  const asAppLoaded = {
    id: 'Qwen3-0.6B-Q8_0.gguf',
    name: 'Qwen3-0.6B-Q8_0.gguf',
    model_path: 'models/Qwen3-0.6B-Q8_0.gguf',
    quant: 'Q8_0',
    provider_kind: 'local',
    status: 'ready',
    loaded_now: true,
    generation_ready: true,
    lane_class: 'supported',
  }
  const gate = getChatGateState(capabilities, asAppLoaded, runtime)
  assert.equal(gate.contractSupported, true, 'the backend exact-artifact verdict must reach the chat gate')
  assert.equal(gate.chatMode, 'supported', 'a row the card calls validated must not chat as experimental')
  assert.equal(gate.chatUnlocked, true)

  // Same file, same verdict, whichever path named it.
  const asAutoLoaded = { ...asAppLoaded, id: 'Qwen3 0.6B Instruct', name: 'Qwen3 0.6B Instruct' }
  const autoGate = getChatGateState(capabilities, asAutoLoaded, { ...runtime, active_model_id: 'Qwen3 0.6B Instruct' })
  assert.equal(autoGate.chatMode, 'supported', 'the load path must not change what a file IS')

  // The verdict adds evidence; it can never substitute for runtime readiness.
  // (`chatMode` names the LANE a model belongs to; `chatUnlocked` owns readiness.)
  const notReady = getChatGateState(capabilities, asAppLoaded, { status: 'online', loaded_now: false, generation_ready: false })
  assert.equal(notReady.contractSupported, true)
  assert.equal(notReady.runtimeReady, false)
  assert.equal(notReady.chatUnlocked, false, 'a supported row still may not chat before it is loaded and ready')
  assert.equal(notReady.experimentalUnlocked, false, 'and it must not fall through to the experimental lane either')

  // And it must not promote a file the backend did NOT mark supported.
  const experimental = { ...asAppLoaded, lane_class: 'experimental_implemented' }
  assert.equal(
    getChatGateState(capabilities, experimental, runtime).chatMode,
    'experimental',
    'only an exact supported_* artifact verdict may unlock the supported lane',
  )
  const noVerdict = { ...asAppLoaded }
  delete noVerdict.lane_class
  assert.equal(
    getChatGateState(capabilities, noVerdict, runtime).chatMode,
    'experimental',
    'absent a verdict the gate falls back to contract matching, unchanged',
  )
}

/* --- the load protocol ---------------------------------------------------- */
assert.equal(modelFilenameFromPath('C:\\models\\Qwen3-0.6B-Q8_0.gguf'), 'Qwen3-0.6B-Q8_0.gguf')
assert.equal(modelFilenameFromPath('/srv/models/Qwen3-0.6B-Q8_0.gguf'), 'Qwen3-0.6B-Q8_0.gguf')
assert.equal(modelFilenameFromPath(null), '')

{
  const calls = []
  const stages = []
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test/',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    onStage: (stage) => stages.push(stage),
    fetchImpl: async (url, options) => {
      calls.push({ url, body: options?.body ? JSON.parse(options.body) : null })
      if (url.endsWith('/api/models/inspect')) return response({ body: { architecture: 'qwen3' } })
      if (url.endsWith('/api/models/load')) return response({ body: { id: 'Qwen3-0.6B-Q8_0.gguf' } })
      if (url.endsWith('/api/models/current')) return response({ body: { path: 'models/Qwen3-0.6B-Q8_0.gguf' } })
      if (url.endsWith('/v1/health')) {
        return response({ body: { loaded_now: true, generation_ready: true, active_model_id: 'Qwen3-0.6B-Q8_0.gguf' } })
      }
      throw new Error(`unexpected request ${url}`)
    },
  })
  assert.equal(result.ok, true)
  assert.deepEqual(stages, ['checking', 'loading'])
  assert.deepEqual(
    calls.map((call) => call.url),
    [
      'http://camelid.test/api/models/inspect',
      'http://camelid.test/api/models/load',
      'http://camelid.test/api/models/current',
      'http://camelid.test/v1/health',
    ],
    'inspect must precede the load, and readiness must be confirmed after it',
  )
  assert.equal(calls[0].body.path, 'models/Qwen3-0.6B-Q8_0.gguf', 'the load path stays models-relative')
  assert.equal(calls[1].body.replace, true, 'loading a model is a swap, never a second resident copy')
  assert.equal(calls[1].body.set_active, true, 'a generative model becomes the active Chat model')
}

{
  // The supported Nomic encoder is a sidecar: it must not replace the active Chat
  // model, and readiness is a real embedding rather than generation_ready.
  const calls = []
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'nomic-embed-text-v1.5.Q8_0.gguf',
    fetchImpl: async (url, options) => {
      calls.push({ url, body: options?.body ? JSON.parse(options.body) : null })
      if (url.endsWith('/api/models/inspect')) return response({ body: { architecture: 'nomic-bert' } })
      if (url.endsWith('/api/models/load')) return response({ body: {} })
      if (url.endsWith('/v1/embeddings')) {
        return response({ body: { data: [{ embedding: Array(256).fill(0) }] } })
      }
      throw new Error(`unexpected request ${url}`)
    },
  })
  assert.equal(result.ok, true)
  assert.equal(result.embedding, true)
  assert.deepEqual(
    calls.map((call) => call.url),
    [
      'http://camelid.test/api/models/inspect',
      'http://camelid.test/api/models/load',
      'http://camelid.test/v1/embeddings',
    ],
    'the encoder must stop at sidecar readiness instead of checking Chat readiness',
  )
  assert.equal(calls[1].body.replace, false)
  assert.equal(calls[1].body.set_active, false)
}

{
  // A refused architecture must stop before anything reads the weights.
  let loadAttempted = false
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'weird.gguf',
    fetchImpl: async (url) => {
      if (url.endsWith('/api/models/inspect')) {
        return response({ body: { blocker: { code: 'unsupported_model_architecture', message: 'Camelid does not implement mamba.' } } })
      }
      loadAttempted = true
      return response()
    },
  })
  assert.equal(loadAttempted, false, 'a blocked inspect must never reach the load')
  assert.equal(result.ok, false)
  assert.equal(result.stage, 'checking')
  assert.equal(result.code, 'unsupported_model_architecture')
  assert.deepEqual(result.blocker, { code: 'unsupported_model_architecture', message: 'Camelid does not implement mamba.' })
}

{
  // The load-time fit guard's typed 422 must arrive as a code, not as raw HTTP text.
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async (url) => {
      if (url.endsWith('/api/models/inspect')) return response({ body: {} })
      return response({
        ok: false,
        status: 422,
        body: { error: { code: 'model_too_large_for_host', message: 'This model (~0.6 GB) is larger than this machine can hold in memory.' } },
      })
    },
  })
  assert.equal(result.ok, false)
  assert.equal(result.code, 'model_too_large_for_host')
  assert.match(result.message, /larger than this machine/)
  assert.doesNotMatch(result.message, /HTTP 422/, 'the typed message replaces the raw status, it does not append to it')
  assert.equal(firstRunFailureIsRetryable(result.code), false)
}

{
  // A load the engine accepted but did not make ready is a failure, not a success.
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async (url) => {
      if (url.endsWith('/api/models/inspect')) return response({ body: {} })
      if (url.endsWith('/api/models/load')) return response({ body: {} })
      if (url.endsWith('/api/models/current')) return response({ body: { path: 'models/Qwen3-0.6B-Q8_0.gguf' } })
      return response({ body: { loaded_now: true, generation_ready: false, active_model_id: 'Qwen3-0.6B-Q8_0.gguf' } })
    },
  })
  assert.equal(result.ok, false)
  assert.match(result.message, /not generation-ready/)
}

{
  // Identity: the engine must confirm THIS file, not merely return 200.
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async (url) => {
      if (url.endsWith('/api/models/inspect')) return response({ body: {} })
      if (url.endsWith('/api/models/load')) return response({ body: {} })
      return response({ body: { path: 'models/some-other-model.gguf' } })
    },
  })
  assert.equal(result.ok, false)
  assert.match(result.message, /did not confirm/)
}

{
  // An owner that already polls /api/models/current answers identity from it.
  let currentFetches = 0
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    readActiveFilename: async () => 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async (url) => {
      if (url.endsWith('/api/models/current')) currentFetches += 1
      if (url.endsWith('/v1/health')) {
        return response({ body: { loaded_now: true, generation_ready: true, active_model_id: 'Qwen3-0.6B-Q8_0.gguf' } })
      }
      return response({ body: {} })
    },
  })
  assert.equal(result.ok, true)
  assert.equal(currentFetches, 0, 'an injected reader must replace the redundant fetch, not add to it')
}

{
  // Transport failure is reported at the stage it happened, never thrown.
  const result = await loadLocalModelForChat({
    apiBase: 'http://camelid.test',
    filename: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async () => { throw new Error('connection refused') },
  })
  assert.equal(result.ok, false)
  assert.equal(result.stage, 'checking')
  assert.equal(result.message, 'connection refused')
}

/* --- warm-up -------------------------------------------------------------- */
{
  let request = null
  const warmed = await warmGenerationPath({
    apiBase: 'http://camelid.test',
    modelId: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async (url, options) => {
      request = { url, body: JSON.parse(options.body) }
      return response({ body: { choices: [] } })
    },
  })
  assert.equal(warmed, true)
  assert.equal(request.url, 'http://camelid.test/v1/chat/completions')
  assert.equal(request.body.max_tokens, 1, 'the warm-up must build the engine, not generate a reply')
  assert.equal(request.body.stream, false)
  assert.equal(request.body.model, 'Qwen3-0.6B-Q8_0.gguf')

  const failed = await warmGenerationPath({
    apiBase: 'http://camelid.test',
    modelId: 'Qwen3-0.6B-Q8_0.gguf',
    fetchImpl: async () => { throw new Error('cold start') },
  })
  assert.equal(failed, false, 'a failed warm-up degrades to the lazy build; it never breaks activation')
}

console.log('first-run activation smoke: all checks passed')
