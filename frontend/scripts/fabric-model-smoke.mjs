#!/usr/bin/env node
/* Unit coverage for the Cluster view's pure layer.
 *
 * These are the rules that decide whether the page tells the truth, so they are
 * proved here rather than inferred from a screenshot. The load-bearing ones:
 *
 *   - `node_detail` ABSENT (the proxy is not on loopback, so it withholds) must
 *     never read the same as `node_detail: []` (the fabric really is empty).
 *     The fabric's own tests assert the server draws that distinction; this
 *     asserts the client keeps it.
 *   - A field we were not sent stays null. A node that is not ready reports no
 *     load at all, and rendering 0 would invent an idle machine.
 *   - A real 0 must survive as 0, or the rule above would hide working nodes.
 *
 * No browser and no network: every input here is a literal answer body.
 */
import assert from 'node:assert/strict'
import {
  classifyHealthBody,
  corsCommand,
  crossOriginDiagnosis,
  describeFabric,
  describePlacement,
  fabricPosture,
  fabricProblemMessage,
  mixedModeAcceptance,
  modelPlacements,
  provenanceLabel,
  requirementLimits,
  routingCommand,
} from '../src/lib/fabricModel.js'
import {
  clearCachedFabricNodes,
  endpointLabel,
  normalizeEndpoint,
  probeFabric,
  readCachedFabricNodes,
  readFabric,
} from '../src/lib/fabricClient.js'

let checks = 0
function check(name, fn) {
  fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

async function checkAsync(name, fn) {
  await fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

/* The client half needs a network; a stubbed `fetch` stands in for the browser's.
   Restored even when a check throws, so one failure cannot leak into the next. */
async function withFetch(stub, fn) {
  const original = globalThis.fetch
  globalThis.fetch = stub
  try {
    return await fn()
  } finally {
    globalThis.fetch = original
  }
}

// What a browser does with a cross-origin read the server did not allow, and
// with a dead socket: the same rejection, with nothing in it to tell them apart.
const refuse = () => Promise.reject(new TypeError('Failed to fetch'))

const answered = (body, httpStatus = 200) => ({ outcome: 'answered', httpStatus, body })

const READY_NODE = {
  spec: { label: 'win', host: '127.0.0.1', port: 8181 },
  status: {
    state: 'ready',
    active_model_id: 'Llama 3.2 1B Instruct',
    backend: 'cpu_q8_runtime_repack',
    version: 'v0.6.1-267',
    in_flight: 2,
    waiting: 1,
  },
  latency_ms: 3,
}

const NOT_READY_NODE = {
  spec: { label: 'mac', host: 'workstation.local', port: 8181 },
  status: { state: 'not_ready', reason: 'no model loaded' },
  latency_ms: 21,
}

const UNREACHABLE_NODE = {
  spec: { label: 'pi', host: '127.0.0.1', port: 8199 },
  status: { state: 'unreachable', reason: 'connection timed out' },
  latency_ms: null,
}

function fabricBody({ nodes = null, models = [], ready = true, counts = null } = {}) {
  const body = {
    ok: true,
    service: 'camelid-fabric',
    version: '0.6.1',
    build: 'v0.6.1-267-gabc1234',
    ready,
  }
  if (nodes !== null) {
    body.nodes = counts || {
      total: nodes.length,
      ready: nodes.filter((n) => n.status.state === 'ready').length,
      not_ready: nodes.filter((n) => n.status.state === 'not_ready').length,
      unreachable: nodes.filter((n) => n.status.state === 'unreachable').length,
    }
    body.models = models
    body.node_detail = nodes
  }
  return body
}

console.log('fabric model')

/* ---- which process answered ---- */

check('a fabric proxy is recognised by its service name', () => {
  assert.equal(classifyHealthBody({ service: 'camelid-fabric' }), 'fabric')
})

check('an engine is recognised, and never rendered as an empty fabric', () => {
  // The engine carries `engine` and no `service`; the proxy is the other way
  // round. `fabric serve` shaped its payload that way on purpose.
  assert.equal(classifyHealthBody({ ok: true, engine: 'camelid', generation_ready: true }), 'engine')
  const described = describeFabric(answered({ ok: true, engine: 'camelid' }))
  assert.equal(described.problem.code, 'is_an_engine')
  assert.equal(described.nodes, null)
  assert.match(fabricProblemMessage(described.problem, '127.0.0.1:8181'), /not a fabric proxy/)
})

check('anything else is foreign, not an empty fabric', () => {
  assert.equal(classifyHealthBody({ hello: 'world' }), 'foreign')
  assert.equal(classifyHealthBody(null), 'foreign')
  assert.equal(describeFabric(answered({ hello: 'world' })).problem.code, 'not_a_fabric')
})

/* ---- failures are failures, not emptiness ---- */

check('an unreachable proxy yields no counts and no nodes', () => {
  const described = describeFabric({ outcome: 'unreachable', detail: 'No answer within 4000ms.' })
  assert.equal(described.problem.code, 'unreachable')
  assert.equal(described.counts, null)
  assert.equal(described.nodes, null)
  assert.equal(described.ready, null)
  assert.equal(described.detail, 'unknown')
  assert.match(fabricProblemMessage(described.problem, '127.0.0.1:8282'), /No fabric proxy answered at 127\.0\.0\.1:8282/)
})

check('a non-JSON answer is malformed, not unreachable', () => {
  const described = describeFabric({ outcome: 'malformed', detail: 'Answered 200 with something that is not JSON.' })
  assert.equal(described.problem.code, 'malformed_answer')
  assert.equal(described.nodes, null)
})

check('no probe at all is still not an empty fabric', () => {
  const described = describeFabric(undefined)
  assert.equal(described.nodes, null)
  assert.equal(described.counts, null)
})

/* ---- THE distinction: withheld is not empty ---- */

check('a proxy that withholds detail is not reported as having zero nodes', () => {
  const withheld = describeFabric(answered(fabricBody()))
  assert.equal(withheld.detail, 'withheld')
  assert.equal(withheld.nodes, null, 'nodes must be unknown, not []')
  assert.equal(withheld.counts, null, 'counts must be unknown, not 0')
  assert.equal(withheld.models, null, 'models must be unknown, not []')
  assert.equal(withheld.ready, true, 'readiness is still disclosed and must survive')
  assert.equal(withheld.problem, null, 'withholding detail is not a failure')
})

check('a proxy with a genuinely empty fabric reports zero, not unknown', () => {
  const empty = describeFabric(answered(fabricBody({ nodes: [], ready: false }), 503))
  assert.equal(empty.detail, 'disclosed')
  assert.deepEqual(empty.nodes, [], 'an empty fabric is a known empty list')
  assert.equal(empty.counts.total, 0)
  assert.equal(empty.ready, false)
  assert.equal(empty.problem, null, '503 from a reachable proxy is an answer, not a failure')
})

/* ---- unknown never becomes zero, and zero never becomes unknown ---- */

check('a node that is not ready reports no load, rather than an idle one', () => {
  const [node] = describeFabric(answered(fabricBody({ nodes: [NOT_READY_NODE] }))).nodes
  assert.equal(node.state, 'not_ready')
  assert.equal(node.inFlight, null, 'in-flight must be unknown, never 0')
  assert.equal(node.waiting, null)
  assert.equal(node.activeModelId, null)
  assert.equal(node.backend, null)
  assert.equal(node.version, null)
  assert.equal(node.reason, 'no model loaded', 'the proxy reason must reach the operator verbatim')
})

check('an unreachable node keeps its reason and reports nothing else', () => {
  const [node] = describeFabric(answered(fabricBody({ nodes: [UNREACHABLE_NODE] }))).nodes
  assert.equal(node.state, 'unreachable')
  assert.equal(node.reason, 'connection timed out')
  assert.equal(node.inFlight, null)
  assert.equal(node.latencyMs, null, 'an unrecorded probe is unknown, not 0 ms')
})

check('a real zero survives as zero', () => {
  const idle = { ...READY_NODE, status: { ...READY_NODE.status, in_flight: 0, waiting: 0 }, latency_ms: 0 }
  const [node] = describeFabric(answered(fabricBody({ nodes: [idle] }))).nodes
  assert.equal(node.inFlight, 0)
  assert.equal(node.waiting, 0)
  assert.equal(node.latencyMs, 0)
})

check('a ready node surfaces every field the proxy sent', () => {
  const [node] = describeFabric(answered(fabricBody({ nodes: [READY_NODE] }))).nodes
  assert.equal(node.label, 'win')
  assert.equal(node.authority, '127.0.0.1:8181')
  assert.equal(node.activeModelId, 'Llama 3.2 1B Instruct')
  assert.equal(node.backend, 'cpu_q8_runtime_repack')
  assert.equal(node.version, 'v0.6.1-267')
  assert.equal(node.inFlight, 2)
  assert.equal(node.waiting, 1)
  assert.equal(node.latencyMs, 3)
  assert.equal(node.reason, null, 'a ready node has nothing to explain')
})

check('a number-shaped string is not accepted as a number', () => {
  const lying = { ...READY_NODE, status: { ...READY_NODE.status, in_flight: '2' }, latency_ms: '3' }
  const [node] = describeFabric(answered(fabricBody({ nodes: [lying] }))).nodes
  assert.equal(node.inFlight, null)
  assert.equal(node.latencyMs, null)
})

check('an unrecognised node state is unknown rather than guessed', () => {
  const odd = { spec: { label: 'x', host: 'h', port: 1 }, status: { state: 'thinking' }, latency_ms: null }
  const [node] = describeFabric(answered(fabricBody({ nodes: [odd] }))).nodes
  assert.equal(node.state, null)
  assert.equal(node.inFlight, null)
})

check('a missing count is unknown rather than zero', () => {
  const body = fabricBody({ nodes: [READY_NODE], counts: { ready: 1 } })
  const described = describeFabric(answered(body))
  assert.equal(described.counts.ready, 1)
  assert.equal(described.counts.total, null)
  assert.equal(described.counts.notReady, null)
})

/* ---- derived views ---- */

check('models are attributed to the nodes actually serving them', () => {
  const second = {
    spec: { label: 'mac', host: 'm', port: 8181 },
    status: { ...READY_NODE.status, active_model_id: 'Llama 3.2 1B Instruct', in_flight: 0, waiting: 0 },
    latency_ms: 9,
  }
  const third = {
    spec: { label: 'alpha', host: 'a', port: 8181 },
    status: { ...READY_NODE.status, active_model_id: 'Qwen3 4B', in_flight: 0, waiting: 0 },
    latency_ms: 9,
  }
  const described = describeFabric(answered(fabricBody({
    nodes: [READY_NODE, second, third],
    models: ['Llama 3.2 1B Instruct', 'Qwen3 4B'],
  })))
  assert.deepEqual(described.placements, [
    { model: 'Llama 3.2 1B Instruct', labels: ['mac', 'win'] },
    { model: 'Qwen3 4B', labels: ['alpha'] },
  ])
})

check('a model the proxy does not advertise is never listed as served', () => {
  // The proxy withholds models it would refuse to route -- an engine it reads
  // but does not place on. Re-deriving the list from the nodes would put those
  // models back and promise something the fabric would then refuse.
  const foreign = {
    spec: { label: 'studio', host: 's', port: 11434, engine: 'ollama' },
    status: {
      state: 'ready',
      engine: 'ollama',
      active_model_id: null,
      models: ['mistral:latest', 'qwen3:8b'],
      backend: null,
      version: '0.33.3',
    },
    latency_ms: 9,
    placeable: false,
  }
  const described = describeFabric(answered(fabricBody({
    nodes: [READY_NODE, foreign],
    models: ['Llama 3.2 1B Instruct'],
  })))
  assert.deepEqual(described.placements, [
    { model: 'Llama 3.2 1B Instruct', labels: ['win'] },
  ])
  const node = described.nodes.find((entry) => entry.label === 'studio')
  assert.deepEqual(node.models, ['mistral:latest', 'qwen3:8b'], 'the node still shows what it holds')
  assert.equal(node.engine, 'ollama')
  assert.equal(node.placeable, false)
  assert.equal(node.inFlight, null, 'an engine that publishes no load reports none')
})

check('a node with no model contributes no placement', () => {
  assert.deepEqual(modelPlacements([{ label: 'a', activeModelId: null }], []), [])
})

/* ---- capabilities: the answer and how we know it ---- */

check('a capability keeps the provenance the proxy attached to it', () => {
  const node = {
    spec: { label: 'studio', host: 's', port: 1234, engine: 'lmstudio' },
    status: { state: 'ready', engine: 'lmstudio', active_model_id: 'a', models: ['a'] },
    latency_ms: 4,
    placeable: false,
    capabilities: {
      load_reporting: { supported: false, provenance: 'declared', detail: 'no queue depth is published' },
      tool_calls: { supported: null, provenance: 'not_probed', detail: null },
      warm_prefix: { supported: true, provenance: 'measured', detail: 'measured on this build' },
    },
    placement_blockers: ['publishes no queue depth'],
  }
  const [described] = describeFabric(answered(fabricBody({ nodes: [node] }))).nodes
  const byName = Object.fromEntries(described.capabilities.map((c) => [c.name, c]))

  assert.equal(byName.load_reporting.supported, false)
  assert.equal(byName.load_reporting.provenance, 'declared')
  assert.equal(byName.load_reporting.detail, 'no queue depth is published')

  assert.equal(byName.tool_calls.supported, null, 'unchecked must never render as a no')
  assert.equal(byName.tool_calls.provenance, 'not_probed')

  assert.equal(byName.warm_prefix.provenance, 'measured')
  assert.deepEqual(described.placementBlockers, ['publishes no queue depth'])
})

check('a capability the proxy did not send stays absent rather than assumed', () => {
  const [node] = describeFabric(answered(fabricBody({ nodes: [READY_NODE] }))).nodes
  assert.equal(node.capabilities, null, 'no capabilities block means unknown, not "none supported"')
  assert.equal(node.placementBlockers, null)
})

check('an unrecognised provenance is not passed off as a known one', () => {
  const node = {
    ...READY_NODE,
    capabilities: { tool_calls: { supported: true, provenance: 'vibes' } },
  }
  const [described] = describeFabric(answered(fabricBody({ nodes: [node] }))).nodes
  assert.equal(described.capabilities[0].provenance, null)
  assert.equal(described.capabilities[0].supported, true)
})

check('posture separates "not ready" from "we could not look"', () => {
  assert.equal(fabricPosture(describeFabric({ outcome: 'unreachable', detail: 'x' })), 'unknown')
  assert.equal(fabricPosture(describeFabric(answered(fabricBody({ nodes: [], ready: false }), 503))), 'not_ready')
  assert.equal(fabricPosture(describeFabric(answered(fabricBody({ nodes: [READY_NODE] })))), 'ready')
  assert.equal(
    fabricPosture(describeFabric(answered(fabricBody({ nodes: [READY_NODE, NOT_READY_NODE] })))),
    'degraded',
    'one node down out of two is degraded, not healthy',
  )
})

check('a withheld fabric that says it is serving is not called degraded', () => {
  // Counts are unknown there, so "ready < total" cannot be evaluated and must
  // not be guessed.
  assert.equal(fabricPosture(describeFabric(answered(fabricBody()))), 'ready')
})

/* ---- routing: every word about the proxy's behaviour is the proxy's ----
   Screen E renders what these return. The rule under test is that nothing
   here authors a claim: a missing field is unknown, and every sentence is one
   the fixture put there, even when it is a nonsense token. */

check('the routing mode is read from the proxy, and absent is unknown, never Camelid-only', () => {
  assert.equal(describePlacement(undefined).mixedEngines, null)
  assert.equal(describePlacement(null).flag, null)
  assert.equal(describePlacement({ mixed_engines: 'refused' }).mixedEngines, 'refused')
  assert.equal(describePlacement({ mixed_engines: 'allowed' }).mixedEngines, 'allowed')
  assert.equal(describePlacement({ mixed_engines: 'sometimes' }).mixedEngines, null, 'a mode this build does not know is unknown')
  assert.equal(routingCommand(describePlacement({ mixed_engines: 'refused' }), 'mixed'), null, 'no published flag, no command')
  const withheld = describeFabric(answered(fabricBody()))
  assert.equal(withheld.placement.mixedEngines, null, 'a proxy that withholds detail did not say how it routes')
  const older = describeFabric(answered(fabricBody({ nodes: [READY_NODE] })))
  assert.equal(older.placement.mixedEngines, null, 'a proxy that predates the field did not say either')
  assert.equal(older.nodes[0].placementBlockerDetail, null)
  assert.equal(older.nodes[0].requirementLimits, null)
})

check("a node's loaded models are unknown when the proxy did not say, and empty only when it said so", () => {
  const [unsaid] = describeFabric(answered(fabricBody({ nodes: [READY_NODE] }))).nodes
  assert.equal(unsaid.residentModels, null)
  const none = { ...READY_NODE, status: { ...READY_NODE.status, resident_models: [] } }
  assert.deepEqual(describeFabric(answered(fabricBody({ nodes: [none] }))).nodes[0].residentModels, [])
  const gone = { ...NOT_READY_NODE, status: { ...NOT_READY_NODE.status, resident_models: ['x'] } }
  assert.equal(describeFabric(answered(fabricBody({ nodes: [gone] }))).nodes[0].residentModels, null)
})

check("what mixed mode accepts is the proxy's own words", () => {
  const first = {
    spec: { label: 'n1', host: 'h', port: 1, engine: 'zz-engine' },
    status: { state: 'ready', engine: 'zz-engine', version: 'zz-9', models: ['m'] },
    placeable: false,
    capabilities: { 'zz-key-1': { supported: false, provenance: 'declared', detail: 'zz-detail-1' } },
    placement_blockers: ['zz-blocker-1', 'zz-blocker-2'],
    placement_blocker_detail: [
      { key: 'zz-key-1', blocker: 'zz-blocker-1', consequence: 'zz-consequence-1' },
      { key: 'zz-key-2', blocker: 'zz-blocker-2', consequence: 'zz-consequence-2' },
    ],
  }
  const second = {
    ...first,
    spec: { ...first.spec, label: 'n2' },
    placement_blocker_detail: [{ key: 'zz-key-1', blocker: 'zz-blocker-1', consequence: 'zz-consequence-1' }],
  }
  const nodes = describeFabric(answered(fabricBody({ nodes: [first, second] }))).nodes
  const groups = mixedModeAcceptance(nodes)
  assert.deepEqual(groups.map((group) => [group.key, group.blocker, group.consequence]), [
    ['zz-key-1', 'zz-blocker-1', 'zz-consequence-1'],
    ['zz-key-2', 'zz-blocker-2', 'zz-consequence-2'],
  ])
  assert.deepEqual(groups[0].nodes.map((node) => node.label), ['n1', 'n2'])
  assert.equal(groups[0].nodes[0].detail, 'zz-detail-1', "the node's own capability detail travels with it")
  assert.equal(groups[0].nodes[0].provenance, 'declared')
  assert.equal(groups[0].nodes[0].version, 'zz-9')
  assert.equal(groups[1].nodes[0].detail, null, 'no capability entry for that key, no invented detail')

  // A backend this build has never heard of, with one reason only, is one
  // section: the grouping is on what the proxy said, not on an engine name.
  const fourth = {
    spec: { label: 'x', host: 'h', port: 2, engine: 'zz-fourth-engine' },
    status: { state: 'ready', engine: 'zz-fourth-engine', models: ['m'] },
    placeable: true,
    placement_blockers: ['cannot attest a warm prefix'],
    placement_blocker_detail: [{ key: 'warm_prefix', blocker: 'cannot attest a warm prefix', consequence: 'zz-pinned' }],
  }
  const lone = mixedModeAcceptance(describeFabric(answered(fabricBody({ nodes: [fourth] }))).nodes)
  assert.equal(lone.length, 1)
  assert.equal(lone[0].consequence, 'zz-pinned')
  assert.deepEqual(mixedModeAcceptance(describeFabric(answered(fabricBody({ nodes: [READY_NODE] }))).nodes), [])

  const partial = { ...first, placement_blocker_detail: [{ key: 'zz-key-1', blocker: 'zz-blocker-1' }] }
  assert.deepEqual(
    describeFabric(answered(fabricBody({ nodes: [partial] }))).nodes[0].placementBlockerDetail,
    [],
    'an entry missing its consequence is dropped, never rendered half-said',
  )
})

check("tool limits come from the proxy's requirement_limits, not the engine name", () => {
  const foreignWithout = {
    spec: { label: 'studio', host: 's', port: 11434, engine: 'ollama' },
    status: { state: 'ready', engine: 'ollama', models: ['m'] },
    requirement_limits: [{ key: 'rerank_route', consequence: 'zz-no-route' }],
  }
  const oursWith = {
    ...READY_NODE,
    spec: { ...READY_NODE.spec, engine: 'camelid' },
    requirement_limits: [{ key: 'tool_calls', consequence: 'zz-tools-never' }],
  }
  const groups = requirementLimits(describeFabric(answered(fabricBody({ nodes: [foreignWithout, oursWith] }))).nodes)
  const tools = groups.find((group) => group.key === 'tool_calls')
  assert.deepEqual(tools.nodes.map((node) => [node.label, node.consequence]), [['win', 'zz-tools-never']])
  const rerank = groups.find((group) => group.key === 'rerank_route')
  assert.deepEqual(rerank.nodes.map((node) => node.label), ['studio'])
})

check('the command uses the flag the proxy published', () => {
  const placement = describePlacement({ mixed_engines: 'refused', flag: '--zz-flag' })
  const mixed = routingCommand(placement, 'mixed')
  assert.equal(mixed.split('--zz-flag').length - 1, 1, mixed)
  assert.ok(!routingCommand(placement, 'camelid_only').includes('--zz-flag'))
  assert.ok(!mixed.includes('allow-mixed-engines'), 'no spelling of its own')
  assert.equal(provenanceLabel('not_probed'), 'not checked')
  assert.equal(provenanceLabel('zz-way'), 'zz-way', 'an unknown provenance is shown as sent')
  assert.equal(provenanceLabel(null), null)
})

/* ---- endpoint handling ---- */

check('an address is accepted in the forms an operator would type', () => {
  assert.equal(normalizeEndpoint('127.0.0.1:8282'), 'http://127.0.0.1:8282')
  assert.equal(normalizeEndpoint('  127.0.0.1:8282/  '), 'http://127.0.0.1:8282')
  assert.equal(normalizeEndpoint('http://localhost:8282'), 'http://localhost:8282')
  assert.equal(normalizeEndpoint('localhost'), 'http://localhost')
  assert.equal(normalizeEndpoint('[::1]:8282'), 'http://[::1]:8282', 'an IPv6 literal keeps its brackets')
})

check('a port is only dropped when it is the scheme default', () => {
  // `https://host:443` and `https://host` are the same origin, so losing the
  // port there changes nothing. A port that is not the default must survive,
  // because a fabric proxy rarely sits on one.
  assert.equal(normalizeEndpoint('https://fabric.example:443'), 'https://fabric.example')
  assert.equal(normalizeEndpoint('https://fabric.example:8282'), 'https://fabric.example:8282')
  assert.equal(normalizeEndpoint('fabric.example:443'), 'http://fabric.example:443')
})

check('an unusable address is refused rather than fetched as a guess', () => {
  assert.equal(normalizeEndpoint(''), null)
  assert.equal(normalizeEndpoint('   '), null)
  assert.equal(normalizeEndpoint(null), null)
  // A bare scheme names no host. Trimming trailing slashes used to turn this
  // into `http:`, which then parsed as the host `http` and was fetched.
  assert.equal(normalizeEndpoint('http://'), null)
  assert.equal(normalizeEndpoint('http:'), null)
  assert.equal(normalizeEndpoint('https://'), null)
})

check('the displayed address drops only the scheme', () => {
  assert.equal(endpointLabel('http://127.0.0.1:8282'), '127.0.0.1:8282')
  assert.equal(endpointLabel(null), null)
})

/* ---- a proxy that does not allow this page's origin ----
   `camelid fabric serve` sends no CORS headers unless started with
   `--cors-origin`, so this is the default outcome for a page on another origin,
   not an edge case. */

const PAGE = 'http://127.0.0.1:8181'
const PROXY = 'http://127.0.0.1:8282'

check('a read the browser blocked is named as such, with the exact flag that fixes it', () => {
  const blocked = describeFabric({
    outcome: 'blocked', detail: 'The browser did not let this page read the answer.', cause: 'cross_origin',
  })
  assert.equal(blocked.problem.code, 'origin_not_allowed')
  assert.equal(blocked.nodes, null, 'an answer the page could not read is not an empty fabric')
  assert.match(fabricProblemMessage(blocked.problem, '127.0.0.1:8282'), /Something answered at 127\.0\.0\.1:8282/)
  assert.equal(crossOriginDiagnosis(blocked.problem, PAGE, PROXY), 'blocked')
  assert.equal(corsCommand(PAGE), 'camelid fabric serve --cors-origin http://127.0.0.1:8181')
})

check('a plain network failure may be a CORS refusal; a timeout, a bad address or a same-origin read cannot be', () => {
  const network = describeFabric({ outcome: 'unreachable', detail: 'The connection failed.', cause: 'network' }).problem
  assert.equal(network.cause, 'network')
  assert.equal(crossOriginDiagnosis(network, PAGE, PROXY), 'possible')
  const timeout = describeFabric({ outcome: 'unreachable', detail: 'No answer within 4000ms.', cause: 'timeout' }).problem
  assert.equal(crossOriginDiagnosis(timeout, PAGE, PROXY), null, 'a timeout got past CORS to wait')
  const address = describeFabric({ outcome: 'unreachable', detail: 'That is not a usable address.', cause: 'address' }).problem
  assert.equal(crossOriginDiagnosis(address, PAGE, PROXY), null)
  assert.equal(crossOriginDiagnosis(network, PROXY, PROXY), null, 'CORS does not govern a same-origin read')
  assert.equal(crossOriginDiagnosis(null, PAGE, PROXY), null)
})

await checkAsync('an opaque follow-up separates "answered, but not to this page" from "nothing there"', async () => {
  const modes = []
  const blocked = await withFetch((url, init = {}) => {
    modes.push(init.mode)
    // CORS does not govern an opaque request, so a live server answers it.
    return init.mode === 'no-cors' ? Promise.resolve(new Response(null, { status: 200 })) : refuse()
  }, () => probeFabric({ endpoint: '127.0.0.1:8282' }))
  assert.deepEqual(modes, ['cors', 'no-cors'])
  assert.equal(blocked.outcome, 'blocked')

  const dead = await withFetch(refuse, () => probeFabric({ endpoint: '127.0.0.1:8282' }))
  assert.equal(dead.outcome, 'unreachable')
  assert.equal(dead.cause, 'network', 'still possibly CORS, so the view keeps offering the flag')
})

await checkAsync('the shared node cache is cleared by any read that did not disclose, but not by a cancelled one', async () => {
  const disclosed = () => Promise.resolve(new Response(JSON.stringify({
    ok: true, service: 'camelid-fabric', ready: true,
    nodes: { total: 1, ready: 1, not_ready: 0, unreachable: 0 }, models: [], node_detail: [READY_NODE],
  }), { status: 200 }))
  const withheld = () => Promise.resolve(new Response(JSON.stringify({
    ok: true, service: 'camelid-fabric', ready: true,
  }), { status: 200 }))

  clearCachedFabricNodes()
  await withFetch(disclosed, () => readFabric({ endpoint: '127.0.0.1:8282' }))
  assert.equal(readCachedFabricNodes().length, 1)

  // The Observatory draws from this cache. A node kept after the read that
  // should have refreshed it failed is drawn as present when nobody knows.
  await withFetch(refuse, () => readFabric({ endpoint: '127.0.0.1:8282' }))
  assert.deepEqual(readCachedFabricNodes(), [], 'a failed read clears the nodes it can no longer vouch for')

  await withFetch(disclosed, () => readFabric({ endpoint: '127.0.0.1:8282' }))
  await withFetch(withheld, () => readFabric({ endpoint: '127.0.0.1:8282' }))
  assert.deepEqual(readCachedFabricNodes(), [], 'a withheld list is not the old list')

  await withFetch(disclosed, () => readFabric({ endpoint: '127.0.0.1:8282' }))
  const cancelled = new AbortController()
  cancelled.abort()
  await withFetch(refuse, () => readFabric({ endpoint: '127.0.0.1:8282', signal: cancelled.signal }))
  assert.equal(readCachedFabricNodes().length, 1, 'a superseded read says nothing about the fabric')

  clearCachedFabricNodes()
  assert.deepEqual(readCachedFabricNodes(), [], 'changing the proxy address clears it (useFabric.setEndpoint)')
})

console.log(`\nfabric model smoke: ${checks} checks passed`)
