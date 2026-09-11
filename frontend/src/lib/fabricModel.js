/* Turns one raw answer from a fabric proxy's `/v1/health` into what the Cluster
   view renders. Pure on purpose: every rule below is a claim about the operator's
   machines, and a claim deserves a unit test rather than a browser and a guess.

   The rule that shapes the whole file: **absent is not zero, and withheld is not
   empty.** `fabric serve` only discloses `nodes`, `models` and `node_detail` when
   it is bound to loopback, so a proxy on a routable address answers with the
   summary fields alone. Rendering that as "0 nodes" would tell an operator their
   fabric is empty when we simply were not allowed to look. Likewise a node that
   is not ready reports no load at all, and showing 0 in-flight would be inventing
   an idle machine. Anything we were not told stays null all the way to the DOM. */

export const FABRIC_SERVICE = 'camelid-fabric'

/** Node states the proxy can report, matching `NodeStatus`'s serde tag. */
export const NODE_STATES = ['ready', 'not_ready', 'unreachable']

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A number we were actually told. Anything else is null, never 0. */
function numberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function boolOrNull(value) {
  return typeof value === 'boolean' ? value : null
}

/** What kind of process answered, decided on fields only one of them has.
   The proxy carries `service` and no `engine`; the engine carries `engine` and
   no `service`. `fabric serve` shaped its payload that way deliberately so a
   proxy can never be mistaken for something that generates — this reads that
   distinction back, so pointing the view at a node says so instead of showing
   an empty fabric. */
export function classifyHealthBody(body) {
  if (!isPlainObject(body)) return 'foreign'
  if (body.service === FABRIC_SERVICE) return 'fabric'
  if (typeof body.engine === 'string' && body.service === undefined) return 'engine'
  return 'foreign'
}

/** Provenance values the proxy can attach to a capability answer. */
export const PROVENANCE = ['measured', 'declared', 'not_probed']

/* An answer plus how the proxy came by it. `supported` is null exactly when the
   provenance is `not_probed`, and the two must stay distinguishable: an
   unmeasured backend is not a broken one. */
function describeCapabilities(raw) {
  if (!isPlainObject(raw)) return null
  const entries = Object.entries(raw)
    .map(([name, value]) => {
      if (!isPlainObject(value)) return null
      const provenance = PROVENANCE.includes(value.provenance) ? value.provenance : null
      return {
        name,
        supported: boolOrNull(value.supported),
        provenance,
        detail: stringOrNull(value.detail),
      }
    })
    .filter(Boolean)
  return entries.length > 0 ? entries : null
}

function describeNode(entry) {
  const spec = isPlainObject(entry?.spec) ? entry.spec : {}
  const status = isPlainObject(entry?.status) ? entry.status : {}
  const state = NODE_STATES.includes(status.state) ? status.state : null
  const label = stringOrNull(spec.label)
  const host = stringOrNull(spec.host)
  const port = numberOrNull(spec.port)

  return {
    label,
    host,
    port,
    authority: host === null ? null : (port === null ? host : `${host}:${port}`),
    // The engine is declared on the spec, so it is known even for a node that
    // never answered. `status.engine` only exists once one has.
    engine: stringOrNull(spec.engine),
    // Whether this fabric would place work here. A node can be perfectly
    // healthy and still not be somewhere work goes, and only the proxy knows
    // which engines it places on — so this is never derived client-side.
    placeable: boolOrNull(entry?.placeable),
    capabilities: describeCapabilities(entry?.capabilities),
    placementBlockers: Array.isArray(entry?.placement_blockers)
      ? entry.placement_blockers.filter((blocker) => typeof blocker === 'string')
      : null,
    state,
    // Only the failing states carry a reason; a ready node has nothing to explain.
    reason: state === 'ready' ? null : stringOrNull(status.reason),
    activeModelId: state === 'ready' ? stringOrNull(status.active_model_id) : null,
    models: state === 'ready' && Array.isArray(status.models)
      ? status.models.filter((model) => typeof model === 'string')
      : null,
    backend: state === 'ready' ? stringOrNull(status.backend) : null,
    version: state === 'ready' ? stringOrNull(status.version) : null,
    inFlight: state === 'ready' ? numberOrNull(status.in_flight) : null,
    waiting: state === 'ready' ? numberOrNull(status.waiting) : null,
    latencyMs: numberOrNull(entry?.latency_ms),
  }
}

/** Which nodes hold each model the fabric says it will serve.

   `servable` is the proxy's own list, and it is the only thing that decides
   what appears here. A node may hold models the fabric will not route to — an
   engine it reads but does not place on — and advertising those would promise
   something the fabric would then refuse. Which nodes hold a model is read from
   the nodes; *whether it is served at all* is never re-derived client-side. */
export function modelPlacements(nodes, servable) {
  if (!Array.isArray(nodes) || !Array.isArray(servable)) return []
  const holders = new Map()
  for (const node of nodes) {
    const models = node.models && node.models.length
      ? node.models
      : (node.activeModelId ? [node.activeModelId] : [])
    for (const model of models) {
      if (!node.label) continue
      const labels = holders.get(model) || []
      labels.push(node.label)
      holders.set(model, labels)
    }
  }
  return servable
    .map((model) => ({ model, labels: (holders.get(model) || []).slice().sort() }))
    .sort((a, b) => a.model.localeCompare(b.model))
}

const UNKNOWN_FABRIC = {
  answered: false,
  kind: null,
  problem: null,
  service: null,
  version: null,
  build: null,
  ready: null,
  detail: 'unknown',
  counts: null,
  models: null,
  nodes: null,
  placements: null,
}

/**
 * Describe a fabric from one probe result.
 *
 * `probe` is `{ outcome: 'answered', httpStatus, body }`, or
 * `{ outcome: 'unreachable' | 'malformed', detail }`.
 */
export function describeFabric(probe) {
  if (!isPlainObject(probe)) {
    return { ...UNKNOWN_FABRIC, problem: { code: 'no_probe', detail: null } }
  }

  if (probe.outcome !== 'answered') {
    const code = probe.outcome === 'malformed'
      ? 'malformed_answer'
      : (probe.outcome === 'blocked' ? 'origin_not_allowed' : 'unreachable')
    // `cause` separates a network failure, which may be a CORS refusal in
    // disguise, from a timeout or a bad address, which cannot be.
    const cause = stringOrNull(probe.cause)
    return {
      ...UNKNOWN_FABRIC,
      problem: { code, detail: stringOrNull(probe.detail), ...(cause ? { cause } : {}) },
    }
  }

  const body = probe.body
  const kind = classifyHealthBody(body)
  if (kind !== 'fabric') {
    return {
      ...UNKNOWN_FABRIC,
      answered: true,
      kind,
      problem: { code: kind === 'engine' ? 'is_an_engine' : 'not_a_fabric', detail: null },
    }
  }

  // A 503 from a reachable proxy is how it says no node is ready. It is an
  // answer, not a failure, so `ready` below carries it and it must not be
  // folded in with "could not look".
  const summary = isPlainObject(body.nodes) ? body.nodes : null
  const disclosed = Array.isArray(body.node_detail)
  const nodes = disclosed ? body.node_detail.map(describeNode) : null
  const models = Array.isArray(body.models)
    ? body.models.filter((model) => typeof model === 'string')
    : null

  return {
    answered: true,
    kind: 'fabric',
    problem: null,
    service: stringOrNull(body.service),
    version: stringOrNull(body.version),
    build: stringOrNull(body.build),
    ready: boolOrNull(body.ready),
    detail: disclosed ? 'disclosed' : 'withheld',
    counts: summary
      ? {
        total: numberOrNull(summary.total),
        ready: numberOrNull(summary.ready),
        notReady: numberOrNull(summary.not_ready),
        unreachable: numberOrNull(summary.unreachable),
      }
      : null,
    models,
    nodes,
    placements: nodes ? modelPlacements(nodes, models) : null,
  }
}

/** One line an operator can act on. Never invents a cause. */
export function fabricProblemMessage(problem, endpoint) {
  if (!problem) return null
  const where = endpoint ? ` at ${endpoint}` : ''
  switch (problem.code) {
    case 'unreachable':
      return `No fabric proxy answered${where}.${problem.detail ? ` ${problem.detail}` : ''}`
    case 'malformed_answer':
      return `Something answered${where}, but not with a health report we could read.`
    case 'origin_not_allowed':
      return `Something answered${where}, but the browser did not let this page read the answer.`
    case 'is_an_engine':
      return `That address${where} is a Camelid engine, not a fabric proxy. Point this at a \`camelid fabric serve\` address, or add this engine to a fabric as a node.`
    case 'not_a_fabric':
      return `Something answered${where}, but it is not a Camelid fabric proxy.`
    default:
      return `Could not read the fabric${where}.`
  }
}

/** The exact command that lets a page on `pageOrigin` read a fabric proxy.
   `--cors-origin` is off by default, so a proxy started without it answers no
   page but its own origin, and the WebUI is never served from the proxy. */
export function corsCommand(pageOrigin) {
  return `camelid fabric serve --cors-origin ${pageOrigin}`
}

/** Whether a failed read may be the proxy refusing this page's origin.

   `blocked` when an opaque follow-up proved something answered. `possible` on a
   plain network failure, because the browser reports a CORS refusal and a dead
   socket identically and saying only "nothing answered" would be a guess.
   Never for a timeout or a bad address, and never for a same-origin read, which
   CORS does not govern. */
export function crossOriginDiagnosis(problem, pageOrigin, proxyOrigin) {
  if (!problem || !pageOrigin) return null
  if (problem.code === 'origin_not_allowed') return 'blocked'
  if (proxyOrigin && pageOrigin === proxyOrigin) return null
  if (problem.code === 'unreachable' && problem.cause === 'network') return 'possible'
  return null
}

/** Why node detail is missing, when it is. Only ever called for a real fabric. */
export const DETAIL_WITHHELD_REASON =
  'This proxy is not bound to loopback, so it withholds node addresses and model names. Open this page on the machine running the proxy to see them.'

/** Overall state for the header chip. Kept separate from `ready` because a
   fabric we could not reach has no readiness, and "not ready" would be a claim. */
export function fabricPosture(fabric) {
  if (!fabric || fabric.problem) return 'unknown'
  if (fabric.ready === null) return 'unknown'
  if (!fabric.ready) return 'not_ready'
  const counts = fabric.counts
  if (counts && counts.total !== null && counts.ready !== null && counts.ready < counts.total) {
    return 'degraded'
  }
  return 'ready'
}
