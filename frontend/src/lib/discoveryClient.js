/* The three requests the "Find machines" panel makes.
 *
 * Kept out of the hook so every way each one can fail is provable without a
 * browser. Every outcome is either a value or a *named* problem, and never a
 * rejection: "we could not look" must never reach the screen as "nothing is
 * there".
 *
 * One distinction this file exists to keep: a 404 carrying
 * `code: "discovery_disabled"` is a proxy that will not do this, while a 404
 * with no code at all is the proxy's ordinary unknown-route answer — a build
 * from before discovery existed. They need different advice, so they are
 * different problems.
 */
import { describeDiscovery, describeJoin, describePolicy } from './discoveryModel.js'

/* A scan is bounded by the proxy's own 60s wall clock. This is only here so a
   proxy that stops answering cannot leave the page waiting for ever. */
export const SCAN_TIMEOUT_MS = 90 * 1000
const READ_TIMEOUT_MS = 10 * 1000

function headersFor(clientKey, json) {
  const headers = {}
  if (json) headers['content-type'] = 'application/json'
  const key = String(clientKey || '').trim()
  if (key) headers.authorization = `Bearer ${key}`
  return headers
}

/** Read one response into `{ status, body }`, tolerating a non-JSON body. */
async function readBody(response) {
  try {
    return { status: response.status, body: await response.json() }
  } catch {
    return { status: response.status, body: null }
  }
}

/* The proxy's coded refusals, mapped to what the page branches on. The page
   reads `code`, never the message text: prose is for people. */
function problemFor(status, body, clientKey) {
  if (status === 401) {
    return { code: String(clientKey || '').trim() ? 'key_refused' : 'key_required' }
  }
  const code = body?.error?.code
  const detail = body?.error?.message || null
  if (typeof code === 'string' && code.length > 0) {
    /* The three guards answer with their own codes; strip the prefix the
       routes use so the model's table stays readable. */
    const trimmed = code.replace(/^discovery_/, '')
    const known = ['loopback_only', 'host_not_loopback', 'origin_not_allowed'].includes(trimmed)
    return { code: known ? trimmed : code, detail }
  }
  /* A 404 with no code at all is the proxy's ordinary unknown-route body,
     which means this build predates discovery. */
  if (status === 404) return { code: 'old_build', detail }
  return { code: 'refused', detail: detail || `The proxy answered ${status}.` }
}

async function send({ base, path, method, payload, clientKey, timeoutMs, signal, fetchImpl }) {
  const fetcher = fetchImpl || globalThis.fetch
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const onAbort = () => controller.abort()
  if (signal?.aborted) controller.abort()
  else signal?.addEventListener('abort', onAbort)

  /* Our own abort is not the proxy failing, and must not be reported as one. */
  const stopped = () => {
    if (signal?.aborted) return { code: 'cancelled' }
    if (timedOut) return { code: 'timeout', timeoutMs }
    return null
  }

  try {
    let response
    try {
      response = await fetcher(`${base}${path}`, {
        method,
        headers: headersFor(clientKey, payload !== undefined),
        body: payload === undefined ? undefined : JSON.stringify(payload),
        signal: controller.signal,
      })
    } catch (error) {
      return {
        problem: stopped() || {
          code: 'unreachable',
          /* From a browser this may be a CORS refusal, which is reported
             exactly like a dead socket. The view offers the fix on this cause. */
          cause: 'network',
          detail: String(error?.message || error),
        },
      }
    }

    const { status, body } = await readBody(response)
    if (!response.ok) return { problem: problemFor(status, body, clientKey) }
    if (body === null) {
      return { problem: { code: 'malformed', detail: `Answered ${status} with something that is not JSON.` } }
    }
    return { body }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/** What this proxy will scan, and what it will never send. */
export async function getPolicy({ base, clientKey = '', signal, fetchImpl } = {}) {
  const outcome = await send({
    base,
    path: '/v1/fabric/discover',
    method: 'GET',
    clientKey,
    timeoutMs: READ_TIMEOUT_MS,
    signal,
    fetchImpl,
  })
  if (outcome.problem) return outcome
  const policy = describePolicy(outcome.body)
  return policy ? { policy } : { problem: { code: 'malformed', detail: 'That is not a discovery policy.' } }
}

/** Look. Nothing about this adds anything. */
export async function runScan({ base, scope, clientKey = '', signal, fetchImpl } = {}) {
  const outcome = await send({
    base,
    path: '/v1/fabric/discover',
    method: 'POST',
    payload: scope,
    clientKey,
    timeoutMs: SCAN_TIMEOUT_MS,
    signal,
    fetchImpl,
  })
  if (outcome.problem) return outcome
  const discovery = describeDiscovery(outcome.body)
  return discovery
    ? { discovery }
    : { problem: { code: 'malformed', detail: 'That is not a set of findings.' } }
}

/** Add exactly one machine. The only call in this file with a side effect. */
export async function join({ base, request, clientKey = '', signal, fetchImpl } = {}) {
  const outcome = await send({
    base,
    path: '/v1/fabric/discover/join',
    method: 'POST',
    payload: request,
    clientKey,
    timeoutMs: READ_TIMEOUT_MS,
    signal,
    fetchImpl,
  })
  if (outcome.problem) return outcome
  const joined = describeJoin(outcome.body)
  return joined ? { joined } : { problem: { code: 'malformed', detail: 'That is not a completed write.' } }
}
