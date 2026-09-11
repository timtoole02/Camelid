/* Reads a fabric proxy's `/v1/health` straight from the browser.

   The WebUI is served by an engine, not by the proxy, so this is a cross-origin
   request by construction, and `camelid fabric serve` sends no CORS headers
   unless it was started with `--cors-origin <ORIGIN>` naming this page's origin.
   Without that the browser keeps the answer from this page, and it reports the
   refusal exactly as it reports a dead socket. So a failed read is followed by
   one opaque request, which CORS does not govern: if that gets any answer back,
   something is listening and it is the origin rule that stopped us.

   A failed read is reported as a failure, never as an empty fabric: the caller
   needs to tell "no nodes" apart from "could not look". */

import { describeFabric } from './fabricModel.js'
import { appStorage } from './appStorage.js'

/** `camelid fabric serve --addr` default (src/main.rs, docs/CONFIGURATION.md). */
export const DEFAULT_FABRIC_ENDPOINT = '127.0.0.1:8282'

/* Which proxy to ask. Stored because it is an operator *input*, and defined
   once because two screens pointing at different proxies would report on
   different fabrics while looking like one product. */
export const FABRIC_ENDPOINT_KEY = 'camelid.fabricEndpoint'

export function loadFabricEndpoint() {
  try {
    return appStorage.getItem(FABRIC_ENDPOINT_KEY) || DEFAULT_FABRIC_ENDPOINT
  } catch {
    return DEFAULT_FABRIC_ENDPOINT
  }
}

const DEFAULT_TIMEOUT_MS = 4000

/** Accept what an operator would actually type and produce an origin.
   Returns null for input we cannot turn into one, so the caller can say so
   rather than fetching a guess.

   Any path is dropped by rebuilding from protocol/host/port rather than by
   trimming the text: trimming trailing slashes turns `http://` into `http:`,
   which then parses as the host `http`. */
export function normalizeEndpoint(raw) {
  const text = String(raw ?? '').trim()
  if (!text) return null
  // A scheme with nothing after it names no host.
  if (/^[a-z][a-z0-9+.-]*:$/i.test(text)) return null
  const withScheme = /^https?:\/\//i.test(text) ? text : `http://${text}`
  try {
    const url = new URL(withScheme)
    if (!url.hostname) return null
    return url.port ? `${url.protocol}//${url.hostname}:${url.port}` : `${url.protocol}//${url.hostname}`
  } catch {
    return null
  }
}

/** Display form: what the operator typed, minus the scheme noise. */
export function endpointLabel(origin) {
  if (!origin) return null
  return origin.replace(/^https?:\/\//i, '')
}

/* The node list from the most recent read that disclosed one. A read that did
   not — a failure, a withheld list — clears it, and so does pointing the page
   at another proxy: a node drawn from an older answer is a claim that the
   machine is still there. */
let lastGoodNodes = null

/** The most recent disclosed node list, for surfaces that render fabric identity
   but do not own the fetch (the Observatory constellation). Empty until a page
   has successfully looked, and again as soon as a look fails — which is the
   honest default, because then we genuinely do not know of any nodes. */
export function readCachedFabricNodes() {
  return lastGoodNodes ? lastGoodNodes.slice() : []
}

export function clearCachedFabricNodes() {
  lastGoodNodes = null
}

/* Whether anything at all answers `url`. An opaque request is not subject to
   CORS: it resolves wherever a server replied, whatever its headers said, and
   rejects only when no answer came back. Its response is unreadable by design
   and is never looked at. */
async function answersOpaquely(url, signal) {
  try {
    await fetch(url, { mode: 'no-cors', cache: 'no-store', signal })
    return true
  } catch {
    return false
  }
}

/**
 * Probe one fabric proxy.
 *
 * Resolves to `{ outcome: 'answered', httpStatus, body }`; to
 * `{ outcome: 'blocked', detail, cause: 'cross_origin' }` when something
 * answered but the browser would not let this page read it; or to
 * `{ outcome: 'unreachable' | 'malformed', detail, cause }`. It never rejects,
 * because every failure mode here is information the view has to render.
 */
export async function probeFabric({ endpoint, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = {}) {
  const origin = normalizeEndpoint(endpoint)
  if (!origin) return { outcome: 'unreachable', detail: 'That is not a usable address.', cause: 'address' }

  const url = `${origin}/v1/health`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const onAbort = () => controller.abort()
  if (signal?.aborted) controller.abort()
  else signal?.addEventListener('abort', onAbort)

  // Our own abort, not the network: either the caller gave up or we timed out.
  const stopped = () => {
    if (signal?.aborted) return { outcome: 'unreachable', detail: 'Cancelled.', cause: 'cancelled' }
    if (controller.signal.aborted) {
      return { outcome: 'unreachable', detail: `No answer within ${timeoutMs}ms.`, cause: 'timeout' }
    }
    return null
  }

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      mode: 'cors',
      cache: 'no-store',
    })
    const text = await response.text()
    let body
    try {
      body = JSON.parse(text)
    } catch {
      return { outcome: 'malformed', detail: `Answered ${response.status} with something that is not JSON.` }
    }
    return { outcome: 'answered', httpStatus: response.status, body }
  } catch {
    const early = stopped()
    if (early) return early
    if (await answersOpaquely(url, controller.signal)) {
      return {
        outcome: 'blocked',
        detail: 'The browser did not let this page read the answer.',
        cause: 'cross_origin',
      }
    }
    return stopped() || { outcome: 'unreachable', detail: 'The connection failed.', cause: 'network' }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/** Probe and describe in one step, keeping the node cache current. */
export async function readFabric(options = {}) {
  const fabric = describeFabric(await probeFabric(options))
  // A cancelled read — superseded, or its page closed — says nothing about the
  // fabric, so it neither refreshes nor clears what the last read saw.
  if (!options.signal?.aborted) {
    lastGoodNodes = fabric.detail === 'disclosed' ? fabric.nodes : null
  }
  return fabric
}
