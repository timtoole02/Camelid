/* Reads a fabric proxy's `/v1/health` straight from the browser.

   The WebUI is served by an engine, not by the proxy, so this is a cross-origin
   request by construction. That works without a dev hook because Camelid answers
   with a permissive `access-control-allow-origin` — the same route the Cluster
   page already used for per-node telemetry.

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

let lastGoodNodes = null

/** The most recent disclosed node list, for surfaces that render fabric identity
   but do not own the fetch (the Observatory constellation). Empty until the
   Cluster page has successfully looked — which is the honest default, because
   until then we genuinely do not know of any nodes. */
export function readCachedFabricNodes() {
  return lastGoodNodes ? lastGoodNodes.slice() : []
}

export function clearCachedFabricNodes() {
  lastGoodNodes = null
}

/**
 * Probe one fabric proxy.
 *
 * Resolves to `{ outcome: 'answered', httpStatus, body }` or
 * `{ outcome: 'unreachable' | 'malformed', detail }` — it never rejects, because
 * every failure mode here is information the view has to render.
 */
export async function probeFabric({ endpoint, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = {}) {
  const origin = normalizeEndpoint(endpoint)
  if (!origin) return { outcome: 'unreachable', detail: 'That is not a usable address.' }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort)

  try {
    const response = await fetch(`${origin}/v1/health`, {
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
  } catch (error) {
    if (signal?.aborted) return { outcome: 'unreachable', detail: 'Cancelled.' }
    const timedOut = controller.signal.aborted
    return {
      outcome: 'unreachable',
      detail: timedOut ? `No answer within ${timeoutMs}ms.` : 'The connection failed.',
    }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/** Probe and describe in one step, keeping the node cache current. */
export async function readFabric(options) {
  const fabric = describeFabric(await probeFabric(options))
  if (fabric.detail === 'disclosed') lastGoodNodes = fabric.nodes
  return fabric
}
