/* The one request the Compare screen makes, kept out of the hook so every way
 * it can fail is provable without a browser.
 *
 * Every outcome is either a comparison or a named problem. A failure is never
 * returned as an empty comparison, because "we never asked" must not read as
 * "no difference". */
import { describeComparison } from './divergenceModel.js'

/* Generous on purpose. Two nodes each generate every run, possibly serially on
 * a busy machine, and the proxy's own forward timeout decides when a node has
 * failed. This exists only so a hung proxy cannot leave the page waiting
 * forever. */
export const COMPARE_TIMEOUT_MS = 5 * 60 * 1000

/**
 * POST one comparison.
 *
 * Resolves to `{ comparison }` or `{ problem }`, never rejects. `clientKey` is
 * sent as a bearer token when non-empty; the caller owns where it lives.
 */
export async function requestComparison({
  base,
  request,
  clientKey = '',
  timeoutMs = COMPARE_TIMEOUT_MS,
  signal,
  fetchImpl = globalThis.fetch,
}) {
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const onAbort = () => controller.abort()
  if (signal?.aborted) controller.abort()
  else signal?.addEventListener('abort', onAbort)

  // Our own abort is not a network failure, and must not be reported as one.
  const stopped = () => {
    if (signal?.aborted) return { code: 'cancelled' }
    if (timedOut) return { code: 'timeout', timeoutMs }
    return null
  }

  const key = String(clientKey || '').trim()
  const headers = { 'content-type': 'application/json' }
  if (key) headers.authorization = `Bearer ${key}`

  try {
    let response
    try {
      response = await fetchImpl(`${base}/v1/fabric/compare`, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        signal: controller.signal,
      })
    } catch (error) {
      return {
        problem: stopped() || {
          code: 'unreachable',
          // A network failure, which from a browser may be a CORS refusal: the
          // view offers the `--cors-origin` fix on this cause.
          cause: 'network',
          detail: String(error?.message || error),
        },
      }
    }

    // Named before the body is read: the proxy's 401 is the engine's shared
    // refusal, and the only thing worth saying is whether a key was sent.
    if (response.status === 401) {
      return { problem: { code: key ? 'key_refused' : 'key_required' } }
    }

    let body
    try {
      body = await response.json()
    } catch {
      return {
        problem: stopped() || {
          code: 'malformed',
          detail: `Answered ${response.status} with something that is not JSON.`,
        },
      }
    }

    if (!response.ok) {
      return {
        problem: {
          code: 'refused',
          detail: body?.error?.message || body?.message || `The proxy answered ${response.status}.`,
        },
      }
    }

    const comparison = describeComparison(body)
    if (!comparison) {
      return { problem: { code: 'malformed', detail: `Answered ${response.status} with something that is not a comparison.` } }
    }
    return { comparison }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}
