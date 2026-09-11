import { useEffect, useRef, useState } from 'react'
import { loadFabricEndpoint, normalizeEndpoint } from '../lib/fabricClient.js'
import { requestComparison } from '../lib/divergenceClient.js'

const IDLE = { phase: 'idle', comparison: null, problem: null, requested: null }

/* Ask the proxy to compare two nodes.
 *
 * Deliberately not on the 5s poll the Cluster view uses: this request makes two
 * machines generate, so it happens when a person asks for it and at no other
 * time. Leaving the page aborts it, so a slow comparison does not keep a
 * connection open for a screen nobody is looking at. */
export function useDivergence(endpointInput) {
  const [state, setState] = useState(IDLE)
  const inFlight = useRef(null)

  useEffect(() => () => inFlight.current?.abort(), [])

  async function run(request, { clientKey = '' } = {}) {
    const base = normalizeEndpoint(endpointInput || loadFabricEndpoint())
    if (!base) {
      setState({ ...IDLE, phase: 'failed', problem: { code: 'bad_endpoint' } })
      return
    }

    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    setState({ phase: 'running', comparison: null, problem: null, requested: request })

    const outcome = await requestComparison({ base, request, clientKey, signal: controller.signal })
    // Unmounted or superseded: nobody is waiting for this answer.
    if (controller.signal.aborted) return
    inFlight.current = null

    setState(outcome.comparison
      ? { phase: 'settled', comparison: outcome.comparison, problem: null, requested: request }
      : { phase: 'failed', comparison: null, problem: outcome.problem, requested: request })
  }

  return { ...state, run }
}
