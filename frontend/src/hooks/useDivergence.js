import { useState } from 'react'
import { loadFabricEndpoint, normalizeEndpoint } from '../lib/fabricClient.js'
import { describeComparison } from '../lib/divergenceModel.js'

/* Ask the proxy to compare two nodes.
 *
 * Deliberately not on the 5s poll the Cluster view uses: this request makes two
 * machines generate, so it happens when a person asks for it and at no other
 * time. */
export function useDivergence(endpointInput) {
  const [state, setState] = useState({ phase: 'idle', comparison: null, problem: null })

  async function run(request) {
    const base = normalizeEndpoint(endpointInput || loadFabricEndpoint())
    if (!base) {
      setState({ phase: 'failed', comparison: null, problem: { code: 'bad_endpoint' } })
      return
    }
    setState({ phase: 'running', comparison: null, problem: null })

    let response
    try {
      response = await fetch(`${base}/v1/fabric/compare`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(request),
      })
    } catch (error) {
      setState({
        phase: 'failed',
        comparison: null,
        problem: { code: 'unreachable', detail: String(error?.message || error) },
      })
      return
    }

    let body = null
    try {
      body = await response.json()
    } catch {
      setState({
        phase: 'failed',
        comparison: null,
        problem: { code: 'malformed', detail: `Answered ${response.status} with something that is not JSON.` },
      })
      return
    }

    if (!response.ok) {
      // The proxy refuses a comparison it could not set up or run. That is a
      // failure, and must never be rendered as a comparison that found nothing.
      setState({
        phase: 'failed',
        comparison: null,
        problem: {
          code: 'refused',
          detail: body?.error?.message || body?.message || `The proxy answered ${response.status}.`,
        },
      })
      return
    }

    setState({ phase: 'settled', comparison: describeComparison(body), problem: null })
  }

  return { ...state, run }
}
