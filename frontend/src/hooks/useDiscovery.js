import { useCallback, useEffect, useRef, useState } from 'react'
import { getPolicy, join as sendJoin, runScan } from '../lib/discoveryClient.js'

/* Owns one proxy's discovery state, in memory only.
 *
 * Three deliberate absences, each of which was a defect in the view this lane
 * replaced:
 *
 *   - **It never scans on mount, and never polls.** A scan sends traffic to
 *     machines nobody named, so it happens when a person clicks and at no other
 *     time.
 *   - **Nothing is persisted.** Findings describe what answered moments ago; a
 *     stored one would let a reloaded page assert that a machine is there.
 *   - **A join is never assumed to have taken effect.** The row waits until the
 *     proxy's own health lists the label, because the write only matters if the
 *     process actually re-read the file.
 */

const IDLE = { phase: 'idle', discovery: null, problem: null }

export function useDiscovery(base) {
  const [policy, setPolicy] = useState(null)
  const [policyProblem, setPolicyProblem] = useState(null)
  const [scan, setScan] = useState(IDLE)
  const [joins, setJoins] = useState({})
  const scanning = useRef(null)
  const mounted = useRef(true)

  useEffect(() => () => {
    mounted.current = false
    scanning.current?.abort()
  }, [])

  /* Read-only, and the only request made without a click: it is what tells the
     panel whether there is anything to offer at all. */
  const loadPolicy = useCallback(async ({ clientKey = '' } = {}) => {
    if (!base) return null
    const outcome = await getPolicy({ base, clientKey })
    if (!mounted.current) return null
    setPolicy(outcome.policy ?? null)
    setPolicyProblem(outcome.problem ?? null)
    return outcome
  }, [base])

  const start = useCallback(async (scope, { clientKey = '' } = {}) => {
    if (!base) return
    scanning.current?.abort()
    const controller = new AbortController()
    scanning.current = controller
    setScan({ phase: 'scanning', discovery: null, problem: null })

    const outcome = await runScan({ base, scope, clientKey, signal: controller.signal })
    // Unmounted, superseded, or cancelled: nobody is waiting for this.
    if (!mounted.current || controller.signal.aborted) return
    scanning.current = null
    setScan(outcome.discovery
      ? { phase: 'settled', discovery: outcome.discovery, problem: null }
      : { phase: 'failed', discovery: null, problem: outcome.problem })
  }, [base])

  /* Stops the scan at the proxy too: dropping the request is what the server
     reads as "nobody wants this any more". */
  const cancel = useCallback(() => {
    scanning.current?.abort()
    scanning.current = null
    setScan(IDLE)
  }, [])

  const confirm = useCallback(async (id, request, { clientKey = '' } = {}) => {
    if (!base) return null
    setJoins((current) => ({ ...current, [id]: { phase: 'writing', joined: null, problem: null, polls: 0 } }))
    const outcome = await sendJoin({ base, request, clientKey })
    if (!mounted.current) return null
    setJoins((current) => ({
      ...current,
      [id]: outcome.joined
        // Written, but not yet drawn as a node: that waits on the proxy.
        ? { phase: 'waiting', joined: outcome.joined, problem: null, polls: 0 }
        : { phase: 'failed', joined: null, problem: outcome.problem, polls: 0 },
    }))
    return outcome
  }, [base])

  /* Settle a written row only when the proxy's own node list names it. After a
     few polls without it, say so and point at the proxy rather than pretending
     it landed. */
  const settle = useCallback((labels) => {
    setJoins((current) => {
      let changed = false
      const next = {}
      for (const [id, state] of Object.entries(current)) {
        if (state.phase !== 'waiting') {
          next[id] = state
          continue
        }
        const label = state.joined?.line?.split('=')[0]
        if (label && labels.includes(label)) {
          next[id] = { ...state, phase: 'reported' }
          changed = true
        } else if (state.polls >= 3) {
          next[id] = { ...state, phase: 'unreported' }
          changed = true
        } else {
          next[id] = { ...state, polls: state.polls + 1 }
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [])

  const reset = useCallback(() => {
    setScan(IDLE)
    setJoins({})
  }, [])

  return { policy, policyProblem, loadPolicy, scan, start, cancel, joins, confirm, settle, reset }
}

export default useDiscovery
