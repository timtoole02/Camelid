import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DEFAULT_FABRIC_ENDPOINT,
  FABRIC_ENDPOINT_KEY,
  loadFabricEndpoint,
  normalizeEndpoint,
  readFabric,
} from '../lib/fabricClient.js'
import { appStorage } from '../lib/appStorage.js'

/* Owns one fabric proxy address and the polling around it.

   The address is stored because it is an operator *input* — which proxy to ask.
   Nothing about the fabric's state is stored: every status this hook exposes
   comes from the answer to the request it just made, so a stale browser can
   never assert that a machine is up. */

const ENDPOINT_KEY = FABRIC_ENDPOINT_KEY
const POLL_MS = 5000

const loadEndpoint = loadFabricEndpoint

export function useFabric({ pollMs = POLL_MS } = {}) {
  const [endpoint, setEndpointState] = useState(loadEndpoint)
  const [fabric, setFabric] = useState(null)
  // 'never' until the first answer lands, so the view can stay quiet instead of
  // claiming a proxy is unreachable before we have asked.
  const [phase, setPhase] = useState('never')
  const [checkedAt, setCheckedAt] = useState(null)
  const inFlight = useRef(null)
  const mounted = useRef(true)

  useEffect(() => () => { mounted.current = false }, [])

  const refresh = useCallback(async (target) => {
    const address = target ?? endpoint
    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    setPhase((prev) => (prev === 'never' ? 'first' : 'refreshing'))
    const next = await readFabric({ endpoint: address, signal: controller.signal })
    if (!mounted.current || controller.signal.aborted) return null
    setFabric(next)
    setPhase('settled')
    setCheckedAt(new Date().toISOString())
    return next
  }, [endpoint])

  const setEndpoint = useCallback((raw) => {
    const normalized = normalizeEndpoint(raw)
    const stored = normalized ? raw.trim() : raw
    setEndpointState(stored)
    try { appStorage.setItem(ENDPOINT_KEY, stored) } catch { /* storage is best-effort */ }
    // Drop the previous fabric immediately: it describes a different address.
    setFabric(null)
    setPhase('never')
    setCheckedAt(null)
    return normalized
  }, [])

  useEffect(() => {
    refresh()
    if (!pollMs) return undefined
    const timer = window.setInterval(() => { refresh() }, pollMs)
    return () => window.clearInterval(timer)
  }, [refresh, pollMs])

  useEffect(() => () => inFlight.current?.abort(), [])

  return {
    endpoint,
    setEndpoint,
    fabric,
    phase,
    checkedAt,
    refresh,
    valid: normalizeEndpoint(endpoint) !== null,
  }
}
