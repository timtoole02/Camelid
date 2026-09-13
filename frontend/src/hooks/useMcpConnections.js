import { useCallback, useEffect, useState } from 'react'
import { mcpRequest } from '../lib/mcp.js'

export function useMcpConnections(apiBase) {
  const [connections, setConnections] = useState([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const refresh = useCallback(async (signal) => {
    try {
      const data = await mcpRequest(apiBase, '/connections', { signal })
      if (!signal?.aborted) { setConnections(data.connections || []); setError('') }
    } catch (e) { if (!signal?.aborted) { setConnections([]); setError(e.message) } }
  }, [apiBase])
  useEffect(() => {
    setConnections([])
    const controller = new AbortController()
    refresh(controller.signal)
    return () => controller.abort()
  }, [refresh])
  const mutate = async (path, options) => {
    setBusy(true); setError('')
    try { const result = await mcpRequest(apiBase, path, options); await refresh(); return result }
    catch (e) { setError(e.message); return null }
    finally { setBusy(false) }
  }
  return { connections, error, busy, refresh, mutate }
}
