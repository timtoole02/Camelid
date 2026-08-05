import { useCallback, useEffect, useState } from 'react'

const REFRESH_INTERVAL_MS = 5000

export function useRuntimeMemory(apiBase = '') {
  const base = String(apiBase || '').replace(/\/$/, '')
  const [memory, setMemory] = useState(null)
  const [loading, setLoading] = useState(true)
  const [purging, setPurging] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true)
    try {
      const response = await fetch(`${base}/api/runtime/memory`)
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body?.error?.message || `memory request failed (HTTP ${response.status})`)
      setMemory(body)
      setError('')
      return body
    } catch (err) {
      setError(String(err?.message || err))
      return null
    } finally {
      if (!quiet) setLoading(false)
    }
  }, [base])

  const purge = useCallback(async () => {
    if (purging) return
    setPurging(true)
    setNotice('')
    setError('')
    try {
      const response = await fetch(`${base}/api/runtime/kv-cache/purge`, { method: 'POST' })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body?.error?.message || `KV-cache purge failed (HTTP ${response.status})`)
      setNotice(body.purged_entries
        ? `Purged ${body.purged_entries} KV-cache ${body.purged_entries === 1 ? 'entry' : 'entries'}.`
        : 'KV cache was already empty.')
      await refresh({ quiet: true })
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setPurging(false)
    }
  }, [base, purging, refresh])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(() => refresh({ quiet: true }), REFRESH_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [refresh])

  return { memory, loading, purging, error, notice, refresh, purge, clearNotice: () => setNotice('') }
}

export default useRuntimeMemory
