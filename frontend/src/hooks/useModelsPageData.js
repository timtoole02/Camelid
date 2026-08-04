import { useCallback, useEffect, useRef, useState } from 'react'
import { localModelDeleteRequest } from '../lib/modelDeletion'

/* Single data spine for the Models page. Owns fetch + refresh for the four
   backend truths the page renders from — /api/models/local (disk scan with lane
   facts), /api/models/current (loaded model), /api/models/catalog/downloads
   (live download progress), and /api/models/default (next-launch selection) —
   so every zone reads the same snapshot instead of fetching privately.
   Capabilities stay a prop (already lifted in
   useDashboardData); this hook never touches localStorage: "downloaded" is only
   ever the live disk scan.

   The downloads poll runs only while there is (or was just started) an active
   download and backs off when idle. When a download leaves the backend's active
   list (terminal statuses are reported once, then dropped), the local scan is
   refreshed so the file appears in its derived lane without a page reload. */

const DOWNLOAD_POLL_MS = 1500
// Ticks the poller keeps running after the list goes empty, so a just-started
// install (kicked before the backend registers it) isn't missed.
const IDLE_TICKS_BEFORE_STOP = 3

export function useModelsPageData({ apiBase = '' } = {}) {
  const base = (apiBase || '').replace(/\/$/, '')

  const [local, setLocal] = useState(null) // { models_dir, models: [...] }
  const [localLoading, setLocalLoading] = useState(false)
  const [localError, setLocalError] = useState('')
  const [current, setCurrent] = useState(null) // { path } from /api/models/current
  const [loadedModels, setLoadedModels] = useState([]) // /v1/models resident registry
  const [startupDefault, setStartupDefault] = useState(null) // { filename, configured }
  const [downloads, setDownloads] = useState([])
  const [polling, setPolling] = useState(true) // one initial pass on mount

  const idleTicksRef = useRef(0)
  const prevDownloadIdsRef = useRef(new Set())
  const refreshLocalPromiseRef = useRef(null)
  const localRequestSequenceRef = useRef(0)
  const localAppliedSequenceRef = useRef(0)
  const localRequestsPendingRef = useRef({ base, count: 0 })
  const baseRef = useRef(base)
  baseRef.current = base

  const refreshLocal = useCallback(async ({ force = false } = {}) => {
    if (!force && refreshLocalPromiseRef.current?.base === base) return refreshLocalPromiseRef.current.promise
    const sequence = ++localRequestSequenceRef.current
    if (force) localAppliedSequenceRef.current = sequence
    const request = (async () => {
      if (localRequestsPendingRef.current.base !== base) {
        localRequestsPendingRef.current = { base, count: 0 }
      }
      localRequestsPendingRef.current.count += 1
      setLocalLoading(true)
      setLocalError('')
      try {
        const res = await fetch(`${base}/api/models/local`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const next = await res.json()
        if (baseRef.current === base && sequence >= localAppliedSequenceRef.current) {
          localAppliedSequenceRef.current = sequence
          setLocal(next)
        }
        return next
      } catch (err) {
        if (baseRef.current === base) setLocalError(String(err?.message || err))
        return null
      } finally {
        if (localRequestsPendingRef.current.base === base) {
          localRequestsPendingRef.current.count -= 1
          if (localRequestsPendingRef.current.count === 0) setLocalLoading(false)
        }
      }
    })()
    const trackedRequest = { base, promise: request }
    refreshLocalPromiseRef.current = trackedRequest
    try {
      return await request
    } finally {
      if (refreshLocalPromiseRef.current === trackedRequest) refreshLocalPromiseRef.current = null
    }
  }, [base])

  const refreshCurrent = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/current`)
      if (!res.ok) {
        if (baseRef.current === base) setCurrent(null)
        return null
      }
      const next = await res.json()
      if (baseRef.current !== base) return null
      setCurrent(next)
      return next
    } catch {
      if (baseRef.current === base) setCurrent(null)
      return null
    }
  }, [base])

  const refreshLoadedModels = useCallback(async () => {
    try {
      const res = await fetch(`${base}/v1/models`)
      if (!res.ok) return null
      const next = await res.json()
      if (baseRef.current !== base) return null
      const models = Array.isArray(next?.data) ? next.data : []
      setLoadedModels(models)
      return models
    } catch {
      return null
    }
  }, [base])

  const refreshDownloads = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/catalog/downloads`)
      if (!res.ok) return null
      const list = await res.json()
      if (baseRef.current !== base) return null
      setDownloads(Array.isArray(list) ? list : [])
      return Array.isArray(list) ? list : []
    } catch {
      return null
    }
  }, [base])

  const refreshDefault = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/default`)
      if (!res.ok) return null
      const next = await res.json()
      if (baseRef.current !== base) return null
      setStartupDefault(next)
      return next
    } catch {
      return null
    }
  }, [base])

  const setDefaultModel = useCallback(
    async (filename) => {
      const res = await fetch(`${base}/api/models/default`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(body?.error?.message || `default model update failed (HTTP ${res.status})`)
      }
      if (baseRef.current === base) setStartupDefault(body)
      return body
    },
    [base],
  )

  const refreshLocalAndDefault = useCallback(async () => {
    const [nextLocal] = await Promise.all([refreshLocal(), refreshDefault()])
    return nextLocal
  }, [refreshDefault, refreshLocal])

  /* Wake the downloads poller — call right after starting an install. */
  const kickDownloadsPoll = useCallback(() => {
    idleTicksRef.current = 0
    setPolling(true)
  }, [])

  const cancelDownload = useCallback(
    async (id) => {
      const res = await fetch(`${base}/api/models/catalog/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      await refreshDownloads()
      await refreshLocal()
      return res.ok
    },
    [base, refreshDownloads, refreshLocal],
  )

  const deleteLocalModel = useCallback(
    async (entry) => {
      const payload = localModelDeleteRequest(entry)
      if (!payload) throw new Error('Refresh Models before deleting this file.')
      const res = await fetch(`${base}/api/models/local/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        await Promise.all([
          refreshLocal({ force: true }),
          refreshCurrent(),
          refreshDefault(),
          refreshDownloads(),
        ])
        const error = new Error(body?.error?.message || `delete failed (HTTP ${res.status})`)
        error.code = body?.error?.code || 'model_delete_failed'
        throw error
      }
      const [verified] = await Promise.all([
        refreshLocal({ force: true }),
        refreshDefault(),
      ])
      if (!verified) {
        setLocal((current) => current
          ? { ...current, models: current.models.filter((model) => model.filename !== entry.filename) }
          : current)
      }
      return body
    },
    [base, refreshCurrent, refreshDefault, refreshDownloads, refreshLocal],
  )

  const refreshAll = useCallback(async () => {
    await Promise.all([
      refreshLocal(),
      refreshCurrent(),
      refreshLoadedModels(),
      refreshDefault(),
      refreshDownloads(),
    ])
  }, [refreshLocal, refreshCurrent, refreshLoadedModels, refreshDefault, refreshDownloads])

  useEffect(() => {
    setLocal(null)
    setCurrent(null)
    setLoadedModels([])
    setStartupDefault(null)
    setDownloads([])
    setLocalLoading(false)
    refreshLocalPromiseRef.current = null
    localRequestsPendingRef.current = { base, count: 0 }
    localRequestSequenceRef.current += 1
    localAppliedSequenceRef.current = localRequestSequenceRef.current
    idleTicksRef.current = 0
    setPolling(true)
    refreshLocal()
    refreshCurrent()
    refreshLoadedModels()
    refreshDefault()
  }, [base, refreshLocal, refreshCurrent, refreshLoadedModels, refreshDefault])

  useEffect(() => {
    if (!polling) return undefined
    let cancelled = false
    const tick = async () => {
      const list = await refreshDownloads()
      if (cancelled || list === null) return
      const ids = new Set(list.filter((d) => d.status === 'downloading').map((d) => d.id))
      // Any download that left the active list (completed, failed, or canceled)
      // may have landed a file — re-scan disk so lanes update without a reload.
      const settled = [...prevDownloadIdsRef.current].some((id) => !ids.has(id))
      prevDownloadIdsRef.current = ids
      if (settled) {
        refreshLocalAndDefault()
      }
      if (ids.size === 0) {
        idleTicksRef.current += 1
        if (idleTicksRef.current >= IDLE_TICKS_BEFORE_STOP) setPolling(false)
      } else {
        idleTicksRef.current = 0
      }
    }
    tick()
    const timer = setInterval(tick, DOWNLOAD_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [polling, refreshDownloads, refreshLocalAndDefault])

  const activeFilename = String(current?.path || '').split(/[\\/]/).pop() || ''
  const localFilenames = new Set((local?.models || []).map((m) => m.filename))
  const loadedModelIds = new Set(loadedModels.map((model) => model?.id).filter(Boolean))

  return {
    base,
    local,
    localLoading,
    localError,
    current,
    activeFilename,
    loadedModels,
    loadedModelIds,
    defaultFilename: startupDefault?.filename || '',
    defaultConfigured: Boolean(startupDefault?.configured),
    localFilenames,
    downloads,
    refreshLocal,
    refreshLocalAndDefault,
    refreshCurrent,
    refreshLoadedModels,
    refreshDefault,
    refreshDownloads,
    refreshAll,
    kickDownloadsPoll,
    cancelDownload,
    deleteLocalModel,
    setDefaultModel,
  }
}
