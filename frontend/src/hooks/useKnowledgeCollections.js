import { useCallback, useEffect, useRef, useState } from 'react'
import { listCollections } from '../lib/knowledgeCollections.js'
import { getApiBase } from '../lib/apiBase.js'

/* The server's collections, fetched once when enabled and on request. `null`
   until the first answer, so callers never mistake "not loaded" for "none". */
export function useKnowledgeCollections(enabled = true, apiBase = getApiBase()) {
  const currentBase = useRef(apiBase)
  currentBase.current = apiBase
  const [collections, setCollections] = useState(null)
  const [error, setError] = useState('')
  const refresh = useCallback(async () => {
    try {
      const next = await listCollections(apiBase)
      if (currentBase.current !== apiBase) return null
      setCollections(next)
      setError('')
      return next
    } catch (failure) {
      if (currentBase.current === apiBase) setError(failure?.message || 'Collections are unavailable.')
      return null
    }
  }, [apiBase])
  useEffect(() => {
    setCollections(null)
    setError('')
    if (enabled) refresh()
  }, [enabled, refresh])
  return { collections, error, refresh, setCollections }
}
