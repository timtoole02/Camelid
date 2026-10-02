import { useCallback, useEffect, useState } from 'react'
import { listCollections } from '../lib/knowledgeCollections.js'

/* The server's collections, fetched once when enabled and on request. `null`
   until the first answer, so callers never mistake "not loaded" for "none". */
export function useKnowledgeCollections(enabled = true) {
  const [collections, setCollections] = useState(null)
  const [error, setError] = useState('')
  const refresh = useCallback(async () => {
    try {
      const next = await listCollections()
      setCollections(next)
      setError('')
      return next
    } catch (failure) {
      setError(failure?.message || 'Collections are unavailable.')
      return null
    }
  }, [])
  useEffect(() => {
    if (enabled) refresh()
  }, [enabled, refresh])
  return { collections, error, refresh, setCollections }
}
