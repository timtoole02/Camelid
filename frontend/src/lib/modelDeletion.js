export function modelDeleteBlockedReason({
  activeFilename = '',
  residentModelsLoaded = false,
  downloads = [],
  loading = false,
  smoking = false,
} = {}) {
  if (residentModelsLoaded) return 'Unload every loaded model before deleting files.'
  if (activeFilename) return 'Unload the current model before deleting files.'
  if (loading) return 'Wait for the current model load to finish.'
  if (smoking) return 'Wait for the current model check to finish.'
  if (downloads.some((download) => download.status === 'downloading')) {
    return 'Cancel or finish active model downloads before deleting files.'
  }
  return ''
}

export function localModelDeleteRequest(entry) {
  if (!entry?.filename || !entry?.delete_token) return null
  return { filename: entry.filename, delete_token: entry.delete_token }
}