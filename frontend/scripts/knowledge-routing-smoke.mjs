#!/usr/bin/env node
import assert from 'node:assert/strict'

const values = new Map()
const requests = []
const nativeFetch = globalThis.fetch
globalThis.window = {
  location: { origin: 'http://ui.example', href: 'http://ui.example/' },
  localStorage: {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
  },
}
globalThis.fetch = window.fetch = async (input, init = {}) => {
  requests.push({ url: new URL(input, window.location.href), init })
  const path = requests.at(-1).url.pathname
  const body = path.endsWith('/collections') && !init.method ? []
    : path.endsWith('/folders') && !init.method ? []
      : path.endsWith('/documents') && !init.method ? []
        : { id: 'collection', name: 'Notes', doc_ids: [], doc_id: 'doc', filename: 'notes.txt', entries: [] }
  return Response.json(body)
}

try {
  const { apiFetch } = await import('../src/lib/apiRequest.js')
  const collections = await import('../src/lib/knowledgeCollections.js')
  const folders = await import('../src/lib/knowledgeFolders.js')
  const { installApiAuthFetch } = await import('../src/lib/apiAuth.js')

  await apiFetch('/api/documents')
  assert.equal(requests.at(-1).url.href, 'http://ui.example/api/documents', 'the default follows the UI origin and Vite proxy')
  values.set('camelid.apiBase', 'https://backend.example/engine/')
  values.set('camelid.apiKey', 'test-api-key')
  await collections.ingestLibraryFile({ name: 'notes.txt', text: async () => 'notes' })
  await collections.listCollections()
  await collections.createCollection('Notes')
  await collections.renameCollection('collection', 'Renamed')
  await collections.addCollectionDocuments('collection', ['doc'])
  await collections.removeCollectionDocument('collection', 'doc')
  await collections.deleteCollection('collection')
  await collections.listLibraryDocuments()
  await folders.listFolders()
  await folders.watchFolder('/work', 'collection')
  await folders.scanFolder('folder')
  await folders.unwatchFolder('folder')
  const controller = new AbortController()
  await apiFetch('/api/documents/doc/source', { signal: controller.signal })
  assert.equal(requests.at(-1).init.signal, controller.signal)
  for (const request of requests.slice(1)) {
    assert.equal(request.url.origin, 'https://backend.example')
    assert.ok(request.url.pathname.startsWith('/engine/api/'))
    assert.equal(new Headers(request.init.headers).get('x-api-key'), 'test-api-key')
  }
  await apiFetch('/api/documents', { headers: { authorization: 'Bearer explicit-key' } })
  assert.equal(new Headers(requests.at(-1).init.headers).get('x-api-key'), null, 'caller credentials are preserved')

  installApiAuthFetch()
  globalThis.fetch = window.fetch
  await folders.browseFolders('/work')
  assert.equal(requests.at(-1).url.href, 'https://backend.example/engine/api/agent/workspace/browse?path=%2Fwork')
  assert.equal(new Headers(requests.at(-1).init.headers).get('x-api-key'), 'test-api-key')
  values.set('camelid.apiBase', 'http://second.example')
  await collections.listCollections()
  assert.equal(requests.at(-1).url.href, 'http://second.example/api/collections', 'requests follow a backend change immediately')
  await fetch('https://unrelated.example/resource')
  assert.equal(new Headers(requests.at(-1).init.headers).get('x-api-key'), null, 'the global wrapper does not send the key to another origin')
  console.log('knowledge routing smoke passed')
} finally {
  globalThis.fetch = nativeFetch
  delete globalThis.window
}
