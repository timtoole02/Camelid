#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createServer as createHttpServer } from 'node:http'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'
import { devRequestOriginAllowed } from './lib/dev-request-origin.mjs'

const request = (headers = {}, remoteAddress = '127.0.0.1') => ({
  headers: { host: '127.0.0.1:4175', ...headers }, socket: { remoteAddress },
})
assert.equal(devRequestOriginAllowed(request()), true)
assert.equal(devRequestOriginAllowed(request({ origin: 'http://127.0.0.1:4175' })), true)
for (const origin of ['https://untrusted.example', 'http://127.0.0.1:4176', 'null', 'invalid']) {
  assert.equal(devRequestOriginAllowed(request({ origin })), false)
}
assert.equal(devRequestOriginAllowed(request({ 'sec-fetch-site': 'cross-site' })), false)
assert.equal(devRequestOriginAllowed(request({}, '192.0.2.1')), false)
assert.equal(devRequestOriginAllowed(request({ host: 'untrusted.example', origin: 'http://untrusted.example' })), false)

const backend = createHttpServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify({ origin: req.headers.origin || null, site: req.headers['sec-fetch-site'] || null }))
})
await new Promise(done => backend.listen(0, '127.0.0.1', done))
const backendOrigin = `http://127.0.0.1:${backend.address().port}`
const priorTarget = process.env.VITE_CAMELID_PROXY_TARGET
process.env.VITE_CAMELID_PROXY_TARGET = `${backendOrigin}/gateway`
let vite
try {
  vite = await createServer({
    root: fileURLToPath(new URL('..', import.meta.url)),
    logLevel: 'silent',
    // This test exercises HTTP middleware only. Do not start a native dependency
    // scan that would still be running when this short-lived server closes.
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { host: '127.0.0.1', port: 0 },
  })
  await vite.listen()
  const uiOrigin = `http://127.0.0.1:${vite.httpServer.address().port}`
  const post = async (path, headers = {}, body) => fetch(`${uiOrigin}${path}`, { method: 'POST', headers, body })
  const foreign = await (await post('/api/models/unload', { origin: 'https://untrusted.example' })).json()
  assert.equal(foreign.origin, 'https://untrusted.example', 'the proxy must not relabel an untrusted request')
  const same = await (await post('/api/models/unload', { origin: uiOrigin })).json()
  assert.equal(same.origin, backendOrigin, 'the local UI still reaches origin-protected backend routes')
  const cli = await (await post('/api/models/unload')).json()
  assert.equal(cli.origin, backendOrigin, 'local CLI proxy clients remain supported')
  const metadata = await (await post('/api/models/unload', { 'sec-fetch-site': 'cross-site' })).json()
  assert.equal(metadata.origin, null)
  assert.equal(metadata.site, 'cross-site')
  for (const path of ['/__camelid/backend/launch', '/__camelid/backend/stop']) {
    const response = await post(path, { origin: 'https://untrusted.example', 'content-type': 'text/plain' }, JSON.stringify({ command: 'exit 0' }))
    assert.equal(response.status, 403, 'development management rejects untrusted browser mutations')
    await response.text()
  }
} finally {
  await vite?.close()
  await new Promise(done => backend.close(done))
  if (priorTarget === undefined) delete process.env.VITE_CAMELID_PROXY_TARGET
  else process.env.VITE_CAMELID_PROXY_TARGET = priorTarget
}
console.log('development origin smoke passed')
