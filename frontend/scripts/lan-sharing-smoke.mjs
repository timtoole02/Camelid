import assert from 'node:assert/strict'
import { createServer } from 'vite'
import { writeFileSync, rmSync } from 'node:fs'
import { launchBrowser } from './lib/launch-browser.mjs'

writeFileSync('.lan-sharing-test.html', '<div id="root"></div><script type="module" src="/.lan-sharing-test.jsx"></script>')
writeFileSync('.lan-sharing-test.jsx', `import React from 'react'; import {createRoot} from 'react-dom/client'; import {LanSharingCard} from './src/components/settings/LanSharingCard'; createRoot(document.getElementById('root')).render(<LanSharingCard online apiBase={location.origin} />);`)
const server = await createServer({ server: { host: '127.0.0.1', port: 0 } })
await server.listen()
const base = server.resolvedUrls.local[0]
const browser = await launchBrowser()
try {
  const page = await browser.newPage()
  let enabled = false
  let fail = false
  await page.setRequestInterception(true)
  page.on('request', async request => {
    if (request.url().endsWith('/api/runtime/lan-sharing')) {
      if (request.method() === 'POST') {
        if (fail) return request.respond({ status: 500, contentType: 'application/json', body: '{}' })
        enabled = JSON.parse(request.postData()).enabled
      }
      return request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify({ enabled, url: enabled ? 'http://192.0.2.5:12345' : null, key: enabled ? 'test-key' : null }) })
    }
    return request.continue()
  })
  await page.goto(`${base}.lan-sharing-test.html`)
  await page.waitForSelector('[role="switch"]')
  assert.equal(await page.$eval('[role="switch"]', element => element.getAttribute('aria-checked')), 'false')
  await page.click('[role="switch"]')
  await page.waitForSelector('[aria-label="Network address"]')
  assert.equal(await page.$eval('[aria-label="Network address"]', element => element.value), 'http://192.0.2.5:12345')
  assert.equal(await page.$eval('[aria-label="Network access key"]', element => element.type), 'password')
  fail = true
  await page.click('[role="switch"]')
  await page.waitForSelector('[role="alert"]')
  assert.equal(await page.$eval('[role="switch"]', element => element.getAttribute('aria-checked')), 'true')
  fail = false
  await page.click('[role="switch"]')
  await page.waitForFunction(() => !document.querySelector('[aria-label="Network address"]'))
  console.log('LAN_SHARING_SMOKE_PASS')
} finally {
  await browser.close()
  await server.close()
  rmSync('.lan-sharing-test.html', { force: true })
  rmSync('.lan-sharing-test.jsx', { force: true })
}
