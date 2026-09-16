#!/usr/bin/env node
// Real ChatWorkspace and microphone/AudioWorklet capture, with controlled API
// responses to test cancellation, permissions, and conversation isolation.
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'vite'
import react from '@vitejs/plugin-react'
import { launchBrowser } from './lib/launch-browser.mjs'

const liveApi = process.env.CAMELID_SPEECH_LIVE_API
const liveWav = process.env.CAMELID_SPEECH_LIVE_WAV
const state = { installed: false, requests: [], delay: 0, text: 'This is a dictated prompt.' }
const handler = async (req, res, next) => {
  if (req.url === '/__speech_fixture.wav' && liveWav) { res.writeHead(200, { 'Content-Type': 'audio/wav' }); res.end(readFileSync(liveWav)); return }
  if (req.url?.startsWith('/api/speech/')) {
    const chunks = []; for await (const chunk of req) chunks.push(chunk)
    state.requests.push({ path: req.url, body: Buffer.concat(chunks) })
    if (liveApi) {
      if (req.url.endsWith('/transcribe')) writeFileSync('../target/voice-input-capture.wav', Buffer.concat(chunks))
      const response = await fetch(`${liveApi}${req.url}`, { method: req.method, headers: { 'Content-Type': 'audio/wav', 'x-api-key': process.env.CAMELID_SPEECH_LIVE_KEY || '' }, body: req.method === 'GET' ? undefined : Buffer.concat(chunks) })
      res.writeHead(response.status, { 'Content-Type': 'application/json' }); res.end(await response.text()); return
    }
    if (req.url.endsWith('/transcribe') && state.delay) await new Promise((r) => setTimeout(r, state.delay))
    let body = { installed: state.installed }
    if (req.url.endsWith('/install')) { state.installed = true; body = { installed: true } }
    if (req.url.endsWith('/transcribe')) body = { text: state.text }
    res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); return
  }
  if (req.url !== '/__voice_test') return next()
  const html = await server.transformIndexHtml('/__voice_test', '<!doctype html><html><body><div id="root"></div><script type="module" src="/__voice_harness.js"></script></body></html>')
  res.writeHead(200, { 'Content-Type': 'text/html' }); res.end(html)
}
mkdirSync('../target', { recursive: true })
const harness = `    import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import ChatWorkspace from '/src/views/ChatWorkspace.jsx';
    import '/src/styles.css';
    const model = {id:'test', name:'Test model', lane_class:'supported', loaded_now:true, generation_ready:true, status:'ready', source:'local',model_path:'/test.gguf'};
    const runtime = {status:'online', model:'test',active_model_id:'test', loaded_now:true, generation_ready:true};
    function Harness() {
      const [draft,setDraft]=useState('Existing draft.'); const [id,setId]=useState('one');
      window.changeChat=()=>{setId('two');setDraft('Other chat.');};
      return React.createElement(ChatWorkspace,{apiBase:location.origin,selectedConversation:{id,messages:[]},selectedModel:model,selectedModelId:'test',setSelectedModelId:()=>{},models:[model],runtime,capabilities:{},composer:draft,setComposer:setDraft,sendMessage:()=>{window.sent=draft},stopGeneration:()=>{},setTab:()=>{},saveToMemory:()=>{},selectedModelRunnable:true});
    }
    createRoot(document.getElementById('root')).render(React.createElement(Harness));
`
const server = await createServer({ configFile: false, plugins: [react(), { name: 'speech-test', resolveId(id) { if (id === '/__voice_harness.js') return '\0voice-harness.js' }, load(id) { if (id === '\0voice-harness.js') return harness }, configureServer(server) { server.middlewares.use(handler) } }], optimizeDeps: { include: ['react', 'react-dom/client'] }, server: { port: 0, host: '127.0.0.1' } })
await server.listen()
const base = `http://127.0.0.1:${server.httpServer.address().port}`
const browser = await launchBrowser({ purpose: 'voice input', headless: true, args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] })
const page = await browser.newPage()
page.setDefaultTimeout(30000)
const errors = []
page.on('pageerror', (error) => { errors.push(error.message); console.error(error.message) })
const mic = '[aria-label="Dictate prompt"]'
const stop = '[aria-label="Stop recording"]'
const send = '[aria-label="Send message"]'
const clickText = (text) => page.evaluate((text) => {
  const button = [...document.querySelectorAll('.cxvoice button')].find((node) => node.textContent.trim() === text)
  if (!button) throw new Error(`Missing button ${text}`)
  button.click()
}, text)
const ready = () => page.waitForFunction((selector) => { const node = document.querySelector(selector); return node && !node.disabled }, {}, mic)
const waitSamples = () => page.waitForFunction(() => /[1-9]\d*s/.test(document.querySelector('.cxvoice__mic')?.textContent || ''))
const countUploads = () => state.requests.filter((r) => r.path.endsWith('/transcribe')).length
try {
  await page.goto(`${base}/__voice_test`)
  await ready()
  if (liveApi) {
    // Inject a known speech MediaStream. This avoids platform-dependent fake
    // audio-file devices while exercising the actual recorder and Rust API.
    await page.evaluate(async () => {
      const context = new AudioContext()
      await context.resume()
      const sound = await context.decodeAudioData(await (await fetch('/__speech_fixture.wav')).arrayBuffer())
      navigator.mediaDevices.getUserMedia = async () => {
        const destination = context.createMediaStreamDestination()
        const source = context.createBufferSource()
        source.buffer = sound
        source.connect(destination)
        source.start(context.currentTime + 0.5)
        return destination.stream
      }
    })
    assert.ok(liveWav, 'Set CAMELID_SPEECH_LIVE_WAV to an 11-second PCM16 speech fixture')
    await page.click(mic); await page.waitForSelector(stop)
    await page.waitForFunction(() => parseInt(document.querySelector('.cxvoice__mic')?.textContent || '0', 10) >= 10, {timeout: 30000})
    await page.click(stop); await ready()
    const draft = await page.$eval('textarea', (node) => node.value)
    assert.ok(draft.startsWith('Existing draft. '), draft)
    assert.ok(draft.toLowerCase().includes(process.env.CAMELID_SPEECH_LIVE_EXPECT || 'ask not what your country'), draft)
    assert.equal(await page.evaluate(() => window.sent), undefined)
    assert.deepEqual(errors, [])
    console.log('PASS: speech MediaStream → AudioWorklet/WAV → Rust resampling/Whisper → editable prompt:', draft)
  } else {
  await page.click(mic); await clickText('Download speech model'); await ready()
  assert.equal(state.installed, true)
  await page.click(mic); await page.waitForSelector(stop); await waitSamples()
  assert.equal(await page.$eval(send, (node) => node.disabled), true)
  await page.click(stop); await ready()
  assert.equal(await page.$eval('textarea', (node) => node.value), 'Existing draft. This is a dictated prompt.')
  assert.equal(await page.evaluate(() => window.sent), undefined)
  const upload = state.requests.find((r) => r.path.endsWith('/transcribe')).body
  assert.equal(upload.toString('ascii', 0, 4), 'RIFF')
  assert.equal(upload.readUInt16LE(22), 1)
  assert.equal(upload.readUInt16LE(34), 16)
  assert.equal(upload.readUInt32LE(40), upload.length - 44)
  assert.ok(upload.length > 1000)
  await page.evaluate(() => { window.realGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('denied', 'NotAllowedError')) })
  await page.click(mic)
  await page.waitForFunction(() => document.querySelector('.cxvoice__panel')?.textContent.includes('was denied'))
  await ready(); assert.equal(countUploads(), 1)
  await page.evaluate(() => { navigator.mediaDevices.getUserMedia = async (...args) => { const s = await window.realGetUserMedia(...args); window.lastStream = s; return s } })
  await page.click(mic); await page.waitForSelector(stop); await clickText('Cancel'); await ready()
  assert.equal(await page.evaluate(() => window.lastStream.getTracks().every((track) => track.readyState === 'ended')), true)
  assert.equal(countUploads(), 1)
  state.delay = 1000
  await page.click(mic); await page.waitForSelector(stop); await waitSamples(); await page.click(stop)
  await page.waitForSelector('[aria-label="Transcribing speech"]')
  await page.evaluate(() => window.changeChat())
  await new Promise((r) => setTimeout(r, 1300)); await ready()
  assert.equal(await page.$eval('textarea', (node) => node.value), 'Other chat.')
  state.delay = 0
  await page.setViewport({width:390,height:844})
  await page.click(mic); await page.waitForSelector(stop)
  const bounds = await page.$eval('.cxvoice__panel', (node) => { const r=node.getBoundingClientRect(); return {left:r.left,right:r.right} })
  assert.ok(bounds.left >= 0 && bounds.right <= 390, JSON.stringify(bounds))
  await page.screenshot({path: '../target/voice-input-mobile.png'})
  await clickText('Cancel')
  await ready()
  const beforeAutoStop = countUploads()
  await page.click(mic); await page.waitForSelector(stop)
  await page.waitForFunction(() => !document.querySelector('[aria-label="Stop recording"]'), {timeout: 40000})
  await ready()
  assert.equal(countUploads(), beforeAutoStop + 1, '30-second limit stops and transcribes')
  assert.equal(await page.evaluate(() => window.lastStream.getTracks().every((t) => t.readyState === 'ended')), true)
  const autoUpload = state.requests.filter((r) => r.path.endsWith('/transcribe')).at(-1).body
  assert.ok(autoUpload.length <= 44 + autoUpload.readUInt32LE(24) * 30 * 2)
  assert.deepEqual(errors, [])
  console.log('PASS: setup, microphone/WAV capture, draft insertion, permission denial, cancellation, chat-switch isolation, mobile layout')
  }
} catch (error) { console.error(await page.evaluate(() => document.body.innerText)); throw error } finally { await browser.close(); await server.close() }
