#!/usr/bin/env node
/* Chat streaming performance profiler (Phase 8B). Drives ONE long real
   streamed response (code-heavy prompt) and records: long tasks (>50ms),
   frame-drop distribution, send→user-message-visible, first-chunk→first-paint,
   and flush count (each flush = one full or partial markdown render).
   Usage: node scripts/chat-perf-profile.mjs --label baseline --out design-evidence/phase-8 */
import { mkdirSync, writeFileSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i]?.replace(/^--/, ''), process.argv[i + 1])
const label = args.get('label') || 'run'
const out = args.get('out') || 'design-evidence/phase-8'
mkdirSync(`${out}/${label}-frames`, { recursive: true })

const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] })
const page = await browser.newPage()
await page.setViewport({ width: 1440, height: 900 })
await page.evaluateOnNewDocument(() => {
  window.localStorage.setItem('camelid-theme', 'dark')
  window.localStorage.removeItem('camelid.conversations')
})
await page.goto('http://127.0.0.1:4175/#chat', { waitUntil: 'domcontentloaded' })
await new Promise(r => setTimeout(r, 4000))

await page.evaluate(() => {
  window.__perf = { longTasks: [], frames: [], sendAt: 0, userVisibleAt: 0, firstChunkAt: 0, firstPaintAt: 0, flushes: 0 }
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) window.__perf.longTasks.push({ start: entry.startTime, dur: entry.duration })
  }).observe({ entryTypes: ['longtask'] })
  let last = performance.now()
  const tick = (now) => { window.__perf.frames.push(now - last); last = now; requestAnimationFrame(tick) }
  requestAnimationFrame(tick)
  // user message visibility + first assistant paint via mutation observer
  const mo = new MutationObserver(() => {
    const p = window.__perf
    if (!p.userVisibleAt && document.querySelector('.cxturn--user, article.cxturn.cxturn--user') && [...document.querySelectorAll('.cxturn--user p')].some(n => n.textContent.includes('PERFPROBE'))) p.userVisibleAt = performance.now()
    const rows = document.querySelectorAll('article.cxturn--assistant')
    const lastRow = rows[rows.length - 1]
    if (!p.firstPaintAt && lastRow && (lastRow.textContent || '').length > 5) p.firstPaintAt = performance.now()
  })
  mo.observe(document.body, { childList: true, subtree: true, characterData: true })
  // count rAF-coalesced content flushes via the assistant row text length changes
  let lastLen = 0
  const lenTick = () => {
    const rows = document.querySelectorAll('article.cxturn--assistant')
    const lastRow = rows[rows.length - 1]
    const len = lastRow ? (lastRow.textContent || '').length : 0
    if (len !== lastLen) { window.__perf.flushes += 1; lastLen = len }
    requestAnimationFrame(lenTick)
  }
  requestAnimationFrame(lenTick)
})

const prompt = 'PERFPROBE Write a python script with three functions (parse_args, run_simulation, plot_results), full docstrings and a main entry point. After the code, explain each function in two sentences. Then write a second short bash script in a code block.'
await page.type('.cxcomposer__input', prompt)
await page.evaluate(() => { window.__perf.sendAt = performance.now() })
await page.keyboard.press('Enter')
const frameShots = setInterval(() => {}, 99999)
let shot = 0
const shotTimer = setInterval(async () => {
  if (shot < 30) { await page.screenshot({ path: `${out}/${label}-frames/f-${String(shot).padStart(2, '0')}.png` }).catch(() => {}); shot += 1 }
}, 400)
await page.waitForFunction(() => !document.querySelector('article.cxturn--assistant.is-streaming') && document.querySelectorAll('.cxturn__meta').length >= 1, { timeout: 600000 })
clearInterval(shotTimer); clearInterval(frameShots)
await new Promise(r => setTimeout(r, 500))

const result = await page.evaluate(() => {
  const p = window.__perf
  const streamFrames = p.frames.filter(f => f > 0)
  const dropped = streamFrames.filter(f => f > 33).length
  const badly = streamFrames.filter(f => f > 100).length
  const meta = document.querySelector('.cxturn__meta')?.textContent || ''
  const chatLongTasks = p.longTasks.filter(t => t.start > p.sendAt)
  return {
    sendToUserVisibleMs: p.userVisibleAt ? +(p.userVisibleAt - p.sendAt).toFixed(1) : null,
    firstPaintFromSendMs: p.firstPaintAt ? +(p.firstPaintAt - p.sendAt).toFixed(1) : null,
    longTasksOver50ms: chatLongTasks.length,
    worstLongTaskMs: chatLongTasks.length ? Math.round(Math.max(...chatLongTasks.map(t => t.dur))) : 0,
    totalFrames: streamFrames.length,
    framesOver33ms: dropped,
    framesOver100ms: badly,
    contentFlushes: p.flushes,
    footerText: meta.slice(0, 90),
  }
})
const tokens = await page.evaluate(() => {
  const m = document.querySelector('.cxturn__meta')?.textContent.match(/tokens[^0-9]*in\s+(\d+)\s*·\s*out\s+(\d+)/)
  return m ? Number(m[2]) : null
})
result.completionTokens = tokens
console.log(JSON.stringify(result, null, 2))
writeFileSync(`${out}/${label}.json`, JSON.stringify(result, null, 2))
await browser.close()
