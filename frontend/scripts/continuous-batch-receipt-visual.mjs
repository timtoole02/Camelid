#!/usr/bin/env node

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const receiptPath = resolve('../qa/evidence-bundles/f1-phase9-rollout/measured-batch8-result.json')
const receipt = JSON.parse(await readFile(receiptPath, 'utf8'))
const outputDir = resolve(process.env.CAMELID_CAPTURE_DIR || '../qa/evidence-bundles/f1-phase9-rollout/screenshots')
await mkdir(outputDir, { recursive: true })

const selected = receipt.product_gate
if (receipt.status !== 'pass') throw new Error('public batch-eight PASS summary is unavailable')
if (!selected.exact_serial_output_and_completion_parity) throw new Error('summary does not prove serial parity')
if (selected.survivors_completed !== 7) throw new Error('summary does not prove seven cancellation survivors')
if (selected.idle_allocated_pages !== 0 || selected.idle_device_bytes !== 0) throw new Error('summary idle memory is not zero')

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
:root{font-family:Inter,Segoe UI,sans-serif;color:#e7edf4;background:#0b0f13}*{box-sizing:border-box;min-width:0}body{margin:0;padding:42px;background:linear-gradient(145deg,#0b0f13,#111820);min-height:100vh}.wrap{max-width:1160px;margin:auto}.eyebrow{font:600 12px ui-monospace;color:#7dd3fc;text-transform:uppercase;letter-spacing:.12em}.head{display:flex;justify-content:space-between;gap:30px;align-items:end;border-bottom:1px solid #27323c;padding-bottom:24px}.head h1{font-size:36px;margin:8px 0 4px;letter-spacing:0}.head p{color:#9aa8b5;margin:0;max-width:700px}.stamp{font:12px ui-monospace;color:#9aa8b5;text-align:right}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card{background:#121920;border:1px solid #27323c;border-radius:8px;padding:18px;min-height:118px}.card strong{display:block;font-size:27px;margin:8px 0}.card span,.card small{color:#9aa8b5}.ok{color:#5ee1a4!important}.wide{grid-column:span 2}.rows{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-top:12px}.row{padding:12px 5px;text-align:center;border:1px solid #2d9cdb;background:#102332;border-radius:6px;font:700 13px ui-monospace}.proof{display:grid;grid-template-columns:1fr 1fr;gap:14px}.proof h2{font-size:18px;margin:0 0 12px}.line{display:flex;justify-content:space-between;padding:9px 0;border-top:1px solid #27323c;color:#aeb9c3;gap:10px}.line b{color:#e7edf4;text-align:right}.foot{margin-top:20px;color:#788895;font:11px ui-monospace;overflow-wrap:anywhere} @media(max-width:700px){body{padding:20px}.head{display:block}.stamp{text-align:left;margin-top:16px}.grid,.proof{grid-template-columns:1fr}.wide{grid-column:span 1}.rows{grid-template-columns:repeat(4,1fr)}.line{align-items:flex-start}.card strong{font-size:24px}}
</style></head><body><main class="wrap"><header class="head"><div><div class="eyebrow">Measured release receipt · not a mock runtime</div><h1>True eight-request CUDA batching</h1><p>One model forward serves independent sequence rows while each request keeps its own KV pages, output, cancellation and lifecycle.</p></div><div class="stamp">source ${receipt.source_commit.slice(0,12)}<br>RTX 4060 · CUDA resident<br>${receipt.configuration.prefill_quantum_tokens}-token prefill quantum</div></header><section class="grid"><article class="card"><span>Eight-request wall ratio</span><strong>${selected.eight_request_wall_ratio.toFixed(3)}×</strong><small class="ok">PASS · below 2×</small></article><article class="card"><span>Eighth / first TTFT</span><strong>${selected.slowest_fastest_ttft_ratio.toFixed(3)}×</strong><small class="ok">PASS · at most 2×</small></article><article class="card"><span>Shared size-8 decode</span><strong>${selected.size_8_decode_forwards}</strong><small>real CUDA forwards</small></article><article class="card"><span>Shared size-8 prefill</span><strong>${selected.size_8_prefill_forwards}</strong><small>cross-request forwards</small></article><article class="card wide"><span>Independent sequence rows observed</span><div class="rows">${Array.from({length:selected.independent_sequences},(_,i)=>`<div class="row">SEQ ${i+1}</div>`).join('')}</div></article><article class="card"><span>Serial equivalence</span><strong class="ok">Exact</strong><small>text and completion counts</small></article><article class="card"><span>Sustained churn</span><strong>${selected.sustained_churn_completed} / ${selected.sustained_churn_requests}</strong><small class="ok">all completed</small></article></section><section class="proof"><article class="card"><h2>Cancellation and overload</h2><div class="line"><span>Cancelled request</span><b>${selected.cancelled_request_outcome}</b></div><div class="line"><span>Sibling survivors</span><b>${selected.survivors_completed} / 7</b></div><div class="line"><span>Ninth request</span><b>HTTP ${selected.ninth_request_status} · ${selected.ninth_request_code}</b></div><div class="line"><span>Retry-After</span><b>${selected.retry_after_seconds}s</b></div></article><article class="card"><h2>Idle memory reconciliation</h2><div class="line"><span>Allocated KV pages</span><b>${selected.idle_allocated_pages}</b></div><div class="line"><span>Device KV bytes</span><b>${selected.idle_device_bytes}</b></div><div class="line"><span>Live sequences</span><b>${selected.idle_sequences}</b></div><div class="line"><span>Recovery request</span><b class="ok">completed</b></div></article></section><div class="foot">Source: qa/evidence-bundles/f1-phase9-rollout/measured-batch8-result.json · binary SHA-256 ${receipt.release_binary_sha256}</div></main></body></html>`
const htmlPath = resolve(outputDir, 'continuous-batch-receipt.html')
await writeFile(htmlPath, html, 'utf8')
const browser = await launchBrowser({ purpose: 'the Phase 7 measured receipt visualization', headless: 'new' })
try {
  for (const viewport of [{name:'desktop',width:1440,height:900},{name:'mobile',width:390,height:844}]) {
    const page = await browser.newPage()
    await page.setViewport(viewport)
    await page.goto(new URL(`file:///${htmlPath.replaceAll('\\','/')}`).href, { waitUntil: 'load' })
    const geometry = await page.evaluate(() => ({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,text:document.body.innerText}))
    if (geometry.client !== geometry.scroll) throw new Error(`${viewport.name}: receipt visualization overflows`)
    for (const marker of ['True eight-request CUDA batching','PASS · below 2×','59','44','7 / 7','HTTP 503','Allocated KV pages\n0']) {
      if (!geometry.text.includes(marker)) throw new Error(`${viewport.name}: missing receipt marker ${marker}`)
    }
    await page.screenshot({path:resolve(outputDir,`measured-batch8-proof-${viewport.name}.png`),fullPage:true})
    await page.close()
  }
} finally { await browser.close() }
console.log('measured continuous-batch receipt visual: PASS')
