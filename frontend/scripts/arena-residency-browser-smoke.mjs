#!/usr/bin/env node

import { mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outputDir = resolve(process.env.CAMELID_CAPTURE_DIR || '../qa/evidence-bundles/f1-phase9-rollout/screenshots')
await mkdir(outputDir, { recursive: true })

const MODEL_A = {
  id: 'phase9-llama-1b',
  name: 'Llama 3.2 1B Instruct Q8_0',
  provider_kind: 'local',
  status: 'ready',
  model_path: 'C:/models/Llama-3.2-1B-Instruct-Q8_0.gguf',
  runtime_model_name: 'phase9-llama-1b',
  architecture: 'llama',
  generation_capable: true,
  chat_capable: true,
}
const MODEL_B = {
  id: 'phase9-llama-3b',
  name: 'Llama 3.2 3B Instruct Q5_K_M',
  provider_kind: 'local',
  status: 'ready',
  model_path: 'C:/models/Llama-3.2-3B-Instruct-Q5_K_M.gguf',
  runtime_model_name: 'phase9-llama-3b',
  architecture: 'llama',
  generation_capable: true,
  chat_capable: true,
}

function health({ capacity = 2, resident = 2, active = 0, legacy = false } = {}) {
  return {
    ok: true,
    engine: 'camelid',
    api_surface: 'full',
    version: '0.7.3',
    build: 'phase9-visual-gate',
    loaded_now: true,
    generation_ready: true,
    active_model_id: MODEL_A.id,
    backend: 'llama',
    model_family: 'llama-family',
    q8_runtime: {},
    execution_plan: { selected_backend: 'cuda_resident_q8_runtime', cuda_resident_active: true },
    continuous_batch_slots: 8,
    ...(!legacy ? {
      cuda_resident_arena: {
        capacity_models: capacity,
        resident_models: resident,
        active_models: active,
        evictions: 0,
        admission_failures: 0,
      },
    } : {}),
  }
}

function localModels(models) {
  return {
    models_dir: 'C:/models',
    models: models.map((model) => ({
      filename: model.model_path.split('/').at(-1),
      size_bytes: model.id === MODEL_A.id ? 1_321_082_528 : 2_322_154_016,
      architecture: 'llama',
      quantization: model.id === MODEL_A.id ? 'Q8_0' : 'Q5_K_M',
      tokenizer_kind: 'gpt2_bpe',
      admitted: true,
      chat_capable: true,
      generation_capable: true,
      lane_class: 'supported',
    })),
  }
}

function modelList(models) {
  return {
    object: 'list',
    data: models.map((model) => ({
      id: model.id,
      object: 'model',
      created: 0,
      owned_by: 'camelid',
      meta: { architecture: 'llama', n_ctx_train: 131072 },
    })),
  }
}

const capabilities = {
  model_compatibility: [
    { id: MODEL_A.id, family: 'llama', quantization: 'Q8_0', status: 'supported_exact_row_smoke' },
    { id: MODEL_B.id, family: 'llama', quantization: 'Q5_K_M', status: 'supported_exact_row_smoke' },
  ],
  api_features: [],
  support_contract: {},
}

function runtimeMemory({ models = [MODEL_A, MODEL_B], capacity = 2, active = 1, longId = false } = {}) {
  const visible = longId
    ? [{ ...MODEL_A, id: 'phase9-extremely-long-model-identity-that-must-wrap-without-horizontal-overflow-llama-3-2-1b-instruct-q8-0' }, MODEL_B]
    : models
  return {
    process_resident_bytes: 3_754_287_104,
    model_weight_bytes_estimate: 3_643_236_544,
    kv_cache_bytes: 5_505_024,
    kv_cache_entries: 1,
    kv_cache_capacity: 8,
    cuda_resident_arena: {
      capacity_models: capacity,
      resident_models: Math.min(capacity, visible.length),
      active_models: active,
      evictions: 0,
      admission_failures: 0,
    },
    models: visible.map((model, index) => ({
      id: model.id,
      active: index === 0,
      weight_bytes_estimate: index === 0 ? 1_321_082_528 : 2_322_154_016,
      kv_cache_bytes: index === 0 ? 5_505_024 : 0,
      kv_cache_entries: index === 0 ? 1 : 0,
      cached_tokens: index === 0 ? 96 : 0,
      cuda_resident: index < capacity,
      cuda_active: index < active,
    })),
  }
}

async function respondJson(request, body, status = 200) {
  await request.respond({
    status,
    contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': '*' },
    body: JSON.stringify(body),
  })
}

const cases = [
  {
    name: 'arena-capacity-two-dark-desktop', hash: 'arena', theme: 'dark', viewport: { width: 1440, height: 900 },
    models: [MODEL_A, MODEL_B], health: health(), memory: runtimeMemory(),
    expect: { heading: 'Two-model CUDA residency enabled', detail: '2 of 2 resident' },
  },
  {
    name: 'arena-default-one-light-mobile', hash: 'arena', theme: 'light', viewport: { width: 390, height: 844 },
    models: [MODEL_A, MODEL_B], health: health({ capacity: 1, resident: 1 }), memory: runtimeMemory({ capacity: 1, active: 0 }),
    expect: { heading: 'Sequential model switching', detail: 'replaces it with Model B' },
  },
  {
    name: 'arena-legacy-backend-dark-desktop', hash: 'arena', theme: 'dark', viewport: { width: 1280, height: 800 },
    models: [MODEL_A, MODEL_B], health: health({ legacy: true }), memory: null,
    expect: { heading: 'Sequential model switching', detail: 'replaces it with Model B' },
  },
  {
    name: 'arena-one-model-empty-light-desktop', hash: 'arena', theme: 'light', viewport: { width: 1280, height: 800 },
    models: [MODEL_A], health: health({ capacity: 2, resident: 1 }), memory: runtimeMemory({ models: [MODEL_A], active: 0 }),
    expect: { heading: 'Two local chat models are needed', detail: 'Open Models' },
  },
  {
    name: 'analytics-resident-active-dark-desktop', hash: 'analytics', theme: 'dark', viewport: { width: 1440, height: 900 },
    models: [MODEL_A, MODEL_B], health: health({ active: 1 }), memory: runtimeMemory({ active: 1 }),
    expect: { heading: 'CUDA residency', detail: 'GPU busy' },
  },
  {
    name: 'analytics-long-id-light-mobile', hash: 'analytics', theme: 'light', viewport: { width: 390, height: 844 },
    models: [MODEL_A, MODEL_B], health: health(), memory: runtimeMemory({ active: 0, longId: true }),
    expect: { heading: 'CUDA residency', detail: 'GPU resident' },
  },
  {
    name: 'analytics-memory-error-dark-desktop', hash: 'analytics', theme: 'dark', viewport: { width: 1280, height: 800 },
    models: [MODEL_A, MODEL_B], health: health(), memoryError: true,
    expect: { heading: 'memory request failed (HTTP 503)', detail: 'not reported by this backend' },
  },
].filter((scenario) => !process.env.CAMELID_VISUAL_CASE || scenario.name === process.env.CAMELID_VISUAL_CASE)

const browser = await launchBrowser({ purpose: 'the Phase 9 arena residency visual gate', headless: 'new' })
try {
  for (const scenario of cases) {
    const page = await browser.newPage()
    await page.setViewport(scenario.viewport)
    await page.evaluateOnNewDocument((theme, models) => {
      localStorage.setItem('camelid-theme', theme)
      localStorage.setItem('camelid.localModels', JSON.stringify(models))
      localStorage.setItem('camelid.activeTab', 'chat')
    }, scenario.theme, scenario.models)

    await page.setRequestInterception(true)
    page.on('request', async (request) => {
      const url = request.url()
      if (url.endsWith('/v1/health')) return respondJson(request, scenario.health)
      if (url.endsWith('/v1/models')) return respondJson(request, modelList(scenario.models))
      if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
      if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
      if (url.endsWith('/api/models/local')) return respondJson(request, localModels(scenario.models))
      if (url.endsWith('/api/models/current')) {
        return respondJson(request, {
          id: MODEL_A.id,
          path: MODEL_A.model_path,
          tokenizer: { status: 'available', model: 'gpt2' },
          llama_config: {},
          llama_tensors: {},
          gguf: { metadata: { general: { file_type: 7 } } },
        })
      }
      if (url.endsWith('/api/runtime/memory')) {
        if (scenario.memoryError) return respondJson(request, { error: { message: 'memory request failed (HTTP 503)' } }, 503)
        return respondJson(request, scenario.memory || runtimeMemory())
      }
      return request.continue()
    })

    await page.goto(`${baseUrl}/#${scenario.hash}`, { waitUntil: 'networkidle2', timeout: 30000 })
    try {
      await page.waitForFunction((expected) => document.body.innerText.toLowerCase().includes(expected.toLowerCase()), { timeout: 10000 }, scenario.expect.heading)
      await page.waitForFunction((expected) => document.body.innerText.toLowerCase().includes(expected.toLowerCase()), { timeout: 10000 }, scenario.expect.detail)
    } catch (error) {
      const diagnostics = await page.evaluate(() => ({
        hash: window.location.hash,
        body: document.body.innerText,
        memoryPanel: document.querySelector('.a-memory')?.innerText || null,
      }))
      throw new Error(`${scenario.name}: expected ${JSON.stringify(scenario.expect)}; rendered ${JSON.stringify(diagnostics)}; ${error.message}`)
    }

    const layout = await page.evaluate((hash) => {
      const root = document.documentElement
      const view = document.querySelector(hash === 'arena' ? '.arena-view' : '.analytics-view')
      const residency = document.querySelector('.arena-residency')
      const cards = [...document.querySelectorAll('.arena-card')].map((node) => {
        const rect = node.getBoundingClientRect()
        return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom }
      })
      return {
        documentWidth: [root.clientWidth, root.scrollWidth],
        viewWidth: view ? [view.clientWidth, view.scrollWidth] : null,
        residencyRole: residency?.getAttribute('role') || null,
        residencyLive: residency?.getAttribute('aria-live') || null,
        cards,
      }
    }, scenario.hash)

    if (layout.documentWidth[0] !== layout.documentWidth[1]) throw new Error(`${scenario.name}: document horizontal overflow ${JSON.stringify(layout)}`)
    if (layout.viewWidth && layout.viewWidth[1] > layout.viewWidth[0] + 1) throw new Error(`${scenario.name}: view horizontal overflow ${JSON.stringify(layout)}`)
    if (scenario.hash === 'arena' && scenario.models.length >= 2) {
      if (layout.residencyRole !== 'status' || layout.residencyLive !== 'polite') throw new Error(`${scenario.name}: residency status is not announced accessibly ${JSON.stringify(layout)}`)
      if (scenario.viewport.width > 980 && layout.cards.length === 2 && layout.cards[0].right > layout.cards[1].left + 1) throw new Error(`${scenario.name}: arena cards overlap ${JSON.stringify(layout)}`)
      if (scenario.viewport.width <= 980 && layout.cards.length === 2 && layout.cards[0].bottom > layout.cards[1].top + 1) throw new Error(`${scenario.name}: stacked arena cards overlap ${JSON.stringify(layout)}`)
    }

    if (scenario.name === 'analytics-long-id-light-mobile') {
      await page.$eval('.a-memory__model', (node) => node.scrollIntoView({ block: 'center' }))
      await new Promise((resolve) => setTimeout(resolve, 150))
    }

    const screenshot = resolve(outputDir, `${scenario.name}.png`)
    await page.screenshot({ path: screenshot, fullPage: true })
    console.log(`${scenario.name}: PASS ${JSON.stringify(layout)}`)
    await page.close()
  }
} finally {
  await browser.close()
}

console.log(`Phase 9 arena residency visual gate passed: ${cases.length} screenshots`)
