#!/usr/bin/env node
import { spawn } from 'node:child_process'
import http from 'node:http'
import https from 'node:https'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

import { renderExpectedPrompt, resolveReferenceContext } from './lib/chat-parity-harness.mjs'
import { sanitizeMessages, sanitizePath, sanitizeText, sanitizeUrl } from './lib/privacy-sanitize.mjs'

const args = parseArgs(process.argv.slice(2))
const backendBase = (args.get('backend') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const llamaBase = (args.get('llama-url') || process.env.LLAMA3_LLAMA_SERVER_URL || 'http://127.0.0.1:8183').replace(/\/$/, '')
const modelPath = resolve(args.get('model') || process.env.LLAMA3_GGUF || '$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf')
const modelId = args.get('model-id') || process.env.LLAMA3_MODEL_ID || 'llama3-small-q8'
const userMessage = args.get('message') ?? process.env.LLAMA3_CHAT_MESSAGE ?? 'hello'
const messagesJson = args.get('messages-json') || process.env.LLAMA3_CHAT_MESSAGES_JSON
const renderMode = args.get('render-mode') || process.env.LLAMA3_CHAT_RENDER_MODE || 'compact'
const maxTokens = Number.parseInt(args.get('max-tokens') || process.env.LLAMA3_CHAT_MAX_TOKENS || '1', 10)
const llamaServerBin = resolve(args.get('llama-server') || process.env.LLAMA3_LLAMA_SERVER || 'target/reference/llama.cpp/build/bin/llama-server')
const llamaTokenizeBin = resolve(args.get('llama-tokenize') || process.env.LLAMA3_LLAMA_TOKENIZE || 'target/reference/llama.cpp/build/bin/llama-tokenize')
const startLlamaServer = args.has('start-llama-server') || process.env.LLAMA3_START_LLAMA_SERVER === '1'
const diagnosticsOut = args.get('diagnostics-out') || process.env.LLAMA3_CHAT_DIAGNOSTICS_OUT
const requirePromptMatch = args.has('require-prompt-match') || process.env.LLAMA3_CHAT_REQUIRE_PROMPT_MATCH === '1'
const requireGeneratedMatch = args.has('require-generated-match') || process.env.LLAMA3_CHAT_REQUIRE_GENERATED_MATCH === '1'
const collectBackendDenseDiagnostics = args.has('backend-dense-diagnostics') || process.env.LLAMA3_CHAT_BACKEND_DENSE_DIAGNOSTICS === '1'
const backendDenseDiagnosticGeneratedIndex = parseOptionalNonNegativeInt(args.get('backend-dense-diagnostic-generated-index') || process.env.LLAMA3_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX, 'backend-dense-diagnostic-generated-index')
const sanitizeDiagnostics = args.has('sanitize-diagnostics') || process.env.LLAMA3_CHAT_SANITIZE_DIAGNOSTICS === '1'
const waitMs = Number.parseInt(args.get('wait-ms') || process.env.LLAMA3_WAIT_MS || '120000', 10)
const explicitLlamaContext = parseOptionalPositiveInt(args.get('llama-context') || process.env.LLAMA3_LLAMA_CONTEXT, 'llama-context')
const llamaFlashAttn = args.get('llama-flash-attn') || process.env.LLAMA3_LLAMA_FLASH_ATTN || 'off'
validateLlamaFlashAttn(llamaFlashAttn)

if (!Number.isInteger(maxTokens) || maxTokens < 1) {
  throw new Error(`--max-tokens must be a positive integer, got ${args.get('max-tokens')}`)
}
if (!Number.isInteger(waitMs) || waitMs < 1) {
  throw new Error(`--wait-ms must be a positive integer, got ${args.get('wait-ms')}`)
}

const messages = await loadMessages({ messagesJson, fallbackMessage: userMessage })
// Camelid's exact-row parity evidence intentionally renders the checked-in prompt
// shape itself. Use llama-server /completion with this explicit prompt instead of
// depending on server-side chat-template expansion, so compact Llama 3,
// TinyLlama marker-template, and Mistral instruct lanes stay exact. Mistral
// prompts already include <s>, so that lane sends token ids to llama-server to
// avoid a second reference-side BOS insertion.
const expectedPrompt = renderExpectedPrompt(messages, renderMode)
const referencePromptTokens = await tokenizeExpectedPrompt()
const llamaCompletionPrompt = renderMode === 'mistral_instruct' ? referencePromptTokens : expectedPrompt
const referenceContext = resolveReferenceContext({
  promptTokenCount: referencePromptTokens.length,
  maxTokens,
  explicitContext: explicitLlamaContext,
})
let child
let childSpawnError
try {
  let llamaServerArgs = null
  if (startLlamaServer) {
    const url = new URL(llamaBase)
    llamaServerArgs = [
      '--host', url.hostname,
      '--port', url.port || '8183',
      '-m', modelPath,
      '-ngl', '0',
      '-c', String(referenceContext),
      '--no-warmup',
      '-fa', llamaFlashAttn,
    ]
    child = spawn(llamaServerBin, llamaServerArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    child.once('error', err => { childSpawnError = err })
    child.stdout.on('data', chunk => process.stderr.write(`[llama-server] ${chunk}`))
    child.stderr.on('data', chunk => process.stderr.write(`[llama-server] ${chunk}`))
  }

  await waitForJson(`${backendBase}/v1/health`, {}, 'backend', waitMs)
  try {
    await waitForJson(`${llamaBase}/health`, {}, 'llama-server', waitMs)
  } catch (err) {
    if (childSpawnError?.code === 'ENOENT') {
      throw new Error(`could not start llama-server binary ${JSON.stringify(llamaServerBin)}; pass --llama-server or set LLAMA3_LLAMA_SERVER to an executable path`)
    }
    throw err
  }

  await fetchJson(`${backendBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify({ path: modelPath, id: modelId }),
  })

  const chatPayload = {
    model: modelId,
    messages,
    max_tokens: maxTokens,
    stream: false,
    temperature: 0,
  }
  const llamaCompletion = await fetchJson(`${llamaBase}/completion`, {
    method: 'POST',
    body: JSON.stringify({
      prompt: llamaCompletionPrompt,
      n_predict: maxTokens,
      temperature: 0,
      cache_prompt: false,
      n_probs: 20,
    }),
  })
  const llamaText = llamaCompletion.content ?? ''
  const llamaLogprobContent = llamaCompletion.completion_probabilities ?? []
  let llamaGeneratedTokens = llamaLogprobContent
    .map(item => Number.isInteger(item?.id) ? item.id : null)
    .filter(token => token !== null)
  if (llamaGeneratedTokens.length === 0 && llamaText.length > 0) {
    llamaGeneratedTokens = await tokenizeReferenceText(llamaText, { noBos: true })
    if (renderMode === 'mistral_instruct' && llamaText.startsWith(' ') && llamaGeneratedTokens[0] === 28705) {
      llamaGeneratedTokens = llamaGeneratedTokens.slice(1)
    }
  }
  const llamaTopLogprobs = llamaLogprobContent.flatMap(item => item?.top_logprobs ?? [])
  const diagnosticTokenIds = uniqueTokenIds([
    ...llamaGeneratedTokens,
    ...llamaTopLogprobs.map(item => item?.id),
  ]).slice(0, 16)

  const backendChat = await fetchJson(`${backendBase}/v1/chat/completions`, {
    method: 'POST',
    body: JSON.stringify({
      ...chatPayload,
      camelid_logit_token_ids: diagnosticTokenIds,
      ...(collectBackendDenseDiagnostics ? { camelid_dense_diagnostics: true } : {}),
      ...(backendDenseDiagnosticGeneratedIndex !== null ? {
        camelid_dense_diagnostics: true,
        camelid_dense_diagnostic_generated_index: backendDenseDiagnosticGeneratedIndex,
      } : {}),
    }),
  })

  const backendPromptTokens = backendChat.camelid?.prompt_token_ids || []
  const backendGeneratedTokens = backendChat.camelid?.generated_token_ids || []
  const backendTopLogits = backendChat.camelid?.top_logits || []
  const promptMatch = JSON.stringify(backendPromptTokens) === JSON.stringify(referencePromptTokens)
  const generatedTokensMatch = JSON.stringify(backendGeneratedTokens) === JSON.stringify(llamaGeneratedTokens)
  const backendText = backendChat.choices?.[0]?.message?.content ?? ''
  const textMatch = backendText === llamaText
  const firstGeneratedTokenLogitComparison = compareFirstGeneratedTokenLogits({
    backendGeneratedTokens,
    backendTopLogits,
    llamaGeneratedTokens,
    llamaLogprobContent,
  })

  const report = {
    backend: backendBase,
    llama_server: llamaBase,
    model: modelPath,
    model_id: modelId,
    message: userMessage,
    messages,
    render_mode: renderMode,
    expected_prompt: expectedPrompt,
    expected_prompt_char_count: expectedPrompt.length,
    reference_prompt_token_count: referencePromptTokens.length,
    reference_context: referenceContext,
    llama_flash_attn: llamaFlashAttn,
    llama_completion_prompt_kind: Array.isArray(llamaCompletionPrompt) ? 'tokens' : 'text',
    llama_server_args: llamaServerArgs,
    prompt_tokens_match: promptMatch,
    generated_tokens_match: generatedTokensMatch,
    generated_text_match: textMatch,
    first_generated_token_diff_index: firstArrayDifference(backendGeneratedTokens, llamaGeneratedTokens),
    backend_dense_diagnostic_generated_index: backendChat.camelid?.dense_diagnostic_generated_index ?? null,
    first_generated_token_logit_comparison: firstGeneratedTokenLogitComparison,
    first_generated_text_diff_index: firstStringDifference(backendText, llamaText),
    backend_prompt_tokens: backendPromptTokens,
    reference_prompt_tokens: referencePromptTokens,
    backend_generated_tokens: backendGeneratedTokens,
    llama_generated_tokens: llamaGeneratedTokens,
    llama_top_logprobs: llamaTopLogprobs,
    backend_diagnostic_token_ids: diagnosticTokenIds,
    backend_text: backendText,
    llama_text: llamaText,
    backend_usage: backendChat.usage,
    llama_usage: llamaCompletion.timings,
    camelid: backendChat.camelid,
  }

  console.log(`backend=${sanitizeDiagnostics ? 'redacted' : backendBase}`)
  console.log(`llama_server=${sanitizeDiagnostics ? 'redacted' : llamaBase}`)
  console.log(`model=${sanitizeDiagnostics ? JSON.stringify(sanitizePath(modelPath)) : modelPath}`)
  console.log(`message=${sanitizeDiagnostics ? JSON.stringify(sanitizeText(userMessage)) : JSON.stringify(userMessage)}`)
  console.log(`messages=${sanitizeDiagnostics ? JSON.stringify(sanitizeMessages(messages)) : JSON.stringify(messages)}`)
  console.log(`render_mode=${renderMode}`)
  console.log(`expected_prompt=${sanitizeDiagnostics ? JSON.stringify(sanitizeText(expectedPrompt)) : JSON.stringify(expectedPrompt)}`)
  console.log(`expected_prompt_char_count=${expectedPrompt.length}`)
  console.log(`reference_prompt_token_count=${referencePromptTokens.length}`)
  console.log(`reference_context=${referenceContext}`)
  console.log(`llama_flash_attn=${llamaFlashAttn}`)
  console.log(`llama_completion_prompt_kind=${Array.isArray(llamaCompletionPrompt) ? 'tokens' : 'text'}`)
  console.log(`backend_prompt_tokens=${JSON.stringify(backendPromptTokens)}`)
  console.log(`reference_prompt_tokens=${JSON.stringify(referencePromptTokens)}`)
  console.log(`prompt_tokens_match=${promptMatch}`)
  console.log(`backend_generated_tokens=${JSON.stringify(backendGeneratedTokens)}`)
  console.log(`llama_generated_tokens=${JSON.stringify(llamaGeneratedTokens)}`)
  console.log(`generated_tokens_match=${generatedTokensMatch}`)
  console.log(`backend_dense_diagnostic_generated_index=${backendChat.camelid?.dense_diagnostic_generated_index ?? null}`)
  console.log(`first_generated_token_logit_comparison=${JSON.stringify(sanitizeDiagnostics ? stripComparisonTokenText(firstGeneratedTokenLogitComparison) : firstGeneratedTokenLogitComparison)}`)
  console.log(`backend_text=${sanitizeDiagnostics ? JSON.stringify(sanitizeText(backendText)) : JSON.stringify(backendText)}`)
  console.log(`llama_text=${sanitizeDiagnostics ? JSON.stringify(sanitizeText(llamaText)) : JSON.stringify(llamaText)}`)
  console.log(`generated_text_match=${textMatch}`)
  console.log(`backend_usage=${JSON.stringify(backendChat.usage)}`)
  console.log(`llama_usage=${JSON.stringify(llamaCompletion.timings)}`)

  if (diagnosticsOut) {
    const diagnosticsPath = resolve(diagnosticsOut)
    await mkdir(dirname(diagnosticsPath), { recursive: true })
    await writeFile(diagnosticsPath, `${JSON.stringify(sanitizeDiagnostics ? sanitizeParityReport(report) : report, null, 2)}\n`)
    console.log(`diagnostics_out=${sanitizeDiagnostics ? JSON.stringify(sanitizePath(diagnosticsPath)) : diagnosticsPath}`)
  }

  if (requirePromptMatch && !promptMatch) process.exitCode = 1
  if (requireGeneratedMatch && !generatedTokensMatch) process.exitCode = 1
} finally {
  if (child) child.kill('SIGTERM')
}


async function loadMessages({ messagesJson, fallbackMessage }) {
  if (!messagesJson) return [{ role: 'user', content: fallbackMessage }]
  const raw = JSON.parse(await readFile(resolve(messagesJson), 'utf8'))
  const messages = Array.isArray(raw) ? raw : raw.messages
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error(`${messagesJson} must contain a non-empty messages array`)
  }
  return messages.map((message, index) => {
    const role = String(message?.role ?? '').trim()
    const content = String(message?.content ?? '')
    if (!role) throw new Error(`${messagesJson} messages[${index}].role must not be empty`)
    if (content.length === 0) throw new Error(`${messagesJson} messages[${index}].content must not be empty`)
    return { role, content }
  })
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++i] : 'true')
    parsed.set(key, value)
  }
  return parsed
}

async function tokenizeExpectedPrompt() {
  return tokenizeReferenceText(expectedPrompt, { noBos: renderMode === 'mistral_instruct' })
}

async function tokenizeReferenceText(text, { noBos = false } = {}) {
  const llamaTokenizeArgs = [
    '-m', modelPath,
    '--ids',
    '--log-disable',
    '-p', text,
  ]
  if (noBos) llamaTokenizeArgs.push('--no-bos')
  const { stdout } = await run(llamaTokenizeBin, llamaTokenizeArgs)
  return JSON.parse(stdout.trim())
}

function parseOptionalPositiveInt(value, name) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`--${name} must be a positive integer, got ${value}`)
  }
  return parsed
}

function parseOptionalNonNegativeInt(value, name) {
  if (value === undefined || value === null || value === '') return null
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`--${name} must be a non-negative integer, got ${value}`)
  }
  return parsed
}

function validateLlamaFlashAttn(value) {
  if (!['off', 'on', 'auto'].includes(value)) {
    throw new Error(`--llama-flash-attn must be off, on, or auto, got ${value}`)
  }
}

async function fetchJson(url, options = {}) {
  const { timeoutMs = waitMs, headers = {}, body, method = body ? 'POST' : 'GET' } = options
  const target = new URL(url)
  const transport = target.protocol === 'https:' ? https : http
  const requestBody = typeof body === 'string' || Buffer.isBuffer(body) ? body : body ? JSON.stringify(body) : null

  return new Promise((resolvePromise, reject) => {
    const request = transport.request(target, {
      method,
      headers: {
        'content-type': 'application/json',
        ...(requestBody ? { 'content-length': Buffer.byteLength(requestBody) } : {}),
        ...headers,
      },
      timeout: timeoutMs,
    }, response => {
      let text = ''
      response.setEncoding('utf8')
      response.on('data', chunk => { text += chunk })
      response.on('end', () => {
        let parsed = null
        try {
          parsed = text ? JSON.parse(text) : null
        } catch (err) {
          reject(new Error(`${url}: invalid JSON response: ${err.message}: ${text.slice(0, 500)}`))
          return
        }
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`${url}: ${response.statusCode} ${response.statusMessage}: ${parsed?.error?.message || text}`))
          return
        }
        resolvePromise(parsed)
      })
    })
    request.on('timeout', () => {
      request.destroy(new Error(`${url}: request timed out after ${timeoutMs} ms`))
    })
    request.on('error', reject)
    if (requestBody) request.write(requestBody)
    request.end()
  })
}

async function waitForJson(url, options, label, waitMs) {
  const deadline = Date.now() + waitMs
  let lastError
  while (Date.now() < deadline) {
    try {
      return await fetchJson(url, { ...options, timeoutMs: Math.min(waitMs, 5000) })
    } catch (err) {
      lastError = err
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
  throw new Error(`${label} did not become reachable at ${url} within ${waitMs} ms: ${lastError?.message}`)
}

async function run(command, commandArgs) {
  return new Promise((resolvePromise, reject) => {
    const childProcess = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    childProcess.stdout.on('data', chunk => { stdout += chunk })
    childProcess.stderr.on('data', chunk => { stderr += chunk })
    childProcess.once('error', reject)
    childProcess.once('close', code => {
      if (code === 0) {
        resolvePromise({ stdout, stderr })
      } else {
        reject(new Error(`${command} exited ${code}: ${stderr || stdout}`))
      }
    })
  })
}

function uniqueTokenIds(ids) {
  const out = []
  for (const id of ids) {
    if (Number.isInteger(id) && !out.includes(id)) out.push(id)
  }
  return out
}

function compareFirstGeneratedTokenLogits({ backendGeneratedTokens, backendTopLogits, llamaGeneratedTokens, llamaLogprobContent }) {
  const backendTokenId = backendGeneratedTokens[0]
  const llamaTokenId = llamaGeneratedTokens[0]
  if (!Number.isInteger(backendTokenId) || !Number.isInteger(llamaTokenId)) return null

  const backendByToken = new Map((backendTopLogits || []).map(row => [row.token_id, row]))
  const llamaFirstTopLogprobs = llamaLogprobContent?.[0]?.top_logprobs || []
  const llamaByToken = new Map(llamaFirstTopLogprobs.map(row => [row.id, row]))
  const backendForBackendToken = backendByToken.get(backendTokenId) || null
  const backendForLlamaToken = backendByToken.get(llamaTokenId) || null
  const llamaForBackendToken = llamaByToken.get(backendTokenId) || null
  const llamaForLlamaToken = llamaByToken.get(llamaTokenId) || null

  const backendMarginLlamaMinusBackend = Number.isFinite(backendForLlamaToken?.logit) && Number.isFinite(backendForBackendToken?.logit)
    ? backendForLlamaToken.logit - backendForBackendToken.logit
    : null
  const llamaMarginLlamaMinusBackend = Number.isFinite(llamaForLlamaToken?.logprob) && Number.isFinite(llamaForBackendToken?.logprob)
    ? llamaForLlamaToken.logprob - llamaForBackendToken.logprob
    : null

  return {
    backend_token_id: backendTokenId,
    llama_token_id: llamaTokenId,
    token_ids_match: backendTokenId === llamaTokenId,
    backend_margin_llama_minus_backend: backendMarginLlamaMinusBackend,
    llama_margin_llama_minus_backend: llamaMarginLlamaMinusBackend,
    margin_disagreement: Number.isFinite(backendMarginLlamaMinusBackend) && Number.isFinite(llamaMarginLlamaMinusBackend)
      ? llamaMarginLlamaMinusBackend - backendMarginLlamaMinusBackend
      : null,
    backend_rows: [backendForBackendToken, backendForLlamaToken].filter(Boolean),
    llama_rows: [llamaForBackendToken, llamaForLlamaToken].filter(Boolean),
  }
}

function sanitizeParityReport(report) {
  const camelid = report.camelid ? {
    prompt_token_ids: report.camelid.prompt_token_ids ?? [],
    generated_token_ids: report.camelid.generated_token_ids ?? [],
    dense_metadata: report.camelid.dense_metadata ?? null,
    dense_diagnostic_generated_index: report.camelid.dense_diagnostic_generated_index ?? null,
    top_logits: stripDiagnosticTokenText(report.camelid.top_logits ?? []),
    step_top_logits: (report.camelid.step_top_logits ?? []).map(stripDiagnosticTokenText),
    output_projection: report.camelid.output_projection ?? [],
    dense: report.camelid.dense ?? null,
  } : null
  return {
    schema: 'camelid.chat-parity-sanitized.v1',
    backend: sanitizeUrl(report.backend),
    llama_server: sanitizeUrl(report.llama_server),
    model: sanitizePath(report.model),
    model_id: 'redacted',
    message: sanitizeText(report.message),
    messages: sanitizeMessages(report.messages),
    render_mode: report.render_mode,
    expected_prompt: sanitizeText(report.expected_prompt),
    expected_prompt_char_count: report.expected_prompt_char_count,
    reference_prompt_token_count: report.reference_prompt_token_count,
    reference_context: report.reference_context,
    llama_flash_attn: report.llama_flash_attn,
    llama_completion_prompt_kind: report.llama_completion_prompt_kind,
    prompt_tokens_match: report.prompt_tokens_match,
    generated_tokens_match: report.generated_tokens_match,
    generated_text_match: report.generated_text_match,
    first_generated_token_diff_index: report.first_generated_token_diff_index,
    first_generated_text_diff_index: report.first_generated_text_diff_index,
    backend_dense_diagnostic_generated_index: report.backend_dense_diagnostic_generated_index,
    first_generated_token_logit_comparison: stripComparisonTokenText(report.first_generated_token_logit_comparison),
    backend_prompt_tokens: report.backend_prompt_tokens,
    reference_prompt_tokens: report.reference_prompt_tokens,
    prompt_position_ids: positionIds(report.backend_prompt_tokens),
    backend_generated_tokens: report.backend_generated_tokens,
    llama_generated_tokens: report.llama_generated_tokens,
    generated_position_ids: positionIds(
      report.backend_generated_tokens,
      Array.isArray(report.backend_prompt_tokens) ? report.backend_prompt_tokens.length : 0,
    ),
    llama_top_logprobs: stripLogprobTokenText(report.llama_top_logprobs),
    backend_diagnostic_token_ids: report.backend_diagnostic_token_ids,
    backend_text: sanitizeText(report.backend_text),
    llama_text: sanitizeText(report.llama_text),
    backend_usage: report.backend_usage,
    llama_usage: report.llama_usage,
    camelid,
  }
}

function stripComparisonTokenText(value) {
  if (!value) return null
  return {
    ...value,
    backend_rows: stripDiagnosticTokenText(value.backend_rows ?? []),
    llama_rows: stripLogprobTokenText(value.llama_rows ?? []),
  }
}

function stripDiagnosticTokenText(rows) {
  return rows.map(({ text, ...row }) => row)
}

function stripLogprobTokenText(rows) {
  return rows.map(({ token, ...row }) => row)
}

function positionIds(tokens, start = 0) {
  return Array.isArray(tokens) ? tokens.map((_, index) => start + index) : null
}

function firstStringDifference(left, right) {
  const max = Math.max(left.length, right.length)
  for (let i = 0; i < max; i += 1) {
    if (left[i] !== right[i]) return i
  }
  return -1
}

function firstArrayDifference(left, right) {
  const max = Math.max(left.length, right.length)
  for (let i = 0; i < max; i += 1) {
    if (left[i] !== right[i]) return i
  }
  return -1
}
