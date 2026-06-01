#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { resolve } from 'node:path'

const argv = process.argv.slice(2)
const relayArgs = [...argv]
if (!hasFlag(argv, 'render-mode')) relayArgs.push('--render-mode', 'mistral_instruct')
if (!hasFlag(argv, 'model') && process.env.MIXTRAL_GGUF) relayArgs.push('--model', process.env.MIXTRAL_GGUF)
if (!hasFlag(argv, 'model-id') && process.env.MIXTRAL_MODEL_ID) relayArgs.push('--model-id', process.env.MIXTRAL_MODEL_ID)
if (!hasFlag(argv, 'llama-url') && process.env.MIXTRAL_LLAMA_SERVER_URL) relayArgs.push('--llama-url', process.env.MIXTRAL_LLAMA_SERVER_URL)
if (!hasFlag(argv, 'llama-server') && process.env.MIXTRAL_LLAMA_SERVER) relayArgs.push('--llama-server', process.env.MIXTRAL_LLAMA_SERVER)
if (!hasFlag(argv, 'llama-tokenize') && process.env.MIXTRAL_LLAMA_TOKENIZE) relayArgs.push('--llama-tokenize', process.env.MIXTRAL_LLAMA_TOKENIZE)
if (!hasFlag(argv, 'messages-json') && process.env.MIXTRAL_CHAT_MESSAGES_JSON) relayArgs.push('--messages-json', process.env.MIXTRAL_CHAT_MESSAGES_JSON)
if (!hasFlag(argv, 'message') && process.env.MIXTRAL_CHAT_MESSAGE) relayArgs.push('--message', process.env.MIXTRAL_CHAT_MESSAGE)
if (!hasFlag(argv, 'max-tokens') && process.env.MIXTRAL_CHAT_MAX_TOKENS) relayArgs.push('--max-tokens', process.env.MIXTRAL_CHAT_MAX_TOKENS)
if (!hasFlag(argv, 'diagnostics-out') && process.env.MIXTRAL_CHAT_DIAGNOSTICS_OUT) relayArgs.push('--diagnostics-out', process.env.MIXTRAL_CHAT_DIAGNOSTICS_OUT)
if (!hasFlag(argv, 'sanitize-diagnostics')) relayArgs.push('--sanitize-diagnostics')
if (!hasFlag(argv, 'backend-dense-diagnostic-generated-index') && process.env.MIXTRAL_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX) relayArgs.push('--backend-dense-diagnostic-generated-index', process.env.MIXTRAL_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX)
if (!hasFlag(argv, 'llama-context') && process.env.MIXTRAL_LLAMA_CONTEXT) relayArgs.push('--llama-context', process.env.MIXTRAL_LLAMA_CONTEXT)
if (!hasFlag(argv, 'wait-ms') && process.env.MIXTRAL_WAIT_MS) relayArgs.push('--wait-ms', process.env.MIXTRAL_WAIT_MS)

const child = spawn(process.execPath, [resolve('scripts/chat-parity-llama3.mjs'), ...relayArgs], {
  stdio: 'inherit',
  env: {
    ...process.env,
    LLAMA3_GGUF: process.env.LLAMA3_GGUF || process.env.MIXTRAL_GGUF,
    LLAMA3_MODEL_ID: process.env.LLAMA3_MODEL_ID || process.env.MIXTRAL_MODEL_ID,
    LLAMA3_LLAMA_SERVER_URL: process.env.LLAMA3_LLAMA_SERVER_URL || process.env.MIXTRAL_LLAMA_SERVER_URL,
    LLAMA3_LLAMA_SERVER: process.env.LLAMA3_LLAMA_SERVER || process.env.MIXTRAL_LLAMA_SERVER,
    LLAMA3_LLAMA_TOKENIZE: process.env.LLAMA3_LLAMA_TOKENIZE || process.env.MIXTRAL_LLAMA_TOKENIZE,
    LLAMA3_CHAT_MESSAGES_JSON: process.env.LLAMA3_CHAT_MESSAGES_JSON || process.env.MIXTRAL_CHAT_MESSAGES_JSON,
    LLAMA3_CHAT_MESSAGE: process.env.LLAMA3_CHAT_MESSAGE || process.env.MIXTRAL_CHAT_MESSAGE,
    LLAMA3_CHAT_MAX_TOKENS: process.env.LLAMA3_CHAT_MAX_TOKENS || process.env.MIXTRAL_CHAT_MAX_TOKENS,
    LLAMA3_CHAT_DIAGNOSTICS_OUT: process.env.LLAMA3_CHAT_DIAGNOSTICS_OUT || process.env.MIXTRAL_CHAT_DIAGNOSTICS_OUT,
    LLAMA3_CHAT_SANITIZE_DIAGNOSTICS: process.env.LLAMA3_CHAT_SANITIZE_DIAGNOSTICS || '1',
    LLAMA3_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX: process.env.LLAMA3_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX || process.env.MIXTRAL_CHAT_BACKEND_DENSE_DIAGNOSTIC_GENERATED_INDEX,
    LLAMA3_LLAMA_CONTEXT: process.env.LLAMA3_LLAMA_CONTEXT || process.env.MIXTRAL_LLAMA_CONTEXT,
    LLAMA3_WAIT_MS: process.env.LLAMA3_WAIT_MS || process.env.MIXTRAL_WAIT_MS,
  },
})

child.once('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal)
  else process.exit(code ?? 0)
})
child.once('error', (err) => {
  console.error(err)
  process.exit(1)
})

function hasFlag(args, flag) {
  return args.some((arg) => arg === `--${flag}` || arg.startsWith(`--${flag}=`))
}
