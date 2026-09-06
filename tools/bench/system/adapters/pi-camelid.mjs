import { readFile } from 'node:fs/promises'
import { isAbsolute } from 'node:path'

export const PINNED_PI_RELEASE = Object.freeze({
  version: '0.84.3',
  sourceCommit: '4e58f324fae8ebfa98a3d45181fb248072a2afac',
  archiveName: 'pi-linux-x64.tar.gz',
  archiveSizeBytes: 42458773,
  archiveSha256: '6f8bb67c21bc6b8a8a106d354f56d7fd4a190a3cd8ad3a32db45f6d281a5d008',
  executableSha256: 'ca858fde375ab91531353b22fac6ebdf29c0a153efe754f5f9b8a72a7423ed08',
  releaseUrl: 'https://github.com/earendil-works/pi/releases/tag/v0.84.3',
})

export const PI_EVENT_STREAM_VERSION = 3
export const PI_PROVIDER_ID = 'camelid-benchmark'
export const PI_SHARED_TOOLS = Object.freeze(['read', 'bash', 'edit', 'write', 'ls'])
export const PI_PROVIDER_EXTENSION_PATH = '/opt/controller/pi-camelid-provider.mjs'
export const PI_BENCHMARK_SYSTEM_PROMPT = [
  'Benchmark execution rules:',
  '- Inspect the workspace with available tools before editing.',
  '- Read a target file and nearby conventions before changing it.',
  '- Never invent workspace facts, file paths, APIs, or source text. Look first.',
  '- If a tool reports an error, inspect the workspace and continue unless genuinely blocked.',
  '- Verify changes with the relevant test or check before claiming completion.',
  '- Keep working until the goal is met or you are genuinely blocked.',
].join('\n')

export function piProviderConfig({ baseUrl, modelId, contextWindow, maxTokens }) {
  assertLoopbackV1Url(baseUrl)
  assertNonEmpty(modelId, 'modelId')
  assertPositiveInteger(contextWindow, 'contextWindow')
  assertPositiveInteger(maxTokens, 'maxTokens')
  if (maxTokens > contextWindow) throw new TypeError('maxTokens cannot exceed contextWindow')

  return {
    providers: {
      [PI_PROVIDER_ID]: {
        name: 'Camelid benchmark',
        baseUrl,
        api: 'openai-completions',
        apiKey: '$CAMELID_PI_API_KEY',
        authHeader: true,
        compat: {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsUsageInStreaming: true,
          supportsFinishReason: true,
          supportsStrictMode: false,
          supportsOpenAIGrammarTools: false,
          maxTokensField: 'max_tokens',
          requiresToolResultName: false,
          requiresAssistantAfterToolResult: false,
        },
        models: [{
          id: modelId,
          name: modelId,
          reasoning: false,
          input: ['text'],
          contextWindow,
          maxTokens,
          samplingParams: { temperature: 0 },
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        }],
      },
    },
  }
}

export function piJsonArgs({ modelId, goal, extensionPath = PI_PROVIDER_EXTENSION_PATH }) {
  assertNonEmpty(modelId, 'modelId')
  assertNonEmpty(goal, 'goal')
  return [
    '--mode', 'json',
    '--provider', PI_PROVIDER_ID,
    '--model', modelId,
    '--thinking', 'off',
    '--append-system-prompt', PI_BENCHMARK_SYSTEM_PROMPT,
    '--no-session',
    '--no-approve',
    '--no-extensions',
    '--extension', extensionPath,
    '--no-skills',
    '--no-prompt-templates',
    '--no-themes',
    '--no-context-files',
    '--tools', PI_SHARED_TOOLS.join(','),
    '--',
    goal,
  ]
}

export function piSandboxEnvironment({
  apiKey,
  agentDir = '/tmp/pi-agent',
  home = '/tmp/home',
  path = '/usr/bin:/bin',
  systemRoot = null,
  temp = null,
}) {
  assertNonEmpty(apiKey, 'apiKey')
  assertAbsolute(agentDir, 'agentDir')
  assertAbsolute(home, 'home')
  assertNonEmpty(path, 'path')
  return compactObject({
    PATH: path,
    HOME: home,
    PI_CODING_AGENT_DIR: agentDir,
    PI_OFFLINE: '1',
    PI_SKIP_VERSION_CHECK: '1',
    PI_TELEMETRY: '0',
    CAMELID_PI_API_KEY: apiKey,
    SYSTEMROOT: systemRoot,
    TEMP: temp,
    TMP: temp,
  })
}

export async function parsePiJsonFile(path) {
  return parsePiJsonLines(await readFile(path, 'utf8'))
}

export function parsePiJsonLines(text) {
  if (typeof text !== 'string' || text.length === 0) throw new PiEventError('Pi JSON stream is empty')
  const lines = text.split(/\r?\n/)
  if (lines.at(-1) === '') lines.pop()
  if (lines.length === 0 || lines.some((line) => line.trim().length === 0)) {
    throw new PiEventError('Pi JSON stream contains an empty record')
  }

  const events = lines.map((line, index) => parseEvent(line, index))
  const session = events[0]
  if (session.type !== 'session') throw new PiEventError('Pi JSON stream must start with a session record')
  if (session.version !== PI_EVENT_STREAM_VERSION) {
    throw new PiEventError(`unsupported Pi JSON stream version ${String(session.version)}`)
  }
  if (events.slice(1).some((event) => event.type === 'session')) {
    throw new PiEventError('Pi JSON stream contains more than one session record')
  }

  let agentStarts = 0
  let agentEnds = 0
  let agentEndIndex = -1
  let modelSteps = 0
  let toolCalls = 0
  let toolErrors = 0
  let currentAssistantUsage = null
  let inputTokens = 0
  let outputTokens = 0
  let usageComplete = true
  const activeTools = new Set()
  const completedTools = new Set()

  for (const [index, event] of events.slice(1).entries()) {
    if (event.type === 'agent_start') agentStarts += 1
    if (event.type === 'agent_end') {
      agentEnds += 1
      agentEndIndex = index + 1
    }
    if (event.type === 'message_start' && event.message?.role === 'assistant') currentAssistantUsage = null
    if (event.type === 'message_update' && event.usage !== undefined) {
      currentAssistantUsage = validateUsage(event.usage)
    }
    if (event.type === 'message_end' && event.message?.role === 'assistant') {
      modelSteps += 1
      const usage = event.message.usage === undefined
        ? currentAssistantUsage
        : validateUsage(event.message.usage)
      if (usage === null) {
        usageComplete = false
      } else {
        inputTokens += usage.input
        outputTokens += usage.output
      }
      currentAssistantUsage = null
    }
    if (event.type === 'tool_execution_start') {
      const id = toolCallId(event)
      if (activeTools.has(id) || completedTools.has(id)) throw new PiEventError(`duplicate Pi tool call ${id}`)
      activeTools.add(id)
      toolCalls += 1
    }
    if (event.type === 'tool_execution_update' && !activeTools.has(toolCallId(event))) {
      throw new PiEventError(`Pi tool update has no matching start: ${toolCallId(event)}`)
    }
    if (event.type === 'tool_execution_end') {
      const id = toolCallId(event)
      if (!activeTools.delete(id)) throw new PiEventError(`Pi tool end has no matching start: ${id}`)
      completedTools.add(id)
      if (event.isError === true) toolErrors += 1
    }
  }

  if (agentStarts !== 1) throw new PiEventError(`Pi JSON stream must contain one agent_start, got ${agentStarts}`)
  if (agentEnds !== 1) throw new PiEventError(`Pi JSON stream must contain one agent_end, got ${agentEnds}`)
  const afterAgentEnd = events.slice(agentEndIndex + 1)
  if (afterAgentEnd.some((event) => event.type !== 'agent_settled')) {
    throw new PiEventError('Pi JSON stream contains work after agent_end')
  }
  if (activeTools.size > 0) throw new PiEventError('Pi JSON stream ended with unfinished tool calls')

  return {
    session,
    events,
    summary: {
      model_steps: modelSteps,
      tool_calls: toolCalls,
      tool_errors: toolErrors,
      input_tokens: usageComplete ? inputTokens : null,
      output_tokens: usageComplete ? outputTokens : null,
    },
  }
}

export class PiEventError extends Error {
  constructor(message) {
    super(message)
    this.name = 'PiEventError'
  }
}

function assertLoopbackV1Url(value) {
  let url
  try {
    url = new URL(value)
  } catch {
    throw new TypeError('baseUrl must be an absolute URL')
  }
  if (url.protocol !== 'http:'
    || url.hostname !== '127.0.0.1'
    || url.port.length === 0
    || url.pathname !== '/v1'
    || url.username.length > 0
    || url.password.length > 0
    || url.search.length > 0
    || url.hash.length > 0) {
    throw new TypeError('baseUrl must be an exact http://127.0.0.1:<port>/v1 URL')
  }
}

function assertNonEmpty(value, name) {
  if (typeof value !== 'string' || value.length === 0) throw new TypeError(`${name} must be a non-empty string`)
}

function assertPositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`${name} must be a positive safe integer`)
}

function assertAbsolute(value, name) {
  if (typeof value !== 'string' || !isAbsolute(value)) throw new TypeError(`${name} must be an absolute path`)
}

function compactObject(value) {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== null && item !== undefined))
}

function parseEvent(line, index) {
  let event
  try {
    event = JSON.parse(line)
  } catch (error) {
    throw new PiEventError(`Pi JSON record ${index + 1} is invalid: ${error.message}`)
  }
  if (event === null || typeof event !== 'object' || Array.isArray(event)) {
    throw new PiEventError(`Pi JSON record ${index + 1} must be an object`)
  }
  if (typeof event.type !== 'string' || event.type.length === 0) {
    throw new PiEventError(`Pi JSON record ${index + 1} has no event type`)
  }
  return event
}

function toolCallId(event) {
  if (typeof event.toolCallId !== 'string' || event.toolCallId.length === 0) {
    throw new PiEventError(`${event.type} has no toolCallId`)
  }
  return event.toolCallId
}

function validateUsage(usage) {
  if (usage === null || typeof usage !== 'object' || Array.isArray(usage)) {
    throw new PiEventError('Pi usage must be an object')
  }
  for (const field of ['input', 'output']) {
    if (!Number.isSafeInteger(usage[field]) || usage[field] < 0) {
      throw new PiEventError(`Pi usage.${field} must be a non-negative safe integer`)
    }
  }
  return usage
}