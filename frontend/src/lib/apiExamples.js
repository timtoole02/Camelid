/* API workbench example generators (Phase 5).

   Every example is pre-filled with the live API base and the loaded model id
   and must work verbatim when pasted into a terminal/script — that is the
   Phase 5 gate. Generation examples mirror the chat surface's request shape
   (greedy temperature=0); the UI gates their try-it buttons exactly like chat.

   Brand note (deliberate, see DESIGN_LOG Phase 5): code samples may name the
   third-party SDK class they instantiate — that is technical compatibility
   content, not product copy. UI copy itself never names competitor brands, and
   this module is excluded from the brand sweep for exactly this line. */

const json = (value) => JSON.stringify(value, null, 2)

export function chatCompletionsBody(modelId) {
  return {
    model: modelId,
    messages: [{ role: 'user', content: 'Reply with one short sentence.' }],
    temperature: 0,
    max_tokens: 64,
    stream: true,
  }
}

export function completionsBody(modelId) {
  return {
    model: modelId,
    prompt: 'Camelid is',
    temperature: 0,
    max_tokens: 32,
  }
}

export function tokenizerEncodeBody() {
  return { text: 'Hello Camelid', add_special: true }
}

export function embeddingsBody(modelId) {
  return {
    model: modelId,
    input: 'search_query: Which Camelid files implement semantic retrieval?',
    dimensions: 256,
  }
}

function curlGet(apiBase, path) {
  return `curl ${apiBase}${path}`
}

function curlPost(apiBase, path, body) {
  return `curl ${apiBase}${path} \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(body)}'`
}

function pythonSdk(apiBase, modelId, kind) {
  if (kind === 'chat') {
    return [
      'from openai import OpenAI',
      '',
      `client = OpenAI(base_url="${apiBase}/v1", api_key="not-needed-locally")`,
      'stream = client.chat.completions.create(',
      `    model="${modelId}",`,
      '    messages=[{"role": "user", "content": "Reply with one short sentence."}],',
      '    temperature=0,',
      '    max_tokens=64,',
      '    stream=True,',
      ')',
      'for chunk in stream:',
      '    print(chunk.choices[0].delta.content or "", end="", flush=True)',
    ].join('\n')
  }
  if (kind === 'completions') {
    return [
      'from openai import OpenAI',
      '',
      `client = OpenAI(base_url="${apiBase}/v1", api_key="not-needed-locally")`,
      'result = client.completions.create(',
      `    model="${modelId}",`,
      '    prompt="Camelid is",',
      '    temperature=0,',
      '    max_tokens=32,',
      ')',
      'print(result.choices[0].text)',
    ].join('\n')
  }
  if (kind === 'models') {
    return [
      'from openai import OpenAI',
      '',
      `client = OpenAI(base_url="${apiBase}/v1", api_key="not-needed-locally")`,
      'for model in client.models.list():',
      '    print(model.id)',
    ].join('\n')
  }
  if (kind === 'embeddings') {
    return [
      'from openai import OpenAI',
      '',
      `client = OpenAI(base_url="${apiBase}/v1", api_key="not-needed-locally")`,
      'result = client.embeddings.create(',
      `    model="${modelId}",`,
      '    input="search_query: Which Camelid files implement semantic retrieval?",',
      '    dimensions=256,',
      ')',
      'print(len(result.data[0].embedding))',
    ].join('\n')
  }
  return null
}

function pythonRequests(apiBase, path, body = null) {
  if (!body) {
    return ['import requests', '', `print(requests.get("${apiBase}${path}").json())`].join('\n')
  }
  return [
    'import requests',
    '',
    `response = requests.post("${apiBase}${path}", json=${JSON.stringify(body)})`,
    'print(response.json())',
  ].join('\n')
}

function jsFetch(apiBase, path, body = null, sse = false) {
  if (!body) {
    return [
      `const response = await fetch("${apiBase}${path}")`,
      'console.log(await response.json())',
    ].join('\n')
  }
  if (sse) {
    return [
      `const response = await fetch("${apiBase}${path}", {`,
      '  method: "POST",',
      '  headers: { "Content-Type": "application/json" },',
      `  body: JSON.stringify(${json(body)}),`,
      '})',
      'const reader = response.body.getReader()',
      'const decoder = new TextDecoder()',
      'for (;;) {',
      '  const { done, value } = await reader.read()',
      '  if (done) break',
      '  process.stdout.write(decoder.decode(value))',
      '}',
    ].join('\n')
  }
  return [
    `const response = await fetch("${apiBase}${path}", {`,
    '  method: "POST",',
    '  headers: { "Content-Type": "application/json" },',
    `  body: JSON.stringify(${json(body)}),`,
    '})',
    'console.log(await response.json())',
  ].join('\n')
}

/* The workbench endpoint table. `gate` values:
   - 'none'      try-it runs whenever the backend answers (read-only routes)
   - 'tokenizer' runs whenever a tokenizer is loaded (typed error otherwise)
   - 'chat'      gated exactly like chat: loaded_now + generation_ready +
                 active_model_id + exact supported row (I1)
   - 'blocked'   fail-closed backend route; never runnable, typed copy only */
export function workbenchEndpoints({ apiBase, modelId }) {
  const base = apiBase || 'http://127.0.0.1:8181'
  const model = modelId || '<loaded-model-id>'
  return [
    {
      id: 'v1_health',
      method: 'GET',
      path: '/v1/health',
      gate: 'none',
      summary: 'Runtime truth: loaded_now, generation_ready, active_model_id — the gate inputs.',
      examples: {
        curl: curlGet(base, '/v1/health'),
        python: pythonRequests(base, '/v1/health'),
        js: jsFetch(base, '/v1/health'),
      },
    },
    {
      id: 'v1_models',
      method: 'GET',
      path: '/v1/models',
      gate: 'none',
      summary: 'Lists the active runtime model. Descriptive meta only — never a support catalog.',
      examples: {
        curl: curlGet(base, '/v1/models'),
        python: pythonSdk(base, model, 'models'),
        js: jsFetch(base, '/v1/models'),
      },
    },
    {
      id: 'v1_chat_completions',
      method: 'POST',
      path: '/v1/chat/completions',
      gate: 'chat',
      sse: true,
      summary: 'The chat surface. Streaming SSE; same request shape the UI sends.',
      body: chatCompletionsBody(model),
      examples: {
        curl: curlPost(base, '/v1/chat/completions', chatCompletionsBody(model)),
        python: pythonSdk(base, model, 'chat'),
        js: jsFetch(base, '/v1/chat/completions', chatCompletionsBody(model), true),
      },
    },
    {
      id: 'v1_completions',
      method: 'POST',
      path: '/v1/completions',
      gate: 'chat',
      summary: 'Raw text completion. The backend route answers for any loaded model, but this workbench keeps generation examples gated to the exact supported row, same as chat.',
      body: completionsBody(model),
      examples: {
        curl: curlPost(base, '/v1/completions', completionsBody(model)),
        python: pythonSdk(base, model, 'completions'),
        js: jsFetch(base, '/v1/completions', completionsBody(model)),
      },
    },
    {
      id: 'api_capabilities',
      method: 'GET',
      path: '/api/capabilities',
      gate: 'none',
      summary: 'The support contract — what the evidence ledger renders.',
      examples: {
        curl: curlGet(base, '/api/capabilities'),
        python: pythonRequests(base, '/api/capabilities'),
        js: jsFetch(base, '/api/capabilities'),
      },
    },
    {
      id: 'api_models_tokenizer_encode',
      method: 'POST',
      path: '/api/models/tokenizer/encode',
      gate: 'tokenizer',
      summary: 'Tokenize text with the loaded tokenizer (feature row tokenizer_encode_decode).',
      body: tokenizerEncodeBody(),
      examples: {
        curl: curlPost(base, '/api/models/tokenizer/encode', tokenizerEncodeBody()),
        python: pythonRequests(base, '/api/models/tokenizer/encode', tokenizerEncodeBody()),
        js: jsFetch(base, '/api/models/tokenizer/encode', tokenizerEncodeBody()),
      },
    },
    {
      id: 'v1_embeddings',
      method: 'POST',
      path: '/v1/embeddings',
      gate: 'none',
      featureRowId: 'openai_embeddings',
      summary: 'Float embeddings for the exact supported Nomic encoder row. A typed model_not_loaded response is expected until that row is registered.',
      body: embeddingsBody(model),
      examples: {
        curl: curlPost(base, '/v1/embeddings', embeddingsBody(model)),
        python: pythonSdk(base, model, 'embeddings'),
        js: jsFetch(base, '/v1/embeddings', embeddingsBody(model)),
      },
    },
    {
      id: 'v1_responses',
      method: 'POST',
      path: '/v1/responses',
      gate: 'blocked',
      featureRowId: 'fail_closed_native_compatibility_routes',
      summary: 'Fail-closed: the responses surface has no contract; the route returns a typed error instead of pretending.',
    },
    {
      id: 'v1_messages',
      method: 'POST',
      path: '/v1/messages',
      gate: 'blocked',
      featureRowId: 'fail_closed_native_compatibility_routes',
      summary: 'Fail-closed: the messages surface has no contract; the route returns a typed error instead of pretending.',
    },
  ]
}
