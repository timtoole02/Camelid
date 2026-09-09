// Separates the two things people call "tok/s":
//   decode tok/s      = completion_tokens / (total - prefill)   <- kernel throughput
//   end-to-end tok/s  = completion_tokens / total               <- what a user feels
// Prefix continuation cannot touch the first and dominates the second whenever the
// reply is short relative to the history.
const base = process.argv[2]
const maxTokens = Number(process.argv[3] || 256)

function filler(seed, words) {
  const vocab = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'sigma', 'tau', 'omega', 'rho', 'phi']
  let s = seed >>> 0
  const out = []
  for (let i = 0; i < words; i++) { s = (s * 1103515245 + 12345) >>> 0; out.push(vocab[s % vocab.length]) }
  return out.join(' ')
}

const model = (await (await fetch(`${base}/v1/models`)).json()).data[0].id
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: `Here are my notes:\n\n${filler(1, 1400)}\n\nAcknowledge briefly.` },
]
for (let t = 1; t <= 3; t++) {
  const started = Date.now()
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0, stream: false }),
  })
  const ms = Date.now() - started
  const j = await res.json()
  const reply = j.choices[0].message.content
  console.log(JSON.stringify({ turn: t, ms, prompt_tokens: j.usage.prompt_tokens, completion_tokens: j.usage.completion_tokens }))
  messages.push({ role: 'assistant', content: reply })
  messages.push({ role: 'user', content: `Now write a long detailed paragraph about ${filler(200 + t, 6)}.` })
}
