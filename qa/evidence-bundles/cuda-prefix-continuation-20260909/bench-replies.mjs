// Same growing conversation as bench-turns.mjs, but records the REPLY TEXT so the
// two arms can be diffed directly instead of inferring agreement from token counts.
const base = process.argv[2]
const turns = Number(process.argv[3] || 5)

function filler(seed, words) {
  const vocab = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'sigma', 'tau', 'omega', 'rho', 'phi']
  let s = seed >>> 0
  const out = []
  for (let i = 0; i < words; i++) {
    s = (s * 1103515245 + 12345) >>> 0
    out.push(vocab[s % vocab.length])
  }
  return out.join(' ')
}

const r = await fetch(`${base}/v1/models`)
const model = (await r.json()).data[0].id
const messages = [
  { role: 'system', content: 'You are a terse assistant. Answer in one short sentence.' },
  { role: 'user', content: `Here are my notes:\n\n${filler(1, 1400)}\n\nAcknowledge briefly.` },
]
for (let t = 1; t <= turns; t++) {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages, max_tokens: 16, temperature: 0, stream: false }),
  })
  const j = await res.json()
  const reply = j.choices[0].message.content
  console.log(`turn ${t} prompt_tokens=${j.usage.prompt_tokens} reply=${JSON.stringify(reply)}`)
  messages.push({ role: 'assistant', content: reply })
  messages.push({ role: 'user', content: `Follow-up ${t}: ${filler(100 + t, 12)}. One sentence.` })
}
