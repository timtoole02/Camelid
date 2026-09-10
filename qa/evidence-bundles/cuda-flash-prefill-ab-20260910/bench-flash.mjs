// Cold-prefill sweep across context lengths.
//
// Flash prefill accelerates the O(n^2) attention inside prefill. Since prefix
// continuation (#737) reduced a FOLLOW-UP turn's prefill to the newly appended
// tokens, flash now matters for the COLD case: turn 1, a long single-shot prompt,
// anything with no resident prefix. So every prompt here is made unique at token 0
// so continuation finds nothing to reuse and we measure a real cold prefill.
const base = process.argv[2]
const lengths = (process.argv[3] || '1500,3000,6000').split(',').map(Number)
const tag = process.argv[4] || 'arm'
const nonce = process.argv[5] || String(Date.now())

function filler(seed, words) {
  const vocab = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'sigma', 'tau', 'omega', 'rho', 'phi', 'chi', 'psi']
  let s = seed >>> 0
  const out = []
  for (let i = 0; i < words; i++) { s = (s * 1103515245 + 12345) >>> 0; out.push(vocab[s % vocab.length]) }
  return out.join(' ')
}

const model = (await (await fetch(`${base}/v1/models`)).json()).data[0].id

for (const words of lengths) {
  // The nonce leads the prompt, so two arms differ at token 0 and neither can
  // continue from the other's resident KV. Same nonce within an arm pair keeps
  // the token count identical across arms.
  const content = `Session ${nonce}-${words}. Notes:\n\n${filler(words, words)}\n\nAcknowledge briefly.`
  const started = Date.now()
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content }],
      max_tokens: 8, temperature: 0, stream: false,
    }),
  })
  const ms = Date.now() - started
  if (!res.ok) {
    console.log(JSON.stringify({ tag, words, error: `http ${res.status}`, body: (await res.text()).slice(0, 200) }))
    continue
  }
  const j = await res.json()
  console.log(JSON.stringify({
    tag, words, ms,
    prompt_tokens: j.usage.prompt_tokens,
    reply: j.choices[0].message.content.slice(0, 60),
  }))
}
