// Greedy token-parity probe for CAMELID_FLASH_PREFILL.
//
// The flag documents itself as "token-parity rather than bit-identical". This
// records the outputs so the two arms can be diffed directly. Distinct prompts and
// two context lengths, because a single prompt cannot tell "always diverges" from
// "sometimes diverges", and the divergence may be length-dependent.
const base = process.argv[2]
const tag = process.argv[3] || 'arm'

function filler(seed, words) {
  const vocab = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'sigma', 'tau', 'omega', 'rho', 'phi', 'chi', 'psi']
  let s = seed >>> 0
  const out = []
  for (let i = 0; i < words; i++) { s = (s * 1103515245 + 12345) >>> 0; out.push(vocab[s % vocab.length]) }
  return out.join(' ')
}

const PROMPTS = [
  { id: 'notes', lead: 'Here are my notes', tail: 'Summarize in one sentence.' },
  { id: 'log', lead: 'Here is a system log', tail: 'What stands out? One sentence.' },
  { id: 'story', lead: 'Here is a draft', tail: 'Give one sentence of feedback.' },
]
const LENGTHS = [1400, 6000]

const model = (await (await fetch(`${base}/v1/models`)).json()).data[0].id

for (const words of LENGTHS) {
  for (const p of PROMPTS) {
    const content = `${p.lead} (${p.id}/${words}):\n\n${filler(words + p.id.length, words)}\n\n${p.tail}`
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model, messages: [{ role: 'user', content }],
        max_tokens: 48, temperature: 0, stream: false,
      }),
    })
    if (!res.ok) { console.log(`${words}\t${p.id}\tHTTP_${res.status}`); continue }
    const j = await res.json()
    console.log(`${words}\t${p.id}\t${j.usage.prompt_tokens}\t${JSON.stringify(j.choices[0].message.content)}`)
  }
}
