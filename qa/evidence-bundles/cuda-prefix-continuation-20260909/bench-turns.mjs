// Multi-turn prefix-continuation A/B driver.
//
// Simulates what an agent or a chat client actually does: every turn re-sends the
// WHOLE conversation, so turn k's prompt is turn k-1's prompt plus the assistant's
// reply plus one new user message. That is the shape prefix continuation exists to
// exploit, and the shape a single-prompt benchmark cannot show.
//
// Prints one JSON line per turn so the caller can pair arms without parsing prose.

const base = process.argv[2] || 'http://127.0.0.1:8099'
const turns = Number(process.argv[3] || 4)
const arm = process.argv[4] || 'unknown'

// Deterministic filler so BOTH arms tokenize to byte-identical prompts. A varying
// prompt would change the prefill length between arms and invalidate the pairing.
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

async function modelId() {
  const r = await fetch(`${base}/v1/models`)
  const j = await r.json()
  return j.data[0].id
}

async function main() {
  const model = await modelId()
  // ~1400 words of filler => a prompt long enough that prefill dominates, which is
  // the regime the whole finding is about.
  const messages = [
    { role: 'system', content: 'You are a terse assistant. Answer in one short sentence.' },
    { role: 'user', content: `Here are my notes:\n\n${filler(1, 1400)}\n\nAcknowledge briefly.` },
  ]

  for (let t = 1; t <= turns; t++) {
    const started = Date.now()
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 16,
        temperature: 0,
        stream: false,
      }),
    })
    const elapsed = Date.now() - started
    if (!res.ok) {
      console.log(JSON.stringify({ arm, turn: t, error: `http ${res.status}`, body: await res.text() }))
      process.exit(1)
    }
    const j = await res.json()
    const reply = j.choices[0].message.content
    console.log(JSON.stringify({
      arm,
      turn: t,
      ms: elapsed,
      prompt_tokens: j.usage ? j.usage.prompt_tokens : null,
      completion_tokens: j.usage ? j.usage.completion_tokens : null,
    }))
    // Grow the conversation exactly as a client would.
    messages.push({ role: 'assistant', content: reply })
    messages.push({ role: 'user', content: `Follow-up ${t}: ${filler(100 + t, 12)}. One sentence.` })
  }
}

main().catch((e) => {
  console.log(JSON.stringify({ arm, error: String(e) }))
  process.exit(1)
})
