// ORACLE_SERVER MODEL CASE OUTPUT.json; scores a full greedy reference continuation.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { resolve } from 'node:path'
import { readFile, writeFile } from 'node:fs/promises'
const [server, model, name, output] = process.argv.slice(2)
assert.ok(server && model && name && output)
const child = spawn(resolve(server), ['-m', resolve(model), '--host', '127.0.0.1', '--port', '18091', '-c', '2048', '-ngl', '0', '-t', '8', '-tb', '8', '-np', '1'], { windowsHide: true })
let log = ''
child.stdout.on('data', data => { log += data })
child.stderr.on('data', data => { log += data })
try {
  let ready = false
  for (let i = 0; i < 180; i++) {
    try { ready = (await fetch('http://127.0.0.1:18091/health')).ok } catch {}
    if (ready) break
    assert.equal(child.exitCode, null, log)
    await new Promise(done => setTimeout(done, 1000))
  }
  assert.ok(ready, log)
  const prompt = await readFile(new URL(name === 'text' ? './text-prompt.txt' : `./${name}.chatml`, import.meta.url), 'utf8')
  const response = await fetch('http://127.0.0.1:18091/completion', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt, n_predict: 32, temperature: 0, top_k: 1, repeat_penalty: 1, return_tokens: true, n_probs: 20, post_sampling_probs: false, cache_prompt: false }),
  })
  const result = await response.json()
  assert.ok(response.ok, JSON.stringify(result))
  await writeFile(output, JSON.stringify(result, null, 2))
  console.log(JSON.stringify({ content: result.content, probabilities: result.completion_probabilities?.length }))
} finally {
  child.kill()
  await writeFile(`${output}.log`, log)
}
