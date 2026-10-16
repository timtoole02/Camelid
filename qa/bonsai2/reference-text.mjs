// ORACLE_SERVER MODEL OUTPUT.json; owns and stops its dedicated reference process.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { basename, resolve } from 'node:path'
import { readFile, writeFile } from 'node:fs/promises'
const [server, model, output] = process.argv.slice(2)
assert.ok(server && model && output)
const child = spawn(resolve(server), ['-m', resolve(model), '--host', '127.0.0.1', '--port', '18091', '-c', '2048', '-ngl', '0', '-t', '8', '-tb', '8', '-np', '1'], { windowsHide: true })
let log = ''
child.stdout.on('data', data => { log += data })
child.stderr.on('data', data => { log += data })
const report = { model: basename(model), reference: 'PrismML-Eng/llama.cpp prism-b10777-e3de715', backend: 'CPU', threads: 8, cases: [] }
try {
  let ready = false
  for (let i = 0; i < 180; i++) {
    try { ready = (await fetch('http://127.0.0.1:18091/health')).ok } catch {}
    if (ready) break
    assert.equal(child.exitCode, null, log)
    await new Promise(done => setTimeout(done, 1000))
  }
  assert.ok(ready, log)
  for (const name of ['text', 'count', 'code', 'explain']) {
    const prompt = await readFile(new URL(name === 'text' ? './text-prompt.txt' : `./${name}.chatml`, import.meta.url), 'utf8')
    const response = await fetch('http://127.0.0.1:18091/completion', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt, n_predict: 32, temperature: 0, top_k: 1, repeat_penalty: 1, return_tokens: true, cache_prompt: false }),
    })
    const result = await response.json()
    assert.ok(response.ok, JSON.stringify(result))
    const ids = result.tokens.at(-1) === 248046 ? result.tokens.slice(0, -1) : result.tokens
    report.cases.push({ name, ids, text: result.content, prompt_tokens: result.tokens_evaluated, timings: result.timings })
    await writeFile(output, JSON.stringify(report, null, 2))
    console.log(JSON.stringify({ name, tokens: ids.length, text: result.content }))
  }
} finally {
  child.kill()
  await writeFile(`${output}.log`, log)
}
