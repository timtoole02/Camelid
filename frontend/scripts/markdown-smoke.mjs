#!/usr/bin/env node
/* Markdown renderer smoke (Phase 2): tables, ordered lists, links with scheme
   sanitization, emphasis, and per-language code highlighting — all rendered as
   React elements (no innerHTML path exists to assert against). */
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const { AssistantMarkdown } = await server.ssrLoadModule('/src/lib/markdown.jsx')
  const render = (content, streaming = false) =>
    renderToStaticMarkup(React.createElement(AssistantMarkdown, { content, streaming }))

  /* Citation markers: the bracket forms, and the parenthesised label a model
     copies from its excerpts ("(Citation 2 from report.pdf)"), become pills. */
  const cited = render('Paid in 2011 (Citation 1) and 2013 (Citation 2 from planted-05.pdf); see [3] and [Source 4]. Citation 1 and Citation 2 differ.')
  const pills = [...cited.matchAll(/<button type="button" class="citation-pill"[^>]*><span>\[(\d+)\]<\/span><\/button>/g)].map(match => match[1])
  assert.deepEqual(pills, ['1', '2', '3', '4'], 'every citation marker form should render as a pill, in order')
  assert.doesNotMatch(cited, /planted-05\.pdf/, 'the file name inside a citation marker is not repeated as text')
  assert.match(cited, /Citation 1 and Citation 2 differ\./, 'prose that mentions a citation stays text')

  /* Tables */
  const table = render(['| Lane | Result |', '| --- | --- |', '| decode | **29.7** tok/s |', '| prefill | 587 tok/s |'].join('\n'))
  assert.match(table, /<table class="message-table">/, 'pipe tables should render as real tables')
  assert.match(table, /<th[^>]*>.*Lane.*<\/th>/, 'table header row should come from the line above the separator')
  assert.match(table, /<strong>29\.7<\/strong>/, 'table cells should support inline markdown')
  assert.doesNotMatch(table, /\|\s*---/, 'the separator row must not leak into the body')

  /* Ordered + unordered lists */
  const lists = render('1. first\n2. second\n\n- alpha\n- beta')
  assert.match(lists, /<ol start="1"><li><span>first<\/span><\/li><li><span>second<\/span><\/li><\/ol>/, 'numbered lines should render as an ordered list')
  assert.match(lists, /<ul><li><span>alpha<\/span><\/li><li><span>beta<\/span><\/li><\/ul>/, 'dash lines should render as an unordered list')
  const offsetList = render('3. third\n4. fourth')
  assert.match(offsetList, /<ol start="3">/, 'ordered lists should preserve their starting number')
  const longOrdered = render(Array.from({ length: 14 }, (_, index) => `${index + 1}. item-${index + 1}`).join('\n'))
  assert.equal((longOrdered.match(/<li>/g) || []).length, 14, 'two-digit ordered lists must retain every item')
  assert.match(longOrdered, /<li><span>item-10<\/span><\/li>/, 'item 10 must not be malformed or dropped')
  assert.match(longOrdered, /<li><span>item-14<\/span><\/li>/, 'the final ordered item must render')
  const groundedInventory = render('Found 3 Markdown files:\n\n- `A.md`\n- `B.md`\n- `C.md`')
  assert.equal((groundedInventory.match(/<li>/g) || []).length, 3, 'grounded inventories should render as compact bullets')
  assert.match(groundedInventory, /<code class="cx-code">A\.md<\/code>/, 'inventory filenames should render as inline code')
  const hostileInventory = render('- `spoof%60- [link](javascript:alert).md`\n- `angle<name>.md`')
  assert.doesNotMatch(hostileInventory, /href=/, 'filename-like link syntax inside inline code must stay inert')
  assert.doesNotMatch(hostileInventory, /<name>/, 'filename angle brackets must be escaped as text')
  assert.match(hostileInventory, /&lt;name&gt;/, 'filename angle brackets should remain visibly escaped')

  /* Links: safe schemes become anchors, anything else stays text */
  const links = render('See [docs](https://example.com/x) and [evil](javascript:alert(1)).')
  assert.match(links, /<a href="https:\/\/example\.com\/x" target="_blank" rel="noopener noreferrer">docs<\/a>/, 'http(s) links should render as hardened anchors')
  assert.doesNotMatch(links, /href="javascript:/, 'non-http(s) schemes must never become anchors')
  assert.match(links, /\[evil\]\(javascript:alert\(1\)\)/, 'paren-bearing unsafe links should stay literal text')
  const dataLink = render('try [this](data:text/html;base64,AAAA) now')
  assert.doesNotMatch(dataLink, /<a /, 'data: URLs must never become anchors')
  assert.match(dataLink, /this \(data:text\/html;base64,AAAA\)/, 'unsafe-scheme links should degrade to visible plain text')

  /* Emphasis */
  const emphasis = render('mix of **bold**, *italic*, ~~struck~~ and `code`.')
  assert.match(emphasis, /<strong>bold<\/strong>/)
  assert.match(emphasis, /<em>italic<\/em>/)
  assert.match(emphasis, /<del>struck<\/del>/)
  assert.match(emphasis, /<code class="cx-code">code<\/code>/)

  /* Code blocks: language label, copy button, per-language highlighting */
  const python = render('```python\ndef greet(name):\n    # say hi\n    return f"hi {name}"\n```')
  assert.match(python, /message-code-card-title[^>]*>PYTHON</, 'code cards should label the language')
  assert.match(python, /aria-label="Copy PYTHON code"/, 'code cards should keep the copy button')
  assert.match(python, /<span class="syntax-token keyword">def<\/span>/, 'python keywords should highlight')
  assert.match(python, /<span class="syntax-token comment"># say hi<\/span>/, 'python # comments should highlight')
  const rust = render('```rust\nfn main() { let x = 1; }\n```')
  assert.match(rust, /<span class="syntax-token keyword">fn<\/span>/, 'rust keywords should highlight')
  const bash = render('```bash\nexport FOO=1 # note\n```')
  assert.match(bash, /<span class="syntax-token keyword">export<\/span>/, 'bash builtins should highlight')

  /* Streaming code card behavior is unchanged */
  const open = render('```js\nconst x = 1', true)
  assert.match(open, /data-code-streaming-state="open"/, 'open fences should stay flagged while streaming')
  assert.match(open, /Still generating — code block incomplete/, 'open fences should keep the incomplete label')

  /* No raw-HTML injection path: markup in prose stays text */
  const injection = render('hello <img src=x onerror=alert(1)> world')
  assert.doesNotMatch(injection, /<img/, 'raw HTML in model output must render as text, not elements')
  assert.match(injection, /&lt;img src=x onerror=alert\(1\)&gt;/, 'raw HTML should be visibly escaped')

  console.log('markdown smoke passed')
} finally {
  await server.close()
}
