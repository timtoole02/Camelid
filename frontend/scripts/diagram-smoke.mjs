#!/usr/bin/env node
/* Mermaid diagram smoke.
 *
 * Mermaid does its drawing in an effect, which renderToStaticMarkup does not
 * run, so this asserts what a static render CAN see: which fences become a
 * diagram figure and which stay code, and that the diagram's own source is
 * the pre-render fallback -- what the reader sees if the lazy chunk never
 * arrives. The browser smoke covers actual drawing.
 */
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
  const markdown = await server.ssrLoadModule('/src/lib/markdown.jsx')
  const { AssistantMarkdown } = markdown
  const render = (content, streaming = false) =>
    renderToStaticMarkup(React.createElement(AssistantMarkdown, { content, streaming }))

  /* ---- a CLOSED mermaid fence becomes a diagram ------------------------- */
  const diagram = render('```mermaid\ngraph TD;\n  A-->B;\n```')
  assert.match(diagram, /class="cx-mermaid"/, 'a closed mermaid fence becomes a diagram figure')
  assert.match(diagram, /graph TD/, 'the diagram source survives as the pre-render fallback')
  assert.match(diagram, /Copy diagram source/, 'the source stays copyable')

  const capitalised = render('```Mermaid\ngraph TD;\n  A-->B;\n```')
  assert.match(capitalised, /class="cx-mermaid"/, 'the language tag is matched case-insensitively')

  /* ---- an OPEN one stays a code card until it closes -------------------- */
  const openDiagram = render('```mermaid\ngraph TD;\n  A-->B;', true)
  assert.doesNotMatch(
    openDiagram,
    /class="cx-mermaid"/,
    'an OPEN mermaid fence stays a code card — handing Mermaid a half-typed diagram flashes parse errors through the stream',
  )
  assert.match(openDiagram, /message-code-card-title/, 'and renders as the ordinary code card meanwhile')

  /* ---- an empty fence has nothing to draw ------------------------------- */
  const empty = render('```mermaid\n```')
  assert.doesNotMatch(empty, /class="cx-mermaid"/, 'an empty mermaid fence is never handed to Mermaid')

  /* ---- every other fence is untouched ----------------------------------- */
  const ordinaryCode = render('```python\nx = 1\n```')
  assert.doesNotMatch(ordinaryCode, /cx-mermaid/, 'a non-mermaid fence is unaffected')
  assert.match(ordinaryCode, /message-code-card-title[^>]*>PYTHON</, 'and still labels its language')

  /* ---- the clipboard helper still reaches every existing call site ------ */
  /* copyText moved to lib/clipboard.js so the diagram component can import it
     without a cycle through markdown.jsx. Everything that already imports it
     from markdown.jsx must get the SAME function, not a copy that could drift. */
  const clipboard = await server.ssrLoadModule('/src/lib/clipboard.js')
  assert.equal(typeof clipboard.copyText, 'function', 'the clipboard module exports copyText')
  assert.equal(markdown.copyText, clipboard.copyText, 'markdown.jsx re-exports the one helper, so existing imports keep working')

  console.log('diagram smoke passed')
} finally {
  await server.close()
}
