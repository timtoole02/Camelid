#!/usr/bin/env node
/* Math + diagram rendering smoke.
 *
 * The delimiter scanner is where this feature lives or dies: model output is
 * full of dollar signs that are not math, and four different delimiter styles
 * that are. Most of this file is the NOT-math half.
 *
 * The renderers themselves are asserted only at the level a static render can
 * see (KaTeX and Mermaid both do their work in an effect, which
 * renderToStaticMarkup does not run) -- that the TeX source and the diagram
 * source survive as the pre-render fallback, which is what the reader sees if
 * the lazy chunk never arrives. The browser smoke covers actual rendering.
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
  const { splitMathSegments, hasMath, displayMathOnlyLine } = await server.ssrLoadModule('/src/lib/mathSegments.js')

  const kinds = (input) => splitMathSegments(input).map((s) => (s.type === 'math' ? (s.display ? 'D' : 'I') : 't')).join('')
  const mathValues = (input) => splitMathSegments(input).filter((s) => s.type === 'math').map((s) => s.value)
  const textOf = (input) => splitMathSegments(input).filter((s) => s.type === 'text').map((s) => s.value).join('')

  /* ---- the NOT-math half: currency and prose must survive untouched ------ */
  const priceList = 'The 8B costs $40 and the 27B costs $90 per month.'
  assert.equal(kinds(priceList), 't', 'two currency amounts in a sentence are not a formula')
  assert.equal(textOf(priceList), priceList, 'and the sentence is preserved byte for byte')

  assert.equal(kinds('It is $5 today'), 't', 'a single price is not an unterminated formula')
  assert.equal(kinds('Range: $10-$20 range'), 't', 'a price range must not become math')
  assert.equal(
    kinds('Pay $100 now, $200 later, $300 never'),
    't',
    'three prices must not pair up into two formulas',
  )
  assert.equal(kinds('costs \\$5 literal'), 't', 'an escaped dollar is a literal dollar')
  assert.equal(kinds('$ x $'), 't', 'whitespace just inside the delimiters means it is not a formula')
  assert.equal(kinds('a $b\n\nc$ d'), 't', 'a candidate span must not cross a blank line')
  assert.equal(kinds('nothing here at all'), 't', 'prose with no delimiters is one text segment')
  assert.equal(kinds(''), '', 'empty input yields no segments')

  /* ---- the math half: all four delimiter styles -------------------------- */
  assert.equal(kinds('Euler: $e^{i\\pi}+1=0$ is neat'), 'tIt', 'bare $...$ is inline math')
  assert.deepEqual(mathValues('Euler: $e^{i\\pi}+1=0$ is neat'), ['e^{i\\pi}+1=0'], 'delimiters are stripped')

  assert.equal(kinds('Block: $$\\int_0^1 x\\,dx$$ done'), 'tDt', '$$...$$ is display math')
  assert.equal(kinds('Llama style \\(a^2+b^2\\) here'), 'tIt', '\\(...\\) is inline math')
  assert.equal(kinds('Llama style \\[c^2\\] here'), 'tDt', '\\[...\\] is display math')
  assert.deepEqual(
    mathValues('\\(a\\) then $b$ then \\[c\\] then $$d$$'),
    ['a', 'b', 'c', 'd'],
    'all four styles can appear in one message',
  )

  assert.equal(kinds('$x^2$ and $y^2$'), 'ItI', 'two real formulas in one line both render')
  assert.equal(kinds('$$\nx = 1 \\\\\ny = 2\n$$'), 'D', 'display math may span lines')

  assert.equal(hasMath('plain text'), false)
  assert.equal(hasMath('$x$'), true)
  assert.equal(displayMathOnlyLine('$$x=1$$'), 'x=1', 'a lone display formula is a block of its own')
  assert.equal(displayMathOnlyLine('see $$x=1$$ here'), null, 'a formula with prose around it is not a block')
  assert.equal(displayMathOnlyLine('$x=1$'), null, 'an inline formula alone on a line is not a display block')

  /* ---- integration with the markdown renderer ---------------------------- */
  const { AssistantMarkdown } = await server.ssrLoadModule('/src/lib/markdown.jsx')
  const render = (content, streaming = false) =>
    renderToStaticMarkup(React.createElement(AssistantMarkdown, { content, streaming }))

  const inlineMath = render('Euler: $e^{i\\pi}+1=0$ is neat.')
  assert.match(inlineMath, /class="cx-math"/, 'inline math renders through the math span')
  assert.match(inlineMath, /e\^\{i\\pi\}\+1=0/, 'the TeX source is the pre-render fallback')

  const displayMath = render('Result:\n\n$$E = mc^2$$')
  assert.match(displayMath, /cx-math--display/, 'a lone display formula gets the display class')

  /* Inline code, links and bold must still win over a $ inside them. */
  const inCode = render('use `$HOME/bin$PATH` carefully')
  assert.doesNotMatch(inCode, /cx-math/, 'a $ inside inline code is shell syntax, not math')
  assert.match(inCode, /<code class="cx-code">\$HOME\/bin\$PATH<\/code>/, 'the code span is intact')

  const inLink = render('see [the $5 tier](https://example.com/pricing$x)')
  assert.doesNotMatch(inLink, /cx-math/, 'link text and targets are not scanned for math')
  assert.match(inLink, /href="https:\/\/example\.com\/pricing\$x"/, 'the link target is untouched')

  const currencyProse = render('The 8B costs $40 and the 27B costs $90.')
  assert.doesNotMatch(currencyProse, /cx-math/, 'currency in rendered prose stays prose')
  assert.match(currencyProse, /costs \$40 and the 27B costs \$90/, 'and reads exactly as written')

  /* ---- mermaid fences ---------------------------------------------------- */
  const diagram = render('```mermaid\ngraph TD;\n  A-->B;\n```')
  assert.match(diagram, /class="cx-mermaid"/, 'a closed mermaid fence becomes a diagram figure')
  assert.match(diagram, /graph TD/, 'the diagram source survives as the pre-render fallback')
  assert.match(diagram, /Copy diagram source/, 'the source stays copyable')

  const openDiagram = render('```mermaid\ngraph TD;\n  A-->B;', true)
  assert.doesNotMatch(
    openDiagram,
    /class="cx-mermaid"/,
    'an OPEN mermaid fence stays a code card — handing Mermaid a half-typed diagram flashes parse errors through the stream',
  )
  assert.match(openDiagram, /message-code-card-title/, 'and renders as the ordinary code card meanwhile')

  const ordinaryCode = render('```python\nx = 1\n```')
  assert.doesNotMatch(ordinaryCode, /cx-mermaid/, 'a non-mermaid fence is unaffected')
  assert.match(ordinaryCode, /message-code-card-title[^>]*>PYTHON</, 'and still labels its language')

  /* Math inside a code fence is code, not math. */
  const mathInFence = render('```tex\n$$x=1$$\n```')
  assert.doesNotMatch(mathInFence, /cx-math/, 'a formula inside a code fence stays code')

  console.log('math + diagram smoke passed')
} finally {
  await server.close()
}
