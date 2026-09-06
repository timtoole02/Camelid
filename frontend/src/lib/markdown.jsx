import { createContext, memo, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react'

/* Assistant markdown + fenced-code rendering.
   Extracted verbatim from the original ChatWorkspace so the parsing/rendering
   behavior (and the markup the CI smokes assert on) is preserved exactly. */

export const normalizeCodeLanguage = (value) => {
  const language = String(value || '').trim().replace(/[^a-zA-Z0-9_+#.-].*$/, '')
  if (!language) return 'Code'
  if (language.toLowerCase() === 'js') return 'JavaScript'
  if (language.toLowerCase() === 'ts') return 'TypeScript'
  if (language.toLowerCase() === 'html') return 'HTML'
  if (language.toLowerCase() === 'css') return 'CSS'
  return language.toUpperCase()
}

/* Returns whether the text actually reached the clipboard. `navigator.clipboard`
   is undefined outside secure contexts — which includes reaching a Camelid served
   with `--addr 0.0.0.0` over a plain-http LAN address — and optional chaining
   would otherwise make that indistinguishable from success, letting a caller show
   a "Copied" confirmation for a copy that never happened. Callers that ignore the
   result behave exactly as before. */
export const copyText = async (text) => {
  try {
    if (!navigator.clipboard?.writeText) return false
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Clipboard access can be denied even in a secure context; rendering still works.
    return false
  }
}

/* Only http(s)/mailto links render as anchors; any other scheme (javascript:,
   data:, file:) stays plain text — model output never picks the protocol. */
const SAFE_LINK_SCHEME = /^(https?:|mailto:)/i
const EMPTY_CITATIONS = []
const CitationContext = createContext(EMPTY_CITATIONS)

function CitationPill({ citeIndex }) {
  const citations = useContext(CitationContext)
  const citation = Array.isArray(citations) ? citations[Number(citeIndex) - 1] : undefined
  return (
    <button
      type="button"
      className="citation-pill"
      title={`View source citation [${citeIndex}]`}
      onClick={() => {
        window.dispatchEvent(new CustomEvent('camelid-citation-click', {
          detail: { index: citeIndex, citation },
        }))
      }}
    >
      <span>[{citeIndex}]</span>
    </button>
  )
}

/* The desktop (Tauri) webview has no browser tabs, so target="_blank" cannot
   open anything there; route http(s) links through the system browser via the
   opener plugin instead. In the web build (no window.__TAURI__) the handler is
   inert and the anchor opens a new tab as usual. */
const openLinkExternally = (event, href) => {
  const opener = typeof window !== 'undefined' ? window.__TAURI__?.opener : undefined
  if (!opener?.openUrl || !/^https?:/i.test(href)) return
  event.preventDefault()
  Promise.resolve(opener.openUrl(href)).catch(() => {})
}

const renderInlineMarkdown = (text, keyPrefix) => {
  const parts = String(text || '')
    .split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|~~[^~]+~~|\[[^\]]+\]\([^()\s]+\)|\[(?:Citation\s+|Source\s+)?\d+\])/gi)
    .filter(Boolean)
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={key} className="cx-code">{part.slice(1, -1)}</code>
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('~~') && part.endsWith('~~')) {
      return <del key={key}>{part.slice(2, -2)}</del>
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>
    }
    const citeMatch = part.match(/^\[(?:Citation\s+|Source\s+)?(\d+)\]$/i)
    if (citeMatch) {
      const citeIndex = citeMatch[1]
      return <CitationPill key={key} citeIndex={citeIndex} />
    }
    const link = part.match(/^\[([^\]]+)\]\(([^()\s]+)\)$/)
    if (link) {
      const [, label, href] = link
      if (SAFE_LINK_SCHEME.test(href)) {
        return <a key={key} href={href} target="_blank" rel="noopener noreferrer" onClick={(event) => openLinkExternally(event, href)}>{label}</a>
      }
      return <span key={key}>{label} ({href})</span>
    }
    return <span key={key}>{part}</span>
  })
}

/* ---- Tables: header + |---| separator + body rows, cells get inline markdown ---- */
const isTableLine = (line) => /^\|.*\|$/.test(line.trim())
const isTableSeparator = (line) => /^\|(\s*:?-{2,}:?\s*\|)+$/.test(line.trim().replace(/\s/g, ''))

const splitTableCells = (line) => {
  const trimmed = line.trim().replace(/^\||\|$/g, '')
  return trimmed.split('|').map((cell) => cell.trim())
}

const renderTable = (tableLines, keyPrefix) => {
  const hasHeader = tableLines.length >= 2 && isTableSeparator(tableLines[1])
  const headerCells = hasHeader ? splitTableCells(tableLines[0]) : []
  const bodyLines = hasHeader ? tableLines.slice(2) : tableLines
  return (
    <div key={`${keyPrefix}-table`} className="message-table-wrap">
      <table className="message-table">
        {hasHeader && (
          <thead>
            <tr>
              {headerCells.map((cell, index) => (
                <th key={`${keyPrefix}-th-${index}`}>{renderInlineMarkdown(cell, `${keyPrefix}-th-${index}`)}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {bodyLines.filter((line) => !isTableSeparator(line)).map((line, rowIndex) => (
            <tr key={`${keyPrefix}-tr-${rowIndex}`}>
              {splitTableCells(line).map((cell, cellIndex) => (
                <td key={`${keyPrefix}-td-${rowIndex}-${cellIndex}`}>{renderInlineMarkdown(cell, `${keyPrefix}-td-${rowIndex}-${cellIndex}`)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const normalizeProseForReading = (text) => String(text || '')
  .replace(/\r\n/g, '\n')
  .replace(/\s+(Page\s+\d+\b)/gi, '\n\n$1')
  .replace(/\s+(References?\s*:)/gi, '\n\n$1')
  .replace(/\s+(Works\s+Cited\s*:)/gi, '\n\n$1')
  .replace(/\s+([•*-]\s+["“])/g, '\n$1')

const splitLongParagraph = (value) => {
  const text = String(value || '').trim()
  if (text.length <= 520) return text ? [text] : []
  const sentences = text.match(/[^.!?]+[.!?]+["”']?|[^.!?]+$/g) || [text]
  const paragraphs = []
  let current = ''

  sentences.forEach((sentence) => {
    const next = `${current}${current ? ' ' : ''}${sentence.trim()}`.trim()
    if (current && (next.length > 620 || current.split(/[.!?]+/).filter(Boolean).length >= 4)) {
      paragraphs.push(current)
      current = sentence.trim()
    } else {
      current = next
    }
  })
  if (current) paragraphs.push(current)
  return paragraphs
}

const pushParagraphBlocks = (blocks, value, keyPrefix) => {
  splitLongParagraph(value).forEach((paragraph) => {
    blocks.push(<p key={`${keyPrefix}-p-${blocks.length}`}>{renderInlineMarkdown(paragraph, `${keyPrefix}-p-${blocks.length}`)}</p>)
  })
}

const renderMarkdownText = (text, keyPrefix) => {
  const lines = normalizeProseForReading(text).split('\n')
  const blocks = []
  let paragraph = []
  let list = []
  let orderedList = []
  let tableLines = []
  let quoteLines = []

  const flushQuote = () => {
    if (!quoteLines.length) return
    const groupIndex = blocks.length
    const quoteParagraphs = []
    let current = []
    quoteLines.forEach((line) => {
      if (line) current.push(line)
      else if (current.length) {
        quoteParagraphs.push(current.join(' '))
        current = []
      }
    })
    if (current.length) quoteParagraphs.push(current.join(' '))
    blocks.push(
      <blockquote key={`${keyPrefix}-q-${groupIndex}`}>
        {quoteParagraphs.map((value, index) => (
          <p key={`${keyPrefix}-q-${groupIndex}-${index}`}>{renderInlineMarkdown(value, `${keyPrefix}-q-${groupIndex}-${index}`)}</p>
        ))}
      </blockquote>,
    )
    quoteLines = []
  }

  const flushParagraph = () => {
    if (paragraph.length) {
      const value = paragraph.join(' ').trim()
      if (value) {
        pushParagraphBlocks(blocks, value, keyPrefix)
      }
      paragraph = []
    }
  }
  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={`${keyPrefix}-ul-${blocks.length}`}>
          {list.map((item, index) => (
            <li key={`${keyPrefix}-li-${blocks.length}-${index}`}>{renderInlineMarkdown(item, `${keyPrefix}-li-${index}`)}</li>
          ))}
        </ul>,
      )
      list = []
    }
    if (orderedList.length) {
      blocks.push(
        <ol key={`${keyPrefix}-ol-${blocks.length}`} start={orderedList[0].start}>
          {orderedList.map((item, index) => (
            <li key={`${keyPrefix}-oli-${blocks.length}-${index}`}>{renderInlineMarkdown(item.text, `${keyPrefix}-oli-${index}`)}</li>
          ))}
        </ol>,
      )
      orderedList = []
    }
    if (tableLines.length) {
      blocks.push(renderTable(tableLines, `${keyPrefix}-tbl-${blocks.length}`))
      tableLines = []
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      flushQuote()
      return
    }
    if (isTableLine(line)) {
      flushParagraph()
      flushQuote()
      if (!tableLines.length) flushList()
      tableLines.push(line)
      return
    }
    if (tableLines.length) flushList()
    /* Blockquote lines group into one <blockquote>; a bare ">" separates
       paragraphs inside it. */
    const quoted = line.match(/^>\s?(.*)$/)
    if (quoted) {
      flushParagraph()
      flushList()
      quoteLines.push(quoted[1].trim())
      return
    }
    flushQuote()
    /* Thematic break: --- / *** / ___ renders as a rule, not literal dashes. */
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      flushParagraph()
      flushList()
      blocks.push(<hr key={`${keyPrefix}-hr-${blocks.length}`} />)
      return
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) {
      flushParagraph()
      flushList()
      const depth = heading[1].length
      const Tag = depth === 1 ? 'h2' : depth <= 3 ? 'h3' : 'h4'
      blocks.push(<Tag key={`${keyPrefix}-h-${blocks.length}`}>{renderInlineMarkdown(heading[2], `${keyPrefix}-h-${blocks.length}`)}</Tag>)
      return
    }
    const pageHeading = line.match(/^(Page\s+\d+)\b[:\s.-]*(.*)$/i)
    if (pageHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-page-${blocks.length}`}>{pageHeading[1]}</h3>)
      if (pageHeading[2]) {
        pushParagraphBlocks(blocks, pageHeading[2], keyPrefix)
      }
      return
    }
    const referencesHeading = line.match(/^(References?|Works\s+Cited)\s*:?(.*)$/i)
    if (referencesHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-ref-${blocks.length}`}>{referencesHeading[1]}</h3>)
      if (referencesHeading[2]) {
        pushParagraphBlocks(blocks, referencesHeading[2].replace(/^\s*[:*-]\s*/, ''), keyPrefix)
      }
      return
    }
    const listItem = line.match(/^[-*]\s+(.+)$/)
    if (listItem) {
      flushParagraph()
      if (!list.length) flushList()
      list.push(listItem[1])
      return
    }
    const orderedItem = line.match(/^(\d+)[.)]\s+(.+)$/)
    if (orderedItem) {
      flushParagraph()
      if (!orderedList.length) flushList()
      orderedList.push({ start: orderedList.length ? orderedList[0].start : Number(orderedItem[1]), text: orderedItem[2] })
      return
    }
    flushList()
    paragraph.push(line)
  })
  flushParagraph()
  flushList()
  flushQuote()
  return blocks
}

const LANG_KEYWORDS = {
  python: 'def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|lambda|None|True|False|self|in|not|and|or|is|pass|raise|yield|global|nonlocal|del|assert|async|await|print',
  rust: 'fn|let|mut|impl|pub|struct|enum|match|use|mod|trait|where|for|while|loop|if|else|return|self|Self|crate|super|const|static|ref|move|async|await|dyn|unsafe|type|Some|None|Ok|Err|Box|Vec|String',
  bash: 'if|then|elif|else|fi|for|do|done|while|until|case|esac|function|echo|export|local|return|source|set|exit|read|shift|trap|true|false',
  json: 'true|false|null',
  js: 'const|let|var|function|return|if|else|for|while|class|new|true|false|null|undefined|import|export|from|async|await|document|window|of|in|typeof|instanceof|switch|case|break|continue|throw|try|catch|finally|yield|extends|super|this|static|get|set',
}

const languageFamily = (language) => {
  const value = String(language || '').toLowerCase()
  if (value.includes('html') || value.includes('xml')) return 'html'
  if (value.includes('css')) return 'css'
  if (value.includes('python') || value === 'py') return 'python'
  if (value.includes('rust') || value === 'rs') return 'rust'
  if (/^(bash|sh|shell|zsh|console)/.test(value)) return 'bash'
  if (value.includes('json')) return 'json'
  return 'js'
}

const syntaxClassForToken = (token, language) => {
  const family = languageFamily(language)
  if (/^\s+$/.test(token)) return ''
  if (/^\/\//.test(token) || /^\/\*/.test(token) || /^<!--/.test(token)) return 'comment'
  if ((family === 'python' || family === 'bash') && /^#/.test(token)) return 'comment'
  if (family === 'json' && /^"(?:\\.|[^"])*"$/.test(token) && token.endsWith('"')) {
    return 'string'
  }
  if (/^['"`]/.test(token)) return 'string'
  if (/^\d/.test(token)) return 'number'
  if (family === 'html' && /^<\/?[\w-]+/.test(token)) return 'tag'
  if (family === 'html' && /^[\w:-]+(?==)/.test(token)) return 'attr'
  const keywords = LANG_KEYWORDS[family] || LANG_KEYWORDS.js
  if (new RegExp(`^(?:${keywords})$`).test(token)) return 'keyword'
  if (family === 'css' && /^[\w-]+(?=\s*:)/.test(token)) return 'attr'
  return ''
}

const renderHighlightedCode = (code, language, keyPrefix) => {
  const lowerLanguage = String(language || '').toLowerCase()
  const family = languageFamily(lowerLanguage)
  const pattern = family === 'html'
    ? /(<!--[\s\S]*?-->|<\/?[\w-]+|\/?>|[\w:-]+(?==)|"(?:\\.|[^"])*"|'(?:\\.|[^'])*')/g
    : family === 'css'
      ? /(\/\*[\s\S]*?\*\/|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|#[\da-fA-F]{3,8}|\b\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw)?\b|[\w-]+(?=\s*:))/g
      : family === 'python' || family === 'bash'
        ? new RegExp(`(#.*|"""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"])*"|'(?:\\\\.|[^'])*'|\\b(?:${LANG_KEYWORDS[family]})\\b|\\b\\d+(?:\\.\\d+)?\\b)`, 'g')
        : family === 'rust'
          ? new RegExp(`(\\/\\/.*|\\/\\*[\\s\\S]*?\\*\\/|"(?:\\\\.|[^"])*"|\\b(?:${LANG_KEYWORDS.rust})\\b|\\b\\d+(?:\\.\\d+)?\\b)`, 'g')
          : family === 'json'
            ? /("(?:\\.|[^"])*")|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g
            : new RegExp(`(\\/\\/.*|\\/\\*[\\s\\S]*?\\*\\/|"(?:\\\\.|[^"])*"|'(?:\\\\.|[^'])*'|\`(?:\\\\.|[^\`])*\`|\\b(?:${LANG_KEYWORDS.js})\\b|\\b\\d+(?:\\.\\d+)?\\b)`, 'g')
  const nodes = []
  let cursor = 0
  let match = pattern.exec(code)
  while (match) {
    if (match.index > cursor) nodes.push(code.slice(cursor, match.index))
    const token = match[0]
    const tokenClass = syntaxClassForToken(token, lowerLanguage)
    nodes.push(tokenClass
      ? <span key={`${keyPrefix}-${nodes.length}`} className={`syntax-token ${tokenClass}`}>{token}</span>
      : token)
    cursor = match.index + token.length
    match = pattern.exec(code)
  }
  if (cursor < code.length) nodes.push(code.slice(cursor))
  return nodes
}

const splitFenceInfo = (value) => {
  const trimmed = String(value || '').trim()
  if (!trimmed) return { language: 'Code', firstCodeLine: '' }
  const [, rawLanguage = '', firstCodeLine = ''] = trimmed.match(/^([a-zA-Z0-9_+#.-]+)?\s*([\s\S]*)$/) || []
  return {
    language: normalizeCodeLanguage(rawLanguage),
    firstCodeLine: firstCodeLine.trimStart(),
  }
}

export const CODE_CARD_STREAMING_LABEL = 'Still generating — code block incomplete'

export function CodeBlockCard({ language, code, keyPrefix, stillGenerating }) {
  const preRef = useRef(null)
  const autoFollowCodeRef = useRef(true)
  const [copied, setCopied] = useState(false)
  const copiedResetRef = useRef(null)

  useEffect(() => () => {
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
  }, [])

  /* Only confirm "Copied" when the text actually reached the clipboard. */
  const handleCopy = async () => {
    if (!(await copyText(code))) return
    setCopied(true)
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current)
    copiedResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  useEffect(() => {
    if (!stillGenerating) return undefined
    autoFollowCodeRef.current = true
    const pre = preRef.current
    if (!pre) return undefined
    const updateAutoFollow = () => {
      const distanceFromBottom = pre.scrollHeight - (pre.scrollTop + pre.clientHeight)
      autoFollowCodeRef.current = distanceFromBottom < 80
    }
    pre.addEventListener('scroll', updateAutoFollow, { passive: true })
    return () => pre.removeEventListener('scroll', updateAutoFollow)
  }, [stillGenerating])

  useLayoutEffect(() => {
    if (!stillGenerating || !autoFollowCodeRef.current) return
    const pre = preRef.current
    if (pre) pre.scrollTop = pre.scrollHeight
  }, [code, stillGenerating])

  return (
    <figure
      className={`message-code-card ${stillGenerating ? 'is-generating' : ''}`}
      aria-busy={stillGenerating ? 'true' : undefined}
      data-code-streaming-state={stillGenerating ? 'open' : undefined}
    >
      <figcaption>
        <span className="message-code-card-title">{language}</span>
        {stillGenerating && <span className="message-code-card-status" aria-live="polite" data-live-status="active">{CODE_CARD_STREAMING_LABEL}</span>}
        <button type="button" onClick={handleCopy} aria-label={`Copy ${language} code`}>{copied ? 'Copied' : 'Copy'}</button>
      </figcaption>
      {/* Highlighting deferred while the fence is open (Phase 8B): plain text
          per flush; full per-language highlighting applies once the fence
          closes and the block becomes a memoized stable segment. */}
      <pre ref={preRef}><code>{stillGenerating ? code : renderHighlightedCode(code, language, keyPrefix)}</code></pre>
    </figure>
  )
}

const pushCodeBlock = (blocks, language, code, keyPrefix, { incomplete = false, streaming = false } = {}) => {
  const trimmedCode = String(code || '').replace(/^\n+|\n+$/g, '')
  const stillGenerating = Boolean(incomplete && streaming)
  blocks.push(
    <CodeBlockCard
      key={`code-${blocks.length}`}
      language={language}
      code={trimmedCode}
      keyPrefix={keyPrefix}
      stillGenerating={stillGenerating}
    />,
  )
}

export const hasOpenCodeFence = (content) => {
  const matches = String(content || '').match(/```/g)
  return Boolean(matches && matches.length % 2 === 1)
}

/* Streaming-path block memoization (Phase 8B): the stable prefix (everything
   up to the last block boundary outside an open fence) renders through a
   React.memo component keyed on its string, so completed blocks parse exactly
   once; only the open tail re-parses per rAF flush. Non-streaming renders are
   a single stable segment. */
const StableMarkdownSegment = memo(function StableMarkdownSegment({ content }) {
  return renderFencedContent(content, false)
})

function splitStableBoundary(content) {
  const fences = [...content.matchAll(/```/g)]
  if (fences.length % 2 === 1) {
    // open fence: everything before it is stable
    return fences[fences.length - 1].index
  }
  const lastBreak = content.lastIndexOf('\n\n')
  return lastBreak === -1 ? 0 : lastBreak + 2
}

function renderFencedContent(normalized, streaming) {
  const blocks = []
  let cursor = 0
  let fenceStart = normalized.indexOf('```', cursor)
  while (fenceStart !== -1) {
    const before = normalized.slice(cursor, fenceStart)
    blocks.push(...renderMarkdownText(before, `md-${blocks.length}`))
    const infoStart = fenceStart + 3
    const nextLine = normalized.indexOf('\n', infoStart)
    const infoEnd = nextLine === -1 ? normalized.length : nextLine
    const { language, firstCodeLine } = splitFenceInfo(normalized.slice(infoStart, infoEnd))
    const codeStart = nextLine === -1 ? infoEnd : nextLine + 1
    const fenceEnd = normalized.indexOf('```', codeStart)
    const incompleteFence = fenceEnd === -1
    const codeEnd = fenceEnd === -1 ? normalized.length : fenceEnd
    const codeBody = normalized.slice(codeStart, codeEnd)
    const code = firstCodeLine ? `${firstCodeLine}${codeBody ? `\n${codeBody}` : ''}` : codeBody
    pushCodeBlock(blocks, language, code, `code-${blocks.length}`, { incomplete: incompleteFence, streaming })
    cursor = fenceEnd === -1 ? normalized.length : fenceEnd + 3
    fenceStart = normalized.indexOf('```', cursor)
  }
  blocks.push(...renderMarkdownText(normalized.slice(cursor), `md-${blocks.length}`))
  return blocks
}

function AssistantMarkdownInner({ content, streaming = false, citations = EMPTY_CITATIONS }) {
  const normalized = String(content || '').replace(/\r\n/g, '\n')
  if (!streaming) {
    const blocks = renderFencedContent(normalized, false)
    return (
      <CitationContext.Provider value={citations}>
        <div className="message-markdown">{blocks.length ? blocks : <p>{content}</p>}</div>
      </CitationContext.Provider>
    )
  }
  const boundary = splitStableBoundary(normalized)
  const stable = normalized.slice(0, boundary)
  const tail = normalized.slice(boundary)
  return (
    <CitationContext.Provider value={citations}>
      <div className="message-markdown">
        {stable && <StableMarkdownSegment content={stable} />}
        {tail && renderFencedContent(tail, true)}
      </div>
    </CitationContext.Provider>
  )
}

export const AssistantMarkdown = memo(AssistantMarkdownInner)
