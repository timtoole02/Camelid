export const MAX_OUTPUT_BYTES = 256 * 1024
const FORMATS = {
  html: ['html', 'text/html', 'html'], svg: ['svg', 'image/svg+xml', 'html'],
  json: ['json', 'application/json', 'json'], csv: ['csv', 'text/csv', 'csv'],
  markdown: ['md', 'text/markdown', 'markdown'], md: ['md', 'text/markdown', 'markdown'],
  javascript: ['js', 'text/javascript', 'text'], js: ['js', 'text/javascript', 'text'],
  typescript: ['ts', 'text/plain', 'text'], ts: ['ts', 'text/plain', 'text'],
  python: ['py', 'text/plain', 'text'], py: ['py', 'text/plain', 'text'],
  css: ['css', 'text/css', 'text'], rust: ['rs', 'text/plain', 'text'],
  bash: ['sh', 'text/plain', 'text'], shell: ['sh', 'text/plain', 'text'],
  yaml: ['yaml', 'text/yaml', 'text'], yml: ['yml', 'text/yaml', 'text'],
  sql: ['sql', 'text/plain', 'text'], xml: ['xml', 'application/xml', 'text'],
}
export const outputBytes = value => new TextEncoder().encode(String(value || '')).length
export function safeOutputName(value, fallback = 'output.txt') {
  const name = String(value || '').split(/[\\/]/).pop().replace(/[\x00-\x1f\x7f<>:"|?*]/g, '_').replace(/^[. ]+|[. ]+$/g, '').slice(0, 120)
  return name && !/^(con|prn|aux|nul|com\d|lpt\d)(\.|$)/i.test(name) ? name : fallback
}
export function textOutput(code, language = '', name) {
  const key = String(language).toLowerCase().split(/\s/)[0]
  const [ext, mime, preview] = FORMATS[key] || ['txt', 'text/plain', 'text']
  return { text: String(code || ''), name: safeOutputName(name, 'output.' + ext), mime, preview, bytes: outputBytes(code) }
}
export function downloadOutput(output, name = output.name) {
  const blob = output.blob || new Blob([output.text], { type: output.mime + ';charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = safeOutputName(name, output.name)
  document.body.appendChild(anchor); anchor.click(); anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
// Opaque-origin, script-free document. CSP is installed before any generated
// markup and prevents asset requests, nested frames, forms, and external CSS.
export function previewDocument(text) {
  return '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'none\'; style-src \'unsafe-inline\'; img-src data:; font-src \'none\'; connect-src \'none\'; frame-src \'none\'; object-src \'none\'; base-uri \'none\'; form-action \'none\'"><meta name="referrer" content="no-referrer"><style>body{margin:20px;overflow-wrap:anywhere}img,svg{max-width:100%}</style></head><body>' + text + '</body></html>'
}
export function csvPreview(text, maxRows = 100, maxColumns = 40) {
  const rows = []; let row = [], cell = '', quoted = false, truncated = false
  for (let i = 0; i <= text.length; i++) {
    const c = text[i]
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++ } else quoted = !quoted
    } else if ((c === ',' || c === '\n' || c === undefined) && !quoted) {
      if (row.length < maxColumns) row.push(cell); else truncated = true
      cell = ''
      if (c !== ',') {
        if (!(c === undefined && row.length === 1 && row[0] === '' && text.endsWith('\n'))) rows.push(row)
        row = []
        if (rows.length >= maxRows && i < text.length) { truncated = true; break }
      }
    } else if (c !== undefined && !(c === '\r' && !quoted && text[i + 1] === '\n')) cell += c
  }
  return { rows, truncated, error: quoted ? 'The CSV has an unclosed quoted field. Download the source to inspect it.' : '' }
}
export function toolOutputs(content) {
  let value
  try { value = typeof content === 'string' ? JSON.parse(content) : content } catch { return [] }
  const outputs = []
  for (const item of (Array.isArray(value?.content) ? value.content : []).slice(0, 16)) {
    const resource = item.type === 'resource' ? item.resource : null
    if (resource && typeof resource.text === 'string' && outputBytes(resource.text) <= MAX_OUTPUT_BYTES) {
      const mime = resource.mimeType || 'text/plain'
      const format = Object.keys(FORMATS).find(key => FORMATS[key][1] === mime) || ''
      outputs.push(textOutput(resource.text, format, resource.uri?.split(/[?#]/)[0]))
    }
    const encoded = item.type === 'image' ? item.data : resource?.blob
    const mime = item.type === 'image' ? item.mimeType : resource?.mimeType
    if (typeof encoded === 'string' && encoded.length <= MAX_OUTPUT_BYTES * 1.4 && /^image\/(png|jpeg|webp|gif)$/.test(mime || '')) {
      try {
        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0))
        outputs.push({ name: safeOutputName(resource?.uri, 'output.' + mime.split('/')[1]), mime, bytes: bytes.length, blob: new Blob([bytes], { type: mime }), preview: 'image' })
      } catch { /* malformed image remains visible in the raw tool result */ }
    }
  }
  return outputs
}
