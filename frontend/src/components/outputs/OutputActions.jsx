import { createContext, lazy, Suspense, useContext, useEffect, useMemo, useState } from 'react'
import { Modal } from '../ui/Modal'
import { Button } from '../ui/Button'
import { csvPreview, downloadOutput, MAX_OUTPUT_BYTES, previewDocument, textOutput, toolOutputs } from '../../lib/outputFiles.js'
import '../../styles/outputs.css'

const MarkdownPreview = lazy(() => import('../../lib/markdown.jsx').then(module => ({ default: module.AssistantMarkdown })))
export const OutputReviewContext = createContext(null)
const OutputPreviewContext = createContext(false)
function ImageOutput({ output }) {
  const [src, setSrc] = useState('')
  useEffect(() => { const url = URL.createObjectURL(output.blob); setSrc(url); return () => URL.revokeObjectURL(url) }, [output.blob])
  return <img className="output-image" src={src} alt={output.name} />
}
function Preview({ output }) {
  if (output.preview === 'image') return <ImageOutput output={output} />
  if (output.preview === 'html') return <><p className="output-note">Static preview. Scripts and external assets are disabled.</p><iframe title="Output preview" sandbox="" referrerPolicy="no-referrer" srcDoc={previewDocument(output.text)} /></>
  if (output.preview === 'markdown') return <div className="output-markdown"><OutputPreviewContext.Provider value={true}><OutputReviewContext.Provider value={null}><Suspense fallback={<p>Loading preview…</p>}><MarkdownPreview content={output.text} /></Suspense></OutputReviewContext.Provider></OutputPreviewContext.Provider></div>
  if (output.preview === 'csv') {
    const parsed = csvPreview(output.text)
    return <>{parsed.error && <p role="alert">{parsed.error}</p>}<div className="output-table"><table><tbody>{parsed.rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div>{parsed.truncated && <p>Preview limited to 100 rows and 40 columns. Download includes the complete file.</p>}</>
  }
  let text = output.text
  if (output.preview === 'json') { try { text = JSON.stringify(JSON.parse(text), null, 2) } catch { return <><p role="alert">This output is not valid JSON.</p><pre>{text}</pre></> } }
  return <pre>{text}</pre>
}
export function OutputActions({ code, language, output: supplied, disabled = false }) {
  const onReview = useContext(OutputReviewContext)
  const insidePreview = useContext(OutputPreviewContext)
  const output = useMemo(() => supplied || textOutput(code, language), [supplied, code, language])
  const [open, setOpen] = useState(false)
  const [source, setSource] = useState(false)
  const [name, setName] = useState(output.name)
  const [error, setError] = useState('')
  useEffect(() => setName(output.name), [output.name])
  const oversized = output.bytes > MAX_OUTPUT_BYTES
  const download = () => { try { downloadOutput(output, name); setError('') } catch { setError('Could not start the download. Try again.') } }
  if (insidePreview) return null
  return <span className="output-actions">
    <Button size="sm" variant="ghost" disabled={disabled || oversized} onClick={() => { setSource(false); setOpen(true) }}>Preview</Button>
    <Button size="sm" variant="ghost" disabled={disabled} onClick={download}>Download</Button>
    {onReview && typeof output.text === 'string' && <Button size="sm" variant="ghost" disabled={disabled || oversized} onClick={() => onReview({ ...output, name })}>Review file change</Button>}
    {error && <span role="alert">{error}</span>}
    {oversized && <small>Preview limit: 256 KB</small>}
    <Modal open={open} onClose={() => setOpen(false)} title="Output preview" labelledById="output-preview-title" size="xl" className="output-modal" footer={<Button variant="primary" onClick={download}>Download file</Button>}>
      <div className="output-meta"><label>Filename<input aria-label="Output filename" value={name} onChange={e => setName(e.target.value)} maxLength={120} /></label><small>{output.mime} · {output.bytes.toLocaleString()} bytes</small>{output.preview !== 'text' && output.preview !== 'image' && <Button onClick={() => setSource(!source)}>{source ? 'Show preview' : 'Show source'}</Button>}</div>
      <div className="output-preview">{source ? <pre>{output.text}</pre> : <Preview output={output} />}</div>
    </Modal>
  </span>
}
export function ToolOutputGallery({ content }) {
  const outputs = useMemo(() => toolOutputs(content), [content])
  if (!outputs.length) return null
  return <div className="tool-outputs">{outputs.map((output, i) => <article key={i}><strong>{output.name}</strong><small>{output.mime} · {output.bytes.toLocaleString()} bytes</small><OutputActions output={output} /></article>)}</div>
}
