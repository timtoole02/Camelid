import { useCallback, useContext, useEffect, useRef, useState } from 'react'
import { OutputPreview, OutputReviewContext } from './OutputActions'
import { Button } from '../ui/Button'
import { IconChevronDown, IconClose, IconFile } from '../ui/icons'
import { downloadOutput, MAX_OUTPUT_BYTES } from '../../lib/outputFiles.js'
import { appStorage } from '../../lib/appStorage.js'
import { wrapDialogFocus } from '../ui/Modal'

export function ConversationFiles({ files, selectedId, onSelect, onClose, conversationId }) {
  const panel = useRef(null)
  const onReview = useContext(OutputReviewContext)
  const [source, setSource] = useState(false)
  const [error, setError] = useState('')
  const [names, setNames] = useState(() => {
    try { return JSON.parse(appStorage.getItem('camelid.outputNames.' + conversationId) || '{}') || {} } catch { return {} }
  })
  const [mobile, setMobile] = useState(() => window.matchMedia('(max-width: 900px)').matches)
  useEffect(() => {
    const media = window.matchMedia('(max-width: 900px)')
    const update = () => setMobile(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    const chat = panel.current?.previousElementSibling
    if (!mobile || !chat) return undefined
    const previous = chat.inert
    chat.inert = true
    return () => { chat.inert = previous }
  }, [mobile])
  useEffect(() => {
    const previous = document.activeElement
    panel.current?.querySelector('button')?.focus({ preventScroll: true })
    return () => { if (previous?.isConnected) previous.focus({ preventScroll: true }) }
  }, [])
  useEffect(() => { setSource(false); setError('') }, [selectedId])
  const output = files.find(file => file.id === selectedId) || files.at(-1)
  const name = output && (names[output.id] ?? output.name)
  const rename = value => {
    const next = { ...names, [output.id]: value }
    setNames(next)
    appStorage.setItem('camelid.outputNames.' + conversationId, JSON.stringify(next))
  }
  const download = useCallback(() => {
    try { downloadOutput(output, name); setError('') } catch { setError('Could not start the download. Try again.') }
  }, [output, name])
  return <aside ref={panel} className="conversation-files" role={mobile ? 'dialog' : 'region'} aria-modal={mobile || undefined} aria-label="Conversation files" tabIndex={-1}
    onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } else if (mobile) wrapDialogFocus(event, panel.current) }}>
    <header className="conversation-files__head"><h2><IconFile size={16} />Files <small>{files.length}</small></h2><button type="button" aria-label="Close conversation files" onClick={onClose}><IconClose size={18} /></button></header>
    {!output ? <p className="conversation-files__empty">Files from Camelid’s replies and tool results will appear here. Documents you attach appear on your message.</p> : <>
      <label className="conversation-files__select">Files in this conversation<select aria-label="Choose conversation file" value={output.id} onChange={event => onSelect(event.target.value)}>{files.map((file, index) => <option key={file.id} value={file.id}>{index + 1}. {names[file.id] || file.name}</option>)}</select></label>
      <div className="output-meta"><label>Filename<input aria-label="Output filename" value={name} onChange={event => rename(event.target.value)} maxLength={120} /></label><small>{output.mime} · {output.bytes.toLocaleString()} bytes</small></div>
      {output.preview !== 'image' && <div className="conversation-files__tabs" role="group" aria-label="File view"><button type="button" aria-pressed={!source} onClick={() => setSource(false)}>Preview</button><button type="button" aria-pressed={source} onClick={() => setSource(true)}>Source</button></div>}
      <div className="output-preview">{output.bytes > MAX_OUTPUT_BYTES ? <p className="output-note">Preview limit: 256 KB. Download to view the complete file.</p> : source ? <pre>{output.text}</pre> : <OutputPreview output={output} />}</div>
      <footer><Button variant="primary" onClick={download}>Download file</Button>{onReview && typeof output.text === 'string' && <Button disabled={output.bytes > MAX_OUTPUT_BYTES} onClick={() => onReview({ ...output, name })}>Review file change</Button>}</footer>
      {error && <p role="alert">{error}</p>}
    </>}
  </aside>
}

// Pinned above the composer so generated files stay in reach however far the thread scrolls.
export function ConversationFilesTray({ files, onOpen }) {
  const [open, setOpen] = useState(true)
  return <section className="files-tray" aria-label="Files from this conversation">
    <button type="button" className="files-tray__head" aria-expanded={open} onClick={() => setOpen(!open)}>
      <IconChevronDown size={14} className={open ? '' : 'is-collapsed'} />
      {files.length} {files.length === 1 ? 'file' : 'files'} from this conversation
    </button>
    {open && <ul className="files-tray__list">{files.map(file => <li key={file.id}>
      <button type="button" className="files-tray__file" onClick={() => onOpen(file.id)} title={`Open ${file.name}`}>
        <IconFile size={14} /><span className="files-tray__name">{file.name}</span><small>{file.mime} · {file.bytes.toLocaleString()} bytes</small>
      </button>
    </li>)}</ul>}
  </section>
}
