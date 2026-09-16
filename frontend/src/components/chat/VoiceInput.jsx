import { useEffect, useRef, useState } from 'react'
import { IconMicrophone, IconStop } from '../ui/icons'
import { microphoneSupport, SpeechRecording } from '../../lib/speechRecording'
import './voice-input.css'

const ACTIVE = new Set(['requesting', 'recording', 'transcribing'])
export function VoiceInput({ apiBase, disabled, onTranscript, onBusyChange }) {
  const [phase, setPhase] = useState('checking')
  const [installed, setInstalled] = useState(false)
  const [panel, setPanel] = useState(false)
  const [message, setMessage] = useState('')
  const [seconds, setSeconds] = useState(0)
  const operation = useRef(0)
  const recording = useRef(null)
  const request = useRef(null)
  const locked = useRef(false)
  const callbacks = useRef({ onTranscript, onBusyChange })
  callbacks.current = { onTranscript, onBusyChange }
  const base = String(apiBase || window.location.origin).replace(/\/$/, '')

  async function api(path, init = {}, timeout = 150_000) {
    const controller = new AbortController()
    request.current = controller
    const timer = setTimeout(() => controller.abort(), timeout)
    try {
      const response = await fetch(`${base}/api/speech/${path}`, { ...init, signal: controller.signal })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        const error = new Error(data.error?.message || (response.status === 403 ? 'Set up voice input on the computer running Camelid first.' : `Voice input is unavailable (${response.status}).`))
        error.status = response.status
        throw error
      }
      return data
    } finally {
      clearTimeout(timer)
      if (request.current === controller) request.current = null
    }
  }
  useEffect(() => {
    const id = ++operation.current
    api('status', {}, 10_000).then((data) => {
      if (id !== operation.current) return
      setInstalled(Boolean(data.installed)); setPhase('idle')
    }).catch(() => { if (id === operation.current) setPhase('idle') })
    return () => {
      operation.current += 1
      request.current?.abort()
      recording.current?.close()
      callbacks.current.onBusyChange(false)
    }
  }, [base])
  useEffect(() => { callbacks.current.onBusyChange(ACTIVE.has(phase)) }, [phase])

  function cancel() {
    operation.current += 1
    request.current?.abort()
    recording.current?.close()
    recording.current = null
    locked.current = false
    setPhase('idle'); setPanel(false); setMessage('')
  }
  function fail(error, id) {
    if (id !== operation.current) return
    recording.current?.close(); recording.current = null
    locked.current = false
    if (error?.status === 412 || (error?.status === 422 && /model.*(file|verification|download)/i.test(error.message))) setInstalled(false)
    const permission = ['NotAllowedError', 'SecurityError'].includes(error?.name)
    setMessage(permission ? 'Microphone access was denied. Allow microphone access in your browser or system settings, then try again.' : error?.name === 'AbortError' ? 'Voice input timed out. Please try again.' : error?.message || 'Voice input failed. Please try again.')
    setPanel(true); setPhase('idle')
  }
  async function install() {
    if (locked.current) return
    locked.current = true
    const id = ++operation.current
    setPhase('downloading'); setMessage('Downloading the English speech model. This only needs an internet connection once.')
    try {
      await api('install', { method: 'POST' }, 650_000)
      if (id !== operation.current) return
      setInstalled(true); setPhase('idle'); setMessage('Voice input is ready. Press the microphone to start speaking.'); locked.current = false
    } catch (error) { fail(error, id) }
  }
  async function finish() {
    const capture = recording.current
    if (!capture) return
    recording.current = null
    const id = operation.current
    setPhase('transcribing'); setMessage('Transcribing on your Camelid server…')
    try {
      const audio = await capture.stop()
      if (id !== operation.current) return
      const result = await api('transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: audio })
      if (id !== operation.current) return
      if (result.text?.trim()) {
        callbacks.current.onTranscript(result.text.trim())
        setMessage('Added to your prompt. Review it and press Send.')
      } else setMessage('No speech detected. Try again closer to the microphone.')
      setPhase('idle'); locked.current = false
    } catch (error) { fail(error, id) }
  }
  async function start() {
    if (locked.current || disabled) return
    const unsupported = microphoneSupport()
    if (unsupported) { setPanel(true); setMessage(unsupported); return }
    setPanel(true); setMessage('')
    if (!installed) return
    locked.current = true
    const id = ++operation.current
    setSeconds(0); setPhase('requesting'); setMessage('Allow microphone access to begin.')
    const capture = new SpeechRecording(finish, setSeconds, (error) => fail(error, id))
    recording.current = capture
    try {
      await capture.start()
      if (id !== operation.current) { capture.close(); return }
      setPhase('recording'); setMessage('Listening. Press Stop when you finish (30 seconds maximum).')
    } catch (error) { fail(error, id) }
  }
  const active = ACTIVE.has(phase)
  const label = phase === 'recording' ? 'Stop recording' : phase === 'transcribing' ? 'Transcribing speech' : 'Dictate prompt'
  return <div className="cxvoice">
    <button type="button" className={`cxcomposer__tool cxvoice__mic ${phase === 'recording' ? 'is-recording' : ''}`}
      aria-label={label} title={label} aria-pressed={phase === 'recording'} aria-expanded={panel}
      disabled={(disabled && !active) || ['checking', 'requesting', 'transcribing', 'downloading'].includes(phase)}
      onClick={phase === 'recording' ? finish : start}>
      {phase === 'recording' ? <IconStop size={20} /> : <IconMicrophone size={20} />}
      {phase === 'recording' && <span>{seconds}s</span>}
    </button>
    {panel && <div className="cxvoice__panel" role="region" aria-label="Voice input">
      <strong>{phase === 'recording' ? `Listening · ${seconds} / 30s` : 'Voice input'}</strong>
      <p role="status" aria-live="polite">{message || (installed ? 'English dictation is processed on your Camelid server.' : 'Download the English speech model (154 MB) to dictate locally. Recordings are processed on your Camelid server and are not saved.')}</p>
      {!installed && phase === 'idle' && !microphoneSupport() && <button type="button" className="cxcomposer__tool" onClick={install}>Download speech model</button>}
      {phase === 'recording' && <button type="button" className="cxcomposer__tool" onClick={finish}>Stop and transcribe</button>}
      <button type="button" className="cxcomposer__tool" onClick={cancel}>{active ? 'Cancel' : phase === 'downloading' ? 'Dismiss' : 'Close'}</button>
    </div>}
  </div>
}
