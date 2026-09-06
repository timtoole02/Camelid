import { useState, useRef, useEffect } from 'react'
import { IconSend, IconStop, IconCopy, IconCheck, IconClose, IconBolt } from '../components/ui/icons'
import { AssistantMarkdown } from '../lib/markdown'
import { formatModelLabel } from '../lib/formatters'

const EMPTY_MODELS = []

export default function SpotlightView({ models = EMPTY_MODELS }) {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [copied, setCopied] = useState(false)
  const [availableModels, setAvailableModels] = useState(models)
  const [activeModel, setActiveModel] = useState(() => models[0]?.id || '')
  const inputRef = useRef(null)
  const abortControllerRef = useRef(null)

  useEffect(() => {
    inputRef.current?.focus()
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (window.__TAURI__) {
          window.__TAURI__.window?.getCurrentWindow()?.hide?.()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    if (models.length > 0) {
      setAvailableModels(models)
      return undefined
    }

    let cancelled = false
    fetch('/v1/models')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((payload) => {
        if (!cancelled && Array.isArray(payload?.data)) setAvailableModels(payload.data)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [models])

  useEffect(() => {
    if (!activeModel && availableModels.length > 0) {
      setActiveModel(availableModels[0].id)
    }
  }, [availableModels, activeModel])

  const handleSend = async (customPrompt) => {
    const textToSend = (customPrompt || query).trim()
    if (!textToSend || isGenerating) return

    setResponse('')
    setIsGenerating(true)
    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          ...(activeModel ? { model: activeModel } : {}),
          messages: [{ role: 'user', content: textToSend }],
          stream: true,
          temperature: 0.7,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let partial = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        partial += chunk
        const lines = partial.split('\n')
        partial = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data:')) continue
          const dataStr = trimmed.slice(5).trim()
          if (dataStr === '[DONE]') break

          try {
            const parsed = JSON.parse(dataStr)
            const token = parsed.choices?.[0]?.delta?.content || ''
            if (token) {
              setResponse((prev) => prev + token)
            }
          } catch {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setResponse((prev) => prev + `\n\n*(Error: ${err.message})*`)
      }
    } finally {
      setIsGenerating(false)
    }
  }

  const handleSummarizeClipboard = async () => {
    try {
      const clipText = await navigator.clipboard.readText()
      if (clipText.trim()) {
        const p = `Summarize the following clipboard text concisely in bullet points:\n\n${clipText}`
        setQuery(p)
        handleSend(p)
      }
    } catch (e) {
      setResponse('*(Clipboard access denied or empty)*')
    }
  }

  const handleTranslateCode = async () => {
    try {
      const clipText = await navigator.clipboard.readText()
      if (clipText.trim()) {
        const p = `Analyze this code snippet, explain what it does, and provide an idiomatic Rust or Python translation:\n\n\`\`\`\n${clipText}\n\`\`\``
        setQuery(p)
        handleSend(p)
      }
    } catch (e) {
      setResponse('*(Clipboard access denied or empty)*')
    }
  }

  const handleCopy = async () => {
    if (!response) return
    await navigator.clipboard.writeText(response)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      margin: 0,
      padding: '12px',
      boxSizing: 'border-box',
      background: 'rgba(15, 23, 42, 0.85)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(56, 189, 248, 0.25)',
      borderRadius: '16px',
      display: 'flex',
      flexDirection: 'column',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, sans-serif',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.75)',
      overflow: 'hidden'
    }}>
      {/* Top Bar / Drag handle */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 8px 8px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#38bdf8', boxShadow: '0 0 8px #38bdf8' }} />
          <span style={{ fontSize: '12px', fontWeight: 700, letterSpacing: '0.5px' }}>CAMELID SPOTLIGHT</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <select
            value={activeModel}
            onChange={(e) => setActiveModel(e.target.value)}
            disabled={isGenerating}
            style={{ background: 'rgba(0, 0, 0, 0.4)', color: '#94a3b8', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '6px', fontSize: '11px', padding: '2px 6px' }}
          >
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>{formatModelLabel(m.name || m.id)}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => window.__TAURI__?.window?.getCurrentWindow()?.hide?.()}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', display: 'flex' }}
            title="Dismiss (Esc)"
          >
            <IconClose size={14} />
          </button>
        </div>
      </div>

      {/* Input bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 4px' }}>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSend() }}
          placeholder="Ask a quick question, summarize clipboard, or translate code..."
          disabled={isGenerating}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: '#fff',
            fontSize: '15px',
            lineHeight: 1.4,
          }}
        />
        {isGenerating ? (
          <button
            type="button"
            onClick={() => abortControllerRef.current?.abort()}
            style={{ background: '#ef4444', color: '#fff', border: 'none', borderRadius: '8px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '12px' }}
          >
            <IconStop size={14} /> Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => handleSend()}
            disabled={!query.trim()}
            style={{ background: '#38bdf8', color: '#000', border: 'none', borderRadius: '8px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '4px', cursor: query.trim() ? 'pointer' : 'default', opacity: query.trim() ? 1 : 0.4, fontWeight: 600, fontSize: '12px' }}
          >
            <IconSend size={14} /> Ask
          </button>
        )}
      </div>

      {/* Action Pills */}
      <div style={{ display: 'flex', gap: '6px', padding: '0 4px 10px', borderBottom: response ? '1px solid rgba(255, 255, 255, 0.08)' : 'none' }}>
        <button
          type="button"
          onClick={handleSummarizeClipboard}
          disabled={isGenerating}
          style={{ background: 'rgba(255, 255, 255, 0.06)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', padding: '4px 10px', fontSize: '11px', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
        >
          📋 Summarize Clipboard
        </button>
        <button
          type="button"
          onClick={handleTranslateCode}
          disabled={isGenerating}
          style={{ background: 'rgba(255, 255, 255, 0.06)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px', padding: '4px 10px', fontSize: '11px', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
        >
          🔤 Translate Code
        </button>
      </div>

      {/* Response Box */}
      {response && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 8px', fontSize: '13px', lineHeight: 1.6, position: 'relative' }}>
          <AssistantMarkdown content={response} />
          <div style={{ position: 'sticky', bottom: 0, right: 0, display: 'flex', justifyContent: 'flex-end', paddingTop: '8px' }}>
            <button
              type="button"
              onClick={handleCopy}
              style={{ background: 'rgba(30, 41, 59, 0.9)', border: '1px solid rgba(255, 255, 255, 0.15)', borderRadius: '6px', padding: '4px 10px', fontSize: '11px', color: copied ? '#4ade80' : '#cbd5e1', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              {copied ? <IconCheck size={12} /> : <IconCopy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
