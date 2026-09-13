import { useEffect, useState } from 'react'
import { ConfirmDialog } from '../ui/ConfirmDialog'
import {
  SAMPLING_PARAMS,
  isSamplingParamSupported,
  loadSavedSamplingParams,
  saveSamplingParams,
} from '../../lib/samplingContract'
import { appStorage } from '../../lib/appStorage.js'

const SYSTEM_PROMPT_STORAGE_KEY = 'camelid.systemPrompt'
const SYSTEM_PROMPT_PRESETS_KEY = 'camelid.systemPromptPresets'

const readStoredPrompt = () => {
  if (typeof window === 'undefined') return ''
  return appStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY) || ''
}

const readPresets = () => {
  if (typeof window === 'undefined') return []
  try {
    const saved = JSON.parse(appStorage.getItem(SYSTEM_PROMPT_PRESETS_KEY) || '[]')
    return Array.isArray(saved) ? saved.filter((p) => p?.name && typeof p.content === 'string') : []
  } catch {
    return []
  }
}

/* Generation controls drawer (Phase 2).

   System prompt: editable — system messages are part of the supported chat
   surface (the code-first policy already sends one) and apply to the next send.

   Sampling parameters: rendered from the /api/capabilities contract. A control
   is editable ONLY when the contract advertises a supported row for that exact
   parameter; otherwise the row is visibly guarded (I3). Today the contract
   advertises none, so chat stays greedy temperature=0 — exactly the lane the
   parity evidence covers. */
export function ChatControls({ capabilities, modelId, onClose, globalPrompt, onGlobalPromptChange, busy = false }) {
  const apiFeatures = capabilities?.api_features || []
  const [localSystemPrompt, setSystemPrompt] = useState(readStoredPrompt)
  const systemPrompt = globalPrompt ?? localSystemPrompt
  const [presets, setPresets] = useState(readPresets)
  const [presetName, setPresetName] = useState('')
  const [confirmClearPresets, setConfirmClearPresets] = useState(false)
  const [savedParams, setSavedParams] = useState(() => loadSavedSamplingParams(modelId))

  useEffect(() => {
    setSavedParams(loadSavedSamplingParams(modelId))
  }, [modelId])

  const persistPrompt = (value) => {
    if (busy) return
    setSystemPrompt(value)
    onGlobalPromptChange?.(value)
    if (typeof window !== 'undefined') appStorage.setItem(SYSTEM_PROMPT_STORAGE_KEY, value)
  }
  const persistPresets = (next) => {
    setPresets(next)
    if (typeof window !== 'undefined') appStorage.setItem(SYSTEM_PROMPT_PRESETS_KEY, JSON.stringify(next))
  }
  const savePreset = () => {
    const name = presetName.trim()
    if (!name || !systemPrompt.trim()) return
    persistPresets([...presets.filter((p) => p.name !== name), { name, content: systemPrompt }])
    setPresetName('')
  }
  const updateParam = (key, rawValue) => {
    const next = { ...savedParams }
    if (rawValue === '' || rawValue === null) delete next[key]
    else next[key] = Number.isNaN(Number(rawValue)) ? rawValue : Number(rawValue)
    setSavedParams(next)
    saveSamplingParams(modelId, next)
  }

  return (
    <section className="chat-controls" aria-label="Generation controls">
      <header className="chat-controls__head">
        <h3>Generation controls</h3>
        <button type="button" className="cxturn__action" onClick={onClose}>Close</button>
      </header>

      <div className="chat-controls__group">
        <div className="chat-controls__group-head">
          <span className="cx-caption chat-controls__label">Global system prompt</span>
          <span className="chat-controls__note">applies to the next send · stored locally</span>
        </div>
        <textarea
          className="chat-controls__prompt"
          rows={3}
          disabled={busy}
          value={systemPrompt}
          placeholder="Optional system prompt for local chat (leave empty for default behavior)"
          onChange={(event) => persistPrompt(event.target.value)}
          aria-label="System prompt"
        />
        <div className="chat-controls__presets">
          {presets.length > 0 && (
            <select
              className="chat-controls__preset-select"
              aria-label="Load system prompt preset"
              value=""
              onChange={(event) => {
                const preset = presets.find((p) => p.name === event.target.value)
                if (preset) persistPrompt(preset.content)
              }}
            >
              <option value="" disabled>Load preset…</option>
              {presets.map((preset) => <option key={preset.name} value={preset.name}>{preset.name}</option>)}
            </select>
          )}
          <input
            className="chat-controls__preset-name"
            placeholder="Save as preset…"
            value={presetName}
            onChange={(event) => setPresetName(event.target.value)}
            aria-label="Preset name"
          />
          <button type="button" className="cxturn__action" onClick={savePreset} disabled={!presetName.trim() || !systemPrompt.trim()}>Save</button>
          {presets.length > 0 && (
            <button
              type="button"
              className="cxturn__action"
              onClick={() => setConfirmClearPresets(true)}
              title="Delete all presets"
            >
              Clear presets
            </button>
          )}
        </div>
        <ConfirmDialog
          open={confirmClearPresets}
          title="Delete all presets?"
          detail={`This deletes ${presets.length === 1 ? 'your saved system prompt preset' : `all ${presets.length} saved system prompt presets`}. This can't be undone.`}
          confirmLabel="Delete all"
          onCancel={() => setConfirmClearPresets(false)}
          onConfirm={() => {
            persistPresets([])
            setConfirmClearPresets(false)
          }}
        />
      </div>

      <div className="chat-controls__group">
        <div className="chat-controls__group-head">
          <span className="cx-caption chat-controls__label">Sampling</span>
          <span className="chat-controls__note">These unlock when the local server supports adjusting them.</span>
        </div>
        <ul className="chat-controls__params">
          {SAMPLING_PARAMS.map((param) => {
            const supported = isSamplingParamSupported(apiFeatures, param.key)
            return (
              <li key={param.key} className="chat-controls__param">
                <span className="chat-controls__param-name">{param.label}</span>
                <span className="chat-controls__param-current">{param.current}</span>
                {supported ? (
                  <input
                    className="chat-controls__param-input"
                    value={savedParams[param.key] ?? ''}
                    onChange={(event) => updateParam(param.key, event.target.value)}
                    aria-label={`${param.label} value`}
                  />
                ) : (
                  <span className="chat-controls__param-current" title="Locked" aria-label={`${param.label} locked`}>—</span>
                )}
              </li>
            )
          })}
        </ul>
      </div>
    </section>
  )
}

export default ChatControls
