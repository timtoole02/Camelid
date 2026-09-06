import { useEffect, useMemo, useRef, useState } from 'react'
import { IconBolt, IconCheck, IconScale, IconSend, IconStop } from '../components/ui/icons'
import { AssistantMarkdown } from '../lib/markdown'
import { formatModelLabel } from '../lib/formatters'
import { loadLocalModelForChat, modelFilenameFromPath } from '../lib/modelActivation'
import { modelRuntimeIdMatches } from '../lib/modelState'
import { readStreamingChatCompletion } from '../lib/chatCompletionStream'
import {
  arenaDefaultModelA,
  arenaModelIsAlreadyReady,
  arenaModelChoices,
  arenaSelectionsAreReady,
  runArenaSequentially,
} from '../lib/arenaModels'

const PRESET_PROMPTS = [
  'Explain how speculative decoding works in local LLMs in 3 concise bullet points.',
  'Write a clean, memory-safe LRU cache implementation in Rust with unit tests.',
  'Compare the architectural trade-offs of Mixtral MoE vs dense Llama 3.2.',
  'Draft a persuasive, professional email proposing local offline AI inference for our engineering team.',
]

const EMPTY_RESULT = {
  output: '',
  ttft: null,
  tokensPerSec: null,
  totalTokens: 0,
  elapsedMs: 0,
  status: 'idle',
  error: '',
}

function phaseLabel(status) {
  if (status === 'checking') return 'Checking model'
  if (status === 'loading') return 'Loading model'
  if (status === 'generating') return 'Generating'
  if (status === 'queued') return 'Waiting for Model A'
  if (status === 'done') return 'Complete'
  if (status === 'stopped') return 'Stopped'
  if (status === 'error') return 'Needs attention'
  return ''
}

function modelOptionLabel(model, runtime) {
  const label = formatModelLabel(model.name || model.id)
  return modelRuntimeIdMatches(model, runtime) ? `${label} · Loaded` : label
}

function ArenaPane({ side, color, modelId, setModelId, models, runtime, state, disabled, otherModelId, onChange }) {
  return (
    <section className="arena-card" aria-label={`Model ${side} result`}>
      <div className="arena-card__header">
        <div className="arena-card__identity">
          <span className="arena-card__side" style={{ background: color, color: side === 'A' ? '#000' : '#fff' }}>Model {side}</span>
          <label className="sr-only" htmlFor={`arena-model-${side.toLowerCase()}`}>Choose Model {side}</label>
          <select
            id={`arena-model-${side.toLowerCase()}`}
            aria-label={`Choose Model ${side}`}
            value={modelId}
            onChange={(event) => {
              setModelId(event.target.value)
              onChange()
            }}
            disabled={disabled}
            className="arena-model-select"
          >
            <option value="">Choose a local model…</option>
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={model.id === otherModelId}>
                {modelOptionLabel(model, runtime)}
              </option>
            ))}
          </select>
        </div>

        <div className="arena-card__telemetry" aria-live="polite">
          {phaseLabel(state.status) && <span className={`arena-phase arena-phase--${state.status}`}>{phaseLabel(state.status)}</span>}
          {state.ttft !== null && <span title="Time to first generated token">TTFT <strong>{state.ttft}ms</strong></span>}
          {state.tokensPerSec !== null && <span title="Generation speed">Speed <strong>{state.tokensPerSec} tok/s</strong></span>}
          {state.totalTokens > 0 && <span>{state.totalTokens} tokens</span>}
        </div>
      </div>

      <div className="arena-card__body" aria-busy={['checking', 'loading', 'generating', 'queued'].includes(state.status) || undefined}>
        {state.output ? (
          <AssistantMarkdown content={state.output} streaming={state.status === 'generating'} />
        ) : state.error ? (
          <div className="arena-card__error" role="alert">{state.error}</div>
        ) : ['checking', 'loading', 'generating', 'queued'].includes(state.status) ? (
          <div className="arena-card__placeholder">{phaseLabel(state.status)}…</div>
        ) : (
          <div className="arena-card__placeholder">Choose a model and run the same prompt on both sides.</div>
        )}
      </div>
    </section>
  )
}

export default function ArenaView({ models = [], runtime, apiBase = '', loadDashboard, setTab }) {
  const eligibleModels = useMemo(() => arenaModelChoices(models, runtime), [models, runtime])
  const eligibleIds = useMemo(() => new Set(eligibleModels.map((model) => model.id)), [eligibleModels])
  const [modelA, setModelA] = useState('')
  const [modelB, setModelB] = useState('')
  const [prompt, setPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [stateA, setStateA] = useState(EMPTY_RESULT)
  const [stateB, setStateB] = useState(EMPTY_RESULT)
  const [vote, setVote] = useState(null)
  const activeControllerRef = useRef(null)
  const runIdRef = useRef(0)

  useEffect(() => {
    setModelA((current) => eligibleIds.has(current) ? current : arenaDefaultModelA(models, runtime))
    setModelB((current) => eligibleIds.has(current) ? current : '')
  }, [eligibleIds, models, runtime])

  useEffect(() => () => {
    runIdRef.current += 1
    activeControllerRef.current?.abort()
  }, [])

  const clearResults = () => {
    setStateA(EMPTY_RESULT)
    setStateB(EMPTY_RESULT)
    setVote(null)
  }

  const stopAll = () => {
    runIdRef.current += 1
    activeControllerRef.current?.abort()
    activeControllerRef.current = null
    setIsGenerating(false)
    setStateA((state) => ({ ...state, status: ['checking', 'loading', 'generating'].includes(state.status) ? 'stopped' : state.status }))
    setStateB((state) => ({ ...state, status: ['checking', 'loading', 'generating', 'queued'].includes(state.status) ? 'stopped' : state.status }))
  }

  const runModel = async (model, side, promptText, controller, runId) => {
    const setState = side === 'a' ? setStateA : setStateB
    const fetchWithSignal = (input, init = {}) => fetch(input, { ...init, signal: controller.signal })
    const base = String(apiBase || '').replace(/\/$/, '')
    const updateIfCurrent = (updater) => {
      if (runIdRef.current === runId && !controller.signal.aborted) setState(updater)
    }

    setState({ ...EMPTY_RESULT, status: 'checking' })
    const filename = modelFilenameFromPath(model.model_path)
    let health = {}
    try {
      const healthResponse = await fetchWithSignal(`${base}/v1/health`)
      health = healthResponse.ok ? await healthResponse.json().catch(() => ({})) : {}
    } catch (error) {
      if (error?.name === 'AbortError') return false
      // The authoritative load helper below will surface a useful connection
      // or backend error if this inexpensive active-model check could not run.
    }
    const activeIdentity = health.active_model_id || ''
    const alreadyReady = arenaModelIsAlreadyReady(model, health)
    const loadResult = alreadyReady
      ? { ok: true, id: activeIdentity }
      : await loadLocalModelForChat({
          apiBase,
          filename,
          path: model.model_path,
          modelId: model.id,
          model,
          fetchImpl: fetchWithSignal,
          onStage: (stage) => updateIfCurrent((state) => ({ ...state, status: stage === 'loading' ? 'loading' : 'checking' })),
        })

    if (controller.signal.aborted || runIdRef.current !== runId) return false
    if (!loadResult.ok) {
      setState({ ...EMPTY_RESULT, status: 'error', error: loadResult.message || 'Camelid could not load this model.' })
      return false
    }

    setState({ ...EMPTY_RESULT, status: 'generating' })
    const requestStartedAt = performance.now()
    let firstTokenAt = null
    let firstTokenCount = 0

    try {
      const response = await fetchWithSignal(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: loadResult.id || model.id,
          messages: [{ role: 'user', content: promptText }],
          stream: true,
          stream_options: { include_usage: true },
          max_tokens: 256,
          temperature: 0,
        }),
      })

      const streamed = await readStreamingChatCompletion(response, (_delta, fullContent) => {
        updateIfCurrent((state) => ({ ...state, output: fullContent }))
      }, {
        onStreamEvent(event) {
          if (!['content', 'reasoning', 'usage'].includes(event.type)) return
          const now = performance.now()
          const tokens = Number(event.completionTokens) || 0
          if (tokens > 0 && firstTokenAt === null) {
            firstTokenAt = now
            firstTokenCount = tokens
          }
          const decodeTokens = firstTokenAt === null ? 0 : Math.max(0, tokens - firstTokenCount)
          const decodeMs = firstTokenAt === null ? 0 : now - firstTokenAt
          const speed = decodeTokens > 0 && decodeMs > 0 ? (decodeTokens / (decodeMs / 1000)).toFixed(1) : null
          updateIfCurrent((state) => ({
            ...state,
            ttft: firstTokenAt === null ? state.ttft : Math.round(firstTokenAt - requestStartedAt),
            tokensPerSec: speed ?? state.tokensPerSec,
            totalTokens: tokens,
            elapsedMs: Math.round(now - requestStartedAt),
          }))
        },
      })

      if (controller.signal.aborted || runIdRef.current !== runId) return false
      const elapsedMs = Math.round(performance.now() - requestStartedAt)
      if (!streamed.content) {
        setState((state) => ({
          ...state,
          status: 'error',
          totalTokens: streamed.completionTokens || state.totalTokens,
          elapsedMs,
          error: 'The model completed without any visible answer text.',
        }))
        return false
      }
      setState((state) => ({
        ...state,
        output: streamed.content,
        status: 'done',
        totalTokens: streamed.completionTokens || state.totalTokens,
        elapsedMs,
      }))
      return true
    } catch (error) {
      if (error?.name === 'AbortError') {
        setState((state) => ({ ...state, status: 'stopped' }))
      } else {
        setState((state) => ({ ...state, status: 'error', error: error?.message || 'Generation failed.' }))
      }
      return false
    }
  }

  const handleSend = async (event) => {
    event?.preventDefault()
    if (!prompt.trim() || isGenerating || !arenaSelectionsAreReady(modelA, modelB)) return

    const selectedA = eligibleModels.find((model) => model.id === modelA)
    const selectedB = eligibleModels.find((model) => model.id === modelB)
    if (!selectedA || !selectedB) return

    const runId = runIdRef.current + 1
    runIdRef.current = runId
    const controller = new AbortController()
    activeControllerRef.current = controller
    const promptText = prompt.trim()
    setVote(null)
    setIsGenerating(true)
    setStateA({ ...EMPTY_RESULT, status: 'checking' })
    setStateB({ ...EMPTY_RESULT, status: 'queued' })

    try {
      await runArenaSequentially({
        modelA: selectedA,
        modelB: selectedB,
        signal: controller.signal,
        runModel: (model, side) => runModel(model, side, promptText, controller, runId),
      })
    } finally {
      if (runIdRef.current === runId) {
        activeControllerRef.current = null
        setIsGenerating(false)
        await loadDashboard?.({ silent: true })
      }
    }
  }

  const selectionsReady = arenaSelectionsAreReady(modelA, modelB)
  const canCompare = Boolean(prompt.trim() && selectionsReady && !isGenerating)
  const bothCompleted = stateA.status === 'done' && stateB.status === 'done' && stateA.output && stateB.output

  return (
    <div className="arena-view">
      <header className="arena-header">
        <div className="arena-title">
          <span className="arena-title__icon"><IconScale size={20} /></span>
          <div>
            <h1>Model Arena</h1>
            <p>Compare two local models with the same prompt and live performance details.</p>
          </div>
        </div>
        {vote && <div className="arena-vote-receipt"><IconCheck size={14} /> Voted: {vote}</div>}
      </header>

      <div className="arena-safety-note" role="note">
        Camelid runs Model A, then safely switches to Model B. This avoids loading two large models into memory at once.
      </div>

      {eligibleModels.length < 2 ? (
        <div className="arena-empty">
          <h2>Two local chat models are needed</h2>
          <p>Download or register another generation-capable GGUF before starting a comparison.</p>
          {setTab && <button type="button" className="cx-btn cx-btn--tonal cx-btn--md" onClick={() => setTab('library')}>Open Models</button>}
        </div>
      ) : (
        <>
          <div className="arena-panes">
            <ArenaPane
              side="A"
              color="#38bdf8"
              modelId={modelA}
              setModelId={setModelA}
              models={eligibleModels}
              runtime={runtime}
              state={stateA}
              disabled={isGenerating}
              otherModelId={modelB}
              onChange={clearResults}
            />
            <ArenaPane
              side="B"
              color="#a855f7"
              modelId={modelB}
              setModelId={setModelB}
              models={eligibleModels}
              runtime={runtime}
              state={stateB}
              disabled={isGenerating}
              otherModelId={modelA}
              onChange={clearResults}
            />
          </div>

          {bothCompleted && !vote && (
            <div className="arena-voting" aria-label="Choose the better response">
              <span>Which model gave a better answer?</span>
              <button type="button" className="cxcomposer__tool" onClick={() => setVote('Model A')}>Model A was better</button>
              <button type="button" className="cxcomposer__tool" onClick={() => setVote('Tie')}>Both were equal</button>
              <button type="button" className="cxcomposer__tool" onClick={() => setVote('Model B')}>Model B was better</button>
            </div>
          )}

          <footer className="arena-footer">
            <div className="arena-presets" aria-label="Prompt suggestions">
              {PRESET_PROMPTS.map((preset) => (
                <button key={preset} type="button" onClick={() => setPrompt(preset)} disabled={isGenerating}>
                  <IconBolt size={12} /> {preset.length > 40 ? `${preset.slice(0, 40)}…` : preset}
                </button>
              ))}
            </div>

            <form onSubmit={handleSend} className="arena-composer">
              <label className="sr-only" htmlFor="arena-prompt">Comparison prompt</label>
              <input
                id="arena-prompt"
                type="text"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                disabled={isGenerating}
                placeholder={selectionsReady ? 'Type one prompt for both models…' : 'Choose two different models first…'}
              />
              {isGenerating ? (
                <button type="button" className="arena-stop" onClick={stopAll}><IconStop size={16} /> Stop</button>
              ) : (
                <button type="submit" className="arena-compare" disabled={!canCompare}><IconSend size={16} /> Compare</button>
              )}
            </form>
          </footer>
        </>
      )}
    </div>
  )
}
