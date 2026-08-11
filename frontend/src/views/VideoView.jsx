import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button } from '../components/ui/Button'
import { StatusDot } from '../components/ui/StatusDot'
import {
  IconExternal,
  IconPlay,
  IconRefresh,
  IconStop,
  IconVideo,
} from '../components/ui/icons'
import { formatBytes } from '../lib/formatters'

function endpoint(apiBase, path) {
  return `${String(apiBase || '').replace(/\/$/, '')}${path}`
}

async function readJson(response, fallback) {
  const payload = await response.json().catch(() => null)
  if (!response.ok) throw new Error(payload?.error?.message || fallback || `Request failed (${response.status})`)
  return payload
}

function paths(value) {
  return value.split(/[,\n]/).map((item) => item.trim()).filter(Boolean)
}

function timestamp(value) {
  if (!value) return 'just now'
  return new Date(value * 1000).toLocaleString()
}

function statusTone(status) {
  if (status === 'succeeded') return 'ready'
  if (status === 'failed') return 'error'
  if (status === 'running' || status === 'queued') return 'warn'
  return 'neutral'
}

export default function VideoView({ apiBase = '', showNotice }) {
  const [capabilities, setCapabilities] = useState(null)
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState('')
  const [modelsDir, setModelsDir] = useState('')
  const [loading, setLoading] = useState(true)
  const [choosingBundle, setChoosingBundle] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const [form, setForm] = useState({
    prompt: '',
    variant: 'fl2va',
    width: 640,
    height: 384,
    frames: 25,
    steps: 4,
    seed: 11,
    includeAudio: true,
    initImage: '',
    endImage: '',
    referenceImages: '',
    referenceVideos: '',
    referenceAudios: '',
  })

  const refresh = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true)
    try {
      const query = new URLSearchParams({ variant: form.variant, include_audio: String(form.includeAudio) })
      if (modelsDir.trim()) query.set('models_dir', modelsDir.trim())
      const [caps, list] = await Promise.all([
        fetch(endpoint(apiBase, `/api/video/capabilities?${query}`)).then((response) => readJson(response, 'Video readiness is unavailable.')),
        fetch(endpoint(apiBase, '/api/video/jobs')).then((response) => readJson(response, 'Video jobs are unavailable.')),
      ])
      const nextJobs = list?.data || []
      setCapabilities(caps)
      setJobs(nextJobs)
      setSelectedJobId((current) => {
        if (current && nextJobs.some((job) => job.id === current)) return current
        return nextJobs.find((job) => job.status === 'succeeded')?.id || nextJobs[0]?.id || ''
      })
      setError('')
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      if (!quiet) setLoading(false)
    }
  }, [apiBase, form.includeAudio, form.variant, modelsDir])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    const timer = window.setInterval(() => refresh({ quiet: true }), 4000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const selectedJob = jobs.find((job) => job.id === selectedJobId) || null
  const percent = capabilities?.expected_bytes
    ? Math.min(100, (Number(capabilities.downloaded_bytes || 0) / Number(capabilities.expected_bytes)) * 100)
    : 0
  const verifyingArtifact = capabilities?.artifacts?.find((artifact) => artifact.stage === 'verifying')
  const effectiveFrames = form.frames <= 5 ? 5 : Math.ceil((form.frames - 5) / 17) * 17 + 5
  const ref2va = form.variant === 'ref2va'
  const refsReady = !ref2va || Boolean(paths(form.referenceImages).length || paths(form.referenceVideos).length || paths(form.referenceAudios).length)
  const ready = Boolean(capabilities?.artifacts_ready && capabilities?.backend_ready)
  const canSubmit = ready && form.prompt.trim() && refsReady && !submitting
  const activeJobs = jobs.filter((job) => job.status === 'queued' || job.status === 'running')

  const update = (field, value) => setForm((current) => ({ ...current, [field]: value }))

  const chooseBundle = async () => {
    const invoke = window.__TAURI__?.core?.invoke
    if (!invoke) return
    setChoosingBundle(true)
    setError('')
    try {
      const selected = await invoke('choose_video_models_directory')
      if (selected) setModelsDir(selected)
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setChoosingBundle(false)
    }
  }

  const generate = async (event) => {
    event.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    setError('')
    try {
      const payload = {
        prompt: form.prompt.trim(),
        variant: form.variant,
        models_dir: modelsDir.trim() || capabilities?.models_dir,
        width: Number(form.width),
        height: Number(form.height),
        frames: Number(form.frames),
        steps: Number(form.steps),
        seed: Number(form.seed),
        include_audio: form.includeAudio,
        init_image: form.initImage.trim() || null,
        end_image: form.endImage.trim() || null,
        reference_images: paths(form.referenceImages),
        reference_videos: paths(form.referenceVideos),
        reference_audios: paths(form.referenceAudios),
        offload_to_cpu: true,
      }
      const response = await fetch(endpoint(apiBase, '/api/video/jobs'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const job = await readJson(response, 'Video generation could not start.')
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)])
      setSelectedJobId(job.id)
      showNotice?.('MiniMax-H3 video job queued.', 'ready')
    } catch (err) {
      const message = String(err?.message || err)
      setError(message)
      showNotice?.(message, 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const cancel = async (id) => {
    try {
      const response = await fetch(endpoint(apiBase, `/api/video/jobs/${id}/cancel`), { method: 'POST' })
      const job = await readJson(response, 'Video job could not be canceled.')
      setJobs((current) => current.map((item) => item.id === id ? job : item))
    } catch (err) {
      setError(String(err?.message || err))
    }
  }

  const readinessLabel = ready
    ? 'Ready to generate'
    : verifyingArtifact
      ? `Verifying ${String(verifyingArtifact.role || 'model').replaceAll('_', ' ')}`
    : capabilities?.downloaded_bytes > 0 && !capabilities?.artifacts_ready
      ? `Downloading bundle · ${percent.toFixed(1)}%`
      : 'Setup incomplete'

  return (
    <section className="video-view cxv">
      <header className="cxv-head video-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconVideo size={15} /> Experimental creation lane</p>
          <h1>Video Studio</h1>
          <p className="cxv-sub">Create local MiniMax-H3 video with the pinned 25&nbsp;GiB FL2VA bundle and a capability-checked local backend.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={ready ? 'ready' : percent > 0 ? 'warn' : 'offline'} pulse={ready || activeJobs.length > 0} label={readinessLabel} />
          <Button size="sm" variant="outline" icon={<IconRefresh size={16} />} loading={loading} onClick={() => refresh()}>Refresh</Button>
        </div>
      </header>

      {error && <div className="video-alert" role="alert">{error}</div>}

      <div className="video-readiness cxv-card">
        <div className="video-readiness__summary">
          <div>
            <span className="video-eyebrow">MiniMax-H3 bundle</span>
            <strong>{readinessLabel}</strong>
            <small>{formatBytes(capabilities?.downloaded_bytes || 0)} of {formatBytes(capabilities?.expected_bytes || 0)} on external storage</small>
          </div>
          <div className="video-readiness__flags">
            <StatusDot tone={capabilities?.artifacts_ready ? 'ready' : 'warn'} label={capabilities?.artifacts_ready ? 'Models ready' : verifyingArtifact ? 'Models verifying' : 'Models downloading'} />
            <StatusDot tone={capabilities?.backend_ready ? 'ready' : 'warn'} label={capabilities?.backend_ready ? 'H3 backend ready' : 'Building backend'} />
          </div>
        </div>
        <div className="video-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(percent)}>
          <span style={{ width: `${percent}%` }} />
        </div>
        <div className="video-paths">
          <label className="cx-field">
            <span className="cx-field__label">Model bundle folder</span>
            <span className="video-path-picker">
              <input value={modelsDir || capabilities?.models_dir || ''} onChange={(event) => setModelsDir(event.target.value)} placeholder="Auto-detect external drive" />
              {window.__TAURI__?.core?.invoke && <Button type="button" size="sm" variant="outline" loading={choosingBundle} onClick={chooseBundle}>Choose</Button>}
            </span>
          </label>
          <div><span>Backend</span><code>{capabilities?.backend || 'Checking sd-cli…'}</code></div>
          <div><span>Finished videos</span><code>{capabilities?.output_dir || 'Checking output folder…'}</code></div>
        </div>
        {!capabilities?.backend_ready && capabilities?.backend_error && <p className="video-readiness__note">{capabilities.backend_error}</p>}
      </div>

      <div className="video-workbench">
        <form className="cxv-card video-compose" onSubmit={generate}>
          <div className="cxv-section__head">
            <div><span className="video-eyebrow">New generation</span><h2>Direct the clip</h2></div>
            <span className="cxv-tag">24 FPS · WebM</span>
          </div>

          <label className="cx-field video-prompt">
            <span className="cx-field__label">Prompt</span>
            <textarea value={form.prompt} onChange={(event) => update('prompt', event.target.value)} placeholder="A cinematic close-up of a red panda walking through mist, soft morning light, slow camera dolly…" rows="5" />
          </label>

          <div className="video-mode-tabs" role="group" aria-label="Generation model variant">
            <button type="button" className={form.variant === 'fl2va' ? 'is-active' : ''} onClick={() => update('variant', 'fl2va')}>
              <strong>Text / first frame</strong><span>FL2VA · installed bundle</span>
            </button>
            <button type="button" className={form.variant === 'ref2va' ? 'is-active' : ''} onClick={() => update('variant', 'ref2va')}>
              <strong>Reference media</strong><span>REF2VA · separate model</span>
            </button>
          </div>

          {!ref2va && (
            <div className="video-grid video-grid--two">
              <label className="cx-field"><span className="cx-field__label">First-frame image path <em>optional</em></span><input value={form.initImage} onChange={(event) => update('initImage', event.target.value)} placeholder="Local path to start.png" /></label>
              <label className="cx-field"><span className="cx-field__label">Last-frame image path <em>optional</em></span><input value={form.endImage} onChange={(event) => update('endImage', event.target.value)} placeholder="Local path to end.png" /></label>
            </div>
          )}

          {ref2va && (
            <div className="video-reference-fields">
              <label className="cx-field"><span className="cx-field__label">Reference image paths</span><textarea rows="2" value={form.referenceImages} onChange={(event) => update('referenceImages', event.target.value)} placeholder="One local path per line" /></label>
              <label className="cx-field"><span className="cx-field__label">Reference video frame folders</span><textarea rows="2" value={form.referenceVideos} onChange={(event) => update('referenceVideos', event.target.value)} placeholder="One frame-directory path per line" /></label>
              <label className="cx-field"><span className="cx-field__label">Reference audio paths</span><textarea rows="2" value={form.referenceAudios} onChange={(event) => update('referenceAudios', event.target.value)} placeholder="One local audio path per line" /></label>
            </div>
          )}

          <button type="button" className="video-advanced-toggle" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}>
            {advanced ? 'Hide generation settings' : 'Show generation settings'}
          </button>

          {advanced && (
            <div className="video-grid video-grid--settings">
              <label className="cx-field"><span className="cx-field__label">Width</span><input type="number" min="64" max="4096" step="32" value={form.width} onChange={(event) => update('width', event.target.value)} /></label>
              <label className="cx-field"><span className="cx-field__label">Height</span><input type="number" min="64" max="4096" step="32" value={form.height} onChange={(event) => update('height', event.target.value)} /></label>
              <label className="cx-field"><span className="cx-field__label">Requested frames</span><input type="number" min="5" max="360" value={form.frames} onChange={(event) => update('frames', event.target.value)} /><span className="cx-field__hint">H3 renders {effectiveFrames} frames ({(effectiveFrames / 24).toFixed(1)}s)</span></label>
              <label className="cx-field"><span className="cx-field__label">Steps</span><input type="number" min="1" max="100" value={form.steps} onChange={(event) => update('steps', event.target.value)} /></label>
              <label className="cx-field"><span className="cx-field__label">Seed</span><input type="number" value={form.seed} onChange={(event) => update('seed', event.target.value)} /></label>
              <label className="video-check"><input type="checkbox" checked={form.includeAudio} onChange={(event) => update('includeAudio', event.target.checked)} /><span><strong>Generate stereo audio</strong><small>Uses the bundled audio VAE</small></span></label>
            </div>
          )}

          <div className="video-compose__foot">
            <div>
              {!ready && <small>Generation unlocks when the model files and backend are ready.</small>}
              {ref2va && !refsReady && <small>Add at least one reference input.</small>}
            </div>
            <Button type="submit" size="lg" variant="primary" icon={<IconPlay size={17} />} loading={submitting} disabled={!canSubmit}>Generate video</Button>
          </div>
        </form>

        <section className="cxv-card video-preview">
          <div className="cxv-section__head">
            <div><span className="video-eyebrow">Preview</span><h2>{selectedJob ? 'Generation output' : 'Your video appears here'}</h2></div>
            {selectedJob && <StatusDot tone={statusTone(selectedJob.status)} pulse={selectedJob.status === 'running'} label={selectedJob.status} />}
          </div>
          {selectedJob?.status === 'succeeded' ? (
            <video key={selectedJob.id} controls playsInline preload="metadata" src={endpoint(apiBase, selectedJob.content_url)} />
          ) : (
            <div className="video-preview__empty">
              <IconVideo size={42} />
              <strong>{selectedJob ? selectedJob.status === 'failed' ? 'Generation failed' : selectedJob.status === 'canceled' ? 'Generation canceled' : 'Rendering locally' : 'No completed video yet'}</strong>
              <p>{selectedJob?.error || (selectedJob ? 'MiniMax-H3 jobs are serialized to protect the GPU working set.' : 'Write a prompt and start a generation when setup finishes.')}</p>
            </div>
          )}
          {selectedJob && (
            <div className="video-preview__meta">
              <span>{selectedJob.width}×{selectedJob.height}</span><span>{selectedJob.effective_frames} frames</span><span>seed {selectedJob.seed}</span>
              <a href={endpoint(apiBase, selectedJob.log_url)} target="_blank" rel="noreferrer">Open log <IconExternal size={13} /></a>
            </div>
          )}
        </section>
      </div>

      <section className="cxv-card video-jobs">
        <div className="cxv-section__head"><div><span className="video-eyebrow">This runtime</span><h2>Video jobs</h2></div><span className="cxv-section__count">{jobs.length}</span></div>
        {jobs.length === 0 ? <p className="cxv-sub">No MiniMax-H3 jobs have been submitted in this server session.</p> : (
          <div className="video-job-list">
            {jobs.map((job) => (
              <div key={job.id} className={`video-job ${selectedJobId === job.id ? 'is-active' : ''}`}>
                <button type="button" className="video-job__select" onClick={() => setSelectedJobId(job.id)} aria-label={`Open ${job.prompt}`}>
                  <StatusDot tone={statusTone(job.status)} pulse={job.status === 'running'} />
                  <span className="video-job__copy"><strong>{job.prompt}</strong><small>{timestamp(job.created_at)} · {job.width}×{job.height} · {job.effective_frames} frames</small></span>
                </button>
                <span className="video-job__status">{job.status}</span>
                {(job.status === 'queued' || job.status === 'running') && <Button size="sm" variant="ghost" icon={<IconStop size={14} />} onClick={() => cancel(job.id)}>Cancel</Button>}
              </div>
            ))}
          </div>
        )}
      </section>

      <p className="video-license">Experimental external-backend bridge. MiniMax-H3 is governed by its <a href={capabilities?.license || 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE'} target="_blank" rel="noreferrer">Community License Agreement</a>. Text chat support is unchanged.</p>
    </section>
  )
}
