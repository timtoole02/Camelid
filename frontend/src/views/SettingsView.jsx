import { useEffect, useRef, useState } from 'react'
import { Card, CardHeader, CardBody } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Chip } from '../components/ui/Chip'
import { StatusDot } from '../components/ui/StatusDot'
import { Field } from '../components/ui/Field'
import { CamelidMark } from '../components/ui/CamelidMark'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { IconPlay, IconStop, IconCopy, IconCheck, IconServer, IconSettings, IconMonitor, IconSun, IconMoon, IconNetwork, IconChevronRight } from '../components/ui/icons'
import { copyText } from '../lib/markdown'
import { getConfiguredMaxTokens, setConfiguredMaxTokens } from '../lib/responseLimits'
import { getStoredApiKey, setStoredApiKey } from '../lib/apiAuth'
import { appStorage } from '../lib/appStorage.js'
import { isLanChatOnly } from '../lib/apiSurface.js'
import { ResponseLengthControl } from '../components/settings/ResponseLengthControl'

const THEME_OPTS = [
  { value: 'system', label: 'System', Icon: IconMonitor },
  { value: 'light', label: 'Light', Icon: IconSun },
  { value: 'dark', label: 'Dark', Icon: IconMoon },
]

export default function SettingsView({
  authRequired = false,
  apiSurface = 'full',
  runtime,
  apiBase,
  setApiBase,
  refreshDashboard = null,
  backend,
  showNotice,
  themePreference = 'system',
  setThemePreference = () => {},
  onOpenCluster = () => {},
  conversationCount = 0,
  deleteAllConversations = null,
  selectedModel = null,
  capabilities = null,
}) {
  const [confirmWipe, setConfirmWipe] = useState(false)
  const [wiping, setWiping] = useState(false)
  const online = runtime?.status === 'online'
  const { status, command, setCommand, resolvedCommand, starting, start, stop } = backend
  const [copied, setCopied] = useState(false)
  const [apiBaseDraft, setApiBaseDraft] = useState(apiBase || 'http://127.0.0.1:8181')
  const [apiKeyDraft, setApiKeyDraft] = useState(getStoredApiKey)
  const [showAdvanced, setShowAdvanced] = useState(Boolean(command))
  const [showLogs, setShowLogs] = useState(false)
  const [maxTokens, setMaxTokens] = useState(() => getConfiguredMaxTokens(selectedModel?.id))
  const [gpu, setGpu] = useState(null)
  const [gpuBusy, setGpuBusy] = useState(false)
  const copyResetRef = useRef(null)
  const apiKeyRef = useRef(null)

  useEffect(() => () => { if (copyResetRef.current) window.clearTimeout(copyResetRef.current) }, [])

  useEffect(() => {
    if (!authRequired) return
    apiKeyRef.current?.focus()
  }, [authRequired])

  // GPU (CUDA) acceleration availability + on/off state. Only present on a
  // CUDA-capable build/host; the card stays hidden everywhere else.
  useEffect(() => {
    if (!online || isLanChatOnly(apiSurface)) { setGpu(null); return }
    let cancelled = false
    const base = (apiBase || '').replace(/\/$/, '')
    fetch(`${base}/api/runtime/gpu`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled) setGpu(d) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [online, apiBase, apiSurface])

  const toggleGpu = async (enabled) => {
    const base = (apiBase || '').replace(/\/$/, '')
    setGpuBusy(true)
    try {
      const r = await fetch(`${base}/api/runtime/gpu`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      const d = await r.json()
      setGpu(d)
      showNotice?.(`GPU acceleration ${d.enabled ? 'enabled' : 'disabled'}${d.device ? ` (${d.device})` : ''}.`, 'success')
    } catch {
      showNotice?.('Could not change GPU acceleration.', 'error')
    } finally {
      setGpuBusy(false)
    }
  }

  const handleCopy = async () => {
    await copyText(resolvedCommand || 'camelid serve')
    setCopied(true)
    if (copyResetRef.current) window.clearTimeout(copyResetRef.current)
    copyResetRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  const handleSaveApiBase = () => {
    const next = apiBaseDraft.trim()
    if (!next) { showNotice?.('API base cannot be empty.', 'error'); return }
    setApiBase(next)
    showNotice?.(`API base set to ${next.replace(/\/$/, '')}.`, 'success')
  }

  const handleSaveApiKey = () => {
    const next = apiKeyDraft.trim()
    setStoredApiKey(next)
    showNotice?.(
      next
        ? authRequired ? 'API key saved. Checking access…' : 'API key saved in this browser.'
        : 'API key cleared.',
      next && !authRequired ? 'success' : 'info',
    )
    refreshDashboard?.()
  }

  const handleMaxTokens = (value) => {
    setMaxTokens(value)
    setConfiguredMaxTokens(selectedModel?.id || '', value)
    // keep the legacy global key as the fallback for other models
    if (typeof window !== 'undefined') appStorage.setItem('camelid.maxTokens', String(value))
  }

  const statusTone = online ? 'ready' : status.running ? 'warn' : 'offline'
  const statusLabel = online ? 'Online' : status.running ? 'Starting…' : 'Offline'
  const noBinary = status.available && !resolvedCommand

  return (
    <div className="settings-view">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconSettings size={14} /> Settings</p>
          <h1>Settings</h1>
          <p className="cxv-sub">Start the local Camelid backend, point the UI at it, and tune appearance.</p>
        </div>
      </header>

      <Card>
        <CardHeader icon={<IconServer size={20} />} eyebrow="Backend" title="Camelid inference server" actions={
          <span className="settings-status"><StatusDot tone={statusTone} pulse={online} /> <strong>{statusLabel}</strong></span>
        } />
        <CardBody>
          <p className="settings-help">
            {online
              ? `Connected at ${(apiBase || '').replace(/\/$/, '') || 'the configured API base'}. Load a model from the Models page to start chatting.`
              : status.running
                ? 'The launched process is running — waiting for it to bind the port…'
                : status.available
                  ? 'Not running yet. Start it below — Camelid finds the built binary for you.'
                  : 'Not running. Start it from a terminal with the command below.'}
          </p>

          {status.available ? (
            <>
              <div className="settings-actions">
                <Button variant="primary" icon={<IconPlay size={16} />} onClick={start} loading={starting} disabled={online || starting || status.running}>
                  {online ? 'Running' : status.running ? 'Starting…' : 'Start backend'}
                </Button>
                {status.running && <Button variant="danger" icon={<IconStop size={16} />} onClick={stop}>Stop</Button>}
                <Button variant="ghost" icon={copied ? <IconCheck size={16} /> : <IconCopy size={16} />} onClick={handleCopy}>
                  {copied ? 'Copied' : 'Copy command'}
                </Button>
              </div>

              <p className={`settings-run ${noBinary ? 'is-warn' : ''}`}>
                {noBinary
                  ? 'No built camelid binary found automatically. Add a launch command under Advanced, or run cargo build --release.'
                  : <>Runs <code>{resolvedCommand}</code>{command.trim() ? '' : ' (auto-detected)'}.</>}
              </p>

              <button type="button" className="settings-disclosure" onClick={() => setShowAdvanced((v) => !v)} aria-expanded={showAdvanced}>
                {showAdvanced ? 'Hide' : 'Advanced'} — override launch command
              </button>
              {showAdvanced && (
                <Field hint="Leave blank to auto-detect the built binary. Runs from the repo root; e.g. cargo run --release -- serve.">
                  <input type="text" value={command} spellCheck={false} placeholder={status.detected || 'camelid serve'} onChange={(e) => setCommand(e.target.value)} />
                </Field>
              )}

              <div className="settings-logs">
                <button type="button" className="settings-disclosure" onClick={() => setShowLogs((v) => !v)} aria-expanded={showLogs}>
                  {showLogs ? 'Hide' : 'Show'} backend log
                </button>
                {showLogs && <pre className="settings-logs__pre">{status.logTail?.trim() || 'No output yet. Start the backend to see logs.'}</pre>}
              </div>
            </>
          ) : (
            <>
              <div className="settings-actions">
                <Button variant="primary" icon={copied ? <IconCheck size={16} /> : <IconCopy size={16} />} onClick={handleCopy}>
                  {copied ? 'Copied' : 'Copy restart command'}
                </Button>
              </div>
              <p className="settings-help settings-help--muted">
                Run <code>{resolvedCommand || 'camelid serve'}</code> in a terminal, then this turns Online
                automatically. This page is served by the engine itself, so it cannot start one for you.
              </p>
            </>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Connection" title={authRequired ? 'API key required' : 'API base URL'} />
        <CardBody>
          {authRequired && (
            <p className="settings-help" role="status" id="api-key-required-message">
              Camelid is online and requires the laptop&apos;s LAN Chat key. Ask the laptop owner to run <code>camelid lan-key</code> and share the displayed key directly.
            </p>
          )}
          <p className="settings-help">Where the UI sends requests. Change this to reach a backend on another host or port.</p>
          <div className="settings-inline">
            <input type="text" value={apiBaseDraft} spellCheck={false} onChange={(e) => setApiBaseDraft(e.target.value)} placeholder="http://127.0.0.1:8181" />
            <Button variant="tonal" onClick={handleSaveApiBase} disabled={apiBaseDraft.trim() === (apiBase || '')}>Save</Button>
          </div>
          <p className="settings-help">API key for authenticated LAN or remote backends. It is stored only in this browser.</p>
          <div className="settings-inline">
            <input
              ref={apiKeyRef}
              type="password"
              value={apiKeyDraft}
              autoComplete="off"
              spellCheck={false}
              aria-describedby={authRequired ? 'api-key-required-message' : undefined}
              onChange={(e) => setApiKeyDraft(e.target.value)}
              placeholder={authRequired ? 'Enter the server API key' : 'Optional API key'}
            />
            <Button variant="tonal" onClick={handleSaveApiKey}>Save key</Button>
          </div>
        </CardBody>
      </Card>

      {gpu?.available && (
        <Card>
          <CardHeader icon={<IconServer size={20} />} eyebrow="Hardware" title="GPU acceleration" actions={
            <span className="settings-status"><StatusDot tone={gpu.enabled ? 'ready' : 'offline'} pulse={gpu.enabled} /> <strong>{gpu.enabled ? 'On' : 'Off'}</strong></span>
          } />
          <CardBody>
            <p className="settings-help">
              Use {gpu.device || 'your GPU'} to speed up generation. Output is identical either way.
            </p>
            <p className="settings-help settings-help--muted">
              Smaller models benefit most; very large models may not speed up much on this engine version.
            </p>
            <div className="settings-actions">
              <Button
                variant={gpu.enabled ? 'tonal' : 'primary'}
                onClick={() => toggleGpu(!gpu.enabled)}
                loading={gpuBusy}
                disabled={gpuBusy}
              >
                {gpu.enabled ? 'Disable GPU acceleration' : 'Enable GPU acceleration'}
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {!isLanChatOnly(apiSurface) && (
        <Card interactive className="settings-cluster-card" onClick={onOpenCluster}>
          <CardHeader
            icon={<IconNetwork size={20} />}
            eyebrow="Infrastructure"
            title="Cluster"
            actions={<IconChevronRight size={20} />}
          />
          <CardBody>See the machines a Camelid fabric proxy routes across — which are serving, which are not, and why — read live from the proxy itself.</CardBody>
        </Card>
      )}

      <Card>
        <CardHeader eyebrow="Chat" title="Response length" />
        <CardBody>
          <p className="settings-help">How many tokens Camelid can generate per reply. Higher limits give more complete answers but take longer.</p>
          <ResponseLengthControl
            value={Number(maxTokens)}
            onChange={(next) => handleMaxTokens(next)}
            model={selectedModel}
            capabilities={capabilities}
            gemma4ServeLane={runtime?.gemma4_serve_lane}
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Chat" title="Local data" />
        <CardBody>
          <p className="settings-help">Conversations live only in this browser&apos;s storage. Deleting them does not touch models, memories, or anything on disk.</p>
          <div className="settings-select-row">
            {deleteAllConversations && (
              <Button
                variant="ghost"
                className="cxv-danger"
                disabled={conversationCount === 0}
                onClick={() => setConfirmWipe(true)}
              >
                {conversationCount === 0 ? 'No conversations stored' : `Delete all ${conversationCount} conversations…`}
              </Button>
            )}
          </div>
          <ConfirmDialog
            open={confirmWipe}
            title="Delete all conversations?"
            detail={`This removes ${conversationCount} conversation${conversationCount === 1 ? '' : 's'} from this browser only. Models, memories, and files on disk are not affected.`}
            confirmLabel="Delete all"
            busy={wiping}
            onConfirm={async () => {
              setWiping(true)
              try { await deleteAllConversations() } finally {
                setWiping(false)
                setConfirmWipe(false)
              }
            }}
            onCancel={() => setConfirmWipe(false)}
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader eyebrow="Appearance" title="Theme" />
        <CardBody>
          <div className="settings-theme">
            {THEME_OPTS.map(({ value, label, Icon }) => (
              <button
                key={value}
                type="button"
                className={`settings-theme__opt ${themePreference === value ? 'is-active' : ''}`}
                aria-pressed={themePreference === value}
                onClick={() => setThemePreference(value)}
              >
                <Icon size={18} /> <span>{label}</span>
              </button>
            ))}
          </div>
        </CardBody>
      </Card>

      <Card tone="muted">
        <div className="settings-about">
          <CamelidMark size={28} />
          <div>
            <strong>Camelid</strong>
            <p>Local, proof-carrying LLM inference. The UI talks to the backend over the OpenAI-compatible HTTP API.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
