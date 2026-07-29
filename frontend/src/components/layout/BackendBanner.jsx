import { useEffect, useRef, useState } from 'react'
import { Button } from '../ui/Button'
import { StatusDot } from '../ui/StatusDot'
import { IconPlay, IconCopy, IconCheck, IconSettings } from '../ui/icons'
import { copyText } from '../../lib/markdown'

/* Shown on the copy button when the engine never told us where it lives.

   NOT presented as "the" restart command. `camelid serve` only runs when the
   binary is on PATH, which it is NOT for the population that actually hits this
   banner: someone who unzipped a release and double-clicked camelid.exe. Handing
   them a command that errors with "'camelid' is not recognized" is the same
   class of dead end this component exists to remove. */
const RESTART_COMMAND = 'camelid serve'

/* Where the engine's own path is cached while it is still answering. Written by
   App.jsx from the health payload; read here, because by the time this banner
   renders there is nobody left to ask. */
export const ENGINE_PATH_STORAGE_KEY = 'camelid.enginePath'
export const ENGINE_ADDR_STORAGE_KEY = 'camelid.engineAddr'

/* `serve` binds this when given no --addr. Anything else has to be passed back
   explicitly or the restart lands on the wrong port. */
const DEFAULT_ADDR = '127.0.0.1:8181'

function readStored(key) {
  if (typeof window === 'undefined') return ''
  try {
    return (window.localStorage.getItem(key) || '').trim()
  } catch {
    return ''
  }
}

function readEnginePath() {
  return readStored(ENGINE_PATH_STORAGE_KEY)
}

/* Build a command the user can paste and run, from an absolute executable path.

   Quoting is not cosmetic here. On Windows a path with a space MUST be quoted,
   and PowerShell — the default Windows shell — will not execute a quoted string
   as a command without the `&` call operator, so the naive quoted form fails on
   the very shell most Windows users are in. Paths without a space need no
   quoting and then run in both cmd and PowerShell unchanged, so that form is
   preferred whenever it is available.

   `--addr` is appended for any non-default bind. Leaving it off was a real
   defect: a bare `serve` falls back to 127.0.0.1:8181, which fails outright when
   something else already owns that port, and even on success comes up somewhere
   the tab showing this banner is not looking. */
export function restartCommandFor(executablePath, listenAddr) {
  const path = (executablePath || '').trim()
  if (!path) return RESTART_COMMAND
  const isWindowsPath = /^[A-Za-z]:[\\/]/.test(path) || path.startsWith('\\\\')
  const launcher = !path.includes(' ')
    ? path
    : (isWindowsPath ? `& "${path}"` : `"${path}"`)
  const addr = (listenAddr || '').trim()
  const addrFlag = addr && addr !== DEFAULT_ADDR ? ` --addr ${addr}` : ''
  return `${launcher} serve${addrFlag}`
}

const LOOPBACK_HOSTS = new Set(['localhost', '[::1]', '::1'])

/* Which machine is the engine supposed to be on? `camelid serve` is only sound
   advice when the UI is talking to THIS machine. Someone who reached a Camelid
   started with `--addr 0.0.0.0` from a different box must not be told to run the
   command on their own. An unset API base means same-origin, so the page's own
   host is the engine's host.

   The whole 127.0.0.0/8 block is loopback, not just 127.0.0.1. */
function isLoopbackHost(host) {
  if (!host) return true
  if (LOOPBACK_HOSTS.has(host)) return true
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host)
}

function engineHostFrom(apiBase, pageHref) {
  try {
    return new URL(apiBase || pageHref, pageHref).hostname
  } catch {
    return ''
  }
}

/* Shown across the app whenever the backend is offline.

   Three situations, and conflating them produced a real dead end:

   1. Dev server (`npm run dev`). The Vite plugin in vite.config.js can spawn the
      binary, so `status.available` is true and one-click Start genuinely works.

   2. Anything a user downloads, talking to an engine on this machine. That
      plugin is registered `apply: 'serve'`, so it does not exist in the shipped
      build and `status.available` is always false. This banner then rendered a
      lone "Settings" button, which navigated to a page whose only explanation
      was that the launcher needs the dev server: a control that looks like the
      fix, is not the fix, and explains nothing to someone who downloaded a
      release. Worse, this page is served BY the process that died, so it can
      never restart it — only the user, or the desktop app's supervisor, can.

   3. A UI pointed at an engine on another host. Same missing launcher, but the
      restart command would be advice for the wrong machine. */
export function BackendBanner({ backend, apiBase, onOpenSettings }) {
  const canStart = backend.status.available
  const running = backend.status.running
  const enginePath = readEnginePath()
  const restartCommand = enginePath
    ? restartCommandFor(enginePath, readStored(ENGINE_ADDR_STORAGE_KEY))
    : backend.resolvedCommand || RESTART_COMMAND
  const host = engineHostFrom(apiBase, typeof window === 'undefined' ? '' : window.location.href)
  const isLocalEngine = isLoopbackHost(host)
  // 'idle' | 'copied' | 'failed'. Distinguishing the last two matters: the
  // clipboard API is absent outside secure contexts, and claiming "Copied"
  // there would be the same class of lie this whole change is removing.
  const [copyState, setCopyState] = useState('idle')
  const copyResetRef = useRef(null)

  useEffect(() => () => { if (copyResetRef.current) window.clearTimeout(copyResetRef.current) }, [])

  const handleCopy = async () => {
    const copied = await copyText(restartCommand)
    setCopyState(copied ? 'copied' : 'failed')
    if (copyResetRef.current) window.clearTimeout(copyResetRef.current)
    copyResetRef.current = window.setTimeout(() => setCopyState('idle'), 2400)
  }

  const copyLabel = { copied: 'Copied', failed: 'Couldn’t copy' }[copyState]
    || (enginePath ? 'Copy restart command' : 'Copy terminal command')

  return (
    <div className="backend-banner" role="status" aria-live="polite">
      <StatusDot tone="offline" />
      <div className="backend-banner__copy">
        <strong>Camelid engine isn’t running</strong>
        <span>
          {canStart && 'Start it to load a model and chat — no setup needed.'}
          {!canStart && isLocalEngine && enginePath && (
            <>This page is served by the engine, so it can’t restart it. Start it again with <code>{restartCommand}</code> — or just run <code>camelid.exe</code> from where you unzipped Camelid.</>
          )}
          {!canStart && isLocalEngine && !enginePath && (
            <>This page is served by the engine, so it can’t restart it. Start <code>camelid.exe</code> again from the folder you unzipped Camelid into — or re-run the command you launched it with.</>
          )}
          {!canStart && !isLocalEngine && (
            <>The engine this UI points at — <code>{host}</code> — isn’t answering. Check that Camelid is running on that machine, or point this UI somewhere else in Settings.</>
          )}
        </span>
      </div>
      <div className="backend-banner__actions">
        {canStart && (
          <Button
            variant="primary"
            size="sm"
            icon={<IconPlay size={16} />}
            onClick={backend.start}
            loading={backend.starting}
            disabled={backend.starting || running}
          >
            {running ? 'Starting…' : 'Start Camelid'}
          </Button>
        )}
        {!canStart && isLocalEngine && (
          <Button
            variant="primary"
            size="sm"
            icon={copyState === 'copied' ? <IconCheck size={16} /> : <IconCopy size={16} />}
            onClick={handleCopy}
          >
            {copyLabel}
          </Button>
        )}
        <Button
          variant={!canStart && !isLocalEngine ? 'primary' : 'ghost'}
          size="sm"
          icon={<IconSettings size={16} />}
          onClick={onOpenSettings}
        >
          Settings
        </Button>
      </div>
    </div>
  )
}

export default BackendBanner
