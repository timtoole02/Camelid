import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'

const DEV_API_TARGET = process.env.VITE_CAMELID_PROXY_TARGET || 'http://127.0.0.1:8181'

function apiProxy() {
  return {
    target: DEV_API_TARGET,
    changeOrigin: true,
    configure(proxy) {
      proxy.on('proxyReq', (request) => request.setHeader('Origin', DEV_API_TARGET))
    },
  }
}

/* Probe common locations for a built `camelid` binary so Settings can suggest a
   working launch command (the binary often isn't on the dev server's PATH). */
function detectCamelidCommand(repoRoot) {
  const home = process.env.HOME || ''
  const candidates = [
    join(repoRoot, '..', 'cargo-targets', 'Camelid-push', 'release', 'camelid'),
    join(repoRoot, 'target', 'release', 'camelid'),
    join(repoRoot, 'target', 'debug', 'camelid'),
    home && join(home, '.cargo', 'bin', 'camelid'),
  ].filter(Boolean)
  for (const candidate of candidates) {
    try { if (existsSync(candidate)) return `${candidate} serve` } catch { /* noop */ }
  }
  return null
}

/**
 * Dev-only backend launcher.
 * Exposes localhost endpoints the Settings page uses to actually start/stop
 * `camelid serve` from the UI when running `npm run dev`:
 *   POST /__camelid/backend/launch  { command }   -> spawns the command (process group)
 *   POST /__camelid/backend/stop                   -> SIGTERMs the launched process group
 *   GET  /__camelid/backend/status                 -> { available, running, pid, logTail }
 * In a static build these routes don't exist; the Settings page detects that and
 * falls back to a copyable command. Only attached in `serve` (dev) mode.
 */
function camelidBackendLauncher() {
  let child = null
  let childExited = false
  const logs = []
  const pushLog = (line) => {
    logs.push(line)
    while (logs.length > 300) logs.shift()
  }
  // exitCode stays null when a process is killed by a signal (e.g. the OOM killer's
  // SIGKILL), so track an explicit flag set in the 'exit' handler instead.
  const running = () => Boolean(child && !childExited)
  const status = () => ({ available: true, running: running(), pid: running() ? child.pid : null, logTail: logs.slice(-60).join('') })
  const killChild = () => {
    if (!running()) return
    try { process.kill(-child.pid, 'SIGTERM') } catch { try { child.kill('SIGTERM') } catch {} }
  }
  const readBody = (req) => new Promise((res) => {
    let data = ''
    req.on('data', (c) => { data += c })
    req.on('end', () => res(data))
    req.on('error', () => res(''))
  })

  return {
    name: 'camelid-backend-launcher',
    apply: 'serve',
    configureServer(server) {
      const repoRoot = resolve(server.config.root, '..')
      const json = (res, code, obj) => {
        res.statusCode = code
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(obj))
      }
      server.httpServer?.on('close', killChild)
      process.on('exit', killChild)

      server.middlewares.use('/__camelid/backend/status', (req, res) => json(res, 200, { ...status(), detected: detectCamelidCommand(repoRoot) }))

      server.middlewares.use('/__camelid/backend/launch', async (req, res) => {
        if (req.method !== 'POST') return json(res, 405, { error: 'POST only' })
        if (running()) return json(res, 200, { ...status(), note: 'already running' })
        let command = ''
        try { command = String(JSON.parse((await readBody(req)) || '{}').command || '').trim() } catch { /* noop */ }
        // No explicit command → auto-detect the built binary so users never need to know the path.
        if (!command) command = detectCamelidCommand(repoRoot) || ''
        if (!command) return json(res, 400, { error: 'No camelid binary found. Build it (cargo build --release) or set a launch command in Settings.' })
        pushLog(`\n$ ${command}\n`)
        try {
          child = spawn(command, { cwd: repoRoot, shell: true, detached: true, env: process.env })
          childExited = false
        } catch (error) {
          child = null
          return json(res, 500, { error: String(error?.message || error) })
        }
        child.stdout?.on('data', (d) => pushLog(d.toString()))
        child.stderr?.on('data', (d) => pushLog(d.toString()))
        child.on('exit', (code, signal) => { childExited = true; pushLog(`\n[backend exited code=${code} signal=${signal}${signal === 'SIGKILL' ? ' — likely out of memory' : ''}]\n`) })
        child.on('error', (error) => pushLog(`\n[spawn error: ${error?.message || error}]\n`))
        return json(res, 200, status())
      })

      server.middlewares.use('/__camelid/backend/stop', (req, res) => {
        if (req.method !== 'POST') return json(res, 405, { error: 'POST only' })
        killChild()
        child = null
        return json(res, 200, { available: true, running: false, pid: null, logTail: logs.slice(-60).join('') })
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), camelidBackendLauncher()],
  server: {
    host: '127.0.0.1',
    port: 4175,
    proxy: {
      '/api': apiProxy(),
      '/v1': apiProxy(),
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4175,
  },
})
