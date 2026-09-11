/* ClusterConstellation — the local compute fabric as a small constellation
   in the lower-left sky. Node identity comes from the last fabric the Cluster
   page actually read; node activity comes only from real worker telemetry
   (worker_node_active/idle/error). The serving machine is always present.
   Nodes with no live signal render as quiet stars — never as "working".

   Until the Cluster page has successfully read a proxy there are no known
   nodes, and the constellation stays empty rather than drawing a fabric that
   was only ever sketched in browser storage. */

import { readCachedFabricNodes } from '../fabricClient.js'

const REFRESH_MS = 5000

export class ClusterConstellation {
  constructor() {
    this.nodes = []
    this.lastLoad = 0
    this.pulse = new Map() // key -> intensity
  }

  refresh(t) {
    if (t - this.lastLoad < REFRESH_MS && this.nodes.length) return
    this.lastLoad = t
    try {
      this.nodes = readCachedFabricNodes()
        .filter((node) => node.label)
        .slice(0, 12)
        .map((node) => ({
          id: node.label,
          display_name: node.label,
          hostname: node.host,
          ip_address: node.host,
          status: node.state === 'ready' ? 'online' : 'offline',
        }))
    } catch {
      this.nodes = []
    }
  }

  matchKey(workerNode) {
    // Worker telemetry identifies nodes by "host:port"; fabric nodes by host.
    const host = String(workerNode || '').split(':')[0]
    const hit = this.nodes.find((n) => n.ip_address === host || n.hostname === host)
    return hit ? hit.id : workerNode
  }

  onEvent(evt) {
    if (evt.event === 'worker_node_active') this.pulse.set(this.matchKey(evt.node), 1)
    if (evt.event === 'worker_node_error') this.pulse.set(this.matchKey(evt.node), -1)
    if (evt.event === 'worker_node_idle') {
      const key = this.matchKey(evt.node)
      if ((this.pulse.get(key) || 0) > 0.4) this.pulse.set(key, 0.4)
    }
  }

  draw(ctx, frame) {
    const { w, h, t, dt, run, workers, connection, showLabels } = frame
    this.refresh(t)

    // Stars: the local serving node first, then topology nodes.
    const stars = [{ id: '__local', label: 'this machine', live: run.active }]
    this.nodes.forEach((n) => stars.push({ id: n.id, label: n.display_name || n.hostname || n.id, status: n.status }))
    // Workers seen on telemetry but absent from the topology still appear.
    workers.forEach((state, node) => {
      const key = this.matchKey(node)
      if (!stars.some((s) => s.id === key)) stars.push({ id: key, label: node, status: 'unknown' })
    })
    if (stars.length === 1 && !run.active && workers.size === 0) {
      // Single quiet machine and nothing distributed: skip the constellation
      // rather than imply a cluster exists.
      return
    }

    const originX = w * 0.085
    const originY = h * 0.86
    ctx.save()
    stars.forEach((star, i) => {
      const angle = -0.2 - i * 0.42
      const dist = i === 0 ? 0 : 46 + (i % 3) * 30
      const x = originX + Math.cos(angle) * dist + i * 16
      const y = originY + Math.sin(angle) * dist * 0.5
      star.x = x
      star.y = y

      let intensity = this.pulse.get(star.id) || 0
      if (star.id === '__local' && run.active) intensity = Math.max(intensity, 0.85)
      const isError = intensity < 0
      const mag = Math.abs(intensity)

      // Link lines back to the local star.
      if (i > 0) {
        ctx.strokeStyle = `rgba(126, 231, 255, ${0.05 + mag * 0.25})`
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(stars[0].x, stars[0].y)
        ctx.lineTo(x, y)
        ctx.stroke()
      }

      const base = connection === 'live' ? 0.35 : 0.15
      const color = isError ? '255, 107, 107' : star.status === 'offline' ? '110, 120, 140' : '126, 231, 255'
      ctx.globalCompositeOperation = 'lighter'
      ctx.fillStyle = `rgba(${color}, ${base + mag * 0.6})`
      ctx.beginPath()
      ctx.arc(x, y, 2.2 + mag * 2.4, 0, Math.PI * 2)
      ctx.fill()
      ctx.globalCompositeOperation = 'source-over'

      if (showLabels || isError) {
        ctx.fillStyle = `rgba(196, 208, 224, ${isError ? 0.9 : 0.45})`
        ctx.font = '10px "SF Mono", "JetBrains Mono", Menlo, monospace'
        ctx.fillText(isError ? `${star.label} · error` : star.label, x + 7, y + 3)
      }

      if (intensity > 0) this.pulse.set(star.id, Math.max(0, intensity - dt * 0.0006))
      if (intensity < 0) this.pulse.set(star.id, Math.min(0, intensity + dt * 0.0002))
    })
    ctx.restore()
  }
}
