import { useState } from 'react'
import { formatDurationMs, formatRate } from '../../../lib/formatters'
import { IconChart } from '../../ui/icons'

/* Developer diagnostics panel — extracted verbatim from ChatWorkspace.
   Surfaces TTFT, decode rate, generation time, weight-load, and a per-layer
   attention-vs-FFN latency breakdown from the camelid generation diagnostics. */
export function DeveloperDiagnosticsBlock({ message }) {
  const [isOpen, setIsOpen] = useState(false)

  if (!message.camelid && !message.tokens_out_per_sec && !message.first_content_ms) return null

  const metrics = message.camelid?.timings_ms || {}
  const layers = metrics.layers || []
  const maxLayerTime = layers.reduce((max, layer) => Math.max(max, layer.total || 0), 0.0001)

  const ttft = message.first_content_ms !== null && message.first_content_ms !== undefined
    ? `${(Number(message.first_content_ms) / 1000).toFixed(2)}s`
    : null
  const outputRate = formatRate(message.tokens_out_per_sec)

  const weightLoadTime = metrics.weight_load ? formatDurationMs(metrics.weight_load) : null
  const totalGenTime = metrics.generate ? formatDurationMs(metrics.generate) : null

  return (
    <div className="developer-diagnostics-container">
      <button
        type="button"
        className={`developer-diagnostics-trigger ${isOpen ? 'is-open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="trigger-icon" aria-hidden="true"><IconChart size={14} /></span>
        <span>Developer Diagnostics</span>
        {outputRate && <span className="trigger-badge">{outputRate}</span>}
      </button>

      {isOpen && (
        <div className="developer-diagnostics-panel">
          <div className="diagnostics-grid-summary">
            {ttft && (
              <div className="summary-card">
                <span className="card-label">Time to First Token (TTFT)</span>
                <strong className="card-value">{ttft}</strong>
              </div>
            )}
            {outputRate && (
              <div className="summary-card">
                <span className="card-label">Decode Rate</span>
                <strong className="card-value">{outputRate}</strong>
              </div>
            )}
            {totalGenTime && (
              <div className="summary-card">
                <span className="card-label">Generation Time</span>
                <strong className="card-value">{totalGenTime}</strong>
              </div>
            )}
            {weightLoadTime && (
              <div className="summary-card">
                <span className="card-label" title="Time to map the model weights into memory (VM map)">Model load time</span>
                <strong className="card-value">{weightLoadTime}</strong>
              </div>
            )}
          </div>

          {layers.length > 0 && (
            <div className="layer-breakdown-section">
              <h4>Layer Latency Breakdown</h4>
              <p className="section-meta">Active transformer computation spent in Attention vs. Feed-Forward networks across {layers.length} layers.</p>

              <div className="layer-bars-container">
                {layers.map((layer) => {
                  const attnTime = (layer.attention_q || 0) + (layer.attention_k || 0) + (layer.attention_v || 0) + (layer.attention_context || 0) + (layer.attention_output || 0)
                  const ffnTime = (layer.ffn_gate || 0) + (layer.ffn_up || 0) + (layer.ffn_down || 0)
                  const otherTime = Math.max(0, (layer.total || 0) - (attnTime + ffnTime))

                  const attnPercent = layer.total > 0 ? (attnTime / layer.total) * 100 : 0
                  const ffnPercent = layer.total > 0 ? (ffnTime / layer.total) * 100 : 0
                  const otherPercent = layer.total > 0 ? (otherTime / layer.total) * 100 : 0

                  const totalPercent = Math.max(2, (layer.total / maxLayerTime) * 100)

                  return (
                    <div key={layer.layer_index} className="layer-bar-row">
                      <div className="layer-label">
                        <span>L{layer.layer_index}</span>
                        <small>{formatDurationMs(layer.total)}</small>
                      </div>
                      <div className="layer-bar-track">
                        <div className="layer-bar-fill" style={{ width: `${totalPercent}%` }}>
                          {attnPercent > 0 && (
                            <div className="segment-attn" style={{ width: `${attnPercent}%` }} title={`Attention: ${formatDurationMs(attnTime)}`} />
                          )}
                          {ffnPercent > 0 && (
                            <div className="segment-ffn" style={{ width: `${ffnPercent}%` }} title={`Feed-Forward: ${formatDurationMs(ffnTime)}`} />
                          )}
                          {otherPercent > 0 && (
                            <div className="segment-other" style={{ width: `${otherPercent}%` }} title={`Residual / Overhead: ${formatDurationMs(otherTime)}`} />
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              <div className="layer-legend">
                <span className="legend-item"><span className="legend-dot dot-attn" /> Attention</span>
                <span className="legend-item"><span className="legend-dot dot-ffn" /> Feed-Forward</span>
                <span className="legend-item"><span className="legend-dot dot-other" /> Residual / Norm</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
