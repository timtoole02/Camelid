import { useRuntimeMemory } from '../../hooks/useRuntimeMemory'
import { formatBytes } from '../../lib/formatters'
import { Button } from '../ui/Button'
import { Notice } from '../ui/Notice'
import { IconMemory, IconRefresh, IconTrash } from '../ui/icons'

function memoryValue(value, loading) {
  if (loading && value === undefined) return '…'
  return formatBytes(value)
}

export function RuntimeMemoryPanel({ apiBase }) {
  const { memory, loading, purging, error, notice, refresh, purge, clearNotice } = useRuntimeMemory(apiBase)
  const modelCount = memory?.models?.length || 0
  const cacheEntries = Number(memory?.kv_cache_entries || 0)

  return (
    <section className="cxv-card cxv-panel a-memory">
      <div className="a-memory__head">
        <div>
          <div className="cxv-section__head"><h2>Runtime memory</h2><span className="cxv-section__count">live · refreshes every 5s</span></div>
          <p className="cxv-sub">Current Camelid process memory, loaded-model weight estimates, and retained prompt KV cache.</p>
        </div>
        <div className="a-memory__actions">
          <Button
            variant="outline"
            size="sm"
            icon={<IconRefresh size={15} />}
            onClick={() => refresh()}
            loading={loading}
          >
            Refresh
          </Button>
          <Button
            variant="danger"
            size="sm"
            icon={<IconTrash size={16} />}
            onClick={purge}
            loading={purging}
            disabled={loading || purging || cacheEntries === 0}
            title={cacheEntries ? 'Release retained prompt KV-cache entries' : 'KV cache is empty'}
          >
            Purge KV cache
          </Button>
        </div>
      </div>

      {error ? <Notice notice={error} tone="error" /> : null}
      {notice ? <Notice notice={notice} tone="success" onDismiss={clearNotice} /> : null}

      <div className="cxv-stat-grid a-memory__stats">
        <div className="cxv-stat">
          <span>Camelid memory</span>
          <strong>{memoryValue(memory?.process_resident_bytes, loading)}</strong>
          <small>current process footprint</small>
        </div>
        <div className="cxv-stat">
          <span>Model weights</span>
          <strong>{memoryValue(memory?.model_weight_bytes_estimate, loading)}</strong>
          <small>{modelCount} loaded {modelCount === 1 ? 'model' : 'models'} · estimated</small>
        </div>
        <div className="cxv-stat">
          <span>KV cache</span>
          <strong>{memoryValue(memory?.kv_cache_bytes, loading)}</strong>
          <small>{cacheEntries} of {memory?.kv_cache_capacity || 0} retained entries</small>
        </div>
      </div>

      <div className="a-memory__models">
        <div className="a-memory__models-head">
          <strong>Loaded models</strong>
          <span>weights estimate · retained KV</span>
        </div>
        {memory?.models?.length ? memory.models.map((model) => (
          <article className="a-memory__model" key={model.id}>
            <div className="a-memory__model-name">
              <IconMemory size={16} />
              <strong title={model.id}>{model.id}</strong>
              {model.active ? <span className="cxv-tag cxv-tag--ready">Active</span> : null}
            </div>
            <div className="a-memory__model-metrics">
              <span><small>Weights</small><strong>{formatBytes(model.weight_bytes_estimate)}</strong></span>
              <span><small>KV cache</small><strong>{formatBytes(model.kv_cache_bytes)}</strong></span>
              <span><small>Cached tokens</small><strong>{model.cached_tokens?.toLocaleString() || '0'}</strong></span>
            </div>
          </article>
        )) : (
          <p className="a-memory__empty">{loading ? 'Reading loaded models…' : 'No models are loaded.'}</p>
        )}
      </div>
    </section>
  )
}

export default RuntimeMemoryPanel
