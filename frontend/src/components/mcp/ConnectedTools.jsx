import { useEffect, useRef } from 'react'
import { Button } from '../ui/Button'
import { MAX_MCP_TOOLS } from '../../lib/mcp.js'
import '../../styles/mcp.css'

export function ConnectedTools({ connections, selectedKeys, onToggle, onManage, disabled, capability }) {
  const pickerRef = useRef(null)
  useEffect(() => { if (disabled && pickerRef.current) pickerRef.current.open = false }, [disabled])
  const tools = connections.filter(c => c.connected).flatMap(c => c.tools.map(t => ({ ...t, connection: c.config.name })))
  return <details ref={pickerRef} className="mcp-picker"><summary>Connected tools{selectedKeys.length ? ` (${selectedKeys.length})` : ''}</summary>
    <div className="mcp-picker__body"><div className="mcp-actions"><span>Choose tools for this conversation. Every call requires your approval.</span><Button variant="ghost" onClick={onManage}>Manage connections</Button></div>
      {!capability.capable && <p>{capability.reason}</p>}
      {!tools.length && <p>No connected tools. Add a server in Connections, then choose Connect.</p>}
      <div className="mcp-tool-options">{tools.map(t => <label key={t.key}><input type="checkbox" checked={selectedKeys.includes(t.key)} disabled={disabled || !capability.capable || (!selectedKeys.includes(t.key) && selectedKeys.length >= MAX_MCP_TOOLS)} onChange={() => onToggle(t.key)} /><span><strong>{t.name}</strong><small>{t.connection}</small><span>{t.description}</span></span></label>)}</div>
      {selectedKeys.some(key => !tools.some(t => t.key === key)) && <p className="mcp-error">Some selected tools are disconnected. Reconnect their server or clear the selection.</p>}
      {!!selectedKeys.length && <Button variant="ghost" disabled={disabled} onClick={() => onToggle(null)}>Clear selection</Button>}
    </div>
  </details>
}
export function McpRunPanel({ activity, approval, onDecision, onStop }) {
  if (!activity || activity.phase === 'idle') return null
  return <section className="mcp-run" aria-label="Connected tool activity">
    <div className="mcp-actions"><strong role="status">{activity.phase === 'approval' ? 'Tool approval needed' : activity.phase === 'executing' ? `Running ${activity.tool}` : 'Working with connected tools'}</strong><Button variant="ghost" onClick={onStop}>Stop</Button></div>
    {approval && <div className="mcp-approval"><p><strong>{approval.connection_name}</strong> wants to run <code>{approval.tool}</code>.</p><pre>{JSON.stringify(approval.arguments, null, 2)}</pre><p>These arguments will be sent to the connected server.</p><div className="mcp-actions"><Button variant="primary" onClick={() => onDecision(true)}>Allow once</Button><Button onClick={() => onDecision(false)}>Deny</Button></div></div>}
  </section>
}
