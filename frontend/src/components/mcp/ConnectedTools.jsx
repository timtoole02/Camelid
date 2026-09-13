import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Button } from '../ui/Button'
import { IconCheck, IconChevronDown, IconClose, IconExternal, IconLink, IconPlus, IconPlug, IconRefresh, IconSearch, IconServer, IconShield, IconTrash } from '../ui/icons'
import { MAX_MCP_TOOLS } from '../../lib/mcp.js'
import { readMcpToolSets, sameMcpSelection, toggleMcpGroup, upsertMcpToolSet, writeMcpToolSets } from '../../lib/mcpToolSets.js'
import { ToolsPopover } from './ToolsPopover'
import '../../styles/mcp.css'

const toolLabel = name => name.replace(/[_-]+/g, ' ').replace(/^./, letter => letter.toUpperCase())

function GroupSelection({ count, total, name, disabled, onChange }) {
  const ref = useCallback(element => { if (element) element.indeterminate = count > 0 && count < total }, [count, total])
  return <label className="mcp-group-selection"><span>{count}/{total}</span><input ref={ref} type="checkbox"
    checked={total > 0 && count === total} disabled={disabled || !total} onChange={onChange}
    aria-label={(count ? 'Clear tools from ' : 'Select all tools from ') + name} /></label>
}

export function ConnectedTools({
  connections = [], selectedKeys = [], onSelectionChange, onManage, onConnect, connectionBusy = false,
  error: connectionError = '', onRetry, disabled = false, capability, manualEnabled = false,
  onManualEnabledChange, manualText = '', onManualTextChange, manualReadiness, structuredMode = 'off',
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState(null)
  const [sets, setSets] = useState(readMcpToolSets)
  const [activeSet, setActiveSet] = useState('')
  const [savingSet, setSavingSet] = useState(false)
  const [setName, setSetName] = useState('')
  const [error, setError] = useState('')
  const triggerRef = useRef(null)
  useEffect(() => { if (disabled) setOpen(false) }, [disabled])
  const id = useId()
  const unavailable = selectedKeys.filter(key => !connections.some(connection => connection.connected && connection.tools.some(tool => tool.key === key)))
  const query = search.trim().toLocaleLowerCase()
  const selectedSet = sets.find(set => set.id === activeSet && sameMcpSelection(set.keys, selectedKeys))
  const selectionDisabled = disabled || connectionBusy || !capability?.capable
  const close = useCallback((restoreFocus = true) => {
    setOpen(false)
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true })
  }, [])
  const changeSelection = keys => {
    if (disabled || connectionBusy) return
    onSelectionChange?.(keys)
    onManualEnabledChange?.(false)
    setActiveSet('')
    setError('')
  }
  const saveSet = event => {
    event.preventDefault()
    try {
      const result = upsertMcpToolSet(sets, setName, selectedKeys)
      writeMcpToolSets(result.sets)
      setSets(result.sets); setActiveSet(result.saved.id); setSavingSet(false); setSetName(''); setError('')
    } catch (failure) { setError(failure.message) }
  }
  const removeSet = () => {
    try {
      const next = sets.filter(set => set.id !== selectedSet.id)
      writeMcpToolSets(next)
      setSets(next); setActiveSet(''); setError('')
    } catch (failure) { setError(failure.message) }
  }
  const toggleGroup = connection => {
    if (selectionDisabled) return
    try { changeSelection(toggleMcpGroup(selectedKeys, connection.tools.map(tool => tool.key))) }
    catch (failure) { setError(failure.message) }
  }
  const groups = connections.map(connection => ({
    ...connection,
    visibleTools: connection.tools.filter(tool => (connection.config.name + ' ' + tool.name + ' ' + (tool.description || '')).toLocaleLowerCase().includes(query)),
  })).filter(connection => !query || connection.visibleTools.length || connection.config.name.toLocaleLowerCase().includes(query))
  const expandedId = expanded ?? connections.find(connection => connection.connected)?.config.id
  // Busy changes unmount the picker immediately, including throughout MCP
  // continuation and approval. Returning to idle never reopens it.
  const visible = open && !disabled
  return <div className="mcp-composer-control">
    <button ref={triggerRef} type="button" className={'cxcomposer__tool mcp-trigger' + (selectedKeys.length || manualEnabled ? ' is-on' : '')}
      aria-label={'Tools' + (selectedKeys.length ? ' · ' + selectedKeys.length + ' selected' : manualEnabled ? ' · manual definitions on' : '')}
      aria-expanded={visible} aria-haspopup="dialog" aria-controls={visible ? id : undefined} disabled={disabled}
      onClick={() => {
        if (visible) close()
        else { setOpen(true); setSearch(''); setSavingSet(false); setError(''); setSets(readMcpToolSets()) }
      }}>
      <IconPlug size={16} /><span>Tools</span>
      {selectedKeys.length > 0 && <span className="mcp-trigger__count">{selectedKeys.length}</span>}
      {manualEnabled && !selectedKeys.length && <span className="mcp-trigger__count">Manual</span>}
      {unavailable.length > 0 && <span className="mcp-trigger__unavailable" aria-label="Some selected tools are unavailable" />}
      <IconChevronDown size={12} />
    </button>
    {visible && <ToolsPopover anchorRef={triggerRef} id={id} titleId={id + '-title'} onClose={close}>
      <header className="mcp-picker__head"><div><h2 id={id + '-title'}>Tools for this chat</h2><p aria-live="polite">{selectedKeys.length} selected · up to {MAX_MCP_TOOLS} tools</p></div>
        <button type="button" className="mcp-icon-button" onClick={() => close()} aria-label="Close tool picker"><IconClose size={17} /></button>
      </header>
      <div className="mcp-picker__body">
        <div className="mcp-picker__controls">
          <label className="mcp-search"><IconSearch size={16} /><input type="search" aria-label="Search tools" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search tools…" /></label>
          <div className="mcp-set-row"><label htmlFor={id + '-set'}>Tool set</label><select id={id + '-set'} aria-label="Tool set" value={selectedSet?.id || (selectedKeys.length ? '' : 'none')}
            disabled={selectionDisabled} onChange={event => {
              if (event.target.value === 'none') { changeSelection([]); return }
              const set = sets.find(item => item.id === event.target.value)
              if (set) { changeSelection([...set.keys]); setActiveSet(set.id) }
            }}>
            <option value="" disabled>Custom selection</option>
            <option value="none">No tools</option>
            {sets.map(set => <option value={set.id} key={set.id}>{set.name}</option>)}
          </select>
          <button type="button" className="mcp-icon-button" aria-label="Save selection as a tool set" disabled={!selectedKeys.length || selectionDisabled}
            onClick={() => { setSavingSet(value => !value); setSetName(selectedSet?.name || ''); setError('') }}><IconPlus size={16} /></button>
          {selectedSet && <button type="button" className="mcp-icon-button" aria-label={'Delete ' + selectedSet.name + ' tool set'} disabled={disabled} onClick={removeSet}><IconTrash size={16} /></button>}
          </div>
          {savingSet && <form className="mcp-save-set" onSubmit={saveSet}><label>Set name<input aria-label="Tool set name" maxLength={80} value={setName} onChange={event => setSetName(event.target.value)} placeholder="e.g. Code review" required /></label>
            <div className="mcp-actions"><Button type="submit" size="sm" disabled={selectionDisabled || !selectedKeys.length}>Save set</Button><Button size="sm" variant="ghost" onClick={() => setSavingSet(false)}>Cancel</Button></div><p className="mcp-muted">Saved on this device. A matching name replaces that set.</p></form>}
        </div>
        {!capability?.capable && <p className="mcp-picker__message" role="status">{capability?.reason || 'Choose a model that supports tools to enable selections.'}</p>}
        {structuredMode !== 'off' && <p className="mcp-picker__message">Turn off structured output before using tools.</p>}
        {connectionError && <div className="mcp-picker__message mcp-error" role="alert">{connectionError} <button type="button" className="mcp-text-button" disabled={connectionBusy} onClick={onRetry}>Retry</button></div>}
        {unavailable.length > 0 && <div className="mcp-picker__message" role="status">{unavailable.length} selected {unavailable.length === 1 ? 'tool is' : 'tools are'} unavailable. Reconnect the server or remove the unavailable selection before sending.
          <button type="button" className="mcp-text-button" disabled={disabled || connectionBusy} onClick={() => changeSelection(selectedKeys.filter(key => !unavailable.includes(key)))}>Remove unavailable</button>
        </div>}
        <div className="mcp-tool-groups">
          {groups.map(connection => {
            const { config, connected, tools, visibleTools } = connection
            const count = tools.filter(tool => selectedKeys.includes(tool.key)).length
            const isExpanded = Boolean(query) || expandedId === config.id
            return <section className="mcp-tool-group" key={config.id}>
              <div className="mcp-group-head">
                <button type="button" className="mcp-group-toggle" onClick={() => setExpanded(isExpanded ? '' : config.id)} disabled={!connected}
                  aria-expanded={connected ? isExpanded : undefined} aria-controls={connected && isExpanded ? id + '-' + config.id : undefined}>
                  {config.transport === 'stdio' ? <IconServer size={16} /> : <IconLink size={16} />}
                  <span>{config.name}</span>{connected && <IconChevronDown size={12} className={isExpanded ? 'is-expanded' : ''} />}
                </button>
                {connected ? <GroupSelection count={count} total={tools.length} name={config.name} disabled={selectionDisabled} onChange={() => toggleGroup(connection)} /> : <small>Disconnected</small>}
              </div>
              {connected && isExpanded && <div className="mcp-tool-options" id={id + '-' + config.id}>
                {visibleTools.map(tool => <label className="mcp-tool-option" key={tool.key}>
                  <span><strong title={tool.name}>{toolLabel(tool.name)}</strong><small>{tool.description || tool.name}</small></span>
                  <input type="checkbox" data-tool-key={tool.key} aria-label={'Use ' + tool.name} checked={selectedKeys.includes(tool.key)}
                    disabled={selectionDisabled || (!selectedKeys.includes(tool.key) && selectedKeys.length >= MAX_MCP_TOOLS)}
                    onChange={() => changeSelection(selectedKeys.includes(tool.key) ? selectedKeys.filter(key => key !== tool.key) : [...selectedKeys, tool.key])} />
                </label>)}
                {!tools.length && <p className="mcp-muted">No tools were discovered on this server.</p>}
              </div>}
              {!connected && <div className="mcp-disconnected"><span>{config.transport === 'stdio' ? 'Local command · starts a program' : 'Remote endpoint'}</span>
                <button type="button" className="mcp-text-button" disabled={connectionBusy || disabled} onClick={() => onConnect?.(config.id)}><IconRefresh size={14} />Connect</button>
              </div>}
              {connection.error && <p className="mcp-error">{connection.error}</p>}
            </section>
          })}
          {!groups.length && <div className="mcp-picker__empty"><IconLink size={24} /><strong>{connections.length ? 'No matching tools' : 'Connect your first server'}</strong>
            <p>{connections.length ? 'Try another tool or connection name.' : 'Add a local command or remote endpoint in Connections.'}</p>
            {!connections.length && <Button size="sm" onClick={() => { close(false); onManage() }}>Add connection</Button>}
          </div>}
        </div>
        <div className="mcp-picker__secondary">
          {!!selectedKeys.length && <button type="button" className="mcp-text-button" disabled={disabled || connectionBusy} onClick={() => changeSelection([])}>Clear selection</button>}
          {onManualEnabledChange && <details className="mcp-manual"><summary>Advanced</summary>
            <label className="mcp-manual__toggle"><input type="checkbox" checked={manualEnabled} disabled={selectionDisabled} onChange={event => {
              if (event.target.checked) changeSelection([])
              onManualEnabledChange(event.target.checked)
            }} />Manual tool definitions</label>
            <p className="mcp-muted">For testing custom definitions. Calls are returned to you and are not executed by Camelid.</p>
            {manualEnabled && <><textarea aria-label="Tool definitions" rows={5} value={manualText} onChange={event => onManualTextChange?.(event.target.value)} disabled={selectionDisabled} spellCheck={false} />
              <p className={'mcp-muted' + (!manualReadiness?.ready ? ' mcp-error' : '')}>{manualReadiness?.ready ? 'Definitions will be offered on the next message.' : manualReadiness?.reason}</p></>}
          </details>}
          {error && <p role="alert" className="mcp-error">{error}</p>}
        </div>
      </div>
      <footer className="mcp-picker__footer"><span><IconShield size={15} />Every call asks first</span><button type="button" className="mcp-text-button" onClick={() => { close(false); onManage() }}>Manage connections<IconExternal size={13} /></button></footer>
    </ToolsPopover>}
  </div>
}

export function McpRunPanel({ activity, approval, onDecision, onStop }) {
  if (!activity || activity.phase === 'idle') return null
  return <section className={'mcp-run' + (approval ? ' mcp-run--approval' : '')} aria-label="Connected tool activity">
    <header className="mcp-run__head"><span className="mcp-run__icon"><IconLink size={18} /></span><div><strong role="status">{approval ? 'Review tool request' : activity.phase === 'executing' ? 'Running ' + activity.tool : 'Working with connected tools'}</strong>
      <p>{approval ? approval.connection_name + ' · ' + approval.tool : activity.connection || 'Selections are fixed for this turn.'}</p></div><Button size="sm" variant="ghost" onClick={onStop}>Stop</Button></header>
    {approval && <div className="mcp-approval"><p>Review the arguments before they are sent to <strong>{approval.connection_name}</strong>.</p>
      <pre>{JSON.stringify(approval.arguments, null, 2)}</pre><div className="mcp-actions"><Button variant="primary" icon={<IconCheck size={16} />} onClick={() => onDecision(true)}>Allow once</Button><Button onClick={() => onDecision(false)}>Deny</Button></div>
    </div>}
  </section>
}
