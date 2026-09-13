import { useCallback, useState } from 'react'
import { Button } from '../components/ui/Button'
import { Modal } from '../components/ui/Modal'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { IconChevronDown, IconInfo, IconLink, IconPlus, IconRefresh, IconServer } from '../components/ui/icons'
import '../styles/mcp.css'

const EMPTY = { name: '', transport: 'stdio', command: '', args: '', url: '', env_vars: '', bearer_env: '' }

export default function ConnectionsView({ mcp }) {
  const [form, setForm] = useState(EMPTY)
  const [formError, setFormError] = useState('')
  const [showForm, setShowForm] = useState(false)
  const [removeId, setRemoveId] = useState(null)
  const [filter, setFilter] = useState('all')
  const [expanded, setExpanded] = useState(null)
  const closeForm = useCallback(() => { if (!mcp.busy) setShowForm(false) }, [mcp.busy])
  const openForm = () => { setForm(EMPTY); setFormError(''); setShowForm(true) }
  const field = name => ({ value: form[name], onChange: event => setForm(current => ({ ...current, [name]: event.target.value })) })
  const save = async event => {
    event.preventDefault()
    if (mcp.busy) return
    setFormError('')
    if (!form.name.trim()) { setFormError('Enter a connection name.'); return }
    let args = []
    try {
      args = form.transport === 'stdio' && form.args.trim() ? JSON.parse(form.args) : []
      if (!Array.isArray(args) || args.some(argument => typeof argument !== 'string')) throw new Error()
    } catch { setFormError('Arguments must be a JSON array of strings, for example ["server.js", "--read-only"].'); return }
    const result = await mcp.mutate('/connections', { method: 'POST', body: {
      ...form, name: form.name.trim(), args,
      env_vars: form.env_vars.split(',').map(name => name.trim()).filter(Boolean),
    } })
    if (result) { setForm(EMPTY); setShowForm(false); setFilter('all') }
  }
  const visible = mcp.connections.filter(connection => filter === 'all' || (filter === 'connected' ? connection.connected : !connection.connected))
  return <section className="cxv mcp-view">
    <header className="cxv-head"><div className="cxv-head__copy"><p className="cxv-kicker"><IconLink size={14} />Connected tools</p><h1>Your connections</h1><p className="cxv-sub">Connect once. Choose tools for each conversation.</p></div>
      <Button variant="primary" onClick={openForm} icon={<IconPlus size={16} />}>Add server</Button>
    </header>
    {mcp.error && !showForm && <p className="mcp-error mcp-connection-error" role="alert">{mcp.error} <Button variant="ghost" disabled={mcp.busy} onClick={() => mcp.refresh()}>Retry</Button></p>}
    {mcp.connections.length > 0 && <div className="mcp-filters" role="group" aria-label="Filter connections">
      {[['all', 'All connections'], ['connected', 'Connected'], ['disconnected', 'Disconnected']].map(([value, label]) => <button type="button" key={value} aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
    </div>}
    {!mcp.connections.length && !mcp.error && <div className="mcp-empty"><IconLink size={28} /><h2>Connect your first tool server</h2><p>Add a local command or a remote MCP endpoint. Then choose its tools in chat.</p><Button onClick={openForm}>Add server</Button></div>}
    {mcp.connections.length > 0 && <div className="mcp-connection-list">
      {visible.map(connection => {
        const { config, connected, tools } = connection
        const local = config.transport === 'stdio'
        const open = expanded === config.id
        return <article className="mcp-connection" key={config.id}>
          <div className="mcp-connection__row"><span className="mcp-connection__icon">{local ? <IconServer size={21} /> : <IconLink size={21} />}</span>
            <div className="mcp-connection__copy"><h2>{config.name}</h2><p>{local ? 'Local command' : 'Remote endpoint'} · {connected ? tools.length + ' tools discovered' : 'Connect to discover tools'}</p></div>
            <span className={'mcp-connection__status' + (connected ? ' is-connected' : '')}><i />{connected ? 'Connected' : 'Disconnected'}</span>
            <div className="mcp-connection__buttons">
              {!connected && <Button size="sm" disabled={mcp.busy} icon={<IconRefresh size={14} />} onClick={() => mcp.mutate('/connections/' + config.id + '/connect', { method: 'POST' })}>Connect</Button>}
              <Button size="sm" variant="ghost" aria-expanded={open} aria-label={'Details for ' + config.name} onClick={() => setExpanded(open ? null : config.id)}
                iconRight={<IconChevronDown size={14} className={open ? 'is-expanded' : ''} />}>Details</Button>
            </div>
          </div>
          {connection.error && <p role="alert" className="mcp-error mcp-connection-error">{connection.error}</p>}
          {open && <div className="mcp-connection__details">
            <code className="mcp-endpoint">{local ? config.command : config.url}</code>
            {local && <p className="mcp-muted">Connecting starts this program with your user account.</p>}
            {connected && <details className="mcp-available-tools"><summary>Available tools ({tools.length})</summary>
              <ul className="mcp-tool-list">{tools.map(tool => <li key={tool.key}><strong>{tool.name}</strong><p>{tool.description || 'No description supplied.'}</p></li>)}</ul>
            </details>}
            <div className="mcp-actions">
              {connected && <Button size="sm" variant="outline" disabled={mcp.busy} onClick={() => mcp.mutate('/connections/' + config.id + '/disconnect', { method: 'POST' })}>Disconnect</Button>}
              <Button size="sm" disabled={mcp.busy} variant="ghost" onClick={() => setRemoveId(config.id)}>Remove connection</Button>
            </div>
          </div>}
        </article>
      })}
      {!visible.length && <p className="mcp-filter-empty">No {filter} connections.</p>}
    </div>}
    <p className="mcp-connection-note"><IconInfo size={15} /><span>Saved servers start disconnected. Connecting a local server starts its program with your account. Every tool call needs your approval.</span></p>
    <Modal open={showForm} onClose={closeForm} title="Connect a tool server" labelledById="mcp-add-server-title" className="mcp-connection-modal">
      <form className="mcp-form" onSubmit={save}>
        <p className="mcp-muted">Give Camelid a new set of capabilities.</p>
        <fieldset disabled={mcp.busy}>
          <label>Connection name<input {...field('name')} required maxLength={80} placeholder="e.g. My project tools" /></label>
          <label>Connection type<select {...field('transport')}><option value="stdio">Local command (stdio)</option><option value="http">Remote endpoint (Streamable HTTP)</option></select></label>
          {form.transport === 'stdio' ? <>
            <label>Executable<input {...field('command')} required placeholder="node or /path/to/server" spellCheck={false} /></label>
            <label>Arguments<textarea {...field('args')} rows={2} placeholder={'["/path/to/server.js"]'} spellCheck={false} /></label>
          </> : <label>MCP endpoint URL<input {...field('url')} required type="url" placeholder="https://example.com/mcp" spellCheck={false} /></label>}
          <details className="mcp-form__advanced"><summary>Authentication &amp; environment</summary>
            {form.transport === 'stdio' ? <><label>Environment variable names<input {...field('env_vars')} placeholder="GITHUB_TOKEN, MY_SERVICE_KEY" spellCheck={false} /></label><p>Separate names with commas. Values come from the environment where Camelid was started.</p></>
              : <><label>Bearer token environment variable<input {...field('bearer_env')} placeholder="MY_MCP_TOKEN (optional)" spellCheck={false} /></label><p>The token stays in the engine. This connection uses bearer tokens; OAuth sign-in is not available here.</p></>}
            <p>Put credentials in environment variables, not URLs or command arguments.</p>
          </details>
        </fieldset>
        <p className="mcp-muted">Saving does not start the server. Choose Connect when you’re ready.</p>
        {(formError || mcp.error) && <p role="alert" className="mcp-error">{formError || mcp.error}</p>}
        <div className="mcp-form__footer"><Button onClick={closeForm} variant="ghost" disabled={mcp.busy}>Cancel</Button><Button type="submit" variant="primary" loading={mcp.busy}>Save server</Button></div>
      </form>
    </Modal>
    <ConfirmDialog open={Boolean(removeId)} title="Remove connection?" detail="This disconnects the server and removes its saved configuration." confirmLabel="Remove" busy={mcp.busy}
      onCancel={() => setRemoveId(null)} onConfirm={async () => {
        const result = await mcp.mutate('/connections/' + removeId, { method: 'DELETE' })
        if (result) setRemoveId(null)
      }} />
  </section>
}
