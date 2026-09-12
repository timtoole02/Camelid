import { useState } from 'react'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { IconBolt, IconPlus } from '../components/ui/icons'
import '../styles/mcp.css'

const EMPTY = { name: '', transport: 'stdio', command: '', args: '', url: '', env_vars: '', bearer_env: '' }
export default function ConnectionsView({ mcp }) {
  const [form, setForm] = useState(EMPTY)
  const [formError, setFormError] = useState('')
  const [showForm, setShowForm] = useState(false)
  const [removeId, setRemoveId] = useState(null)
  const field = (name) => ({ value: form[name], onChange: e => setForm({ ...form, [name]: e.target.value }) })
  const save = async e => {
    e.preventDefault(); setFormError('')
    let args = []
    try { args = form.transport === 'stdio' && form.args.trim() ? JSON.parse(form.args) : []; if (!Array.isArray(args) || args.some(a => typeof a !== 'string')) throw new Error() }
    catch { setFormError('Arguments must be a JSON array of strings, for example ["server.js", "--read-only"].'); return }
    const result = await mcp.mutate('/connections', { method: 'POST', body: { ...form, args, env_vars: form.env_vars.split(',').map(s => s.trim()).filter(Boolean) } })
    if (result) { setForm(EMPTY); setShowForm(false) }
  }
  return <section className="cxv mcp-view">
    <header className="cxv-head"><div className="cxv-head__copy"><p className="cxv-kicker"><IconBolt size={14} /> Connected tools</p><h1>Connections</h1><p className="cxv-sub">Give local models access to tools through Model Context Protocol.</p></div><Button onClick={() => setShowForm(!showForm)} icon={<IconPlus size={16} />}>Add connection</Button></header>
    <div className="mcp-intro"><strong>Your tools, under your control</strong><p>Connect a server, choose its tools in chat, and review each requested action before it runs. Local servers run with your user account; connecting starts the configured program.</p></div>
    {mcp.error && <p className="mcp-error" role="alert">{mcp.error} <Button variant="ghost" onClick={() => mcp.refresh()}>Retry</Button></p>}
    {showForm && <form className="mcp-form cxv-card" onSubmit={save}>
      <h2>Add MCP server</h2>
      <label>Name<input {...field('name')} required maxLength={80} placeholder="My tools" /></label>
      <label>Connection type<select {...field('transport')}><option value="stdio">Local command (stdio)</option><option value="http">Remote endpoint (Streamable HTTP)</option></select></label>
      {form.transport === 'stdio' ? <>
        <label>Executable<input {...field('command')} required placeholder="/path/to/server or an executable on PATH" /></label>
        <label>Arguments<textarea {...field('args')} rows={3} placeholder={'["/path/to/server.js"]'} spellCheck={false} /></label>
        <label>Environment variables to pass<input {...field('env_vars')} placeholder="GITHUB_TOKEN, MY_SERVICE_KEY" /></label>
        <p>Enter variable names separated by commas. Values come from the environment where Camelid was started. Put credentials in environment variables, rather than command arguments.</p>
      </> : <>
        <label>MCP endpoint URL<input {...field('url')} required type="url" placeholder="https://example.com/mcp" /></label>
        <label>Bearer token environment variable<input {...field('bearer_env')} placeholder="MY_MCP_TOKEN (optional)" /></label>
        <p>The token stays in the engine. This version supports bearer tokens; services that require an OAuth sign-in flow need that configured separately.</p>
      </>}
      {formError && <p role="alert" className="mcp-error">{formError}</p>}
      <div className="mcp-actions"><Button type="submit" variant="primary" loading={mcp.busy}>Save connection</Button><Button onClick={() => setShowForm(false)}>Cancel</Button></div>
    </form>}
    {!mcp.connections.length && !showForm && !mcp.error && <div className="mcp-empty"><IconBolt size={28} /><h2>Connect your first tool server</h2><p>Add a local MCP command or an HTTPS endpoint. Saved servers stay disconnected until you choose Connect.</p><Button onClick={() => setShowForm(true)}>Add connection</Button></div>}
    <div className="mcp-grid">{mcp.connections.map(c => <article className="cxv-card mcp-connection" key={c.config.id}>
      <div className="mcp-connection__head"><h2>{c.config.name}</h2><span className={c.connected ? 'mcp-connected' : ''}>{c.connected ? 'Connected' : 'Disconnected'}</span></div>
      <code className="mcp-endpoint">{c.config.transport === 'stdio' ? c.config.command : c.config.url}</code>
      <p>{c.config.transport === 'stdio' ? 'Local command' : 'Streamable HTTP'} · {c.connected ? `${c.tools.length} ${c.tools.length === 1 ? 'tool' : 'tools'} available` : 'Connect to discover tools'}</p>
      {c.error && <p role="alert" className="mcp-error">{c.error}</p>}
      {c.connected && <details><summary>Available tools</summary><ul className="mcp-tool-list">{c.tools.map(t => <li key={t.key}><strong>{t.name}</strong><p>{t.description || 'No description supplied.'}</p></li>)}</ul></details>}
      <div className="mcp-actions"><Button variant={c.connected ? 'outline' : 'primary'} loading={mcp.busy} onClick={() => mcp.mutate(`/connections/${c.config.id}/${c.connected ? 'disconnect' : 'connect'}`, { method: 'POST' })}>{c.connected ? 'Disconnect' : 'Connect'}</Button><Button disabled={mcp.busy} variant="ghost" onClick={() => setRemoveId(c.config.id)}>Remove</Button></div>
    </article>)}</div>
    <ConfirmDialog open={Boolean(removeId)} title="Remove connection?" detail="This disconnects the server and removes its saved configuration." confirmLabel="Remove" onCancel={() => setRemoveId(null)} onConfirm={async () => { await mcp.mutate(`/connections/${removeId}`, { method: 'DELETE' }); setRemoveId(null) }} />
  </section>
}
