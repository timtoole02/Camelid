import { useEffect, useMemo, useRef, useState } from 'react'
import { CodingProjectPanel } from '../components/coding/CodingProjectPanel.jsx'
import { useCodingSession } from '../hooks/useCodingSession.js'
import { codingActive, codingContext, createCodingFolder } from '../lib/codingSessions.js'
import { readModelToolCapability } from '../lib/toolCalling.js'
import { changeRequest } from '../lib/changeReviews.js'
import { appStorage } from '../lib/appStorage.js'
import { AssistantMarkdown } from '../lib/markdown.jsx'
import { MessageTurn } from '../components/chat/MessageTurn.jsx'
import { describeCodingMessage, describeCodingAction, describeCodingEvent } from '../lib/codingPresentation.js'
import { FolderPicker } from './WorkspaceView.jsx'
import { ConversationContext } from '../components/context/ContextEditors.jsx'
import { Button } from '../components/ui/Button.jsx'
import { ConfirmDialog } from '../components/ui/ConfirmDialog.jsx'
import { CamelidMark } from '../components/ui/CamelidMark.jsx'
import { IconApi, IconCheck, IconChevronRight, IconCpu, IconFile, IconFolder, IconHistory, IconPlus, IconReceipt, IconSearch, IconSend, IconShield, IconSidebar, IconStop, IconTrash } from '../components/ui/icons.jsx'

const labels = { running: 'Working', queued: 'Queued', working: 'Working', waiting_approval: 'Needs approval', waiting_helpers: 'Waiting for helpers', paused: 'Paused', stopping: 'Stopping', completed: 'Turn finished', done: 'Done', failed: 'Failed', cancelled: 'Stopped', interrupted: 'Interrupted' }
const label = value => labels[value] || value || 'Ready'
const agentName = agent => agent.id === 'lead' ? 'Lead' : agent.id.replaceAll('-', ' ').replace(/^./, c => c.toUpperCase())
const relativeName = path => path?.split(/[\\/]/).at(-1) || path
const ignore = promise => promise?.catch(() => {})
const FileDiff = ({ diff }) => <pre className="coding-diff">{String(diff || '').split('\n').map((line, index) => <span key={index} className={line.startsWith('+') ? 'is-added' : line.startsWith('-') ? 'is-removed' : ''}>{line}{'\n'}</span>)}</pre>

export default function CodingWorkspace({ apiBase, runtime, selectedModel, capabilities, projects, chatContext, contextSources: chatContextSources, updateChatContext, globalPrompt, setTab, onActivity, active }) {
  /* Memory is for chat: the user approved those facts for conversations, not
     as instructions for a coding session. */
  const contextSources = useMemo(() => chatContextSources.filter((source) => source.id !== 'memory'), [chatContextSources])
  const coding = useCodingSession(apiBase, runtime?.active_model_id, runtime?.loaded_now)
  const { snapshot, busy, connection, selectedId } = coding
  const [workspace, setWorkspace] = useState(() => appStorage.getItem('camelid.codingWorkspace') || '')
  const [goal, setGoal] = useState('')
  const [allowCommands, setAllowCommands] = useState(false)
  const [inputMode, setInputMode] = useState('steer')
  const [optionsOpen, setOptionsOpen] = useState(false)
  const optionsRef = useRef(null)
  const optionsButton = useRef(null)
  const [projectSettings, setProjectSettings] = useState({ verification_command: '', preview_entry: 'index.html', workflow: '', max_run_seconds: 1800, browser_check: false })
  const [folderOpen, setFolderOpen] = useState(false)
  const [sideOpen, setSideOpen] = useState(true)
  const [panel, setPanel] = useState('agents')
  const [agentId, setAgentId] = useState('lead')
  const [reviewId, setReviewId] = useState(null)
  const [commandOpen, setCommandOpen] = useState(false)
  const [snippetsOpen, setSnippetsOpen] = useState(false)
  const [review, setReview] = useState(null)
  const [reviewError, setReviewError] = useState('')
  const [reviewBusy, setReviewBusy] = useState(false)
  const [reviewOverrides, setReviewOverrides] = useState({})
  const [confirmRemove, setConfirmRemove] = useState(false)
  const sidebar = useRef(null)
  const bottom = useRef(null)
  const userAway = useRef(false)
  const running = codingActive(snapshot?.phase)
  const capability = readModelToolCapability(capabilities, selectedModel, runtime)
  const ready = capability.capable && runtime?.loaded_now && runtime?.generation_ready && coding.toolCapableModel === runtime.active_model_id
  const agents = Object.values(snapshot?.agents || {}).sort((a, b) => a.id === 'lead' ? -1 : b.id === 'lead' ? 1 : a.id.localeCompare(b.id))
  const selectedAgent = agents.find(a => a.id === agentId) || agents[0]
  const reviews = (snapshot?.reviews || []).map(r => reviewOverrides[r.id] || r)
  const pendingReview = snapshot?.approval?.detail?.review
  const approvalIsSelected = snapshot?.approval && (!reviewId || pendingReview?.id === reviewId)
  const working = agents.filter(a => ['working', 'queued', 'waiting_approval'].includes(a.status))
  const canSend = Boolean(goal.trim() && !busy && (running ? connection === 'connected' && snapshot?.phase !== 'stopping' : ready) && (selectedId ? snapshot : workspace.trim()))
  const project = projects.find(p => p.id === (snapshot?.config.project_id || chatContext?.project_id))
  const selectedEvents = useMemo(() => (snapshot?.events || []).filter(e => e.agent_id === selectedAgent?.id && e.kind !== 'model.timing').slice(-20), [snapshot?.events, selectedAgent?.id])
  useEffect(() => { onActivity?.(snapshot ? { id: snapshot.id, title: snapshot.title, phase: snapshot.phase } : null) }, [snapshot?.id, snapshot?.title, snapshot?.phase, onActivity])
  useEffect(() => { setReviewId(null); setCommandOpen(false); setSnippetsOpen(false); setAgentId('lead'); setReviewOverrides({}); setReview(null) }, [selectedId])
  useEffect(() => {
    if (!active || running || !snapshot?.reviews?.length) return
    const controller = new AbortController()
    const ids = new Set(snapshot.reviews.map(review => review.id))
    changeRequest(apiBase, '', { signal: controller.signal }).then(data => {
      setReviewOverrides(current => ({ ...current, ...Object.fromEntries((data.reviews || []).filter(review => ids.has(review.id)).map(review => [review.id, review])) }))
    }).catch(error => { if (error.name !== 'AbortError') setReviewError(error.message) })
    return () => controller.abort()
  }, [apiBase, selectedId, running, active, snapshot?.reviews?.length])
  useEffect(() => {
    if (active && !userAway.current) bottom.current?.scrollIntoView({ block: 'end' })
  }, [snapshot?.turns?.length, snapshot?.agents?.lead?.output, snapshot?.approval?.id, active])
  useEffect(() => {
    setReviewError(''); setReview(null)
    if (!reviewId) return
    const controller = new AbortController()
    changeRequest(apiBase, '/' + encodeURIComponent(reviewId), { signal: controller.signal }).then(value => { if (controller.signal.aborted) return; setReview(value); setReviewOverrides(current => ({ ...current, [value.id]: value })) }).catch(e => { if (e.name !== 'AbortError') setReviewError(e.message) })
    return () => controller.abort()
  }, [apiBase, reviewId, snapshot?.approval?.id, snapshot?.phase])
  useEffect(() => {
    if (sideOpen && (reviewId || commandOpen || snippetsOpen) && window.matchMedia('(max-width: 700px)').matches) sidebar.current?.scrollIntoView({ block: 'start' })
  }, [sideOpen, reviewId, commandOpen, snippetsOpen])
  useEffect(() => { setOptionsOpen(false); setInputMode('steer') }, [selectedId])
  useEffect(() => {
    if (!optionsOpen) return
    const outside = event => { if (!optionsRef.current?.contains(event.target)) setOptionsOpen(false) }
    const escape = event => {
      if (event.key === 'Escape') { setOptionsOpen(false); optionsButton.current?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape) }
  }, [optionsOpen])
  const submit = async () => {
    if (!canSend) return
    const options = { workspace: workspace.trim(), model_id: runtime.active_model_id, project_id: chatContext?.project_id || '', ...codingContext(contextSources), allow_commands: allowCommands, project: { ...projectSettings, engine_id: coding.executionEngine?.id || '' }, ...(selectedId ? { mode: running ? inputMode : 'follow_up' } : {}) }
    try { await coding.send(goal.trim(), options); setGoal(''); setInputMode('steer'); setOptionsOpen(false); userAway.current = false; appStorage.setItem('camelid.codingWorkspace', workspace.trim()) } catch { /* hook presents the error */ }
  }
  const decide = async approved => { try { await coding.decide(approved); setReviewId(null); setCommandOpen(false) } catch { /* hook presents the error */ } }
  const undo = async () => {
    setReviewBusy(true); setReviewError('')
    try { const next = await changeRequest(apiBase, '/' + encodeURIComponent(reviewId) + '/undo', { method: 'POST' }); setReview(next); setReviewOverrides(current => ({ ...current, [next.id]: next })) }
    catch (e) { setReviewError(e.message) }
    finally { setReviewBusy(false) }
  }
  const decisionButtons = <div className="coding-actions"><Button variant="primary" size="sm" disabled={busy || Boolean(coding.decidingId) || connection !== 'connected' || snapshot?.phase === 'paused'} onClick={() => decide(true)}>{pendingReview ? 'Approve & apply' : 'Allow command once'}</Button><Button variant="outline" size="sm" disabled={busy || Boolean(coding.decidingId) || connection !== 'connected' || snapshot?.phase === 'paused'} onClick={() => decide(false)}>Deny</Button></div>
  const openReview = id => { setReview(null); setReviewError(''); setReviewId(id); setCommandOpen(false); setSnippetsOpen(false); setSideOpen(true); setPanel('changes') }
  const closeDetail = () => { setReview(null); setReviewId(null); setCommandOpen(false); setSnippetsOpen(false) }
  const openApproval = () => {
    if (pendingReview) openReview(pendingReview.id)
    else { closeDetail(); setCommandOpen(true); setSideOpen(true) }
  }
  const sourceSnippets = (snapshot?.turns || []).flatMap((turn, index) => ['user', 'assistant'].flatMap(role => describeCodingMessage(turn[role]).snippets.map((snippet, i) => ({ ...snippet, key: `${index}-${role}-${i}`, title: `${role === 'user' ? 'Your message' : 'Camelid'} · turn ${index + 1}` }))))
  const renderReview = () => <section className="coding-review" aria-label="Selected coding file review">
    {reviewError && <p role="alert">{reviewError}</p>}
    {!review && !reviewError && <p role="status">Loading file review…</p>}
    {review && <>
      <div className="coding-row"><IconFile size={17} /><h2>{relativeName(review.path)}</h2><span className="coding-status">{review.status}</span></div>
      <p className="coding-filepath">{review.path}</p><FileDiff diff={review.diff} />
      <details><summary>Complete before and after</summary><h3>Before</h3><pre>{review.before ?? '(New file)'}</pre><h3>After</h3><pre>{review.after}</pre></details>
      {approvalIsSelected && pendingReview?.id === review.id && <div className="coding-approval"><p>Applies this exact file version. Its original is saved for Undo.</p>{decisionButtons}</div>}
      {review.status === 'applied' && <div className="coding-approval"><p>Undo restores the original if this file has not changed since application.</p><Button variant="outline" size="sm" disabled={reviewBusy || running} onClick={undo}>Undo change</Button>{running && <p className="coding-muted">Wait for the run to finish before undoing its changes.</p>}</div>}
      {review.status === 'pending' && !approvalIsSelected && <p className="coding-muted">This saved review is not a pending action in this run. Open Changes to review it separately.</p>}
    </>}
  </section>
  return <section className={`coding-workspace ${sideOpen ? '' : 'coding-workspace--wide'}`} aria-label="Agentic coding workspace">
    <header className="coding-toolbar">
      <div className="coding-title"><IconApi size={18} /><strong>{snapshot?.title || 'New coding session'}</strong>{snapshot && <span className={'coding-status is-' + snapshot.phase}>{label(snapshot.phase)}</span>}</div>
      <div className="coding-actions">
        <label className="coding-session-picker"><span className="sr-only">Saved coding sessions</span><IconHistory size={15} /><select aria-label="Saved coding sessions" value={selectedId} disabled={busy || running} onChange={e => coding.select(e.target.value)}><option value="">New session</option>{coding.sessions.map(s => <option value={s.id} key={s.id}>{s.title} · {label(s.phase)}</option>)}</select></label>
        <Button size="sm" variant="ghost" icon={<IconPlus size={16} />} aria-label="New coding session" disabled={busy || running} onClick={() => { coding.select(''); setGoal('') }} />
        <Button size="sm" variant="ghost" icon={<IconSidebar size={16} />} aria-label="Toggle agent sidebar" aria-expanded={sideOpen} aria-controls="coding-team" onClick={() => setSideOpen(v => !v)} />
      </div>
    </header>
    {(coding.error || snapshot?.error || ['reconnecting','disconnected'].includes(connection)) && <div className="coding-alert" role="status"><span>{coding.error || snapshot?.error || 'Connection interrupted. Reconnecting to the server-owned run; no actions will be replayed.'}</span><Button variant="ghost" size="sm" onClick={() => ignore(coding.retry())} disabled={busy}>Refresh</Button></div>}
    <div className="coding-body">
      <div className="coding-chat-pane">
        <div className="coding-conversation" onScroll={event => { const e = event.currentTarget; userAway.current = e.scrollHeight - e.scrollTop - e.clientHeight > 100 }}>
          <div className="cxchat__column"><div className="cxchat__thread">
            {!snapshot && <div className="coding-empty"><CamelidMark size={38} /><h2>What are we building?</h2><p>Choose a project and describe the work. Follow your agents, review changes, and keep the conversation going.</p></div>}
            {snapshot?.turns.map((turn, index) => {
              const user = describeCodingMessage(turn.user)
              const answer = describeCodingMessage(turn.assistant)
              const isLatest = index === snapshot.turns.length - 1
              const message = answer.description || (answer.snippets.length ? 'Code is ready to inspect in the sidebar.' : isLatest && running ? describeCodingAction(snapshot.agents.lead?.action, snapshot.agents.lead?.status) + '…' : label(turn.outcome))
              return <div className={'coding-turn' + (isLatest && running && !turn.assistant ? ' is-working' : '')} key={index}>
                <MessageTurn message={{ id: `user-${index}`, role: 'user', content: user.description || 'Code shared in the sidebar.' }} generationElapsedSeconds={0} />
                <MessageTurn message={{ id: `assistant-${index}`, role: 'assistant', content: message }} generationElapsedSeconds={0} />
                {(user.snippets.length > 0 || answer.snippets.length > 0) && <button className="coding-source-link" type="button" onClick={() => { closeDetail(); setSnippetsOpen(true); setSideOpen(true) }}><IconApi size={14} />View code in sidebar<IconChevronRight size={14} /></button>}
              </div>
            })}
            {reviews.length > 0 && <section className="coding-proposals" aria-label="Proposed changes"><button type="button" onClick={() => { closeDetail(); setPanel('changes'); setSideOpen(true) }}><IconReceipt size={17} /><span>{reviews.length} file {reviews.length === 1 ? 'change' : 'changes'}<small>{reviews.filter(r => r.status === 'applied').length} applied · review details in the sidebar</small></span><IconChevronRight size={14} /></button></section>}
            {snapshot?.approval && <section className="coding-approval coding-approval-summary" aria-label="Pending agent approval"><div className="coding-row"><IconShield size={18} /><strong>Your review is needed</strong></div><p>{pendingReview ? `Lead wants to ${pendingReview.before == null ? 'create' : 'update'} ${relativeName(pendingReview.path)}.` : 'Lead wants to run a command in your project.'}</p><Button variant="primary" size="sm" onClick={openApproval}>{pendingReview ? 'Review file change' : 'Review command'}<IconChevronRight size={14} /></Button></section>}
            {running && <div className="coding-runline" role="status"><span className="coding-dot" /><span>{snapshot.phase === 'paused' ? 'Paused before the next action. The current request may finish.' : snapshot.phase === 'stopping' ? 'Stopping…' : describeCodingAction(snapshot.agents.lead?.action, snapshot.agents.lead?.status)}</span><div className="coding-actions"><Button size="sm" variant="ghost" disabled={busy || snapshot.phase === 'stopping'} onClick={() => ignore(coding.control(snapshot.phase === 'paused' ? 'resume' : 'pause'))}>{snapshot.phase === 'paused' ? 'Resume' : 'Pause'}</Button><Button size="sm" variant="ghost" icon={<IconStop size={14} />} disabled={busy || snapshot.phase === 'stopping'} onClick={() => ignore(coding.control('stop'))}>Stop</Button></div></div>}
            {snapshot && !running && <div className="coding-runline"><span>{label(snapshot.phase)} · conversation saved</span><Button size="sm" variant="ghost" icon={<IconTrash size={14} />} disabled={busy} onClick={() => setConfirmRemove(true)}>Remove session</Button></div>}
          {(snapshot?.incoming || []).filter(m => m.mode === 'steer' || m.status === 'accepted').slice(-6).map(m => <p className="coding-incoming" key={m.id}>{m.mode === 'steer' ? (m.status === 'consumed' ? 'Message received' : 'Message sent') : 'Queued'}: {describeCodingMessage(m.text).description}{!running && m.mode === 'queue' && <Button size="sm" variant="ghost" onClick={() => ignore(coding.send(m.text, { message_id: m.id }))}>Continue queued task</Button>}</p>)}
            <div ref={bottom} />
          </div></div>
        </div>
        <div className="coding-composer-area cxchat__column">
          {!snapshot && <div className="coding-setup">
            <div className="coding-folder-field"><label htmlFor="coding-folder">Project folder</label><div className="coding-row"><input id="coding-folder" value={workspace} onChange={e => setWorkspace(e.target.value)} placeholder="Choose a local folder" disabled={busy} /><Button size="sm" variant="outline" icon={<IconFolder size={15} />} disabled={busy} onClick={() => setFolderOpen(true)}>Browse</Button></div></div>

          </div>}
          {!snapshot && <ConversationContext compact={false} context={chatContext} projects={projects} sources={contextSources} globalPrompt={globalPrompt || ''} onSave={updateChatContext} onManageProjects={() => setTab('projects')} busy={busy} />}
          {!ready && <p className="coding-readiness" role="status">{capability.reason || 'Code needs an exact tool-capable model artifact with a certified digest.'} <button type="button" onClick={() => setTab('library')}>Open Models</button></p>}
          <form className="coding-composer cxcomposer" onSubmit={e => { e.preventDefault(); submit() }}>
            <div className="cxcomposer__box">
            {snapshot && <div className="coding-composer-project"><IconFolder size={14} /><span>{project?.name || relativeName(snapshot.config.workspace)}</span></div>}
            <textarea className="cxcomposer__input" aria-label="Message coding agents" value={goal} maxLength={16000} onChange={e => setGoal(e.target.value)} placeholder={running && inputMode === 'queue' ? 'What should Camelid do next?' : 'Message Camelid…'} disabled={busy || snapshot?.phase === 'stopping'} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit() } }} rows={2} />
            <div className="coding-composer-tools cxcomposer__toolbar">
              <span className="coding-composer-model" title={snapshot?.config.model_id || selectedModel?.name || 'No model'}><span className="coding-dot" />{snapshot?.config.model_id || selectedModel?.name || 'No model'}</span>
              <div className="coding-composer-options" ref={optionsRef}>
                <button ref={optionsButton} type="button" className="coding-options-trigger" aria-label="Coding options" aria-expanded={optionsOpen} aria-controls="coding-options" onClick={() => setOptionsOpen(v => !v)}><IconShield size={14} /><span>{snapshot?.auto_approve_files ? 'Auto-approve edits' : 'Review edits'}</span></button>
                <div id="coding-options" className="coding-options-popover" hidden={!optionsOpen}>
                  <strong>Project permissions</strong>
                  {snapshot ? <>
                    <p className="coding-options-path"><IconFolder size={14} />{snapshot.config.workspace}</p>
                    <label className="coding-command-choice coding-file-approval-choice"><input type="checkbox" checked={Boolean(snapshot.auto_approve_files)} onChange={e => ignore(coding.setAutoApproveFiles(e.target.checked))} disabled={busy || connection !== 'connected' || snapshot.phase === 'stopping'} /><span>Auto-approve file changes<small>Apply pending and future edits in this session. You can review and undo them.</small></span></label>
                    <p>Commands still ask for approval. Auto-approval turns off when the engine restarts.</p>
                  </> : <label className="coding-command-choice"><input type="checkbox" checked={allowCommands} onChange={e => setAllowCommands(e.target.checked)} disabled={busy} /><span>Allow command requests<small>Every command asks first and runs with your account permissions. Command changes are not covered by Undo.</small></span></label>}
                  <p>Agents can read, search, and edit project files.</p>
                </div>
              </div>
              <div className="coding-send-controls">
                {running && <select className="coding-send-mode" aria-label="Coding message mode" value={inputMode} onChange={e => setInputMode(e.target.value)}><option value="steer">Send now</option><option value="queue">Send after task</option></select>}
                <Button type="submit" size="sm" variant="primary" icon={<IconSend size={17} />} aria-label={running ? (inputMode === 'steer' ? 'Add to current task' : 'Queue follow-up') : snapshot ? 'Send coding follow-up' : 'Start coding'} disabled={!canSend} />
              </div>
            </div>
            </div>
          </form>
        </div>
      </div>
      <aside ref={sidebar} id="coding-team" className="coding-team" hidden={!sideOpen} aria-label="Agent assignments">
        <div className="coding-machine"><IconCpu size={14} /><span>Runs on {snapshot?.config.project?.engine_name || coding.executionEngine?.name || 'connected engine'}</span><button type="button" aria-pressed={panel === 'project'} onClick={() => setPanel('project')}>Project</button></div>
        <div className="coding-team-heading"><span className="coding-team-emblem"><IconApi size={19} /></span><div><strong>Project activity</strong><span>{snapshot ? relativeName(snapshot.config.workspace) : 'Your coding workspace'}</span></div><span className="coding-count">{working.length} active</span></div>
        {(reviewId || commandOpen || snippetsOpen) ? <div className="coding-side-detail">
          <Button size="sm" variant="ghost" onClick={closeDetail}>← Back to activity</Button>
          {reviewId && renderReview()}
          {commandOpen && snapshot?.approval && !pendingReview && <section className="coding-approval" aria-label="Command review"><div className="coding-row"><IconShield size={17} /><strong>Review command</strong></div><pre>{snapshot.approval.detail.command}</pre><p className="coding-filepath">{snapshot.approval.detail.workspace}</p><p>{snapshot.approval.detail.execution} Timeout: {snapshot.approval.detail.timeout_seconds} seconds.</p>{decisionButtons}</section>}
          {snippetsOpen && <section className="coding-snippets"><h3>Code from the conversation</h3><p className="coding-muted">These are message excerpts. Applied changes are recorded under Changes.</p>{sourceSnippets.map(snippet => <details key={snippet.key}><summary>{snippet.title}{snippet.language ? ` · ${snippet.language}` : ''}</summary><pre>{snippet.content}</pre></details>)}</section>}
        </div> : <>
          <div className="coding-tabs" role="group" aria-label="Coding sidebar"><button type="button" aria-pressed={panel === 'agents'} onClick={() => setPanel('agents')}>Overview <span>{agents.length}</span></button><button type="button" aria-pressed={panel === 'changes'} onClick={() => setPanel('changes')}>Changes <span>{reviews.length}</span></button><button type="button" aria-pressed={panel === 'checks'} onClick={() => setPanel('checks')}>Checks <span>{snapshot?.checks?.length || 0}</span></button><button type="button" aria-pressed={panel === 'preview'} onClick={() => setPanel('preview')}>Preview</button></div>
          {['checks', 'preview', 'project'].includes(panel) ? <CodingProjectPanel panel={panel} coding={coding} settings={projectSettings} setSettings={setProjectSettings} running={running} /> : panel === 'agents' ? <>
            {snapshot?.plan?.length > 0 && <section className="coding-plan-section"><div className="coding-section-heading"><strong>Work plan</strong><span>{snapshot.plan.filter(s => s.status === 'done').length}/{snapshot.plan.length}</span></div><ol className="coding-plan" aria-label="Agent task plan">{snapshot.plan.map((step, i) => <li key={i} className={'is-' + step.status}><span>{step.status === 'done' ? <IconCheck size={12} /> : step.status === 'in_progress' ? <span className="coding-dot" /> : i + 1}</span>{describeCodingMessage(step.text).description}</li>)}</ol></section>}
            <div className="coding-section-heading coding-agent-heading"><strong>Agents</strong><span>{agents.length || 'Ready'}</span></div>
            {!agents.length && <p className="coding-team-summary">Assignments will appear here when the work starts.</p>}
            <div className="coding-agent-list">{agents.map(a => <button type="button" className="coding-agent" key={a.id} aria-pressed={selectedAgent?.id === a.id} onClick={() => setAgentId(a.id)}><span className="coding-row"><span className="coding-agent-icon">{a.id === 'lead' ? <CamelidMark size={20} /> : <IconSearch size={17} />}</span><strong>{agentName(a)}</strong><span className={'coding-status is-' + a.status}>{label(a.status)}</span></span><span>{describeCodingMessage(a.goal).description || 'Review the shared code'}</span><small><span className={`coding-agent-dot is-${a.status}`} />{describeCodingAction(a.action, a.status)}</small></button>)}</div>
            {selectedAgent && <section className="coding-agent-detail"><div className="coding-section-heading"><strong>{agentName(selectedAgent)} activity</strong><span>{selectedAgent.files.length} files</span></div><p>{selectedAgent.parent_id ? 'Read-only helper · reports to Lead' : 'Lead · coordinates changes and checks'}</p>
              {selectedAgent.id !== 'lead' && selectedAgent.output && <div className="coding-findings"><AssistantMarkdown content={describeCodingMessage(selectedAgent.output).description || 'Findings include code. Expand to inspect.'} />{describeCodingMessage(selectedAgent.output).snippets.map((snippet, i) => <details key={i}><summary>Code excerpt {i + 1}</summary><pre>{snippet.content}</pre></details>)}</div>}
              <ol className="coding-event-list">{selectedEvents.slice(-8).map(e => <li key={e.seq}><span className={'coding-event-marker ' + (e.detail?.ok === false ? 'is-error' : '')} /><div><strong>{describeCodingEvent(e)}</strong><small>{new Date(e.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</small><details><summary>Details</summary><pre>{e.detail?.content || e.detail?.detail || e.detail?.message || JSON.stringify(e.detail, null, 2)}</pre></details></div></li>)}</ol>
              {selectedAgent.files.length > 0 && <details className="coding-touched-files"><summary>Files inspected or changed</summary>{selectedAgent.files.map(f => <div className="coding-agent-file" key={f}><IconFile size={13} /><span>{f}</span></div>)}</details>}
            </section>}
            <div className="coding-team-footer"><IconCpu size={15} /><span>Local model · up to 2 helpers<br />File reviews and Undo stay available.</span></div>
          </> : <div className="coding-review-list">{reviews.length ? reviews.map(r => <button type="button" key={r.id} onClick={() => openReview(r.id)}><IconFile size={16} /><span><strong>{relativeName(r.path)}</strong><small>{r.path}</small></span><span className={'coding-status is-' + r.status}>{r.status}</span><IconChevronRight size={14} /></button>) : <p>Proposed file changes will appear here.</p>}</div>}
          {sourceSnippets.length > 0 && <button type="button" className="coding-source-link coding-sidebar-sources" onClick={() => setSnippetsOpen(true)}><IconApi size={14} />Conversation code ({sourceSnippets.length})<IconChevronRight size={14} /></button>}
        </>}
      </aside>
    </div>
    {folderOpen && <FolderPicker apiBase={apiBase} initialPath={workspace || null} onCreate={(parent, name) => createCodingFolder(apiBase, parent, name)} onPick={value => { setWorkspace(value); setFolderOpen(false) }} onClose={() => setFolderOpen(false)} />}
    <ConfirmDialog open={confirmRemove} title="Remove this coding session?" detail="This removes the saved conversation and agent activity. Reviewed file changes and their Undo history remain in Changes." confirmLabel="Remove session" onCancel={() => setConfirmRemove(false)} onConfirm={async () => { try { await coding.remove(); setConfirmRemove(false) } catch { /* visible hook error */ } }} />
  </section>
}
