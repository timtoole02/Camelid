import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import SidebarRail from './components/layout/SidebarRail'
import TopBar from './components/TopBar'
import { BackendBanner, ENGINE_PATH_STORAGE_KEY, ENGINE_ADDR_STORAGE_KEY } from './components/layout/BackendBanner'
import { FirstRunCard } from './components/onboarding/FirstRunCard'
import { Notice } from './components/ui/Notice'
import { ConfirmDialog } from './components/ui/ConfirmDialog'
import { isFirstRunHost } from './lib/firstRunActivation'
import { formatPreview, formatSidebarDate } from './lib/formatters'
import { hasOverlayTitleBar } from './lib/desktopShell'
import { useDashboardData } from './hooks/useDashboardData'
import { useBackendLauncher } from './hooks/useBackendLauncher'
import { useNotice } from './hooks/useNotice'
import { useTheme } from './hooks/useTheme'
import { ensureInferenceTelemetryConnected } from './hooks/useInferenceTelemetry'
import ChatWorkspace from './views/ChatWorkspace'
import { CommandPalette } from './components/CommandPalette'
import { ShortcutsOverlay } from './components/ShortcutsOverlay'

/* Route-level code splitting (Phase 7): chat is the default surface and stays
   eager; every other view loads on first visit. */
const AnalyticsView = lazy(() => import('./views/AnalyticsView'))
const HistoryView = lazy(() => import('./views/HistoryView'))
const MemoryView = lazy(() => import('./views/MemoryView'))
const ModelsView = lazy(() => import('./views/ModelsView'))
const DownloadedModelsView = lazy(() => import('./views/DownloadedModelsView'))
const ApiView = lazy(() => import('./views/ApiView'))
const SystemView = lazy(() => import('./views/SystemView'))
const SettingsView = lazy(() => import('./views/SettingsView'))
const ClusterView = lazy(() => import('./views/ClusterView'))
const CompatibilityView = lazy(() => import('./views/CompatibilityView'))
const TelemetryView = lazy(() => import('./views/TelemetryView'))
const InferenceObservatoryView = lazy(() => import('./views/InferenceObservatoryView'))
const WorkspaceView = lazy(() => import('./views/WorkspaceView'))

const DEMO_UI = import.meta.env?.VITE_CAMELID_DEMO_UI === 'true'
const HASH_TABS = new Set(['chat', 'workspace', 'library', 'downloads', 'api', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'observatory', 'compatibility', 'telemetry'])

function App() {
  const { notice, noticeTone, showNotice, clearNotice } = useNotice()
  const { preference, resolved, cyclePreference, setPreference } = useTheme()

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (DEMO_UI) return true
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem('camelid.sidebarCollapsed') === 'true'
  })
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(max-width: 860px)').matches,
  )
  const [pendingDeleteConversationId, setPendingDeleteConversationId] = useState(null)
  const [ledgerFocusRow, setLedgerFocusRow] = useState(null)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [modelsVisited, setModelsVisited] = useState(false)
  const [firstRunCardActive, setFirstRunCardActive] = useState(false)
  const viewRef = useRef(null)
  const hamburgerRef = useRef(null)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined
    const media = window.matchMedia('(max-width: 860px)')
    const sync = () => { setIsMobile(media.matches); if (!media.matches) setMobileNavOpen(false) }
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [])

  const dash = useDashboardData({ showNotice, clearNotice })
  const {
    dashboard, tab, setTab, selectedConversationId, setSelectedConversationId,
    selectedModelId, setSelectedModelId, search, setSearch, memorySearch, setMemorySearch,
    composer, setComposer, newChatTitle, setNewChatTitle, sending, receiptMode, setReceiptMode,
    thinkingMode, setThinkingMode,
    loadingModelId, registerForm, setRegisterForm,
    conversations, memories, filteredConversations, models, runtime, selectedConversation,
    selectedModel, selectedModelRunnable, selectedModelExperimental, latestAssistantMessage, pendingConversation,
    createConversation, showNewChatLanding, sendMessage, resendFromMessage, stopGeneration, saveToMemory,
    createMemory, updateMemory, deleteMemory, renameConversation, deleteConversation, deleteAllConversations,
    activateModel, unloadCurrentModel,
    registerModel, loadDashboard, stoppingGeneration,
    apiBase, setApiBase,
  } = dash

  const backend = useBackendLauncher({ showNotice, loadDashboard })

  useEffect(() => {
    if (tab === 'library') setModelsVisited(true)
  }, [tab])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('camelid.sidebarCollapsed', String(sidebarCollapsed))
  }, [sidebarCollapsed])

  // Deep-link the active tab via the URL hash (e.g. #library) — shareable and
  // handy for direct navigation. Runs once on mount if a valid tab hash is present.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const hash = window.location.hash.replace('#', '')
    if (HASH_TABS.has(hash)) setTab(hash)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const closeMobileNav = () => setMobileNavOpen(false)

  /* The persistent .camelid-view div is the scroll container for page views;
     because the node survives tab switches, each navigation must reset it or
     the previous view's scroll offset leaks into the next one. */
  useEffect(() => {
    viewRef.current?.scrollTo(0, 0)
  }, [tab])

  /* Mobile drawer: move focus into the drawer when it opens so keyboard and
     screen-reader users land where the navigation is. */
  useEffect(() => {
    if (!mobileNavOpen) return undefined
    const frame = window.requestAnimationFrame(() => {
      document.getElementById('camelid-sidebar')?.querySelector('button, input')?.focus()
    })
    return () => window.cancelAnimationFrame(frame)
  }, [mobileNavOpen])

  /* Observatory stream listens from app start (Phase 6.1 DEFECT 1): runs made
     before the view's first mount must not be invisible. */
  useEffect(() => {
    if (apiBase) ensureInferenceTelemetryConnected(apiBase)
  }, [apiBase])

  /* Remember where the engine lives while it is still answering. Once it stops,
     the offline banner needs this to offer a command that actually runs, and by
     then there is nobody left to ask. Loopback servers only — a remote engine
     reports no path, so nothing is stored and the banner stays generic. */
  useEffect(() => {
    const executable = runtime?.executable
    if (!executable || typeof window === 'undefined') return
    try {
      window.localStorage.setItem(ENGINE_PATH_STORAGE_KEY, executable)
      if (runtime?.listen_addr) window.localStorage.setItem(ENGINE_ADDR_STORAGE_KEY, runtime.listen_addr)
    } catch { /* private mode */ }
  }, [runtime?.executable, runtime?.listen_addr])

  /* Evidence Chips anywhere in the app deep-link to their ledger row through
     this event — no prop drilling through every chip call site. */
  useEffect(() => {
    const onOpenLedger = (event) => {
      setLedgerFocusRow(event.detail?.rowId || null)
      setTab('compatibility')
      if (typeof window !== 'undefined') window.history.replaceState(null, '', '#compatibility')
    }
    window.addEventListener('camelid:open-ledger', onOpenLedger)
    return () => window.removeEventListener('camelid:open-ledger', onOpenLedger)
  }, [setTab])

  /* Global keys: Cmd/Ctrl+K command palette; "?" shortcut map outside inputs;
     Escape closes the mobile nav drawer (palette/shortcuts handle their own). */
  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setShortcutsOpen(false)
        setPaletteOpen((value) => !value)
        return
      }
      if (event.key === 'Escape' && mobileNavOpen && !paletteOpen && !shortcutsOpen) {
        event.preventDefault()
        setMobileNavOpen(false)
        hamburgerRef.current?.focus()
        return
      }
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable
      if (event.key === '?' && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        setPaletteOpen(false)
        setShortcutsOpen((value) => !value)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [mobileNavOpen, paletteOpen, shortcutsOpen])

  const navigateTab = (next) => {
    setTab(next)
    if (typeof window !== 'undefined' && HASH_TABS.has(next)) {
      window.history.replaceState(null, '', next === 'chat' ? window.location.pathname : `#${next}`)
    }
    closeMobileNav()
  }

  const selectConversation = (id) => {
    setSelectedConversationId(id)
    setTab('chat')
    closeMobileNav()
  }

  const startNewChat = () => {
    showNewChatLanding()
    closeMobileNav()
  }

  const requestDeleteConversation = (id) => {
    setPendingDeleteConversationId(id)
    setDeleteBusy(false)
  }

  const pendingDeleteConversation = useMemo(
    () => conversations.find((c) => c.id === pendingDeleteConversationId) || null,
    [conversations, pendingDeleteConversationId],
  )

  const pendingDeleteTitle = useMemo(() => {
    if (!pendingDeleteConversation) return 'Delete chat?'
    const trimmed = pendingDeleteConversation.title?.trim()
    if (trimmed && trimmed.toLowerCase() !== 'new conversation') return `Delete “${trimmed}”?`
    return `Delete untitled chat · ${formatSidebarDate(pendingDeleteConversation.updated_at) || 'new chat'}?`
  }, [pendingDeleteConversation])

  const pendingDeleteDetail = useMemo(() => {
    if (!pendingDeleteConversation) return ''
    const latest = [...(pendingDeleteConversation.messages || [])].reverse()
      .find((m) => typeof m?.content === 'string' && m.content.trim())
    const preview = formatPreview(latest?.content, 80)
    return preview === 'No messages yet' ? 'This conversation will be permanently removed.' : `“${preview}” — this conversation will be permanently removed.`
  }, [pendingDeleteConversation])

  const handleDeleteConfirm = async () => {
    if (!pendingDeleteConversationId || deleteBusy) return
    setDeleteBusy(true)
    const ok = await deleteConversation(pendingDeleteConversationId)
    if (ok) setPendingDeleteConversationId(null)
    setDeleteBusy(false)
  }

  /* First-run activation. Mounted here rather than inside the chat view so an
     in-flight download keeps its watcher when the user navigates away, and kept
     mounted while the card still owns the flow: the moment the file lands the host
     stops looking like a fresh install, so `firstRun` alone would drop the card on
     the next refresh -- abandoning the load, or deleting a failure's Retry button
     while the downloaded artifact sits there unused. */
  const firstRun = isFirstRunHost({ runtime, models })
  const firstRunActive = firstRun || firstRunCardActive
  const handleFirstRunActivated = useCallback(async () => {
    await loadDashboard({ silent: true })
    navigateTab('chat')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadDashboard])

  /* Above the loading early-return: every hook must run on every render, and
     the shell this window lives in cannot change while it is open. */
  const [overlayTitleBar] = useState(hasOverlayTitleBar)

  if (!dashboard) {
    return (
      <div className="loading-shell">
        <div className="loading-shell-stack">
          <Notice notice={notice} tone={noticeTone} />
          <div>Loading Camelid…</div>
        </div>
      </div>
    )
  }

  const shellClasses = [
    'camelid-app',
    sidebarCollapsed ? 'is-collapsed' : '',
    mobileNavOpen ? 'is-mobile-open' : '',
    DEMO_UI ? 'is-demo' : '',
    overlayTitleBar ? 'has-overlay-chrome' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className={shellClasses}>
      {/* macOS desktop only: the window draws no title bar of its own, so the
          traffic lights float over the top-left of our content. This strip is
          the room they sit in and the surface the window is dragged by —
          without it the app showed the OS bar stacked above our own header,
          which is what makes an app look like a web page in a frame. */}
      {overlayTitleBar && <div className="app-drag-strip" data-tauri-drag-region />}
      {!DEMO_UI && (
        <SidebarRail
          collapsed={!isMobile && sidebarCollapsed}
          onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
          showNewChatLanding={startNewChat}
          search={search}
          setSearch={setSearch}
          tab={tab}
          setTab={navigateTab}
          filteredConversations={filteredConversations}
          selectedConversationId={selectedConversation?.id || null}
          onSelectConversation={selectConversation}
          renameConversation={renameConversation}
          requestDeleteConversation={requestDeleteConversation}
          runtime={runtime}
          themePreference={preference}
          themeResolved={resolved}
          onCycleTheme={cyclePreference}
        />
      )}

      {!DEMO_UI && mobileNavOpen && (
        <button type="button" className="camelid-app__scrim" aria-label="Close navigation" onClick={closeMobileNav} />
      )}

      <main className="camelid-main" data-view={tab}>
        <TopBar
          tab={tab}
          setTab={navigateTab}
          selectedConversationTitle={selectedConversation?.title || ''}
          runtime={runtime}
          capabilities={dashboard?.capabilities}
          selectedModelId={selectedModelId}
          models={models}
          onToggleSidebar={DEMO_UI ? null : () => setMobileNavOpen((v) => !v)}
          mobileNavOpen={mobileNavOpen}
          menuButtonRef={hamburgerRef}
          demoMode={DEMO_UI}
        />

        {notice && (
          <div className="camelid-notice-slot">
            <Notice notice={notice} tone={noticeTone} onDismiss={clearNotice} />
          </div>
        )}

        {!DEMO_UI && runtime?.status === 'offline' && tab !== 'settings' && (
          <BackendBanner backend={backend} apiBase={apiBase} onOpenSettings={() => navigateTab('settings')} />
        )}

        {!DEMO_UI && firstRunActive && (
          <div className="camelid-firstrun-slot" hidden={tab !== 'chat'}>
            <FirstRunCard
              apiBase={apiBase}
              capabilities={dashboard?.capabilities}
              onActivated={handleFirstRunActivated}
              onActiveChange={setFirstRunCardActive}
              onOpenModels={() => navigateTab('library')}
            />
          </div>
        )}

        {/* --chat is the full-bleed frame for views that own their own edges and
           manage their own height (chat, workspace, cluster canvas). Every
           other view is a .cxv page and needs the padded page frame. */}
        <div ref={viewRef} className={`camelid-view ${(tab === 'chat' || tab === 'workspace' || tab === 'cluster') ? 'camelid-view--chat' : 'camelid-view--page'}`}>
          <Suspense fallback={<div className="view-loading" role="status" aria-label="Loading view">Loading view…</div>}>
          {tab === 'chat' && (
            <ChatWorkspace
              selectedConversation={selectedConversation}
              selectedModel={selectedModel}
              selectedModelId={selectedModelId}
              setSelectedModelId={setSelectedModelId}
              activateModel={activateModel}
              loadingModelId={loadingModelId}
              models={models}
              runtime={runtime}
              capabilities={dashboard?.capabilities}
              latestAssistantMessage={latestAssistantMessage}
              pendingConversation={pendingConversation}
              composer={composer}
              setComposer={setComposer}
              saveToMemory={saveToMemory}
              sendMessage={sendMessage}
              resendFromMessage={resendFromMessage}
              stopGeneration={stopGeneration}
              sending={sending}
              receiptMode={receiptMode}
              setReceiptMode={setReceiptMode}
              thinkingMode={thinkingMode}
              setThinkingMode={setThinkingMode}
              stoppingGeneration={stoppingGeneration}
              selectedModelRunnable={selectedModelRunnable}
              selectedModelExperimental={selectedModelExperimental}
              setTab={navigateTab}
              showNewChatLanding={startNewChat}
              firstRunActive={firstRunActive}
              demoMode={DEMO_UI}
            />
          )}

          {tab === 'workspace' && (
            <WorkspaceView
              apiBase={apiBase}
              capabilities={dashboard?.capabilities}
              selectedModel={selectedModel}
              runtime={runtime}
              setTab={navigateTab}
            />
          )}

          {tab === 'analytics' && (
            <AnalyticsView apiBase={apiBase} conversations={conversations} models={models} runtime={runtime} capabilities={dashboard?.capabilities} />
          )}

          {tab === 'history' && (
            <HistoryView
              filteredConversations={filteredConversations}
              setSelectedConversationId={selectConversation}
              setTab={navigateTab}
              deleteConversation={requestDeleteConversation}
            />
          )}

          {tab === 'memory' && (
            <MemoryView
              memories={memories}
              memorySearch={memorySearch}
              setMemorySearch={setMemorySearch}
              selectedConversation={selectedConversation}
              latestAssistantMessage={latestAssistantMessage}
              saveToMemory={saveToMemory}
              createMemory={createMemory}
              updateMemory={updateMemory}
              deleteMemory={deleteMemory}
              setTab={navigateTab}
            />
          )}

          {modelsVisited && (
            <div hidden={tab !== 'library'}>
              <ModelsView
                runtime={runtime}
                capabilities={dashboard?.capabilities}
                refreshDashboard={loadDashboard}
                onOpenChat={() => navigateTab('chat')}
                unloadCurrentModel={unloadCurrentModel}
                loadingModelId={loadingModelId}
                registerForm={registerForm}
                setRegisterForm={setRegisterForm}
                registerModel={registerModel}
                apiBase={apiBase}
              />
            </div>
          )}

          {tab === 'downloads' && (
            <DownloadedModelsView
              runtime={runtime}
              apiBase={apiBase}
              refreshDashboard={loadDashboard}
              onOpenModels={() => navigateTab('library')}
            />
          )}

          {tab === 'api' && <ApiView runtime={runtime} selectedModel={selectedModel} capabilities={dashboard?.capabilities} />}

          {tab === 'telemetry' && <TelemetryView />}

          {tab === 'compatibility' && (
            <CompatibilityView
              capabilities={dashboard?.capabilities}
              focusRowId={ledgerFocusRow}
              onFocusConsumed={() => setLedgerFocusRow(null)}
            />
          )}

          {tab === 'system' && <SystemView runtime={runtime} selectedModel={selectedModel} capabilities={dashboard?.capabilities} />}

          {tab === 'settings' && (
            <SettingsView
              runtime={runtime}
              apiBase={apiBase}
              setApiBase={setApiBase}
              refreshDashboard={loadDashboard}
              backend={backend}
              showNotice={showNotice}
              themePreference={preference}
              setThemePreference={setPreference}
              onOpenCluster={() => navigateTab('cluster')}
              conversationCount={conversations.length}
              deleteAllConversations={deleteAllConversations}
              selectedModel={selectedModel}
              capabilities={dashboard?.capabilities}
            />
          )}

          {tab === 'cluster' && <ClusterView showNotice={showNotice} />}

          {tab === 'observatory' && <InferenceObservatoryView apiBase={apiBase} runtime={runtime} selectedModel={selectedModel} capabilities={dashboard?.capabilities} />}
          </Suspense>
        </div>
      </main>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        setTab={navigateTab}
        showNewChatLanding={startNewChat}
        cyclePreference={cyclePreference}
        models={models}
        capabilities={dashboard?.capabilities?.model_compatibility || []}
        setSelectedModelId={setSelectedModelId}
      />
      <ShortcutsOverlay open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

            <ConfirmDialog
        open={Boolean(pendingDeleteConversation)}
        title={pendingDeleteTitle}
        detail={pendingDeleteDetail}
        confirmLabel="Delete"
        busy={deleteBusy}
        onCancel={() => { if (!deleteBusy) setPendingDeleteConversationId(null) }}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  )
}

export default App
