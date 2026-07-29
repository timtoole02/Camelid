import { useRef, useState } from 'react'
import { useClusterTopology } from '../hooks/useClusterTopology'
import { TopologyCanvas } from '../components/cluster/TopologyCanvas'
import { NodeInventory } from '../components/cluster/NodeInventory'
import { NodeInspector } from '../components/cluster/NodeInspector'
import { ClusterDrawer } from '../components/cluster/ClusterDrawer'
import { AddServerWizard } from '../components/cluster/AddServerWizard'
import { DiscoverDevices } from '../components/cluster/DiscoverDevices'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import {
  IconPlus, IconNetwork, IconCheckCircle, IconCheck, IconDownload, IconGrid, IconChart, IconTrash,
} from '../components/ui/icons'

export default function ClusterView({ showNotice }) {
  const cluster = useClusterTopology({ showNotice })
  const canvasRef = useRef(null)
  const [wizardOpen, setWizardOpen] = useState(false)
  const [discoverOpen, setDiscoverOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [clearOpen, setClearOpen] = useState(false)

  const resetLayout = () => { cluster.applyAutoLayout(); window.setTimeout(() => canvasRef.current?.fit(), 60) }

  const inspectorActions = {
    testNode: cluster.testNode,
    startWorker: cluster.startWorker,
    stopWorker: cluster.stopWorker,
    restartWorker: cluster.restartWorker,
    removeNode: cluster.removeNode,
    updateNode: cluster.updateNode,
    testConnection: cluster.testConnection,
    updateConnection: cluster.updateConnection,
    removeConnection: cluster.removeConnection,
  }

  return (
    <div className="cluster-view">
      <header className="cluster-header">
        <div className="cluster-header__copy">
          <h1>Cluster Topology</h1>
          <p>Connect Macs, Windows PCs, Linux servers, and Raspberry Pis into one local Camelid compute fabric.</p>
        </div>
        <div className="cluster-header__actions">
          <div className="cluster-header__primary">
            <Button variant="primary" icon={<IconPlus size={16} />} onClick={() => setWizardOpen(true)}>Add Server</Button>
            <Button variant="tonal" icon={<IconNetwork size={16} />} onClick={() => setDiscoverOpen(true)}>Discover Devices</Button>
            <Button variant="tonal" icon={<IconCheckCircle size={16} />} onClick={cluster.validateCluster}>Validate Cluster</Button>
            <Button variant="tonal" icon={<IconCheck size={16} />} onClick={cluster.save}>Save Topology</Button>
          </div>
          <div className="cluster-header__secondary">
            <Button variant="ghost" size="sm" icon={<IconDownload size={15} />} onClick={cluster.exportTopology}>Export Diagram</Button>
            <Button variant="ghost" size="sm" icon={<IconGrid size={15} />} onClick={resetLayout}>Reset Layout</Button>
            <Button variant="ghost" size="sm" icon={<IconChart size={15} />} onClick={() => setDrawerOpen(true)}>View Logs</Button>
            {cluster.nodes.length > 0 && (
              <Button variant="ghost" size="sm" icon={<IconTrash size={15} />} onClick={() => setClearOpen(true)}>Clear Cluster</Button>
            )}
          </div>
        </div>
      </header>

      <div className="cluster-body">
        <NodeInventory
          nodes={cluster.nodes}
          summary={cluster.summary}
          selection={cluster.selection}
          onSelect={cluster.select}
          onAddServer={() => setWizardOpen(true)}
        />

        <TopologyCanvas
          ref={canvasRef}
          nodes={cluster.nodes}
          connections={cluster.connections}
          selection={cluster.selection}
          busyIds={cluster.busyIds}
          onSelect={cluster.select}
          onMoveNode={cluster.moveNode}
          onAddConnection={cluster.addConnection}
          onAutoLayout={resetLayout}
          onAddServer={() => setWizardOpen(true)}
          onLoadSample={cluster.loadSample}
        />

        <NodeInspector
          selectedNode={cluster.selectedNode}
          selectedConnection={cluster.selectedConnection}
          nodes={cluster.nodes}
          actions={inspectorActions}
          onViewLogs={() => setDrawerOpen(true)}
        />
      </div>

      <ClusterDrawer
        events={cluster.events}
        issues={cluster.issues}
        summary={cluster.summary}
        open={drawerOpen}
        onToggle={() => setDrawerOpen((v) => !v)}
      />

      <AddServerWizard open={wizardOpen} onClose={() => setWizardOpen(false)} onAdd={(node) => { cluster.addNode(node); setWizardOpen(false) }} />
      <DiscoverDevices open={discoverOpen} onClose={() => setDiscoverOpen(false)} onDiscover={cluster.discover} onAdd={(partial) => cluster.addNode(partial)} />

      <ConfirmDialog
        open={clearOpen}
        title="Clear the cluster topology?"
        detail={`This removes all ${cluster.nodes.length} node${cluster.nodes.length === 1 ? '' : 's'} and ${cluster.connections.length} link${cluster.connections.length === 1 ? '' : 's'} from this diagram, including the sample fabric. It only clears what is drawn here — no machine is touched and nothing is uninstalled. Export the diagram first if you want a copy.`}
        confirmLabel="Clear topology"
        onCancel={() => setClearOpen(false)}
        onConfirm={() => { cluster.resetTopology(); setClearOpen(false); showNotice?.('Cluster topology cleared.', 'info') }}
      />
    </div>
  )
}
