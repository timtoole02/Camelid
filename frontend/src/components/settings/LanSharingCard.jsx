import { useEffect, useState } from 'react'
import { Card, CardHeader, CardBody } from '../ui/Card'
import { Button } from '../ui/Button'
import { copyText } from '../../lib/markdown'
import { getStoredApiKey } from '../../lib/apiAuth'

export function LanSharingCard({ apiBase, online }) {
  const [sharing, setSharing] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const endpoint = `${(apiBase || '').replace(/\/$/, '')}/api/runtime/lan-sharing`
  const headers = () => {
    const key = getStoredApiKey()
    return { 'content-type': 'application/json', ...(key ? { Authorization: `Bearer ${key}` } : {}) }
  }
  useEffect(() => {
    setSharing(null)
    if (!online) return
    let cancelled = false
    fetch(endpoint, { headers: headers(), cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(value => { if (!cancelled) setSharing(value) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [endpoint, online])

  async function toggle() {
    setBusy(true)
    setError('')
    try {
      const response = await fetch(endpoint, {
        method: 'POST', headers: headers(), body: JSON.stringify({ enabled: !sharing.enabled }),
      })
      if (!response.ok) throw new Error('Could not change network sharing. Try again.')
      setSharing(await response.json())
    } catch (error) { setError(error.message) }
    finally { setBusy(false) }
  }

  async function copy(value) {
    try { if (!(await copyText(value))) setError('Could not copy. Select and copy the text manually.') }
    catch { setError('Could not copy. Select and copy the text manually.') }
  }

  if (!sharing) return null
  return <Card>
    <CardHeader eyebrow="Network" title="Share on local network" />
    <CardBody>
      <p className="settings-help">Let devices on your network chat using this computer’s loaded model. Remote users need the access key and can switch local models.</p>
      <p className="settings-help settings-help--muted">Use on a trusted network: the connection uses HTTP, so messages and the key are not encrypted. Sharing turns off when Camelid closes.</p>
      <div className="settings-actions">
        <Button role="switch" aria-checked={sharing.enabled} onClick={toggle} disabled={busy} loading={busy}>
          {sharing.enabled ? 'Turn off network sharing' : 'Turn on network sharing'}
        </Button>
      </div>
      {sharing.enabled && <>
        <p className="settings-help">Open this address on the other device, then enter the access key in Settings. Keep Camelid open.</p>
        <div className="settings-inline">
          <input aria-label="Network address" readOnly value={sharing.url || `http://<this computer’s IP>:${sharing.port}`} />
          {sharing.url && <Button onClick={() => copy(sharing.url)}>Copy address</Button>}
        </div>
        <div className="settings-inline">
          <input aria-label="Network access key" readOnly type="password" value={sharing.key || ''} />
          <Button onClick={() => copy(sharing.key)}>Copy key</Button>
        </div>
        <p className="settings-help settings-help--muted">If the address does not connect, check that both devices use the same network and your firewall allows Camelid. A VPN may change the displayed address.</p>
      </>}
      {error && <p role="alert" className="settings-help">{error}</p>}
    </CardBody>
  </Card>
}
