import { useState } from 'react'
import { Button } from '../ui/Button'
import { IconCopy } from '../ui/icons'
import { copyText } from '../../lib/clipboard.js'

/* A command the operator runs themselves. This page cannot start or restart a
   process on the machine that hosts the proxy, so it hands over the command
   rather than a button that would only pretend to act. */
export function CopyableCommand({ command }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="fabric-cmd">
      <code>{command}</code>
      <Button
        variant="ghost"
        size="sm"
        icon={<IconCopy size={14} />}
        onClick={async () => {
          const ok = await copyText(command)
          setCopied(ok)
          window.setTimeout(() => setCopied(false), 2000)
        }}
      >
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  )
}

export default CopyableCommand
