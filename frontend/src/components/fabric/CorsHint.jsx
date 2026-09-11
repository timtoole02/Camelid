import { CopyableCommand } from './CopyableCommand.jsx'
import { corsCommand } from '../../lib/fabricModel.js'

/* What to do when the browser may have kept a proxy's answer from this page.

   `camelid fabric serve` sends no CORS headers unless it was started with
   `--cors-origin`, and a browser reports that refusal exactly as it reports a
   dead address. `blocked` means an opaque follow-up proved something answered;
   `possible` means the page genuinely cannot tell, and says so. */
export function CorsHint({ pageOrigin, diagnosis }) {
  if (!pageOrigin || !diagnosis) return null
  return (
    <div className="fabric-cors-hint" data-testid="fabric-cors-hint" data-diagnosis={diagnosis}>
      <p className="fabric-note">
        {diagnosis === 'blocked'
          ? <>If it is a fabric proxy, it does not allow this page's origin (<code>{pageOrigin}</code>).</>
          : (
            <>
              If a proxy is running there, it may not allow this page's origin (<code>{pageOrigin}</code>).
              A browser reports that exactly like a dead address, so this page cannot tell the two apart.
            </>
          )}
        {' '}A fabric proxy only answers a page on another origin when it was started with{' '}
        <code>--cors-origin</code> naming that origin. Restart it with this flag added to its usual{' '}
        <code>--node</code> arguments:
      </p>
      <CopyableCommand command={corsCommand(pageOrigin)} />
    </div>
  )
}

export default CorsHint
