import { isIP } from 'node:net'

const loopback = value => value === '::1' || value === '[::1]'
  || (isIP(value.replace(/^::ffff:/, '')) === 4 && value.replace(/^::ffff:/, '').startsWith('127.'))

/* Local developer administration must not turn an untrusted page into a
   same-origin request to the backend or a command-launch request. */
export function devRequestOriginAllowed(request) {
  if (!loopback(request.socket?.remoteAddress || '')) return false
  try {
    const host = new URL(`http://${request.headers.host}`)
    if (host.username || host.password || (host.hostname !== 'localhost' && !loopback(host.hostname))) return false
    const origin = request.headers.origin
    if (origin !== undefined) {
      const parsed = new URL(origin)
      return ['http:', 'https:'].includes(parsed.protocol)
        && !parsed.username && !parsed.password && !parsed.search && !parsed.hash
        && parsed.pathname === '/' && parsed.host === host.host
    }
    return !request.headers['sec-fetch-site'] || ['same-origin', 'none'].includes(request.headers['sec-fetch-site'])
  } catch { return false }
}
