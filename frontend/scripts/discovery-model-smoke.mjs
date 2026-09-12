#!/usr/bin/env node
/* Unit coverage for the discovery panel's pure layer.
 *
 * Discovery is the one surface that sends traffic to machines the user did not
 * name, and the page's job is to keep saying exactly what the proxy
 * established and nothing more. These checks pin the rules that stop it
 * helping a reader conclude more than that.
 *
 * No browser and no network: every input is a literal proxy answer.
 */
import assert from 'node:assert/strict'
import {
  describeDiscovery,
  describeFinding,
  describeJoin,
  describePolicy,
  groupFindings,
  groupOf,
  problemMessage,
} from '../src/lib/discoveryModel.js'
import { getPolicy, join, runScan } from '../src/lib/discoveryClient.js'

let checks = 0
function check(name, fn) {
  fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}
async function checkAsync(name, fn) {
  await fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

const PROPOSAL = {
  label: 'host-100-64-0-37-ollama',
  engine: 'ollama',
  host: '100.64.0.37',
  port: 11434,
  line: 'host-100-64-0-37-ollama=ollama://100.64.0.37:11434',
  comment_preview: '# joined by fabric discover <time of writing>: 100.64.0.37:11434 answered like ollama 0.33.2',
  scanned_address: '100.64.0.37:11434',
  engine_choices: [],
  host_alternatives: [],
  warnings: ['cleartext'],
}

function finding(overrides = {}) {
  return {
    id: 'f1',
    address: '100.64.0.37',
    port: 11434,
    addresses: ['100.64.0.37'],
    via_name: null,
    this_machine: false,
    identity_basis: 'unauthenticated_answer',
    engines: [
      { engine: 'camelid', verdict: 'ruled_out', detail: '/v1/health answered HTTP 404' },
      { engine: 'ollama', verdict: 'matched', version: '0.33.2' },
      { engine: 'lmstudio', verdict: 'ruled_out' },
    ],
    classification: { kind: 'answers_like', engine: 'ollama', version: '0.33.2', withheld_elsewhere: [] },
    evidence: [{ request: 'GET /api/version', status: 200, content_type: 'application/json', fact: 'a JSON object naming a non-empty version', matched: ['ollama'] }],
    name: { name: null, why: 'no reverse DNS record', rejected: null, resolved: [] },
    in_fabric: null,
    possibly_same_as: [],
    proposal: PROPOSAL,
    not_proposed: null,
    ...overrides,
  }
}

function discovery(overrides = {}) {
  return {
    scope: { addresses: 254, ports: [11434], limits: {} },
    transport: 'cleartext (direct remote transport explicitly acknowledged)',
    credentials_presented: 'none',
    user_agent: 'camelid-fabric-discover/0.7.3',
    elapsed_ms: 9800,
    planned: 254,
    probes: 254,
    not_listed: { refused: 12, timed_out: 240, unreachable: 0, other: 0 },
    not_scanned: 0,
    nodes_file: { path: '/etc/camelid/nodes', sha256: 'abc123' },
    findings: [finding()],
    hint: null,
    ...overrides,
  }
}

/* ---- what may be added, and what may not ---- */

check('an unknown classification is never a candidate', () => {
  // A future proxy kind must not fall through to "looks addable", even when
  // the body carries a perfectly well-formed proposal.
  const row = describeFinding(finding({ classification: { kind: 'teleport' }, proposal: PROPOSAL }))
  assert.equal(row.kind, 'unrecognized')
  assert.equal(row.canJoin, false, 'an unrecognised kind must never be joinable')
  assert.equal(groupOf(row), 'unrecognized')

  // ...and the kinds this build does know behave as stated.
  assert.equal(describeFinding(finding()).canJoin, true)
  for (const kind of ['other_http', 'fabric_proxy', 'requires_credentials', 'incomplete', 'not_http', 'silent_after_connect', 'tls_not_authenticated']) {
    const other = describeFinding(finding({ classification: { kind }, proposal: PROPOSAL }))
    assert.equal(other.canJoin, false, `${kind} must not be joinable even with a proposal`)
  }
})

check('a row with no proposal is never joinable whatever its kind', () => {
  const row = describeFinding(finding({ proposal: null }))
  assert.equal(row.canJoin, false)
  // A proposal missing the fields a line is built from is not a proposal.
  assert.equal(describeFinding(finding({ proposal: { label: 'x' } })).canJoin, false)
})

check('an ambiguous row carries the choice and never resolves it', () => {
  const row = describeFinding(finding({
    classification: { kind: 'ambiguous', engines: ['camelid', 'ollama'] },
    proposal: { ...PROPOSAL, engine_choices: ['camelid', 'ollama'] },
  }))
  assert.equal(row.canJoin, true)
  assert.deepEqual(row.proposal.engineChoices, ['camelid', 'ollama'])
  assert.equal(groupOf(row), 'ambiguous')
})

check('an unsettled row carries no engine, and neither does the page', () => {
  // The server sends no engine for a row that matched two, so there is nothing
  // here to fall back to a first match with.
  const row = describeFinding(finding({
    classification: { kind: 'ambiguous', engines: ['camelid', 'ollama'] },
    proposal: {
      ...PROPOSAL,
      engine: null,
      engine_choices: ['camelid', 'ollama'],
      line: 'host-100-64-0-37-ollama=<engine>://100.64.0.37:11434',
    },
  }))
  assert.equal(row.canJoin, true, 'the choice is the thing the confirm panel collects')
  assert.equal(row.proposal.engine, null, 'the page must not invent one either')
  assert.deepEqual(row.proposal.engineChoices, ['camelid', 'ollama'])
  assert.match(row.proposal.line, /<engine>/)

  // No engine and nothing to choose between is not a proposal at all.
  const empty = describeFinding(finding({ proposal: { ...PROPOSAL, engine: null, engine_choices: [] } }))
  assert.equal(empty.proposal, null)
  assert.equal(empty.canJoin, false)
})

check('the socket the scan reached is read from the body, never composed', () => {
  const row = describeFinding(finding({
    address: 'fd00::1',
    proposal: { ...PROPOSAL, host: '[fd00::1]', scanned_address: '[fd00::1]:11434' },
  }))
  assert.equal(row.proposal.scannedAddress, '[fd00::1]:11434')
  // Exactly the shape a page gets wrong by building it out of address and
  // port: `fd00::1:11434` is not an address anything can read.
  assert.notEqual(row.proposal.scannedAddress, `${row.address}:${row.port}`)
})

check('an incomplete row names what it could not rule out', () => {
  const row = describeFinding(finding({
    classification: { kind: 'incomplete', unanswered: ['/api/tags'], matched_so_far: ['camelid'] },
    proposal: null,
  }))
  assert.equal(row.canJoin, false)
  assert.match(row.summary, /camelid/)
  assert.match(row.summary, /\/api\/tags/)
  assert.match(row.summary, /has not been established/)
})

check('a row already in the fabric is grouped as such and not offered again', () => {
  const row = describeFinding(finding({
    in_fabric: { label: 'studio', declared_engine: 'ollama', agrees: true, unknown: null },
    proposal: null,
    not_proposed: 'already in this fabric as `studio`',
  }))
  assert.equal(groupOf(row), 'in_fabric')
  assert.equal(row.inFabric.label, 'studio')
  assert.equal(row.canJoin, false)
})

check('an unresolvable existing spec leaves membership unknown, never false', () => {
  const row = describeFinding(finding({
    in_fabric: { label: null, declared_engine: null, agrees: null, unknown: '`studio` did not resolve' },
  }))
  assert.equal(row.inFabric.label, null)
  assert.match(row.inFabric.unknown, /did not resolve/)
})

/* ---- counts and failures ---- */

check('absent counts stay unknown and a failed scan is never empty', () => {
  const described = describeDiscovery(discovery({ not_listed: {}, not_scanned: undefined, planned: undefined }))
  assert.equal(described.notListed.refused, null, 'an absent count is unknown, not zero')
  assert.equal(described.notListed.timedOut, null)
  assert.equal(described.notScanned, null)
  assert.equal(described.planned, null)

  // A body that is not a set of findings produces nothing, so the caller has
  // to report a problem rather than an empty table.
  assert.equal(describeDiscovery({}), null)
  assert.equal(describeDiscovery(null), null)
  assert.equal(describeDiscovery({ findings: 'lots' }), null)
})

check('joining is off when there is nothing to write against', () => {
  assert.equal(describeDiscovery(discovery()).canJoin, true)
  assert.equal(describeDiscovery(discovery({ nodes_file: null })).canJoin, false)
  assert.equal(describeDiscovery(discovery({ nodes_file: { path: '/etc/camelid/nodes' } })).canJoin, false)
})

check('warnings pass through verbatim, and are never derived here', () => {
  const row = describeFinding(finding({
    proposal: { ...PROPOSAL, engine: 'camelid', warnings: ['cleartext', 'bearer_will_be_sent'] },
  }))
  assert.deepEqual(row.proposal.warnings, ['cleartext', 'bearer_will_be_sent'])
  // The same engine with no warning from the server carries none.
  const quiet = describeFinding(finding({ proposal: { ...PROPOSAL, engine: 'camelid', warnings: [] } }))
  assert.deepEqual(quiet.proposal.warnings, [])
})

check('a device-claimed name is an alternative, never the host', () => {
  const row = describeFinding(finding({
    name: { name: 'mini2.lan', source: 'reverse_dns', source_trust: 'device_claimed', proof: 'resolves_to_this_address', resolved: ['100.64.0.37'] },
    proposal: {
      ...PROPOSAL,
      host_alternatives: [{ host: 'mini2.lan', label: 'mini2-ollama', source_trust: 'device_claimed', warning: 'this name comes from your router' }],
    },
  }))
  assert.equal(row.proposal.host, '100.64.0.37', 'the address is the default host')
  assert.equal(row.proposal.hostAlternatives[0].host, 'mini2.lan')
  assert.equal(row.proposal.hostAlternatives[0].trust, 'device_claimed')
  assert.match(row.proposal.hostAlternatives[0].warning, /router/)
})

check('a name that was not proven is reported with its reason', () => {
  const row = describeFinding(finding({
    name: { name: null, why: 'the name resolves somewhere else', rejected: { proof: 'resolves_elsewhere', escaped: 'mini2.lan' } },
  }))
  assert.equal(row.name.name, null)
  assert.match(row.name.why, /somewhere else/)
  assert.equal(row.name.rejected.proof, 'resolves_elsewhere')
})

check('rows are grouped, and every group that shows has rows', () => {
  const rows = [
    describeFinding(finding()),
    describeFinding(finding({ id: 'f2', classification: { kind: 'other_http', statuses: {} }, proposal: null })),
    describeFinding(finding({ id: 'f3', classification: { kind: 'fabric_proxy' }, proposal: null })),
  ]
  const groups = groupFindings(rows)
  assert.deepEqual(groups.map((group) => group.key), ['addable', 'proxies', 'other'])
  for (const group of groups) assert.ok(group.findings.length > 0)
})

/* ---- the policy and the write ---- */

check('a suggestion always says where its prefix came from', () => {
  const policy = describePolicy({
    enabled: true,
    nodes_file: { path: '/etc/camelid/nodes', sha256: 'abc', labels: ['local'] },
    default_ports: [8181, 11434, 1234],
    allowed_ranges: ['100.64.0.0/10'],
    suggestions: [{ cidr: '100.64.0.0/24', interface: 'en0', address: '100.64.0.20', prefix_source: 'interface_netmask' }],
    transport: { description: 'cleartext restricted to loopback/tunnels', lan_permitted: false, flag_needed: '--allow-cleartext-node-transport' },
    fabric_bearer_configured: true,
    credentials_presented: 'none',
  })
  assert.equal(policy.enabled, true)
  assert.equal(policy.suggestions[0].prefixSource, 'interface_netmask')
  assert.equal(policy.transport.lanPermitted, false)
  assert.equal(policy.bearerConfigured, true)
  // A suggestion with no range is not a suggestion.
  assert.equal(describePolicy({ suggestions: [{ interface: 'en0' }] }).suggestions.length, 0)
})

check('a write is only a write when the proxy said it wrote', () => {
  const joined = describeJoin({
    written: true,
    path: '/etc/camelid/nodes',
    appended: '# joined\nhost-x=ollama://100.64.0.37:11434\n',
    line: 'host-x=ollama://100.64.0.37:11434',
    answered_from: '100.64.0.37:11434',
    sha256_after: 'def456',
    note: 'the proxy re-reads this file within 1 s',
  })
  assert.equal(joined.line, 'host-x=ollama://100.64.0.37:11434')
  assert.match(joined.appended, /^# joined/)
  assert.equal(describeJoin({ written: false }), null)
  assert.equal(describeJoin({}), null)
})

check('every refusal code is distinct and names a next step', () => {
  const codes = [
    'discovery_disabled', 'old_build', 'key_required', 'key_refused', 'origin_not_allowed',
    'loopback_only', 'host_not_loopback', 'transport_refused', 'scope_refused', 'scope_too_large',
    'scan_in_progress', 'file_changed', 'no_longer_answers', 'name_reaches_another_address',
    'duplicate_endpoint', 'duplicate_label', 'invalid_label', 'invalid_host', 'name_not_proven',
    'not_an_engine', 'file_does_not_parse', 'write_failed',
  ]
  const messages = codes.map((code) => problemMessage({ code }))
  assert.equal(new Set(messages).size, codes.length, 'two refusals share a message')
  for (const message of messages) assert.ok(message && message.length > 20, message)
  assert.equal(problemMessage(null), null)
})

/* ---- the transport ---- */

const json = (status, value) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } })
const base = 'http://127.0.0.1:8282'

await checkAsync('a disabled proxy and an old build are different problems', async () => {
  const disabled = await getPolicy({
    base,
    fetchImpl: () => Promise.resolve(json(404, { error: { message: 'not started with --discovery', type: 'fabric_error', code: 'discovery_disabled' } })),
  })
  assert.equal(disabled.problem.code, 'discovery_disabled')

  // The proxy's ordinary unknown-route body carries no code at all.
  const old = await getPolicy({
    base,
    fetchImpl: () => Promise.resolve(json(404, { error: { message: 'this fabric proxy does not serve that route', type: 'fabric_error' } })),
  })
  assert.equal(old.problem.code, 'old_build')
})

await checkAsync('a 401 says whether a key was sent, and the key is sent as a bearer', async () => {
  const seen = []
  const deny = (url, init) => {
    seen.push(init.headers.authorization ?? null)
    return Promise.resolve(json(401, { error: { message: 'unauthorized' } }))
  }
  assert.equal((await getPolicy({ base, fetchImpl: deny })).problem.code, 'key_required')
  assert.equal((await getPolicy({ base, clientKey: '  k-9f3a  ', fetchImpl: deny })).problem.code, 'key_refused')
  assert.deepEqual(seen, [null, 'Bearer k-9f3a'])
})

await checkAsync('each route guard keeps its own code', async () => {
  for (const [code, expected] of [
    ['discovery_loopback_only', 'loopback_only'],
    ['discovery_host_not_loopback', 'host_not_loopback'],
    ['discovery_origin_not_allowed', 'origin_not_allowed'],
    ['transport_refused', 'transport_refused'],
    ['scan_in_progress', 'scan_in_progress'],
  ]) {
    const outcome = await runScan({
      base,
      scope: {},
      fetchImpl: () => Promise.resolve(json(403, { error: { message: 'no', type: 'fabric_error', code } })),
    })
    assert.equal(outcome.problem.code, expected, code)
    assert.equal(outcome.discovery, undefined, 'a refusal is never a set of findings')
  }
})

await checkAsync('a network failure is named, and never an empty result', async () => {
  const offline = await runScan({ base, scope: {}, fetchImpl: () => Promise.reject(new TypeError('Failed to fetch')) })
  assert.equal(offline.problem.code, 'unreachable')
  assert.equal(offline.problem.cause, 'network', 'which from a browser may be a CORS refusal')
  assert.equal(offline.discovery, undefined)

  const garbage = await runScan({ base, scope: {}, fetchImpl: () => Promise.resolve(json(200, { nope: true })) })
  assert.equal(garbage.problem.code, 'malformed')
  assert.equal(garbage.discovery, undefined)
})

await checkAsync('leaving the page is not the proxy failing', async () => {
  const hang = (url, init) => new Promise((_, reject) => {
    init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
  })
  const left = new AbortController()
  const pending = runScan({ base, scope: {}, fetchImpl: hang, signal: left.signal })
  left.abort()
  assert.equal((await pending).problem.code, 'cancelled')
})

await checkAsync('a join reports the refusal the proxy gave it', async () => {
  for (const code of ['file_changed', 'no_longer_answers', 'name_reaches_another_address', 'duplicate_endpoint', 'invalid_host']) {
    const outcome = await join({
      base,
      request: {},
      fetchImpl: () => Promise.resolve(json(409, { error: { message: 'no', type: 'fabric_error', code } })),
    })
    assert.equal(outcome.problem.code, code)
    assert.equal(outcome.joined, undefined, 'a refused join is never a completed write')
  }
})

await checkAsync('a scan sends exactly the scope it was given', async () => {
  let sent = null
  await runScan({
    base,
    scope: { ranges: ['100.64.0.0/24'], hosts: [], ports: [8080], loopback: false, default_ports: true },
    fetchImpl: (url, init) => {
      sent = { url, body: JSON.parse(init.body), method: init.method }
      return Promise.resolve(json(200, discovery()))
    },
  })
  assert.equal(sent.url, `${base}/v1/fabric/discover`)
  assert.equal(sent.method, 'POST')
  assert.deepEqual(sent.body.ranges, ['100.64.0.0/24'])
  assert.equal(sent.body.loopback, false)
})

console.log(`\ndiscovery model smoke: ${checks} checks passed`)
