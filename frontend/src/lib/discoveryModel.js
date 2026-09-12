/* The pure layer of the "Find machines" panel.
 *
 * This screen renders answers about machines nobody has identified yet, so the
 * rules here are all versions of one rule: **the page never concludes anything
 * the proxy did not.**
 *
 *   - A classification this build does not recognise is `unrecognized`, and is
 *     never a candidate for adding — not even if the body carried a proposal.
 *     A future proxy kind must not fall through to "looks addable".
 *   - A count the proxy did not send is null, never 0. "Nothing was refused"
 *     and "we were not told how many were refused" are different facts.
 *   - A failed scan is a named problem, never an empty result list. "We could
 *     not look" must not render as "nothing is there".
 *   - Warnings are passed through verbatim. The page never derives one from the
 *     engine name: whether a node would be shown the fabric's bearer depends on
 *     how the *proxy* was started, which only the proxy knows.
 */

/** Classification kinds this build knows how to render. */
const KNOWN_KINDS = [
  'answers_like',
  'ambiguous',
  'fabric_proxy',
  'incomplete',
  'requires_credentials',
  'other_http',
  'not_http',
  'silent_after_connect',
  'tls_not_authenticated',
]

/** The only two kinds a machine can be added from. */
const ADDABLE_KINDS = ['answers_like', 'ambiguous']

/** Where each kind is shown, and what that group is called. */
const GROUPS = [
  { key: 'addable', title: 'Can be added', kinds: ['answers_like'] },
  { key: 'ambiguous', title: 'Answered like more than one engine', kinds: ['ambiguous'] },
  { key: 'in_fabric', title: 'Already in this fabric', kinds: [] },
  { key: 'incomplete', title: 'Could not finish checking', kinds: ['incomplete'] },
  { key: 'proxies', title: 'Fabric proxies', kinds: ['fabric_proxy'] },
  { key: 'credentials', title: 'Wants a credential', kinds: ['requires_credentials'] },
  { key: 'other', title: 'Other HTTP services', kinds: ['other_http'] },
  { key: 'not_http', title: 'Not HTTP', kinds: ['not_http', 'silent_after_connect', 'tls_not_authenticated'] },
  { key: 'unrecognized', title: 'Not recognised by this page', kinds: ['unrecognized'] },
]

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null
}

/** A count we were actually told. Absent stays absent; it is not a zero. */
function countOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function boolOrNull(value) {
  return typeof value === 'boolean' ? value : null
}

function listOf(value, map = (entry) => entry) {
  return Array.isArray(value) ? value.map(map) : []
}

function describeName(raw) {
  if (!isPlainObject(raw)) return { name: null, why: null, trust: null, proof: null, resolved: [], rejected: null }
  return {
    name: stringOrNull(raw.name),
    why: stringOrNull(raw.why),
    trust: stringOrNull(raw.source_trust),
    proof: stringOrNull(raw.proof),
    resolved: listOf(raw.resolved, (entry) => String(entry)),
    rejected: isPlainObject(raw.rejected)
      ? { proof: stringOrNull(raw.rejected.proof), escaped: stringOrNull(raw.rejected.escaped) }
      : null,
  }
}

function describeProposal(raw) {
  if (!isPlainObject(raw)) return null
  const line = stringOrNull(raw.line)
  const label = stringOrNull(raw.label)
  const host = stringOrNull(raw.host)
  const engine = stringOrNull(raw.engine)
  /* Non-empty only when the address matched more than one engine, and then a
     choice is required rather than defaulted. */
  const engineChoices = listOf(raw.engine_choices, (entry) => String(entry))
  /* An unsettled row carries no engine at all — the server does not pick one,
     and neither does this. It is still a proposal, because the choice is the
     thing the confirm panel exists to collect. */
  if (!line || !label || !host || (!engine && engineChoices.length === 0)) return null
  return {
    label,
    engine,
    host,
    port: countOrNull(raw.port),
    line,
    commentPreview: stringOrNull(raw.comment_preview),
    /* Composed by the server, sent back verbatim on a join. The page never
       spells a socket itself: `::1` and `[::1]` are one machine written two
       ways, and only one of them is an address. */
    scannedAddress: stringOrNull(raw.scanned_address),
    engineChoices,
    hostAlternatives: listOf(raw.host_alternatives, (entry) => ({
      host: stringOrNull(entry?.host),
      label: stringOrNull(entry?.label),
      trust: stringOrNull(entry?.source_trust),
      warning: stringOrNull(entry?.warning),
    })).filter((entry) => entry.host),
    /* Verbatim from the server. The page never adds to or filters this list. */
    warnings: listOf(raw.warnings, (entry) => String(entry)),
  }
}

/** What this row could not rule out, in the words the page shows. */
function incompleteSummary(classification) {
  const unanswered = listOf(classification.unanswered, (entry) => String(entry))
  const matched = listOf(classification.matched_so_far, (entry) => String(entry))
  const rivals = unanswered.length > 0 ? unanswered.join(', ') : 'a check'
  const so_far = matched.length > 0
    ? `It answered like ${matched.join(' and ')}, but `
    : ''
  return `${so_far}${rivals} never answered, so what this is has not been established. Scan again.`
}

export function describeFinding(raw) {
  if (!isPlainObject(raw)) return null
  const classification = isPlainObject(raw.classification) ? raw.classification : {}
  const rawKind = stringOrNull(classification.kind)
  /* A kind this build has never heard of is its own state, and is never
     addable. Trusting a proposal on an unknown kind is exactly how a future
     "this is probably an engine" would become a node nobody vouched for. */
  const kind = KNOWN_KINDS.includes(rawKind) ? rawKind : 'unrecognized'
  const proposal = describeProposal(raw.proposal)
  const inFabric = isPlainObject(raw.in_fabric)
    ? {
        label: stringOrNull(raw.in_fabric.label),
        declaredEngine: stringOrNull(raw.in_fabric.declared_engine),
        agrees: boolOrNull(raw.in_fabric.agrees),
        unknown: stringOrNull(raw.in_fabric.unknown),
      }
    : null

  return {
    id: stringOrNull(raw.id) || stringOrNull(raw.address) || 'finding',
    address: stringOrNull(raw.address),
    port: countOrNull(raw.port),
    addresses: listOf(raw.addresses, (entry) => String(entry)),
    viaName: stringOrNull(raw.via_name),
    thisMachine: boolOrNull(raw.this_machine),
    identityBasis: stringOrNull(raw.identity_basis),
    tlsNameUsed: stringOrNull(raw.tls_name_used),
    kind,
    engine: stringOrNull(classification.engine),
    version: stringOrNull(classification.version),
    engines: listOf(raw.engines, (entry) => ({
      engine: stringOrNull(entry?.engine),
      verdict: stringOrNull(entry?.verdict),
      version: stringOrNull(entry?.version),
      detail: stringOrNull(entry?.detail),
    })),
    evidence: listOf(raw.evidence, (entry) => ({
      request: stringOrNull(entry?.request),
      status: countOrNull(entry?.status),
      contentType: stringOrNull(entry?.content_type),
      fact: stringOrNull(entry?.fact),
      matched: listOf(entry?.matched, (engine) => String(engine)),
    })),
    withheldElsewhere: listOf(classification.withheld_elsewhere, (entry) => String(entry)),
    statuses: isPlainObject(classification.statuses) ? classification.statuses : null,
    summary: kind === 'incomplete' ? incompleteSummary(classification) : null,
    name: describeName(raw.name),
    inFabric,
    possiblySameAs: listOf(raw.possibly_same_as, (entry) => String(entry)),
    proposal,
    notProposed: stringOrNull(raw.not_proposed),
    /* Two conditions, both required: a kind we recognise as addable, and a
       proposal the server actually built. */
    canJoin: ADDABLE_KINDS.includes(kind) && proposal !== null,
  }
}

/** Which group a row belongs in. */
export function groupOf(finding) {
  if (finding.inFabric?.label) return 'in_fabric'
  const group = GROUPS.find((candidate) => candidate.kinds.includes(finding.kind))
  return group ? group.key : 'unrecognized'
}

/** The groups that have rows, in the order they are shown. */
export function groupFindings(findings) {
  return GROUPS.map((group) => ({
    ...group,
    findings: findings.filter((finding) => groupOf(finding) === group.key),
  })).filter((group) => group.findings.length > 0)
}

export function describeDiscovery(body) {
  if (!isPlainObject(body) || !Array.isArray(body.findings)) return null
  const notListed = isPlainObject(body.not_listed) ? body.not_listed : {}
  const nodesFile = isPlainObject(body.nodes_file) ? body.nodes_file : null
  return {
    findings: body.findings.map(describeFinding).filter(Boolean),
    planned: countOrNull(body.planned),
    probes: countOrNull(body.probes),
    elapsedMs: countOrNull(body.elapsed_ms),
    notScanned: countOrNull(body.not_scanned),
    notListed: {
      refused: countOrNull(notListed.refused),
      timedOut: countOrNull(notListed.timed_out),
      unreachable: countOrNull(notListed.unreachable),
      other: countOrNull(notListed.other),
    },
    transport: stringOrNull(body.transport),
    credentialsPresented: stringOrNull(body.credentials_presented),
    userAgent: stringOrNull(body.user_agent),
    hint: stringOrNull(body.hint),
    addresses: countOrNull(body.scope?.addresses),
    ports: listOf(body.scope?.ports, (entry) => Number(entry)),
    nodesFile: nodesFile
      ? { path: stringOrNull(nodesFile.path), sha256: stringOrNull(nodesFile.sha256) }
      : null,
    /* Without the hash of the file somebody is agreeing to add a line to,
       there is nothing to write against, so joining is off rather than
       attempted and refused. */
    canJoin: Boolean(nodesFile && stringOrNull(nodesFile.sha256)),
  }
}

export function describePolicy(body) {
  if (!isPlainObject(body)) return null
  const nodesFile = isPlainObject(body.nodes_file) ? body.nodes_file : null
  const transport = isPlainObject(body.transport) ? body.transport : {}
  return {
    enabled: body.enabled === true,
    nodesFile: nodesFile
      ? {
          path: stringOrNull(nodesFile.path),
          sha256: stringOrNull(nodesFile.sha256),
          labels: listOf(nodesFile.labels, (entry) => String(entry)),
        }
      : null,
    defaultPorts: listOf(body.default_ports, (entry) => Number(entry)),
    allowedRanges: listOf(body.allowed_ranges, (entry) => String(entry)),
    suggestions: listOf(body.suggestions, (entry) => ({
      cidr: stringOrNull(entry?.cidr),
      interface: stringOrNull(entry?.interface),
      address: stringOrNull(entry?.address),
      /* Always rendered: two of the three values are guesses, and a suggestion
         that hid which it was would be a claim about the network. */
      prefixSource: stringOrNull(entry?.prefix_source),
    })).filter((entry) => entry.cidr),
    limits: isPlainObject(body.limits) ? body.limits : null,
    transport: {
      description: stringOrNull(transport.description),
      lanPermitted: boolOrNull(transport.lan_permitted),
      flagNeeded: stringOrNull(transport.flag_needed),
    },
    bearerConfigured: boolOrNull(body.fabric_bearer_configured),
    credentialsPresented: stringOrNull(body.credentials_presented),
  }
}

export function describeJoin(body) {
  if (!isPlainObject(body) || body.written !== true) return null
  return {
    path: stringOrNull(body.path),
    appended: typeof body.appended === 'string' ? body.appended : null,
    line: stringOrNull(body.line),
    answeredFrom: stringOrNull(body.answered_from),
    sha256After: stringOrNull(body.sha256_after),
    note: stringOrNull(body.note),
  }
}

/** What each refusal means, and what to do about it. */
const PROBLEMS = {
  discovery_disabled: 'This proxy is running, but it was not started with --discovery, so it will not look for machines.',
  old_build: 'This proxy is running a build from before discovery existed.',
  key_required: 'This proxy requires a client key.',
  key_refused: 'That client key was not accepted.',
  origin_not_allowed: "This proxy does not allow this page's origin.",
  loopback_only: "Discovery runs only for a caller on the proxy's own machine. Open this page there.",
  host_not_loopback: 'That request did not address the proxy by a loopback name.',
  transport_refused: 'Reaching another machine in the clear needs an explicit acknowledgement.',
  scope_refused: 'That scan was refused before anything was sent.',
  scope_too_large: 'That range is larger than one scan covers. It is refused rather than cut short.',
  scan_in_progress: 'This proxy is already looking. Wait for that scan to finish.',
  file_changed: 'The nodes file changed since it was read. Scan again, then add it.',
  no_longer_answers: 'That machine no longer answers like the engine it was going to be added as. Nothing was written.',
  name_reaches_another_address: 'That name now reaches a different machine than the one that was scanned. Nothing was written.',
  invalid_scanned_address: 'That request did not say, in a form this proxy can read, which machine the scan reached. Nothing was written; scan again.',
  duplicate_endpoint: 'That machine is already in this fabric under another label. Nothing was written.',
  duplicate_label: 'That label is already used in the nodes file.',
  invalid_label: 'That label is not one this build will write into a nodes file.',
  invalid_host: 'That host is not one this build will write into a nodes file.',
  name_not_proven: 'That name could not be shown to reach that machine from here.',
  not_an_engine: 'That is not an engine this fabric can read.',
  file_does_not_parse: 'The nodes file does not parse, so a line added to it would never take effect.',
  write_failed: 'The nodes file could not be written.',
  unreachable: 'Nothing answered at that address.',
  malformed: 'That answer was not something this page could read.',
  cancelled: 'That scan was stopped.',
  timeout: 'That scan did not finish in time.',
}

/** One sentence for a problem, always naming a next step. */
export function problemMessage(problem) {
  if (!problem) return null
  const known = PROBLEMS[problem.code]
  if (known) return problem.detail ? `${known} ${problem.detail}` : known
  return problem.detail || 'This proxy refused that request.'
}

export { KNOWN_KINDS, ADDABLE_KINDS, GROUPS }
