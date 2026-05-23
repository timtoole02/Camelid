#!/usr/bin/env node
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const rootDir = resolve(args.get('root') || 'qa/evidence-bundles')
const outPath = args.get('out') ? resolve(args.get('out')) : null
const strict = args.has('strict')

const textExtensions = new Set(['.json', '.md', '.txt', '.log', '.tsv'])
const textFilenames = new Set(['SHA256SUMS'])
const findings = []

const patterns = [
  {
    id: 'linux_home_path',
    description: 'Linux home path leaked into durable bundle content',
    regex: /(?:file:\/\/)?\/home\/[^/\s"']+\/[^\s"']*/g,
  },
  {
    id: 'mac_home_path',
    description: 'macOS home path leaked into durable bundle content',
    regex: /\/Users\/[^/\n]+\/[^"]*/g,
  },
  {
    id: 'mac_mounted_volume_path',
    description: 'macOS mounted-volume path leaked into durable bundle content',
    regex: /\/Volumes\/[^/\n]+\/[^"]*/g,
  },
  {
    id: 'ipv4_literal',
    description: 'Literal IPv4 address leaked into durable bundle content',
    regex: /\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d?)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b/g,
  },
  {
    id: 'ec2_hostname',
    description: 'Literal EC2 private hostname leaked into durable bundle content',
    regex: /\bip-(?:\d+-){3}\d+\b/g,
  },
  {
    id: 'raw_ssh_command',
    description: 'Raw SSH command leaked into durable bundle content',
    regex: /(^|[\s"'])ssh\s+(?=[^\n]*(?:-i\b|StrictHostKeyChecking|BatchMode|IdentitiesOnly))[^\n"']*/g,
  },
  {
    id: 'raw_ssh_timeout',
    description: 'Raw SSH timeout stderr leaked into durable bundle content',
    regex: /Operation\s+timed\s+out/g,
  },
  {
    id: 'raw_ssh_rc_255',
    description: 'Raw SSH connection status leaked into durable bundle content',
    regex: /\brc=25[5]\b/g,
  },
  {
    id: 'ssh_key_path',
    description: 'SSH private-key path leaked into durable bundle content',
    regex: /[^\s"']*[.]pem\b/g,
  },
]

await walk(rootDir)

const findingsByBundle = new Map()
for (const finding of findings) {
  const bucket = findingsByBundle.get(finding.bundle) || []
  bucket.push(finding)
  findingsByBundle.set(finding.bundle, bucket)
}

const report = {
  schema: 'camelid.evidence_bundle_privacy_audit.v1',
  generated_at: new Date().toISOString(),
  root: relative(process.cwd(), rootDir) || '.',
  strict,
  finding_count: findings.length,
  bundle_count_with_findings: findingsByBundle.size,
  bundles: [...findingsByBundle.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([bundle, bundleFindings]) => ({
      bundle,
      finding_count: bundleFindings.length,
      findings: bundleFindings,
    })),
}

const output = `${JSON.stringify(report, null, 2)}\n`
if (outPath) {
  await mkdir(dirname(outPath), { recursive: true })
  await writeFile(outPath, output, 'utf8')
}
process.stdout.write(output)
if (strict && findings.length > 0) process.exit(1)

async function walk(currentDir) {
  const entries = await readdir(currentDir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = join(currentDir, entry.name)
    if (entry.isDirectory()) {
      await walk(fullPath)
      continue
    }
    const info = await stat(fullPath)
    if (!info.isFile()) continue
    if (!hasTextExtension(entry.name)) continue
    await scanFile(fullPath)
  }
}

async function scanFile(fullPath) {
  const relPath = relative(rootDir, fullPath)
  const bundle = relPath.split('/')[0] || relPath
  const text = await readFile(fullPath, 'utf8')
  const lines = text.split(/\r?\n/)
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    for (const pattern of patterns) {
      pattern.regex.lastIndex = 0
      const matches = [...line.matchAll(pattern.regex)]
      for (const match of matches) {
        const sample = match[0]
        if (pattern.id === 'ipv4_literal' && sample === '127.0.0.1') continue
        if (sample === 'canonical-private-ubuntu-validation-host') continue
        findings.push({
          bundle,
          file: relPath,
          line: index + 1,
          pattern: pattern.id,
          description: pattern.description,
          sample,
        })
      }
    }
  }
}

function hasTextExtension(name) {
  if (textFilenames.has(name)) return true
  const dot = name.lastIndexOf('.')
  return dot >= 0 && textExtensions.has(name.slice(dot).toLowerCase())
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    if (inline !== undefined) {
      parsed.set(key, inline)
      continue
    }
    if (!next || next.startsWith('--')) {
      parsed.set(key, 'true')
      continue
    }
    parsed.set(key, next)
    i += 1
  }
  return parsed
}
