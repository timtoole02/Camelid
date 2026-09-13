import { appStorage } from './appStorage.js'
import { MAX_MCP_TOOLS } from './mcp.js'

export const MCP_TOOL_SETS_KEY = 'camelid.mcpToolSets'
export const MAX_MCP_TOOL_SETS = 20
const validKey = key => typeof key === 'string' && /^mcp_[a-zA-Z0-9_]+$/.test(key) && key.length <= 160

export function normalizeMcpSelection(keys) {
  return [...new Set((Array.isArray(keys) ? keys : []).filter(validKey))].slice(0, MAX_MCP_TOOLS)
}

export function sameMcpSelection(left, right) {
  return left.length === right.length && left.every(key => right.includes(key))
}

// Group selection is atomic: never silently select only the first 16 tools.
export function toggleMcpGroup(selected, keys) {
  if (keys.some(key => selected.includes(key))) return selected.filter(key => !keys.includes(key))
  const next = [...new Set([...selected, ...keys])]
  if (next.length > MAX_MCP_TOOLS) throw new Error('Choose individual tools: a conversation can use up to ' + MAX_MCP_TOOLS + ' tools.')
  return next
}

export function readMcpToolSets(storage = appStorage) {
  try {
    const raw = storage.getItem(MCP_TOOL_SETS_KEY)
    if (!raw || raw.length > 64000) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    const ids = new Set()
    return parsed.filter(set => {
      if (!set || typeof set.id !== 'string' || !set.id || set.id.length > 80 || ids.has(set.id)
        || typeof set.name !== 'string' || !set.name.trim() || set.name.length > 80
        || !Array.isArray(set.keys) || !set.keys.length || set.keys.length > MAX_MCP_TOOLS
        || set.keys.some(key => !validKey(key)) || new Set(set.keys).size !== set.keys.length) return false
      ids.add(set.id)
      return true
    }).slice(0, MAX_MCP_TOOL_SETS).map(set => ({ id: set.id, name: set.name.trim(), keys: [...set.keys] }))
  } catch { return [] }
}

export function writeMcpToolSets(sets, storage = appStorage) {
  const serialized = JSON.stringify(sets)
  try {
    storage.setItem(MCP_TOOL_SETS_KEY, serialized)
    if (storage.getItem(MCP_TOOL_SETS_KEY) !== serialized) throw new Error('Storage unavailable')
  } catch {
    throw new Error('Could not save tool sets on this device. Check available storage and try again.')
  }
}

export function upsertMcpToolSet(sets, name, keys) {
  const trimmed = name.trim()
  if (!trimmed || trimmed.length > 80) throw new Error('Name your tool set using 1–80 characters.')
  const normalized = normalizeMcpSelection(keys)
  if (!normalized.length || normalized.length !== keys.length) throw new Error('Choose 1–16 valid tools before saving a set.')
  const existing = sets.find(set => set.name.toLocaleLowerCase() === trimmed.toLocaleLowerCase())
  if (!existing && sets.length >= MAX_MCP_TOOL_SETS) throw new Error('You can save up to ' + MAX_MCP_TOOL_SETS + ' tool sets. Remove a set first.')
  const saved = { id: existing?.id || crypto.randomUUID(), name: trimmed, keys: normalized }
  return { saved, sets: [...sets.filter(set => set.id !== saved.id), saved] }
}
