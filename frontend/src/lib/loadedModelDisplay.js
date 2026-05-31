import { LLAMA32_3B_ACCEPTANCE_TARGET } from './acceptanceTargets.js'
import { ggufFileTypeValueFromLabel, quantLabelFromGgufFileType } from './capabilities.js'

function pathBasename(value) {
  return String(value || '').split(/[\\/]/).filter(Boolean).pop() || ''
}

function normalizeQuantLabel(value) {
  const text = String(value || '').trim()
  const fileType = ggufFileTypeValueFromLabel(text)
  return String(fileType ? quantLabelFromGgufFileType(fileType) : text).trim().toUpperCase().replace(/[^A-Z0-9]+/g, '_').replace(/^_+|_+$/g, '')
}

const LLAMA32_3B_ACCEPTANCE_FILENAME = pathBasename(LLAMA32_3B_ACCEPTANCE_TARGET.source)

function isExactLlama32ThreeBLoadedGguf(modelPath, quantLabel) {
  return pathBasename(modelPath).toLowerCase() === LLAMA32_3B_ACCEPTANCE_FILENAME.toLowerCase()
    && normalizeQuantLabel(quantLabel) === 'Q8_0'
}

export function resolveLoadedModelDisplayName({ fallbackName, modelPath, quantLabel }) {
  if (isExactLlama32ThreeBLoadedGguf(modelPath, quantLabel)) return LLAMA32_3B_ACCEPTANCE_TARGET.name
  return fallbackName
}
