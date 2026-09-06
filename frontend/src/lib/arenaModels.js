import { isGenerationCapableModel } from './modelCapabilities.js'
import { canLoadIntoRuntime, isExternalModel, modelRuntimeIdMatches } from './modelState.js'

/**
 * The Arena can only compare local GGUFs that Camelid can intentionally load.
 * Catalog-only, hosted, embedding-only, and companion records must never become
 * implicit on-demand model loads just because they sort first in the dashboard.
 */
export function arenaModelChoices(models = [], runtime = null) {
  return models.filter((model) => (
    !isExternalModel(model)
    && isGenerationCapableModel(model, runtime)
    && canLoadIntoRuntime(model)
  ))
}

export function arenaDefaultModelA(models = [], runtime = null) {
  const choices = arenaModelChoices(models, runtime)
  return choices.find((model) => modelRuntimeIdMatches(model, runtime))?.id
    || choices[0]?.id
    || ''
}

export function arenaSelectionsAreReady(modelA, modelB) {
  return Boolean(modelA && modelB && modelA !== modelB)
}

export function arenaModelIsAlreadyReady(model, health) {
  if (!model || !health?.generation_ready) return false
  const filename = String(model.model_path || '').split(/[\\/]/).filter(Boolean).pop() || ''
  return [model.id, model.runtime_model_name, filename]
    .filter(Boolean)
    .includes(health.active_model_id || '')
}

/** Run the two sides in a defined order so a one-active-model runtime can swap
 * safely instead of racing two expensive on-demand loads. */
export async function runArenaSequentially({ modelA, modelB, runModel, signal }) {
  const resultA = await runModel(modelA, 'a')
  if (signal?.aborted) return { resultA, resultB: null }
  const resultB = await runModel(modelB, 'b')
  return { resultA, resultB }
}
