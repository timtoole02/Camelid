/**
 * Session-scoped context policies offered by Code mode.
 *
 * The values are part of the workspace-session wire contract. Keep the labels
 * here as the single source of truth for both the composer and inspector.
 */
export const CONTEXT_PROFILES = Object.freeze([
  Object.freeze({
    value: 'auto',
    label: 'Auto',
    detail: 'Use the best validated context window for the loaded model.',
  }),
  Object.freeze({
    value: 'q8_16k',
    label: 'Q8 16K',
    detail: 'Use the 16K paged window available only to Qwen3 4B Q8_0.',
  }),
  Object.freeze({
    value: 'standard',
    label: 'Q4 8K',
    detail: 'Use the standard 8K window for Q4 and other supported models.',
  }),
])

export const DEFAULT_CONTEXT_PROFILE = CONTEXT_PROFILES[0].value

export function contextProfileMeta(value) {
  return CONTEXT_PROFILES.find((profile) => profile.value === value) || CONTEXT_PROFILES[0]
}
