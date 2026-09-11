/* Regenerated replies kept side by side.

   Regenerate used to be destructive: the reply you had was replaced by the
   one you got, and if the first was better it was gone. Keeping both makes
   Regenerate safe to press, which is the only way it is actually useful.

   SHAPE, and it is the whole design: a variant is the message MINUS its
   identity fields, and the active variant's fields are ALSO mirrored onto the
   message itself. Every existing reader -- the renderer, the exporter, the
   context-budget estimator, the request builder, the telemetry footer -- keeps
   reading `message.content` and `message.usage` and never learns that variants
   exist. Only the code that switches between them looks at the array.

   Defining a variant by subtraction rather than by an explicit field list is
   deliberate: replies grow fields often (receipts, tool calls, research
   sources, continuation counts), and an allow-list would silently stop
   carrying each new one across a switch. */

const IDENTITY_FIELDS = ['id', 'role', 'variants', 'active_variant']

export function snapshotVariant(message) {
  const variant = {}
  for (const [key, value] of Object.entries(message || {})) {
    if (IDENTITY_FIELDS.includes(key)) continue
    variant[key] = value
  }
  return variant
}

/* A message stored before variants existed has none; it IS its own first
   variant. Migrating on read rather than rewriting stored conversations means
   an older transcript needs no upgrade step and a downgrade loses only the
   alternatives, never the reply. */
export function variantsOf(message) {
  const stored = Array.isArray(message?.variants) ? message.variants : null
  if (stored && stored.length) return stored
  return message ? [snapshotVariant(message)] : []
}

export function variantCountOf(message) {
  return variantsOf(message).length
}

export function activeVariantIndexOf(message) {
  const count = variantCountOf(message)
  if (count === 0) return 0
  const stored = Number(message?.active_variant)
  if (!Number.isFinite(stored)) return count - 1
  return Math.min(Math.max(Math.trunc(stored), 0), count - 1)
}

export function hasVariants(message) {
  return variantCountOf(message) > 1
}

/* Only assistant replies branch. A user turn has Edit & resend, which is a
   different operation: it changes the question rather than re-asking it. */
export function canBranchMessage(message) {
  return Boolean(message)
    && message.role === 'assistant'
    && !message.streaming
    && Boolean(String(message.content || '').trim())
}

export function withActiveVariant(message, index) {
  const variants = variantsOf(message)
  if (!variants.length) return message
  const bounded = Math.min(Math.max(Math.trunc(Number(index) || 0), 0), variants.length - 1)
  return {
    id: message.id,
    role: message.role,
    variants,
    active_variant: bounded,
    ...variants[bounded],
  }
}

/* Append a freshly generated reply as the newest variant and select it.
   `next` is an ordinary message object; its identity fields are dropped. */
export function withVariantAppended(message, next) {
  const variants = [...variantsOf(message), snapshotVariant(next)]
  return {
    id: message.id,
    role: message.role,
    variants,
    active_variant: variants.length - 1,
    ...variants[variants.length - 1],
  }
}

/* Discard the active variant and select a neighbour. Returns the message
   unchanged when it is the last one standing: deleting it would leave a reply
   with no content, and the transcript has no way to render that. */
export function withActiveVariantRemoved(message) {
  const variants = variantsOf(message)
  if (variants.length <= 1) return message
  const active = activeVariantIndexOf(message)
  const remaining = variants.filter((_, index) => index !== active)
  const nextActive = Math.min(active, remaining.length - 1)
  return {
    id: message.id,
    role: message.role,
    variants: remaining,
    active_variant: nextActive,
    ...remaining[nextActive],
  }
}

/* Storage normalization: keep `variants`/`active_variant` internally
   consistent, and keep the mirrored top-level fields in step with the active
   variant. A transcript edited by hand, restored from an export, or written by
   an older build must not render one variant's text beside another's token
   counts. */
export function normalizeMessageVariants(message) {
  if (!message || message.role !== 'assistant') return message
  if (!Array.isArray(message.variants) || message.variants.length <= 1) {
    if (message.variants === undefined && message.active_variant === undefined) return message
    // A single or empty array carries no alternatives worth storing.
    const { variants, active_variant: activeVariant, ...rest } = message
    void variants
    void activeVariant
    return rest
  }
  return withActiveVariant(message, activeVariantIndexOf(message))
}
