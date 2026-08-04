import assert from 'node:assert/strict'
import { localModelDeleteRequest, modelDeleteBlockedReason } from '../src/lib/modelDeletion.js'

assert.equal(modelDeleteBlockedReason(), '')
assert.match(modelDeleteBlockedReason({ activeFilename: 'active.gguf' }), /Unload/)
assert.match(modelDeleteBlockedReason({ residentModelsLoaded: true }), /every loaded model/)
assert.match(modelDeleteBlockedReason({ loading: true }), /load/)
assert.match(modelDeleteBlockedReason({ smoking: true }), /check/)
assert.match(
  modelDeleteBlockedReason({ downloads: [{ status: 'downloading' }] }),
  /downloads/,
)
assert.equal(modelDeleteBlockedReason({ downloads: [{ status: 'failed' }] }), '')
assert.deepEqual(
  localModelDeleteRequest({ filename: 'model.gguf', delete_token: 'opaque-token' }),
  { filename: 'model.gguf', delete_token: 'opaque-token' },
)
assert.equal(localModelDeleteRequest({ filename: 'model.gguf' }), null)
assert.equal(localModelDeleteRequest(null), null)

console.log('model deletion smoke: all checks passed')