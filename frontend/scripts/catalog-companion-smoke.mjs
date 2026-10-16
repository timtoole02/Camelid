import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { catalogDownloadSettlement } from '../src/lib/catalogActivation.js'
import { findCompatibilityHint, quantLabelFromGgufFileType } from '../src/lib/capabilities.js'
import { catalogQualification } from '../src/lib/catalogBrowse.js'
import { getChatGateState } from '../src/lib/chatGate.js'
import {
  catalogBundleInstalled,
  catalogDownloadBytes,
  missingCatalogArtifacts,
} from '../src/lib/catalogCompanions.js'
import {
  BONSAI_27B_VISION_PROJECTOR,
  SUPPORTED_MODELS,
} from '../src/lib/supportedModels.js'

const q1 = SUPPORTED_MODELS.find((item) => item.catalog_id === 'bonsai_27b_q1_0')
for (const size of [2, 4, 9, 27]) {
  const model = SUPPORTED_MODELS.find((item) => item.catalog_id === `ternary_bonsai_2_${size}b_ptq1_0`)
  assert.ok(model, `Bonsai 2 ${size}B is offered on the download page`)
  const [projector] = model.companion_artifacts
  assert.equal(projector.filename, `Ternary-Bonsai-2-${size}B-mmproj-Q8_0.gguf`)
  assert.equal(projector.repo_id, model.repo_id)
  assert.equal(catalogDownloadBytes(model), model.size_bytes + projector.size_bytes)
  assert.equal(catalogBundleInstalled(model, new Set([model.filename])), false)
  assert.equal(catalogBundleInstalled(model, new Set([model.filename, projector.filename])), true)
  assert.equal(catalogBundleInstalled(model, new Set([model.filename, BONSAI_27B_VISION_PROJECTOR.filename])), false)
  const capabilities = { model_compatibility: [{ id: model.catalog_id, quantization: 'PTQ1_0', status: 'experimental_implemented' }] }
  assert.equal(catalogQualification(model, capabilities).kind, 'runnable')
  const loaded = { ...model, id: model.catalog_id, runtime_model_name: model.catalog_id, model_path: model.filename, provider_kind: 'local', status: 'ready', loaded_now: true, generation_ready: true, lane_class: 'experimental_implemented' }
  const runtime = { status: 'online', loaded_now: true, generation_ready: true, active_model_id: loaded.id, backend: 'runnable-runtime', vision_ready: true }
  assert.equal(getChatGateState(capabilities, loaded, runtime).experimentalUnlocked, true)
  assert.equal(getChatGateState(capabilities, loaded, { ...runtime, generation_ready: false }).experimentalUnlocked, false)
  for (const quant of ['PTQ1_0', 'file_type 143', undefined]) {
    const hint = findCompatibilityHint(capabilities, { model_path: `C:/models/${model.filename}`, quant })
    assert.equal(hint?.target?.id, model.catalog_id)
    assert.equal(hint?.exact, true)
  }
}
assert.equal(quantLabelFromGgufFileType(143), 'PTQ1_0')
const q2 = SUPPORTED_MODELS.find((item) => item.catalog_id === 'ternary_bonsai_27b_q2_0')
assert.ok(q1 && q2, 'both exact 27B catalog rows must be decorated')
assert.deepEqual(q1.companion_artifacts, [BONSAI_27B_VISION_PROJECTOR])
assert.deepEqual(q2.companion_artifacts, [BONSAI_27B_VISION_PROJECTOR])
assert.equal(
  `https://huggingface.co/${BONSAI_27B_VISION_PROJECTOR.repo_id}/resolve/main/${BONSAI_27B_VISION_PROJECTOR.filename}`,
  'https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-mmproj-Q8_0.gguf',
)

const empty = new Set()
assert.equal(catalogBundleInstalled(q1, empty), false)
assert.deepEqual(
  missingCatalogArtifacts(q1, empty).map((artifact) => artifact.role),
  ['model', 'vision_projector'],
)
assert.equal(
  catalogDownloadBytes(q1, empty),
  q1.size_bytes + BONSAI_27B_VISION_PROJECTOR.size_bytes,
)

const projectorOnly = new Set([BONSAI_27B_VISION_PROJECTOR.filename])
assert.deepEqual(
  missingCatalogArtifacts(q1, projectorOnly).map((artifact) => artifact.role),
  ['model'],
  'an existing shared projector must not be downloaded again',
)

const modelOnly = new Set([q1.filename])
assert.deepEqual(
  missingCatalogArtifacts(q1, modelOnly).map((artifact) => artifact.role),
  ['vision_projector'],
  'an older text-only installation must be repairable with a projector-only acquisition',
)
assert.equal(catalogBundleInstalled(q1, new Set([q1.filename, BONSAI_27B_VISION_PROJECTOR.filename])), true)

assert.equal(
  catalogDownloadSettlement({
    downloading: false,
    failed: true,
    installed: false,
    sawDownload: true,
    startedAt: Date.now(),
  }).action,
  'failed',
  'a failed companion must stop activation immediately',
)

const browseSource = readFileSync(new URL('../src/components/models/CatalogLaneBrowse.jsx', import.meta.url), 'utf8')
assert.doesNotMatch(
  browseSource,
  /Ternary-Bonsai-27B-mmproj-Q8_0/,
  'the Models UI must render backend companion metadata instead of branching on a filename',
)
assert.match(browseSource, /catalogBundleInstalled\(item, localFilenames\)/)
assert.match(browseSource, /enables image prompts only after both files land/)

console.log('catalog companion smoke: ok')
