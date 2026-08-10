# Catalog expansion: Mistral Nemo, Qwen3 14B, DeepSeek R1 0528 Qwen3 8B

This bundle records the exact public artifacts inspected for the August 2026 catalog
expansion. All three GGUFs parse, admit, and execute bounded raw generation in Camelid.

Only Qwen3 14B reproduced the same eight greedy tokens on the Metal and deterministic CPU
K-quant lanes. That is useful bring-up evidence, but it is not the repository's full
support-promotion bar: external llama.cpp prompt/generated-token parity, the ChatML API
surface, and frontend smoke remain outstanding.

Mistral Nemo passed the committed 39-case Tekken token oracle and whole-file admission
test, then generated successfully on both Camelid lanes. Its five-token common prefix
diverged at generated index 5, and the locally installed llama.cpp binary crashed while
loading the file on both attempted backends. DeepSeek likewise executes, but diverged at
index 2 and carries the original DeepSeek R1 marker template rather than Qwen ChatML.

Accordingly, these three rows are downloadable catalog additions with explicit active-
validation contracts. None inherits a supported badge from its architecture, quant, or a
smaller sibling. The exact blockers are machine-readable in `manifest.json`.
