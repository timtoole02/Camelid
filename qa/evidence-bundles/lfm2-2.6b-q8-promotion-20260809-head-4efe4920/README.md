# LFM2.5 2.6B Q8_0 exact-row promotion

This bundle promotes only `LFM2.5-2.6B-Q8_0.gguf`, SHA-256
`36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`
(2,874,779,456 bytes), from `LiquidAI/LFM2.5-2.6B-GGUF`.

On the clean source head recorded in `manifest.json`, the release-mode real-model test
passed both gates. Camelid's tokenizer matched the frozen llama.cpp token ids for the raw
and rendered-chat fixtures, and the resident Metal graph reproduced all four 24-token
greedy continuations (96/96 tokens) from llama.cpp b9632 (`acd79d603`). The three-shape
chat-template fixture and the existing Windows non-streaming serve smoke close the exact
renderer, reasoning split, multi-turn replay, and typed tool-refusal surfaces.

The support boundary remains deliberately narrow: exact bytes, Q8_0, greedy generation,
and short raw/chat smoke only. No streaming, tools, sampling, bounded/model-native context,
CUDA, other LFM2 artifacts, or production-throughput claim is made.
