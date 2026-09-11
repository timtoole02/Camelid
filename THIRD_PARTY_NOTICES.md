# Third-Party Notices

Last updated: 2026-09-10

## Scope note

This file records the third-party notices Camelid currently needs to keep visible in source and release distributions.

It is intentionally explicit rather than exhaustive today. Camelid should expand this notice set as the project adds more redistributed source, bundled binaries, shipped fixtures, or material third-party runtime and build dependencies. A broader dependency inventory may later live in a separate generated notice or SBOM workflow, but that future inventory does not replace the current obligation to preserve visible credit wherever Camelid's public evidence trail depends on external work.

Practical rule: documentation cleanup, branding polish, or repository renaming must not remove third-party acknowledgement when public claims, parity evidence, tokenizer references, or reference benchmarks still depend on that external work.

## Current credited reference work

Camelid is an independent Rust-native local inference project. Its implementation is original, but parts of its public credibility story — especially compatibility comparisons, tokenizer references, parity harnesses, and benchmark evidence — rely on important open-source reference work. Those references travel through the README, compatibility matrix, status ledger, and release-note claims whenever Camelid cites parity-backed evidence.

### llama.cpp / ggml

Camelid uses llama.cpp as a compatibility and parity reference for GGUF model behavior, tokenizer fixtures, and local inference validation. This is not incidental credit: those references remain part of Camelid's documented evidence trail and should stay explicitly credited wherever that evidence is summarized or redistributed.

- Project: <https://github.com/ggml-org/llama.cpp>
- License: MIT

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NVIDIA CUDA NVRTC runtime (prebuilt Windows GPU release)

Camelid's prebuilt Windows release bundles the NVIDIA CUDA NVRTC runtime-compilation
redistributable libraries (`nvrtc64_*.dll` and `nvrtc-builtins64_*.dll`) alongside the
executable, so the GPU path works on a host that has only the NVIDIA display driver
installed (no CUDA Toolkit). These libraries are NVIDIA's own work, are shipped
unmodified, and are not part of Camelid. They are redistributed under the NVIDIA CUDA
Toolkit End User License Agreement, which lists NVRTC among its redistributable
components. The CUDA driver itself (`nvcuda.dll`) is **not** redistributed — it is
provided by the user's installed NVIDIA GPU driver.

This notice and the bundled libraries apply only to the prebuilt Windows download;
building Camelid from source pulls no NVIDIA libraries (the CUDA runtime is loaded
dynamically at runtime when present).

- Product: NVIDIA CUDA NVRTC (runtime compilation library), a component of the NVIDIA CUDA Toolkit
- Source: <https://developer.nvidia.com/cuda-toolkit>
- License: NVIDIA CUDA Toolkit EULA — redistributable components (see Attachment A)
- License text: <https://docs.nvidia.com/cuda/eula/index.html>

### tokio-util (direct Rust dependency)

Camelid's API server uses `tokio-util`'s `CancellationToken` for the generation
cancellation contract: every decode observes a cooperative stop signal so a dropped
request (client disconnect, timeout) can never orphan compute against the shared
GPU-resident decode state (see `docs/recon/ENGINE_INVERSION_CONDUCTOR.md`). The crate
was already present in the dependency graph transitively (via the axum/tokio
ecosystem); this records its promotion to a direct, load-bearing dependency.

- Project: <https://github.com/tokio-rs/tokio>
- License: MIT

### Bundled WebUI dependencies

The WebUI is compiled to static assets and embedded in the engine binary by `rust-embed`, so every runtime npm dependency below is **redistributed inside every Camelid release**, not merely used at build time. That is what puts them in this file rather than leaving them to the lockfile.

KaTeX is loaded on demand rather than in the main bundle, but on-demand loading is a size decision, not a distribution one: its chunk and font files still ship in the binary.

| Dependency | License | Used for |
| --- | --- | --- |
| [React](https://github.com/facebook/react) / React DOM | MIT | the WebUI itself |
| [KaTeX](https://github.com/KaTeX/KaTeX) | MIT | typesetting TeX in assistant replies (JS plus its bundled fonts) |
| [Inter](https://github.com/rsms/inter), [Space Grotesk](https://github.com/floriankarsten/space-grotesk), [IBM Plex Mono](https://github.com/IBM/plex) (via Fontsource) | OFL-1.1 | the WebUI's typefaces |

The OFL-1.1 fonts are redistributed unmodified under their reserved names, which that licence permits; Camelid does not sell the fonts on their own.

## Maintenance note

Keep this file in sync with any third-party source, binary, fixture, or reference tooling Camelid redistributes or materially depends on for public evidence. Documentation polish, branding cleanup, or repository renaming work must not remove these credits while the underlying technical reliance still exists.
