# MiniMax H3 local video (experimental)

Camelid can generate MiniMax H3 video through a capability-checked
[`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) `sd-cli`
backend. This is an experimental backend bridge, not a claim that Camelid's
autoregressive Rust engine natively executes the H3 diffusion graph.

## What the lane supports

- `fl2va`: text-to-video, first-frame image-to-video, and first/last-frame video.
- `ref2va`: image, video-frame-directory, and WAV reference conditioning.
- Native 24 fps WebM or AVI output; WebM is recommended for H3's generated stereo audio.
- A pinned, SHA-256-verified starter bundle for either checkpoint.
- A dry-run plan and doctor command that do not load the model.

The **Video Studio** page uses a loopback-only asynchronous HTTP job queue so a
long render never blocks Camelid's text decode worker. Chat model loading stays
separate: H3's GGUFs are diffusion/text-encoder components, not chat-completion
models.

## Requirements

The default `fl2va` bundle is 26,978,253,672 bytes (about 25.13 GiB):

| Component | File | Bytes |
| --- | --- | ---: |
| Diffusion transformer | `minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf` | 8,063,029,344 |
| H3 text encoder | `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` | 13,102,161,024 |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 |

The `ref2va` starter bundle is about 23.84 GiB because its Q2_K diffusion
transformer is smaller. Runtime memory depends heavily on resolution, frame count,
backend, and offload. Camelid defaults the text encoder to CPU and enables CPU
offload; this lowers the GPU-memory requirement but needs substantial system RAM
and fast local storage.

MiniMax H3 is governed by the
[MiniMax-H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
including its applicable-territory terms. Camelid requires an explicit license
acknowledgement before downloading and does not decide whether the license applies
to a particular user or location.

## Install an H3-capable backend

Use a current `sd-cli` build from `stable-diffusion.cpp`. The bridge was checked
against source revision `c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12` and performs a
runtime capability check for the required CLI flags. Put `sd-cli` beside the
Camelid executable, on `PATH`, in `CAMELID_SD_CLI`, in
`/Volumes/<drive>/Camelid/bin/sd-cli` on macOS, or pass `--sd-cli <path>`.

On Windows, use `sd-cli.exe` and keep any DLLs from the same upstream build
beside it. Camelid launches both the capability check and video render without
opening a console window. Video Studio defaults to
`<serve --models-dir>\minimax-h3`, rather than the process working directory,
so an installed desktop app uses its resolved sidecar model store reliably.
The **Choose** button can point it at `D:\Camelid\models\minimax-h3` (or another
local/removable drive) without changing the text-model store.

Follow the upstream
[build instructions](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md) for
the desired Metal, CUDA, Vulkan, or CPU backend. Camelid does not download or
execute build scripts for this third-party binary.

## Download and verify the model bundle

Read the H3 license first, then download the text/first/last-frame bundle:

```bash
camelid video pull --variant fl2va --accept-license
camelid video doctor --variant fl2va --verify-sha256
```

Use `--no-audio` with both commands for a silent-video bundle. To use reference
media instead:

```bash
camelid video pull --variant ref2va --accept-license
camelid video doctor --variant ref2va --verify-sha256
```

Downloads are resumable. Camelid writes `.part` files, checks exact byte counts
and pinned SHA-256 digests, and only then promotes each artifact to its final name.

On macOS, Video Studio automatically discovers a bundle at
`/Volumes/<drive>/Camelid/models/minimax-h3`. Its readiness card reports partial
download bytes while `.part` files are still being filled and writes completed
clips to the sibling `Camelid/outputs` folder. In the desktop app, use **Choose**
beside the bundle path the first time macOS asks Camelid to access a removable
volume; selecting the `minimax-h3` folder grants that access without changing the
separate text-model storage preference.

Desktop release builders can bundle the small backend executable inside the
signed app while leaving the model files external:

```bash
CAMELID_DESKTOP_SD_CLI=/path/to/sd-cli ./scripts/build-macos-desktop.sh
```

For a Windows desktop bundle, stage a checked upstream build and its sibling
runtime DLLs before invoking Tauri's existing resource overlay:

```powershell
.\scripts\stage-windows-h3-backend.ps1 C:\path\to\stable-diffusion.cpp\bin\sd-cli.exe
npm.cmd --prefix frontend run build
cargo build --release --locked --bin camelid
Copy-Item target\release\camelid.exe camelid-desktop\sidecar\camelid.exe -Force
Push-Location camelid-desktop
npx.cmd --yes '@tauri-apps/cli@^2' build --config tauri.bundle.conf.json
Pop-Location
```

`tauri.bundle.conf.json` already includes `sidecar/*`, so the NSIS resource
layout places `sd-cli.exe` beside the bundled `camelid.exe`; the portable layout
uses the same sibling discovery rule. The 24–27 GB model bundle remains outside
the installer.

## Use Video Studio

Start the local server and open **Video Studio** from the Workspace section of
the sidebar. The page provides:

- FL2VA text-to-video and optional first/last-frame paths.
- REF2VA image, frame-directory, and audio reference paths when its separate
  diffusion model is installed.
- Resolution, frame, step, seed, and generated-audio controls.
- A serialized asynchronous job list with cancellation, backend logs, and
  seekable in-browser WebM playback.

The page polls bundle/backend readiness, so it can remain open while a resumable
`camelid video pull` runs in another terminal.

## Local video API

Video Studio uses these local endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/video/capabilities` | Bundle bytes, artifact/backend readiness, and discovered paths |
| `GET/POST /api/video/jobs` | List or queue jobs |
| `GET /api/video/jobs/:id` | Read job state |
| `POST /api/video/jobs/:id/cancel` | Cancel a queued/running job |
| `GET /api/video/jobs/:id/content` | Range-capable WebM playback |
| `GET /api/video/jobs/:id/log` | Backend log |

These filesystem-bearing routes fail closed when Camelid is not listening on a
loopback address. They are a Camelid-local control surface, not an OpenAI
`/v1/videos` compatibility claim.

## Generate a video

Text-to-video with native stereo audio:

```bash
camelid video generate \
  --prompt "a red panda stepping along a mossy log in a misty forest, cinematic" \
  --output red-panda.webm \
  --width 640 --height 384 --frames 25 --steps 4 --seed 11
```

Image-to-video adds an initial image; add `--end-image` for first/last-frame
conditioning:

```bash
camelid video generate \
  --prompt "the camera slowly circles the subject as snow begins to fall" \
  --init-image start.png \
  --end-image finish.png \
  --output snow.webm
```

Reference-conditioned generation uses the other checkpoint:

```bash
camelid video generate \
  --variant ref2va \
  --prompt "Use <Picture 1> as the main character in a cinematic tracking shot" \
  --reference-image subject.png \
  --output subject.webm
```

`--reference-image`, `--reference-video`, and `--reference-audio` may be repeated.
The upstream backend currently represents a reference video as a directory of
lexicographically sorted frames; reference audio must be a supported WAV file.

Inspect the exact process invocation without running it:

```bash
camelid video generate --prompt "test" --dry-run
```

H3 fixes fps at 24 and uses a `17k + 5` temporal frame grid. Camelid reports both
the requested and effective frame counts in its JSON plan. It pins `cfg-scale=1.0`
because H3 is distilled and classifier-free-guidance values above 1.0 are invalid.

## Current boundary

This lane is groundwork/experimental until a reproducible video-and-audio receipt
is captured on named hardware against the pinned artifacts and backend revision.
The UI and local job API are orchestration surfaces, not model-support evidence.
There is no claim yet for output parity, quality, throughput, portable memory fit,
other H3 quants, `/v1/videos` compatibility, or a Camelid-native H3 runtime.
