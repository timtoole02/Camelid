# Camelid Desktop (add-on, Windows and macOS)

**Camelid Desktop is an additive native app.** It gives users a desktop chat
experience with no web browser, by embedding the **same `camelid` engine** that ships as the
server binary and hosting the existing web UI in a native WebView2 window on Windows or
WebKit window on macOS via [Tauri v2](https://v2.tauri.app/).

It is an add-on only. It does **not** modify, gate, or relax any existing support claim,
parity contract, or the `camelid` server binary. **The web path remains the canonical path.**

## What it inherits (and does not change)

- **Identical engine.** The desktop process spawns the shipped `camelid serve` as a
  loopback-only sidecar (`127.0.0.1:<ephemeral>`). It does not reimplement tokenization,
  decoding, GGUF parsing, or sampling. Generation is byte-identical to `camelid serve`.
- **Identical support contract.** The window points at the engine's already-embedded UI, so
  model availability and the **runtime-ready + exact-supported-row** chat gate come from the
  same authority as the web UI (`/api/capabilities`, the compatibility ledger). A model the
  existing gate refuses is refused here too — the gate is reused, not re-derived.
- **Identical GPU acceleration.** Because the sidecar *is* the shipped `camelid` engine, it
  uses the engine's GPU path unchanged: on a machine with an NVIDIA GPU it auto-engages the
  bundled CUDA runtime (the same Windows CUDA-resident decode path the engine validates — the
  Qwen3 Q8_0 rows), and falls back to the CPU otherwise. The Gemma 4 E4B-It Q8_0 CUDA lane is
  opt-in behind `CAMELID_GEMMA4_CUDA=1`, not auto-engaged. The
  app adds no GPU code and makes no separate performance claim; the authoritative supported-row
  and GPU list is the engine's [`README.md`](../README.md) (*Windows CUDA*).
- **No fabricated metrics.** Any tokens/sec or status readout is sourced from the same real
  generation events the server emits (the SSE `camelid.decode_tps` field). If a metric is
  unavailable it is shown as unavailable, never as a placeholder.

This app makes **no broader claims** than the engine it embeds about supported models,
performance, or compatibility.

## Architecture (sidecar; see `../DECISIONS.md` D11)

```
camelid-desktop ──spawns──▶ camelid serve --addr 127.0.0.1:<ephemeral> --no-open
        │                                  │  (loopback only)
        │  poll /v1/health (backoff)       │
        ▼                                  ▼
   Native webview   ──navigates to──▶  http://127.0.0.1:<ephemeral>/
   (splash first)                      (UI + API are same-origin; the engine serves the
                                        embedded React UI from its `*` fallback route)
```

On window close the sidecar is terminated cleanly. On Windows, a **job object** with
`KILL_ON_JOB_CLOSE` also prevents a desktop crash from orphaning a `camelid` process.

## Windows in-place upgrades

The NSIS installer overwrites the files it ships but, like any overwrite-only installer, cannot
by itself remove a file an **older** version installed that the current one no longer ships.
`windows/installer-hooks.nsh` supplies an `NSIS_HOOK_PREINSTALL` that deletes the NVRTC
redistributables (`nvrtc64_*.dll`, `nvrtc-builtins64_*.dll`) from `sidecar\` before the file
copy, so every upgrade re-lays exactly the set that version ships. That covers both the
`nvrtc64_120_0.alt.dll` orphan left by pre-filter releases and any future CUDA version bump,
which renames these DLLs and would otherwise strand the previous ones.

The hook is deliberately narrow, and widening it to clear `sidecar\` wholesale would **destroy
user data**: the desktop's model store is the `models\` folder beside the engine binary
(`sidecar_models_dir` in `src/engine.rs`), i.e. `sidecar\models\`, holding multi-GB downloaded
GGUF weights. Only files the packaging scripts stage may be removed there.

## Windows code signing

Release artifacts are signed with Azure Artifact Signing. The subtlety is that
`camelid-desktop.exe` exists as **two distinct copies**, and only one of them is reachable
from an ordinary post-build signing pass:

| copy | signed by | when |
| --- | --- | --- |
| `sidecar\camelid.exe` | signing action, folder pass | before bundling; copied verbatim as a resource |
| the exe **inside** the installer | `bundle.windows.signCommand` | during bundling |
| portable exe + NSIS installer | signing action, folder pass | after bundling |

The middle row needs its own mechanism because Tauri patches the binary with the bundle type,
rewriting `__TAURI_BUNDLE_TYPE_VAR_UNK` to `..._NSS` so the installed app knows how it was
installed, and signs only afterwards. A signature applied before `tauri build` is invalidated by
that rewrite, and one applied afterwards cannot reach a binary already sealed inside the
installer. `windows/sign-artifact-signing.ps1` runs in the only window where the bytes are final.

Three shipped releases got some part of this wrong, which is why the guards below exist:

| release | what users got |
| --- | --- |
| v0.4.6 | installed exe `NotSigned` — signed only after bundling |
| v0.4.7 | installed exe `HashMismatch` — signed only before bundling; a broken signature is worse than none |
| v0.4.8 | **no Windows installer at all** — `signCommand` used a project-relative script path |

**`signCommand` paths must be absolute, and the release workflow generates them.** Tauri invokes
the hook **seven times** per bundle — the app binary, five NSIS plugin DLLs, and the uninstaller
staged as a `%TEMP%\nst*.tmp` — and the working directory is *not* constant: six run from the
Tauri project directory, but the uninstaller call runs from `target\release\nsis\x64`. A
project-relative path resolves for six of seven and fails the build on the last. The workflow
therefore writes `tauri.signing.conf.json` at release time with absolute paths (and re-states
`installerHooks`, read from `tauri.conf.json`, so the overlay cannot drop it), and passes it as an
extra `--config`. It is deliberately absent from the committed config so a developer's
`tauri build` needs no signing tooling; the script also no-ops when its environment is unset.

Two independent guards close the loop: the release workflow installs the built installer on the
runner and asserts the unpacked binary verifies, and `verify-release-assets` demotes any release
whose asset set is incomplete so `releases/latest` can never point at one.

## Startup failures

The splash is fail-closed: it stays visible until the sidecar returns `200` from
`/v1/health`. It polls every 350 ms for up to 40 seconds. Native startup state is retained and
replayed after the splash listener registers, so a fast failure cannot be lost before the page
loads. A failure shows an actionable title and next step first, followed by the engine's actual
error and captured stderr under **Technical details**; it never navigates to a fake-ready UI.

| Splash error | Meaning | Next step |
| --- | --- | --- |
| **Camelid engine is missing** | The platform engine was not found beside the desktop executable, in the bundled sidecar resources, or on `PATH`. | Reinstall Camelid Desktop or restore its bundled Camelid engine, then retry. |
| **Sidecar port unavailable** | The engine reported that it could not bind Camelid's selected ephemeral loopback port. This is not a fixed `8181` port conflict. | Close the conflicting local process and retry. |
| **Engine startup timed out** | The sidecar did not pass the 40-second `/v1/health` gate. | Retry, then use the visible technical details to diagnose a persistent failure. |
| **Engine startup failed** | The sidecar exited before it became healthy for another reason. | Review the visible technical details and retry. |

Model readiness is separate from sidecar startup. Once `/v1/health` passes, Desktop navigates to
the engine's existing UI. If local GGUFs exist, the sidecar loads the saved default from the
configured models directory; without a saved preference, it loads the first local GGUF. The Models
page labels that row **Starts automatically** and offers **Make default** on other loadable rows.
If no eligible model exists, the UI remains the authority and shows its normal model-required
state. Desktop does not claim a ready model or manufacture a model error on the splash.

## Requirements

- **Windows:** Windows 10/11 with the **WebView2 runtime** (preinstalled on current Windows 10/11; the
  Tauri bundle ships the bootstrapper otherwise).
- **macOS:** Apple Silicon running macOS 12 or newer. The current macOS desktop bundle is
  ad-hoc signed and not notarized; it ships prebuilt as the release DMG (see
  [macOS install](#macos-install)).
- A bundled platform engine. The portable ZIP, Windows installer, and macOS app bundle
  include it automatically.

## macOS install

**Prebuilt (recommended).** On an Apple Silicon Mac, one command downloads the release DMG
(`camelid-desktop-macos-arm64.dmg`, built by the additive `desktop-macos` release job), verifies
its published SHA-256, installs `/Applications/Camelid Desktop.app`, and launches it — no
toolchain required:

```sh
curl -fsSL https://raw.githubusercontent.com/timtoole02/Camelid/main/scripts/get-desktop-macos.sh | bash
```

The app is ad-hoc signed and not notarized, so a browser-downloaded DMG is quarantined and
Gatekeeper blocks the first launch (approve under **System Settings → Privacy & Security** — on
macOS 12, **System Preferences → Security & Privacy → General** — or
`xattr -cr` the installed app). The script path avoids that: command-line downloads carry no
quarantine attribute. Pass a tag to pin a version (`... | bash -s -- v0.4.5`).

**From source.** From the repository root on an Apple Silicon Mac:

```sh
./scripts/install-macos-desktop.sh
```

This builds the frontend, release engine, app bundle, and DMG; closes an existing Camelid Desktop
instance cleanly; installs the new app at `/Applications/Camelid Desktop.app`; verifies its
ad-hoc signature; and launches it. The script uses `sudo` only when `/Applications` is not writable.
Prerequisites are macOS 12 or newer, the Xcode Command Line Tools, Rust, and Node.js 22 with npm.

Neither path replaces the default Application Support model directory or a custom model
directory: installed models survive updates and reinstalls.

## Building (developers)

```sh
# From the workspace root. Keep the UI shell debuggable, but always run the
# optimized inference engine; an unoptimized CPU fallback can take minutes per
# token on agent prompts.
cargo build --release --locked --bin camelid
$env:CAMELID_ENGINE_PATH = (Resolve-Path target/release/camelid.exe)
cargo run -p camelid-desktop
```

On macOS/Linux, set the same override with
`CAMELID_ENGINE_PATH="$(pwd)/target/release/camelid"`. The override must be an
absolute existing path and takes precedence over an engine beside the desktop
executable. For packaging, build the release sidecar explicitly:

```sh
cargo build --release --locked --bin camelid
```

The debug `camelid.exe` is supported for local desktop development. On Windows, the server's
link configuration reserves sufficient stack for the large CLI parser; CI exercises its
`--version` startup path directly. When the sidecar fails to come up, the desktop surfaces the
real error and engine stderr on the splash rather than faking a ready state.

The server build is unaffected by this crate: `cargo build --release --locked --bin camelid`
does not pull `camelid-desktop` into its graph (workspace `resolver = "2"`,
`default-members = ["."]`).

For the shipped bundles — Windows installer + portable zip, and the macOS DMG — see the
additive `desktop-windows` and `desktop-macos` jobs in `../.github/workflows/release.yml`.

### Building the macOS app and DMG

On an Apple Silicon Mac:

```sh
./scripts/build-macos-desktop.sh
```

The script builds the real frontend, the release Metal-enabled `camelid` sidecar, and the
Tauri `.app` and `.dmg`. It uses an ad-hoc signature (`-`), not a Developer ID signature,
and performs no notarization. macOS may therefore require the user to approve the app in
**System Settings → Privacy & Security** (on macOS 12, **System Preferences → Security &
Privacy → General**) after downloading it.

Downloaded models are stored under the app's per-user Application Support directory rather
than inside the app bundle by default. The **Downloaded models** tab can save a different local
folder for the next launch. Existing GGUFs are never moved automatically, so changing the folder
does not risk an implicit multi-gigabyte copy or deletion.

## Scope notes (intentionally deferred)

v1 deliberately keeps the native shell thin and ships the engine's real UI as-is:

- **No fabricated metrics, by construction.** The splash shows only real lifecycle status;
  all chat metrics (e.g. tokens/sec) come from the embedded UI rendering the engine's real
  generation/telemetry events. Nothing in this crate computes or smooths a metric.
- **Native tray / arbitrary GGUF file-picker are deferred.** The loopback-origin page receives
  only a scoped native folder chooser for configuring model storage; it does not receive broad
  filesystem access. Local/catalog model loading still goes through the engine's existing API.
