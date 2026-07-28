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
the engine's existing UI; if no eligible model is loaded, that UI remains the authority and shows
its normal model-required state. Desktop does not claim a ready model or manufacture a model error
on the splash.

## Requirements

- **Windows:** Windows 10/11 with the **WebView2 runtime** (preinstalled on current Windows 10/11; the
  Tauri bundle ships the bootstrapper otherwise).
- **macOS:** Apple Silicon running macOS 12 or newer. The current macOS desktop bundle is
  ad-hoc signed for local/developer distribution and is not notarized.
- A bundled platform engine. The portable ZIP, Windows installer, and macOS app bundle
  include it automatically.

## macOS command-line install

From the repository root on an Apple Silicon Mac:

```sh
./scripts/install-macos-desktop.sh
```

This builds the frontend, release engine, app bundle, and DMG; closes an existing Camelid Desktop
instance cleanly; installs the new app at `/Applications/Camelid Desktop.app`; verifies its
ad-hoc signature; and launches it. The script uses `sudo` only when `/Applications` is not writable.
The model directory under the user's Application Support folder is not replaced.

Prerequisites are macOS 12 or newer, the Xcode Command Line Tools, Rust, and Node.js 22 with npm.
The app is not notarized yet, so this path is intended for local testing and developer
distribution.

## Building (developers)

```sh
# From the workspace root. Build the debug server sidecar, then build and run
# the desktop app. Both executables land in target/debug/:
cargo build --locked --bin camelid
cargo build -p camelid-desktop
cargo run -p camelid-desktop
```

For packaging, build the release sidecar explicitly:

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

For a bundled installer + portable zip, see the additive `desktop-windows` job in
`../.github/workflows/release.yml`.

### Building the macOS app and DMG

On an Apple Silicon Mac:

```sh
./scripts/build-macos-desktop.sh
```

The script builds the real frontend, the release Metal-enabled `camelid` sidecar, and the
Tauri `.app` and `.dmg`. It uses an ad-hoc signature (`-`), not a Developer ID signature,
and performs no notarization. macOS may therefore require the user to approve the app in
**System Settings → Privacy & Security** after downloading it.

Downloaded models are stored under the app's per-user Application Support directory rather
than inside the app bundle.

## Scope notes (intentionally deferred)

v1 deliberately keeps the native shell thin and ships the engine's real UI as-is:

- **No fabricated metrics, by construction.** The splash shows only real lifecycle status;
  all chat metrics (e.g. tokens/sec) come from the embedded UI rendering the engine's real
  generation/telemetry events. Nothing in this crate computes or smooths a metric.
- **Native tray / native GGUF file-picker are deferred.** Both would require granting Tauri
  IPC to the loopback-origin page, widening the attack surface this design intentionally
  avoids — and the embedded UI already loads local/catalog models via the existing
  `/api/models/load` path, so a native picker adds no capability. They can be added later
  behind a scoped capability if desired.
