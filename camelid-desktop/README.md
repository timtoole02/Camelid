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
        │                                  │  [--exit-when-stdin-closes]  (loopback only)
        │  poll /v1/health (backoff)       │
        ▼                                  ▼
   Native webview   ──navigates to──▶  http://127.0.0.1:<ephemeral>/
   (splash first)                      (UI + API are same-origin; the engine serves the
                                        embedded React UI from its `*` fallback route)
```

## Closing the window, the tray, and Quit

By default, **closing the main window quits Camelid Desktop and stops the engine**, which is
the documented behaviour through v0.6.x. From v0.7.0 to v0.7.3 a close left the desktop and
its engine running with no window (see [the v0.7.0 regression](#the-v070-regression)). This
release restores close = quit in code; [What is NOT claimed](#what-is-not-claimed) says what
has been observed live.

**Keep engine running when window closes**, a check item in the tray menu, is off by default.
With it on, closing the main window hides it, the engine keeps serving, and the tray states
what the engine is doing. The first such close shows a one-time notice saying so and where
Quit is. The setting is stored in `desktop-lifetime.json` in the app-data directory
(`~/Library/Application Support/app.camelid.desktop/` on macOS,
`%APPDATA%\app.camelid.desktop\` on Windows). A missing, unreadable or unknown-version file
means off. If saving fails, the check mark stays where it was and the menu shows
`Could not save preference: …`.

Background mode is honoured only when it can be controlled and contained:

- a tray icon must exist, on both OSes, because it is the only surface that states the
  engine's status and offers Quit once the window is hidden;
- on Windows the engine must be inside the kill-on-close job object.

When either is missing, the check item is unchecked, disabled and says why, and closing the
window quits. The check mark always shows what the next close will do.

The tray (menu bar on macOS, notification area on Windows):

- **Left click** toggles Spotlight, as it has since v0.7.0. **Right click** opens the menu.
- The status line is observed, never assumed: `Engine starting…`, `Engine restarting…`,
  `Engine running on 127.0.0.1:<port> (loopback only)`, `Checking engine on
  127.0.0.1:<port>…`, `Engine not answering on 127.0.0.1:<port>`, `Engine stopped (exit
  code <n>)` or `(killed by signal <n>)`, `Engine failed to start: <reason>`. Running needs a
  live process and a `/v1/health` 200 at most 12 s old. An older answer reads Checking, two
  failed probes in a row read not answering, and an observed exit outranks any answer; once
  an engine is seen to stop, no later status for it reads as running. Pointing at or
  clicking the icon also requests an immediate re-read of the exit status. Whether that
  re-read lands before macOS draws the menu has not been observed.
- The model line comes from `/v1/health`: `Model: <id>`, `Model: <id> (not ready)`, or
  `No model ready`. It never says "No model loaded": while it switches models, a busy
  engine reports no active model whatever is loaded.
- After the engine stops, `Last engine message: …` shows the last line of its stderr.
  Stderr is drained into a 16 KiB tail once the health gate passes, so a chatty engine
  cannot block on a full pipe.
- Open Camelid, Show Spotlight, Restart engine, Keep engine running when window closes, and
  **Quit Camelid (stops the engine)**.

Every quit path stops the engine: the tray's Quit, Cmd+Q and the app menu,
`osascript -e 'tell application "Camelid Desktop" to quit'` (the Apple Event logout sends,
and what both macOS upgrade scripts rely on), and a close with background mode off. Nothing
vetoes an exit. A sidecar still inside its 40-second health gate is stopped too; it is held
in the engine slot from the moment it is spawned.

Relaunching the running app is meant to show the main window rather than start a second
engine and a second model load. On macOS a relaunch through the Dock icon, Finder or
`open -a` arrives as Reopen, which shows the main window. On Windows,
`tauri-plugin-single-instance` (pinned `=2.4.4`, Windows only) hands a second launch to the
running instance through a named mutex and a window class that are local to the user's
session, and ignores the arguments the second process passes. The plugin is not used on
macOS (see [What is NOT claimed](#what-is-not-claimed)). Dev builds share the installed
app's identifier and app data. On Windows a dev build focuses a running installed app and
exits, so **quit the installed app before `cargo run -p camelid-desktop`**. On macOS it
starts beside the installed app with a second engine.

Crash backstops, for when the desktop process dies without running any of its own code:

- **Windows:** the engine is in a job object with `KILL_ON_JOB_CLOSE`, so the OS kills it.
- **Every OS:** if the engine's `serve --help` lists `--exit-when-stdin-closes`, the desktop
  passes it and holds the engine's stdin open. The OS closes the pipe when the desktop dies
  and the engine exits. The desktop probes `serve --help` before every start, so an older
  engine, or one found on `PATH`, starts exactly as before, without the flag.

While backgrounded on macOS, the app holds an App Nap activity (user-initiated, idle system
sleep still allowed) so the tray's checks are not throttled. It is released when the window
is shown, when background mode is turned off, and on quit, so App Nap behaviour with
background mode off is unchanged. The hidden main webview stays resident in background mode,
on top of the engine and its model.

The sidecar stays bound to `127.0.0.1`. Serving other devices is not offered here.

### What is NOT claimed

- **Serving other devices.** The engine never binds a non-loopback address from the
  desktop. That needs its own explicit confirmation, a generated API key file, TLS or a
  shown cleartext acknowledgement, `--lan-chat-only`, and a stable port.
- **"Only this computer can reach it."** `(loopback only)` is the literal bind address, not
  an access boundary: a web page in a local browser can reach a loopback port through DNS
  rebinding, and the engine's generation and health routes do not check the `Host` header.
  Background mode lengthens how long that exposure lasts. This release adds no Host check.
- **A macOS crash backstop with an engine that predates `--exit-when-stdin-closes`.**
  Without the flag, a desktop that is killed leaves the engine running on macOS.
- **The macOS behaviour of this build, observed.** Close quits, background hide, the
  one-time notice, reopen, the quit paths, the stdin crash backstop and the tray's refresh
  timing are described here as designed and unit-tested. None of it has been observed live
  on this build yet. The only live receipt is the 0.7.3 regression below.
- **Any Windows behaviour of this build.** The `cfg(windows)` code (the kill-on-close job and
  its test, the single-instance registration) has been reviewed but not yet compiled. The
  CI smoke that drives the real desktop with a stand-in engine
  (`scripts/desktop-lifetime-smoke.ps1`) is written but has not yet run on a runner. On top
  of both, a receipt from real Windows hardware with the real engine is owed by the PR
  author.
- **A single instance on macOS outside LaunchServices.** Starting the executable directly,
  `open -n`, or a second copy of the app at another path starts a second desktop with its
  own engine and model load. `tauri-plugin-single-instance` would cover that, but on macOS
  its socket is a fixed path in the shared `/tmp` (`/tmp/app_camelid_desktop_si.sock`). On a
  Mac with several accounts, another user's socket there makes a launch skip the check
  silently, and a socket someone else binds first swallows every launch before a window
  appears. So it is registered on Windows only.
- **Launch at login.** Not offered.

### The v0.7.0 regression

48c3c261 (between v0.6.1 and v0.7.0) added the hidden `spotlight` window. It is never
destroyed, so closing `main` never emptied the window set and the app never exited.
Measured on an installed 0.7.3: after the close button every app window was off screen, yet
the desktop and its sidecar kept running and `/v1/health` still answered 200. `open -a
"Camelid Desktop"` did not bring the window back, the only tray icon toggled Spotlight, and
the app menu's Quit stopped both processes within 3 s. Closing the window now requests an
exit explicitly, so Spotlight no longer outlives the main window unless background mode is
on.

The sidecar receives a new ephemeral port on each launch, so browser-origin storage cannot be
the desktop app's durable authority. Before React starts, the shell hydrates Camelid-owned UI
state from `ui-storage-v1.json` in the per-user application-data directory. Writes are mirrored
there through scoped commands; the regular browser build continues to use `localStorage`.

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
- **An arbitrary GGUF file-picker is deferred.** The loopback-origin page receives
  a scoped native folder chooser plus Camelid-keyed UI-state commands confined to the app-data
  directory; it does not receive broad filesystem access. Local/catalog model loading still
  goes through the engine's existing API.
