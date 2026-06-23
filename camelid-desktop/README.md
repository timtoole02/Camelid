# Camelid Desktop (add-on, Windows)

**Camelid Desktop is an additive native Windows app.** It gives users a desktop chat
experience with no web browser, by embedding the **same `camelid` engine** that ships as the
server binary and hosting the existing web UI in a native [WebView2](https://developer.microsoft.com/microsoft-edge/webview2/)
window via [Tauri v2](https://v2.tauri.app/).

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
  Qwen3 Q8_0 rows and the Gemma 4 E4B-It Q8_0 row), and falls back to the CPU otherwise. The
  app adds no GPU code and makes no separate performance claim; the authoritative supported-row
  and GPU list is the engine's [`README.md`](../README.md) (*Windows CUDA*).
- **No fabricated metrics.** Any tokens/sec or status readout is sourced from the same real
  generation events the server emits (the SSE `camelid.decode_tps` field). If a metric is
  unavailable it is shown as unavailable, never as a placeholder.

This app makes **no broader claims** than the engine it embeds about supported models,
performance, or compatibility.

## Architecture (sidecar; see `../DECISIONS.md` D11)

```
camelid-desktop.exe ──spawns──▶ camelid.exe serve --addr 127.0.0.1:<ephemeral> --no-open
        │                                  │  (loopback only)
        │  poll /v1/health (backoff)       │
        ▼                                  ▼
   WebView2 window  ──navigates to──▶  http://127.0.0.1:<ephemeral>/
   (splash first)                      (UI + API are same-origin; the engine serves the
                                        embedded React UI from its `*` fallback route)
```

On window close the sidecar is terminated cleanly; a Windows **job object** with
`KILL_ON_JOB_CLOSE` is the backstop so a desktop crash cannot orphan a `camelid` process.

## Requirements

- Windows 10/11 with the **WebView2 runtime** (preinstalled on current Windows 10/11; the
  Tauri bundle ships the bootstrapper otherwise).
- A `camelid.exe` next to `camelid-desktop.exe` (the portable zip and installer bundle it).

## Building (developers)

```sh
# From the workspace root. Build the server in RELEASE so a working camelid.exe lands in
# target/release/ (see the debug caveat below), then build + run the desktop app:
cargo build --release --locked --bin camelid
cargo build -p camelid-desktop

# Run the desktop app, pointing it at the release sidecar that sits beside it. In dev the
# desktop exe is in target/debug/, so place (or symlink/copy) the release camelid.exe there:
cp target/release/camelid.exe target/debug/camelid.exe   # one-time, for dev
cargo run -p camelid-desktop
```

> **Debug-server caveat (pre-existing, server-side).** The *debug* `camelid.exe` overflows
> its main-thread stack on startup (it crashes even on `camelid --version`) — a large stack
> frame in the server's `main.rs` that release optimization elides. This is unrelated to the
> desktop app and out of scope here (the brief forbids modifying the server). Always pair the
> desktop app with a **release** `camelid.exe`; the shipped artifact does exactly that. When
> the sidecar fails to come up, the desktop surfaces the real error + engine stderr on the
> splash rather than faking a ready state — which is the intended fail-closed behavior.

The server build is unaffected by this crate: `cargo build --release --locked --bin camelid`
does not pull `camelid-desktop` into its graph (workspace `resolver = "2"`,
`default-members = ["."]`).

For a bundled installer + portable zip, see the additive `desktop-windows` job in
`../.github/workflows/release.yml`.

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
