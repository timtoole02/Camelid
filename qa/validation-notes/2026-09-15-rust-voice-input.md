# Rust voice input validation — 2026-09-15

Implementation: Rust Candle 0.9.2 CPU Whisper tiny.en, hound WAV parsing,
Rust resampling/mel filters, existing Rust tokenizers, authenticated speech API,
and React microphone capture through AudioWorklet. No C++ speech dependency.

Checks on Apple M4 macOS:

- `cargo build --release --bin camelid`: passed with the final frontend assets.
- `cargo check --lib`: passed (existing unrelated dead-code warnings).
- `cargo test --lib speech -- --nocapture`: five tests passed; opt-in real-model
  unit test excluded from ordinary test runs.
- `npm run build`: passed; recording worklet emitted as a separate JS asset.
- `npm run smoke:ui`: passed.
- `npm run smoke:voice`: passed using the real ChatWorkspace and browser
  AudioWorklet with a synthetic microphone and controlled speech API responses.
  Covers model setup, WAV encoding, preserving the draft, no automatic send,
  permission denial, stopping all tracks on cancel, late response isolation
  after switching chats, mobile panel bounds, and automatic stop at 30 seconds.
- `api::server::tests::lan_chat_surface_is_an_exact_method_and_path_allowlist`: passed.
- Live browser-to-Rust check passed: a known speech MediaStream went through the
  real AudioWorklet, PCM16 encoding, Rust 48-to-16 kHz resampling and Whisper, and
  appeared in the existing editable prompt without sending. Chrome's file-backed
  fake microphone yielded silence on this host, so the live check uses a
  deterministic AudioBuffer-to-MediaStream fixture instead of that device.
- Mobile capture inspected at 390 × 844: panel and controls fit the viewport.
- macOS Info.plist and Entitlements.plist passed `plutil -lint`;
  `scripts/build-macos-desktop.sh` passed `bash -n`.
- Candle dependency feature tree contains no native acceleration, onig,
  C/C++ compiler, or whisper.cpp dependency. HTTPS uses Camelid's already
  existing Rustls/AWS-LC provider; no new native crypto backend was added.

Real release binary tested on `127.0.0.1:8198`, with API-key authentication and
an isolated models directory:

- Unauthenticated speech status returned 401.
- Rust model installer downloaded and verified all three pinned artifacts in
  3.41 seconds on this connection.
- A standard 11-second JFK WAV fixture returned in 0.94 seconds:
  "And so my fellow Americans ask not what your country can do for you, ask
  what you can do for your country."
- Repeating the transcription produced the same text.
- Malformed audio returned 400; an oversized request returned 413.
- A second of silent 48 kHz PCM16 audio returned an empty transcript.
- Concurrent requests returned one successful transcription and one 409.
- Corrupt model bytes were rejected with 422, and the installer repaired them.

Fixture source (downloaded for testing, not bundled):
https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav
This is an audio fixture only; no whisper.cpp code or executable is used.

These are local CPU/browser checks. Actual microphone permission prompts in
packaged macOS and Windows desktop applications, and Windows CPU speed, have
not been validated on physical devices in this run. Initial speech support is
English and limited to 30-second recordings, with user review before sending.
