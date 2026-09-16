# Voice input

Chat's microphone button records up to 30 seconds of English speech and inserts
an editable transcript into the current prompt. Press the microphone, speak,
then press **Stop and transcribe**. Review the text and press Send yourself.
Existing draft text is preserved. Cancel discards the recording; changing chats
or leaving Chat closes the microphone and discards any late response.

On first use, choose **Download speech model**. Camelid downloads approximately
154 MB from `openai/whisper-tiny.en` on Hugging Face. The three files are pinned to
revision `87c7102498dcde7456f24cfd30239ca606ed9063`, checked against fixed SHA-256
hashes, and stored under `<models-dir>/speech/whisper-tiny.en/`. Subsequent
transcription works offline. Model files are verified again before inference.

Speech preprocessing and Whisper inference run in Rust, using hound, Candle
0.9.2's CPU implementation, and the existing tokenizers crate with its Rust
regex backend. No whisper.cpp, C++ speech runtime, Python service, system BLAS,
or browser/cloud speech-recognition service is used. Candle's acceleration
features are disabled. HTTPS downloads reuse Camelid's existing Rustls crypto
provider; this does not change the native dependencies already used elsewhere
in Camelid.

The browser/WebView captures mono audio and encodes PCM16 WAV; Rust validates,
resamples to 16 kHz, computes the mel spectrogram, and decodes the transcript.
Only one speech download or transcription runs per server at a time. Model and
inference buffers are released after each transcription. Digital silence is
rejected, and decoding has token/time bounds. As with other speech models,
noise, accents, music, and very quiet speech can produce incorrect text.

Recordings are sent only to the configured Camelid server and are not saved by
this feature. With a local server they remain on your computer. With a remote
Camelid server, that server processes the recording. Audio is never included in
model-download requests. Browser microphone access requires HTTPS or localhost;
plain HTTP over a LAN is insufficient. A restricted LAN Chat listener permits
speech status and transcription after authentication, but model installation
must be done on the host's full interface.

macOS bundles include `NSMicrophoneUsageDescription` and the hardened-runtime
`com.apple.security.device.audio-input` entitlement. The final DMG staging
signature preserves that entitlement. Windows WebView2 and browser users must
allow microphone access when prompted. Real-device desktop checks remain
necessary when releasing on each platform.

## API

- `GET /api/speech/status`: model installation state, busy state, language,
  download size, and recording limit.
- `POST /api/speech/install`: downloads and verifies the fixed English model.
  Honors the server's configured download ceiling.
- `POST /api/speech/transcribe`: raw `audio/wav` body, mono signed PCM16,
  8–96 kHz, 0.2–30 seconds. Returns `{ "text": "..." }`; silence returns empty
  text. Upload size is capped independently of the overall request ceiling.

Routes inherit the server's existing authentication and CORS policy. Concurrent
speech requests return 409. Missing models return 412; invalid audio returns
400. Aborting a transcription request discards the result in the UI, while the
bounded Rust worker retains its admission slot until it finishes.

## Validation

Run `cargo test --lib speech` for audio validation, resampling, checksum,
silence, and admission tests. Run `npm run smoke:voice` in `frontend/` for the
real ChatWorkspace/AudioWorklet browser flow with controlled server responses.

A real-model test can be enabled explicitly:

```sh
CAMELID_SPEECH_TEST_MODEL_DIR=/path/to/models/speech/whisper-tiny.en \
CAMELID_SPEECH_TEST_WAV=/path/to/mono-pcm16.wav \
CAMELID_SPEECH_TEST_EXPECT='expected words' \
cargo test --release --lib speech::tests::real_recording -- --ignored --nocapture
```

Use release builds for interactive CPU transcription. Debug inference is much
slower. This initial implementation supports English dictation, not live
streaming captions or spoken assistant replies.

To test the browser recorder against a running Rust server with the model
installed, provide an 11-second speech WAV fixture (the standard JFK fixture is
used by the default expected-text check):

```sh
cd frontend
CAMELID_SPEECH_LIVE_API=http://127.0.0.1:8181 \
CAMELID_SPEECH_LIVE_WAV=/absolute/path/to/jfk.wav \
CAMELID_SPEECH_LIVE_KEY=your-test-api-key \
npm run smoke:voice
```

This injects a known speech MediaStream into the real AudioWorklet recorder and
asserts the Rust API's transcript appears in the draft without being sent. It
does not record a physical microphone. `CAMELID_SPEECH_LIVE_EXPECT` overrides the
expected text fragment. The default smoke separately uses Chrome's fake
microphone for permissions, lifecycle, and cancellation coverage.
