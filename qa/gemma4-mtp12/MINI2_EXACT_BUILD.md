# Reproducing the mini2 target stream

The mini2 takeover measurements use the official Gemma 4 12B MTP assistant
with the QAT Q4_0 target and its existing native Q4 sidecar. The target
checks every emitted draft token. Shortlisting changes only assistant
proposals; qualification still compares the complete generated token ID
array with the saved old-binary response for each request.

## Compiler settings are part of the reference

The existing mini2 release binary was compiled with Homebrew Rust 1.94.1,
release LTO disabled, and 16 codegen units. Reproduce those settings:

```sh
export PATH=/opt/homebrew/bin:$PATH
export CARGO_BUILD_JOBS=1
export CARGO_PROFILE_RELEASE_LTO=off
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
cargo build --release --bin camelid
```

Use `~/bin/cam-lock.sh` around builds and model/GPU jobs on mini2. Run only
one heavy job at a time. Record the source commit and executable SHA-256
with every measurement. Source archives must set `CAMELID_GIT_COMMIT` and
`CAMELID_GIT_DESCRIBE` explicitly so the executable identifies its source.

The repository's default fat-LTO/single-codegen-unit build produced a
different inference stream at output token 65 in the takeover gate. The
difference persisted with the new matrix-unit head disabled. Rebuilding
the same source with the original settings restored the original token
IDs on both short-chat controls. Therefore, the default-profile build is
not qualified as a replacement for this particular reference binary.

## Measurement rules

Use fresh chat requests at temperature zero with the original prompt text
and output budget. Compare nonempty integer token ID arrays, response
text, and finish reasons. Missing baseline files or missing token arrays
must fail the gate. Repeated runs may compare against repeat zero of the
same saved request.

Report the request's `camelid.mtp12.decode_tokens_per_second` value and
acceptance together. The nested `native_receipt_qualification` field is a
historical qualification record, not the speed of the current request.
Report each prompt separately; a coding result above 50 tok/s does not
establish ordinary short-chat performance above 50 tok/s.

The production launcher uses a larger KV capacity than the short bench.
Validate the intended launcher capacity separately before deploying a new
executable or selector combination.
