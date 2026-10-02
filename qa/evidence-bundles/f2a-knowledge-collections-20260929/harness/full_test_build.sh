#!/usr/bin/env bash
# Full test suite as CI (all targets, all features), then the frontend + release eval build.
cd ~/Camelid || exit 1
source ~/.cargo/env
echo "== cargo test --all-targets --all-features"
cargo test --all-targets --all-features > /tmp/f2c/fulltest.log 2>&1
echo "TEST_EXIT=$?"
grep -E "^test result:" /tmp/f2c/fulltest.log | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i, "binaries", NR}'
grep -E "^test .* FAILED$|panicked" /tmp/f2c/fulltest.log | head -20
echo "== release eval build"
(cd frontend && npm run build > /tmp/f2c/fe_build_release.log 2>&1) && echo FE_OK
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release --bin camelid --target-dir target/eval > /tmp/f2c/release_build.log 2>&1
echo "BUILD_EXIT=$?"
mkdir -p /tmp/f2c/bin
cp target/eval/release/camelid /tmp/f2c/bin/camelid-collections
/tmp/f2c/bin/camelid-collections --version
echo FULL_DONE
