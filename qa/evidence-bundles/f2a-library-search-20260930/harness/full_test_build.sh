#!/usr/bin/env bash
# Full test suite as CI (all targets, all features), then gates, then the frontend + release eval build.
cd ~/Camelid || exit 1
source ~/.cargo/env
echo "commit=$(git rev-parse --short HEAD) tracked_changes=$(git status --porcelain --untracked-files=no | wc -l)"
# smoke_admits_tinyllama is documented as release-only; in a debug build it runs for over an hour.
cargo test --all-targets --all-features --no-fail-fast -- --skip smoke_admits_tinyllama > /tmp/f2l/fulltest2.log 2>&1
echo "TEST_EXIT=$?"
grep -E "^test result:" /tmp/f2l/fulltest2.log | awk '{p+=$4; f+=$6; i+=$8} END {print "passed", p, "failed", f, "ignored", i, "binaries", NR}'
grep -E "^test .* FAILED$" /tmp/f2l/fulltest2.log | sort -u
bash /tmp/f2h/run_gates.sh
(cd frontend && npm run build > /tmp/f2l/fe_build_release.log 2>&1) && echo FE_OK
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release --bin camelid --target-dir target/eval > /tmp/f2l/release_build.log 2>&1
echo "BUILD_EXIT=$?"
mkdir -p /tmp/f2l/bin
cp target/eval/release/camelid /tmp/f2l/bin/camelid-library
/tmp/f2l/bin/camelid-library --version
echo "tracked changes after: $(git status --porcelain --untracked-files=no | wc -l)"
echo FULL_DONE
