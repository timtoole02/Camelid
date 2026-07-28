#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
desktop_dir="$repo_root/camelid-desktop"
build_target="${CAMELID_DESKTOP_TARGET_DIR:-$repo_root/target}"
app_version="$(
  node -e 'const fs = require("fs"); const config = JSON.parse(fs.readFileSync(process.argv[1], "utf8")); process.stdout.write(config.version)' \
    "$desktop_dir/tauri.conf.json"
)"
app_path="$build_target/release/bundle/macos/Camelid Desktop.app"
dmg_dir="$build_target/release/bundle/dmg"
dmg_path="$dmg_dir/Camelid Desktop_${app_version}_aarch64.dmg"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "error: the Camelid Desktop macOS bundle must currently be built on Apple Silicon" >&2
  exit 1
fi

export CARGO_TARGET_DIR="$build_target"

clean_bundle_metadata() {
  local bundle_path="$1"
  xattr -cr "$bundle_path"
  xattr -d com.apple.FinderInfo "$bundle_path" 2>/dev/null || true
  xattr -d com.apple.ResourceFork "$bundle_path" 2>/dev/null || true
  xattr -d 'com.apple.fileprovider.fpfs#P' "$bundle_path" 2>/dev/null || true
}

(
  cd "$repo_root/frontend"
  npm ci
  npm run build
)

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --release \
  --locked \
  --bin camelid

mkdir -p "$desktop_dir/sidecar"
cp "$build_target/release/camelid" "$desktop_dir/sidecar/camelid"
chmod 755 "$desktop_dir/sidecar/camelid"

(
  cd "$desktop_dir"
  npx --yes @tauri-apps/cli@^2 build --config tauri.macos.bundle.conf.json
)

# Tauri's styled DMG workflow uses Finder to arrange the image, which can attach FinderInfo
# metadata to the already-signed app on newer macOS hosts. Build a simple drag-to-Applications
# image in /tmp instead. Keeping the final staging copy outside Documents/iCloud also prevents
# file-provider metadata from being reattached between cleaning and signature verification.
# This can be replaced by Tauri's native DMG target when Developer ID notarization is added.
stage_dir="$(mktemp -d /tmp/camelid-dmg-stage.XXXXXX)"
cleanup() {
  if [[ "$stage_dir" == /tmp/camelid-dmg-stage.* ]]; then
    rm -rf -- "$stage_dir"
  fi
}
trap cleanup EXIT

ditto --noextattr --noqtn "$app_path" "$stage_dir/Camelid Desktop.app"
ln -s /Applications "$stage_dir/Applications"
clean_bundle_metadata "$stage_dir/Camelid Desktop.app"
codesign --force --deep --sign - --options runtime "$stage_dir/Camelid Desktop.app"
codesign --verify --deep --strict "$stage_dir/Camelid Desktop.app"

mkdir -p "$dmg_dir"
hdiutil create \
  -volname "Camelid Desktop" \
  -srcfolder "$stage_dir" \
  -ov \
  -format UDZO \
  "$dmg_path"

echo
echo "macOS app (intermediate): $app_path"
echo "macOS DMG: $dmg_path"
