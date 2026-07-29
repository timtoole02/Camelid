#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
build_target="${CAMELID_DESKTOP_TARGET_DIR:-$repo_root/target}"
built_app="$build_target/release/bundle/macos/Camelid Desktop.app"
applications_dir="/Applications"
installed_app="$applications_dir/Camelid Desktop.app"
desktop_process="$installed_app/Contents/MacOS/camelid-desktop"
sidecar_process="$installed_app/Contents/Resources/sidecar/camelid serve"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "error: Camelid Desktop currently requires an Apple Silicon Mac" >&2
  exit 1
fi

"$script_dir/build-macos-desktop.sh"

if [ ! -d "$built_app" ]; then
  echo "error: the macOS build did not produce $built_app" >&2
  exit 1
fi

# Give the running app a chance to terminate its loopback sidecar cleanly before
# replacing the bundle. Model files live in Application Support and are untouched.
osascript -e 'tell application "Camelid Desktop" to quit' 2>/dev/null || true
for _ in {1..20}; do
  if ! pgrep -f "$desktop_process" >/dev/null \
    && ! pgrep -f "$sidecar_process" >/dev/null; then
    break
  fi
  sleep 0.25
done

if pgrep -f "$desktop_process" >/dev/null || pgrep -f "$sidecar_process" >/dev/null; then
  echo "error: Camelid Desktop did not exit cleanly; close it and run this script again" >&2
  exit 1
fi

if [ -w "$applications_dir" ]; then
  ditto --noextattr --noqtn "$built_app" "$installed_app"
  xattr -cr "$installed_app"
else
  sudo ditto --noextattr --noqtn "$built_app" "$installed_app"
  sudo xattr -cr "$installed_app"
fi

codesign --verify --deep --strict "$installed_app"
open "$installed_app"

echo "Installed and launched $installed_app"
