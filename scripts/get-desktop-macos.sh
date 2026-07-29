#!/usr/bin/env bash
# get-desktop-macos.sh — install the prebuilt Camelid Desktop app on an Apple
# Silicon Mac with one command, no toolchain required:
#
#   curl -fsSL https://raw.githubusercontent.com/timtoole02/Camelid/main/scripts/get-desktop-macos.sh | bash
#
# Downloads the release DMG, verifies its published SHA-256, installs the app
# bundle into /Applications, and launches it. Pass a tag to pin a version
# (`... | bash -s -- v0.4.5`), or set CAMELID_DESKTOP_TAG. Model files under
# the app's Application Support directory are never touched.
#
# Why a script instead of "download the DMG and double-click": the macOS app is
# ad-hoc signed and not notarized, so a browser-downloaded copy is quarantined
# and Gatekeeper blocks the first launch. Command-line downloads carry no
# quarantine attribute, so this path installs an app that opens immediately.
# To build from source instead, see scripts/install-macos-desktop.sh.
set -euo pipefail

repo="${CAMELID_REPO:-timtoole02/Camelid}"
tag="${1:-${CAMELID_DESKTOP_TAG:-latest}}"
asset="camelid-desktop-macos-arm64.dmg"
app_name="Camelid Desktop.app"
applications_dir="/Applications"
installed_app="$applications_dir/$app_name"
desktop_process="$installed_app/Contents/MacOS/camelid-desktop"
sidecar_process="$installed_app/Contents/Resources/sidecar/camelid serve"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "error: Camelid Desktop currently requires an Apple Silicon Mac" >&2
  exit 1
fi

if [ "$tag" = "latest" ]; then
  base_url="https://github.com/$repo/releases/latest/download"
else
  base_url="https://github.com/$repo/releases/download/$tag"
fi

work_dir="$(mktemp -d /tmp/camelid-desktop-get.XXXXXX)"
mount_point="$work_dir/mount"

# Freshly mounted images can stay transiently busy (Spotlight/XProtect scan new
# volumes), so detach with retries — and never let an unmount hiccup fail an
# install that already succeeded.
is_mounted() {
  [ -d "$mount_point" ] \
    && [ "$(stat -f %d "$mount_point" 2>/dev/null)" != "$(stat -f %d "$work_dir" 2>/dev/null)" ]
}
detach_image() {
  local attempt
  for attempt in 1 2 3 4 5; do
    is_mounted || return 0
    if hdiutil detach "$mount_point" -quiet 2>/dev/null; then return 0; fi
    sleep 1
  done
  hdiutil detach "$mount_point" -force -quiet 2>/dev/null || true
  ! is_mounted
}
cleanup() {
  detach_image || true
  # Never delete through a mount point that refused to detach.
  if [[ "$work_dir" == /tmp/camelid-desktop-get.* ]] && ! is_mounted; then
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT

echo "Downloading $asset ($tag) ..."
if ! curl -fL --retry 3 --progress-bar -o "$work_dir/$asset" "$base_url/$asset"; then
  echo "error: could not download $base_url/$asset" >&2
  echo "The requested release may not include the macOS desktop app (the DMG first shipped" >&2
  echo "after v0.4.4), or the network request failed. To build from source instead:" >&2
  echo "  git clone https://github.com/$repo.git && cd ${repo##*/} && ./scripts/install-macos-desktop.sh" >&2
  exit 1
fi

# The sidecar checksum names the asset file, so verification must run beside it.
curl -fsSL -o "$work_dir/$asset.sha256" "$base_url/$asset.sha256"
(cd "$work_dir" && shasum -a 256 -c "$asset.sha256")

# Give a running app a chance to terminate its loopback sidecar cleanly before
# the bundle is replaced (same shutdown handshake as install-macos-desktop.sh).
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

mkdir -p "$mount_point"
hdiutil attach "$work_dir/$asset" -nobrowse -readonly -mountpoint "$mount_point" -quiet

if [ ! -d "$mount_point/$app_name" ]; then
  echo "error: the DMG does not contain $app_name" >&2
  exit 1
fi

# Replace (not merge-over) any existing bundle so no stale files from an older
# version survive inside the app. Model files live in Application Support, not
# in the bundle, so they are unaffected.
if [ -w "$applications_dir" ] && { [ ! -e "$installed_app" ] || [ -w "$installed_app" ]; }; then
  rm -rf -- "$installed_app"
  ditto --noextattr --noqtn "$mount_point/$app_name" "$installed_app"
  xattr -cr "$installed_app"
else
  echo "Installing into $applications_dir requires administrator access."
  sudo rm -rf -- "$installed_app"
  sudo ditto --noextattr --noqtn "$mount_point/$app_name" "$installed_app"
  sudo xattr -cr "$installed_app"
fi

detach_image \
  || echo "note: the installer image at $mount_point is still busy; it will detach on its own" >&2
codesign --verify --deep --strict "$installed_app"
open "$installed_app"

echo "Installed and launched $installed_app"
