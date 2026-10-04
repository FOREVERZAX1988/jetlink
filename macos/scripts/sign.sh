#!/usr/bin/env bash
#
# Sign Jetlink.app.
#
#   scripts/sign.sh macos/build/Jetlink.app
#   SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" scripts/sign.sh ...
#
# Gotchas:
#   - codesign --deep is not used for signing: it would sign any nested code
#     with the app's entitlements. The frameworks the app embeds (the Swift
#     server's onnxruntime, Sparkle) are signed here first, deepest first; any
#     other nested code stops the script.
#   - Ad hoc ("-") cannot carry a secure timestamp, so --timestamp is dropped
#     in that case. --options runtime stays on either way so a local build
#     behaves like a release one.
#   - The hardened runtime's library validation lets the app load Sparkle only
#     when both carry the same Team ID, and an ad hoc signature has none. An
#     ad hoc app is signed with library validation off; a Developer ID one
#     keeps it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

APP="${1:-$MACOS_DIR/build/Jetlink.app}"
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
APP_ENTITLEMENTS="$MACOS_DIR/Resources/Jetlink.entitlements"

[ -d "$APP" ] || { echo "error: no app bundle at $APP" >&2; exit 1; }

TIMESTAMP=--timestamp
if [ "$SIGN_IDENTITY" = "-" ]; then
  TIMESTAMP=--timestamp=none
  echo "==> signing ad hoc; this build is for local use only"
  ADHOC_ENTITLEMENTS="$(mktemp -t jetlink-entitlements)"
  trap 'rm -f "$ADHOC_ENTITLEMENTS"' EXIT
  cp "$APP_ENTITLEMENTS" "$ADHOC_ENTITLEMENTS"
  /usr/libexec/PlistBuddy -c "Add :com.apple.security.cs.disable-library-validation bool true" "$ADHOC_ENTITLEMENTS" >/dev/null
  APP_ENTITLEMENTS="$ADHOC_ENTITLEMENTS"
fi

NESTED="$(find "$APP/Contents" \( -path "$APP/Contents/MacOS" -o -path "$APP/Contents/Frameworks" \) -prune -o -type f \( -name '*.dylib' -o -name '*.so' \) -print)"
if [ -n "$NESTED" ]; then
  echo "error: nested code this script does not sign yet:" >&2
  printf '  %s\n' "$NESTED" >&2
  exit 1
fi

FRAMEWORKS="$APP/Contents/Frameworks"
if [ -d "$FRAMEWORKS" ]; then
  # The app is built ad hoc (build-app.sh), so the frameworks it embeds, the
  # Swift server's onnxruntime among them, are signed here with the app's
  # identity: their loose dylibs first, then each framework bundle.
  echo "==> signing the frameworks"
  while IFS= read -r -d '' f; do
    codesign --force --options runtime "$TIMESTAMP" --sign "$SIGN_IDENTITY" "$f"
  done < <(find "$FRAMEWORKS" -type f -name '*.dylib' -print0)
  # Sparkle's two helpers, before the framework that seals them, in the order
  # Sparkle's documentation gives. Its XPC services would need their
  # entitlements kept; Jetlink is not sandboxed, so build-app.sh removes them.
  SPARKLE="$FRAMEWORKS/Sparkle.framework"
  if [ -d "$SPARKLE" ]; then
    if [ -n "$(find "$SPARKLE" -name '*.xpc' -print -quit)" ]; then
      echo "error: Sparkle's XPC services are in the bundle; build-app.sh should have removed them" >&2
      exit 1
    fi
    for helper in "$SPARKLE/Versions/B/Autoupdate" "$SPARKLE/Versions/B/Updater.app"; do
      [ -e "$helper" ] || { echo "error: no $helper; has Sparkle's layout changed?" >&2; exit 1; }
      codesign --force --options runtime "$TIMESTAMP" --sign "$SIGN_IDENTITY" "$helper"
    done
  fi
  for f in "$FRAMEWORKS"/*.framework; do
    [ -e "$f" ] || continue
    codesign --force --options runtime "$TIMESTAMP" --sign "$SIGN_IDENTITY" "$f"
  done
fi
echo "==> signing the app"
codesign --force --options runtime "$TIMESTAMP" \
  --entitlements "$APP_ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$APP"

echo "==> verifying"
codesign --verify --deep --strict --verbose=2 "$APP"

if [ "$SIGN_IDENTITY" != "-" ]; then
  # Before notarization this reports "Unnotarized Developer ID". That is
  # expected; notarize.sh is the next step.
  spctl --assess --type execute --verbose "$APP" || true
fi
