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
#     server's onnxruntime) are signed here first, deepest first; any other
#     nested code stops the script.
#   - Ad hoc ("-") cannot carry a secure timestamp, so --timestamp is dropped
#     in that case. --options runtime stays on either way so a local build
#     behaves like a release one.
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
