#!/usr/bin/env bash
#
# Sign the DMG for Sparkle and write the update feed for it:
#
#   build/appcast.xml   one signed item: this version's DMG on its GitHub release
#
# The release workflow attaches the feed to the release, and installed copies
# read it from releases/latest/download/appcast.xml. Run it after make-dmg.sh:
# stapling changes the DMG, so it is signed for Sparkle once it is final.
#
#   SPARKLE_ED_PRIVATE_KEY=<base64 seed> scripts/make-appcast.sh build/Jetlink.app
#   SPARKLE_ED_KEY_FILE=path/to/key scripts/make-appcast.sh build/Jetlink.app
#
# The notes are CHANGELOG.md's, this release and the ones before it
# (scripts/changelog.py history). The download URL is GITHUB_REPOSITORY's
# release (zoompilot/jetlink's outside CI); UPDATE_DOWNLOAD_BASE puts the DMG
# somewhere else, for a test feed.
#
# Gotchas:
#   - The signature is checked against the SUPublicEDKey inside the app before
#     anything is written: a secret that is not that key's other half would
#     publish an update every installed copy refuses.
#   - sign_update is the one the Sparkle package brought into DerivedData, so
#     it is the version the app embeds; `make app` has to have run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$MACOS_DIR")"

APP="${1:-$MACOS_DIR/build/Jetlink.app}"
[ -d "$APP" ] || { echo "error: no app bundle at $APP" >&2; exit 1; }

# The DMG's name follows make-dmg.sh: JETLINK_VERSION (the Makefile sets it
# from the tag), else the app's own. The feed takes its versions from the app.
APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
VERSION="${JETLINK_VERSION:-$APP_VERSION}"
if [ "$VERSION" != "$APP_VERSION" ]; then
  echo "error: the app is $APP_VERSION, not $VERSION; build it again before make appcast" >&2
  exit 1
fi
PUBLIC_KEY="$(/usr/libexec/PlistBuddy -c 'Print :SUPublicEDKey' "$APP/Contents/Info.plist")"
TAG="v$VERSION"
DMG="$MACOS_DIR/build/Jetlink-$VERSION-macOS.dmg"
OUT="$MACOS_DIR/build/appcast.xml"
REPO="${GITHUB_REPOSITORY:-zoompilot/jetlink}"
DOWNLOAD_BASE="${UPDATE_DOWNLOAD_BASE:-https://github.com/$REPO/releases/download/$TAG}"
SIGN_UPDATE="$MACOS_DIR/build/DerivedData/SourcePackages/artifacts/sparkle/Sparkle/bin/sign_update"

[ -f "$DMG" ] || { echo "error: no $DMG; run make dmg first" >&2; exit 1; }
[ -x "$SIGN_UPDATE" ] || { echo "error: no sign_update at $SIGN_UPDATE; run make app first" >&2; exit 1; }
if [ -z "${SPARKLE_ED_PRIVATE_KEY:-}" ] && [ -z "${SPARKLE_ED_KEY_FILE:-}" ]; then
  echo "error: set SPARKLE_ED_PRIVATE_KEY or SPARKLE_ED_KEY_FILE" >&2
  exit 1
fi

# The key goes to sign_update on stdin, never on a command line.
key() {
  if [ -n "${SPARKLE_ED_PRIVATE_KEY:-}" ]; then
    printf '%s\n' "$SPARKLE_ED_PRIVATE_KEY"
  else
    cat "$SPARKLE_ED_KEY_FILE"
  fi
}

echo "==> signing $(basename "$DMG") for Sparkle"
SIGNATURE="$(key | "$SIGN_UPDATE" --ed-key-file - -p "$DMG")"

echo "==> checking it against the app's SUPublicEDKey"
xcrun swift "$SCRIPT_DIR/check-update-signature.swift" "$PUBLIC_KEY" "$SIGNATURE" "$DMG"

NOTES="$(mktemp -t jetlink-notes)"
trap 'rm -f "$NOTES"' EXIT
python3 "$REPO_ROOT/scripts/changelog.py" history "$TAG" > "$NOTES"
if [ ! -s "$NOTES" ]; then
  echo "warning: CHANGELOG.md has no $TAG section; the update window links to the release page instead"
fi

echo "==> writing $OUT"
python3 "$SCRIPT_DIR/make-appcast.py" \
  --app "$APP" --archive "$DMG" --signature "$SIGNATURE" \
  --url "$DOWNLOAD_BASE/$(basename "$DMG")" \
  --release-page "https://github.com/$REPO/releases/tag/$TAG" \
  --history "https://github.com/$REPO/releases" \
  --notes "$NOTES" --output "$OUT"

echo "==> signing the feed"
key | "$SIGN_UPDATE" --ed-key-file - "$OUT"
