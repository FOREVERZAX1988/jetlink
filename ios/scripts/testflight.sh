#!/usr/bin/env bash
#
# Archive the app for the App Store and upload it to App Store Connect, where
# it appears in TestFlight once Apple has processed it.
#
#   JETLINK_TEAM=ABCDE12345 ASC_KEY_ID=... ASC_ISSUER_ID=... ASC_KEY_PATH=AuthKey_XXXX.p8 \
#     ios/scripts/testflight.sh
#
# The API key signs as well as uploads: xcodebuild makes the distribution
# certificate and profile itself, which needs a key with the Admin role. The
# notary key's NOTARY_* variables stand in for ASC_* when one key does both.
# UPLOAD=0 stops at an exported .ipa in ios/build/export.
#
# The version is jetlink.__version__ and the build number the commit count, as
# on the Mac; App Store Connect refuses a build number it already has for that
# version, so JETLINK_BUILD overrides it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$IOS_DIR")"
BUILD="$IOS_DIR/build"

command -v xcodegen >/dev/null 2>&1 || { echo "error: xcodegen is missing; run: brew install xcodegen" >&2; exit 1; }

ASC_KEY_ID="${ASC_KEY_ID:-${NOTARY_KEY_ID:-}}"
ASC_ISSUER_ID="${ASC_ISSUER_ID:-${NOTARY_ISSUER_ID:-}}"
ASC_KEY_PATH="${ASC_KEY_PATH:-${NOTARY_KEY_PATH:-}}"
for var in JETLINK_TEAM ASC_KEY_ID ASC_ISSUER_ID ASC_KEY_PATH; do
  if [ -z "${!var:-}" ]; then
    echo "error: $var is not set" >&2
    exit 1
  fi
done
[ -f "$ASC_KEY_PATH" ] || { echo "error: no key file at $ASC_KEY_PATH" >&2; exit 1; }
# xcodebuild wants the key's path absolute
ASC_KEY_PATH="$(cd "$(dirname "$ASC_KEY_PATH")" && pwd)/$(basename "$ASC_KEY_PATH")"

# The App Store takes up to three dot-separated integers, so a prerelease
# version (0.8.0rc1) cannot go up.
VERSION="$(sed -n -E "s/^__version__[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/p" "$REPO_ROOT/jetlink/__init__.py" | head -n 1)"
case "$VERSION" in
  *[!0-9.]* | "" | .* | *. | *..*)
    echo "error: version '$VERSION' is not a plain X.Y.Z the App Store takes" >&2
    exit 1
    ;;
esac
BUILD_NUMBER="${JETLINK_BUILD:-$(git -C "$REPO_ROOT" rev-list --count HEAD)}"
BUNDLE_ID="${JETLINK_BUNDLE_ID:-io.zoompilot.jetlink}"

AUTH=(
  -allowProvisioningUpdates
  -authenticationKeyPath "$ASC_KEY_PATH"
  -authenticationKeyID "$ASC_KEY_ID"
  -authenticationKeyIssuerID "$ASC_ISSUER_ID"
)

echo "==> generating the project"
# with the placeholder version, as make project does: the real one goes to
# xcodebuild below
JETLINK_VERSION=0.0.0 JETLINK_BUILD=1 xcodegen generate --spec "$IOS_DIR/project.yml" --project "$IOS_DIR" --quiet

echo "==> archiving $BUNDLE_ID $VERSION ($BUILD_NUMBER) for team $JETLINK_TEAM"
ARCHIVE="$BUILD/Jetlink.xcarchive"
rm -rf "$ARCHIVE" "$BUILD/export"
xcodebuild archive \
  -project "$IOS_DIR/Jetlink.xcodeproj" \
  -scheme Jetlink \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$BUILD/DerivedData" \
  -archivePath "$ARCHIVE" \
  "${AUTH[@]}" \
  "DEVELOPMENT_TEAM=$JETLINK_TEAM" \
  "PRODUCT_BUNDLE_IDENTIFIER=$BUNDLE_ID" \
  "MARKETING_VERSION=$VERSION" \
  "CURRENT_PROJECT_VERSION=$BUILD_NUMBER"

if [ "${UPLOAD:-1}" = 0 ]; then
  DESTINATION="export"
else
  DESTINATION="upload"
fi
OPTIONS="$BUILD/ExportOptions.plist"
cat >"$OPTIONS" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key>
	<string>app-store-connect</string>
	<key>destination</key>
	<string>$DESTINATION</string>
	<key>teamID</key>
	<string>$JETLINK_TEAM</string>
	<key>signingStyle</key>
	<string>automatic</string>
	<key>uploadSymbols</key>
	<true/>
	<key>manageAppVersionAndBuildNumber</key>
	<false/>
</dict>
</plist>
PLIST

echo "==> exporting ($DESTINATION)"
# The export runs /usr/bin/rsync, which starts its other end as whatever rsync
# is first on PATH; Homebrew's 3.x rejects Apple's -E and the export fails
# with "Copy failed".
PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH" xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$OPTIONS" \
  -exportPath "$BUILD/export" \
  "${AUTH[@]}"

if [ "$DESTINATION" = upload ]; then
  echo "uploaded $VERSION ($BUILD_NUMBER); it shows in TestFlight once App Store Connect has processed it"
else
  echo "exported: $BUILD/export"
fi
