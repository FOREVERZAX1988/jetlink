#!/usr/bin/env bash
# Builds libjetlink.so, the jetlink server the Android app runs in process:
# JetlinkKit's `jetlink` product, cross-compiled with the Swift SDK for
# Android, the Swift runtime and Foundation linked in statically. Copies it
# and the NDK's libc++_shared.so into OUT/arm64-v8a for the APK.
#
#   swift-build.sh OUT ONNXRUNTIME_AAR [release|debug]
#
# The app's Gradle build runs this (the swiftBuild task) with the
# onnxruntime AAR it resolved; the build needs only the AAR's C headers,
# because the library opens libonnxruntime.so at run time.
#
# Needs the open-source Swift toolchain and the Swift SDK for Android of the
# same version, and the NDK the SDK was built with (android/README.md):
#   SWIFT              the swift binary, if not the one found below
#   ANDROID_NDK_HOME   the NDK, if not $ANDROID_HOME/ndk/$NDK_VERSION
set -euo pipefail

SWIFT_VERSION=6.4.0
NDK_VERSION=30.0.16248370
API=31
ABI=arm64-v8a
TRIPLE=aarch64-unknown-linux-android$API
SDK_NAME="swift-$SWIFT_VERSION-RELEASE_android"

OUT=${1:?usage: swift-build.sh OUT ONNXRUNTIME_AAR [release|debug]}
AAR=${2:?usage: swift-build.sh OUT ONNXRUNTIME_AAR [release|debug]}
CONFIG=${3:-release}

ANDROID_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ANDROID_DIR/.." && pwd)"
BUILD="$ANDROID_DIR/build/swift"

fail() {
  echo "swift-build.sh: $*" >&2
  exit 1
}

# The toolchain: Xcode's swift cannot use a swift.org SDK, so look for the
# open-source one first.
if [[ -z "${SWIFT:-}" ]]; then
  for candidate in \
    "$HOME/Library/Developer/Toolchains/swift-$SWIFT_VERSION-RELEASE.xctoolchain/usr/bin/swift" \
    "$HOME/.local/share/swiftly/toolchains/$SWIFT_VERSION/usr/bin/swift" \
    "$(command -v swift || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      SWIFT=$candidate
      break
    fi
  done
fi
[[ -n "${SWIFT:-}" ]] || fail "no Swift $SWIFT_VERSION toolchain; install it with swiftly (android/README.md)"
"$SWIFT" --version 2>&1 | grep -q "Swift version ${SWIFT_VERSION%.0}" ||
  fail "$SWIFT is not Swift $SWIFT_VERSION (the SDK for Android must match the toolchain exactly)"

# The SDK, and its link to the NDK.
BUNDLE=""
for root in "$HOME/Library/org.swift.swiftpm/swift-sdks" "$HOME/.swiftpm/swift-sdks" "${XDG_CONFIG_HOME:-$HOME/.config}/swiftpm/swift-sdks"; do
  if [[ -d "$root/$SDK_NAME.artifactbundle" ]]; then
    BUNDLE="$root/$SDK_NAME.artifactbundle/swift-android"
    break
  fi
done
[[ -n "$BUNDLE" ]] || fail "the Swift SDK for Android ($SDK_NAME) is not installed (android/README.md)"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  SDK_ROOT=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}
  ANDROID_NDK_HOME="$SDK_ROOT/ndk/$NDK_VERSION"
fi
[[ -d "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" ]] || fail "no NDK at $ANDROID_NDK_HOME (NDK $NDK_VERSION)"
export ANDROID_NDK_HOME
if [[ ! -d "$BUNDLE/ndk-sysroot" ]]; then
  bash "$BUNDLE/scripts/setup-android-sdk.sh"
fi

# onnxruntime's C headers, from the AAR the app ships.
HEADERS="$BUILD/onnxruntime/include"
mkdir -p "$HEADERS/onnxruntime"
unzip -o -q -j "$AAR" 'headers/*' -d "$HEADERS/onnxruntime"

args=(
  --package-path "$REPO/JetlinkKit"
  --product jetlink
  -c "$CONFIG"
  --swift-sdk "$TRIPLE"
  --build-path "$BUILD"
  -Xcc "-I$HEADERS"
  -Xswiftc -static-stdlib
  -Xswiftc -resource-dir -Xswiftc "$BUNDLE/swift-resources/usr/lib/swift_static-aarch64"
  # Static FoundationNetworking's curl uses the SDK's OpenSSL, which nothing
  # else names (nor the zlib it inflates with): without these the library
  # fails to load on X509_free.
  -Xlinker -lssl -Xlinker -lcrypto -Xlinker -lz
)
export JETLINK_ANDROID=1
"$SWIFT" build "${args[@]}"
BIN=$("$SWIFT" build "${args[@]}" --show-bin-path)

mkdir -p "$OUT/$ABI"
cp "$BIN/libjetlink.so" "$OUT/$ABI/libjetlink.so"
cp "$(echo "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so)" "$OUT/$ABI/"
echo "swift-build.sh: $OUT/$ABI/libjetlink.so ($CONFIG)"
