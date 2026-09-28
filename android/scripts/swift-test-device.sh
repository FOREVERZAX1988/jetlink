#!/usr/bin/env bash
# Runs JetlinkKit's Swift tests on an Android device or emulator: the same
# suites as on the Mac and Linux, built with the Swift SDK for Android, pushed
# with their fixtures and the libraries they load, and run through adb.
#
#   swift-test-device.sh ONNXRUNTIME_AAR [TARGET...]
#
# TARGET is JetlinkKitTests, JetlinkONNXTests, JetlinkRegistryTests or
# JetlinkServerTests; all four by default. The fixtures are read in place on
# the Mac, so the device gets a copy of those folders and JETLINK_TEST_ROOT
# points the tests at it (Tests/JetlinkTestSupport/SourceTree.swift).
#
# Uses the toolchain, SDK and NDK swift-build.sh finds, and the one adb device.
set -euo pipefail

SWIFT_VERSION=6.4.0
NDK_VERSION=30.0.16248370
TRIPLE=aarch64-unknown-linux-android31
SDK_NAME="swift-$SWIFT_VERSION-RELEASE_android"
DEVICE_DIR=/data/local/tmp/jetlink-tests

AAR=${1:?usage: swift-test-device.sh ONNXRUNTIME_AAR [TARGET...]}
shift
TARGETS=("$@")
[[ ${#TARGETS[@]} -gt 0 ]] || TARGETS=(JetlinkKitTests JetlinkONNXTests JetlinkRegistryTests JetlinkServerTests)

ANDROID_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ANDROID_DIR/.." && pwd)"
BUILD="$ANDROID_DIR/build/swift-tests"
STAGE="$BUILD/device"

fail() {
  echo "swift-test-device.sh: $*" >&2
  exit 1
}

SWIFT=${SWIFT:-}
if [[ -z "$SWIFT" ]]; then
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
[[ -n "$SWIFT" ]] || fail "no Swift $SWIFT_VERSION toolchain"
BUNDLE=""
for root in "$HOME/Library/org.swift.swiftpm/swift-sdks" "$HOME/.swiftpm/swift-sdks" "${XDG_CONFIG_HOME:-$HOME/.config}/swiftpm/swift-sdks"; do
  [[ -d "$root/$SDK_NAME.artifactbundle" ]] && BUNDLE="$root/$SDK_NAME.artifactbundle/swift-android" && break
done
[[ -n "$BUNDLE" ]] || fail "the Swift SDK for Android ($SDK_NAME) is not installed"
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  ANDROID_NDK_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}/ndk/$NDK_VERSION"
fi
export ANDROID_NDK_HOME
[[ -d "$BUNDLE/ndk-sysroot" ]] || bash "$BUNDLE/scripts/setup-android-sdk.sh"
ADB=${ADB:-$(command -v adb || echo "${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools/adb")}

HEADERS="$BUILD/onnxruntime/include"
mkdir -p "$HEADERS/onnxruntime"
unzip -o -q -j "$AAR" 'headers/*' -d "$HEADERS/onnxruntime"

# The tests link the Swift runtime dynamically: swift-testing has no static
# build in the SDK. Its libraries go to the device beside the runners.
args=(--package-path "$REPO/JetlinkKit" --build-tests --swift-sdk "$TRIPLE" --build-path "$BUILD" -Xcc "-I$HEADERS")
JETLINK_ANDROID=1 "$SWIFT" build "${args[@]}"
PRODUCTS=$(JETLINK_ANDROID=1 "$SWIFT" build "${args[@]}" --show-bin-path)

rm -rf "$STAGE"
mkdir -p "$STAGE/repo/JetlinkKit/Tests" "$STAGE/repo/tests"
for target in "${TARGETS[@]}"; do
  cp "$PRODUCTS/$target-test-runner" "$PRODUCTS/$target.so" "$STAGE/"
done
cp -R "$PRODUCTS"/*.bundle "$STAGE/" 2>/dev/null || true
cp "$BUNDLE"/swift-resources/usr/lib/swift-aarch64/android/*.so "$STAGE/"
cp "$(echo "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so)" "$STAGE/"
unzip -o -q -j "$AAR" 'jni/arm64-v8a/libonnxruntime.so' -d "$STAGE"
for fixtures in JetlinkServerTests JetlinkONNXTests; do
  mkdir -p "$STAGE/repo/JetlinkKit/Tests/$fixtures"
  cp -R "$REPO/JetlinkKit/Tests/$fixtures/Fixtures" "$STAGE/repo/JetlinkKit/Tests/$fixtures/"
done
cp -R "$REPO/tests/fixtures" "$STAGE/repo/tests/"

"$ADB" shell rm -rf "$DEVICE_DIR"
"$ADB" shell mkdir -p "$DEVICE_DIR"
"$ADB" push "$STAGE/." "$DEVICE_DIR/" >/dev/null

status=0
for target in "${TARGETS[@]}"; do
  echo "== $target"
  "$ADB" shell "cd $DEVICE_DIR && chmod +x $target-test-runner && JETLINK_TEST_ROOT=$DEVICE_DIR/repo LD_LIBRARY_PATH=. ./$target-test-runner --testing-library swift-testing" || status=1
done
exit $status
