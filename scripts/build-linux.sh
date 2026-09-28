#!/usr/bin/env bash
# Builds jetlink for Linux: the TensorRT shim, its selftest, and the server
# tarball. One entry point for CI, local containers and a build on a device.
#
#   scripts/build-linux.sh [--container] <aarch64|x86_64> [step...]
#
# Steps, run in the order given (default: server):
#   headers    fetch and unpack the flavor's pinned TensorRT + CUDA headers
#   shim       compile CTrt/jl_trt.cpp against them, and check it links
#              nothing of NVIDIA's (everything is dlopened at run time)
#   selftest   link tools/jl_trt_selftest against the shim; run it on a GPU
#   fake-test  build and run the fake shim's tests, and the selftest on it
#   server     swift build -c release --static-swift-stdlib, then
#              dist/jetlink-server-<version>-linux-<flavor>.tar.gz + .sha256
#
# aarch64 is TensorRT 10 for the Jetson, compiled against 10.3 (JetPack 6.2's,
# the oldest it runs on); x86_64 is TensorRT 11.3 for PCs. --container runs
# the steps in a container of the flavor's platform (docker, else podman):
# ubuntu:22.04 for the C and C++ steps, $JETLINK_SWIFT_IMAGE (swift:6.3-jammy)
# for the server. Downloads are cached in $JETLINK_BUILD_CACHE, by default
# ${XDG_CACHE_HOME:-~/.cache}/jetlink-build: NVIDIA's headers never enter the
# repo. Objects go to build/linux-<flavor>, tarballs to dist/.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${JETLINK_BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/jetlink-build}
CTRT=$ROOT/JetlinkKit/Sources/CTrt
SWIFT_IMAGE=${JETLINK_SWIFT_IMAGE:-swift:6.3-jammy}
C_IMAGE=ubuntu:22.04

die() {
  echo "build-linux: $*" >&2
  exit 1
}

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# --- the header bundles ------------------------------------------------------------
# Only headers: TensorRT's, its ONNX parser's, and the CUDA runtime and driver
# API headers TensorRT's include (cuda.h lives in cudart-dev; cuda-driver-dev
# holds only a stub library). Pinned by URL and sha256.

JETSON=https://repo.download.nvidia.com/jetson/common/pool/main
CUDA_X86=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64

# Sets BUNDLE (its cache directory's name) and DEBS ("url sha256" each).
pins() {
  case $FLAVOR in
  aarch64)
    BUNDLE=trt10.3.0.30-cuda12.6-aarch64
    DEBS=(
      "$JETSON/t/tensorrt/libnvinfer-headers-dev_10.3.0.30-1+cuda12.5_arm64.deb 40a4fa566218f71176144a0eafa5aef8cf0af7ee9211b20487f1970d5344bbb8"
      "$JETSON/t/tensorrt/libnvonnxparsers-dev_10.3.0.30-1+cuda12.5_arm64.deb 8a6675f3d444dead0e0430fb4e7e6b546af9d9a72414ee8964b18edf1fb28756"
      "$JETSON/c/cuda-cudart/cuda-cudart-dev-12-6_12.6.68-1_arm64.deb 2ddcaeec93f2c508533f52bea7ad3f339d223d8724537e4695cf14a034b65193"
      "$JETSON/c/cuda-nvcc/cuda-crt-12-6_12.6.68-1_arm64.deb b95d26f0e63f113e6e2f5b8578841644de549293679d65c70e112164d2080311"
    )
    ;;
  x86_64)
    # 11.x keeps NvOnnxParser.h in libnvonnxparsers-dev, not the headers package
    BUNDLE=trt11.3.0.99-cuda13.4-x86_64
    DEBS=(
      "$CUDA_X86/libnvinfer-headers-dev_11.3.0.99-1+cuda13.4_amd64.deb 6383e49c3753e6ed8d3bdb95c0c6607d92d259b4e923407e817b5b4e0c483324"
      "$CUDA_X86/libnvonnxparsers-dev_11.3.0.99-1+cuda13.4_amd64.deb dc8c998f95b3fd9deae65d353650562fa12fefa9f43267facaa9aa057384b0ec"
      "$CUDA_X86/cuda-cudart-dev-13-4_13.4.92-1_amd64.deb 71bf3d001f2560a4710bdaba9297fcd3f775a50e3b320ec1c5fcc880d458b0df"
      "$CUDA_X86/cuda-crt-13-4_13.4.92-1_amd64.deb ec05400c48dafc7d2186f34c3b000596513895aa8d3c2d84446385c6d51f6d49"
    )
    ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# A deb's files into $2: dpkg-deb where there is one, else bsdtar (macOS),
# which reads the ar archive a deb is.
unpack_deb() {
  if command -v dpkg-deb >/dev/null; then
    dpkg-deb -x "$1" "$2"
  else
    tar -xOf "$1" 'data.tar.*' | tar -xJf - -C "$2"
  fi
}

bundle_dir() {
  echo "$CACHE/$BUNDLE"
}

step_headers() {
  local dir stamp
  dir=$(bundle_dir)
  stamp=$(printf '%s\n' "${DEBS[@]}")
  if [[ -f $dir/.complete && $(cat "$dir/.complete") == "$stamp" ]]; then
    echo "headers: $dir/include"
    return
  fi
  mkdir -p "$CACHE/debs"
  rm -rf "$dir" "$dir.tmp"
  mkdir -p "$dir.tmp" "$dir/include"
  local entry url want file
  for entry in "${DEBS[@]}"; do
    url=${entry% *}
    want=${entry#* }
    file=$CACHE/debs/$(basename "$url")
    if [[ ! -f $file || $(sha256 "$file") != "$want" ]]; then
      echo "headers: fetching $(basename "$url")"
      curl -fsSL --retry 3 -o "$file.part" "$url"
      mv "$file.part" "$file"
    fi
    [[ $(sha256 "$file") == "$want" ]] || die "$(basename "$url") does not match its pinned sha256"
    unpack_deb "$file" "$dir.tmp"
  done
  # one flat include dir: TensorRT's from usr/include/<triplet>, CUDA's from
  # usr/local/cuda-*/targets/<triplet>/include
  cp "$dir.tmp"/usr/include/*-linux-gnu/*.h "$dir/include/"
  cp -R "$dir.tmp"/usr/local/cuda-*/targets/*/include/. "$dir/include/"
  rm -rf "$dir.tmp"
  [[ -f $dir/include/NvInfer.h && -f $dir/include/NvOnnxParser.h && -f $dir/include/cuda.h &&
    -f $dir/include/cuda_runtime_api.h ]] || die "the $FLAVOR bundle is missing a header"
  echo "$stamp" >"$dir/.complete"
  echo "headers: $dir/include"
}

# --- the shim -----------------------------------------------------------------------

native_or_die() {
  local arch
  arch=$(uname -m)
  [[ $arch == arm64 ]] && arch=aarch64
  if [[ $arch != "$FLAVOR" && -z ${CXX:-} ]]; then
    die "this is $arch: build $FLAVOR with --container, or set CXX and CC to a cross compiler"
  fi
  [[ $(uname -s) == Linux ]] || die "the shim builds on Linux; use --container"
}

step_shim() {
  native_or_die
  local out=$ROOT/build/linux-$FLAVOR inc
  inc=$(bundle_dir)/include
  [[ -f $(bundle_dir)/.complete ]] || step_headers
  mkdir -p "$out"
  local cxx=${CXX:-g++}
  echo "shim: $cxx $($cxx -dumpversion) against $(basename "$(bundle_dir)")"
  "$cxx" -std=c++17 -O2 -fPIC -Wall -Wextra -Werror -I"$CTRT/include" -isystem "$inc" \
    -c "$CTRT/jl_trt.cpp" -o "$out/jl_trt.o"
  # Everything of NVIDIA's comes through dlopen: a reference here would make
  # the binary need TensorRT or CUDA just to start.
  local leaks
  leaks=$(nm -u "$out/jl_trt.o" | awk '{print $NF}' | grep -E '^cu[A-Z]|nvinfer|nvonnx|createInfer|getInferLib|initLibNvInfer' || true)
  [[ -z $leaks ]] || die "jl_trt.o references NVIDIA symbols directly: $leaks"
  echo "shim: $out/jl_trt.o, no NVIDIA symbol referenced"
}

step_selftest() {
  native_or_die
  local out=$ROOT/build/linux-$FLAVOR
  [[ -f $out/jl_trt.o ]] || step_shim
  local cc=${CC:-gcc} cxx=${CXX:-g++}
  "$cc" -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -I"$CTRT/include" -c "$CTRT/tools/jl_trt_selftest.c" \
    -o "$out/jl_trt_selftest.o"
  "$cxx" "$out/jl_trt_selftest.o" "$out/jl_trt.o" -ldl -pthread -o "$out/jl_trt_selftest"
  local needed
  needed=$(readelf -d "$out/jl_trt_selftest" | awk '/NEEDED/ {gsub(/[][]/, "", $NF); print $NF}' | tr '\n' ' ')
  echo "selftest: $out/jl_trt_selftest (needs: $needed)"
  if echo "$needed" | grep -qE 'nvinfer|nvonnx|cuda'; then
    die "jl_trt_selftest links against NVIDIA libraries: $needed"
  fi
}

step_fake_test() {
  local out cc=${CC:-cc}
  out=$ROOT/build/fake-$(uname -s)-$(uname -m)
  mkdir -p "$out"
  local flags=(-std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror -I"$CTRT/include")
  if [[ ${JETLINK_SANITIZE:-1} == 1 ]]; then
    flags+=("-fsanitize=address,undefined" -fno-omit-frame-pointer)
  fi
  "$cc" "${flags[@]}" "$CTRT/jl_trt_fake.c" "$CTRT/tools/jl_trt_fake_test.c" -pthread -o "$out/jl_trt_fake_test"
  "$cc" "${flags[@]}" -DJL_TRT_FAKE "$CTRT/jl_trt_fake.c" "$CTRT/tools/jl_trt_selftest.c" -pthread \
    -o "$out/jl_trt_selftest_fake"
  echo "fake-test: $(uname -s) $(uname -m), $("$cc" --version | head -1)"
  "$out/jl_trt_fake_test" "$out"
  TMPDIR=$out "$out/jl_trt_selftest_fake"
}

# --- the server ---------------------------------------------------------------------

step_server() {
  [[ $(uname -s) == Linux ]] || die "the server builds on Linux; use --container"
  [[ -f $(bundle_dir)/.complete ]] || step_headers
  local version name stage
  version=$(sed -n "s/^__version__ = '\(.*\)'$/\1/p" "$ROOT/jetlink/__init__.py")
  [[ -n $version ]] || die "no __version__ in jetlink/__init__.py"
  name=jetlink-server-$version-linux-$FLAVOR
  JETLINK_TENSORRT=$(bundle_dir)/include swift build --package-path "$ROOT/JetlinkKit" -c release \
    --static-swift-stdlib --product jetlink-server
  local bin
  bin=$(swift build --package-path "$ROOT/JetlinkKit" -c release --show-bin-path)
  stage=$ROOT/dist/$name
  rm -rf "$stage"
  mkdir -p "$stage/bin" "$stage/share/jetlink/systemd" "$stage/share/jetlink/udev" "$stage/share/jetlink/web"
  cp "$bin/jetlink-server" "$stage/bin/"
  # SwiftPM looks for a target's resources in a bundle beside the executable
  find "$bin" -maxdepth 1 -name '*.resources' -exec cp -R {} "$stage/bin/" \;
  cp "$ROOT"/scripts/*.service "$stage/share/jetlink/systemd/"
  cp "$ROOT"/scripts/*.rules "$stage/share/jetlink/udev/"
  if [[ -d $ROOT/JetlinkKit/Sources/JetlinkStatusPage/Resources ]]; then
    cp -R "$ROOT/JetlinkKit/Sources/JetlinkStatusPage/Resources/." "$stage/share/jetlink/web/"
  fi
  cp "$ROOT/LICENSE" "$stage/"
  echo "$version" >"$stage/VERSION"
  tar -C "$ROOT/dist" -czf "$ROOT/dist/$name.tar.gz" "$name"
  (cd "$ROOT/dist" && sha256 "$name.tar.gz" | sed "s/\$/  $name.tar.gz/" >"$name.tar.gz.sha256")
  rm -rf "$stage"
  echo "server: dist/$name.tar.gz"
}

# --- containers ---------------------------------------------------------------------

container() {
  local image=$1
  shift
  local engine
  engine=$(command -v docker || command -v podman) || die "no docker or podman for --container"
  local platform=linux/arm64
  [[ $FLAVOR == x86_64 ]] && platform=linux/amd64
  local prepare=true
  if [[ $image == "$C_IMAGE" ]]; then
    prepare="apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++ binutils >/dev/null"
  fi
  mkdir -p "$CACHE"
  "$engine" run --rm --platform "$platform" -v "$ROOT:/src" -v "$CACHE:/cache" -e JETLINK_BUILD_CACHE=/cache \
    -e JETLINK_SANITIZE="${JETLINK_SANITIZE:-1}" -w /src "$image" \
    bash -c "$prepare && scripts/build-linux.sh $FLAVOR $*"
}

IN_CONTAINER=0
if [[ ${1:-} == --container ]]; then
  IN_CONTAINER=1
  shift
fi
FLAVOR=${1:-}
[[ $FLAVOR == aarch64 || $FLAVOR == x86_64 ]] || usage
shift
pins
STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(server)

for step in "${STEPS[@]}"; do
  case $step in
  headers | shim | selftest | fake-test | server) ;;
  *) usage ;;
  esac
done

if [[ $IN_CONTAINER == 1 ]]; then
  # the headers come down on this side, where curl is; each run of steps
  # goes to the container its tools are in
  group=()
  group_image=
  for step in "${STEPS[@]}"; do
    case $step in
    headers)
      step_headers
      continue
      ;;
    server) image=$SWIFT_IMAGE ;;
    *) image=$C_IMAGE ;;
    esac
    [[ $step == shim || $step == selftest || $step == server ]] && step_headers >/dev/null
    if [[ -n $group_image && $image != "$group_image" ]]; then
      container "$group_image" "${group[@]}"
      group=()
    fi
    group_image=$image
    group+=("$step")
  done
  if [[ ${#group[@]} -gt 0 ]]; then
    container "$group_image" "${group[@]}"
  fi
  exit 0
fi

for step in "${STEPS[@]}"; do
  "step_${step//-/_}"
done
