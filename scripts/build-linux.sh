#!/usr/bin/env bash
# Builds jetlink for Linux: the server tarball the installer takes, the
# TensorRT shim and its selftest. One entry point for CI (plain ubuntu-22.04
# and ubuntu-22.04-arm runners), local Docker or podman, and a device.
#
#   scripts/build-linux.sh [--release] [--container] <linux-aarch64|linux-x86_64> [step...]
#
# Steps, run in the order given (default: server):
#   headers    fetch and unpack the flavor's pinned TensorRT + CUDA headers
#   ort        fetch onnxruntime's pinned CPU tarball for the tests and print
#              its directory: include/ for -Xcc -I, lib/ for LD_LIBRARY_PATH
#   shim       compile CTrt/jl_trt.cpp against them, and check it links
#              nothing of NVIDIA's (everything is dlopened at run time)
#   selftest   link tools/jl_trt_selftest against the shim; run it on a GPU
#   fake-test  build the selftest against the fake shim and run it
#   server     dist/jetlink-server-<version>-<flavor>.tar.gz and .sha256
#
# linux-aarch64 is TensorRT 10 for the Jetson, compiled against 10.3 (JetPack
# 6.2's, the oldest it runs on); linux-x86_64 is TensorRT 11.3 for PCs.
#
# The version is jetlink/__init__.py's __version__ for a release (--release,
# or HEAD tagged v<__version__>), else <__version__>-dev.<short sha>.
#
# A step runs where its tools are: `server` in $JETLINK_SWIFT_IMAGE
# (swift:6.3.3-jammy, CI's) unless already inside it, the C and C++ steps in
# ubuntu:22.04 unless this is a Linux host of the flavor's architecture.
# --container sends every step but the fetches to a container. Downloads are
# cached in $JETLINK_BUILD_CACHE, by default ${XDG_CACHE_HOME:-~/.cache}/jetlink-build:
# NVIDIA's headers never enter the repo. Objects go to build/, tarballs to dist/.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${JETLINK_BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/jetlink-build}
CTRT=$ROOT/JetlinkKit/Sources/CTrt
SWIFT_IMAGE=${JETLINK_SWIFT_IMAGE:-swift:6.3.3-jammy}
SWIFT_VERSION=6.3.3
C_IMAGE=ubuntu:22.04

die() {
  echo "build-linux: $*" >&2
  exit 1
}

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# --- the header bundles ------------------------------------------------------------
# Only headers: TensorRT's, its ONNX parser's, and the CUDA runtime and driver
# API headers TensorRT's include (cuda.h lives in cudart-dev; cuda-driver-dev
# holds only a stub library). Pinned by URL and sha256.

JETSON=https://repo.download.nvidia.com/jetson/common/pool/main
CUDA_X86=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
ORT_RELEASES=https://github.com/microsoft/onnxruntime/releases/download/v1.29.0

# Sets BUNDLE (its cache directory's name) and DEBS ("url sha256" each), and
# ORT: onnxruntime's official tarball ("url sha256"), whose C headers COrt
# compiles against; the server opens the library at run time.
pins() {
  case $FLAVOR in
  aarch64)
    ORT="$ORT_RELEASES/onnxruntime-linux-aarch64-1.29.0.tgz e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f"
    BUNDLE=trt10.3.0.30-cuda12.6-aarch64
    DEBS=(
      "$JETSON/t/tensorrt/libnvinfer-headers-dev_10.3.0.30-1+cuda12.5_arm64.deb 40a4fa566218f71176144a0eafa5aef8cf0af7ee9211b20487f1970d5344bbb8"
      "$JETSON/t/tensorrt/libnvonnxparsers-dev_10.3.0.30-1+cuda12.5_arm64.deb 8a6675f3d444dead0e0430fb4e7e6b546af9d9a72414ee8964b18edf1fb28756"
      "$JETSON/c/cuda-cudart/cuda-cudart-dev-12-6_12.6.68-1_arm64.deb 2ddcaeec93f2c508533f52bea7ad3f339d223d8724537e4695cf14a034b65193"
      "$JETSON/c/cuda-nvcc/cuda-crt-12-6_12.6.68-1_arm64.deb b95d26f0e63f113e6e2f5b8578841644de549293679d65c70e112164d2080311"
    )
    ;;
  x86_64)
    ORT="$ORT_RELEASES/onnxruntime-linux-x64-1.29.0.tgz c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba"
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

fetch() {
  if command -v curl >/dev/null; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null; then
    wget -q -O "$2" "$1"
  else
    python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
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

ort_dir() {
  echo "$CACHE/$(basename "${ORT% *}" .tgz)"
}

# onnxruntime's headers under include/onnxruntime/, where COrt looks for them,
# and its library under lib/, which the tests open.
ort_headers() {
  local url=${ORT% *} want=${ORT#* } file dir
  dir=$(ort_dir)
  [[ -f $dir/.complete && $(cat "$dir/.complete") == "$ORT lib" ]] && return
  file=$CACHE/debs/$(basename "$url")
  mkdir -p "$CACHE/debs"
  if [[ ! -f $file || $(sha256 "$file") != "$want" ]]; then
    echo "headers: fetching $(basename "$url")" >&2
    fetch "$url" "$file.part"
    mv "$file.part" "$file"
  fi
  [[ $(sha256 "$file") == "$want" ]] || die "$(basename "$url") does not match its pinned sha256"
  rm -rf "$dir"
  mkdir -p "$dir/include/onnxruntime"
  tar -xzf "$file" -C "$dir/include/onnxruntime" --strip-components 2 "$(basename "$url" .tgz)/include"
  tar -xzf "$file" -C "$dir" --strip-components 1 "$(basename "$url" .tgz)/lib"
  echo "$ORT lib" >"$dir/.complete"
}

step_ort() {
  ort_headers
  ort_dir
}

step_headers() {
  local dir=$CACHE/$BUNDLE stamp entry url want file
  ort_headers
  stamp=$(printf '%s\n' "${DEBS[@]}")
  if [[ -f $dir/.complete && $(cat "$dir/.complete") == "$stamp" ]]; then
    echo "headers: $dir/include"
    return
  fi
  mkdir -p "$CACHE/debs"
  rm -rf "$dir" "$dir.tmp"
  mkdir -p "$dir.tmp" "$dir/include"
  for entry in "${DEBS[@]}"; do
    url=${entry% *}
    want=${entry#* }
    file=$CACHE/debs/$(basename "$url")
    if [[ ! -f $file || $(sha256 "$file") != "$want" ]]; then
      echo "headers: fetching $(basename "$url")"
      fetch "$url" "$file.part"
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

step_shim() {
  local out=$ROOT/build/linux-$FLAVOR inc=$CACHE/$BUNDLE/include cxx=${CXX:-g++}
  [[ -f $CACHE/$BUNDLE/.complete ]] || step_headers
  mkdir -p "$out"
  echo "shim: $cxx $($cxx -dumpversion) against $BUNDLE"
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
  local out=$ROOT/build/linux-$FLAVOR cc=${CC:-gcc} cxx=${CXX:-g++}
  [[ -f $out/jl_trt.o ]] || step_shim
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
  local flags=(-std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror -DJL_TRT_FAKE -I"$CTRT/include")
  if [[ ${JETLINK_SANITIZE:-1} == 1 ]]; then
    flags+=("-fsanitize=address,undefined" -fno-omit-frame-pointer -fno-sanitize-recover=all)
  fi
  echo "fake-test: $(uname -s) $(uname -m), $("$cc" --version | head -1)"
  "$cc" "${flags[@]}" "$CTRT/jl_trt_fake.c" "$CTRT/tools/jl_trt_selftest.c" -pthread -lm -o "$out/jl_trt_selftest_fake"
  TMPDIR=$out "$out/jl_trt_selftest_fake"
}

# --- the server ---------------------------------------------------------------------

version() {
  if [[ -n ${JETLINK_VERSION:-} ]]; then
    echo "$JETLINK_VERSION"
    return
  fi
  local base tag sha
  base=$(sed -n "s/^__version__ = '\(.*\)'$/\1/p" "$ROOT/jetlink/__init__.py")
  [[ -n $base ]] || die "no __version__ in jetlink/__init__.py"
  tag=$(git -C "$ROOT" -c safe.directory='*' describe --exact-match --tags HEAD 2>/dev/null || true)
  if [[ $RELEASE == 1 || $tag == "v$base" ]]; then
    echo "$base"
    return
  fi
  sha=$(git -C "$ROOT" -c safe.directory='*' rev-parse --short HEAD 2>/dev/null) ||
    die "no git here to name a dev build: pass --release or set JETLINK_VERSION"
  echo "$base-dev.$sha"
}

step_server() {
  [[ -f $CACHE/$BUNDLE/.complete ]] || step_headers
  local version name stage bin scratch=$ROOT/build/swift-linux-$FLAVOR ort
  ort=$(ort_dir)
  ort_headers
  version=$(version)
  name=jetlink-server-$version-linux-$FLAVOR
  # its own scratch path, so a Mac's JetlinkKit/.build is never touched
  JETLINK_TENSORRT=$CACHE/$BUNDLE/include swift build --package-path "$ROOT/JetlinkKit" --scratch-path "$scratch" \
    -c release --static-swift-stdlib --product jetlink-server -Xcc -I"$ort/include"
  bin=$(swift build --package-path "$ROOT/JetlinkKit" --scratch-path "$scratch" -c release --show-bin-path)
  stage=$ROOT/dist/$name
  rm -rf "$stage"
  mkdir -p "$stage/bin" "$stage/share/jetlink/systemd" "$stage/share/jetlink/udev" "$stage/share/jetlink/web"
  # stripped: the symbol table is a third of the binary
  strip -o "$stage/bin/jetlink-server" "$bin/jetlink-server"
  # SwiftPM looks for a target's resources in a bundle beside the executable
  find "$bin" -maxdepth 1 -name '*.resources' -exec cp -R {} "$stage/bin/" \;
  # the installer's own unit and rules, as they are in this tree
  cp "$ROOT/scripts/jetlink-server.service" "$stage/share/jetlink/systemd/"
  cp "$ROOT"/scripts/99-jetlink-*.rules "$stage/share/jetlink/udev/"
  if [[ -d $ROOT/JetlinkKit/Sources/JetlinkStatusPage/Resources ]]; then
    cp -R "$ROOT/JetlinkKit/Sources/JetlinkStatusPage/Resources/." "$stage/share/jetlink/web/"
  fi
  cp "$ROOT/LICENSE" "$stage/"
  echo "$version" >"$stage/VERSION"
  # no onnxruntime: a TensorRT host needs none
  tar -C "$ROOT/dist" -czf "$ROOT/dist/$name.tar.gz" "$name"
  (cd "$ROOT/dist" && echo "$(sha256 "$name.tar.gz")  $name.tar.gz" >"$name.tar.gz.sha256")
  rm -rf "$stage"
  echo "server: dist/$name.tar.gz"
}

# --- where each step runs -------------------------------------------------------------

in_swift_container() {
  [[ ${JETLINK_IN_CONTAINER:-} == 1 ]] && return 0
  [[ -f /.dockerenv || -f /run/.containerenv ]] && command -v swift >/dev/null &&
    swift --version 2>/dev/null | grep -q "Swift version $SWIFT_VERSION"
}

native_c() {
  local arch
  arch=$(uname -m)
  [[ ${JETLINK_IN_CONTAINER:-} == 1 ]] ||
    { [[ $(uname -s) == Linux && ${arch/arm64/aarch64} == "$FLAVOR" ]] && command -v "${CXX:-g++}" >/dev/null; }
}

# The image a step needs, or "" to run it here.
image_for() {
  case $1 in
  headers | ort) echo "" ;;
  server) in_swift_container && echo "" || echo "$SWIFT_IMAGE" ;;
  fake-test) [[ $FORCE_CONTAINER == 1 ]] && echo "$C_IMAGE" || echo "" ;;
  *) [[ $FORCE_CONTAINER == 0 ]] && native_c && echo "" || echo "$C_IMAGE" ;;
  esac
}

container() {
  local image=$1 engine platform=linux/arm64 prepare=true swift=()
  shift
  engine=$(command -v docker || command -v podman) || die "$* needs docker or podman, or a matching host"
  [[ $FLAVOR == x86_64 ]] && platform=linux/amd64
  if [[ $image == "$C_IMAGE" ]]; then
    prepare="apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++ binutils >/dev/null"
  else
    # Swift needs no packages, so it runs as the caller and leaves files the
    # caller owns; the version comes from here, where git is
    swift=(--user "$(id -u):$(id -g)" -e HOME=/tmp/home -e JETLINK_VERSION="$(version)")
  fi
  mkdir -p "$CACHE"
  # ${a[@]+...}: bash 3.2, the Mac's, calls an empty array unbound
  "$engine" run --rm --platform "$platform" ${swift[@]+"${swift[@]}"} -v "$ROOT:/src" -v "$CACHE:/cache" \
    -e JETLINK_BUILD_CACHE=/cache -e JETLINK_IN_CONTAINER=1 -e JETLINK_SANITIZE="${JETLINK_SANITIZE:-1}" -w /src "$image" \
    bash -c "$prepare && scripts/build-linux.sh linux-$FLAVOR $*"
}

RELEASE=0
FORCE_CONTAINER=0
while [[ ${1:-} == --* ]]; do
  case $1 in
  --release) RELEASE=1 ;;
  --container) FORCE_CONTAINER=1 ;;
  *) usage ;;
  esac
  shift
done
case ${1:-} in
linux-aarch64 | aarch64) FLAVOR=aarch64 ;;
linux-x86_64 | x86_64) FLAVOR=x86_64 ;;
*) usage ;;
esac
shift
pins
STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(server)
for step in "${STEPS[@]}"; do
  case $step in
  headers | ort | shim | selftest | fake-test | server) ;;
  *) usage ;;
  esac
done
# Consecutive steps for the same container go in one run of it.
group=()
group_image=
flush() {
  if [[ ${#group[@]} -gt 0 ]]; then
    container "$group_image" "${group[@]}"
  fi
  group=()
}
for step in "${STEPS[@]}"; do
  image=$(image_for "$step")
  if [[ -z $image ]]; then
    flush
    "step_${step//-/_}"
    continue
  fi
  # the headers come down here, where curl is, before the container needs them
  [[ $step == fake-test ]] || step_headers >/dev/null
  [[ $image == "$group_image" ]] || flush
  group_image=$image
  group+=("$step")
done
flush
