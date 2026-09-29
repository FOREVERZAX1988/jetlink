#!/usr/bin/env bash
# Run the installer scenarios (scenarios.sh) in throwaway Ubuntu 24.04 and
# 22.04 containers. Needs Docker; changes nothing on this machine.
#
#   tests/installer/run.sh            both releases
#   tests/installer/run.sh 24.04      one
#   SHOW_OUTPUT=1 tests/installer/run.sh    every scenario's output, not only a failure's
#   JETLINK_REAL_SERVER=dist/jetlink-server-<version>-linux-aarch64.tar.gz tests/installer/run.sh
#       also installs and runs that real server (scripts/build-linux.sh makes
#       it), in a container of its processor: linux-x86_64 runs in an amd64
#       one, emulated on an Arm Mac. Several tarballs, space separated, run
#       one after another.
#
# The tree under test is what git tracks plus new files it does not ignore, so
# uncommitted work is tested and the multi-gigabyte model caches stay out. The
# releases that ran the server in Docker come from their tags (fetched when a
# shallow clone lacks them), so the moves from them run their real installers.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ $# -gt 0 ] || set -- 24.04 22.04
OLD_RELEASES="v0.4.3 v0.5.0 v0.6.0"

tree="$(mktemp -d)"
old="$(mktemp -d)"
trap 'rm -rf "$tree" "$old"' EXIT
git ls-files -z --cached --others --exclude-standard | xargs -0 tar -cf - | tar -xf - -C "$tree"
for tag in $OLD_RELEASES; do
  if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    git fetch -q --depth 1 "$(git remote | head -n 1)" "refs/tags/$tag:refs/tags/$tag" \
      || { echo "run.sh: cannot find release $tag; fetch the tags (git fetch --tags)" >&2; exit 1; }
  fi
  mkdir -p "$old/$tag"
  git archive "$tag" install.sh scripts docker | tar -xf - -C "$old/$tag"
done

# a run per real server named, else one with the stand-ins alone
runs=()
for tarball in ${JETLINK_REAL_SERVER:-}; do
  [ -f "$tarball" ] || { echo "run.sh: no such server tarball: $tarball" >&2; exit 1; }
  runs+=("$(cd "$(dirname "$tarball")" && pwd)/$(basename "$tarball")")
done
[ ${#runs[@]} -gt 0 ] || runs=('')

status=0
for real in "${runs[@]}"; do
  platform='' mount='' packages=git tag=jetlink-installer-test
  if [ -n "$real" ]; then
    case "$real" in
      *-linux-aarch64.tar.gz) platform=linux/arm64 ;;
      *-linux-x86_64.tar.gz) platform=linux/amd64 ;;
      *) echo "run.sh: not a linux-aarch64 or linux-x86_64 server tarball: $real" >&2; exit 1 ;;
    esac
    # the real server needs libcurl, and the stand-in GPU a compiler
    packages="git libcurl4 gcc libc6-dev" tag="jetlink-installer-test-real-${platform#linux/}"
    mount="$real:/real/$(basename "$real"):ro"
  fi
  for release in "$@"; do
    image="$tag:$release"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      printf 'FROM ubuntu:%s\nRUN apt-get update && apt-get install -y --no-install-recommends %s && rm -rf /var/lib/apt/lists/*\n' \
        "$release" "$packages" | docker build -q ${platform:+--platform "$platform"} -t "$image" - >/dev/null
    fi
    docker run --rm ${platform:+--platform "$platform"} -e SHOW_OUTPUT -v "$tree:/src:ro" -v "$old:/releases:ro" \
      ${mount:+-v "$mount"} "$image" bash /src/tests/installer/scenarios.sh || status=1
  done
done
exit "$status"
