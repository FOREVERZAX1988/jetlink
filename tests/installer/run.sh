#!/usr/bin/env bash
# Run the installer scenarios (scenarios.sh) in throwaway Ubuntu 24.04 and
# 22.04 containers. Needs Docker; changes nothing on this machine.
#
#   tests/installer/run.sh            both releases
#   tests/installer/run.sh 24.04      one
#   SHOW_OUTPUT=1 tests/installer/run.sh    every scenario's output, not only a failure's
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

status=0
for release in "$@"; do
  image="jetlink-installer-test:$release"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    printf 'FROM ubuntu:%s\nRUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*\n' "$release" \
      | docker build -q -t "$image" - >/dev/null
  fi
  docker run --rm -e SHOW_OUTPUT -v "$tree:/src:ro" -v "$old:/releases:ro" "$image" bash /src/tests/installer/scenarios.sh || status=1
done
exit "$status"
