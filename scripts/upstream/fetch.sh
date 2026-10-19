#!/usr/bin/env bash
# Fetches the part of commaai/openpilot that jetlink/openpilot/stock.py and
# the modeld seam rely on, as data for check_paths.py and check_imports.py:
# openpilot's sparse paths, and opendbc's car.capnp (CarControl) at the commit
# openpilot's opendbc_repo gitlink names, where a submodule checkout puts it.
# About 2.5 MB and ten seconds: one commit each, blobs only for the sparse
# paths, LFS never smudged and modeld/models/ left out (with either missing,
# the same checkout pulls 1.7 GB, the chestnut's big model among it).
#
#   scripts/upstream/fetch.sh <dest> [<sha or branch>]
#
# The revision defaults to the pin in UPSTREAM_SHA. The checkout is read, never
# run: no hook, filter or script of it executes.
set -euo pipefail

dest=$1
rev=${2:-$(tr -d '[:space:]' < "$(dirname "$0")/UPSTREAM_SHA")}
url=${UPSTREAM_URL:-https://github.com/commaai/openpilot.git}
opendbc_url=${OPENDBC_URL:-https://github.com/commaai/opendbc.git}
export GIT_LFS_SKIP_SMUDGE=1

rm -rf "$dest"
git init -q "$dest"
git -C "$dest" remote add origin "$url"
git -C "$dest" sparse-checkout set --no-cone \
  '/openpilot/common/' \
  '/openpilot/selfdrive/modeld/' '!/openpilot/selfdrive/modeld/models/' \
  '/openpilot/system/camerad/cameras/' \
  '/openpilot/cereal/'
git -C "$dest" fetch -q --depth 1 --filter=blob:none -- origin "$rev"
git -C "$dest" checkout -q FETCH_HEAD

# the gitlink is in the tree, which the blob filter keeps
opendbc=$(git -C "$dest" ls-tree HEAD opendbc_repo | awk '$2 == "commit" { print $3 }')
if [ -n "$opendbc" ]; then
  git init -q "$dest/opendbc_repo"
  git -C "$dest/opendbc_repo" remote add origin "$opendbc_url"
  git -C "$dest/opendbc_repo" sparse-checkout set --no-cone '/opendbc/car/car.capnp'
  git -C "$dest/opendbc_repo" fetch -q --depth 1 --filter=blob:none -- origin "$opendbc"
  git -C "$dest/opendbc_repo" checkout -q FETCH_HEAD
else
  echo "no opendbc_repo gitlink: CarControl is not checked" >&2
fi
git -C "$dest" rev-parse HEAD
