#!/usr/bin/env bash
# Fetches the part of commaai/openpilot that jetlink/openpilot/stock.py and
# the modeld seam rely on, as data for check_paths.py and check_imports.py.
# About 2.5 MB and a few seconds: one commit, blobs only for the sparse paths,
# LFS never smudged and modeld/models/ left out (with either missing, the same
# checkout pulls 1.7 GB, the chestnut's big model among it).
#
#   scripts/upstream/fetch.sh <dest> [<sha or branch>]
#
# The revision defaults to the pin in UPSTREAM_SHA. The checkout is read, never
# run: no hook, filter or script of it executes.
set -euo pipefail

dest=$1
rev=${2:-$(tr -d '[:space:]' < "$(dirname "$0")/UPSTREAM_SHA")}
url=${UPSTREAM_URL:-https://github.com/commaai/openpilot.git}
export GIT_LFS_SKIP_SMUDGE=1

rm -rf "$dest"
git init -q "$dest"
git -C "$dest" remote add origin "$url"
git -C "$dest" sparse-checkout set --no-cone \
  '/openpilot/common/' \
  '/openpilot/selfdrive/modeld/' '!/openpilot/selfdrive/modeld/models/' \
  '/openpilot/system/camerad/cameras/' \
  '/openpilot/cereal/'
git -C "$dest" fetch -q --depth 1 --filter=blob:none origin "$rev"
git -C "$dest" checkout -q FETCH_HEAD
git -C "$dest" rev-parse HEAD
