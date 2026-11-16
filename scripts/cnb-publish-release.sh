#!/usr/bin/env bash
#
# Publishes build files as a CNB (cnb.cool) release, the update feed the
# Android app checks (android/.../update/CnbReleases.kt). Companion to
# cn-build-tag.sh: the same "<version>-<MMDD>-<n>" tag, published on the
# CNB repo instead of (or alongside) the GitHub one.
#
# A same-day rerun reuses the existing release and replaces the files,
# exactly like the GitHub "gh release upload --clobber" path.
#
# Usage: cnb-publish-release.sh <tag> <file>...
# Needs: CNB_TOKEN (a CNB access token with repo-contents read/write),
#        CNB_REPO (default mouxancnb/jetlink),
#        TARGET_SHA (optional; the commit the tag points at, falling back
#        to the repo's cn branch when the mirror does not have it yet).
set -euo pipefail

TAG="${1:?usage: cnb-publish-release.sh <tag> <file>...}"
shift
[ $# -ge 1 ] || { echo "error: no files to publish" >&2; exit 1; }
[ -n "${CNB_TOKEN:-}" ] || { echo "error: CNB_TOKEN is not set" >&2; exit 1; }

REPO="${CNB_REPO:-mouxancnb/jetlink}"
API="https://api.cnb.cool/${REPO}"

jsonget() { python3 -c "import json,sys;print(json.load(sys.stdin).get('$1',''))"; }

api() { # api METHOD PATH [JSON-BODY]
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -fsS -X "$method" -H "Authorization: Bearer ${CNB_TOKEN}" \
      -H "Accept: application/json" -H "Content-Type: application/json" \
      -d "$body" "${API}${path}"
  else
    curl -fsS -X "$method" -H "Authorization: Bearer ${CNB_TOKEN}" \
      -H "Accept: application/json" "${API}${path}"
  fi
}

# The release for this tag: reuse it on a rerun, create it otherwise.
RID="$(api GET "/-/releases/tags/${TAG}" 2>/dev/null | jsonget id 2>/dev/null || true)"
if [ -z "$RID" ]; then
  TARGET="${TARGET_SHA:-$(git rev-parse HEAD 2>/dev/null || echo cn)}"
  TITLE="Jetlink cn ${TAG}"
  NOTES="Builds of the cn branch. The Android APK is signed with the release key, so it installs over an earlier build; the app's updater compares this tag."
  RID="$(api POST "/-/releases" \
    "{\"tag_name\":\"${TAG}\",\"name\":\"${TITLE}\",\"prerelease\":false,\"body\":\"${NOTES}\",\"target_commitish\":\"${TARGET}\"}" \
    2>/dev/null | jsonget id 2>/dev/null || true)"
  if [ -z "$RID" ]; then
    # the commit may not exist on the CNB mirror yet; tag the branch instead
    RID="$(api POST "/-/releases" \
      "{\"tag_name\":\"${TAG}\",\"name\":\"${TITLE}\",\"prerelease\":false,\"body\":\"${NOTES}\",\"target_commitish\":\"cn\"}" \
      | jsonget id)"
  fi
fi
[ -n "$RID" ] || { echo "error: could not resolve the CNB release id for ${TAG}" >&2; exit 1; }
echo "cnb release ${TAG}: id ${RID}"

for FILE in "$@"; do
  NAME="$(basename "$FILE")"
  SIZE="$(wc -c < "$FILE" | tr -d '[:space:]')"
  RESP="$(api POST "/-/releases/${RID}/asset-upload-url" \
    "{\"asset_name\":\"${NAME}\",\"size\":${SIZE},\"overwrite\":true}")"
  UPLOAD_URL="$(printf '%s' "$RESP" | jsonget upload_url)"
  VERIFY_URL="$(printf '%s' "$RESP" | jsonget verify_url)"
  [ -n "$UPLOAD_URL" ] && [ -n "$VERIFY_URL" ] || {
    echo "error: no upload url for ${NAME}" >&2; exit 1;
  }
  curl -fsS -X PUT --upload-file "$FILE" "$UPLOAD_URL" >/dev/null
  # confirming finalizes the asset; the verify URL carries its token/path
  curl -fsS -X POST -H "Authorization: Bearer ${CNB_TOKEN}" \
    -H "Accept: application/json" "$VERIFY_URL" >/dev/null
  echo "uploaded ${NAME} (${SIZE} bytes)"
done

echo "done: https://cnb.cool/${REPO}/-/releases/tag/${TAG}"
