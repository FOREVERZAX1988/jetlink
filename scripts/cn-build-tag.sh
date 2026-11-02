#!/usr/bin/env bash
#
# Build tag for a cn-branch prerelease: "<appVersion>-<MMDD>-<n>", e.g.
# "0.8.7-1015-1" for the first build of October 15 while the app is 0.8.7,
# "0.8.7-1015-4" for the fourth that day, "0.8.8-1016-1" once either bumps.
#
# The date is China time (UTC+8), the team's calendar day. <n> is one more than
# the highest same-day, same-version build already published, so parallel /
# re-run jobs agree without any state in the checkout. Prints the tag only.
#
# Needs: gh authenticated (CI provides GH_TOKEN/GITHUB_TOKEN) or `git` for the
# version. Used by .github/workflows/ci.yml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
REPO="${REPO:-$(git -C "$REPO_ROOT" config --get remote.origin.url \
                 | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')}"

VERSION="$(sed -n -E "s/^__version__[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/p" \
             "$REPO_ROOT/jetlink/__init__.py" | head -n 1)"
[ -n "$VERSION" ] || { echo "error: no __version__ in jetlink/__init__.py" >&2; exit 1; }

# MMDD in UTC+8 regardless of the runner's zone.
DAY="$(TZ=Asia/Shanghai date +%m%d)"
PREFIX="${VERSION}-${DAY}-"

# Highest <n> among already-published builds "<version>-<day>-<n>". gh lists
# newest first; take the max of matching tags so an older rerun cannot lower it.
seq=0
if command -v gh >/dev/null 2>&1; then
  for tag in $(gh release list --repo "$REPO" --limit 100 --json tagName \
                 --jq '.[].tagName' 2>/dev/null); do
    case "$tag" in
      "$PREFIX"*)
        n="${tag#"$PREFIX"}"
        case "$n" in
          ''|*[!0-9]*) ;;  # not a bare sequence number
          *) [ "$n" -gt "$seq" ] && seq="$n" ;;
        esac
        ;;
    esac
  done
fi
seq=$((seq + 1))

echo "${PREFIX}${seq}"
