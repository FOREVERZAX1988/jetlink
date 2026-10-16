#!/usr/bin/env bash
#
# Copyright (c) 2026-, Zeph Leggett.
# This file is part of jetlink and is licensed under the MIT License.
#
# The root script is jetlink/comma/jetlink-root.sh now. This forwards to it for
# one release, for an owner still running the jetlink from before the move (it
# names this path) while the checkout under it updates. Remove after v0.9.x.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../jetlink/comma/jetlink-root.sh" "$@"
