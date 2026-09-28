"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink: run openpilot's large driving models on an attached Jetson.
"""
# The one place the version is written: pyproject.toml reads it from here, and
# so does the release check (macos/scripts/check-version.sh). The comma runs
# jetlink from a checkout, not an install, so this is also all it has.
__version__ = '0.5.0'
