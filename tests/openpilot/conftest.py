"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A guard: no jetlink.openpilot test may touch the machine's own params store
or device files. A suite run on a comma once cleared the live JetlinkSpec and
turned the accelerator off without a word; a read is harmless, but a test
that reads the live store passes or fails on whatever the device has set.

Many of the paths are read inside `except OSError`, so refusing the access
would only hide it. The guard records every open, listing, socket connect or
root-script run that reaches a forbidden path, and fails the test after it
runs (jetlink.openpilot.testing.Guard, which forks install the same way).
tests/openpilot/fakes.isolate points everything at a temporary directory
instead.
"""
from __future__ import annotations

import pytest

from jetlink.openpilot.testing import install_guard

GUARD = install_guard()


@pytest.fixture(autouse=True)
def nothing_of_the_machines_own():
  GUARD.start()
  try:
    yield
    touched = GUARD.stop()
    assert not touched, f"the test reached the machine's own files: {touched}"
  finally:
    GUARD.stop()
