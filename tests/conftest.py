"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
import pytest

from jetlink.comma import wifi
from tests.aio_fakes import FakeAio


@pytest.fixture(autouse=True)
def fake_aio(request, monkeypatch):
  """Gadget writes go through FakeAio: only Linux has the real one, and only a
  comma has the endpoint it was made for. tests/test_aio.py, marked real_aio,
  runs the real one on Linux."""
  if request.node.get_closest_marker('real_aio'):
    return
  from jetlink.transport import ffs
  monkeypatch.setattr(ffs, '_open_aio', FakeAio)


@pytest.fixture(autouse=True)
def isolated_wifi_state(tmp_path, monkeypatch):
  """No test may write the device's real Wi-Fi state files.

  ``link.open_link`` remembers the host it dialed for real, and a mocked client
  hands it a Mock: before this fixture that wrote
  ``/data/.jetlink_wifi_peer`` with ``<Mock name='mock.peer_host'>`` as the host,
  which the next join would then have dialed (seen 2026-10-22, after a suite run
  clobbered the real 192.168.10.65 entry). The blocklist is isolated for the same
  reason: it is the person's list, not the suite's.
  """
  monkeypatch.setattr(wifi, 'PEER_FILE', tmp_path / 'peer')
  monkeypatch.setattr(wifi, 'BLOCK_FILE', tmp_path / 'blocked')
  wifi.forget_unusable()
  yield
  wifi.forget_unusable()
