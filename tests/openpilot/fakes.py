"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The openpilot a jetlink test sees: the fork's adapter, faked.

No Params, no cloudlog, no modeld: a dict holds the JSON params, files in a
temporary directory hold the two settings jetlink reads as files, and a list
holds the log. Nothing here can reach a live params store, which is what
cleared JetlinkSpec on a comma once when a suite ran on the device.

The module is also an adapter module, adapter() and owner_config(), for the
entry points that take --adapter; JETLINK_FAKE_ROOT says where its files go.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from jetlink.openpilot.interface import MODES, Keys, ModelFace, OwnerConfig

# the fork's names, so a log line or an assertion reads as it would on a comma
KEYS = Keys(link='JetlinkLink', offroad='IsOffroad', progress='AcceleratorProgress', spec='JetlinkSpec',
            pointers='JetlinkModelPointers', big_model='ModelManager_ActiveBundleChestnut',
            catalog='ModelManager_ModelsCache_Chestnut')
CHESTNUT_IDS = frozenset({(0xADD1, 0x0001), (0x3801, 0x0001), (0x174C, 0x2464), (0x174C, 0x2463)})
# (cam_w, cam_h, model_w, model_h): a comma 3X, and a comma four
TICI = (1928, 1208, 512, 256)
MICI = (1344, 760, 512, 256)


class RecordingLog:
  """cloudlog's face, keeping every line: (level, message with its arguments)."""

  def __init__(self):
    self.records: list[tuple[str, str]] = []

  def _add(self, level: str, msg, *args, **kwargs) -> None:
    text = str(msg)
    if args:
      try:
        text = text % args
      except (TypeError, ValueError):
        text = f"{text} {args}"
    self.records.append((level, text))

  def debug(self, msg, *args, **kwargs):
    self._add('debug', msg, *args)

  def info(self, msg, *args, **kwargs):
    self._add('info', msg, *args)

  def warning(self, msg, *args, **kwargs):
    self._add('warning', msg, *args)

  def error(self, msg, *args, **kwargs):
    self._add('error', msg, *args)

  def exception(self, msg, *args, **kwargs):
    self._add('exception', msg, *args)

  def critical(self, msg, *args, **kwargs):
    self._add('critical', msg, *args)

  def lines(self, level: str | None = None) -> list[str]:
    return [text for lvl, text in self.records if level is None or lvl == level]

  def has(self, fragment: str, level: str | None = None) -> bool:
    return any(fragment in text for text in self.lines(level))


class FakeParser:
  """modeld's Parser, as far as jetlink uses it: the outputs as they came."""

  def parse_outputs(self, outputs: dict) -> dict:
    return dict(outputs)


def nv12_info(width: int, height: int) -> tuple[int, int, int, int]:
  # stride, the Y plane's height, the UV plane's offset and the buffer size,
  # in the layout get_nv12_info returns; jetlink only reads the size
  return width, height, width * height, width * height * 3 // 2


def get_action_from_model(*args):
  return ('action', args)


FACE = ModelFace(parser=FakeParser, nv12_info=nv12_info, desire_len=8,
                 constants=SimpleNamespace(MODEL_FREQ=20, DESIRE_LEN=8), lat_smooth_seconds=0.0,
                 long_smooth_seconds=0.3, get_action_from_model=get_action_from_model, lat_delay=lambda: 0.2,
                 telemetry_every=2)


class FakeOpenpilot:
  """Every side of the interface, over a temporary directory."""

  def __init__(self, root: Path | None = None, *, camera=TICI, chestnut: bool = False, catalog_selector: int = 19,
               keys: Keys = KEYS):
    self.root = Path(root if root is not None else tempfile.mkdtemp())
    self.keys = keys
    self.log = RecordingLog()
    self.catalog_selector = catalog_selector
    self.basedir = self.root / 'openpilot'
    self.basedir.mkdir(parents=True, exist_ok=True)
    self.params_dir = self.root / 'params' / 'd'
    self.params_dir.mkdir(parents=True, exist_ok=True)
    self.store: dict[str, object] = {}
    self.events: list[tuple[str, dict]] = []
    self.chestnut = chestnut
    self.geometry = camera
    self.engaged = True
    self.put_error: Exception | None = None
    self.face = FACE
    self.worker = (sys.executable, '-m', 'jetlink.openpilot.provision', '--adapter', __name__)

  # -- the readers ------------------------------------------------------------

  def get(self, key: str):
    # a round trip through JSON, as Params hands back a fresh decoded value
    value = self.store.get(key)
    return None if value is None else json.loads(json.dumps(value))

  def owner(self) -> OwnerConfig:
    return OwnerConfig(params_dir=self.params_dir, keys=self.keys, chestnut_ids=CHESTNUT_IDS, worker=self.worker,
                       cwd=str(self.basedir), env={'PYTHONPATH': str(self.basedir)},
                       log_file=self.root / 'jetlink-owner.log')

  def chestnut_present(self) -> bool:
    return self.chestnut

  def camera(self) -> tuple[int, int, int, int]:
    return self.geometry

  def warp_path(self, cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
    return self.root / 'warps' / f'warp_{cam_w}x{cam_h}_{model_w}x{model_h}_tinygrad.pkl'

  # -- the provisioning run ----------------------------------------------------

  def put(self, key: str, value, block: bool = False) -> None:
    if self.put_error is not None:
      raise self.put_error
    self.store[key] = json.loads(json.dumps(value))

  def remove(self, key: str) -> None:
    self.store.pop(key, None)

  def event(self, name: str, **fields) -> None:
    self.events.append((name, fields))

  def model_dir(self) -> Path:
    return self.root / 'models' / 'jetlink'

  # -- modeld ------------------------------------------------------------------

  def model_face(self) -> ModelFace:
    return self.face

  def engagement(self):
    def engaged(timeout_ms: int) -> bool:
      # as SubMaster.update does, wait for news; a poller that returned at
      # once would spin the thread that calls it
      time.sleep(min(timeout_ms, 20) / 1000)
      return self.engaged
    return engaged

  # -- the build ---------------------------------------------------------------

  def make_warp(self, cam_w: int, cam_h: int, model_w: int, model_h: int):
    def warp(tfm, big_tfm, frame, big_frame):
      return SimpleNamespace(tfm=tfm, big_tfm=big_tfm, frame=frame, big_frame=big_frame)
    return warp, nv12_info(cam_w, cam_h)[3]

  # -- what a test sets --------------------------------------------------------

  def set_mode(self, mode: str | None) -> None:
    """The Accelerator Link setting as the panels write it; None unsets it."""
    path = self.params_dir / self.keys.link
    if mode is None:
      path.unlink(missing_ok=True)
    else:
      path.write_bytes(str(MODES.index(mode)).encode())

  def set_offroad(self, parked: bool | None) -> None:
    path = self.params_dir / self.keys.offroad
    if parked is None:
      path.unlink(missing_ok=True)
    else:
      path.write_bytes(b'1' if parked else b'0')


def _root() -> Path:
  return Path(os.environ.get('JETLINK_FAKE_ROOT') or tempfile.mkdtemp())


def adapter() -> FakeOpenpilot:
  return FakeOpenpilot(_root())


def owner_config() -> OwnerConfig:
  return adapter().owner()
