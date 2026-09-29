"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink on an openpilot device: everything between the fork and the comma's
device layer (jetlink.comma).

The fork implements the interface in interface.py once, in an adapter module
that holds every openpilot import, binds it here (bind) and calls the
methods of what comes back. That is the whole contract between the two repos,
and API names its version: the adapter checks it exactly and treats any other
value as jetlink being absent, an offroad alert and no link rather than a
crash.

API changes only for a breaking change to a name exported here or to the
interface. A new Status field, a new Jetlink method, or a new interface member
that jetlink reads with getattr and a fallback, is additive and keeps it.

Imported by the resident gadget owner, so nothing heavy at module level: the
parts a heavy process uses are imported when it first uses them.
"""
from __future__ import annotations

import importlib
import threading
import time
from functools import cached_property

from jetlink.comma import gadget
from jetlink.openpilot.interface import (MODES, STATES, BuildSide, Keys, ModelFace, ModelSide, Openpilot, OwnerConfig,
                                         StatusSide, WorkerSide, conformance)
from jetlink.openpilot.settings import FileParams, Settings
from jetlink.openpilot.status import Status

API = 1

__all__ = ['API', 'MODES', 'STATES', 'BuildSide', 'Jetlink', 'Keys', 'ModelFace', 'ModelSide', 'Openpilot',
           'OwnerConfig', 'Status', 'StatusSide', 'WorkerSide', 'bind', 'conformance', 'load_adapter']

# the chestnut runs the big model natively and the link stays off beside it,
# whatever the setting says. Cached: the UI asks five times a second and the
# answer is a walk of the USB bus
CHESTNUT_TTL = 2.0
# how long hardwared waits for the owner's run to shut the Jetson down. Wake
# from suspend is ~8 s to a server
SHUTDOWN_TIMEOUT = 25.0


def bind(op) -> Jetlink:
  """jetlink for this process, over the fork's adapter. One per process: it
  keeps the process's caches, and points jetlink.comma's log at op.log, so a
  heavy process's lines reach the drive's log while the owner's go to its file."""
  gadget.set_logger(op.log)
  return Jetlink(op)


def load_adapter(module: str):
  """The adapter an adapter module makes: `module.adapter()`. How the entry
  points that run as their own process (the provisioning run, the warp build)
  find the fork's."""
  return importlib.import_module(module).adapter()


class Jetlink:
  """What the fork calls: one per process, from bind()."""

  def __init__(self, op):
    self.op = op
    self._settings: Settings | None = None
    self._chestnut: tuple[float, bool] | None = None
    # prepare() said yes in this process, so attach() may join modeld
    self._prepared = False
    self._status_error: str | None = None

  @property
  def log(self):
    return self.op.log

  # -- the parts, made on first use ----------------------------------------------

  @property
  def settings(self) -> Settings:
    """The link setting and whether the car is parked, off the params files,
    as the owner reads them. Follows the directory the adapter names, which
    moves with OPENPILOT_PREFIX as Params does."""
    directory = self.op.owner().params_dir
    if self._settings is None or self._settings.params.directory != directory:
      self._settings = Settings(FileParams(directory), self.op.keys)
    return self._settings

  @cached_property
  def models(self):
    from jetlink.openpilot.models import Models
    return Models(self.op)

  @cached_property
  def spec(self):
    from jetlink.openpilot.state import SpecRecord
    return SpecRecord(self.op)

  @cached_property
  def progress(self):
    from jetlink.openpilot.status import Progress
    return Progress(self.op, size=lambda: (self.models.selected_model() or {}).get('size'))

  @cached_property
  def presence(self):
    from jetlink.openpilot.status import Presence
    return Presence()

  @cached_property
  def warps(self):
    from jetlink.openpilot.warp import Warps
    return Warps(self.op)

  def chestnut_fitted(self) -> bool:
    now = time.monotonic()
    if self._chestnut is None or now - self._chestnut[0] > CHESTNUT_TTL:
      self._chestnut = (now, bool(self.op.chestnut_present()))
    return self._chestnut[1]

  # -- API 1 ------------------------------------------------------------------------

  def enabled(self) -> bool:
    """Has the user turned the link on, with no chestnut fitted? Configuration
    only, never link state or readiness. A chestnut runs the big model natively
    and the link stays off beside it, so jetlinkd never takes the USB
    controller from it. manager's should_run for jetlinkd."""
    return self.settings.mode() != 'off' and not self.chestnut_fitted()

  def status(self) -> Status:
    """One snapshot for the UI, hardwared and the panels. Never raises: they
    read it on their own threads, which have nothing to do with jetlink."""
    from jetlink.openpilot import status
    try:
      return status.read(self)
    except Exception as e:
      error = f"{type(e).__name__}: {e}"
      if error != self._status_error:
        # once per distinct failure: the UI asks five times a second
        self._status_error = error
        self.log.exception("jetlink: could not read the status")
      return status.failed(error)

  def prepare(self) -> bool:
    """Will the link join this modeld? modeld only, before it goes realtime:
    enabled(), then the process-wide setup that has to happen before then,
    which is also a last veto."""
    from jetlink.openpilot.status import NO_WARP
    from jetlink.openpilot.warp import init_device
    self._prepared = False
    if not self.enabled():
      return False
    # the link is not worth waiting for: attach() joins in the background.
    # enabled() is the setting alone, so this is where a device that cannot
    # present a gadget at all says so; nothing here would ever reach a Jetson
    if not gadget.link_configured():
      self.log.warning("jetlink: no usable gadget (%s), staying on the small model",
                       gadget.gadget_error() or 'not set up')
      return False
    # the warp is a build product and nothing compiles one at runtime, so one
    # missing now stays missing, and saying no keeps modeld on the plain small
    # model; the offroad alert has said why
    if not self.warps.built():
      self.log.warning("jetlink: %s, staying on the small model", NO_WARP)
      return False
    # the last hook before modeld goes SCHED_FIFO on core 7, and the GPU's init
    # spawns a thread that would inherit that. See warp.init_device
    init_device(self.log)
    self._prepared = True
    return True

  def attach(self, small, cam_w: int, cam_h: int):
    """Join the link to modeld, once the camera is up and `small` is built:
    the joining model, which drives as `small` until the Jetson is there.

    None unless prepare() said yes in this process: without it the GPU's
    thread would start on modeld's realtime core. If the joining model cannot
    be built, `small`, and the failure is logged.
    """
    if not self._prepared:
      return None
    from jetlink.openpilot.joining import join
    try:
      return join(self, cam_w, cam_h, small)
    except Exception:
      self.log.exception("jetlink load failed")
      return small

  def shutdown(self, reason: str = '', timeout: float = SHUTDOWN_TIMEOUT) -> None:
    """The device is powering off for good. Tell the Jetson, within `timeout`.

    hardwared calls this before DoShutdown and publishes no deviceState until it
    returns, so the request runs on a thread and is abandoned at the deadline.
    """
    if not self.enabled():
      return

    def request():
      try:
        self._request_shutdown(reason, timeout)
      except Exception:
        self.log.exception("jetlink: shutdown request failed")

    t = threading.Thread(target=request, name='accelerator-shutdown', daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
      self.log.warning("jetlink: shutdown request still pending after %.0f s, going on without it", timeout)

  def _request_shutdown(self, reason: str, timeout: float) -> None:
    """Take the Jetson down with the comma. hardwared cannot touch the link:
    the owner holds the gadget, wakes a sleeping Jetson and starts a
    provisioning run that asks it. Hand the request over and wait; the wake
    and one round trip take ~10 s, and manager will not stop the owner until
    this returns.

    Skipped when no Jetson is known to be there (dormant counts as there). A run
    busy in a long provision will not see the request; the timeout covers that.
    """
    if self.settings.mode() == 'off' or not self.presence.present():
      return
    self.log.warning("jetlink: asking the jetson to power off: %s", reason)
    if not gadget.request_shutdown(reason):
      return
    if self._await_shutdown(timeout):
      self.log.warning("jetlink: shutdown request handed to the jetson")
    else:
      self.log.warning("jetlink: nobody took the shutdown request within %.0f s", timeout)

  @staticmethod
  def _await_shutdown(timeout: float) -> bool:
    """Wait for the owner's run to take the request. False if nobody did in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if not gadget.SHUTDOWN_REQUEST.exists():
        return True
      time.sleep(0.25)
    gadget.finish_shutdown()
    return False

  def extends_catalog(self) -> bool:
    """Should the model manager's big-model catalog carry the newer catalogs'
    models? With no chestnut fitted. Hardware, not the link setting: the model
    manager drops a pick its catalog does not list."""
    return not self.chestnut_fitted()

  def extend_catalog(self, catalog: dict) -> dict:
    """The big-model catalog the model manager fetched, with the models newer
    catalogs list folded in. Never raises."""
    return self.models.big_catalog(catalog)
