"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink on an openpilot device: everything between the fork and the comma's
device layer (jetlink.comma).

The fork implements Openpilot (interface.py) once, in an adapter module that
holds every openpilot import, binds it here (bind) and calls the methods of
the Jetlink that comes back. That is the whole contract between the two repos:
the names in __all__, the Jetlink methods and Status, which
tests/openpilot/test_api.py and test_interface.py pin. API names its version:
the adapter checks it exactly and treats any other value as jetlink being
absent, an offroad alert and no link rather than a crash. Everything else in
this package is jetlink's own and changes without notice.

API changes only for a breaking change to the contract. A new Status field, a
new Jetlink method, or a new interface member that jetlink reads with getattr
and a fallback, is additive and keeps it.

Imported by the resident gadget owner, so nothing heavy at module level: the
parts a heavy process uses are imported when it first uses them.
"""
from __future__ import annotations

import threading
import time

from jetlink.comma import gadget
from jetlink.openpilot.interface import MODES, STATES, Keys, ModelFace, Openpilot, OwnerConfig, conformance
from jetlink.openpilot.parts import Parts, for_this_process
from jetlink.openpilot.status import Status

API = 1

__all__ = ['API', 'MODES', 'STATES', 'Jetlink', 'Keys', 'ModelFace', 'Openpilot', 'OwnerConfig', 'Status', 'bind',
           'conformance']

# how long hardwared waits for the owner's run to shut the Jetson down. Wake
# from suspend is ~8 s to a server
SHUTDOWN_TIMEOUT = 25.0


def bind(op) -> Jetlink:
  """jetlink for this process, over the fork's adapter. One per process: it
  keeps the process's caches, and points jetlink.comma's log at op.log, so a
  heavy process's lines reach the drive's log while the owner's go to its file."""
  return Jetlink(for_this_process(op))


class Jetlink:
  """What the fork calls: one per process, from bind(). Its public methods are
  the API; the parts behind it are not."""

  def __init__(self, parts: Parts):
    self._parts = parts
    self._log = parts.log
    # prepare() said yes in this process, so attach() may join modeld
    self._prepared = False
    # what -> the last failure logged reading it; cleared by a read that works
    self._failures: dict[str, str] = {}

  def _failed(self, what: str, e: Exception) -> str:
    """Log a reader's failure once per distinct error: the readers are asked
    several times a second, from threads that have nothing to do with jetlink."""
    error = f"{type(e).__name__}: {e}"
    if self._failures.get(what) != error:
      self._failures[what] = error
      self._log.exception("jetlink: could not read %s", what)
    return error

  def _worked(self, what: str) -> None:
    self._failures.pop(what, None)

  def _mode(self) -> str:
    """The link setting, or 'off' when it cannot be read."""
    try:
      mode = self._parts.settings.mode()
    except Exception as e:
      self._failed('the link setting', e)
      return 'off'
    self._worked('the link setting')
    return mode

  def enabled(self) -> bool:
    """Has the user turned the link on, with no chestnut fitted? Configuration
    only, never link state or readiness. A chestnut runs the big model natively
    and the link stays off beside it, so jetlinkd never takes the USB
    controller from it. manager's should_run for jetlinkd, on every device, so
    it never raises: a failure is False."""
    mode = self._mode()
    try:
      on = self._parts.enabled(mode)
    except Exception as e:
      self._failed('whether the link is on', e)
      return False
    self._worked('whether the link is on')
    return on

  def status(self) -> Status:
    """One snapshot for the UI and the panels. Never raises: they read it on
    their own threads, which have nothing to do with jetlink."""
    from jetlink.openpilot import status
    mode = self._mode()
    try:
      snapshot = status.read(self._parts, mode)
    except Exception as e:
      return status.failed(self._failed('the status', e), mode)
    self._worked('the status')
    return snapshot

  def reason(self) -> str | None:
    """Why the link the user asked for cannot run: hardwared's offroad alert.
    None with the link off. The files the gadget and the build leave, nothing
    else: hardwared asks twice a second on every device. Never raises."""
    from jetlink.openpilot import status
    mode = self._mode()
    try:
      why = status.reason(self._parts, mode)
    except Exception as e:
      return status.failure(self._failed('why the link cannot run', e), mode)
    self._worked('why the link cannot run')
    return why

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
      self._log.warning("jetlink: no usable gadget (%s), staying on the small model",
                        gadget.gadget_error() or 'not set up')
      return False
    # the warp is a build product and nothing compiles one at runtime, so one
    # missing now stays missing, and saying no keeps modeld on the plain small
    # model; the offroad alert has said why
    if not self._parts.warps.built():
      self._log.warning("jetlink: %s, staying on the small model", NO_WARP)
      return False
    # the last hook before modeld goes SCHED_FIFO on core 7, and the GPU's init
    # spawns a thread that would inherit that. See warp.init_device
    init_device(self._log)
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
      return join(self._parts, cam_w, cam_h, small)
    except Exception:
      self._log.exception("jetlink load failed")
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
        self._log.exception("jetlink: shutdown request failed")

    t = threading.Thread(target=request, name='accelerator-shutdown', daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
      self._log.warning("jetlink: shutdown request still pending after %.0f s, going on without it", timeout)

  def _request_shutdown(self, reason: str, timeout: float) -> None:
    """Take the Jetson down with the comma. hardwared cannot touch the link:
    the owner holds the gadget, wakes a sleeping Jetson and starts a
    provisioning run that asks it. Hand the request over and wait; the wake
    and one round trip take ~10 s, and manager will not stop the owner until
    this returns.

    Skipped when no Jetson is known to be there (dormant counts as there). A run
    busy in a long provision will not see the request; the timeout covers that.
    """
    from jetlink.openpilot.status import Presence
    # a presence of its own: one this process's readers keep refreshed would
    # count a Jetson seen seconds before the power-off, and the run would wait
    # 20 s for a host that has gone
    if self._parts.settings.mode() == 'off' or not Presence().present():
      return
    self._log.warning("jetlink: asking the jetson to power off: %s", reason)
    if not gadget.request_shutdown(reason):
      return
    if self._await_shutdown(timeout):
      self._log.warning("jetlink: shutdown request handed to the jetson")
    else:
      self._log.warning("jetlink: nobody took the shutdown request within %.0f s", timeout)

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

  def should_extend_catalog(self) -> bool:
    """Should the model manager's big-model catalog carry the newer catalogs'
    models? With no chestnut fitted. Hardware, not the link setting: the model
    manager drops a pick its catalog does not list."""
    return not self._parts.chestnut_fitted()

  def extend_catalog(self, catalog: dict) -> dict:
    """The big-model catalog the model manager fetched, with the models newer
    catalogs list folded in. Never raises."""
    return self._parts.models.big_catalog(catalog)
