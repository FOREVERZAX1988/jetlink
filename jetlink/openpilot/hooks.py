"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What a fork's hook lines call: Jetlink's methods, behind a net that never
raises.

manager, hardwared, the UI and the model manager call these on every device,
link on or off, and modeld on every drive: whatever jetlink does wrong turns
the link off and is logged, and never takes one of them down. Each hook has
the answer it gets with the link off, and gives it on any failure. They are
not all False: attach() keeps modeld's model, extend_catalog() the catalog,
and shutdown_step() says yes to a power-off that is wanted, since a broken
jetlink must never keep a comma on to drain the car's battery. selfdrived's
swap events (swap_events()) answer no event at all.

The fork's guard, inside a function so a jetlink that fails to import lands
in it too:

  try:
    from jetlink.openpilot.hooks import Hooks
    HOOKS = Hooks(API, adapter, link_on=link_on)
  except Exception as e:
    HOOKS = Off(str(e), link_on=link_on)

Off is here for a jetlink that imports but cannot be used (another API). The
fork still needs its own answer for no jetlink at all, a checkout missing or
too old to have this module: that one cannot live in jetlink. link_on() is
the fork's stdlib read of the Jetlink setting, so the reason a link cannot
run is told only to someone who turned it on.

Imported by every process that imports the fork's hook module, the resident
gadget owner among them: the standard library and jetlink's light half only.
Construction checks the API and nothing more; the adapter is built and bound
on the first hook that needs it, once per process.
"""
from __future__ import annotations

import threading
from collections import namedtuple
from collections.abc import Callable

from jetlink.openpilot import API, bind

# the APIs a fork's adapter may be written to: today's, and after a bump the
# previous one too, for one minor release (jetlink/openpilot/__init__.py)
COMPATIBLE = frozenset({3})

# what selfdrived is told when the swap events cannot run: none at all. The
# fields of events.Flags, read by name
_Flags = namedtuple('_Flags', 'offer link_lost no_entry chime settling')
NO_EVENTS = _Flags(False, False, False, False, False)


class _Hooks:
  """The hooks and the answer each gets with the link off. _call(name,
  default, *args) is Jetlink.<name>(*args), or default."""

  def __init__(self, reason: str = '', link_on: Callable[[], bool] | None = None):
    self._why = reason
    self._link_on = link_on

  def _call(self, name: str, default, *args):
    return default

  def _unusable(self) -> str | None:
    """Why jetlink cannot run here, told to someone who turned the link on."""
    if not self._why:
      return None
    try:
      return self._why if self._link_on is None or self._link_on() else None
    except Exception:
      return None

  def should_run(self, started, params, CP) -> bool:
    """manager's should_run for the gadget owner."""
    return self._call('enabled', False)

  def status(self):
    return self._call('status', None)

  def reason(self) -> str | None:
    answer = self._call('reason', None)
    return self._unusable() or answer

  def prepare(self, chestnut: bool) -> bool:
    """modeld, before config_realtime_process. A chestnut runs the large
    model itself and never asks jetlink."""
    return not chestnut and self._call('prepare', False)

  def attach(self, model, small, cam_w: int, cam_h: int):
    """modeld, once `small` is built: the model to run, `model` unless the link joined."""
    joined = self._call('attach', None, small, cam_w, cam_h)
    return model if joined is None else joined

  def handed_over(self, model) -> bool:
    return self._call('handed_over', False, model)

  def state(self, model) -> str:
    return self._call('state', 'none', model)

  def shutdown_step(self, should_shutdown: bool, reason: str = '') -> bool:
    return self._call('shutdown_step', should_shutdown, should_shutdown, reason)

  def request_shutdown(self, reason: str = '') -> bool:
    return self._call('request_shutdown', False, reason)

  def shutdown_pending(self) -> bool:
    return self._call('shutdown_pending', False)

  def model_state(self, ref: str) -> str | None:
    return self._call('model_state', None, ref)

  def should_extend_catalog(self) -> bool:
    return self._call('should_extend_catalog', False)

  def extend_catalog(self, catalog: dict) -> dict:
    return self._call('extend_catalog', catalog, catalog)

  def swap_events(self, tick_s: float) -> SwapEvents:
    """selfdrived's swap-event tracker, once, every `tick_s` (events.SwapEvents)."""
    return SwapEvents(None, None)


class SwapEvents:
  """events.SwapEvents behind the net: update() is no event at all when it
  fails, and settling False; the failure is logged as the hooks' are. Inert
  without a tracker."""

  def __init__(self, events, hooks: Hooks | None):
    self._events = events
    self._hooks = hooks
    self._ok = events is not None

  @property
  def settling(self) -> bool:
    try:
      return self._ok and bool(self._events.settling)
    except Exception:
      return False

  def update(self, fresh, state, big, in_control):
    if self._events is None:
      return NO_EVENTS
    try:
      flags, self._ok = self._events.update(fresh, state, big, in_control), True
      self._hooks._failed.pop('swap_events.update', None)
      return flags
    except Exception as e:
      self._ok = False
      self._hooks._failure('swap_events.update', e)
      return NO_EVENTS


class Off(_Hooks):
  """The link off on every hook; reason() says why, when the link is on."""


class Hooks(_Hooks):
  """Jetlink over the fork's adapter, for an adapter written to `api`.
  adapter_factory() makes it, on first use. Raises at once for an API this
  jetlink does not accept."""

  def __init__(self, api: int, adapter_factory: Callable[[], object], *, link_on: Callable[[], bool] | None = None,
               log=None):
    if api not in COMPATIBLE:
      raise ValueError(f"jetlink package API {API}, this build expects {api}")
    super().__init__('', link_on)
    self._factory = adapter_factory
    # the adapter's once bound; until then the fork's, or the standard library's
    self._log = log
    self._bound = None
    self._binding = threading.Lock()
    # hook -> the failure last logged for it, cleared by a call that works
    self._failed: dict[str, str] = {}

  def _jetlink(self):
    """Bound once, the failure included: one Jetlink per process, so that
    prepare() and attach() reach the same one."""
    if self._bound is None:
      with self._binding:
        if self._bound is None:
          self._bound = self._bind()
    return self._bound or None

  def _bind(self):
    try:
      op = self._factory()
      self._log = getattr(op, 'log', None) or self._log
      return bind(op)
    except Exception as e:
      self._why = f"jetlink failed to start: {type(e).__name__}: {e}"
      self._logger().exception("jetlink: %s", self._why)
      return False

  def _logger(self):
    if self._log is None:
      import logging
      self._log = logging.getLogger('jetlink')
    return self._log

  def _call(self, name: str, default, *args):
    try:
      jetlink = self._jetlink()
      if jetlink is not None:
        result = getattr(jetlink, name)(*args)
        self._failed.pop(name, None)
        return result
    except Exception as e:
      self._failure(name, e)
    return default

  def _failure(self, name: str, error: Exception) -> None:
    # once per distinct error: the UI would log a failing status five times a second
    text = f"{type(error).__name__}: {error}"
    if self._failed.get(name) != text:
      self._failed[name] = text
      self._logger().exception("jetlink: %s() failed: %s", name, text)

  def swap_events(self, tick_s: float) -> SwapEvents:
    """Pure bookkeeping, so it needs no adapter, and imported here so that
    selfdrived never imports jetlink at its top."""
    try:
      from jetlink.openpilot.events import SwapEvents as Tracker
      return SwapEvents(Tracker(tick_s), self)
    except Exception as e:
      self._failure('swap_events', e)
      return SwapEvents(None, None)
