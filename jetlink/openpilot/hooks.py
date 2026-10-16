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

The fork's guard, which builds Hooks or else Off inside a function, and its
own fallback for no jetlink at all: docs/porting.md, "The adapter module".
Off is for a jetlink that imports but cannot be used (another API); link_on()
is the fork's stdlib read of the setting, so the reason a link cannot run is
told only to someone who turned it on.

Imported by every process that imports the fork's hook module, the resident
gadget owner among them: the standard library and jetlink's light half only.
Construction checks the API and nothing more; the adapter is built and bound
on the first hook that needs it, once per process. handed_over() and state()
read the model alone and never bind, and with link_on() saying the link is
off, nor do should_run(), reason(), runtime() and in_control(): manager,
hardwared and selfdrived load nothing of jetlink's heavy half then. status()
binds either way: the snapshot names the big model from the model library
(the catalog param and jetlink.registry's default), which the UI shows with
the link off too.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from jetlink.openpilot import API, Flags, Jetlink, SwapEvents, bind

# the APIs a fork's adapter may be written to: today's, and after a bump the
# previous one too, for one minor release (jetlink/openpilot/__init__.py)
COMPATIBLE = frozenset({API})

# what selfdrived is told when the swap events cannot run: none at all
NO_EVENTS = Flags(False, False, False, False, False)


def _text(error: BaseException) -> str:
  """An error as a log line says it, from an error whose str() may raise too."""
  try:
    return f"{type(error).__name__}: {error}"
  except Exception:
    return type(error).__name__


class _Hooks:
  """The hooks and the answer each gets with the link off. _call(name,
  default, *args) is Jetlink.<name>(*args), or default."""

  def __init__(self, reason: str = '', link_on: Callable[[], bool] | None = None):
    self._why = reason
    self._link_on = link_on
    # the joining model's handovers as handed_over() last saw them, in this process
    self._handovers = 0
    self._prepared = False

  def _call(self, name: str, default, *args):
    return default

  def _off(self) -> bool:
    """link_on() says the link is off. Unknown, or a link_on() that raises,
    is not off: the hook asks jetlink, as without one."""
    try:
      return self._link_on is not None and not self._link_on()
    except Exception:
      return False

  def should_run(self, started, params, CP) -> bool:
    """manager's should_run for the gadget owner."""
    return False if self._off() else self._call('enabled', False)

  def status(self):
    return self._call('status', None)

  def reason(self) -> str | None:
    """Why jetlink cannot run here, told to someone who turned the link on,
    else Jetlink.reason's. The call first: a bind that fails sets the why."""
    try:
      on = self._link_on is None or bool(self._link_on())
    except Exception:
      on = None   # unknown: jetlink's answer, and the why told to nobody
    if on is False:
      return None
    answer = self._call('reason', None)
    return self._why if self._why and on else answer

  def prepare(self, chestnut: bool) -> bool:
    """modeld, before config_realtime_process. A chestnut runs the large
    model itself and never asks jetlink."""
    self._prepared = bool(not chestnut and self._call('prepare', False))
    return self._prepared

  def attach(self, model, small, cam_w: int, cam_h: int):
    """modeld, once `small` is built: the model to run, `model` unless the link
    joined. Binds nothing unless prepare() said yes in this process."""
    if not self._prepared:
      return model
    joined = self._call('attach', None, small, cam_w, cam_h)
    return model if joined is None else joined

  # Jetlink's own two, which read the model and self._handovers alone: modeld
  # asks every frame, so they never bind
  def handed_over(self, model) -> bool:
    try:
      return bool(Jetlink.handed_over(self, model))
    except Exception:
      return False

  def state(self, model) -> str:
    try:
      return Jetlink.state(self, model)
    except Exception:
      return 'none'

  def runtime(self):
    """What modeld runs, from its runtime record (Jetlink.runtime): a
    status.Runtime, or None with the link off or unusable, or a chestnut
    fitted. Cheap enough for selfdrived's loop."""
    return None if self._off() else self._call('runtime', None)

  def in_control(self) -> bool:
    """The adapter's in_control(), the rule the swap waits on, for
    selfdrived's swap events. True with the link off or unusable, and on a
    failure: something may be in control."""
    return True

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

  def swap_events(self, tick_s: float) -> SwapTracker:
    """selfdrived's swap-event tracker, updated every `tick_s` (events.SwapEvents)."""
    return SwapTracker(None, None)


class Off(_Hooks):
  """The link off on every hook; reason() says why, when the link is on."""


class Hooks(_Hooks):
  """Jetlink over the fork's adapter, for an adapter written to `api`.
  adapter_factory() makes it, on first use. Raises at once for an API this
  jetlink does not accept. `log` is a logger, or a function that returns one,
  for what fails before the adapter's log is there."""

  def __init__(self, api: int, adapter_factory: Callable[[], object], *, link_on: Callable[[], bool] | None = None,
               log=None):
    if api not in COMPATIBLE:
      raise ValueError(f"jetlink package API {API}, this build expects {api}")
    super().__init__('', link_on)
    self._factory = adapter_factory
    # the adapter's once bound; until then the fork's, or the standard library's
    self._log = log
    self._bound = None
    self._adapter = None
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
      jetlink = bind(op)
      self._adapter = op
      return jetlink
    except Exception as e:
      self._why = f"jetlink failed to start: {_text(e)}"
      self._report("jetlink: %s", self._why)
      return False

  def _report(self, msg: str, *args) -> None:
    """A failure to the log, in the except block that caught it. A log that
    fails is let go: nothing a hook does may raise."""
    try:
      if not hasattr(self._log, 'exception'):
        try:
          log = self._log()   # a factory; None or a logger raises here too
        except Exception:
          log = None
        if not hasattr(log, 'exception'):
          import logging
          log = logging.getLogger('jetlink')
        self._log = log
      self._log.exception(msg, *args)
    except Exception:
      pass

  def _net(self, name: str, default, fn, *args):
    """fn(*args), or default when it raises. Logged once per distinct error
    (the UI would log a failing status five times a second), again after a
    call that works."""
    try:
      result = fn(*args)
    except Exception as e:
      text = _text(e)
      if self._failed.get(name) != text:
        self._failed[name] = text
        self._report("jetlink: %s() failed: %s", name, text)
      return default
    self._failed.pop(name, None)
    return result

  def _call(self, name: str, default, *args):
    def call():
      jetlink = self._jetlink()
      return default if jetlink is None else getattr(jetlink, name)(*args)
    return self._net(name, default, call)

  def in_control(self) -> bool:
    if self._off():
      return True
    return self._net('in_control', True, lambda: self._jetlink() is None or bool(self._adapter.in_control()))

  def swap_events(self, tick_s: float) -> SwapTracker:
    """Bookkeeping alone, so it binds nothing."""
    return SwapTracker(self._net('swap_events', None, SwapEvents, tick_s), self)


class SwapTracker:
  """events.SwapEvents behind the net: update() is no event at all when it
  fails, and settling False; the failure is logged as the hooks' are. Inert
  without a tracker."""

  def __init__(self, events: SwapEvents | None, hooks: Hooks | None):
    self._events = events
    self._hooks = hooks
    self._ok = events is not None
    self._warm = False

  @property
  def settling(self) -> bool:
    try:
      return self._ok and bool(self._events.settling)
    except Exception:
      return False

  def update(self, fresh, state, big, in_control) -> Flags:
    """SwapEvents.update. An in_control function is asked once on the first
    update as well, idle or not: a reader it starts lazily (Stock's
    SubMaster, StarPilot's AOL grace) answers True until it has heard
    anything, which would be a false offer had a state gone straight from
    'none' to 'ready' between two of the caller's reads."""
    if self._events is None:
      return NO_EVENTS
    if not self._warm and callable(in_control):
      self._warm = True
      self._hooks._net('in_control', None, in_control)
    flags = self._hooks._net('swap_events.update', None, self._events.update, fresh, state, big, in_control)
    self._ok = flags is not None
    return NO_EVENTS if flags is None else flags
