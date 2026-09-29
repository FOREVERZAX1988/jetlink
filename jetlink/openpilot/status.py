"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the UI, hardwared and manager are told, in one snapshot, and the
progress the panels show while something provisions or joins.

Every answer here is files and params, no link IO: the UI asks five times a
second. The fork maps the snapshot onto its own widgets; the one mapping that
is jetlink's knowledge, which icon a state is, is Status.icon.

Light at module level: the package imports it, and the owner imports the package.
"""
from __future__ import annotations

import time
from typing import NamedTuple

from jetlink.comma import gadget

# -- progress -----------------------------------------------------------------
# Written by the provisioning run and the joining state, read by the UI: a
# param today (Keys.progress), because the writer is another process.

# Each report is a file write, and the UI reads it at 5 Hz. An upload reports
# once per 4 MB chunk, 440 of them for a 1.7 GB model, and onroad that is IO a
# recording would have to share the disk with.
PROGRESS_MIN_INTERVAL = 0.25


def estimated_build_seconds(size: int) -> int:
  """Orin Nano Super, TensorRT 10.3: the 766 MB models built in 102 to 166 s,
  the 1.75 GB ones in 230 to 294 s."""
  return int(60 + 130 * size / 1e9)


def _eta(seconds: float) -> str:
  if seconds >= 90:
    return f"about {seconds / 60:.0f} min left"
  return f"about {max(seconds, 1):.0f}s left"


class Progress:
  def __init__(self, op, size=None):
    self.op = op
    # () -> the picked model's size in bytes, or None: what the build's estimate is made from
    self._size = size
    self._last = ('', 0.0)

  def read(self) -> dict | None:
    """{stage, frac, msg} while something provisions, else None.

    Read from the UI's param thread, so nothing may escape, UnknownKeyName included.
    """
    try:
      value = self.op.get(self.op.keys.progress)
    except Exception:
      return None
    return value if isinstance(value, dict) else None

  def report(self, stage: str, frac: float, msg: str = '') -> None:
    """Never raises: called from except handlers.

    Held to 4 Hz within a stage. The end of one always goes through, so the last
    thing the panel is told is never dropped.
    """
    last_stage, last_at = self._last
    now = time.monotonic()
    if frac < 1.0 and stage == last_stage and now - last_at < PROGRESS_MIN_INTERVAL:
      return
    self._last = (stage, now)
    try:
      self.op.put(self.op.keys.progress, {'stage': stage, 'frac': round(frac, 4), 'msg': msg})
    except Exception:
      self.op.log.exception("jetlink: could not report progress")

  def clear(self) -> None:
    try:
      self.op.remove(self.op.keys.progress)
    except Exception:
      self.op.log.exception("jetlink: could not clear progress")

  def report_with_eta(self, stage: str, frac: float, msg: str = '') -> None:
    """Progress, with how long the build still has to run.

    Estimated from the model's size, on measurements of this hardware. It
    belongs here rather than in the UI, which knows nothing about jetlink. Only
    the build is estimated: the upload reports MB of MB and a connect has
    nothing to predict.
    """
    if stage == 'build':
      size = self._size() if self._size is not None else None
      if size:
        msg = _eta(estimated_build_seconds(size) * max(0.0, 1.0 - frac))
    self.report(stage, frac, msg)


# -- what is on the other end -------------------------------------------------

# jetlinkd, the owner, holds the gadget for as long as the link is enabled, so
# presence no longer blinks at every handover. What is left to bridge is a USB3
# link recovery passing through "addressed", and a bounce made on purpose when
# a host will not enumerate (gadget.wait_for_host)
PRESENCE_HOLD = 5.0


class Presence:
  def __init__(self):
    self._last_configured = 0.0

  def present(self) -> bool:
    """Is a Jetson actually on the other end right now?

    True once something holds the gadget open and a host has configured us,
    held for PRESENCE_HOLD after that stops. A phone on the cable is a host on
    the gadget like any other.
    """
    if gadget.dormant():
      # no enumeration during suspend; the CC line still tells a sleeping host from an unplugged one
      return gadget.port_has_host()
    now = time.monotonic()
    if gadget.host_attached():
      self._last_configured = now
      return True
    return now - self._last_configured < PRESENCE_HOLD


def link_transport() -> str:
  """What carries the link, for the panels: the gadget the owner built, a
  Jetson, a Linux PC or a Mac on the vendor interface or an iPhone dialed in
  over the network interface. Never raises: the panels read it on their tick."""
  try:
    if gadget.link_kind() == 'cable':
      peer = gadget.link_peer()
      return f"iOS over USB ({peer})" if peer else "iOS over USB"
  except Exception:
    pass
  return "USB"


def usb_port() -> str | None:
  """What the comma's USB-C port controller sees on the CC pin: 'empty', or
  'host' for a cable with something live behind it (it cannot say what). None
  where the kernel does not expose it, rather than claiming an empty port."""
  try:
    raw = gadget.CC_ORIENTATION.read_text().strip()
  except OSError:
    return None
  return 'empty' if raw == '0' else 'host'


# -- whether the link can run --------------------------------------------------

# the offroad alert's text for a device whose build made no warp for its camera
NO_WARP = "no warp built for this camera"


def unavailable(jl) -> str | None:
  """Why an enabled link cannot run the large model, or None: a file read and
  a stat, since the UI asks at 5 Hz."""
  error = gadget.gadget_error()
  if error is not None:
    return error
  return None if jl.warps.built() else NO_WARP


def _built_for_the_pick(jl) -> bool:
  spec = jl.spec.load()
  selected = jl.models.selected_model()
  return (spec is not None and selected is not None and spec.sha256 == selected['oid']
          and jl.spec.engine_ready_for(spec.sha256))


def ready(jl) -> bool:
  """Can the large model run right now? Params only, what the UI calls
  'compiled': the provisioning has already recorded the answer."""
  return jl.enabled() and unavailable(jl) is None and _built_for_the_pick(jl)


def unavailable_reason(jl) -> str | None:
  """For someone who asked for the link only: with it off, a device that
  cannot present the gadget simply does not offer the feature."""
  return unavailable(jl) if jl.enabled() else None


# -- the snapshot ----------------------------------------------------------------

class Status(NamedTuple):
  """Everything a reader shows, taken at once. Fields are only ever added."""
  enabled: bool                 # the setting is on and no chestnut is fitted
  mode: str                     # the Accelerator Link setting, one of MODES
  transport: str                # 'USB', or 'iOS over USB (<peer>)'
  present: bool                 # a host is on the gadget now, or asleep and known to be there
  port: str | None              # 'host' or 'empty' off the CC pin; None where the kernel does not say
  ready: bool                   # the picked model's engine is built (records only, no link IO)
  reason: str | None            # why an enabled link cannot run: the offroad alert's text
  progress: dict | None         # {stage, frac, msg} while provisioning or joining
  model: str | None             # the big model it will run: the pick, else jetlink's default
  default_model: str | None     # jetlink's default, named like the chestnut's (no build date)

  @property
  def active_model(self) -> str | None:
    """model, once it can run."""
    return self.model if self.ready else None

  def icon(self, started: bool, model_seen: bool, running_big: bool, state: str) -> str:
    """The chestnut icon's state for the link: a ChestnutState value.

    Offroad it is progress and the records; onroad, modelV2 and the joining
    model's state (modelDataV2SP.acceleratorState). `model_seen` is a modelV2
    since this drive started, `running_big` a live one that says big, and
    `state` the acceleratorState name.
    """
    if not started:
      stage = str((self.progress or {}).get('stage', ''))
      if not self.present:
        return 'disconnected'
      if stage and stage != 'ready':
        return 'failed' if stage == 'failed' else 'loading'
      return 'ready' if self.ready else 'uncompiled'

    if model_seen and running_big:
      return 'active'
    if not self.present:
      return 'disconnected'
    # attached, a pending join is loading, not a failed model
    if state in ('joining', 'retrying') or not model_seen:
      return 'loading'
    # the engine is up and only the swap window is missing, which on a MADS car
    # is the rest of the drive unless the driver stops
    if state == 'ready':
      return 'waiting'
    if not self.ready:
      return 'uncompiled'
    if state == 'running':
      return 'active'
    return 'failed'


def read(jl) -> Status:
  """ready() and unavailable_reason() as one pass: each file is read once."""
  enabled = jl.enabled()
  reason = unavailable(jl) if enabled else None
  return Status(
    enabled=enabled,
    mode=jl.settings.mode(),
    transport=link_transport(),
    present=jl.presence.present(),
    port=usb_port(),
    ready=enabled and reason is None and _built_for_the_pick(jl),
    reason=reason,
    progress=jl.progress.read(),
    model=jl.models.selected_model_name(),
    default_model=jl.models.default_model_name(),
  )


# what a reader gets when the snapshot itself failed: nothing to show, and the
# reason where the offroad alert puts it
def failed(error: str) -> Status:
  return Status(enabled=False, mode='off', transport='USB', present=False, port=None, ready=False,
                reason=f"jetlink status failed: {error}", progress=None, model=None, default_model=None)
