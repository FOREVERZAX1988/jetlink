"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

One provisioning run: `python -m jetlink.openpilot.provision --adapter MODULE`.

The heavy half of jetlink, numpy and the client, which is why it is a run and
not a daemon. The owner (jetlink.openpilot.owner, as jetlinkd) holds the gadget
for the whole time the link is enabled and starts a run when something
changes; the run borrows the endpoint files from it exactly as modeld does
(jetlink.comma.lending), so the gadget never leaves the bus and a parked car
keeps one resident jetlink process of about 13 MB instead of a run's 47. Only
the owner ever holds ep0: a run that gets no loan logs it and exits, and the
owner starts another.

The result is cached on the Jetson, recorded in the spec record and left
loaded on the server. The owner stops a run at the onroad transition with
SIGTERM, so every long wait polls `stop`; the server's build thread carries on
regardless and modeld picks the engine up over its own link.
"""
from __future__ import annotations

import argparse
import json
import signal

from jetlink.comma import gadget, lending
from jetlink.openpilot import link

# how long to wait for the Jetson to enumerate before giving up on this run.
# The owner presented the gadget; a box that is asleep answers the bind in
# about 8 s, one that is off never does and the next run will find it
WAKE_TIMEOUT = 20.0


class ProvisioningRun:
  def __init__(self, parts):
    self.parts = parts
    self.log = parts.log
    self.client = None
    self.stop = False
    self.fetch_failed = False
    # does the far end suspend when the gadget goes? From the server's hello.
    # The owner needs it to decide whether letting go is worth what it costs,
    # and cannot ask: it never speaks the protocol. None until a hello says:
    # a run with nothing to do never asks
    self.server_sleeps: bool | None = None

  # -- lifecycle ------------------------------------------------------------

  def request_stop(self, *_) -> None:
    self.stop = True

  def close_link(self) -> None:
    """Always go through this: a FunctionFS owner that exits without closing
    can wedge the driver until a reboot."""
    client, self.client = self.client, None
    if client is not None:
      try:
        client.close()
      except Exception:
        self.log.exception("jetlink: error closing the link")

  def open_link(self) -> bool:
    """Borrow the link from the owner that started this run. Without a loan
    there is nothing to open: only the owner ever holds ep0."""
    if self.client is not None:
      return True
    try:
      loan = lending.borrow('provision')
      if loan is None:
        self.log.error("jetlink: the owner lent no link, nothing to provision over")
        return False
      self.client = link.connect(self.log, loan, deadline=5.0, name='provision')
      return True
    except Exception:
      self.log.exception("jetlink: could not open the link")
      return False

  # -- provisioning ---------------------------------------------------------

  def fetch_model(self):
    """Download the pinned large model, once.

    Minutes on a slow link, so it reports progress and stops when the owner
    stops the run; it is the one call in a run that blocks for long.
    """
    if self.fetch_failed:
      return None
    try:
      path = self.parts.models.fetch_shipped_model(
        progress=lambda frac: self.parts.progress.report('download', frac, 'downloading the large model'),
        should_stop=lambda: self.stop,
      )
    except Exception:
      self.log.exception("jetlink: could not fetch the large model")
      self.parts.progress.report('failed', 1.0, 'could not download the large model')
      # one attempt per run; retrying a gigabyte on a loop is worse than staying small
      self.fetch_failed = True
      return None
    return path

  def provision(self) -> bool:
    """Make the Jetson ready for the selected model. Host must be attached.

    The identity comes from the catalog model's LFS pointer (the oid is the
    sha256, size the byte count), so the comma can ask without holding or
    hashing the ONNX.
    The file is only fetched when the server asks for the bytes; the Jetson
    keeps its own copy of every ONNX and never prunes it.
    """
    from jetlink.client import EngineMissing
    parts = self.parts
    entry = parts.models.selected_model()
    if entry is None:
      # no catalog yet; not an error
      parts.spec.clear_ready()
      parts.progress.clear()
      return False
    sha256, nbytes = link.identity(parts, entry)

    # only needed if the server turns out not to have this model; None is a
    # legitimate state here, see EngineMissing below
    model_path = parts.models.shipped_model_path()

    self.log.warning("jetlink: provisioning %s (%d MB, sha %s)",
                     entry.get('name', sha256[:16]), nbytes >> 20, sha256[:16])
    parts.progress.report('connect', 0.0, 'talking to the accelerator')

    hello = self.client.hello(timeout=10.0)
    self.note_sleep_after(hello)
    self.log.warning("jetlink: server %s trt %s", hello.get('device'), hello.get('trt_version'))
    try:
      spec = link.ensure(parts, self.client, sha256, nbytes, model_path,
                         progress=parts.progress.report_with_eta, should_stop=lambda: self.stop)
    except EngineMissing:
      # the server has nothing to build from, and neither have we: fetch the
      # model and hand it over in this run. The owner lends the link for the
      # whole run, so leaving after the download only put the build a
      # WORKER_BACKOFF later, five minutes of a finished download doing nothing
      if model_path is not None or (model_path := self.fetch_model()) is None:
        raise
      # a download of minutes can outlast the session; a hello starts a new one
      self.client.hello(timeout=10.0)
      spec = link.ensure(parts, self.client, sha256, nbytes, model_path,
                         progress=parts.progress.report_with_eta, should_stop=lambda: self.stop)

    parts.progress.report('ready', 1.0, 'engine ready')
    self.log.warning("jetlink: engine ready for %s", spec.sha256[:16])
    return True

  # -- the parked car -------------------------------------------------------

  def note_sleep_after(self, hello: dict) -> None:
    """Record whether the server suspends itself when the gadget goes.

    Letting go is only worth what it costs if the Jetson sleeps when it is
    orphaned. On ignition power it does not, and releasing anyway meant a
    powered, awake box spent the whole parked period unenumerated: the icon
    read DISCONNECTED five seconds later, and every handover after that was an
    unplug the server had to recover from.
    """
    try:
      after = hello.get('sleep_after')
      self.server_sleeps = True if after is None else float(after) > 0
    except (TypeError, ValueError):
      self.server_sleeps = True
    self.log.warning("jetlink: the jetson %s when the gadget goes",
                     "sleeps" if self.server_sleeps else "stays up")

  def has_work(self) -> bool:
    """Is there a reason to wake the Jetson? Only things the link can fix count."""
    spec = self.parts.spec.load()
    if spec is None or not self.parts.spec.engine_ready_for(spec.sha256):
      return True
    selected = self.parts.models.selected_model()
    return selected is not None and selected.get('oid') != spec.sha256

  def shutdown_jetson(self, reason: str) -> None:
    """hardwared is shutting the comma down and wants the Jetson off too.
    The request file is removed whatever happens: hardwared is waiting on it."""
    self.log.warning("jetlink: shutting the jetson down: %s", reason)
    try:
      if not self.open_link():
        raise RuntimeError("could not open the link")
      if not gadget.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce, should_stop=lambda: self.stop,
                                  mode=self.parts.settings.mode()):
        raise TimeoutError(f"no jetson attached within {WAKE_TIMEOUT:.0f} s")
      resp = self.client.shutdown(reason, timeout=5.0)
      self.log.warning("jetlink: jetson answered the shutdown request: %s", resp)
    except Exception:
      self.log.exception("jetlink: could not shut the jetson down")
    finally:
      gadget.finish_shutdown()

  # -- one run --------------------------------------------------------------

  def bounce(self) -> bool:
    """Ask the owner to bounce the gadget, over the lease. On a phone's cable
    the client's rebind is a no-op: nothing is stuck in an endpoint file."""
    try:
      return bool(self.client.rebind()) if self.client is not None else False
    except Exception:
      self.log.exception("jetlink: could not bounce the gadget")
      return False

  def note_state(self, unfinished: bool) -> None:
    """What the owner cannot work out for itself: whether the far end sleeps
    when the gadget goes, and whether this run left anything undone.

    A run that never heard a hello keeps what an earlier one learned. Writing
    the default instead told the owner a phone or an always-on Jetson sleeps
    after every run with nothing to do, and it let the gadget go."""
    sleeps = gadget.far_end_sleeps() if self.server_sleeps is None else self.server_sleeps
    try:
      gadget.STATE.write_text(json.dumps({
        'sleep_after': 1.0 if sleeps else 0.0,
        'unfinished': unfinished,
      }))
    except OSError:
      self.log.exception("jetlink: could not record what the owner needs")

  def run(self) -> bool:
    """One provisioning round. True when there is nothing left to do."""
    # the setting alone, as the owner that started this run reads it
    if self.parts.settings.mode() == 'off':
      return True

    reason = gadget.pending_shutdown()
    if reason is not None:
      self.shutdown_jetson(reason)
      return True

    # without a warp for this camera the engine would never run; the offroad
    # alert says so, and waking the Jetson to build one would not change it
    if not self.parts.warps.built():
      self.log.warning("jetlink: no warp built for this camera, nothing to provision for")
      self.note_state(unfinished=False)
      return True

    if not self.has_work():
      self.log.warning("jetlink: nothing to provision")
      self.note_state(unfinished=False)
      return True

    finished = False
    try:
      if not self.open_link():
        return False
      if not gadget.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce, should_stop=lambda: self.stop,
                                  mode=self.parts.settings.mode()):
        self.log.warning("jetlink: no jetson within %.0f s, leaving it for the next run", WAKE_TIMEOUT)
        return False
      finished = self.provision()
    except Exception:
      self.log.exception("jetlink: provisioning failed")
      self.parts.progress.report('failed', 1.0, 'see the log')
    finally:
      self.note_state(unfinished=not finished)
      self.close_link()
    return finished


def main(argv: list[str] | None = None) -> None:
  """One provisioning round over the fork's adapter, as the owner starts it."""
  from jetlink.openpilot.interface import load_adapter
  from jetlink.openpilot.parts import for_this_process
  p = argparse.ArgumentParser(prog='python -m jetlink.openpilot.provision', description=main.__doc__)
  p.add_argument('--adapter', required=True, help="the fork's adapter module")
  args = p.parse_args(argv)
  d = ProvisioningRun(for_this_process(load_adapter(args.adapter)))
  # the owner stops this at the onroad transition; closing the link properly is
  # what keeps the driver healthy for modeld
  signal.signal(signal.SIGTERM, d.request_stop)
  signal.signal(signal.SIGINT, d.request_stop)
  d.run()


if __name__ == "__main__":
  main()
