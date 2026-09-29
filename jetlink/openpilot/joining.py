"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A ModelState that starts as the small model and upgrades to the Jetson.

modeld's large-model load is a one-shot with BIG_MODEL_TIMEOUT and a silent,
one-way fallback. That suits a chestnut, powered from the comma. A Jetson is
on the ignition rail: cranking browns it out, so its boot starts about when
the comma goes onroad and takes 45 to 100 s, long after modeld gave up.

So modeld is handed the small model it already loaded, and the Jetson is
swapped in underneath once the link, the engine and the warp are all there.
modeld re-reads `model` every frame, and `modelV2.big` keeps its meaning.

Two rules the swap keeps:

- tinygrad work happens on modeld's thread. The joining thread does link IO
  only; building the JetlinkModelState unpickles a TinyJit, and doing that
  next to the small model running frames on the same device is not safe.
- never swap while the plan is steering. The two models disagree by ~195 m of
  planned path, and the swap costs a frame or two. Require fresh, fully
  disengaged controls; standstill alone is not enough, longitudinal control
  may still hold the brake.
"""
from __future__ import annotations

import threading
import time

from jetlink.comma import gadget
from jetlink.transport.priority import background_thread

# after a join fails or the large model dies mid-drive. Long enough not to
# thrash a booting Jetson, short enough to catch one that finished a moment later
REJOIN_DELAY = 5.0
# doubled per consecutive failure, reset by a join that lasted STABLE_SECONDS.
# A link that dies on its first frame every time must not cost a swap, a
# demote and an alert every few seconds
REJOIN_DELAY_MAX = 60.0
STABLE_SECONDS = 60.0
# after a join that held: a link that ran for minutes and then went is a USB
# drop, and the host re-enumerates a rebound gadget in under a second. The
# rest of the old delay was the driver's time, not the Jetson's
REJOIN_DELAY_QUICK = 1.0
# how often a backoff looks at the gadget. A host that configures it again
# after it went away is a replug, which the backoff is not for: one waited 16 s
# for a Jetson that was back in 0.4 (2026-09-29)
REPLUG_POLL = 0.25
# link drops in one drive before the status names the cable. One is weather;
# the 2026-09-07 evening drive had six in twelve minutes, every one the
# USB-C port losing its host, and nothing the driver could see said so
DROPS_TO_BLAME_CABLE = 2
ENGAGEMENT_POLL_MS = 100
ENGAGEMENT_MAX_AGE = 0.25
# a large-model frame is ~30 ms and the worst seen on the current stack ~55.
# One that took LATE_FRAME, or a second past SLOW_FRAME within LAG_WINDOW of
# the last, is a fault and is handled as a loss: modeld would otherwise count
# the dropped camera frames into modeldLagging with the slow model still
# steering. A frame past link.INFERENCE_TIMEOUT never returns; it fails
LATE_FRAME = 0.1
SLOW_FRAME = 0.075
LAG_WINDOW = 10.0
# the first frames after every swap are never counted as slow: they carry the
# history reset, and a Mac's first after a join is ~100 ms of CoreML warm-up
SETTLING_FRAMES = 3
# the small model drives the first frames of every modeld start, even with the
# large model ready: its first run in a process costs ~1.3 s, which the first
# fallback frame paid (25 dropped frames, commIssue) when the large model had
# driven from frame one. Nothing is in control at a modeld start
SMALL_WARMUP_FRAMES = 3
# how often a link that is ready but has nowhere to land gets checked, and
# how long the check may take; both off the frame loop
KEEPALIVE_PERIOD = 10.0
PING_TIMEOUT = 2.0
# why a demote happened, as the log and the UI say it
LOST = 'lost the accelerator'
BEHIND = 'the accelerator fell behind'


class JoiningModelState:
  """Duck-types openpilot's modeld ModelState, with a second one inside.

  `progress` is where the join says what it is waiting on (status.Progress),
  `engagement` makes the poller the swap window reads (the adapter's), and
  `log` is cloudlog on a comma.
  """

  def __init__(self, small, connect, build, prepare=None, reset_small=None, *, progress, engagement, log):
    self._small = small
    self._active = small
    self._connect = connect
    self._build = build
    self._reset_small = reset_small
    self._progress = progress
    self._engagement = engagement
    self._log = log

    # whatever the swap would otherwise do on the frame loop, done now on
    # modeld's main thread before the frame loop exists; at the swap it cost
    # a 1.5 s frame
    if prepare is not None:
      try:
        t0 = time.monotonic()
        prepare()
        log.warning("jetlink: prepared the large model ahead of the swap in %.2f s", time.monotonic() - t0)
      except Exception:
        log.exception("jetlink: could not prepare the large model ahead of the swap")
        raise

    # handed over by the joining thread, consumed by the first frame that finds
    # it safe to swap. Only ever assigned under the lock
    self._joined: tuple[object, object] | None = None
    # kept true during a keepalive ping, which temporarily takes _joined
    self._available = False
    self._retired = None
    self._lock = threading.Lock()
    self._rejoin = threading.Event()
    self._rejoin.set()
    # earliest the join loop may try again. Without a backoff a fault that
    # recurs on the first frame after every swap is a connect, build and demote
    # every 700 ms for the drive, each costing modeld a 100 ms frame
    self._rejoin_at = 0.0
    self._failures = 0
    self._joined_at = 0.0
    # links lost and lag demotes after a swap this drive, and why the last one
    # went while the join thread still owes the log and the UI a word about it.
    # Only a lost link says anything about the cable
    self._drops = 0
    self._lags = 0
    self._demoted: str | None = None
    # changes of the model that drives, and decisions to change it: modeld
    # compares it across run() (handovers)
    self._handovers = 0
    # set on a frame the large model fell behind on; the next frame demotes.
    # _slow_at is when the last slow frame was, for the second strike
    self._lagging = False
    self._slow_at: float | None = None
    # frames each model has run: the small one's since start, the large one's
    # since it swapped in
    self._small_frames = 0
    self._big_frames = 0
    # whether the host had let go of the gadget when the last link was lost,
    # and whether this failure streak has already skipped a backoff for a replug
    self._host_left = False
    self._replugged = False

    # assume engaged and moving until a message says otherwise, so a swap can
    # never happen on no information
    self._engaged = True
    self._engagement_updated = 0.0
    self._stop = threading.Event()

    # whether the large model has produced a frame; a first inference that
    # fails never announces readiness. Travels in modelV2.big and
    # modelDataV2SP, not a param: a chestnut's load is over once, this never is
    self._loading = True

    self._threads = [threading.Thread(target=self._join_loop, daemon=True),
                     threading.Thread(target=self._watch_engagement, daemon=True)]
    for t in self._threads:
      t.start()

  # -- what modeld reads ------------------------------------------------------
  # A read not defined here follows the model that is driving (__getattr__):
  # vision_input_names, and modeld_v2's constants, smoothing and action
  # function, which sunnypilot keeps on the ModelState so a custom small bundle
  # can carry its own. Swaps and demotes happen inside run(), so what is read
  # after it belongs to the model whose output it is. What the loop writes
  # lands on both. The frame's inputs are built for the small bundle (its
  # desire name, its optional slots); the large model reads the desire under
  # any name and the loop always supplies what it needs, so a swap can land on
  # any frame

  @property
  def big_model_available(self) -> bool:
    """Connected and waiting to switch: what big_model_state calls ready."""
    return not self._stop.is_set() and self._available and self._active is self._small

  @property
  def chestnut(self) -> bool:
    # modelV2.big. False while proxying, as the small model would report
    return getattr(self._active, 'chestnut', False)

  @property
  def handovers(self) -> int:
    """Moves on every swap and demote, and on the frame that decides a lag
    demote. modeld resets its dropped-frame filter when this changes across
    run(), so the stall of a handover is not lag, as for a chestnut's fallback.
    A count and not modelV2.big: a swap whose first frame fails and demotes in
    the same run() leaves `chestnut` as it was."""
    return self._handovers

  @property
  def big_model_state(self) -> str:
    """modelDataV2SP.acceleratorState, one of its enum names (STATES)."""
    if self._stop.is_set():
      return 'unavailable'
    if self._active is not self._small:
      return 'running'
    if self.big_model_available:
      # nothing left to wait for but a window. The icon says so rather than
      # pulsing "loading" for the rest of a drive with no stop in it
      return 'ready'
    return 'retrying' if self._failures else 'joining'

  @property
  def client(self):
    # read by the status publisher on every send, so the telemetry follows the link
    return getattr(self._active, 'client', None)

  @property
  def desire_key(self) -> str:
    return self._small.desire_key

  @property
  def numpy_inputs(self):
    return self._small.numpy_inputs

  @property
  def lat_delay(self):
    return self._active.lat_delay

  @lat_delay.setter
  def lat_delay(self, value):
    # modeld writes this every frame. Set on both, so a model that joins
    # mid-drive does not start on a stale delay
    self._small.lat_delay = value
    if self._active is not self._small:
      self._active.lat_delay = value

  @property
  def PLANPLUS_CONTROL(self):
    return self._active.PLANPLUS_CONTROL

  @PLANPLUS_CONTROL.setter
  def PLANPLUS_CONTROL(self, value):
    self._small.PLANPLUS_CONTROL = value
    if self._active is not self._small:
      self._active.PLANPLUS_CONTROL = value

  def __getattr__(self, name):
    # Only for names this class does not define. Without it, a comma or
    # sunnypilot sync that adds one read was an AttributeError on the frame
    # thread, which modeld re-raises: modeld dead for the drive, on jetlink
    # devices only. Reads only; a new write lands here and nowhere else, so
    # each one modeld makes has its own setter above
    if name.startswith('_'):
      # also what keeps __init__ from recursing before _active exists
      raise AttributeError(name)
    return getattr(self._active, name)

  # -- the frame path ---------------------------------------------------------

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    if self._lagging:
      # the last frame was the large model's last, published as it came
      self._lagging = False
      self._demote(BEHIND)
    self._maybe_swap()
    active = self._active
    started = time.monotonic()
    try:
      result = active.run(bufs, transforms, inputs, after_enqueue)
    except Exception:
      if active is self._small:
        # nothing to do with the link; modeld's own handler owns this
        raise
      failed = time.monotonic()
      self._log.exception("jetlink: large model failed mid-drive, back to the small model")
      self._demote(LOST)
      demoted = time.monotonic()
      # re-run the frame rather than propagate: modeld's fallback is permanent,
      # this one is retryable. after_enqueue is dropped, the large model may
      # already have called it
      result = self._small.run(bufs, transforms, inputs, None)
      # all of it is one modeld frame, 42 to 107 ms on the 2026-09-29 drives
      # with the small model warm. The dropped camera frames are forgiven
      # (modeld sees the model change), but the log should say which part it was
      done = time.monotonic()
      self._log.warning("jetlink: fallback frame %.0f ms: link %.0f, demote %.0f, small model %.0f",
                        (done - started) * 1e3, (failed - started) * 1e3,
                        (demoted - failed) * 1e3, (done - demoted) * 1e3)
      return result
    if active is self._small:
      self._small_frames += 1
      return result
    took = time.monotonic() - started
    self._big_frames += 1
    if self._loading:
      # a connected engine can still fail its first inference; only announce
      # readiness after a frame the caller can publish
      self._joined_at = time.monotonic()
      self._loading = False
      self._progress.clear()
      self._log.warning("jetlink: large model joined mid-drive, modelV2.big is now true")
    if self._big_frames > SETTLING_FRAMES and self._fell_behind(took):
      # this frame's output is published as it came; its stall is forgiven now
      self._lagging = True
      self._handovers += 1
      self._log.warning("jetlink: large model frame took %.0f ms, the small model drives from the next", took * 1e3)
    return result

  def _fell_behind(self, took: float) -> bool:
    if took > LATE_FRAME:
      return True
    if took <= SLOW_FRAME:
      return False
    now = time.monotonic()
    second = self._slow_at is not None and now - self._slow_at <= LAG_WINDOW
    self._slow_at = now
    return second

  @property
  def _window_open(self) -> bool:
    # standstill does not make an active longitudinal controller safe to swap
    fresh = 0 <= time.monotonic() - self._engagement_updated < ENGAGEMENT_MAX_AGE
    return fresh and not self._engaged

  def _maybe_swap(self) -> None:
    if self._joined is None or not self._window_open or self._small_frames < SMALL_WARMUP_FRAMES:
      return
    with self._lock:
      joined, self._joined = self._joined, None
      if joined is not None:
        self._available = False
    if joined is None:
      return
    client, spec = joined
    try:
      # everything tinygrad touches happens here, on modeld's thread. No
      # warmup: the first real frame carries the reset (~30 ms on the server),
      # where a warmup frame over the link was two more dropped frames
      t0 = time.monotonic()
      big = self._build(client, spec)
      big.lat_delay = self._small.lat_delay
      self._log.warning("jetlink: built the large model state in %.0f ms", (time.monotonic() - t0) * 1000)
    except Exception:
      self._log.exception("jetlink: could not bring up the large model, staying small")
      with self._lock:
        self._retired = client
      # backed off like a demote, or a build that fails the same way every
      # time is a connect and a build per second for the drive
      self._back_off()
      return
    self._big_frames = 0
    self._handovers += 1
    self._active = big

  def _demote(self, why: str) -> None:
    """Back to the small model, from a reset history. On the frame thread, so
    nothing here waits: the join thread reads the port, reports and closes
    the link, moments later."""
    big, self._active = self._active, self._small
    self._handovers += 1
    self._loading = True
    self._slow_at = None
    if why == BEHIND:
      self._lags += 1
    else:
      self._drops += 1
    self._demoted = why
    with self._lock:
      self._retired = big
    self._back_off()
    if self._reset_small is not None:
      self._reset_small()

  def _close_retired(self) -> None:
    with self._lock:
      retired, self._retired = self._retired, None
    if retired is not None:
      try:
        retired.close()
      except Exception:
        self._log.exception('jetlink: closing the retired link')

  def _back_off(self) -> None:
    """Push the next attempt out, further each time one fails on its heels.

    Each failed cycle is a swap frame, a demote frame and the alerts that go
    with them. A join that held for STABLE_SECONDS is retried at once, as
    failure one; a failure on its heels is the second rung.
    """
    held = time.monotonic() - self._joined_at if self._joined_at else 0.0
    stable = bool(self._joined_at) and held > STABLE_SECONDS
    self._failures = 1 if stable else self._failures + 1
    if stable:
      self._replugged = False   # a new streak may skip a backoff for a replug again
    self._joined_at = 0.0
    if stable:
      delay = REJOIN_DELAY_QUICK
    else:
      delay = min(REJOIN_DELAY * 2 ** (self._failures - 1), REJOIN_DELAY_MAX)
    self._rejoin_at = time.monotonic() + delay
    self._rejoin.set()
    self._log.warning("jetlink: next attempt in %.0f s (failure %d, link held %.0f s, drop %d, lag %d this drive)",
                      delay, self._failures, held, self._drops, self._lags)

  # -- background -------------------------------------------------------------

  def _report(self, stage: str, msg: str) -> None:
    """Tell the UI what the join is waiting on.

    Offroad the record carries a provisioning run's progress; without this, a
    Jetson that is not plugged in looked like one six seconds from loading. No
    fraction to give, and the panel does not invent one.
    """
    # the count is the diagnosis. A link that runs for minutes and then goes,
    # again and again, is the cable, and the cable is the one thing the
    # driver can do something about
    if self._drops >= DROPS_TO_BLAME_CABLE:
      msg = f"{msg}; link dropped {self._drops} times this drive, check the USB cable or the phone app"
    self._progress.report(stage, 0.0, msg)

  def _note_link_loss(self, why: str) -> None:
    """Off the frame thread: what the comma's USB-C port sees now, next to the
    failure (gadget.cc_orientation). A host still on the cable means the data
    link alone went; the kernel logs the same edge as a Type-C disconnect."""
    cc = gadget.cc_orientation()
    if cc is None:
      port = "port state unknown"
    else:
      port = f"port sees a host (cc {cc})" if cc else "port sees no host (cc 0)"
    # read now, before the teardown: a host back by the time the backoff starts
    # has still been replugged
    self._host_left = not gadget.host_attached()
    self._log.warning("jetlink: %s, %s; drop %d, lag %d this drive", why, port, self._drops, self._lags)
    self._report('connect', f'{why}, reconnecting')

  def _join_loop(self) -> None:
    """Open the link and get the engine ready. No tinygrad in here."""
    # off modeld's realtime core first. A thread started after
    # config_realtime_process(7, 54) inherits SCHED_FIFO and the single-core
    # affinity, and an equal-priority thread that wakes takes the core until
    # it blocks. Measured with these threads left as created: exec p95
    # 90.8 ms, max 159.9, 5% frame drops, enough for modeldLagging
    background_thread()
    while not self._stop.is_set():
      # no timeout: once joined there is nothing to poll for, and close() sets
      # this. An idle wake per second is not free on modeld's core
      self._rejoin.wait()
      if self._demoted and not self._stop.is_set():
        # before the teardown below, which can block: the port is read about
        # when it let go. Not after close(), which has cleared the progress
        why, self._demoted = self._demoted, None
        self._note_link_loss(why)
      # unbind and reader joins can block; only this thread does teardown,
      # and it finishes before opening another link
      self._close_retired()
      if self._stop.is_set():
        return
      self._rejoin.clear()
      if self._wait_out_back_off():
        return
      self._report('connect', 'waiting for the accelerator')
      try:
        # the connect can take minutes when the picked model still has to be
        # built, so it is handed the flag close() sets rather than polled
        client, spec = self._connect(self._stop.is_set)
      except Exception as e:
        # expected while the Jetson boots. Not exception(): a stack trace every
        # 5 s for the first minute of every drive is noise
        self._log.warning("jetlink: not joined yet (%s), retrying in %.0fs", e, REJOIN_DELAY)
        if self._stop.wait(REJOIN_DELAY):
          return
        self._rejoin.set()
        continue
      with self._lock:
        if self._stop.is_set():
          client.close()
          return
        self._available = True
        self._joined = (client, spec)
      self._log.warning("jetlink: link ready, waiting for a window to swap")
      self._report('connect', 'ready; re-engage to switch models')
      self._keep_alive()

  def _wait_out_back_off(self) -> bool:
    """Until the next attempt is due, or a host configures the gadget again
    after it went away. Once per failure streak: a host that flaps, a marginal
    cable, would otherwise cycle the swap and its alerts with no backoff at
    all. True once closed."""
    host_left, self._host_left = self._host_left or not gadget.host_attached(), False
    while (left := self._rejoin_at - time.monotonic()) > 0:
      if self._stop.wait(min(left, REPLUG_POLL)):
        return True
      attached = gadget.host_attached()
      if attached and host_left and not self._replugged:
        self._replugged = True
        self._log.warning("jetlink: the host configured the gadget again, retrying now")
        return False
      host_left = host_left or not attached
    return self._stop.is_set()

  def _keep_alive(self) -> None:
    """Ping a link that is waiting for a swap window.

    On a drive with no stop and no disengage that is the whole drive, and a
    Jetson that reboots in there would otherwise be found at the swap: a build
    on a dead link, a demote and the backoff, all on modeld's thread.

    The client is taken out of _joined for the ping and put back after, so the
    frame loop sees a whole one or none, and never waits on the lock.
    """
    while not self._stop.is_set():
      if self._rejoin.wait(KEEPALIVE_PERIOD):
        return
      with self._lock:
        joined, self._joined = self._joined, None
      if joined is None:
        return  # swapped in on a frame, or closed under us
      try:
        joined[0].ping(timeout=PING_TIMEOUT)
      except Exception as e:
        self._available = False
        self._log.warning("jetlink: the link died before it could be used (%s), reopening", e)
        try:
          joined[0].close()
        except Exception:
          pass
        self._rejoin_at = time.monotonic() + REJOIN_DELAY
        self._rejoin.set()
        return
      with self._lock:
        if self._stop.is_set():
          # close() ran during the ping and found nothing to close
          joined[0].close()
          return
        self._joined = joined

  def _watch_engagement(self) -> None:
    background_thread()   # off modeld's realtime core; see _join_loop
    # made on this thread: the poller's sockets (a SubMaster) belong to the
    # thread that made them
    engaged = self._engagement()
    while not self._stop.is_set():
      # rechecked on every poll, including ones with no news: that is when
      # alive turns false. The frame thread expires the answer too
      self._engaged = engaged(ENGAGEMENT_POLL_MS)
      self._engagement_updated = time.monotonic()

  def close(self) -> None:
    self._stop.set()
    self._available = False
    self._rejoin.set()
    self._progress.clear()
    with self._lock:
      joined, self._joined = self._joined, None
    if joined is not None:
      joined[0].close()
    close = getattr(self._active, 'close', None)
    if close is not None and self._active is not self._small:
      close()


def join(parts, cam_w: int, cam_h: int, small) -> JoiningModelState:
  """The model modeld runs: the small model now, the Jetson once it is there.
  Returns straight away; the join runs in the background.

  The warp is loaded and warmed here, before the frame loop exists, rather
  than at the swap on a driving frame. Sized from the recorded spec, which is
  what the link will hand back; another geometry is rejected.
  """
  from jetlink.openpilot import link as links
  # here rather than in build(): the import then costs modeld's main thread
  # before the frame loop, not the frame the swap lands on
  from jetlink.openpilot.model_state import JetlinkModelState
  op = parts.op
  face = op.model_face()
  ready: dict = {}
  link = links.Link(parts.log)

  def prepare():
    from jetlink.openpilot.warp import prepare_reset, warm
    # the gadget first, so the Jetson enumerates while the warp loads. Left to
    # the join thread the bind landed ~3 s later, behind the small model's
    # first frame, and one ignition had a 655 ms frame during the bind
    links.present_early(link, background_thread)
    cached = parts.spec.load()
    if cached is not None:
      img_h, img_w = cached.model_hw
      geometry = (img_w * 2, img_h * 2)
    else:
      geometry = parts.warps.geometry()[2:]
    try:
      ready['reset_small'] = prepare_reset(small)
      warp = parts.warps.load(cam_w, cam_h, *geometry)
      warm(warp, face.frame_size(cam_w, cam_h))
    except Exception:
      link.close()
      raise
    ready.update(warp=warp, geometry=geometry)

  def build(client, spec):
    img_h, img_w = spec.model_hw
    warp = ready.get('warp') if ready.get('geometry') == (img_w * 2, img_h * 2) else None
    if warp is None:
      raise RuntimeError('no prepared warp for the server model geometry')
    return JetlinkModelState(cam_w, cam_h, client, spec, warp, face=face, log=parts.log, event=op.event)

  def connect(should_stop=None):
    return links.open_link(parts, link, should_stop)

  return JoiningModelState(small, connect, build, prepare, reset_small=lambda: ready['reset_small'](),
                           progress=parts.progress, engagement=op.engagement, log=parts.log)
