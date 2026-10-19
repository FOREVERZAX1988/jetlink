"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A ModelState whose policy runs on the Jetson.

Everything that touches the car stays on the comma: cameras, calibration, the
warp, the parser, controlsd, panda, CAN. The Jetson is a pure function, warped
frames and context in, 18452 floats out, and holds no control state.

The split is at `run_policy`. The warp stays on the comma's GPU: its input is a
2 MB camera buffer already there and its output the 393 KB the link carries
anyway. The history queues live on the Jetson, shipping them would cost ~10 MB
a frame instead of ~0.5 MB. modeld runs its warp as a JIT of its own, and
the link runs the same one; see warp.

A model that keeps its own history (openpilot #38916, Cinque Terre V3 on)
takes the same warped frame and the same scalars; its hidden state never
leaves the Jetson, so there is no prev_feat to send back. The spec says which.

What openpilot's modeld reads off a ModelState comes from the fork's adapter
(ModelFace): comma's constants, parser, smoothing and action function.
tinygrad is warp.Warp's to import: only modeld has it.
"""
from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jetlink.openpilot import stages

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
SLOW_FRAME = 0.05  # the full 20 Hz budget, not just the largest outliers
# How far into a frame end() waits for the reply before publishing the
# previous frame's output again (a held frame). Past this the camera frame
# would be dropped: modeld's own work around run() takes the rest of the 50 ms.
# A held frame is a plan one frame old, which the 10 s plan and the actuator
# delay both dwarf, where a dropped frame is the same stale plan plus a count
# toward selfdrived's modeldLagging. None waits the client's deadline, as before
HOLD_FRAME = 0.046
# What the frames before the host's first answer may wait, which the hold above
# cannot cover: there is no output to publish again, so there is nothing to fall
# back on and the wait would be the client's deadline (link.INFERENCE_TIMEOUT,
# 0.2 s), a number written for one 20 Hz frame. The first frame of a join is the
# slowest one there is: it carries the history reset and the host's own cold
# start (a Mac's CoreML ~100 ms, a phone's LiteRT more), and no reply yet is a
# link that never joined, not a host that is behind. So those frames wait this
# instead (client.FRAME_TIMEOUT's own reason: past it the far end is gone, not
# slow), and Wi-Fi the wider one: an extra round trip, a radio that can lose the
# request itself, and no USB edge to catch a reply that is late.
# Bounded on purpose: end() blocks the frame thread for as long as this lasts,
# so modelV2 stops, and a host that never answers would pin modeld where the
# small model should have taken over. A join that fails this way is retried,
# and the retry finds the host warm.
FIRST_FRAME_TIMEOUT = 3.0
WIFI_FIRST_FRAME_TIMEOUT = 6.0
# How long a preview may go with no answer at all before the link is taken for
# lost. A preview never takes control, so holding is no hazard in it: a slow
# host's frames publish the previous output, the plan goes on being drawn, and
# the picture refreshes at the host's rate. That is the point of a preview, and
# the hold counts below (written for the wheel) would take it away from the
# first frame past SETTLING_FRAMES. Only silence past this hands the picture
# back to the small model, so a plan that has stopped moving is replaced by the
# comma's own.
PREVIEW_SILENCE = 5.0
# Held frames the large model may make before it is behind rather than
# tailing: this many in a row, or more than HOLDS_ALLOWED within HOLD_WINDOW.
# A host at 40 ms a frame with a tail past 46 (an iPhone) holds a few frames
# in a hundred, each a plan one frame old; one at 60 ms holds every frame.
# More in a row than the client's deadline (link.INFERENCE_TIMEOUT, 0.2 s)
# lets a quiet host run to, so a run of holds is always a host that answers,
# late; one that has stopped answering fails the link first
# (JetlinkClient.infer_begin), as a lost link
HOLDS_IN_A_ROW = 5
HOLDS_ALLOWED = 20
HOLD_WINDOW = 10.0
# The first frames after a swap, which the fork holds engagement off for (a
# second at 20 Hz): a single held frame among them is behind. A host cold from
# a reconnect proves itself here, before anyone can engage, rather than at the
# wheel, where an iPhone's every return was a soft disable within a second
# (2026-10-04)
PROVING_FRAMES = 20
# the first of them never hand back, held or not: the first after a join
# carries the history reset, and a Mac's is ~100 ms of CoreML warm-up. A
# single slow frame later is the comma's own stall, a warp or a send, which
# handing back would not fix; only held frames say the host is behind
SETTLING_FRAMES = 3
# A phone's NPU runs the inference at ~33-43 ms; with the Wi-Fi round trip
# and the comma's own warp/send (~5 ms) a frame's wait is ~45 ms, right at the
# 20 Hz budget. The USB-tuned 46 ms hold leaves only ~41 ms after that work, so
# a phone behind either link holds most frames. These link-specific budgets let
# the large model stay swapped on a phone without weakening the strict defaults
# a fast Jetson/Mac keeps. A held frame still publishes the previous output
# (a plan one frame old), not a dropped camera frame; the 0.2 s client deadline
# and modeld's camera-drop filter hand back on a real stall.
#
# Over the USB cable (usb/ios): the CDC-NCM link adds almost no network time,
# but the same phone NPU does, so a modestly wider hold and a tolerant proof.
CABLE_HOLD_FRAME = 0.05
CABLE_SETTLING_FRAMES = 5
CABLE_HOLDS_IN_A_ROW = 10
CABLE_HOLDS_ALLOWED = 80
# Over Wi-Fi the hold deadline has to fit inside the camera's 50 ms frame, or
# the frame the wait was holding for is dropped behind the large model instead
# of published a frame late -- and a drop ratio past DROP_LIMIT is what hands a
# drive's picture back, which is how a ~55 ms reply kept the link in a preview
# and lost it on the road. 0.046 is upstream's HOLD_FRAME and that bound: a wait
# ending here still leaves modeld's own work its ~4 ms, so a measured ~55 ms
# reply publishes the previous answer (a held frame, a plan one frame old) and
# costs no camera frame. More settling frames and a wider steady-state
# tolerance for marginal holds. The counts below are pinned by joining and kept
# for the tests, but over Wi-Fi they no longer hand back at all (see _note_hold):
# a phone that tails the frame budget keeps the link, and only the client's own
# deadline ends it.
WIFI_HOLD_FRAME = 0.046
WIFI_SETTLING_FRAMES = 8
WIFI_HOLDS_IN_A_ROW = 15
WIFI_HOLDS_ALLOWED = 150
# How long a Wi-Fi link may go with no answer at all and still be tailing rather
# than behind. prepare() stamps _last_answer whenever a reply has landed, so on a
# phone that answers late but answers, this stays fresh every frame and the
# counts above never fill. A host that answers slower than this publishes a plan
# more than four frames old, and that is where the counts take the link back.
WIFI_TAILING_ANSWER = 0.2
# frames between asks for the server's telemetry, which rides on the response:
# every second one, as modeld sent a chestnut's state (20 Hz over 10 Hz)
TELEMETRY_EVERY = 2
# telemetry goes to the log at most this often
TELEMETRY_PERIOD = 1.0
# frames Trips keeps the timings of: a minute at 20 Hz
TRIPS_KEPT = 1200
# the whole run's summary (Trips.summary) goes to the log this often. The
# per-frame line is one sample: a drive's p99 is read off this one, not
# inferred from a stream of them
SUMMARY_PERIOD = 30.0


class FrameFailed(Exception):
  """A frame the host answered, and the answer carried no output the session
  can publish: an inference that failed (Status.NOT_FINITE on an output the host
  could not compute, or a shape it could not fill) on a frame with none earlier
  in the session to publish again. The link is not the failure.

  Only a preview gets here: it sets client.tolerate_failed, so such an answer
  does not end the link, and it never takes control, so serving this frame from
  the small model is safe (joining.JoiningModelState._run_big). The wheel's
  rules are untouched -- there such an answer is still the link's (client).
  """


class Trips:
  """What the comma measures of its frames over the link: the whole run, warp
  to parsed output, which is the frame as modeld waits on it, against the
  server's own total, which is all the server can see. Summarised for the
  leave (JetlinkClient.leave), so a phone's log carries the half of the frame
  budget it cannot measure. Bounded to the last TRIPS_KEPT frames."""

  def __init__(self):
    self.frames = 0          # frames waited for and published
    self.over = 0            # frames past SLOW_FRAME, the whole 20 Hz budget
    self.held = 0            # frames whose reply was late and published the previous one
    self.pipelined = 0       # frames that published the previous one without waiting at all
    self.started = time.monotonic()
    self._whole_ms: deque[float] = deque(maxlen=TRIPS_KEPT)
    self._server_ms: deque[float] = deque(maxlen=TRIPS_KEPT)

  def record(self, whole_s: float, server_us: int) -> None:
    self.frames += 1
    if whole_s > SLOW_FRAME:
      self.over += 1
    self._whole_ms.append(whole_s * 1e3)
    self._server_ms.append(server_us / 1e3)

  def summary(self) -> dict:
    """frames, over, held, pipelined, span_s, and over the frames kept:
    p50_ms, p99_ms, max_ms of the whole frame and server_ms, the server's mean
    total."""
    out = {'frames': self.frames, 'over': self.over, 'held': self.held,
           'pipelined': self.pipelined, 'span_s': round(time.monotonic() - self.started, 1)}
    if self._whole_ms:
      whole = sorted(self._whole_ms)
      out.update(p50_ms=round(whole[len(whole) // 2], 1), p99_ms=round(whole[int(0.99 * (len(whole) - 1))], 1),
                 max_ms=round(whole[-1], 1), server_ms=round(sum(self._server_ms) / len(self._server_ms), 1))
    return out


def summary_text(summary: dict) -> str:
  """Trips.summary() as one log line: frames, over, held, pipelined, span_s,
  then p50/p99/max of the whole frame and the server's mean total."""
  return ' '.join(f"{k} {v}" for k, v in summary.items())


@dataclass
class Frame:
  """One frame on its way to the host: when its warp started, was launched
  and finished, then its seq and send time once sent. The bytes are the
  warp's output (Warp.output)."""
  t0: float
  t1: float
  t2: float
  seq: int | None = None
  sent: float = 0.0
  telemetry: bool = False
  # where its six segments are filled in as it goes (openpilot.stages), and
  # finished by whichever frame reads the reply
  slot: stages.Slot | None = None


class JetlinkModelState:
  """Duck-types openpilot's modeld ModelState, and modeld_v2's where it differs.

  The small model is whatever bundle the user picked, on whichever modeld that
  bundle needs. sunnypilot's modeld_v2 reads constants, smoothing and the
  action function off the ModelState; stock modeld has them as module
  constants. This is comma's large model, so they are comma's.

  A frame is prepare(), send() and end(); run() is all three, as modeld calls
  it. The joining model (joining.JoiningModelState) reads `behind` after each
  frame.
  """

  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, client, spec, warp, *, face, log, event, preview: bool = False,
               pipeline: bool | None = None,
               hold_frame: float | None = None, settling_frames: int | None = None,
               holds_in_a_row: int | None = None, holds_allowed: int | None = None,
               proving_holds_fatal: bool = True):
    self._log = log
    # Per-link overrides. None means "read the module constant each frame", so
    # the strict USB/Jetson defaults stay live constants the tests can mock and
    # a Wi-Fi/USB phone build pins a wider value at construction (see joining).
    self._hold_frame_override = hold_frame
    self._settling_frames_override = settling_frames
    self._holds_in_a_row_override = holds_in_a_row
    self._holds_allowed_override = holds_allowed
    # USB: any held frame in the proof window proves the host too slow. Over
    # Wi-Fi a phone NPU sits right at the frame budget and holds most frames by
    # a millisecond or two (still a fresh-enough, one-frame-old plan), so the
    # proof uses the steady-state rules (a run of holds / too many holds); only
    # a genuinely stalling host hands back.
    self._proving_holds_fatal = bool(proving_holds_fatal)
    # (name, **fields): a structured log line, cloudlog.event on a comma
    self._event = event
    self.face = face
    # the joining model sets it from the small model before the first frame,
    # and modeld writes it every frame after
    self.lat_delay = 0.0
    self.constants = face.constants
    self.LAT_SMOOTH_SECONDS = face.lat_smooth_seconds
    self.LONG_SMOOTH_SECONDS = face.long_smooth_seconds
    self.PLANPLUS_CONTROL = 1.0
    self.get_action_from_model = face.get_action_from_model

    self.client = client
    # the six segments of every frame, filed as its reply is read: the frame
    # line below is one sample and has no reply segment for a frame that never
    # waited, which is the frame a demotion is made of (openpilot.stages)
    self.stages = stages.StageTimes()
    client.on_answer = self.stages.answer
    # when the last frame ended: what the next frame's pickup is measured from
    self._prev_end: float | None = None
    self._last_stages_logged = 0.0
    self.spec = spec
    transport = getattr(self.client, 't', None)
    on_cable = getattr(transport, 'on_the_cable', None)
    self._wifi = callable(on_cable) and not on_cable()
    # Whether anything can be in control, written by the joining state before
    # every frame it runs on this state (upstream asks the same question of its
    # adapter). It bounds the frames before the host's first answer: a join
    # budget written for a parked comma is not one a drive can be blocked on
    self.in_control = False
    # the fork's onroad preview (its IsOnroadPreview, declared through
    # interface.Keys.preview); see the property. A callable is the param itself,
    # read every frame; a bool is that frame's answer, kept. Either way it tells
    # the link, since a preview keeps a host whose frame carried no output
    self._preview_source = preview if callable(preview) else None
    self._preview = bool(preview()) if callable(preview) else bool(preview)
    self._tell_the_link()
    # Whether a frame's own reply is waited for at all; a Wi-Fi link's is not
    # (see the property), and a preview's is not either
    self._pipeline = True if pipeline is None else bool(pipeline)
    # not chestnut hardware, but the same role: modelV2.big, the UI and the
    # model manager key off this flag
    self.chestnut = True

    # a warp.Warp, loaded and warmed ahead: its first call links the JIT, and
    # this runs on modeld's frame thread. Frames go out of its output as they
    # are, over either transport
    self.warp = warp

    # a lossless warp's errors go packed to a server that takes them
    # (jetlink.lossless): half the bytes of the frame, for Wi-Fi
    self._packer = None
    if getattr(warp, 'errors', None) is not None and getattr(client, 'takes_lossless', False):
      from jetlink.lossless import Packer
      self._packer = Packer(spec.warped_shape)

    self.input_shapes = spec.input_shapes
    self.output_slices = spec.output_slices
    # from the spec, not ModelConstants: the server derives its history stride
    # from the same field
    self.frame_skip = spec.frame_skip

    # compile_modeld.make_input_queues' packed_npy_inputs, minus the warp's
    # inputs and the GPU queues the server owns; for a stateful graph, minus
    # prev_feat too
    self.packed = np.zeros(spec.packed_nelem, dtype=np.float32)
    self.npy = {k: self.packed[at].reshape(shape) for k, (at, shape) in spec.packed_layout.items()}

    self.prev_desire = np.zeros(face.desire_len, dtype=np.float32)
    self.parser = face.parser()
    # the camera buffers modeld hands over, whatever the graph calls its inputs
    self.vision_input_names = ['img', 'big_img']
    self._need_reset = True
    self._frame_id = 0
    self._last_logged = 0.0
    self._last_hold_logged = 0.0
    # perf_counter, like the frame times it is compared with: the first line
    # comes a period after the swap frame, not on it
    self._last_summary_logged = time.perf_counter()
    self.trips = Trips()
    # the frame prepared and not yet ended, if any
    self._frame: Frame | None = None
    # the newest output parsed, with what it parsed to: a held frame
    # publishes the dict again rather than parsing the array a second time,
    # which it must not, the parser's softmax working in place
    self._parsed: tuple[np.ndarray, dict] | None = None
    # when the large model held a frame, within HOLD_WINDOW, and how many in
    # a row; `behind` says why it should hand back after the last frame
    self._holds: deque[float] = deque()
    self._holds_in_a_row = 0
    self.behind: str | None = None
    # when the host last answered, for the preview's silence rule. The state is
    # built on the swap frame, so it starts counting from there
    self._last_answer = time.monotonic()

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k, v in output_slices.items()}

  def log_telemetry(self) -> None:
    """The server's health, piggybacked on the previous response, to the log
    at 1 Hz. chestnutState is comma's board on the wire and a Jetson's
    telemetry has no message of its own yet."""
    telemetry = self.client.last_state
    if not telemetry:
      return
    now = time.monotonic()
    if now - self._last_logged < TELEMETRY_PERIOD:
      return
    self._last_logged = now
    self._event("jetlinkTelemetry", dead=bool(self.client.dead), **telemetry)
    send = getattr(self.client.t, 'last_send', None)
    if send:
      self._event("jetlinkSend", nonce=self.client.nonce, frame=self._frame_id, totals=self.client.t.send_totals.copy(), **send)

  def prepare(self, bufs: dict, transforms: dict[str, np.ndarray], inputs: dict[str, np.ndarray]) -> None:
    """Take this frame up: start the warp, and while the GPU runs it, read
    what the host has answered so far (a held frame's late reply) and pack
    the rest of the inputs. send() puts it on the link and end() waits for
    its answer."""
    t0 = time.perf_counter()
    # the param's answer for this frame: a preview that ended between frames
    # must not keep the link's patience one frame longer
    self._tell_the_link()
    self.warp.start(_address(bufs['img']), _address(bufs['big_img']), transforms['img'], transforms['big_img'])
    t1 = time.perf_counter()
    # the GPU is warping; nothing below is an input to it
    if self.client.drain():
      # the host answered since the last frame. For a pipelined frame, which
      # never waits for its own reply, this is the only clock the silence rule
      # has: the frames that arrive late are the normal shape there, not a host
      # that has gone quiet
      self._last_answer = time.monotonic()
    self.log_telemetry()

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge.
    # Under whichever name the loop keyed it: stock modeld's desire_pulse, or a modeld_v2 bundle's own
    desire = inputs[next(k for k in inputs if k.startswith('desire'))]
    desire[0] = 0
    self.npy['desire'][:] = np.where(desire - self.prev_desire > .99, desire, 0)
    self.prev_desire[:] = desire
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']

    # free again once infer_begin returns: the gadget's io_submit and the
    # cable's sendmsg have copied it by then, before the next warp writes it
    self.warp.wait()
    self._frame = Frame(t0, t1, time.perf_counter())
    # pickup is the camera and modeld's own publish: the frame budget is spent
    # out of it before the warp is even launched, and nothing else reports it
    slot = self._frame.slot = self.stages.start(None if self._prev_end is None else t0 - self._prev_end)
    slot.warp = t1 - t0
    # everything from the launch to the warp's output being free again: the
    # packed inputs, the drain, and the wait itself, which the six do not
    # separate because at this size they are one number
    slot.pair = self._frame.t2 - t1

  def send(self, want_telemetry: bool = False) -> None:
    """The prepared frame to the host. It asks for the server's telemetry
    when the log is due for it, or when told to: modeld asks on the frames it
    would have sent a chestnut's state on.

    With an output to hold, a frame the link cannot take without waiting for
    the host to drain earlier ones is not sent, and end() holds it: the same
    plan one frame old a late reply publishes, where waiting stalls the frame
    loop on a host that is behind."""
    frame = self._frame
    self._frame_id += 1
    frame.telemetry = want_telemetry or time.monotonic() - self._last_logged >= TELEMETRY_PERIOD
    # packing a lossless warp is CPU on the frame thread, and it is the one
    # segment whose cost we could remove: timed apart from the send it precedes
    lossless = None
    if self._packer is not None:
      encode_from = time.perf_counter()
      lossless = self._packer.pack(self.warp.errors)
      frame.slot.encode = time.perf_counter() - encode_from
    try:
      if self._packer is None:
        frame.seq = self.client.infer_begin(self.warp.output, self.packed, self._frame_id, reset=self._need_reset,
                                            want_state=frame.telemetry, skip_if_busy=self._can_hold)
      else:
        frame.seq = self.client.infer_begin(None, self.packed, self._frame_id, reset=self._need_reset,
                                            want_state=frame.telemetry, skip_if_busy=self._can_hold,
                                            lossless=lossless)
    except Exception as e:
      # the exception first: a link that has no send stats of its own (TCP)
      # logged an empty dict here, and the reason went with it
      stats = getattr(self.client.t, 'last_send', None)
      self._log.warning("jetlink: frame %d send failed: %s%s", self._frame_id, e,
                        f" ({stats})" if stats else "")
      raise
    frame.sent = time.perf_counter()
    frame.slot.send = frame.sent - frame.t2
    # filed under the id the reply will name; an unsent frame keeps the slot
    # unused, and its reply never comes
    self.stages.track(frame.slot, self._frame_id, frame.sent)
    if frame.seq is not None:
      self._need_reset = False

  def end(self, after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    """The sent frame's output, as modeld blocks on a chestnut's, but not past
    the hold once there is an output to publish again: a frame that long
    is a dropped camera frame. A stall past the client's deadline raises, into
    the joining state's demotion to the small model. In a preview a frame the
    host answered without an output raises FrameFailed instead, which the
    joining state takes without handing the link back."""
    frame, self._frame = self._frame, None
    # publish health while the Jetson works
    if after_enqueue is not None:
      after_enqueue()
    waiting_from = time.perf_counter()
    pipelined = False
    if frame.seq is None:
      model_output = None   # not sent (see send): held
    elif self.pipeline and self.client.last_output is not None:
      # a pipelined frame does not wait for its own reply: the newest answer is
      # published (prepare()'s drain() read it) and this frame's reply is read
      # by a later frame. Nothing waited, so nothing here is late -- the host
      # is working on this frame while the camera brings up the next one
      model_output = None
      pipelined = True
      self.trips.pipelined += 1
    elif self._can_hold:
      model_output = self.client.infer_end(frame.seq, hold=self._hold_left(frame))
    else:
      # nothing has come back yet this session, so there is no output to publish
      # again: this frame waits its own budget, not one 20 Hz frame
      model_output = self.client.infer_end(frame.seq, deadline=self._first_frame_budget)
    t4 = time.perf_counter()
    # where the next frame's pickup starts from: modeld publishes this one and
    # comes back, and none of that is inside any segment below
    self._prev_end = t4
    # a frame that published the previous output, for one of two reasons: its
    # reply was late (a held frame, and the host is behind) or it was not
    # waited for at all (a pipelined frame, where being an answer behind is the
    # shape of the loop)
    earlier = model_output is None
    held = earlier and not pipelined
    self.behind = self._note_pipeline() if pipelined else self._note_hold(held)
    if earlier and self.client.last_failed and self.client.last_output is None:
      # the host answered and the answer carried no output, on a frame with
      # none earlier in this session to publish again: there is nothing to hold.
      # A preview keeps the link and the joining state serves this frame from
      # the small model, rather than handing the picture back to it
      raise FrameFailed(f"inference failed on frame {self._frame_id} before any output")
    outputs = self._outputs(self.client.last_output if earlier else model_output)
    self.trips.record(t4 - frame.t0, self.client.last_timings[2])
    # a frame past the budget is a dropped camera frame; send against reply
    # says which end it was. Held frames are a steady few on a marginal host
    # and a pipeline's are every frame, so after the first they go to the log
    # at TELEMETRY_PERIOD
    stale = held or pipelined
    log_hold = stale and (self.trips.held + self.trips.pipelined <= 3 or t4 - self._last_hold_logged >= TELEMETRY_PERIOD)
    if self._frame_id <= 3 or t4 - frame.t0 > SLOW_FRAME or log_hold:
      note = ''
      if stale:
        self._last_hold_logged = t4
        note = (', pipelined' if pipelined else
                f', held ({self.trips.held} so far{", not sent" if frame.seq is None else ""})')
      # persisted on the comma so a drive can separate server execution from
      # receive stalls once the Jetson is offline; server total excludes USB
      gpu_us, queue_us, total_us = self.client.last_timings
      receive = getattr(self.client.t, 'last_receive', {})
      # over the cable: how frames go, how many never came back, and what the
      # kernel dropped on the way in (softnet), since the link came up
      drops = getattr(self.client.t, 'net_drops', lambda: None)()
      cable = (f"; frames {'as datagrams' if self.client.t.datagrams else 'on the stream'}, "
               f"{self.client.frames_lost} lost, net drops {drops}" if drops is not None else '')
      self._log.warning("jetlink: frame %d warp %.1f data %.1f send %.1f wait %.1f ms%s; "
                        "server gpu %.1f queue %.1f total %.1f ms; ffs maxima prepare %.1f read_wait %.1f handoff %.1f ms%s",
                        self._frame_id, (frame.t1 - frame.t0) * 1e3, (frame.t2 - frame.t1) * 1e3,
                        (frame.sent - frame.t2) * 1e3, (t4 - waiting_from) * 1e3,
                        note,
                        gpu_us / 1e3, queue_us / 1e3, total_us / 1e3, receive.get('prepare', 0.0) * 1e3,
                        receive.get('read_wait', 0.0) * 1e3, receive.get('handoff', 0.0) * 1e3, cable)
    # the six segments of the newest frame that has them, at this line's
    # cadence. The line above splits a frame at the link; this one splits it,
    # and it is the only one that can split a frame that never waited: there
    # the reply segment is the frames the host was behind, not a wait that
    # ended when the hold did
    row = self.stages.last
    if row is not None and (self._frame_id <= 3 or t4 - frame.t0 > SLOW_FRAME
                            or t4 - self._last_stages_logged >= TELEMETRY_PERIOD):
      self._last_stages_logged = t4
      self._log.warning("jetlink: stages %s", stages.text(row))
    # the run itself, on the comma's log: the line above is one frame, and what
    # a drive's tail latency was is what this says it was
    if t4 - self._last_summary_logged >= SUMMARY_PERIOD:
      self._last_summary_logged = t4
      self._log.warning("jetlink: trips %s", summary_text(self.trips.summary()))
      # and which segment the drive's tail was in, over the frames kept: the
      # frame lines say it for one frame, and a demotion is decided on a run
      self._log.warning("jetlink: stages %s", stages.text(self.stages.summary()))
    return outputs

  def run(self, bufs: dict, transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    self.prepare(bufs, transforms, inputs)
    self.send(after_enqueue is not None)
    return self.end(after_enqueue)

  @property
  def _hold_frame(self) -> float | None:
    if self._hold_frame_override is not None:
      return self._hold_frame_override
    # nothing pinned by the caller (joining pins a per-link budget): a
    # Wi-Fi link keeps its own wider one, the cable the 46 ms above
    return WIFI_HOLD_FRAME if self._wifi else HOLD_FRAME

  @property
  def _settling_frames(self) -> int:
    return SETTLING_FRAMES if self._settling_frames_override is None else self._settling_frames_override

  @property
  def _holds_in_a_row_limit(self) -> int:
    return HOLDS_IN_A_ROW if self._holds_in_a_row_override is None else self._holds_in_a_row_override

  @property
  def _holds_allowed(self) -> int:
    return HOLDS_ALLOWED if self._holds_allowed_override is None else self._holds_allowed_override

  @property
  def _can_hold(self) -> bool:
    """Is there an output to publish again instead of waiting?"""
    return bool(self._hold_frame) and self.client.last_output is not None

  @property
  def preview(self) -> bool:
    """Is the comma drawing the onroad UI while parked (the fork's
    IsOnroadPreview)? Nothing in a preview can take control, so the large model
    is held to PREVIEW_SILENCE and a frame the host cannot answer is that
    frame's failure, not the link's.

    Read from the param every frame, not taken at construction. Upstream has no
    preview state at all -- what a link may do there is decided by its medium --
    and a snapshot taken at construction is a state upstream cannot have: a
    large model built while the preview was on kept the preview's rules after
    the fork cleared its param, which is a drive running under rules written for
    a parked car (joining clears it too, on the first frame anything can be in
    control, but the param is the fact and the clear is the fallback)."""
    return bool(self._preview_source()) if self._preview_source is not None else self._preview

  @preview.setter
  def preview(self, value: bool) -> None:
    # an explicit write wins over the param for the rest of the session: the
    # joining state's clear has to outrank a param that still reads true
    self._preview_source = None
    self._preview = bool(value)
    self._tell_the_link()

  def _tell_the_link(self) -> None:
    """Hand the preview's two relaxations to the link, or take them back.

    Both are the link's own rules: a host that is there but cannot answer a
    frame keeps its link where the wheel would hand one back
    (client.tolerate_failed), and a host that is behind rather than gone holds
    the frames it owes instead of costing the link (client.patient). A stall a
    preview can wait out is not the small model's cue, and the silence rule is
    what hands the picture back in the end."""
    preview = self.preview
    self.client.tolerate_failed = preview
    self.client.patient = preview

  @property
  def pipeline(self) -> bool:
    """Is this frame loop pipelined: never waiting for the frame's own reply,
    publishing the newest answer the host has sent instead?

    The wait it skips is not free: end() blocks modeld's frame thread, and a
    wait that runs into the camera's 50 ms is camera frames dropped behind the
    large model -- and a drop ratio past DROP_LIMIT is the joining state's
    hand-back. With the wait gone the host's work overlaps the camera's own
    frame: each frame goes out as its warp finishes, prepare()'s drain() reads
    whatever the host has answered since, and the loop's own cost falls to the
    warp and the pack. The host's own queue stops being empty, which is the
    point -- it was the accelerator that sat idle, not the link that was slow.

    What a pipelined frame publishes is what a held one publishes: the answer
    before this one. That is upstream's own output on a Wi-Fi link, where the
    wait window (WIFI_HOLD_FRAME) expires before a phone answers and the frame
    publishes the previous answer anyway; the pipeline is that same plan
    without the block that costs the camera frame.

    Which links get it is upstream's rule, the medium, not the fork's preview:
    upstream keys every Wi-Fi relaxation -- the hold window, the settling
    frames, the hold counts, pinned by joining from `mode == 'wifi'` -- on the
    medium, so its preview and its drive run on the same rules. A preview-only
    pipeline was the fork's one divergence from that, and it is what left the
    road with a frame loop that had no margin: measured 2026-10-18, the same
    phone and link, 7.7 ms pipelined and 50 to 56 ms waiting for its own reply
    against a 50 ms camera. So a Wi-Fi link pipelines whether anything can be
    in control or not, and the cable keeps the frame loop it always had.

    What stays preview's, and only preview's, is the link-keeping
    (client.patient, client.tolerate_failed) and the silence rule: those are
    the wheel's to refuse, and PREVIEW_SILENCE here reads the last answer
    drain() took rather than any one frame's own reply.
    """
    return self._pipeline and (self.preview or self._wifi)

  def _note_pipeline(self) -> str | None:
    """Why a frame should hand the picture back: silence, and only silence,
    which is what a preview is held to. Read by a pipelined frame and by a held
    frame of a preview that still waits (the frames before the host's first
    answer, and any state built with pipeline=False)."""
    if self.client.last_failed:
      # the host answered; only what it said was unusable. The link is there,
      # and a preview is held to the link, so the clock restarts and it goes on
      # being the model the app is running
      self._last_answer = time.monotonic()
      return None
    silence = time.monotonic() - self._answered_at()
    return f'no answer for {silence:.0f} s' if silence > PREVIEW_SILENCE else None

  @property
  def _first_frame_budget(self) -> float:
    """What the frames before the host's first answer of this session may wait.

    Parked, the wider one: the frame carries the history reset and the host's
    cold start, and there is no output to publish again, so nothing is lost by
    waiting (end() blocks modeld's frame thread, and a parked comma has no
    picture to keep moving). With anything in control, the client's own
    deadline, as upstream's first frame waits: end() blocking the frame thread
    is modelV2 stopping, and a drive cannot be parked on a host that may be
    gone. A join that fails there is retried, and the retry finds the host warm.
    """
    if self.in_control:
      return getattr(self.client, 'deadline', FIRST_FRAME_TIMEOUT)
    return WIFI_FIRST_FRAME_TIMEOUT if self._wifi else FIRST_FRAME_TIMEOUT

  def _hold_left(self, frame) -> float:
    """What is left of the hold for `frame`, from its warp's start."""
    return self._hold_frame - (time.perf_counter() - frame.t0)

  def _answered_at(self) -> float:
    """When the host last answered, by either clock.

    prepare()'s drain() stamps _last_answer when an answer is waiting at the
    top of a frame, but on a phone that tails by one the answer to frame N
    lands after frame N+1 has already drained and is taken by N+1's infer_end
    instead (client._take_reply), which stamps client.last_answer_at. Counting
    only the drain clock made a phone that answers every frame look silent:
    15 held frames in a row, the link handed back ~1 s into every join, while
    the wire lost nothing. Whichever clock is newer is the host's last word.
    """
    return max(self._last_answer, getattr(self.client, 'last_answer_at', 0.0))

  def _note_hold(self, held: bool) -> str | None:
    """Why the large model should hand back after this frame, if it should:
    any held frame among its first PROVING_FRAMES past SETTLING_FRAMES,
    HOLDS_IN_A_ROW held frames running, or more than HOLDS_ALLOWED in
    HOLD_WINDOW. None while it is keeping up.

    A preview is held to PREVIEW_SILENCE instead, and to nothing else: it never
    takes control, so a host that is slow but answering is the feature.

    Over Wi-Fi none of those counts hand back: see the note in the body. What a
    Wi-Fi host is doing that cannot keep up looks like a link that has stopped
    answering, and that is a lost link, not a held frame."""
    if not held:
      self._holds_in_a_row = 0
      self._last_answer = time.monotonic()
      return None
    self.trips.held += 1
    if self._wifi and (time.monotonic() - self._answered_at()) <= WIFI_TAILING_ANSWER:
      # A phone's reply lands at the frame budget, and the wait a held frame can
      # report is capped by that frame's own hold budget: on this link every held
      # frame waits the same ~42 ms whether the host is 10 ms behind or 200. So
      # the counts above are not evidence of a slow host; what is, is whether an
      # answer is still arriving (prepare() stamps _last_answer when one lands).
      # Measured 2026-10-18, SM8975 phone over a 5 GHz hotspot: ~55 ms every
      # reply, so every frame after the first published the previous output, and
      # the counts tore the link down about a second into every join -- nine
      # joins in one drive, "held 15 frames in a row", 0 lost and not one net
      # drop on the wire. Below this the phone is tailing, and the plan it
      # publishes is one frame old.
      return None
    now = time.monotonic()
    self._holds.append(now)
    while now - self._holds[0] > HOLD_WINDOW:
      self._holds.popleft()
    self._holds_in_a_row += 1
    if self.preview:
      return self._note_pipeline()
    if self._frame_id <= self._settling_frames:
      return None
    if self._proving_holds_fatal and self._frame_id <= PROVING_FRAMES:
      return f'held frame {self._frame_id} of the first {PROVING_FRAMES}'
    if self._holds_in_a_row >= self._holds_in_a_row_limit:
      return f'held {self._holds_in_a_row} frames in a row'
    if len(self._holds) > self._holds_allowed:
      return f'held {len(self._holds)} frames in {HOLD_WINDOW:.0f} s'
    return None

  def _outputs(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
    """The output as modeld reads it. An array parsed before (a held frame's)
    gives the same dict again: the parser's softmax works in place on it."""
    if self._parsed is not None and self._parsed[0] is model_output:
      return self._parsed[1]
    # the non-finite check runs on the server (Status.NOT_FINITE -> LinkError),
    # so modeld's big->small failover fires as it does for a chestnut
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    self._parsed = (model_output, outputs_dict)
    return outputs_dict

  def close(self) -> None:
    """Let go of the link; the joining state calls this on a model it retires."""
    self.client.close()


def _address(buf) -> int:
  """Where a camera buffer modeld hands over (a VisionBuf) keeps its bytes."""
  return np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
