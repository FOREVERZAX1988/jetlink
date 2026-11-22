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
from pathlib import Path

import numpy as np

from jetlink import codecs, hevc
from jetlink.openpilot import stream as stream_mod
from jetlink.openpilot.latency import LateTracker
from jetlink.openpilot.pace import Governor
from jetlink.openpilot.stages import FrameStages as StageStages
from jetlink.protocol import NotFinite


class NoPose(RuntimeError):
  """The large model answers, but without a pose: its camera odometry comes back
  as zeros at a confidence no camera can have. Not a slow host and not a link
  problem -- a model whose outputs cannot drive, which must not be swapped in."""


class NotReady(RuntimeError):
  """This frame has no output to publish, and no link failure to blame.

  An HEVC stream opens with frames withheld until both cameras have produced a
  keyframe (stream.StreamGate), and for the first of them there is no earlier
  output to publish again either. That is a frame the small model must run, and
  nothing more: the stream is coming up, not failing.

  Distinct from a NotFinite, which is a host that answered wrongly and hands the
  drive back; a caller that catches this runs the small model for one frame and
  keeps the link.
  """
  retryable = True

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
SLOW_FRAME = 0.05  # the full 20 Hz budget, not just the largest outliers
# How far into a frame end() waits for the reply before publishing the
# previous frame's output again (a held frame). Past this the camera frame
# would be dropped: modeld's own work around run() takes the rest of the 50 ms.
# A held frame is a plan one frame old, which the 10 s plan and the actuator
# delay both dwarf, where a dropped frame is the same stale plan plus a count
# toward selfdrived's modeldLagging. None waits the client's full deadline
HOLD_FRAME = 0.046
# The first frame of a session waits its own, wider budget. Nothing has come
# back yet, so there is no earlier output to publish again and a hold is not
# available; on a cold phone that frame also carries the engine's preload (a
# 100-140 ms inference) and the radio waking from power save. Charging that
# one frame against the 0.2 s link deadline abandons the link before the host
# has answered, and every retry starts cold again -- a phone that was freshly
# installed can never join. Kept under modelV2's 0.5 s alive limit; later
# frames still use client.deadline so a host that has gone silent fails fast.
# DIAGNOSTIC MODE 2026-11-22: the "Jetlink remote diagnostic" toggle in
# developer settings raises this to 2.0 s for offroad/bench sessions, where
# the two video encoders cost ~1.2 s to open. That time is spent before the
# first frame is sent, so it fits a parked session but not a drive. Default
# stays at 0.45 s, under modelV2's 0.5 s alive limit.
def _remote_diag() -> bool:
  try:
    return Path("/data/params/d/JetlinkDiagMode").read_bytes().strip() == b"1"
  except OSError:
    return False


FIRST_FRAME_BUDGET = 2.0 if _remote_diag() else 0.45
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
# wheel, where a slow host's return is a soft disable within a second
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
# Over Wi-Fi: a 55 ms wait window covers the measured ~45 ms reply; more
# settling frames and a wider steady-state tolerance for marginal holds.
WIFI_HOLD_FRAME = 0.055
WIFI_SETTLING_FRAMES = 8
WIFI_HOLDS_IN_A_ROW = 15
WIFI_HOLDS_ALLOWED = 150
# How many frames a pipelined Wi-Fi link may leave unanswered. Without a bound
# a cold or busy phone piles frames up until the oldest crosses the client's
# 0.2 s deadline (link abandoned) or the TCP send buffer backs up (write
# timeout). Three at 20 Hz is 150 ms, inside the deadline, and a phone that
# answers in ~22 ms keeps only one or two out at a time.
# A model that cannot drive has no plan. Measured 2026-11-18: the healthy
# chestnut model plans 21.53 m ahead while parked in the preview (pose_stds
# 0.07344, lane_lines 4.98), and the one that failed in motion carried
# position.x[5] under 0.05 m on 99 % of frames. Two metres separates them with
# room to spare. The confidence on the pose cannot be the yardstick -- a parked
# car's rotation std is 0.0011 on the comma's own model too, which is right.
MISSING_PLAN_M = 2.0
MISSING_POSE_FRAMES = 5
# The heads worth reporting on the first output of a model: a missing one is a
# model exported without it, a zero one is a model exported with it that computes
# nothing. Logged once per model so the phone build can be told from the export.
OUTPUT_SUMMARY_KEYS = ('pose', 'pose_stds', 'plan', 'plan_stds',
                       'lane_lines', 'lane_lines_prob', 'lead', 'lead_prob')
PIPELINE_IN_FLIGHT = 3
# How long the frame thread may spend handing a frame to a pipelined link
# before that is the link and not the host. A frame costs the camera 50 ms:
# warp ~3 ms, the host's answer read later, the rest is modeld's own work. When
# send() blocks past this, the socket buffer is full because the link cannot
# carry the frames the cap asks for.
# The governor steps the rate down on this, so the link carries fewer frames
# instead of modeld dropping camera frames behind it (pace.py). Pipelined
# frames never hold, so without this the governor has no signal at all on Wi-Fi.
#
# 12 ms was measured with LOSSLESS frames (~187-393 kB each: a phone hotspot
# needed ~30 Mbps at 20 Hz and the 2026-11-08 drive saw 20-57 ms sends). HEVC
# made a frame ~25 kB, so a healthy Wi-Fi send now lands at 12-16 ms and used
# to sit exactly on the old line: every pipelined frame was called backpressure,
# the governor stepped down to 10 Hz within a second, and it could not climb
# back because that needs 10 quiet seconds. Measured 2026-11-17 on a phone
# hotspot: send 12.3-15.8 ms while the server answered in 42.8 ms -- the link
# was fine, the budget was stale. A 25 kB frame sent in 30 ms is still under
# 7 Mbps, far below what either link carries, so a real queue still overruns it.
SEND_BUDGET = 0.030
# frames between asks for the server's telemetry, which rides on the response:
# every second one, as modeld sent a chestnut's state (20 Hz over 10 Hz)
TELEMETRY_EVERY = 2
# telemetry goes to the log at most this often
TELEMETRY_PERIOD = 1.0
# frames Trips keeps the timings of: a minute at 20 Hz
TRIPS_KEPT = 1200


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
    """frames, over, held, span_s, and over the frames kept: p50_ms, p99_ms,
    max_ms of the whole frame and server_ms, the server's mean total."""
    out = {'frames': self.frames, 'over': self.over, 'held': self.held,
           'span_s': round(time.monotonic() - self.started, 1)}
    if self._whole_ms:
      whole = sorted(self._whole_ms)
      out.update(p50_ms=round(whole[len(whole) // 2], 1), p99_ms=round(whole[int(0.99 * (len(whole) - 1))], 1),
                 max_ms=round(whole[-1], 1), server_ms=round(sum(self._server_ms) / len(self._server_ms), 1))
    return out


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
  # held back because the pipeline's in-flight bound was reached (not a slow
  # host): end() publishes the previous plan as a pipelined frame does
  skipped: bool = False
  # the link, not the host: send() blocked past SEND_BUDGET on a pipelined
  # frame, so the governor steps the rate down (see SEND_BUDGET)
  slow_send: bool = False


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

  def __init__(self, client, spec, warp, *, face, log, event,
               hold_frame: float | None = None, settling_frames: int | None = None,
               holds_in_a_row: int | None = None, holds_allowed: int | None = None,
               proving_holds_fatal: bool = True, stages: object | None = None,
               late: object | None = None, pace: bool = False,
               pipeline: bool = False):
    self._log = log
    # A Wi-Fi link pipelines: end() publishes the newest answer drain() read
    # and never blocks on the frame's own reply (see end). Waiting for a phone
    # costs 30-50 ms of modeld's frame thread against a 50 ms camera, and the
    # dropped-frame ratio past DROP_LIMIT hands the drive back to the small
    # model. The cable keeps the strict frame loop it always had.
    self._pipeline = bool(pipeline)
    # the answer the last pipelined frame published, so publishing it twice is
    # visible as a host that missed a whole frame period (see end)
    self._published = None
    # why this model's outputs cannot be used, once enough of them are parsed
    self._bad_outputs: str | None = None
    self._outputs_checked = False
    # the plan magnitude of the first answers, judged by its peak (see _judge_pose)
    self._plan_peaks: list[float] = []
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
    self.spec = spec
    # not chestnut hardware, but the same role: modelV2.big, the UI and the
    # model manager key off this flag
    self.chestnut = True

    # a warp.Warp or the adapter's model_warp, loaded and warmed ahead: its
    # first call links the JIT, and this runs on modeld's frame thread. Frames
    # go out of its output as they are, over either transport
    self.warp = warp

    # a lossless warp's errors go packed to a server that takes them
    # (jetlink.lossless): half the bytes of the frame, for Wi-Fi
    self._packer = None
    if getattr(warp, 'errors', None) is not None and getattr(client, 'takes_lossless', False):
      from jetlink.lossless import Packer
      self._packer = Packer(spec.warped_shape)

    # an HEVC link's two encoders and its gate (jetlink.hevc, stream): the
    # frame path encodes instead of sending the warped bytes, and the gate
    # decides when a keyframe is owed. Built only where the server listed the
    # codec and the caller chose it (client.codec, codecs.pick)
    self._hevc = None
    self._gate = None
    self._metas = None
    self._epoch = 0
    if getattr(client, 'codec', None) == codecs.HEVC and getattr(client, 'takes_hevc', False):
      self._build_hevc(spec)

    # the frame rate's governor, on phone hosts only (joining passes pace for
    # wifi and cable-to-phone links): a host that falls behind (its holds) is
    # sent fewer frames instead of work it answers late, and paced frames are
    # never counted toward the hold limits that hand the drive back (pace.py)
    self._pace = Governor(log=self._log) if pace else None

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
    # each camera's luma as the last measured frame left it, and when that
    # was, for the frame-to-frame motion line (_log_frame_motion)
    self._last_luma = None
    self._motion_logged = 0.0
    self._last_hold_logged = 0.0
    # when the comma last told the host its own six stages (report_stages)
    self._stages_reported = 0.0
    self.trips = Trips()
    # The comma's own six stages (stages.FrameStages): where the frame's half of
    # the budget went, which neither the server's telemetry nor the round trip
    # can say. A caller may pass one to share (the join's, across rebuilds); one
    # is made here otherwise, so a frame is always timed and nothing is optional
    self.stages = stages if stages is not None else StageStages()
    # late replies, by the watermark rule (latency.LateTracker): a reply past
    # the hold is late, not lost, and a link that is late is one about to drop
    self.late = late if late is not None else LateTracker()
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
    # whether the last frame published the one before it: the runtime record's
    self.held = False

  def _build_hevc(self, spec) -> None:
    """The HEVC path's state: two encoders (one camera each), the gate that
    owes keyframes, and the cache that finds a frame's context when its encoder
    answers.

    The encoders come from the adapter a fork supplies (codecs.HEVCEncoder):
    the comma's is the Venus block, reachable only from the device, so this
    builds nothing itself and a link whose fork has no encoder never negotiates
    the codec (codecs.available at the pick).

    `_encoders` is empty and the path stays off where the builder refuses,
    which is how a link falls back to the codec it was already using rather
    than sending frames a server cannot decode.
    """
    h, w = spec.model_hw
    self._gate = stream_mod.StreamGate()
    self._metas = hevc.MetaCache()
    self._epoch = 0
    self._encoders: dict[str, object] = {}
    # two different questions about the same frame: how many bytes it is, and
    # how those bytes are arranged. An encoder is configured from the second,
    # and a length cannot be split back into it.
    self._hevc_size = hevc.nv12_shape((h, w))
    self._hevc_geometry = hevc.nv12_geometry((h, w))
    self._warp_shape = spec.warped_shape
    self._hevc = True
    # The two encoders are built here, before the first frame, and not on it.
    #
    # Opening the Venus block -- two V4L2 devices, their formats, their buffers
    # -- costs about 1.2 s once per state (measured 2026-11-21: an 1134 ms and
    # a 1293 ms first frame, every rejoin, with the encoders' log lines printed
    # in between). Built on the first frame, that second is spent with a frame
    # outstanding, and every frame pipelined behind it misses the link's
    # deadline: 0.35 s is what keeps modelV2 from going stale on a drive, and
    # 1.2 s cannot fit inside it. Here it is spent before anything is sent.
    self._encoder_for()
    # a connection is up and the stream starts here
    self._open_hevc_stream()

  def _open_hevc_stream(self) -> None:
    """A connection is up, or a stream must restart: the gate owes IDRs and
    the encoders drop their references, so the first frames out are ones a
    receiver with nothing can decode."""
    if not self._hevc:
      return
    self._epoch += 1
    self._metas.clear()
    self._gate.open(self._epoch)

  def _encode_frame(self, frame_id: int) -> tuple | None:
    """The two cameras' HEVC access units for this frame, or None while the
    gate is shut or an encoder has nothing ready.

    The warp's output is both cameras as one (2, 6, H, W); each is reassembled
    into NV12 and through its own encoder. A frame that comes back short (an
    encoder still buffering) is not a failure: it is a frame the server is not
    sent, and the gate is told, so the next one owes an IDR if it was lossy.
    """
    if not self._hevc or self._gate is None:
      return None
    step = self._gate.begin(self._epoch, frame_id)
    encoders = self._encoder_for()
    if encoders is None:
      return None
    try:
      planes = self._planes()
    except Exception:
      self._log.exception("jetlink: could not read the warped frame for HEVC")
      return None
    units: dict[str, bytes] = {}
    idrs: dict[str, bool] = {}
    for name, enc in encoders.items():
      nv12 = hevc.to_nv12(planes[int(name == stream_mod.STREAMS[1])])
      # DIAGNOSTIC 2026-11-21: a bright, structured picture instead of the
      # camera's, while /data/local/tmp/jetlink_test_pattern exists. The scene
      # at the bench went flat and dark (narrow 19/1), which makes a dark
      # picture on the far end ambiguous: a decoder that works and a decoder
      # that returns nothing look the same. A ramp cannot be either.
      if os.path.exists('/data/local/tmp/jetlink_test_pattern'):
        yy = np.fromfunction(lambda r, c: (3 * r + c) % 251, (256, 512)).astype(np.uint8)
        uv = np.full((128, 256, 2), 128, np.uint8)
        nv12 = np.concatenate([yy.reshape(-1), uv.reshape(-1)])
      # DIAGNOSTIC 2026-11-21: an IDR every 8 frames, so a receiver that joined
      # mid-stream recovers within a fraction of a second. Remove once the gate
      # is shown to open with one on its own.
      forced = (name in step.request_idr) or (frame_id % 8 == 0)
      data, key = enc.encode(nv12, force_keyframe=forced)
      # DIAGNOSTIC 2026-11-21: what this side actually feeds and sends, so the
      # coded picture can be checked against the warped one somewhere else.
      if 12 <= frame_id <= 24 and name == stream_mod.STREAMS[0]:
        try:
          import os as _os
          if _os.path.exists('/data/local/tmp/jetlink_au_dump'):
            with open('/data/local/tmp/au_%03d.hevc' % frame_id, 'wb') as f:
              f.write(bytes(data))
            with open('/data/local/tmp/nv12_%03d.raw' % frame_id, 'wb') as f:
              f.write(nv12.tobytes())
        except Exception:
          pass
      # DIAGNOSTIC 2026-11-21: what the encoder was actually handed, against the
      # `warp frame ... narrow` line that says what the warp produced. A gap
      # between the two is a gap between the warp and the encoder, not between
      # the encoder and the decoder.
      if frame_id % 10 == 0 and name == stream_mod.STREAMS[0]:
        try:
          a = np.frombuffer(nv12, dtype=np.uint8)[: 512 * 256][::7]
          self._log.warning('jetlink hevc in: %.1f/%.1f au=%d key=%s',
                            float(a.mean()), float(a.std()), len(data), key)
        except Exception:
          pass
      if not data:
        # Nothing out of this encoder yet, which the opening frames of a
        # hardware stream always are: the block holds several frames before
        # its first output (codecs.HEVCEncoder.encode says so, and waiting
        # there would deadlock). This is not a gap, and reporting it as one
        # livelocks the gate: gap() shuts it and demands a fresh IDR, the
        # next frame's output is still a buffered earlier picture rather than
        # that IDR, so the gate never opens and no frame is ever sent. Leave
        # the gate as it stands and let the next frame try; a link that
        # genuinely never produces output is caught by the hold limits.
        return None
      units[name], idrs[name] = bytes(data), bool(key)
    acted = self._gate.confirm(self._epoch, frame_id, **{f'{k}_idr': v for k, v in idrs.items()})
    if not acted.send:
      # the gate is still waiting for a keyframe: the frame is dropped, and
      # the caller reads this as a frame not sent (a hold, not a link failure)
      return None
    return units[stream_mod.STREAMS[0]], units[stream_mod.STREAMS[1]]

  def _planes(self):
    """The warp's output as (2, 6, H, W) uint8, a view and not a copy."""
    out = self.warp.output
    return np.frombuffer(out, np.uint8).reshape(*self._warp_shape)

  def _encoder_for(self) -> dict | None:
    """This link's two encoders, made on first use. None where the adapter
    supplies no builder, which keeps the link on the codec it had.

    The builder is handed `(rows, cols)` and the flat byte count: the first is
    what an encoder configures itself from, the second is what a test compares a
    produced frame against. Handing it the byte count twice, as this did, left
    every backend with nothing to take a width and a height out of.
    """
    if self._encoders:
      return self._encoders
    maker = getattr(self.face, 'hevc_encoder', None)
    if maker is None:
      self._hevc = False
      return None
    try:
      for name in stream_mod.STREAMS:
        self._encoders[name] = maker(self._hevc_geometry, tuple(self._hevc_size))
    except Exception:
      self._log.exception("jetlink: could not build the HEVC encoder, staying on the frame carrier in use")
      # Close whatever did come up before giving the link back its old carrier.
      # An encoder holds a hardware session for as long as its fd is open, and
      # the block allows very few of them. Left open, one failed attempt costs
      # a session for good: the next attempt -- this trip's, or the next
      # connection's -- finds the block full and fails too, so the link never
      # gets HEVC again without a restart. Measured 2026-11-16: one modeld held
      # 16 /dev/video33 handles, all from failed attempts, and every retry was
      # rejected with "Instance count reached Max limit".
      for enc in self._encoders.values():
        try:
          enc.close()
        except Exception:
          self._log.exception("jetlink: could not close a half-built HEVC encoder")
      self._hevc = False
      self._encoders = {}
      return None
    return self._encoders

  def _frame_eof(self, inputs: dict) -> float | None:
    """The camera's own read-out stamp for this frame, on modeld's monotonic
    clock, when the loop supplies one: the capture stage is measured from it.
    None when it does not, and then the stage is measured from the frame's
    start and the sensor's own delay is not counted (see stages.FrameStages)."""
    stamp = inputs.get('timestamp_eof') or inputs.get('timestampEof')
    if stamp is None:
      return None
    try:
      # modeld's stamp is monotonic nanoseconds, the same clock perf_counter
      # reads: only the units differ
      return float(stamp) / 1e9
    except (TypeError, ValueError):
      return None

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
    # the frame's own timing starts here; the sensor's stamp, when the caller
    # passes one, is where the capture stage is measured from
    self.stages.start(self._frame_id + 1, t0, t_eof=self._frame_eof(inputs))
    self.warp.start(_address(bufs['img']), _address(bufs['big_img']), transforms['img'], transforms['big_img'])
    t1 = time.perf_counter()
    self.stages.captured(self._frame_id + 1, t0)
    # the GPU is warping; nothing below is an input to it
    self.client.drain()
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
    self.stages.warped(self._frame_id + 1, time.perf_counter())
    self._frame = Frame(t0, t1, time.perf_counter())
    self._log_frame_motion()

  # How often the warped frame is measured, in seconds: the measurement is a
  # mean, a spread and a difference over one plane each, which is nothing
  # beside the warp it follows.
  MOTION_PERIOD = 1.0

  def _log_frame_motion(self) -> None:
    """Say whether the picture this frame is about to send differs from the
    last one, and by how much.

    The big model has been seen publishing `transx 0.07` for a whole drive --
    the answer the graph gives when its frames do not change (measured
    2026-11-21: replaying one frame 120 times gives 0.081, the drive's own
    number). Everything upstream of the wire is ours, and none of it could be
    cleared by reading: this line is the one that says which side is still.
    A `moved` near 0 on a moving car means the warp or the frame it reads is
    frozen; a `moved` of a few counts with the host still answering 0.07 means
    the loss is after the encoder.

    Both views: 0 is the narrow camera, 1 the wide one, and the mean is what a
    black plane would show as 0-16.
    """
    now = time.monotonic()
    if now - self._motion_logged < self.MOTION_PERIOD:
      return
    self._motion_logged = now
    try:
      # the warped frame is (2, 6, h, w): two cameras, six planes each, so it
      # splits into twelve equal planes without knowing h and w at all
      out = self.warp.output
      planes = np.frombuffer(out, np.uint8).reshape(12, -1)
      if self._last_luma is None:
        self._last_luma = [None, None]
      view = []
      for slot in (0, 1):
        # every 8th byte: 4k samples carry a mean, and this must stay cheaper
        # than the warp it follows
        luma = planes[slot * 6][::8]
        prev = self._last_luma[slot]
        moved = 0.0 if prev is None else float(np.abs(luma.astype(np.int16) - prev.astype(np.int16)).mean())
        view.append((float(luma.mean()), float(luma.std()), moved))
        self._last_luma[slot] = luma.copy()
    except Exception as e:
      # never the frame's problem: say why once a second, then carry on
      self._log.warning('jetlink: warp frame probe unavailable: %s', e)
      return
    self._log.warning(
      'jetlink: warp frame %d narrow %.0f/%.0f moved %.2f, wide %.0f/%.0f moved %.2f',
      self._frame_id, view[0][0], view[0][1], view[0][2], view[1][0], view[1][1], view[1][2])

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
    # At the head of a session only one frame may be outstanding.
    #
    # The host's first frame after a join costs about 1.2 s -- the model's first
    # run on a phone that has just been handed a picture, not the frame itself --
    # and it is allowed the first-frame budget. Frames pipelined behind it are
    # judged by the steady-state deadline, which exists so modelV2 never goes
    # stale on a drive; they would all miss it and their loss would end the
    # link. So the head of a session does not pipeline, and the rest does.
    inflight_limit = PIPELINE_IN_FLIGHT if self._frame_id > 3 else 1
    if self._pipeline and self.client.unanswered >= inflight_limit:
      # Read what has arrived before deciding: a reply that just landed makes
      # room without dropping a camera frame. If the host really is that far
      # behind, hold this one (publish the previous plan) rather than push it
      # out and let the oldest reply cross the link's deadline.
      self.client.drain()
      if self.client.unanswered >= inflight_limit:
        frame.skipped = True
        frame.sent = time.perf_counter()
        self.stages.sent(self._frame_id, frame.sent)
        return
    try:
      if self._hevc:
        # the two cameras' access units, or a frame the gate kept back, which
        # is a hold and not a failure (send() returning with seq None)
        units = self._encode_frame(self._frame_id)
        if units is None:
          frame.sent = time.perf_counter()
          return
        frame.seq = self.client.infer_begin(None, self.packed, self._frame_id, reset=self._need_reset,
                                            want_state=frame.telemetry, skip_if_busy=self._can_hold,
                                            hevc=units)
      elif self._packer is None:
        frame.seq = self.client.infer_begin(self.warp.output, self.packed, self._frame_id, reset=self._need_reset,
                                            want_state=frame.telemetry, skip_if_busy=self._can_hold)
      else:
        frame.seq = self.client.infer_begin(None, self.packed, self._frame_id, reset=self._need_reset,
                                            want_state=frame.telemetry, skip_if_busy=self._can_hold,
                                            lossless=self._packer.pack(self.warp.errors))
    except Exception as e:
      # the exception first: a link with no send stats of its own (TCP) logged
      # an empty dict here, and the reason -- a write timeout or a piled-up
      # pipeline -- went with it
      stats = getattr(self.client.t, 'last_send', None)
      self._log.warning("jetlink: frame %d send failed: %s%s", self._frame_id, e,
                        f" ({stats})" if stats else "")
      raise
    frame.sent = time.perf_counter()
    # a send that took this long was the link's queue, not the host's work
    frame.slow_send = self._pipeline and frame.sent - frame.t2 > SEND_BUDGET
    self.stages.sent(self._frame_id, frame.sent)
    if frame.seq is not None:
      self._need_reset = False

  def end(self, after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    """The sent frame's output, as modeld blocks on a chestnut's, but not past
    HOLD_FRAME once there is an output to publish again: a frame that long
    is a dropped camera frame. Only a stall past the client's deadline, or an
    output that is not finite, raises, into the joining state's demotion to
    the small model."""
    frame, self._frame = self._frame, None
    # publish health while the Jetson works
    if after_enqueue is not None:
      after_enqueue()
    waiting_from = time.perf_counter()
    pipelined = False
    if frame.seq is None:
      # Not sent: either the link refused it (a hold), or the pipeline's
      # in-flight bound held it back. The latter publishes the previous plan
      # just as a pipelined frame does, and must not count as a host behind.
      model_output = None
      pipelined = frame.skipped
    elif self._pipeline and self.client.last_output is not None:
      # A pipelined frame does not wait for its own reply: publish the newest
      # answer prepare()'s drain() already read, and leave this frame's reply
      # for a later drain. Nothing waited, so the camera's 50 ms is intact --
      # the host works on this frame while the camera brings up the next one.
      # This publishes the same one-frame-old plan a held frame does, without
      # the block that drops camera frames behind the large model.
      model_output = None
      pipelined = True
    else:
      try:
        if self._can_hold:
          # The link budget is measured from the reply, not from the warp's
          # start: WIFI_HOLD_FRAME (55 ms) is documented as covering the
          # ~45 ms phone reply. Discounting the whole prepare/send/
          # after_enqueue span left as little as 27 ms, so a healthy link
          # held almost every frame and the pace governor could never see
          # that the host had recovered.
          model_output = self.client.infer_end(frame.seq, hold=self._hold_frame)
        else:
          # no answer yet this session: no output to publish again, so no hold.
          # infer_end() with hold=None judges the link by client.deadline; give
          # this one frame the wider budget and put the normal one straight
          # back. Passing deadline= instead would change the client interface.
          saved = getattr(self.client, 'deadline', FIRST_FRAME_BUDGET)
          self.client.deadline = FIRST_FRAME_BUDGET
          try:
            model_output = self.client.infer_end(frame.seq, hold=None)
          finally:
            self.client.deadline = saved
      except NotFinite:
        # answered, so the leave's summary counts it
        self.trips.record(time.perf_counter() - frame.t0, self.client.last_timings[2])
        raise
    t4 = time.perf_counter()
    # Only a frame that waited and gave up is a host that fell behind. The
    # answer a pipelined frame publishes took client.last_reply_s to arrive, and
    # that number is kept and logged -- but it is *not* a verdict here: the hold
    # budget (55 ms) was set for lossless frames, and HEVC puts a decode on the
    # phone inside the same round trip, so a healthy link measures 75-111 ms.
    # Judging it against 55 ms made every frame a hold, stepped the governor to
    # 10 Hz and held the drive at 8-10 fps (measured 2026-11-18 16:27). What
    # actually says the host cannot keep up is the in-flight bound and the
    # camera frames it drops behind it, which are judged where they belong.
    held = model_output is None and not pipelined
    reply_s = getattr(self.client, 'last_reply_s', 0.0)
    late = pipelined and (self.client.last_output is self._published
                          or reply_s > self._hold_frame)
    self._published = self.client.last_output if pipelined else None
    # The first frames of an HEVC stream have nothing to publish and nothing to
    # publish again: the gate is waiting for a keyframe on both cameras. That is
    # a stream coming up, not a host that is behind, so it is raised before any
    # hold is counted and never moves `behind`: the caller runs the small model
    # for the frame and keeps the link (joining catches NotReady).
    if held and self.client.last_output is None:
      self.held = False
      self.stages.replied(self._frame_id, t4)
      self.stages.finished(self._frame_id, time.perf_counter(), held=False)
      raise NotReady(f"frame {self._frame_id}: nothing to publish and nothing to hold")
    self.held = held
    # A slow *link* -- send() blocked past SEND_BUDGET, or the in-flight bound
    # holding a frame back -- is not a host that fell behind: it asks the
    # governor for fewer frames and never hands the drive back.
    pressured = pipelined and (frame.skipped or frame.slow_send)
    self.behind = self._note_hold(held, pressured)
    self.trips.record(t4 - frame.t0, self.client.last_timings[2])
    # the reply's own arrival, and whether the hold gave up before it: the
    # watermark rule, which says a late reply is late and a missing one lost
    # The watermark classifies replies the frame path waited on. A pipelined
    # frame neither waited nor gave up, so it must not move the watermark (or
    # every later reply would read as late) nor claim its reply arrived.
    if frame.seq is not None and not pipelined:
      (self.late.gave_up if held else self.late.arrived)(self._frame_id)
    self.stages.replied(self._frame_id, t4)
    outputs = self._outputs(self.client.last_output if (held or pipelined) else model_output)
    self.stages.finished(self._frame_id, time.perf_counter(), held=held)
    # a frame past the budget is a dropped camera frame; send against reply
    # says which end it was. Held frames are a steady few on a marginal host,
    # so after the first they go to the log at TELEMETRY_PERIOD
    log_hold = (held or late) and (self.trips.held <= 3 or t4 - self._last_hold_logged >= TELEMETRY_PERIOD)
    if self._frame_id <= 3 or t4 - frame.t0 > SLOW_FRAME or log_hold:
      note = ''
      if held:
        self._last_hold_logged = t4
        note = f', held ({self.trips.held} so far{", not sent" if frame.seq is None else ""})'
      elif late:
        # the number a blocking loop would have hit the hold rule with
        self._last_hold_logged = t4
        note = ', pipelined, reply %.0f ms past the %.0f ms hold' % (
          getattr(self.client, 'last_reply_s', 0.0) * 1e3, self._hold_frame * 1e3)
      elif pipelined:
        note = ', pipelined, send past budget' if frame.slow_send else ', pipelined'
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
    self._report_stages(t4)
    return outputs

  def _report_stages(self, now: float) -> None:
    """The comma's six stage means to the host, once a second of frames.

    Sent off the ring's running sum, so it costs a divide and not a sort, and
    from the frame path because that is the only thread holding the link: a
    second thread would race the frame's seq and could have its own message
    taken for a replay. A second that has run no frame says nothing rather than
    six zeros, which would read as a link with no cost at all."""
    if now - self._stages_reported < TELEMETRY_PERIOD:
      return
    self._stages_reported = now
    means = self.stages.means()
    if means.get('frames'):
      self.client.report_stages(means)

  def run(self, bufs: dict, transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    # a paced frame is not prepared, sent or run on the host: the last plan
    # publishes again, the held frame's semantics without a frame the host
    # queues. Never counted as a hold -- the governor asked for it (pace.py)
    if self._pace is not None and self.client.last_output is not None \
            and self._pace.skip(time.monotonic()):
      # A paced frame publishes the last plan without sending, but the replies
      # to frames already in flight still arrive. Without draining them here,
      # a stretch of paced frames (15 Hz) lets `unanswered` pile up with no
      # reader: send() then skips frames as "in-flight bound", and when the
      # governor restores 20 Hz the first prepared frame's drain can land in
      # a state where infer_end waits for a seq whose reply was already read
      # ("no answer to 2 frames in 0.2s; link abandoned", 2026-11-16). Draining
      # is non-blocking (recv timeout 0) and keeps last_output / the in-flight
      # ledger current, exactly as a sent frame's prepare() would.
      self.client.drain()
      self.held = True
      self.behind = None
      if after_enqueue is not None:
        after_enqueue()
      return self._outputs(self.client.last_output)
    self.prepare(bufs, transforms, inputs)
    self.send(after_enqueue is not None)
    return self.end(after_enqueue)

  @property
  def _hold_frame(self) -> float | None:
    return HOLD_FRAME if self._hold_frame_override is None else self._hold_frame_override

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
    """Is there an output to publish again instead of waiting (HOLD_FRAME)?"""
    return bool(self._hold_frame) and self.client.last_output is not None

  def _hold_left(self, frame) -> float:
    """What is left of the hold for `frame`, from its warp's start."""
    return self._hold_frame - (time.perf_counter() - frame.t0)

  def _note_hold(self, held: bool, pressured: bool = False) -> str | None:
    """Why the large model should hand back after this frame, if it should:
    any held frame among its first PROVING_FRAMES past SETTLING_FRAMES,
    HOLDS_IN_A_ROW held frames running, or more than HOLDS_ALLOWED in
    HOLD_WINDOW. None while it is keeping up.

    `pressured` is the pipelined link's own backpressure (SEND_BUDGET): the
    governor steps the rate down on it, the demote rule does not."""
    # the governor watches the same rhythm and steps the frame rate; its own
    # paced frames never come through here (run() answers them itself)
    if self._pace is not None:
      self._pace.note(held or pressured)
    if not held:
      self._holds_in_a_row = 0
      return None
    self.trips.held += 1
    now = time.monotonic()
    self._holds.append(now)
    while now - self._holds[0] > HOLD_WINDOW:
      self._holds.popleft()
    self._holds_in_a_row += 1
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
    # the server checks first (Status.NOT_FINITE, which the client raises as
    # this). This is for one that did not, or bytes the link changed: a raw
    # inf is a corrupt output even where the parser would clip it finite. A
    # sum allocates nothing, and any NaN or inf makes it one; these outputs
    # are far from overflow
    if not np.isfinite(model_output.sum()):
      raise NotFinite(f"frame {self._frame_id}: the output is not finite")
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    if not self._outputs_checked:
      self._outputs_checked = True
      self._log_output_summary(outputs_dict)
    self._judge_pose(outputs_dict)
    if self._bad_outputs is not None:
      raise NoPose(f"frame {self._frame_id}: {self._bad_outputs}")
    # a finite output can still parse to inf, by a parser that does not clip
    if not np.isfinite(sum(float(value.sum()) for value in outputs_dict.values())):
      raise NotFinite(f"frame {self._frame_id}: the parsed output is not finite")
    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    self._parsed = (model_output, outputs_dict)
    return outputs_dict

  def _judge_pose(self, outputs: dict[str, np.ndarray]) -> None:
    """Decide whether this model produces a usable plan, over its first answers.

    The confidence on a pose cannot be the yardstick: a parked car's rotation is
    zero and the model is *right* to be sure of it. Measured 2026-11-18 in the
    parked preview, on the same chestnut model route 00000041 called dead:

        pose=max0.09766  pose_stds=max0.07344  plan=max21.53125
        lane_lines=max4.97656  lane_lines_prob=max0.89479

    and the comma's own model parked reports rotation std 0.0011 -- the same
    0.0011 the chestnut model showed at 5-10 m/s, which is what made a
    rotation-std test refuse a perfectly healthy model.

    What a model that cannot drive does not have is a plan. A healthy one plans
    ~20 m ahead even standing still (21.53 m above); the one that failed in
    motion carried position.x[5] under 0.05 m on 99 % of frames. So: the peak
    plan magnitude over the first few answers, and the peak rather than a median
    because one honest answer is proof the head works.
    """
    if self._bad_outputs is not None or len(self._plan_peaks) >= MISSING_POSE_FRAMES:
      return
    plan = outputs.get('plan')
    if plan is None:
      return
    self._plan_peaks.append(float(np.max(np.abs(np.asarray(plan)))))
    if len(self._plan_peaks) < MISSING_POSE_FRAMES:
      return
    peak = float(np.max(self._plan_peaks))
    if peak < MISSING_PLAN_M:
      self._bad_outputs = ('the model returns no usable plan (peak %.2f m over %d frames)'
                           % (peak, MISSING_POSE_FRAMES))

  def _log_output_summary(self, outputs: dict[str, np.ndarray]) -> None:
    """One line per model saying what it actually produced.

    A head the model was not exported with is simply absent from the parsed
    dict; a head it was exported with, but computes nothing for, comes back as
    zeros. Both look like "no pose" from locationd's side, and telling them
    apart is the difference between a broken phone build and a broken export.
    Route 00000041 had pose_stds 0.0010 on all 14740 chestnut frames and an
    empty plan, with nothing on the device to say whether the heads existed.
    """
    try:
      parts = []
      for key in OUTPUT_SUMMARY_KEYS:
        value = outputs.get(key)
        if value is None:
          parts.append('%s=absent' % key)
        elif value.size == 0:
          # declared by the spec and sliced to nothing: an export gap that would
          # otherwise raise here and cost the whole line
          parts.append('%s=empty' % key)
        else:
          parts.append('%s=max%.5f' % (key, float(np.max(np.abs(value)))))
      self._log.warning(
        'jetlink: model %s output: %d elements, slices=[%s] %s',
        str(getattr(self.spec, 'sha256', '?'))[:12],
        max(sl.stop or 0 for sl in self.output_slices.values()),
        ' '.join(sorted(self.output_slices)), ' '.join(parts))
    except Exception:
      self._log.exception('jetlink: could not describe the model output')

  def release_encoders(self) -> None:
    """Close this state's HEVC encoders, keeping the link.

    An encoder holds a hardware session while its fd is open, and the block
    allows only a handful. The encoders belong to the state, not to the link:
    the next attempt builds its own two, so a retired state that kept them
    would strand sessions for the rest of the drive. Measured 2026-11-16: one
    modeld accumulated 14 /dev/video33 handles over a handful of rejoins and
    then every new attempt failed with "Instance count reached Max limit",
    which this link reads as ENOMEM and treats as "no HEVC here".

    Idempotent, and a no-op on a link that never negotiated HEVC.
    """
    encoders, self._encoders = getattr(self, '_encoders', {}) or {}, {}
    for enc in encoders.values():
      try:
        enc.close()
      except Exception:
        self._log.exception("jetlink: closing an HEVC encoder")

  def close(self) -> None:
    """Let go of the link; the joining state calls this on a model it retires."""
    self.release_encoders()
    self.client.close()


def _address(buf) -> int:
  """Where a camera buffer modeld hands over (a VisionBuf) keeps its bytes."""
  return np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
