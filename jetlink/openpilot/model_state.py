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
a frame instead of ~0.5 MB. Upstream fused warp and policy into one JIT, so
the fork's build compiles a standalone warp; see warp.

A model that keeps its own history (openpilot #38916, Cinque Terre V3 on)
takes the same warped frame and the same scalars; its hidden state never
leaves the Jetson, so there is no prev_feat to send back. The spec says which.

What openpilot's modeld reads off a ModelState comes from the fork's adapter
(ModelFace): comma's constants, parser, smoothing and action function.
tinygrad is imported when a model is built, never at module level: only
modeld and the build have it.
"""
from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from jetlink.openpilot.warp import call_warp
from jetlink.transport.base import LinkError

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
SLOW_FRAME = 0.05  # the full 20 Hz budget, not just the largest outliers
# How far into a frame run() waits for the reply before publishing the
# previous frame's output again (a held frame). Past this the camera frame
# would be dropped: modeld's own work around run() takes the rest of the 50 ms.
# A held frame is a plan one frame old, which the 10 s plan and the actuator
# delay both dwarf, where a dropped frame is the same stale plan plus a count
# toward selfdrived's modeldLagging. The joining model bounds how often
# (joining.HOLDS_ALLOWED). None waits the client's deadline, as before
HOLD_FRAME = 0.046
# A host that has answered nothing for this long while the small model drove
# is gone, not slow: the frames keep going out, so a quiet host shows here
# before any frame waits on it. Under the cable's 4 MB socket buffer, which
# holds ten frames, or the send would block the frame thread first
SHADOW_TIMEOUT = 0.4
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
    self.frames = 0
    self.over = 0            # frames past SLOW_FRAME, the whole 20 Hz budget
    self.held = 0            # frames whose reply was late and published the previous one
    self.shadowed = 0        # frames sent while the small model drove, never waited for
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
    """frames, over, held, shadowed, held_s, and over the frames kept: p50_ms,
    p99_ms, max_ms of the whole frame and server_ms, the server's mean total."""
    out = {'frames': self.frames, 'over': self.over, 'held': self.held, 'shadowed': self.shadowed,
           'held_s': round(time.monotonic() - self.started, 1)}
    if self._whole_ms:
      whole = sorted(self._whole_ms)
      out.update(p50_ms=round(whole[len(whole) // 2], 1), p99_ms=round(whole[int(0.99 * (len(whole) - 1))], 1),
                 max_ms=round(whole[-1], 1), server_ms=round(sum(self._server_ms) / len(self._server_ms), 1))
    return out


class JetlinkModelState:
  """Duck-types openpilot's modeld ModelState, and modeld_v2's where it differs.

  The small model is whatever bundle the user picked, on whichever modeld that
  bundle needs. sunnypilot's modeld_v2 reads constants, smoothing and the
  action function off the ModelState; stock modeld has them as module
  constants. This is comma's large model, so they are comma's.
  """

  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, client, spec, warp, *, face, log, event):
    from tinygrad.device import Device
    from tinygrad.tensor import Tensor
    self._tensor = Tensor
    self._log = log
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
    # Over a phone's cable, hand the socket the warp's GPU mapping itself: the
    # kernel copies it while the first segments are already on the wire, where
    # copying it here first held the send back 2.5 ms. On the Mac stand-in the
    # comma's side of a frame was 0.6 ms faster at p50 and 2 ms at p99. USB
    # keeps the host copy it was measured with. The client's transport says
    # which, as the owner lent it
    self.send_from_gpu = client.t.link_info().get('kind') == 'cable'
    # not chestnut hardware, but the same role: modelV2.big, the UI and the
    # model manager key off this flag
    self.chestnut = True

    # a warm warp, loaded ahead: the first call costs ~2 s and this can run on
    # modeld's frame thread (warp.warm)
    self.warp = warp

    self.input_shapes = spec.input_shapes
    self.output_slices = spec.output_slices
    # from the spec, not ModelConstants: the server derives its history stride
    # from the same field
    self.frame_skip = spec.frame_skip

    # the warp's own two inputs stay separate NPY tensors; everything past the
    # warp is the server's. Writing the numpy array is what feeds the JIT
    self.npy = {'tfm': np.zeros((3, 3), dtype=np.float32),
                'big_tfm': np.zeros((3, 3), dtype=np.float32)}
    self.warp_inputs = {k: Tensor(v, device='NPY').realize() for k, v in self.npy.items()}

    # compile_modeld.make_input_queues' packed_npy_inputs, minus the GPU queues
    # the server owns; for a stateful graph, minus prev_feat too
    self.packed = np.zeros(spec.packed_nelem, dtype=np.float32)
    self.npy.update({k: self.packed[at].reshape(shape) for k, (at, shape) in spec.packed_layout.items()})

    # read once: it must be the device the cached JIT was compiled against
    self.warp_dev = Device.DEFAULT
    self.prev_desire = np.zeros(face.desire_len, dtype=np.float32)
    self.parser = face.parser()
    self.frame_size = face.frame_size(cam_w, cam_h)
    # the camera buffers modeld hands over, whatever the graph calls its inputs
    self.vision_input_names = ['img', 'big_img']
    self.full_frames: dict = {}
    self._blob_cache: dict = {}
    self._need_reset = True
    self._frame_id = 0
    self._last_logged = 0.0
    self.trips = Trips()
    # whether the last run() published the frame before it (HOLD_FRAME)
    self.frame_held = False

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

  def _take_inputs(self, bufs: dict, transforms: dict[str, np.ndarray], inputs: dict[str, np.ndarray]) -> None:
    for key in bufs.keys():
      ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
      cache_key = (key, ptr)
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = self._tensor.from_blob(ptr, (self.frame_size,), dtype='uint8', device=self.warp_dev)
      self.full_frames[key] = self._blob_cache[cache_key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge.
    # Under whichever name the loop keyed it: stock modeld's desire_pulse, or a modeld_v2 bundle's own
    desire = inputs[next(k for k in inputs if k.startswith('desire'))]
    desire[0] = 0
    self.npy['desire'][:] = np.where(desire - self.prev_desire > .99, desire, 0)
    self.prev_desire[:] = desire
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']
    self.npy['tfm'][:, :] = transforms['img'][:, :]
    self.npy['big_tfm'][:, :] = transforms['big_img'][:, :]

  def _warp(self):
    """The warped frame as the wire takes it, and when the warp and the
    readback each finished."""
    warped = call_warp(self.warp, **self.warp_inputs, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])
    t1 = time.perf_counter()
    # .data() rather than .numpy(): same ~2.5 ms mean (a write-combined GPU
    # mapping read), but no per-frame allocation and no 52 ms outlier. The
    # mapping is safe to send: infer_begin returns only once the socket has
    # copied all of it, before the next warp can write it
    if self.send_from_gpu:
      data = warped._buffer().as_memoryview(allow_zero_copy=True)
    else:
      data = warped.data()
    return data, t1, time.perf_counter()

  def _send(self, data, want_telemetry: bool) -> tuple[int, bool]:
    """This frame to the host. Its seq, and whether it asked for the server's
    telemetry: on the frames modeld would have sent a chestnut's state on,
    its own callback's, or every TELEMETRY_EVERY-th."""
    self._frame_id += 1
    telemetry = want_telemetry or self._frame_id % TELEMETRY_EVERY == 0
    seq = self.client.infer_begin(data, self.packed, self._frame_id, reset=self._need_reset, want_state=telemetry)
    self._need_reset = False
    return seq, telemetry

  def shadow(self, bufs: dict, transforms: dict[str, np.ndarray], inputs: dict[str, np.ndarray]) -> None:
    """This frame to the host, without waiting for its answer: what the
    joining model does with every frame the small model drives, from the join
    on and after a hand-back. The host's model sees every frame, so its
    history and hidden state are current and warm at the swap, where a model
    that sat idle cost 110 to 120 ms on its first frame after every rejoin
    (an iPad, 2026-10-03). The answers are read here, a frame later, and
    dropped, bar the newest, which the first driving frame may hold. The
    frame thread's cost is the warp and the send, 5 to 7 ms. Raises LinkError
    when the host has answered nothing for SHADOW_TIMEOUT."""
    self.client.drain()
    waiting = self.client.waiting_for()
    if waiting > SHADOW_TIMEOUT:
      raise LinkError(f"the host has answered nothing for {waiting * 1e3:.0f} ms; link abandoned")
    self._take_inputs(bufs, transforms, inputs)
    t0 = time.perf_counter()
    data, t1, t2 = self._warp()
    _, telemetry = self._send(data, False)
    t3 = time.perf_counter()
    self.trips.shadowed += 1
    if telemetry:
      self.log_telemetry()
    if self.trips.shadowed <= 3 or t3 - t0 > SLOW_FRAME / 2:
      self._log.warning("jetlink: shadow frame %d warp %.1f data %.1f send %.1f ms; %d frames unanswered",
                        self._frame_id, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                        self.client.unanswered)

  def run(self, bufs: dict, transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
    self._take_inputs(bufs, transforms, inputs)
    t0 = time.perf_counter()
    data, t1, t2 = self._warp()
    seq, telemetry = self._send(data, after_enqueue is not None)
    t3 = time.perf_counter()
    # publish health while the Jetson works
    if after_enqueue is not None:
      after_enqueue()
    elif telemetry:
      self.log_telemetry()
    callback_done = time.perf_counter()
    # blocks like a chestnut frame, but not past HOLD_FRAME once there is an
    # output to publish again: a frame that long is a dropped camera frame.
    # Only a stall past the client's deadline raises, into the joining
    # state's demotion to the small model
    hold = None
    if HOLD_FRAME and self.client.last_output is not None:
      hold = HOLD_FRAME - (callback_done - t0)
    model_output = self.client.infer_end(seq, hold=hold)
    t4 = time.perf_counter()
    self.frame_held = model_output is None
    if self.frame_held:
      self.trips.held += 1
      model_output = self.client.last_output
    self.trips.record(t4 - t0, self.client.last_timings[2])
    # a frame past the budget is a dropped camera frame and three in a row
    # are modeldLagging; send against reply says which end it was
    if self._frame_id <= 3 or t4 - t0 > SLOW_FRAME or self.frame_held:
      # persisted on the comma so a drive can separate server execution from
      # receive stalls once the Jetson is offline; server total excludes USB
      gpu_us, queue_us, total_us = self.client.last_timings
      self._log.warning("jetlink: frame %d warp %.1f data %.1f send %.1f reply %.1f ms%s; "
                        "server gpu %.1f queue %.1f total %.1f ms", self._frame_id,
                        (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3,
                        ' held, the previous output again' if self.frame_held else '',
                        gpu_us / 1e3, queue_us / 1e3, total_us / 1e3)
      receive = getattr(self.client.t, 'last_receive', {})
      self._log.warning("jetlink: frame %d health %.1f wait %.1f ms; "
                        "ffs maxima prepare %.1f read_wait %.1f handoff %.1f ms", self._frame_id,
                        (callback_done - t3) * 1e3, (t4 - callback_done) * 1e3,
                        receive.get('prepare', 0.0) * 1e3, receive.get('read_wait', 0.0) * 1e3,
                        receive.get('handoff', 0.0) * 1e3)

    # the non-finite check runs on the server (Status.NOT_FINITE -> LinkError),
    # so modeld's big->small failover fires as it does for a chestnut
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    self.spec.feed_back(self.packed, model_output)
    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict

  def close(self) -> None:
    """Let go of the link; the joining state calls this on a model it retires."""
    self.client.close()

