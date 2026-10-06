"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma-side warp JIT: how it is built, where it lives, what loads it, and
the small model's reset for a fallback.

The warp stays on the comma (see model_state), and upstream's fused run_model
JIT (openpilot #38684) has no warp to borrow, so comma's make_warp graph is
JIT-compiled as a build target: the fork's build runs
`python -m jetlink.openpilot.warp` once per camera it builds for. A source
build makes the one for its own camera, a prebuilt release one for every
camera it installs on. Nothing compiles one at runtime: in modeld the ~9 s
compile would hold back the first frame on every ignition, and in a
provisioning run, which only runs offroad, it was lost to ignition. A device
without one runs the small model.

load() is what stands between a bad pickle and the car.

tinygrad is the fork's, and only modeld and the build have it: every import of
it here is inside the function that needs it.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# what TinyJit records for the keyword call in call_warp: sorted(kwargs)
WARP_INPUT_NAMES = ['big_frame', 'big_tfm', 'frame', 'tfm']
# KGSL allocation flags (msm_kgsl.h) for memory the GPU's accesses snoop the
# CPU's caches on, mapped write-back: KGSL_MEMFLAGS_IOCOHERENT, which
# tinygrad's kgsl bindings lack, and KGSL_CACHEMODE_WRITEBACK (3 << 26). See
# coherent_output
COHERENT_WRITEBACK = (1 << 31) | (3 << 26)
# input sets Replay keeps: modeld's two camera pools of 18 buffers advance
# together, so a drive sees a few dozen; past this, TinyJit checks every call
KNOWN_INPUT_SETS = 256


def call_warp(warp, tfm, big_tfm, frame, big_frame):
  """Call a warp JIT. Every caller goes through here, capture included.

  TinyJit names inputs from enumerate(args) plus sorted(kwargs) and refuses a
  call whose names differ from the capture. A positional compile and a keyword
  call raised JitError on the first frame of a drive.
  """
  return warp(tfm=tfm, big_tfm=big_tfm, frame=frame, big_frame=big_frame)


class Replay:
  """A warp JIT for modeld's frame loop, minus TinyJit's checks on inputs it
  has passed before. Called as the JIT is, through call_warp.

  TinyJit prepares every call's inputs and checks them against the capture
  before it replays it, a graph rewrite per input: 0.97 ms of a 1.93 ms warp
  call on the comma, on modeld's frame thread. The frame loop passes the same
  tensors frame after frame: the two NPY transforms, written in place, and
  the camera buffers, a fixed pool per stream. So the first call with a set
  goes through TinyJit, checks and all, and later ones replay the capture
  with the buffers prepared for it then, which is all TinyJit does with them.
  A JIT that has not captured, or a tinygrad without the same internals,
  goes through TinyJit every call.
  """

  def __init__(self, jit):
    self.jit = jit
    self._captured = getattr(jit, 'captured', None)
    try:
      from tinygrad.engine.jit import _prepare_jit_inputs
      self._prepare = _prepare_jit_inputs
    except ImportError:
      self._prepare = None
    # id()s of an input set -> (the set, its prepared buffers); the set is
    # kept so an id cannot be reused under it
    self._known: dict[tuple[int, ...], tuple[tuple, list]] = {}

  def __call__(self, *, tfm, big_tfm, frame, big_frame):
    inputs = (tfm, big_tfm, frame, big_frame)
    key = tuple(map(id, inputs))
    known = self._known.get(key)
    if known is not None and all(a is b for a, b in zip(known[0], inputs, strict=True)):
      return self._captured(known[1], {})
    out = call_warp(self.jit, tfm, big_tfm, frame, big_frame)
    if self._captured is not None and self._prepare is not None and len(self._known) < KNOWN_INPUT_SETS:
      bufs, var_vals, _, _ = self._prepare((), {'tfm': tfm, 'big_tfm': big_tfm, 'frame': frame, 'big_frame': big_frame})
      if not var_vals:
        self._known[key] = (inputs, bufs)
    return out


def init_device(log) -> None:
  """Bring the GPU up now, on the caller's thread.

  tinygrad initialises the device on its first kernel run and spawns a libusb
  event thread doing it. A thread created after config_realtime_process(7, 54)
  inherits SCHED_FIFO 54 and the core-7 pin and preempts the frame loop; that
  cost 5% of frames once. Failure is not a reason to refuse the accelerator.
  """
  try:
    from tinygrad.tensor import Tensor
    Tensor([0.0]).realize()
  except Exception:
    log.exception("jetlink: could not bring the gpu up before modeld goes realtime")
  # the same trap for tinygrad's compile pool (engine/worker.py): created on
  # the first compile, after modeld goes realtime, its handler threads sat at
  # SCHED_FIFO 54 on core 7. An older tinygrad or PARALLEL=0 is not a failure
  try:
    from tinygrad.engine.worker import get_worker_pool
    get_worker_pool()
  except Exception:
    log.exception("jetlink: could not start tinygrad's compile pool before modeld goes realtime")


class Warps:
  """The warps the build made for this device, as the fork's adapter says where."""

  def __init__(self, op):
    self.op = op
    # nothing compiles a warp at runtime and the build runs before manager, so
    # the answer holds for the life of the process; the UI asks at 5 Hz
    self._built: bool | None = None

  def geometry(self) -> tuple[int, int, int, int]:
    """(cam_w, cam_h, model_w, model_h) for this device: the choice modeld's
    build makes, so the warp built is the one modeld asks for. If they
    disagree, load() raises and the drive is small-model."""
    return tuple(self.op.camera())

  def path(self, cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
    return Path(self.op.warp_path(cam_w, cam_h, model_w, model_h))

  def is_cached(self, cam_w: int, cam_h: int, model_w: int, model_h: int) -> bool:
    """Is there a warp for this geometry?

    Presence only. Staleness is the build's job: the target depends on tinygrad
    and the capture sources. A pickle from an incompatible tinygrad raises in load().
    """
    return self.path(cam_w, cam_h, model_w, model_h).is_file()

  def built(self) -> bool:
    """Is there a warp for this device's camera? Without one the link cannot
    run the large model, which the offroad alert says (status.NO_WARP)."""
    if self._built is None:
      self._built = self.is_cached(*self.geometry())
    return self._built

  def load(self, cam_w: int, cam_h: int, model_w: int, model_h: int):
    """The built warp JIT. Raises if it is not there or is stale.

    modeld's big-model load is wrapped in the fallback to the small model, and a
    warp that cannot be trusted must not reach the car.
    """
    if not self.is_cached(cam_w, cam_h, model_w, model_h):
      raise RuntimeError(f"no warp built for {cam_w}x{cam_h} -> {model_w}x{model_h}; "
                         "the fork's build makes it (python -m jetlink.openpilot.warp)")
    with open(self.path(cam_w, cam_h, model_w, model_h), 'rb') as f:
      warp = pickle.load(f)

    # a JIT pickled before TinyJit captured loads fine and computes nothing; one
    # captured with a different call convention raises JitError on the first
    # frame of a drive. Both have happened
    captured = getattr(warp, 'captured', None)
    if captured is None:
      raise RuntimeError("cached warp was pickled before it captured; it computes nothing")
    names = list(getattr(captured, 'expected_names', []))
    if names != WARP_INPUT_NAMES:
      raise RuntimeError(f"cached warp expects {names}, call_warp passes {WARP_INPUT_NAMES}")
    return warp


def warm(warp, frame_size: int) -> None:
  """Run a loaded warp JIT until it is cheap to call. `frame_size` is the
  camera's NV12 buffer size (ModelFace.frame_size).

  Measured: loading 0.3 s, the first call 1.9 s, the second 5 ms. Paid on
  modeld's frame loop that was ~26 dropped frames and 16 s of modeldLagging
  after every join.
  """
  import numpy as np
  from tinygrad.device import Device
  from tinygrad.tensor import Tensor

  frames = [np.zeros(frame_size, dtype=np.uint8) for _ in range(2)]
  blobs = [Tensor.from_blob(f.ctypes.data, (frame_size,), dtype='uint8', device=Device.DEFAULT) for f in frames]
  eye = [np.eye(3, dtype=np.float32) for _ in range(2)]
  tfm, big_tfm = (Tensor(e, device='NPY').realize() for e in eye)
  for _ in range(2):
    call_warp(warp, tfm, big_tfm, blobs[0], blobs[1]).realize()
  Device.default.synchronize()


def coherent_output(warp, log) -> bool:
  """Put the warp's output in GPU memory the CPU reads through its cache.

  tinygrad maps a QCOM buffer write-combined, so the CPU reads it uncached:
  copying the 393 KB warp output out took 2.9 ms, and sending it straight
  from that mapping 7.3 ms (the kernel's copy). The Adreno 630 is
  IO-coherent, so a write-back buffer flagged KGSL_MEMFLAGS_IOCOHERENT is
  right to read the moment the GPU is done: 0.1 ms to copy, 0.22 ms to send,
  the GPU's own time unchanged (2026-10-06, comma four, real warp, every
  frame checked against a GPU-side copy).

  Call it before the warp's first call, which binds its buffers' addresses
  into the graph. False leaves the warp as it was: a device or kernel that
  keeps either flag back would hand out write-back memory that is not
  coherent, and the CPU would read stale frames from it.
  """
  try:
    from tinygrad.device import Device
    out = warp.captured.ret.uop.base.buffer
    if not out.device.startswith('QCOM') or out._base is not None:
      return False
    if _coherent(out._buf):
      return True
    dev = Device[out.device]
    mem = dev._gpu_alloc(out.nbytes, flags=COHERENT_WRITEBACK)
    if not _coherent(mem):
      dev._gpu_free(mem)
      log.warning("jetlink: the GPU driver kept back IO-coherent memory; the warp's output is read uncached")
      return False
    out.deallocate()
    out.allocate(opaque=mem)
    return True
  except Exception:
    log.exception("jetlink: could not move the warp's output to IO-coherent memory")
    return False


def coherent_view(warp):
  """(the tensor every warp call returns, the CPU's view of its bytes, the
  GPU's synchronize) once coherent_output moved it; None otherwise. The
  view is right to read, or send, once synchronize returns."""
  try:
    from tinygrad.device import Device
    ret = warp.captured.ret
    out = ret.uop.base.buffer
    if not _coherent(out._buf):
      return None
    return ret, out.as_memoryview(allow_zero_copy=True, no_sync=True), Device[out.device].synchronize
  except Exception:
    return None


def _coherent(mem) -> bool:
  """Does this QCOM allocation have both flags? The kernel's answer, which
  is what the allocation's record holds, not the request."""
  return getattr(mem.meta[0], 'flags', 0) & COHERENT_WRITEBACK == COHERENT_WRITEBACK


def prepare_reset(model):
  """Capture the small model's reset before driving, keeping its JIT's buffer identities.

  Its history is stale after the Jetson ran, so a fallback starts from the same
  zero history as modeld startup. Nothing is allocated or compiled on the
  failure frame.

  The small model is whatever bundle the user picked, stock modeld's or a
  modeld_v2 one, so the history is whatever GPU queues it has; the NPY tensors
  are zeroed through their numpy views. Duck-typed on openpilot's ModelState:
  input_queues, numpy_inputs (modeld_v2) or npy (modeld), prev_desire. The
  fork's tests pin those names.
  """
  from tinygrad import Tensor, TinyJit

  queues = tuple(q for q in model.input_queues.values() if q.device != 'NPY')
  npy = model.numpy_inputs if hasattr(model, 'numpy_inputs') else model.npy

  @TinyJit
  def clear():
    Tensor.realize(*(q.assign(0) for q in queues))

  for _ in range(3):
    clear()

  def reset():
    clear()
    model.prev_desire.fill(0)
    for array in npy.values():
      array.fill(0)

  return reset


# -- the build ------------------------------------------------------------------

def compile_warp(graph, frame_size: int, out: Path) -> Path:
  """JIT the warp graph and pickle it to `out`. Holds the GPU while it runs.

  Three runs before pickling: TinyJit captures on the second call, and a
  pickle taken earlier is an empty jit that silently does nothing.
  """
  import numpy as np
  from tinygrad.device import Device
  from tinygrad.engine.jit import TinyJit
  from tinygrad.tensor import Tensor

  warp_jit = TinyJit(graph, prune=True)

  # one set of input tensors: TinyJit captures against the buffers it is
  # first handed. Random so nothing constant-folds
  rng = np.random.default_rng(42)
  tfm_npy, big_tfm_npy = np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
  tfm = Tensor(tfm_npy, device='NPY')
  big_tfm = Tensor(big_tfm_npy, device='NPY')
  frame = Tensor.randint(frame_size, low=0, high=256, dtype='uint8', device=Device.DEFAULT).realize()
  big_frame = Tensor.randint(frame_size, low=0, high=256, dtype='uint8', device=Device.DEFAULT).realize()
  for _ in range(3):
    tfm_npy[:] = rng.standard_normal((3, 3)).astype(np.float32)
    big_tfm_npy[:] = rng.standard_normal((3, 3)).astype(np.float32)
    call_warp(warp_jit, tfm, big_tfm, frame, big_frame).realize()
  Device.default.synchronize()

  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  # through a temporary: a compile killed mid-write leaves nothing under the
  # name load() opens
  tmp = out.with_suffix('.pkl.tmp')
  with open(tmp, 'wb') as f:
    pickle.dump(warp_jit, f)
  tmp.replace(out)
  return out


def size(text: str) -> tuple[int, int]:
  """'WxH' as (w, h)."""
  w, sep, h = text.lower().partition('x')
  if not sep:
    raise argparse.ArgumentTypeError(f"expected WxH, not {text!r}")
  try:
    return int(w), int(h)
  except ValueError:
    raise argparse.ArgumentTypeError(f"expected WxH, not {text!r}") from None


def main(argv: list[str] | None = None) -> None:
  """Build the warp for one camera, as the fork's build runs it."""
  from jetlink.openpilot.interface import load_adapter
  p = argparse.ArgumentParser(prog='python -m jetlink.openpilot.warp', description=main.__doc__)
  p.add_argument('--adapter', required=True, help="the fork's adapter module")
  p.add_argument('--camera', type=size, required=True, help='camera resolution, WxH')
  p.add_argument('--model', type=size, required=True, help='model input, WxH')
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args(argv)

  (cam_w, cam_h), (model_w, model_h) = args.camera, args.model
  op = load_adapter(args.adapter)
  print(f"Compiling jetlink warp for {cam_w}x{cam_h} -> {model_w}x{model_h}...")
  # before anything of tinygrad's is imported here: comma's graph module
  # patches tinygrad's firmware fetch as it loads
  graph, frame_size = op.make_warp(cam_w, cam_h, model_w, model_h)
  out = compile_warp(graph, frame_size, args.output)
  print(f"  Saved to {out}")


if __name__ == "__main__":
  sys.exit(main())
