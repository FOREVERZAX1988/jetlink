"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's half of lossless frames (jetlink.lossless): MED errors on the GPU
from the warp's output, and Huffman packing on the CPU.

Measured on the mici (2026-10-06, scripts/comma/bench_lossless.py), warp
p50 0.96 ms alone:
- the errors as a second graph after the warp: +0.5 ms, bit for bit with
  jetlink.lossless.encode
- zstd at level -20 with Huffman forced on the literals, reading the errors
  from coherent memory: +1.8 ms, 2.1x on real frames. Negative levels leave
  literals raw (1.2-1.5x) unless ZSTD_c_literalCompressionMode says otherwise

tinygrad is the fork's: imported inside the functions that need it.
"""
from __future__ import annotations

import ctypes

import numpy as np

from jetlink.lossless import SHAPE

# zstd.h: ZSTD_c_compressionLevel, and ZSTD_c_literalCompressionMode
# (ZSTD_c_experimentalParam5, accepted by the shared library) set to
# ZSTD_ps_enable
_ZSTD_C_LEVEL = 100
_ZSTD_C_LITERAL_MODE = 1002
_ZSTD_PS_ENABLE = 1
# matching is not worth its time on these errors; Huffman alone is the gain
LEVEL = -20


def med(x):
  """The MED errors of a uint8 tensor (..., H, W), folded, as uint8: the
  tinygrad form of jetlink.lossless.encode."""
  from tinygrad import dtypes
  x = x.cast(dtypes.int32)
  p = x.pad(((0, 0),) * (x.ndim - 2) + ((1, 0), (1, 0)))
  a, b, c = p[..., 1:, :-1], p[..., :-1, 1:], p[..., :-1, :-1]
  mx, mn = a.maximum(b), a.minimum(b)
  r = (x - (c >= mx).where(mn, (c <= mn).where(mx, a + b - c)) + 256) % 256
  return (r < 128).where(r * 2, (256 - r) * 2 - 1).cast(dtypes.uint8)


class Errors:
  """The MED errors of a warp's output, on the GPU, into memory the CPU reads
  through its cache (warp._make_coherent): run() after the warp is started,
  then wait on the device, and `view` holds them."""

  def __init__(self, warped):
    from tinygrad import Tensor, TinyJit, dtypes
    from jetlink.openpilot.warp import _make_coherent
    self._out = Tensor.empty(*SHAPE, dtype=dtypes.uint8, device=warped.device).realize()
    buf = self._out.uop.base.buffer
    buf.ensure_allocated()
    if buf.device.startswith('QCOM'):
      # before the capture, which binds the graph to the buffer's address
      _make_coherent(buf)
    self._warped = warped
    self._jit = TinyJit(lambda x: self._out.assign(med(x)).realize())
    for _ in range(3):
      self._jit(warped)
    self.view = np.frombuffer(buf.as_memoryview(force_zero_copy=True, no_sync=True), np.uint8)

  def run(self) -> None:
    self._jit(self._warped)


class Packer:
  """Huffman packing of the errors through the system's libzstd."""

  def __init__(self, capacity: int = 2 * int(np.prod(SHAPE))):
    lib = ctypes.CDLL('libzstd.so.1')
    lib.ZSTD_createCCtx.restype = ctypes.c_void_p
    lib.ZSTD_CCtx_setParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ZSTD_CCtx_setParameter.restype = ctypes.c_size_t
    lib.ZSTD_compress2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_compress2.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    self._lib = lib
    self._cctx = lib.ZSTD_createCCtx()
    for key, value in ((_ZSTD_C_LEVEL, LEVEL), (_ZSTD_C_LITERAL_MODE, _ZSTD_PS_ENABLE)):
      if lib.ZSTD_isError(lib.ZSTD_CCtx_setParameter(self._cctx, key, value)):
        raise RuntimeError(f"libzstd refused parameter {key}={value}")
    self.out = np.empty(capacity, np.uint8)

  def pack(self, src: np.ndarray) -> memoryview:
    """The packed bytes of src, valid until the next pack()."""
    n = self._lib.ZSTD_compress2(self._cctx, self.out.ctypes.data, self.out.size, src.ctypes.data, src.nbytes)
    if self._lib.ZSTD_isError(n):
      raise RuntimeError("zstd could not pack the frame")
    return memoryview(self.out)[:n]
