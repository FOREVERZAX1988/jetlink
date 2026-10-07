#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Time lossless frames on the comma: the built warp alone, with the MED errors
on the GPU, and with Huffman packing, and check every step bit for bit
against jetlink.lossless.

    cd /data/openpilot && PYTHONPATH=/data/openpilot \\
      taskset -c 7 /usr/local/venv/bin/python3 jetlink_repo/scripts/comma/bench_lossless.py

The camera frames are synthetic (a gradient and noise), so the timings stand
and the packed size does not; scratchpad lossless/medbench.c measured real
frames.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time

import numpy as np

from jetlink import lossless
from jetlink.openpilot.lossless import Errors, Packer
from jetlink.openpilot.warp import Warp

WARP = '/data/openpilot/openpilot/sunnypilot/jetlink_adapter/models/warp_{w}x{h}_512x256_tinygrad.pkl'


def frames(size: int, width: int, n: int = 4) -> list[np.ndarray]:
  rng = np.random.default_rng(0)
  rows = size // width + 1
  yy, xx = np.mgrid[0:rows, 0:width]
  return [np.ascontiguousarray(((xx * (k + 1) // 7 + yy // 3) % 256 + rng.integers(0, 6, xx.shape))
                               .astype(np.uint8).ravel()[:size]) for k in range(n)]


def quantiles(ts: list[float]) -> str:
  ts = sorted(ts)
  return '  '.join(f"{name} {ts[min(len(ts) - 1, int(q * len(ts)))]:.2f}" for name, q in (('p50', .5), ('p90', .9), ('p99', .99)))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--camera', default='1344x760', help='WxH (mici 1344x760, tici 1928x1208)')
  p.add_argument('--n', type=int, default=300)
  args = p.parse_args()

  from openpilot.selfdrive.modeld.compile_modeld import NV12Frame
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from tinygrad.device import Device
  w, h = (int(v) for v in args.camera.split('x'))
  nv12 = NV12Frame(w, h, *get_nv12_info(w, h))
  log = logging.getLogger('bench_lossless')

  with open(WARP.format(w=w, h=h), 'rb') as f:
    jit = pickle.load(f)
  warp = Warp(jit, nv12.size, log)
  errors = Errors(jit.captured.ret)
  packer = Packer()
  sync = Device[jit.captured.ret.device].synchronize
  cams = frames(nv12.size, w)
  tfm = np.array([[1.3, 0, 100], [0, 1.3, 60], [0, 0, 1]], np.float32)

  for i in range(4):
    warp.start(cams[i].ctypes.data, cams[(i + 1) % 4].ctypes.data, tfm, tfm)
    errors.run()
    sync()
    warped = np.frombuffer(warp.output, np.uint8).reshape(lossless.SHAPE)
    assert (errors.view.reshape(lossless.SHAPE) == lossless.encode(warped)).all(), "GPU errors differ from the reference"
    packer.pack(errors.view)
  print("GPU errors match jetlink.lossless.encode")

  for name, step in (('warp', 0), ('warp + errors', 1), ('warp + errors + pack', 2)) * 2:
    ts, sizes = [], []
    for i in range(args.n):
      t0 = time.perf_counter()
      warp.start(cams[i % 4].ctypes.data, cams[(i + 1) % 4].ctypes.data, tfm, tfm)
      if step:
        errors.run()
      sync()
      if step == 2:
        sizes.append(len(packer.pack(errors.view)))
      ts.append((time.perf_counter() - t0) * 1000)
    extra = f"  {np.mean(sizes) / 1024:.0f} KB (synthetic)" if sizes else ''
    print(f"{name:22s} {quantiles(ts)} ms{extra}")


if __name__ == '__main__':
  main()
