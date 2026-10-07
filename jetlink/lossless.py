"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Lossless frames for links slower than USB (Wi-Fi): what the comma sends in
place of the warped frame, and how a server gets the frame back bit for bit.

Each 128x256 plane of the warp's output, (2 cameras, 6 planes, 128, 256)
uint8, is predicted pixel by pixel from its left (a), upper (b) and upper-left
(c) neighbours with JPEG-LS's MED predictor, zero outside the plane:

    pred = min(a, b) if c >= max(a, b)
           max(a, b) if c <= min(a, b)
           a + b - c otherwise

and the error (x - pred) mod 256 is folded so small errors either way are
small bytes: 0, -1, 1, -2, 2 ... -> 0, 1, 2, 3, 4 ... A Huffman coder then
packs those bytes: 2.1x on real drive frames (route 286, 2026-10-06).

Zero padding gives the edges JPEG-LS's rules without cases: the first row
predicts from the left, the first column from above, the first pixel from 0.

The comma computes the errors on its GPU with the warp (openpilot.lossless);
this module is the reference both ends and the tests check against, in numpy.
Decoding is sequential (each pixel needs its decoded neighbours), so a
server's decoder is native code; decode() here is for tests.
"""
from __future__ import annotations

import numpy as np

# the warp's output: two cameras of six 128x256 planes (4 Y phases, U, V)
SHAPE = (2, 6, 128, 256)


def predict(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
  """MED's prediction from the left, upper and upper-left neighbours."""
  mx, mn = np.maximum(a, b), np.minimum(a, b)
  return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


def fold(r: np.ndarray) -> np.ndarray:
  """(x - pred) mod 256 as a byte, small either way first: 0, -1, 1, -2 ..."""
  r = r % 256
  return np.where(r < 128, r * 2, (256 - r) * 2 - 1).astype(np.uint8)


def unfold(z: np.ndarray) -> np.ndarray:
  """fold's inverse, as an error mod 256."""
  z = z.astype(np.int32)
  return np.where(z % 2 == 0, z // 2, 256 - (z + 1) // 2)


def encode(frame: np.ndarray) -> np.ndarray:
  """The folded errors of planes (..., H, W) uint8, the same shape."""
  x = frame.astype(np.int32)
  pad = [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)]
  p = np.pad(x, pad)
  return fold(x - predict(p[..., 1:, :-1], p[..., :-1, 1:], p[..., :-1, :-1]))


def decode(errors: np.ndarray) -> np.ndarray:
  """encode's inverse. Row by row, pixel by pixel: slow, for tests."""
  r = unfold(errors)
  h, w = errors.shape[-2:]
  out = np.zeros(errors.shape[:-2] + (h + 1, w + 1), np.int32)   # zero row and column first
  for i in range(1, h + 1):
    for j in range(1, w + 1):
      pred = predict(out[..., i, j - 1], out[..., i - 1, j], out[..., i - 1, j - 1])
      out[..., i, j] = (pred + r[..., i - 1, j - 1]) % 256
  return out[..., 1:, 1:].astype(np.uint8)
