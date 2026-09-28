"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Which runtime runs the model.

    trt       TensorRT on an NVIDIA GPU: the Jetson, and a desktop or laptop
    ort       onnxruntime: CoreML on Apple silicon, CUDA or CPU elsewhere

Runtimes are imported only when chosen, never here: the tests import the
server on machines with none of them, and a Jetson must not pay for a Mac's
imports.

`auto` prefers TensorRT, then onnxruntime: CoreML on a Mac, CUDA or the CPU
elsewhere.
"""
from __future__ import annotations

import importlib.util
import logging

from jetlink.server.backends.base import Backend

log = logging.getLogger('jetlink.backends')

NAMES = ('trt', 'ort')


def _importable(module: str) -> bool:
  try:
    return importlib.util.find_spec(module) is not None
  except (ImportError, ValueError):
    return False


def _make(name: str, device: str) -> Backend:
  if name == 'trt':
    from jetlink.server.backends.trt import TrtBackend
    return TrtBackend(device)
  if name == 'ort':
    from jetlink.server.backends.ort import OrtBackend
    return OrtBackend(device)
  raise ValueError(f"unknown backend {name!r}; one of {NAMES} or auto")


def available() -> list[str]:
  """Backends whose runtime is installed, in auto's order of preference."""
  found = []
  if _importable('tensorrt') and (_importable('cuda.bindings') or _importable('cuda')):
    found.append('trt')
  if _importable('onnxruntime'):
    found.append('ort')
  return found


def _candidates(device: str) -> list[tuple[str, str]]:
  """(backend, device) pairs auto tries, in order: every installed backend,
  each given the --device the user gave, which the ones it means nothing to
  skip."""
  return [(name, device) for name in available()]


def select(name: str = 'auto', device: str = 'auto') -> Backend:
  """The backend to serve with. A named backend that will not come up raises;
  `auto` moves on to the next and says why."""
  if name != 'auto':
    return _make(name, device)
  candidates = _candidates(device)
  if not candidates:
    raise RuntimeError("no inference runtime is installed: pip install one of "
                       "'jetlink[trt]' or 'jetlink[ort]'")
  reasons = []
  for candidate, dev in candidates:
    try:
      backend = _make(candidate, dev)
    except Exception as e:
      reasons.append(f"{candidate} ({dev}): {type(e).__name__}: {e}")
      log.warning("backend %s (%s) not used: %s", candidate, dev, e)
      continue
    log.info("backend %s %s on %s", backend.name, backend.describe()['runtime_version'],
             backend.describe()['device'])
    return backend
  raise RuntimeError("no backend came up:\n  " + "\n  ".join(reasons))
