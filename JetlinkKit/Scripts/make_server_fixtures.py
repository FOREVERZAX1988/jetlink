#!/usr/bin/env python3
"""
Golden frames for the Swift server's tests, from the Python server's own parts.

Writes, into Tests/JetlinkServerTests/Fixtures:

  tiny_queued.onnx, tiny_stateful.onnx   tests/tiny_model.py's graphs, with the
                                          shapes onnx infers recorded, so the
                                          Swift preparation (which has no shape
                                          inferrer) rewrites what Python does
  <name>.spec.json                        spec_from_onnx(...).to_dict()
  <name>.frames.bin                       per frame: warped uint8, then the packed
                                          float32 protocol 2 sent: for the queued
                                          graph the scalars and then a prev_feat
  <name>.expected.bin                     per frame: the driving output, float32
  tiny_queued.fed.expected.bin            the same, the server feeding back its
                                          own hidden state as protocol 3 does

The expected outputs are onnxruntime's CPU provider on the graph as the
simplify branch prepares it for CoreML (whole, not split), fed by the
server's own staging: PolicyQueues for the queued graph, each next_state_
output fed back for the stateful one. The Swift server runs the same prepared
graph on the same provider, so its outputs must match bit for bit.

The files from protocol 2 are unchanged: the queued graph's frames carry a
prev_feat of their own, which the server's feedback path takes in here as a
hidden state kept from the frame before, giving protocol 2's outputs bit for
bit. Over the link a comma can no longer send one, so the served outputs are
the fed ones.

  .venv/bin/python JetlinkKit/Scripts/make_server_fixtures.py [--out DIR]

from the root of this checkout, so the fixtures pin the Python beside them.
tests/test_conformance.py runs it into a temporary directory and compares.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
# This checkout's jetlink and tests, ahead of any jetlink the environment has
# installed: a venv's editable install can point at another checkout.
sys.path.insert(0, str(ROOT))
from tests import tiny_model  # noqa: E402

from jetlink.queues import PolicyQueues  # noqa: E402
from jetlink.server.backends.ort import _prepared_model  # noqa: E402
from jetlink.spec import DRIVING_OUTPUT, spec_from_onnx  # noqa: E402

OUT = ROOT / 'JetlinkKit' / 'Tests' / 'JetlinkServerTests' / 'Fixtures'
FRAMES = 8
ORT_DTYPES = {'tensor(float16)': np.float16, 'tensor(float)': np.float32, 'tensor(uint8)': np.uint8}


def session(path: Path):
  prepared = _prepared_model(path, for_coreml=True)
  return ort.InferenceSession(prepared.SerializeToString(), providers=['CPUExecutionProvider'])


def write_outputs(path: Path, outputs: list[np.ndarray]) -> None:
  with open(path, 'wb') as f:
    for output in outputs:
      f.write(np.asarray(output, np.float32).reshape(-1).tobytes())


def write(out: Path, name: str, spec, frames: list[tuple[np.ndarray, np.ndarray]], outputs: list[np.ndarray],
          packed_nelem: int) -> None:
  (out / f'{name}.spec.json').write_text(json.dumps(spec.to_dict(), indent=2) + '\n')
  with open(out / f'{name}.frames.bin', 'wb') as f:
    for warped, packed in frames:
      assert warped.dtype == np.uint8 and warped.nbytes == spec.warped_nbytes
      assert packed.dtype == np.float32 and packed.size == packed_nelem
      f.write(warped.tobytes())
      f.write(packed.tobytes())
  write_outputs(out / f'{name}.expected.bin', outputs)


def kept(spec, hidden: np.ndarray) -> dict[str, np.ndarray]:
  """A driving output carrying `hidden` as its hidden state, for after_run."""
  output = np.zeros(spec.output_nelem, np.float32)
  output[slice(*spec.hidden_range)] = hidden
  return {DRIVING_OUTPUT: output}


def queued(out: Path, rng) -> None:
  path = out / 'tiny_queued.onnx'
  tiny_model.write(path, shapes=True)
  spec = spec_from_onnx(str(path))
  sess = session(path)
  types = {i.name: ORT_DTYPES[i.type] for i in sess.get_inputs()}

  def run(queues, warped, packed):
    feed = {n: v.astype(types[n]).reshape(spec.input_shapes[n]) for n, v in queues.step(warped, packed).items()}
    return sess.run(['outputs'], feed)[0]

  # protocol 2's frames: the scalars, then a prev_feat the comma sent,
  # taken in through the feedback path
  queues = PolicyQueues(spec)
  n = spec.packed_nelem
  frames, outputs = [], []
  for _ in range(FRAMES):
    warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
    sent = (rng.standard_normal(n + spec.feat_dim) * 0.5).astype(np.float32)
    queues.after_run(kept(spec, sent[n:]), {})
    outputs.append(run(queues, warped, sent[:n]))
    frames.append((warped, sent))
  write(out, 'tiny_queued', spec, frames, outputs, n + spec.feat_dim)

  # the same images and scalars over protocol 3: the server feeds back its own
  queues = PolicyQueues(spec)
  fed = []
  for warped, sent in frames:
    output = run(queues, warped, sent[:n])
    assert np.all(np.isfinite(output))
    queues.after_run({DRIVING_OUTPUT: output}, {})
    fed.append(output)
  write_outputs(out / 'tiny_queued.fed.expected.bin', fed)


def stateful(out: Path, rng) -> None:
  path = out / 'tiny_stateful.onnx'
  tiny_model.write_stateful(path, shapes=True)
  spec = spec_from_onnx(str(path))
  sess = session(path)
  types = {i.name: ORT_DTYPES[i.type] for i in sess.get_inputs()}
  pairs = spec.state_pairs
  state = {n: np.zeros(spec.input_shapes[n], types[n]) for n in pairs}
  frames, outputs = [], []
  for _ in range(FRAMES):
    warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
    packed = (rng.standard_normal(spec.packed_nelem) * 0.5).astype(np.float32)
    feed = dict(state)
    feed['new_img'] = warped.astype(types['new_img'])
    for name, (s, _shape) in spec.packed_layout.items():
      feed[name] = packed[s].reshape(spec.input_shapes[name]).astype(types[name])
    names = ['outputs'] + list(pairs.values())
    results = dict(zip(names, sess.run(names, feed), strict=True))
    outputs.append(results['outputs'])
    state = {n: results[nxt].astype(types[n]) for n, nxt in pairs.items()}
    frames.append((warped, packed))
  write(out, 'tiny_stateful', spec, frames, outputs, spec.packed_nelem)


def generate(out: Path = OUT) -> None:
  out.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(20260926)
  queued(out, rng)
  stateful(out, rng)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--out', type=Path, default=OUT, help='where to write (default: the Swift tests\' Fixtures)')
  out = parser.parse_args().out
  generate(out)
  for p in sorted(out.iterdir()):
    print(f'{p.stat().st_size:>8}  {p.name}')
