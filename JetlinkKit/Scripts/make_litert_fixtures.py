#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The tiny graphs as LiteRT runs them, for the Swift LiteRT backend's tests:
JetlinkKit/Tests/JetlinkServerTests/Fixtures/tiny_queued.tflite and
tiny_stateful.tflite, from the committed .onnx beside them. The tests convert
nothing: they hand the backend these in place of the on-device conversion.
tiny_stateful_5d.tflite is the stateful graph with its frame queue left 5-D,
which LiteRT's GPU cannot run, for the test that a build says so.

The conversion is the LiteRT spike's (plans/litert/spike, convert.sh), which
the on-device LiteRTPreparation follows for the real models:

  - the tinygrad Contiguous op bypassed (it is an identity);
  - the uint8 frame queue [2,5,6,H,W] as the 4-D view [2,30,H,W], the same
    bytes, its frames read by Slice instead of Gather (LiteRT's GPU takes
    nothing above rank 4);
  - onnx2tf 2.6.9 with every input in ONNX's layout (-kat), so the server
    stages the same bytes it stages for onnxruntime and each next_state_
    output has its state_ input's shape;
  - float32 weights stored as float16 behind a DEQUANTIZE, the layout of
    TFLite's float16 quantization, which the CPU and the GPU fold back.

  uv venv -p python3.12 .venv-litert && uv pip install -p .venv-litert onnx2tf==2.6.9 onnxruntime
  PATH="$PWD/.venv-litert/bin:$PATH" .venv-litert/bin/python JetlinkKit/Scripts/make_litert_fixtures.py

from the root of this checkout (onnx2tf runs onnxsim, which must be on PATH).
Each model is checked against onnxruntime on the original graph before it is
written.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'JetlinkKit' / 'Tests' / 'JetlinkServerTests' / 'Fixtures'
# (output, source graph, frame queue as the 4-D view)
MODELS = (('tiny_queued', 'tiny_queued', False), ('tiny_stateful', 'tiny_stateful', True),
          ('tiny_stateful_5d', 'tiny_stateful', False))


def strip_tinygrad(model: onnx.ModelProto) -> None:
  """Bypasses org.tinygrad nodes, identities that only order the graph."""
  g = model.graph
  rename = {}
  keep = []
  for n in g.node:
    if n.domain == 'org.tinygrad':
      assert len(n.input) == 1 and len(n.output) == 1, n
      rename[n.output[0]] = n.input[0]
    else:
      keep.append(n)
  for n in keep:
    for k, name in enumerate(n.input):
      n.input[k] = rename.get(name, name)
  del g.node[:]
  g.node.extend(keep)
  opsets = [o for o in model.opset_import if o.domain != 'org.tinygrad']
  del model.opset_import[:]
  model.opset_import.extend(opsets)


def frame_queue_4d(model: onnx.ModelProto) -> None:
  """tiny_stateful's uint8 frame queue as [B, F*C, H, W]: the next queue is
  the last F-1 frames and the new pair, and each camera's frames 0 and F-1
  are channel slices of it. The nodes this replaces are named by
  tests/tiny_model.py."""
  g = model.graph
  q = next(i for i in g.input if i.name == 'state_img_q')
  B, F, C, H, W = (d.dim_value for d in q.type.tensor_type.shape.dim)
  replaced = {'unsqueeze', 'img_tail', 'next_state_img_q', 'road', 'wide', 'road_pair', 'wide_pair', 'road_img', 'wide_img'}
  assert replaced <= {o for n in g.node for o in n.output}, 'not tiny_stateful'

  def const(name, values):
    g.initializer.append(numpy_helper.from_array(np.array(values, np.int64), name))
    return name

  nodes = [helper.make_node('Slice', ['state_img_q', const('q_from', [C]), const('q_to', [F * C]), const('q_axis', [1])], ['img_tail4']),
           helper.make_node('Concat', ['img_tail4', 'new_img'], ['next_state_img_q'], axis=1)]
  for b, camera in enumerate(('road_img', 'wide_img')):
    parts = []
    for f in (0, F - 1):
      part = f'{camera}_f{f}'
      nodes.append(helper.make_node('Slice', ['next_state_img_q', const(f'{part}_from', [b, f * C]),
                                              const(f'{part}_to', [b + 1, (f + 1) * C]), const(f'{part}_axes', [0, 1])], [part]))
      parts.append(part)
    nodes.append(helper.make_node('Concat', parts, [camera], axis=1))
  first = next(k for k, n in enumerate(g.node) if n.output[0] in replaced)
  rest = [n for n in g.node if n.output[0] not in replaced]
  del g.node[:]
  g.node.extend(rest[:first] + nodes + rest[first:])
  for v in list(g.input) + list(g.output):
    if v.name in ('state_img_q', 'next_state_img_q'):
      dims = v.type.tensor_type.shape.dim
      del dims[:]
      for d in (B, F * C, H, W):
        dims.add().dim_value = d
  del g.value_info[:]


def fp16_weights(src: Path, dst: Path, minimum: int = 1024) -> None:
  """Every float32 constant of `minimum` elements or more stored as float16,
  read through a DEQUANTIZE (the spike's fp16_weights.py)."""
  import flatbuffers
  from ai_edge_litert import schema_py_generated as S

  buf = src.read_bytes()
  m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(buf, 0))
  deq = next((i for i, oc in enumerate(m.operatorCodes)
              if max(oc.builtinCode, oc.deprecatedBuiltinCode) == S.BuiltinOperator.DEQUANTIZE), None)
  if deq is None:
    oc = S.OperatorCodeT()
    oc.builtinCode = oc.deprecatedBuiltinCode = S.BuiltinOperator.DEQUANTIZE
    oc.version = 3  # float16 to float32
    m.operatorCodes.append(oc)
    deq = len(m.operatorCodes) - 1
  for sg in m.subgraphs:
    constants = sorted({int(t) for op in sg.operators for t in (op.inputs if op.inputs is not None else []) if t >= 0})
    added = []
    for ti in constants:
      t = sg.tensors[ti]
      data = m.buffers[t.buffer].data if t.buffer else None
      if t.type != S.TensorType.FLOAT32 or data is None or len(data) == 0:
        continue
      values = np.frombuffer(bytes(data), np.float32)
      if values.size < minimum:
        continue
      half = S.BufferT()
      half.data = np.frombuffer(values.astype(np.float16).tobytes(), np.uint8)
      m.buffers.append(half)
      m.buffers[t.buffer].data = None
      t16 = S.TensorT()
      t16.shape, t16.type, t16.buffer = t.shape, S.TensorType.FLOAT16, len(m.buffers) - 1
      t16.name = (t.name.decode() if isinstance(t.name, bytes) else t.name) + '_fp16'
      sg.tensors.append(t16)
      op = S.OperatorT()
      op.opcodeIndex = deq
      op.inputs = np.array([len(sg.tensors) - 1], np.int32)
      op.outputs = np.array([ti], np.int32)
      added.append(op)
    sg.operators = added + sg.operators
  builder = flatbuffers.Builder(1 << 20)
  builder.Finish(m.Pack(builder), file_identifier=b'TFL3')
  dst.write_bytes(builder.Output())


def convert(name: str, out: Path, four_d: bool) -> None:
  model = onnx.load(str(FIXTURES / f'{name}.onnx'))
  strip_tinygrad(model)
  if four_d:
    frame_queue_4d(model)
  model = shape_inference.infer_shapes(model, strict_mode=True)
  onnx.checker.check_model(model)
  inputs = [i.name for i in model.graph.input]
  with tempfile.TemporaryDirectory() as tmp:
    prepared = Path(tmp) / f'{name}.onnx'
    onnx.save(model, str(prepared))
    run = subprocess.run(['onnx2tf', '-i', str(prepared), '-o', tmp, '-kat', *inputs], capture_output=True, text=True)
    if run.returncode != 0:
      sys.exit(f'{name}: onnx2tf failed:\n{run.stdout}{run.stderr}')
    fp16_weights(Path(tmp) / f'{name}_float32.tflite', out)


def check(name: str, tflite: Path, frames: int = 3) -> float:
  """The worst correlation of the converted model's outputs with
  onnxruntime's on the original graph (less the tinygrad op, which
  onnxruntime does not know), over a few frames of random inputs: closed loop
  for the stateful graph, its queue read through the 4-D view."""
  import onnxruntime as ort
  from ai_edge_litert.interpreter import Interpreter

  original = onnx.load(str(FIXTURES / f'{name}.onnx'))
  strip_tinygrad(original)
  session = ort.InferenceSession(original.SerializeToString())
  lite = Interpreter(model_path=str(tflite))
  lite.allocate_tensors()
  runner = lite.get_signature_runner()
  details = runner.get_input_details()
  rng = np.random.default_rng(5)
  state = {i.name: np.zeros(i.shape, np.uint8 if i.type == 'tensor(uint8)' else np.float32)
           for i in session.get_inputs() if i.name.startswith('state_')}
  worst = 1.0
  for _ in range(frames):
    feed = {}
    for i in session.get_inputs():
      if i.name in state:
        feed[i.name] = state[i.name]
      elif i.type == 'tensor(uint8)':
        feed[i.name] = rng.integers(0, 256, i.shape, dtype=np.uint8)
      else:
        feed[i.name] = rng.standard_normal(i.shape).astype(np.float16 if i.type == 'tensor(float16)' else np.float32)
    expected = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
    got = runner(**{k: v.astype(details[k]['dtype']).reshape(details[k]['shape']) for k, v in feed.items()})
    a, b = expected['outputs'].astype(np.float64).ravel(), got['outputs'].astype(np.float64).ravel()
    worst = min(worst, float(np.corrcoef(a, b)[0, 1]))
    for k in state:
      nxt = expected[f'next_{k}']
      assert np.array_equal(got[f'next_{k}'].reshape(nxt.shape), nxt) if nxt.dtype == np.uint8 else True, k
      state[k] = nxt
  return worst


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--out', type=Path, default=FIXTURES, help='where to write the .tflite files')
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  for output, name, four_d in MODELS:
    out = args.out / f'{output}.tflite'
    convert(name, out, four_d)
    worst = check(name, out)
    print(f'{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {out.stat().st_size} bytes, outputs correlate {worst:.6f}')
    if worst < 0.999:
      sys.exit(f'{name}: the conversion does not match onnxruntime')


if __name__ == '__main__':
  main()
