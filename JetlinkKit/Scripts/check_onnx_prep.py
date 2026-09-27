"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Checks the Swift ONNX preparation (JetlinkONNX, through the jetlink-onnx CLI)
against the Python one it ports: the `simplify` branch's
`_prepared_model(for_coreml=True)`, then `split_vision_policy` unless
--whole, then `_with_cache_key` per part.

    PYTHONPATH=../jetlink-simplify ../jetlink/.venv/bin/python \\
      JetlinkKit/Scripts/check_onnx_prep.py [--whole | --both] [--ort] [--cli PATH] model.onnx ...

For each model and layout it compares, with the onnx package:
  - nodes, in order: op_type, name, domain, inputs, outputs, attributes by value;
  - initializers by name: dims, data_type and bit-equal data (and their order);
  - graph inputs and outputs: name, elem type, dims;
  - value_info by name (the whole message);
  - opset_import, ir_version, metadata_props, and the other model and graph fields;
  - the weight bytes each side reports per part;
and says whether the files are byte for byte the same. With --ort (always on
for graphs under 64 MB) it runs both chains with onnxruntime's CPU provider on
the same random inputs, vision's outputs fed into policy by name, and needs
every output bit for bit equal. When Python raises, the Swift side must fail
with the same message.

The Python parts stay in memory; only the Swift side writes files, into a
temporary directory removed after each model. Prepared real models are about
0.8 GB, so free disk is checked first.

Two of the test fixtures fail here on purpose, because Swift has no shape
inferrer: noshape.onnx [split] (the cut tensor's shape is not recorded; Python
infers it, Swift refuses) and unrecorded.onnx [whole] (a Gather whose input
shape is not recorded; Python infers it and rewrites the index, Swift leaves
it). The Swift tests hold both to the Swift behaviour.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI = ROOT / '.build' / 'release' / 'jetlink-onnx'
IMG_INPUTS = ('img', 'big_img', 'new_img', 'state_img_q')
ORT_ALWAYS_BELOW = 64 << 20
NUMPY = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64,
         9: np.bool_, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64}


def cache_key(stem: str, part: str) -> str:
  # jetlink.server.backends.ort._cache_key, with the stem given directly
  return re.sub(r'[^A-Za-z0-9]', '', stem + part)[:63]


def python_parts(src: Path, whole: bool, prefix: str) -> dict[str, onnx.ModelProto]:
  from jetlink.onnx_patch import split_vision_policy
  from jetlink.server.backends.ort import _prepared_model, _with_cache_key
  model = _prepared_model(src, for_coreml=True)
  names = ('model',) if whole else ('vision', 'policy')
  parts = (model,) if whole else split_vision_policy(model)
  return {name: _with_cache_key(part, cache_key(prefix, name)) for name, part in zip(names, parts, strict=True)}


def swift_parts(cli: Path, src: Path, out: Path, whole: bool, prefix: str) -> tuple[dict[str, Path], dict, str]:
  """(part files, {part: weight bytes}, the CLI's output); raises with the CLI's error."""
  cmd = [str(cli), 'prepare', str(src), str(out), '--key-prefix', prefix] + (['--whole'] if whole else [])
  run = subprocess.run(cmd, capture_output=True, text=True)
  if run.returncode != 0:
    raise SwiftFailed(run.stderr.strip().removeprefix('error: '))
  files, weights = {}, {}
  for line in run.stdout.splitlines():
    m = re.match(r'\s+(\w+)\s+(\S+)\s+\d+ bytes, weights (\d+) bytes', line)
    if m:
      files[m.group(1)] = Path(m.group(2))
      weights[m.group(1)] = int(m.group(3))
  return files, weights, run.stdout


class SwiftFailed(Exception):
  pass


# -- the structural comparison ------------------------------------------------

def attr_value(a: onnx.AttributeProto):
  v = helper.get_attribute_value(a)
  if isinstance(v, onnx.TensorProto):
    arr = numpy_helper.to_array(v)
    return ('tensor', arr.dtype.str, arr.shape, arr.tobytes())
  if isinstance(v, (onnx.GraphProto, onnx.TypeProto, onnx.SparseTensorProto)):
    return ('proto', v.SerializeToString())
  if isinstance(v, list) and v and hasattr(v[0], 'SerializeToString'):
    return ('protos', [x.SerializeToString() for x in v])
  if isinstance(v, float):
    return ('float', np.float32(v).tobytes())
  if isinstance(v, list) and v and isinstance(v[0], float):
    return ('floats', np.array(v, np.float32).tobytes())
  return v


def node_key(n: onnx.NodeProto):
  return (n.op_type, n.name, n.domain, list(n.input), list(n.output),
          [(a.name, a.type, attr_value(a)) for a in n.attribute])


def io_key(vi: onnx.ValueInfoProto):
  tt = vi.type.tensor_type
  dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in tt.shape.dim]
  return (vi.name, tt.elem_type, dims)


def tensor_values(t: onnx.TensorProto):
  arr = numpy_helper.to_array(t)
  return arr.dtype.str, arr.shape, arr.tobytes()


def compare(py: onnx.ModelProto, sw: onnx.ModelProto, limit: int = 8) -> list[str]:
  diffs: list[str] = []

  def differ(what: str, a, b):
    if a != b:
      diffs.append(f'{what}: python {short(a)} / swift {short(b)}')

  differ('ir_version', py.ir_version, sw.ir_version)
  differ('opset_import', [(o.domain, o.version) for o in py.opset_import],
         [(o.domain, o.version) for o in sw.opset_import])
  differ('metadata_props', [(p.key, p.value) for p in py.metadata_props], [(p.key, p.value) for p in sw.metadata_props])
  for field in ('producer_name', 'producer_version', 'domain', 'model_version', 'doc_string'):
    differ(field, getattr(py, field), getattr(sw, field))
  differ('functions', [f.SerializeToString() for f in py.functions], [f.SerializeToString() for f in sw.functions])
  differ('training_info', len(py.training_info), len(sw.training_info))
  pg, sg = py.graph, sw.graph
  differ('graph name', pg.name, sg.name)
  differ('graph doc_string', pg.doc_string, sg.doc_string)
  differ('graph metadata_props', [(p.key, p.value) for p in pg.metadata_props],
         [(p.key, p.value) for p in sg.metadata_props])
  differ('sparse_initializer', len(pg.sparse_initializer), len(sg.sparse_initializer))
  differ('quantization_annotation', len(pg.quantization_annotation), len(sg.quantization_annotation))

  differ('node count', len(pg.node), len(sg.node))
  bad = [i for i, (a, b) in enumerate(zip(pg.node, sg.node, strict=False)) if node_key(a) != node_key(b)]
  for i in bad[:limit]:
    diffs.append(f'node {i}: python {short(node_key(pg.node[i]))} / swift {short(node_key(sg.node[i]))}')
  if len(bad) > limit:
    diffs.append(f'... and {len(bad) - limit} more nodes differ')

  differ('initializer order', [t.name for t in pg.initializer], [t.name for t in sg.initializer])
  pi, si = {t.name: t for t in pg.initializer}, {t.name: t for t in sg.initializer}
  differ('initializer names', sorted(pi), sorted(si))
  bad = []
  for name in sorted(pi.keys() & si.keys()):
    a, b = pi[name], si[name]
    if list(a.dims) != list(b.dims) or a.data_type != b.data_type or tensor_values(a) != tensor_values(b):
      bad.append(name)
  for name in bad[:limit]:
    diffs.append(f'initializer {name}: dims {list(pi[name].dims)}/{list(si[name].dims)}, '
                 f'type {pi[name].data_type}/{si[name].data_type}, data differs')
  if len(bad) > limit:
    diffs.append(f'... and {len(bad) - limit} more initializers differ')

  differ('graph inputs', [io_key(v) for v in pg.input], [io_key(v) for v in sg.input])
  differ('graph outputs', [io_key(v) for v in pg.output], [io_key(v) for v in sg.output])
  differ('value_info order', [v.name for v in pg.value_info], [v.name for v in sg.value_info])
  pv = {v.name: v.SerializeToString() for v in pg.value_info}
  sv = {v.name: v.SerializeToString() for v in sg.value_info}
  differ('value_info names', sorted(pv), sorted(sv))
  bad = [n for n in sorted(pv.keys() & sv.keys()) if pv[n] != sv[n]]
  for n in bad[:limit]:
    diffs.append(f'value_info {n} differs')
  if len(bad) > limit:
    diffs.append(f'... and {len(bad) - limit} more value_info entries differ')
  return diffs


def short(v, width: int = 160) -> str:
  s = repr(v)
  return s if len(s) <= width else s[:width] + '...'


# -- onnxruntime ----------------------------------------------------------------

def random_feeds(parts: list[onnx.ModelProto], seed: int = 0) -> dict[str, np.ndarray]:
  """Inputs for the chain: every part's inputs that no earlier part produces."""
  rng = np.random.default_rng(seed)
  produced: set[str] = set()
  feeds = {}
  for part in parts:
    for vi in part.graph.input:
      if vi.name in produced or vi.name in feeds:
        continue
      tt = vi.type.tensor_type
      shape = [d.dim_value if d.HasField('dim_value') else 1 for d in tt.shape.dim]
      dtype = NUMPY[tt.elem_type]
      if vi.name in IMG_INPUTS or dtype == np.uint8:
        feeds[vi.name] = rng.integers(0, 256, shape).astype(dtype)
      elif np.issubdtype(dtype, np.floating):
        feeds[vi.name] = (rng.standard_normal(shape) * 0.5).astype(dtype)
      elif dtype == np.bool_:
        feeds[vi.name] = rng.integers(0, 2, shape).astype(dtype)
      else:
        feeds[vi.name] = rng.integers(0, 4, shape).astype(dtype)
    produced.update(o.name for o in part.graph.output)
  return feeds


def run_chain(sources: list, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  import onnxruntime as ort
  opts = ort.SessionOptions()
  opts.log_severity_level = 3
  values = dict(feeds)
  results = {}
  for source in sources:
    sess = ort.InferenceSession(source, opts, providers=['CPUExecutionProvider'])
    outs = sess.run(None, {i.name: values[i.name] for i in sess.get_inputs()})
    for o, v in zip(sess.get_outputs(), outs, strict=True):
      values[o.name] = v
      results[o.name] = v
    del sess
  return results


def compare_runs(py: dict[str, np.ndarray], sw: dict[str, np.ndarray]) -> list[str]:
  diffs = []
  if sorted(py) != sorted(sw):
    diffs.append(f'ort outputs: python {sorted(py)} / swift {sorted(sw)}')
  for name in sorted(py.keys() & sw.keys()):
    a, b = py[name], sw[name]
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
      diffs.append(f'ort output {name}: not bit-equal (max abs diff '
                   f'{np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) if a.shape == b.shape else "shape"})')
  return diffs


# -- one model -----------------------------------------------------------------

def check(src: Path, whole: bool, cli: Path, workdir: Path, prefix: str, ort_run: bool) -> bool:
  layout = 'whole' if whole else 'split'
  label = f'{src.name} [{layout}]'
  out = workdir / f'{src.stem}.{layout}'
  shutil.rmtree(out, ignore_errors=True)
  try:
    t0 = time.perf_counter()
    try:
      py = python_parts(src, whole, prefix)
      py_error = None
    except Exception as e:
      py, py_error = None, str(e)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
      files, weights, _ = swift_parts(cli, src, out, whole, prefix)
      sw_error = None
    except SwiftFailed as e:
      files, weights, sw_error = {}, {}, str(e)
    t_sw = time.perf_counter() - t0

    if py_error or sw_error:
      if py_error == sw_error:
        print(f'PASS {label}: both refuse it: {py_error}')
        return True
      print(f'FAIL {label}: python {"raised " + repr(py_error) if py_error else "succeeded"}, '
            f'swift {"failed with " + repr(sw_error) if sw_error else "succeeded"}')
      return False

    diffs = []
    if list(py) != list(files):
      diffs.append(f'parts: python {list(py)} / swift {list(files)}')
    identical = []
    sw_models = {}
    for name, part in py.items():
      if name not in files:
        continue
      py_weights = sum(len(t.raw_data) for t in part.graph.initializer)
      if weights.get(name) != py_weights:
        diffs.append(f'{name} weight bytes: python {py_weights} / swift {weights.get(name)}')
      raw = files[name].read_bytes()
      identical.append(f'{name} {"identical" if raw == part.SerializeToString() else "NOT byte-identical"}')
      sw_model = onnx.load_from_string(raw)
      del raw
      diffs += [f'{name}: {d}' for d in compare(part, sw_model)]
      sw_models[name] = sw_model

    t_ort = None
    if ort_run and not diffs:
      t0 = time.perf_counter()
      parts = list(py.values())
      feeds = random_feeds(parts)
      py_out = run_chain([p.SerializeToString() for p in parts], feeds)
      sw_out = run_chain([str(files[n]) for n in py], feeds)
      diffs += compare_runs(py_out, sw_out)
      t_ort = time.perf_counter() - t0
      ort_note = f', ort: {len(py_out)} outputs bit-equal in {t_ort:.1f} s' if not diffs else ''
    else:
      ort_note = ''

    status = 'PASS' if not diffs else 'FAIL'
    print(f'{status} {label}: {", ".join(identical)}; python {t_py:.1f} s, swift {t_sw:.1f} s{ort_note}')
    for d in diffs:
      print(f'    {d}')
    return not diffs
  finally:
    shutil.rmtree(out, ignore_errors=True)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('models', nargs='+', type=Path)
  layout = ap.add_mutually_exclusive_group()
  layout.add_argument('--whole', action='store_true', help='the whole layout only')
  layout.add_argument('--both', action='store_true', help='the split and the whole layouts')
  ap.add_argument('--ort', action='store_true', help='also run onnxruntime on large models')
  ap.add_argument('--cli', type=Path, default=DEFAULT_CLI, help=f'the jetlink-onnx binary (default {DEFAULT_CLI})')
  ap.add_argument('--key-prefix', default='check')
  ap.add_argument('--workdir', type=Path, default=None)
  args = ap.parse_args()

  if not args.cli.exists():
    print(f'no jetlink-onnx at {args.cli}; build it (swift build -c release --product jetlink-onnx) or pass --cli',
          file=sys.stderr)
    return 2
  workdir = args.workdir or Path(tempfile.mkdtemp(prefix='check-onnx-prep-'))
  workdir.mkdir(parents=True, exist_ok=True)
  layouts = (False, True) if args.both else ((True,) if args.whole else (False,))
  ok = True
  try:
    for src in args.models:
      size = src.stat().st_size
      free = shutil.disk_usage(workdir).free
      # the Swift side's files, about the model's size, plus room to spare
      if free < size + (2 << 30):
        print(f'SKIP {src.name}: {free / 2**30:.1f} GB free, need {(size + (2 << 30)) / 2**30:.1f}', file=sys.stderr)
        ok = False
        continue
      for whole in layouts:
        ok &= check(src, whole, args.cli, workdir, args.key_prefix, args.ort or size < ORT_ALWAYS_BELOW)
  finally:
    if args.workdir is None:
      shutil.rmtree(workdir, ignore_errors=True)
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
