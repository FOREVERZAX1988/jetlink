"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The whole graph on the Neural Engine (`--device ane-whole`): the two passes
after the Mac's CoreML preparation that make one program of it, checked on a
small graph built to take every branch, not just the ones one driving model
happens to take. A Swift port has to reproduce these bytes, so the counts,
names and order are pinned here.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from jetlink.onnx_patch import (  # noqa: E402
  HEAD_MAX_NODES,
  heads_in_fp32,
  prescale_layernorm,
  vision_heads,
)
from jetlink.server.backends.ort import ANE_WHOLE_LAYOUT, _prepared_model  # noqa: E402

REAL_MODEL = Path.home() / 'Library/Application Support/Jetlink/cache/models/09d080f36965bb2a.onnx'

rng = np.random.default_rng(3)


def f16(shape, scale=0.1):
  return (rng.standard_normal(shape) * scale).astype(np.float16)


def variants() -> onnx.ModelProto:
  """comma's layout with every branch of the preparation in it:
  Concat(img, big_img) -> Cast, a passthrough feeding the graph output,
  negative Gather indices (a shared scalar on axes of different sizes, an
  int32 vector), LayerNorms in the vision trunk and the policy (a shared
  input, no bias, one already fp32), MatMul+Add at rank 2, 3 and 4 and the
  MatMuls that must be left alone (no Add, a small weight, an Add that is not
  a bias), an Expand that becomes a Tile and one that must not, and heads off
  the vision trunk that go to fp32 (a MatMul+Add the Gemm rewrite transposes,
  a plain MatMul, a weight the policy reads too, a tensor read inside the
  heads and by the output Concat)."""
  T = TensorProto
  inits = {
    'vshape': np.array([1, 24, 32], np.int64),
    'vs': f16([32], 1), 'vb': f16([32]),
    'W1': f16([32, 64]), 'b1': f16([64]),
    's': f16([64], 1), 'b': f16([64]), 's32': (rng.standard_normal(64)).astype(np.float32),
    'idx': np.array(-1, np.int64),
    'idxv': np.array([0, -1, -5], np.int32),
    'W2': f16([64, 32]), 'b2': f16([32]),
    'W3': f16([64, 32]),
    'Ws': f16([64, 8]), 'bs': f16([8]),
    'W4': f16([64, 32]),
    'r4': np.array([1, 2, 15, 64], np.int64),
    'W5': f16([64, 16]), 'b5': f16([16]),
    'eshape': np.array([1, 1, 4, 1], np.int64),
    'uaxis': np.array([2], np.int64),
    'eshape_up': np.array([2, 1, 64], np.int64),
    'raxes': np.array([1], np.int64),
    'hs': f16([64], 1), 'hb': f16([64]),
    'hW1': f16([64, 128]), 'hb1': f16([128]),
    'hW2': f16([128, 64]),
  }
  nodes = [
    helper.make_node('Concat', ['img', 'big_img'], ['cat'], axis=1, name='cat'),
    helper.make_node('Cast', ['cat'], ['catf'], to=T.FLOAT16, name='cast'),
    # the vision trunk: reads the images only
    helper.make_node('Reshape', ['catf', 'vshape'], ['v'], name='vreshape'),
    helper.make_node('LayerNormalization', ['v', 'vs', 'vb'], ['vln'], axis=-1, epsilon=1e-5, name='vln'),
    helper.make_node('MatMul', ['vln', 'W1'], ['vmm'], name='vmm'),
    helper.make_node('Add', ['vmm', 'b1'], ['vout'], name='vadd'),
    # the policy: reads features too
    helper.make_node('Concat', ['vout', 'feat'], ['p'], axis=1, name='pcat'),
    helper.make_node('LayerNormalization', ['p', 's', 'b'], ['ln1'], axis=-1, epsilon=1e-5, name='ln1'),
    helper.make_node('LayerNormalization', ['p', 's', 'b'], ['ln2'], axis=-1, epsilon=1e-5, name='ln2'),
    helper.make_node('LayerNormalization', ['p', 's'], ['ln3'], axis=-1, epsilon=1e-5, name='ln3'),
    helper.make_node('Cast', ['p'], ['p32'], to=T.FLOAT, name='p32'),
    helper.make_node('LayerNormalization', ['p32', 's32'], ['ln4'], axis=-1, epsilon=1e-5, name='ln4'),
    helper.make_node('Cast', ['ln4'], ['ln4h'], to=T.FLOAT16, name='ln4h'),
    helper.make_node('Add', ['ln1', 'ln2'], ['q0'], name='q0'),
    helper.make_node('Add', ['q0', 'ln3'], ['q1'], name='q1'),
    helper.make_node('Add', ['q1', 'ln4h'], ['q'], name='q'),
    helper.make_node('Gather', ['q', 'idx'], ['g1'], axis=1, name='g1'),
    helper.make_node('Gather', ['q', 'idx'], ['g2'], axis=2, name='g2'),
    helper.make_node('Gather', ['q', 'idxv'], ['g3'], axis=1, name='g3'),
    helper.make_node('MatMul', ['g1', 'W2'], ['m2mm'], name='m2mm'),
    helper.make_node('Add', ['m2mm', 'b2'], ['m2'], name='m2'),
    helper.make_node('MatMul', ['g3', 'W3'], ['m3'], name='m3'),
    helper.make_node('MatMul', ['g1', 'Ws'], ['smm'], name='smm'),
    helper.make_node('Add', ['smm', 'bs'], ['small'], name='small'),
    helper.make_node('MatMul', ['g1', 'W4'], ['nb'], name='nbmm'),
    helper.make_node('Add', ['nb', 'm2'], ['notbias'], name='notbias'),
    helper.make_node('Reshape', ['q', 'r4'], ['q4'], name='q4'),
    helper.make_node('MatMul', ['q4', 'W5'], ['m5mm'], name='m5mm'),
    helper.make_node('Add', ['m5mm', 'b5'], ['m5'], name='m5'),
    # an Expand that only repeats a size-1 axis becomes a Tile...
    helper.make_node('Unsqueeze', ['g3', 'uaxis'], ['g3u'], name='g3u'),
    helper.make_node('Expand', ['g3u', 'eshape'], ['ex'], name='ex'),
    # ...one that broadcasts a lower-rank input stays an Expand
    helper.make_node('Expand', ['g1', 'eshape_up'], ['ex2'], name='ex2'),
    # heads off the vision trunk, read only by each other and the output
    helper.make_node('ReduceMean', ['vout', 'raxes'], ['vm'], keepdims=0, name='vm'),
    helper.make_node('LayerNormalization', ['vm', 'hs', 'hb'], ['hln'], axis=-1, epsilon=1e-5, name='hln'),
    helper.make_node('MatMul', ['hln', 'hW1'], ['h1mm'], name='h1mm'),
    helper.make_node('Add', ['h1mm', 'hb1'], ['h1'], name='h1'),
    helper.make_node('Gelu', ['h1'], ['hg'], approximate='tanh', name='hg'),
    helper.make_node('MatMul', ['hg', 'hW2'], ['h2'], name='h2'),
    helper.make_node('Add', ['h2', 'vm'], ['hres'], name='hres'),
    helper.make_node('Mul', ['hres', 's'], ['hsc'], name='hsc'),
  ]
  flat = []
  for name in ('m2', 'm3', 'small', 'notbias', 'm5', 'g2', 'ex', 'ex2'):
    # ex2 broadcast its batch to 2; flattened whole, it still concatenates.
    nodes.append(helper.make_node('Flatten', [name], [f'{name}_f'], axis=0 if name == 'ex2' else 1, name=f'{name}_flat'))
    flat.append(f'{name}_f')
  nodes.append(helper.make_node('Concat', flat + ['hres', 'hsc'], ['pre'], axis=1, name='pre'))
  # a layout hint feeding the graph output: the output has to keep its name
  nodes.append(helper.make_node('Contiguous', ['pre'], ['outputs'], domain='org.tinygrad', name='hint'))
  n_out = 32 + 3 * 32 + 8 + 32 + 2 * 15 * 16 + 30 + 3 * 4 * 64 + 2 * 64 + 2 * 64
  graph = helper.make_graph(
    nodes, 'variants',
    [helper.make_tensor_value_info('img', T.UINT8, [1, 12, 4, 8]),
     helper.make_tensor_value_info('big_img', T.UINT8, [1, 12, 4, 8]),
     helper.make_tensor_value_info('feat', T.FLOAT16, [1, 6, 64])],
    [helper.make_tensor_value_info('outputs', T.FLOAT16, [1, n_out])],
    initializer=[numpy_helper.from_array(v, k) for k, v in inits.items()])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20), helper.make_opsetid('org.tinygrad', 1)])
  model.ir_version = 10
  # Record every intermediate shape, as the driving models' exporter does;
  # the unknown-domain op at the end is simply left without one.
  inferred = onnx.shape_inference.infer_shapes(model)
  model.graph.value_info.extend(v for v in inferred.graph.value_info if v.name != 'outputs')
  return model


@pytest.fixture(scope='module')
def variants_path(tmp_path_factory) -> Path:
  path = tmp_path_factory.mktemp('ane') / 'variants.onnx'
  onnx.save(variants(), str(path))
  return path


@pytest.fixture
def coreml_prepared(variants_path):
  """The Mac's CoreML preparation, which the two passes run after."""
  return _prepared_model(variants_path, for_coreml=True)


def _names(model, op=None):
  return [n.name for n in model.graph.node if op is None or n.op_type == op]


def _init(model, name):
  return next(t for t in model.graph.initializer if t.name == name)


class TestPrescaleLayerNorm:
  def test_the_policy_norms_share_one_mul_per_input(self, coreml_prepared):
    """ln1, ln2 and ln3 read p, so one Mul feeds all three; ln4 is fp32 and
    the trunk's vln and the heads' hln are vision, all left alone."""
    m = coreml_prepared
    assert prescale_layernorm(m) == 3
    muls = [n for n in m.graph.node if n.op_type == 'Mul' and n.input[1] == '__layernorm_prescale_8']
    assert [(n.name, list(n.input), list(n.output)) for n in muls] == [('p__prescale', ['p', '__layernorm_prescale_8'], ['p__scaled'])]
    norms = {n.name: n.input[0] for n in m.graph.node if n.op_type == 'LayerNormalization'}
    assert norms == {'vln': 'v', 'ln1': 'p__scaled', 'ln2': 'p__scaled', 'ln3': 'p__scaled', 'ln4': 'p32', 'hln': 'vm'}
    # inserted directly before the first norm that reads p
    names = _names(m)
    assert names.index('p__prescale') == names.index('ln1') - 1

  def test_the_constant_is_fp16_one_eighth_appended_last(self, coreml_prepared):
    m = coreml_prepared
    prescale_layernorm(m)
    const = m.graph.initializer[-1]
    assert const.name == '__layernorm_prescale_8'
    assert const.data_type == TensorProto.FLOAT16
    assert list(const.dims) == []
    assert const.raw_data == np.float16(0.125).tobytes()
    assert numpy_helper.to_array(const) == np.float16(1 / 8)

  def test_epsilon_is_left_alone(self, coreml_prepared):
    m = coreml_prepared
    prescale_layernorm(m)
    for n in m.graph.node:
      if n.op_type == 'LayerNormalization':
        assert helper.get_attribute_value(next(a for a in n.attribute if a.name == 'epsilon')) == pytest.approx(1e-5)

  def test_running_it_twice_changes_nothing(self, coreml_prepared):
    m = coreml_prepared
    prescale_layernorm(m)
    once = m.SerializeToString()
    assert prescale_layernorm(m) == 0
    assert m.SerializeToString() == once

  def test_a_graph_without_policy_norms_is_untouched(self, tmp_path):
    from tests import tiny_model
    m = _prepared_model(tiny_model.write(tmp_path / 'tiny.onnx'), for_coreml=True)
    before = m.SerializeToString()
    assert prescale_layernorm(m) == 0
    assert m.SerializeToString() == before


class TestVisionHeads:
  def test_the_heads_are_the_six_nodes_between_the_reduce_and_the_concat(self, coreml_prepared):
    """After the Gemm rewrite the MatMul+Add is one Gemm; the plain MatMul
    stays; the ReduceMean that makes vm is not a head op and ends the region."""
    m = coreml_prepared
    names = _names(m)
    assert [names[i] for i in vision_heads(m)] == ['hln', 'h1mm__gemm', 'hg', 'h2', 'hres', 'hsc']

  def test_without_an_output_concat_there_are_none(self, coreml_prepared):
    m = coreml_prepared
    concat = next(n for n in m.graph.node if n.name == 'pre')
    concat.op_type = 'Sum'
    assert vision_heads(m) == []

  def test_a_region_over_the_cap_is_refused(self, coreml_prepared, monkeypatch):
    import jetlink.onnx_patch as patch
    monkeypatch.setattr(patch, 'HEAD_MAX_NODES', 5)
    assert vision_heads(coreml_prepared) == []
    assert HEAD_MAX_NODES == 64


class TestHeadsInFp32:
  def test_the_heads_move_between_casts(self, coreml_prepared):
    m = coreml_prepared
    prescale_layernorm(m)
    assert heads_in_fp32(m) == 6
    names = _names(m)
    # one Cast up before the first head reads vm, the heads, one Cast down per
    # exit right after the node that makes it
    i = names.index('vm')
    assert names[i + 1:i + 10] == ['vm__cast_fp32', 'hln', 'h1mm__gemm', 'hg', 'h2', 'hres',
                                    'hres__cast_fp16', 'hsc', 'hsc__cast_fp16']
    nodes = {n.name: n for n in m.graph.node}
    assert list(nodes['vm__cast_fp32'].input) == ['vm'] and list(nodes['vm__cast_fp32'].output) == ['vm__fp32']
    assert helper.get_attribute_value(nodes['vm__cast_fp32'].attribute[0]) == TensorProto.FLOAT
    assert list(nodes['hln'].input) == ['vm__fp32', 'hs__fp32', 'hb__fp32']
    assert list(nodes['h1mm__gemm'].input) == ['hln', 'h1mm__wt__fp32', 'hb1__fp32']
    assert list(nodes['h2'].input) == ['hg', 'hW2__fp32']
    assert list(nodes['hres'].input) == ['h2', 'vm__fp32'] and list(nodes['hres'].output) == ['hres__fp32']
    assert list(nodes['hsc'].input) == ['hres__fp32', 's__fp32'] and list(nodes['hsc'].output) == ['hsc__fp32']
    assert list(nodes['hres__cast_fp16'].input) == ['hres__fp32'] and list(nodes['hres__cast_fp16'].output) == ['hres']
    assert helper.get_attribute_value(nodes['hres__cast_fp16'].attribute[0]) == TensorProto.FLOAT16
    # the output Concat still reads the fp16 names
    assert list(nodes['pre'].input)[-2:] == ['hres', 'hsc']

  def test_weights_widen_in_name_order_and_the_shared_one_keeps_its_fp16(self, coreml_prepared):
    """The copies are appended in the order of the fp16 names (code points,
    so h1mm__wt before hW2), and the fp16 originals nothing else reads go."""
    m = coreml_prepared
    before = [t.name for t in m.graph.initializer]
    original = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    heads_in_fp32(m)
    after = [t.name for t in m.graph.initializer]
    wide = [n for n in after if n.endswith('__fp32')]
    assert wide == ['h1mm__wt__fp32', 'hW2__fp32', 'hb__fp32', 'hb1__fp32', 'hs__fp32', 's__fp32']
    assert after == [n for n in before if n not in ('hW2', 'h1mm__wt', 'hb1', 'hb', 'hs')] + wide
    # s is read by the policy norms too, so its fp16 stays beside the copy
    assert 's' in after
    for name in wide:
      t = _init(m, name)
      assert t.data_type == TensorProto.FLOAT
      np.testing.assert_array_equal(numpy_helper.to_array(t), original[name.removesuffix('__fp32')].astype(np.float32))

  def test_inside_value_infos_go_and_the_exits_stay(self, coreml_prepared):
    m = coreml_prepared
    heads_in_fp32(m)
    vi = {v.name for v in m.graph.value_info}
    assert not vi & {'hln', 'hg', 'h2'}
    assert {'vm', 'hres', 'hsc'} <= vi

  def test_running_it_twice_changes_nothing_and_says_so(self, coreml_prepared, caplog):
    m = coreml_prepared
    heads_in_fp32(m)
    once = m.SerializeToString()
    with caplog.at_level(logging.WARNING, logger='jetlink.onnx_patch'):
      assert heads_in_fp32(m) == 0
    assert 'no heads found' in caplog.text
    assert m.SerializeToString() == once

  def test_a_graph_without_heads_warns(self, tmp_path, caplog):
    from tests import tiny_model
    m = _prepared_model(tiny_model.write(tmp_path / 'tiny.onnx'), for_coreml=True)
    before = m.SerializeToString()
    with caplog.at_level(logging.WARNING, logger='jetlink.onnx_patch'):
      assert heads_in_fp32(m) == 0
    assert 'no heads found' in caplog.text
    assert m.SerializeToString() == before

  def test_an_entry_that_is_not_fp16_refuses(self, coreml_prepared, caplog):
    m = coreml_prepared
    vi = next(v for v in m.graph.value_info if v.name == 'vm')
    vi.type.tensor_type.elem_type = TensorProto.FLOAT
    with caplog.at_level(logging.WARNING, logger='jetlink.onnx_patch'):
      assert heads_in_fp32(m) == 0
    assert 'not fp16' in caplog.text


class TestTheLayout:
  def test_the_layout_runs_both_passes_after_the_coreml_ones(self, variants_path):
    m = _prepared_model(variants_path, for_coreml=True, layout=ANE_WHOLE_LAYOUT)
    names = _names(m)
    assert 'p__prescale' in names and 'vm__cast_fp32' in names
    assert 'hint' not in names                                      # stripped
    assert [n.op_type for n in m.graph.node if n.name == 'ex'] == ['Tile']
    assert [n.op_type for n in m.graph.node if n.name == 'ex2'] == ['Expand']
    assert 'h1mm__gemm' in names
    onnx.checker.check_model(m, full_check=True)
    # and the same bytes as the passes run by hand, in that order
    by_hand = _prepared_model(variants_path, for_coreml=True)
    prescale_layernorm(by_hand)
    heads_in_fp32(by_hand)
    assert m.SerializeToString() == by_hand.SerializeToString()

  def test_the_layout_is_a_coreml_preparation(self, variants_path):
    with pytest.raises(ValueError, match='CoreML'):
      _prepared_model(variants_path, for_coreml=False, layout=ANE_WHOLE_LAYOUT)
    with pytest.raises(ValueError, match='layout'):
      _prepared_model(variants_path, for_coreml=True, layout='phone')


@pytest.mark.skipif(importlib.util.find_spec('onnxruntime') is None, reason='needs onnxruntime')
def test_the_whole_graph_computes_what_the_plain_one_does(variants_path, tmp_path):
  """On the CPU provider, a build that retypes a tensor and misses a reader
  is a model onnxruntime refuses, and one that scales the wrong thing is off
  by whole units. onnxruntime runs in the backend's worker, never here (see
  backends.ort.quiet)."""
  from jetlink.server.backends.base import infer
  from jetlink.server.backends.ort import MANIFEST, OrtBackend
  providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

  def staged(device):
    artifact = tmp_path / f'{device}.ortcache'
    artifact.mkdir()
    manifest = OrtBackend(device, providers=providers)._stage(variants_path, artifact, artifact)
    for entry in manifest:
      if entry['cache']:
        (artifact / entry['cache']).rmdir()
      entry['units'] = entry['cache'] = None
    (artifact / MANIFEST).write_text(json.dumps(manifest))
    return artifact

  backend = OrtBackend('cpu')
  plain = backend.load(staged('cpu'))
  whole = backend.load(staged('ane-whole'))
  try:
    assert set(whole.inputs) == set(plain.inputs) == {'img', 'big_img', 'feat'}
    assert set(whole.outputs) == set(plain.outputs) == {'outputs'}
    for seed in range(3):
      r = np.random.default_rng(seed)
      feed = {'img': r.integers(0, 256, [1, 12, 4, 8], dtype=np.uint8),
              'big_img': r.integers(0, 256, [1, 12, 4, 8], dtype=np.uint8),
              'feat': r.standard_normal([1, 6, 64]).astype(np.float16)}
      want = np.asarray(infer(plain, feed)['outputs'], np.float32)
      got = np.asarray(infer(whole, feed)['outputs'], np.float32)
      scale = float(np.abs(want).max())
      assert float(np.abs(got - want).max()) <= 0.02 * max(1.0, scale)
  finally:
    whole.close()
    plain.close()


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason=f'needs the cached driving model at {REAL_MODEL}')
def test_the_driving_model_counts(caplog):
  """The 766 MB chestnut model the passes were measured on: 41 policy norms
  prescaled and 24 head nodes in fp32 (iPhone 17 Pro, 2026-09-25). A port
  producing other numbers has changed what runs where."""
  m = _prepared_model(REAL_MODEL, for_coreml=True)
  assert prescale_layernorm(m) == 41
  with caplog.at_level(logging.WARNING, logger='jetlink.onnx_patch'):
    assert heads_in_fp32(m) == 24
  assert not caplog.text
  assert sum(1 for t in m.graph.initializer if t.name == '__layernorm_prescale_8') == 1
  onnx.checker.check_model(m, full_check=False)
