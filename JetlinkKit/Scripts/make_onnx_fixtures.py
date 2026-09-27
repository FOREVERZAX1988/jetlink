"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Writes the small ONNX graphs JetlinkONNX's tests read, and what the Python
preparation makes of them, into Tests/JetlinkONNXTests/Fixtures/.

    PYTHONPATH=. ../jetlink/.venv/bin/python JetlinkKit/Scripts/make_onnx_fixtures.py

from the jetlink checkout whose preparation the fixtures should pin (the
script imports jetlink.onnx_patch, the ORT backend and tests.test_ane_whole).

The graphs are shaped like the driving models, shrunk: the same input names,
uint8 images behind the head Cast, a tinygrad Contiguous (with the local
function the real exports carry for it), a vision trunk that narrows to one
tensor before its heads, MatMul+Add in two and three dimensions, a negative
Gather index, a repeat-only Expand, value_info for every tensor, metadata_props
on nodes, value infos and the graph, and fields the preparation does not know.
The arithmetic is float32 except one fp16 head, so onnxruntime's CPU provider
runs all of them.

variants.onnx is tests/test_ane_whole.py's graph, built to take every branch
of the ane-whole passes (a shared LayerNorm input, a norm without a bias, an
fp32 norm, an Expand that must stay, a head weight the policy reads too).

noshape.onnx and unrecorded.onnx leave a shape out of value_info, where
Python's preparation asks onnx's shape inferrer and Swift, which has none,
refuses the split or skips the rewrite. notype.onnx and noentry.onnx leave a
type out (a policy LayerNorm's input, a head's entry), where Python infers it
and Swift refuses the ane-whole layout. Every other graph must come out the
same from both.

Next to each graph go the files Python's preparation writes for it in each
layout, split, whole and ane-whole (<name>.<layout>.<part>.expected.onnx, cache
keys from the prefix "fixture") and python.json: the counts, the weight bytes,
or the error Python raised.
The Swift tests hold the Swift preparation to those, byte for byte.
"""
from __future__ import annotations

import codecs
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

FIXTURES = Path(__file__).resolve().parents[1] / 'Tests' / 'JetlinkONNXTests' / 'Fixtures'
KEY_PREFIX = 'fixture'
F, H, U8, I64 = TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.UINT8, TensorProto.INT64
TINYGRAD = 'org.tinygrad'


def rng(seed):
  return np.random.default_rng(seed)


def w(r, shape, dtype=np.float32, scale=0.1):
  return (r.standard_normal(shape) * scale).astype(dtype)


def const(name, values, dtype=np.int64):
  return numpy_helper.from_array(np.array(values, dtype), name)


def contiguous_function():
  """The local function the real exports define for tinygrad's Contiguous."""
  return helper.make_function(TINYGRAD, 'Contiguous', ['input'], ['output'],
                              [helper.make_node('Identity', ['input'], ['output'])],
                              [helper.make_opsetid('', 17)])


def finish(nodes, name, inputs, outputs, inits, *, slices=None, functions=(), extras=True,
           drop_value_info=()) -> onnx.ModelProto:
  """A model as an exporter writes it: every tensor's shape in value_info."""
  graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
  opsets = [helper.make_opsetid('', 17)]
  if any(n.domain == TINYGRAD for n in nodes):
    opsets.append(helper.make_opsetid(TINYGRAD, 1))
  model = helper.make_model(graph, opset_imports=opsets, functions=list(functions))
  model.ir_version = 8
  model = shape_inference.infer_shapes(model)
  g = model.graph
  known = {vi.name for vi in (*g.input, *g.value_info, *g.output)}
  shapes = {vi.name: vi for vi in (*g.input, *g.value_info, *g.output)}
  for node in g.node:
    # onnx's inferrer does not see through tinygrad's op; its output is its input's shape.
    if node.domain == TINYGRAD and node.output[0] not in known:
      vi = onnx.ValueInfoProto()
      vi.CopyFrom(shapes[node.input[0]])
      vi.name = node.output[0]
      g.value_info.append(vi)
      known.add(vi.name)
  missing = [o for n in g.node for o in n.output if o not in known]
  assert not missing, f'{name}: no shape for {missing}'
  for n in drop_value_info:
    [vi] = [vi for vi in g.value_info if vi.name == n]
    g.value_info.remove(vi)

  if slices:
    model.metadata_props.add(key='output_slices', value=codecs.encode(pickle.dumps(slices), 'base64').decode())
  model.metadata_props.add(key='model_checkpoint', value=f'{name}-test')
  if extras:
    # What the real exports carry besides the graph, and what nobody knows.
    model.producer_name, model.producer_version, model.doc_string = 'pytorch', '2.13.0', 'a test graph'
    g.metadata_props.add(key='pkg.torch.export.ExportedProgram.range_constraints', value='{}')
    g.node[0].metadata_props.add(key='namespace', value=': test.Model')
    g.value_info[0].metadata_props.add(key='pkg.torch.onnx.original_node_name', value='p_test')
    g.doc_string = 'the graph'
    model.metadata_props.add(key='CACHE_KEY', value='stale')
    # An unknown field in a node, in the graph and in the model: Python keeps
    # them and writes them after the fields it knows.
    node = onnx.NodeProto()
    node.ParseFromString(g.node[1].SerializeToString() + bytes([0xf8, 0x06, 0x2a]))   # field 111, varint 42
    g.node[1].CopyFrom(node)
    graph = onnx.GraphProto()
    graph.ParseFromString(g.SerializeToString() + bytes([0xa2, 0x06, 0x03]) + b'abc')  # field 100, bytes
    model.graph.CopyFrom(graph)
    raw = model.SerializeToString() + bytes([0x98, 0x06, 0x07])                         # field 99, varint 7
    model = onnx.ModelProto()
    model.ParseFromString(raw)
  onnx.checker.check_model(model)
  return model


# -- the graphs ---------------------------------------------------------------

def queued() -> onnx.ModelProto:
  """V2's layout: img and big_img concatenated as uint8, then one head Cast."""
  r = rng(1)
  inputs = [helper.make_tensor_value_info('img', U8, [1, 12, 8, 16]),
            helper.make_tensor_value_info('big_img', U8, [1, 12, 8, 16]),
            helper.make_tensor_value_info('desire_pulse', F, [1, 4, 8]),
            helper.make_tensor_value_info('traffic_convention', F, [1, 2]),
            helper.make_tensor_value_info('features_buffer', F, [1, 4, 16])]
  pol_w = w(r, (32 + 64 + 32 + 2 + 64, 48), scale=0.05)
  inits = [
    numpy_helper.from_array(w(r, (8, 24, 3, 3), scale=0.02), 'conv_w'),
    numpy_helper.from_array(w(r, (8,)), 'conv_b'),
    const('trunk_shape', [1, 8, 32]),
    numpy_helper.from_array(w(r, (32, 64), np.float16), 'head_w'),
    numpy_helper.from_array(w(r, (64,), np.float16), 'head_b'),
    const('last', -1),
    # float_data rather than raw_data, as an older exporter writes it
    helper.make_tensor('pol_w', F, pol_w.shape, pol_w.reshape(-1).tolist()),
    numpy_helper.from_array(w(r, (48,)), 'pol_b'),
    const('pol4_shape', [1, 6, 1, 8]),
    const('expand_shape', [1, 6, 4, 8]),
    numpy_helper.from_array(w(r, (48, 8)), 'small_w'),    # 384 elements: stays a MatMul
    numpy_helper.from_array(w(r, (8,)), 'small_b'),
  ]
  nodes = [
    helper.make_node('Concat', ['img', 'big_img'], ['cat'], name='cat', axis=1),
    helper.make_node('Cast', ['cat'], ['cat_h'], name='head_cast', to=H),
    helper.make_node('Cast', ['cat_h'], ['cat_f'], name='to_f32', to=F),
    helper.make_node('Conv', ['cat_f', 'conv_w', 'conv_b'], ['conv'], name='conv', pads=[1, 1, 1, 1], strides=[2, 2]),
    helper.make_node('Relu', ['conv'], ['relu'], name='relu'),
    helper.make_node('Reshape', ['relu', 'trunk_shape'], ['trunk'], name='trunk'),
    helper.make_node('Contiguous', ['trunk'], ['trunk_c'], name='contiguous', domain=TINYGRAD),
    helper.make_node('Cast', ['trunk_c'], ['trunk_h'], name='to_f16', to=H),
    helper.make_node('MatMul', ['trunk_h', 'head_w'], ['head_mm'], name='head_mm'),
    helper.make_node('Add', ['head_mm', 'head_b'], ['head_a'], name='head_add'),
    helper.make_node('Gather', ['head_a', 'last'], ['head_last_h'], name='select_last', axis=1),
    helper.make_node('Cast', ['head_last_h'], ['head_last'], name='head_to_f32', to=F),
    helper.make_node('ReduceMean', ['trunk_c'], ['head_mean'], name='head_mean', axes=[1], keepdims=0),
    helper.make_node('Flatten', ['desire_pulse'], ['desire_flat'], name='desire_flat', axis=1),
    helper.make_node('Flatten', ['features_buffer'], ['feat_flat'], name='feat_flat', axis=1),
    helper.make_node('Concat', ['head_mean', 'head_last', 'desire_flat', 'traffic_convention', 'feat_flat'],
                     ['policy_in'], name='policy_in', axis=1),
    helper.make_node('MatMul', ['policy_in', 'pol_w'], ['pol_mm'], name='pol_mm'),
    helper.make_node('Add', ['pol_b', 'pol_mm'], ['pol'], name='pol_add'),      # the bias first
    helper.make_node('Reshape', ['pol', 'pol4_shape'], ['pol4'], name='pol4'),
    helper.make_node('Expand', ['pol4', 'expand_shape'], ['pol_exp'], name='expand'),
    helper.make_node('ReduceMean', ['pol_exp'], ['pol_red'], name='pol_red', axes=[2], keepdims=0),
    helper.make_node('Flatten', ['pol_red'], ['pol_flat'], name='pol_flat', axis=1),
    helper.make_node('MatMul', ['pol_flat', 'small_w'], ['small_mm'], name='small_mm'),
    helper.make_node('Add', ['small_mm', 'small_b'], ['small'], name='small_add'),
    helper.make_node('Concat', ['pol', 'pol_flat', 'small'], ['outputs'], name='outputs', axis=1),
  ]
  outputs = [helper.make_tensor_value_info('outputs', F, [1, 104])]
  slices = {'plan': slice(0, 48), 'lead': slice(48, 96), 'hidden_state': slice(96, 104)}
  return finish(nodes, 'queued', inputs, outputs, inits, slices=slices, functions=[contiguous_function()])


def stateful() -> onnx.ModelProto:
  """V3's layout: a uint8 frame queue the graph hands back, frames picked by
  Gather, a desire queue and a feature queue."""
  r = rng(2)
  inputs = [helper.make_tensor_value_info('new_img', U8, [2, 6, 8, 16]),
            helper.make_tensor_value_info('desire', F, [8]),
            helper.make_tensor_value_info('traffic_convention', F, [1, 2]),
            helper.make_tensor_value_info('action_t', F, [1, 2]),
            helper.make_tensor_value_info('state_img_q', U8, [2, 5, 6, 8, 16]),
            helper.make_tensor_value_info('state_desire_q', F, [6, 1, 8]),
            helper.make_tensor_value_info('state_feat_q', F, [4, 1, 16])]
  big = 2 ** 62
  inits = [
    const('i0', 0), const('ax0', [0]), const('ax1', [1]), const('one', [1]), const('zero', [0]),
    const('end', [big]), const('four', [4]), const('h0', [32]), const('h1', [48]),
    # int64_data rather than raw_data
    helper.make_tensor('wide_index', I64, [], [-1]),
    const('cam_shape', [1, 12, 8, 16]), const('desire_row', [1, 1, 8]), const('desire_flat', [1, 48]),
    const('feat_flat', [1, 64]), const('hidden_row', [1, 1, 16]),
    numpy_helper.from_array(np.array(1 / 255, np.float32), 'scale'),
    numpy_helper.from_array(w(r, (8, 24, 3, 3), scale=0.05), 'conv_w'),
    numpy_helper.from_array(w(r, (8,)), 'conv_b'),
    const('trunk_shape', [1, 8, 32]),
    numpy_helper.from_array(w(r, (32, 48)), 'head_w'),
    numpy_helper.from_array(w(r, (48,)), 'head_b'),
    const('last', [-1]),
    numpy_helper.from_array(w(r, (48 + 32 + 48 + 2 + 2 + 64, 64), scale=0.05), 'W'),
    numpy_helper.from_array(w(r, (64,)), 'B'),
    const('p4_shape', [1, 4, 1, 16]), const('p_expand', [1, 4, 2, 16]),
  ]
  nodes = [
    helper.make_node('Unsqueeze', ['new_img', 'ax1'], ['unsqueeze'], name='unsqueeze'),
    helper.make_node('Slice', ['state_img_q', 'one', 'end', 'ax1'], ['img_tail'], name='img_tail'),
    helper.make_node('Concat', ['img_tail', 'unsqueeze'], ['next_state_img_q'], name='img_q', axis=1),
    helper.make_node('Gather', ['next_state_img_q', 'i0'], ['road'], name='select', axis=0),
    helper.make_node('Gather', ['next_state_img_q', 'wide_index'], ['wide'], name='select_1', axis=0),
    helper.make_node('Slice', ['road', 'zero', 'end', 'ax0', 'four'], ['road_pair'], name='road_pair'),
    helper.make_node('Slice', ['wide', 'zero', 'end', 'ax0', 'four'], ['wide_pair'], name='wide_pair'),
    helper.make_node('Reshape', ['road_pair', 'cam_shape'], ['road_img'], name='road_img'),
    helper.make_node('Reshape', ['wide_pair', 'cam_shape'], ['wide_img'], name='wide_img'),
    helper.make_node('Concat', ['road_img', 'wide_img'], ['imgs'], name='imgs', axis=1),
    helper.make_node('Cast', ['imgs'], ['imgs_f16'], name='head_cast', to=H),
    helper.make_node('Cast', ['imgs_f16'], ['imgs_f32'], name='to_f32', to=F),
    helper.make_node('Mul', ['imgs_f32', 'scale'], ['imgs_n'], name='normalize'),
    helper.make_node('Conv', ['imgs_n', 'conv_w', 'conv_b'], ['conv'], name='conv', pads=[1, 1, 1, 1], strides=[2, 2]),
    helper.make_node('Relu', ['conv'], ['relu'], name='relu'),
    helper.make_node('Reshape', ['relu', 'trunk_shape'], ['trunk'], name='trunk'),
    helper.make_node('Contiguous', ['trunk'], ['trunk_c'], name='contiguous', domain=TINYGRAD),
    helper.make_node('MatMul', ['trunk_c', 'head_w'], ['head_mm'], name='head_mm'),
    helper.make_node('Add', ['head_mm', 'head_b'], ['head_a'], name='head_add'),
    helper.make_node('Gather', ['head_a', 'last'], ['head_last3'], name='select_last', axis=1),
    helper.make_node('Flatten', ['head_last3'], ['head_last'], name='head_last', axis=1),
    helper.make_node('ReduceMean', ['trunk_c'], ['head_mean'], name='head_mean', axes=[1], keepdims=0),
    helper.make_node('Reshape', ['desire', 'desire_row'], ['desire_new'], name='desire_new'),
    helper.make_node('Slice', ['state_desire_q', 'one', 'end', 'ax0'], ['desire_tail'], name='desire_tail'),
    helper.make_node('Concat', ['desire_tail', 'desire_new'], ['next_state_desire_q'], name='desire_q', axis=0),
    helper.make_node('Reshape', ['next_state_desire_q', 'desire_flat'], ['desire_in'], name='desire_in'),
    helper.make_node('Reshape', ['state_feat_q', 'feat_flat'], ['feat_in'], name='feat_in'),
    helper.make_node('Concat', ['head_last', 'head_mean', 'desire_in', 'traffic_convention', 'action_t', 'feat_in'],
                     ['features'], name='features', axis=1),
    helper.make_node('MatMul', ['features', 'W'], ['mm'], name='mm'),
    helper.make_node('Add', ['mm', 'B'], ['policy_out'], name='policy_add'),
    helper.make_node('Reshape', ['policy_out', 'p4_shape'], ['p4'], name='p4'),
    helper.make_node('Expand', ['p4', 'p_expand'], ['p_exp'], name='expand'),
    helper.make_node('ReduceMax', ['p_exp'], ['p_red'], name='p_red', axes=[2], keepdims=0),
    helper.make_node('Flatten', ['p_red'], ['p_flat'], name='p_flat', axis=1),
    helper.make_node('Add', ['policy_out', 'p_flat'], ['outputs'], name='outputs'),
    helper.make_node('Slice', ['outputs', 'h0', 'h1', 'ax1'], ['hidden'], name='hidden'),
    helper.make_node('Reshape', ['hidden', 'hidden_row'], ['hidden_new'], name='hidden_new'),
    helper.make_node('Slice', ['state_feat_q', 'one', 'end', 'ax0'], ['feat_tail'], name='feat_tail'),
    helper.make_node('Concat', ['feat_tail', 'hidden_new'], ['next_state_feat_q'], name='feat_q', axis=0),
  ]
  outputs = [helper.make_tensor_value_info('outputs', F, [1, 64]),
             helper.make_tensor_value_info('next_state_img_q', U8, [2, 5, 6, 8, 16]),
             helper.make_tensor_value_info('next_state_desire_q', F, [6, 1, 8]),
             helper.make_tensor_value_info('next_state_feat_q', F, [4, 1, 16])]
  slices = {'plan': slice(0, 16), 'lead_prob': slice(16, 19), 'hidden_state': slice(32, 48), 'pad': slice(48, 64)}
  return finish(nodes, 'stateful', inputs, outputs, inits, slices=slices, functions=[contiguous_function()])


def nocut() -> onnx.ModelProto:
  """sunnypilot's layout, a Cast per image, and two image branches that never
  meet: the trunk never narrows to one tensor, so the cut is every tensor the
  policy reads. One vision-only graph output."""
  inputs = [helper.make_tensor_value_info('img', U8, [1, 6, 8, 8]),
            helper.make_tensor_value_info('big_img', U8, [1, 6, 8, 8]),
            helper.make_tensor_value_info('desire', F, [1, 8])]
  inits = [numpy_helper.from_array(np.array(2.0, np.float32), 'two')]
  nodes = [
    helper.make_node('Cast', ['img'], ['img_h'], name='img_cast', to=H),
    helper.make_node('Cast', ['img_h'], ['img_f'], name='img_f32', to=F),
    helper.make_node('Cast', ['big_img'], ['big_h'], name='big_cast', to=H),
    helper.make_node('Cast', ['big_h'], ['big_f'], name='big_f32', to=F),
    helper.make_node('ReduceMean', ['img_f'], ['a'], name='a', axes=[2, 3], keepdims=0),
    helper.make_node('ReduceMean', ['big_f'], ['b'], name='b', axes=[2, 3], keepdims=0),
    helper.make_node('Mul', ['a', 'two'], ['vision_out'], name='vision_out'),
    helper.make_node('Concat', ['a', 'desire'], ['pa'], name='pa', axis=1),
    helper.make_node('Concat', ['pa', 'b'], ['outputs'], name='outputs', axis=1),
  ]
  outputs = [helper.make_tensor_value_info('outputs', F, [1, 20]),
             helper.make_tensor_value_info('vision_out', F, [1, 6])]
  return finish(nodes, 'nocut', inputs, outputs, inits, extras=False)


def noshape() -> onnx.ModelProto:
  """queued without the trunk's recorded shape. Python asks onnx's shape
  inferrer; Swift has none and refuses the split."""
  return without_value_info(queued(), 'trunk')


def variants() -> onnx.ModelProto:
  """tests/test_ane_whole.py's graph, the one the Python tests pin the
  ane-whole passes on: comma's layout with every branch of the preparation
  in it (see its docstring). Imported, not copied, so the bytes the Swift
  port is held to are the ones those tests check."""
  from tests.test_ane_whole import variants as build
  return build()


def notype() -> onnx.ModelProto:
  """variants without the recorded type of p, the input three policy
  LayerNorms share. Python's prescale_layernorm asks the inferrer and scales
  them; Swift refuses the ane-whole layout. The other layouts never look."""
  return without_value_info(variants(), 'p')


def noentry() -> onnx.ModelProto:
  """variants without the recorded type of vm, the tensor the heads read
  from the trunk. Python's heads_in_fp32 asks the inferrer; Swift refuses the
  ane-whole layout."""
  return without_value_info(variants(), 'vm')


def without_value_info(model: onnx.ModelProto, name: str) -> onnx.ModelProto:
  [vi] = [vi for vi in model.graph.value_info if vi.name == name]
  model.graph.value_info.remove(vi)
  return model


# -- graphs the preparation refuses -------------------------------------------

def small(nodes, inputs, outputs, inits=(), name='bad', infer=True) -> onnx.ModelProto:
  """A few nodes, with the shapes onnx's inferrer finds recorded as value_info,
  as an export records them."""
  graph = helper.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
  opsets = [helper.make_opsetid('', 17)]
  if any(n.domain == TINYGRAD for n in nodes):
    opsets.append(helper.make_opsetid(TINYGRAD, 1))
  model = helper.make_model(graph, opset_imports=opsets)
  model.ir_version = 8
  return shape_inference.infer_shapes(model) if infer else model


def unrecorded() -> onnx.ModelProto:
  """A negative Gather index on a tensor whose shape the file does not record.
  Python's _static_dims asks onnx's shape inferrer and rewrites it; Swift has
  no inferrer and leaves the Gather alone. The one place the two differ by
  design, and why the real exports' recorded shapes matter."""
  return small([helper.make_node('Cast', ['img'], ['img_h'], name='head_cast', to=H),
                helper.make_node('Cast', ['img_h'], ['img_f'], name='to_f32', to=F),
                helper.make_node('Gather', ['img_f', 'last'], ['g'], name='select', axis=1),
                helper.make_node('Unsqueeze', ['g', 'ax'], ['outputs'], name='u')],
               [IMG], [out(shape=(1, 1))], inits=[const('last', -1), const('ax', [1])], name='unrecorded',
               infer=False)


IMG = helper.make_tensor_value_info('img', U8, [1, 4])
DESIRE = helper.make_tensor_value_info('desire', F, [1, 4])


def out(name='outputs', t=F, shape=(1, 4)):
  return helper.make_tensor_value_info(name, t, list(shape))


def errors() -> dict[str, onnx.ModelProto]:
  cast = helper.make_node('Cast', ['img'], ['img_h'], name='head_cast', to=H)
  to_f = helper.make_node('Cast', ['img_h'], ['img_f'], name='to_f32', to=F)
  return {
    'err_tinygrad_op': small(
      [cast, to_f, helper.make_node('Frobnicate', ['img_f'], ['outputs'], name='f', domain=TINYGRAD)],
      [IMG], [out()]),
    'err_tinygrad_arity': small(
      [cast, to_f, helper.make_node('Contiguous', ['img_f', 'img_f'], ['outputs'], name='c', domain=TINYGRAD)],
      [IMG], [out()]),
    'err_cast_fp32': small(
      [helper.make_node('Cast', ['img'], ['img_f'], name='head_cast', to=F),
       helper.make_node('Relu', ['img_f'], ['outputs'], name='relu')],
      [IMG], [out()]),
    'err_cast_output': small([cast], [IMG], [out('img_h', H)]),
    'err_uint8_arith': small(
      [helper.make_node('Add', ['img', 'img'], ['doubled'], name="it's add"),
       helper.make_node('Cast', ['doubled'], ['outputs'], name='cast', to=H)],
      [IMG], [out(t=H)]),
    'err_no_cast': small([helper.make_node('Identity', ['img'], ['outputs'], name='id')], [IMG], [out(t=U8)]),
    # Once the head Cast goes, the Add reads the image input itself.
    'err_policy_reads_images': small(
      [cast, helper.make_node('Add', ['img_h', 'desire_h'], ['outputs'], name='mix')],
      [IMG, helper.make_tensor_value_info('desire_h', H, [1, 4])], [out(t=H)]),
    'err_no_images': small([helper.make_node('Relu', ['desire'], ['outputs'], name='relu')], [DESIRE], [out()]),
    'err_nothing_crosses': small(
      [cast, to_f, helper.make_node('Relu', ['img_f'], ['vision_out'], name='relu'),
       helper.make_node('Relu', ['desire'], ['outputs'], name='relu_2')],
      [IMG, DESIRE], [out(), out('vision_out')]),
    'err_gather_range': small(
      [cast, to_f, helper.make_node('Gather', ['img_f', 'bad_index'], ['g'], name='select', axis=1),
       helper.make_node('Unsqueeze', ['g', 'ax'], ['outputs'], name='u')],
      [IMG], [out(shape=(1, 1))],
      inits=[const('bad_index', -9), const('ax', [1])]),
  }


# -- what Python makes of them ------------------------------------------------

LAYOUTS = ('split', 'whole', 'ane-whole')


def python_prepare(path: Path, layout: str):
  """The Python preparation as the ORT backend's _stage runs it, with the
  counts _prepared_model logs."""
  from jetlink import onnx_patch as p
  from jetlink.server.backends.ort import ANE_WHOLE_LAYOUT, _prepared_model, _with_cache_key

  model = onnx.load(str(path))
  counts = {'stripped': p.strip_tinygrad_ops(model)}
  counts['retypedImages'] = p.needs_patch(model)
  if counts['retypedImages']:
    p.patch_uint8_inputs(model)
  counts['gathers'] = p.normalize_gather_indices(model)
  counts['gemms'] = p.gemm_with_transposed_weight(model)
  counts['tiles'] = p.expand_to_tile(model)
  if layout == ANE_WHOLE_LAYOUT:
    counts['norms'] = p.prescale_layernorm(model)
    counts['heads'] = p.heads_in_fp32(model)

  prepared = _prepared_model(path, for_coreml=True, layout=ANE_WHOLE_LAYOUT if layout == ANE_WHOLE_LAYOUT else None)
  assert prepared.SerializeToString() == model.SerializeToString()
  names = ('vision', 'policy') if layout == 'split' else ('model',)
  parts = p.split_vision_policy(prepared) if layout == 'split' else (prepared,)
  out = {}
  for name, part in zip(names, parts, strict=True):
    key = re.sub(r'[^A-Za-z0-9]', '', KEY_PREFIX + name)[:63]
    out[name] = _with_cache_key(part, key)
  return counts, out


def main() -> None:
  FIXTURES.mkdir(parents=True, exist_ok=True)
  for old in FIXTURES.glob('*.onnx'):
    old.unlink()
  models = {'queued': queued(), 'stateful': stateful(), 'nocut': nocut(), 'noshape': noshape(),
            'unrecorded': unrecorded(), 'variants': variants(), 'notype': notype(), 'noentry': noentry(),
            **errors()}
  results = {}
  for name, model in models.items():
    path = FIXTURES / f'{name}.onnx'
    onnx.save(model, str(path))
    for layout in LAYOUTS:
      try:
        counts, parts = python_prepare(path, layout)
      except Exception as e:
        results[f'{name}.{layout}'] = {'error': str(e)}
        continue
      entry = dict(counts)
      entry['parts'] = {}
      for part, m in parts.items():
        onnx.save(m, str(FIXTURES / f'{name}.{layout}.{part}.expected.onnx'))
        entry['parts'][part] = sum(len(t.raw_data) for t in m.graph.initializer)
      results[f'{name}.{layout}'] = entry
  (FIXTURES / 'python.json').write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
  for k, v in results.items():
    print(f'{k:36} {v.get("error") or v}')


if __name__ == '__main__':
  sys.exit(main())
