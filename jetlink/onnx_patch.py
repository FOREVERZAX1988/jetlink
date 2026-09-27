"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Make an openpilot driving model acceptable to TensorRT's ONNX parser.

TensorRT 10.3 rejects UINT8 graph inputs ("Found unsupported input type of
UINT8"), so the image inputs are declared FP16 and the head Cast deleted.
Feeding 0..255 as fp16 is exact and free; the weights are FP16 already. A
stateful graph also hands its uint8 frame queue back as an output, which is
retyped with it.

Runs on the Jetson at build time. The shipped model is never modified in place.
"""
from __future__ import annotations

import logging

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

log = logging.getLogger('jetlink.onnx_patch')

# where the vision trunk starts: img and big_img in a queued graph, the newest
# frame and the frame queue in a stateful one (openpilot #38916). The uint8
# patch needs no names; it starts from whatever inputs are uint8.
IMG_INPUTS = ('img', 'big_img', 'new_img', 'state_img_q')
# Ops that only move image bytes around. Each one's output has its input's
# type, so a retyped input retypes the output, and 0..255 comes through every
# one of them exactly.
LAYOUT_OPS = frozenset(('Concat', 'Slice', 'Gather', 'Reshape', 'Unsqueeze', 'Squeeze',
                        'Transpose', 'Flatten', 'Identity', 'Expand'))

# tinygrad's exporter leaves layout hints in its own domain, and TensorRT rejects
# any op in a domain it does not know. Contiguous is about tinygrad's buffers,
# not the arithmetic, so bypassing it is safe. Whether a model has one varies
# with how it was exported.
TINYGRAD_DOMAIN = 'org.tinygrad'
PASSTHROUGH_OPS = ('Contiguous',)


def _uint8_inputs(model: onnx.ModelProto) -> list:
  return [vi for vi in model.graph.input if vi.type.tensor_type.elem_type == TensorProto.UINT8]


def needs_patch(model: onnx.ModelProto) -> bool:
  return bool(_uint8_inputs(model))


def strip_tinygrad_ops(model: onnx.ModelProto) -> int:
  """Bypass tinygrad's layout-hint nodes. In place, returns how many went.

  Refuses anything it has not been told is a passthrough rather than guessing:
  silently dropping an op that did something would change what the car sees.
  """
  g = model.graph
  graph_outputs = {o.name for o in g.output}
  removed = 0

  for node in [n for n in g.node if n.domain == TINYGRAD_DOMAIN]:
    if node.op_type not in PASSTHROUGH_OPS:
      raise ValueError(f"unknown {TINYGRAD_DOMAIN} op {node.op_type!r}; it may not "
                       "be a no-op, so dropping it is not safe")
    if len(node.input) != 1 or len(node.output) != 1 or node.attribute:
      raise ValueError(f"{node.op_type} is not a plain one-in one-out passthrough")

    source, produced = node.input[0], node.output[0]
    if produced in graph_outputs:
      # The output keeps its name: output_slices and the parity tools address
      # it. So the producer takes the name over; renaming both ends leaves the
      # output with no producer at all.
      for n in g.node:
        for i, name in enumerate(n.output):
          if name == source:
            n.output[i] = produced
      for n in g.node:
        for i, name in enumerate(n.input):
          if name == source:
            n.input[i] = produced
    else:
      for n in g.node:
        for i, name in enumerate(n.input):
          if name == produced:
            n.input[i] = source
    g.node.remove(node)
    removed += 1

  if removed:
    for i, opset in enumerate(model.opset_import):
      if opset.domain == TINYGRAD_DOMAIN:
        del model.opset_import[i]
        break
    live = {n for node in g.node for n in list(node.input) + list(node.output)}
    for i in reversed(range(len(g.value_info))):
      if g.value_info[i].name not in live:
        del g.value_info[i]
  return removed


def normalize_gather_indices(model: onnx.ModelProto) -> int:
  """Rewrite Gather nodes whose constant index is negative to the positive
  equivalent. In place, returns how many were rewritten.

  Apple's Neural Engine returns garbage for Gather with a scalar index of -1
  on a large axis: measured on the driving model's `select_2` (add_53[:, -1]
  over 288 tokens), correlation 0.03 against the CPU, exact with the index
  written as 287. ONNX defines a negative index as counting from the end, so
  the rewrite changes nothing about the graph's meaning; it needs the data's
  static size along the axis, which the exporter's value_info carries, and
  leaves any node whose size it cannot see alone rather than guess.

  Each rewritten node gets its own index initializer: the exporter shares one
  constant between Gathers on axes of different sizes.
  """
  g = model.graph
  init = {t.name: t for t in g.initializer}
  gathers = [n for n in g.node if n.op_type == 'Gather' and len(n.input) > 1 and n.input[1] in init]
  dims = _static_dims(model, {n.input[0] for n in gathers})
  rewritten = 0
  for node in gathers:
    index = numpy_helper.to_array(init[node.input[1]])
    if not np.issubdtype(index.dtype, np.integer) or index.size == 0 or index.min() >= 0:
      continue
    axis = next((helper.get_attribute_value(a) for a in node.attribute if a.name == 'axis'), 0)
    shape = dims.get(node.input[0])
    if shape is None or axis >= len(shape) or shape[axis] <= 0:
      continue
    size = shape[axis]
    fixed = np.where(index < 0, index + size, index).astype(index.dtype)
    if (fixed < 0).any() or (fixed >= size).any():
      raise ValueError(f"{node.name}: Gather index {index.tolist()} out of range for axis {axis} of size {size}")
    name = f"{node.output[0]}__index"
    g.initializer.append(numpy_helper.from_array(fixed, name))
    node.input[1] = name
    rewritten += 1
  return rewritten


def _vision_mask(model: onnx.ModelProto) -> list[bool]:
  """Per node, in graph order: whether it depends on the image inputs alone,
  the vision trunk and the heads that hang off it. Found by dataflow: a node
  is vision when every tensor it reads is an image input, an initializer, or
  another vision node's output. Nodes reading only initializers count as
  neither. By position, because an exporter need not name its nodes."""
  g = model.graph
  init = {t.name for t in g.initializer}
  image_inputs = {vi.name for vi in g.input if vi.name in IMG_INPUTS}
  if not image_inputs:
    raise ValueError(f"model has none of {IMG_INPUTS} as graph inputs")
  vision_tensors = set(image_inputs)
  mask = []
  for node in g.node:
    data = [x for x in node.input if x and x not in init]
    vision = bool(data) and all(x in vision_tensors for x in data)
    if vision:
      vision_tensors.update(node.output)
    mask.append(vision)
  return mask


def _trunk_end(g, mask: list[bool], images: list[str], handed: list[str]) -> str | None:
  """The latest vision tensor every one of `handed` is computed from and
  nothing else of the images: where the trunk narrows to one tensor before
  it fans out into heads. Never a graph output, which the worker has to
  return rather than pass on. None if there is no such tensor.

  Walks the vision nodes backwards, replacing each needed tensor by what its
  node reads: at every step the needed set separates the images from
  `handed`, so a step where it is one tensor has found a place to cut."""
  init = {t.name for t in g.initializer}
  never = {o.name for o in g.output} | set(images)
  live = set(handed)
  for node, vision in zip(reversed(g.node), reversed(mask), strict=True):
    if len(live) == 1 and not live & never:
      return next(iter(live))
    if vision and live.intersection(node.output):
      live.difference_update(node.output)
      live.update(i for i in node.input if i and i not in init)
  return None


def split_vision_policy(model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
  """The graph cut where the image-only trunk ends, as (vision, policy).

  vision takes the image inputs and returns the trunk's output, plus any
  graph output made from the images alone (a stateful graph's
  next_state_img_q). policy takes that and the other graph inputs and returns
  the remaining graph outputs. Run as a chain, feeding by name
  (backends/ort/worker.py), the two compute what the whole graph does.

  The cut is where the trunk narrows to one tensor before fanning out into
  heads: on the driving models the last conv's output, 32 KB a frame, so the
  hand-off costs nothing. Everything after it goes with the policy, including
  the image-only heads `_vision_mask` counts as vision: they are a residual
  MLP whose LayerNormalizations lose too much in the Neural Engine's fp16
  (Cinque Terre V3's road_transform at corr 0.9988 over 32 frames with them
  on it, under the parity gate's 0.999). A graph without such a tensor is cut
  at `_vision_mask`'s whole boundary instead.
  """
  from onnx.shape_inference import infer_shapes
  from onnx.utils import Extractor

  g = model.graph
  mask = _vision_mask(model)
  made = {o for n, vision in zip(g.node, mask, strict=True) if vision for o in n.output}
  policy_nodes = [n for n, vision in zip(g.node, mask, strict=True) if not vision]
  images = [i.name for i in g.input if i.name in IMG_INPUTS]
  others = [i.name for i in g.input if i.name not in IMG_INPUTS]
  if any(i in images for n in policy_nodes for i in n.input):
    raise ValueError("a policy node reads the image inputs directly; the graph has no clean vision trunk")
  handed = sorted({i for n in policy_nodes for i in n.input if i in made})
  if not handed:
    raise ValueError("nothing crosses from the vision trunk to the policy")
  cut = _trunk_end(g, mask, images, handed)
  ends = [cut] if cut is not None else handed
  outputs = [o.name for o in g.output]
  # The hand-off tensors become graph inputs and outputs, which need a type
  # and a fixed shape. The driving models' exports record them; the shape
  # inferrer, 1.3 s and a second copy of the weights, only for one that does not.
  typed = {vi.name for vi in (*g.input, *g.value_info, *g.output) if vi.type.tensor_type.HasField('shape')}
  extractor = Extractor(model if set(ends) <= typed else infer_shapes(model))
  vision_model = extractor.extract_model(images, ends + [o for o in outputs if o in made])
  policy_model = extractor.extract_model(ends + others, [o for o in outputs if o not in made])
  return vision_model, policy_model


def expand_to_tile(model: onnx.ModelProto) -> int:
  """Rewrite an Expand with a constant shape of the input's rank, which only
  repeats size-1 axes, as the equivalent Tile. In place; returns how many.

  onnxruntime's CoreML provider does not take Expand, and the policy has two
  (a [1, 9, 1, 512] input expanded to [1, 9, 32, 512]). Each one it refuses
  splits the graph: on an M1 Pro with `--device ane` the model ran as two
  CoreML programs with the Expands on the CPU between them. As Tiles, which it
  does take, the model is one program again, and a 20 Hz paced frame went
  from 29.5 ms mean and 31.1 p99 to 27.3 and 28.7 (1,200 frames, 2026-09-25).
  An Expand that broadcasts a lower-rank input, or whose shape is not a
  constant, is left alone."""
  g = model.graph
  init = {t.name: t for t in g.initializer}
  dims = _static_dims(model, {n.input[0] for n in g.node if n.op_type == 'Expand'})
  done = 0
  for n in g.node:
    if n.op_type != 'Expand' or len(n.input) != 2 or n.input[1] not in init:
      continue
    shape = dims.get(n.input[0])
    target = [int(v) for v in numpy_helper.to_array(init[n.input[1]]).reshape(-1)]
    if shape is None or len(shape) != len(target) or any(d <= 0 for d in shape):
      continue
    repeats = []
    for have, want in zip(shape, target, strict=True):
      if want in (1, have):
        repeats.append(1)
      elif have == 1:
        repeats.append(want)
      else:
        repeats = None
        break
    if repeats is None:
      continue
    name = f"{n.output[0]}__repeats"
    g.initializer.append(numpy_helper.from_array(np.array(repeats, np.int64), name))
    n.op_type = 'Tile'
    n.input[1] = name
    done += 1
  return done


def _static_dims(model: onnx.ModelProto, wanted: set[str]) -> dict[str, tuple[int, ...]]:
  """Static shapes of the graph's tensors, from what the file carries; the
  shape inferrer only when one of `wanted` is missing. Fails open: a tensor
  still unknown afterwards is simply absent, and the caller decides what it
  cannot do without it."""
  g = model.graph
  dims: dict[str, tuple[int, ...]] = {}

  def take(values):
    for vi in values:
      tt = vi.type.tensor_type
      if tt.HasField('shape'):
        dims[vi.name] = tuple(d.dim_value if d.HasField('dim_value') else -1 for d in tt.shape.dim)
  take(g.input)
  take(g.value_info)
  take(g.output)
  if wanted - dims.keys():
    try:
      take(onnx.shape_inference.infer_shapes(model).graph.value_info)
    except Exception:
      pass
  return dims


def _static_types(model: onnx.ModelProto, wanted: set[str]) -> dict[str, int]:
  """Element types of the graph's tensors, as `_static_dims` finds shapes:
  from what the file carries, the shape inferrer only when one of `wanted`
  is missing, and fails open the same way."""
  g = model.graph
  types: dict[str, int] = {}

  def take(values):
    for vi in values:
      if vi.type.tensor_type.elem_type:
        types[vi.name] = vi.type.tensor_type.elem_type
  take(g.input)
  take(g.value_info)
  take(g.output)
  if wanted - types.keys():
    try:
      take(onnx.shape_inference.infer_shapes(model).graph.value_info)
    except Exception:
      pass
  return types


# -- the whole graph on the Neural Engine ------------------------------------
# `--device ane` keeps everything after the vision trunk off the Neural Engine
# (split_vision_policy). `--device ane-whole` runs the graph as one CoreML
# program with every compute unit allowed instead, which is what a phone,
# whose GPU is far weaker than its Neural Engine, wants; these two passes are
# what that takes. Measured on an iPhone 17 Pro and an M1 Pro, 2026-09-24/25.

# LayerNorm(x / k) is LayerNorm(x) with epsilon scaled by k^2: 1e-5 becomes
# 6.4e-4 at k = 8, well under the variance of anything the norms see, so the
# epsilon is left alone. The policy's inputs reach 1189, whose square
# overflows fp16, while the row sums of (x / 8)^2 stay under 34,000.
LAYERNORM_PRESCALE = 8
# The ops heads_in_fp32 moves to fp32: the small MLPs and linear layers that
# end the vision trunk. A reduction, a reshape or anything else ends the region.
HEAD_OPS = frozenset(('Gemm', 'MatMul', 'LayerNormalization', 'Gelu', 'Add', 'Sub', 'Mul', 'Div',
                      'Relu', 'Sigmoid', 'Tanh'))
# Larger than that is not the heads but the trunk itself, which belongs in fp16.
HEAD_MAX_NODES = 64


def prescale_layernorm(model: onnx.ModelProto, k: int = LAYERNORM_PRESCALE) -> int:
  """Feed every fp16 LayerNormalization on the policy side its input times
  1/k, one Mul per distinct input. In place; returns how many norms.

  The Neural Engine's fp16 LayerNormalization squares its input before it
  reduces, and the policy's residual stream reaches values whose square
  overflows fp16. Scaling the input first keeps the sum in range and changes
  the result only through epsilon (see LAYERNORM_PRESCALE). Vision-side
  norms (`_vision_mask`) are left alone: their inputs are small, and on
  `ane-whole` they go to fp32 with the heads. A norm whose input type the
  file does not record and the inferrer cannot find is left alone too.

  Deterministic, so a port can reproduce the bytes: the constant is one fp16
  initializer `__layernorm_prescale_{k}` holding np.float16(1 / k), appended
  after the existing initializers; a norm reading `x` reads `x__scaled`
  instead, made by `Mul(x, const)` named `x__prescale` inserted directly
  before the first norm that reads `x`, in graph order. A norm already fed by
  such a Mul is not scaled again, so the pass is idempotent.
  """
  g = model.graph
  const = f"__layernorm_prescale_{k}"
  mask = _vision_mask(model)
  norms = {i for i, (n, vision) in enumerate(zip(g.node, mask, strict=True))
           if n.op_type == 'LayerNormalization' and not vision}
  already = {n.output[0] for n in g.node if n.op_type == 'Mul' and len(n.input) == 2 and n.input[1] == const}
  types = _static_types(model, {g.node[i].input[0] for i in norms})
  new, scaled, done = [], {}, 0
  for i, node in enumerate(g.node):
    if i in norms and node.input[0] not in already and types.get(node.input[0]) == TensorProto.FLOAT16:
      x = node.input[0]
      if x not in scaled:
        scaled[x] = f"{x}__scaled"
        new.append(helper.make_node('Mul', [x, const], [scaled[x]], name=f"{x}__prescale"))
      node.input[0] = scaled[x]
      done += 1
    new.append(node)
  if done:
    g.initializer.append(numpy_helper.from_array(np.array(1.0 / k, np.float16), const))
    del g.node[:]
    g.node.extend(new)
  return done


def vision_heads(model: onnx.ModelProto) -> list[int]:
  """The indices, in graph order, of the heads that end the vision trunk:
  the largest set of vision nodes (`_vision_mask`) with ops in HEAD_OPS
  whose outputs are all read, and read only, by each other or by a Concat
  that makes a graph output. Empty when there is no such Concat or the set
  would exceed HEAD_MAX_NODES.

  Found by growing backwards to a fixed point: a pass over the nodes from
  last to first adds each one whose every output has a reader and every
  reader is already in the region or is such a Concat, and passes repeat
  until one adds nothing. A node that makes a graph output itself never
  joins: the worker returns it as it is.
  """
  g = model.graph
  mask = _vision_mask(model)
  outputs = {o.name for o in g.output}
  ends = {i for i, n in enumerate(g.node) if n.op_type == 'Concat' and any(o in outputs for o in n.output)}
  if not ends:
    return []
  readers: dict[str, set[int]] = {}
  for i, n in enumerate(g.node):
    for x in n.input:
      readers.setdefault(x, set()).add(i)
  region: set[int] = set()
  grew = True
  while grew:
    grew = False
    for i in reversed(range(len(g.node))):
      n = g.node[i]
      if i in region or not mask[i] or n.op_type not in HEAD_OPS or any(o in outputs for o in n.output):
        continue
      read = [readers.get(o, set()) for o in n.output]
      if all(r and r <= region | ends for r in read):
        region.add(i)
        grew = True
  return sorted(region) if len(region) <= HEAD_MAX_NODES else []


def heads_in_fp32(model: onnx.ModelProto) -> int:
  """Run `vision_heads` in fp32: cast what they read from the trunk up,
  their fp16 weights to fp32 copies, and what they hand the output Concat
  back down. The Neural Engine cannot run fp32, so CoreML places them on the
  GPU or CPU. In place; returns how many nodes moved, 0 with a warning when
  it found no heads it could move.

  The heads are small (24 nodes and 4 MB of weights in the 766 MB chestnut
  model), but in fp16 on the Neural Engine their LayerNormalization, Gelu
  and 1024-wide Gemms lose enough that road_transform fails the parity gate
  on an iPhone 17 Pro (worst column 0.9989). Computed exactly from the
  Neural Engine's own trunk output, every column is 0.9996 or better.
  Measured 2026-09-25. `split_vision_policy` keeps them off the Neural
  Engine for the same reason.

  Deterministic, so a port can reproduce the bytes. An entry is a tensor a
  head reads that no head makes and no initializer is; an exit is a tensor
  a head makes that something outside the heads reads. Every entry has to
  be fp16 and every head weight fp16 or fp32, or nothing is done. In name
  order, each fp16 weight `w` gets an fp32 copy `w__fp32` appended to the
  initializers (an existing tensor of that name is reused). Then, in graph
  order: the first head to read an entry `x` is preceded by
  `Cast(x) -> x__fp32` named `x__cast_fp32`; a head that makes an exit `o`
  writes `o__fp32` and is followed by `Cast(o__fp32) -> o` named
  `o__cast_fp16`, one per exit in the node's output order; head inputs are
  renamed to the `__fp32` tensor throughout. The fp16 weights nothing
  outside the heads reads are dropped, and the value_infos of the tensors
  made and consumed inside the heads, which said fp16, with them. Running it
  again finds no heads (the Casts fence them off) and changes nothing.
  """
  g = model.graph
  index = vision_heads(model)
  if not index:
    log.warning("heads_in_fp32: no heads found after the vision trunk (no output Concat they feed, "
                "or more than %d nodes); the whole graph stays fp16", HEAD_MAX_NODES)
    return 0
  heads = set(index)
  init = {t.name: t for t in g.initializer}
  produced = {o for i in index for o in g.node[i].output}
  entries = {x for i in index for x in g.node[i].input if x and x not in init and x not in produced}
  types = _static_types(model, entries)
  if any(types.get(x) != TensorProto.FLOAT16 for x in entries):
    log.warning("heads_in_fp32: a head reads %s, which is not fp16; the whole graph stays fp16",
                sorted(x for x in entries if types.get(x) != TensorProto.FLOAT16))
    return 0
  weights = {x for i in index for x in g.node[i].input if x in init}
  odd = sorted(w for w in weights if init[w].data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT))
  if odd:
    log.warning("heads_in_fp32: head weights %s are neither fp16 nor fp32; the whole graph stays fp16", odd)
    return 0
  read_outside = {x for j, n in enumerate(g.node) if j not in heads for x in n.input}
  exits = produced & read_outside
  wide = {}
  for w in sorted(weights):
    if init[w].data_type == TensorProto.FLOAT16:
      wide[w] = f"{w}__fp32"
      if wide[w] not in init:
        g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(init[w]).astype(np.float32), wide[w]))
  new, cast = [], set()
  for j, n in enumerate(g.node):
    if j not in heads:
      new.append(n)
      continue
    for i, x in enumerate(n.input):
      if x in entries:
        if x not in cast:
          cast.add(x)
          new.append(helper.make_node('Cast', [x], [f"{x}__fp32"], name=f"{x}__cast_fp32", to=TensorProto.FLOAT))
        n.input[i] = f"{x}__fp32"
      elif x in exits:
        n.input[i] = f"{x}__fp32"
      elif x in wide:
        n.input[i] = wide[x]
    back = [o for o in n.output if o in exits]
    for i, o in enumerate(n.output):
      if o in exits:
        n.output[i] = f"{o}__fp32"
    new.append(n)
    for o in back:
      new.append(helper.make_node('Cast', [f"{o}__fp32"], [o], name=f"{o}__cast_fp16", to=TensorProto.FLOAT16))
  del g.node[:]
  g.node.extend(new)
  # The fp16 originals, unless something outside the heads reads them too.
  keep = [t for t in g.initializer if t.name not in wide or t.name in read_outside]
  del g.initializer[:]
  g.initializer.extend(keep)
  # What the heads compute inside is fp32 now; the value infos said fp16.
  keep_vi = [v for v in g.value_info if v.name not in produced - exits]
  del g.value_info[:]
  g.value_info.extend(keep_vi)
  return len(index)


def patch_uint8_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
  """Retype the uint8 inputs, the images, to fp16 and drop the head Cast. In place.

  Everything between the inputs and the Cast is followed, so the fix is the
  same whatever the exporter put there or called the inputs:

    comma      Concat(img, big_img) -> cat -> Cast(fp16)
    sunnypilot Cast(img), Cast(big_img) -> Concat -> cat
    stateful   Concat(Slice(state_img_q), Unsqueeze(new_img)) -> next_state_img_q
                 -> Gather, Slice, Reshape -> Concat -> Cast(fp16)

  Every tensor on the way is retyped, next_state_img_q included, so the queue
  the graph hands back is fp16 too and feeds straight back into its input.
  """
  g = model.graph

  img_inputs = _uint8_inputs(model)
  if not img_inputs:
    raise ValueError("model has no uint8 inputs; already patched?")

  retyped, casts = _image_dataflow(g, {vi.name for vi in img_inputs})
  if not casts:
    raise ValueError("could not find the head Cast: no Cast ends the uint8 image chain")
  graph_outputs = {vi.name for vi in g.output}
  for cast in casts:
    to = next(onnx.helper.get_attribute_value(a) for a in cast.attribute if a.name == 'to')
    if to != TensorProto.FLOAT16:
      raise ValueError(f"head Cast targets {to}, expected FLOAT16 ({TensorProto.FLOAT16})")
    if cast.output[0] in graph_outputs:
      raise ValueError(f"head Cast {cast.name} feeds a graph output; dropping it would rename one")

  for cast in casts:
    # Everything downstream reads the cast's source directly now.
    source, produced = cast.input[0], cast.output[0]
    for n in g.node:
      for i, name in enumerate(n.input):
        if name == produced:
          n.input[i] = source
    g.node.remove(cast)

  # the inputs, and every tensor on the way to the Cast: TensorRT tolerates a
  # stale uint8 value_info, onnxruntime rejects the model
  for vi in [*g.input, *g.value_info, *g.output]:
    if vi.name in retyped:
      vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  return model


def _image_dataflow(g, sources: set[str]) -> tuple[set[str], list]:
  """The tensors carrying image bytes from `sources`, and the Casts that end
  them. Nodes are in topological order, as ONNX requires, so one pass sees
  every producer before its consumers.

  Anything else reading the bytes is refused: an op that does arithmetic on
  uint8 would change meaning under fp16, and guessing is not safe."""
  retyped = set(sources)
  casts = []
  for node in g.node:
    if not any(i in retyped for i in node.input):
      continue
    if node.op_type == 'Cast':
      casts.append(node)
    elif node.op_type in LAYOUT_OPS:
      retyped.update(node.output)
    else:
      raise ValueError(f"{node.op_type} {node.name!r} reads the uint8 images; only layout ops "
                       "and the head Cast are expected there")
  return retyped, casts


# A weight smaller than this stays as it is. onnxruntime writes an initializer
# of ten elements or more to the weight file, but rewriting a graph is only
# worth it where the MIL text would be large, and a bias-sized constant costs
# a few hundred bytes either way.
BLOB_MIN_ELEMENTS = 1024


def gemm_with_transposed_weight(model: onnx.ModelProto) -> int:
  """Rewrite `MatMul(x, W)` followed by `Add(b)` as `Gemm(x, W.T, b,
  transB=1)`. In place, returns how many were rewritten.

  onnxruntime's MatMulAddFusion already does this fusion at optimization
  level 1, but it emits `transB=0` and leaves W as it was. The CoreML EP's
  Gemm builder then transposes W on the host and adds it through
  `AddConstant`, which is always an immediate, so the weight is written into
  `model.mil` as hex float text: 6.06 bytes per fp16 value measured on an M1
  Pro, which is how the trunk's MIL reached 4.1 GB against a 47 MB
  weight.bin. Every load parses all of it. Handed W already transposed with
  `transB=1`, the builder passes the initializer through as a TensorProto and
  it lands in the weight file instead. The MIL op is `linear` either way, so
  this moves bytes and changes no arithmetic.

  Doing it here rather than leaving it to onnxruntime leaves the fusion
  nothing to fuse. ORT flattens a rank-3 or rank-4 A to 2-D around the Gemm
  and reshapes the result back, because ONNX Gemm is 2-D only; this inserts
  the same pair, so the graph onnxruntime receives is the one its own fusion
  would have built. `session.disable_specified_optimizers` was tried first
  and does not reach this transformer in onnxruntime 1.29.0: the 106 fused
  nodes keep their MatMulAddFusion names under every spelling of it.
  """
  g = model.graph
  init = {t.name: t for t in g.initializer}
  consumers: dict[str, list] = {}
  for node in g.node:
    for name in node.input:
      consumers.setdefault(name, []).append(node)
  outputs = {vi.name for vi in g.output}

  candidates = [n for n in g.node if n.op_type == 'MatMul' and len(n.input) == 2
                and n.input[1] in init and n.output[0] not in outputs]
  dims = _static_dims(model, {n.input[0] for n in candidates})

  replacements: dict[int, list] = {}
  drop: set[int] = set()
  order = {id(n): i for i, n in enumerate(g.node)}
  rewritten = 0
  for node in candidates:
    weight = init[node.input[1]]
    if len(weight.dims) != 2 or weight.dims[0] * weight.dims[1] < BLOB_MIN_ELEMENTS:
      continue
    after = consumers.get(node.output[0], [])
    if len(after) != 1 or after[0].op_type != 'Add':
      continue
    add = after[0]
    bias = next((i for i in add.input if i in init), None)
    # The bias has to be the one that broadcasts over the output's last axis;
    # anything else is a real elementwise Add and not a Gemm's C.
    if bias is None or list(init[bias].dims) != [weight.dims[1]]:
      continue
    shape = dims.get(node.input[0])
    if shape is None or len(shape) < 2 or shape[-1] != weight.dims[0] or any(d <= 0 for d in shape):
      continue

    stem = node.output[0]
    array = numpy_helper.to_array(weight)
    transposed = f"{stem}__wt"
    g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array.T), transposed))
    del array

    new: list = []
    a_name = node.input[0]
    if len(shape) > 2:
      flat = f"{stem}__flat_shape"
      g.initializer.append(numpy_helper.from_array(
        np.array([-1, weight.dims[0]], dtype=np.int64), flat))
      a_name = f"{stem}__flat"
      new.append(helper.make_node('Reshape', [node.input[0], flat], [a_name],
                                  name=f"{stem}__reshape_in"))

    gemm_out = add.output[0] if len(shape) == 2 else f"{stem}__gemm"
    new.append(helper.make_node('Gemm', [a_name, transposed, bias], [gemm_out],
                                name=f"{stem}__gemm", transB=1))
    if len(shape) > 2:
      back = f"{stem}__out_shape"
      g.initializer.append(numpy_helper.from_array(
        np.array(list(shape[:-1]) + [weight.dims[1]], dtype=np.int64), back))
      new.append(helper.make_node('Reshape', [gemm_out, back], [add.output[0]],
                                  name=f"{stem}__reshape_out"))

    replacements[order[id(node)]] = new
    drop.add(order[id(add)])
    rewritten += 1

  if not rewritten:
    return 0

  rebuilt = []
  for i, node in enumerate(g.node):
    if i in drop:
      continue
    rebuilt.extend(replacements.get(i, [node]))
  del g.node[:]
  g.node.extend(rebuilt)
  _drop_unused_initializers(model)
  return rewritten


def _drop_unused_initializers(model: onnx.ModelProto) -> int:
  """The weights the rewrite left behind. A transposed copy replaces the
  original, and the model would otherwise carry 671 MB of both."""
  g = model.graph
  used = {name for node in g.node for name in node.input}
  stale = [t for t in g.initializer if t.name not in used]
  for t in stale:
    g.initializer.remove(t)
  return len(stale)


def patch_file(src: str, dst: str, check: bool = True) -> str:
  model = onnx.load(src)
  strip_tinygrad_ops(model)
  if needs_patch(model):
    patch_uint8_inputs(model)
  if check:
    onnx.checker.check_model(model, full_check=False)
  onnx.save(model, dst)
  return dst


if __name__ == '__main__':
  import sys
  print(patch_file(sys.argv[1], sys.argv[2]))
