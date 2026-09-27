#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the Python does, written down for the Swift to match. Every file comes
from the Python's own code paths, run on fixed inputs:

  wire      JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance/wire*
            headers and INFER bodies from protocol.py, and the byte streams
            StreamTransport frames for TCP, for a USB host (UsbBulkTransport)
            and for the gadget (FfsTransport's 16 KB bursts), with the reads a
            USB host posts to take the gadget's stream in
  staging   .../conformance/staging*: the tensors PolicyQueues.step feeds for
            the tiny queued graph at frame_skip 1, 2 and 4, a reset included
  stats     .../conformance/stats.json: FrameStats.summary on fixed samples
  control   JetlinkKit/Tests/JetlinkKitTests/Fixtures/python_control_events.jsonl:
            lines a real ControlServer writes, over a real Registry and cache
  registry  tests/fixtures/conformance/registry.json: LFS pointers, catalog
            parsing and merging, and the catalog and inventory payloads the
            Registry makes of one cache directory

  .venv/bin/python JetlinkKit/Scripts/make_conformance_fixtures.py [--root DIR]

from the root of this checkout. --root writes the same tree somewhere else;
tests/test_conformance.py does that and compares byte for byte.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# This checkout's jetlink and tests, ahead of any jetlink the environment has
# installed: a venv's editable install can point at another checkout.
sys.path.insert(0, str(ROOT))

SERVER = Path('JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance')
CONTROL = Path('JetlinkKit/Tests/JetlinkKitTests/Fixtures/python_control_events.jsonl')
REGISTRY = Path('tests/fixtures/conformance/registry.json')

SHA_A = 'a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525'
SHA_B = '4f3c1a2b9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a'
REF_A = 'f877d7a0ccc3cce943c76e285214c020cd65c899'
TAG = 'ort1.29.0.coreml-Apple_M1_Pro'
CATALOG_FILE = 'catalog_chestnut_v25.json'


def dump(value) -> str:
  return json.dumps(value, indent=2, sort_keys=True) + '\n'


# -- wire ---------------------------------------------------------------------

def payload(seq: int, n: int) -> bytes:
  """The bytes of a binary payload; the Swift tests make the same ones."""
  return bytes((seq * 31 + i * 7) % 251 for i in range(n))


# (type, seq, flags, parts) with parts a list of lengths, or a JSON text. The
# lengths put header plus payload on each side of the 1024 byte packet and the
# 16 KB burst, where the PADDED byte and the gadget's padding change.
WIRE_MESSAGES = [
  ('HELLO_REQ', 1, 0, '{"client":{"name":"modeld","nonce":7}}'),
  ('HELLO_RESP', 1, 0, '{"protocol":2,"engine_state":"none","sleep_after":0.0}'),
  ('PING', 2, 0, []),
  ('PONG', 2, 0, []),
  ('ENGINE_REQ', 3, 0, '{"sha256":"' + SHA_A + '","nbytes":765953504,"frame_skip":4}'),
  ('INFER_REQ', 4, 3, [8, 800, 184]),       # 1024 on the wire: padded
  ('INFER_RESP', 4, 0, [20, 971]),          # 1023
  ('UPLOAD_CHUNK', 5, 0, [8, 985]),         # 1025
  ('UPLOAD_CHUNK', 6, 0, [8, 2008]),        # 2048: padded
  ('STATE_RESP', 7, 0, [16352]),            # 16384: padded, and a whole burst
  ('ERROR', 8, 0, [16353]),                 # 16385
  ('INFER_RESP', 9, 0, [20, 73808, 64]),    # the driving output and telemetry
  ('SHUTDOWN_REQ', 10, 0, []),
  ('PROGRESS', 0, 0, '{"stage":"build","frac":0.5,"msg":"half"}'),
]

FRAMING = ('packet_size', 'read_chunk', 'tx_align', 'rx_align', 'write_chunk', 'read_slack')


def _memory(base, framing_from=None):
  """A transport over bytes in memory with `framing_from`'s framing rules, or
  `base` itself subclassed when its constructor opens nothing."""
  from jetlink.transport.base import LinkError, StreamTransport

  attrs = {a: getattr(framing_from, a) for a in FRAMING} if framing_from is not None else {}

  class Memory(base if framing_from is None else StreamTransport):
    def attach(self, incoming: bytes = b''):
      self.sent = bytearray()
      self.incoming = memoryview(incoming)
      self.pos = 0
      self.reads: list[int] = []
      return self

    def _write(self, bufs) -> int:
      n = 0
      for b in bufs:
        self.sent += b
        n += memoryview(b).nbytes
      return n

    def _read_into(self, dest, timeout) -> int:
      # a USB host clamps to whole packets before it reads, TCP does not
      n = self._clamp_read(dest) if self.packet_size else dest.nbytes
      self.reads.append(n)
      left = self.incoming.nbytes - self.pos
      if left <= 0:
        raise LinkError('end of the fixture stream')
      n = min(n, left)
      dest[:n] = self.incoming[self.pos:self.pos + n]
      self.pos += n
      return n

    def close(self) -> None:
      pass

  for name, value in attrs.items():
    setattr(Memory, name, value)
  return Memory


def _message_bytes(spec) -> tuple[list[bytes], bytes]:
  _, seq, _, parts = spec
  if isinstance(parts, str):
    return [parts.encode()], parts.encode()
  body = payload(seq, sum(parts))
  out, at = [], 0
  for n in parts:
    out.append(body[at:at + n])
    at += n
  return out, body


def wire(root: Path) -> None:
  from jetlink import protocol as P
  from jetlink.transport.ffs import FfsTransport
  from jetlink.transport.tcp import TcpTransport
  from jetlink.transport.usbbulk import UsbBulkTransport

  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)

  headers = []
  for i, msg in enumerate(P.Msg):
    fields = {'msg_type': int(msg), 'seq': 1000 * i + 7, 'flags': [0, 1, 2, 128, 131][i % 5],
              'length': [0, 1, 1023, 1 << 20, 16 << 20][i % 5], 'reserved': [0, 99, 2 ** 63][i % 3]}
    raw = P.pack_header(fields['msg_type'], fields['seq'], fields['length'], fields['flags'], fields['reserved'])
    headers.append({'name': msg.name, **fields, 'hex': raw.hex()})
  infer_req = [{'frame_id': f, 'flags': fl, 'hex': P.pack_infer_req(f, fl).hex()}
               for f, fl in ((0, 0), (1, 1), (4242, 3), (2 ** 32 - 1, 2))]
  infer_resp = [{'frame_id': f, 'status': int(s), 'gpu_us': g, 'queue_us': q, 'total_us': t,
                 'hex': P.pack_infer_resp(f, s, g, q, t).hex()}
                for f, s, g, q, t in ((42, P.Status.NOT_FINITE, 28000, 130, 29000), (0, P.Status.NOT_READY, 0, 0, 0),
                                      (2 ** 32 - 1, P.Status.OK, 2 ** 32 - 1, 1, 2 ** 31))]

  # usb_host: what a USB host sends and the gadget reads; usb_gadget: the
  # other way, in whole bursts; tcp: both ways over a socket
  senders = {
    'tcp': (_memory(TcpTransport, TcpTransport), _memory(TcpTransport, TcpTransport)),
    'usb_host': (_memory(UsbBulkTransport), _memory(FfsTransport, FfsTransport)),
    'usb_gadget': (_memory(FfsTransport, FfsTransport), _memory(UsbBulkTransport)),
  }
  streams = {}
  for name, (sender_cls, receiver_cls) in senders.items():
    sender = _new(sender_cls).attach()
    for spec in WIRE_MESSAGES:
      parts, _ = _message_bytes(spec)
      sender.send(P.Msg[spec[0]], spec[1], parts, spec[2])
    stream = bytes(sender.sent)
    (out / f'wire.{name}.bin').write_bytes(stream)

    # read it back the way the other end does: the fixture must be a stream
    # the receiving side of the Python takes in whole
    receiver = _new(receiver_cls).attach(stream)
    offsets = []
    for spec in WIRE_MESSAGES:
      m = receiver.recv(timeout=None)
      _, body = _message_bytes(spec)
      assert (m.msg_type, m.seq, bytes(m.payload)) == (P.Msg[spec[0]], spec[1], body), (name, spec[0])
      offsets.append(receiver.pos)
    assert receiver.pos == len(stream), name
    streams[name] = {'file': f'wire.{name}.bin', 'bytes': len(stream), 'ends': offsets,
                     'reads': receiver.reads if name == 'usb_gadget' else None}

  messages = [{'type': P.Msg[t].value, 'name': t, 'seq': s, 'flags': f,
               'parts': p if isinstance(p, list) else None, 'json': p if isinstance(p, str) else None}
              for t, s, f, p in WIRE_MESSAGES]
  (out / 'wire.json').write_text(dump({
    'payload': 'byte i of a binary payload is (seq * 31 + i * 7) % 251',
    'headers': headers, 'infer_req': infer_req, 'infer_resp': infer_resp,
    'messages': messages, 'streams': streams,
  }))


def _new(cls):
  """An instance without the real constructor's side effects, where it has any."""
  from jetlink.transport.base import StreamTransport
  from jetlink.transport.tcp import TcpTransport
  from jetlink.transport.usbbulk import UsbBulkTransport
  if issubclass(cls, UsbBulkTransport):
    return cls(handle=None)   # opens nothing; sets the 2 MB receive buffer the host uses
  if issubclass(cls, TcpTransport):
    obj = cls.__new__(cls)
    StreamTransport.__init__(obj)
    return obj
  obj = cls.__new__(cls)
  StreamTransport.__init__(obj, rx_size=256 << 10)   # FfsTransport's
  return obj


# -- staging ------------------------------------------------------------------

STAGING_FRAMES = 12
STAGING_RESET_BEFORE = 8
STAGING_INPUTS = ('img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t')


def staging(root: Path) -> None:

  from jetlink.queues import PolicyQueues
  from jetlink.spec import spec_from_onnx
  from tests import tiny_model

  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)
  cases = []
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'tiny_queued.onnx'
    tiny_model.write(path, shapes=True)
    for skip in (1, 2, 4):
      spec = spec_from_onnx(str(path), frame_skip=skip)
      rng = np.random.default_rng(20260927 + skip)
      queues = PolicyQueues(spec)
      frames, staged = bytearray(), bytearray()
      for f in range(STAGING_FRAMES):
        warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
        packed = (rng.standard_normal(spec.packed_nelem) * 2.0).astype(np.float32)
        if f == STAGING_RESET_BEFORE:
          queues.reset()
        feed = queues.step(warped, packed)
        frames += warped.tobytes() + packed.tobytes()
        for name in STAGING_INPUTS:
          assert feed[name].dtype == np.float16, name
          staged += feed[name].tobytes()
      stem = f'staging.fs{skip}'
      (out / f'{stem}.spec.json').write_text(dump(spec.to_dict()))
      (out / f'{stem}.frames.bin').write_bytes(bytes(frames))
      (out / f'{stem}.staged.bin').write_bytes(bytes(staged))
      cases.append({'frame_skip': skip, 'spec': f'{stem}.spec.json', 'frames': f'{stem}.frames.bin',
                    'staged': f'{stem}.staged.bin',
                    'inputs': [{'name': n, 'shape': list(spec.input_shapes[n])} for n in STAGING_INPUTS]})
  (out / 'staging.json').write_text(dump({
    'frames': STAGING_FRAMES, 'reset_before': STAGING_RESET_BEFORE, 'dtype': 'float16',
    'frame_layout': 'warped uint8 then packed float32, per frame',
    'staged_layout': 'each input in `inputs` order, float16, per frame',
    'cases': cases,
  }))


# -- stats --------------------------------------------------------------------

def _stats_cases():
  """(name, window, now, frames_total, samples as (at, total, gpu, queue, send) us)."""
  steady = [(10.0 + 0.05 * i, 30000 + 97 * i, 28000 + 13 * i, 500 + i, 300 + 3 * i) for i in range(20)]
  return [
    ('steady', 1.0, 11.0, 1234, steady),
    # 1.125 and 2.675: Python's round is half to even on the exact binary
    # value, and 1.125 is exact
    ('ties', 1.0, 5.0, 2, [(4.5, 1125, 1125, 0, 0), (4.6, 1125, 1125, 0, 0)]),
    ('inexact', 1.0, 5.0, 1, [(4.5, 2675, 2675, 1005, 15)]),
    # a window of 1.25 s says 1.2, not 1.3
    ('window_tie', 1.25, 5.0, 3, [(4.0, 30000, 29000, 400, 200), (4.9, 31000, 29500, 410, 210)]),
    ('cutoff', 0.5, 20.0, 9, [(19.4, 40000, 1, 1, 1), (19.5, 50000, 2, 2, 2), (19.9, 61000, 3, 3, 3)]),
    ('slow', 1.0, 3.0, 5, [(2.1, 60000, 1000, 0, 0), (2.2, 60001, 1000, 0, 0), (2.3, 90000, 1000, 0, 70000)]),
    ('empty', 1.0, 100.0, 0, [(10.0, 30000, 29000, 400, 200)]),
    ('many', 1.0, 50.0, 400, [(49.0 + i / 400, 20000 + (i * 7919) % 30000, 18000 + (i * 104729) % 9000, (i * 31) % 900,
                               (i * 17) % 700) for i in range(400)]),
  ]


def stats(root: Path) -> None:
  from jetlink.server import session as S

  out = root / SERVER
  out.mkdir(parents=True, exist_ok=True)
  cases = []
  real = S.time
  try:
    for name, window, now, total, samples in _stats_cases():
      clock = [0.0]
      S.time = SimpleNamespace(perf_counter=lambda clock=clock: clock[0])
      stats = S.FrameStats()
      for at, *us in samples:
        clock[0] = at
        stats.record(*us)
      clock[0] = now
      cases.append({'name': name, 'window': window, 'now': now, 'frames_total': total,
                    'samples': [list(s) for s in samples], 'expected': stats.summary(window, frames_total=total)})
  finally:
    S.time = real
  (out / 'stats.json').write_text(dump({'slow_frame_us': S.SLOW_FRAME_US, 'cases': cases}))


# -- a cache directory for the registry and the control channel --------------

def _seed_cache(root: Path) -> dict:
  """Writes one cache directory with something in every corner the registry
  reads, and returns it as {relative path: content} for the Swift tests to
  write the same tree. Content is {'json': ...}, {'text': ...} or {'bytes': n}."""
  catalog = json.loads((ROOT / 'tests/fixtures' / CATALOG_FILE).read_text())
  from jetlink.registry import CATALOG_URL
  newest = sorted((b for b in catalog['bundles'] if isinstance(b, dict) and b.get('is_big')),
                  key=lambda b: int(b.get('index', 0)), reverse=True)[0]['ref']
  tree = {
    'registry/catalog.json': {'json': {'fetched_at': 1757440000.0, 'url': CATALOG_URL}, 'raw_file': CATALOG_FILE},
    'registry/pointers.json': {'json': {REF_A: {'oid': SHA_A, 'size': 1234}, newest: {'oid': SHA_B, 'size': 777},
                                        'not-a-ref': {'oid': 'x', 'size': 1}}},
    'registry/local-models.json': {'json': [
      {'sha256': '0123456789abcdef' + '0' * 48, 'bytes': 5, 'name': 'My import', 'added_at': 1757440100.0},
      {'sha256': 'broken'},
    ]},
    f'models/{SHA_A[:16]}.onnx': {'bytes': 1234},
    f'models/{SHA_B[:16]}.onnx': {'bytes': 777},
    'models/0123456789abcdef.onnx': {'bytes': 5},
    'models/fedcba9876543210.onnx': {'bytes': 3},
    'models/not-a-model.onnx': {'bytes': 9},
    f'engines/{SHA_A[:16]}.{TAG}.json': {'json': {
      'backend': 'ort', 'onnxruntime': '1.29.0', 'device': 'coreml-Apple_M1_Pro', 'prepare': 5,
      'built_at': '2026-09-08T21:19:15Z', 'build_seconds': 20.4,
      'spec': {'sha256': SHA_A, 'checkpoint': 'b9facbcc-3a1e-4d3f-9a55-6d1a0f2c8e77'}}},
    f'engines/{SHA_A[:16]}.{TAG}.ortcache/sessions.json': {'bytes': 100},
    f'engines/{SHA_A[:16]}.{TAG}.ortcache/model.onnx': {'bytes': 4000},
    f'engines/{SHA_B[:16]}.trt10.16.2.Orin.json': {'json': {
      'backend': 'trt', 'trt_version': '10.16.2', 'device': 'Orin', 'built_at': '2026-09-01T10:00:00Z',
      'build_seconds': 180.0, 'spec': {'sha256': SHA_B, 'checkpoint': None}}},
    f'engines/{SHA_B[:16]}.trt10.16.2.Orin.plan': {'bytes': 2048},
    'engines/garbage.json': {'text': 'not json'},
    f'engines/{SHA_B[:16]}.orphan.json': {'json': {'backend': 'ort', 'spec': {'sha256': SHA_B}}},
    'last-loaded.json': {'json': {'sha256': SHA_A, 'frame_skip': 4, 'backend': 'ort'}},
  }
  for rel, content in tree.items():
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if 'raw_file' in content:
      path.write_text(json.dumps({**content['json'], 'raw': catalog}))
    elif 'json' in content:
      path.write_text(json.dumps(content['json']))
    elif 'text' in content:
      path.write_text(content['text'])
    else:
      path.write_bytes(bytes(content['bytes']))
  return tree


def _normalized(value, root: Path):
  """The payload with the cache root as $ROOT and the free space as 0: the
  two things that depend on where and when the fixture was made."""
  text = json.dumps(value).replace(str(root), '$ROOT')
  out = json.loads(text)
  if isinstance(out, dict) and isinstance(out.get('disk'), dict):
    out['disk']['free_bytes'] = 0
  return out


# -- registry -----------------------------------------------------------------

POINTER_TEXTS = [
  'version https://git-lfs.github.com/spec/v1\noid sha256:' + SHA_A + '\nsize 765953504\n',
  'version x\r\noid sha256:' + SHA_A + '\r\nsize 765_953_504\r\n',
  'oid sha256:' + SHA_A + '\n',
  'oid sha256:nothex\nsize 12\n',
  'oid sha256:' + SHA_A + '\nsize twelve\n',
  'oid sha256:' + SHA_A + '\nsize 0\n',
  'oid sha256:' + SHA_A + '\nsize -4\n',
  'oid sha256:' + SHA_A.upper() + '\nsize 12\n',
  'oid ' + SHA_A + '\nsize 12\n',
  'size 12\noid sha256:' + SHA_A + '\n',
  'oid sha256:' + SHA_A + '\nsize  12\n',
  'oid sha256:' + SHA_A + '\nsize 12\nsize 13\n',
  'x' * 5000,
  'not a pointer at all',
  '',
]


def _catalog_cases(catalog: dict) -> list[tuple[str, object]]:
  bundles = catalog['bundles']
  first = dict(bundles[0])

  def with_first(**change):
    return {**catalog, 'bundles': [{**first, **change}, *bundles[1:]]}

  return [
    ('first_three', catalog),
    ('selector_18', with_first(minimum_selector_version='18')),
    ('selector_int', with_first(minimum_selector_version=19)),
    ('selector_text', with_first(minimum_selector_version=' 19 ')),
    ('selector_rubbish', with_first(minimum_selector_version='nineteen')),
    ('not_big', with_first(is_big=False)),
    ('bad_ref', with_first(ref='not-a-commit')),
    ('index_text', with_first(index='42')),
    ('index_float', with_first(index=4.7)),
    ('index_rubbish', with_first(index='x')),
    ('no_name', with_first(display_name='', short_name=None, build_time=None)),
    ('duplicate', {**catalog, 'bundles': [*bundles, {**first, 'display_name': 'the second one'}]}),
    ('rubbish_bundles', {**catalog, 'bundles': [1, 'two', None, [], first]}),
    ('no_bundles', {'version': 1}),
    ('not_a_list', {'bundles': {'a': 1}}),
    ('not_an_object', ['bundles']),
  ]


def registry(root: Path) -> None:
  from jetlink.registry import Registry
  from jetlink.registry.catalog import is_ref, is_sha256, merge_catalogs, parse_catalog
  from jetlink.registry.lfs import parse_pointer_text
  from jetlink.server.cache import EngineCache

  out = root / REGISTRY
  out.parent.mkdir(parents=True, exist_ok=True)
  published = json.loads((ROOT / 'tests/fixtures' / CATALOG_FILE).read_text())
  catalog = {**published, 'bundles': published['bundles'][:3]}

  pointers = []
  for text in POINTER_TEXTS:
    p = parse_pointer_text(text)
    pointers.append({'text': text, 'expected': None if p is None else {'oid': p.oid, 'size': p.size}})
  identities = [{'value': v, 'is_ref': is_ref(v), 'is_sha256': is_sha256(v)}
                for v in (REF_A, SHA_A, REF_A.upper(), SHA_A[:63], SHA_A + '0', '', 'g' * 40, 'g' * 64)]
  parses = [{'name': 'published', 'catalog_file': CATALOG_FILE, 'expected': [vars(m) for m in parse_catalog(published)]}]
  parses += [{'name': name, 'catalog': data,
              'expected': [vars(m) for m in parse_catalog(data)]} for name, data in _catalog_cases(catalog)]

  newer = {'bundles': [
    {**catalog['bundles'][0], 'minimum_selector_version': '20', 'display_name': 'moved on'},
    {'ref': 'e' * 40, 'is_big': True, 'minimum_selector_version': '20', 'display_name': 'only newer',
     'short_name': 'ON', 'index': 99, 'build_time': '2026-09-20T00:00:00Z', 'models': [{'x': 1}]},
    {'ref': 'd' * 40, 'is_big': False, 'minimum_selector_version': '20', 'index': 98},
  ]}
  merges = [{'name': name, 'catalogs': cats, 'expected': merge_catalogs(cats)}
            for name, cats in (('one', [catalog]), ('with_newer', [catalog, newer]), ('none', []))]

  with tempfile.TemporaryDirectory() as tmp:
    cache_root = Path(tmp) / 'cache'
    tree = _seed_cache(cache_root)
    backend = SimpleNamespace(name='ort', suffix='.ortcache', tag=lambda: TAG)
    reg = Registry(cache_root)
    cached_catalog = _normalized(reg.catalog(refresh=False, max_age=float('inf')), cache_root)
    inventory = _normalized(reg.inventory(EngineCache(cache_root, backend=backend)), cache_root)

  out.write_text(dump({
    'pointers': pointers, 'identities': identities, 'parses': parses, 'merges': merges,
    'cache': {'tree': tree, 'artifact_tag': TAG, 'artifact_suffix': '.ortcache',
              'catalog': cached_catalog, 'inventory': inventory},
  }))


# -- control ------------------------------------------------------------------

def control(root: Path) -> None:
  from jetlink.server import control as C
  from jetlink.server import session as S
  from jetlink.server.cache import EngineCache
  from jetlink.registry import Registry
  from tests.fake_backend import FakeBackend

  out = root / CONTROL
  out.parent.mkdir(parents=True, exist_ok=True)
  real = (C.time, S.time)
  clock = [1757440000.0]
  fake_time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0], perf_counter=lambda: clock[0])
  try:
    C.time = S.time = fake_time
    with tempfile.TemporaryDirectory() as tmp:
      cache_root = Path(tmp) / 'cache'
      _seed_cache(cache_root)
      cache = EngineCache(cache_root, backend=FakeBackend(version='0.1', device='test'))
      host = S.EngineHost(cache)
      server = C.ControlServer(str(Path(tmp) / 'control.sock'), host, cache, registry=Registry(cache_root),
                               info={'version': '0.4.3', 'python': '3.14.7', 'platform': 'darwin',
                                     'cache': '/Users/me/Library/Application Support/Jetlink/cache',
                                     'transport': 'usb', 'port': None})
      lines: list[bytes] = []
      client = SimpleNamespace(offer=lambda line: lines.append(line) or True)
      server._clients.append(client)

      def tick(seconds: float = 0.1) -> None:
        clock[0] = round(clock[0] + seconds, 3)

      server._catalog_kicked = True    # the catalog is on disk; no fetch to kick
      server._on_connect(client)
      tick()
      server._on_host('link', {'state': 'connected', 'detail': '', 'peer': 'usb', 'medium': 'usb3'})
      tick()
      host.job = S.Job(SHA_A, load_only=False)
      host._last_stage = ('build', 0.42, 'compiling the graph')
      server._on_host('progress', {})
      tick()
      host.job = S.Job(SHA_A, load_only=True, state='failed', detail='ArtifactInvalid: made invalid')
      host._last_stage = ('failed', 1.0, '')
      server._on_host('engine', {})
      tick()
      host.job = S.Job(SHA_A, load_only=True, state='ready')
      host.loaded = SimpleNamespace(sha256=SHA_A)
      host._last_stage = ('load', 1.0, 'loaded in 1.8 s')
      server._on_host('engine', {})
      tick()
      for i in range(20):
        tick(0.05)
        host.frame_stats.record(30000 + 97 * i, 28000 + 13 * i, 500 + i, 300 + 3 * i)
      server.publish('stats', host.frame_stats.summary(1.0, frames_total=1234))
      tick()
      download = C._Download(sha256=SHA_A, ref=REF_A, total=765953504,
                             source='https://gitlab.com/commaai/openpilot-lfs.git/info/lfs')
      server._download_event(download, 'queued')
      tick(0.6)
      server._download_event(download, 'progress', frac=0.42)
      tick(0.6)
      server._download_event(download, 'progress', frac=0.9)
      tick()
      server._download_event(download, 'failed', detail='NetworkError: could not fetch')
      tick()
      server._import_event(Path('/Users/me/Downloads/big.onnx'), 'hashing', frac=0.3)
      tick()
      server._import_event(Path('/Users/me/Downloads/big.onnx'), 'done', frac=1.0, sha256=SHA_B)
      tick()
      for raw in (b'{"id": 7, "cmd": "inventory"}', b'{"id": 8, "cmd": "nothing"}', b'not json',
                  b'{"id": true, "cmd": "status"}', b'{"id": 9, "cmd": "cancel_download", "sha256": "' + SHA_A.encode() + b'"}'):
        server._handle(client, raw)
        tick()
      server._reply(client, 10, True, None, {'queued': True, 'sha256': SHA_A})
      server.publish('server', server._server_payload('stopping'))
  finally:
    C.time, S.time = real

  events = []
  for line in b''.join(lines).decode().splitlines():
    event = _normalized(json.loads(line), cache_root)
    if event['event'] == 'hello':
      event['pid'] = 4242
    events.append(json.dumps(event, separators=(',', ':')))
  out.write_text('\n'.join(events) + '\n')


# -----------------------------------------------------------------------------

PARTS = {'wire': wire, 'staging': staging, 'stats': stats, 'control': control, 'registry': registry}


def generate(root: Path = ROOT, parts=tuple(PARTS)) -> None:
  for name in parts:
    PARTS[name](root)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='The conformance fixtures: what the Python does, for the Swift to match.')
  parser.add_argument('--root', type=Path, default=ROOT, help='write the tree under this directory (default: this checkout)')
  parser.add_argument('parts', nargs='*', metavar='PART', help=f"some of {', '.join(PARTS)} (default: all)")
  args = parser.parse_args()
  unknown = set(args.parts) - set(PARTS)
  if unknown:
    parser.error(f"no such fixtures: {', '.join(sorted(unknown))}")
  generate(args.root, args.parts or tuple(PARTS))
