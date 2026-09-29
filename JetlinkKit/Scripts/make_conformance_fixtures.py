#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma's Python does, written down for the Swift server to match.
Every file comes from the code the comma runs, on fixed inputs:

  wire      JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance/wire*
            headers and INFER bodies from protocol.py, and the byte streams
            StreamTransport frames for TCP, for a USB host and for the gadget
            (FfsTransport's 16 KB bursts), with the reads a USB host posts to
            take the gadget's stream in
  staging   .../conformance/staging*: the tensors PolicyQueues.step feeds for
            the tiny queued graph at frame_skip 1, 2 and 4, with the hidden
            state each frame's output feeds back, a hello, a non-finite frame
            and a reset included; checked here against protocol 2's staging,
            where modeld fed the hidden state back through prev_feat
  registry  tests/fixtures/conformance/registry.json: LFS pointers, model
            identities, catalog parsing and merging. Its `cache` block (one
            cache directory's catalog and inventory payloads) was written by
            the Python server's registry and is frozen: this copies it through.

  .venv/bin/python JetlinkKit/Scripts/make_conformance_fixtures.py [--root DIR]

from the root of this checkout. --root writes the same tree somewhere else;
tests/test_conformance.py does that and compares byte for byte. The server's
own goldens beside these (stats.json, the golden frames, the control events)
are the Swift's, written once and never regenerated (docs/conformance.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# This checkout's jetlink and tests, ahead of any jetlink the environment has
# installed: a venv's editable install can point at another checkout.
sys.path.insert(0, str(ROOT))

SERVER = Path('JetlinkKit/Tests/JetlinkServerTests/Fixtures/conformance')
REGISTRY = Path('tests/fixtures/conformance/registry.json')

SHA_A = 'a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525'
REF_A = 'f877d7a0ccc3cce943c76e285214c020cd65c899'
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
  ('HELLO_REQ', 1, 0, '{"client":{"name":"modeld","nonce":7,"protocol":3}}'),
  ('HELLO_RESP', 1, 0, '{"protocol":3,"engine_state":"none","sleep_after":0.0}'),
  ('PING', 2, 0, []),
  ('PONG', 2, 0, []),
  ('ENGINE_REQ', 3, 0, '{"sha256":"' + SHA_A + '","nbytes":765953504,"frame_skip":4}'),
  ('INFER_REQ', 4, 3, [8, 800, 184]),       # 1024 on the wire: padded
  ('INFER_RESP', 4, 0, [20, 971]),          # 1023
  ('UPLOAD_CHUNK', 5, 0, [8, 985]),         # 1025
  ('UPLOAD_CHUNK', 6, 0, [8, 2008]),        # 2048: padded
  ('STATE_RESP', 7, 0, [16352]),            # 16384: padded, and a whole burst
  ('ERROR', 8, 0, [16353]),                 # 16385
  ('INFER_RESP', 9, 0, [20, 2066 * 4, 2 * 4, 64]),  # the big models' outputs either side of hidden_state, telemetry
  ('INFER_RESP', 11, 0, [20, 73808]),       # the whole vector, on WANT_HIDDEN
  ('SHUTDOWN_REQ', 10, 0, []),
  ('PROGRESS', 0, 0, '{"stage":"build","frac":0.5,"msg":"half"}'),
]

FRAMING = ('packet_size', 'read_chunk', 'tx_align', 'rx_align', 'write_chunk', 'read_slack')


def _usb_host():
  """The USB host's framing, as the comma's gadget expects it: whole-packet
  reads of up to USB_READ_CHUNK into a 2 MB buffer with a packet of slack, the
  gadget's 16 KB bursts stripped, and the PADDED byte on what it sends. The
  server is the only USB host now; this is the framing it is held to."""
  from jetlink import protocol as P
  from jetlink.transport.base import StreamTransport

  class UsbHost(StreamTransport):
    packet_size = P.USB_MAX_PACKET
    read_chunk = P.USB_READ_CHUNK
    rx_align = P.GADGET_TX_ALIGN
    read_slack = P.USB_MAX_PACKET

    def __init__(self):
      super().__init__(rx_size=2 << 20)

  return UsbHost


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

  host = _usb_host()
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
    'usb_host': (_memory(host), _memory(FfsTransport, FfsTransport)),
    'usb_gadget': (_memory(FfsTransport, FfsTransport), _memory(host)),
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
    'version': P.VERSION, 'envelope_version': P.ENVELOPE_VERSION, 'envelope': sorted(int(m) for m in P.ENVELOPE),
    'headers': headers, 'infer_req': infer_req, 'infer_resp': infer_resp,
    'messages': messages, 'streams': streams,
  }))


def _new(cls):
  """An instance without the real constructor's side effects, where it has any."""
  from jetlink.transport.base import StreamTransport
  from jetlink.transport.ffs import FfsTransport
  from jetlink.transport.tcp import TcpTransport
  if not issubclass(cls, (FfsTransport, TcpTransport)):
    return cls()   # the USB host: opens nothing
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
# a new client: nothing fed back on its first frame, the queues kept
STAGING_HELLO_BEFORE = 5
# an output with a NaN in it: its hidden state is not fed back
STAGING_NOT_FINITE = 3
STAGING_INPUTS = ('img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t')


def _protocol2_staging(spec, frames) -> list[dict[str, np.ndarray]]:
  """What protocol 2 staged for the same frames: modeld sent the hidden state
  back as prev_feat, zero for a new ModelState (so on its first frame, the one
  that resets) and for a new client, and replaced only after a frame that
  reached it, which a non-finite one never did. openpilot's rolling queues in
  numpy, not PolicyQueues, so the fixture checks one against the other."""
  fs = spec.frame_skip
  hidden = slice(*spec.hidden_range)
  queues = {'img': np.zeros(spec.img_buf_shape, np.float16), 'big_img': np.zeros(spec.img_buf_shape, np.float16),
            'feat': np.zeros(spec.feat_q_shape, np.float16), 'desire': np.zeros(spec.desire_q_shape, np.float16)}
  prev_feat = np.zeros(spec.prev_feat_shape, np.float32)

  def shift(name, row):
    buf = queues[name]
    buf[:-1] = buf[1:].copy()
    buf[-1] = np.asarray(row).reshape(buf.shape[1:]).astype(np.float16)

  staged = []
  layout = spec.packed_layout
  for f, (warped, packed, output) in enumerate(frames):
    if f == STAGING_RESET_BEFORE:
      for buf in queues.values():
        buf[...] = 0
      prev_feat[...] = 0
    if f == STAGING_HELLO_BEFORE:
      prev_feat[...] = 0
    shift('img', warped[0])
    shift('big_img', warped[1])
    shift('desire', packed[layout['desire'][0]])
    shift('feat', prev_feat)
    des = queues['desire']
    # copies: the next frame shifts the queues under any view
    staged.append({
      'img': queues['img'][::fs].reshape(spec.input_shapes['img']).copy(),
      'big_img': queues['big_img'][::fs].reshape(spec.input_shapes['big_img']).copy(),
      'features_buffer': queues['feat'][::fs].reshape(spec.input_shapes['features_buffer']).copy(),
      'desire_pulse': des.reshape(-1, fs, *des.shape[1:]).max(axis=1).reshape(spec.input_shapes['desire_pulse']),
      'traffic_convention': packed[layout['traffic_convention'][0]].astype(np.float16),
      'action_t': packed[layout['action_t'][0]].astype(np.float16),
    })
    if np.all(np.isfinite(output)):
      prev_feat.reshape(-1)[...] = output[hidden]
  return staged


def staging(root: Path) -> None:

  from jetlink.queues import PolicyQueues
  from jetlink.spec import DRIVING_OUTPUT, spec_from_onnx
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
      frames, staged, sent = bytearray(), bytearray(), []
      for f in range(STAGING_FRAMES):
        warped = rng.integers(0, 256, spec.warped_shape, dtype=np.uint8)
        packed = (rng.standard_normal(spec.packed_nelem) * 2.0).astype(np.float32)
        # what the engine returned for this frame, which the next feeds back
        output = (rng.standard_normal(spec.output_nelem) * 2.0).astype(np.float32)
        if f == STAGING_NOT_FINITE:
          output[spec.hidden_range[0]] = np.nan
        if f == STAGING_RESET_BEFORE:
          queues.reset()
        if f == STAGING_HELLO_BEFORE:
          queues.new_client()
        feed = queues.step(warped, packed)
        if np.all(np.isfinite(output)):
          queues.after_run({DRIVING_OUTPUT: output}, feed)
        frames += warped.tobytes() + packed.tobytes() + output.tobytes()
        sent.append((warped, packed, output))
        for name in STAGING_INPUTS:
          assert feed[name].dtype == np.float16, name
          staged += feed[name].tobytes()
      want = b''.join(frame[name].tobytes() for frame in _protocol2_staging(spec, sent) for name in STAGING_INPUTS)
      assert bytes(staged) == want, f'frame_skip {skip}: protocol 3 stages what protocol 2 did not'
      stem = f'staging.fs{skip}'
      (out / f'{stem}.spec.json').write_text(dump(spec.to_dict()))
      (out / f'{stem}.frames.bin').write_bytes(bytes(frames))
      (out / f'{stem}.staged.bin').write_bytes(bytes(staged))
      cases.append({'frame_skip': skip, 'spec': f'{stem}.spec.json', 'frames': f'{stem}.frames.bin',
                    'staged': f'{stem}.staged.bin',
                    'inputs': [{'name': n, 'shape': list(spec.input_shapes[n])} for n in STAGING_INPUTS]})
  (out / 'staging.json').write_text(dump({
    'frames': STAGING_FRAMES, 'reset_before': STAGING_RESET_BEFORE, 'hello_before': STAGING_HELLO_BEFORE,
    'dtype': 'float16',
    'frame_layout': 'warped uint8, packed float32, then the driving output float32 the frame returned, per frame',
    'feedback': 'the output\'s hidden_state is fed back after the frame when every value is finite',
    'staged_layout': 'each input in `inputs` order, float16, per frame',
    'cases': cases,
  }))


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
  from jetlink.registry.catalog import is_ref, is_sha256, merge_catalogs, parse_catalog
  from jetlink.registry.lfs import parse_pointer_text

  # frozen: the Python server's registry wrote it, and nothing here makes it now
  frozen = json.loads((ROOT / REGISTRY).read_text())['cache']
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

  out.write_text(dump({
    'pointers': pointers, 'identities': identities, 'parses': parses, 'merges': merges, 'cache': frozen,
  }))


# -----------------------------------------------------------------------------

PARTS = {'wire': wire, 'staging': staging, 'registry': registry}


def generate(root: Path = ROOT, parts=tuple(PARTS)) -> None:
  for name in parts:
    PARTS[name](root)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="The conformance fixtures: what the comma's Python does, for the Swift to match.")
  parser.add_argument('--root', type=Path, default=ROOT, help='write the tree under this directory (default: this checkout)')
  parser.add_argument('parts', nargs='*', metavar='PART', help=f"some of {', '.join(PARTS)} (default: all)")
  args = parser.parse_args()
  unknown = set(args.parts) - set(PARTS)
  if unknown:
    parser.error(f"no such fixtures: {', '.join(sorted(unknown))}")
  generate(args.root, args.parts or tuple(PARTS))
