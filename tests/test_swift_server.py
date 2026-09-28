"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's client against the Swift server, live.

The comma is the peer that matters, so the contract is checked from its side:
jetlink.client over TCP against a real `jetlink-server` on onnxruntime's CPU
provider, serving tests/tiny_model.py's graphs (the committed copies the Swift
golden frames use). Outputs are held to Python's own staging (jetlink.queues)
and a numpy run of the graph.

The binary comes from JETLINK_SERVER_BIN, else the newest SwiftPM build in
JetlinkKit/.build; JETLINK_SERVER_BUILD=1 builds it first. Without one this
skips.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import EngineMissing, JetlinkClient
from jetlink.queues import PolicyQueues
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError, LinkTimeout
from jetlink.transport.tcp import TcpTransport
from tests import tiny_model

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'JetlinkKit'
FIXTURES = PACKAGE / 'Tests' / 'JetlinkServerTests' / 'Fixtures'
QUEUED, STATEFUL = FIXTURES / 'tiny_queued.onnx', FIXTURES / 'tiny_stateful.onnx'
# In the cache, it makes a SHUTDOWN_REQ log instead of powering the host off
DRY_RUN = 'poweroff-dry-run'
HELLO_KEYS = {'protocol', 'backend', 'runtime_version', 'device', 'engine_state', 'loaded', 'frames_served', 'cached_models',
              'telemetry', 'sleep_after'}


def _binary() -> Path | None:
  named = os.environ.get('JETLINK_SERVER_BIN')
  if named:
    if not os.access(named, os.X_OK):
      raise RuntimeError(f'JETLINK_SERVER_BIN={named} is not an executable')
    return Path(named)
  if os.environ.get('JETLINK_SERVER_BUILD') == '1':
    subprocess.run(['swift', 'build', '--package-path', str(PACKAGE), '--product', 'jetlink-server'], check=True)
  built = [p for p in (PACKAGE / '.build' / c / 'jetlink-server' for c in ('debug', 'release')) if os.access(p, os.X_OK)]
  return max(built, key=lambda p: p.stat().st_mtime, default=None)


BIN = _binary()
pytestmark = pytest.mark.skipif(BIN is None, reason='no jetlink-server: set JETLINK_SERVER_BIN, or build the jetlink-server product')


def spec_of(path: Path) -> ModelSpec:
  """The spec Python derives, as make_server_fixtures.py recorded it: no onnx
  package needed here, so this runs anywhere numpy does."""
  return ModelSpec.from_dict(json.loads(path.with_suffix('.spec.json').read_text()))


@dataclass
class Server:
  proc: subprocess.Popen
  port: int
  cache: Path
  log: Path
  returncode: int | None = None
  provisioned: dict = field(default_factory=dict)

  def connect(self) -> JetlinkClient:
    return JetlinkClient(TcpTransport.connect('127.0.0.1', self.port), deadline=10.0, name='test_swift_server')

  def tail(self) -> str:
    return '\n'.join(self.log.read_text(errors='replace').splitlines()[-40:])

  def provision(self, path: Path) -> tuple[ModelSpec, list[str]]:
    """Upload and build `path` once, as a provisioning run does; (served spec, stages)."""
    if path not in self.provisioned:
      spec, stages = spec_of(path), []
      client = self.connect()
      try:
        served = client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=path, progress=lambda s, f, m: stages.append(s),
                                      build_timeout=120.0)
      finally:
        client.close()
      self.provisioned[path] = (served, stages)
    return self.provisioned[path]


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    return s.getsockname()[1]


def _wait_listening(server: Server, timeout: float = 30.0) -> None:
  end = time.monotonic() + timeout
  while time.monotonic() < end:
    if server.proc.poll() is not None:
      raise AssertionError(f'jetlink-server exited with {server.proc.returncode}:\n{server.tail()}')
    try:
      socket.create_connection(('127.0.0.1', server.port), timeout=0.2).close()
      return
    except OSError:
      time.sleep(0.1)
  raise AssertionError(f'jetlink-server never listened on {server.port}:\n{server.tail()}')


@contextmanager
def running(tmp: Path, *extra: str):
  cache = tmp / 'cache'
  cache.mkdir(parents=True)
  (cache / DRY_RUN).touch()   # before the server starts: a shutdown test must never power a machine off
  # and a stand-in on PATH, so a server that ignored the file would call this
  fake = tmp / 'bin'
  fake.mkdir()
  (fake / 'systemctl').write_text(f'#!/bin/sh\necho "$@" >> "{tmp / "systemctl.called"}"\n')
  (fake / 'systemctl').chmod(0o755)
  env = dict(os.environ, PATH=f'{fake}{os.pathsep}{os.environ.get("PATH", "")}', JETLINK_CACHE=str(cache))
  log = tmp / 'server.log'
  port = _free_port()
  with open(log, 'wb') as out:
    proc = subprocess.Popen([str(BIN), 'serve', '--backend', 'ort', '--device', 'cpu', '--listen', '--host', '127.0.0.1',
                             '--port', str(port), '--cache', str(cache), '--log-level', 'debug', *extra],
                            stdout=out, stderr=subprocess.STDOUT, env=env)
  server = Server(proc, port, cache, log)
  try:
    _wait_listening(server)
    yield server
  finally:
    proc.send_signal(signal.SIGTERM)
    try:
      server.returncode = proc.wait(10.0)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()


@pytest.fixture(scope='module')
def server(tmp_path_factory):
  with running(tmp_path_factory.mktemp('swift-server')) as s:
    yield s


@pytest.fixture(scope='module')
def bare(tmp_path_factory):
  """A server that has never seen a model."""
  with running(tmp_path_factory.mktemp('swift-bare'), '--no-preload') as s:
    yield s


@contextmanager
def ready(server: Server, path: Path):
  server.provision(path)
  spec = spec_of(path)
  client = server.connect()
  try:
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=60.0)
    yield client
  finally:
    client.close()


@pytest.fixture
def queued(server):
  with ready(server, QUEUED) as client:
    yield client


@pytest.fixture
def stateful(server):
  with ready(server, STATEFUL) as client:
    yield client


def close_enough(got: np.ndarray, want: np.ndarray, atol: float) -> None:
  assert np.all(np.isfinite(got))
  assert np.corrcoef(got, want)[0, 1] > 0.999
  np.testing.assert_allclose(got, want, atol=atol, rtol=atol)


# -- the queued graph: the server keeps the history ---------------------------

def queued_frames(n: int, seed: int = 0, prev_feat: dict[int, float] | None = None):
  """(warped, packed) per frame; prev_feat {frame: value} or random.

  Pixels up to 15, not 255: at full range the image terms are 80 and fp16's
  rounding of them (0.1) is as big as a scalar the queues dropped (0.2).
  """
  spec, rng = spec_of(QUEUED), np.random.default_rng(seed)
  frames = []
  for i in range(n):
    warped = rng.integers(0, 16, spec.warped_shape, dtype=np.uint8)
    packed = (rng.standard_normal(spec.packed_nelem) * 0.5).astype(np.float32)
    if prev_feat is not None:
      packed[spec.packed_layout['prev_feat'][0]] = prev_feat.get(i, 0.0)
    frames.append((warped, packed))
  return frames


def queued_reference(frames, resets=(0,)) -> list[np.ndarray]:
  """Python's staging (the reference the Swift is locked to) into a numpy run."""
  queues, outs = PolicyQueues(spec_of(QUEUED)), []
  for i, (warped, packed) in enumerate(frames):
    if i in resets:
      queues.reset()
    outs.append(tiny_model.reference(queues.step(warped, packed)))
  return outs


def run(client, frames, resets=(0,)) -> list[np.ndarray]:
  return [client.infer(w, p, frame_id=i + 1, reset=i in resets) for i, (w, p) in enumerate(frames)]


def test_upload_build_and_load_through_the_link(server):
  spec, stages = server.provision(QUEUED)
  assert spec.to_dict() == spec_of(QUEUED).to_dict()   # the spec the server derives is Python's
  assert {'upload', 'load'} <= set(stages)
  # the sidecar carries the spec, so a later load needs neither the ONNX nor a parser
  sidecars = [json.loads(p.read_text()) for p in (server.cache / 'engines').glob(f'{spec.sha256[:16]}.*.json')]
  assert [m['backend'] for m in sidecars] == ['ort']
  assert ModelSpec.from_dict(sidecars[0]['spec']).to_dict() == spec.to_dict()
  assert (server.cache / 'models' / f'{spec.sha256[:16]}.onnx').stat().st_size == spec.nbytes


def test_infer_round_trip(queued):
  frames = queued_frames(10)
  outs = run(queued, frames)
  assert all(o.shape == (queued.spec.output_nelem,) and o.dtype == np.float32 for o in outs)
  for got, want in zip(outs, queued_reference(frames), strict=True):
    close_enough(got, want, atol=0.02)
  gpu_us, queue_us, total_us = queued.last_timings
  assert gpu_us >= 0 and queue_us >= 0 and total_us >= gpu_us


def test_hidden_state_feeds_back_into_the_queues(queued):
  """prev_feat goes out and comes back as features_buffer frames later."""
  frames = queued_frames(12, seed=1, prev_feat={1: 3.0})
  outs = run(queued, frames)
  for got, want in zip(outs, queued_reference(frames), strict=True):
    close_enough(got, want, atol=0.02)
  without = queued_reference(queued_frames(12, seed=1, prev_feat={}))
  assert max(np.abs(o - w).max() for o, w in zip(outs, without, strict=True)) > 0.5, "prev_feat never reached the engine"


def test_queues_reset_flag_clears_history(queued):
  frames = queued_frames(9, seed=2)
  for got, want in zip(run(queued, frames, resets=(0, 6)), queued_reference(frames, resets=(0, 6)), strict=True):
    close_enough(got, want, atol=0.02)


def test_nonfinite_output_is_reported_not_returned(queued):
  warped, packed = queued_frames(1)[0]
  packed[queued.spec.packed_layout['traffic_convention'][0]] = np.inf
  with pytest.raises(LinkError, match='NOT_FINITE'):
    queued.infer(warped, packed, reset=True)


def test_telemetry_piggybacks_only_when_asked(queued):
  warped, packed = queued_frames(1)[0]
  queued.infer(warped, packed, reset=True, want_state=False)
  assert queued.last_state is None
  queued.infer(warped, packed, want_state=True)
  assert isinstance(queued.last_state, dict)   # {} where the host has no sensors: never zeros


def test_ping_and_state_requests(queued):
  assert queued.ping(timeout=5) < 5.0
  state = queued.state(timeout=5)
  assert state['engine_state'] == 'ready'
  assert 'frames_served' in state
  assert queued.hello(timeout=5)['protocol'] == P.VERSION


def test_frame_deadline_includes_time_spent_sending(queued, monkeypatch):
  send = queued.t.send

  def delayed_send(*args, **kwargs):
    send(*args, **kwargs)
    time.sleep(0.08)

  monkeypatch.setattr(queued.t, 'send', delayed_send)
  queued.deadline = 0.05
  warped, packed = queued_frames(1)[0]
  with pytest.raises(LinkError, match='link abandoned'):
    queued.infer(warped, packed)
  assert queued.dead


def test_wrong_sized_request_is_rejected(queued):
  seq = queued._next_seq()
  queued.t.send(P.Msg.INFER_REQ, seq, (P.pack_infer_req(1, 0), b'\x00' * 1000))
  _, status, _, _, _ = P.unpack_infer_resp(queued._expect(P.Msg.INFER_RESP, seq, 5.0).payload)
  assert status == P.Status.BAD_SHAPE


def _may_power_off() -> bool:
  """A booted systemd host outside CI: a server that got the dry run wrong could take it down."""
  return Path('/run/systemd/system').exists() and not (os.environ.get('CI') or os.environ.get('JETLINK_TEST_SHUTDOWN'))


@pytest.mark.skipif(_may_power_off(), reason='a real systemd host; JETLINK_TEST_SHUTDOWN=1 runs it anyway')
# TODO(WS-C): drop with LinuxHost's poweroff hook, until which Linux replies ok:false
@pytest.mark.xfail(sys.platform.startswith('linux') and not os.environ.get('JETLINK_LINUX_POWEROFF'), reason='no Linux poweroff hook yet',
                   raises=AssertionError, strict=True)
def test_shutdown_replies_and_a_dry_run_stays_up(server, queued):
  assert (server.cache / DRY_RUN).exists()
  resp = queued.shutdown('car battery', timeout=5)
  assert isinstance(resp.get('ok'), bool) and isinstance(resp.get('detail'), str)
  if sys.platform.startswith('linux'):
    assert resp['ok'] is True
  time.sleep(0.5)
  assert server.proc.poll() is None, server.tail()
  assert not (server.cache.parent / 'systemctl.called').exists(), 'the dry run called systemctl'
  assert queued.ping(timeout=5) < 5.0   # the host is what goes down, not the session


# -- the hello ------------------------------------------------------------------

def test_the_hello_carries_the_link_and_answers_with_what_the_comma_reads(server, monkeypatch):
  client = server.connect()
  try:
    monkeypatch.setattr(client.t, 'link_info', lambda: {'kind': 'cable', 'usb_speed': 'high-speed'})
    hello = client.hello(timeout=5)
  finally:
    client.close()
  assert HELLO_KEYS <= set(hello), f'missing {HELLO_KEYS - set(hello)}'
  assert hello['protocol'] == P.VERSION
  assert hello['backend'] == 'ort' and 'trt_version' not in hello
  # a missing sleep_after reads as 1.0 on the comma, which then lets the gadget go
  assert isinstance(hello['sleep_after'], (int, float)) and hello['sleep_after'] >= 0


def test_a_hello_restarts_the_seqs_and_replays_are_dropped(server):
  client = server.connect()
  try:
    client.seq = 5000
    client.ping(timeout=5)
    client.seq = 0                  # a new process on the same link starts at 1
    client.hello(timeout=5)         # seq 1: answered, whatever came before
    client.t.send(P.Msg.PING, 1)    # the hello's own seq is a replay
    with pytest.raises(LinkTimeout):
      client._expect(P.Msg.PONG, 1, 0.5)
    assert client.ping(timeout=5) < 5.0
  finally:
    client.close()


# -- the stateful graph: the engine keeps the history ------------------------------

def packed_for(frame: dict) -> np.ndarray:
  return np.concatenate([frame['desire'].ravel(), frame['traffic_convention'].ravel(), frame['action_t'].ravel()]).astype(np.float32)


def stateful_reference(frames: list[dict]) -> list[np.ndarray]:
  state, outs = tiny_model.empty_state(), []
  for f in frames:
    out, state = tiny_model.stateful_step(state, **f)
    outs.append(out)
  return outs


def test_the_state_carries_from_frame_to_frame(stateful):
  frames = tiny_model.stateful_frames(8, seed=2)
  want = stateful_reference(frames[:3]) + stateful_reference(frames[3:])
  for i, (f, w) in enumerate(zip(frames, want, strict=True)):
    out = stateful.infer(f['new_img'], packed_for(f), frame_id=i + 1, reset=i in (0, 3))
    assert out.shape == (stateful.spec.output_nelem,)
    close_enough(out, w, atol=1e-4)   # float32 throughout: a frame off in the queue is far outside this


def test_a_request_is_the_frame_and_twelve_floats(stateful):
  spec = stateful.spec
  assert spec.infer_req_nbytes == P.INFER_REQ_SIZE + spec.warped_nbytes + 12 * 4
  seq = stateful.infer_begin(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32), 1, reset=True)
  assert stateful.infer_end(seq).shape == (spec.output_nelem,)


# -- a server with nothing loaded -------------------------------------------------

def test_not_ready_is_reported_rather_than_crashing(bare):
  client = bare.connect()
  client.spec = spec_of(QUEUED)
  try:
    with pytest.raises(LinkError, match='NOT_READY'):
      client.infer(np.zeros(client.spec.warped_shape, np.uint8), np.zeros(client.spec.packed_nelem, np.float32))
  finally:
    client.close()


def test_the_ping_does_not_need_an_engine(bare):
  client = bare.connect()
  try:
    assert client.ping(timeout=5) < 5.0
    assert client.state(timeout=5)['engine_state'] == 'none'
    assert client.hello(timeout=5)['protocol'] == P.VERSION
  finally:
    client.close()


def test_a_missing_engine_with_nothing_to_upload_is_engine_missing(bare):
  # modeld never carries the ONNX: this is what sends the comma back to provisioning
  spec = spec_of(STATEFUL)
  client = bare.connect()
  try:
    with pytest.raises(EngineMissing):
      client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  finally:
    client.close()


def test_sigterm_stops_it_cleanly(tmp_path):
  with running(tmp_path) as s:
    client = s.connect()
    assert client.ping(timeout=5) < 5.0
    client.close()
  assert s.returncode == 0, s.tail()
