"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The Python half of the conformance suite (docs/conformance.md).

The Swift tests hold the Swift to fixtures the Python wrote. This file holds
the Python to the same fixtures: every generator runs again into a temporary
directory and must write the committed bytes. So a Python change that moves
an output fails here, and regenerating to make it pass fails the Swift until
the Swift moves too.

The generators run in a fresh interpreter each, as they do from the command
line: this process may not load onnxruntime (test_ort_backend), and a graph
built from another test module's helpers must not see what earlier tests did
to that module's state.

Two kinds of file depend on the tools as well as on this code, and are
compared only where the tools match what made them: anything onnx serialises
or shape-infers, and onnxruntime's CPU outputs, which are compared on Apple
arm64 under the onnxruntime the Swift package links. Both releases come from
JetlinkKit/Scripts/fixture-pins.txt.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'JetlinkKit' / 'Scripts'


def _script(name: str):
  """A generator, imported, for where it writes by default. Only the ones
  whose top level is light: numpy at most."""
  spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


# where each generator writes by default, relative to the checkout
CONFORMANCE_SCRIPT = _script('make_conformance_fixtures')
CONFORMANCE = CONFORMANCE_SCRIPT.SERVER
CONTROL = CONFORMANCE_SCRIPT.CONTROL
REGISTRY = CONFORMANCE_SCRIPT.REGISTRY
PINS_SCRIPT = _script('make_pins')
PINNED = PINS_SCRIPT.OUT.relative_to(ROOT)
# the releases the committed fixtures were made with
FIXTURE_PINS = PINS_SCRIPT.FIXTURE_PINS.relative_to(ROOT)
PINS = PINS_SCRIPT.fixture_pins()
FIXTURE_ONNX = PINS['onnx']
APPLE_ONNXRUNTIME = PINS['onnxruntime']
# Not imported: those two load onnxruntime, which this process must not
# (test_ort_backend checks).
SERVER_FIXTURES = Path('JetlinkKit/Tests/JetlinkServerTests/Fixtures')
ONNX_FIXTURES = Path('JetlinkKit/Tests/JetlinkONNXTests/Fixtures')


def _run(script: str, *args) -> None:
  env = {**os.environ, 'PYTHONPATH': str(ROOT)}
  result = subprocess.run([sys.executable, str(SCRIPTS / f'{script}.py'), *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
  assert result.returncode == 0, f'{script} failed:\n{result.stdout}\n{result.stderr}'


def _same_tree(made: Path, committed: Path, skip=lambda name: False) -> list[str]:
  """What differs between two directories, file by file, as sentences."""
  problems = []
  made_files = {p.relative_to(made) for p in made.rglob('*') if p.is_file()}
  committed_files = {p.relative_to(committed) for p in committed.rglob('*') if p.is_file()}
  for rel in sorted(made_files | committed_files):
    if skip(rel.name):
      continue
    if rel not in committed_files:
      problems.append(f'{rel} is made but not committed')
    elif rel not in made_files:
      problems.append(f'{rel} is committed but no longer made')
    elif (made / rel).read_bytes() != (committed / rel).read_bytes():
      problems.append(f'{rel} differs')
  return problems


def _version(package: str) -> str | None:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return None


def _onnx_matches() -> bool:
  return _version('onnx') == FIXTURE_ONNX


def _apple_runtime() -> str | None:
  """Why golden onnxruntime outputs cannot be compared here, or None if they can."""
  if sys.platform != 'darwin' or platform.machine() != 'arm64':
    return 'onnxruntime CPU outputs are pinned on Apple arm64'
  if _version('onnxruntime') != APPLE_ONNXRUNTIME:
    return f'onnxruntime {_version("onnxruntime")} is not the pinned {APPLE_ONNXRUNTIME}'
  return None


def test_pinned_swift_is_current(tmp_path):
  made = tmp_path / 'Pinned.swift'
  _run('make_pins', '--out', made)
  assert made.read_text() == (ROOT / PINNED).read_text(), \
    'JetlinkKit/Sources/JetlinkKit/Pinned.swift is stale: run JetlinkKit/Scripts/make_pins.py and commit it'


def test_the_swift_package_links_the_pinned_onnxruntime():
  package = (ROOT / 'JetlinkKit' / 'Package.swift').read_text()
  assert f'pod-archive-onnxruntime-c-{APPLE_ONNXRUNTIME}.zip' in package
  requirements = ROOT / 'macos' / 'Python' / 'requirements.txt'
  if requirements.exists():
    assert re.search(rf'^onnxruntime=={re.escape(APPLE_ONNXRUNTIME)}\b', requirements.read_text(), re.M)


def test_ci_regenerates_with_the_releases_the_fixtures_record():
  """CI's macOS test job installs the pins file, or it regenerates with
  something else and fails for no reason, or skips what it should compare."""
  ci = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text()
  assert f'-r {FIXTURE_PINS}' in ci, f'.github/workflows/ci.yml does not install {FIXTURE_PINS}'
  assert {'numpy', 'onnx', 'onnxruntime', 'protobuf'} <= set(PINS)


@pytest.mark.parametrize('part', list(CONFORMANCE_SCRIPT.PARTS))
def test_conformance_fixtures_are_what_the_python_makes(tmp_path, part):
  if part == 'staging' and not _onnx_matches():
    pytest.skip(f'the staging spec comes from onnx shape inference; fixtures made with onnx {FIXTURE_ONNX}')
  _run('make_conformance_fixtures', '--root', tmp_path, part)
  where = {'control': CONTROL, 'registry': REGISTRY}.get(part, CONFORMANCE)
  if where.suffix:
    made, committed = tmp_path / where, ROOT / where
    assert made.read_bytes() == committed.read_bytes(), f'{where} differs; see docs/conformance.md'
    return
  # wire, staging and stats share one directory: compare what this part writes
  problems = _same_tree(tmp_path / where, ROOT / where, skip=lambda name: not name.startswith(part))
  assert not problems, problems


def test_server_fixtures_are_what_the_python_makes(tmp_path):
  if not _onnx_matches():
    pytest.skip(f'the graphs are shape-inferred by onnx; fixtures made with onnx {FIXTURE_ONNX}')
  _run('make_server_fixtures', '--out', tmp_path)
  runtime = _apple_runtime()
  problems = _same_tree(tmp_path, ROOT / SERVER_FIXTURES, skip=lambda name: name.endswith('.expected.bin') and runtime is not None)
  # the conformance directory beside them is test_conformance_fixtures' business
  problems = [p for p in problems if not p.startswith('conformance')]
  assert not problems, problems


def test_onnx_fixtures_are_what_the_python_makes(tmp_path):
  if not _onnx_matches():
    pytest.skip(f'fixtures made with onnx {FIXTURE_ONNX}')
  _run('make_onnx_fixtures', '--out', tmp_path)
  problems = _same_tree(tmp_path, ROOT / ONNX_FIXTURES)
  assert not problems, problems
