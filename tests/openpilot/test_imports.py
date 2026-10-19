"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink imports no openpilot, and the owner's path stays small.

Three ways, since each misses something the others see. A fresh interpreter
that refuses openpilot's packages proves each module imports without them,
even where openpilot is importable (the fork's venv puts its checkout on every
path). The owner's path is held to the standard library and jetlink's light
half, because it stays resident for the whole drive. And a scan of the source
finds an import tucked inside a function, which no import runs.

stock.py is the one exception (the plan's rule 1): it imports openpilot, but
only inside a method, and its top level stays on the owner's path.
"""
from __future__ import annotations

import ast
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import jetlink.openpilot
from jetlink.openpilot.testing import HEAVY_JETLINK

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'jetlink'
# openpilot and what comes with it; tinygrad is the fork's, and jetlink CI has none
BLOCKED = ('openpilot', 'cereal', 'msgq', 'capnp', 'zmq', 'opendbc', 'tinygrad')
MODULES = sorted({'jetlink.openpilot', *(m.name for m in pkgutil.walk_packages(jetlink.openpilot.__path__,
                                                                              'jetlink.openpilot.'))})
# what stays resident for the whole drive, at about 10 MB: the gadget owner and
# everything it opens. The provisioning run and modeld are the heavy half
OWNER_PATH = ('jetlink.openpilot.owner', 'jetlink.openpilot', 'jetlink.openpilot.interface', 'jetlink.openpilot.settings',
              'jetlink.openpilot.stock', 'jetlink.comma.owner', 'jetlink.comma.gadget', 'jetlink.comma.lending', 'jetlink.comma.port',
              'jetlink.comma.root', 'jetlink.transport.ffs')
# a fork's hook module imports the hooks, and the owner imports that module
HOOKS = ('jetlink.openpilot.hooks',)
# the base adapter, which may import openpilot inside a method and nowhere else
STOCK = PACKAGE / 'openpilot' / 'stock.py'

BLOCKER = '''
import importlib, json, sys
from importlib.abc import MetaPathFinder

BLOCKED = set(sys.argv[2].split(','))

class Refuse(MetaPathFinder):
  def find_spec(self, name, path=None, target=None):
    if name.split('.')[0] in BLOCKED:
      raise ImportError(f'jetlink must not import {name}')
    return None

sys.meta_path.insert(0, Refuse())
importlib.import_module(sys.argv[1])
print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & BLOCKED)))
'''

LOADS = '''
import importlib, json, sys
before = set(sys.modules)
for name in sys.argv[1:]:
  importlib.import_module(name)
print(json.dumps(sorted(set(sys.modules) - before)))
'''


def fresh(code: str, *args: str) -> subprocess.CompletedProcess:
  # cwd first on the path: this checkout's jetlink, whatever is installed
  return subprocess.run([sys.executable, '-c', code, *args], cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_the_package_is_walked():
  # the parametrisation below is only as good as the walk
  assert {'jetlink.openpilot', 'jetlink.openpilot.interface', 'jetlink.openpilot.settings'} <= set(MODULES)


@pytest.mark.parametrize('module', MODULES)
def test_a_module_imports_without_openpilot(module):
  run = fresh(BLOCKER, module, ','.join(BLOCKED))
  assert run.returncode == 0, run.stderr
  assert json.loads(run.stdout) == [], f"{module} loaded {run.stdout}"


@pytest.mark.parametrize('path', [OWNER_PATH, HOOKS], ids=['owner', 'hooks'])
def test_the_owner_path_is_the_standard_library_and_jetlink(path):
  run = fresh(LOADS, *path)
  assert run.returncode == 0, run.stderr
  loaded = json.loads(run.stdout)
  foreign = sorted({m.split('.')[0] for m in loaded} - set(sys.stdlib_module_names) - {'jetlink'})
  assert foreign == [], f"the owner's path loads {foreign}; it stays resident for the whole drive"
  heavy = sorted(m for m in loaded if m.startswith(HEAVY_JETLINK))
  assert heavy == [], f"the owner's path loads jetlink's heavy half: {heavy}"
  # the kit is the fork's tests', never a drive's
  assert 'jetlink.openpilot.testing' not in loaded


HEAP = '''
import importlib, sys, tracemalloc
tracemalloc.start()
importlib.import_module(sys.argv[1])
importlib.import_module('jetlink.transport.ffs')
print(tracemalloc.get_traced_memory()[0])
'''


def test_the_owner_path_costs_under_a_megabyte_more_than_the_comma_owner():
  # a proxy for the owner's PSS on the comma: the Python heap its imports
  # allocate in a fresh interpreter, over the comma layer's own owner
  # (about 0.6 MB, typing 0.4 of it)
  def heap(module: str) -> int:
    run = subprocess.run([sys.executable, '-S', '-c', HEAP, module], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    return int(run.stdout)
  extra = heap('jetlink.openpilot.owner') - heap('jetlink.comma.owner')
  assert extra < 1 << 20, f"the owner's path grew {extra >> 10} KB over the comma owner's"


def _imports(path: Path) -> list[tuple[str, int, bool]]:
  """(module, line, at module level) for every import in a file."""
  tree = ast.parse(path.read_text(), filename=str(path))
  top = set(map(id, tree.body))
  found = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      found += [(alias.name, node.lineno, id(node) in top) for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
      found.append((node.module, node.lineno, id(node) in top))
  return found


def _in_a_function(path: Path) -> set[int]:
  """The lines of every import nested in a function: a module-level try or if is not."""
  lines = set()
  for fn in ast.walk(ast.parse(path.read_text(), filename=str(path))):
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
      lines |= {node.lineno for node in ast.walk(fn) if isinstance(node, (ast.Import, ast.ImportFrom))}
  return lines


def test_no_source_in_jetlink_names_openpilot():
  offenders = []
  for path in sorted(PACKAGE.rglob('*.py')):
    lazy = _in_a_function(path) if path == STOCK else set()
    for module, line, _ in _imports(path):
      if module.split('.')[0] == 'openpilot' and line in lazy:
        continue
      if module.split('.')[0] in BLOCKED and module.split('.')[0] != 'tinygrad':
        offenders.append(f"{path.relative_to(ROOT)}:{line} imports {module}")
  assert offenders == []


def test_tinygrad_is_only_ever_imported_where_it_is_used():
  # it is the fork's, and only modeld has it; a module-level
  # import would fail every other process and jetlink's CI
  offenders = [f"{path.relative_to(ROOT)}:{line}"
               for path in sorted(PACKAGE.rglob('*.py'))
               for module, line, top in _imports(path) if module.split('.')[0] == 'tinygrad' and top]
  assert offenders == []


def test_stock_names_openpilot_only_inside_a_method():
  # the one file the guard above lets through, and only this far
  lazy = [module for module, line, _ in _imports(STOCK) if module.split('.')[0] == 'openpilot' and line in _in_a_function(STOCK)]
  assert lazy, "stock.py imports no openpilot: drop its exception above"
  assert 'jetlink.openpilot.stock' in MODULES   # so the blocked import above runs it
