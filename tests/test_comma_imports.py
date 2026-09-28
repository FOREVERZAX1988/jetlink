"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma imports, and that it loads nothing else.

openpilot runs jetlink from a checkout on the comma (only jetlink/ and
scripts/comma/ ship), imports the names below and patches some of them in its
tests. Each module is imported in a fresh interpreter, so one module's imports
cannot hide another's: none may load the server or a package beyond numpy and
the standard library.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# What openpilot imports from each module (danger-unstable, 2026-09-28)
FORK_IMPORTS = {
  'jetlink.client': ('EngineMissing', 'FRAME_TIMEOUT', 'JetlinkClient'),
  'jetlink.spec': ('ModelSpec', 'sha256_file'),
  'jetlink.queues': ('PolicyQueues',),
  'jetlink.comma': ('gadget', 'lending', 'owner', 'port'),
  'jetlink.comma.owner': ('WATCHED',),
  'jetlink.registry.catalog': ('DEFAULT_BIG_MODEL_REF', 'NetworkError', 'RegistryError', 'fetch_catalogs', 'merge_catalogs'),
  'jetlink.registry.lfs': ('LFS_ENDPOINTS', 'POINTER_URL', 'Pointer', 'fetch_pointer', 'lfs_download', 'lfs_resolve'),
  'jetlink.transport.tcp': ('CABLE_ADDRESS', 'TcpTransport'),
}
# and the rest of what runs there
COMMA_MODULES = (*FORK_IMPORTS, 'jetlink.protocol', 'jetlink.comma.gadget', 'jetlink.comma.lending', 'jetlink.comma.port',
                 'jetlink.comma.root', 'jetlink.transport.ffs')

PROBE = '''
import importlib, json, sys
before = set(sys.modules)
module = importlib.import_module(sys.argv[1])

def resolves(name):
  if hasattr(module, name):
    return True
  try:   # `from package import submodule`
    importlib.import_module(f'{sys.argv[1]}.{name}')
    return True
  except ImportError:
    return False

missing = [n for n in sys.argv[2:] if not resolves(n)]
loaded = set(sys.modules) - before
print(json.dumps({
  'missing': missing,
  'server': sorted(n for n in loaded if n == 'jetlink.server' or n.startswith('jetlink.server.')),
  'foreign': sorted({n.split('.')[0] for n in loaded} - set(sys.stdlib_module_names) - {'jetlink', 'numpy'}),
}))
'''


@pytest.mark.parametrize('module', COMMA_MODULES)
def test_a_comma_module_loads_only_what_the_comma_has(module):
  # cwd first on the path: this checkout's jetlink, whatever is installed
  run = subprocess.run([sys.executable, '-c', PROBE, module, *FORK_IMPORTS.get(module, ())],
                       cwd=ROOT, capture_output=True, text=True, check=True)
  found = json.loads(run.stdout)
  assert found['missing'] == [], f"openpilot imports {found['missing']} from {module}"
  assert found['server'] == [], f"{module} loads {found['server']}"
  assert found['foreign'] == [], f"{module} needs {found['foreign']}; the comma has numpy and the standard library"

