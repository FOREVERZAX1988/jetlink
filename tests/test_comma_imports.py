"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma imports, and that it loads nothing else.

openpilot runs jetlink from a checkout on the comma (only jetlink/ and
scripts/comma/ ship), imports the names below and patches some of them in its
tests. Each module is imported in a fresh interpreter, so one module's imports
cannot hide another's: none may load a package beyond numpy and the standard
library.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jetlink import protocol as P

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
# What openpilot calls on a client and reads off one (helpers.py, backend.py,
# joining.py, provision.py, model_state.py)
CLIENT_CALLS = ('open_socket', 'open_borrowed_ffs', 'open_ffs', 'hello', 'ensure_engine', 'infer_begin', 'infer_end',
                'shutdown', 'rebind', 'close')
CLIENT_FIELDS = ('t', 'dead', 'deadline', 'last_timings', 'last_state')
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
  assert found['foreign'] == [], f"{module} needs {found['foreign']}; the comma has numpy and the standard library"


def test_the_client_has_what_openpilot_calls():
  from jetlink.client import JetlinkClient
  assert [n for n in CLIENT_CALLS if not callable(getattr(JetlinkClient, n, None))] == []
  client = JetlinkClient(SimpleNamespace())
  assert [n for n in CLIENT_FIELDS if not hasattr(client, n)] == []


def test_the_gadget_presents_what_a_host_looks_for():
  # The server finds the comma by these; nothing on the comma reads protocol.py's copy.
  script = (ROOT / 'scripts' / 'comma' / 'jetlink-root.sh').read_text()
  ids = {k: int(v, 16) for k, v in re.findall(r'^(VID|PID)=(0x[0-9a-fA-F]+)', script, re.M)}
  assert ids == {'VID': P.USB_VID, 'PID': P.USB_PID}
