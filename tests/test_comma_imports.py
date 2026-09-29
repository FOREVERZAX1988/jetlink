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

# What openpilot imports from each module, and the members it calls, reads or
# patches in its tests (danger-unstable 5be617a394, 2026-09-29). The fork keeps
# its own copy of the integration until it moves onto jetlink.openpilot, and
# runs against whatever jetlink_repo is pinned: a name dropped here breaks
# that build.
FORK_IMPORTS = {
  'jetlink.client': ('EngineMissing', 'FRAME_TIMEOUT', 'JetlinkClient'),
  'jetlink.spec': ('ModelSpec', 'sha256_file'),
  'jetlink.queues': ('PolicyQueues',),
  'jetlink.comma': ('gadget', 'lending', 'owner', 'port'),
  'jetlink.comma.gadget': (
    # production: the backend, helpers, joining, provision and spec_cache
    'enabled', 'link_mode', 'link_kind', 'link_peer', 'link_configured', 'gadget_error', 'host_attached',
    'port_has_host', 'dormant', 'wait_for_host', 'request_shutdown', 'pending_shutdown', 'finish_shutdown',
    'far_end_sleeps', 'set_logger', 'SHUTDOWN_REQUEST', 'STATE', 'CC_ORIENTATION', 'P_SPEC',
    # its tests, which call or patch these
    'set_dormant', 'owner_state', 'note_lender_error', 'udc_state', 'ios', 'params_dir', 'raw_param',
    'P_LINK', 'P_BIG_MODEL', 'P_OFFROAD', 'LINK_MODES', 'LINK', 'DORMANT', 'GADGET_STATUS', 'LENDER_STATUS',
    'time',
  ),
  'jetlink.comma.lending': ('borrow', 'BORROW_TIMEOUT'),
  'jetlink.comma.owner': ('main', 'WATCHED'),
  'jetlink.comma.port': ('CHESTNUT_IDS',),
  'jetlink.registry.catalog': ('DEFAULT_BIG_MODEL_REF', 'NetworkError', 'RegistryError', 'fetch_catalogs', 'merge_catalogs'),
  'jetlink.registry.lfs': ('LFS_ENDPOINTS', 'POINTER_URL', 'Pointer', 'fetch_pointer', 'lfs_download', 'lfs_resolve'),
  'jetlink.transport.tcp': ('CABLE_ADDRESS', 'TcpTransport'),
}
# What openpilot calls on a client and reads off one (helpers.py, backend.py,
# joining.py, provision.py, model_state.py). open_loan is what jetlink.openpilot
# opens a loan with
CLIENT_CALLS = ('open_socket', 'open_borrowed_ffs', 'open_ffs', 'open_loan', 'hello', 'ensure_engine', 'infer_begin',
                'infer_end', 'ping', 'shutdown', 'rebind', 'close')
CLIENT_FIELDS = ('t', 'dead', 'deadline', 'last_timings', 'last_state')
# what it reads off a loan (backend._Link, helpers.connect) and off the
# transport a client rides on (model_state)
LOAN_MEMBERS = ('sock', 'mount', 'udc', 'bounce', 'closed', 'renew', 'close')
TRANSPORT_CALLS = ('link_info',)
# and the rest of what runs there
COMMA_MODULES = tuple(dict.fromkeys((*FORK_IMPORTS, 'jetlink.protocol', 'jetlink.comma.gadget', 'jetlink.comma.lending',
                                     'jetlink.comma.port', 'jetlink.comma.root', 'jetlink.transport.ffs')))

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


def test_a_loan_has_what_openpilot_reads():
  import socket

  from jetlink.comma.lending import Loan
  a, b = socket.socketpair()
  try:
    loan = Loan(a, bytearray(), '/dev/ffs-jetlink', 'udc0')
    assert [n for n in LOAN_MEMBERS if not hasattr(loan, n)] == []
  finally:
    a.close()
    b.close()


def test_every_transport_says_what_it_is():
  from jetlink.transport.ffs import FfsTransport
  from jetlink.transport.tcp import TcpTransport
  for transport in (FfsTransport, TcpTransport):
    assert [n for n in TRANSPORT_CALLS if not callable(getattr(transport, n, None))] == [], transport


def test_the_gadget_presents_what_a_host_looks_for():
  # The server finds the comma by these; nothing on the comma reads protocol.py's copy.
  script = (ROOT / 'scripts' / 'comma' / 'jetlink-root.sh').read_text()
  ids = {k: int(v, 16) for k, v in re.findall(r'^(VID|PID)=(0x[0-9a-fA-F]+)', script, re.M)}
  assert ids == {'VID': P.USB_VID, 'PID': P.USB_PID}
