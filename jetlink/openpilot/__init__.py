"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink on an openpilot device: everything between the fork and the comma's
device layer (jetlink.comma).

The fork implements the interface in interface.py once, in an adapter module
that holds every openpilot import, and calls what this package exports. That
is the whole contract between the two repos, and API names its version: the
adapter checks it exactly and treats any other value as jetlink being absent,
an offroad alert and no link rather than a crash.

API changes only for a breaking change to a name exported here or to the
interface. A new Status field, a new method, or a new interface member that
jetlink reads with getattr and a fallback, is additive and keeps it.

Imported by the resident gadget owner, so nothing heavy at module level.
"""
from jetlink.openpilot.interface import (MODES, STATES, BuildSide, Keys, ModelFace, ModelSide, Openpilot, OwnerConfig,
                                         StatusSide, WorkerSide, conformance)

API = 1

__all__ = ['API', 'MODES', 'STATES', 'BuildSide', 'Keys', 'ModelFace', 'ModelSide', 'Openpilot', 'OwnerConfig',
           'StatusSide', 'WorkerSide', 'conformance']
