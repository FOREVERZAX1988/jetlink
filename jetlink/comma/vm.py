"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The VM tuning the gadget needs, and how to put it back: jetlink-root.sh vm
apply|restore, which holds the values, the record of the stock ones in
/dev/shm/jetlink-sysctl-prev, and why each is what it is.

loggerd's dirty pages pile up until the kernel reclaims them synchronously,
right while a FunctionFS transfer allocates its buffer: gadget reads stalled
200-350 ms, past the fork's backend.INFERENCE_TIMEOUT, and the big model fell
back. Capping dirty memory and holding a free-memory floor took the worst frame
from 244 to 72 ms with no lagging frames over 20 min.

System-wide, since the gadget read shares the kernel with every writer. The
owner applies it while the link is on, so a device with the link off runs
stock values, which are put back only on disable. Never restored on exit:
manager stops the owner at ignition, exactly when the contention starts, so
modeld would get stock values every drive. A reboot resets them.
"""
from __future__ import annotations

from jetlink.comma import gadget, root


def apply() -> bool:
  """Record the stock values once, then apply ours. AGNOS only."""
  return gadget.AGNOS and root.run('vm', 'apply', timeout=root.VM_TIMEOUT)


def restore() -> bool:
  """Put the recorded values back and drop the record. AGNOS only."""
  return gadget.AGNOS and root.run('vm', 'restore', timeout=root.VM_TIMEOUT)
