"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlinkd, the resident gadget owner, as an openpilot device runs it.

manager runs the fork's adapter module under the name jetlinkd, and its main()
builds an OwnerConfig and calls main() here. The owner itself is
jetlink.comma.owner; this hands it the settings, read off the params files the
config names, the chestnut's USB ids, and the provisioning run to start when
there is work.

The owner stays resident for the whole drive at about 10 MB, so this module
and everything it imports is the standard library and jetlink's light half;
tests/openpilot/test_imports.py holds the line.
"""
from __future__ import annotations

from jetlink.comma import owner as comma_owner
from jetlink.openpilot.interface import OwnerConfig
from jetlink.openpilot.settings import FileParams, Settings


def settings(config: OwnerConfig) -> Settings:
  """What the owner reads: the link setting, whether the car is parked, and
  when the pick and the built model last changed."""
  return Settings(FileParams(config.params_dir), config.keys)


def main(config: OwnerConfig) -> None:
  """Hold the gadget until manager stops this process (SIGINT, or SIGTERM)."""
  comma_owner.main(list(config.worker), cwd=config.cwd, env=dict(config.env), log_file=config.log_file,
                   settings=settings(config), chestnut_ids=config.chestnut_ids)
