"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The HEVC decoder's build flags, as SwiftPM actually passes them.

`Package.swift` picks the decoder with a `.define` per platform, and
`jl_hevc.c` chooses its implementation with `#if defined(...)` on those same
names (directly, or through a one-line alias it defines itself). Nothing in
either file's compiler or test run notices when the two drift apart: the C
still compiles, the library still links, and the #else branch -- a decoder
whose `create` returns NULL -- is what ships. On the device that showed up as
`HevcFrame` never being built and every HEVC frame answered BAD_SHAPE, which
looks like a protocol or codec fault and is neither.

These tests are the missing link check: every macro the C branches on is one
SwiftPM defines for some platform, every platform that gets a define has a
branch for it, and the alias names the C may still use are aliases of a real
define rather than free-floating.

Run on any machine: it reads the two files as text.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'JetlinkKit' / 'Package.swift'
SOURCE = ROOT / 'JetlinkKit' / 'Sources' / 'CHevc' / 'jl_hevc.c'

# `.define("JL_HEVC_X", .when(platforms: [.android]))`
DEFINE = re.compile(r'\.define\("(JL_HEVC_\w+)"\s*,\s*\.when\(platforms:\s*\[([^\]]*)\]\)')
# `#if defined(X)` / `#elif defined(X)`
IF_DEFINED = re.compile(r'^\s*#(?:if|elif)\s+defined\(\s*(JL_HEVC_\w+)\s*\)', re.M)
# `#ifdef X`
IFDEF = re.compile(r'^\s*#ifdef\s+(JL_HEVC_\w+)', re.M)
# `#define X 1`, the alias the C makes for itself
SELF_DEFINE = re.compile(r'^\s*#define\s+(JL_HEVC_\w+)\s+1\s*$', re.M)


def read(path: Path) -> str:
  if not path.exists():
    raise AssertionError(f'{path} is missing; did the decoder move?')
  return path.read_text(encoding='utf-8', errors='replace')


class TestHevcBuildFlags(unittest.TestCase):

  def setUp(self):
    self.package = read(PACKAGE)
    self.source = read(SOURCE)

  def swiftpm_defines(self) -> dict:
    """macro -> the platforms its .define applies to."""
    found = {}
    for name, platforms in DEFINE.findall(self.package):
      found[name] = platforms
    return found

  def c_branch_macros(self) -> set:
    """Macros the C chooses an implementation by, aliases excluded."""
    aliases = set(SELF_DEFINE.findall(self.source))
    used = set(IF_DEFINED.findall(self.source)) | set(IFDEF.findall(self.source))
    return used - aliases

  def test_swiftpm_defines_a_decoder_for_android_and_linux(self):
    defines = self.swiftpm_defines()
    self.assertTrue(defines, 'Package.swift defines no JL_HEVC_* for the CHevc target')
    joined = ' '.join(defines.values())
    self.assertIn('android', joined, 'no decoder is defined for android')
    self.assertIn('linux', joined, 'no decoder is defined for linux')

  def test_every_branch_macro_is_one_swiftpm_defines(self):
    defines = set(self.swiftpm_defines())
    branches = self.c_branch_macros()
    self.assertTrue(branches, 'jl_hevc.c chooses no implementation at all')
    unknown = branches - defines
    self.assertFalse(
      unknown,
      'jl_hevc.c branches on ' + ', '.join(sorted(unknown)) +
      ' but Package.swift defines only ' + ', '.join(sorted(defines)) +
      '. The branch would never be compiled: no decoder, every HEVC frame '
      'answered BAD_SHAPE.')

  def test_every_swiftpm_define_has_a_branch(self):
    # a define nothing reads is dead weight and, worse, hides a rename: the
    # platform it was for silently gets the #else branch
    branches = self.c_branch_macros() | set(SELF_DEFINE.findall(self.source))
    unused = set(self.swiftpm_defines()) - branches
    self.assertFalse(
      unused,
      'Package.swift defines ' + ', '.join(sorted(unused)) +
      ' but jl_hevc.c never reads it: that platform gets the no-decoder branch')

  def test_the_alias_names_map_onto_a_real_define(self):
    # JL_HEVC_MC and JL_HEVC_AV are the names the branches may still use; each
    # alias has to hang off a define SwiftPM actually passes, or it is a
    # free-floating name that no build ever sets.
    defines = set(self.swiftpm_defines())
    for alias in SELF_DEFINE.findall(self.source):
      guarded = re.search(
        r'#if\s+defined\(\s*(\w+)\s*\)\s*\n\s*#define\s+' + re.escape(alias), self.source)
      self.assertIsNotNone(guarded, f'{alias} is defined with no #if around it')
      self.assertIn(guarded.group(1), defines,
                    f'{alias} is an alias of {guarded.group(1)}, which SwiftPM never defines')


if __name__ == '__main__':
  unittest.main()
