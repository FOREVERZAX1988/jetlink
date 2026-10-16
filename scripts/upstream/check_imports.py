"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Does every openpilot import in jetlink/openpilot/stock.py resolve in a
checkout? Each `from openpilot.X import Y` in the file, method bodies included,
read off its AST, so the list keeps itself as stock.py changes; check_paths.py
holds the parameters and values.

By AST alone: the checkout is data, never imported or run.

  python3 -I scripts/upstream/check_imports.py <checkout> [<adapter.py>]

<checkout> is the repository root; the adapter file defaults to stock.py, and
may be a fork's own. Exits 1 on any import that does not resolve.
"""
import ast
import sys
from pathlib import Path

# -I leaves this directory off the path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_paths import STOCK, lookup


def openpilot_imports(source: Path) -> list[tuple[str, str]]:
  found = set()
  for node in ast.walk(ast.parse(source.read_text(), str(source))):
    if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'openpilot':
      found |= {(node.module, alias.name) for alias in node.names}
    elif isinstance(node, ast.Import):
      found |= {(alias.name, '') for alias in node.names if alias.name.split('.')[0] == 'openpilot'}
  return sorted(found)


def resolves(package: Path, module: str, name: str) -> str | None:
  """None when `from module import name` would find something, else why not."""
  if not name:   # `import openpilot.x.y`
    base = package.joinpath(*module.split('.')[1:])
    return None if base.with_suffix('.py').is_file() or (base / '__init__.py').is_file() else f"module {module} missing"
  _, err = lookup(package, module, name)
  if err:
    # `from openpilot.cereal import messaging` names a submodule
    sub = package.joinpath(*module.split('.')[1:], name)
    if sub.with_suffix('.py').is_file() or (sub / '__init__.py').is_file():
      return None
  return err


def main(argv: list[str]) -> int:
  if not argv:
    print(__doc__.strip().split('\n\n')[-2])
    return 2
  checkout, source = Path(argv[0]), Path(argv[1]) if len(argv) > 1 else STOCK
  found = openpilot_imports(source)
  bad = [f"{module}:{name} ({err})" for module, name in found if (err := resolves(checkout / 'openpilot', module, name))]
  print(f"{source.name} against {checkout}: {len(found)} imports, {len(bad)} missing")
  for line in bad:
    print(f"  MISSING {line}")
  return 1 if bad or not found else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
