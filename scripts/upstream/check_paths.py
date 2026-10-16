"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Does an openpilot checkout still have what jetlink/openpilot/stock.py relies
on? The names it calls, with the parameters it calls them with, the values
jetlink was tested against, the ModelState attributes its small_reset zeroes,
and the modeld seam the fork's hook lines patch. The literals stock.py writes
out (the chestnut ids, the warp's name, the services, the offroad key) are read
from stock.py itself, so the two cannot drift apart.

By AST and text alone: the checkout is data, never imported or run.

  python3 -I scripts/upstream/check_paths.py <checkout> [-v]

<checkout> is the repository root (what fetch.sh makes), which holds the
openpilot package. Exits 1 on anything missing or different.
"""
import ast
import sys
from pathlib import Path

STOCK = Path(__file__).resolve().parents[2] / 'jetlink' / 'openpilot' / 'stock.py'

# (module, name, parameters): a parameter list ending in '...' is a prefix,
# None any binding. What Stock imports and how it calls it, and the modeld
# names jetlink's JoinedModel stands in for
NAMES = [
  ('openpilot.common.params', 'Params', None),
  ('openpilot.common.params', 'Params.get', ['self', 'key', 'block', 'return_default']),
  ('openpilot.common.params', 'Params.put', ['self', 'key', 'dat', 'block']),
  ('openpilot.common.params', 'Params.remove', ['self', 'key']),
  ('openpilot.common.swaglog', 'cloudlog', None),
  ('openpilot.common.logging_extra', 'SwagLogger.event', ['self', 'event', '*args', '**kwargs']),
  ('openpilot.common.basedir', 'BASEDIR', None),
  ('openpilot.common.hardware', 'HARDWARE', None),
  ('openpilot.common.hardware.base', 'HardwareBase.get_device_type', ['self']),
  ('openpilot.common.hardware.hw', 'Paths', None),
  ('openpilot.common.hardware.hw', 'Paths.comma_home', []),
  ('openpilot.common.transformations.camera', '_ar_ox_fisheye', None),
  ('openpilot.common.transformations.camera', '_os_fisheye', None),
  ('openpilot.common.transformations.model', 'MEDMODEL_INPUT_SIZE', None),
  ('openpilot.selfdrive.modeld.constants', 'ModelConstants', None),
  ('openpilot.selfdrive.modeld.modeld', 'get_action_from_model',
   ['model_output', 'prev_action', 'lat_action_t', 'long_action_t', 'v_ego']),
  ('openpilot.selfdrive.modeld.modeld', 'ModelState.run', ['self', 'bufs', 'transforms', 'inputs', 'after_enqueue']),
  ('openpilot.selfdrive.modeld.parse_model_outputs', 'Parser.parse_outputs', ['self', 'outs']),
  ('openpilot.system.camerad.cameras.nv12_info', 'get_nv12_info', ['width', 'height']),
  ('openpilot.cereal.messaging', 'SubMaster.__init__', ['self', 'services', '...']),
  ('openpilot.cereal.messaging', 'SubMaster.update', ['self', 'timeout']),
  ('openpilot.cereal.messaging', 'SubMaster.all_alive', ['self', 'service_list']),
  ('openpilot.cereal.messaging', 'SubMaster.all_valid', ['self', 'service_list']),
]
# names Stock reads with a fallback: reported, never a failure
OPTIONAL = [
  ('openpilot.common.hardware.hw', 'Paths.model_root'),   # sunnypilot's
]
# (module, name, value) jetlink was tested against
LITERALS = [
  ('openpilot.common.transformations.model', 'MEDMODEL_INPUT_SIZE', (512, 256)),
  ('openpilot.selfdrive.modeld.modeld', 'LAT_SMOOTH_SECONDS', 0.0),
  ('openpilot.selfdrive.modeld.modeld', 'LONG_SMOOTH_SECONDS', 0.3),
  ('openpilot.selfdrive.modeld.constants', 'ModelConstants.DESIRE_LEN', 8),
]
# what Stock.small_reset zeroes, as the class assigns it
MODEL_STATE = ('openpilot.selfdrive.modeld.modeld', 'ModelState', ('input_queues', 'state_pairs', 'packed_input', 'prev_desire'))
# the modeld seam the hook lines go into (docs/porting.md): (file under the package, text)
ANCHORS = [
  ('selfdrive/modeld/modeld.py', 'small_model = ModelState('),
  ('selfdrive/modeld/modeld.py', 'config_realtime_process(7, 54)'),   # prepare() goes before it
  ('selfdrive/modeld/modeld.py', 'frame_drop_ratio = '),
  ('selfdrive/modeld/modeld.py', 'modelV2.big = model.chestnut'),
  ('selfdrive/modeld/modeld.py', 'ChestnutActive'),
  ('selfdrive/modeld/SConscript', '--frames 2'),                       # the two-frame warp jetlink checks
  ('cereal/log.capnp', 'selfdriveState @130'),
]


def module_file(package: Path, module: str) -> Path | None:
  base = package.joinpath(*module.split('.')[1:])   # past 'openpilot'
  for p in (base.with_suffix('.py'), base / '__init__.py'):
    if p.is_file():
      return p
  return None


def top_level(tree: ast.Module) -> dict[str, ast.AST]:
  names: dict[str, ast.AST] = {}
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      names[node.name] = node
    elif isinstance(node, ast.Assign):
      for target in node.targets:
        for n in ast.walk(target):
          if isinstance(n, ast.Name):
            names[n.id] = node
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      names[node.target.id] = node
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      for alias in node.names:
        names[(alias.asname or alias.name).split('.')[0]] = node
    elif isinstance(node, ast.If):   # `if PC: HARDWARE = Pc() else: ...`
      for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
          for target in sub.targets:
            if isinstance(target, ast.Name):
              names.setdefault(target.id, sub)
  return names


def lookup(package: Path, module: str, dotted: str) -> tuple[ast.AST | None, str | None]:
  """The node binding `dotted` (a name, or Class.member) in `module`, or why not."""
  f = module_file(package, module)
  if f is None:
    return None, f"module {module} missing"
  head, *rest = dotted.split('.')
  node = top_level(ast.parse(f.read_text(), str(f))).get(head)
  if node is None:
    return None, f"{module}.{head} missing"
  for part in rest:
    if not isinstance(node, ast.ClassDef):
      return None, f"{module}.{dotted}: {head} is not a class"
    found = None
    for sub in node.body:
      if isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and sub.name == part:
        found = sub
      elif isinstance(sub, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in sub.targets):
        found = sub
      elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) and sub.target.id == part:
        found = sub
    if found is None:
      return None, f"{module}.{dotted} missing"
    node = found
  return node, None


def params_of(fn: ast.FunctionDef) -> list[str]:
  a = fn.args
  out = [x.arg for x in a.posonlyargs + a.args]
  if a.vararg:
    out.append('*' + a.vararg.arg)
  out += [x.arg for x in a.kwonlyargs]
  if a.kwarg:
    out.append('**' + a.kwarg.arg)
  return out


def literal(node: ast.AST):
  """A literal's value; frozenset({...}) included."""
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'frozenset':
    return frozenset(ast.literal_eval(node.args[0]))
  return ast.literal_eval(node)


def stock_values() -> dict:
  """Stock's literal class attributes, KEYS as its keywords."""
  tree = ast.parse(STOCK.read_text(), str(STOCK))
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Stock')
  values = {}
  for node in cls.body:
    target = node.targets[0] if isinstance(node, ast.Assign) else node.target if isinstance(node, ast.AnnAssign) else None
    if not isinstance(target, ast.Name) or node.value is None:
      continue
    if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == 'Keys':
      values[target.id] = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
      continue
    try:
      values[target.id] = literal(node.value)
    except (ValueError, TypeError, SyntaxError):
      pass   # Path(...) and the like: not checked against the tree
  return values


def text_has(package: Path, rel: str, needle: str) -> tuple[str, str]:
  p = package / rel
  if not p.is_file():
    return 'MISSING', 'no file'
  return ('OK', '') if needle in p.read_text(errors='replace') else ('DIFF', 'text not found')


def check(checkout: Path) -> list[tuple[str, str, str]]:
  package = checkout / 'openpilot'
  results = []
  for module, name, expect in NAMES:
    node, err = lookup(package, module, name)
    if err:
      results.append(('MISSING', f"{module}:{name}", err))
      continue
    if expect is not None:
      if not isinstance(node, ast.FunctionDef):
        results.append(('DIFF', f"{module}:{name}", 'not a function'))
        continue
      got = params_of(node)
      prefix = expect[-1:] == ['...']
      want = expect[:-1] if prefix else expect
      if (got[:len(want)] if prefix else got) != want:
        results.append(('DIFF', f"{module}:{name}", f"params {got} != {expect}"))
        continue
    results.append(('OK', f"{module}:{name}", ''))
  for module, name in OPTIONAL:
    _, err = lookup(package, module, name)
    results.append(('ABSENT' if err else 'OK', f"{module}:{name} (optional)", 'Stock falls back' if err else ''))
  for module, name, expected in LITERALS:
    node, err = lookup(package, module, name)
    if err:
      results.append(('MISSING', f"{module}:{name}=", err))
      continue
    try:
      value = literal(node.value)
    except (ValueError, TypeError, SyntaxError, AttributeError):
      results.append(('DIFF', f"{module}:{name}=", 'not a literal'))
      continue
    results.append(('OK', f"{module}:{name}=", '') if value == expected else ('DIFF', f"{module}:{name}=", repr(value)))

  # what stock.py writes out, against the tree
  stock = stock_values()
  ids = set()
  for name in ('CHESTNUT_USB_IDS', 'CHESTNUT_ROM_USB_IDS'):
    node, err = lookup(package, 'openpilot.common.hardware.usb', name)
    if err:
      results.append(('MISSING', f"usb:{name}", err))
    else:
      ids |= set(literal(node.value))
  results.append(('OK', 'Stock.CHESTNUT_IDS', '') if ids == stock['CHESTNUT_IDS'] else
                 ('DIFF', 'Stock.CHESTNUT_IDS', f"usb has {', '.join(f'{v:04x}:{d:04x}' for v, d in sorted(ids))}"))
  # the warp: modeld's SConscript, beside WARP_DIR, compiles WARP_NAME (a
  # fork may put a prefix before it, as zoompilot's big_ for a chestnut)
  sconscript = Path(stock['WARP_DIR']).parent.relative_to('openpilot') / 'SConscript'
  status, why = text_has(package, str(sconscript), stock['WARP_NAME'])
  results.append((status, f"Stock.WARP_DIR/WARP_NAME ~ {sconscript}", why))
  for service in stock['IN_CONTROL_SERVICES']:
    status, why = text_has(package, 'cereal/services.py', f'"{service}":')
    results.append((status, f"Stock.IN_CONTROL_SERVICES {service}", why))
  status, why = text_has(package, 'common/params_keys.h', f'{{"{stock["KEYS"]["offroad"]}",')
  results.append((status, f"Stock.KEYS.offroad {stock['KEYS']['offroad']}", why))

  module, cls, attrs = MODEL_STATE
  node, err = lookup(package, module, cls)
  if err:
    results.append(('MISSING', f"{module}:{cls}", err))
  else:
    assigned = {t.attr for sub in ast.walk(node) if isinstance(sub, (ast.Assign, ast.AnnAssign))
                for t in (sub.targets if isinstance(sub, ast.Assign) else [sub.target])
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self'}
    for attr in attrs:
      results.append(('OK', f"{cls}.{attr}", '') if attr in assigned else ('MISSING', f"{cls}.{attr}", 'never assigned'))
  for rel, needle in ANCHORS:
    status, why = text_has(package, rel, needle)
    results.append((status, f"{rel} ~ {needle}", why))
  return results


def main(argv: list[str]) -> int:
  verbose = '-v' in argv
  trees = [a for a in argv if a != '-v']
  if not trees:
    print(__doc__.strip().split('\n\n')[-2])
    return 2
  failed = False
  for tree in trees:
    results = check(Path(tree))
    bad = [r for r in results if r[0] in ('MISSING', 'DIFF')]
    failed |= bool(bad)
    print(f"{tree}: {len(results)} checks, {len(bad)} failed")
    for status, what, why in results:
      if verbose or status != 'OK':
        print(f"  {status:7} {what} {why}".rstrip())
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
