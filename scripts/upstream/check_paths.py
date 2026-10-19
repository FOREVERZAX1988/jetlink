"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Does an openpilot checkout still have what jetlink/openpilot/stock.py relies
on? The parameters it calls openpilot's functions with, the values jetlink was
tested against, the message fields its engaged() reads, the ModelState
attributes its small_reset zeroes, the upstream rules it copies rather than
imports (the params path, the /AGNOS test), and the modeld seam the forks' hook
lines go into. The literals stock.py writes out (the chestnut ids, the warp's
directory and name, the services, the offroad key, the engaged fields) are read
from stock.py itself, so the two cannot drift apart. That every name it
imports exists is check_imports.py's.

By AST and text alone: the checkout is data, never imported or run.

  python3 -I scripts/upstream/check_paths.py <checkout> [-v]

<checkout> is the repository root (what fetch.sh makes), which holds the
openpilot package and, fetched beside it, opendbc's car.capnp. Exits 1 on
anything missing or different.
"""
import ast
import functools
import re
import string
import sys
from pathlib import Path

STOCK = Path(__file__).resolve().parents[2] / 'jetlink' / 'openpilot' / 'stock.py'

# (module, name, parameters) for what Stock calls, and the modeld names
# jetlink's JoinedModel stands in for. A list ending in '...' is a prefix that
# any further parameter with a default may follow: a harmless addition upstream
NAMES = [
  ('openpilot.common.params', 'Params.get', ['self', 'key', '...']),
  ('openpilot.common.params', 'Params.put', ['self', 'key', 'dat', 'block', '...']),   # block by keyword
  ('openpilot.common.params', 'Params.remove', ['self', 'key', '...']),
  ('openpilot.common.logging_extra', 'SwagLogger.event', ['self', 'event', '*args', '**kwargs']),
  ('openpilot.common.hardware.base', 'HardwareBase.get_device_type', ['self', '...']),
  ('openpilot.common.hardware.hw', 'Paths.comma_home', ['...']),
  ('openpilot.selfdrive.modeld.modeld', 'get_action_from_model',
   ['model_output', 'prev_action', 'lat_action_t', 'long_action_t', 'v_ego', '...']),
  ('openpilot.selfdrive.modeld.modeld', 'ModelState.run', ['self', 'bufs', 'transforms', 'inputs', 'after_enqueue', '...']),
  ('openpilot.selfdrive.modeld.parse_model_outputs', 'Parser.__init__', ['self', '...']),
  ('openpilot.selfdrive.modeld.parse_model_outputs', 'Parser.parse_outputs', ['self', 'outs', '...']),
  ('openpilot.system.camerad.cameras.nv12_info', 'get_nv12_info', ['width', 'height', '...']),
  ('openpilot.cereal.messaging', 'SubMaster.__init__', ['self', 'services', '...']),
  ('openpilot.cereal.messaging', 'SubMaster.update', ['self', 'timeout', '...']),
  ('openpilot.cereal.messaging', 'SubMaster.all_alive', ['self', 'service_list', '...']),
  ('openpilot.cereal.messaging', 'SubMaster.all_valid', ['self', 'service_list', '...']),
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
# where each service engaged() reads is defined, under the checkout. carControl
# is opendbc's, which fetch.sh fetches at openpilot's gitlink
STRUCTS = {
  'carControl': ('opendbc_repo/opendbc/car/car.capnp', 'CarControl'),
  'selfdriveState': ('openpilot/cereal/log.capnp', 'SelfdriveState'),
}
# where Stock.small_reset's ModelState is: it zeroes stock's _MODEL_STATE and prev_desire
MODEL_STATE = ('openpilot.selfdrive.modeld.modeld', 'ModelState')
# (file under the package, text): upstream behaviour stock.py copies, and the
# modeld seam the hook lines go into (docs/porting.md)
ANCHORS = [
  # Stock.params_dir(): hw.h's Path::params() and comma_home(), params.cc's prefix
  ('common/hardware/hw.h', 'util::getenv("OPENPILOT_PREFIX", "")'),
  ('common/hardware/hw.h', 'util::getenv("HOME") + "/.comma" + Path::openpilot_prefix()'),
  ('common/hardware/hw.h', 'util::getenv("PARAMS_ROOT", Hardware::PC() ? (Path::comma_home() + "/params") : "/data/params")'),
  ('common/params.cc', '"/" + util::getenv("OPENPILOT_PREFIX", "d")'),
  # jetlink.comma.root.AGNOS, which Stock takes for openpilot's
  ('common/hardware/__init__.py', "AGNOS = os.path.isfile('/AGNOS')"),
  # attach() goes after the small model is built, and prepare() before the
  # realtime call: tinygrad's threads inherit SCHED_FIFO 54 on core 7
  ('selfdrive/modeld/modeld.py', 'small_model = ModelState('),
  ('selfdrive/modeld/modeld.py', 'config_realtime_process(7, 54)'),
  # the hook writes model.frame_drop_ratio from it; modelV2.big reads JoinedModel.chestnut
  ('selfdrive/modeld/modeld.py', 'frame_drop_ratio = '),
  ('selfdrive/modeld/modeld.py', 'modelV2.big = model.chestnut'),
  # the two-frame warp jetlink's Warps.load checks
  ('selfdrive/modeld/SConscript', '--frames 2'),
]


@functools.cache
def source(path: Path) -> str | None:
  return path.read_text(errors='replace') if path.is_file() else None


@functools.cache
def parsed(path: Path) -> ast.Module:
  """Each file parsed once a run."""
  return ast.parse(source(path), str(path))


def module_file(package: Path, module: str) -> Path | None:
  base = package.joinpath(*module.split('.')[1:])   # past 'openpilot'
  for p in (base.with_suffix('.py'), base / '__init__.py'):
    if p.is_file():
      return p
  return None


@functools.cache
def top_level(path: Path) -> dict[str, ast.AST]:
  names: dict[str, ast.AST] = {}
  for node in parsed(path).body:
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
  node = top_level(f).get(head)
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


def params_of(fn: ast.FunctionDef) -> list[tuple[str, bool]]:
  """(name, may be left out) for each parameter, *args and **kwargs as '*args', '**kwargs'."""
  a = fn.args
  positional = a.posonlyargs + a.args
  first_default = len(positional) - len(a.defaults)
  out = [(x.arg, i >= first_default) for i, x in enumerate(positional)]
  if a.vararg:
    out.append(('*' + a.vararg.arg, True))
  out += [(x.arg, d is not None) for x, d in zip(a.kwonlyargs, a.kw_defaults, strict=True)]
  if a.kwarg:
    out.append(('**' + a.kwarg.arg, True))
  return out


def signature_ok(fn: ast.FunctionDef, expect: list[str]) -> bool:
  got = params_of(fn)
  if expect[-1:] != ['...']:
    return [name for name, _ in got] == expect
  want = expect[:-1]
  return [name for name, _ in got[:len(want)]] == want and all(optional for _, optional in got[len(want):])


def literal(node: ast.AST):
  """A literal's value; frozenset({...}) included."""
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'frozenset':
    return frozenset(ast.literal_eval(node.args[0]))
  return ast.literal_eval(node)


def stock_values() -> dict:
  """Stock's literal class attributes (KEYS as its keywords), and stock.py's module-level ones."""
  tree = parsed(STOCK)
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Stock')
  values = {}
  for node in [*tree.body, *cls.body]:
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


def name_pattern(template: str) -> str:
  """A regex for a str.format template's text, whatever its fields are called."""
  return ''.join(re.escape(text) + (r'\{\w+\}' if field is not None else '')
                 for text, field, _, _ in string.Formatter().parse(template))


def text_check(path: Path, what: str, needle: str = '', pattern: str = '') -> tuple[str, str, str]:
  text = source(path)
  if text is None:
    return 'MISSING', what, f'no file {path.name}'
  found = re.search(pattern, text) if pattern else needle in text
  return ('OK', what, '') if found else ('DIFF', what, 'text not found')


def capnp_fields(text: str, struct: str) -> set[str] | None:
  """The field names declared directly in `struct` (not in a struct nested in it)."""
  start = re.search(rf'^\s*struct\s+{struct}\s*\{{', text, re.M)
  if start is None:
    return None
  depth, fields = 1, set()
  for line in text[start.end():].splitlines():
    code = line.split('#', 1)[0]
    if depth == 1 and (m := re.match(r'\s*(\w+)\s*@\d+', code)):
      fields.add(m.group(1))
    depth += code.count('{') - code.count('}')
    if depth <= 0:
      break
  return fields


def check(checkout: Path) -> list[tuple[str, str, str]]:
  package = checkout / 'openpilot'
  results = []
  for module, name, expect in NAMES:
    node, err = lookup(package, module, name)
    if err:
      results.append(('MISSING', f"{module}:{name}", err))
    elif not isinstance(node, ast.FunctionDef):
      results.append(('DIFF', f"{module}:{name}", 'not a function'))
    elif not signature_ok(node, expect):
      results.append(('DIFF', f"{module}:{name}", f"params {[n for n, _ in params_of(node)]} != {expect}"))
    else:
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
  ids, unreadable = set(), False
  for name in ('CHESTNUT_USB_IDS', 'CHESTNUT_ROM_USB_IDS'):
    node, err = lookup(package, 'openpilot.common.hardware.usb', name)
    try:
      ids |= set(literal(node.value)) if err is None else set()
    except (ValueError, TypeError, SyntaxError, AttributeError):
      err = 'not a literal'
    if err:
      unreadable = True
      results.append(('MISSING' if 'missing' in err else 'DIFF', f"usb:{name}", err))
  if not unreadable:
    results.append(('OK', 'Stock.CHESTNUT_IDS', '') if ids == stock['CHESTNUT_IDS'] else
                   ('DIFF', 'Stock.CHESTNUT_IDS', f"usb has {', '.join(f'{v:04x}:{d:04x}' for v, d in sorted(ids))}"))
  # the warp: modeld beside WARP_DIR keeps its models in that directory, its
  # SConscript compiles WARP_NAME into it (a fork may prefix the name, as
  # zoompilot's big_ for a chestnut), and modeld loads it from there
  warp_dir = Path(stock['WARP_DIR'])
  modeld = checkout / warp_dir.parent
  where = f"Stock.WARP_DIR {warp_dir}"
  name = name_pattern(stock['WARP_NAME'])
  results.append(text_check(modeld / 'helpers.py', f"{where} ~ helpers.py MODELS_DIR",
                            pattern=rf"MODELS_DIR\s*=.*\.parent\s*/\s*['\"]{re.escape(warp_dir.name)}['\"]"))
  results.append(text_check(modeld / 'SConscript', f"{where}/WARP_NAME ~ SConscript",
                            pattern=rf"['\"]{re.escape(warp_dir.name)}/(\{{[^{{}}]*\}})?{name}['\"]"))
  results.append(text_check(modeld / 'modeld.py', "Stock.WARP_NAME ~ modeld.py loads it from MODELS_DIR",
                            pattern=rf"MODELS_DIR\s*/\s*f['\"](\{{[^{{}}]*\}})?{name}['\"]"))
  for service in stock['IN_CONTROL_SERVICES']:
    results.append(text_check(package / 'cereal/services.py', f"Stock.IN_CONTROL_SERVICES {service}", f'"{service}":'))
  results.append(text_check(package / 'common/params_keys.h', f"Stock.KEYS.offroad {stock['KEYS']['offroad']}",
                            f'{{"{stock["KEYS"]["offroad"]}",'))
  for service, fields in stock['_ENGAGED'].items():
    if service not in STRUCTS:
      results.append(('DIFF', f"engaged() {service}", 'no struct known for it here: add it to STRUCTS'))
      continue
    rel, struct = STRUCTS[service]
    text = source(checkout / rel)
    declared = None if text is None else capnp_fields(text, struct)
    if declared is None:
      why = f"no {rel} (fetch.sh fetches opendbc's): not covered" if text is None else f"no struct {struct}"
      results.append(('MISSING', f"engaged() {service} ~ {rel}", why))
      continue
    for field in fields:
      results.append(('OK', f"engaged() {struct}.{field}", '') if field in declared else
                     ('MISSING', f"engaged() {struct}.{field}", f"not in {rel}"))

  (module, cls), attrs = MODEL_STATE, (*stock['_MODEL_STATE'], 'prev_desire')
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
    results.append(text_check(package / rel, f"{rel} ~ {needle}", needle))
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
