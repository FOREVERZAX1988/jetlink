"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Who may do endpoint IO on the gadget, while one process owns it throughout.

The owner (owner.py) holds ep0 for as long as the link is on, and nothing else
ever does: when two processes took turns at it, every change of owner was an
unplug, a fresh libusb open and a fresh server session as the Jetson saw it.

What still has to change hands is the right to read the endpoint files.
FunctionFS keeps a queued read queued until something completes it, so a second
reader would sit behind the first and take its reply. This is the handshake for
that: modeld borrows for the length of a drive, the owner stays off the
endpoints while it does, and the connection is the lease, so a modeld that is
killed returns it by dying. There is no "give it back" message: the socket
closing is the only signal, because it is the only one a killed process sends.

An iPhone cannot take the endpoint files: it is on the gadget's network
interface and dials the owner (CableListener). The loan then carries the
accepted socket instead, sent over the same unix socket with SCM_RIGHTS, and
the owner closes its copy when the loan ends so the phone dials again.
"""
from __future__ import annotations

import errno
import json
import os
import select
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from jetlink.comma import gadget

SOCKET = Path('/dev/shm/jetlink-lend.sock')
# how long a borrower waits for the owner to put the gadget down. It only has
# something to put down if it was mid-provision at ignition, and then it is one
# re-enumeration; the usual answer is immediate
BORROW_TIMEOUT = 8.0
# a stuck write is already 15 s old by the time this is asked for
BOUNCE_TIMEOUT = 10.0
# the owner records a note at once; this only bounds one that is wedged
NOTE_TIMEOUT = 5.0
RETRY = 0.25
POLL = 0.5
# what a borrower passes on of the server's hello (Loan.note_server): the owner
# never speaks the protocol, and sleep_after is how it knows whether letting
# go of the gadget lets the Jetson sleep. The rest is for its status record
SERVER_FIELDS = ('protocol', 'device', 'backend', 'runtime_version', 'trt_version', 'sleep_after')


def _send(conn: socket.socket, msg: dict) -> None:
  conn.sendall(json.dumps(msg).encode() + b'\n')


def _recv_line(conn: socket.socket, buf: bytearray, deadline: float, fds: list[int] | None = None) -> dict | None:
  """One json message off the socket, or None if the deadline passes first.

  Both ends speak newline-delimited json over a stream, so a message can arrive
  in pieces or with the next one behind it, and the recv timeout is a poll
  rather than a refusal: the lender answers one borrower at a time and can be a
  moment late while it lets go of the one before.

  The peer going away raises, because that is the one thing neither end may
  read as "nothing yet": for the lender it is the whole lease ending.

  With `fds`, file descriptors sent along with the bytes land there: a plain
  recv would have the kernel close them unseen.
  """
  while time.monotonic() < deadline:
    if b'\n' in buf:
      line, _, rest = bytes(buf).partition(b'\n')
      buf[:] = rest
      return json.loads(line)
    try:
      if fds is None:
        chunk = conn.recv(4096)
      else:
        chunk, got, _, _ = socket.recv_fds(conn, 4096, 4)
        fds.extend(got)
    except TimeoutError:
      continue
    if not chunk:
      raise ConnectionResetError('the peer closed the link socket')
    buf.extend(chunk)
  return None


class Loan:
  """The right to do endpoint IO on a gadget the owner holds.

  Held for the length of a drive: modeld is stopped at every ignition-off and
  SIGKILLed if it lingers, so the socket closing is how the link is handed
  back, and a modeld that crashed hands it back the same way.
  """

  def __init__(self, conn: socket.socket, buf: bytearray, mount: str, udc: str,
               sock: socket.socket | None = None, name: str = 'modeld'):
    self.conn = conn
    self.mount = mount
    self.udc = udc
    self.name = name
    # a phone's dial, accepted by the owner: the link rides on this and the
    # endpoint files are left alone. None on a USB link
    self.sock = sock
    self._buf = buf
    self._lock = threading.Lock()
    self._closed = False

  @property
  def closed(self) -> bool:
    return self._closed

  def bounce(self) -> bool:
    """Ask the owner to take the gadget down and put it back up.

    The only thing that dequeues a FunctionFS write nobody is reading is the
    unbind, and the unbind belongs to whoever holds ep0. Called from the write
    watchdog on a link that is already 15 s stuck, so the re-enumeration it
    costs is not the expensive part.
    """
    with self._lock:
      if self._closed:
        return False
      try:
        _send(self.conn, {'op': 'bounce'})
        reply = _recv_line(self.conn, self._buf, time.monotonic() + BOUNCE_TIMEOUT)
      except OSError:
        gadget.log.exception("jetlink: could not ask for a gadget bounce")
        return False
      return bool(reply and reply.get('ok'))

  def note_server(self, hello: dict) -> bool:
    """Tell the owner what the server said in its hello, which the owner
    cannot ask for itself. Every borrower sends it after every hello, so
    modeld's join refreshes it each drive and a Jetson moved to another
    power supply is known by the next one. False when the owner did not
    take it: an older owner answers "unknown op", which is not an error.

    An answer that does not come within NOTE_TIMEOUT, or is not one, closes
    the loan. The exchange has no ids, so an answer that came later would be
    read as the answer to the next request on it, a renewal's or a bounce's,
    and every one after that would be one behind."""
    fields = {k: hello[k] for k in SERVER_FIELDS if k in hello}
    with self._lock:
      if self._closed:
        return False
      try:
        _send(self.conn, {'op': 'server', **fields})
        reply = _recv_line(self.conn, self._buf, time.monotonic() + NOTE_TIMEOUT)
      except (OSError, ValueError) as e:
        reply, why = None, str(e) or type(e).__name__
      else:
        why = 'no answer' if reply is None else f'the answer {reply!r}'
      if not isinstance(reply, dict):
        gadget.log.warning("jetlink: the owner did not take what the server said (%s), letting the loan go", why)
        self._closed = True
        _shut(self.conn)
        return False
    return bool(reply.get('ok'))

  def renew(self, timeout: float = BORROW_TIMEOUT) -> bool:
    """Ask again which link this loan is for, before another attempt at a join.

    The owner answers as it would a new borrower. The loan lasts the drive and
    the answer changes under it: a phone that dialed after the first answer was
    never used, and a dial whose session ended with the last attempt is spent.
    Without this a borrower that took the endpoint files once wrote a hello to
    a phone every attempt, 15 s and a bounce each, and every bounce took the
    phone's network interface down before it could dial.

    False when the owner gave no link in time or refused; the loan is closed
    only when the owner is gone.
    """
    with self._lock:
      if self._closed:
        return False
      spent, self.sock = self.sock, None
      _close(spent)   # the client that used it closed it too; closing twice is harmless
      reply = self._take(timeout)
      if reply is not None and (self.sock is None) != (spent is None):
        gadget.log.warning("jetlink: the loan is now %s", _what_was_lent(reply))
      return reply is not None

  def _take(self, timeout: float) -> dict | None:
    """Ask the owner for the link until it lends one or `timeout` passes; on a
    lend, what this loan carries now and the reply that lent it. The loan is
    closed when the owner is gone."""
    try:
      got = _ask(self.conn, self._buf, self.name, time.monotonic() + timeout)
    except (OSError, ValueError, KeyError):
      if not self._closed:   # a close() from another thread wakes the ask this way
        gadget.log.exception("jetlink: could not ask the owner for the link")
      self._closed = True
      _close(self.conn)
      return None
    if got is None:
      return None
    reply, self.sock = got
    self.mount, self.udc = str(reply['mount']), str(reply['udc'])
    return reply

  def close(self) -> None:
    # not behind the lock: a renewal holds it through a whole hold, and the
    # shutdown is what wakes that renewal
    self._closed = True
    _shut(self.conn)
    with self._lock:
      _close(self.conn)
      _close(self.sock)


def borrow(name: str = 'modeld', timeout: float = BORROW_TIMEOUT, path: Path | None = None) -> Loan | None:
  """Ask the owner for the link, or None if there is nobody to ask: the link
  was only just turned on, or the owner died or cannot listen, which it says
  in gadget.gadget_error(). The caller asks again later; only the owner ever
  holds ep0. `path` defaults to SOCKET as it is when called, so a test that
  points SOCKET elsewhere is never lent the real owner's link.
  """
  try:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(POLL)
    conn.connect(str(SOCKET if path is None else path))
  except OSError:
    return None   # no owner listening
  loan = Loan(conn, bytearray(), '', '', name=name)
  reply = loan._take(timeout)
  if reply is None:
    loan.close()
    return None
  gadget.log.warning("jetlink: borrowed %s from the owner", _what_was_lent(reply))
  return loan


def _ask(conn: socket.socket, buf: bytearray, name: str,
         deadline: float) -> tuple[dict, socket.socket | None] | None:
  """Ask for the link until the owner lends it or `deadline` passes: the reply
  that lent it and, on the cable, the phone's socket. None when out of time or
  refused. For a first borrow and a renewal alike."""
  fds: list[int] = []
  try:
    while time.monotonic() < deadline:
      _send(conn, {'op': 'borrow', 'name': name})
      reply = _recv_line(conn, buf, deadline, fds)
      if reply is None:
        return None   # out of time
      if reply.get('ok'):
        if not reply.get('cable'):
          return reply, None
        if not fds:
          gadget.log.warning("jetlink: the owner lent the cable link without its socket")
          return None
        return reply, socket.socket(fileno=fds.pop(0))
      if not reply.get('retry'):
        gadget.log.warning("jetlink: the owner would not lend the gadget (%s)", reply.get('detail'))
        return None
      time.sleep(RETRY)
    return None
  finally:
    for fd in fds:
      os.close(fd)


def _what_was_lent(reply: dict) -> str:
  if reply.get('cable'):
    return f"the cable link ({reply.get('peer')})"
  return f"the gadget (udc {reply.get('udc')})"


# between attempts to bind the cable address: usb0 has no address until
# jetlink-root.sh net has run, and a bind that keeps failing is not worth a log
# line twice a second
CABLE_BIND_BACKOFF = 5.0


class CableListener:
  """The owner's ear for a phone: one accept socket on CABLE_ADDR, open while
  the gadget is presented, holding at most one dial at a time.

  A dial is accepted from the owner's step, never read: what it proves is that
  a phone is on the cable, and the bytes belong to whoever borrows the link.
  A newer dial replaces an older one, so a phone whose app restarted is not
  stuck behind its own dead connection.
  """

  def __init__(self):
    self._srv: socket.socket | None = None
    self._sock: socket.socket | None = None
    self.peer: str | None = None
    self.bound: tuple | None = None
    self.next_open = 0.0
    # we hung up on the phone because its borrower finished, so its next dial
    # is the same phone coming back, not news; `news` says which the last
    # accepted dial was
    self.redial_expected = False
    self.news = False
    self._last_error = ''
    self._lock = threading.Lock()

  @property
  def listening(self) -> bool:
    return self._srv is not None

  @property
  def held(self) -> bool:
    """Is a phone's dial in hand?"""
    return self._sock is not None

  def open(self) -> bool:
    """Bind, or say why not. Never raises: the address is usb0's, which may
    not exist yet, and the owner's loop must carry on without it."""
    if self._srv is not None:
      return True
    now = time.monotonic()
    if now < self.next_open:
      return False
    self.next_open = now + CABLE_BIND_BACKOFF
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      srv.bind(gadget.CABLE_ADDR)
      srv.listen(2)
      srv.setblocking(False)
    except OSError as e:
      srv.close()
      why = ('usb0 has no address yet' if e.errno == errno.EADDRNOTAVAIL else str(e))
      if why != self._last_error:
        self._last_error = why
        gadget.log.warning("jetlink: cannot listen for a phone on %s:%d (%s)", *gadget.CABLE_ADDR, why)
      return False
    self._srv = srv
    self.bound = srv.getsockname()
    self._last_error = ''
    gadget.log.warning("jetlink: listening for a phone on %s:%d", *self.bound[:2])
    return True

  def poll(self) -> str | None:
    """Take a dial if one is waiting. The peer's address when one arrived,
    else None. A held dial the phone has since dropped is let go here too,
    so a borrower is never handed a dead socket."""
    if self._srv is None:
      return None
    self._drop_if_dead()
    try:
      conn, addr = self._srv.accept()
    except (BlockingIOError, InterruptedError):
      return None
    except OSError:
      gadget.log.exception("jetlink: the cable listener failed")
      return None
    conn.setblocking(True)
    with self._lock:
      old, self._sock = self._sock, conn
      self.peer = str(addr[0])
      self.news, self.redial_expected = not self.redial_expected, False
    _close(old)
    return self.peer

  def _drop_if_dead(self) -> None:
    """A phone that closed its end. It sends nothing until a hello."""
    with self._lock:
      sock = self._sock
      if sock is None or not hung_up(sock):
        return
      self._sock, self.peer = None, None
    gadget.log.warning("jetlink: the phone hung up")
    _close(sock)

  def lend(self, conn: socket.socket, msg: dict) -> bool:
    """Send the held dial along with `msg` on the lend connection. False
    with nothing held. From the lender's thread."""
    with self._lock:
      sock = self._sock
      if sock is None:
        return False
      msg = {**msg, 'cable': True, 'peer': self.peer}
      socket.send_fds(conn, [json.dumps(msg).encode() + b'\n'], [sock.fileno()])
      return True

  def release(self, expect_redial: bool = False) -> None:
    """Close the held dial. The phone dials again and the next accept
    replaces it: a borrower that has finished, or died, must not leave the
    phone talking to nobody. `expect_redial` marks that next dial as ours to
    expect rather than a phone turning up."""
    with self._lock:
      sock, self._sock = self._sock, None
      self.peer = None
      self.redial_expected = expect_redial and sock is not None
    _close(sock)

  def close(self) -> None:
    self.release()
    srv, self._srv = self._srv, None
    self.bound = None
    _close(srv)


def _close(sock: socket.socket | None) -> None:
  if sock is not None:
    try:
      sock.close()
    except OSError:
      pass


def _shut(sock: socket.socket) -> None:
  """End the connection for both ends, so the owner sees the lease end."""
  try:
    sock.shutdown(socket.SHUT_RDWR)
  except OSError:
    pass


def hung_up(sock: socket.socket, wait: float = 0.0) -> bool:
  """Whether the far end closed `sock`, waiting up to `wait` s to see. Bytes
  waiting read as alive, and the peek leaves them for whoever shares the socket;
  a peer that sends nothing unasked is readable only at EOF."""
  try:
    # select first: a socket with a timeout would wait that long in recv
    if not select.select([sock], [], [], wait)[0]:
      return False
    return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
  except (BlockingIOError, InterruptedError, TimeoutError):
    return False   # a sharer took the bytes between the two
  except OSError:
    return True


class Lender:
  """The owner's side: one borrower at a time, for as long as it stays connected.

  `lendable` says whether the gadget is in the state a borrower can take over
  from, bound with no endpoint file open here; while it is not, a borrow is
  answered "retry" and the daemon's own loop puts it there. `holding` says the
  host is a phone (Accelerator Link iOS): "retry" until it dials, so nobody
  writes a hello over FunctionFS to a phone. With `cable` holding a dial, the
  loan carries the phone's socket instead of the endpoint files. `server`
  takes what a borrower passes on of the server's hello (Loan.note_server),
  with the borrower's name, on this thread.
  """

  def __init__(self, lendable: Callable[[], bool], bounce: Callable[[], bool],
               path: Path | None = None, holding: Callable[[], bool] | None = None,
               cable: CableListener | None = None, server: Callable[[str, dict], None] | None = None):
    self._lendable = lendable
    self._bounce = bounce
    self._holding = holding or (lambda: False)
    self._cable = cable
    self._server = server
    self._cable_lent = False
    # what this borrower was last told it has, so each change is logged once
    self._told = ''
    # as it is when made, for the same reason as borrow's
    self.path = SOCKET if path is None else path
    self.borrower = ''
    self._sock: socket.socket | None = None
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._lent = threading.Event()
    # why the last start could not listen, for the owner to report
    self.error: str | None = None

  @property
  def listening(self) -> bool:
    """Can anybody ask us for the endpoints? If not, nobody can use the link,
    and the owner says so and starts this again; see Owner.ensure_lender."""
    return self._sock is not None

  @property
  def lent(self) -> bool:
    """Is somebody using the endpoints? True from the first ask, not the first
    successful one: the daemon has to get off them before it can say yes."""
    return self._lent.is_set()

  def start(self) -> bool:
    """Listen for borrowers. False, with the reason in `error`, on a read-only
    /dev/shm or a path somebody else owns; the caller logs it and tries again,
    so this does not."""
    if self._thread is not None:
      return True
    sock = None
    try:
      self._clear_stale()
      sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      sock.bind(str(self.path))
      sock.listen(1)
      sock.settimeout(POLL)
    except OSError as e:
      if sock is not None:
        sock.close()
      self.error = str(e) or type(e).__name__
      return False
    self.error = None
    self._sock = sock
    self._thread = threading.Thread(target=self._serve, name='jetlink_lend', daemon=True)
    self._thread.start()
    return True

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(2.0)
      self._thread = None
    if self._sock is None:
      return   # never listened: the path, if any, is somebody else's
    self._sock.close()
    self._sock = None
    try:
      self.path.unlink(missing_ok=True)
    except OSError:
      pass

  def _clear_stale(self) -> None:
    """A socket file a dead daemon left behind. Proven dead by a connect that
    is refused, so a second owner cannot take the link from a live one."""
    if not self.path.exists():
      return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      probe.settimeout(0.5)
      probe.connect(str(self.path))
    except OSError:
      os.unlink(self.path)
    finally:
      probe.close()

  def _serve(self) -> None:
    while not self._stop.is_set():
      try:
        conn, _ = self._sock.accept()
      except (TimeoutError, OSError):
        continue
      try:
        self._handle(conn)
      except Exception:
        gadget.log.exception("jetlink: the borrower's connection failed")
      finally:
        conn.close()
        if self._lent.is_set():
          gadget.log.warning("jetlink: %s handed the %s back", self.borrower or 'the borrower',
                             'cable link' if self._cable_lent else 'gadget')
        # the phone's session ended with the borrower; let it dial again
        self._release_cable()
        self._told = ''
        self._lent.clear()
        self.borrower = ''

  def _release_cable(self) -> None:
    """Let a lent dial go, so the phone dials again: its session went with the
    borrower, or with the attempt a renewal follows. The owner's hold keeps a
    phone that has dialed from being lent the endpoint files meanwhile."""
    if self._cable_lent and self._cable is not None:
      self._cable.release(expect_redial=True)
    self._cable_lent = False

  def _tell(self, what: str, *msg) -> None:
    """Log a lend when it is not what this borrower already had: a renewal
    that changes nothing is not news."""
    if self._told != what:
      self._told = what
      gadget.log.warning(*msg)

  def _handle(self, conn: socket.socket) -> None:
    conn.settimeout(POLL)
    buf = bytearray()
    while not self._stop.is_set():
      try:
        msg = _recv_line(conn, buf, time.monotonic() + POLL)
      except (OSError, ValueError):
        return   # the borrower exited, was killed mid-drive, or is not one
      # None is only the poll coming round again; the lease ends on EOF, which
      # is what a modeld that manager stopped or SIGKILLed sends
      if msg is not None:
        self._answer(conn, msg)

  def _answer(self, conn: socket.socket, msg: dict) -> None:
    op = msg.get('op')
    if op == 'borrow':
      self.borrower = str(msg.get('name') or 'a borrower')
      self._lent.set()
      # a renewal: the dial went with the attempt that used it
      self._release_cable()
      if self._cable is not None and self._cable.held:
        # a phone: the link is its dial, and the endpoint files stay put
        if self._cable.lend(conn, {'ok': True, 'udc': gadget.bound_udc() or '', 'mount': str(gadget.FFS_MOUNT)}):
          self._cable_lent = True
          self._tell('cable', "jetlink: lending the cable link to %s (%s)", self.borrower, self._cable.peer)
          return
      if self._holding():
        _send(conn, {'ok': False, 'retry': True, 'detail': 'waiting for a phone to dial'})
        return
      udc = gadget.bound_udc()
      if not (udc and self._lendable()):
        # the daemon is mid-exchange, or has not bound yet. It sees `lent` on
        # its next cycle and puts the endpoints down for us
        _send(conn, {'ok': False, 'retry': True, 'detail': 'the gadget is still in use here'})
        return
      self._tell('gadget', "jetlink: lending the gadget to %s, udc %s", self.borrower, udc)
      _send(conn, {'ok': True, 'udc': udc, 'mount': str(gadget.FFS_MOUNT)})
    elif op == 'bounce':
      gadget.log.warning("jetlink: %s asked for a gadget bounce", self.borrower)
      _send(conn, {'ok': bool(self._bounce())})
    elif op == 'server' and self._server is not None:
      try:
        self._server(self.borrower or 'a borrower', {k: msg[k] for k in SERVER_FIELDS if k in msg})
      except Exception:
        # a note is never worth the lease: a raise here would end the loan
        # under a borrower that is using the endpoints
        gadget.log.exception("jetlink: could not record what the server said")
      _send(conn, {'ok': True})
    else:
      _send(conn, {'ok': False, 'detail': f'unknown op {op!r}'})
