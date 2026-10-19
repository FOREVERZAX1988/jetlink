"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Where one frame's time went, in the same six segments every time: pickup, warp,
pair, encode, send, reply.

model_state's frame line carries four of them, one frame at a time, and it reads
them off the end of a wait: a frame that never waited -- a pipelined one, or one
the link could not take -- has no reply segment there at all, and that is
exactly the frame a demotion is made of. So the segments are kept here, per
frame, and finished when the reply is read, whichever frame reads it: a frame
that waited is answered in its own infer_end, a pipelined one by the drain of a
later frame (client._take_reply calls the hook this sets on the client). Nothing
here waits, sends, or decides anything; it is a record.

Bounded twice. A reply that comes back after RING frames is not timed: a host
answers in order, so only a link that came back from a hello gets that far
behind. The percentiles are over the last KEPT timed frames -- a drive's shape
rather than a sample of it, which is what the 30 s summary line is read for.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

# frames a slot is held for: four frames is the deadline at 20 Hz, and a host
# that answers in order overwrites a slot only with a frame newer than every
# reply it still owes
RING = 64
# frames the percentiles are over: three seconds at 20 Hz
KEPT = 64

# the six, in the order a frame spends them. The first five are marks the
# frame takes as it goes; reply is the one that is only known when the answer
# is read, and it is what the four-segment frame line cannot report at all
MEASURED = ('pickup', 'warp', 'pair', 'encode', 'send')
STAGES = (*MEASURED, 'reply')


def _ms(seconds: float | None) -> float | None:
  return None if seconds is None else round(seconds * 1e3, 1)


@dataclass
class Slot:
  """One frame's segments so far, in seconds; None where the segment has not
  happened (a frame with no lossless packer has no encode, the first frame of a
  session has nothing to be picked up after)."""
  pickup: float | None = None
  warp: float | None = None
  pair: float | None = None
  encode: float | None = None
  send: float | None = None
  frame_id: int = -1
  sent_at: float = 0.0   # perf_counter, when the frame was on the wire


class StageTimes:
  """The last RING frames' segments, finished as their replies come back, and
  the percentiles of the last KEPT that were."""

  def __init__(self, ring: int = RING, kept: int = KEPT):
    self._slots: list[Slot | None] = [None] * ring
    self._recent: deque[dict] = deque(maxlen=kept)
    self.frames = 0     # frames whose reply was read and timed
    self.untimed = 0    # replies read for a frame whose slot was already reused
    self.last: dict | None = None

  def start(self, pickup: float | None) -> Slot:
    """A slot for the frame being taken up. `pickup` is how long it was from
    the previous frame's end to this one's start: the camera and modeld's own
    publish, which the frame budget is spent out of before any of the rest."""
    return Slot(pickup=pickup)

  def track(self, slot: Slot, frame_id: int, sent_at: float) -> None:
    """The frame is on its way out under this id: the reply names the id, so
    that is what its segments are filed under."""
    slot.frame_id = frame_id
    slot.sent_at = sent_at
    self._slots[frame_id % len(self._slots)] = slot

  def answer(self, frame_id: int, at: float | None = None) -> dict | None:
    """The host's reply to this frame was read at `at` (perf_counter, default
    now): finish its segments and keep them. None when that frame is not in the
    ring any more, which is a link that fell far enough behind to be worth
    counting and not worth guessing at."""
    slot = self._slots[frame_id % len(self._slots)]
    if slot is None or slot.frame_id != frame_id:
      self.untimed += 1
      return None
    self._slots[frame_id % len(self._slots)] = None
    at = time.perf_counter() if at is None else at
    row = {'frame': frame_id}
    for name in MEASURED:
      row[f'{name}_ms'] = _ms(getattr(slot, name))
    # the round trip as the link made it, not as the wait ended: a frame that
    # published an older output still waited this long for its own answer
    row['reply_ms'] = _ms(max(0.0, at - slot.sent_at))
    self.frames += 1
    self.last = row
    self._recent.append(row)
    return row

  def summary(self) -> dict:
    """frames, untimed, then p50/p99/max in ms of each segment over the frames
    kept -- `-` where no kept frame had that segment at all."""
    out: dict = {'frames': self.frames, 'untimed': self.untimed}
    for name in STAGES:
      column = sorted(row[f'{name}_ms'] for row in self._recent if row[f'{name}_ms'] is not None)
      if column:
        out[name] = (column[len(column) // 2], column[int(0.99 * (len(column) - 1))], column[-1])
      else:
        out[name] = None
    return out


def text(values: dict) -> str:
  """One line of segments: the frame's own (`frame 42 pickup 50.1 ... reply
  43.5`, a `-` for a segment that did not happen) or a summary (`pickup
  50.0/50.4/73.2`, p50/p99/max)."""
  parts = []
  for key, value in values.items():
    if isinstance(value, tuple):
      value = '/'.join(f'{v:.1f}' for v in value)
    elif isinstance(value, float):
      value = f'{value:.1f}'
    elif value is None:
      value = '-'
    parts.append(f'{key.replace("_ms", "")} {value}')
  return ' '.join(parts)
