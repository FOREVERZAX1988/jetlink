"""The frame rate's governor: steps down on a host that holds, up on one that
keeps up, and never paces at the full 20 Hz cap."""
from __future__ import annotations

import pytest

from jetlink.openpilot import pace
from jetlink.openpilot.pace import CAPS, Governor


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
  monkeypatch.delenv('JETLINK_NO_ADAPTIVE_PACE', raising=False)


def test_never_paces_at_the_full_cap():
  """A camera's frames are the 20 Hz budget themselves: until a step down has
  happened, nothing is paced, jitter included."""
  g = Governor()
  t = 1000.0
  assert g.skip(t) is False
  assert g.skip(t + 0.01) is False   # 10 ms later: the budget, not a skip
  assert g.skip(t + 0.09) is False
  assert g.paced == 0


def test_holds_in_a_row_step_down_and_pacing_starts():
  g = Governor()
  t = 1000.0
  for _ in range(3):
    g.note(True, now=t)
  assert g.cap == CAPS[1]
  assert g.skip(t + 0.01) is True    # inside the 15 Hz window: paced
  assert g.skip(t + 0.067) is False  # past it: sent
  assert g.paced == 1


def test_continued_holds_reach_the_floor_and_no_lower():
  g = Governor()
  t = 1000.0
  for _ in range(9):
    g.note(True, now=t)
  assert g.cap == CAPS[-1]


def test_holds_across_the_window_step_down_without_a_run():
  g = Governor()
  t = 2000.0
  for i in range(8):
    g.note(True, now=t + i * 0.5)
    g.note(False, now=t + i * 0.5 + 0.01)  # each answered: never a run
  assert g.cap == CAPS[1]


def test_quiet_time_steps_back_up_one_level_at_a_time():
  g = Governor()
  t = 3000.0
  for _ in range(6):
    g.note(True, now=t)
  assert g.cap == CAPS[-1]
  # one quiet window: a level up, not two
  g.note(False, now=t + 11.0)
  assert g.cap == CAPS[-1]           # the first answered frame opens the window
  g.note(False, now=t + 22.0)
  assert g.cap == CAPS[1]
  g.note(False, now=t + 33.0)
  assert g.cap == CAPS[0]


def test_an_answered_frame_opens_the_quiet_window_again():
  g = Governor()
  t = 4000.0
  g.note(True, now=t)
  g.note(True, now=t + 0.1)
  g.note(False, now=t + 0.2)
  g.note(True, now=t + 9.0)          # inside the quiet window: nothing yet
  assert g.cap == CAPS[0]


def test_isolated_holds_no_longer_trap_the_cap_at_the_floor():
  """Regression: a phone that stepped down while the link was bad must climb
  back once it recovers. A stray hold every couple of seconds is normal on a
  Wi-Fi link; the old rule restarted a 10 s quiet clock on every one of them,
  so the governor never stepped up (2026-11-07: 224 downs, zero ups)."""
  g = Governor()
  t = 6000.0
  for _ in range(6):
    g.note(True, now=t)
  assert g.cap == CAPS[-1]
  # 30 s of healthy 20 Hz frames, with one isolated late frame every 2 s
  now = t
  for i in range(600):
    now += 0.05
    g.note(i % 40 == 0, now=now)
  assert g.cap == CAPS[0]


def test_a_run_of_holds_still_restarts_the_quiet_wait():
  """Sustained trouble must not be papered over by the stray allowance."""
  g = Governor()
  t = 7000.0
  for _ in range(6):
    g.note(True, now=t)
  assert g.cap == CAPS[-1]
  g.note(False, now=t + 1.0)       # open the quiet window
  g.note(True, now=t + 2.0)
  g.note(True, now=t + 2.1)        # a run, not jitter: clock restarted
  for i in range(1, 240):
    g.note(False, now=t + 2.1 + i * 0.05)
  assert g.cap == CAPS[1]          # at most one level, not back to 20 Hz


def test_the_env_switch_turns_the_governor_off(monkeypatch):
  monkeypatch.setenv('JETLINK_NO_ADAPTIVE_PACE', '1')
  import importlib
  importlib.reload(pace)
  try:
    g = pace.Governor()
    t = 5000.0
    for _ in range(6):
      g.note(True, now=t)
    assert g.cap == pace.CAPS[0]
    assert g.skip(t + 0.01) is False
  finally:
    monkeypatch.delenv('JETLINK_NO_ADAPTIVE_PACE', raising=False)
    importlib.reload(pace)
