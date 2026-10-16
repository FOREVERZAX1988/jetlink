"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The swap events as flags, tick by tick. Ported from zoompilot's
test_accelerator_events.py, a test here for each of its counter tests under the
same name: offer is its bigModelAvailable, link_lost bigModelLinkLost, no_entry
bigModelLoading and chime bigModelReady. What runs openpilot itself (the state
machines, the alert texts, selfdrived's source) stays in the fork.
"""
import unittest

from jetlink.openpilot.events import Flags, SwapEvents
from jetlink.openpilot.interface import STATES

DT_CTRL = 0.01   # zoompilot's selfdrived
EVENTS = ('offer', 'link_lost', 'no_entry', 'chime')


class SwapEventsTest(unittest.TestCase):
  tick_s = DT_CTRL

  def setUp(self):
    self.swap = SwapEvents(self.tick_s)
    self.OFFER_TICKS, self.HANDBACK_TICKS, self.SWITCHING_TICKS = (self.swap.offer_ticks, self.swap.handback_ticks,
                                                                   self.swap.switching_ticks)

  def step(self, state='none', big=False, alive=True, fresh=True, enabled=False, mads=False) -> set[str]:
    """One tick, `alive` being modelV2's and `fresh` the state message's, wired
    as zoompilot's adapter reads them; what was raised, by name."""
    self.flags = self.swap.update(fresh and alive, state, big and alive, enabled or mads)
    return {name for name in EVENTS if getattr(self.flags, name)}

  def drive_big(self, **kwargs) -> None:
    """The big model swapped in and past its first second."""
    for _ in range(self.SWITCHING_TICKS + 1):
      self.step(state='running', big=True, **kwargs)

  def ticks_with(self, event: str, **kwargs) -> int:
    """How many ticks in a row `event` is raised from here on, up to a minute."""
    for n in range(6000):
      if event not in self.step(**kwargs):
        return n
    return 6000


class TestSettling(SwapEventsTest):
  def test_a_swap_in_settles_for_a_second(self):
    self.step()
    self.assertFalse(self.swap.settling)
    settled = [(self.step(state='running', big=True), self.swap.settling)[1] for _ in range(self.SWITCHING_TICKS + 5)]
    self.assertEqual(settled, [True] * self.SWITCHING_TICKS + [False] * 5)

  def test_a_hand_back_settles_for_a_second(self):
    self.drive_big()
    self.assertFalse(self.swap.settling)
    settled = [(self.step(state='retrying', big=False), self.flags.settling)[1] for _ in range(self.SWITCHING_TICKS + 5)]
    self.assertEqual(settled, [True] * self.SWITCHING_TICKS + [False] * 5)

  def test_the_flag_is_the_property(self):
    # zoompilot's selfdrived reads the property after the tick
    for state, big in [('none', False)] + [('running', True)] * 50 + [('retrying', False)] * 150:
      self.step(state=state, big=big)
      self.assertEqual(self.flags.settling, self.swap.settling)


class TestHandBack(SwapEventsTest):
  def test_a_fall_while_engaged_warns_for_five_seconds(self):
    self.drive_big(enabled=True)
    # the tick of the fall is the first of them
    self.assertEqual(self.step(state='retrying', enabled=True), {'link_lost'})
    self.assertEqual(1 + self.ticks_with('link_lost', state='retrying', enabled=True), self.HANDBACK_TICKS)
    self.assertEqual(self.HANDBACK_TICKS, 500)

  def test_a_fall_with_mads_lateral_paused_warns_once_it_steers(self):
    # paused is still in control, in zoompilot's meaning: MADS steers again on its own
    self.drive_big(mads=True)
    self.assertEqual(self.step(state='retrying', mads=True), {'link_lost'})

  def test_a_mads_only_fall_warns(self):
    # lateral only, cruise off: the 2026-09-29 drives lost the link three
    # times like this with no alert at all
    self.drive_big(mads=True)
    self.assertEqual(self.step(state='retrying', mads=True), {'link_lost'})
    self.assertEqual(1 + self.ticks_with('link_lost', state='retrying', mads=True), self.HANDBACK_TICKS)

  def test_a_disengage_ends_it_and_a_reengage_does_not_bring_it_back(self):
    self.drive_big(enabled=True)
    for _ in range(100):
      self.assertEqual(self.step(state='retrying', enabled=True), {'link_lost'})
    self.assertEqual(self.step(state='retrying'), set())
    self.assertEqual(self.step(state='retrying', enabled=True), set())

  def test_a_fall_while_disengaged_is_silent(self):
    self.drive_big()
    self.assertEqual(self.step(state='retrying'), set())
    # the fall was consumed: engaging afterwards does not replay it
    self.assertEqual(self.step(state='retrying', enabled=True), set())

  def test_a_chestnut_fall_is_not_the_adapters(self):
    # comma's board has its fork's own failure event for this, and nothing here adds to it
    self.step(state='none', big=True, enabled=True)
    self.assertEqual(self.step(state='none', enabled=True), set())

  def test_the_adapter_never_raises_big_model_failed(self):
    # upstream's text for that says to restart the car, which the 2026-09-29
    # drive showed after a Mac was unplugged while engaged. Nothing here
    # disengages: there is no failure flag, and a fall is link_lost alone
    self.assertEqual(Flags._fields, (*EVENTS, 'settling'))
    self.drive_big(enabled=True, mads=True)
    for state in [s for s in STATES if s != 'running']:
      self.assertLessEqual(self.step(state=state, enabled=True, mads=True), {'link_lost'})
      self.assertLessEqual(self.step(state=state, alive=False, enabled=True, mads=True), {'link_lost'})


class TestSwitching(SwapEventsTest):
  """For a second after a swap nothing engages: the big model starts from an
  empty history."""

  def test_a_second_of_no_entry_after_a_swap_then_the_chime(self):
    self.step(state='ready')
    self.assertEqual(self.step(state='running', big=True), {'no_entry'})
    self.assertEqual(1 + self.ticks_with('no_entry', state='running', big=True), self.SWITCHING_TICKS)
    self.assertEqual(self.SWITCHING_TICKS, 100)

  def test_the_chime_waits_for_the_end_of_the_second(self):
    self.step(state='ready')
    heard = ['chime' in self.step(state='running', big=True) for _ in range(self.SWITCHING_TICKS + 10)]
    self.assertEqual([i for i, h in enumerate(heard) if h], [self.SWITCHING_TICKS])

  def test_a_fall_inside_the_second_ends_it_without_the_chime(self):
    self.step(state='ready')
    for _ in range(10):
      self.step(state='running', big=True)
    self.assertEqual(self.step(state='retrying'), set())
    for _ in range(self.SWITCHING_TICKS):
      self.assertEqual(self.step(state='retrying'), set())

  def test_a_chestnut_never_switches_like_this(self):
    self.step(state='none')
    self.assertEqual(self.step(state='none', big=True), set())


class TestOffer(SwapEventsTest):
  """The offer to switch. The big model swaps in only while nothing is in
  control; ready while something is, the driver is told once to re-engage;
  ready while nothing is, it swaps at once and the chime says so."""

  def offered(self, available=False, big=False, alive=True, fresh=True, enabled=False, mads=False) -> bool:
    """One tick, and whether the offer is up."""
    state = 'ready' if available else ('running' if big else 'none')
    return 'offer' in self.step(state, big=big, alive=alive, fresh=fresh, enabled=enabled, mads=mads)

  def test_ready_while_engaged_offers_once_for_three_seconds(self):
    self.assertFalse(self.offered(enabled=True))
    self.assertEqual(self.ticks_with('offer', state='ready', enabled=True), self.OFFER_TICKS)
    for _ in range(1000):
      self.assertFalse(self.offered(available=True, enabled=True))

  def test_mads_alone_is_in_control(self):
    self.assertTrue(self.offered(available=True, mads=True))

  def test_ready_while_nothing_is_in_control_never_offers(self):
    # the swap happens at once, and the chime is what the driver hears
    for _ in range(100):
      self.assertFalse(self.offered(available=True))
    self.assertFalse(self.offered(big=True))

  def test_engaging_while_it_still_waits_offers(self):
    # it became ready in the gap before an engagement took the window away
    self.assertFalse(self.offered(available=True))
    self.assertTrue(self.offered(available=True, enabled=True))

  def test_no_repeat_at_every_stop(self):
    self.assertTrue(self.offered(available=True, mads=True))
    for _ in range(2 * self.OFFER_TICKS):
      self.offered(available=True, mads=True)
    for _ in range(3):
      self.assertFalse(self.offered(available=True, mads=True))

  def test_the_swap_ends_it_at_once(self):
    # 2026-09-29: the offer came back for ~0.9 s after the chime had expired,
    # reading as if the switch had not happened
    self.assertTrue(self.offered(available=True, mads=True))
    self.assertTrue(self.offered(available=True, mads=True))
    self.assertFalse(self.offered(big=True))
    for _ in range(self.OFFER_TICKS):
      self.assertFalse(self.offered(big=True))

  def test_on_screen_it_never_outlives_the_swap(self):
    # the fork's version runs the alert manager; the flag is gone from the swap's tick
    shown = [self.offered(available=frame < 50, big=frame >= 50, enabled=True) for frame in range(4 * self.OFFER_TICKS)]
    self.assertTrue(any(shown[:50]))
    self.assertFalse(any(shown[50:]))

  def test_it_rearms_after_the_link_goes_and_comes_back(self):
    self.assertTrue(self.offered(available=True, enabled=True))
    self.assertFalse(self.offered(big=True))   # swapped at a window, then...
    self.assertFalse(self.offered())           # lost, waiting to reconnect
    self.assertTrue(self.offered(available=True, enabled=True))

  def test_after_a_loss_while_engaged_it_waits_for_the_warning(self):
    # a join that held rejoins in a second, inside the 5 s link-lost warning,
    # which is the higher priority: an offer under it is never seen
    self.offered(big=True, enabled=True)
    self.assertFalse(self.offered(enabled=True))
    waited = 0
    while not self.offered(available=True, enabled=True):
      waited += 1
      self.assertLess(waited, 2 * self.HANDBACK_TICKS)
    self.assertEqual(waited, self.HANDBACK_TICKS - 1)
    self.assertEqual(self.ticks_with('offer', state='ready', enabled=True), self.OFFER_TICKS - 1)

  def test_chestnut_and_old_messages_do_not_announce_availability(self):
    # 'none' is the default of the fork's state field
    self.assertFalse(self.offered(enabled=True))
    self.assertFalse(self.offered(big=True, enabled=True))
    self.assertFalse(self.offered(enabled=True))

  def test_running_big_suppresses_a_pending_status_from_previous_frame(self):
    self.assertFalse(self.swap.update(True, 'ready', True, True).offer)

  def test_missing_invalid_or_stale_messages_never_announce(self):
    for wiring in ({'alive': False}, {'fresh': False}):
      with self.subTest(**wiring):
        self.assertFalse(self.offered(available=True, enabled=True, **wiring))
    self.assertTrue(self.offered(available=True, enabled=True))

  def test_stale_gap_does_not_repeat_the_offer(self):
    # the fork's version cuts the offer short by hand; here it runs out
    self.assertTrue(self.offered(available=True, enabled=True))
    for _ in range(self.OFFER_TICKS):
      self.offered(available=True, enabled=True)
    self.assertFalse(self.offered(fresh=False, enabled=True))
    self.assertFalse(self.offered(available=True, enabled=True))
    self.assertFalse(self.offered(enabled=True))  # an explicit loss rearms it
    self.assertTrue(self.offered(available=True, enabled=True))


class TestSeconds(SwapEventsTest):
  """At another loop rate, every event lasts the same seconds."""
  tick_s = 0.05

  def test_the_ticks_are_the_seconds_at_the_callers_rate(self):
    self.assertEqual((self.OFFER_TICKS, self.HANDBACK_TICKS, self.SWITCHING_TICKS), (60, 100, 20))

  def test_a_period_that_cannot_count_the_switch_is_refused(self):
    # at construction, never in the loop
    for tick_s in (0, -0.01, float('nan'), 2.0, 5.0):
      with self.subTest(tick_s=tick_s), self.assertRaises(ValueError):
        SwapEvents(tick_s)
    self.assertEqual(SwapEvents(1.0).switching_ticks, 1)

  def test_a_swap_is_a_second_of_no_entry_and_settling_then_the_chime(self):
    self.step(state='ready')
    raised = []
    for _ in range(25):
      raised.append((self.step(state='running', big=True), self.flags.settling))
    self.assertEqual(raised, [({'no_entry'}, True)] * 20 + [({'chime'}, False)] + [(set(), False)] * 4)

  def test_a_fall_warns_for_five_seconds_and_settles_for_one(self):
    self.drive_big(enabled=True)
    raised = [(self.step(state='retrying', enabled=True), self.flags.settling) for _ in range(110)]
    self.assertEqual(raised, [({'link_lost'}, True)] * 20 + [({'link_lost'}, False)] * 80 + [(set(), False)] * 10)

  def test_the_offer_is_three_seconds(self):
    self.assertEqual(self.ticks_with('offer', state='ready', enabled=True), 60)


if __name__ == '__main__':
  unittest.main()
