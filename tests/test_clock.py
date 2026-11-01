"""BcClock: the offset's sign, ok(), and the step check.  Run: .venv\\Scripts\\python.exe -m pytest tests -q"""
import time

from bcbot.clock import BcClock


def feed(clock: BcClock, offset: float, n: int = 60, jitter=None, end=None) -> None:
    """n ticks, one per 0.5 s, ending now: server time = arrival - offset (offset = PC minus bc.game)."""
    end = time.time() if end is None else end
    for i in range(n):
        arrival = end - 0.5 * (n - 1 - i)
        late = jitter(i) if jitter else 0.0
        clock.note(arrival - offset - late, arrival + late)   # a late tick arrives later than it was stamped


def test_offset_sign_pc_ahead():
    c = BcClock()
    feed(c, +0.7)                                   # this PC's clock is 0.7 s AHEAD of bc.game
    assert abs(c.offset() - 0.7) < 1e-6
    assert abs(c.now() - (time.time() - 0.7)) < 0.05
    assert c.ok()


def test_offset_sign_pc_behind():
    c = BcClock()
    feed(c, -0.7)                                   # this PC's clock is 0.7 s BEHIND bc.game
    assert abs(c.offset() + 0.7) < 1e-6
    assert abs(c.now() - (time.time() + 0.7)) < 0.05
    assert c.ok()


def test_deadline_on_bc_clock():
    """An order deadline compared on bc.game's clock: with the PC 0.7 s behind, time.time() says there is time
    left when bc.game's clock says the deadline has passed (and the other way round)."""
    behind = BcClock()
    feed(behind, -0.7)
    cutoff = behind.now() + 1.0                     # bc.game: the cutoff is 1.0 s away
    deadline = cutoff - 1.5                         # the send margin: already past on bc.game's clock
    assert behind.now() > deadline                  # -> TooLate
    assert time.time() < deadline + 0.3             # the PC clock would still have clicked
    ahead = BcClock()
    feed(ahead, +0.7)
    cutoff = ahead.now() + 2.0
    assert ahead.now() < cutoff - 1.5               # still allowed on bc.game's clock
    assert time.time() > cutoff - 1.5               # the PC clock would have skipped it


def test_not_ok_with_few_or_stale_ticks():
    c = BcClock()
    feed(c, 0.0, n=10)
    assert not c.ok() and "ticks" in c.why_not_ok()
    c = BcClock()
    feed(c, 0.0, end=time.time() - 3.0)             # fresh enough samples, but the last tick is 3 s old
    assert not c.ok() and "ago" in c.why_not_ok()


def test_late_ticks_do_not_move_the_median_much():
    c = BcClock()
    feed(c, 0.2, n=120, jitter=lambda i: 1.0 if i % 10 == 0 else 0.0)   # every 10th tick arrives 1 s late
    assert abs(c.offset() - 0.2) < 0.05
    assert c.ok()


def test_disagreement_without_exchange_move_is_not_a_step():
    c = BcClock()
    now = time.time()
    feed(c, 0.0, n=80, end=now - 20)                # 40 s of normal ticks ...
    feed(c, 1.0, n=40, end=now)                     # ... then 20 s of ticks 1 s "late" (a latency burst)
    assert not c.ok()                               # the medians disagree: not ok, as it should be
    c.check_step(0.0)                               # exchanges did not move
    c._disagree_since = now - 30                    # pretend it has persisted
    c.check_step(0.0)
    assert c.steps == 0 and c.stepped_at is None


def test_step_with_exchange_move_restarts_window():
    c = BcClock()
    now = time.time()
    feed(c, 0.0, n=80, end=now - 20)
    feed(c, 1.0, n=40, end=now)                     # the PC clock really jumped +1 s
    c.check_step(0.0)                               # first sight of the disagreement
    c._disagree_since = now - 30
    c.check_step(1.0)                               # the exchange clocks moved the same way
    assert c.steps == 1
    assert abs(c.offset() - 1.0) < 1e-6             # the window now holds only the post-step ticks
