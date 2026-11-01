"""bc.game's clock, measured from its own price ticks.

Every tick of bc.game's price feed carries bc.game's server time, and the bot notes when it arrived here.  The median
of (arrival - server) over the last minute is this PC's offset from bc.game: the PC clock drifts ~0.05 s/h against
it and jumps ~1 s about once a day, and the arrival jitter is one-sided (ticks arrive late, never early), which is
why a median and not a mean.  now() = time.time() - offset is bc.game's time; every trigger, order deadline and log
stamp is taken on it, never on this PC's clock.

The clock is "ok" while it has enough fresh ticks and its 20 s and 60 s medians agree.  It is never rejected for the
size of a stable offset (a 0.8 s offset is harmless once subtracted; on four days of ticks the offset was beyond
0.5 s a fifth of the time).  A real step of the PC clock shows as a lasting disagreement of the medians AND the same
move in the exchanges' clocks; then the window restarts at the step.  A disagreement without the exchanges moving is
a bc.game latency burst: the clock just stays not-ok until the medians agree again.
"""
import statistics
import time
from collections import deque

WINDOW_S = 60.0
MIN_SAMPLES = 20
MAX_TICK_AGE_S = 1.0
MED_DIFF_S = 0.3          # 20 s median vs 60 s median: beyond this the clock is not ok
STEP_S = 0.5              # 15 s median vs 60 s median beyond this, for STEP_PERSIST_S, with the exchange clocks
STEP_PERSIST_S = 10.0     # moved the same way: a PC clock step


class BcClock:
    def __init__(self, window_s: float = WINDOW_S, min_samples: int = MIN_SAMPLES,
                 max_tick_age_s: float = MAX_TICK_AGE_S, med_diff_s: float = MED_DIFF_S,
                 step_s: float = STEP_S, step_persist_s: float = STEP_PERSIST_S):
        self.window_s, self.min_samples = window_s, min_samples
        self.max_tick_age_s, self.med_diff_s = max_tick_age_s, med_diff_s
        self.step_s, self.step_persist_s = step_s, step_persist_s
        self.samples: deque = deque()          # (arrival, arrival - server), the last window_s of ticks
        self.last_server: float | None = None
        self.last_arrival: float | None = None
        self.ticks = 0
        self.stepped_at: float | None = None
        self.steps = 0
        self._disagree_since: float | None = None
        self._ex_at_disagree: float | None = None

    def note(self, server_s: float, arrival_s: float | None = None) -> None:
        """One bc.game tick: its server stamp and when it arrived (this PC's clock)."""
        arrival = time.time() if arrival_s is None else arrival_s
        self.samples.append((arrival, arrival - server_s))
        self.last_server, self.last_arrival = server_s, arrival
        self.ticks += 1
        cut = arrival - self.window_s
        while self.samples and self.samples[0][0] < cut:
            self.samples.popleft()

    def _median(self, seconds: float, now: float | None = None, least: int = 5) -> float | None:
        now = time.time() if now is None else now
        vals = [o for a, o in self.samples if a >= now - seconds]
        return statistics.median(vals) if len(vals) >= least else None

    def offset(self) -> float | None:
        """This PC's clock minus bc.game's (seconds), or None until there are enough ticks."""
        if len(self.samples) < self.min_samples:
            return None
        return statistics.median(o for _, o in self.samples)

    def now(self) -> float | None:
        """bc.game's time now, or None while the clock is unknown."""
        off = self.offset()
        return None if off is None else time.time() - off

    def tick_age(self) -> float | None:
        return None if self.last_arrival is None else time.time() - self.last_arrival

    def med_diff(self) -> float | None:
        """20 s median minus 60 s median: the clock's own consistency check."""
        m20, m60 = self._median(20.0), self.offset()
        return None if m20 is None or m60 is None else m20 - m60

    def ok(self) -> bool:
        if self.offset() is None:
            return False
        age = self.tick_age()
        if age is None or age > self.max_tick_age_s:
            return False
        d = self.med_diff()
        return d is not None and abs(d) <= self.med_diff_s

    def why_not_ok(self) -> str | None:
        if self.offset() is None:
            return f"only {len(self.samples)} ticks in the last {self.window_s:.0f} s"
        age = self.tick_age()
        if age is None or age > self.max_tick_age_s:
            return f"last tick {age:.1f} s ago" if age is not None else "no tick yet"
        d = self.med_diff()
        if d is None or abs(d) > self.med_diff_s:
            return f"medians disagree by {d:+.2f} s" if d is not None else "medians unknown"
        return None

    def check_step(self, ex_offset: float | None) -> None:
        """Call about once a second with the exchanges' offset (this PC's clock minus theirs, ourprice.clock()).
        A lasting disagreement of the medians that the exchange clocks share is a PC clock step: restart there."""
        m15, m60 = self._median(15.0), self.offset()
        now = time.time()
        if m15 is None or m60 is None or abs(m15 - m60) <= self.step_s:
            self._disagree_since = None
            return
        if self._disagree_since is None:
            self._disagree_since, self._ex_at_disagree = now, ex_offset
            return
        if now - self._disagree_since < self.step_persist_s:
            return
        if ex_offset is not None and self._ex_at_disagree is not None:
            moved = ex_offset - self._ex_at_disagree
            if abs(moved) > self.step_s and (moved > 0) == (m15 - m60 > 0):
                since = self._disagree_since
                self.samples = deque((a, o) for a, o in self.samples if a >= since)
                self.stepped_at, self.steps = now, self.steps + 1
                self._disagree_since = None

    def dict(self) -> dict:
        off, d, age = self.offset(), self.med_diff(), self.tick_age()
        return {"offset": round(off, 3) if off is not None else None,
                "med20_minus_med60": round(d, 3) if d is not None else None,
                "tick_age": round(age, 3) if age is not None else None,
                "n": len(self.samples), "ok": self.ok(), "why": self.why_not_ok(),
                "stepped_at": self.stepped_at, "steps": self.steps}
