"""The Tap picker trusts measured hit rates: a far cell with a high label but a low measured hit rate is never
picked over the current band; an unmeasured cell kind is never picked."""
from bcbot.tap import TapAnalyzer


def test_grid_picks_by_measured_hit_rate():
    now, price, gap, step = 1_000_000_000.0, 70000.0, 2000, 5.0
    hits = {(2, 0): 0.55, (2, 2): 0.10, (3, 0): 0.56}
    bc = {}
    first = (now - now % gap) + gap
    base = (price // step) * step
    for k in (2, 3):
        for r in (-2, -1, 0, 1, 2):
            bc[(first + k * gap, round(base + r * step, 2))] = 8.0 if abs(r) == 2 else 2.8 if r == 0 else 4.0
    g = TapAnalyzer._grid(now, price, gap, step, 3e-5, 3e-5, 8.0, bc_odds=bc, calib={}, hits=hits)
    cells = {(c["k"], c["r"]): c for c in g["cells"]}
    assert cells[(2, 0)]["p_emp"] == 0.55 and abs(cells[(2, 0)]["ev_emp"] - 0.55 * 2.8) < 1e-6
    assert cells[(2, 2)]["ev_emp"] == 0.8                     # 0.10 x 8: a loser however big the label
    assert cells[(2, 1)]["ev_emp"] is None                    # not measured
    best = max((c for c in g["cells"] if c["ev_emp"] is not None), key=lambda c: c["ev_emp"])
    assert (best["k"], best["r"]) == (3, 0)


def test_regime_factor_from_trailing_hits_and_range(tmp_path):
    from collections import deque
    from bcbot.tap import REGIME_FACTOR
    a = TapAnalyzer.__new__(TapAnalyzer)                     # no browser: only the two inputs the regime reads
    a.band_hits = deque([(i, i % 2 == 0) for i in range(60)])            # 50% trailing: hot
    t0 = 1_000_000_000
    a.ticks = deque([(t0 + i * 500, 70000.0 + (i % 4)) for i in range(130)])   # range $3 at $5 bands: calm
    r = a._regime(5.0)
    assert r["name"] == "calm+hot" and r["factor"] == REGIME_FACTOR[("calm", "hot")] and r["range_bands"] == 0.6
    a.ticks = deque([(t0 + i * 500, 70000.0 + 30 * (i % 2)) for i in range(130)])   # range $30 = 6 bands: wild
    a.band_hits = deque([(i, i % 5 == 0) for i in range(60)])            # 20% trailing: cold
    r = a._regime(5.0)
    assert r["name"] == "wild+cold" and r["factor"] == REGIME_FACTOR[("wild", "cold")]
    a.band_hits = deque()
    assert a._regime(5.0)["factor"] == 1.0                   # unknown: no change


def test_trailing_rate_caps_the_estimate():
    from collections import deque
    a = TapAnalyzer.__new__(TapAnalyzer)
    a.band_hits = deque([(i, i % 4 == 0) for i in range(60)])          # 25% trailing (a wild night)
    a.regime = {"factor": 0.76, "trailing": 0.25}
    est = a._shrunk(0.55, 5.0)                                          # history 55%: (30*0.55 + 60*0.25)/90 = 0.35
    assert abs(est - 0.35) < 1e-9 and est < 0.55 * 0.76
    a.band_hits = deque([(i, i % 5 != 0) for i in range(60)])          # 80% trailing (calm): the factor rules
    a.regime = {"factor": 1.20, "trailing": 0.8}
    assert abs(a._shrunk(0.55, 5.0) - 0.66) < 1e-9
    a.regime = {"factor": 1.0, "trailing": None}                        # unknown trailing: the long-run rate
    assert a._shrunk(0.55, 5.0) == 0.55
