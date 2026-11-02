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
