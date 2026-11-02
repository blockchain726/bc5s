"""Candidate pickers run side by side on paper: each places at most one tap per column by its own score and bar;
"label" is the live picker and feeds the paper book, the others only their own books (and their own log)."""
import json
from collections import deque

from bcbot.tap import TapAnalyzer, MODELS


class Live:
    mode = "off"
    min_ev = 1.35
    ratio_boot = 1.0

    def ratio(self):
        return None

    def payout(self, k=None):
        return {2: 0.6, 4: 0.75}.get(k, 0.68)


def _analyzer(tmp_path, range_bands=2.0):
    a = TapAnalyzer.__new__(TapAnalyzer)                     # no browser: only what the pick and settle read
    a.live = Live()
    a.regime = {"name": "calm+hot", "factor": 1.0, "trailing": 0.5, "range_bands": range_bands}
    a.paper_pending = []
    a.pending = []
    a.band_hits = deque(maxlen=200)
    a.paper_recent = deque(maxlen=50)
    a.paper_path = tmp_path / "tap_paper.jsonl"
    a.models_path = tmp_path / "tap_models.jsonl"
    a.model_hour = {}
    a.stats = {"paper": {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0},
               "paper_strict": {"bets": 0, "wins": 0, "pnl": 0.0}, "models": {}}
    a.grid = None
    return a


def _cells():
    base = {"r": 0, "tappable": True, "odds_src": "bc", "po": 0.5, "po_cal": 0.5, "t1": 0}
    a = dict(base, k=2, t0=1000, lo=100.0, hi=105.0, odds=3.0, ev=1.5, ev_emp=2.0)   # label: best (2.0); paid 1.2
    b = dict(base, k=4, t0=5000, lo=100.0, hi=105.0, odds=3.8, ev=1.5, ev_emp=1.9)   # paid: 1.9 x 0.75 = 1.425
    return [a, b]


def test_each_picker_places_its_own_tap(tmp_path):
    a = _analyzer(tmp_path)
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 500.0, 102.0)
    assert grid["pick"] == (2, 0)                                        # the live picker: by the label EV
    assert grid["picks"] == {"label": (2, 0), "paid": (4, 0), "paid-lo": (4, 0), "paid-calm": (4, 0),
                             "paid-far": (4, 0), "calm-label": (2, 0), "big-label": (4, 0),
                             "paid-r6": (4, 0), "label-r6": (2, 0)}
    assert sorted(c["model"] for c in a.paper_pending) == sorted(grid["picks"])
    a._pick_models(grid, 600.0, 102.0)                                   # the same columns again: nothing new
    assert len(a.paper_pending) == len(MODELS)


def test_calm_only_pickers_sit_out_a_wild_market(tmp_path):
    a = _analyzer(tmp_path, range_bands=6.0)
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 500.0, 102.0)
    assert "paid-calm" not in grid["picks"] and "calm-label" not in grid["picks"] and "paid" in grid["picks"]
    assert "paid-r6" not in grid["picks"] and "label-r6" not in grid["picks"]       # 6 bands is not under 6


def test_candidates_book_in_their_own_books(tmp_path):
    a = _analyzer(tmp_path)
    cell = {"k": 4, "r": 0, "odds": 4.0, "odds_src": "bc", "po": 0.5, "po_cal": 0.5, "ev": 1.5, "ev_emp": 1.9,
            "created": 0, "lo": 100.0, "hi": 105.0, "t0": 1000, "t1": 3000, "price": 102.0}
    a.paper_pending = [dict(cell, model="label"), dict(cell, model="paid")]
    a.ticks = deque([(0, 102.0), (2000, 102.0), (4500, 102.0)])            # a HIT for both
    a._settle()
    assert a.stats["paper"]["bets"] == 1                                 # only the live picker feeds the paper book
    assert a.stats["models"]["label"]["bets"] == 1 and a.stats["models"]["paid"]["bets"] == 1
    assert abs(a.stats["models"]["paid"]["pnl"] - (4.0 * 0.75 - 1.0)) < 1e-9   # at the paid odds
    assert json.loads(a.models_path.read_text(encoding="utf-8").splitlines()[0])["model"] == "paid"
    assert len(a.paper_path.read_text(encoding="utf-8").splitlines()) == 1
    names = [m["name"] for m in a.models_summary()]
    assert names[0] == "label" and len(names) == len(MODELS)
    assert a.models_summary()[1]["hour_n"] == 1
