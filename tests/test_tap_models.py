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
    a.model_recent = {}
    a.show_model = "mid"
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
    assert grid["pick"] is None                                          # the live picker (mid) needs ticks
    assert grid["picks"] == {"paid": (4, 0), "paid-lo": (4, 0), "paid-calm": (4, 0),
                             "paid-far": (4, 0), "calm-label": (2, 0), "big-label": (4, 0),
                             "paid-r6": (4, 0), "label-r6": (2, 0), "label-r0": (2, 0), "best": (4, 0),
                             "best-lo": (4, 0)}
    assert sorted(c["model"] for c in a.paper_pending) == sorted(grid["picks"])
    a._pick_models(grid, 600.0, 102.0)                                   # the same columns again: nothing new
    assert len(a.paper_pending) == len(MODELS) - 4                       # mid, mid-lo, lr, lr-calm need ticks


def test_mid_band_pickers_need_a_quiet_mid_band_price(tmp_path):
    a = _analyzer(tmp_path)
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a.ticks = deque([(0.0, 102.0), (400.0, 102.0)])                      # price 102 in [100, 105): pos 0.4, no move
    a._pick_models(grid, 2500.0, 102.0)
    assert grid["picks"]["mid"] == (4, 0) and grid["picks"]["mid-lo"] == (2, 0)   # paid: 1.425 >= 1; label: A
    assert grid["pick"] == (4, 0)                                        # mid is the live picker
    a = _analyzer(tmp_path)
    a.ticks = deque([(0.0, 100.3), (400.0, 100.3)])                      # moved 0.34 band in 2 s: no
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 2500.0, 102.0)
    assert "mid" not in grid["picks"] and "mid-lo" not in grid["picks"]
    a = _analyzer(tmp_path)
    a.ticks = deque([(0.0, 104.6), (400.0, 104.6)])                      # at the top edge (pos 0.92): no
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 2500.0, 104.6)
    assert "mid" not in grid["picks"]


def test_calm_only_pickers_sit_out_a_wild_market(tmp_path):
    a = _analyzer(tmp_path, range_bands=6.0)
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 500.0, 102.0)
    assert "paid-calm" not in grid["picks"] and "calm-label" not in grid["picks"] and "paid" in grid["picks"]
    assert "paid-r6" not in grid["picks"] and "label-r6" not in grid["picks"]       # 6 bands is not under 6
    assert "best" not in grid["picks"] and "best-lo" not in grid["picks"] and grid["picks"]["label-r0"] == (2, 0)


def test_candidates_book_in_their_own_books(tmp_path):
    a = _analyzer(tmp_path)
    cell = {"k": 4, "r": 0, "odds": 4.0, "odds_src": "bc", "po": 0.5, "po_cal": 0.5, "ev": 1.5, "ev_emp": 1.9,
            "created": 0, "lo": 100.0, "hi": 105.0, "t0": 1000, "t1": 3000, "price": 102.0}
    a.paper_pending = [dict(cell, model="mid"), dict(cell, model="paid")]
    a.ticks = deque([(0, 102.0), (2000, 102.0), (4500, 102.0)])            # a HIT for both
    a._settle()
    assert a.stats["paper"]["bets"] == 1                                 # only the live picker feeds the paper book
    assert a.stats["models"]["mid"]["bets"] == 1 and a.stats["models"]["paid"]["bets"] == 1
    assert abs(a.stats["models"]["paid"]["pnl"] - (4.0 * 0.75 - 1.0)) < 1e-9   # at the paid odds
    assert json.loads(a.models_path.read_text(encoding="utf-8").splitlines()[0])["model"] == "paid"
    assert len(a.paper_path.read_text(encoding="utf-8").splitlines()) == 1
    names = [m["name"] for m in a.models_summary()]
    assert names[0] == "mid" and len(names) == len(MODELS)
    assert a.models_summary()[1]["hour_n"] == 1


def test_page_features_the_chosen_picker(tmp_path):
    a = _analyzer(tmp_path)
    a.show_model = "paid"                                                # feature a candidate, not the live picker
    a.cfg = {"timeIncrement": 2}
    a.ticks = deque([(1000.0, 102.0)])
    a.grid = {"cells": [], "pick": (2, 0), "picks": {"mid": (2, 0), "paid": (4, 0)}, "ev_bar": 1.35}
    cell = {"t1": 3000, "lo": 100.0, "hi": 105.0, "odds": 3.0, "odds_src": "bc"}
    a.paper_pending = [dict(cell, t0=1000, model="mid"), dict(cell, t0=5000, model="paid"),
                       dict(cell, t0=5000, model="big-label")]
    a.model_recent = {"paid": deque([{"t0": 0, "lo": 100.0, "hi": 105.0, "k": 3, "r": 0, "odds": 3.0, "hit": True,
                                      "pnl": 1.1}])}
    a.paper_recent = deque([{"t0": 0, "lo": 200.0, "hi": 205.0, "k": 3, "r": 0, "odds": 3.0, "hit": False,
                             "pnl": -1.0}])
    boxes = a.overlay_data()["cells"]
    picks = [b for b in boxes if b["cls"] == "pick"]
    assert len(picks) == 1 and picks[0]["t0"] == 5000 and picks[0]["label"].startswith("PAID")
    assert not any(b["t0"] == 1000 for b in boxes)                      # the live picker's own box is hidden
    assert [b["cls"] for b in boxes if b["cls"] in ("hit", "miss")] == ["hit"]   # paid's result, not the book's
    assert any(b["cls"] == "mpick" and b["label"] == "big-label" for b in boxes)
    a.show_model = "mid"                                                 # featuring the live picker: its box shows
    boxes = a.overlay_data()["cells"]
    assert [b["t0"] for b in boxes if b["cls"] == "pick"] == [1000]


def test_lr_pickers_follow_the_fitted_model(tmp_path, monkeypatch):
    import bcbot.tap as tap
    a = _analyzer(tmp_path)
    a.ticks = deque([(0.0, 102.0)] + [(100.0 + i * 100.0, 102.0) for i in range(3000)])   # 5 min flat at 102
    a.cfg = {"volatility": 1e-5}
    a.vol_real = 5e-6
    names = ["1", "pos_c", "pos_c^2", "|mv2|", "out_move", "|mv10|", "rng60", "rng300", "trail", "volr",
             "log step", "k", "|mv2|k", "pos_c k"]
    monkeypatch.setattr(tap, "LR_MODEL", {"names": names, "w": [0.0] * 14, "mu": [0.0] * 14, "sd": [1.0] * 14})
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 300100.0, 102.0)
    assert grid["picks"]["lr"] == (4, 0)                 # P = 0.5 everywhere: 0.5 x 3.8 x 0.75 = 1.43 >= 1.05
    assert grid["picks"]["lr-calm"] == (4, 0)            # flat 5 minutes: range 0
    monkeypatch.setattr(tap, "LR_MODEL", None)           # no model file: the lr pickers place nothing
    a = _analyzer(tmp_path)
    a.ticks = deque([(0.0, 102.0)] + [(100.0 + i * 100.0, 102.0) for i in range(3000)])
    a.cfg = {"volatility": 1e-5}
    a.vol_real = 5e-6
    grid = {"cells": _cells(), "pick": None, "picks": {}, "ev_bar": 1.35}
    a._pick_models(grid, 300100.0, 102.0)
    assert "lr" not in grid["picks"] and "lr-calm" not in grid["picks"]
