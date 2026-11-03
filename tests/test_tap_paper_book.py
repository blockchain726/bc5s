"""The paper book prices a win at what bc.game's server pays (a fraction of the grid label), keeps the label P&L
alongside, and keeps a stricter book of the picks that would also clear the bar at the paid odds."""
import json
from collections import deque

from bcbot.tap import TapAnalyzer
from bcbot.tap_live import TapLive, PAYOUT_DEFAULT, PAYOUT_OTHER, MIN_AHEAD_MS


class Live:
    min_ev = 1.35

    def payout(self, k=None):
        return 0.7


class FakeAnalyzer:
    def __init__(self):
        self.ticks = [(1_000_000.0, 70000.0)]
        self.connected = True
        self.bc_read = {"n": 10}
        self.grid = {"cells": [{"t0": 1_000_000.0 + MIN_AHEAD_MS + 2000, "lo": 70000.0, "hi": 70010.0, "k": 2, "r": 0,
                                "tappable": True, "odds_src": "bc", "odds": 2.0, "po_cal": 0.7, "p_emp": 0.7}]}
        self.page = None

    def market_open(self):
        return True


def _analyzer(tmp_path):
    a = TapAnalyzer.__new__(TapAnalyzer)                     # no browser: only what _settle reads
    a.live = Live()
    a.stats = {"paper": {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0},
               "paper_strict": {"bets": 0, "wins": 0, "pnl": 0.0}}
    a.paper_recent = deque(maxlen=50)
    a.paper_path = tmp_path / "tap_paper.jsonl"
    a.pending = []
    a.band_hits = deque(maxlen=200)
    a.model_hour = {}
    a.model_recent = {}
    a.show_model = "mid"
    a.models_path = tmp_path / "tap_models.jsonl"
    return a


def test_paper_wins_are_booked_at_the_payout(tmp_path):
    a = _analyzer(tmp_path)
    cell = {"k": 3, "r": 0, "odds": 3.0, "odds_src": "bc", "po": 0.5, "po_cal": 0.5, "ev": 1.5, "ev_emp": 1.6,
            "created": 0}
    a.paper_pending = [dict(cell, lo=100.0, hi=105.0, t0=1000, t1=3000, price=102.0),             # inside: HIT
                       dict(cell, lo=200.0, hi=205.0, t0=1000, t1=3000, price=102.0, ev_emp=2.1)]  # far: MISS
    a.ticks = deque([(0, 102.0), (2000, 102.0), (4500, 102.0)])
    a._settle()
    p = a.stats["paper"]
    assert p["bets"] == 2 and p["wins"] == 1
    assert abs(p["pnl"] - ((3.0 * 0.7 - 1.0) - 1.0)) < 1e-9          # the hit pays label x payout, the miss -1
    assert abs(p["pnl_label"] - ((3.0 - 1.0) - 1.0)) < 1e-9          # what the label promised
    s = a.stats["paper_strict"]                 # 1.6 x 0.7 = 1.12 < 1.35: not strict; 2.1 x 0.7 = 1.47: strict
    assert s["bets"] == 1 and s["wins"] == 0 and s["pnl"] == -1.0
    recs = [json.loads(l) for l in a.paper_path.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["pay"] == 0.7 and recs[0]["pnl"] == 1.1 and recs[0]["pnl_label"] == 2.0
    assert recs[0]["strict"] is False and recs[1]["strict"] is True


def test_payout_is_the_default_until_measured(tmp_path, monkeypatch):
    monkeypatch.delenv("TAP_LIVE", raising=False)
    lv = TapLive(FakeAnalyzer(), tmp_path)
    assert lv.payout(2) == PAYOUT_DEFAULT[2] and lv.payout(9) == PAYOUT_OTHER
    for i in range(25):
        lv.st["orders"][str(i)] = {"k": 3, "ratio": 0.5, "result": "WIN"}
    assert lv.payout(3) == 0.5                               # measured three columns ahead
    assert lv.payout(2) == 0.5                               # too few of its own: every order's median
    assert lv.payout_table() == {"2": 0.5, "3": 0.5, "4": 0.5}


def test_old_label_priced_book_is_repriced(tmp_path):
    a = _analyzer(tmp_path)
    recs = [{"k": 3, "odds": 3.0, "hit": True, "ev": 1.6}, {"k": 3, "odds": 3.0, "hit": False, "ev": 1.6},
            {"k": 4, "odds": 4.0, "hit": True, "ev": 2.0}]
    a.paper_path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    a.stats["paper"] = {"bets": 3, "wins": 2, "pnl": 4.0}      # the old book: (3-1) - 1 + (4-1) = 4 at the label
    a._rebook_paper()
    p = a.stats["paper"]
    assert p["bets"] == 3 and p["wins"] == 2 and p["pnl_label"] == 4.0
    assert abs(p["pnl"] - ((3.0 * 0.7 - 1) - 1 + (4.0 * 0.7 - 1))) < 1e-9
    assert a.stats["paper_strict"] == {"bets": 1, "wins": 1, "pnl": round(4.0 * 0.7 - 1, 4)}   # only 2.0 x 0.7 >= 1.35
