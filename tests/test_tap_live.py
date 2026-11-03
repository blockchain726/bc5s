"""Tap live: mode and stake switches, and the bootstrap ratio before the server/label ratio is measured."""
import pytest

from bcbot.tap_live import TapLive, MIN_AHEAD_MS


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


def test_mode_and_stake_switches(tmp_path, monkeypatch):
    monkeypatch.delenv("TAP_LIVE", raising=False)
    monkeypatch.setenv("TAP_STAKE", "0.2")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    assert lv.mode == "off" and lv.stake == 0.2
    with pytest.raises(ValueError):
        lv.set_mode("live")
    with pytest.raises(ValueError):
        lv.set_stake(0.01)
    with pytest.raises(ValueError):
        lv.set_stake(10)
    lv.set_mode("real"); lv.set_stake(0.5)
    assert lv.mode == "real" and lv.stake == 0.5
    events = (tmp_path / "tap_live.jsonl").read_text(encoding="utf-8")
    assert '"event": "mode"' in events and '"event": "stake"' in events
    lv.set_mode("off")
    assert lv._why_not(1_000_000.0) == "off"


def test_live_tap_is_the_analyzers_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("TAP_LIVE", "real")
    monkeypatch.setenv("TAP_STAKE", "0.2")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    assert lv._why_not(1_000_000.0) is None
    assert lv._choose(1_000_000.0) is None                   # the analyzer picked nothing this column
    lv.a.grid["pick"] = (2, 0)
    cell = lv._choose(1_000_000.0)                           # its pick, priced at what bc.game pays 2 columns ahead
    assert cell and abs(cell["ev_exp"] - 0.7 * 2.0 * lv.payout(2)) < 1e-6 and cell["ratio_used"] == lv.payout(2)
    lv.a.grid["pick"] = (3, 0)                               # a pick with no such cell: nothing
    assert lv._choose(1_000_000.0) is None


def test_live_gates_still_apply_to_the_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("TAP_LIVE", "real")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    lv.a.grid["pick"] = (2, 0)
    c = lv.a.grid["cells"][0]
    c["p_emp"] = None                                        # not a measured cell kind: no
    assert lv._choose(1_000_000.0) is None
    c["p_emp"] = 0.7
    lv.st["orders"]["x"] = {"t0": c["t0"]}                   # the column already has a live tap: no
    assert lv._choose(1_000_000.0) is None
    lv.st["orders"] = {}
    assert lv._choose(c["t0"] - MIN_AHEAD_MS) is None       # too close to click: no
    assert lv._choose(1_000_000.0) is not None


def test_stake_and_mode_from_the_page_survive_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("TAP_LIVE", "off"); monkeypatch.setenv("TAP_STAKE", "0.1")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    lv.set_stake(0.3); lv.set_mode("real")
    lv2 = TapLive(FakeAnalyzer(), tmp_path)                 # a restart with the same .env: the page's values win
    assert lv2.stake == 0.3 and lv2.mode == "real"
