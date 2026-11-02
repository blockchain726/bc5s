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


def test_real_taps_bootstrap_with_the_assumed_ratio(tmp_path, monkeypatch):
    monkeypatch.setenv("TAP_LIVE", "real")
    monkeypatch.setenv("TAP_STAKE", "0.2")
    monkeypatch.setenv("TAP_RATIO_BOOT", "0.85")
    monkeypatch.setenv("TAP_MIN_EV", "1.15")                  # the test's own settings, whatever .env says
    monkeypatch.setenv("TAP_USE_RATIO", "1")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    assert lv.ratio() is None
    assert lv._why_not(1_000_000.0) is None                 # no longer refused for the unknown ratio
    cell = lv._choose(1_000_000.0)                          # EV = 0.7 x 2.0 x 0.85 = 1.19 >= 1.15
    assert cell and abs(cell["ev_exp"] - 1.19) < 1e-6 and cell["ratio_used"] == 0.85
    lv.st["ratios"] = [0.5] * 5                             # measured: the server pays half the label
    assert lv.ratio() == 0.5 and lv._choose(1_000_000.0) is None


def test_live_skips_uncalibrated_far_cells(tmp_path, monkeypatch):
    monkeypatch.setenv("TAP_LIVE", "real")
    monkeypatch.setenv("TAP_MIN_EV", "1.15"); monkeypatch.setenv("TAP_USE_RATIO", "1"); monkeypatch.setenv("TAP_RATIO_BOOT", "0.85")
    lv = TapLive(FakeAnalyzer(), tmp_path)
    far = dict(lv.a.grid["cells"][0], r=3, lo=70015.0, hi=70020.0, odds=8.0, po_cal=0.3, p_emp=None)   # not measured
    lv.a.grid["cells"] = [far]
    assert lv._choose(1_000_000.0) is None
    far["p_emp"] = 0.05                                      # measured: hits 5% -> EV 0.4 x 0.85: no
    assert lv._choose(1_000_000.0) is None
    far["p_emp"] = 0.3                                       # measured 30% at x8 -> EV 2.0: yes
    assert lv._choose(1_000_000.0) is not None
