"""The separate new price: its own recipe file, logged next to our price, read by one paper model and nothing else."""
import json
import time
from collections import deque

import pytest

from bcbot.auto import MODEL_REGISTERED, AutoTrader, _fade_fair_call
from bcbot.ourprice import MIX, MidFeed, OurPrice, load_fair_recipe, read_fair_recipe
from bcbot.trader import TradeError
from tests.test_safety import FakeTrader, make_cfg, trader_for, write_rounds


def test_our_price_class_takes_another_mix():
    t0 = 1_700_000_000.0
    flat, rising = MidFeed("binance", "", [], None), MidFeed("kraken", "", [], None)
    for j in range(300):                                   # 30 s at 0.1 s: one flat, one rising 1 cent per row
        flat.mids.append((t0 + j * 0.1, 70100.0))
        rising.mids.append((t0 + j * 0.1, 70200.0 + 0.01 * j))
    ticks = deque((t0 + j * 0.5, 70000.0) for j in range(60))
    new = OurPrice({"binance": flat, "kraken": rising}, ticks, mix={"binance": (0.25, 0.0), "kraken": (0.75, 0.0)},
                   gap_window=30.0)
    assert new.mix == {"binance": (0.25, 0.0), "kraken": (0.75, 0.0)} and new.gap_window == 30.0
    assert abs(new.move(t0 + 5, t0 + 15) - 0.75 * 1.0) < 1e-6          # only the rising exchange moved: $1 in 10 s
    assert abs(new.value(t0 + 29) - 70000) < 3                          # at bc.game's level, not at 70100 / 70200
    assert set(new.exchange_series(t0 + 5, t0 + 6)) == {"binance", "kraken"} and new.stale(t0 + 29) == []
    old = OurPrice({}, deque())
    assert old.mix is MIX and old.gap_window == 60.0                    # our price: the same recipe as before


def test_recipe_file_is_checked(tmp_path):
    p = tmp_path / "fair_price.json"
    assert load_fair_recipe(p) is None                                  # no file: no new price
    good = {"version": 3, "registered": 1794430000, "level_window": 45, "fitted_utc": "2026-11-11 21:00",
            "mix": {"binance_perp": [2, 0.0], "kraken": [1, 0.1], "okx_perp": [1, 0.05]}}
    p.write_text(json.dumps(good), encoding="utf-8")
    r = load_fair_recipe(p)
    assert r["version"] == 3 and r["registered"] == 1794430000 and r["level_window"] == 45
    assert r["mix"] == {"binance_perp": (0.5, 0.0), "kraken": (0.25, 0.1), "okx_perp": (0.25, 0.05)}
    assert read_fair_recipe(p) == (r, None) and r["fitted_utc"] == "2026-11-11 21:00"
    for note in (float("nan"), float("inf"), [float("nan")], {"r2": float("nan")}, 12, True):   # only text is kept:
        p.write_text(json.dumps({**good, "fitted_utc": note}), encoding="utf-8")                # the answers stay JSON
        assert load_fair_recipe(p)["fitted_utc"] is None, note
    p.write_text(json.dumps({**good, "fitted_utc": "x" * 5000}), encoding="utf-8")
    assert load_fair_recipe(p)["fitted_utc"] == "x" * 40
    p.write_text(json.dumps(good), encoding="utf-8-sig")                # saved by Notepad / PowerShell: a mark first
    assert load_fair_recipe(p) == r
    inf = float("inf")
    for bad in ({**good, "mix": {"nasdaq": [1, 0]}}, {**good, "mix": {"kraken": [-1, 0]}}, {**good, "mix": {"kraken": [1, 5]}},
                {**good, "mix": {}}, {**good, "version": 0}, {**good, "registered": 0}, {**good, "level_window": 5},
                {"version": 1}, [1, 2],
                {**good, "mix": {"kraken": [0, 0], "okx": [1, 0]}},                     # a fit leaves such an exchange out
                {**good, "mix": {"kraken": [inf, 0], "okx": [1, 0]}}, {**good, "mix": {"kraken": [1e308, 0], "okx": [1e308, 0]}},
                {**good, "mix": {"kraken": [float("nan"), 0]}}, {**good, "mix": {"kraken": "12"}},
                {**good, "mix": {"kraken": [True, 0]}}, {**good, "mix": {"kraken": [1, 0, 0]}},
                {**good, "version": inf}, {**good, "version": 1.9}, {**good, "version": "2"},
                {**good, "registered": 1794430000000}, {**good, "registered": inf}, {**good, "level_window": None}):
        p.write_text(json.dumps(bad), encoding="utf-8")
        recipe, why = read_fair_recipe(p)
        assert recipe is None and why, bad                              # refused, and it says why (the page shows it)
    assert read_fair_recipe(p)[1] == "level_window is not a number"
    for text in ("{not json", '{"version": 1' + "0" * 400 + "}", json.dumps(good).replace("1794430000", "1" + "0" * 400),
                 "[" * 100000 + "]" * 100000):                          # none of these may stop the bot from starting
        p.write_text(text, encoding="utf-8")
        recipe, why = read_fair_recipe(p)
        assert recipe is None and why, text[:40]
    p.write_text(json.dumps(good), encoding="utf-16")                   # PowerShell's '>'
    assert load_fair_recipe(p) is None
    p.unlink()
    assert read_fair_recipe(p) == (None, None)                          # no file is not a refusal


def _fill(a, now):
    """One minute of prices: bc.game around 70000 (+-$0.5 every tick), every exchange flat at its own level, and
    Kraken (only the new price reads it) $8 higher over the last 2 s."""
    a.bc_ticks.clear()
    a.bc_ticks.extend((now - 60 + 0.5 * j, 70000.0 + (0.5 if j % 2 else -0.5)) for j in range(121))
    for i, (k, f) in enumerate({**a.ex_feeds, **a.fair_feeds}.items()):
        f.mids.clear()
        for j in range(700):
            t = now - 70 + j * 0.1
            f.mids.append((t, 70050.0 + 10 * i + (8.0 if k == "kraken" and t >= now - 2 else 0.0)))
    a.our._gaps_at = 0.0
    if a.fair is not None:
        a.fair._gaps_at = 0.0


def test_new_price_is_logged_next_to_our_price_and_read_only_by_its_model(tmp_path):
    now = time.time()
    b = AutoTrader(FakeTrader(), make_cfg(tmp_path))                    # no recipe: no new price at all
    assert b.fair is None and b.fair_v == 0 and b.fair_info() is None and "fade_fair" not in MODEL_REGISTERED
    _fill(b, now)
    fb = b._features(now, {"cutoff": now + 2.9})
    assert fb["dq"]["stale"] is not None and "fair" not in fb and "dev_fair" not in fb and _fade_fair_call(fb) is None
    assert b.prices(60)["fair"] is None and b.summary()["fair_price"] is None
    assert "fade_fair" in AutoTrader.CALLERS                            # selectable, silent until there is a recipe

    (tmp_path / "fair_price.json").write_text(json.dumps(
        {"version": 2, "registered": now - 100, "mix": {"binance": [1, 0.0], "kraken": [1, 0.0]}}), encoding="utf-8")
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    assert a.fair is not None and a.fair_v == 2 and set(a.fair_feeds) == {"binance", "kraken"}
    assert set(a.ex_feeds) == set(MIX) and a.our.mix is MIX            # our price and its feeds: as they were
    assert MODEL_REGISTERED["fade_fair"] == now - 100
    _fill(a, now)
    fa = a._features(now, {"cutoff": now + 2.9})
    for k, v in fb.items():                                             # every input the other models read is the same
        if k not in ("dq", "clock"):
            assert fa[k] == v, k
    assert fa["fair_v"] == 2 and -5.5 < fa["dev_fair"] < -2.5 and abs(fa["dev_tick"]) <= 1.01      # Kraken ran $8 up
    assert 0.5 < fa["vol60"] < 1.5 and _fade_fair_call(fa) == "UP"
    rec = {"id": 1, "cutoff": now, "win": "UP", "tie": False, "feat": fa, "fair_move": 3.0, "fair_v": 2, "mix_move": -1.0,
           "mix_v": 3}
    assert a._model_calls(rec)["fade_fair"] == "UP"
    assert a._model_calls(dict(rec, feat=dict(fa, fair_v=1)))["fade_fair"] is None      # another recipe's round
    assert b._model_calls(rec)["fade_fair"] is None                                     # no recipe loaded: not scored
    a._count(rec)
    s = a.stats
    assert (s["fair_n"], s["fair_ok"], s["mix_n"], s["mix_ok"]) == (1, 1, 1, 0)
    assert (s["cmp_n"], s["cmp_fair_ok"], s["cmp_mix_ok"]) == (1, 1, 0) and s["fade_fair_fwd_called"] == 1
    a._count(dict(rec, id=2, fair_v=1))                                 # a round of an older recipe: not this one's
    assert s["fair_n"] == 1 and s["cmp_n"] == 1 and s["mix_n"] == 2
    p = a.prices(60)
    assert len(p["fair"]) > 100 and abs(p["fair"][-1][1] - 70004) < 1.5 and p["our_vs_bc"]["fair_n"] == 1
    info = a.summary()["fair_price"]
    assert info["version"] == 2 and info["mix"]["kraken"]["label"] == "Kraken" and info["same_rounds"]["n"] == 1
    a.fair_feeds["kraken"].last_msg_at, a.fair_feeds["binance"].last_msg_at = time.time(), time.time() - 30
    mix = a.fair_info()["mix"]                                          # no socket in a test: not live; fresh = a price
    assert (mix["kraken"]["fresh"], mix["binance"]["fresh"]) == (True, False) and not mix["kraken"]["live"]   # under 5 s old
    a.model = "fade_fair"
    a.on = True
    a.round = {"id": 1}
    a.recs[1] = {"call": "UP", "feat": fa}
    assert "from the new price" in a.panel_html()


def _recipe(tmp_path, version, mix):
    (tmp_path / "fair_price.json").write_text(json.dumps(
        {"version": version, "registered": time.time() - 100, "mix": mix}), encoding="utf-8")


def test_new_price_reads_an_exchange_without_a_time_stamp_on_our_prices_clock(tmp_path):
    """Kraken's messages carry no time, so its rows are placed with this PC's measured clock offset.  The new price
    must take that offset from our price (as the recording its recipe is fitted on does), not from whichever
    exchanges its own recipe happens to name: with only Kraken in the recipe it would otherwise be 0."""
    _recipe(tmp_path, 1, {"kraken": [1, 0.0]})
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    for f in a.ex_feeds.values():                                       # this PC's clock is 0.2 s ahead of the exchanges
        f.offsets.extend([0.2] * 10)
    assert a.fair_feeds["kraken"].offset() is None
    assert a.our.clock() == 0.2 and a.fair.clock() == 0.2 and a.fair._snapshot()["kraken"][2] == 0.2
    assert OurPrice(a.fair_feeds, a.bc_ticks, mix=a.fair.mix).clock() == 0.0       # what it was on its own
    assert a.our._clock is None                                         # our price: its own feeds, as before
    # while our price's clock is not known (its exchanges sent no time yet), the clock the recipe's own exchanges give
    _recipe(tmp_path, 2, {"kraken": [1, 0.0], "hyper_perp": [1, 0.0]})
    b = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    b.fair_feeds["hyper_perp"].offsets.extend([0.45] * 10)
    assert b.our.clock() == 0.0 and b.fair.clock() == 0.45 and b.fair._snapshot()["kraken"][2] == 0.45
    for f in b.ex_feeds.values():
        f.offsets.extend([0.2] * 10)
    assert b.fair.clock() == 0.2 and b.fair._snapshot()["hyper_perp"][2] == 0.45        # a stamped feed: its own


def test_a_new_price_without_prices_or_with_an_error_does_not_hold_up_the_chart(tmp_path):
    now = time.time()
    _recipe(tmp_path, 1, {"kraken": [1, 0.0]})
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    _fill(a, now)
    a.fair_feeds["kraken"].mids.clear()                                 # its only exchange never sent a price
    a.fair._gaps_at = 0.0
    worked_out, gaps = [], a.fair.gaps
    a.fair.gaps = lambda snap=None: (worked_out.append(1), gaps(snap))[1]
    p = a.prices(60)
    assert p["fair"] == [] and len(worked_out) == 1                     # once, not once for each of 300 points
    assert len(p["our"]) > 100 and p["fair_problem"] is None

    def broken(*_):
        raise RuntimeError("boom")
    a.fair.series = broken
    p = a.prices(60)
    assert p["fair"] == [] and len(p["our"]) > 100 and len(p["bc"]) > 100          # the rest of the chart is there


def test_a_refused_recipe_file_is_reported_not_taken_for_no_file(tmp_path):
    _recipe(tmp_path, 1, {"kraken": [0, 0.0], "okx": [1, 0.0]})
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    assert a.fair is None and a.fair_v == 0 and a.fair_problem == "the weight of kraken must be above 0"
    assert a.prices(60)["fair_problem"] == a.fair_problem and a.prices(60)["fair_info"] is None
    (tmp_path / "fair_price.json").unlink()
    assert AutoTrader(FakeTrader(), make_cfg(tmp_path)).fair_problem is None


def test_an_arming_on_the_new_price_ends_with_its_recipe(tmp_path):
    """The model on the new price is defined by the recipe file, and a refit is put in place by a restart.  Real
    orders armed on one recipe must not go on by themselves on another one (its record starts at zero), nor stay
    'armed' on no recipe; an arming on any other model is not touched by the recipe."""
    today = time.strftime("%Y-%m-%d")
    settings, state = tmp_path / "auto_settings.json", tmp_path / "live_state.json"

    def start():
        return trader_for(tmp_path, auto_resume_arm=True)

    def disarmed_by_restart(t):
        return [e["reason"] for e in t.events if e["event"] == "disarm"]

    write_rounds(tmp_path / "auto_rounds.jsonl", list("WL"), time.time() - 3600)
    settings.write_text(json.dumps({"model": "fade_fair"}), encoding="utf-8")
    a = start()                                                         # no recipe: the model cannot call
    a.on = True
    with pytest.raises(TradeError, match="no recipe"):
        a.arm(1, today, force=True)
    assert a.state == "DISARMED"
    _recipe(tmp_path, 2, {"binance_perp": [1, 0.1]})
    a = start()
    a.on = True
    a.arm(1, today, force=True)
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert a.state == "ARMED" and saved["model"] == "fade_fair" and saved["fair_v"] == 2
    b = start()                                                         # the same recipe: the arming goes on
    assert b.state == "ARMED" and not disarmed_by_restart(b)
    _recipe(tmp_path, 3, {"kraken": [1, 0.0]})                          # a refit
    c = start()
    assert c.state == "DISARMED" and c.fair_v == 3
    assert disarmed_by_restart(c) == ["the calling model changed (fade_fair, recipe 2 -> fade_fair, recipe 3)"]
    assert start().state == "DISARMED" and not disarmed_by_restart(start())        # said once
    c.on = True
    c.arm(1, today, force=True)                                         # the owner arms recipe 3
    (tmp_path / "fair_price.json").unlink()                             # ... and the file is gone at the next start
    d = start()
    assert d.state == "DISARMED" and d.fair is None
    assert disarmed_by_restart(d) == ["the calling model changed (fade_fair, recipe 3 -> fade_fair, recipe none)"]
    # the same version number over another recipe (a refit that forgot to raise it): not the recipe that was armed
    _recipe(tmp_path, 3, {"kraken": [1, 0.0]})
    f = start()
    f.on = True
    f.arm(1, today, force=True)
    assert json.loads(state.read_text(encoding="utf-8"))["fair_id"] == f.fair_id != 0 and start().state == "ARMED"
    _recipe(tmp_path, 3, {"kraken": [1, 0.0], "okx": [1, 0.0]})
    g = start()
    assert g.state == "DISARMED" and disarmed_by_restart(g) == [
        "the calling model changed (fade_fair, recipe 3 -> fade_fair, recipe 3 (the file is not the one armed))"]
    settings.write_text(json.dumps({"model": "fade"}), encoding="utf-8")           # another model: the recipe is
    _recipe(tmp_path, 4, {"kraken": [1, 0.0]})                                      # none of its business
    e = start()
    e.on = True
    e.arm(1, today, force=True)
    _recipe(tmp_path, 5, {"okx": [1, 0.0]})
    assert start().state == "ARMED"
    (tmp_path / "fair_price.json").unlink()
    assert start().state == "ARMED"
