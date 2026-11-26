"""The Up/Down call rules (the file keeps its first name: it began with the 'mom' model, removed on 2026-11-11)."""
import time

import pytest

from bcbot.auto import AutoTrader


def test_removed_models_cannot_be_chosen_or_scored(tmp_path):
    """Removed on 2026-11-11 after losing on their forward records: no entry in any list, and no call."""
    from bcbot.auto import MODEL_REGISTERED, REMOVED_MODELS
    from bcbot.trader import TradeError
    from tests.test_safety import make_cfg, FakeTrader
    assert set(REMOVED_MODELS) == {"mom", "momvol", "jump", "book", "bookmom", "mhh", "mhh_hi", "agree", "time", "gbm"}
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    rec = {"id": 1, "cutoff": time.time(), "win": "UP", "pred": {"ready": True, "sign_p_up": 0.6, "time_call": "UP"},
           "prob": 0.7, "prob_gbm": 0.7, "prob_ens": 0.7,       # gbm's P(UP) is still logged: the ensemble uses it
           "mhh": {"call": "UP", "conf": 0.9, "p_up": 0.9}, "feat": {"our_mom5": 2.0, "our_mom10": 20.0, "obi20": 0.9, "vol60": 3.0}}
    scored = a._model_calls(rec)
    for name in REMOVED_MODELS:
        assert name not in AutoTrader.MODELS and name not in AutoTrader.CALLERS and name not in MODEL_REGISTERED
        assert name not in scored and name not in a.ladders and name not in a.summary()["records"]
        with pytest.raises(TradeError):
            a.set_model(name)
    # HawkesSign is the six-market forecast now (the order-book service's answer): the old one-book forecast in
    # rec["pred"] is no bet of its any more.  Every selectable model and "pool" are scored
    assert scored["sign"] is None and set(scored) == set(AutoTrader.MODELS) | {"pool"}
    a._count(dict(rec, result=None))                       # an old round with their fields in it is counted without them
    assert not any(k.startswith(("mom_", "momvol_", "mhh_", "mhh_hi_", "agree_", "time_", "book_", "bookmom_", "jump_", "gbm_")) for k in a.stats)
    assert scored["prob"] == "UP" and scored["ens"] == "UP" and a.stats["ens_called"] == 1      # the ensemble goes on
    assert "MHH" not in a.panel_html() and "Momentum" not in a.panel_html()


def test_real_orders_resume_only_with_the_model_that_was_armed(tmp_path):
    """The owner's rule is that a restart keeps the arming.  The one exception: the calls are no longer made by the
    model that was armed (it was removed, so the bot fell back to HawkesSign): real orders by a model the owner did
    not choose must not start.  Once the owner arms the model that calls now, that arming is kept like any other."""
    import json
    from tests.test_safety import make_cfg, FakeTrader, trader_for, write_rounds
    settings, state = tmp_path / "auto_settings.json", tmp_path / "live_state.json"
    today = time.strftime("%Y-%m-%d")

    def saved(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def old_state_file(**change):                 # as written before 2026-11-11: it does not name the armed model
        st = saved(state)
        st.pop("model", None)
        st.update(change)
        state.write_text(json.dumps(st), encoding="utf-8")

    write_rounds(tmp_path / "auto_rounds.jsonl", list("WL"), time.time() - 3600)
    settings.write_text(json.dumps({"model": "fade"}), encoding="utf-8")
    a = trader_for(tmp_path, auto_resume_arm=True)
    a.on = True
    a.arm(1, today, force=True)
    assert a.state == "ARMED" and a.model == "fade" and saved(state)["model"] == "fade"
    assert trader_for(tmp_path, auto_resume_arm=True).state == "ARMED"          # the same model: the arming goes on
    settings.write_text(json.dumps({"model": "mom"}), encoding="utf-8")         # the armed model is gone
    b = trader_for(tmp_path, auto_resume_arm=True)
    assert b.model == "sign" and b.model_replaced == "mom" and b.state == "DISARMED"
    assert any(e["event"] == "disarm" and "fade -> sign" in e.get("reason", "") for e in b.events)
    assert saved(settings)["model"] == "sign" and "model_replaced" not in saved(settings)   # kept, not redone
    c = trader_for(tmp_path, auto_resume_arm=True)                              # the next restart: nothing to say
    assert c.model == "sign" and c.model_replaced is None and c.state == "DISARMED"
    assert not any(e["event"] == "disarm" for e in c.events)
    c.on = True
    c.arm(1, today, force=True)                                                 # the owner arms HawkesSign himself
    assert trader_for(tmp_path, auto_resume_arm=True).state == "ARMED"          # ... and that arming is kept
    # an old state file, no saved model (.env decides): a model that still exists resumes, a removed one does not
    settings.unlink()
    old_state_file()
    assert trader_for(tmp_path, auto_resume_arm=True, auto_model="sign").state == "ARMED"
    old_state_file()
    d = trader_for(tmp_path, auto_resume_arm=True, auto_model="mhh")
    assert d.model == "sign" and d.model_replaced == "mhh" and d.state == "DISARMED"
    assert not settings.exists()                  # no settings file is made: it would beat a later correction of .env
    assert trader_for(tmp_path, auto_resume_arm=True, auto_model="fade").model == "fade"
    # the fallback is remembered even when a setting is saved before the start check has run
    old_state_file(state="ARMED", armed_at=time.time())
    settings.write_text(json.dumps({"model": "jump"}), encoding="utf-8")
    e = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_resume_arm=True))      # this start dies before its check
    e.set_progression(1.5)                                                      # ... after saving a setting
    assert saved(settings)["model"] == "sign" and saved(settings)["model_replaced"] == "jump"
    g = trader_for(tmp_path, auto_resume_arm=True)
    assert g.model == "sign" and g.model_replaced == "jump" and g.state == "DISARMED"
    assert "model_replaced" not in saved(settings)
    # nothing is said when nothing would have resumed anyway (the switch is off)
    old_state_file(state="ARMED", armed_at=time.time())
    settings.write_text(json.dumps({"model": "book"}), encoding="utf-8")
    h = trader_for(tmp_path, auto_resume_arm=False)
    assert h.state == "DISARMED" and not any(ev["event"] == "disarm" for ev in h.events)
    # a settings file that cannot be written must not stop the start (the model is replaced again next time)
    settings.write_text(json.dumps({"model": "mom"}), encoding="utf-8")
    k = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_resume_arm=True))

    def locked():
        raise PermissionError(13, "Permission denied")
    k._save_settings = locked
    k._load_state()
    k._load_log()
    k._check_start()
    assert k.model == "sign" and k.state == "DISARMED" and any(ev["event"] == "start" for ev in k.events)
    assert saved(settings)["model"] == "mom"


def test_revert_quiet_only():
    from bcbot.auto import _revert_call
    assert _revert_call({"vol60": 1.0, "bc_mom2": 1.2}) == "DOWN"
    assert _revert_call({"vol60": 1.0, "bc_mom2": -0.8}) == "UP"
    assert _revert_call({"vol60": 1.0, "bc_mom2": 0.3}) is None       # move too small
    assert _revert_call({"vol60": 3.0, "bc_mom2": 1.2}) is None       # busy market
    assert _revert_call({}) is None


def test_overshoot_rules():
    from bcbot.auto import _overshoot_call
    # bc.game moved up $1 and sits $3 above the exchanges, quiet market: bet DOWN
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 3.0, "vol60": 1.0, "obi20": 0.0}) == "DOWN"
    assert _overshoot_call({"bc_mom2": -1.0, "dev_tick": -3.0, "vol60": 1.0, "obi20": 0.0}) == "UP"
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": -3.0, "vol60": 1.0}) is None    # exchanges moved too
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 3.0, "vol60": 3.0}) is None     # busy market
    # with the book: any market, book against the move
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 1.5, "vol60": 3.0, "obi20": -0.6}, with_book=True) == "DOWN"
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 1.5, "vol60": 3.0, "obi20": 0.6}, with_book=True) is None


def test_runs_today_counts_only_rounds_since_midnight(tmp_path):
    import time
    from tests.test_safety import make_cfg, FakeTrader
    from bcbot.auto import AutoTrader
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    now = time.time()
    lt = time.localtime(now)
    day0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    for won in (False, False, False, True):              # yesterday: a run of 3
        a._run_step("overshoot", won, day0 - 3600)
    for won in (False, True, False, False):              # today: a run of 1, then 2 still running
        a._run_step("overshoot", won, day0 + 60)
    assert a.runs["overshoot"]["lengths"] == {3: 1, 1: 1} and a.runs["overshoot"]["current"] == 2
    td = a.runs_today["overshoot"]
    assert td["lengths"] == {1: 1} and td["current"] == 2 and td["bets"] == 4
    yesterday = time.strftime("%Y-%m-%d", time.localtime(day0 - 3600))
    assert a.runs_days[yesterday]["overshoot"] == {"current": 0, "longest": 3, "bets": 4}
    a.reset_stats(who="test")                             # 'P&L -> 0' keeps the calendar-day tables
    assert a.runs == {} and a.runs_today["overshoot"]["bets"] == 4 and a.runs_days[yesterday]["overshoot"]["longest"] == 3
    assert yesterday in a.summary()["runs_days"]


def test_trees_mid_is_symmetric_and_only_in_medium_markets(monkeypatch):
    import bcbot.auto as a

    class Booster:
        def predict(self, X):                   # P(UP) from the first signed feature (dev), mirrored rows flip it
            return [0.5 + 0.1 * (1 if row[0] > 0 else -1) for row in X]
    order = list(a.UD_SIGNED) + list(a.UD_UNSIGNED) + ["pay_lean", "pool_total", "abs_dev", "abs_mom2", "same_way"]
    monkeypatch.setattr(a, "_ud_booster", lambda: {"booster": Booster(), "features": order})
    f = {"dev": 2.0, "vol60": 2.0, "pay_up": 1.94, "pay_dn": 1.94}
    assert a._treesmid_call(f) == "UP"                          # 0.6 x 1.94 = 1.16 >= 1.03
    assert a._treesmid_call(dict(f, dev=-2.0)) == "DOWN"        # the mirror image
    assert a._treesmid_call(dict(f, vol60=1.0)) is None         # calm
    assert a._treesmid_call(dict(f, vol60=3.5)) is None         # busy
    assert a._treesmid_call(dict(f, pay_up=1.5, pay_dn=1.5)) is None   # 0.6 x 1.5 = 0.9: no value at these payouts


def test_overshoot_rel_scales_with_the_volatility():
    from bcbot.auto import _overshoot_rel_call, OVERSHOOT_REL_K
    assert OVERSHOOT_REL_K == 2.0
    assert _overshoot_rel_call({"dev_tick": 2.5, "vol60": 1.0}) == "DOWN"     # bc.game $2.5 above the market
    assert _overshoot_rel_call({"dev_tick": -8.0, "vol60": 3.7}) == "UP"      # a busy market needs a bigger gap
    assert _overshoot_rel_call({"dev_tick": 5.0, "vol60": 3.7}) is None
    assert _overshoot_rel_call({"dev_tick": 2.5}) is None and _overshoot_rel_call(None) is None
    assert _overshoot_rel_call({"dev_tick": 2.5, "vol60": 0.0}) is None
    assert _overshoot_rel_call({"dev_tick": 2.5, "vol60": 1.0, "error": "x"}) is None


def test_fade_calls_from_one_volatility():
    from bcbot.auto import _fade_call, _overshoot_rel_call, FADE_K
    assert FADE_K == 1.0
    f = {"dev_tick": 4.5, "vol60": 3.7}
    assert _fade_call(f) == "DOWN" and _overshoot_rel_call(f) is None     # 1.2 x: fade calls, the 2 x rule does not
    assert _fade_call({"dev_tick": -4.5, "vol60": 3.7}) == "UP"
    assert _fade_call({"dev_tick": 3.0, "vol60": 3.7}) is None and _fade_call({}) is None
    assert "fade" in AutoTrader.CALLERS


def test_fade_strict_calls_from_2_45_volatilities_with_a_floor():
    """The outcome of the 2026-11-11 model search: the Gap fade's bet from 2.45 x max(vol60, $1.25)."""
    from bcbot.auto import _fade_call, _fade_strict_call, FADE_STRICT_K, FADE_VOL_FLOOR
    assert (FADE_STRICT_K, FADE_VOL_FLOOR) == (2.45, 1.25)
    assert _fade_strict_call({"dev_tick": 9.1, "vol60": 3.7}) == "DOWN"          # 2.46 x
    assert _fade_strict_call({"dev_tick": -9.1, "vol60": 3.7}) == "UP"
    assert _fade_strict_call({"dev_tick": 9.0, "vol60": 3.7}) is None            # 2.43 x: the plain fade calls, this one not
    assert _fade_call({"dev_tick": 9.0, "vol60": 3.7}) == "DOWN"
    assert _fade_strict_call({"dev_tick": 2.0, "vol60": 0.5}) is None            # 4 x a tiny volatility is a $2 gap: the floor
    assert _fade_strict_call({"dev_tick": 3.1, "vol60": 0.5}) == "DOWN"          # 2.45 x $1.25 = $3.06
    assert _fade_strict_call({}) is None and _fade_strict_call({"dev_tick": 9.1, "vol60": 3.7, "error": "x"}) is None
    assert _fade_strict_call({"dev_tick": 9.1, "vol60": 0}) is None
    assert "fade_strict" in AutoTrader.CALLERS


def test_every_caller_is_registered_scored_and_dispatched():
    """A model that can be selected must be able to call: trees_mid was once selectable but had no branch in
    _step, so choosing it made no call at all."""
    import inspect
    from bcbot.auto import MODEL_REGISTERED
    step = inspect.getsource(AutoTrader._step) + inspect.getsource(AutoTrader._hawkes)
    count = inspect.getsource(AutoTrader._model_calls)
    from bcbot.obfeat import OB_MODELS
    ob = inspect.getsource(AutoTrader._ob_fetch)             # the order-book models: called from the service's answer
    assert "self._ob_fetch(rid, rec)" in step and "if self.model in OB_MODELS:" in ob and "await self._call(rid, rec," in ob
    assert "for name, key in OB_ANSWER.items():" in count     # ... and scored from it (HawkesSign is one of them)
    for name in AutoTrader.CALLERS:
        assert name in AutoTrader.MODELS
        if name in OB_MODELS:
            continue
        assert f'"{name}"' in step, f"{name} can be selected but _step never calls it"
        assert f'"{name}":' in count, f"{name} is not scored"
    for name in ("fade", "fade_strict", "overshoot_rel", "trees_mid"):
        assert name in MODEL_REGISTERED


def test_pnl_reset_counts_from_now_and_keeps_the_forward_records(tmp_path):
    """'P&L -> 0': the shown record starts again and stays so after a restart; the round log, the gate's forward
    records and the real-loss counters are not touched."""
    import json, time
    from tests.test_safety import make_cfg, FakeTrader
    cfg = make_cfg(tmp_path)
    now = time.time()
    old = {"id": 1, "cutoff": now - 600, "win": "UP", "tie": False, "call": "UP", "result": "WIN", "payout": 1.94,
           "feat": {"dev_tick": -4.5, "vol60": 3.7}}               # the Gap fade calls UP here
    cfg.auto_log.write_text(json.dumps(old) + chr(10), encoding="utf-8")
    a = AutoTrader(FakeTrader(), cfg)
    a._load_log()
    assert a.stats["rounds"] == 1 and a.stats["called"] == 1 and a.stats["pnl"] > 0 and a.stats["fade_called"] == 1
    run, campaign = a.real_run, a.campaign_pnl
    a.reset_stats(who="test")
    assert a.stats["rounds"] == 0 and a.stats["called"] == 0 and a.stats["pnl"] == 0.0 and "fade_called" not in a.stats
    assert a.stats["fade_fwd_called"] == 1 and a.runs == {} and not a.settled
    assert (a.real_run, a.campaign_pnl) == (run, campaign)
    a._count(dict(old, id=2, cutoff=time.time() + 1, win="DOWN", result="LOSS", pnl=-1.0))       # a round after the reset
    assert a.stats["rounds"] == 1 and a.stats["fade_called"] == 1 and a.stats["fade_fwd_called"] == 2
    b = AutoTrader(FakeTrader(), cfg)                                                # a restart: the old round stays out
    b._load_log()
    assert b.stats_since == a.stats_since and b.stats["rounds"] == 0 and b.stats["fade_fwd_called"] == 1


def test_hawkes_forecast_stops_at_its_time_budget():
    """A busy order book made a forecast take 3-4.7 s and starve the round loop; it now stops simulating in time."""
    import time
    from collections import deque
    import numpy as np
    import bcbot.predictor as P

    class Feed:
        ev_time = deque([(0.0, 0.0)]); ev_sign = deque([(0.0, 0.0)]); bi = deque([(0.0, 0.2)])
        def now(self):
            return 1.0

    class SlowMHP:
        def excitation(self, t, typ, now):
            return None
        def simulate(self, E, horizon, rng):
            time.sleep(0.004)
            return [(1.0, P.EC), (2.0, P.EC_UP)]

    class Coe:
        def forecast(self, bt, bv, now, offsets):
            return np.full(len(offsets), 1e-6)

    p = P.HawkesPredictor(Feed())
    p.mhp_time = p.mhp_sign = SlowMHP()
    p.coe = Coe()
    old = P.PREDICT_BUDGET_S
    P.PREDICT_BUDGET_S = 0.3
    try:
        t0 = time.perf_counter()
        out = p.predict(1.0, 6.0)
        took = time.perf_counter() - t0
    finally:
        P.PREDICT_BUDGET_S = old
    assert out["ready"] and took < 1.0                             # 400 simulations would take 1.6 s or more
    assert all(P.MIN_SIMS <= n < P.N_SIMS for n in out["sims"])
    assert 0.0 <= out["sign_p_up"] <= 1.0 and out["time_call"] == "UP"
