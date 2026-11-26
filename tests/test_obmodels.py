"""The order-book models (logistic regression / LightGBM / XGBoost / DeepLOB on the books and trades of the six
markets of our price): their inputs (bcbot/obfeat.py) and how the bot scores them (scored only, per fitted version)."""
import asyncio
import time

import numpy as np
import pytest

from bcbot import obfeat as F
from bcbot.auto import AutoTrader
from bcbot.ourprice import MIX
from bcbot.trader import TradeError
from tests.test_safety import FakeTrader, make_cfg


def rows(t0, n, bid=100.0, ask=100.2, bid_size=1.0, ask_size=1.0, drift=0.0, buy=0.0, sell=0.0, age=0.05):
    """n rows 100 ms apart ending at t0: a flat book of 10 levels a side, the price moving `drift` $ a row."""
    a = np.zeros(n, dtype=F.ROW)
    for i in range(n):
        a[i]["t"] = t0 - (n - 1 - i) * F.STEP
        a[i]["bid"], a[i]["ask"] = bid + drift * i, ask + drift * i
        a[i]["bp"] = a[i]["ap"] = np.arange(F.LEVELS) * 0.1
        a[i]["bs"], a[i]["as_"] = bid_size, ask_size
        a[i]["buy"], a[i]["sell"], a[i]["age"] = buy, sell, age
    return a


def test_the_markets_and_shares_are_those_of_our_price():
    assert F.SHARES == {k: w for k, (w, _) in MIX.items()}
    assert len(F.NAMES) == len(F.SHARES) * len(F.PER) + 11 + 2 == 85
    assert [n for n, s in zip(F.NAMES, F.SIGN) if s > 0] == [f"{k}_spread" for k in F.SHARES] + ["vol"]


def test_inputs_of_one_market_have_the_sign_of_what_they_say():
    t = 1_000_000.0
    col = {n: i for i, n in enumerate(F.PER)}
    x = F.market_features(rows(t, 150, bid_size=3.0, ask_size=1.0, drift=0.01, buy=0.2, sell=0.1), [t], [2.0])[0]
    assert x[col["imb1"]] == pytest.approx(0.5) and x[col["imb10"]] == pytest.approx(0.5)      # more bids than asks
    assert x[col["press1"]] == pytest.approx(1 / 3) and x[col["press10"]] == pytest.approx(1 / 3)   # more bought than sold
    assert x[col["ret1"]] == pytest.approx(0.01 * 10 / 2.0) and x[col["ret10"]] == pytest.approx(0.01 * 100 / 2.0)
    assert x[col["spread"]] == pytest.approx(0.2 / 2.0, abs=1e-6) and x[col["dimb5"]] == pytest.approx(0.0)
    quiet = F.market_features(rows(t, 150), [t], [2.0])[0]
    assert quiet[col["imb1"]] == 0 and quiet[col["press3"]] == 0 and quiet[col["ret3"]] == pytest.approx(0)   # no trade: 0


def test_a_silent_or_missing_book_is_unknown_not_balanced():
    t = 1_000_000.0
    assert np.isnan(F.market_features(rows(t, 150, age=5.0), [t], [2.0])).all()            # the book stopped updating
    assert np.isnan(F.market_features(rows(t - 2.0, 150), [t], [2.0])).all()               # the last row is 2 s old
    assert np.isnan(F.market_features(np.zeros(0, dtype=F.ROW), [t], [2.0])).all()         # nothing recorded
    half = rows(t, 150); half["bid"][-1] = np.nan
    assert np.isnan(F.market_features(half, [t], [2.0])).all()                             # no book in the last row


def test_the_whole_row_mixes_by_share_and_mirrors():
    t = 1_000_000.0
    books = {k: rows(t, 150, bid_size=3.0, ask_size=1.0) for k in F.SHARES}
    X = F.build(books, [t], [3.0], [2.0])
    name = {n: i for i, n in enumerate(F.NAMES)}
    assert X.shape == (1, 85) and not np.isnan(X).any()
    assert X[0, name["mix_imb1"]] == pytest.approx(0.5) and X[0, name["gap"]] == pytest.approx(1.5)
    M = F.mirror(X)
    assert M[0, name["mix_imb1"]] == pytest.approx(-0.5) and M[0, name["gap"]] == pytest.approx(-1.5)
    assert M[0, name["okx_spread"]] == X[0, name["okx_spread"]] and M[0, name["vol"]] == X[0, name["vol"]]
    # one market against the rest: the mix follows the shares (Bitfinex futures 43.3%)
    books["bitfinex_perp"] = rows(t, 150, bid_size=1.0, ask_size=3.0)
    assert F.build(books, [t], [0.0], [2.0])[0, name["mix_imb1"]] == pytest.approx(0.5 * (1 - 2 * 0.433))
    # markets with less than half the share known: no mixed input
    few = {"coinbase": books["coinbase"], "okx": books["okx"]}
    assert np.isnan(F.build(few, [t], [0.0], [2.0])[0, name["mix_imb1"]])


def test_a_model_calls_only_when_it_is_far_enough_from_50():
    assert F.side(0.53, 0.02) == "UP" and F.side(0.47, 0.02) == "DOWN" and F.side(0.51, 0.02) is None
    assert F.side(None, 0.02) is None and F.side(0.6, None) is None and F.side(float("nan"), 0.02) is None
    m = {"idx": [0, 1], "sd": [1.0, 2.0], "w": [0.5, -0.25]}
    X = np.array([[2.0, 4.0, 9.0], [np.nan, 0.0, 1.0]])
    assert F.logit_predict(m, X) == pytest.approx([1 / (1 + np.exp(-0.5)), 0.5])            # missing input = 0
    assert F.tree_predict(lambda Z: np.full(len(Z), 0.7), np.zeros((2, 85))) == pytest.approx([0.5, 0.5])


def test_the_order_book_models_bet_in_every_round():
    """The owner's choice (2026-11-16: "I was thinking that will bet every round"): whatever the fit's level and the
    model's earlier answers, a model bets the side its P(UP) is on; only an answer of exactly 50% is no bet."""
    assert F.BET_SHARE == 1.0
    for past, fit_thr in (([], 0.0777), ([0.05] * 240, 0.019), ([], None)):
        level = F.live_level(past, fit_thr)
        assert level == F.MIN_THR
        assert F.side(0.50003, level) == "UP" and F.side(0.49998, level) == "DOWN" and F.side(0.5, level) is None


def test_a_share_under_one_follows_the_models_own_answers():
    """With a share under 1 (0.25 until the owner chose every round): the fit's level let LightGBM bet in 8% and
    DeepLOB in 54% of the rounds, so the level is the one that the surest share of the model's EARLIER answers
    reached, whatever the spread of its answers."""
    rng = np.random.default_rng(3)
    q = dict(share=0.25, min_thr=0.005)
    share = {}
    for spread in (0.002, 0.02, 0.09):                                # a model that says 50%, a middling one, a loud one
        far = np.abs(rng.normal(0, spread, 4000))
        bets = sum(far[i] >= F.live_level(far[i - F.LEVEL_N:i], 0.5, **q) for i in range(F.LEVEL_N, len(far)))
        share[spread] = bets / (len(far) - F.LEVEL_N)
    assert 0.22 < share[0.02] < 0.28 and 0.22 < share[0.09] < 0.28
    assert share[0.002] < 0.05                                        # half a point from 50% is still asked for
    assert F.live_level([0.01] * (F.LEVEL_MIN - 1), 0.033, **q) == 0.033              # too few answers: the fit's level
    assert F.live_level([], None, **q) is None                                        # ... and none: no bet (F.side)
    assert F.live_level([0.0001] * 100, 0.033, **q) == 0.005                          # never under the least level
    assert F.live_level(list(np.linspace(0, 0.1, 101)), 0.5, **q) == pytest.approx(0.075)


def settled(i, win, ob, t0):
    t = t0 + 30 * i
    return {"id": 9000 + i, "cutoff": t, "start": t, "end": t + 5, "win": win, "tie": False, "payouts": [1.94, 1.94],
            "pools": [100.0, 100.0], "ob": ob, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))}


def test_the_bot_scores_the_order_book_models_per_fitted_version(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    t0 = time.time() - 600
    v1 = {"v": 1, "p": {"ob_logit": 0.53, "ob_lgbm": 0.51, "ob_xgb": 0.44}, "thr": {"ob_logit": 0.02, "ob_lgbm": 0.02, "ob_xgb": 0.02}}
    a._count(settled(0, "UP", v1, t0))
    calls = a._model_calls(settled(0, "UP", v1, t0))
    assert (calls["ob_logit"], calls["ob_lgbm"], calls["ob_xgb"], calls["ob_deeplob"]) == ("UP", None, "DOWN", None)
    s = a.stats
    assert (s["ob_logit_called"], s["ob_logit_correct"]) == (1, 1) and (s["ob_xgb_called"], s["ob_xgb_correct"]) == (1, 0)
    assert s["ob_logit_pnl"] == pytest.approx(0.94) and s["ob_xgb_pnl"] == pytest.approx(-1.0) and "ob_lgbm_called" not in s
    # a round answered by a newer fit: the records start again, and rounds of the old fit no longer count
    v2 = dict(v1, v=2, p={"ob_logit": 0.40, "ob_lgbm": 0.60, "ob_xgb": 0.50})
    a._count(settled(1, "DOWN", v2, t0))
    assert a.ob_v == 2 and (s["ob_logit_called"], s["ob_logit_correct"]) == (1, 1) and "ob_xgb_called" not in s
    assert (s["ob_lgbm_called"], s["ob_lgbm_correct"]) == (1, 0)
    assert a._model_calls(settled(0, "UP", v1, t0))["ob_logit"] is None
    # no answer from the service, or no fit yet: no call
    assert a._model_calls(settled(2, "UP", None, t0))["ob_logit"] is None
    assert a._model_calls(settled(2, "UP", {"v": 0, "p": {}, "thr": {}}, t0))["ob_lgbm"] is None
    out = a.summary()                                        # what the dashboard reads: a row for each of them
    assert all(k in out["records"] for k in F.OB_MODELS) and out["ob"]["counted_version"] == 2
    assert out["records"]["ob_logit"] == {"called": 1, "correct": 1, "pnl": 0.94}


def test_hawkes_sign_is_the_six_market_model_and_its_record_starts_again(tmp_path):
    """HawkesSign ("sign") is the six-market forecast of the order-book service (ob_hawkes in its answer) since
    SIGN_SIX_SINCE.  The owner had its P&L and history started again with the change: no earlier round counts for
    it.  It is fitted again every 20 s by design, so a new fit of the other order-book models does not start its
    record again, and it is scored without any fit in place."""
    from bcbot.auto import MODEL_REGISTERED, SIGN_SIX_SINCE
    assert MODEL_REGISTERED["sign"] == SIGN_SIX_SINCE == 1794843360 and F.OB_ANSWER["sign"] == "ob_hawkes"
    assert "sign" in F.OB_MODELS and "sign" not in F.OB_FITTED and "ob_hawkes" not in AutoTrader.MODELS
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    thr = {"ob_logit": 0.02, "ob_hawkes": F.MIN_THR}
    old = settled(0, "UP", {"v": 1, "p": {"ob_hawkes": 0.6}, "thr": thr}, SIGN_SIX_SINCE - 3600)
    old["pred"] = {"ready": True, "sign_p_up": 0.7}              # the one-book forecast of the old HawkesSign
    assert a._model_calls(old)["sign"] is None                   # before the change: nothing counts
    a._count(old)
    assert a.stats.get("sign_called", 0) == 0 and "sign" not in a.runs
    t0 = SIGN_SIX_SINCE + 60
    a._count(settled(0, "UP", {"v": 0, "p": {"ob_hawkes": 0.515}, "thr": thr}, t0))             # no fit yet
    a._count(settled(1, "UP", {"v": 1, "p": {"ob_logit": 0.53, "ob_hawkes": 0.48}, "thr": thr}, t0))
    a._count(settled(2, "DOWN", {"v": 2, "p": {"ob_logit": 0.40, "ob_hawkes": 0.5}, "thr": thr}, t0))   # 50%: no bet
    a._count(settled(3, "DOWN", {"v": 2, "p": {"ob_logit": 0.40, "ob_hawkes": 0.31}, "thr": thr}, t0))
    a._count(dict(settled(4, "UP", None, t0), pred={"ready": True, "sign_p_up": 0.9}))          # no answer: no bet
    s = a.stats
    assert (s["sign_called"], s["sign_correct"]) == (3, 2) and (s["sign_fwd_called"], s["sign_fwd_correct"]) == (3, 2)
    assert (s["ob_logit_called"], s["ob_logit_correct"]) == (2, 2)                              # only the rounds of fit 2
    out = a.summary()
    assert out["records"]["sign"]["called"] == 3 and "ob_hawkes" not in out["records"] and out["sign_accuracy"] == 0.6667
    a.set_model("sign")
    assert "since it reads the six markets' order books (2026-11-16 15:36 UTC)" in a.gate_status()["text"]


def test_the_order_book_models_can_be_selected_and_make_the_calls(tmp_path):
    """The owner looked for them in the page's Model list (2026-11-16).  The chosen one's call is made as soon as the
    service's answer is in; without an answer there is no bet and the round says why."""
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url="http://127.0.0.1:1/ob", auto_stake_usd=1.0))
    a.on = True
    a.bc_now = lambda: 1000.0
    page = open("bcbot/api.py", encoding="utf-8").read()
    for name in F.OB_MODELS:
        assert name in a.CALLERS and name in a.MODELS and name in a.ladders
        assert f'<option value="{name}">' in page
        a.set_model(name)
        assert a.model == name
    a.set_model("ob_xgb")
    ans = {"v": 3, "p": {"ob_xgb": 0.47, "ob_hawkes": 0.6}, "thr": {"ob_xgb": F.MIN_THR, "ob_hawkes": F.MIN_THR},
           "n_ok": 6, "hk": {"per": {"okx": 0.6, "coinbase": 0.55}, "share": 1.0}, "status": {"version": 3}}

    def ask(i):
        rec = {"id": i, "cutoff": 1002.9, "start": 1002.6, "end": 1007.6, "feat": {"dev_tick": 3.5, "vol60": 2.0}}
        a.round, a.recs[i] = {"id": i}, rec
        asyncio.run(a._ob_fetch(i, rec))
        return rec
    a._ob_http = _Http(ans)
    rec = ask(7)
    assert rec["call"] == "DOWN" and rec["stake"] == 1.0 and a.ob_call_v == 3          # XGBoost said 47%: DOWN
    assert "P(UP) 0.470 &middot; fit 3" in a.panel_html()
    g = a.gate_status()
    assert g["open"] is False and "a new fit starts the record again" in g["text"]
    a.set_model("sign")                                                                 # HawkesSign: ob_hawkes in the answer
    rec = ask(8)
    assert rec["call"] == "UP" and "P(UP) 0.600 &middot; 2 of 6 order books" in a.panel_html()
    a._ob_http = _Http(dict(ans, p={"ob_hawkes": 0.5}))                                 # 50% exactly: no bet
    assert ask(9).get("skip") == "no call"
    a._ob_http = _Http(fail=OSError("refused"))                                         # the service is down
    rec = ask(10)
    assert rec["skip"] == "the order-book service did not answer" and "call" not in rec
    assert "the order-book service did not answer" in a.panel_html()
    a.set_model("ob_deeplob")                                                           # not in the answer
    a._ob_http = _Http(ans)
    assert ask(11)["skip"] == "the order-book service has no answer from this model"
    a.set_model("fade")                                                                 # another model calls:
    rec = ask(12)                                                                       # the answer is only logged
    assert "call" not in rec and "skip" not in rec and rec["ob"]["v"] == 3
    off = AutoTrader(FakeTrader(), make_cfg(tmp_path))                                  # no service configured
    off.set_model("ob_lgbm")
    r = {"id": 1}
    off._ob_no_call(r, "the order-book service is switched off (AUTO_OB_URL is empty)")
    assert r["skip"].startswith("the order-book service is switched off") and off.stats["skipped"] == 1


def test_the_bots_own_hawkes_forecast_waits_until_the_order_is_out(tmp_path):
    """2026-11-17, LIVE on XGBoost: of 15 rounds 6 got their real order.  The bot's own Hawkes forecast (pure-Python
    simulations in a thread, about 0.7 s) started at the decision and starved the loop while the order-book answer,
    the balance read and the click had to happen.  Unless it makes the call itself it now starts after the last
    moment for an order and the late snapshot, and is made as of the decision."""
    import time as _time

    class Pred:
        ready = True

        def __init__(self):
            self.feed = type("Feed", (), {"now": staticmethod(lambda: 5000.0 + _time.time() - t0)})()
            self.asked = []

        def data_span(self):
            return 999.0

        def predict(self, a, b, at=None):
            self.asked.append((_time.time() - t0, a, b, at))
            return {"ready": True, "exp_ret": 0.0, "sign_p_up": 0.6, "n_changes": 1.0, "lead_s": a, "calc_ms": 1}

    # the last moment for an order 2.8 s and the late snapshot 2.7 s before the cutoff, so that the test is short
    cfg = make_cfg(tmp_path, auto_ob_url="", auto_stake_usd=1.0, auto_send_margin_s=2.8, auto_late_snapshot_s=2.7)
    for model, waits in (("ob_xgb", True), ("fade", True), ("fade_hawkes", False)):
        a = AutoTrader(FakeTrader(), cfg)
        a.on, a.model = True, model
        t0 = _time.time()
        a.bc_now = lambda: 1000.0 + _time.time() - t0
        a.pred = Pred()
        rec = {"id": 7, "cutoff": 1002.9, "start": 1002.6, "end": 1007.6, "feat": {"dev_tick": 3.0, "vol60": 1.0}}
        a.round, a.recs[7] = {"id": 7}, rec                 # the decision, 2.9 s before the cutoff
        asyncio.run(a._hawkes(7, rec))
        (asked_after, w_a, w_b, at), = a.pred.asked
        if waits:                                           # starts 0.15 s after cutoff - 2.7 s = 0.35 s from now
            assert 0.3 <= asked_after < 0.8, (model, asked_after)
        else:                                               # Gap fade + Hawkes: its call needs the forecast at once
            assert asked_after < 0.1, (model, asked_after)
        assert abs(at - 5000.0) < 0.05 and abs(w_a - 2.6) < 0.05 and abs(w_b - 7.6) < 0.05    # as of the decision
        assert rec["pred"]["sign_p_up"] == 0.6


def test_the_forecast_as_of_an_earlier_instant_uses_only_the_events_up_to_it():
    from bcbot.lob import LobFeed
    from bcbot.predictor import HawkesPredictor
    import numpy as np
    feed = LobFeed()
    rng = np.random.default_rng(1)
    t_end = 10_000.0
    for q, types in ((feed.ev_time, 3), (feed.ev_sign, 4)):
        for t in np.sort(rng.uniform(t_end - 600, t_end, 1500)):
            q.append((float(t), int(rng.integers(types))))
    for t in np.arange(t_end - 600, t_end, 0.5):
        feed.bi.append((float(t), float(rng.uniform(-1, 1))))
        feed.ret.append((float(t), float(rng.normal(0, 1e-5))))
    feed.now = lambda: t_end - 100                           # "now" for the fit and the first forecast
    p = HawkesPredictor(feed, seed=3)
    p.refit()
    assert p.ready
    first = p.predict(2.9, 7.9)
    feed.now = lambda: t_end                                 # 100 s later, with 100 s more events in the feed
    p.rng = np.random.default_rng(3)
    p2 = HawkesPredictor(feed, seed=3)
    p2.mhp_time, p2.mhp_sign, p2.coe = p.mhp_time, p.mhp_sign, p.coe
    again = p2.predict(2.9, 7.9, at=t_end - 100)             # asked later, as of the earlier instant
    p3 = HawkesPredictor(feed, seed=3)
    p3.mhp_time, p3.mhp_sign, p3.coe = p.mhp_time, p.mhp_sign, p.coe
    feed.now = lambda: t_end - 100
    same = p3.predict(2.9, 7.9)
    for k in ("sign_p_up", "exp_ret", "bi", "n_changes"):
        assert again[k] == same[k], k                        # the same forecast, whenever it is computed
    assert first["ready"] and again["lead_s"] == 2.9


def test_a_new_fit_starts_the_fitted_models_stake_books_again(tmp_path):
    from bcbot.auto import SIGN_SIX_SINCE
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    a.since = SIGN_SIX_SINCE
    t0 = SIGN_SIX_SINCE + 60
    thr = {"ob_logit": 0.02, "ob_hawkes": F.MIN_THR}
    a._count(settled(0, "UP", {"v": 1, "p": {"ob_logit": 0.53, "ob_hawkes": 0.52}, "thr": thr}, t0))
    assert a.ladders["ob_logit"]["staked"] > 0 and a.ladders["sign"]["staked"] > 0
    a._count(settled(1, "UP", {"v": 2, "p": {"ob_hawkes": 0.52}, "thr": thr}, t0))
    assert a.ladders["ob_logit"]["staked"] == 0 and a.ladders["sign"].get("bets") == 2          # HawkesSign's goes on


class _Answer:
    def __init__(self, data):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self.data


class _Http:
    closed = False

    def __init__(self, data=None, fail=None):
        self.data, self.fail, self.asked = data, fail, []

    def get(self, url, params=None):
        self.asked.append((url, params))
        if self.fail:
            raise self.fail
        return _Answer(self.data)


def test_the_answer_of_the_service_is_logged_with_the_round(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url="http://127.0.0.1:1/ob"))
    a.bc_now = lambda: 1000.0
    hk = {"per": {"okx": 0.61}, "share": 1.0, "ms": 60, "age_s": 12.0}
    a._ob_http = _Http({"v": 3, "p": {"ob_xgb": 0.55}, "thr": {"ob_xgb": 0.03}, "n_ok": 6, "hk": hk,
                        "status": {"hours": 30.5, "version": 3, "markets_ok": 6, "markets": 6}})
    rec = {"cutoff": 1002.9, "start": 1002.6, "end": 1007.6, "feat": {"dev_tick": 3.5, "vol60": 2.0}}
    asyncio.run(a._ob_fetch(7, rec))                           # a, b: the round's window from now, for ob_hawkes
    assert a._ob_http.asked == [("http://127.0.0.1:1/ob", {"gap": "3.5000", "vol": "2.0000", "a": "2.6000", "b": "7.6000"})]
    assert rec["ob"] == {"v": 3, "p": {"ob_xgb": 0.55}, "thr": {"ob_xgb": 0.03}, "n_ok": 6, "lead_s": 2.9, "hk": hk}
    assert a.ob_info["ok"] and a.ob_info["hours"] == 30.5
    # the service is down: nothing is logged for the round, and the dashboard is told why
    a._ob_http = _Http(fail=OSError("connection refused"))
    rec2 = {"cutoff": 1002.9, "feat": {}}
    asyncio.run(a._ob_fetch(8, rec2))
    assert "ob" not in rec2 and a.ob_info["ok"] is False and "refused" in a.ob_info["error"]
    assert a._ob_http.asked == [("http://127.0.0.1:1/ob", {})]
