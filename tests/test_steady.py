"""The lifetime of an order-book reading (the owner, 2026-11-17: "the current OB has a lifetime, but we do not calc
this and follow the current OB only"): the reading and its age (bcbot/obfeat.py), the paper model "ob_steady" that
follows a strong reading of 5 s or more, and the same numbers as inputs of the fitted models."""
import asyncio
import importlib.util
import json
import sys
import time

import numpy as np
import pytest

from bcbot import obfeat as F
from bcbot.auto import AutoTrader
from tests.test_obmodels import _Http, rows, settled
from tests.test_safety import FakeTrader, make_cfg

T = 1_000_000.0


def books(n=200, **kw):
    return {k: rows(T, n, **kw) for k in F.SHARES}


def test_a_reading_has_a_side_a_size_and_an_age():
    bk = books(200, bid_size=3.0, ask_size=1.0)                   # 75% of the best-price size on the bid, for 20 s
    r, age, move, share = F.life_mixed(bk, [T])[0]
    assert r == pytest.approx(0.5) and share == pytest.approx(1.0) and move == pytest.approx(0.0)
    assert age == pytest.approx(19.9 - 0.4)                       # counted from the first row that has a 1 s average
    assert F.steady_call(r, age) == "UP"
    # the book turned 3 s ago: the reading is on the new side, and young
    for a in bk.values():
        a["bs"][-30:], a["as_"][-30:] = 1.0, 3.0
    r, age, _, _ = F.life_mixed(bk, [T])[0]
    assert r == pytest.approx(-0.5) and 2.0 < age < 3.0 and F.steady_call(r, age) is None
    # ... and at 6 s it is old enough
    for a in bk.values():
        a["bs"][-60:], a["as_"][-60:] = 1.0, 3.0
    r, age, _, _ = F.life_mixed(bk, [T])[0]
    assert 5.0 < age < 6.0 and F.steady_call(r, age) == "DOWN"


def test_a_weak_reading_or_an_unknown_one_is_no_bet():
    assert F.steady_call(0.49, 20.0) is None and F.steady_call(-0.49, 20.0) is None        # under 75% on one side
    assert F.steady_call(0.9, 4.9) is None                                                  # strong, but young
    assert F.steady_call(0.5, 5.0) == "UP" and F.steady_call(-0.8, 30.0) == "DOWN"
    assert F.steady_call(float("nan"), 9.0) is None and F.steady_call(0.8, None) is None
    r = F.life_mixed(books(200, bid_size=3.0, age=5.0), [T])[0]                             # the books went silent
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert np.isnan(F.life_mixed({}, [T])).all()


def test_a_hole_in_the_rows_ends_the_readings_life():
    bk = books(200, bid_size=3.0, ask_size=1.0)
    keep = np.ones(200, dtype=bool)
    keep[150:170] = False                                         # 2 s without rows, 3 s before the decision
    r, age, _, _ = F.life_mixed({k: a[keep] for k, a in bk.items()}, [T])[0]
    assert r == pytest.approx(0.5) and 2.0 < age < 3.0            # the same side all along, but only known since the hole


def test_the_reading_is_of_five_markets_and_follows_the_price():
    assert F.LIFE_MARKETS == ("bybit_perp", "okx_perp", "okx", "binance_perp", "coinbase")
    bk = books(200, bid_size=3.0, ask_size=1.0, drift=0.01)       # every market climbs 1 cent a row
    bk["bitfinex_perp"] = rows(T, 200, bid_size=1.0, ask_size=9.0)          # the other way: not read
    r, age, move, share = F.life_mixed(bk, [T])[0]
    assert r == pytest.approx(0.5) and share == pytest.approx(1.0)
    assert move == pytest.approx(0.01 * age / F.STEP, abs=1e-6)   # the price's move since the reading took its side
    del bk["okx"], bk["coinbase"]                                 # two markets missing: still over half the five's share
    r, _, _, share = F.life_mixed(bk, [T])[0]
    assert r == pytest.approx(0.5) and share == pytest.approx((0.172 + 0.160 + 0.068) / 0.567, abs=1e-3)
    del bk["bybit_perp"]                                          # under half: no reading
    assert np.isnan(F.life_mixed(bk, [T])[0][0])


def test_the_new_inputs_come_after_the_85_and_mirror():
    assert F.NAMES2[:85] == F.NAMES and len(F.NAMES2) == 85 + len(F.SHARES) * 2 + 4 == 101
    assert F.NAMES2[-4:] == F.MIX5 == ("mix5_avg1", "mix5_age1", "mix5_old1", "mix5_resp1")
    bk = books(200, bid_size=3.0, ask_size=1.0, drift=0.01)
    X = F.build2(bk, [T], [3.0], [2.0])
    assert X.shape == (1, 101) and np.allclose(X[:, :85], F.build(bk, [T], [3.0], [2.0]), equal_nan=True)
    name = {n: i for i, n in enumerate(F.NAMES2)}
    age = 19.5
    assert X[0, name["mix5_avg1"]] == pytest.approx(0.5) and X[0, name["mix5_age1"]] == pytest.approx(age / 10)
    assert X[0, name["mix5_old1"]] == pytest.approx(0.5 * age / 10)
    assert X[0, name["mix5_resp1"]] == pytest.approx(0.01 * 195 / 2.0, abs=1e-6)            # in units of vol60
    assert X[0, name["okx_age1"]] == pytest.approx(age / 10) and X[0, name["bitfinex_perp_old1"]] == pytest.approx(0.5 * age / 10)
    M = F.mirror(X)
    assert np.allclose(M[:, 85:], -X[:, 85:]) and np.allclose(F.mirror(X[:, :85]), M[:, :85])
    down = F.build2(books(200, bid_size=1.0, ask_size=3.0, drift=-0.01), [T], [-3.0], [2.0])  # the same market upside down
    assert np.allclose(down[:, 85:], M[:, 85:], atol=1e-6)


def test_the_service_and_the_fit_get_the_same_numbers():
    """The fit reads the whole recording, the service the last 40 s of rows: an age is counted up to 30 s, so both
    see the same reading, age and inputs at a decision."""
    rng = np.random.default_rng(5)
    n = 1300
    t = T + np.arange(n) * F.STEP
    bk = {}
    for k in F.SHARES:
        a = np.zeros(n, dtype=F.ROW)
        mid = 60000 + np.cumsum(rng.normal(0, 0.5, n))
        a["t"], a["bid"], a["ask"] = t, mid - 0.5, mid + 0.5
        slow = np.repeat(rng.lognormal(0, 1.0, (n // 50 + 1, 2)), 50, axis=0)[:n]           # sizes that last 5 s
        a["bs"] = (slow[:, :1] * rng.lognormal(0, 0.3, (n, F.LEVELS))).astype("f4")
        a["as_"] = (slow[:, 1:] * rng.lognormal(0, 0.3, (n, F.LEVELS))).astype("f4")
        a["bp"] = a["ap"] = np.arange(F.LEVELS) * 0.1
        a["buy"], a["sell"] = rng.exponential(0.1, n), rng.exponential(0.1, n)
        a["age"] = np.where(rng.random(n) < 0.01, 5.0, 0.05)
        a["bid"][rng.random(n) < 0.01] = np.nan
        bk[k] = a
    keep = np.ones(n, dtype=bool)
    keep[700:730] = False                                          # the service stood still for 3 s
    bk = {k: a[keep] for k, a in bk.items()}
    dec = T + np.array([45.03, 52.2, 61.0, 69.97, 71.5, 73.3, 80.0, 95.55, 110.0, 129.9])
    vol, gap = np.full(len(dec), 2.0), np.linspace(-3, 3, len(dec))
    whole = F.build2(bk, dec, gap, vol)
    mixed = F.life_mixed(bk, dec)
    assert (~np.isnan(mixed[:, 0])).sum() >= 8 and np.nanmax(mixed[:, 1]) > 3.0
    for i, d in enumerate(dec):
        ring = {k: a[a["t"] <= d][-400:] for k, a in bk.items()}  # what the service has in memory at that instant
        assert np.allclose(F.build2(ring, [d], [gap[i]], [vol[i]]), whole[i:i + 1], equal_nan=True, atol=1e-6)
        assert np.allclose(F.life_mixed(ring, [d]), mixed[i:i + 1], equal_nan=True, atol=1e-6)
        slim = {k: F.slim(a) for k, a in ring.items()}            # the fit keeps only these columns
        assert np.allclose(F.life_inputs(slim, [d], [vol[i]]), whole[i:i + 1, 85:], equal_nan=True, atol=1e-6)


def test_steady_is_a_model_of_the_page_scored_from_its_start(tmp_path):
    from bcbot.auto import MODEL_REGISTERED, STEADY_SINCE
    assert MODEL_REGISTERED["ob_steady"] == STEADY_SINCE == 1794961200
    assert "ob_steady" in F.OB_MODELS and "ob_steady" not in F.OB_FITTED and F.OB_ANSWER["ob_steady"] == "ob_steady"
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url="http://127.0.0.1:1/ob", auto_stake_usd=1.0))
    assert "ob_steady" in a.CALLERS and "ob_steady" in a.ladders
    page = open("bcbot/api.py", encoding="utf-8").read()
    assert '<option value="ob_steady">Steady order book (strong 5 s+)</option>' in page and "ob_steady: 'Steady book'" in page
    thr = {"ob_steady": F.MIN_THR, "ob_xgb": F.MIN_THR}
    a._count(settled(0, "UP", {"v": 5, "p": {"ob_steady": 0.54}, "thr": thr}, STEADY_SINCE - 3600))   # before it was in the bot
    assert a.stats.get("ob_steady_called", 0) == 0
    t0 = STEADY_SINCE + 60
    a._count(settled(0, "UP", {"v": 5, "p": {"ob_steady": 0.54, "ob_xgb": 0.6}, "thr": thr}, t0))      # UP, right
    a._count(settled(1, "UP", {"v": 6, "p": {"ob_steady": 0.46, "ob_xgb": 0.6}, "thr": thr}, t0))      # a new fit: its
    a._count(settled(2, "UP", {"v": 6, "p": {"ob_steady": 0.5, "ob_xgb": 0.6}, "thr": thr}, t0))       # record goes on
    a._count(settled(3, "DOWN", {"v": 6, "p": {"ob_xgb": 0.6}, "thr": thr}, t0))                       # no reading
    s = a.stats
    assert (s["ob_steady_called"], s["ob_steady_correct"]) == (2, 1) and s["ob_xgb_called"] == 3       # 50%: no bet
    assert a.summary()["records"]["ob_steady"]["called"] == 2
    a.set_model("ob_steady")
    assert "2026-11-18 00:20 UTC" in a.gate_status()["text"]


def test_steady_makes_the_call_and_says_why_it_does_not_bet(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url="http://127.0.0.1:1/ob", auto_stake_usd=1.0))
    a.on = True
    a.bc_now = lambda: 1000.0
    a.set_model("ob_steady")
    thr = {"ob_steady": F.MIN_THR}

    def ask(i, p, life):
        rec = {"id": i, "cutoff": 1002.9, "start": 1002.6, "end": 1007.6, "feat": {"dev_tick": 3.5, "vol60": 2.0}}
        a.round, a.recs[i] = {"id": i}, rec
        a._ob_http = _Http({"v": 5, "p": p, "thr": thr, "n_ok": 6, "status": {"version": 5}, **({"life": life} if life else {})})
        asyncio.run(a._ob_fetch(i, rec))
        return rec
    rec = ask(1, {"ob_steady": 0.54}, {"r": 0.56, "age": 7.2, "move": 1.5, "share": 1.0})
    assert rec["call"] == "UP" and rec["stake"] == 1.0 and rec["ob"]["life"] == {"r": 0.56, "age": 7.2, "move": 1.5, "share": 1.0}
    assert "the books 78% UP for 7.2 s" in a.panel_html()
    rec = ask(2, {"ob_steady": 0.46}, {"r": -0.8, "age": 30.0, "move": -4.0, "share": 0.8})
    assert rec["call"] == "DOWN"
    rec = ask(3, {"ob_steady": 0.5}, {"r": -0.22, "age": 2.1, "move": 0.0, "share": 1.0})
    assert "call" not in rec and rec["skip"] == "no steady reading: the books 61% DOWN for 2.1 s (a bet needs 75% for 5 s)"
    assert "no steady reading: the books 61% DOWN for 2.1 s" in a.panel_html()
    rec = ask(4, {}, None)                                         # the books are not known
    assert rec["skip"] == "the order-book service has no answer from this model"
    a.set_model("ob_xgb")                                          # another model calls: the reading is still logged
    rec = ask(5, {"ob_xgb": 0.55, "ob_steady": 0.5}, {"r": 0.1, "age": 0.3, "move": 0.0, "share": 1.0})
    assert rec["ob"]["life"]["age"] == 0.3


def _service(tmp_path, monkeypatch):
    """tools/ob_server.py as a module, on an empty folder (it reads its folder and port from the command line)."""
    monkeypatch.setattr(sys, "argv", ["ob_server.py", str(tmp_path), "18766"])
    spec = importlib.util.spec_from_file_location("ob_server_under_test", "tools/ob_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Req:
    def __init__(self, **q):
        self.query = {k: str(v) for k, v in q.items()}


def test_the_service_answers_the_steady_model_from_its_rows(tmp_path, monkeypatch):
    S = _service(tmp_path, monkeypatch)
    now = time.time()

    def fill(**kw):
        for k, m in S.M.items():
            a = rows(now, 200, **kw)
            m.ring[:200], m.n, m.ready, m.book_at = a, 200, True, now
            m.bids, m.asks = {100.0: 1.0}, {100.2: 1.0}
    fill(bid_size=3.0, ask_size=1.0)
    out = json.loads(asyncio.run(S.handle_ob(_Req(gap=1.0, vol=2.0, probe=1))).text)
    assert out["p"]["ob_steady"] == 0.54 and out["thr"]["ob_steady"] == F.MIN_THR          # strong and old: UP
    assert out["life"]["r"] == 0.5 and 19.0 < out["life"]["age"] < 20.0 and out["life"]["share"] == 1.0
    assert "ob_steady" in out["status"]["models"]
    fill(bid_size=1.0, ask_size=1.2)
    out = json.loads(asyncio.run(S.handle_ob(_Req(gap=1.0, vol=2.0, probe=1))).text)
    assert out["p"]["ob_steady"] == 0.5 and out["life"]["r"] == pytest.approx(-0.0909, abs=1e-4)   # weak: no bet
    for m in S.M.values():
        m.n = 0                                                    # no rows yet (just started): no answer from it
    out = json.loads(asyncio.run(S.handle_ob(_Req(gap=1.0, vol=2.0, probe=1))).text)
    assert "ob_steady" not in out["p"] and "life" not in out


def test_the_service_serves_a_fit_with_85_inputs_and_one_with_the_new_ones(tmp_path, monkeypatch):
    S = _service(tmp_path, monkeypatch)
    mdir = tmp_path / "models"
    (mdir / "v1").mkdir(parents=True)
    now = time.time()
    bk = {k: rows(now, 200, bid_size=3.0, ask_size=1.0) for k in F.SHARES}
    X = F.build2(bk, [now], [1.0], [2.0])
    for names, idx in ((F.NAMES, [F.NAMES.index("mix_imb1")]), (F.NAMES2, [F.NAMES2.index("mix5_old1")])):
        (mdir / "v1" / "logit.json").write_text(json.dumps({"idx": idx, "sd": [1.0], "w": [1.0]}))
        (mdir / "current.json").write_text(json.dumps({"version": 1, "names": list(names), "files": {"ob_logit": "v1/logit.json"},
                                                       "thr": {"ob_logit": 0.0}, "hours": 1.0}))
        m = S.Models(tmp_path)
        m.load()
        assert m.error is None and m.names() == ["ob_logit"]
        p = m.predict(X, bk, now, 1.0, 2.0)["ob_logit"]
        assert p == pytest.approx(1 / (1 + np.exp(-X[0, idx[0]])), abs=1e-5) and p > 0.5
    (mdir / "current.json").write_text(json.dumps({"version": 2, "names": ["a", "b"], "files": {}}))
    m = S.Models(tmp_path)
    m.load()
    assert "another list of inputs" in m.error and not m.fn
