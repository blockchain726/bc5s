"""The page's P&L graph of the calling model: its paper bets one by one (the amounts its row in the Models table adds
up) and the real orders of the same time, and the losing-runs table behind a button."""
import time

import pytest

from bcbot.api import DASHBOARD, build_app
from bcbot.auto import AutoTrader, _running
from bcbot.obfeat import MIN_THR
from tests.test_obmodels import settled
from tests.test_safety import FakeTrader, make_cfg, trader_for, write_rounds


def ob(v, **p):
    return {"v": v, "p": p, "thr": {k: MIN_THR for k in p}}


def trader(tmp_path, **over):
    return AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url="http://127.0.0.1:1/ob", auto_stake_usd=1.0,
                                             auto_model="ob_xgb", **over))


def test_the_paper_line_is_the_models_row_bet_by_bet(tmp_path):
    a = trader(tmp_path)
    t0 = time.time() - 600
    for i, win in enumerate(["UP", "DOWN", "UP", "UP", "DOWN"]):               # XGBoost says UP every time
        a._count(settled(i, win, ob(5, ob_xgb=0.6, ob_lgbm=0.4), t0))
    row, d = a.summary()["records"]["ob_xgb"], a.pnl_curve("all")
    assert d["model"] == "ob_xgb" and d["range"] == "all"                       # the calling model, since its row began
    p = d["paper"]
    assert (p["n"], p["won"], p["pnl"]) == (row["called"], row["correct"], row["pnl"]) == (5, 3, pytest.approx(0.82))
    assert p["v"] == [0.94, -0.06, 0.88, 1.82, 0.82] and p["v"][-1] == round(row["pnl"], 2)
    assert p["t"] == [round(t0 + 30 * i, 1) for i in range(5)] and d["from"] < p["t"][0] <= d["now"]
    assert d["real"] == {"t": [], "v": [], "n": 0, "won": 0, "pnl": 0.0}       # no real order: nothing invented
    other = a.pnl_curve("all", "ob_lgbm")["paper"]                              # another model can be asked for
    assert (other["n"], other["won"], other["v"][-1]) == (5, 2, -1.12)
    assert a.pnl_curve("all", "no_such_model")["model"] == "ob_xgb"
    assert a.pnl_curve("all", "ob_deeplob")["paper"]["n"] == 0                  # a model without a bet: an empty line


def test_a_range_starts_both_lines_at_zero_at_its_left_edge(tmp_path):
    a = trader(tmp_path)
    now = time.time()
    for i, (age, win) in enumerate([(5 * 3600, "UP"), (2 * 3600, "DOWN"), (1200, "UP"), (600, "DOWN")]):
        a._count(settled(0, win, ob(5, ob_xgb=0.6), now - age) | {"id": 100 + i})
    assert a.pnl_curve("all")["paper"]["v"] == [0.94, -0.06, 0.88, -0.12]
    assert a.pnl_curve("6h")["paper"]["v"] == [0.94, -0.06, 0.88, -0.12]
    assert a.pnl_curve("1h")["paper"]["v"] == [0.94, -0.06]                     # only the last hour's two bets
    d = a.pnl_curve("1h")
    assert d["range"] == "1h" and d["now"] - d["from"] == pytest.approx(3600, abs=1)
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    today = a.pnl_curve("today")
    assert today["from"] == round(midnight, 1)
    assert today["paper"]["n"] == sum(now - age >= midnight for age in (5 * 3600, 2 * 3600, 1200, 600))
    assert a.pnl_curve("nonsense")["range"] == "all"


def test_the_real_line_is_the_real_orders_and_a_pnl_reset_does_not_cut_it(tmp_path):
    t0 = time.time() - 1800
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WLPLW"), t0, stake=2.0)   # four real orders and a paper round
    a = trader_for(tmp_path)
    real = a.pnl_curve("1h")["real"]
    assert (real["n"], real["won"]) == (4, 2) and real["v"] == [1.88, -0.12, -2.12, -0.24]
    assert real["t"] == [round(t0 + 30 * i, 1) for i in (0, 1, 3, 4)] and real["pnl"] == pytest.approx(-0.24)
    assert a.pnl_curve("all")["real"]["n"] == 0        # no paper bet by the calling model: its row has not begun
    a.reset_stats()                                     # 'P&L -> 0': the rows and their lines start again ...
    a._count(settled(0, "UP", ob(5, ob_xgb=0.6), time.time() + 1) | {"id": 9009, "call": "UP", "result": "WIN", "pnl": 0.94,
                                                                      "live": {"ok": True, "stake": 1.0}, "live_pnl": 0.94})
    d = a.pnl_curve("1h", "ob_xgb")
    assert d["paper"]["v"] == [0.94] and d["real"]["n"] == 5 and d["real"]["v"][-1] == 0.7    # ... the real orders stay
    b = trader_for(tmp_path)                           # and a restart reads them from the log again
    assert b.pnl_curve("1h")["real"]["n"] == 4


def test_a_new_fit_starts_the_fitted_models_lines_again_but_not_steadys(tmp_path):
    from bcbot.auto import STEADY_SINCE
    a = trader(tmp_path)
    t0 = max(time.time() - 600, STEADY_SINCE + 60)
    a._count(settled(0, "UP", ob(5, ob_xgb=0.6, ob_steady=0.54), t0))
    a._count(settled(1, "UP", ob(5, ob_xgb=0.6, ob_steady=0.54), t0))
    a._count(settled(2, "DOWN", ob(6, ob_xgb=0.6, ob_steady=0.54), t0))       # answered by fit 6: the rows start again
    assert a.pnl_curve("all", "ob_xgb")["paper"]["v"] == [-1.0] and a.stats["ob_xgb_called"] == 1
    assert a.pnl_curve("all", "ob_steady")["paper"]["v"] == [0.94, 1.88, 0.88]


def test_a_long_line_is_thinned_and_keeps_its_end():
    ts, ps = list(range(5000)), [0.94 if i % 2 else -1.0 for i in range(5000)]
    d = _running(ts, ps, 1200)
    assert d["n"] == 5000 and d["won"] == 2500 and d["pnl"] == pytest.approx(-150.0)
    assert 1000 <= len(d["t"]) <= 1260 and len(d["t"]) == len(d["v"])
    assert d["t"][0] == 0 and d["t"][-1] == 4999 and d["v"][-1] == -150.0 and d["t"] == sorted(d["t"])
    short = _running([1.0, 2.0], [0.94, -1.0], 1200)
    assert short == {"t": [1.0, 2.0], "v": [0.94, -0.06], "n": 2, "won": 1, "pnl": pytest.approx(-0.06)}


def test_the_page_draws_the_graph_and_keeps_the_losing_runs_behind_a_button(tmp_path):
    assert '<canvas id="pnl"></canvas>' in DASHBOARD and 'id="pnlrange"' in DASHBOARD
    assert '<button class="btn" id="runsbtn"' in DASHBOARD
    hidden = DASHBOARD.index('<div id="runswrap" hidden>')                     # closed until the button is clicked
    assert hidden < DASHBOARD.index('<h2>Losing runs today</h2>') < DASHBOARD.index('<div id="runs"')
    assert "kpis(d); modelsTable(d); runsBox(d); pnlState(d);" in DASHBOARD
    app = build_app(FakeTrader(), trader(tmp_path), make_cfg(tmp_path))
    assert "/auto/pnl" in {r.resource.canonical for r in app.router.routes()}
