"""HawkesSign on the order books of the six markets of our price (bcbot/hawkes6.py): the events it reads from a
market's rows, the runs it makes of each market's model, and how the six markets are mixed by their share."""
import numpy as np
import pytest

from bcbot import hawkes6 as H
from bcbot import obfeat as F
from bcbot.lob import EB, EC_DN, EC_UP, ES


def rows(seq, t0=1000.0, age=0.05):
    """seq: (bid, ask, bid size, ask size) per row, 100 ms apart."""
    return np.array([(t0 + 0.1 * i, b, a, bs, as_, age) for i, (b, a, bs, as_) in enumerate(seq)], float)


def test_a_markets_rows_become_the_four_kinds_of_events():
    c = rows([(100.0, 100.2, 5, 5),
              (100.1, 100.2, 5, 5),          # the mid is higher: a price move up, $0.05
              (100.1, 100.2, 5, 5),          # nothing changed: no event
              (100.1, 100.2, 7, 5),          # more size at the best bid: buy pressure
              (100.1, 100.2, 7, 9),          # more size at the best ask: sell pressure
              (100.1, 100.2, 4, 9),          # less at the best bid: sell pressure
              (100.1, 100.2, 4, 6),          # less at the best ask: buy pressure
              (99.9, 100.0, 4, 6),           # both prices lower: a price move down, $0.20
              (99.8, 100.1, 4, 6)])          # wider on both sides, the mid stays: no event
    t, typ, step = H.events(c)
    assert list(typ) == [EC_UP, EB, ES, ES, EB, EC_DN]
    assert t == pytest.approx([1000.1, 1000.3, 1000.4, 1000.5, 1000.6, 1000.7])
    assert step == pytest.approx([0.05, 0, 0, 0, 0, 0.2])


def test_an_unknown_or_silent_book_and_a_stall_make_no_events():
    nan = float("nan")
    c = rows([(100.0, 100.2, 5, 5), (nan, nan, nan, nan), (100.5, 100.7, 5, 5), (100.6, 100.8, 5, 5)])
    assert list(H.events(c)[1]) == [EC_UP]                 # only the last pair: both rows known
    stale = rows([(100.0, 100.2, 5, 5), (100.1, 100.3, 5, 5)], age=F.MAX_AGE + 1)
    assert len(H.events(stale)[0]) == 0                    # the book has been silent: its rows say nothing
    gap = rows([(100.0, 100.2, 5, 5), (100.1, 100.3, 5, 5)])
    gap[1, 0] += 5.0                                       # the recording stalled for 5 s between the two rows
    assert len(H.events(gap)[0]) == 0
    assert len(H.events(rows([(100.0, 100.2, 5, 5)]))[0]) == 0


def model(up=0.0, dn=0.0, step=1.0, alpha=None, beta=4.0):
    return {"mu": np.array([up, dn, 0.0, 0.0]) + 1e-9, "alpha": np.zeros((4, 4)) if alpha is None else alpha,
            "beta": np.full((4, 4), beta), "step": step, "at": 1000.0}


def test_the_runs_count_up_minus_down_moves_inside_the_window():
    rng = np.random.default_rng(1)
    net = H.simulate_net(model(up=2.0, dn=1.0), np.zeros((4, 4)), 1.0, 4.0, 4000, rng)
    assert net.mean() == pytest.approx((2.0 - 1.0) * 3.0, abs=0.2)           # 3 s of window, 2 up and 1 down a second
    assert net.var() == pytest.approx((2.0 + 1.0) * 3.0, rel=0.15)
    # an up move that brings on more up moves (half an extra one each): twice as many in the long run
    alpha = np.zeros((4, 4))
    alpha[EC_UP, EC_UP] = 2.0
    net = H.simulate_net(model(up=1.0, alpha=alpha), np.zeros((4, 4)), 10.0, 20.0, 2000, rng)
    assert net.mean() == pytest.approx(20.0, abs=1.5)
    # what happened just before the decision counts: five up moves a moment ago bring more
    hot = np.zeros((4, 4))
    hot[:, EC_UP] = 5.0
    cold = H.simulate_net(model(up=1.0, alpha=alpha), np.zeros((4, 4)), 0.0, 1.0, 4000, rng).mean()
    assert H.simulate_net(model(up=1.0, alpha=alpha), hot, 0.0, 1.0, 4000, rng).mean() > cold + 1.0


def quiet(t0=1000.0):
    return {k: rows([(100.0, 100.2, 5, 5)] * 5, t0 - 0.5) for k in F.SHARES}


def test_the_six_markets_are_mixed_by_their_share_in_our_price():
    heavy = ("bitfinex_perp", "bybit_perp")                                    # 60.5% of our price
    models = {k: model(up=3.0) if k in heavy else model(dn=3.0) for k in F.SHARES}
    out = H.predict(models, quiet(), 1000.0, 2.9, 7.9, seed=7)
    assert out["share"] == pytest.approx(1.0) and out["p"] > 0.85              # the heavier side decides
    assert all(out["per"][k] == (1.0 if k in heavy else 0.0) for k in F.SHARES)
    # ... in $: the same counts with a tenth of the step on the heavy side, and the others decide
    small = {k: model(up=3.0, step=0.1) if k in heavy else model(dn=3.0) for k in F.SHARES}
    assert H.predict(small, quiet(), 1000.0, 2.9, 7.9, seed=7)["p"] < 0.05
    # markets with a model must make up half of our price, else there is no answer for the mix
    few = {k: models[k] for k in ("okx", "binance_perp", "coinbase")}          # 23.5%
    out = H.predict(few, quiet(), 1000.0, 2.9, 7.9, seed=7)
    assert out["p"] is None and set(out["per"]) == set(few) and out["share"] == pytest.approx(0.235)
    assert H.predict({}, quiet(), 1000.0, 2.9, 7.9)["p"] is None
    # nobody moves: 50% exactly, which is no bet
    still = H.predict({k: model() for k in F.SHARES}, quiet(), 1000.0, 2.9, 7.9, seed=7)
    assert still["p"] == 0.5 and F.side(still["p"], F.MIN_THR) is None


def test_a_markets_model_is_fitted_on_its_last_300_s_of_rows():
    rng = np.random.default_rng(5)
    n = 3200
    mid = 100.0 + np.cumsum(rng.choice([0.0, 0.0, 0.5, -0.5], n))            # a price move in half of the rows
    size = 5 + rng.integers(0, 3, n)
    c = np.column_stack([1000.0 + 0.1 * np.arange(n), mid - 0.1, mid + 0.1, size, np.full(n, 5.0), np.full(n, 0.05)])
    now = c[-1, 0] + 0.05
    m = H.fit_market(c, now)
    assert m["step"] == pytest.approx(0.5) and m["moves"] == pytest.approx(1500, abs=120) and m["at"] == now
    assert m["mu"].shape == (4,) and m["alpha"].shape == m["beta"].shape == (4, 4) and (m["alpha"] >= 0).all()
    assert m["mu"][EC_UP] + m["alpha"][EC_UP].sum() > 0                         # it expects up moves to go on coming
    out = H.predict({k: m for k in F.SHARES}, {k: c for k in F.SHARES}, now, 2.9, 7.9, seed=3)
    assert 0.2 < out["p"] < 0.8 and out["ms"] < 2000                            # a coin's rows: no strong opinion
    assert H.fit_market(c[:30], c[29, 0] + 0.05) is None                        # too few events: no model
    assert H.fit_market(c, now + 1000) is None                                  # ... or rows that are too old
