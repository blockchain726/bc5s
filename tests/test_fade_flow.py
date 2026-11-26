"""The "fade_flow" model: the Gap fade's bet, placed only when the futures' order flow is on the same side."""
from bcbot.auto import MODEL_REGISTERED, AutoTrader, _fade_call, _fade_flow_call
from tests.test_safety import FakeTrader, make_cfg

GAP_UP = {"dev_tick": 4.0, "vol60": 2.0}           # bc.game $4 above the market: the fade bets DOWN


def flow(binance, okx, bybit, **base):
    return {**(base or GAP_UP), "ofi_binance_perp_3s": binance, "ofi_okx_perp_3s": okx, "ofi_bybit_perp_3s": bybit}


def test_fade_flow_needs_two_of_three_futures_on_the_bets_side():
    assert _fade_call(GAP_UP) == "DOWN"
    assert _fade_flow_call(flow(-5.0, -1.0, -2.0)) == "DOWN"            # all three sold: the bet stands
    assert _fade_flow_call(flow(-5.0, 3.0, -2.0)) == "DOWN"             # two of three
    assert _fade_flow_call(flow(-5.0, 3.0, 2.0)) is None                # one of three: no bet
    assert _fade_flow_call(flow(5.0, 3.0, 2.0)) is None                 # the futures bought: no bet
    assert _fade_flow_call(flow(-5.0, None, -2.0)) == "DOWN"            # one feed silent: the other two decide
    assert _fade_flow_call(flow(-5.0, None, None)) is None              # one opinion is not two
    assert _fade_flow_call(flow(-5.0, 0, 0.0)) is None and _fade_flow_call(flow(-5.0, float("nan"), True)) is None
    assert _fade_flow_call(GAP_UP) is None                              # no flow logged: no bet
    assert _fade_flow_call(flow(-5.0, -1.0, -2.0, dev_tick=1.0, vol60=2.0)) is None     # no Gap fade bet: nothing to confirm
    assert _fade_flow_call(flow(5.0, 1.0, -2.0, dev_tick=-4.0, vol60=2.0)) == "UP"      # the mirror case
    assert _fade_flow_call(None) is None and _fade_flow_call({}) is None
    assert "fade_flow" in AutoTrader.CALLERS


def test_fade_flow_is_scored_only_after_it_was_fixed(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    t = MODEL_REGISTERED["fade_flow"]
    rec = {"id": 1, "cutoff": t + 60, "win": "DOWN", "tie": False, "feat": flow(-5.0, -1.0, 2.0)}
    calls = a._model_calls(rec)
    assert calls["fade_flow"] == "DOWN" and calls["fade"] == "DOWN"
    assert a._model_calls(dict(rec, cutoff=t - 60))["fade_flow"] is None            # the rounds it was chosen on
    assert a._model_calls(dict(rec, feat=flow(5.0, 1.0, 2.0)))["fade_flow"] is None and \
        a._model_calls(dict(rec, feat=flow(5.0, 1.0, 2.0)))["fade"] == "DOWN"       # the plain fade still bets
    a._count(rec)
    s = a.stats
    assert (s["fade_flow_called"], s["fade_flow_correct"], s["fade_flow_fwd_called"]) == (1, 1, 1)
    a.on = True
    a.set_model("fade_flow")
    a.round = {"id": 1}
    a.recs[1] = {"call": "DOWN", "feat": rec["feat"]}
    assert "from the market" in a.panel_html()


def test_fade_hawkes_bets_only_when_hawkes_sign_agrees(tmp_path):
    from bcbot.auto import _fade_hawkes_call
    up, down = {"sign_p_up": 0.61, "ready": True}, {"sign_p_up": 0.42, "ready": True}
    assert _fade_hawkes_call(GAP_UP, down) == "DOWN"                    # the fade says DOWN and so does Hawkes
    assert _fade_hawkes_call(GAP_UP, up) is None                        # Hawkes says UP: no bet
    assert _fade_hawkes_call(GAP_UP, {"sign_p_up": 0.5}) is None        # no opinion
    assert _fade_hawkes_call(GAP_UP, {"sign_p_up": 0.2, "ready": False}) is None     # still warming up
    assert _fade_hawkes_call(GAP_UP, None) is None and _fade_hawkes_call(GAP_UP, {}) is None
    assert _fade_hawkes_call({"dev_tick": 1.0, "vol60": 2.0}, down) is None          # no Gap fade bet: nothing to confirm
    assert _fade_hawkes_call({"dev_tick": -4.0, "vol60": 2.0}, up) == "UP"
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    t = MODEL_REGISTERED["fade_hawkes"]
    rec = {"id": 1, "cutoff": t + 60, "win": "DOWN", "tie": False, "feat": dict(GAP_UP), "pred": down}
    assert a._model_calls(rec)["fade_hawkes"] == "DOWN"                 # it still reads the bot's own one-book forecast
    assert a._model_calls(rec)["sign"] is None                          # which is no longer HawkesSign's bet (six markets now)
    assert a._model_calls(dict(rec, cutoff=t - 60))["fade_hawkes"] is None          # before it was added
    assert a._model_calls(dict(rec, pred=up))["fade_hawkes"] is None and a._model_calls(dict(rec, pred=up))["fade"] == "DOWN"
    a._count(rec)
    assert (a.stats["fade_hawkes_called"], a.stats["fade_hawkes_correct"]) == (1, 1)
    a.on = True
    a.set_model("fade_hawkes")
    a.round = {"id": 1}
    a.recs[1] = {"call": "DOWN", "feat": dict(GAP_UP), "pred": down}
    html = a.panel_html()
    assert "from the market" in html and "HawkesSign P(up) 0.42" in html


def test_fade_flow2_needs_the_flow_and_a_gap_of_two_volatilities(tmp_path):
    from bcbot.auto import FADE_FLOW2_K, _fade_flow2_call
    assert FADE_FLOW2_K == 2.0
    assert _fade_flow2_call(flow(-5.0, -1.0, -2.0)) == "DOWN"                              # a $4 gap at vol 2: exactly 2 x
    assert _fade_flow2_call(flow(-5.0, -1.0, -2.0, dev_tick=3.9, vol60=2.0)) is None       # 1.95 x: the flow model bets, this one not
    assert _fade_flow_call(flow(-5.0, -1.0, -2.0, dev_tick=3.9, vol60=2.0)) == "DOWN"
    assert _fade_flow2_call(flow(5.0, 1.0, -2.0)) is None                                  # a big gap, the flow against it
    assert _fade_flow2_call(flow(5.0, 1.0, -2.0, dev_tick=-6.0, vol60=2.0)) == "UP"
    assert _fade_flow2_call(None) is None and _fade_flow2_call({}) is None and _fade_flow2_call(GAP_UP) is None
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    t = MODEL_REGISTERED["fade_flow2"]
    rec = {"id": 1, "cutoff": t + 60, "win": "DOWN", "tie": False, "feat": flow(-5.0, -1.0, 2.0)}
    assert a._model_calls(rec)["fade_flow2"] == "DOWN" and a._model_calls(dict(rec, cutoff=t - 60))["fade_flow2"] is None
    a.on = True
    a.set_model("fade_flow2")
    assert a.model == "fade_flow2"
