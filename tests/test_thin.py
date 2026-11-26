"""The "thin" model: no prediction, the side the page pays 2.10 or more on; paid from the round's final pools."""
from bcbot.auto import MODEL_REGISTERED, POOL_FEE_K, THIN_MIN_PAY, AutoTrader, _pool_pay, _thin_call
from tests.test_safety import FakeTrader, make_cfg

T = MODEL_REGISTERED["thin"] + 100                 # a round after the rule was fixed


def test_thin_calls_the_side_that_pays_2_10_or_more():
    assert _thin_call({"pay_up": 2.13, "pay_dn": 1.79}) == "UP"
    assert _thin_call({"pay_up": 1.70, "pay_dn": 2.31}) == "DOWN"
    assert _thin_call({"pay_up": 2.10, "pay_dn": 1.81}) == "UP"
    assert _thin_call({"pay_up": 2.09, "pay_dn": 1.82}) is None         # under the threshold it does not pay
    assert _thin_call({"pay_up": 1.94, "pay_dn": 1.94}) is None and _thin_call({}) is None and _thin_call(None) is None
    assert _thin_call({"pay_up": None, "pay_dn": 2.5}) is None          # one pool unknown: no bet
    assert THIN_MIN_PAY == 2.10 and POOL_FEE_K == 0.95 and "thin" in AutoTrader.CALLERS


def test_thin_is_paid_from_the_final_pools_with_our_stake_in_them(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    rec = {"id": 1, "cutoff": T, "win": "UP", "tie": False, "pools": [90.0, 120.0],
           "feat": {"pay_up": 2.255, "pay_dn": 1.706}}                  # no page payout: the calling model made no call
    paid = POOL_FEE_K * 120 / 91                                        # x2.25 on the thin side with our $1 in it
    assert a._model_calls(rec)["thin"] == "UP"
    assert a._model_calls(dict(rec, cutoff=MODEL_REGISTERED["thin"] - 100))["thin"] is None     # before its time
    assert abs(_pool_pay(rec, "UP", 1.0) - (1 + paid)) < 1e-9 and _pool_pay({"pools": [0, 5]}, "UP", 1) is None
    assert _pool_pay({}, "UP", 1) is None
    assert abs(a._pnl(rec, "UP", pools=True) - paid) < 1e-9
    assert abs(a._pnl(rec, "UP") - 0.94) < 1e-9                         # every other model: booked as before
    assert a._pnl(rec, "DOWN", pools=True) == -1.0 and a._pnl(dict(rec, tie=True), "UP", pools=True) == 0.0
    assert abs(a._pnl(dict(rec, pools=None), "UP", pools=True) - 0.94) < 1e-9   # pools not seen: the average
    before = {k: v for k, v in a.stats.items() if not k.startswith("thin")}
    a._count(rec)
    s = a.stats
    assert (s["thin_called"], s["thin_correct"], s["thin_fwd_called"], s["thin_fwd_correct"]) == (1, 1, 1, 1)
    assert abs(s["thin_pnl"] - paid) < 1e-3
    assert {k: v for k, v in s.items() if not k.startswith("thin") and k != "rounds"} == \
        {k: v for k, v in before.items() if k != "rounds"}              # a round only thin calls moves no other record
    a._count(dict(rec, id=2, win="DOWN"))
    assert s["thin_called"] == 2 and s["thin_correct"] == 1 and abs(s["thin_pnl"] - (paid - 1)) < 1e-3
    a.on = True
    a.set_model("thin")                                                 # selectable like any model
    assert a.model == "thin"
    a.round = {"id": 1}
    a.recs[1] = {"call": "UP", "feat": rec["feat"]}
    assert "the smaller pool" in a.panel_html()


def test_thin_ladder_shows_what_a_big_stake_does_to_its_own_payout(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    a.progression, a.since = 2.0, 0.0
    lad = a.ladders["thin"]
    lad["step"] = 5                                                     # five losses in a row: the next stake is $32
    rec = {"id": 1, "cutoff": T, "win": "UP", "tie": False, "pools": [105.0, 131.0]}
    a._ladder_step(lad, rec, "UP", pools=True)
    assert lad["biggest"] == 32 and lad["step"] == 0
    assert abs(lad["pnl"] - 32 * POOL_FEE_K * 131 / (105 + 32)) < 1e-3  # paid x1.91: under even money
    assert abs(lad["flat"] - POOL_FEE_K * 131 / 106) < 1e-3             # the same call at $1: x2.17
    assert lad["pnl"] < 32 < lad["pnl_shown"]                           # the page's payout would have promised more


def test_thin_record_starts_after_the_log_the_rule_was_chosen_on(tmp_path):
    # that log's last round has its cutoff at 1794427514.030 (one of the finding's bets); the next one at ...544.533
    assert 1794427514.030 < MODEL_REGISTERED["thin"] <= 1794427544.533
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    feat = {"pay_up": 2.122192, "pay_dn": 1.789903}
    assert a._model_calls({"id": 1, "cutoff": 1794427514.03, "feat": feat})["thin"] is None
    assert a._model_calls({"id": 2, "cutoff": 1794427544.533, "feat": feat})["thin"] == "UP"


def test_thin_does_not_call_on_another_rounds_pools(tmp_path):
    assert _thin_call({"pay_up": 2.32, "pay_dn": 1.67, "pools_prev": True}) is None
    assert _thin_call({"pay_up": 2.32, "pay_dn": 1.67}) == "UP"                 # an older record: as before
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    a.bc_ticks.extend((1000.0 + 0.5 * i, 77000.0 + i % 3) for i in range(60))
    a.pools = {"upPoolAmount": 100.0, "downPoolAmount": 140.0}                  # the previous round's last frame
    f = a._features(1030.0, {"id": 5, "cutoff": 1033.0})
    assert f["pools_prev"] is True and f["pay_up"] > 2.3 and _thin_call(f) is None
    f = a._features(1030.0, {"id": 5, "cutoff": 1033.0, "pools": [100.0, 140.0]})
    assert "pools_prev" not in f and _thin_call(f) == "UP"                      # the round's own pools: a call


def test_thin_other_side_empty_pays_the_stake_back(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    rec = {"id": 1, "cutoff": T, "win": "UP", "tie": False, "pools": [16.000006, 0.0]}
    assert _pool_pay(rec, "UP", 1.0) == 1.0 and a._pnl(rec, "UP", pools=True) == 0.0
    assert _pool_pay(rec, "DOWN", 1.0) is None and _pool_pay({"pools": [0.0, 0.0]}, "UP", 1.0) is None
    assert _pool_pay({"pools": [None, None]}, "UP", 1.0) is None and _pool_pay({"pools": {"up": 1}}, "UP", 1.0) is None
    lad = a.ladders["thin"]
    a._ladder_step(lad, rec, "UP", pools=True)
    assert lad["pnl"] == 0.0 and lad["flat"] == 0.0 and lad["won"] == 1


def test_pool_pay_does_not_add_a_real_order_of_ours_twice():
    rec = {"pools": [127.0, 125.0], "call": "UP", "live": {"ok": True, "stake": 32.0}}    # others 95 / 125 + our real $32
    assert abs(_pool_pay(rec, "UP", 32.0) - (1 + POOL_FEE_K * 125 / 127)) < 1e-9         # what bc.game pays that order
    assert abs(_pool_pay(rec, "UP", 1.0) - (1 + POOL_FEE_K * 125 / 96)) < 1e-9           # the same call at $1
    assert abs(_pool_pay(rec, "DOWN", 1.0) - (1 + POOL_FEE_K * 127 / 126)) < 1e-9        # the other side: pools as they are
    failed = {"pools": [96.0, 125.0], "call": "UP", "live": {"ok": False, "stake": 1.0}}
    assert abs(_pool_pay(failed, "UP", 1.0) - (1 + POOL_FEE_K * 125 / 97)) < 1e-9        # not in the pool: nothing taken out


def test_thin_gate_text_arming_answer_and_shown_results(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    a.on = True
    a.set_model("fade")
    a.stats["fade_fwd_called"], a.stats["fade_fwd_correct"] = 100, 50
    assert a.gate_status()["text"].endswith("vs break-even 51.55%, needs 3000 calls")       # every other model: as before
    a.set_model("thin")
    a.stats["thin_fwd_called"], a.stats["thin_fwd_correct"] = 3000, 1500
    g = a.gate_status()
    assert "breaks even near 45.5% wins" in g["text"] and g["open"] is False and g["break_even"] == 0.5155
    a.progression = 2.0
    x = {"id": 9, "cutoff": T, "call": "UP", "win": "UP", "result": "WIN", "tie": False, "pnl": 1.17, "stake": 32.0,
         "payout": 2.17, "pools": [105.0, 131.0], "by": "thin"}
    a.settled.append(x)
    a.round = {"id": 10}
    assert f"{32 * POOL_FEE_K * 131 / 137:+.2f}" in a.panel_html()               # +29.07: what thin's ladder books
    a.settled.clear()
    a.settled.append({k: v for k, v in x.items() if k != "by"})                 # another model's call: as before
    assert "+28.44" in a.panel_html()
    a.settled.clear()
    a.recs[10] = {"id": 10, "cutoff": T, "call": "UP", "stake": 1.0, "payout": 2.24, "result": "WIN", "win": "UP",
                  "pools": [103.0, 125.0], "feat": {"pay_up": 2.2388, "pay_dn": 1.7155}, "by": "thin"}
    html = a.panel_html()
    assert "booked x2.14 at the final pools on $1" in html and "booked 224%" not in html
    a.recs[10].pop("by")                                                        # the same round called by another model
    assert "booked 224% on $1" in a.panel_html()
