"""The "value" model: the chance of each side (from the gap and the pools) times what the page pays on it; a bet from
an expected return of 5%; paid from the round's final pools."""
import asyncio

from bcbot.auto import (MODEL_REGISTERED, PAY_K, POOL_BOOKED, POOL_FEE_K, VALUE_FITTED_UNTIL, VALUE_GAP_HI, VALUE_GAP_LO,
                        VALUE_MIN_EV, VALUE_W_GAP, VALUE_W_POOL, AutoTrader, _thin_call, _value_call, _value_read)
from tests.test_safety import FakeTrader, make_cfg

T = MODEL_REGISTERED["value"] + 100                # a round after the rule went into the bot


def feat(pay_up, gap=0.0, vol=2.0, **more):
    """A snapshot where the page pays `pay_up` on UP (the other side follows from the same pools) and bc.game is
    `gap` volatilities above the market (a negative gap: below it, so the bet against the gap is UP)."""
    return dict({"pay_up": pay_up, "pay_dn": 1 + PAY_K ** 2 / (pay_up - 1), "dev_tick": gap * vol, "vol60": vol}, **more)


def flip(f):
    """The same snapshot with UP and DOWN exchanged."""
    return dict(f, pay_up=f["pay_dn"], pay_dn=f["pay_up"], dev_tick=-f["dev_tick"])


def test_value_needs_less_payout_the_bigger_the_gap_on_its_side():
    assert (VALUE_GAP_LO, VALUE_GAP_HI, VALUE_W_GAP, VALUE_W_POOL, VALUE_MIN_EV) == (1.0, 4.0, 0.1154, 0.0831, 0.05)
    assert _value_call(feat(2.12)) == "UP" and _value_call(feat(2.11)) is None           # no gap: the payout alone
    assert _value_call(feat(2.12, gap=-0.9)) == "UP" and _value_call(feat(2.11, gap=-0.9)) is None   # under 1 x: no gap
    assert _value_call(feat(2.00, gap=-2.0)) == "UP" and _value_call(feat(1.97, gap=-2.0)) is None
    assert _value_call(feat(1.89, gap=-3.0)) == "UP" and _value_call(feat(1.86, gap=-3.0)) is None
    assert _value_call(feat(1.80, gap=-4.0)) == "UP" and _value_call(feat(1.75, gap=-4.0)) is None
    assert _value_read(feat(1.80, gap=-9.0))[1:] == _value_read(feat(1.80, gap=-4.0))[1:]   # the gap counts up to 4 x
    assert _value_read(feat(2.00, gap=-0.9))[1:] == _value_read(feat(2.00))[1:]
    for f in (feat(2.12), feat(2.00, gap=-2.0), feat(1.80, gap=-4.0)):                   # UP and DOWN are symmetric
        assert _value_call(flip(f)) == "DOWN"
        assert abs(_value_read(flip(f))[2] - _value_read(f)[2]) < 1e-12
    assert _value_call(flip(feat(1.97, gap=-2.0))) is None


def test_value_reads_the_chance_from_the_gap_and_the_pools():
    side, p, ev = _value_read(feat(1.9415))                                  # even pools, no gap: a coin at x1.94
    assert abs(p - 0.5) < 1e-9 and abs(ev - (0.5 * 1.9415 - 1)) < 1e-9 and _value_call(feat(1.9415)) is None
    side, p, ev = _value_read(feat(1.9415, gap=-2.0))                        # 2 x below the market: UP 52.9%
    assert side == "UP" and abs(p - 0.5288) < 5e-4 and abs(ev - (p * 1.9415 - 1)) < 1e-12
    assert abs(_value_read(feat(1.9415, gap=-3.0))[1] - 0.5574) < 5e-4       # 3 x: 55.7%
    assert abs(_value_read(feat(1.9415, gap=-4.0))[1] - 0.5857) < 5e-4       # 4 x: 58.6%
    assert _value_read(feat(1.9415, gap=2.0))[0] == "DOWN"                   # above the market: the bet is DOWN
    side, p, ev = _value_read(feat(2.20))                                    # the smaller pool wins a little less often
    assert side == "UP" and 0.49 < p < 0.5 and abs(ev - (p * 2.20 - 1)) < 1e-12
    # a side that pays well with a big gap against it: neither side is worth a bet
    f = feat(2.15, gap=3.0)
    assert _thin_call(f) == "UP" and _value_call(f) is None and _value_read(f)[2] < 0


def test_value_makes_no_call_without_its_inputs():
    good = feat(2.20, gap=-2.0)
    assert _value_call(good) == "UP"
    assert _value_call(None) is None and _value_call({}) is None and _value_read(None) is None
    assert _value_call(dict(good, error="boom")) is None
    assert _value_call(dict(good, pools_prev=True)) is None                  # the previous round's pools: no bet
    for k in ("dev_tick", "vol60", "pay_up", "pay_dn"):
        assert _value_call(dict(good, **{k: None})) is None
        assert _value_call({x: v for x, v in good.items() if x != k}) is None
        assert _value_call(dict(good, **{k: float("nan")})) is None
        assert _value_call(dict(good, **{k: True})) is None
    assert _value_call(dict(good, vol60=0.0)) is None and _value_call(dict(good, vol60=-1.0)) is None
    assert _value_call(dict(good, pay_up=1.0)) is None and _value_call(dict(good, pay_dn=0.0)) is None
    assert _value_call(dict(good, dev_tick=0, pay_up=2.2, pay_dn=1.74)) == "UP"          # a whole-number gap of 0: fine


def test_value_is_paid_from_the_final_pools_and_counts_only_later_rounds(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    assert "value" in AutoTrader.CALLERS and POOL_BOOKED == ("thin", "value")
    rec = {"id": 1, "cutoff": T, "win": "UP", "tie": False, "pools": [100.0, 104.0], "feat": feat(1.98, gap=-3.0)}
    paid = POOL_FEE_K * 104 / 101                                           # our $1 inside our side's final pool
    assert a._model_calls(rec)["value"] == "UP" and a._model_calls(rec)["thin"] is None
    assert a._model_calls(dict(rec, cutoff=MODEL_REGISTERED["value"] - 100))["value"] is None     # before its time
    # its record counts from the time it went into the bot (2026-11-16 02:28 UTC), not from the end of its fit: the
    # 817 bets in between were its test on the log, and the page must not show them as bets the bot made
    assert MODEL_REGISTERED["value"] == 1794796080 > VALUE_FITTED_UNTIL == MODEL_REGISTERED["thin"]
    assert a._model_calls(dict(rec, cutoff=VALUE_FITTED_UNTIL + 3600))["value"] is None
    a._count(rec)
    s = a.stats
    assert (s["value_called"], s["value_correct"], s["value_fwd_called"], s["value_fwd_correct"]) == (1, 1, 1, 1)
    assert abs(s["value_pnl"] - paid) < 1e-3
    a._count(dict(rec, id=2, win="DOWN"))
    assert s["value_called"] == 2 and s["value_correct"] == 1 and abs(s["value_pnl"] - (paid - 1)) < 1e-3
    lad = a.ladders["value"]
    assert lad["bets"] == 2 and abs(lad["flat"] - (paid - 1)) < 1e-3        # its paper series: the same final pools


def test_value_can_be_selected_and_its_bet_is_shown_with_its_reason(tmp_path):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_stake_usd=1.0))
    a.on = True
    a.set_model("value")
    assert a.model == "value"
    g = a.gate_status()
    assert "judge it by its P&L" in g["text"] and "47-50% wins" in g["text"] and g["open"] is False
    rec = {"id": 7, "cutoff": T, "start": T - 0.3, "end": T + 4.7, "feat": feat(2.00, gap=-2.0)}
    a.round, a.recs[7] = {"id": 7}, rec
    asyncio.run(a._call(7, rec, _value_call(rec["feat"])))
    assert rec["call"] == "UP" and rec["by"] == "value" and rec["stake"] == 1.0
    html = a.panel_html()
    assert "chance 53% &times; pays x2.00 = +6% expected" in html and "gap 2.0 x volatility" in html
    rec.update(result="WIN", win="UP", pools=[100.0, 104.0])                 # won: booked at the final pools, not the page's
    assert f"booked x{1 + POOL_FEE_K * 104 / 101:.2f} at the final pools on $1" in a.panel_html()
    skipped = {"id": 8, "cutoff": T + 30, "feat": feat(1.95)}
    asyncio.run(a._call(8, skipped, _value_call(skipped["feat"])))
    assert skipped.get("skip") == "no call" and "call" not in skipped
    a.set_model("fade")                                                      # another model's call is not marked
    other = {"id": 9, "cutoff": T + 60, "feat": feat(2.00, gap=-2.0)}
    asyncio.run(a._call(9, other, "UP"))
    assert "by" not in other


def test_value_arming_answer_warns_about_a_stake_series(tmp_path):
    import time

    from tests.test_safety import trader_for
    a = trader_for(tmp_path, auto_resume_arm=False, auto_progression=1.0, auto_real_ladder=(1.0,))
    a.on = True
    a.set_model("value")
    msg = a.arm(a.real_run, time.strftime("%Y-%m-%d"), force=True)           # flat: no note, said as any model
    assert "Value bet with a stake rule" not in msg and "by Value bet" in msg and "its edge is not proven" in msg
    a.set_progression(1.5)                                                   # the owner may: it is a note, not a refusal
    assert a.state == "ARMED"
    a.disarm("test")
    msg = a.arm(a.real_run, time.strftime("%Y-%m-%d"), force=True)
    assert "Value bet with a stake rule" in msg and a.state == "ARMED"
