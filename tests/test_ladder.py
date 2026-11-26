"""The real compensate ladder (AUTO_REAL_LADDER): used only while the go-live gate is open, its step follows the
persistent run of real losses, and the day's budget still refuses a stake that does not fit."""
import time

from tests.test_safety import make_cfg, FakeTrader
from bcbot.auto import AutoTrader


def bot(tmp_path, ladder, gate_open):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_real_ladder=ladder, auto_stake_usd=1.0))
    a.gate_status = lambda: {"open": gate_open, "text": "test"}
    a.progression = 1.0                 # "Flat" on the page: the .env ladder and the gate decide, as before
    return a


def test_flat_when_gate_closed(tmp_path):
    a = bot(tmp_path, (1.0, 2.0), gate_open=False)
    for run in range(6):
        a.real_run = run
        assert a.stake() == 1.0 and not a.ladder_on()


def test_ladder_steps_when_gate_open(tmp_path):
    a = bot(tmp_path, (1.0, 2.0), gate_open=True)
    assert a.ladder_on()
    assert [(setattr(a, "real_run", r), a.stake())[1] for r in range(5)] == [1.0, 2.0, 1.0, 2.0, 1.0]


def test_three_step_ladder(tmp_path):
    a = bot(tmp_path, (1.0, 2.0, 4.0), gate_open=True)
    assert [(setattr(a, "real_run", r), a.stake())[1] for r in range(4)] == [1.0, 2.0, 4.0, 1.0]


def test_flat_config_never_ladders(tmp_path):
    a = bot(tmp_path, (1.0,), gate_open=True)
    a.real_run = 3
    assert a.stake() == 1.0 and not a.ladder_on()


def test_ladder_forced_at_arming_while_gate_closed(tmp_path):
    a = bot(tmp_path, (1.0, 2.0), gate_open=False)
    a.on = True
    today = time.strftime("%Y-%m-%d")
    a.arm(0, today, force=True)                           # force alone: flat
    assert a.state == "ARMED" and not a.ladder_on() and a.stake() == 1.0
    a.disarm("test")
    a.arm(0, today, force=True, ladder=True)              # force + ladder: the ladder, for this arming
    assert a.ladder_on() and a.ladder_forced and a.stake() == 1.0
    a.real_run = 1
    assert a.stake() == 2.0                               # doubled after a real loss
    a.disarm("test")                                      # ends with the arming
    assert not a.ladder_forced and not a.ladder_on() and a.stake() == 1.0
    a.arm(1, today, force=True, ladder=True)
    a._hard_stop("test stop")                             # and with a stop
    assert not a.ladder_forced and a.state == "STOPPED"


def test_ladder_switched_while_armed(tmp_path):
    import pytest
    from bcbot.trader import TradeError
    a = bot(tmp_path, (1.0, 2.0), gate_open=False)
    a.on = True
    with pytest.raises(TradeError):
        a.set_ladder(True)                                # not armed: chosen at arming instead
    a.arm(0, time.strftime("%Y-%m-%d"), force=True)      # armed flat
    a.real_run = 1
    assert a.stake() == 1.0
    a.set_ladder(True)                                    # switched on while live: x2 after the loss
    assert a.state == "ARMED" and a.ladder_on() and a.stake() == 2.0
    a.set_ladder(False)                                   # and off again, still live
    assert a.state == "ARMED" and not a.ladder_on() and a.stake() == 1.0


def test_live_follows_the_stake_rule_of_the_page(tmp_path):
    """The owner's decision of 2026-11-10: x2 on the page doubles the REAL stake after each real loss too, for as
    many losses as the losses-in-a-row stop allows (no other cap); no gate is needed.  The limits are set on the page."""
    import pytest
    from bcbot.trader import TradeError
    a = bot(tmp_path, (1.0, 2.0), gate_open=False)
    a.set_progression(2.0)
    assert a.cfg.auto_max_real_run == 5 and a.ladder_on()
    assert [(setattr(a, "real_run", r), a.stake())[1] for r in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]     # stop after 5
    a.set_limit("max_real_run", 6)
    assert [(setattr(a, "real_run", r), a.stake())[1] for r in range(6)] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert a.summary()["real_ladder"] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0] and a.stake_cap() == 32.0
    a.set_limit("daily_loss_usd", 60)
    assert a.cfg.auto_daily_loss_usd == 60
    with pytest.raises(TradeError):
        a.set_limit("daily_loss_usd", 100000)
    lad = {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0}      # paper: the same series, then from $1 again
    for _ in range(7):
        a._ladder_step(lad, {"win": "UP", "payout": 1.94}, "DOWN")
    assert lad["biggest"] == 32.0 and lad["step"] == 1
    again = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_real_ladder=(1.0, 2.0), auto_stake_usd=1.0))
    assert again.progression == 2.0 and again.cfg.auto_max_real_run == 6 and again.cfg.auto_daily_loss_usd == 60
    a.real_run, a.on = 0, True
    a.arm(0, time.strftime("%Y-%m-%d"), force=True)
    # the owner's decision of 2026-11-12: the stake rule can be changed while real orders are on.  The next real
    # stake follows the new rule at the present run of losses, and the change is in the events
    a.real_run = 3
    assert a.stake() == 8.0
    a.set_progression(1.0, who="api")                    # Flat while LIVE: $1 whatever the run
    assert a.state == "ARMED" and a.stake() == 1.0 and not a.ladder_on()
    ev = [e for e in a.events if e["event"] == "stake_rule"]
    assert len(ev) == 1 and (ev[0]["was"], ev[0]["now"], ev[0]["stake"], ev[0]["who"]) == (2.0, 1.0, 1.0, "api")
    a.set_progression(1.0)                               # the same rule again: nothing to log
    assert len([e for e in a.events if e["event"] == "stake_rule"]) == 1
    a.set_progression(2.0)                               # back to x2 at 3 losses in a row: the next stake is $8 again
    assert a.state == "ARMED" and a.stake() == 8.0 and a.ladder_on() and a.stake_cap() == 32.0
    assert [e for e in a.events if e["event"] == "stake_rule"][-1]["stakes"] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_real_ladder=(1.0, 2.0), auto_stake_usd=1.0)).progression == 2.0
    with pytest.raises(TradeError):
        a.set_progression(5.0)                           # out of range: refused as before, and nothing changes
    assert a.progression == 2.0
    a.disarm("test")
    a.set_progression(1.0)                               # on Paper: as before, and no event
    assert a.stake() == 1.0 and not a.ladder_on()
    assert len([e for e in a.events if e["event"] == "stake_rule"]) == 2
