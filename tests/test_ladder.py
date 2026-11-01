"""The real compensate ladder (AUTO_REAL_LADDER): used only while the go-live gate is open, its step follows the
persistent run of real losses, and the day's budget still refuses a stake that does not fit."""
import time

from tests.test_safety import make_cfg, FakeTrader
from bcbot.auto import AutoTrader


def bot(tmp_path, ladder, gate_open):
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_real_ladder=ladder, auto_stake_usd=1.0))
    a.gate_status = lambda: {"open": gate_open, "text": "test"}
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
