"""The real-money rules of AutoTrader without a browser: the persistent run of real losses rebuilt from the log,
the stop at the limit, no resume, the one-order re-arm, the pre-order checklist and the arming command.
Run: .venv\\Scripts\\python.exe -m pytest tests -q"""
import asyncio
import json
import time
from pathlib import Path

import pytest

from bcbot import config
from bcbot.auto import AutoTrader, BETTING
from bcbot.trader import TradeError

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(tmp_path: Path, **over) -> config.Config:
    base = config.load()
    fields = {k: getattr(base, k) for k in base.__dataclass_fields__}
    fields.update(auto_log=tmp_path / "auto_rounds.jsonl", log_file=tmp_path / "trades.jsonl",
                  auto_state_file=tmp_path / "live_state.json", auto_events=tmp_path / "live_events.jsonl",
                  auto_kill_file=tmp_path / "KILL", flow_model=ROOT / "models" / "flow5s_logit.json",
                  live_allowed=True, tos_risk_accepted=True, auto_model="sign")
    fields.update(over)
    return config.Config(**fields)


class FakeTrader:
    page_listeners: list = []

    def __init__(self):
        self.calls = []

    async def status(self, light=False):
        from bcbot.trader import Status
        return Status(True, "REAL", 50.0, "USDT", "open", 10, 194, 194, 77000.0, "5s $1-50", None)

    async def usd_rate(self, currency):
        return 1.0


def write_rounds(path: Path, results: list, t0: float, stake: float = 1.0) -> None:
    """A round log with one real order per entry: results are "W" / "L" / "P" (paper only) / "T" (tie)."""
    rows = []
    for i, r in enumerate(results):
        t = t0 + 30 * i
        rec = {"id": 1000 + i, "cutoff": t, "start": t - 0.27, "end": t + 4.68, "win": "UP", "tie": r == "T",
               "call": "UP" if r in ("W", "T") else "DOWN", "payout": 1.94,
               "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))}
        rec["result"] = "TIE" if r == "T" else "WIN" if r == "W" else "LOSS"
        if r != "P":
            rec["live"] = {"ok": True, "order_id": str(5000 + i), "round_id": str(1000 + i), "stake": stake,
                           "amount": stake, "currency": "USDT", "payout_pct": 194, "side": rec["call"].lower()}
            rec["live_pnl"] = 0.0 if r == "T" else (0.94 * stake if r == "W" else -stake)
        rows.append(rec)
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def trader_for(tmp_path, **over):
    cfg = make_cfg(tmp_path, **over)
    a = AutoTrader(FakeTrader(), cfg)
    a._load_state()
    a._load_log()
    a._check_start()
    return a


def test_run_rebuilt_from_log_in_time_order(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WLLPLTL"), t0)      # paper rounds and ties leave it
    a = trader_for(tmp_path)
    assert a.real_run == 4
    assert a.state == "DISARMED"


def test_start_stopped_at_limit_and_not_reset_by_restart(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WLLLLL"), t0)
    a = trader_for(tmp_path)
    assert a.real_run == 5 and a.state == "STOPPED"
    a2 = trader_for(tmp_path)                        # a restart changes nothing
    assert a2.real_run == 5 and a2.state == "STOPPED"


def test_arm_needs_exact_acknowledgement(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WLL"), t0)
    a = trader_for(tmp_path)
    a.on = True
    today = time.strftime("%Y-%m-%d")
    with pytest.raises(TradeError):
        a.arm(0, today, force=True)                  # wrong run
    with pytest.raises(TradeError):
        a.arm(2, "2000-01-01", force=True)           # wrong day
    with pytest.raises(TradeError):
        a.arm(2, today)                              # the gate is closed: needs force
    msg = a.arm(2, today, force=True)
    assert a.state == "ARMED" and "ARMED" in msg and not a.rearm_grant


def test_arm_refused_without_env_flags(tmp_path):
    a = trader_for(tmp_path, live_allowed=False)
    a.on = True
    with pytest.raises(TradeError):
        a.arm(0, time.strftime("%Y-%m-%d"), force=True)


def test_rearm_beyond_limit_allows_one_order_and_a_loss_stops_again(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("LLLLL"), t0)
    a = trader_for(tmp_path, auto_rearm_cooloff_min=0)
    a.on = True
    a.stopped_at = 0.0
    a.arm(5, time.strftime("%Y-%m-%d"), force=True)
    assert a.state == "ARMED" and a.rearm_grant
    rec = {"id": 2000, "cutoff": time.time() + 3, "live": {"ok": True, "stake": 1.0, "amount": 1.0, "payout_pct": 194},
           "live_pnl": -1.0, "result": "LOSS", "call": "UP"}
    a.pending_real[2000] = 1.0
    a._real_settled(rec, "LOSS")                     # the one order lost
    assert a.real_run == 6 and a.state == "STOPPED" and not a.rearm_grant
    # a WIN on the next re-arm resets the run
    a.stopped_at = 0.0
    a.arm(6, time.strftime("%Y-%m-%d"), force=True)
    rec = {"id": 2001, "cutoff": time.time() + 3, "live": {"ok": True, "stake": 1.0, "amount": 1.0, "payout_pct": 194},
           "live_pnl": 0.94, "result": "WIN", "call": "UP"}
    a._real_settled(rec, "WIN")
    assert a.real_run == 0 and a.state == "ARMED"


def test_paper_results_never_touch_the_real_run(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("LLL"), t0)
    a = trader_for(tmp_path)
    a.real_run_before = a.real_run
    for _ in range(10):                              # ten paper wins
        a._shadow_settled({"shadow": {"ok": True}}, "WIN")
    assert a.real_run == 3


def checklist_trader(tmp_path, **over):
    a = trader_for(tmp_path, auto_rearm_cooloff_min=0, **over)
    a.on = True
    a.state, a.armed_at = "ARMED", time.time()
    a.balance, a.balance_currency, a.balance_account, a.balance_usd, a.balance_at = 50.0, "USDT", "REAL", 50.0, time.time()
    a.last_reload_at = 0.0
    # a bc clock that is ok, and an open round 3 s before its cutoff
    now = time.time()
    for i in range(60):
        a.clock.note(now - 0.5 * (59 - i), now - 0.5 * (59 - i))
    rid = 777
    cutoff = a.bc_now() + 3.0
    a.round = {"id": rid, "status": BETTING, "tradeCutoffTime": cutoff * 1000}
    rec = {"id": rid, "cutoff": cutoff, "start": cutoff - 0.27, "end": cutoff + 4.68,
           "pred": {"ready": True, "lead_s": 2.7}, "feat": {"dq": {"ok": True, "stale": []}}}
    a._prepared_rid, a._prepared_stake = rid, 1.0
    return a, rid, rec


def test_checklist_passes_then_each_rule_refuses(tmp_path):
    a, rid, rec = checklist_trader(tmp_path)
    assert a._may_place(rid, rec, 1.0) is None
    a.real_run = 5
    assert "losses in a row" in a._may_place(rid, rec, 1.0)
    a.real_run = 0
    a.pending_real[1] = 1.0
    assert "not settled" in a._may_place(rid, rec, 1.0)
    a.pending_real.clear()
    a.attempts_today = a.cfg.auto_max_real_bets_day
    assert "today" in a._may_place(rid, rec, 1.0)
    a.attempts_today = 0
    a.day_live_pnl = -a.cfg.auto_daily_loss_usd + 0.5
    assert "budget" in a._may_place(rid, rec, 1.0)
    a.day_live_pnl = 0.0
    a.balance_usd = 5.0
    assert "below" in a._may_place(rid, rec, 1.0)
    a.balance_usd = 50.0
    a.balance_at = time.time() - 60
    assert "balance not read" in a._may_place(rid, rec, 1.0)
    a.balance_at = time.time()
    a.balance_account = "DEMO"
    assert "not REAL" in a._may_place(rid, rec, 1.0)
    a.balance_account = "REAL"
    a.dialog = "Risk warning"
    assert "dialog" in a._may_place(rid, rec, 1.0)
    a.dialog = None
    rec["cutoff"] = a.bc_now() + 1.0
    assert "too late" in a._may_place(rid, rec, 1.0)
    rec["cutoff"] = a.bc_now() + 3.0
    a.last_reload_at = time.time()
    assert "reloaded" in a._may_place(rid, rec, 1.0)
    a.last_reload_at = 0.0
    a._prepared_rid = 0
    assert "not prepared" in a._may_place(rid, rec, 1.0)
    a._prepared_rid = rid
    rec["feat"]["dq"] = {"ok": False, "stale": ["bitfinex_perp"]}
    assert "data quality" in a._may_place(rid, rec, 1.0)
    rec["feat"]["dq"] = {"ok": True, "stale": []}
    a.cfg.auto_kill_file.write_text("x")
    assert "kill" in a._may_place(rid, rec, 1.0)
    a.cfg.auto_kill_file.unlink()
    a.state = "DISARMED"
    assert "not armed" in a._may_place(rid, rec, 1.0)


def test_shadow_checks_run_while_disarmed(tmp_path):
    a, rid, rec = checklist_trader(tmp_path)
    a.state = "DISARMED"
    assert a._may_place(rid, rec, 1.0, shadow=True) is None
    a.shadow_run = 5
    assert "losses in a row" in a._may_place(rid, rec, 1.0, shadow=True)


def test_unmatched_trade_is_unknown_and_blocks(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WL"), t0)
    (tmp_path / "trades.jsonl").write_text(json.dumps({"ok": True, "order_id": "999", "side": "up", "amount": 1.0,
                                                       "currency": "USDT", "sent": t0 + 100,
                                                       "time": "2026-10-30 10:00:00"}) + "\n", encoding="utf-8")
    a = trader_for(tmp_path)
    assert a.unknown == ["999"] and a.state == "STOPPED" and a.tech_stop == "unknown"
    assert a.real_run == 2                            # the unknown order counts as a loss until cleared
    a.on = True
    with pytest.raises(TradeError):
        a.arm(2, time.strftime("%Y-%m-%d"), force=True)
    assert "cleared" in a.clear("unknown", "none")
    assert a.real_run == 1 and a.state == "DISARMED" and a.cleared == ["999"]
    a2 = trader_for(tmp_path)                         # the clearing survives a restart
    assert a2.unknown == [] and a2.real_run == 1


def test_kill_file_disarms(tmp_path):
    a, rid, rec = checklist_trader(tmp_path)
    a.cfg.auto_kill_file.write_text("x")
    a._watch()
    assert a.state == "DISARMED"
    a.cfg.auto_kill_file.unlink()
    a._watch()
    assert a.state == "DISARMED"                      # deleting the file does not re-arm


def test_manual_order_refused_when_not_armed(tmp_path):
    a = trader_for(tmp_path)
    a.on = True
    a.round = {"id": 1, "status": BETTING, "tradeCutoffTime": (time.time() + 5) * 1000}
    with pytest.raises(TradeError):
        asyncio.run(a.manual_order("up", 1.0))
