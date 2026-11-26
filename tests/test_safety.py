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
                  auto_kill_file=tmp_path / "KILL",
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

    async def usd_rate(self, currency, max_age=60.0):
        self.rate_asked = getattr(self, "rate_asked", []) + [max_age]
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
    import dataclasses
    a.cfg = dataclasses.replace(a.cfg, auto_max_real_bets_day=0)        # 0 (the default): no daily order count
    a.attempts_today = 500
    assert a._may_place(rid, rec, 1.0) is None
    a.cfg = dataclasses.replace(a.cfg, auto_max_real_bets_day=60)
    a.attempts_today = 60
    assert "today" in a._may_place(rid, rec, 1.0)
    a.attempts_today = 0
    a.set_max_real_run(7, who="test")                           # the operator's own losses-in-a-row stop
    a.real_run = 6
    assert a._may_place(rid, rec, 1.0) is None
    a.real_run = 7
    assert "losses in a row (limit 7)" in a._may_place(rid, rec, 1.0)
    a.real_run = 0
    again = AutoTrader(FakeTrader(), dataclasses.replace(a.cfg, auto_max_real_run=5))
    assert again.cfg.auto_max_real_run == 7                     # kept across a restart, beats .env
    a.day_live_pnl = -a.cfg.auto_daily_loss_usd + 0.5
    assert "budget" in a._may_place(rid, rec, 1.0)
    a.day_live_pnl = 0.0
    a.balance_usd = 5.0                                         # the 20-stakes floor is off since 2026-11-12 (the
    assert a._may_place(rid, rec, 1.0) is None                  # owner's decision): $5 covers a $1 order
    a.balance_usd = 0.5
    assert a._may_place(rid, rec, 1.0) == "balance $0.50 is less than the stake $1"
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


def test_a_late_tick_or_a_refused_order_does_not_hold_the_next_order_back(tmp_path):
    """The owner, 2026-11-17, LIVE on XGBoost: "Rounds where bc.game's price feed is stale ... The round after
    bc.game refuses an order (45 s wait).  Remove this."  What still holds an order back: a clock that cannot time
    it (too few ticks, none for 20 s, medians apart) and another feed being stale."""
    a, rid, rec = checklist_trader(tmp_path)
    assert a._may_place(rid, rec, 1.0) is None
    a.clock.last_arrival = time.time() - 3.0                     # bc.game's last tick came 3 s ago
    assert not a.clock.ok() and a.clock.usable() and a.clock_src() == "bc"
    assert a._may_place(rid, rec, 1.0) is None
    rec["feat"]["dq"] = {"ok": False, "stale": ["bc_ticks"], "held_by": []}        # as _features writes it now
    assert a._may_place(rid, rec, 1.0) is None
    rec["feat"]["dq"] = {"ok": False, "stale": ["bc_ticks"]}                       # a record without held_by
    assert a._may_place(rid, rec, 1.0) is None
    rec["feat"]["dq"] = {"ok": False, "stale": ["binance_book", "bc_ticks"], "held_by": ["binance_book"]}
    assert a._may_place(rid, rec, 1.0) == "data quality: binance_book"             # another feed still does
    rec["feat"]["dq"] = {"ok": True, "stale": []}
    a._rejects.append((rid - 1, time.time()))                    # bc.game refused the order of the round before
    assert a._may_place(rid, rec, 1.0) is None
    for _ in range(40):                                          # no tick in the last 20 s: the clock cannot be used
        a.clock.samples.pop()
    assert not a.clock.usable() and "no tick in the last 20 s" in a._may_place(rid, rec, 1.0)
    a.clock.samples.clear()
    assert "bc.game clock not ok (only 0 ticks" in a._may_place(rid, rec, 1.0)


def test_the_balance_read_before_a_click_does_not_wait_for_the_price_table(tmp_path):
    """2026-11-17: in every second round the read before the click fetched bc.game's price table (0.6-1.6 s), and
    five of 34 real orders came too late to click.  It now uses the table read with the round's preparation."""
    from bcbot.auto import RATE_KEEP_S
    a, rid, rec = checklist_trader(tmp_path)

    async def click():
        return {"ok": True, "sent": time.time(), "acked": time.time(), "rtt": 0.1, "order_id": "1", "round_id": str(rid),
                "amount": 1.0, "currency": "USDT", "side": "up"}

    async def run():
        await a._prepare(0)                                   # the round's preparation: the table may be read
        await a._send(rid, "UP", rec, 1.0, "auto", click)     # right before the click: never
    a.trader.prepare_usd = lambda usd: asyncio.sleep(0)
    a.state = "DISARMED"                                      # (prepare only reads the balance when not armed)
    asyncio.run(run())
    assert a.trader.rate_asked == [60.0, RATE_KEEP_S] and RATE_KEEP_S >= 600
    assert rec["live"]["ok"] and a.balance_usd == 50.0


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


def test_a_technical_stop_says_what_it_is_and_is_cleared_on_the_page(tmp_path):
    """2026-11-17: the owner pressed LIVE and read "stopped: None. Check the site, then 'live clear currency'".  The
    stop was five days old, its reason had been lost when Paper was pressed, and the page had no way to clear it."""
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WL"), t0)
    a = trader_for(tmp_path)
    a.on, a.state = True, "ARMED"
    a._hard_stop("wallet currency changed from USDT to BCD", tech="currency")
    a.disarm("live off", who="api")                   # Paper pressed after the stop: the stop and its reason stay
    assert a.state == "STOPPED" and a.tech_stop == "currency" and a.stop_reason == "wallet currency changed from USDT to BCD"
    st = json.loads(a.cfg.auto_state_file.read_text(encoding="utf-8"))
    st["stop_reason"] = None                          # a state file written before the fix: the reason is gone
    a.cfg.auto_state_file.write_text(json.dumps(st), encoding="utf-8")
    a2 = trader_for(tmp_path)                         # ... and is found again in the audit trail
    assert a2.state == "STOPPED" and a2.stop_reason == "wallet currency changed from USDT to BCD"
    a2.on = True
    for refused in (lambda: a2.arm(a2.real_run, time.strftime("%Y-%m-%d"), force=True),
                    lambda: a2.reset_run(a2.real_run, "new model")):
        with pytest.raises(TradeError) as e:
            refused()
        text = str(e.value)
        assert "None" not in text and "live clear" not in text
        assert "wallet currency changed from USDT to BCD" in text and "Clear stop" in text and "wallet selected" in text
    s = a2.summary()
    assert s["tech_stop"] == "currency" and s["stop_text"] == text and s["stopped_at"]
    assert "cleared" in a2.clear("currency", who="api")
    assert a2.state == "DISARMED" and a2.tech_stop is None and a2.summary()["stop_text"] is None
    a2.arm(a2.real_run, time.strftime("%Y-%m-%d"), force=True)        # cleared: LIVE is the owner's next click
    assert a2.state == "ARMED"
    a2.disarm("live off", who="api")                  # Paper without a technical stop: as before
    assert a2.state == "DISARMED" and a2.stop_reason is None


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


def test_run_reset_is_typed_logged_and_kept_across_restarts(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WLLLLLL"), t0)
    a = trader_for(tmp_path, auto_rearm_cooloff_min=0)
    a.on = True
    a.stopped_at = 0.0
    assert a.real_run == 6 and a.state == "STOPPED"
    with pytest.raises(TradeError):
        a.reset_run(5, "new model")                      # wrong count
    with pytest.raises(TradeError):
        a.reset_run(6, "")                               # no reason
    assert a.real_run == 6
    a.reset_run(6, "new model overshoot")
    assert a.real_run == 0 and a.state == "DISARMED" and a.run_reset_at > 0
    a2 = trader_for(tmp_path)                            # a restart keeps the reset: the old losses no longer count
    assert a2.real_run == 0 and a2.state == "DISARMED"
    events = [json.loads(line) for line in (tmp_path / "live_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("event") == "run_reset" and e.get("was") == 6 for e in events)
    a2.on = True                                         # a real loss after the reset counts again
    a2.arm(0, time.strftime("%Y-%m-%d"), force=True)
    assert a2.state == "ARMED" and not a2.rearm_grant
    rec = {"id": 3000, "cutoff": time.time() + 3, "live": {"ok": True, "stake": 1.0, "amount": 1.0, "payout_pct": 194},
           "live_pnl": -1.0, "result": "LOSS", "call": "UP"}
    a2.pending_real[3000] = 1.0
    a2._real_settled(rec, "LOSS")
    assert a2.real_run == 1 and a2.state == "ARMED"


def trader_for_armed_message(tmp_path):
    """What arm() answers when a time limit is set (AUTO_ARM_TTL_H=4)."""
    t = trader_for(tmp_path, auto_resume_arm=False, auto_arm_ttl_h=4)
    t.on = True
    return t.arm(t.real_run, time.strftime("%Y-%m-%d"), force=True)


def test_arming_goes_on_after_a_restart_until_it_expires(tmp_path):
    t0 = time.time() - 3600
    write_rounds(tmp_path / "auto_rounds.jsonl", list("WL"), t0)
    a = trader_for(tmp_path, auto_resume_arm=True)
    a.on = True
    said = a.arm(1, time.strftime("%Y-%m-%d"), force=True, ladder=True)
    assert a.state == "ARMED" and a.ladder_forced
    assert a.cfg.auto_arm_ttl_h == 0 and "until you switch to Paper" in said     # the owner's decision of 2026-11-12:
    a2 = trader_for(tmp_path, auto_resume_arm=True)        # no time limit.  A restart: the arming goes on, ladder included
    assert a2.state == "ARMED" and a2.ladder_forced and a2.armed_at == a.armed_at
    a2.armed_at -= 5 * 3600                                 # make it look 5 h old and save
    a2._save_state()
    old = trader_for(tmp_path, auto_resume_arm=True)       # still on: LIVE ends with Paper or a stop, not with the clock
    assert old.state == "ARMED" and old._may_place(1, {"cutoff": 0}, 1.0) != "arming expired"
    a3 = trader_for(tmp_path, auto_resume_arm=True, auto_arm_ttl_h=4)       # with a limit set in .env: expired, not resumed
    assert a3.state == "DISARMED"
    assert "for up to 4 h" in trader_for_armed_message(tmp_path)
    a.arm(1, time.strftime("%Y-%m-%d"), force=True)
    a4 = trader_for(tmp_path, auto_resume_arm=False)       # the switch off: every start is disarmed, as before
    assert a4.state == "DISARMED"


def test_a_real_order_only_needs_the_balance_to_cover_its_stake(tmp_path):
    """The owner's decision of 2026-11-12, after a winning $7.59 step was kept on paper at a balance of $18.90: the
    'balance must be at least 20 stakes' rule is off.  The balance must still cover the order itself, and
    AUTO_MIN_BANKROLL_STAKES still works when it is set."""
    import dataclasses
    a, rid, rec = checklist_trader(tmp_path, auto_stake_usd=1.0)
    assert a.cfg.auto_min_bankroll_stakes == 0
    a.set_progression(1.5)
    a.set_max_real_run(9, who="test")
    a.real_run, a.balance_usd = 5, 18.90
    assert a.stake() == 7.59
    a._prepared_stake = 7.59
    assert a._may_place(rid, rec, 7.59) is None                         # placed: $18.90 covers $7.59
    a.balance_usd = 7.0
    assert a._may_place(rid, rec, 7.59) == "balance $7.00 is less than the stake $7.59"
    a.balance_usd = 18.90
    a.cfg = dataclasses.replace(a.cfg, auto_min_bankroll_stakes=20)      # the old floor, when someone sets it
    assert a._may_place(rid, rec, 7.59) == "balance $18.90 is below 20 stakes"
    a.cfg = dataclasses.replace(a.cfg, auto_min_bankroll_stakes=0)
    a.disarm("test")
    a.real_run = 0
    a.arm(0, time.strftime("%Y-%m-%d"), force=True)                     # arming at $18.90: allowed
    assert a.state == "ARMED"
    a.disarm("test")
    a.balance_usd = 0.4
    with pytest.raises(TradeError, match="below the stake"):
        a.arm(0, time.strftime("%Y-%m-%d"), force=True)                 # under one base stake: nothing to bet with
