"""bc.game Up/Down trading bot.

  python main.py              # open browser, log in, accept commands in this terminal + HTTP API
  python main.py --no-repl    # HTTP API only (for running in the background)
  python main.py --no-login   # don't log in (uses the site's demo balance)
  python main.py --auto       # also start the Hawkes auto mode (paper calls; "live arm ..." for real orders)
"""
import argparse
import asyncio
import logging
import threading
import time

from playwright.async_api import async_playwright

from bcbot import api, config
from bcbot.auto import AutoTrader
from bcbot.browser import open_trade_page
from bcbot.tap import TapAnalyzer
from bcbot.trader import TradeError, Trader

log = logging.getLogger("bcbot")

HELP = """Commands (amount is in the account's trading currency, shown by 'status'):
  up [amount]     buy UP    (e.g. "up 5"; only while armed, counted like every real order)
  dn [amount]     buy DOWN  (e.g. "dn 5")
  status          balance, round phase, payouts
  auto            auto mode: accuracy, paper P&L, models
  auto on|off     start / stop the automatic calls (paper: nothing is placed)
  live            the real-money state: losses in a row, orders today, budget, campaign, clock, gate
  live arm run=<n> day=<YYYY-MM-DD> [force] [ladder]   (ladder: the compensate ladder although the gate is closed)
                  allow REAL orders (AUTO_STAKE_USD each, flat). You must type the current run of real
                  losses and today's date. Needs LIVE_ALLOWED=1 and TOS_RISK_ACCEPTED=1 in .env.
                  The bot stops and disarms at AUTO_MAX_REAL_RUN losses in a row (only a real WIN resets
                  the count), at the daily loss limit, at the orders-per-day cap and at the campaign
                  loss; it never resumes by itself. Past the run limit a re-arm allows ONE order.
  live off        disarm (also: create the file logs/KILL)
  live clear <reason> [won|lost|none]   clear a technical stop after you checked the site
  live reset run=<n> <reason>   set the run of real losses back to 0 (your decision; logged, kept across restarts)
  model <name>    which model makes the calls: sign|mhh|agree|mom|book|bookmom|momvol|revert|overshoot|
                  overshoot_book|prob|gbm|ens (P(UP) fitted on the logged rounds: logistic, LightGBM trees,
                  or their average; each calls only when it is
                  at least 56% one way).  Older text: (HawkesSign, MHH flow, agree: only
                         when HawkesSign and MHH call the same side, mom: the direction of our price over
                         the last 5 s, book: with Binance's 20-level book when |imbalance| >= 0.7, bookmom:
                         only when that strong book and the 5 s price trend agree; book and bookmom skip the
                         other rounds); HawkesTime is scored but can't call
  stakes 1|1.5|2  the PAPER ladders' progression after a loss (real orders are always flat)
  login           log in again
  help            this text
  quit            stop the bot (browser stays open)"""


def _stdin_lines(loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    """Read stdin on a daemon thread so a blocked input() never prevents exit."""
    queue: asyncio.Queue = asyncio.Queue()

    def reader():
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                line = None
            loop.call_soon_threadsafe(queue.put_nowait, line)
            if line is None:
                return

    threading.Thread(target=reader, daemon=True).start()
    return queue


async def repl(trader: Trader, auto: AutoTrader, cfg: config.Config) -> None:
    print(HELP)
    lines = _stdin_lines(asyncio.get_running_loop())
    last_amount = cfg.default_amount
    while True:
        print("> ", end="", flush=True)
        line = await lines.get()
        if line is None:
            return
        parts = line.replace("﻿", "").strip().lower().split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]
        try:
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd in ("help", "?"):
                print(HELP)
            elif cmd == "status":
                print((await trader.status()).line())
            elif cmd == "login":
                await trader.login()
            elif cmd == "auto":
                if args and args[0] == "on":
                    await auto.start()
                elif args and args[0] == "off":
                    auto.stop()
                print(auto.line())
            elif cmd == "model" and args:
                auto.set_model(args[0])
                print(auto.line())
            elif cmd == "stakes" and args:
                auto.set_progression(float(args[0]))
                print(f"paper ladders: x{auto.progression:g} after a loss; real orders stay flat ${auto.stake():g}")
            elif cmd == "live":
                if not args:
                    print(auto.live_line())
                elif args[0] == "arm":
                    kv = dict(a.split("=", 1) for a in args[1:] if "=" in a)
                    if "run" not in kv or "day" not in kv:
                        print(f"live arm run=<losses in a row> day=<YYYY-MM-DD> [force]   (now: run={auto.real_run} "
                              f"day={time.strftime('%Y-%m-%d')})")
                        continue
                    print(auto.arm(int(kv["run"]), kv["day"], who="terminal", force="force" in args,
                                   ladder="ladder" in args))
                    print(auto.live_line())
                elif args[0] == "off":
                    auto.disarm("live off", who="terminal")
                    print(auto.live_line())
                elif args[0] == "clear" and len(args) >= 2:
                    print(auto.clear(args[1], args[2] if len(args) > 2 else None, who="terminal"))
                elif args[0] == "reset":
                    kv = dict(a.split("=", 1) for a in args[1:] if "=" in a)
                    if "run" not in kv or len(args) < 3:
                        print(f"live reset run=<losses in a row> <reason>   (now: run={auto.real_run})")
                        continue
                    print(auto.reset_run(int(kv["run"]), " ".join(a for a in args[2:] if "=" not in a), who="terminal"))
                    print(auto.live_line())
                elif args[0] == "on":
                    print(f"'live on' is gone. Arm with: live arm run={auto.real_run} day={time.strftime('%Y-%m-%d')}")
                else:
                    print("live | live arm run=<n> day=<date> [force] | live off | live clear <reason> [won|lost|none] "
                          "| live reset run=<n> <reason>")
            elif cmd in ("up", "u", "dn", "down", "d"):
                amount = float(args[0]) if args else last_amount
                r = await auto.manual_order(cmd, amount, who="terminal")
                last_amount = amount
                print(f"OK  {r['side'].upper()} {amount:g} {r['currency']} (${r['usd']:g}) [{r['account']}] "
                      f"payout ~{r['payout_pct']}%  order {r['order_id']}  round {r['round_id']}")
            else:
                print(f"Unknown command {cmd!r}. Type 'help'.")
        except TradeError as e:
            print(f"FAILED  {e}")
        except ValueError:
            print("Amount must be a number, e.g. 'up 5'")


async def main() -> None:
    parser = argparse.ArgumentParser(description="bc.game Up/Down bot")
    parser.add_argument("--no-repl", action="store_true", help="HTTP API only; don't read commands from stdin")
    parser.add_argument("--no-login", action="store_true", help="skip login (site demo balance)")
    parser.add_argument("--auto", action="store_true", help="start the Hawkes auto mode (paper calls)")
    parser.add_argument("--no-tap", action="store_true", help="don't open the Tap analyzer's tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = config.load()

    async with async_playwright() as pw:
        page = await open_trade_page(pw, cfg)
        trader = Trader(page, cfg, reopen=lambda: open_trade_page(pw, cfg))
        await trader.ensure_ready()
        if not args.no_login:
            await trader.login()
        st = await trader.wait_loaded()
        log.info(st.line())
        if st.account == "REAL":
            log.warning("REAL account: orders spend real funds (%s). Max per order: $%g (MAX_USD). Real orders "
                        "are off until 'live arm ...'%s", st.currency, cfg.max_usd,
                        "" if cfg.live_allowed else " (and LIVE_ALLOWED=1 in .env)")

        auto = AutoTrader(trader, cfg)
        if args.auto:
            await auto.start()
        tap = None
        if not args.no_tap:                      # Tap analyzer: its own bc.game tab, paper only, page /tap
            tap = TapAnalyzer(trader, cfg.auto_log.parent)
            try:
                await tap.start()
            except Exception as e:
                log.warning("Tap analyzer did not start: %r", e)
        runner = await api.start(trader, auto, cfg, tap)
        try:
            if args.no_repl:
                await asyncio.Event().wait()
            else:
                await repl(trader, auto, cfg)
        finally:
            await runner.cleanup()
    log.info("Bot stopped. The browser window stays open; the next start re-attaches to it.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
