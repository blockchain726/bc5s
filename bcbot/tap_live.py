"""Live taps on bc.game's Tap game (BTC/USD), off unless TAP_LIVE is set in .env.

A tap is placed the way a person places it: the bot clicks the chosen cell on the page's grid, and the page builds the
order (POST /api/transaction/tapOrder/create with the cell's time and price limits, the amount and the currency) and
signs it.  The order carries no multiplier: the SERVER sets it.  So the bot records, for every order, the server's
multiplier next to the label the page showed for that cell, and only expects what the server has been paying.

Modes (TAP_LIVE): off | demo (the page's demo account: real server multipliers, no money) | real (account money).

Rules, all checked right before each click:
  * the market is open, bc.game's labels are being read, the page shows the account the mode asks for
  * at most one tap per TAP_MIN_GAP_S, TAP_MAX_OPEN open taps, never the same cell twice
  * the cell starts >= 4.3 s after the last price tick (the page refuses closer ones), and its EV at the multiplier
    the server is expected to pay (label x the median server/label ratio of our recent orders) is >= TAP_MIN_EV
  * real money: the server/label ratio must be known from >= TAP_RATIO_MIN_N orders (demo or real), and the
    real P&L minus the open stakes minus this stake must stay above -TAP_BUDGET_USD: at that point real taps
    stop for good (logs/tap_live_state.json keeps the P&L across restarts)
  * TAP_LOSS_RUN_PAUSE losses in a row: pause TAP_PAUSE_MIN minutes
  * an order sent without an answer, a booking in another cell, or 2 rejections in 10 minutes: stop (technical)
  * logs/TAP_KILL: stop
Every attempt, order and result goes to logs/tap_live.jsonl.
"""
import asyncio
import json
import logging
import os
import statistics
import time
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

ORDER_URL = "/api/transaction/tapOrder/create"
MIN_AHEAD_MS = 4300

# where to click: the centre of a cell, through the same axis fits our boxes are drawn with; null when the axis
# labels are stale or the point is off the chart
CLICK_POINT_JS = r"""(d) => {
  const g = window.__tapGeom, now = performance.now();
  if (!g) return {err: 'no hook'};
  const fresh = o => Object.entries(o).filter(([, v]) => now - v[1] < 1000);
  const T = fresh(g.times), P = fresh(g.prices);
  if (T.length < 2 || P.length < 2) return {err: 'axis labels ' + T.length + '/' + P.length};
  const ref = new Date(d.now);
  const tx = T.map(([s, v]) => {
    const [h, m, sec] = s.split(':').map(Number), dt = new Date(ref);
    dt.setHours(h, m, sec, 0);
    let ms = dt.getTime();
    if (ms - d.now > 43200000) ms -= 86400000;
    if (d.now - ms > 43200000) ms += 86400000;
    return [ms, v[0]];
  });
  const py = P.map(([s, v]) => [parseFloat(s.replace(/,/g, '')), v[0]]);
  const fit = pts => {
    const n = pts.length, mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
    let sxx = 0, sxy = 0;
    pts.forEach(p => { sxx += (p[0] - mx) ** 2; sxy += (p[0] - mx) * (p[1] - my); });
    const b = sxx ? sxy / sxx : 0;
    return v => my + b * (v - mx);
  };
  const X = fit(tx), Y = fit(py);
  const x = X(d.t), y = Y(d.p);
  const c = T[0][1][2], r = c.getBoundingClientRect();
  if (!(x > r.left + 2 && x < r.right - 2 && y > r.top + 2 && y < r.bottom - 2)) return {err: 'off chart', x, y};
  const top = document.elementFromPoint(x, y);
  const chart = c.parentElement;                      // the page's own chart box: its SVG / canvas layers are fine
  const ok = top === c || (top && (top.tagName === 'CANVAS' || (chart && chart.contains(top))));
  return {x, y, onCanvas: ok,
          cover: !ok && top ? (top.id || top.getAttribute('class') || top.tagName).toString().slice(0, 60) : null};
}"""

# the order panel: the amount box, the account (a demo account shows "(Demo)" next to its balance), the currency
PANEL_JS = r"""() => {
  const vis = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const inputs = [...document.querySelectorAll('input[inputmode="decimal"]')].filter(vis);
  const box = document.querySelector('#available-balance-input input') || inputs[0] || null;
  let panel = box;
  for (let i = 0; panel && i < 8; i++) panel = panel.parentElement;
  const text = panel ? panel.innerText.replace(/\s+/g, ' ').slice(0, 400) : '';
  const body = document.body.innerText;
  return {has_box: !!box, value: box ? box.value : null, text, demo: /\(Demo\)/i.test(text) || /\(Demo\)/i.test(body.slice(0, 3000)),
          closed: /Time Until Opening/i.test(body), n_inputs: inputs.length};
}"""


def _flag(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip()


# bc.game's server multiplier as a fraction of the grid label, by columns ahead: measured on 858 real orders
# (2026-11-01).  Used to book paper wins at what the server would have paid; the picker's switch is TAP_USE_RATIO.
PAYOUT_DEFAULT = {2: 0.64, 3: 0.68, 4: 0.70}
PAYOUT_OTHER = 0.68
PAYOUT_MIN_N = 20             # settled real orders of a kind before their own median replaces the default


class TapLive:
    def __init__(self, analyzer, log_dir: Path):
        self.a = analyzer
        self.mode = _flag("TAP_LIVE", "off").lower()
        if self.mode not in ("off", "demo", "real"):
            self.mode = "off"
        self.stake = float(_flag("TAP_STAKE", "1"))
        self.budget = float(_flag("TAP_BUDGET_USD", "30"))
        self.min_ev = float(_flag("TAP_MIN_EV", "1.15"))
        self.max_open = int(_flag("TAP_MAX_OPEN", "2"))
        self.min_gap_s = float(_flag("TAP_MIN_GAP_S", "6"))
        self.ratio_min_n = int(_flag("TAP_RATIO_MIN_N", "5"))
        self.ratio_boot = float(_flag("TAP_RATIO_BOOT", "0.85"))   # assumed server/label ratio before it is measured
        self.use_ratio = _flag("TAP_USE_RATIO", "1") != "0"        # 0: ignore the measured payout, price at the label
        self.lead_extra_ms = int(_flag("TAP_LEAD_EXTRA_MS", "600"))  # click margin over the game's 4.3 s lead
        self.loss_run_pause = int(_flag("TAP_LOSS_RUN_PAUSE", "10"))
        self.pause_min = float(_flag("TAP_PAUSE_MIN", "10"))
        self.state_path = log_dir / "tap_live_state.json"
        self.events_path = log_dir / "tap_live.jsonl"
        self.kill_path = log_dir / "TAP_KILL"
        self.st = {"real": self._book(), "demo": self._book(), "ratios": [], "stopped": None, "orders": {},
                   "settings": {}}
        self._load()
        saved = self.st.get("settings") or {}                   # what the page set last: it beats .env's defaults
        if saved.get("stake"):
            self.stake = float(saved["stake"])
        if saved.get("mode") in ("off", "demo", "real"):
            self.mode = saved["mode"]
        self.busy = False
        self.last_at = 0.0
        self.paused_until = 0.0
        self.panel = {}
        self.panel_at = 0.0
        self.rejects: deque = deque(maxlen=10)
        self.no_request = 0
        self.recent: deque = deque(maxlen=12)
        self.last_reason = None
        if self.mode != "off":
            log.warning("Tap LIVE mode %s: $%g per tap, budget $%g, min EV %.2f (logs/tap_live.jsonl)",
                        self.mode.upper(), self.stake, self.budget, self.min_ev)

    # ---------- persistence ----------

    @staticmethod
    def _book() -> dict:
        return {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0, "run": 0, "longest": 0}

    def _load(self) -> None:
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            for k in self.st:
                if k in saved:
                    self.st[k] = saved[k]
        except (OSError, ValueError):
            pass
        st = self.st.get("stopped")
        why = str((st or {}).get("why", ""))
        if st and not why.startswith("budget"):               # only a budget stop survives a restart
            log.warning("Tap live: dropping the old stop %r (no longer a stop condition)", why)   # an order; a
            self.st["stopped"] = None                              # result missed across a restart)
            for o in self.st["orders"].values():
                if o.get("result") is None and (o.get("stale") or o.get("unknown")):
                    o["result"] = "STALE"

    def _save(self) -> None:
        err = None
        for attempt in range(8):                  # Windows: a scanner or an editor may hold the file for a moment
            try:
                tmp = self.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.st), encoding="utf-8")
                os.replace(tmp, self.state_path)
                return
            except OSError as e:
                err = e
                time.sleep(0.05 * (attempt + 1))
        log.warning("Tap live state: %r (kept in memory; the next save retries)", err)

    def _event(self, event: str, **kw) -> None:
        rec = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "event": event, "mode": self.mode, **kw}
        try:
            with open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _stop(self, why: str) -> None:
        if not self.st["stopped"]:
            self.st["stopped"] = {"why": why, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "mode": self.mode}
            self._save()
            self._event("stop", why=why)
            log.warning("Tap LIVE stopped: %s", why)

    # ---------- the numbers ----------

    def ratio(self) -> float | None:
        """Median server multiplier / page label over our last 20 orders, None before TAP_RATIO_MIN_N."""
        if not self.use_ratio:
            return None
        r = [x for x in self.st["ratios"][-20:] if x and x > 0]
        return statistics.median(r) if len(r) >= self.ratio_min_n else None

    def payout(self, k=None) -> float:
        """What bc.game's server pays as a fraction of the grid label for a cell k columns ahead: the median over our
        settled real orders of that kind (PAYOUT_MIN_N or more), else over all of them, else the measured default.
        Books paper wins at the real payout; independent of use_ratio, which is the picker's switch."""
        settled = [o for o in self.st["orders"].values() if o.get("ratio") and o.get("result") in ("WIN", "LOSS")]
        if k is not None:
            mine = [o["ratio"] for o in settled if o.get("k") == k]
            if len(mine) >= PAYOUT_MIN_N:
                return round(statistics.median(mine), 3)
        if len(settled) >= PAYOUT_MIN_N:
            return round(statistics.median(o["ratio"] for o in settled), 3)
        return PAYOUT_DEFAULT.get(k, PAYOUT_OTHER)

    def payout_table(self) -> dict:
        return {str(k): self.payout(k) for k in (2, 3, 4)}

    def open_orders(self) -> list:
        return [o for o in self.st["orders"].values() if o.get("result") is None]

    def book(self) -> dict:
        return self.st["real" if self.mode == "real" else "demo"]

    # ---------- deciding and clicking ----------

    async def _read_panel(self) -> dict:
        if time.time() - self.panel_at > 5:
            try:
                self.panel = await self.a.page.evaluate(PANEL_JS)
            except Exception as e:
                self.panel = {"err": type(e).__name__}
            self.panel_at = time.time()
        return self.panel

    def _why_not(self, now_ms: float) -> str | None:
        """Everything that doesn't need the page; None when a tap may be considered."""
        if self.mode == "off":
            return "off"
        if self.kill_path.exists():
            self._stop("logs/TAP_KILL exists")
        if self.st["stopped"]:
            return f"stopped: {self.st['stopped']['why']}"
        if time.time() < self.paused_until:
            return f"paused {self.paused_until - time.time():.0f}s after {self.loss_run_pause} losses in a row"
        if not self.a.market_open():
            return "market closed (or its status not read yet)"
        if not self.a.connected or not (self.a.bc_read or {}).get("n"):
            return "grid not read"
        if self.busy or time.time() - self.last_at < self.min_gap_s:
            return "spacing"
        if len(self.open_orders()) >= self.max_open:
            return f"{self.max_open} taps open"
        for o in self.st["orders"].values():        # an unanswered order blocks new taps for 70 s, then is dropped
            if o.get("unknown") and o.get("result") is None:
                if now_ms - o["t0"] > 70000:
                    o["result"] = "STALE"
                    self._save()
                else:
                    return "an order without an answer"
        if self.mode == "real":                   # before TAP_RATIO_MIN_N orders the EV assumes TAP_RATIO_BOOT
            b = self.st["real"]
            at_risk = sum(o.get("usd", self.stake) for o in self.open_orders())
            if b["pnl"] - at_risk - self.stake < -self.budget:
                self._stop(f"budget: real P&L {b['pnl']:+.2f} with {at_risk:.2f} open, limit -{self.budget:g}")
                return "budget"
        return None

    def _choose(self, now_ms: float) -> dict | None:
        """The live tap is the analyzer's pick for this snapshot (its live picker, LIVE_MODEL in tap.py: one cell per
        column, already past that picker's bar), when the live gates allow: labelled by bc.game, far enough ahead
        to click, a measured cell kind, and the column not taken yet.  There is no second rule here."""
        g = self.a.grid or {}
        pick = g.get("pick")
        if not pick:
            return None
        taken = {o["t0"] for o in self.st["orders"].values()}          # one live tap per column
        for c in g.get("cells", []):
            if (c["k"], c["r"]) != tuple(pick):
                continue
            if not c.get("tappable") or c.get("odds_src") != "bc" or c["t0"] - now_ms < MIN_AHEAD_MS + self.lead_extra_ms:
                return None
            if c.get("p_emp") is None or c["t0"] in taken:
                return None
            pay = self.payout(c["k"])                         # the EV at what bc.game really pays, for the record
            return dict(c, ev_exp=round(c["p_emp"] * c["odds"] * pay, 4), ratio_used=round(pay, 4))
        return None

    async def consider(self) -> None:
        """Called after each grid snapshot: maybe place one tap."""
        if not self.a.ticks:
            return
        now_ms = self.a.ticks[-1][0]
        why = self._why_not(now_ms)
        if why:
            self.last_reason = why
            return
        cell = self._choose(now_ms)
        if cell is None:
            self.last_reason = "no live pick this column"
            return
        self.busy = True
        try:
            await self._place(cell)
        except Exception as e:
            self._event("error", err=repr(e)[:300])
            log.warning("Tap live: %r", e)
        finally:
            self.busy = False

    async def _ensure_amount(self, page) -> bool:
        p = await self._read_panel()
        if p.get("closed"):
            self.last_reason = "page shows the market closed"
            return False
        if not p.get("has_box"):
            self.last_reason = "no amount box"
            return False
        want_demo = self.mode == "demo"
        if bool(p.get("demo")) != want_demo:
            self.last_reason = f"page account is {'DEMO' if p.get('demo') else 'REAL'}, mode is {self.mode}"
            return False
        want = f"{self.stake:g}"
        if (p.get("value") or "").replace(",", "") != want:
            box = page.locator('#available-balance-input input, input[inputmode="decimal"]').first
            await box.fill(want)
            await box.press("Tab")
            self.panel_at = 0.0
            p = await self._read_panel()
            if (p.get("value") or "").replace(",", "") != want:
                self.last_reason = f"amount box shows {p.get('value')!r}, wanted {want}"
                return False
        return True

    async def _place(self, cell: dict) -> None:
        page = self.a.page
        if not await self._ensure_amount(page):
            return
        now_ms = self.a.ticks[-1][0]
        if cell["t0"] - now_ms < MIN_AHEAD_MS:
            self.last_reason = "too late for the chosen cell"
            return
        pt = await page.evaluate(CLICK_POINT_JS, {"now": now_ms, "t": (cell["t0"] + cell["t1"]) / 2,
                                                  "p": (cell["lo"] + cell["hi"]) / 2})
        if pt.get("err") or not pt.get("onCanvas"):
            self.last_reason = f"click point: {pt.get('err') or 'covered by ' + str(pt.get('cover'))}"
            return
        sent = {"req": None}

        def on_req(req):
            if ORDER_URL in req.url and req.method == "POST":
                try:
                    sent["req"] = req.post_data_json
                except Exception:
                    sent["req"] = {"raw": (req.post_data or "")[:300]}

        page.on("request", on_req)
        self.last_at = time.time()
        intent = {"t0": cell["t0"], "lo": cell["lo"], "hi": cell["hi"], "k": cell["k"], "r": cell["r"],
                  "label": cell["odds"], "po_cal": cell["po_cal"], "ev_exp": cell["ev_exp"], "ratio_used": cell["ratio_used"],
                  "stake": self.stake, "price": self.a.ticks[-1][1], "tick_ms": now_ms, "x": round(pt["x"], 1), "y": round(pt["y"], 1)}
        self._event("intent", **intent)
        resp = None
        try:
            async with page.expect_response(lambda r: ORDER_URL in r.url, timeout=15000) as info:
                await page.mouse.click(pt["x"], pt["y"])      # the server's answer takes 2-7 s (measured)
            resp = await info.value
        except Exception:
            resp = None
        finally:
            page.remove_listener("request", on_req)
        if resp is None:
            if sent["req"] is None:
                self.no_request += 1
                self._event("no_request", **intent)
                self.last_reason = "the click sent no order"
                if self.no_request >= 3:
                    log.warning("Tap live: 3 clicks in a row sent no order (the page's layout or rules changed?); "
                                "pausing 10 s")
                    self.last_at = time.time() + 10
                    self.no_request = 0
                return
            oid = f"unknown-{int(time.time() * 1000)}"
            self.st["orders"][oid] = dict(intent, id=oid, unknown=True, sent=sent["req"], result=None, mode=self.mode)
            self._save()
            self._event("no_answer", id=oid, **intent)
            log.warning("Tap live: an order was sent and got no answer in 15 s (held as unknown; booked if the "
                        "answer comes later, dropped after 70 s)")
            self.last_reason = "an order without an answer"

            def late(r):                                   # the answer arriving after the wait: book the order
                if ORDER_URL in r.url:
                    page.remove_listener("response", late)
                    asyncio.ensure_future(self._book_late(oid, intent, r))
            page.on("response", late)
            return
        self.no_request = 0
        try:
            body = await resp.json()
        except Exception:
            body = {"raw": (await resp.text())[:300]}
        code = body.get("code")
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        if code != 0 or not data.get("id"):
            self._event("rejected", code=code, msg=body.get("msg"), sent=sent["req"], **intent)
            self.last_reason = f"rejected: {code} {body.get('msg')}"
            self.rejects.append(time.time())      # a rejection is never a stop: 2024 = the band size changed under
            self.last_at = time.time() + 3        # the order, 2022 = the column's lead time passed on the way;
            if code == 2022:                      # the next click gets a larger margin
                self.lead_extra_ms = min(2000, self.lead_extra_ms + 200)
            return
        odds = float(data.get("odds") or 0)
        booked_t0, booked_lo = data.get("lowerTimeLimit"), data.get("lowerPriceLimit")
        order = dict(intent, id=str(data["id"]), odds=odds, ratio=round(odds / cell["odds"], 4) if odds and cell["odds"] else None,
                     booked_t0=booked_t0, booked_lo=booked_lo, amount=data.get("amount"), currency=data.get("currency"),
                     usd=self.stake, result=None, mode=self.mode, sent=sent["req"])
        self.st["orders"][order["id"]] = order
        if order["ratio"]:
            self.st["ratios"].append(order["ratio"])
            self.st["ratios"] = self.st["ratios"][-200:]
        b = self.book()
        b["bets"] += 1
        b["staked"] = round(b["staked"] + self.stake, 4)
        self._save()
        self._event("order", **{k: v for k, v in order.items() if k != "sent"})
        log.info("Tap %s order %s: $%g on %.0f-%.0f at %s (+%d col %+d band), server x%.2f vs label x%.2f",
                 self.mode.upper(), order["id"], self.stake, cell["lo"], cell["hi"],
                 time.strftime("%H:%M:%S", time.gmtime(cell["t0"] / 1000)), cell["k"], cell["r"], odds, cell["odds"])
        if booked_t0 is not None and (abs(float(booked_t0) - cell["t0"]) > 1 or abs(float(booked_lo) - cell["lo"]) > 0.01):
            step = cell["hi"] - cell["lo"]                      # the click landed at the edge of the lead time and the
            order["t0"], order["lo"] = float(booked_t0), float(booked_lo)   # server booked the next column: keep the
            order["t1"], order["hi"] = float(booked_t0) + (cell["t1"] - cell["t0"]), float(booked_lo) + step   # booked cell
            self._save()
            self._event("rebooked", id=order["id"], t0=order["t0"], lo=order["lo"], wanted_t0=cell["t0"], wanted_lo=cell["lo"])
            log.warning("Tap live: order %s was booked at %s/%s, not %s/%s: kept as booked", order["id"], booked_t0,
                        booked_lo, cell["t0"], cell["lo"])

    # ---------- results ----------

    def on_end(self, r: dict) -> None:
        """A /tapOrder/end frame: settle it when it is one of ours."""
        oid = str(r.get("id"))
        o = self.st["orders"].get(oid)
        if o is None or o.get("result") is not None:
            return
        try:
            profit = float(r.get("profit") or 0)
        except (TypeError, ValueError):
            profit = 0.0
        won = profit > 0
        o.update(result="WIN" if won else "LOSS", profit=profit, end_price=r.get("endPrice"),
                 settled_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        b = self.st[o.get("mode", self.mode) if o.get("mode") in ("real", "demo") else "demo"]
        b["wins"] += won
        b["pnl"] = round(b["pnl"] + profit, 4)
        b["run"] = 0 if won else b["run"] + 1
        b["longest"] = max(b["longest"], b["run"])
        if self.loss_run_pause > 0 and b["run"] >= self.loss_run_pause:    # 0 = never pause (the user's choice)
            self.paused_until = time.time() + self.pause_min * 60
            self._event("pause", run=b["run"], minutes=self.pause_min)
        self.recent.append({k: o.get(k) for k in ("id", "k", "r", "label", "odds", "result", "profit", "mode")})
        # our paper hit rule against the real result: does the analyzer settle cells the way bc.game does?
        hit, _ = self.a.touched_cell(o["lo"], o["hi"], o["t0"], o["t0"] + (self.a.cfg["timeIncrement"] * 1000))
        o["our_hit"] = hit
        self._save()
        self._event("result", id=oid, result=o["result"], profit=profit, odds=o.get("odds"), label=o.get("label"),
                    our_hit=hit, pnl=b["pnl"], run=b["run"])
        log.info("Tap %s order %s: %s %+.2f (x%.2f) | %s P&L %+.2f, %d/%d won, losses in a row %d",
                 o.get("mode", "").upper(), oid, o["result"], profit, o.get("odds") or 0, o.get("mode"), b["pnl"],
                 b["wins"], b["bets"], b["run"])

    async def _book_late(self, oid: str, intent: dict, resp) -> None:
        """An order's answer that came after the 15 s wait: book it in place of the unknown order."""
        try:
            body = await resp.json()
        except Exception:
            return
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        if body.get("code") != 0 or not data.get("id"):
            self.st["orders"].pop(oid, None)                       # rejected after all: nothing placed
            self._event("rejected", code=body.get("code"), msg=body.get("msg"), late=True, **intent)
            self._save()
            return
        odds = float(data.get("odds") or 0)
        order = dict(intent, id=str(data["id"]), odds=odds,
                     ratio=round(odds / intent["label"], 4) if odds and intent.get("label") else None,
                     booked_t0=data.get("lowerTimeLimit"), booked_lo=data.get("lowerPriceLimit"), amount=data.get("amount"),
                     currency=data.get("currency"), usd=self.stake, result=None, mode=self.mode, late=True)
        self.st["orders"].pop(oid, None)
        self.st["orders"][order["id"]] = order
        if order["ratio"]:
            self.st["ratios"].append(order["ratio"])
            self.st["ratios"] = self.st["ratios"][-200:]
        b = self.book()
        b["bets"] += 1
        b["staked"] = round(b["staked"] + self.stake, 4)
        self._save()
        self._event("order", **{k: v for k, v in order.items() if k != "sent"})
        log.warning("Tap live: late answer booked: order %s x%.2f", order["id"], odds)

    def check_stale(self) -> None:
        """An order not settled 60 s after its window ended (its end message was missed, e.g. across a restart):
        it is dropped from the open list as STALE (the server settled it; its result is on the page's History) and
        live taps go on.  The bot is not stopped: it placed the order, it just did not see the result."""
        if not self.a.ticks:
            return
        now_ms = self.a.ticks[-1][0]
        for o in self.open_orders():
            if not o.get("unknown") and now_ms - o["t0"] > 62000 and not o.get("stale"):
                o["stale"] = True
                o["result"] = "STALE"
                self._event("stale", id=o["id"], t0=o["t0"], usd=o.get("usd"))
                log.warning("Tap live: order %s not settled 60 s after its window: dropped as STALE (see History)", o["id"])
                self._save()

    def set_mode(self, mode: str, who: str = "operator") -> str:
        """Switch live taps off / demo / real at runtime (the dashboard, with the token).  Logged.  A budget stop
        stays; the page's account must match the mode (demo <-> the DEMO account) or no tap is placed."""
        mode = (mode or "").strip().lower()
        if mode not in ("off", "demo", "real"):
            raise ValueError("Tap live mode must be off, demo or real")
        was, self.mode = self.mode, mode
        self.st.setdefault("settings", {})["mode"] = mode
        self._save()
        self._event("mode", was=was, who=who)
        msg = (f"Tap live {mode.upper()}: ${self.stake:g} per tap, budget ${self.budget:g}, min EV {self.min_ev:.2f}"
               + (f", server/label ratio {self.ratio():.2f}" if self.ratio() else
                  f", ratio assumed {self.ratio_boot:.2f} until {self.ratio_min_n} orders"))
        log.warning(msg)
        return msg

    def set_stake(self, usd: float, who: str = "operator") -> str:
        """The stake per tap, $0.1 to $5, at runtime.  Logged."""
        usd = float(usd)
        if not 0.1 <= usd <= 5:
            raise ValueError("the Tap stake must be between $0.1 and $5")
        was, self.stake = self.stake, round(usd, 2)
        self.st.setdefault("settings", {})["stake"] = self.stake
        self._save()
        self._event("stake", was=was, who=who)
        log.warning("Tap live stake $%g per tap (was $%g)", self.stake, was)
        return f"Tap stake ${self.stake:g} per tap"

    def clear(self) -> None:
        """Clear a technical stop (not a budget stop): unknown / stale orders are dropped from the open list."""
        for o in self.st["orders"].values():
            if o.get("result") is None and (o.get("unknown") or o.get("stale")):
                o["result"] = "CLEARED"
        if self.st["stopped"] and not self.st["stopped"]["why"].startswith("budget"):
            self.st["stopped"] = None
        self._save()
        self._event("clear")

    def summary(self) -> dict:
        return {"mode": self.mode, "stake": self.stake, "budget": self.budget, "min_ev": self.min_ev,
                "ratio_boot": self.ratio_boot, "ratio_min_n": self.ratio_min_n, "max_open": self.max_open,
                "stopped": self.st["stopped"], "paused_s": max(0, round(self.paused_until - time.time())),
                "payout": self.payout_table(), "ratio": self.ratio(), "ratios_n": len(self.st["ratios"]), "real": self.st["real"], "demo": self.st["demo"],
                "open": len(self.open_orders()), "last_reason": self.last_reason, "panel": self.panel,
                "recent": list(self.recent)[::-1]}
