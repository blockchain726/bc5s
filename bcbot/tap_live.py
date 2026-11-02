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
  return {x, y, onCanvas: top === c || (top && top.tagName === 'CANVAS'),
          cover: top && top.tagName !== 'CANVAS' ? (top.className || top.tagName).toString().slice(0, 60) : null};
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
        self.loss_run_pause = int(_flag("TAP_LOSS_RUN_PAUSE", "10"))
        self.pause_min = float(_flag("TAP_PAUSE_MIN", "10"))
        self.state_path = log_dir / "tap_live_state.json"
        self.events_path = log_dir / "tap_live.jsonl"
        self.kill_path = log_dir / "TAP_KILL"
        self.st = {"real": self._book(), "demo": self._book(), "ratios": [], "stopped": None, "orders": {}}
        self._load()
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

    def _save(self) -> None:
        try:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.st), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as e:
            log.warning("Tap live state: %r", e)

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
        r = [x for x in self.st["ratios"][-20:] if x and x > 0]
        return statistics.median(r) if len(r) >= self.ratio_min_n else None

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
        if any(o.get("unknown") for o in self.st["orders"].values()):
            return "an order without an answer"
        if self.mode == "real":                   # before TAP_RATIO_MIN_N orders the EV assumes TAP_RATIO_BOOT
            b = self.st["real"]
            at_risk = sum(o.get("usd", self.stake) for o in self.open_orders())
            if b["pnl"] - at_risk - self.stake < -self.budget:
                self._stop(f"budget: real P&L {b['pnl']:+.2f} with {at_risk:.2f} open, limit -{self.budget:g}")
                return "budget"
        return None

    def _choose(self, now_ms: float) -> dict | None:
        g = self.a.grid or {}
        ratio = self.ratio() or self.ratio_boot
        taken = {(o["t0"], o["lo"]) for o in self.st["orders"].values()}
        best = None
        for c in g.get("cells", []):
            if not c.get("tappable") or c.get("odds_src") != "bc" or c["t0"] - now_ms < MIN_AHEAD_MS + 300:
                continue
            if (c["t0"], c["lo"]) in taken:
                continue
            ev = c["po_cal"] * c["odds"] * ratio
            if ev >= self.min_ev and (best is None or ev > best["ev_exp"]):
                best = dict(c, ev_exp=round(ev, 4), ratio_used=round(ratio, 4))
        return best

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
            self.last_reason = f"no cell with EV >= {self.min_ev}"
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
            async with page.expect_response(lambda r: ORDER_URL in r.url, timeout=5000) as info:
                await page.mouse.click(pt["x"], pt["y"])
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
                    self._stop("3 clicks in a row sent no order (the page's layout or rules changed?)")
                return
            oid = f"unknown-{int(time.time() * 1000)}"
            self.st["orders"][oid] = dict(intent, id=oid, unknown=True, sent=sent["req"], result=None, mode=self.mode)
            self._save()
            self._stop("an order was sent and got no answer in 5 s: check the page's Positions, then clear it")
            return
        self.no_request = 0
        try:
            body = await resp.json()
        except Exception:
            body = {"raw": (await resp.text())[:300]}
        code = body.get("code")
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        if code != 0 or not data.get("id"):
            self.rejects.append(time.time())
            self._event("rejected", code=code, msg=body.get("msg"), sent=sent["req"], **intent)
            self.last_reason = f"rejected: {code} {body.get('msg')}"
            if sum(1 for t in self.rejects if time.time() - t < 600) >= 2:
                self._stop(f"2 rejections in 10 minutes (last: {code} {body.get('msg')})")
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
            self._stop(f"order {order['id']} was booked at {booked_t0}/{booked_lo}, not {cell['t0']}/{cell['lo']}")

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
        if b["run"] >= self.loss_run_pause:
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

    def check_stale(self) -> None:
        """An order not settled 60 s after its window ended: technical stop (it must be checked by hand)."""
        if not self.a.ticks:
            return
        now_ms = self.a.ticks[-1][0]
        for o in self.open_orders():
            if not o.get("unknown") and now_ms - o["t0"] > 62000 and not o.get("stale"):
                o["stale"] = True
                self._save()
                self._stop(f"order {o['id']} not settled 60 s after its window")

    def set_mode(self, mode: str, who: str = "operator") -> str:
        """Switch live taps off / demo / real at runtime (the dashboard, with the token).  Logged.  A budget stop
        stays; the page's account must match the mode (demo <-> the DEMO account) or no tap is placed."""
        mode = (mode or "").strip().lower()
        if mode not in ("off", "demo", "real"):
            raise ValueError("Tap live mode must be off, demo or real")
        was, self.mode = self.mode, mode
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
                "ratio": self.ratio(), "ratios_n": len(self.st["ratios"]), "real": self.st["real"], "demo": self.st["demo"],
                "open": len(self.open_orders()), "last_reason": self.last_reason, "panel": self.panel,
                "recent": list(self.recent)[::-1]}
