"""Automatic UP / DOWN calls for bc.game's 5 s rounds from the Hawkes models (predictor.py).

Round timing and results come from the page's own Detrade websocket (the "/contest/BTC/USD/5/ticker" frames:
tradeCutoffTime, priceStartTime, priceEndTime, winSide, ...).  At the start of each betting window the models are
refitted; AUTO_LEAD_MS before the cutoff the call is made and recorded.  Every round is scored against bc.game's
result (paper P&L at the payout the page shows), and logged to logs/auto_rounds.jsonl.

Real orders are OFF at every start.  They need LIVE_ALLOWED and TOS_RISK_ACCEPTED in .env and a typed arming that
names the current run of real losses ("live arm run=0 day=2026-10-30").  Each call is then placed as a flat
AUTO_STAKE_USD order, never into a later round, only after a checklist (_may_place) that is run right before the
click on bc.game's own clock.  One persistent counter of real losses in a row (rebuilt from the log, reset only by a
real WIN) stops and disarms the bot at AUTO_MAX_REAL_RUN; it never resumes by itself, and beyond the limit a re-arm
allows one real order at a time.  Every arm / stop / order goes to logs/live_events.jsonl.  Time, on every path, is
bc.game's clock (clock.py), measured from its own price ticks.
"""
import asyncio
import bisect
import calendar
import html
import json
import logging
import math
import time
import zlib
from collections import deque

import numpy as np

from .clock import BcClock
from .config import Config
from .flow import BinanceBook, BinanceFlow, FlowLogit
from .lob import LobFeed
from .prob import ProbModel, REFIT_S as PROB_REFIT_S
from .ourprice import MIN_WEIGHT, MIX, MIX_VERSION, STALE_SNAPSHOT, OurPrice, make_feeds
from .predictor import HawkesPredictor
from .trader import OrderUnknown, TooLate, TradeError, Trader

log = logging.getLogger(__name__)

TICKER = "/contest/BTC/USD/5/ticker"
PRICE = "/kline/BTC-USD/ticker"
POOLS = "/contest/btc_5/amountPool"     # the round's Up and Down pools ($), as the MHH bot reads them
PAY_K = 0.9415                 # the page's payout = 1 + 0.9415 x other side's pool / own side's pool (579 rounds)
POOL_MIN_PAY = 2.0             # the "pool" model takes the thin side when the page pays more than this
TYPICAL_POOL = 204.0           # median total pool of a round (both sides), for rounds whose pools weren't seen
BETTING = 1001
DEFAULT_PAYOUT = 1.94          # average payout the page showed (x1.942), used when it can't be read
MIN_REFIT_S = 8.0              # start a refit only with this much betting time left, so it can't slow the call
LOG_VERSION = 3                # round records: 3 = bc clock, snapshots A/B with dq, shadow, real-order bookkeeping
PHASE = {1001: "betting", 1003: "5 s window", 1002: "result", 1008: "settling", 1004: "settling",
         1005: "settling", 1006: "settling"}

# The decision panel is drawn into the bot's own BC window (a shadow-DOM box over bc.game's left sidebar, below the
# header, so it covers only the site menu and never the chart or the Up/Down buttons; it scrolls with the mouse
# wheel and keeps its scroll position across refreshes), and is served at / by the local API.
OVERLAY_JS = r"""
(html) => {
  let host = document.getElementById('hawkes-panel');
  if (!host) {
    host = document.createElement('div');
    host.id = 'hawkes-panel';
    host.attachShadow({mode: 'open'});
    document.body.appendChild(host);
  }
  host.style.cssText = 'position:fixed;left:6px;top:66px;z-index:2147483647;';
  const old = host.shadowRoot.querySelector('.p');
  const y = old ? old.scrollTop : 0;              // keep where the user scrolled the panel to
  host.shadowRoot.innerHTML = html;
  const p = host.shadowRoot.querySelector('.p');
  if (p) p.scrollTop = y;
}
"""
CURVE_JS = r"""
(d) => {                                             // our price curve on bc.game's own chart (the visx SVG):
  let st = window.__hawkesCurve;                     // a fixed canvas over the SVG, redrawn 10x a second
  if (!st || !st.host.isConnected) {
    if (st && st.timer) clearInterval(st.timer);
    const host = document.createElement('div');
    host.id = 'hawkes-curve';
    host.attachShadow({mode: 'open'});
    host.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;z-index:2147483646;pointer-events:none;';
    const cv = document.createElement('canvas');
    cv.style.cssText = 'position:fixed;left:0;top:0;pointer-events:none;display:none';
    host.shadowRoot.appendChild(cv);
    document.body.appendChild(host);
    st = window.__hawkesCurve = {host, cv, d: null, at: 0, draw: null};
    st.timer = setInterval(() => { try { st.draw && st.draw(); } catch (e) {} }, 100);
  }
  st.d = d; st.at = performance.now();
  st.draw = () => {
    const cv = st.cv, d = st.d;
    const svg = [...document.querySelectorAll('svg')].find(s => {
      const r = s.getBoundingClientRect(); return r.width > 500 && r.height > 150 && s.querySelector('path.visx-linepath'); });
    if (!d || !d.on || !svg || performance.now() - st.at > 5000) { cv.style.display = 'none'; return; }
    const r = svg.getBoundingClientRect();
    // price -> y: a straight line through the chart's own axis labels
    const ticks = [...svg.querySelectorAll('.visx-axis-tick')].map(g => {
      const l = g.querySelector('line'), t = g.querySelector('text');
      return l && t ? [parseFloat(t.textContent.replace(/[^\d.\-]/g, '')), +l.getAttribute('y1')] : null;
    }).filter(x => x && isFinite(x[0]) && isFinite(x[1]));
    if (ticks.length < 2) { cv.style.display = 'none'; return; }
    let n = ticks.length, sp = 0, sy = 0, spp = 0, spy = 0;
    for (const [pp, yy] of ticks) { sp += pp; sy += yy; spp += pp * pp; spy += pp * yy; }
    const b = (n * spy - sp * sy) / (n * spp - sp * sp), a = (sy - b * sp) / n;
    if (!isFinite(a) || !isFinite(b) || !b) { cv.style.display = 'none'; return; }
    const Y = pp => a + b * pp;
    // time -> x: the chart shows 60 s before its last tick and 30 s after it; the last tick is the line's end
    const nums = (svg.querySelector('path.visx-linepath').getAttribute('d').match(/-?\d+(\.\d+)?/g) || []).map(Number);
    const xNow = nums.length >= 2 ? nums[nums.length - 2] : r.width * 2 / 3;
    const pNow = nums.length >= 2 ? (nums[nums.length - 1] - a) / b : null;
    let tLast = d.t_last;                            // which of the bot's ticks is the chart's last one: same price
    if (pNow != null) for (let i = d.ticks.length - 1; i >= 0; i--) {
      if (Math.abs(d.ticks[i][1] - pNow) < 0.002) { tLast = d.ticks[i][0]; break; } }
    const pxs = r.width / 90, X = t => xNow + (t - tLast) * pxs;
    const dpr = window.devicePixelRatio || 1;
    cv.style.display = 'block'; cv.style.left = r.left + 'px'; cv.style.top = r.top + 'px';
    cv.style.width = r.width + 'px'; cv.style.height = r.height + 'px';
    if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    const pts = (d.our || []).filter(q => q[0] >= tLast - 61 && q[0] <= tLast + 0.6).map(q => [X(q[0]), Y(q[1])]);
    if (pts.length < 2) return;
    for (const [w, c] of [[4, 'rgba(0,0,0,0.55)'], [1.7, '#22d3ee']]) {
      g.strokeStyle = c; g.lineWidth = w; g.lineJoin = 'round'; g.beginPath();
      pts.forEach((q, i) => i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])); g.stroke(); }
    const last = pts[pts.length - 1], pOur = d.our[d.our.length - 1][1];
    g.fillStyle = '#22d3ee'; g.beginPath(); g.arc(last[0], last[1], 3, 0, 7); g.fill();
    g.font = 'bold 12px Consolas, monospace'; g.textAlign = 'right';
    const txt = 'our ' + pOur.toFixed(2) + (pNow != null ? '  (bc ' + (pNow - pOur >= 0 ? '+' : '') + (pNow - pOur).toFixed(2) + ')' : '');
    const tw = g.measureText(txt).width, ly = Math.min(r.height - 8, Math.max(14, last[1] + (last[1] < r.height / 2 ? 18 : -10)));
    g.fillStyle = 'rgba(11,14,19,0.8)'; g.fillRect(last[0] - 10 - tw - 4, ly - 11, tw + 8, 15);
    g.fillStyle = '#22d3ee'; g.fillText(txt, last[0] - 10, ly);
    g.textAlign = 'left'; g.font = '11px Consolas, monospace';
    g.fillStyle = 'rgba(11,14,19,0.8)'; g.fillRect(r.width * 0.3, r.height - 22, 318, 16);   // bottom, clear of the icons
    g.fillStyle = '#22d3ee'; g.fillText('— our price: 6-exchange mix, read 0.15 s earlier, at bc.game\'s level', r.width * 0.3 + 4, r.height - 10);
  };
  st.draw();
}
"""

PANEL_CSS = """<style>
.p{width:230px;font:11px/1.4 Consolas,monospace;color:#d6dde6;max-height:calc(100vh - 76px);overflow-y:auto;
   display:flex;flex-direction:column;gap:6px;scrollbar-width:thin;scrollbar-color:#3a4350 transparent}
.p>*{flex:none}
.hd{display:flex;justify-content:space-between;align-items:center;color:#8793a2;font-size:10px;letter-spacing:.5px;
    padding:2px 4px;background:rgba(11,14,19,.92);border-radius:6px}
.mode{padding:1px 6px;border-radius:4px;font-weight:700;color:#fff}.paper{background:#2c3440}.live{background:#b4232f}
.off{background:#444;color:#aaa}.paused{background:#b7791f}
.c{background:rgba(17,20,26,.97);border:1.5px solid #2c3440;border-radius:9px;padding:7px 9px;box-shadow:0 4px 16px rgba(0,0,0,.45)}
.c.bet{border-color:#facc15;background:rgba(34,29,16,.97)}.c.win{border-color:#22d27a;background:rgba(14,32,26,.97)}
.ct{display:flex;justify-content:space-between;gap:6px;font-weight:700;font-size:10px;letter-spacing:.6px;color:#c3ccd8}
.ph{color:#facc15}.c.win .ph{color:#22d27a}
.big{font:700 28px/1.15 Consolas,monospace;margin:3px 0 4px;white-space:nowrap}
.big small{font:11px Consolas,monospace;color:#c3ccd8;margin-left:6px}.c.bet .big{color:#facc15}.c.win .big{color:#22d27a}
.pbar{height:7px;background:#2c3440;border-radius:4px;overflow:hidden;margin:2px 0 6px}
.pbar i{display:block;height:100%;background:#5b6573}.c.bet .pbar i{background:#facc15}.c.win .pbar i{background:#22d27a}
.row,.two{display:flex;justify-content:space-between;gap:6px}.row span:first-child{white-space:nowrap;color:#8793a2}
.row span:last-child{text-align:right}
.sq{display:flex;gap:2px;flex-wrap:wrap;margin-top:6px}.sq i{width:12px;height:12px;border-radius:2px;font:700 8px/12px Arial;
    text-align:center;color:#0b0e13;font-style:normal}
.sq .U,.sq .W{background:#22d27a}.sq .D,.sq .L{background:#ff5468}.sq .T{background:#5b6573}
.call{font:700 26px/1.2 Segoe UI,Arial;text-align:center;margin:2px 0}
.up{color:#22d27a}.down{color:#ff5468}.mute{color:#8793a2}.sub{text-align:center;color:#8793a2;font-size:10px}
.res{display:flex;align-items:center;gap:5px;font:700 13px/1.2 Segoe UI,Arial;padding:5px 7px;border-radius:6px;margin-top:4px;
     white-space:nowrap}
.res .m{margin-left:auto}.res .sd{color:#e6edf5}
.res.w{background:rgba(34,210,122,.18);color:#22d27a}.res.l{background:rgba(255,84,104,.18);color:#ff5468}
.res.t{background:#1f2630;color:#8793a2}
.tag{font:700 9px Consolas,monospace;padding:1px 4px;border-radius:3px;background:#b4232f;color:#fff;vertical-align:middle}
.ptag{background:#3a4350}
.pay{display:flex;gap:6px;margin:5px 0}.pay div{flex:1;border:1px solid #2c3440;border-radius:6px;padding:3px 6px}
.pay b{display:block;font:700 18px Consolas,monospace}
.pay .dim{opacity:.45}.pay .ours.upb{border-color:#22d27a;background:rgba(34,210,122,.16)}
.pay .ours.downb{border-color:#ff5468;background:rgba(255,84,104,.16)}
.c.betup{border-color:#22d27a;background:rgba(14,40,28,.97)}.c.betdn{border-color:#ff5468;background:rgba(48,16,22,.97)}
.bl{font-size:11px;font-weight:700}.c.betup .bl{color:#22d27a}.c.betdn .bl{color:#ff5468}.note{color:#c3ccd8;font-size:10px}
.c.obup{border-color:#22d27a;background:rgba(14,40,28,.97)}.c.obdn{border-color:#ff5468;background:rgba(48,16,22,.97)}
.obw{font-size:11px;margin-bottom:4px}
.obbar{position:relative;height:8px;background:#2c3440;border-radius:4px;overflow:hidden;margin:3px 0 1px}
.obbar i{position:absolute;top:0;bottom:0}
.obbar:after{content:'';position:absolute;left:50%;top:0;bottom:0;width:1px;background:#8793a2}
.sm{font-size:9px}.spark{display:block;width:100%;height:40px;margin:4px 0 1px}
.pnl{font:700 19px Consolas,monospace}.pos{color:#22d27a}.neg{color:#ff5468}
.sep{border-top:1px solid #2c3440;margin:6px 0}
</style>"""


def _price_at(q, t: float) -> float | None:
    """Last price at or before t in a deque of (time, price)."""
    items = list(q)
    i = bisect.bisect_right([x[0] for x in items], t) - 1
    return items[i][1] if i >= 0 else None


def _diluted(pay: float, stake: float, pools, side: str) -> float:
    """The payout `pay` (as the page shows it, without our bet) once our `stake` is in our side's pool: pari-mutuel,
    so a bet that is big next to the pool mostly pays itself back (a $32 bet into ~$100 a side: x1.94 -> ~x1.7)."""
    if pools and pools[0] > 0 and pools[1] > 0:
        total, mine = pools[0] + pools[1], pools[0] if side == "UP" else pools[1]
    else:
        total = TYPICAL_POOL
        mine = min(total, max(1.0, PAY_K * total / (pay - 1 + PAY_K)))
    return pay * (total + stake) / (mine + stake) * mine / total


def _pool_call(payouts) -> dict | None:
    """The "pool" model: the side with the smaller pool pays more; when the page pays more than POOL_MIN_PAY on it,
    a 50 % coin paid above even money is a positive-expectation bet whatever the models say (the crowd's imbalance
    carried no information in 1,996 rounds).  None without two readable payouts."""
    if not payouts or len(payouts) != 2 or not payouts[0] or not payouts[1] or payouts[0] <= 1 or payouts[1] <= 1:
        return None
    if payouts[0] == payouts[1]:
        return {"side": None, "pay": payouts[0], "call": None}
    side, pay = ("UP", payouts[0]) if payouts[0] > payouts[1] else ("DOWN", payouts[1])
    return {"side": side, "pay": pay, "call": side if pay > POOL_MIN_PAY else None}


def _mom_call(feat: dict | None) -> str | None:
    """The "mom" model (pre-registered 2026-10-31): the sign of our exchange-mix price's move over the last 5 s,
    read at the decision snapshot (~2.9 s before the cutoff).  Why: a 5 s BTC move follows the last few seconds'
    direction 56-59 % of the time on Binance's own price (48 h of 1 s bars), but bc.game's index agrees with
    Binance's window move only ~78 %, which leaves ~51-53 % on bc.game's result; on 1,853 logged rounds this signal
    was 52.6 % [50.3, 54.8] (best of ~15 signals looked at, so it must prove itself forward).  None without it."""
    v = (feat or {}).get("our_mom5")
    if not isinstance(v, (int, float)) or v == 0 or (feat or {}).get("error"):
        return None
    return "UP" if v > 0 else "DOWN"


BOOK_MIN = 0.7     # the "book" model calls only when |Binance 20-level imbalance| is at least this (fixed 2026-10-31)
# Models chosen by looking at the logged rounds: the go-live gate (and so the real compensate ladder) counts only
# their rounds AFTER these times (UTC epoch), i.e. their forward record; older models keep their whole record.
MOMVOL_MIN = 2.8  # "momvol": momentum only when the last 60 s were volatile (vol60 >= this; fixed 2026-11-01)
REVERT_QUIET_MAX = 1.64   # "revert": a quiet market is vol60 below this (the lowest third, fixed 2026-11-01)
REVERT_MIN_MOVE = 0.5     # ... and bc.game's own price moved more than this ($) over the last 2 s
# "overshoot" / "overshoot_book" (fixed 2026-11-01): bc.game's index moved over the last 2 s AND sits ahead of our exchange
# price in that same direction (dev_tick) - it ran ahead of the market and tends to snap back
OVERSHOOT_VOL_MAX, OVERSHOOT_MOVE, OVERSHOOT_DEV = 1.92, 0.5, 2.0
OVERSHOOT_BOOK_MOVE, OVERSHOOT_BOOK_DEV, OVERSHOOT_BOOK_OBI = 0.5, 1.0, 0.5
MODEL_REGISTERED = {"gbm": 1793543190, "ens": 1793543190, "prob": 1793541896, "overshoot": 1793540105, "overshoot_book": 1793540105, "revert": 1793539811, "momvol": 1793516954, "mom": 1793421420, "book": 1793423220, "bookmom": 1793423400}


def _book_call(feat: dict | None) -> str | None:
    """The "book" model (pre-registered 2026-10-31): bet WITH Binance's 20-level order-book imbalance, only when it
    is strongly lopsided (|obi20| >= BOOK_MIN), read at the decision snapshot (~2.9 s before the cutoff).  On 2,002
    logged rounds: 53.4% [50.5, 56.2] on 59% of rounds (in-sample; the contrarian side was 46.6%).  None otherwise."""
    v = (feat or {}).get("obi20")
    if not isinstance(v, (int, float)) or abs(v) < BOOK_MIN or (feat or {}).get("error"):
        return None
    return "UP" if v > 0 else "DOWN"


def _bookmom_call(feat: dict | None) -> str | None:
    """The "bookmom" model (pre-registered 2026-10-31): bet only when Binance's 20-level book is strongly lopsided
    (|obi20| >= BOOK_MIN) AND our price moved the same way over the last 5 s; when they disagree, no call (on 2,000
    logged rounds a buying book with a falling price was a coin flip, 49% UP).  In-sample: 55.4% on 27% of rounds;
    a 30-feature logistic regression scored 49.4% out of sample on the same data, so the simple rule is used."""
    b, m = _book_call(feat), _mom_call(feat)
    return b if b is not None and b == m else None


def _momvol_call(feat: dict | None) -> str | None:
    """The "momvol" model (pre-registered 2026-11-01): momentum, only when the last 60 s were volatile.  Why: about
    half of all rounds are settled by bc.game moves under $2, where even our 6-exchange price agrees with bc.game only
    55-69%; momentum was 49.9% in calm minutes and 53.4% [50.8, 56.0] in the most volatile third (in-sample)."""
    v = (feat or {}).get("vol60")
    if not isinstance(v, (int, float)) or v < MOMVOL_MIN:
        return None
    return _mom_call(feat)


def _revert_call(feat: dict | None) -> str | None:
    """The "revert" model (pre-registered 2026-11-01): in a quiet market, bet that bc.game's own index undoes its last
    2-second move.  Rules picked on the first 60% of 5,445 logged rounds held on the last 40%: 54.1% of 603 then
    54.1% of 828 (|move| > $0.5), 54.0% / 56.0% (> $1), 54.9% / 57.7% (> $2); no effect in busy markets.  None otherwise."""
    f = feat or {}
    v, m = f.get("vol60"), f.get("bc_mom2")
    if not isinstance(v, (int, float)) or not isinstance(m, (int, float)) or f.get("error"):
        return None
    if v >= REVERT_QUIET_MAX or abs(m) <= REVERT_MIN_MOVE:
        return None
    return "DOWN" if m > 0 else "UP"


def _overshoot_call(feat: dict | None, with_book: bool = False) -> str | None:
    """Bet that bc.game's index snaps back after running ahead of the market.  Picked on the first 60% of 4,995 logged
    rounds, judged on the last 40%:  overshoot (not busy: vol60 < 1.92, |bc_mom2| > $0.5, dev_tick the same way
    > $2) 59.6% of 240 -> 57.1% of 226;  overshoot_book (any market, |bc_mom2| > $0.5, dev > $1, Binance book
    |obi20| >= 0.5 AGAINST the move) 60.3% of 199 -> 55.7% of 149.  None otherwise."""
    f = feat or {}
    m, d, v, o = f.get("bc_mom2"), f.get("dev_tick"), f.get("vol60"), f.get("obi20")
    if f.get("error") or not all(isinstance(x, (int, float)) for x in (m, d, v)):
        return None
    if with_book:
        if not isinstance(o, (int, float)) or abs(m) <= OVERSHOOT_BOOK_MOVE or abs(d) <= OVERSHOOT_BOOK_DEV:
            return None
        if (d > 0) != (m > 0) or abs(o) < OVERSHOOT_BOOK_OBI or (o > 0) == (m > 0):
            return None
    else:
        if v >= OVERSHOOT_VOL_MAX or abs(m) <= OVERSHOOT_MOVE or abs(d) <= OVERSHOOT_DEV or (d > 0) != (m > 0):
            return None
    return "DOWN" if m > 0 else "UP"


def _sign_call(pred: dict) -> str | None:
    """HawkesSign's side from a Hawkes forecast: UP if P(up) > 0.5, DOWN if below, None without a forecast."""
    p = pred.get("sign_p_up", 0.5) if pred.get("ready", True) else 0.5
    return "UP" if p > 0.5 else "DOWN" if p < 0.5 else None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a win rate k/n (0..1), (0, 1) without data."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class AutoTrader:
    # every model is scored on every round; only CALLERS may make the calls.  HawkesTime equals -sign(BI) in 92%
    # of rounds and scores 48.5%, so it is scored but never calls.  "agree": bet only when HawkesSign and MHH call
    # the same side, skip the round otherwise
    MODELS = {"time": "HawkesTime", "sign": "HawkesSign", "mhh": "MHH flow", "agree": "Sign + MHH agree",
              "mom": "Momentum 5s", "book": "Strong book", "bookmom": "Book + trend", "momvol": "Momentum, volatile only",
              "revert": "Quiet reversal", "overshoot": "Overshoot", "overshoot_book": "Overshoot + book",
              "prob": "P(UP) model", "gbm": "P(UP) trees", "ens": "P(UP) ensemble"}
    CALLERS = ("sign", "mhh", "agree", "mom", "book", "bookmom", "momvol", "revert", "overshoot", "overshoot_book",
               "prob", "gbm", "ens")
    STATES = ("DISARMED", "ARMED", "STOPPED")
    BREAK_EVEN = 0.5155             # 1 / 1.94: the win rate at which a flat bet breaks even (before pool dilution)
    GATE_MIN_N = 3000               # calls a model must have before its record can open the gate at all
    SIDE_WORDS = {"up": "UP", "u": "UP", "dn": "DOWN", "down": "DOWN", "d": "DOWN"}

    def __init__(self, trader: Trader, cfg: Config):
        self.trader, self.cfg = trader, cfg
        self.feed = LobFeed()
        self.pred = HawkesPredictor(self.feed)
        self.flow = BinanceFlow()                         # the MHH bot's model: Binance order flow -> P(UP)
        self.book = BinanceBook()                         # Binance top-20 book imbalance, for the panel's OB card
        self.flow_model = FlowLogit(cfg.flow_model)
        self.bc_ticks: deque = deque(maxlen=3000)         # (server s, bc.game BTC price) from the page's feed
        self.clock = BcClock()                            # bc.game's clock, from those ticks (clock.py)
        self.pools: dict = {}                             # latest Up / Down pool frame
        self.runs: dict = {}                              # per model: its losing runs by length (see _run_step)
        # our estimate of bc.game's price: a weighted, delayed mix of exchange prices (ourprice.py)
        self.ex_feeds = make_feeds()
        self.our = OurPrice(self.ex_feeds, self.bc_ticks)
        self.prob = ProbModel(cfg.auto_log)               # P(UP) from the logged rounds (bcbot/prob.py)
        self._prob_task = None
        self.on = False
        self.round: dict = {}
        self.recs: dict[int, dict] = {}
        self.history: deque = deque(maxlen=200)          # past expected returns, to scale the confidence z
        self.stats = {"rounds": 0, "called": 0, "correct": 0, "ties": 0, "skipped": 0, "pnl": 0.0,
                      "sign_called": 0, "sign_correct": 0, "bfx_n": 0, "bfx_agree": 0,
                      "live_orders": 0, "live_failed": 0, "live_won": 0, "live_pnl": 0.0}
        self.last: dict = {}
        self.recent: deque = deque(maxlen=30)            # "W" / "L" / "T" of the last called rounds, for the panel
        self.settled: deque = deque(maxlen=6)            # the last settled calls (records), for the panel's result list
        self.winners: deque = deque(maxlen=12)           # "U" / "D" / "T": who won the last rounds (called or not)
        self._open_rid, self._open_t = 0, 0.0            # the current round and when its betting opened (server s)
        self._tasks: list = []
        self._refit_rid = 0
        self._refit: asyncio.Future | None = None
        self._prepared_rid, self._prepared_stake = 0, None
        self._place_lock = asyncio.Lock()
        self.call_streak = 0                              # losing calls in a row (paper + live)
        self.day, self.day_live_pnl = "", 0.0             # today's real-money P&L (rebuilt from the log at start)
        # ---- real money: the state machine and the counters it is judged by (all rebuilt at start) ----
        self.state = "DISARMED"           # DISARMED (default at every start) / ARMED (by a typed command) / STOPPED
        self.stop_reason: str | None = None
        self.tech_stop: str | None = None  # a stop that needs 'live clear <this>' before any re-arm
        self.stopped_at = 0.0
        self.armed_at = 0.0
        self.rearm_grant = False           # beyond the run limit: a re-arm allows exactly one real order
        self.rearms = 0
        self.real_run = 0                  # real losses in a row; only a real WIN resets it, nothing else
        self.real_run_longest_today = 0
        self.attempts_today = 0            # real order attempts today (placed, rejected and unknown)
        self.campaign_since = 0.0          # the first arming; the campaign loss limit counts from here
        self.campaign_pnl = 0.0
        self.pending_real: dict = {}       # rid -> stake of real orders not settled yet
        self.unknown: list = []            # rounds whose real order's outcome is unknown (cleared by the operator)
        self.cleared: list = []            # rounds the operator cleared as "no order was placed"
        self.balance: float | None = None  # the page balance (account currency), when it was read, in USD
        self.balance_currency, self.balance_account, self.balance_at, self.balance_usd = "", "", 0.0, None
        self.dialog: str | None = None     # a site dialog covering the panel (seen at the last balance read)
        self.last_reload_at = 0.0          # the page reloaded / the round feed reconnected
        self._rejects: deque = deque(maxlen=20)   # (rid, time) of rejected real orders
        self._last_order: dict = {}
        self.events: deque = deque(maxlen=10)     # the last audit events, for the dashboard
        # shadow mode: every real-order check runs on paper and the bot logs what it would have done
        self.shadow_run, self.shadow_attempts_today, self.shadow_longest_today, self.shadow_stops_today = 0, 0, 0, 0
        # which model makes the calls and the paper stake progression: from .env, overridden by what was last
        # chosen at runtime (dashboard / API / terminal), kept in logs/auto_settings.json
        self.model, self.progression = cfg.auto_model, cfg.auto_progression
        self.since = 0.0                                  # the ladder test counts rounds from this time (last switch)
        self._settings_path = cfg.auto_log.with_name("auto_settings.json")
        try:
            saved = json.loads(self._settings_path.read_text(encoding="utf-8"))
            self.model = saved.get("model", self.model)
            self.progression = float(saved.get("progression", self.progression))
            self.since = float(saved.get("since", 0.0))
        except (OSError, ValueError):
            pass
        if self.model not in self.CALLERS:
            log.warning("%s can't make the calls (scored only); calls by HawkesSign", self.MODELS.get(self.model, self.model))
            self.model = "sign"
        # paper P&L if each model's stakes followed the progression (like the MHH dashboard's ladder P&L)
        self.ladders = {m: {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0}
                        for m in ("time", "sign", "mhh", "agree", "mom", "book", "bookmom", "momvol", "revert", "overshoot", "overshoot_book", "prob", "gbm", "ens")}

    # ---------- runtime settings ----------

    @property
    def armed(self) -> bool:
        return self.state == "ARMED"

    @property
    def live(self) -> bool:
        """Real orders are being placed (the same as armed; kept for the panel and the API)."""
        return self.state == "ARMED"

    def _save_settings(self) -> None:
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps({"model": self.model, "progression": self.progression,
                                                   "since": self.since}), encoding="utf-8")

    def _restart_test(self) -> None:
        """A new model / progression starts a fresh ladder test from now."""
        self.since = time.time()
        self.ladders = {m: {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0} for m in self.ladders}

    def set_model(self, name: str) -> None:
        name = name.strip().lower()
        if name not in self.CALLERS:
            raise TradeError(f"model must be one of {', '.join(self.CALLERS)} (HawkesTime is scored but can't call)")
        if self.armed:
            raise TradeError("disarm first ('live off'): the calling model can't change while real orders are on")
        if name != self.model:
            log.warning("Calls now made by %s (was %s)", self.MODELS[name], self.MODELS.get(self.model, self.model))
            self._restart_test()
        self.model = name
        self._save_settings()

    def set_progression(self, x: float) -> None:
        """The paper ladders' progression.  Real orders are always the flat base stake."""
        if not 1.0 <= x <= 3.0:
            raise TradeError("progression must be between 1 (flat) and 3")
        if x != self.progression:
            log.warning("Paper ladders: stake x%g after a loss (was x%g). Real orders stay flat $%g.", x,
                        self.progression, self.cfg.auto_stake_usd)
            self._restart_test()
        self.progression = x
        self._save_settings()

    # ---------- start / stop ----------

    async def start(self) -> None:
        """Start the order-book feed and the round loop (once); listen to the page's round feed."""
        if not self._tasks:
            self._load_state()
            self._load_log()
            self._check_start()
            self._prob_task = asyncio.ensure_future(asyncio.to_thread(self.prob.fit))
            await self.trader.ensure_ready()
            await self._attach(self.trader.page)
            self.trader.page_listeners.append(self._attach)              # again whenever the window is reopened
            self._tasks = [asyncio.create_task(self.feed.run()), asyncio.create_task(self.flow.run()),
                           asyncio.create_task(self._loop())]
            self._tasks += [asyncio.create_task(f.run()) for f in self.ex_feeds.values()]
            self._tasks.append(asyncio.create_task(self.book.run()))
        self.on = True
        log.info("Auto ON (calls by %s, paper; real orders %s). Hawkes models need ~5 min of order-book data, the "
                 "MHH flow model ~31 s of Binance seconds.", self.model, self.state.lower())

    def stop(self) -> None:
        self.on = False
        if self.armed:
            self.disarm("auto off", who="operator")
        log.info("Auto OFF")

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self.day != today:
            if self.day:
                self._event("day_roll", day=today)
            self.day, self.day_live_pnl = today, 0.0
            self.attempts_today = self.real_run_longest_today = 0
            self.shadow_attempts_today = self.shadow_longest_today = self.shadow_stops_today = 0

    def budget_left(self) -> float:
        """What today's real orders may still lose: the daily limit plus today's real P&L."""
        self._roll_day()
        return self.cfg.auto_daily_loss_usd + self.day_live_pnl

    def bc_now(self) -> float:
        """Now on bc.game's clock (from its ticks); the exchanges' clock while the tick clock is not ok."""
        t = self.clock.now() if self.clock.ok() else None
        return t if t is not None else self.our.server_now()

    def clock_src(self) -> str:
        return "bc" if self.clock.ok() else "exchange"

    # ---------- real money: arm / disarm / stop ----------

    def _event(self, event: str, **kw) -> None:
        """One line of the audit trail (logs/live_events.jsonl) and the dashboard's event list."""
        row = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "bc_time": round(self.bc_now(), 3),
               "event": event, "state": self.state, "real_run": self.real_run, "attempts_today": self.attempts_today,
               "day_pnl": round(self.day_live_pnl, 4), "campaign_pnl": round(self.campaign_pnl, 4),
               "balance_usd": round(self.balance_usd, 4) if self.balance_usd is not None else None, **kw}
        self.events.append(row)
        try:
            self.cfg.auto_events.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.auto_events.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except OSError as e:
            log.warning("events file: %r", e)

    def _save_state(self) -> None:
        """The real-loss counter and the campaign, kept across restarts (the log is the source; this is a check)."""
        data = {"real_run": self.real_run, "updated": time.time(), "campaign_since": self.campaign_since,
                "campaign_pnl": round(self.campaign_pnl, 4), "state": self.state, "stop_reason": self.stop_reason,
                "tech_stop": self.tech_stop, "stopped_at": self.stopped_at, "rearms": self.rearms,
                "unknown": self.unknown, "cleared": self.cleared, "day": self.day, "attempts_today": self.attempts_today}
        err = None
        for attempt in range(8):                  # Windows: another program (an editor, a scanner) may hold the
            try:                                  # file for a moment; a stop must not be lost to that
                self.cfg.auto_state_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cfg.auto_state_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(self.cfg.auto_state_file)
                if attempt:
                    log.info("state file saved on attempt %d", attempt + 1)
                return
            except OSError as e:
                err = e
                time.sleep(0.05 * (attempt + 1))
        log.warning("state file: %r (the log still has every order; the count is rebuilt from it at start)", err)

    def _load_state(self) -> None:
        try:
            st = json.loads(self.cfg.auto_state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self._saved_state = st
        self.campaign_since = float(st.get("campaign_since") or 0.0)
        self.stop_reason, self.tech_stop = st.get("stop_reason"), st.get("tech_stop")
        self.stopped_at, self.rearms = float(st.get("stopped_at") or 0.0), int(st.get("rearms") or 0)
        self.unknown, self.cleared = list(st.get("unknown") or []), list(st.get("cleared") or [])

    def _check_start(self) -> None:
        """After the log is replayed: compare the rebuilt counter with the saved one, and set the start state
        (never ARMED: every start is disarmed; STOPPED when the run limit or a technical stop is still in force)."""
        saved = getattr(self, "_saved_state", {}).get("real_run")
        if saved is not None and int(saved) != self.real_run:
            log.warning("real losses in a row: %d from the log, %d in the state file; keeping the larger",
                        self.real_run, saved)
            self.real_run = max(self.real_run, int(saved))
        if self.tech_stop or self.unknown:
            self.state = "STOPPED"
            if not self.tech_stop:
                self.tech_stop, self.stop_reason = "unknown", f"outcome of round {self.unknown[0]} unknown"
        elif self.real_run >= self.cfg.auto_max_real_run:
            self.state = "STOPPED"
            self.stop_reason = self.stop_reason or f"{self.real_run} real losses in a row"
        else:
            self.state, self.stop_reason = "DISARMED", None
        self._save_state()
        self._event("start", rebuilt_run=self.real_run)
        log.info("Real orders: %s%s | %d real losses in a row (only a real WIN resets it) | %d attempts today, "
                 "budget left $%.2f | campaign P&L %+.2f%s", self.state.lower(),
                 f" ({self.stop_reason})" if self.stop_reason else "", self.real_run, self.attempts_today,
                 self.budget_left(), self.campaign_pnl, "" if self.cfg.live_allowed else " | LIVE_ALLOWED is not set")

    def arm(self, ack_run: int, day: str, who: str = "operator", force: bool = False) -> str:
        """Allow real orders.  The command must name the current run of real losses and today's date, so nobody
        re-arms without seeing the state they accept; beyond the run limit it allows exactly ONE real order."""
        c = self.cfg
        if not self.on:
            raise TradeError("Turn auto on first")
        if not c.live_allowed or not c.tos_risk_accepted:
            raise TradeError("Real orders are off: set LIVE_ALLOWED=1 and TOS_RISK_ACCEPTED=1 in .env first "
                             "(bc.game's terms list automated play as prohibited; that risk is yours to take)")
        today = time.strftime("%Y-%m-%d")
        if int(ack_run) != self.real_run or day != today:
            raise TradeError(f"acknowledge the current state exactly: live arm run={self.real_run} day={today}")
        if c.auto_kill_file.exists():
            raise TradeError(f"the kill file {c.auto_kill_file} exists; delete it first")
        if self.tech_stop:
            raise TradeError(f"stopped: {self.stop_reason}. Check the site, then 'live clear {self.tech_stop}'")
        if self.state == "STOPPED" and time.time() - self.stopped_at < c.auto_rearm_cooloff_min * 60:
            left = c.auto_rearm_cooloff_min * 60 - (time.time() - self.stopped_at)
            raise TradeError(f"stopped {self.stop_reason}; the cool-off ends in {left / 60:.0f} min")
        if self.model not in self.CALLERS:
            raise TradeError("the calling model can't place orders")
        self._roll_day()
        if self.attempts_today >= c.auto_max_real_bets_day:
            raise TradeError(f"{self.attempts_today} real orders today already (limit {c.auto_max_real_bets_day})")
        if self.budget_left() < c.auto_stake_usd:
            raise TradeError(f"today's budget is used up (real P&L today {self.day_live_pnl:+.2f}, "
                             f"limit -${c.auto_daily_loss_usd:g}); try again tomorrow")
        if self.campaign_pnl <= -c.auto_campaign_loss_usd:
            raise TradeError(f"the campaign lost {self.campaign_pnl:+.2f}, past -${c.auto_campaign_loss_usd:g}: "
                             "real orders are over (a new campaign needs a new state file)")
        if self.balance_usd is not None and self.balance_usd < c.auto_min_bankroll_stakes * c.auto_stake_usd:
            raise TradeError(f"balance ${self.balance_usd:.2f} is below {c.auto_min_bankroll_stakes} stakes")
        gate = self.gate_status()
        if not gate["open"] and not force:
            raise TradeError(f"the gate is closed: {gate['text']}. Add 'force' to arm anyway (your decision; logged)")
        if not self.campaign_since:
            self.campaign_since = time.time()
        grant = self.real_run >= c.auto_max_real_run
        self.state, self.armed_at, self.stop_reason, self.tech_stop = "ARMED", time.time(), None, None
        self.rearm_grant = grant
        self.rearms += int(grant)
        self._save_state()
        self._event("arm", who=who, ack_run=int(ack_run), grant=grant, force=force, gate_open=gate["open"])
        lad_txt = ("compensate ladder " + "-".join(f"{x * c.auto_stake_usd:g}" for x in c.auto_real_ladder)
                   if self.ladder_on() else
                   "flat (the compensate ladder waits for the gate)" if len(c.auto_real_ladder) > 1 else "flat")
        msg = (f"ARMED ({lad_txt}): real orders of ${self.stake():g} by {self.MODELS[self.model]} for up to "
               f"{c.auto_arm_ttl_h:g} h. Stops at {c.auto_max_real_run} real losses in a row, at -${c.auto_daily_loss_usd:g} "
               f"today, after {c.auto_max_real_bets_day} orders a day, at -${c.auto_campaign_loss_usd:g} in all. "
               + (f"You are past the run limit ({self.real_run} losses): this arming allows ONE real order; a loss "
                  f"stops it again." if grant else "")
               + ("" if gate["open"] else f" The gate is closed ({gate['text']}): expect to lose."))
        log.warning(msg)
        return msg

    def disarm(self, why: str, who: str = "operator") -> None:
        if self.state == "ARMED":
            self._event("disarm", reason=why, who=who)
            log.warning("Real orders DISARMED (%s)", why)
        self.state, self.rearm_grant, self._prepared_rid = "DISARMED", False, 0
        self.stop_reason = None
        self._save_state()

    def kill(self, who: str = "operator") -> None:
        """Disarm at once (also what the kill file does)."""
        self.disarm("kill", who)

    def clear(self, what: str, outcome: str | None = None, who: str = "operator") -> str:
        """Clear a technical stop after the operator checked the site.  For an unknown order say what happened:
        'won' (resets the run), 'lost' (already counted) or 'none' (no order was placed: the provisional loss is
        taken back)."""
        if not self.tech_stop or (what != "all" and what != self.tech_stop):
            raise TradeError(f"nothing to clear{': the stop is ' + repr(self.tech_stop) if self.tech_stop else ''}")
        if self.tech_stop == "unknown":
            if outcome not in ("won", "lost", "none"):
                raise TradeError("say what the site's order history shows: live clear unknown won|lost|none")
            if outcome == "won":
                self.real_run = 0
            elif outcome == "none":
                self.real_run = max(0, self.real_run - len(self.unknown))
                self.cleared += self.unknown
            for rid in self.unknown:
                self.pending_real.pop(rid, None)
            self.unknown = []
        self._event("clear", what=self.tech_stop, outcome=outcome, who=who)
        self.tech_stop = None
        self.state, self.stop_reason = "DISARMED", None
        self._save_state()
        return f"cleared; {self.real_run} real losses in a row; disarmed"

    def _hard_stop(self, why: str, tech: str | None = None) -> None:
        """Real orders stop and the bot disarms.  It never resumes by itself: the operator must re-arm, naming the
        current run of losses (and clear the technical reason first, when there is one)."""
        self.state, self.stop_reason, self.stopped_at = "STOPPED", why, time.time()
        self.rearm_grant, self._prepared_rid = False, 0
        if tech:
            self.tech_stop = tech
        self._save_state()
        self._event("stop", reason=why, tech=tech)
        log.warning("REAL ORDERS STOPPED: %s. %s", why,
                    f"Check the site, then 'live clear {tech}'." if tech else
                    f"To go on: 'live arm run={self.real_run} day={time.strftime('%Y-%m-%d')}' "
                    f"(after {self.cfg.auto_rearm_cooloff_min:g} min).")

    def gate_status(self) -> dict:
        """The go-live gate: the calling model's record must beat the break-even with its Wilson 95% lower bound.
        For now this is its record over all logged rounds; the pre-registered forward test (frozen rule, fixed
        looks) is not built yet, so arming while it is closed needs 'force'."""
        s = self.stats
        fwd = self.model in MODEL_REGISTERED
        key = f"{self.model}_fwd" if fwd else self.model
        n, k = s.get(f"{key}_called", 0), s.get(f"{key}_correct", 0)
        lo, hi = _wilson(k, n)
        is_open = n >= self.GATE_MIN_N and lo > self.BREAK_EVEN
        scope = ("rounds since it was fixed (" + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(MODEL_REGISTERED[self.model]))
                 + ")") if fwd else "all logged rounds"
        text = (f"{self.MODELS.get(self.model, self.model)} {k}/{n} = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%}) on {scope} "
                f"vs break-even {self.BREAK_EVEN:.2%}, needs {self.GATE_MIN_N} calls" if n else
                f"no forward record yet ({scope})")
        return {"open": is_open, "n": n, "wins": k, "p": round(k / n, 4) if n else None,
                "lo": round(lo, 4), "hi": round(hi, 4), "break_even": self.BREAK_EVEN, "min_n": self.GATE_MIN_N,
                "text": text}

    # ---------- round feed ----------

    async def _attach(self, page) -> None:
        """Listen to the page's Detrade socket; reload so the socket opens after the listener exists."""
        page.on("websocket", self._on_ws)
        self.last_reload_at = time.time()
        await page.reload(wait_until="domcontentloaded")
        self.round = {}

    def _on_ws(self, ws) -> None:
        if "detrade" in ws.url:
            self.last_reload_at = time.time()           # a new round socket: a reload or a reconnect
            ws.on("framereceived", self._on_frame)

    def _on_frame(self, payload) -> None:
        try:
            raw = payload if isinstance(payload, (bytes, bytearray)) else payload.encode("latin-1")
            m = json.loads(zlib.decompress(raw).decode() if raw[:1] == b"x" else raw.decode())
        except Exception:
            return
        if m.get("cmd") == PRICE and isinstance(m.get("resp"), dict):      # bc.game's own BTC price ticks
            try:
                t = int(m["resp"]["t"]) / 1000
                self.bc_ticks.append((t, float(m["resp"]["p"])))
                self.clock.note(t, time.time())
            except (KeyError, TypeError, ValueError):
                pass
            return
        if m.get("cmd") == POOLS and isinstance(m.get("resp"), dict):
            p = self.pools = m["resp"]
            try:                                  # the round's pools, kept up to date until it settles
                rec = self.recs.get(int(p.get("roundId") or p.get("id") or (self.round or {}).get("id") or 0))
                if rec is not None and not rec.get("settled"):
                    rec["pools"] = [float(p["upPoolAmount"]), float(p["downPoolAmount"])]
            except (KeyError, TypeError, ValueError):
                pass
            return
        if m.get("cmd") != TICKER or not isinstance(m.get("resp"), dict):
            return
        r = m["resp"]
        self.round = r
        rid = int(r.get("id") or 0)
        if rid != self._open_rid:                        # a new round: its betting just opened
            self._open_rid, self._open_t = rid, self.bc_now()
        rec = self.recs.get(rid)
        if rec and not rec.get("settled") and r.get("winSide") in (1, 2) and r.get("endPrice"):
            self._settle(rec, r)

    # ---------- loop ----------

    async def _loop(self) -> None:
        n = 0
        while True:
            await asyncio.sleep(0.1)
            n += 1
            try:
                await self._step()
            except Exception as e:
                log.warning("auto loop: %r", e)
            if n % 10 == 0:
                try:
                    self._watch()
                except Exception as e:
                    log.warning("watch: %r", e)
            if n % 20 == 0:
                try:                     # reopens a closed window / reloads a crashed panel, re-attaching the round feed
                    if self.trader.page_gone() or await self.trader.widget_crashed():
                        self.last_reload_at = time.time()
                        await self.trader.ensure_ready()
                except Exception as e:
                    log.warning("recovering the browser: %r", e)
            if n % 5 == 0:
                try:                                # the panel is re-created after any page reload
                    await self.trader.page.evaluate(OVERLAY_JS, self.panel_html())
                except Exception:
                    pass
            if n % 2 == 0 and self.cfg.auto_page_curve:
                try:                                # our price curve on bc.game's own chart
                    await self.trader.page.evaluate(CURVE_JS, self.curve_data())
                except Exception:
                    pass

    def _watch(self) -> None:
        """Once a second: the clock's step check, the kill file, the arming's expiry, the hourly P(UP) refit."""
        self.clock.check_step(self.our.clock())
        if (time.time() - self.prob.fitted_at > PROB_REFIT_S and (self._prob_task is None or self._prob_task.done())):
            try:
                self._prob_task = asyncio.get_running_loop().create_task(asyncio.to_thread(self.prob.fit))
            except RuntimeError:                  # no event loop (a test): nothing to schedule
                pass
        if self.armed and self.cfg.auto_kill_file.exists():
            self.disarm(f"kill file {self.cfg.auto_kill_file.name} present", who="file")
        if self.armed and time.time() - self.armed_at > self.cfg.auto_arm_ttl_h * 3600:
            self.disarm(f"arming expired after {self.cfg.auto_arm_ttl_h:g} h", who="bot")

    async def _step(self) -> None:
        r = self.round
        if not r or r.get("status") != BETTING:
            return
        rid = int(r.get("id") or 0)
        cutoff_ms = r.get("tradeCutoffTime")
        if not cutoff_ms or not rid:
            return
        cutoff = cutoff_ms / 1000
        now = self.bc_now()
        if cutoff < now - 0.5:                            # a replayed frame after a reload: that round is over
            return
        left = cutoff - now
        if rid != self._refit_rid and (self._refit is None or self._refit.done()) and left > MIN_REFIT_S:
            self._refit_rid = rid                                    # refit at the start of each betting window
            self._refit = asyncio.ensure_future(asyncio.to_thread(self.pred.refit))
            asyncio.ensure_future(self._prepare(rid))                # the balance; the order ready when armed
        if not self.on:
            return
        rec = self.recs.setdefault(rid, {"id": rid})
        start = (r.get("priceStartTime") or cutoff_ms) / 1000
        rec.update(start=start, end=(r.get("priceEndTime") or 0) / 1000 or start + 5, cutoff=cutoff)
        if "hawkes_done" not in rec and now >= cutoff - self.cfg.auto_lead_ms / 1000:
            rec["hawkes_done"] = True
            try:                                  # snapshot A: what is known at the decision instant, taken before
                rec["feat"] = self._features(now, rec)   # the (slow) predict, for the prediction study
            except Exception as e:
                rec["feat"] = {"error": repr(e)[:120]}
            rec["hawkes_task"] = asyncio.ensure_future(self._hawkes(rid, rec))     # never awaited here
            try:                                                   # P(UP) for every round (all three scored)
                ps = self.prob.predict_all(rec["feat"])
                rec["prob"], rec["prob_gbm"], rec["prob_ens"] = ps["prob"], ps["gbm"], ps["ens"]
            except Exception as e:
                rec["prob"] = rec["prob_gbm"] = rec["prob_ens"] = None
                log.warning("P(UP) models: %r", e)
            if self.model in ("prob", "gbm", "ens"):
                key = {"prob": "prob", "gbm": "prob_gbm", "ens": "prob_ens"}[self.model]
                asyncio.ensure_future(self._call(rid, rec, ProbModel.side(rec[key])))
            if self.model == "mom":                                  # momentum calls at snapshot A, at once
                asyncio.ensure_future(self._call(rid, rec, _mom_call(rec["feat"])))
            elif self.model == "book":                               # strong book: at snapshot A, or no call
                asyncio.ensure_future(self._call(rid, rec, _book_call(rec["feat"])))
            elif self.model == "bookmom":                            # strong book + same-way trend, or no call
                asyncio.ensure_future(self._call(rid, rec, _bookmom_call(rec["feat"])))
            elif self.model == "momvol":                             # momentum in volatile minutes only
                asyncio.ensure_future(self._call(rid, rec, _momvol_call(rec["feat"])))
            elif self.model == "revert":                             # quiet market: bc.game undoes its last move
                asyncio.ensure_future(self._call(rid, rec, _revert_call(rec["feat"])))
            elif self.model in ("overshoot", "overshoot_book"):      # bc.game ran ahead of the market: snap-back
                asyncio.ensure_future(self._call(rid, rec, _overshoot_call(rec["feat"], self.model == "overshoot_book")))
        if "mhh_done" not in rec and now >= cutoff - self.flow_model.lead_ms / 1000:   # MHH's own lead (2 s)
            rec["mhh_done"] = True
            asyncio.ensure_future(self._mhh(rid, rec))
        if "feat_late" not in rec and now >= cutoff - self.cfg.auto_late_snapshot_s:
            try:                                  # snapshot B: the late, tick-aligned deviation (paper research)
                rec["feat_late"] = self._features(now, rec, late=True)
            except Exception as e:
                rec["feat_late"] = {"error": repr(e)[:120]}
        for old in [k for k in self.recs if k < rid - 50]:
            self.recs.pop(old, None)

    async def _hawkes(self, rid: int, rec: dict) -> None:
        """HawkesTime + HawkesSign forecasts (always scored); the call when the model is sign."""
        now = self.bc_now()
        a = rec["start"] - now                 # the window starts this long from now (bc.game's clock)
        c = self.cfg
        if not c.auto_min_lead_s <= a <= c.auto_max_lead_s:
            rec["skip"] = f"odd timing (window starts in {a:.1f} s)"     # a replayed or mistimed round: not
            self.stats["skipped"] += 1                                    # predicted, not scored, not in the history
            return
        p = await asyncio.to_thread(self.pred.predict, a, rec["end"] - now)
        p["done_lead_s"] = round(rec["cutoff"] - self.bc_now(), 3)        # how long before the cutoff it finished
        rec["pred"] = p
        calling = self.model == "sign"
        if not p.get("ready"):
            if calling:
                rec["skip"] = "Hawkes models warming up (%.0f s of data)" % self.pred.data_span()
                self.stats["skipped"] += 1
            return
        hist = np.array(self.history)
        z = abs(p["exp_ret"]) / hist.std() if len(hist) >= 20 and hist.std() > 0 else None
        self.history.append(p["exp_ret"])
        rec["z"] = z
        log.info("round %s Hawkes: time %s (exp 5s return %+.2e, z %s)  sign P(up) %.2f  ~%.1f changes  "
                 "(lead %.1fs, calc %d ms, done %.2fs before cutoff)", rid, p["time_call"] or "-", p["exp_ret"],
                 f"{z:.2f}" if z is not None else "-", p["sign_p_up"], p["n_changes"], p["lead_s"], p["calc_ms"],
                 p["done_lead_s"])
        if calling:
            await self._call(rid, rec, _sign_call(p))

    async def _mhh(self, rid: int, rec: dict) -> None:
        """The MHH bot's Binance-flow logit (always scored); the call when the model is mhh, or agree (with
        HawkesSign's forecast, made 1 s earlier)."""
        m = self.flow_model.predict(self.flow)
        rec["mhh"] = m
        if "error" not in m:
            log.info("round %s MHH flow: P(up) %.3f conf %.3f -> %s", rid, m["p_up"], m["conf"], m["call"])
        if self.model == "agree":
            task = rec.get("hawkes_task")
            if task is not None and not task.done():            # wait for HawkesSign, but never past the margin
                timeout = max(0.0, rec["cutoff"] - self.bc_now() - self.cfg.auto_send_margin_s - 0.1)
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout)
                except (asyncio.TimeoutError, Exception):
                    pass
            sign = _sign_call(rec.get("pred") or {})
            if "error" in m or sign is None:
                rec["skip"] = m.get("error") or ("HawkesSign not ready in time" if task and not task.done()
                                                 else "HawkesSign has no forecast")
                self.stats["skipped"] += 1
            elif sign != m["call"]:
                rec["skip"] = f"models disagree (HawkesSign {sign}, MHH {m['call']})"
                self.stats["skipped"] += 1
            else:
                await self._call(rid, rec, sign)
            return
        if self.model != "mhh":
            return
        if "error" in m:
            rec["skip"] = m["error"]
            self.stats["skipped"] += 1
            return
        await self._call(rid, rec, m["call"])

    async def _call(self, rid: int, rec: dict, call: str | None) -> None:
        """Record the calling model's UP / DOWN (paper), place it for real when armed, and in shadow mode run
        every real-order check anyway and log what the bot would have done."""
        if rec.get("call") or rec.get("skip"):          # one call per round, whichever model trigger comes first
            return
        if call is None:
            rec["skip"] = "no call"
            self.stats["skipped"] += 1
            return
        rec["call"] = call
        rec["decided_lead_s"] = round(rec["cutoff"] - self.bc_now(), 3)
        stake = self.stake()
        rec["stake"] = stake if self.armed else self.paper_stake()      # real: stake(); paper: the paper ladder
        if self.armed:
            async with self._place_lock:
                await self._place(rid, call, rec, stake)
        elif self.cfg.auto_shadow:
            self._shadow(rid, rec, stake)
        try:
            st = await self.trader.status(light=True)
        except Exception:
            st = None
        payout = ((st.up_payout if call == "UP" else st.down_payout) or 0) / 100 if st else 0.0
        if payout <= 1:                       # not readable on the page: book at the measured average
            payout = DEFAULT_PAYOUT
            rec["payout_guessed"] = True
        rec.update(payout=payout, account=st.account if st else None,
                   payouts=[(st.up_payout or 0) / 100, (st.down_payout or 0) / 100] if st else None)
        rec["pool"] = _pool_call(rec.get("payouts"))                   # scored like the models (never calls)
        self.last = {"round": rid, "call": call, "payout": payout, "model": self.model}
        log.info("round %s: CALL %s (%s)  payout x%.2f%s", rid, call, self.model, payout,
                 "  shadow: " + ((rec.get("shadow") or {}).get("reason") or "would place") if rec.get("shadow") else "")

    def _features(self, at: float, rec: dict | None = None, late: bool = False) -> dict:
        """The prediction study's inputs at bc.game time `at` (snapshot A at the decision, B late in the window),
        logged with the round (its result is the label).  Prices in $, order flow in each exchange's size units.
        Anything older than its validity window is None, never a stale or zero value; dq says what was stale."""
        now = at
        ticks = list(self.bc_ticks)
        f: dict = {"at": round(now, 3), "clock": self.clock.dict()}
        if rec and rec.get("cutoff"):
            f["lead_s"] = round(rec["cutoff"] - now, 3)
        if len(ticks) < 20:
            f["dq"] = {"ok": False, "stale": ["bc_ticks"]}
            return f
        snap = self.our._snapshot()
        ts = [t for t, _ in ticks]

        def bc_at(t):
            i = bisect.bisect_right(ts, t) - 1
            return ticks[i][1] if i >= 0 else None

        def our(t):
            return self.our.value(t, snap, max_age=STALE_SNAPSHOT, min_weight=MIN_WEIGHT)

        tick_t, bc_now = ticks[-1]
        our_now, our_tick = our(now), our(tick_t)
        f.update(bc=bc_now, our=our_now, tick_t=tick_t, tick_age=now - tick_t)
        if our_tick is not None:
            f["dev_tick"] = bc_now - our_tick                 # bc.game's gap from the market at the tick's own time
            i = bisect.bisect_right(ts, tick_t - 1.0) - 1     # ... and its change over the last second
            if i >= 0 and (o1 := our(ticks[i][0])) is not None:
                f["dev_chg1"] = f["dev_tick"] - (ticks[i][1] - o1)
        if len(ticks) >= 2:
            f["tick_chg"] = bc_now - ticks[-2][1]
        if our_now is not None:
            f["dev"] = bc_now - our_now                       # the gap read at "now" (the earlier definition)
            if not late:
                devs = [(t, p - o) for t, p in ticks if t >= now - 3 and (o := our(t)) is not None]
                for w in (1, 3):
                    d = [v for t, v in devs if t >= now - w]
                    f[f"dev_{w}s"] = sum(d) / len(d) if d else None
        b05 = bc_at(now - 0.5)                                 # bc.game's index over the last half second
        f["bc_mom05"] = bc_now - b05 if b05 else None
        for w in ((1,) if late else (1, 2, 5, 10)):           # recent moves
            b0, o0 = bc_at(now - w), our(now - w)
            f[f"bc_mom{w}"] = bc_now - b0 if b0 else None
            f[f"our_mom{w}"] = our_now - o0 if our_now is not None and o0 is not None else None
        if not late and our_tick is not None:                  # bc.game's gap from the market over the last minute
            d60 = [b - o for i in range(1, 61) if (b := bc_at(now - i)) is not None and (o := our(now - i)) is not None]
            if len(d60) >= 30:
                mean60, std60 = float(np.mean(d60)), float(np.std(d60))
                f["dev_mean60"], f["dev_std60"] = mean60, std60
                f["dev_z60"] = (f["dev_tick"] - mean60) / std60 if std60 > 0.05 else None
        grid = [bc_at(now - 60 + 0.5 * i) for i in range(121)]
        rets = [math.log(b / a) for a, b in zip(grid, grid[1:]) if a and b]
        if len(rets) > 20:
            f["vol60"] = float(np.std(rets)) * bc_now         # bc.game's 0.5 s moves over the last minute ($)
        stale = list(self.our.stale(now, snap))
        ages = {k: round(self.ex_feeds[k].age(), 2) for k in MIX if k in self.ex_feeds}
        if not late:
            for k in ("binance_perp", "bybit_perp", "okx_perp"):  # futures order flow (top of book)
                fd = self.ex_feeds.get(k)
                if fd is not None:
                    f[f"ofi_{k}_1s"], f[f"ofi_{k}_3s"] = fd.ofi(1), fd.ofi(3)
                    if f[f"ofi_{k}_1s"] is None:
                        stale.append(f"ofi_{k}")
            perps = [k for k in self.ex_feeds if k.endswith("_perp")]
            spots = [k for k in self.ex_feeds if not k.endswith("_perp")]

            def basis(t):                                     # futures minus spot (mean mids)
                pv = [v for k in perps if (v := OurPrice._price(snap, k, 0, t, STALE_SNAPSHOT)) is not None]
                sv = [v for k in spots if (v := OurPrice._price(snap, k, 0, t, STALE_SNAPSHOT)) is not None]
                return sum(pv) / len(pv) - sum(sv) / len(sv) if pv and sv else None
            b_now, b_10 = basis(now), basis(now - 10)
            f["basis_chg10"] = b_now - b_10 if b_now is not None and b_10 is not None else None
            bk = self.book                                    # Binance spot top-20 book (the panel's OB card)
            ages["binance_book"] = round(bk.age(), 2)
            if bk.age() <= 2.0:
                f.update(obi1=bk.obi1, obi5=bk.obi5, obi20=bk.obi20, obi_w=bk.obi_w, micro_off=bk.micro_off,
                         spread_binance=bk.spread)
                hist = list(bk.hist)
                if len(hist) > 5:
                    f["obi20_chg5"] = bk.obi20 - hist[-6][1]
            else:
                f.update(obi1=None, obi5=None, obi20=None, obi20_chg5=None)
                stale.append("binance_book")
            ages["binance_flow"] = round(self.flow.age(), 2)
            rows = list(self.flow.ring)                       # Binance spot taker flow (closed seconds)
            fresh = self.flow.age() <= 2.5
            for w in (1, 3, 5, 10):
                if fresh and len(rows) >= w:
                    v, tb = sum(r[2] for r in rows[-w:]), sum(r[3] for r in rows[-w:])
                    f[f"taker{w}"] = (2 * tb - v) / v if v > 0 else None
                else:
                    f[f"taker{w}"] = None
            if not fresh:
                stale.append("binance_flow")
        pools = (rec or {}).get("pools")
        if pools:
            f["pool_up"], f["pool_dn"] = pools
        else:
            p = self.pools or {}
            f["pool_up"], f["pool_dn"] = p.get("upPoolAmount"), p.get("downPoolAmount")
        try:
            up, dn = float(f["pool_up"] or 0), float(f["pool_dn"] or 0)
            f["pay_up"] = 1 + PAY_K * dn / up if up > 0 else None
            f["pay_dn"] = 1 + PAY_K * up / dn if dn > 0 else None
        except (TypeError, ValueError):
            f["pay_up"] = f["pay_dn"] = None
        if now - tick_t > 1.0:
            stale.append("bc_ticks")
        if not self.feed.connected:
            stale.append("bitfinex_lob")
        ages["bc_tick"] = round(now - tick_t, 3)
        f["dq"] = {"ok": now - tick_t <= 1.0 and our_now is not None and self.clock.ok() and self.feed.connected,
                   "stale": stale, "ages": ages}
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}

    def ladder_on(self) -> bool:
        """The real compensate ladder is used only when it is configured (AUTO_REAL_LADDER with 2+ steps) AND the
        go-live gate is open: the calling model has >= GATE_MIN_N calls with its Wilson lower bound above the
        break-even.  Without a proven edge a ladder only loses faster (docs/decision-review, section 7b)."""
        return len(self.cfg.auto_real_ladder) > 1 and self.gate_status()["open"]

    def stake(self) -> float:
        """The real stake: the flat base, or with the ladder on, base x the ladder step for the current run of real
        losses (reset by a real win; after the last step it starts again from the first).  The step follows the
        persistent loss counter, so a restart cannot reset it; every stop and the day's budget still apply."""
        c = self.cfg
        if not self.ladder_on():
            return c.auto_stake_usd
        lad = c.auto_real_ladder
        return round(c.auto_stake_usd * lad[self.real_run % len(lad)], 2)

    def paper_stake(self) -> float:
        """The calling model's next paper stake: the base x progression for each loss in a row of its paper ladder."""
        lad = self.ladders.get(self.model) or {}
        return round(self.cfg.auto_stake_usd * self.progression ** lad.get("step", 0), 2)

    # ---------- real orders: the checklist and the order path ----------

    async def _refresh_balance(self) -> None:
        """Read the page's balance (account currency and USD), the account kind and any dialog."""
        try:
            st = await self.trader.status(light=True)
        except Exception as e:
            log.debug("balance read failed: %r", e)
            return
        self.dialog = st.dialog
        rate = None
        if st.currency:
            try:
                rate = await self.trader.usd_rate(st.currency)
            except Exception:
                rate = None
        if self.armed and self.balance_currency and st.currency and st.currency != self.balance_currency:
            self._hard_stop(f"wallet currency changed from {self.balance_currency} to {st.currency}", tech="currency")
        self.balance, self.balance_currency, self.balance_account = st.balance, st.currency, st.account
        self.balance_usd = st.balance * rate if st.balance is not None and rate else None
        self.balance_at = time.time()

    async def _prepare(self, rid: int) -> None:
        """At the start of a betting window: read the balance (every round), and when armed select the room and
        fill the amount box, so the call only has to click."""
        await self._refresh_balance()
        if not self.armed:
            return
        stake = self.stake()
        try:
            await self.trader.prepare_usd(stake)
            self._prepared_rid, self._prepared_stake = rid, stake
        except Exception as e:
            self._prepared_rid = 0
            log.warning("round %s: could not prepare the order (%s); no real order this round", rid, e)

    def _may_place(self, rid: int, rec: dict, stake: float, shadow: bool = False, manual: bool = False) -> str | None:
        """Why a real order must NOT be placed now, or None when every check passes.  Run immediately before the
        click on every route (automatic, manual, shadow); the round is never retried after a refusal."""
        c = self.cfg
        now = self.bc_now()
        run = self.shadow_run if shadow else self.real_run
        attempts = self.shadow_attempts_today if shadow else self.attempts_today
        if not shadow:
            if self.state != "ARMED":
                return f"not armed ({self.state.lower()}{': ' + self.stop_reason if self.stop_reason else ''})"
            if self.tech_stop:
                return f"stopped: {self.stop_reason} (needs 'live clear {self.tech_stop}')"
            if time.time() - self.armed_at > c.auto_arm_ttl_h * 3600:
                return "arming expired"
            if c.auto_kill_file.exists():
                return "kill file present"
            if self.pending_real:
                return f"real order of round {next(iter(self.pending_real))} not settled yet"
            if self.unknown:
                return f"outcome of round {self.unknown[0]} unknown"
        if run >= c.auto_max_real_run and not (self.rearm_grant and not shadow):
            return f"{run} real losses in a row (limit {c.auto_max_real_run})"
        self._roll_day()
        if attempts >= c.auto_max_real_bets_day:
            return f"{attempts} real orders today (limit {c.auto_max_real_bets_day})"
        budget = c.auto_daily_loss_usd + self.day_live_pnl
        if stake > budget + 1e-9:
            return f"today's budget left ${budget:.2f} is less than the stake"
        if self.campaign_pnl <= -c.auto_campaign_loss_usd:
            return f"campaign loss {self.campaign_pnl:+.2f} reached -${c.auto_campaign_loss_usd:g}"
        if stake > c.auto_max_stake_usd:
            return f"stake ${stake:g} above AUTO_MAX_STAKE_USD {c.auto_max_stake_usd:g}"
        if self.balance_usd is None or time.time() - self.balance_at > c.auto_balance_max_age_s:
            return f"balance not read in the last {c.auto_balance_max_age_s:g} s"
        if self.balance_account != "REAL":
            return f"account is {self.balance_account or 'unknown'}, not REAL"
        if self.balance_usd < c.auto_min_bankroll_stakes * stake:
            return f"balance ${self.balance_usd:.2f} is below {c.auto_min_bankroll_stakes} stakes"
        if self.dialog:
            return f"a dialog is open on the page ({self.dialog[:30]})"
        if not self.clock.ok():
            return f"bc.game clock not ok ({self.clock.why_not_ok()})"
        if now > rec["cutoff"] - c.auto_send_margin_s:
            return f"too late ({rec['cutoff'] - now:.2f} s before the cutoff, margin {c.auto_send_margin_s:g} s)"
        r = self.round or {}
        if int(r.get("id") or 0) != rid or r.get("status") != BETTING:
            return "betting closed or another round open"
        if time.time() - self.last_reload_at < 60:
            return "page reloaded or round feed reconnected less than 60 s ago"
        if self._rejects and time.time() - self._rejects[-1][1] < 45:
            return "an order was rejected in the previous round"
        if not manual:
            p = rec.get("pred") or {}
            if p.get("ready") and not (c.auto_min_lead_s <= p.get("lead_s", 0) <= c.auto_max_lead_s):
                return f"odd forecast lead ({p.get('lead_s')} s)"
            dq = (rec.get("feat") or {}).get("dq") or {}
            if dq and not dq.get("ok", True):
                return "data quality: " + ", ".join(dq.get("stale") or ["?"])
            if not shadow and (self._prepared_rid != rid or self._prepared_stake != stake):
                return "order not prepared for this round"
        return None

    def _shadow(self, rid: int, rec: dict, stake: float) -> None:
        """Shadow mode: what a real order would have met at this instant, logged with the round."""
        reason = self._may_place(rid, rec, stake, shadow=True)
        rec["shadow"] = {"ok": reason is None, "reason": reason, "lead_s": round(rec["cutoff"] - self.bc_now(), 3)}
        if reason is None:
            self._roll_day()
            self.shadow_attempts_today += 1

    async def _place(self, rid: int, call: str, rec: dict, stake: float) -> bool:
        """The automatic real order: the checklist, then the prepared click before the deadline on bc.game's
        clock.  One attempt per round, never a slow path, never a retry."""
        reason = self._may_place(rid, rec, stake)
        if reason:
            rec["live"] = {"ok": False, "skipped": reason}
            log.warning("round %s REAL %s not placed: %s", rid, call, reason)
            return False
        deadline = rec["cutoff"] - self.cfg.auto_send_margin_s
        return await self._send(rid, call, rec, stake, "auto",
                                lambda: self.trader.click_prepared(call, deadline, now=self.bc_now))

    async def _send(self, rid: int, call: str, rec: dict, stake: float, who: str, sender) -> bool:
        """Bookkeeping around one real order: the intent is written before the click, the outcome after it."""
        grant, self.rearm_grant = self.rearm_grant, False       # a grant is spent by the attempt
        self.attempts_today += 1
        self.pending_real[rid] = stake
        await self._refresh_balance()                           # the balance right before the click, to reconcile
        rec["balance_before"] = [self.balance, self.balance_currency]
        self._event("order_intent", rid=rid, side=call, stake=stake, who=who)
        try:
            res = await sender()
        except TooLate as e:                                    # nothing was sent: not an attempt
            self.pending_real.pop(rid, None)
            self.attempts_today -= 1
            self.rearm_grant = grant
            rec["live"] = {"ok": False, "skipped": str(e)}
            self._event("order_skipped", rid=rid, reason=str(e))
            log.warning("round %s REAL %s skipped: %s", rid, call, e)
            return False
        except OrderUnknown as e:                               # clicked, no answer: maybe placed, counted as lost
            rec["live"] = {"ok": None, "unknown": True, "error": str(e)[:200], "stake": stake, "side": call.lower()}
            self.unknown.append(rid)
            self.real_run += 1
            self.stats["live_failed"] += 1
            self._event("order_unknown", rid=rid, error=str(e)[:200])
            self._hard_stop(f"outcome of the order in round {rid} unknown", tech="unknown")
            return False
        except Exception as e:                                  # rejected (balance, late, status): no retry
            self.pending_real.pop(rid, None)
            self.rearm_grant = grant
            self.stats["live_failed"] += 1
            rec["live"] = {"ok": False, "error": str(e)[:200]}
            self._rejects.append((rid, time.time()))
            self._event("order_rejected", rid=rid, error=str(e)[:200])
            log.warning("round %s REAL %s REJECTED (%.2fs before cutoff): %s", rid, call,
                        rec["cutoff"] - self.bc_now(), e)
            recent = [t for _, t in self._rejects if time.time() - t < 600]
            if len(recent) >= self.cfg.auto_reject_stop:
                self._hard_stop(f"{len(recent)} orders rejected within 10 min", tech="rejected")
            return False
        self.stats["live_orders"] += 1
        rec["live"] = {**res, "stake": stake, "grant": grant, "who": who}
        self._last_order = {k: res.get(k) for k in ("sent", "acked", "rtt", "code", "msg", "order_id", "round_id")}
        self._event("order_sent", rid=rid, order_id=res.get("order_id"), round_id=res.get("round_id"),
                    rtt=res.get("rtt"), lead_s=round(rec["cutoff"] - res["sent"] + (self.clock.offset() or 0), 3))
        log.info("round %s REAL %s placed: order %s (%g %s), %.2fs before cutoff, rtt %.2fs%s", rid, call,
                 res["order_id"], res["amount"], res["currency"], rec["cutoff"] - self.bc_now(), res.get("rtt") or 0,
                 "  (the one order of this re-arm)" if grant else "")
        booked = int(res.get("round_id") or 0)
        if booked and booked != rid:            # booked in another round: settle it against that round's result
            self.pending_real.pop(rid, None)
            self.pending_real[booked] = stake
            brec = self.recs.setdefault(booked, {"id": booked})
            brec.update(live=rec.pop("live"), call=call, stake=stake, misbooked_from=rid,
                        balance_before=rec.get("balance_before"))
            rec["live"] = {"ok": False, "misbooked": booked}
            self._hard_stop(f"order booked in round {booked} instead of {rid}", tech="misbooked")
        return True

    async def manual_order(self, side: str, amount: float, who: str = "operator") -> dict:
        """A typed / API order in the account currency: refused unless armed, and counted like every other real
        order (the run of losses, today's attempts and budget, the campaign)."""
        call = self.SIDE_WORDS.get(side.strip().lower())
        if call is None:
            raise TradeError("Side must be 'up' or 'dn'")
        r = self.round or {}
        rid = int(r.get("id") or 0)
        if r.get("status") != BETTING or not rid or not r.get("tradeCutoffTime"):
            raise TradeError("Betting is closed; a manual order goes into the open round only")
        await self._refresh_balance()
        rate = await self.trader.usd_rate(self.balance_currency) if self.balance_currency else 1.0
        stake = round(amount * rate, 4)
        rec = self.recs.setdefault(rid, {"id": rid})
        cutoff = r["tradeCutoffTime"] / 1000
        start = (r.get("priceStartTime") or r["tradeCutoffTime"]) / 1000
        rec.setdefault("start", start)
        rec.setdefault("end", (r.get("priceEndTime") or 0) / 1000 or start + 5)
        rec.setdefault("cutoff", cutoff)
        if rec.get("live"):
            raise TradeError("there is already a real order in this round")
        async with self._place_lock:
            reason = self._may_place(rid, rec, stake, manual=True)
            if reason:
                raise TradeError(f"refused: {reason}")
            rec.update(call=call, stake=stake, manual=True)
            deadline = cutoff - self.cfg.auto_send_margin_s
            ok = await self._send(rid, call, rec, stake, who,
                                  lambda: self.trader.buy(side, amount, wait_window=False, deadline=deadline,
                                                          now=self.bc_now))
        lv = rec.get("live") or {}
        if not ok:
            raise TradeError(lv.get("error") or lv.get("skipped") or "not placed")
        return lv

    # ---------- results ----------

    def _settle(self, rec: dict, r: dict) -> None:
        rec["settled"] = True
        win = "UP" if r["winSide"] == 1 else "DOWN"
        sp, ep = float(r.get("startPrice") or 0), float(r.get("endPrice") or 0)
        tie = sp == ep
        ps, pe = r.get("priceStartTime"), r.get("priceEndTime")
        if ps and pe:                               # the real window (the betting frame carried the planned one)
            rec["start_planned"], rec["start"], rec["end"] = rec.get("start"), ps / 1000, pe / 1000
        if "start" in rec:                          # did Bitfinex move the same way as bc.game's price?
            m0, m1 = self.feed.mid_at(rec["start"]), self.feed.mid_at(rec["end"])
            if m0 and m1 and m0 != m1 and not tie:
                rec["bfx_ret"] = (m1 - m0) / m0
            # our price (the chart's Binance + Bitfinex average) over the same window: its Up/Down vs bc.game's
            b0, b1 = _price_at(self.flow.mids, rec["start"]), _price_at(self.flow.mids, rec["end"])
            moves = [x1 - x0 for x0, x1 in ((b0, b1), (m0, m1)) if x0 and x1]
            if moves and not tie:
                rec["our_move"] = round(sum(moves) / len(moves), 2)          # old recipe: Binance + Bitfinex
            mix = self.our.move(rec["start"], rec["end"])                       # new: the weighted, delayed mix
            if mix is not None and not tie:
                rec["mix_move"], rec["mix_v"] = mix, MIX_VERSION
        call = rec.get("call")
        result = ("TIE" if tie else "WIN" if call == win else "LOSS") if call else None
        rec.update(win=win, tie=tie, result=result, start_price=sp, end_price=ep)
        pnl = rec["pnl"] = self._pnl(rec, call)                       # paper: the base stake
        lv = rec.get("live") or {}
        if lv.get("ok"):                                                # real: the stake actually placed
            rec["live_pnl"] = round(pnl * lv.get("stake", self.cfg.auto_stake_usd) / self.cfg.auto_stake_usd, 4)
        self._count(rec)
        if call:
            s = self.stats
            log.info("round %s result %s (%s -> %s): call %s -> %s %+.2f%s | accuracy %d/%d  paper P&L %+.2f",
                     rec["id"], win, sp, ep, call, result, pnl,
                     f" (REAL ${lv.get('stake', 0):g}: {rec['live_pnl']:+.2f})" if lv.get("ok") else "",
                     s["correct"], s["called"], s["pnl"])
            if result in ("WIN", "LOSS"):
                self.call_streak = self.call_streak + 1 if result == "LOSS" else 0
            if lv.get("ok"):
                self._real_settled(rec, result)
            elif (rec.get("shadow") or {}).get("ok") and result in ("WIN", "LOSS"):
                self._shadow_settled(rec, result)
        self._record(rec)

    def _real_settled(self, rec: dict, result: str | None) -> None:
        """A real order's result: the counters, the state file, the stops, the reconciliation."""
        c = self.cfg
        rid = rec["id"]
        self.pending_real.pop(rid, None)
        self._roll_day()
        self.day_live_pnl += rec["live_pnl"]
        self.campaign_pnl += rec["live_pnl"]
        if result == "LOSS":
            self.real_run += 1
            self.real_run_longest_today = max(self.real_run_longest_today, self.real_run)
        elif result == "WIN":
            self.real_run = 0
        self._save_state()
        self._event("real_settled", rid=rid, result=result, pnl=rec["live_pnl"])
        if self.armed:
            if self.real_run >= c.auto_max_real_run:
                self._hard_stop(f"{self.real_run} real losses in a row")
            elif self.day_live_pnl <= -c.auto_daily_loss_usd:
                self._hard_stop(f"daily loss limit: real P&L today {self.day_live_pnl:+.2f}")
            elif self.campaign_pnl <= -c.auto_campaign_loss_usd:
                self._hard_stop(f"campaign loss {self.campaign_pnl:+.2f} reached -${c.auto_campaign_loss_usd:g}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._reconcile(rec))
            if self.armed and self.ladder_on() and self._prepared_rid and self._prepared_stake != self.stake():
                loop.create_task(self._prepare(self._prepared_rid))      # the ladder step changed: new amount
        except RuntimeError:                      # no event loop (a replay / a test): nothing to reconcile against
            pass

    def _shadow_settled(self, rec: dict, result: str) -> None:
        """Shadow mode's own run of losses: how often the real rules would have stopped the bot."""
        self.shadow_run = self.shadow_run + 1 if result == "LOSS" else 0
        self.shadow_longest_today = max(self.shadow_longest_today, self.shadow_run)
        if self.shadow_run >= self.cfg.auto_max_real_run:
            self.shadow_stops_today += 1
            rec["shadow"]["stop"] = True
            log.info("SHADOW: %d losses in a row - with real money the bot would have stopped here (stop #%d today)",
                     self.shadow_run, self.shadow_stops_today)
            self.shadow_run = 0

    async def _reconcile(self, rec: dict) -> None:
        """A few seconds after a real settle: does the page balance agree with the booked result?  Anything else
        (another order, a deposit, a wrong payout read) stops real orders until the operator has looked."""
        lv = rec.get("live") or {}
        before = rec.get("balance_before") or [None, None]
        if before[0] is None or not lv.get("ok"):
            return
        amount = float(lv.get("amount") or 0)
        pay = (lv.get("payout_pct") or 0) / 100
        if rec.get("result") == "WIN":
            expected = amount * ((pay if pay > 1 else DEFAULT_PAYOUT) - 1)
        elif rec.get("result") == "LOSS":
            expected = -amount
        else:
            expected = 0.0
        delta = None
        for wait in (5, 6):
            await asyncio.sleep(wait)
            await self._refresh_balance()
            if self.balance is None or self.balance_currency != before[1]:
                continue
            delta = self.balance - before[0]
            rec["real_delta"] = round(delta, 6)
            if abs(delta - expected) <= 0.1 * amount + 0.01:
                self._event("reconciled", rid=rec["id"], delta=round(delta, 6), expected=round(expected, 6))
                return
        self._event("mismatch", rid=rec["id"], delta=delta, expected=round(expected, 6), currency=before[1])
        if self.armed:
            self._hard_stop(f"balance moved {delta if delta is None else round(delta, 4)} {before[1]} but the "
                            f"order's result was {expected:+.4f}", tech="mismatch")

    def _pnl(self, rec: dict, side: str | None, diluted: bool = False) -> float:
        """$-stake P&L of calling `side` in a settled round: that side's page payout when it was read (the called
        side's, else from the pair logged at call time), else the average; `diluted` books it as a pari-mutuel bet
        of our stake into that side's pool.  (Recomputed on replay: early records logged payout 0.)"""
        if not side or rec.get("tie"):
            return 0.0
        stake = self.cfg.auto_stake_usd
        if side != rec.get("win"):
            return -stake
        pay = rec.get("payout") or 0
        if side != rec.get("call") or pay <= 1:
            ps = rec.get("payouts") or []
            pay = (ps[0] if side == "UP" else ps[1]) if len(ps) == 2 and all(ps) else 0
        if not pay or pay <= 1:
            pay = DEFAULT_PAYOUT
        if diluted:
            pay = _diluted(pay, stake, rec.get("pools"), side)
        return stake * (pay - 1)

    @staticmethod
    def _rec_time(rec: dict) -> float:
        if rec.get("cutoff"):
            return float(rec["cutoff"])
        if rec.get("start"):
            return float(rec["start"])
        try:
            return float(calendar.timegm(time.strptime(rec["utc"], "%Y-%m-%d %H:%M:%S")))
        except (KeyError, ValueError):
            return time.time()                        # a round settling right now

    def _ladder_step(self, lad: dict, rec: dict, side: str) -> None:
        """One paper bet at the progression stake: base x progression^losses-in-a-row, back to base after a win.
        It never gives up unless AUTO_LADDER_MAX_USD is set: then back to base when the next stake would pass it.
        P&L twice: at the payout the page showed (how the MHH dashboard counts it), and pool-adjusted (pnl): our
        stake added to our side's pool lowers our own payout, which matters once the ladder bets are big."""
        base = self.cfg.auto_stake_usd
        stake = base * self.progression ** lad["step"]
        won = side == rec.get("win")
        pay = 1 + self._pnl(rec, side) / base
        lad["pnl_shown"] = round(lad.get("pnl_shown", 0.0) + stake * (pay - 1), 4)
        lad["flat"] = round(lad.get("flat", 0.0) + self._pnl(rec, side), 4)       # the same calls at the base stake
        lad["bets"] = lad.get("bets", 0) + 1
        lad["won"] = lad.get("won", 0) + int(won)
        if won:
            pay = _diluted(pay, stake, rec.get("pools"), side)
        lad["pnl"] = round(lad["pnl"] + stake * (pay - 1), 4)
        lad["staked"] = round(lad["staked"] + stake, 4)
        lad["biggest"] = max(lad["biggest"], round(stake, 2))
        lad["step"] = 0 if won else lad["step"] + 1
        cap = self.cfg.auto_ladder_max_usd
        if cap and base * self.progression ** lad["step"] > cap:
            lad["step"] = 0
            lad["gave_up"] = lad.get("gave_up", 0) + 1

    def _run_step(self, name: str, won: bool, t: float) -> None:
        """Losing runs of one model: how many runs of each length (a run ends at the next win), and the current one."""
        r = self.runs.setdefault(name, {"lengths": {}, "current": 0, "bets": 0, "first": t, "last": t})
        r["bets"] += 1
        r["last"] = t
        if won:
            if r["current"]:
                r["lengths"][r["current"]] = r["lengths"].get(r["current"], 0) + 1
            r["current"] = 0
        else:
            r["current"] += 1

    def _count(self, rec: dict, replay: bool = False) -> None:
        """Add one settled round to the stats (also used to rebuild them from the log)."""
        s = self.stats
        if replay and rec.get("result"):
            rec["pnl"] = self._pnl(rec, rec.get("call"))
        s["rounds"] += 1
        win, result = rec.get("win"), rec.get("result")
        if win:
            self.winners.append("T" if rec.get("tie") else win[0])
        if rec.get("bfx_ret") is not None:
            s["bfx_n"] += 1
            s["bfx_agree"] += int((rec["bfx_ret"] > 0) == (win == "UP"))
        for prefix, field in (("our", "our_move"), ("mix", "mix_move")):   # did our price give bc.game's Up/Down?
            om = rec.get(field)
            if om is None or not win or rec.get("tie") or (prefix == "mix" and rec.get("mix_v") != MIX_VERSION):
                continue                            # (rounds of an earlier mix are not counted for this one)
            if om == 0:
                s[f"{prefix}_flat"] = s.get(f"{prefix}_flat", 0) + 1     # our price didn't move: no answer
                continue
            same = int((om > 0) == (win == "UP"))
            size = "small" if abs(om) <= 1 else "mid" if abs(om) <= 5 else "big"
            for key in (prefix, f"{prefix}_{size}"):
                s[f"{key}_n"] = s.get(f"{key}_n", 0) + 1
                s[f"{key}_ok"] = s.get(f"{key}_ok", 0) + same
        lv = rec.get("live") or {}
        if replay:                                  # live counts are taken at order time while running
            s["live_orders"] += int(bool(lv.get("ok")))
            s["live_failed"] += int(bool(lv.get("error")))
        if win and not rec.get("tie"):              # each model's own record, whichever one makes the calls
            pred, m = rec.get("pred") or {}, rec.get("mhh") or {}
            sign = _sign_call(pred)
            calls = {"time": pred.get("time_call"),
                     "sign": sign,
                     "mhh": m.get("call"),
                     "mhh_hi": m.get("call") if m.get("conf", 0) >= self.flow_model.q75 else None,
                     "agree": sign if sign and sign == m.get("call") else None,
                     "mom": _mom_call(rec.get("feat")),
                     "book": _book_call(rec.get("feat")),
                     "bookmom": _bookmom_call(rec.get("feat")),
                     "momvol": _momvol_call(rec.get("feat")),
                     "revert": _revert_call(rec.get("feat")),
                     "overshoot": _overshoot_call(rec.get("feat")),
                     "prob": ProbModel.side(rec.get("prob")),
                     "gbm": ProbModel.side(rec.get("prob_gbm")),
                     "ens": ProbModel.side(rec.get("prob_ens")),
                     "overshoot_book": _overshoot_call(rec.get("feat"), with_book=True),
                     "pool": (rec.get("pool") or _pool_call(rec.get("payouts")) or {}).get("call")}
            for name, side in calls.items():
                if side:
                    s[f"{name}_called"] = s.get(f"{name}_called", 0) + 1
                    s[f"{name}_correct"] = s.get(f"{name}_correct", 0) + int(side == win)
                    s[f"{name}_pnl"] = round(s.get(f"{name}_pnl", 0.0)
                                             + self._pnl(rec, side, diluted=name == "pool"), 4)
                    if self._rec_time(rec) >= MODEL_REGISTERED.get(name, 0):     # the forward record
                        s[f"{name}_fwd_called"] = s.get(f"{name}_fwd_called", 0) + 1
                        s[f"{name}_fwd_correct"] = s.get(f"{name}_fwd_correct", 0) + int(side == win)
                    self._run_step(name, side == win, self._rec_time(rec))
                    if name in self.ladders and self._rec_time(rec) >= self.since:
                        self._ladder_step(self.ladders[name], rec, side)
        if not result:
            return
        self.recent.append(result[0])
        self.settled.append(rec)
        if result == "TIE":
            s["ties"] += 1
        else:
            s["called"] += 1
            s["correct"] += int(result == "WIN")
            s["pnl"] = round(s["pnl"] + rec["pnl"], 4)
        if lv.get("ok"):
            s["live_won"] += int(result == "WIN")
            s["live_pnl"] = round(s["live_pnl"] + rec.get("live_pnl", rec["pnl"]), 4)

    def _load_log(self) -> None:
        """Rebuild the record from logs/auto_rounds.jsonl, so a restart keeps accuracy, P&L and - above all - the
        run of real losses, today's real attempts and the campaign P&L; then cross-check the orders in
        logs/trades.jsonl and the intents in the events file, so no real order can hide from the counters."""
        self._roll_day()
        today = self.day
        reals: list = []                              # (time, result, live) of every real order
        seen_orders: set = set()
        if self.cfg.auto_log.exists():
            n = 0
            for line in self.cfg.auto_log.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "win" not in rec:
                    continue
                self._count(rec, replay=True)
                lv = rec.get("live") or {}
                if lv.get("ok") or lv.get("unknown"):
                    t = self._rec_time(rec)
                    reals.append((t, rec.get("result"), lv, rec.get("id")))
                    if lv.get("order_id"):
                        seen_orders.add(str(lv["order_id"]))
                    if lv.get("ok"):
                        pnl = rec.get("live_pnl", rec.get("pnl", 0.0))
                        if time.strftime("%Y-%m-%d", time.localtime(t)) == today:
                            self.day_live_pnl += pnl
                        if self.campaign_since and t >= self.campaign_since:
                            self.campaign_pnl += pnl
                if "exp_ret" in (rec.get("pred") or {}):
                    self.history.append(rec["pred"]["exp_ret"])
                n += 1
            if n:
                s = self.stats
                log.info("Record loaded from %s: %d rounds, calls %d/%d, paper P&L %+.2f, live orders %d (P&L %+.2f), "
                         "real P&L today %+.2f (daily limit -$%g)", self.cfg.auto_log.name, n, s["correct"],
                         s["called"], s["pnl"], s["live_orders"], s["live_pnl"], self.day_live_pnl,
                         self.cfg.auto_daily_loss_usd)
        reals.sort(key=lambda x: x[0])
        run = 0
        for t, result, lv, rid in reals:              # the run of real losses, in time order; ties leave it
            if lv.get("unknown"):
                if rid not in self.cleared:
                    run += 1                          # an order that may have been placed counts as lost
            elif result == "LOSS":
                run += 1
            elif result == "WIN":
                run = 0
        self.real_run = run
        self._load_trades(seen_orders, today)
        self._check_events()

    def _load_trades(self, seen_orders: set, today: str) -> None:
        """logs/trades.jsonl holds every attempt at order time (the round log only at settle): today's attempts,
        and any placed order the round log never saw (a crash between the click and the settle)."""
        if not self.cfg.log_file.exists():
            return
        for line in self.cfg.log_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            t = row.get("sent")
            if not t:
                try:
                    t = time.mktime(time.strptime(row["time"], "%Y-%m-%d %H:%M:%S"))
                except (KeyError, ValueError):
                    t = 0
            if t and time.strftime("%Y-%m-%d", time.localtime(t)) == today:
                self.attempts_today += 1
            oid = str(row.get("order_id") or "")
            if row.get("ok") and oid and oid not in seen_orders and oid not in self.cleared:
                log.warning("order %s (%s %s %s) is in trades.jsonl but its round never settled in the round log: "
                            "outcome unknown", oid, row.get("side"), row.get("amount"), row.get("currency"))
                self.unknown.append(oid)
                self.real_run += 1

    def _check_events(self) -> None:
        """An order intent in the events file with no outcome after it: the process died at the click."""
        if not self.cfg.auto_events.exists():
            return
        open_intents: dict = {}
        for line in self.cfg.auto_events.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            rid = ev.get("rid")
            if ev.get("event") == "order_intent":
                open_intents[rid] = ev
            elif ev.get("event") in ("order_sent", "order_skipped", "order_rejected", "order_unknown"):
                open_intents.pop(rid, None)
        for rid, ev in open_intents.items():
            if rid in self.cleared or rid in self.unknown:
                continue
            log.warning("round %s: an order was about to be clicked when the bot died (%s); outcome unknown",
                        rid, ev.get("utc"))
            self.unknown.append(rid)
            self.real_run += 1

    def _record(self, rec: dict) -> None:
        keep = {k: v for k, v in rec.items() if k not in ("decided", "settled", "hawkes_task")}
        keep["utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        keep["v"] = LOG_VERSION
        self.cfg.auto_log.parent.mkdir(parents=True, exist_ok=True)
        with self.cfg.auto_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(keep, default=float) + "\n")

    # ---------- decision panel ----------

    def panel_html(self) -> str:
        """The decision panel (bc.game's left bar and the local dashboard), as cards like the MHH dashboard:
        ROUND (countdown, prices, pools, last winners), CALL (this round's call and the last results),
        BET · P&L (next stake, payouts, ladder P&L, losses in a row) and MODELS."""
        e = html.escape
        r, s = self.round or {}, self.stats
        now = self.bc_now() * 1000                             # bc.game's round times are its server's clock
        rid = int(r.get("id") or 0)
        rec = self.recs.get(rid, {})
        status = r.get("status")
        cut, end = r.get("tradeCutoffTime"), r.get("priceEndTime")
        mode = ('<span class="mode live">LIVE</span>' if self.armed else
                '<span class="mode paused">STOPPED</span>' if self.state == "STOPPED" else
                '<span class="mode paper">PAPER</span>' if self.on else '<span class="mode off">OFF</span>')

        def row(k, v):
            return f'<div class="row"><span>{k}</span><span>{v}</span></div>'

        def usd(v):
            return f'{"+" if v >= 0 else "-"}${abs(v):,.2f}'

        # ---- ROUND: countdown, prices, pools, the last rounds' winners ----
        if status == BETTING and cut:
            left = max(0.0, (cut - now) / 1000)
            opened = self._open_t * 1000 if self._open_rid == rid and self._open_t else cut - 15000
            frac = (now - opened) / max(1.0, cut - opened)
            big, small, phase, cls = f"{left:.1f}s", "until bets close", "BETTING OPEN", "bet"
        elif status == 1003 and end:
            left = max(0.0, (end - now) / 1000)
            frac = 1 - left / 5
            big, small, phase, cls = f"{left:.1f}s", "5 s window running", "WINDOW", "win"
        else:
            frac, big, small, cls = 1.0, "&ndash;", PHASE.get(status, "waiting"), ""
            phase = small.upper()
        frac = min(1.0, max(0.0, frac))
        bc_now = self.bc_ticks[-1][1] if self.bc_ticks else None
        ours = self.our.now_value()
        start = float(r.get("startPrice") or 0)
        try:
            up_pool = float((self.pools or {}).get("upPoolAmount") or 0)
            dn_pool = float((self.pools or {}).get("downPoolAmount") or 0)
        except (TypeError, ValueError):
            up_pool = dn_pool = 0.0
        pay_up = 1 + PAY_K * dn_pool / up_pool if up_pool > 0 else None
        pay_dn = 1 + PAY_K * up_pool / dn_pool if dn_pool > 0 else None
        winners = "".join(f'<i class="{w}">{ {"U": "&#9650;", "D": "&#9660;"}.get(w, "=") }</i>'
                          for w in reversed(self.winners))
        round_card = (
            f'<div class="c {cls}"><div class="ct"><span>ROUND</span><span class="ph">{phase}</span></div>'
            f'<div class="big">{big}<small>{small}</small></div><div class="pbar"><i style="width:{frac * 100:.0f}%"></i></div>'
            + row("bc.game", f"{bc_now:,.2f}" if bc_now else "&ndash;")
            + row("our price", f"{ours:,.2f}" if ours else "&ndash;")
            + row("start", f"{start:,.2f}" if start and status != BETTING else "&ndash;")
            + row("pool UP / DN", f"${up_pool:,.0f} / ${dn_pool:,.0f}" if up_pool or dn_pool else "&ndash;")
            + (f'<div class="sq" title="last rounds, newest left">{winners}</div>' if winners else "") + '</div>')

        # ---- the call: what the BET card says before / without a call ----
        call, p, m = rec.get("call"), rec.get("pred") or {}, rec.get("mhh") or {}
        mhh = self.model in ("mhh", "agree")                      # these call at MHH's time, 2 s before cutoff
        lead_s = (self.flow_model.lead_ms if mhh else self.cfg.auto_lead_ms) / 1000
        if not self.on:
            sub = "auto is off"
        elif call:
            if self.model == "agree":
                sub = f"both say {call}: Sign P(up) {p.get('sign_p_up', 0):.2f} &middot; MHH P(up) {m.get('p_up', 0):.3f}"
            elif mhh:
                sub = f"MHH P(up) {m.get('p_up', 0):.3f} &middot; conf {m.get('conf', 0):.3f}"
            elif self.model == "mom":
                sub = f"our price moved {((rec.get('feat') or {}).get('our_mom5') or 0):+.2f} $ in the last 5 s"
            elif self.model == "book":
                sub = f"Binance book imbalance {((rec.get('feat') or {}).get('obi20') or 0):+.2f} (calls at |x| &ge; {BOOK_MIN})"
            elif self.model == "bookmom":
                f_ = rec.get("feat") or {}
                sub = (f"book {(f_.get('obi20') or 0):+.2f} &middot; price 5 s {(f_.get('our_mom5') or 0):+.2f} $ "
                       f"(both the same way, book &ge; {BOOK_MIN})")
            elif self.model in ("prob", "gbm", "ens"):
                key = {"prob": "prob", "gbm": "prob_gbm", "ens": "prob_ens"}[self.model]
                mg = ProbModel.side.__defaults__[0]
                sub = (f"P(UP) {rec[key]:.2f} (calls outside {0.5 - mg:.2f}-{0.5 + mg:.2f})"
                       if isinstance(rec.get(key), float) else "P(UP) models " + ("ready" if self.prob.model else "not fitted yet"))
            else:
                sub = f"HawkesSign P(up) {p.get('sign_p_up', 0):.2f}"
        elif mhh and self.flow.arrays(31) is None:
            sub = "warming up: waiting for 31 s of Binance flow"
        elif self.model == "sign" and not self.pred.ready:
            sub = f"warming up: order-book data {self.pred.data_span():.0f} / 300 s"
        elif status == BETTING:
            sub = f"call {lead_s:g} s before bets close"
        else:
            sub = "no call this round"

        # ---- OB · top 20 levels: Binance BTCUSDT book imbalance, as on the MHH dashboard ----
        bk = self.book
        v = bk.obi20
        side_cls = "up" if v >= 0.2 else "down" if v <= -0.2 else ""
        color = "#22d27a" if v >= 0.2 else "#ff5468" if v <= -0.2 else "#c3ccd8"
        words = ("balanced book" if abs(v) < 0.2 else ("strongly " if abs(v) >= 0.6 else "")
                 + ("bid-heavy &middot; buyers stacked" if v > 0 else "ask-heavy &middot; sellers stacked"))
        fill = (f"left:50%;width:{v * 50:.1f}%;background:#22d27a" if v >= 0 else
                f"left:{50 + v * 50:.1f}%;width:{-v * 50:.1f}%;background:#ff5468")
        hist = [h for _, h in list(bk.hist)[-120:]]
        sw, sh = 212, 40
        spark = ""
        if len(hist) >= 2:
            pts = " ".join(f"{i * sw / (len(hist) - 1):.1f},{sh / 2 - h * sh / 2 * 0.92:.1f}" for i, h in enumerate(hist))
            spark = (f'<svg class="spark" viewBox="0 0 {sw} {sh}" preserveAspectRatio="none">'
                     f'<line x1="0" y1="{sh / 2}" x2="{sw}" y2="{sh / 2}" stroke="#5b6573" stroke-dasharray="3 3"/>'
                     f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')
        ob_card = (
            f'<div class="c {"ob" + side_cls if side_cls else ""}"><div class="ct"><span>OB &middot; TOP 20 LEVELS</span>'
            f'<span class="mute">{"live" if bk.connected else "reconnecting"}</span></div>'
            f'<div class="big {side_cls}">{v * 100:+.1f}%</div><div class="obw {side_cls}">{words}</div>'
            f'<div class="obbar"><i style="{fill}"></i></div>'
            f'<div class="two mute sm"><span>asks -100%</span><span>0</span><span>bids +100%</span></div>{spark}'
            f'<div class="two mute sm"><span>L1 {bk.obi1:+.2f} &middot; L1-5 {bk.obi5:+.2f}</span><span>last 2 min</span></div>'
            '</div>')

        # ---- BET, like the MHH dashboard: "next $X" until the call, then "UP $X" / "DN $X" with what it pays,
        # then WON / LOST; the next stake follows the progression (x2: doubles after each loss, back to base on a win)
        lad = self.ladders.get(self.model) or {}
        base, prog = self.cfg.auto_stake_usd, self.progression
        step = self.real_run if self.armed else lad.get("step", 0)
        run_lost = base * ((prog ** step - 1) / (prog - 1) if prog != 1 else step)
        plabel = {1.0: "flat", 1.5: "x1.5", 2.0: "double"}.get(prog, f"x{prog:g}")
        runs = self.runs.get(self.model) or {}
        longest = max([*map(int, runs.get("lengths", {})), runs.get("current", 0)], default=0)
        bets, won = lad.get("bets", 0), lad.get("won", 0)
        pnl = lad.get("pnl", 0.0)
        since = time.strftime("%H:%M", time.localtime(self.since)) if self.since else "start"
        strip = "".join(f'<i class="{x}"></i>' for x in reversed(self.recent))

        def amt(v):
            return f"{v:,.2f}".rstrip("0").rstrip(".")

        lv = rec.get("live") or {}
        result = rec.get("result")
        if call:
            stake = lv.get("stake") if lv.get("ok") else rec.get("stake", self.paper_stake())
            pay = rec.get("payout") if (rec.get("payout") or 0) > 1 else (pay_up if call == "UP" else pay_dn)
            gain = stake * (pay - 1) if pay else None
            bet_cls = "betup" if call == "UP" else "betdn"
            arrow = "&#9650; UP" if call == "UP" else "&#9660; DN"
            bet_big = f'<div class="big {"up" if call == "UP" else "down"}">{arrow} ${amt(stake)}</div>'
            state = {"WIN": " &middot; WON", "LOSS": " &middot; LOST", "TIE": " &middot; TIE (stake back)"}.get(
                result, " &middot; waiting for the result")
            bet_line = (f'{"Up" if call == "UP" else "Down"} Wins {f"{pay * 100:.0f}%" if pay else "&ndash;"}{state}'
                        + (' <span class="tag">REAL</span>' if lv.get("ok") else
                           ' <span class="tag">real order failed</span>' if lv.get("error") else ""))
            if result == "WIN" and gain is not None:
                note = f"booked {pay * 100:.0f}% on ${amt(stake)} &rarr; {usd(gain)}"
            elif result == "LOSS":
                note = f"lost {usd(-stake)}"
            elif gain is not None:
                note = f"our {call} ${amt(stake)} pays {usd(gain)} if it wins"
            else:
                note = ""
            note += f" &middot; {sub}" if note else sub
        elif rec.get("skip") and self.on:
            bet_cls, bet_big = "", '<div class="big mute">NO BET</div>'
            bet_line, note = e(rec["skip"]), ""
        else:
            nxt = self.stake() if self.live else self.paper_stake()
            bet_cls, bet_big = "", f'<div class="big mute">next ${amt(nxt)}</div>'
            bet_line, note = sub, ""
        last = self.settled[-1] if self.settled else None
        if last is not None and last is not rec and last.get("call") and last.get("win"):
            llv = last.get("live") or {}
            lstake = llv.get("stake") if llv.get("ok") else last.get("stake", base)
            lpnl = last.get("live_pnl") if llv.get("ok") else last.get("pnl", 0.0) * lstake / base
            last_txt = (f'last: {last["call"]} {last.get("result", "")} ${amt(lstake)} {usd(lpnl or 0.0)}'
                        + (" (real)" if llv.get("ok") else ""))
        else:
            last_txt = ""

        def pay_box(side, pay):
            cls_ = ("ours" if call == side else "dim") if call else ""
            name, color_ = ("Up Wins", "up") if side == "UP" else ("Down Wins", "down")
            return (f'<div class="{cls_} {color_}b"><span class="{color_}">{name}</span>'
                    f'<b class="{color_}">{f"{pay * 100:.0f}%" if pay else "&ndash;"}</b></div>')

        bet_card = (
            f'<div class="c {bet_cls}"><div class="ct"><span>BET &middot; {e(self.MODELS.get(self.model, self.model))}'
            f'</span><span class="ph">{plabel} &middot; step {step}</span></div>'
            f'{bet_big}<div class="bl">{bet_line}</div>'
            f'<div class="pay">{pay_box("UP", pay_up)}{pay_box("DOWN", pay_dn)}</div>'
            + (f'<div class="note">{note}</div>' if note else "")
            + (f'<div class="mute">{last_txt}</div>' if last_txt else "") + '</div>')

        # ---- LAST RESULTS: our call -> the winner, WIN / LOSS, money (no times) ----
        def result_row(x):
            xlv = x.get("live") or {}
            cls_ = {"WIN": "w", "LOSS": "l"}.get(x["result"], "t")
            side = "&#9650;UP" if x["call"] == "UP" else "&#9660;DN"
            winner = {"UP": "UP", "DOWN": "DN"}.get(x["win"], x["win"])
            if xlv.get("ok"):
                money = f'{x.get("live_pnl", x.get("pnl", 0.0)):+.2f} <span class="tag">REAL ${xlv.get("stake", 0):g}</span>'
            else:                                 # paper: at the stake the ladder bet that round
                xstake = x.get("stake", base)
                money = (f'{x.get("pnl", 0.0) * xstake / base:+.2f} <span class="tag ptag">${xstake:g}</span>'
                         + (' <span class="tag">failed</span>' if xlv.get("error") else ""))
            return (f'<div class="res {cls_}"><span class="sd">{side}&rarr;{winner}</span><b>{x["result"]}</b>'
                    f'<span class="m">{money}</span></div>')

        settled = [x for x in reversed(self.settled) if x.get("call") and x.get("win")][:5]
        results_card = (
            '<div class="c"><div class="ct"><span>LAST RESULTS</span><span class="mute">our call &rarr; winner</span></div>'
            + "".join(result_row(x) for x in settled) + '</div>') if settled else ""

        # ---- P&L: the progression's paper P&L, bets, losses in a row, real money ----
        nxt_after = self.stake() if self.live else self.paper_stake()
        c = self.cfg
        real = (f"LIVE &middot; real next ${self.stake():g}" if self.armed else
                f"STOPPED: {e(self.stop_reason or '')}" if self.state == "STOPPED" else "real orders off")
        clk = self.clock.dict()
        safety = (f'<div class="mute">real stake ${self.stake():g} &middot; '
                  + ("compensate " + "-".join(f"{x * c.auto_stake_usd:g}" for x in c.auto_real_ladder) + " ON"
                     if self.ladder_on() else "flat" + (" (ladder waits for the gate)" if len(c.auto_real_ladder) > 1 else ""))
                  + '</div>'
                  f'<div class="mute">real losses in a row <b>{self.real_run}</b> / {c.auto_max_real_run} &middot; '
                  f'orders today {self.attempts_today} / {c.auto_max_real_bets_day} &middot; '
                  f'budget left ${self.budget_left():.2f}</div>'
                  f'<div class="mute">campaign {usd(self.campaign_pnl)} (stop -${c.auto_campaign_loss_usd:g}) &middot; '
                  f'balance {f"${self.balance_usd:,.2f}" if self.balance_usd is not None else "&ndash;"} &middot; '
                  f'clock {"bc " + f"{clk["offset"]:+.2f}s" if clk["ok"] else "exchange (" + e(clk["why"] or "?") + ")"}</div>'
                  + (f'<div class="mute">shadow: would-bet {self.shadow_attempts_today} today &middot; run {self.shadow_run} '
                     f'&middot; longest {self.shadow_longest_today} &middot; stops {self.shadow_stops_today}</div>'
                     if c.auto_shadow and not self.armed else ""))
        pnl_card = (
            f'<div class="c"><div class="ct"><span>P&amp;L &middot; {plabel}</span><span class="mute">since {since}</span></div>'
            f'<div class="two mute"><span>paper P&amp;L</span><span>bets &middot; won</span></div>'
            f'<div class="two"><b class="pnl {"pos" if pnl >= 0 else "neg"}">{usd(pnl)}</b>'
            f'<b class="pnl">{bets} &middot; {won}</b></div>'
            f'<div class="mute">at page payouts {usd(lad.get("pnl_shown", 0.0))} &middot; flat $1 {usd(lad.get("flat", 0.0))}'
            + (f' &middot; won {won / bets:.1%}' if bets else "") + '</div>'
            f'<div class="ph" style="margin-top:3px">next bet ${amt(nxt_after)} &middot; losses in a row {step} &middot; '
            f'run lost ${amt(run_lost)} &middot; longest {longest}</div>'
            f'<div class="mute">{real} &middot; real today {usd(self.day_live_pnl)} (limit -${self.cfg.auto_daily_loss_usd:g})</div>'
            f'{safety}'
            f'<div class="sq" title="our last calls, newest left">{strip}</div></div>')

        # ---- MODELS: every model's record (flat $1 per call, all logged rounds) ----
        def model_row(name, label):
            n, ok = s.get(f"{name}_called", 0), s.get(f"{name}_correct", 0)
            mark = " &#9664;" if self.model == name else ""
            return row(label + mark, f"{ok / n:.1%} of {n} &middot; {s.get(f'{name}_pnl', 0.0):+.0f}" if n else "&ndash;")

        mix_n, mix_ok = s.get("mix_n", 0), s.get("mix_ok", 0)
        models_card = (
            '<div class="c"><div class="ct"><span>MODELS</span><span class="mute">right &middot; flat $1 P&amp;L</span></div>'
            + model_row("sign", "HawkesSign") + model_row("time", "HawkesTime") + model_row("mhh", "MHH flow")
            + model_row("agree", "Sign+MHH agree") + model_row("mom", "Momentum 5s") + model_row("book", "Strong book") + model_row("bookmom", "Book + trend") + model_row("momvol", "Mom, volatile") + model_row("revert", "Quiet reversal") + model_row("overshoot", "Overshoot") + model_row("overshoot_book", "Overshoot+book") + model_row("prob", "P(UP) model") + model_row("gbm", "P(UP) trees") + model_row("ens", "P(UP) ensemble")
            + row("our price wrong", f"{(mix_n - mix_ok) / mix_n:.1%} of {mix_n}" if mix_n else "&ndash;") + '</div>')
        return (f'{PANEL_CSS}<div class="p"><div class="hd"><span>BC BOT &middot; ROUND #{rid or "&ndash;"}</span>{mode}</div>'
                f'{round_card}{ob_card}{bet_card}{results_card}{pnl_card}{models_card}</div>')

    def curve_data(self) -> dict:
        """What the page needs to draw our price on bc.game's chart: our last 75 s, the last ticks (the page finds
        the chart's last tick among them by price), and the time of the newest tick."""
        now = self.bc_now()
        ticks = [[t, p] for t, p in list(self.bc_ticks)[-40:]]
        return {"on": True, "our": self.our.series(now - 75, now), "ticks": ticks,
                "t_last": ticks[-1][0] if ticks else now}

    def prices(self, window: float = 60.0, exchanges: bool = False) -> dict:
        """The last `window` seconds of bc.game's BTC price and of our price (the weighted, delayed exchange mix),
        the rounds in view with their results, and how often our price got the Up/Down wrong.  With `exchanges`
        also each exchange's own curve (delayed and moved to bc.game's level, as the mix uses it)."""
        now = self.bc_now()                           # bc.game's clock (from its ticks); this PC's clock may be off
        cut = now - window

        def last(q):
            return [[round(t, 3), p] for t, p in list(q) if t >= cut]

        r = self.round or {}
        return {
            "now": now,
            "bc": last(self.bc_ticks),
            "our": self.our.series(cut, now),         # our price: the weighted, delayed exchange mix (ourprice.py)
            "exchanges": self.our.exchange_series(cut, now) if exchanges else None,
            "mix": {k: {"label": self.ex_feeds[k].name, "weight": w, "lag": lag,
                        "gap": round(self.our.gaps().get(k, float("nan")), 2), "live": bool(self.ex_feeds[k].connected)}
                    for k, (w, lag) in MIX.items()},
            "clock": round(self.our.clock(), 3),      # this PC minus the exchanges' clocks
            "clock_bc": self.clock.dict(),            # this PC minus bc.game's clock, and whether it is in use
            "round": {"cutoff": (r.get("tradeCutoffTime") or 0) / 1000, "start": (r.get("priceStartTime") or 0) / 1000,
                      "end": (r.get("priceEndTime") or 0) / 1000, "start_price": float(r.get("startPrice") or 0),
                      "status": PHASE.get(r.get("status"), "")},
            # every round whose window is in view, with its result once settled (the chart colors it)
            "rounds": [{"start": rec["start"], "end": rec["end"], "win": rec.get("win"), "tie": rec.get("tie", False),
                        "start_price": rec.get("start_price"), "end_price": rec.get("end_price"),
                        "our_move": rec.get("mix_move", rec.get("our_move"))}
                       for rec in list(self.recs.values()) if rec.get("start") and rec.get("end", 0) >= cut],
            # how often our price gave a different Up/Down from bc.game (all logged rounds)
            "our_vs_bc": {f"{p}{k}": self.stats.get(f"{p}{k}", 0) for p in ("our", "mix") for k in
                          ("_n", "_ok", "_flat", "_small_n", "_small_ok", "_mid_n", "_mid_ok", "_big_n", "_big_ok")},
        }

    def _ladder_text(self) -> str:
        lad = self.ladders.get(self.model)
        if not lad or not lad["staked"]:
            return "&ndash;"
        return (f"{lad['pnl']:+.2f} (page payouts {lad.get('pnl_shown', 0):+.2f}) &middot; max bet {lad['biggest']:g}"
                + (f" &middot; gave up {lad['gave_up']}" if lad.get("gave_up") else ""))

    # ---------- status ----------

    def summary(self) -> dict:
        s, c = self.stats, self.cfg
        rec = self.recs.get(int((self.round or {}).get("id") or 0)) or {}
        return {
            "on": self.on, "live": self.armed, "live_wanted": self.armed,
            # ---- real money: the state machine and every counter it is judged by ----
            "state": self.state, "stop_reason": self.stop_reason, "tech_stop": self.tech_stop,
            "armed_at": self.armed_at or None, "rearm_grant": self.rearm_grant, "rearms": self.rearms,
            "real_run": self.real_run, "max_real_run": c.auto_max_real_run,
            "real_run_longest_today": self.real_run_longest_today,
            "attempts_today": self.attempts_today, "max_real_bets_day": c.auto_max_real_bets_day,
            "today_live_pnl": round(self.day_live_pnl, 4), "daily_loss_usd": c.auto_daily_loss_usd,
            "budget_left": round(self.budget_left(), 4),
            "campaign_pnl": round(self.campaign_pnl, 4), "campaign_since": self.campaign_since or None,
            "campaign_loss_usd": c.auto_campaign_loss_usd,
            "balance": self.balance, "balance_currency": self.balance_currency, "balance_usd": self.balance_usd,
            "balance_age": round(time.time() - self.balance_at, 1) if self.balance_at else None,
            "balance_account": self.balance_account, "dialog": self.dialog,
            "clock": self.clock.dict(), "clock_src": self.clock_src(),
            "pending": list(self.pending_real), "unknown": self.unknown, "last_order": self._last_order,
            "events": list(self.events), "gate": self.gate_status(), "break_even": self.BREAK_EVEN,
            "live_allowed": c.live_allowed, "tos_risk_accepted": c.tos_risk_accepted,
            "real_ladder": list(c.auto_real_ladder), "ladder_on": self.ladder_on(), "next_real_stake": self.stake(),
            "shadow": {"on": c.auto_shadow, "run": self.shadow_run, "attempts_today": self.shadow_attempts_today,
                       "longest_today": self.shadow_longest_today, "stops_today": self.shadow_stops_today,
                       "last": rec.get("shadow")},
            "dq": (rec.get("feat") or {}).get("dq"),
            "progression": self.progression, "next_stake": self.stake(), "ladders": self.ladders,
            "since": self.since, "max_usd": c.max_usd, "ladder_cap": c.auto_ladder_max_usd or None,
            "model": self.model,
            # each model's losing runs over all logged rounds: {"lengths": {run length: how many}, "current", ...}
            "runs": self.runs,
            "feed": {"connected": self.feed.connected, "msgs": self.feed.msgs, "data_s": round(self.pred.data_span()),
                     "events": len(self.feed.ev_time), "mid": self.feed.mid},
            "models_ready": self.pred.ready, "models": self.pred.info,
            "accuracy": round(s["correct"] / s["called"], 4) if s["called"] else None,
            "sign_accuracy": round(s["sign_correct"] / s["sign_called"], 4) if s["sign_called"] else None,
            "bitfinex_agrees_with_bc": round(s["bfx_agree"] / s["bfx_n"], 4) if s["bfx_n"] else None,
            "records": {name: {"called": s.get(f"{name}_called", 0), "correct": s.get(f"{name}_correct", 0),
                               "pnl": s.get(f"{name}_pnl", 0.0)}
                        for name in ("time", "sign", "mhh", "mhh_hi", "agree", "mom", "book", "bookmom", "momvol", "revert", "overshoot", "overshoot_book", "prob", "gbm", "ens", "pool")},
            "prob_model": self.prob.info(),
            "stats": s, "last": self.last,
        }

    def line(self) -> str:
        d = self.summary()
        acc = f"{d['accuracy']:.1%}" if d["accuracy"] is not None else "-"
        sacc = f"{d['sign_accuracy']:.1%}" if d["sign_accuracy"] is not None else "-"
        bfx = f"{d['bitfinex_agrees_with_bc']:.1%}" if d["bitfinex_agrees_with_bc"] is not None else "-"
        s = d["stats"]
        return (f"auto {'ON' if d['on'] else 'off'} ({d['model']}) | real orders {d['state'].lower()}"
                f"{' (' + d['stop_reason'] + ')' if d['stop_reason'] else ''} | book data "
                f"{d['feed']['data_s']}s, models {'ready' if d['models_ready'] else 'warming up'} | calls "
                f"{s['correct']}/{s['called']} = {acc} (HawkesSign {sacc}), ties {s['ties']}, skipped {s['skipped']} | "
                f"paper P&L {s['pnl']:+.2f} | Bitfinex agrees with bc.game {bfx} | real orders {s['live_orders']} "
                f"(failed {s['live_failed']}) P&L {s['live_pnl']:+.2f}")

    def live_line(self) -> str:
        """One line on the real-money state, for 'live' in the terminal."""
        c, clk = self.cfg, self.clock.dict()
        gate = self.gate_status()
        return (f"real orders {self.state}{' (' + self.stop_reason + ')' if self.stop_reason else ''}"
                f"{' - needs: live clear ' + self.tech_stop if self.tech_stop else ''} | losses in a row {self.real_run}"
                f"/{c.auto_max_real_run}{' (one order per re-arm)' if self.real_run >= c.auto_max_real_run else ''} | "
                f"orders today {self.attempts_today}/{c.auto_max_real_bets_day}, real today {self.day_live_pnl:+.2f}, "
                f"budget left ${self.budget_left():.2f} | campaign {self.campaign_pnl:+.2f} (stop -${c.auto_campaign_loss_usd:g}) | "
                f"balance {f'${self.balance_usd:.2f}' if self.balance_usd is not None else '?'} "
                f"({self.balance_account or '?'}) | clock {'bc ' + f'{clk["offset"]:+.2f}s' if clk['ok'] else 'exchange: ' + str(clk['why'])} | "
                f"gate {'OPEN' if gate['open'] else 'closed'}: {gate['text']} | "
                f"LIVE_ALLOWED={int(c.live_allowed)} TOS_RISK_ACCEPTED={int(c.tos_risk_accepted)} shadow={int(c.auto_shadow)}")
