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
import dataclasses
import calendar
import html
import json
import os
import logging
import math
import time
import zlib
from array import array
from collections import deque

import aiohttp
import numpy as np

from .clock import BcClock
from .config import Config
from .flow import BinanceBook, BinanceFlow
from .lab_display import read_status as read_lab_status, panel_card as lab_panel_card
from .lob import LobFeed
from .obfeat import LIFE_CAP_S, OB_ANSWER, OB_FITTED, OB_MODELS, STEADY_AGE_S, STEADY_MIN, side as _ob_side
from .prob import ProbModel, REFIT_S as PROB_REFIT_S
from .ourprice import MIN_WEIGHT, MIX, MIX_VERSION, STALE_SNAPSHOT, OurPrice, make_feeds, read_fair_recipe
from .predictor import HawkesPredictor
from .trader import OrderUnknown, TooLate, TradeError, Trader

log = logging.getLogger(__name__)

TICKER = "/contest/BTC/USD/5/ticker"
PRICE = "/kline/BTC-USD/ticker"
POOLS = "/contest/btc_5/amountPool"     # the round's Up and Down pools ($), as the MHH bot reads them
PAY_K = 0.9415                 # the page's payout = 1 + 0.9415 x other side's pool / own side's pool (579 rounds)
POOL_MIN_PAY = 2.0             # the "pool" model takes the thin side when the page pays more than this
THIN_MIN_PAY = 2.10            # the "thin" model bets the side the page pays at least this on (_thin_call)
POOL_FEE_K = 0.95              # what bc.game pays on a win: 1 + 0.95 x other pool / own pool at the round's FINAL pools,
                               # our stake inside our pool (9 real wins by balance move, within 0.007).  PAY_K above is
                               # the same with the $1 the page has typed in: 0.95 x 110 / 111
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
    // each round's 5 s window as on the dashboard: start line (dashed) and end line (solid), gray while it runs,
    // green / red once bc.game settled it, with the result at the end line
    const vline = (x, w) => { g.lineWidth = w; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, r.height); g.stroke(); };
    g.font = 'bold 11px Consolas, monospace'; g.textAlign = 'left';
    for (const rd of (d.rounds || [])) {
      if (!rd.start || !rd.end || rd.end < tLast - 61 || rd.start > tLast + 29) continue;
      const xs = Math.max(0, X(rd.start)), xe = Math.min(r.width, X(rd.end));
      const rgb = rd.tie ? '148,163,184' : rd.win === 'UP' ? '34,210,122' : rd.win === 'DOWN' ? '255,84,104' : null;
      g.fillStyle = rgb ? `rgba(${rgb},0.16)` : 'rgba(148,163,184,0.14)';
      g.fillRect(xs, 0, Math.max(0, xe - xs), r.height);
      g.strokeStyle = rgb ? `rgb(${rgb})` : '#cbd5e1';
      if (X(rd.start) >= 0 && X(rd.start) <= r.width) {
        g.setLineDash([4, 3]); vline(X(rd.start), 1.2); g.setLineDash([]);
        g.fillStyle = rgb ? `rgb(${rgb})` : '#cbd5e1'; g.fillText('start', X(rd.start) + 3, 40); }
      if (X(rd.end) >= 0 && X(rd.end) <= r.width) {
        vline(X(rd.end), rgb ? 2.5 : 1.5);
        g.fillStyle = rgb ? `rgb(${rgb})` : '#cbd5e1';
        g.fillText(rd.tie ? 'TIE' : rd.win === 'UP' ? '▲ UP' : rd.win === 'DOWN' ? '▼ DOWN' : 'end', X(rd.end) + 4, 40); }
    }
    const pts = (d.our || []).filter(q => q[0] >= tLast - 61 && q[0] <= tLast + 0.6).map(q => [X(q[0]), Y(q[1])]);
    if (pts.length < 2) return;
    for (const [w, c] of [[4, 'rgba(0,0,0,0.55)'], [1.7, '#22d3ee']]) {
      g.strokeStyle = c; g.lineWidth = w; g.lineJoin = 'round'; g.beginPath();
      pts.forEach((q, i) => i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])); g.stroke(); }
    const last = pts[pts.length - 1], pOur = d.our[d.our.length - 1][1];
    g.fillStyle = '#22d3ee'; g.beginPath(); g.arc(last[0], last[1], 3, 0, 7); g.fill();
    g.font = 'bold 12px Consolas, monospace'; g.textAlign = 'right';
    const pBc = d.ticks.length ? d.ticks[d.ticks.length - 1][1] : null;   // bc.game's exact last tick, not pixels
    const txt = 'our ' + pOur.toFixed(2) + (pBc != null ? '  (bc ' + (pBc - pOur >= 0 ? '+' : '') + (pBc - pOur).toFixed(2) + ')' : '');
    const tw = g.measureText(txt).width, ly = Math.min(r.height - 8, Math.max(14, last[1] + (last[1] < r.height / 2 ? 18 : -10)));
    g.fillStyle = 'rgba(11,14,19,0.8)'; g.fillRect(last[0] - 70 - tw - 4, ly - 11, tw + 8, 15);
    g.fillStyle = '#22d3ee'; g.fillText(txt, last[0] - 70, ly);
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


def _thin_call(feat: dict | None) -> str | None:
    """The "thin" model (2026-11-11, paper): no prediction.  The game is pari-mutuel, so the side with the smaller
    pool pays more, and on the log it wins about as often as the other (49.7% of 10,654 rounds).  A flat $1 on the
    side the page pays at least THIN_MIN_PAY on at the decision returned +14.4% per $1 on 470 bets, paid at the final
    pools (a fair coin at those payouts: +9.9%; under 2.10 it does not pay: 2.00-2.10 lost 3.9%).  Small flat stakes
    only: our own stake goes into the thin pool and lowers its payout ($32 into a $105 pool: x2.2 -> x1.9), so a
    doubling series keeps about a third of the return per $1 (+3 to +4% at a fair coin) and ends 470 bets with a
    loss about one time in three (flat: 1 in 40).  None when the snapshot has no pools or neither side pays that
    much, and when the pools in it are not the round's own (pools_prev: the round had no pool frame yet, so
    _features took the last frame, usually the previous round's).  A record without that key counts as it did."""
    if not feat or feat.get("pools_prev"):
        return None
    up, dn = feat.get("pay_up"), feat.get("pay_dn")
    if not up or not dn or up == dn:
        return None
    side, pay = ("UP", up) if up > dn else ("DOWN", dn)
    return side if pay >= THIN_MIN_PAY else None


def _pool_pay(rec: dict, side: str, stake: float) -> float | None:
    """What bc.game pays on `side` for `stake` $ of ours: 1 + POOL_FEE_K x other pool / (own pool + stake) at the
    round's final pools (a paper bet is not in them, so it is added; a real order of ours on that side is, so it
    is taken out first).  An empty other side pays the stake back (x1.0).  None when the round's pools were not seen
    or our own side's pool is empty in them."""
    pools = rec.get("pools")
    try:
        if not pools or len(pools) != 2 or not pools[0] >= 0 or not pools[1] >= 0:
            return None
        own, other = (pools[0], pools[1]) if side == "UP" else (pools[1], pools[0])
        lv = rec.get("live") or {}
        if lv.get("ok") and rec.get("call") == side:
            own = max(own - (lv.get("stake") or 0.0), 0.0)
        if not own > 0:
            return None
        return 1 + POOL_FEE_K * other / (own + stake)
    except TypeError:                               # a pool that is not a number: not seen
        return None


# A technical stop in the owner's words: what happened, and what to look at on bc.game before clearing it on the page.
TECH_STOPS = {
    "currency": ("the wallet on bc.game changed to another currency while real orders were on",
                 "that the wallet selected on bc.game is the one you want to bet with"),
    "unknown": ("bc.game did not confirm an order, so its result is not known",
                "in the bet history on bc.game whether that order was placed and whether it won"),
    "rejected": ("bc.game refused several orders within 10 minutes",
                 "on bc.game that a bet can be placed by hand (balance, wallet, any message on the page)"),
    "misbooked": ("an order was booked in another round than the one it was sent for",
                  "that order in the bet history on bc.game"),
    "mismatch": ("the balance did not move as the order's result said",
                 "the balance and the last bets in the bet history on bc.game"),
}
# The bot's own Hawkes forecast (Bitfinex spot, predictor.py) is pure-Python simulation in a thread: for about 0.7 s
# it holds the interpreter and this loop gets little time.  Started at the decision, that is exactly when a real order
# has to go out.  Measured 2026-11-17 on 2,570 rounds with an order-book model calling: the order-book service answers
# in 0.10 s (asked from outside the bot), but inside the bot the answer was there after 0.38 s (median), 0.75 s (90%),
# and not within 0.8 s in 5.6% of the rounds; of the first 15 rounds LIVE on XGBoost only 6 got their real order (3 no
# answer, 1 too late to click, 1 refused by bc.game as closed and the round after it, 2 stale bc.game ticks, 1 the
# arming round).  So unless this forecast makes the call itself (Gap fade + Hawkes), it starts only this long after
# the last moment for an order and the late snapshot are over, and is made as of the decision (HawkesPredictor.predict
# `at`): the logged forecast is the same one, computed later.
HAWKES_AFTER_S = 0.15
# The balance is read right before a real order's click.  That read also asked bc.game for its price table whenever
# the copy was older than 60 s, i.e. in every second round, and the answer takes 0.6-1.6 s: of 34 real attempts on
# 2026-11-17 five came "too late to click" for it (1.30-1.64 s) and eleven went out 0.6-0.8 s later than they could.
# The read before a click now uses the table read with the round's preparation (25 s earlier), however old.
RATE_KEEP_S = 900.0
REVERT_QUIET_MAX = 1.64   # "revert": a quiet market is vol60 below this (the lowest third, fixed 2026-11-01)
REVERT_MIN_MOVE = 0.5     # ... and bc.game's own price moved more than this ($) over the last 2 s
# "overshoot" / "overshoot_book" (fixed 2026-11-01): bc.game's index moved over the last 2 s AND sits ahead of our exchange
# price in that same direction (dev_tick) - it ran ahead of the market and tends to snap back
OVERSHOOT_VOL_MAX, OVERSHOOT_MOVE, OVERSHOOT_DEV = 1.92, 0.5, 2.0
OVERSHOOT_BOOK_MOVE, OVERSHOOT_BOOK_DEV, OVERSHOOT_BOOK_OBI = 0.5, 1.0, 0.5
FADE_K = 1.0                   # "fade" (fixed 2026-11-10): the same bet from 1 x vol60, about a quarter of all rounds
FADE_STRICT_K, FADE_VOL_FLOOR = 2.45, 1.25   # "fade_strict" (fixed 2026-11-11): from 2.45 x max(vol60, $1.25), fewer bets
OVERSHOOT_REL_K = 2.0          # "overshoot_rel" (fixed 2026-11-10): bc.game's gap from the market is at least this x vol60
TREES_VOL_LO, TREES_VOL_HI, TREES_EV = 1.38, 2.77, 1.03   # "trees_mid": medium activity only, EV at the snapshot payout
# Models chosen by looking at the logged rounds: the go-live gate (and so the real compensate ladder) counts only
# their rounds AFTER these times (UTC epoch), i.e. their forward record; older models keep their whole record.
# HawkesSign ("sign") reads the order books of the six markets of our price since this time (bcbot/hawkes6.py, run by
# the order-book service), no longer Bitfinex spot alone.  The owner had its P&L and history started again with it
# (2026-11-16: "we are doing it from 6 markets again so we need to monitor from the first again"): no round before
# this time counts for it, in any of its numbers.  The old forecast is still made and logged (rec["pred"]): Gap fade
# + Hawkes uses it.  On Bitfinex spot it had been 9,579 of 19,007 (50.4%).
SIGN_SIX_SINCE = 1794843360       # 2026-11-16 15:36 UTC
# "ob_steady" ("Steady order book", the owner's "the order book has a lifetime", 2026-11-17; obfeat.steady_call):
# in the bot from this time.  No earlier round has its answer, and none counts for it.
STEADY_SINCE = 1794961200         # 2026-11-18 00:20 UTC
MODEL_REGISTERED = {"ob_steady": STEADY_SINCE, "sign": SIGN_SIX_SINCE, "value": 1794796080, "fade_flow2": 1794462784, "fade_hawkes": 1794461209, "fade_flow": 1794460320, "thin": 1794427520, "fade_strict": 1794413954, "fade": 1794340955, "overshoot_rel": 1794322458, "trees_mid": 1794217964,
                    "ens": 1793543190, "prob": 1793541896, "overshoot": 1793540105,
                    "overshoot_book": 1793540105, "revert": 1793539811}
# Removed on 2026-11-11 at the owner's request, after losing at a flat $1 on every call.  The dashboard's record since
# the last 'P&L -> 0' (2026-11-10 15:26 UTC): mom 49.0% of 3,065 (-$147), mhh 49.5% of 2,985 (-$114), momvol 49.1% of
# 2,094 (-$97), jump 48.0% of 1,312 (-$89), book 49.1% of 1,646 (-$77), time 50.4% of 2,943 (-$66), bookmom 49.2% of
# 1,381 (-$62), mhh_hi 48.9% of 814 (-$42), agree 50.8% of 1,439 (-$21).  Their forward records (since each was fixed)
# say the same: 48.4-51.0% on 1,505-13,882 calls each, -$30 to -$558.  Betting against them is no better (49-52%):
# the trend, the order book and the order flow carry no signal here either way.  Their code is in git before this change.
# HawkesTime's forecast (pred.exp_ret / time_call, rec["z"]) is still computed in predictor.py and logged with every
# round; only its score is gone (dropping the computation would change HawkesSign's share of the time budget).
# "gbm" (P(UP) trees) was removed on 2026-11-16 at the owner's request: it hardly ever bet (5 bets since the last
# 'P&L -> 0', 99 since it was fixed on 2026-11-01, 56.6% of them right: too few to tell anything).  Its P(UP) is
# still computed and logged with every round (prob_gbm): the ensemble ("ens") is the average of it and the logistic.
REMOVED_MODELS = ("mom", "momvol", "jump", "book", "bookmom", "mhh", "mhh_hi", "agree", "time", "gbm")


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


def _overshoot_rel_call(feat: dict | None, k: float = OVERSHOOT_REL_K, key: str = "dev_tick") -> str | None:
    """The "overshoot_rel" model (pre-registered 2026-11-10): the snap-back with no fixed $ thresholds.  Bet against
    bc.game's gap from our exchange price (dev_tick) when it is at least OVERSHOOT_REL_K times the last minute's
    volatility (vol60), in any market.  Why: Overshoot's $ thresholds (vol60 < 1.92, gap > $2) made no call in 370
    rounds on 2026-11-09, when the median vol60 was 3.72.  On 11,534 logged rounds: first 60% 56% of 1,021, last 40%
    58% of 732 (multipliers 1.5 / 2.5 / 3 gave 55-58% on both parts), about 2.6 times Overshoot's calls.  The
    multiplier was looked at on the whole log, so only the forward record counts.  None otherwise."""
    f = feat or {}
    d, v = f.get(key), f.get("vol60")
    if f.get("error") or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (d, v)):
        return None
    if v <= 0 or d == 0 or abs(d) < k * v:
        return None
    return "DOWN" if d > 0 else "UP"


def _fade_call(feat: dict | None) -> str | None:
    """The "fade" model (pre-registered 2026-11-10): overshoot_rel from FADE_K = 1 x vol60, so that the bot calls in
    about a quarter of all rounds instead of once in hours.  On 8,876 logged rounds, by day (10-30 .. 11-10): 51.9%
    of 243, 54.9% of 1,438, 52.7% of 1,564, 59.3% of 361, 53.8% of 91, 54.5% of 134: 54.3% of 3,831 in all, against
    a break-even of 51.55%.  The multiplier was looked at on the whole log (0.75 and 1.25 give the same picture), and
    the edge is thin: only the forward record counts.  No extra condition (book, quiet / busy, bc.game's last move,
    the late snapshot) was better on both halves."""
    return _overshoot_rel_call(feat, FADE_K)


def _fade_fair_call(feat: dict | None) -> str | None:
    """The "fade_fair" model (2026-11-11, paper): the Gap fade's own rule (the gap is at least FADE_K x vol60, bet
    against it) with the gap measured against the separate new price (dev_fair) instead of our price (dev_tick).
    The two run side by side on the same rounds, so their records say whether the new price is the better one.
    None while there is no new price (bcbot/fair_price.json absent) or it had no value at the snapshot."""
    return _overshoot_rel_call(feat, FADE_K, "dev_fair")


def _fade_strict_call(feat: dict | None) -> str | None:
    """The "fade_strict" model (pre-registered 2026-11-11): the Gap fade's bet, only when the gap is at least
    FADE_STRICT_K = 2.45 times max(vol60, $1.25).  It is what was left of a search for a model that bets often and
    rarely loses 5-6 times in a row: four independent searches (fitted models, bc.game's price mechanics, losing
    runs, agreement of signals and payouts; 2,218 variants, everything chosen on 10-30 .. 11-01, each result re-run by
    a second reviewer) produced 12 candidates and none beat the plain Gap fade on the later days (11-02, 11-09 ..
    11-11).  The gap is the only input with signal; losses are independent (after 3 / 4 / 5 losses in a row the next
    bet wins 48 / 60 / 53%), so no pause rule shortens the runs: only a higher win rate on the bets taken does, and
    the gap's size is the one thing that raises it.  This rule: 61.2% of 719 on the first three days, 56.9% of 253
    (6% of rounds) on the later four, 95% range 50.4-63.2%: about 180 bets a day and a run of 6 losses 3.6 times per
    1,000 bets (Gap fade: 5.3), i.e. still about one every day and a half.  The $1.25 floor (in a quiet market a
    "2 x volatility" gap is under $2 and won 50%) did not bind on the later days, so it is untested out of sample;
    there the rule equals the plain fade at 2.45 x.  Only the forward record counts."""
    f = feat or {}
    d, v = f.get("dev_tick"), f.get("vol60")
    if f.get("error") or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (d, v)):
        return None
    if v <= 0 or d == 0 or abs(d) < FADE_STRICT_K * max(v, FADE_VOL_FLOOR):
        return None
    return "DOWN" if d > 0 else "UP"


FADE_FLOW_KEYS = ("ofi_binance_perp_3s", "ofi_okx_perp_3s", "ofi_bybit_perp_3s")


def _fade_flow_call(feat: dict | None) -> str | None:
    """The "fade_flow" model (2026-11-12, paper; the owner's idea: bet only when a second opinion says the same
    side).  The Gap fade's bet, placed only when at least 2 of the 3 perpetual futures (Binance, OKX, Bybit) had
    order flow on the bet's side over the last 3 s.  29 second opinions were measured on the Gap fade's 5,250 logged
    bets: HawkesSign, the P(UP) models, the trend and the order book moved its win rate by a point or less; the
    futures' order flow was the only one that did.  With this rule: 55.7% of 1,952 bets (+8.4% per $1 at the final
    pools) against 52.1% of the 3,298 it leaves out (first half of the log 56.2 / 52.5, second half 55.0 / 51.8;
    better on 5 of 7 days).  It keeps about 37% of the bets and does NOT make losing runs rare: 9.7 runs of 5+ per
    1,000 bets (Gap fade 11.4).  Picked after looking at 29 candidates, so only its forward record counts."""
    side = _fade_call(feat)
    if not side:
        return None
    votes = [("UP" if v > 0 else "DOWN") for k in FADE_FLOW_KEYS
             if isinstance(v := feat.get(k), (int, float)) and not isinstance(v, bool) and v == v and v != 0]
    return side if sum(x == side for x in votes) >= 2 else None


FADE_FLOW2_K = 2.0             # "fade_flow2": the gap is at least this x vol60 (as overshoot_rel) AND the flow agrees


def _fade_flow2_call(feat: dict | None) -> str | None:
    """The "fade_flow2" model (2026-11-12, paper): the Gap fade's bet with BOTH things that raise its win rate on
    the log: the gap is at least FADE_FLOW2_K = 2 x vol60, and at least 2 of the 3 futures had order flow on the
    bet's side (_fade_flow_call).  Asked for as a way to make long losing runs rarer.  Losses are independent, so runs
    per day = bets per day x p x (1 - p)^k: only a higher win rate and fewer bets lower them.  On the log: 57.6% of
    802 bets (older days 57.7% of 709, recent days 57.0% of 93), +12.1% per $1 at the final pools, about 230 bets on a
    full day, and so about 1.8 runs of five losses and 0.8 runs of six a day, against 15 and 7 for the plain Gap
    fade (seen on the log: 6 and 2 in 802 bets, the longest 6).  Runs do not vanish: at 58% six in a row still comes
    every day or two.  A fitted score on the same inputs did not hold from one part of the log to the other, so the
    rule is kept this simple.  Thresholds picked after looking at 18 such rules: only its forward record counts."""
    side = _fade_flow_call(feat)
    if not side:
        return None
    return side if abs(feat["dev_tick"]) >= FADE_FLOW2_K * feat["vol60"] else None


# "value" (2026-11-15, paper): the chance of each side times what the page pays on it.  Fitted on the rounds up to
# 2026-11-11 20:05 UTC (10,780 rounds): the gap counts only beyond 1 x vol60 (under it the bet against the gap wins
# 49-51%) and up to 4 x; the bigger pool wins a little more often than the smaller one.
VALUE_GAP_LO, VALUE_GAP_HI = 1.0, 4.0
VALUE_W_GAP, VALUE_W_POOL = 0.1154, 0.0831
VALUE_MIN_EV = 0.05            # bet when chance x payout - 1 is at least this
VALUE_FITTED_UNTIL = 1794427520   # 2026-11-11 20:05 UTC: the end of the rounds it was fitted on.  Its record on the
#                                   page counts from MODEL_REGISTERED["value"] = 2026-11-16 02:28 UTC, when it went into
#                                   the bot: the 817 bets in between were its test on the log, not bets the bot made
POOL_BOOKED = ("thin", "value")   # models whose bet is about the payout: booked from the round's final pools (_pool_pay)


def _value_read(feat: dict | None) -> tuple | None:
    """(side, its chance, its expected return per $1) for the better of the two sides at the decision, or None when
    the gap, the volatility or the round's own pools are not known (pools_prev: see _thin_call).  The chance of UP is
    a logistic of two inputs and no constant (UP and DOWN are symmetric): the part of bc.game's gap from the market
    beyond 1 x vol60 (against the gap, as the Gap fade bets) and the log of the pools' ratio, read from the payouts."""
    f = feat or {}
    if f.get("error") or f.get("pools_prev"):
        return None
    vals = [f.get(k) for k in ("dev_tick", "vol60", "pay_up", "pay_dn")]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in vals):
        return None
    d, v, up, dn = vals
    if v <= 0 or up <= 1 or dn <= 1:
        return None
    g = d / v
    beyond = math.copysign(max(0.0, min(abs(g), VALUE_GAP_HI) - VALUE_GAP_LO), g)
    pools = 0.5 * math.log((dn - 1) / (up - 1))            # log(UP pool / DOWN pool): pay_up - 1 = PAY_K x dn / up
    p_up = 1 / (1 + math.exp(VALUE_W_GAP * beyond - VALUE_W_POOL * pools))
    ev_up, ev_dn = p_up * up - 1, (1 - p_up) * dn - 1
    return ("UP", p_up, ev_up) if ev_up >= ev_dn else ("DOWN", 1 - p_up, ev_dn)


def _value_call(feat: dict | None) -> str | None:
    """The "value" model (2026-11-15, paper): bet the side whose chance x payout is at least 1 + VALUE_MIN_EV.

    Why: on 14,895 logged rounds only two things made money at a flat $1 paid at the final pools: the gap (the bet
    against it wins 53-58% from 1.5 x vol60 up) and the payout (the smaller pool wins as often and pays more).  The
    trend, the order book and the order flow did not (CLAUDE.md).  This rule puts both on one scale, so a small gap
    is enough on a side that pays 2.05 and a big gap is needed on a side that pays 1.85.  With nothing but the payout
    it bets from x2.12; with a gap of 2 / 3 / 4 x vol60 from x1.99 / x1.88 / x1.79.

    How it was chosen: four shapes for the chance x seven thresholds, each fitted on the days before and scored on
    the day, inside the rounds up to 2026-11-11 20:05 UTC only; this shape at 5% had the largest profit / sqrt(bets)
    there (1,391 bets, +$142.72).  Then one look at the 4,115 rounds after it (to 11-13 06:38 UTC, about 35 h):
    817 bets, 23 an hour, 51.7% won, +$73.84 = +9.0% per $1 (11-11 +$13.93, 11-12 +$57.15, 11-13 +$2.75; on the same
    rounds Thin side +$43.84, the Gap fade from 2 x +$37.01, the Gap fade +$19.18).  All 28 shape / threshold pairs
    were positive there (+3% to +15% per $1), so the choice of pair is not what carries it.

    What it is not: safe.  It wins about half its bets, so it has losing runs (about 3 runs of six a day, the longest
    on the log 9; worst drop at a flat $1: $24), and a bigger stake fills the pool it bets into (flat $1 +9.0% per
    $1, $5 +6.9%, $10 +4.4% on the later rounds).  A bigger flat stake did better than a series at the same worst
    drop (all rounds: flat $2 +$502, drop $49; x1.5 giving up after 5 +$374, drop $51; x2 after 6 +$423, drop $207).
    Its record on the page counts the rounds since it went into the bot (2026-11-16 02:28 UTC).  The 817 bets on the
    log between the end of its fit and that time were its test (not fitted or chosen on, but seen once before the rule
    was put in the bot); since 2026-11-16 the page no longer shows them as its bets (the owner read them as bets the
    bot had made).  It is paid x2.0-2.1 on average, so break-even is 47-50% wins: judge it by its P&L."""
    r = _value_read(feat)
    return r[0] if r and r[2] >= VALUE_MIN_EV else None


UD_SIGNED = ("dev", "dev_1s", "dev_3s", "dev_chg1", "dev_mean60", "dev_tick", "dev_z60", "tick_chg", "bc_mom05", "bc_mom1",
             "bc_mom2", "bc_mom5", "bc_mom10", "our_mom1", "our_mom2", "our_mom5", "our_mom10", "basis_chg10", "obi1", "obi5",
             "obi20", "obi20_chg5", "obi_w", "micro_off", "taker1", "taker3", "taker5", "taker10",
             "ofi_binance_perp_1s", "ofi_binance_perp_3s", "ofi_bybit_perp_1s", "ofi_bybit_perp_3s",
             "ofi_okx_perp_1s", "ofi_okx_perp_3s")
UD_UNSIGNED = ("vol60", "dev_std60", "spread_binance", "lead_s", "tick_age")
_ud_trees = None                                        # the model, loaded on first use; False when it can't be


def _ud_booster():
    global _ud_trees
    if _ud_trees is None:
        try:
            import lightgbm as lgb
            from pathlib import Path
            here = Path(__file__).parent
            meta = json.loads((here / "ud_trees.json").read_text(encoding="utf-8"))
            _ud_trees = {"booster": lgb.Booster(model_file=str(here / "ud_trees.txt")), "features": meta["features"]}
        except Exception as e:                          # noqa: BLE001 - optional model: the model just makes no calls
            log.warning("Up/Down tree model not loaded: %r", e)
            _ud_trees = False
    return _ud_trees


def _ud_vector(fe: dict, order: list, mirror: bool) -> list:
    """The tree model's features for a snapshot (same as the research script); mirrored = UP and DOWN swapped."""
    def num(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")
    s = -1.0 if mirror else 1.0
    x = {k: s * num(fe.get(k)) for k in UD_SIGNED}
    x.update({k: num(fe.get(k)) for k in UD_UNSIGNED})
    up, dn = num(fe.get("pool_up")), num(fe.get("pool_dn"))
    if mirror:
        up, dn = dn, up
    x["pay_lean"] = (up - dn) / (up + dn) if up + dn > 0 else float("nan")
    x["pool_total"] = up + dn
    d, m = num(fe.get("dev_tick")), num(fe.get("bc_mom2"))
    x["abs_dev"], x["abs_mom2"] = abs(d), abs(m)
    x["same_way"] = 1.0 if d * m > 0 else 0.0
    return [x[k] for k in order]


def _treesmid_call(feat: dict | None) -> str | None:
    """The "trees_mid" model (pre-registered 2026-11-09): the tree model of the logged rounds, used only in medium
    activity, where it won 55.9% of 771 walk-forward bets (+10% per $; calm third 52.7%, busy third 53.0%).  The side
    with the higher expected value at the snapshot's payouts, when that is at least TREES_EV.  None otherwise."""
    f = feat or {}
    v = f.get("vol60")
    if f.get("error") or "dev" not in f or not isinstance(v, (int, float)) or not TREES_VOL_LO <= v < TREES_VOL_HI:
        return None
    m = _ud_booster()
    if not m:
        return None
    a = np.array([_ud_vector(f, m["features"], False), _ud_vector(f, m["features"], True)], dtype=np.float64)
    pr = m["booster"].predict(a)
    p_up = (float(pr[0]) + 1.0 - float(pr[1])) / 2.0    # exactly symmetric between UP and DOWN
    pay = {k: (f.get(k) if isinstance(f.get(k), (int, float)) and f.get(k) > 1 else DEFAULT_PAYOUT) for k in ("pay_up", "pay_dn")}
    ev_up, ev_dn = p_up * pay["pay_up"], (1.0 - p_up) * pay["pay_dn"]
    side, ev = ("UP", ev_up) if ev_up >= ev_dn else ("DOWN", ev_dn)
    return side if ev >= TREES_EV else None


def _sign_call(pred: dict) -> str | None:
    """HawkesSign's side from a Hawkes forecast: UP if P(up) > 0.5, DOWN if below, None without a forecast."""
    p = pred.get("sign_p_up", 0.5) if pred.get("ready", True) else 0.5
    return "UP" if p > 0.5 else "DOWN" if p < 0.5 else None


def _fade_hawkes_call(feat: dict | None, pred: dict | None) -> str | None:
    """The "fade_hawkes" model (2026-11-12, paper; the owner's rule): the Gap fade's bet, placed only when
    HawkesSign's forecast (the order-book model) is on the same side.  On the log HawkesSign's agreement did NOT
    change the Gap fade's win rate: 53.8% of 2,463 bets when it agreed, 53.1% of 2,435 when it did not (recent days
    50.9% when it agreed), so this model is here to be watched side by side, not because the log supports it.  About
    half of the Gap fade's bets.  None while the Hawkes models are warming up (about 5 min after every start)."""
    side = _fade_call(feat)
    if not side or not pred:
        return None
    return side if _sign_call(pred) == side else None


OB_LIFE_EVERY_S = 1.0          # the order-book card asks the service for the books' reading this often (GET /life)
OB_LIFE_STALE_S = 3.5          # ... and says "no reading" when the last answer is older than this
OB_LIFE_SHOW_S = 120.0         # the card's small graph: the reading over this long


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _reading_text(r: float, age: float) -> str:
    """A reading on the order-book card: "78% UP for 7.2 s" (the share of the best-price size on the side it is on;
    an age is counted up to LIFE_CAP_S)."""
    side = " UP" if r > 0 else " DOWN" if r < 0 else ""
    return f"{50 + 50 * abs(r):.0f}%{side} for " + (f"{LIFE_CAP_S:.0f} s+" if age >= LIFE_CAP_S else f"{age:.1f} s")


def _steady_text(life: dict | None) -> str:
    """What the five markets' best prices show, in words: "the books 78% UP for 7.2 s"."""
    r, age = (life or {}).get("r"), (life or {}).get("age")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in (r, age)):
        return "the books' reading is not known"
    return f"the books {50 + 50 * abs(r):.0f}% {'UP' if r > 0 else 'DOWN'} for {age:.1f} s"


def _running(ts, ps, max_pts: int) -> dict:
    """Bets one by one (times, P&L of each) as a running total for a graph: the times and the totals (at most about
    max_pts of them, the last one always), how many bets, how many won, and their sum.  Added up as the Models
    table adds a row up (rounded to 4 places at every bet), so the sum of a whole row is the row's P&L."""
    acc, won, tot = 0.0, 0, []
    for p in ps:
        acc = round(acc + p, 4)
        won += p > 0
        tot.append(acc)
    n = len(tot)
    keep = range(n) if n <= max_pts else sorted({*range(0, n, -(-n // max_pts)), n - 1})
    return {"t": [round(ts[k], 1) for k in keep], "v": [round(tot[k], 2) for k in keep],
            "n": n, "won": int(won), "pnl": acc}


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
    # every model is scored on every round and any of them may make the calls (the "pool" rule is scored only:
    # _model_calls).  The models removed on 2026-11-11 and 2026-11-16 are listed at REMOVED_MODELS
    MODELS = {"sign": "HawkesSign", "revert": "Quiet reversal", "overshoot": "Overshoot",
              "overshoot_book": "Overshoot + book", "prob": "P(UP) model",
              "ens": "P(UP) ensemble", "trees_mid": "Trees, medium market", "overshoot_rel": "Overshoot, scaled",
              "fade": "Gap fade", "fade_strict": "Gap fade, strict", "fade_fair": "Gap fade, new price",
              "thin": "Thin side", "fade_flow": "Gap fade + flow",
              "fade_hawkes": "Gap fade + Hawkes", "fade_flow2": "Gap fade + flow, 2x", "value": "Value bet",
              # the order-book models (obfeat.OB_MODELS): their P(UP) comes from the order-book service at the
              # decision (_ob_fetch).  Selectable since 2026-11-16: the owner looked for them in the Model list
              "ob_logit": "Logistic regression (order books)", "ob_lgbm": "LightGBM (order books)",
              "ob_xgb": "XGBoost (order books)", "ob_deeplob": "DeepLOB (order books)",
              # follow the five markets' best-price reading when it is strong and 5 s or older (obfeat.steady_call)
              "ob_steady": "Steady order book"}
    CALLERS = tuple(MODELS)
    STATES = ("DISARMED", "ARMED", "STOPPED")
    BREAK_EVEN = 0.5155             # 1 / 1.94: the win rate at which a flat bet breaks even (before pool dilution)
    GATE_MIN_N = 3000               # calls a model must have before its record can open the gate at all
    SIDE_WORDS = {"up": "UP", "u": "UP", "dn": "DOWN", "down": "DOWN", "d": "DOWN"}

    def __init__(self, trader: Trader, cfg: Config):
        self.trader, self.cfg = trader, cfg
        self.feed = LobFeed()
        self.pred = HawkesPredictor(self.feed)
        self.flow = BinanceFlow()                         # Binance 1 s klines + best bid/ask: the taker-flow inputs
        self.book = BinanceBook()                         # Binance top-20 book imbalance, for the panel's OB card
        self.bc_ticks: deque = deque(maxlen=3000)         # (server s, bc.game BTC price) from the page's feed
        self.clock = BcClock()                            # bc.game's clock, from those ticks (clock.py)
        self.pools: dict = {}                             # latest Up / Down pool frame
        self.runs: dict = {}                              # per model: its losing runs by length (see _run_step)
        # the order-book models: P(UP) from the order-book service at each decision (_ob_fetch)
        self.ob_v = 0                                     # the fitted version their records count (_ob_roll)
        self.ob_call_v = 0                                # the fit that made the calls last (when one of them calls)
        self.ob_info: dict = {}                           # what the service last said, for the dashboard
        self._ob_http = None
        # the order-book card: the five markets' reading right now, asked once a second (_ob_life_loop); shown only
        self.ob_life: dict = {}                           # {at, r, age, per: {market: its own reading}}
        self.ob_life_hist: deque = deque(maxlen=int(OB_LIFE_SHOW_S / OB_LIFE_EVERY_S) + 10)     # (time, reading)
        self._life_http = None
        self.runs_today: dict = {}                        # the same, only rounds since local midnight
        self._runs_day0 = 0.0
        self.runs_days: dict = {}                         # earlier days: {date: {model: {longest, bets, current}}}
        # the page's P&L graph (pnl_curve): each model's paper bets one by one, the amounts its row in the Models
        # table adds up, and every settled real order
        self.curves: dict = {}                            # per model: (times, P&L of each bet), two arrays
        self.real_pts: list = []                          # (time, P&L in $) of every settled real order
        # our estimate of bc.game's price: a weighted, delayed mix of exchange prices (ourprice.py)
        self.ex_feeds = make_feeds()
        self.our = OurPrice(self.ex_feeds, self.bc_ticks)
        # the separate new price (bcbot/ourprice.py, FAIR_FILE): its own recipe and, where the recipe names an exchange
        # our price does not use, its own feed.  Nothing above or below reads it except _fade_fair_call.  It keeps
        # time by our price's clock (the recording its recipe is fitted on does too)
        self.fair_recipe, self.fair_problem = read_fair_recipe()      # fair_problem: why a file was refused (the page)
        self.fair: OurPrice | None = None
        self.fair_feeds: dict = {}
        self.fair_v = 0
        self.fair_id = 0                                  # a fingerprint of the loaded recipe (0: none), kept with an arming
        self._fair_warned = False
        if self.fair_recipe:
            mix = self.fair_recipe["mix"]
            self.fair_feeds = {k: self.ex_feeds[k] if k in self.ex_feeds else make_feeds([k])[k] for k in mix}
            self.fair = OurPrice(self.fair_feeds, self.bc_ticks, mix=mix, gap_window=self.fair_recipe["level_window"],
                                 clock=self.our.clock)
            self.fair_v = self.fair_recipe["version"]
            self.fair_id = zlib.crc32(json.dumps([self.fair_recipe["registered"], self.fair_recipe["level_window"],
                                                  sorted(mix.items())]).encode()) or 1
            MODEL_REGISTERED["fade_fair"] = self.fair_recipe["registered"]     # its record counts from the recipe's time
        self.prob = ProbModel(cfg.auto_log)               # P(UP) from the logged rounds (bcbot/prob.py)
        self._prob_task = None
        self.on = False
        self.round: dict = {}
        self.recs: dict[int, dict] = {}
        self.history: deque = deque(maxlen=200)          # past expected returns, to scale the confidence z
        self.stats = self._new_stats()
        self.stats_since = 0.0                           # 'P&L -> 0': the shown record counts rounds from this time
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
        self.run_reset_at = 0.0            # 'live reset': real orders before this time don't count in the run
        self.armed_at = 0.0
        self.rearm_grant = False           # beyond the run limit: a re-arm allows exactly one real order
        self.ladder_forced = False         # 'live arm ... ladder': the compensate ladder on although the gate is closed
        self.arm_live_pnl = 0.0            # real P&L since the current arming (the panel's "this run")
        self.other_money_active = lambda: (os.getenv("TAP_LIVE", "off").strip().lower() != "off"     # Tap live on,
                                           or os.getenv("AUTO_BALANCE_STOP", "0").strip() != "1")  # or the check is off
        self.arm_orders = 0
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
        self.limits_set: dict = {}                        # the real-money limits the operator set on the page (LIMITS)
        self._settings_path = cfg.auto_log.with_name("auto_settings.json")
        self.model_replaced: str | None = None            # the chosen model no longer exists (see _check_start)
        self._start_checked = False                       # _check_start has run (the saved state is dealt with)
        try:
            saved = json.loads(self._settings_path.read_text(encoding="utf-8"))
            self.model = saved.get("model", self.model)
            self.model_replaced = saved.get("model_replaced") or None     # a fallback an earlier start did not finish
            self.progression = float(saved.get("progression", self.progression))
            self.since = float(saved.get("since", 0.0))
            self.stats_since = float(saved.get("stats_since", 0.0))
            for key, (_, typ, lo, hi) in self.LIMITS.items():     # the limits set from the page beat .env
                if saved.get(key) and lo <= typ(saved[key]) <= hi:
                    self.limits_set[key] = typ(saved[key])
            self._apply_limits()
            cfg = self.cfg
        except (OSError, ValueError):
            pass
        if self.model not in self.CALLERS:
            log.warning("model %r %s; calls by HawkesSign until another model is chosen", self.model,
                        "was removed on 2026-11-11" if self.model in REMOVED_MODELS else "can't make the calls")
            self.model_replaced, self.model = self.model, "sign"
        # paper P&L if each model's stakes followed the progression (like the MHH dashboard's ladder P&L)
        self.ladders = {m: {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0}
                        for m in self.MODELS}

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
                                                   "since": self.since, "stats_since": self.stats_since,
                                                   **({"model_replaced": self.model_replaced}      # until the start
                                                      if self.model_replaced and not self._start_checked else {}),
                                                   **self.limits_set}),                           # check has run
                                       encoding="utf-8")

    @staticmethod
    def _new_stats() -> dict:
        return {"rounds": 0, "called": 0, "correct": 0, "ties": 0, "skipped": 0, "pnl": 0.0,
                "sign_called": 0, "sign_correct": 0, "bfx_n": 0, "bfx_agree": 0,
                "live_orders": 0, "live_failed": 0, "live_won": 0, "live_pnl": 0.0}

    def reset_stats(self, who: str = "operator") -> str:
        """'P&L -> 0': the shown record (accuracy and P&L of the calls and of every model, the losing runs, the paper
        ladders, the real-order totals on the panel) starts again from now, and stays so across restarts
        (stats_since in auto_settings.json).  The round log is not touched.  NOT reset, because they are evidence or
        safety and not a display: each model's forward record (the go-live gate), the run of real losses, today's
        real P&L and attempts, the campaign P&L."""
        fwd = {k: v for k, v in self.stats.items() if "_fwd_" in k}
        self.stats = {**self._new_stats(), **fwd}
        self.stats_since = time.time()
        self.runs = {}                             # (today's runs and the earlier days' longest stay: calendar days)
        self.curves = {}                           # the graph's paper lines are the rows' bets (the real orders stay)
        self.recent.clear()
        self.settled.clear()
        self.call_streak = 0
        self._restart_test()                       # the paper ladders too (sets `since`)
        self._save_settings()
        self._event("stats_reset", who=who)
        log.warning("Shown record reset by %s: accuracy and P&L count from now (the log and the gate's forward "
                    "records are kept)", who)
        return "accuracy and P&L count from now"

    def _restart_test(self) -> None:
        """A new model / progression starts a fresh ladder test from now."""
        self.since = time.time()
        self.ladders = {m: {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0} for m in self.ladders}

    def set_model(self, name: str) -> None:
        name = name.strip().lower()
        if name not in self.CALLERS:
            raise TradeError(f"model must be one of {', '.join(self.CALLERS)}")
        if self.armed:
            raise TradeError("disarm first ('live off'): the calling model can't change while real orders are on")
        if name != self.model:
            log.warning("Calls now made by %s (was %s)", self.MODELS[name], self.MODELS.get(self.model, self.model))
            self._restart_test()
        self.model = name
        self._save_settings()

    def set_progression(self, x: float, who: str = "operator") -> None:
        """The stake rule after a loss: 1 = flat, 2 = double.  Since 2026-11-10 (the owner's decision) real orders
        follow it like the paper bets: base x progression for each real loss in a row, back to the base after a
        win, for as many losses in a row as the losses-in-a-row stop allows (stake(), _real_ladder()).
        Since 2026-11-12 (the owner's decision) it can be changed while real orders are on: the next real stake then
        follows the new rule at the present run of real losses (x2 chosen at 3 losses in a row makes it $8; Flat makes
        it $1), the change is written to the events file, and an amount already typed for the round being prepared
        is typed again."""
        if not 1.0 <= x <= 3.0:
            raise TradeError("progression must be between 1 (flat) and 3")
        changed, was = x != self.progression, self.progression
        before = self.stake() if self.armed else None
        if changed:
            log.warning("Stake x%g after a loss (was x%g), on paper and for real orders.", x, self.progression)
            self._restart_test()
        self.progression = x
        self._sync_trader_cap()
        self._save_settings()
        if changed and self.armed:                 # real orders are on: say what the next one will be, and log it
            self._event("stake_rule", who=who, was=was, now=x, stake=self.stake(),
                        stakes=[v * self.cfg.auto_stake_usd for v in self._real_ladder()] if self.ladder_on() else None)
            log.warning("Real orders are ON: the next real stake is now $%g (real losses in a row: %d).",
                        self.stake(), self.real_run)
            try:
                if self._prepared_rid and self.stake() != before:
                    asyncio.get_running_loop().create_task(self._prepare(self._prepared_rid))
            except RuntimeError:                   # no event loop (a test)
                pass

    # ---------- start / stop ----------

    def restore_history(self) -> None:
        """Restore saved records once, even when the site has not connected yet."""
        if not getattr(self, "_history_loaded", False):
            self._load_state()
            self._load_log()
            self._check_start()
            self._history_loaded = True

    async def start(self) -> None:
        """Start the order-book feed and the round loop (once); listen to the page's round feed."""
        if not self._tasks:
            self.restore_history()
            self._prob_task = asyncio.ensure_future(asyncio.to_thread(self.prob.fit))
            await self.trader.ensure_ready()
            await self._attach(self.trader.page)
            self.trader.page_listeners.append(self._attach)              # again whenever the window is reopened
            self._tasks = [asyncio.create_task(self.feed.run()), asyncio.create_task(self.flow.run()),
                           asyncio.create_task(self._loop())]
            self._tasks += [asyncio.create_task(f.run()) for f in self.ex_feeds.values()]
            self._tasks += [asyncio.create_task(f.run()) for k, f in self.fair_feeds.items() if k not in self.ex_feeds]
            self._tasks.append(asyncio.create_task(self.book.run()))
            if self.cfg.auto_ob_url:                                     # the order-book card's reading
                self._tasks.append(asyncio.create_task(self._ob_life_loop()))
        self.on = True
        log.info("Auto ON (calls by %s, paper; real orders %s). The Hawkes models need ~5 min of order-book data.",
                 self.model, self.state.lower())

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
        """Now on bc.game's clock (from its ticks); the exchanges' clock while the tick clock cannot be used (too
        few ticks in the last minute, none in the last 20 s, or its medians disagree).  A tick that is a second or
        two late does not change the clock: its offset is a median over the last minute."""
        t = self.clock.now() if self.clock.usable() else None
        return t if t is not None else self.our.server_now()

    def clock_src(self) -> str:
        return "bc" if self.clock.usable() else "exchange"

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
                "campaign_pnl": round(self.campaign_pnl, 4), "state": self.state, "model": self.model,
                "fair_v": self.fair_v, "fair_id": self.fair_id, "stop_reason": self.stop_reason,
                "armed_at": self.armed_at, "ladder_forced": self.ladder_forced, "arm_live_pnl": round(self.arm_live_pnl, 4),
                "arm_orders": self.arm_orders, "rearm_grant": self.rearm_grant,
                "tech_stop": self.tech_stop, "stopped_at": self.stopped_at, "rearms": self.rearms,
                "run_reset_at": self.run_reset_at,
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
        if self.tech_stop and not self.stop_reason:     # lost when Paper was pressed after the stop (until 2026-11-17)
            self.stop_reason = self._stop_reason_from_events()
        self.stopped_at, self.rearms = float(st.get("stopped_at") or 0.0), int(st.get("rearms") or 0)
        self.run_reset_at = float(st.get("run_reset_at") or 0.0)
        self.unknown, self.cleared = list(st.get("unknown") or []), list(st.get("cleared") or [])

    def _stop_reason_from_events(self) -> str | None:
        """Why the technical stop in force was made: the last 'stop' of its kind in the audit trail."""
        try:
            lines = self.cfg.auto_events.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "stop" and e.get("tech") == self.tech_stop:
                return e.get("reason") or None
        return None

    def _check_start(self) -> None:
        """After the log is replayed: compare the rebuilt counter with the saved one, and set the start state:
        STOPPED when the run limit or a technical stop is still in force; ARMED again only when a saved arming is
        resumed (AUTO_RESUME_ARM, inside its time limit, the same calling model); else DISARMED."""
        saved = getattr(self, "_saved_state", {}).get("real_run")
        if saved is not None and int(saved) != self.real_run:
            log.warning("real losses in a row: %d from the log, %d in the state file; keeping the larger",
                        self.real_run, saved)
            self.real_run = max(self.real_run, int(saved))
        if self.tech_stop == "mismatch" and self.other_money_active():
            log.warning("dropping the saved stop %r: Tap live moves the same balance (the balance check is only a "
                        "note while it is on)", self.stop_reason)
            self.tech_stop, self.stop_reason = None, None
        if self.tech_stop or self.unknown:
            self.state = "STOPPED"
            if not self.tech_stop:
                self.tech_stop, self.stop_reason = "unknown", f"outcome of round {self.unknown[0]} unknown"
        elif self.real_run >= self.cfg.auto_max_real_run:
            self.state = "STOPPED"
            self.stop_reason = self.stop_reason or f"{self.real_run} real losses in a row"
        else:
            self.state, self.stop_reason = "DISARMED", None
            saved = getattr(self, "_saved_state", {}) or {}
            armed_at = float(saved.get("armed_at") or saved.get("updated") or 0)
            would_resume = (saved.get("state") == "ARMED" and self.cfg.auto_resume_arm
                            and (not self.cfg.auto_arm_ttl_h                       # 0: no time limit
                                 or time.time() - armed_at < self.cfg.auto_arm_ttl_h * 3600))
            # ... but only with the model that was armed: real orders by a model the owner did not choose must not
            # start.  The state file names the armed model; one written before 2026-11-11 does not, and then a model
            # that had to be replaced at this start (it was removed) is the sign.
            armed_model = saved.get("model")
            same_model = armed_model == self.model if armed_model else not self.model_replaced
            was, now_by = armed_model or self.model_replaced, self.model
            # the new price's model is its recipe (a refit starts its record again): armed on one recipe, it does not
            # go on with another one, or with none.  The fingerprint catches a file changed under the same version
            if same_model and self.model == "fade_fair" and (not self.fair_v or saved.get("fair_v") != self.fair_v
                                                             or saved.get("fair_id") != self.fair_id):
                same_model = False
                was = f"fade_fair, recipe {saved.get('fair_v') or 'none'}"
                now_by = f"fade_fair, recipe {self.fair_v or 'none'}" + (
                    " (the file is not the one armed)" if self.fair_v and saved.get("fair_v") == self.fair_v else "")
            if would_resume and not same_model:
                self._event("disarm", who="restart", reason=f"the calling model changed ({was} -> {now_by})")
                log.warning("Real orders NOT resumed after the restart: they were armed with %r and the calls are now "
                            "made by %r. Switch to LIVE again if that is what you want.", was, now_by)
            elif would_resume:
                self.state, self.armed_at = "ARMED", armed_at          # the arming goes on across the restart
                self.ladder_forced = bool(saved.get("ladder_forced"))
                self.rearm_grant = bool(saved.get("rearm_grant"))
                self.arm_live_pnl = float(saved.get("arm_live_pnl") or 0.0)
                self.arm_orders = int(saved.get("arm_orders") or 0)
                if not self.campaign_since:
                    self.campaign_since = armed_at
                self._event("arm", who="restart", resumed=True, ladder=self.ladder_on())
                log.warning("Real orders ARMED again after the restart (AUTO_RESUME_ARM=1; armed at %s, ladder %s)",
                            time.strftime("%H:%M", time.localtime(armed_at)), "on" if self.ladder_on() else "off")
        self._save_state()
        self._start_checked = True
        if self.model_replaced and self._settings_path.exists():    # the fallback is dealt with: keep the model that
            try:                                   # calls now, so that the next restart has nothing to replace.  (No
                self._save_settings()              # settings file: .env decides, and a file would beat a later fix of
            except OSError as e:                   # it.)  A start must not fail on this file: the model is replaced
                log.warning("settings file: %r", e)                                      # again at the next start
        self._event("start", rebuilt_run=self.real_run)
        log.info("Real orders: %s%s | %d real losses in a row (only a real WIN resets it) | %d attempts today, "
                 "budget left $%.2f | campaign P&L %+.2f%s", self.state.lower(),
                 f" ({self.stop_reason})" if self.stop_reason else "", self.real_run, self.attempts_today,
                 self.budget_left(), self.campaign_pnl, "" if self.cfg.live_allowed else " | LIVE_ALLOWED is not set")

    def arm(self, ack_run: int, day: str, who: str = "operator", force: bool = False, ladder: bool = False) -> str:
        """Allow real orders.  The command must name the current run of real losses and today's date, so nobody
        re-arms without seeing the state they accept; beyond the run limit it allows exactly ONE real order.
        `ladder` (with `force`, while the gate is closed) switches the real compensate ladder on for this arming:
        the operator's decision, logged; it ends with the arming (a stop, a disarm or the time limit)."""
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
            raise TradeError(self.stop_text())
        if self.state == "STOPPED" and time.time() - self.stopped_at < c.auto_rearm_cooloff_min * 60:
            left = c.auto_rearm_cooloff_min * 60 - (time.time() - self.stopped_at)
            raise TradeError(f"stopped {self.stop_reason}; the cool-off ends in {left / 60:.0f} min")
        if self.model not in self.CALLERS:
            raise TradeError("the calling model can't place orders")
        if self.model == "fade_fair" and not self.fair_v:
            raise TradeError("the new price has no recipe yet, so its model makes no calls; choose another model")
        self._roll_day()
        if c.auto_max_real_bets_day and self.attempts_today >= c.auto_max_real_bets_day:
            raise TradeError(f"{self.attempts_today} real orders today already (limit {c.auto_max_real_bets_day})")
        if self.budget_left() < c.auto_stake_usd:
            raise TradeError(f"today's budget is used up (real P&L today {self.day_live_pnl:+.2f}, "
                             f"limit -${c.auto_daily_loss_usd:g}); try again tomorrow")
        if self.campaign_pnl <= -c.auto_campaign_loss_usd:
            raise TradeError(f"the campaign lost {self.campaign_pnl:+.2f}, past -${c.auto_campaign_loss_usd:g}: "
                             "real orders are over (a new campaign needs a new state file)")
        if self.balance_usd is not None and self.balance_usd < max(1, c.auto_min_bankroll_stakes) * c.auto_stake_usd:
            raise TradeError(f"balance ${self.balance_usd:.2f} is below " + (
                f"{c.auto_min_bankroll_stakes} stakes" if c.auto_min_bankroll_stakes > 1 else f"the stake ${c.auto_stake_usd:g}"))
        gate = self.gate_status()
        if not gate["open"] and not force:
            raise TradeError(f"the gate is closed: {gate['text']}. Add 'force' to arm anyway (your decision; logged)")
        if ladder and len(c.auto_real_ladder) < 2:
            raise TradeError("no ladder is configured: set AUTO_REAL_LADDER (e.g. 1,2) in .env first")
        if not self.campaign_since:
            self.campaign_since = time.time()
        self._sync_trader_cap()
        grant = self.real_run >= c.auto_max_real_run
        self.state, self.armed_at, self.stop_reason, self.tech_stop = "ARMED", time.time(), None, None
        self.rearm_grant = grant
        self.ladder_forced = bool(ladder) and not gate["open"]
        self.arm_live_pnl, self.arm_orders = 0.0, 0           # a fresh P&L view from this arming
        self._restart_test()                                   # the paper P&L "since" marker too
        self.rearms += int(grant)
        self._save_state()
        self._event("arm", who=who, ack_run=int(ack_run), grant=grant, force=force, gate_open=gate["open"],
                    ladder=self.ladder_on(), stakes=[x * c.auto_stake_usd for x in self._real_ladder()] if self.ladder_on() else None)
        lad_txt = ("compensate ladder " + "-".join(f"{x * c.auto_stake_usd:g}" for x in self._real_ladder())
                   + (" (your decision: the gate is closed)" if self.ladder_forced else "")
                   if self.ladder_on() else
                   "flat (the compensate ladder waits for the gate; add 'ladder' to use it anyway)"
                   if len(c.auto_real_ladder) > 1 else "flat")
        msg = (f"ARMED ({lad_txt}): real orders of ${self.stake():g} by {self.MODELS[self.model]} "
               + (f"for up to {c.auto_arm_ttl_h:g} h" if c.auto_arm_ttl_h else "until you switch to Paper")
               + f". Stops at {c.auto_max_real_run} real losses in a row, at -${c.auto_daily_loss_usd:g} "
               f"today, {f'after {c.auto_max_real_bets_day} orders a day, ' if c.auto_max_real_bets_day else ''}"
               f"at -${c.auto_campaign_loss_usd:g} in all. "
               + (f"You are past the run limit ({self.real_run} losses): this arming allows ONE real order; a loss "
                  f"stops it again." if grant else "")
               + ("" if gate["open"] else f" The gate is closed ({gate['text']})"
                  + (": its edge is not proven." if self.model in POOL_BOOKED else ": expect to lose."))
               + (" Thin side with a stake rule: your own stake goes into the thin pool, so the bigger steps are paid "
                  "less ($32 into a $105 pool: x2.2 -> x1.9); the finding was made at a flat $1."
                  if self.model == "thin" and self.ladder_on() else "")
               + (" Value bet with a stake rule: it was measured at a flat $1. It wins about half its bets (runs of "
                  "six losses about 3 times a day, the longest on the log 9), and a bigger stake is paid less."
                  if self.model == "value" and self.ladder_on() else ""))
        log.warning(msg)
        return msg

    def disarm(self, why: str, who: str = "operator") -> None:
        if self.state == "ARMED":
            self._event("disarm", reason=why, who=who)
            log.warning("Real orders DISARMED (%s)", why)
        # a technical stop outlives Paper: STOPPED, with its reason, until the owner clears it on the page
        self.state = "STOPPED" if self.tech_stop else "DISARMED"
        self.rearm_grant, self._prepared_rid, self.ladder_forced = False, 0, False
        if not self.tech_stop:
            self.stop_reason = None
        self._save_state()

    def kill(self, who: str = "operator") -> None:
        """Disarm at once (also what the kill file does)."""
        self.disarm("kill", who)

    def set_ladder(self, on: bool, who: str = "operator") -> str:
        """Switch the real compensate ladder on or off for the current arming, without disarming.  Logged.  If the
        stake of the round being prepared changes, the amount box is filled again (else that round is skipped)."""
        c = self.cfg
        if not self.armed:
            raise TradeError("not armed: tick the box and press ARM (the ladder is chosen when arming)")
        if on and len(c.auto_real_ladder) < 2:
            raise TradeError("no ladder is configured: set AUTO_REAL_LADDER (e.g. 1,2) in .env first")
        before = self.stake()
        self.ladder_forced = bool(on)
        self._event("ladder", who=who, on=self.ladder_on(), stake=self.stake())
        try:
            if self._prepared_rid and self.stake() != before:
                asyncio.get_running_loop().create_task(self._prepare(self._prepared_rid))
        except RuntimeError:                      # no event loop (a test)
            pass
        msg = (f"compensate ladder {'ON' if self.ladder_on() else 'OFF'} for this arming: next real stake "
               f"${self.stake():g} (losses in a row {self.real_run})")
        log.warning(msg)
        return msg

    # the real-money limits the operator sets on the page: name -> (Config field, type, lowest, highest)
    LIMITS = {"max_real_run": ("auto_max_real_run", int, 1, 20),          # stop at this many real losses in a row
              "daily_loss_usd": ("auto_daily_loss_usd", float, 1, 2000)}  # the most real money lost in a day

    def _apply_limits(self) -> None:
        for key, v in self.limits_set.items():
            self.cfg = dataclasses.replace(self.cfg, **{self.LIMITS[key][0]: v})
        self._sync_trader_cap()

    def stake_cap(self) -> float:
        """The largest real stake.  Flat on the page: AUTO_MAX_STAKE_USD.  With a stake rule (x1.5, x2) there is no
        separate cap (the owner removed it on 2026-11-10): the series runs for as many losses as the losses-in-a-row
        stop allows, so its last step is the largest stake."""
        c = self.cfg
        if self.progression == 1.0:
            return c.auto_max_stake_usd
        return round(c.auto_stake_usd * self.progression ** (max(1, c.auto_max_real_run) - 1), 2)

    def _sync_trader_cap(self) -> None:
        tc = getattr(self.trader, "cfg", None)             # the order path has its own cap (MAX_USD): keep it in step
        if tc is not None and tc.max_usd < self.stake_cap():
            self.trader.cfg = dataclasses.replace(tc, max_usd=self.stake_cap())

    def set_limit(self, key: str, value, who: str = "operator") -> str:
        """One of the operator's own real-money limits (LIMITS).  Kept across restarts (auto_settings.json, beats
        .env) and written to the events log.  A stop that already happened is not undone by it."""
        field, typ, lo, hi = self.LIMITS[key]
        v = typ(value)
        if not lo <= v <= hi:
            raise TradeError(f"{key} must be between {lo} and {hi}")
        was = getattr(self.cfg, field)
        self.limits_set[key] = v
        self._apply_limits()
        self._save_settings()
        self._event("limit", who=who, key=key, was=was, now=v)
        log.warning("Real-money limit %s is now %g (was %g), set by %s", key, v, was, who)
        return f"{key} is now {v:g}"

    def set_max_real_run(self, n: int, who: str = "operator") -> str:
        return self.set_limit("max_real_run", n, who)

    def reset_run(self, ack_run: int, reason: str, who: str = "operator") -> str:
        """Set the run of real losses back to 0 by hand: the operator's decision (a new model, a new experiment),
        typed with the current count and a reason, written to the events log and kept across restarts (real orders
        before it no longer count).  Today's budget, the order count and the campaign loss are untouched."""
        c = self.cfg
        if int(ack_run) != self.real_run:
            raise TradeError(f"acknowledge the current count exactly: live reset run={self.real_run} <reason>")
        reason = (reason or "").strip()
        if len(reason) < 3:
            raise TradeError("say why (for example 'new model overshoot'); it goes into the events log")
        if self.tech_stop:
            raise TradeError(self.stop_text())
        if self.state == "STOPPED" and time.time() - self.stopped_at < c.auto_rearm_cooloff_min * 60:
            left = c.auto_rearm_cooloff_min * 60 - (time.time() - self.stopped_at)
            raise TradeError(f"stopped {self.stop_reason}; the cool-off ends in {left / 60:.0f} min")
        was = self.real_run
        self.real_run, self.run_reset_at, self.rearm_grant = 0, time.time(), False
        if self.state == "STOPPED":
            self.state, self.stop_reason = "DISARMED", None
        self._save_state()
        self._event("run_reset", who=who, was=was, reason=reason)
        msg = (f"real losses in a row set to 0 (was {was}): {reason}. Real orders before now no longer count in the "
               f"run; the budget, the order count and the campaign loss still do. Arm with: live arm run=0 "
               f"day={time.strftime('%Y-%m-%d')}")
        log.warning(msg)
        return msg

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

    def stop_text(self) -> str | None:
        """The technical stop in force, in plain words for the page: what happened and when, what to check on
        bc.game, how to go on.  None when there is none."""
        if not self.tech_stop:
            return None
        what, check = TECH_STOPS.get(self.tech_stop, (self.tech_stop, "bc.game"))
        when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.stopped_at)) if self.stopped_at else "an earlier day"
        return (f"Real orders have been stopped since {when}: {self.stop_reason or what}. Check {check}. "
                "Then press 'Clear stop' on this page and switch to LIVE again.")

    def _hard_stop(self, why: str, tech: str | None = None) -> None:
        """Real orders stop and the bot disarms.  It never resumes by itself: the operator must re-arm, naming the
        current run of losses (and clear the technical reason first, when there is one)."""
        self.state, self.stop_reason, self.stopped_at = "STOPPED", why, time.time()
        self.ladder_forced = False
        self.rearm_grant, self._prepared_rid = False, 0
        if tech:
            self.tech_stop = tech
        self._save_state()
        self._event("stop", reason=why, tech=tech)
        log.warning("REAL ORDERS STOPPED: %s. %s", why,
                    f"Check the site, then clear the stop: 'Clear stop' on the page (or 'live clear {tech}')." if tech else
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
        if self.model in OB_FITTED:                 # fitted again as the recording grows: each fit has its own record
            scope = f"the rounds of its fit {self.ob_v} (a new fit starts the record again)"
        elif self.model == "sign":
            scope = ("rounds since it reads the six markets' order books ("
                     + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(SIGN_SIX_SINCE)) + ")")
        text = (f"{self.MODELS.get(self.model, self.model)} {k}/{n} = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%}) on {scope} "
                f"vs break-even {self.BREAK_EVEN:.2%}, needs {self.GATE_MIN_N} calls" if n else
                f"no forward record yet ({scope})")
        if self.model == "thin":                    # its payout is not the x1.94 this gate assumes
            text += (". Thin side is paid about x2.2, so it breaks even near 45.5% wins, not "
                     f"{self.BREAK_EVEN:.2%}: judge it by its P&L, not by this win rate")
        elif self.model == "value":                 # it bets where the payout is high: x2.0-2.1 on average
            text += (". Value bet picks the side by chance x payout and is paid x2.0-2.1 on average, so it breaks "
                     f"even at 47-50% wins, not {self.BREAK_EVEN:.2%}: judge it by its P&L, not by this win rate")
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
                    elif time.time() - self.last_reload_at > 90 and await self.trader.launch_refused():
                        log.warning("bc.game refused to start the game (request_too_fast): reloading the page")
                        self.last_reload_at = time.time()           # at most once per 90 s, or it stays refused
                        await self.trader.page.reload(wait_until="domcontentloaded")
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
        if self.armed and self.cfg.auto_arm_ttl_h and time.time() - self.armed_at > self.cfg.auto_arm_ttl_h * 3600:
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
            if self.cfg.auto_ob_url:                               # the order-book models' P(UP) at this instant
                asyncio.ensure_future(self._ob_fetch(rid, rec))    # (its own process; the round loop never waits for
            elif self.model in OB_MODELS:                          # it; a call by one of them is made in there)
                self._ob_no_call(rec, "the order-book service is switched off (AUTO_OB_URL is empty)")
            rec["hawkes_task"] = asyncio.ensure_future(self._hawkes(rid, rec))     # never awaited here
            try:                                                   # P(UP) for every round (all three scored)
                ps = self.prob.predict_all(rec["feat"])
                rec["prob"], rec["prob_gbm"], rec["prob_ens"] = ps["prob"], ps["gbm"], ps["ens"]
            except Exception as e:
                rec["prob"] = rec["prob_gbm"] = rec["prob_ens"] = None
                log.warning("P(UP) models: %r", e)
            if self.model in ("prob", "ens"):
                key = {"prob": "prob", "ens": "prob_ens"}[self.model]
                asyncio.ensure_future(self._call(rid, rec, ProbModel.side(rec[key])))
            if self.model == "revert":                               # quiet market: bc.game undoes its last move
                asyncio.ensure_future(self._call(rid, rec, _revert_call(rec["feat"])))
            elif self.model in ("overshoot", "overshoot_book"):      # bc.game ran ahead of the market: snap-back
                asyncio.ensure_future(self._call(rid, rec, _overshoot_call(rec["feat"], self.model == "overshoot_book")))
            elif self.model == "fade_fair":                          # the Gap fade against the separate new price
                asyncio.ensure_future(self._call(rid, rec, _fade_fair_call(rec["feat"])))
            elif self.model == "fade_strict":                        # the same bet from 2.45 x the volatility: fewer bets
                asyncio.ensure_future(self._call(rid, rec, _fade_strict_call(rec["feat"])))
            elif self.model == "fade_flow2":                         # ... and the gap is at least 2 x the volatility
                asyncio.ensure_future(self._call(rid, rec, _fade_flow2_call(rec["feat"])))
            elif self.model == "fade_flow":                          # the Gap fade, when the futures' flow agrees
                asyncio.ensure_future(self._call(rid, rec, _fade_flow_call(rec["feat"])))
            elif self.model == "fade":                               # the same bet from 1 x the volatility
                asyncio.ensure_future(self._call(rid, rec, _fade_call(rec["feat"])))
            elif self.model == "overshoot_rel":                      # the snap-back, scaled by the volatility
                asyncio.ensure_future(self._call(rid, rec, _overshoot_rel_call(rec["feat"])))
            elif self.model == "trees_mid":                          # the tree model, medium activity only
                asyncio.ensure_future(self._call(rid, rec, _treesmid_call(rec["feat"])))
            elif self.model == "thin":                               # no prediction: the side that pays 2.10 or more
                asyncio.ensure_future(self._call(rid, rec, _thin_call(rec["feat"])))
            elif self.model == "value":                              # chance (from the gap) x payout at least 1.05
                asyncio.ensure_future(self._call(rid, rec, _value_call(rec["feat"])))
        if "feat_mid" not in rec and now >= cutoff - self.cfg.auto_mid_snapshot_s:
            try:                                  # snapshot M (paper research): the same short list as snapshot B,
                rec["feat_mid"] = self._features(now, rec, late=True)   # at a lead a real order could still make
            except Exception as e:
                rec["feat_mid"] = {"error": repr(e)[:120]}
        if "feat_late" not in rec and now >= cutoff - self.cfg.auto_late_snapshot_s:
            try:                                  # snapshot B: the late, tick-aligned deviation (paper research)
                rec["feat_late"] = self._features(now, rec, late=True)
            except Exception as e:
                rec["feat_late"] = {"error": repr(e)[:120]}
        for old in sorted(self.recs)[:-50]:          # keep the newest 50 rounds (ids grow by ~1e6 per round,
            self.recs.pop(old, None)                # so "rid - 50" kept only the current one; the charts need more)

    def _ob_no_call(self, rec: dict, why: str) -> None:
        """An order-book model makes the calls and has no answer for this round: no bet, and the round says why."""
        if not rec.get("call") and not rec.get("skip"):
            rec["skip"] = why
            self.stats["skipped"] += 1

    async def _ob_fetch(self, rid: int, rec: dict) -> None:
        """Ask the order-book service (tools/ob_server.py: the books and trades of the six markets of our price,
        its own process) for its models' P(UP) at the decision, with the gap and vol60 of this snapshot.  The answer
        is logged with the round (rec["ob"]: version, P(UP) and betting level per model) and every one of them is
        scored on it.  When one of them is the calling model, its call is made here, as soon as the answer is in
        (0.04-0.5 s after the decision; no answer in 0.8 s = no call in this round)."""
        f = rec.get("feat") or {}
        now = self.bc_now()                 # a, b: when the round's window starts and ends, from now (for ob_hawkes)
        q = {k: f"{v:.4f}" for k, v in (("gap", f.get("dev_tick")), ("vol", f.get("vol60")),
                                        ("a", rec["start"] - now if "start" in rec else None),
                                        ("b", rec["end"] - now if "end" in rec else None))
             if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        try:
            if self._ob_http is None or self._ob_http.closed:
                self._ob_http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.8))
            async with self._ob_http.get(self.cfg.auto_ob_url, params=q) as r:
                d = await r.json()
        except Exception as e:
            self.ob_info = {"ok": False, "error": (repr(e) or type(e).__name__)[:100], "at": time.time()}
            if self.model in OB_MODELS:
                self._ob_no_call(rec, "the order-book service did not answer")
            return
        rec["ob"] = {"v": d.get("v") or 0, "p": d.get("p") or {}, "thr": d.get("thr") or {}, "n_ok": d.get("n_ok"),
                     "lead_s": round(rec["cutoff"] - self.bc_now(), 3)}
        if d.get("hk"):                     # ob_hawkes: each market's own P(UP), the share of our price it covered
            rec["ob"]["hk"] = d["hk"]
        if d.get("life"):                   # the five markets' best-price reading, its age, the price's move since
            rec["ob"]["life"] = d["life"]   # (what ob_steady decides on; logged in every round)
        self.ob_info = {"ok": True, "at": time.time(), **(d.get("status") or {})}
        if self.model in OB_MODELS:         # this model makes the calls (paper, and real orders when armed)
            v = rec["ob"]["v"]
            if self.model in OB_FITTED and v and v != self.ob_call_v:
                if self.ob_call_v and self.armed:       # a new fit took the model's place while real orders are on:
                    self._event("ob_fit", model=self.model, version=v, was=self.ob_call_v)   # written down, no stop
                    log.warning("%s: fit %s replaced fit %s while real orders are on", self.MODELS[self.model], v, self.ob_call_v)
                self.ob_call_v = v
            key = OB_ANSWER[self.model]         # "sign" is ob_hawkes in the answer
            p = rec["ob"]["p"].get(key)
            if p is None:
                self._ob_no_call(rec, "the order-book service has no answer from this model"
                                 + ("" if v or self.model not in OB_FITTED else " (not fitted yet)"))
            else:
                call = _ob_side(p, rec["ob"]["thr"].get(key))
                if call is None and self.model == "ob_steady":      # no bet: the round says what the books showed
                    self._ob_no_call(rec, "no steady reading: " + _steady_text(rec["ob"].get("life"))
                                     + f" (a bet needs {50 + 50 * STEADY_MIN:.0f}% for {STEADY_AGE_S:g} s)")
                else:
                    await self._call(rid, rec, call)

    def _ob_roll(self, rec: dict) -> None:
        """The order-book models are fitted again as the recording grows.  A round answered by a newer version
        starts their records again: each version is judged on its own rounds, all of them after its fit.  Not
        ob_hawkes: it is fitted again every 20 s by design and keeps one record."""
        v = (rec.get("ob") or {}).get("v") or 0
        if v > self.ob_v:
            self.ob_v = v
            for name in OB_FITTED:
                for key in ("called", "correct", "pnl", "fwd_called", "fwd_correct"):
                    self.stats.pop(f"{name}_{key}", None)
                self.runs.pop(name, None)
                self.runs_today.pop(name, None)
                self.curves.pop(name, None)
                if name in self.ladders:
                    self.ladders[name] = {"pnl": 0.0, "staked": 0.0, "biggest": 0.0, "step": 0}

    # ---------- the order-book card: what the books show right now ----------

    async def _ob_life_loop(self) -> None:
        """Once a second: ask the order-book service what the five markets' best prices show right now (GET /life),
        for the order-book card.  It is shown only: nothing here is logged, scored or used for a call.  The reading a
        round is decided on is the one in the service's answer to the bot's question (rec["ob"]["life"])."""
        while True:
            await asyncio.sleep(OB_LIFE_EVERY_S)
            try:
                await self._ob_life_fetch()
            except Exception as e:                       # the loop never ends; the card says "no reading" by itself
                log.debug("order-book card: %r", e)

    async def _ob_life_fetch(self) -> None:
        url = self.cfg.auto_ob_url.rsplit("/", 1)[0] + "/life"
        try:
            if self._life_http is None or self._life_http.closed:    # its own connection: never the question's
                self._life_http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.5))
            async with self._life_http.get(url) as r:
                d = await r.json()
        except Exception:
            return                                       # no answer: the last one goes stale (OB_LIFE_STALE_S)
        self._ob_life_take(d)

    def _ob_life_take(self, d, at: float | None = None) -> None:
        """Keep one answer of GET /life: the reading for the card, and one point of its small graph."""
        if not isinstance(d, dict):
            return
        at = time.time() if at is None else at
        r, age = (float(d[k]) if _is_num(d.get(k)) else None for k in ("r", "age"))
        per = d.get("per") if isinstance(d.get("per"), dict) else {}
        self.ob_life = {"at": at, "r": r, "age": age, "per": {str(k): float(v) for k, v in per.items() if _is_num(v)}}
        if r is not None:
            self.ob_life_hist.append((at, max(-1.0, min(1.0, r))))

    def _ob_card(self) -> str:
        """The order-book card: the best-price reading of five of the six markets the order-book models read (Bybit,
        OKX and Binance futures, OKX spot, Coinbase, by their share in our price; Bitfinex perp's best price is
        thin), averaged over 1 s, and how long it has been on its present side.  Under the graph, for the last round
        that was decided: the reading at the decision (the one the round was decided on, rec["ob"]["life"]) and the
        reading when that round's window opened, with "turned" when it had changed side by then (the owner's case:
        "the OB was showing Buying, but when the bet round is starting that has already ended").
        The same numbers as "ob_steady" and the fitted models' lifetime inputs (obfeat.life_mixed).
        Until 2026-11-18 this card showed Binance SPOT's top 20 levels: not one of the six, a market bc.game's price
        does not follow, and above 71% on one side in half of the rounds; that feed still gives the logged inputs
        (obi20 ...).  The owner watches this card against the result, so it must show what the models read."""
        lf, now = self.ob_life, time.time()
        r, age = lf.get("r"), lf.get("age")
        head = ('<div class="ct"><span title="Sizes at the best bid and ask of Bybit, OKX and Binance futures, OKX '
                'spot and Coinbase, mixed by their share in our price, averaged over the last second. The order-book '
                'models read these books.">ORDER BOOKS &middot; 5 MARKETS</span><span class="mute">{}</span></div>')
        stale = now - lf.get("at", 0.0) > OB_LIFE_STALE_S
        if not self.cfg.auto_ob_url or stale or r is None or age is None:
            why = ("the order-book service is switched off" if not self.cfg.auto_ob_url else
                   "the order-book service does not answer" if stale else "the books are not known right now")
            return (f'<div class="c">{head.format("no reading")}<div class="big mute">&ndash;</div>'
                    f'<div class="obw mute">{why}</div></div>')
        pct = round(50 + 50 * abs(r))                   # as the big number shows it: the words and the colour go by it
        side_cls = ("up" if r > 0 else "down") if pct >= 60 else ""            # coloured from 60% on one side
        color = {"up": "#22d27a", "down": "#ff5468"}.get(side_cls, "#c3ccd8")
        who = "buyers" if r > 0 else "sellers"
        words = "balanced" if pct < 60 else f"mostly {who}" if pct >= 75 else f"more {who}"
        per = [v for v in lf.get("per", {}).values() if v]
        if per and r:                                   # how many of the markets are on the mixed reading's side
            words += f" &middot; {sum(v * r > 0 for v in per)} of {len(per)} markets"
        fill = (f"left:50%;width:{r * 50:.1f}%;background:#22d27a" if r >= 0 else
                f"left:{50 + r * 50:.1f}%;width:{-r * 50:.1f}%;background:#ff5468")
        hist = [(t, h) for t, h in self.ob_life_hist if now - t <= OB_LIFE_SHOW_S]
        sw, sh = 212, 40
        spark = ""
        if len(hist) >= 2:
            pts = " ".join(f"{(1 - (now - t) / OB_LIFE_SHOW_S) * sw:.1f},{sh / 2 - h * sh / 2 * 0.92:.1f}" for t, h in hist)
            spark = (f'<svg class="spark" viewBox="0 0 {sw} {sh}" preserveAspectRatio="none">'
                     f'<line x1="0" y1="{sh / 2}" x2="{sw}" y2="{sh / 2}" stroke="#5b6573" stroke-dasharray="3 3"/>'
                     f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')
        rec = next((x for _, x in sorted(self.recs.items(), reverse=True) if (x.get("ob") or {}).get("life")), {})
        was = (rec.get("ob") or {}).get("life") or {}
        then = began = "&ndash;"
        if _is_num(was.get("r")) and _is_num(was.get("age")):
            then = _reading_text(was["r"], was["age"])
            if _is_num(rec.get("start")):               # the round's times are on bc.game's clock, the graph's on ours
                t0 = rec["start"] + now - self.bc_now()
                at = next((h for t, h in self.ob_life_hist if t0 <= t <= t0 + 2 * OB_LIFE_EVERY_S + 0.5), None)
                if at is not None:
                    began = (_reading_text(at, 0.0).partition(" for ")[0]
                             + (" &middot; <b>turned</b>" if at * was["r"] < 0 else ""))
        big, _, since = _reading_text(r, age).partition(" for ")
        return (
            f'<div class="c {"ob" + ("up" if side_cls == "up" else "dn") if side_cls else ""}">{head.format("live")}'
            f'<div class="big {side_cls}">{big}<small>for {since}</small></div><div class="obw {side_cls}">{words}</div>'
            f'<div class="obbar"><i style="{fill}"></i></div>'
            f'<div class="two mute sm"><span>all sellers</span><span>even</span><span>all buyers</span></div>{spark}'
            f'<div class="two mute sm"><span>best price &middot; 1 s average</span><span>last 2 min</span></div>'
            f'<div class="row"><span>at decision</span><span>{then}</span></div>'
            f'<div class="row"><span title="When the 5 s window of that round opened">at start</span>'
            f'<span>{began}</span></div></div>')

    async def _hawkes(self, rid: int, rec: dict) -> None:
        """The Hawkes forecast on Bitfinex spot's book (the bot's own predictor): logged with every round, and the
        call when the model is Gap fade + Hawkes.  HawkesSign itself ("sign") is the six-market forecast since
        SIGN_SIX_SINCE and is called from the order-book service's answer (_ob_fetch)."""
        now = self.bc_now()
        a = rec["start"] - now                 # the window starts this long from now (bc.game's clock)
        c = self.cfg
        if not c.auto_min_lead_s <= a <= c.auto_max_lead_s:
            rec["skip"] = f"odd timing (window starts in {a:.1f} s)"     # a replayed or mistimed round: not
            self.stats["skipped"] += 1                                    # predicted, not scored, not in the history
            return
        calling = self.model == "fade_hawkes"
        b, at = rec["end"] - now, self.pred.feed.now()     # the forecast is made as of this instant, whenever it runs
        if not calling:                                    # not needed for the call: keep the loop free for the order
            wait = rec["cutoff"] - min(c.auto_send_margin_s, c.auto_late_snapshot_s) + HAWKES_AFTER_S - self.bc_now()
            if wait > 0:
                await asyncio.sleep(wait)
        p = await asyncio.to_thread(self.pred.predict, a, b, at)
        p["done_lead_s"] = round(rec["cutoff"] - self.bc_now(), 3)        # how long before the cutoff it finished
        rec["pred"] = p
        if not p.get("ready"):
            if calling:
                rec["skip"] = "Hawkes models warming up (%.0f s of data)" % self.pred.data_span()
                self.stats["skipped"] += 1
            return
        hist = np.array(self.history)
        z = abs(p["exp_ret"]) / hist.std() if len(hist) >= 20 and hist.std() > 0 else None
        self.history.append(p["exp_ret"])
        rec["z"] = z
        log.info("round %s Hawkes on Bitfinex spot P(up) %.2f  ~%.1f changes  (lead %.1fs, calc %d ms, done %.2fs before cutoff)",
                 rid, p["sign_p_up"], p["n_changes"], p["lead_s"], p["calc_ms"], p["done_lead_s"])
        if calling:                                # fade_hawkes: the Gap fade's side, when this forecast agrees
            await self._call(rid, rec, _fade_hawkes_call(rec.get("feat"), p))

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
        if self.model in POOL_BOOKED:
            rec["by"] = self.model                      # its paper bet is shown as booked from the final pools
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
            bk = self.book                                    # Binance spot top-20 book (logged inputs only)
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
        else:                                                 # no pool frame of this round yet: the last frame,
            p = self.pools or {}                              # usually the previous round's final pools
            f["pool_up"], f["pool_dn"] = p.get("upPoolAmount"), p.get("downPoolAmount")
            f["pools_prev"] = True                            # read by _thin_call only
        try:
            up, dn = float(f["pool_up"] or 0), float(f["pool_dn"] or 0)
            f["pay_up"] = 1 + PAY_K * dn / up if up > 0 else None
            f["pay_dn"] = 1 + PAY_K * up / dn if dn > 0 else None
        except (TypeError, ValueError):
            f["pay_up"] = f["pay_dn"] = None
        held_by = list(stale) + ([] if our_now is not None else ["our price"])   # what holds a real order back:
        if now - tick_t > 1.0:                                # ... not bc.game's own ticks being late (the owner,
            stale.append("bc_ticks")                          # 2026-11-17): logged as stale, the order goes out
        if not self.feed.connected:
            stale.append("bitfinex_lob")
            held_by.append("bitfinex_lob")
        ages["bc_tick"] = round(now - tick_t, 3)
        f["dq"] = {"ok": now - tick_t <= 1.0 and our_now is not None and self.clock.ok() and self.feed.connected,
                   "stale": stale, "ages": ages, "held_by": held_by}
        if self.fair is not None:                             # the separate new price: logged next to our price,
            try:                                              # read by no model but fade_fair.  Last, so that nothing
                fsnap = self.fair._snapshot()                 # above is read later than it was without it
                f["fair"] = self.fair.value(now, fsnap, max_age=STALE_SNAPSHOT, min_weight=MIN_WEIGHT)
                f["fair_v"] = self.fair_v
                fair_tick = self.fair.value(tick_t, fsnap, max_age=STALE_SNAPSHOT, min_weight=MIN_WEIGHT)
                if fair_tick is not None:
                    f["dev_fair"] = bc_now - fair_tick        # bc.game's gap from the new price at the tick's time
            except Exception as e:                            # noqa: BLE001 - it must never break a snapshot
                f["fair_error"] = repr(e)[:80]
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}

    def ladder_on(self) -> bool:
        """The real compensate ladder is used only when it is configured (AUTO_REAL_LADDER with 2+ steps) AND the
        go-live gate is open: the calling model has >= GATE_MIN_N calls with its Wilson lower bound above the
        break-even.  Without a proven edge a ladder only loses faster (docs/decision-review, section 7b).
        The operator can override that for one arming ('live arm ... force ladder'); it is logged."""
        if self.progression != 1.0:                # the page's stake rule: the operator's choice, no gate needed
            return len(self._real_ladder()) > 1
        return len(self.cfg.auto_real_ladder) > 1 and (self.gate_status()["open"] or (self.ladder_forced and self.armed))

    def _real_ladder(self) -> tuple:
        """The real stake series as multiples of the base stake.  With a stake rule chosen on the page (x1.5, x2):
        1, p, p^2, ... one step for each loss the losses-in-a-row stop allows (no other cap).
        Flat on the page: AUTO_REAL_LADDER from .env, as before."""
        c = self.cfg
        if self.progression == 1.0:
            return tuple(c.auto_real_ladder)
        return tuple(round(self.progression ** i, 4) for i in range(max(1, c.auto_max_real_run)))

    def stake(self) -> float:
        """The real stake: the flat base, or with the ladder on, base x the ladder step for the current run of real
        losses (reset by a real win; after the last step it starts again from the first).  The step follows the
        persistent loss counter, so a restart cannot reset it; every stop and the day's budget still apply."""
        c = self.cfg
        if not self.ladder_on():
            return c.auto_stake_usd
        lad = self._real_ladder()
        return round(c.auto_stake_usd * lad[self.real_run % len(lad)], 2)

    def paper_stake(self) -> float:
        """The calling model's next paper stake: the base x progression for each loss in a row of its paper ladder."""
        lad = self.ladders.get(self.model) or {}
        return round(self.cfg.auto_stake_usd * self.progression ** lad.get("step", 0), 2)

    # ---------- real orders: the checklist and the order path ----------

    async def _refresh_balance(self, quick: bool = False) -> None:
        """Read the page's balance (account currency and USD), the account kind and any dialog.  quick (right
        before a click): never wait for bc.game's price table, see RATE_KEEP_S."""
        try:
            st = await self.trader.status(light=True)
        except Exception as e:
            log.debug("balance read failed: %r", e)
            return
        self.dialog = st.dialog
        rate = None
        if st.currency:
            try:
                rate = await (self.trader.usd_rate(st.currency, RATE_KEEP_S) if quick
                              else self.trader.usd_rate(st.currency))
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
                return f"stopped: {self.stop_reason or self.tech_stop} (needs 'Clear stop' on the page)"
            if c.auto_arm_ttl_h and time.time() - self.armed_at > c.auto_arm_ttl_h * 3600:
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
        if c.auto_max_real_bets_day and attempts >= c.auto_max_real_bets_day:
            return f"{attempts} real orders today (limit {c.auto_max_real_bets_day})"
        budget = c.auto_daily_loss_usd + self.day_live_pnl
        if stake > budget + 1e-9:
            return f"today's budget left ${budget:.2f} is less than the stake"
        if self.campaign_pnl <= -c.auto_campaign_loss_usd:
            return f"campaign loss {self.campaign_pnl:+.2f} reached -${c.auto_campaign_loss_usd:g}"
        if stake > self.stake_cap() + 1e-9:
            return f"stake ${stake:g} above the largest stake allowed ${self.stake_cap():g}"
        if self.balance_usd is None or time.time() - self.balance_at > c.auto_balance_max_age_s:
            return f"balance not read in the last {c.auto_balance_max_age_s:g} s"
        if self.balance_account != "REAL":
            return f"account is {self.balance_account or 'unknown'}, not REAL"
        if self.balance_usd < stake:              # the balance must at least cover this order
            return f"balance ${self.balance_usd:.2f} is less than the stake ${stake:g}"
        if self.balance_usd < c.auto_min_bankroll_stakes * min(stake, c.auto_stake_usd):     # off unless set in .env
            return f"balance ${self.balance_usd:.2f} is below {c.auto_min_bankroll_stakes} stakes"
        if self.dialog:
            return f"a dialog is open on the page ({self.dialog[:30]})"
        if not self.clock.usable():               # a late tick alone is no reason (the owner, 2026-11-17): only a
            return f"bc.game clock not ok ({self.clock.why_not_usable()})"      # clock that cannot time the order
        if now > rec["cutoff"] - c.auto_send_margin_s:
            return f"too late ({rec['cutoff'] - now:.2f} s before the cutoff, margin {c.auto_send_margin_s:g} s)"
        r = self.round or {}
        if int(r.get("id") or 0) != rid or r.get("status") != BETTING:
            return "betting closed or another round open"
        if time.time() - self.last_reload_at < 60:
            return "page reloaded or round feed reconnected less than 60 s ago"
        # (no pause after an order bc.game refused: removed on 2026-11-17 at the owner's request.  AUTO_REJECT_STOP
        # refusals within 10 minutes still stop real orders, see _send.)
        if not manual:
            p = rec.get("pred") or {}
            if p.get("ready") and not (c.auto_min_lead_s <= p.get("lead_s", 0) <= c.auto_max_lead_s):
                return f"odd forecast lead ({p.get('lead_s')} s)"
            dq = (rec.get("feat") or {}).get("dq") or {}
            if dq and not dq.get("ok", True):     # an exchange feed is stale or our price is unknown; bc.game's own
                held = (dq["held_by"] if "held_by" in dq       # ticks being late is not among them any more
                        else [s for s in (dq.get("stale") or ["?"]) if s != "bc_ticks"])
                if held:
                    return "data quality: " + ", ".join(held)
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
        await self._refresh_balance(quick=True)                 # the balance right before the click, to reconcile
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
            if self.fair is not None:                                           # the same for the separate new price
                try:
                    fm = self.fair.move(rec["start"], rec["end"])
                    if fm is not None and not tie:
                        rec["fair_move"], rec["fair_v"] = fm, self.fair_v
                except Exception as e:                                          # noqa: BLE001 - never breaks a settle
                    log.warning("new price move: %r", e)
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
        self.arm_live_pnl += rec["live_pnl"]
        self.arm_orders += 1
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
        if self.armed and not self.other_money_active():
            self._hard_stop(f"balance moved {delta if delta is None else round(delta, 4)} {before[1]} but the "
                            f"order's result was {expected:+.4f}", tech="mismatch")
        elif self.armed:                          # Tap live moves the same balance: the check can't tell them apart
            log.warning("round %s: balance moved %s but the order's result was %+.4f (Tap live is on, so the balance "
                        "check is only a note)", rec["id"], delta if delta is None else round(delta, 4), expected)

    def _pnl(self, rec: dict, side: str | None, diluted: bool = False, pools: bool = False) -> float:
        """$-stake P&L of calling `side` in a settled round: that side's page payout when it was read (the called
        side's, else from the pair logged at call time), else the average; `diluted` books it as a pari-mutuel bet
        of our stake into that side's pool.  (Recomputed on replay: early records logged payout 0.)
        `pools` (the "thin" model): paid as bc.game pays, from the round's final pools with our stake in them; its
        bet is about the payout, and the page's payout is not read in rounds the calling model does not call."""
        if not side or rec.get("tie"):
            return 0.0
        stake = self.cfg.auto_stake_usd
        if side != rec.get("win"):
            return -stake
        if pools and (pay := _pool_pay(rec, side, stake)):
            return stake * (pay - 1)
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

    def _ladder_step(self, lad: dict, rec: dict, side: str, pools: bool = False) -> None:
        """One paper bet at the progression stake: base x progression^losses-in-a-row, back to base after a win.
        It never gives up unless AUTO_LADDER_MAX_USD is set: then back to base when the next stake would pass it.
        P&L twice: at the payout the page showed (how the MHH dashboard counts it), and pool-adjusted (pnl): our
        stake added to our side's pool lowers our own payout, which matters once the ladder bets are big.
        `pools` (the "thin" model): both from the round's final pools (see _pnl)."""
        base = self.cfg.auto_stake_usd
        stake = base * self.progression ** lad["step"]
        won = side == rec.get("win")
        pay = 1 + self._pnl(rec, side, pools=pools) / base
        lad["pnl_shown"] = round(lad.get("pnl_shown", 0.0) + stake * (pay - 1), 4)
        lad["flat"] = round(lad.get("flat", 0.0) + self._pnl(rec, side, pools=pools), 4)   # the same calls at the base stake
        lad["bets"] = lad.get("bets", 0) + 1
        lad["won"] = lad.get("won", 0) + int(won)
        if won and pools and (exact := _pool_pay(rec, side, stake)):
            pay = exact
        elif won:
            pay = _diluted(pay, stake, rec.get("pools"), side)
        lad["pnl"] = round(lad["pnl"] + stake * (pay - 1), 4)
        lad["staked"] = round(lad["staked"] + stake, 4)
        lad["biggest"] = max(lad["biggest"], round(stake, 2))
        lad["step"] = 0 if won else lad["step"] + 1
        cap = self.cfg.auto_ladder_max_usd or self.stake_cap()                 # the same series as the real orders
        if cap and base * self.progression ** lad["step"] > cap + 1e-9:
            lad["step"] = 0
            lad["gave_up"] = lad.get("gave_up", 0) + 1

    def _run_step(self, name: str, won: bool, t: float) -> None:
        """Losing runs of one model: how many runs of each length (a run ends at the next win), and the current one.
        Kept twice: over all logged rounds (runs) and over today's rounds only (runs_today, reset at midnight)."""
        self._run_into(self.runs, name, won, t)
        self._run_day(name, won, t)

    def _run_day(self, name: str, won: bool, t: float) -> None:
        """Today's losing runs and every earlier day's longest one.  'P&L -> 0' does not cut these: they are counted
        from the round log by calendar day."""
        lt = time.localtime()
        day0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        if day0 != self._runs_day0:
            if self._runs_day0 and self.runs_today:            # midnight passed while running: file the day away
                self.runs_days[time.strftime("%Y-%m-%d", time.localtime(self._runs_day0))] = {
                    k: {"current": 0, "bets": r["bets"], "longest": max([r["current"], *r["lengths"]])}
                    for k, r in self.runs_today.items()}
            self.runs_today, self._runs_day0 = {}, day0
        if t >= day0:
            self._run_into(self.runs_today, name, won, t)
            return
        d = self.runs_days.setdefault(time.strftime("%Y-%m-%d", time.localtime(t)), {}).setdefault(
            name, {"current": 0, "longest": 0, "bets": 0})
        d["bets"] += 1
        d["current"] = 0 if won else d["current"] + 1
        d["longest"] = max(d["longest"], d["current"])

    @staticmethod
    def _run_into(table: dict, name: str, won: bool, t: float) -> None:
        r = table.setdefault(name, {"lengths": {}, "current": 0, "bets": 0, "first": t, "last": t})
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
        self._ob_roll(rec)
        s = self.stats
        if replay and rec.get("result"):
            rec["pnl"] = self._pnl(rec, rec.get("call"))
        win, result = rec.get("win"), rec.get("result")
        if (rec.get("live") or {}).get("ok") and result:      # a settled real order, for the page's P&L graph; real
            self.real_pts.append((self._rec_time(rec), float(rec.get("live_pnl", rec.get("pnl")) or 0.0)))   # money is not cut by 'P&L -> 0'
        if self._rec_time(rec) < self.stats_since:  # before 'P&L -> 0': only the gate's forward records count it
            if win and not rec.get("tie"):
                for name, side in self._model_calls(rec).items():
                    if side and self._rec_time(rec) >= MODEL_REGISTERED.get(name, 0):
                        s[f"{name}_fwd_called"] = s.get(f"{name}_fwd_called", 0) + 1
                        s[f"{name}_fwd_correct"] = s.get(f"{name}_fwd_correct", 0) + int(side == win)
                    if side:
                        self._run_day(name, side == win, self._rec_time(rec))
            return
        s["rounds"] += 1
        if win:
            self.winners.append("T" if rec.get("tie") else win[0])
        if rec.get("bfx_ret") is not None:
            s["bfx_n"] += 1
            s["bfx_agree"] += int((rec["bfx_ret"] > 0) == (win == "UP"))
        fm, mm = rec.get("fair_move"), rec.get("mix_move")      # the new price against our price on the same rounds
        if (self.fair_v and rec.get("fair_v") == self.fair_v and rec.get("mix_v") == MIX_VERSION and win
                and not rec.get("tie") and fm and mm):
            s["cmp_n"] = s.get("cmp_n", 0) + 1
            s["cmp_fair_ok"] = s.get("cmp_fair_ok", 0) + int((fm > 0) == (win == "UP"))
            s["cmp_mix_ok"] = s.get("cmp_mix_ok", 0) + int((mm > 0) == (win == "UP"))
        for prefix, field in (("our", "our_move"), ("mix", "mix_move"), ("fair", "fair_move")):   # did the price
            om = rec.get(field)                                                    # give bc.game's Up/Down?
            if (om is None or not win or rec.get("tie") or (prefix == "mix" and rec.get("mix_v") != MIX_VERSION)
                    or (prefix == "fair" and (not self.fair_v or rec.get("fair_v") != self.fair_v))):
                continue                            # (rounds of an earlier mix / recipe are not counted for this one)
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
            for name, side in self._model_calls(rec).items():
                if side:
                    s[f"{name}_called"] = s.get(f"{name}_called", 0) + 1
                    s[f"{name}_correct"] = s.get(f"{name}_correct", 0) + int(side == win)
                    bet = self._pnl(rec, side, diluted=name == "pool", pools=name in POOL_BOOKED)
                    s[f"{name}_pnl"] = round(s.get(f"{name}_pnl", 0.0) + bet, 4)
                    ts, ps = self.curves.setdefault(name, (array("d"), array("d")))     # the same bet, for the graph
                    ts.append(self._rec_time(rec))
                    ps.append(bet)
                    if self._rec_time(rec) >= MODEL_REGISTERED.get(name, 0):     # the forward record
                        s[f"{name}_fwd_called"] = s.get(f"{name}_fwd_called", 0) + 1
                        s[f"{name}_fwd_correct"] = s.get(f"{name}_fwd_correct", 0) + int(side == win)
                    self._run_step(name, side == win, self._rec_time(rec))
                    if name in self.ladders and self._rec_time(rec) >= self.since:
                        self._ladder_step(self.ladders[name], rec, side, pools=name in POOL_BOOKED)
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

    def _model_calls(self, rec: dict) -> dict:
        """What every model called in a settled round (None = no call), from the record."""
        calls = {"revert": _revert_call(rec.get("feat")),
                 "overshoot": _overshoot_call(rec.get("feat")),
                 "prob": ProbModel.side(rec.get("prob")),
                 "ens": ProbModel.side(rec.get("prob_ens")),
                 "overshoot_book": _overshoot_call(rec.get("feat"), with_book=True),
                 "overshoot_rel": _overshoot_rel_call(rec.get("feat")),
                 "fade": _fade_call(rec.get("feat")),
                 # the new price: only rounds read with the recipe that is loaded now (a refit starts its record again)
                 "fade_fair": (_fade_fair_call(rec.get("feat")) if self.fair_v
                               and (rec.get("feat") or {}).get("fair_v") == self.fair_v else None),
                 "fade_strict": _fade_strict_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["fade_strict"] else None,
                 # the tree model was fitted on the older rounds: only rounds after that count
                 "trees_mid": _treesmid_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["trees_mid"] else None,
                 "fade_flow": _fade_flow_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["fade_flow"] else None,
                 "fade_flow2": _fade_flow2_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["fade_flow2"] else None,
                 "fade_hawkes": (_fade_hawkes_call(rec.get("feat"), rec.get("pred"))
                                 if self._rec_time(rec) >= MODEL_REGISTERED["fade_hawkes"] else None),
                 # the payout rule was chosen on the rounds before its time: only later rounds count
                 "thin": _thin_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["thin"] else None,
                 # fitted and chosen on the rounds before its time: only later rounds count
                 "value": _value_call(rec.get("feat")) if self._rec_time(rec) >= MODEL_REGISTERED["value"] else None,
                 "pool": (rec.get("pool") or _pool_call(rec.get("payouts")) or {}).get("call")}
        ob = rec.get("ob") or {}
        for name, key in OB_ANSWER.items():
            if name in OB_FITTED:   # the fitted ones: only rounds answered by the version that is in place now
                counted = ob.get("v") and ob.get("v") == self.ob_v
            else:                   # HawkesSign: the six-market forecast, and no round from before it
                counted = self._rec_time(rec) >= MODEL_REGISTERED[name]
            calls[name] = _ob_side((ob.get("p") or {}).get(key), (ob.get("thr") or {}).get(key)) if counted else None
        return calls

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
            if t < self.run_reset_at:                 # before a typed 'live reset': not counted
                continue
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
        frac = min(1.0, max(0.0, frac))              # time used; the bar shows the time LEFT, so it empties like a timer
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
            f'<div class="big">{big}<small>{small}</small></div><div class="pbar"><i style="width:{(1 - frac if cls else 1.0) * 100:.0f}%"></i></div>'
            + row("bc.game", f"{bc_now:,.2f}" if bc_now else "&ndash;")
            + row("our price", f"{ours:,.2f}" if ours else "&ndash;")
            + row("start", f"{start:,.2f}" if start and status != BETTING else "&ndash;")
            + row("pool UP / DN", f"${up_pool:,.0f} / ${dn_pool:,.0f}" if up_pool or dn_pool else "&ndash;")
            + (f'<div class="sq" title="last rounds, newest left">{winners}</div>' if winners else "") + '</div>')

        # ---- the call: what the BET card says before / without a call ----
        call, p = rec.get("call"), rec.get("pred") or {}
        lead_s = self.cfg.auto_lead_ms / 1000
        if not self.on:
            sub = "auto is off"
        elif call:
            if self.model in ("fade", "overshoot_rel", "fade_strict", "fade_fair", "fade_flow", "fade_hawkes", "fade_flow2"):
                f_ = rec.get("feat") or {}
                vol_ = max(f_.get("vol60") or 0, FADE_VOL_FLOOR if self.model == "fade_strict" else 1e-9)
                gap_ = f_.get("dev_fair" if self.model == "fade_fair" else "dev_tick") or 0
                sub = (f"bc.game {gap_:+.2f} $ from the {'new price' if self.model == 'fade_fair' else 'market'} &middot; "
                       f"{abs(gap_) / vol_:.1f} x volatility"
                       + (f" &middot; HawkesSign P(up) {p.get('sign_p_up', 0):.2f}" if self.model == "fade_hawkes" else ""))
            elif self.model == "thin":
                f_ = rec.get("feat") or {}
                sub = f"the smaller pool: the page pays x{max(f_.get('pay_up') or 0, f_.get('pay_dn') or 0):.2f} on it"
            elif self.model == "value":
                f_ = rec.get("feat") or {}
                vr = _value_read(f_)
                sub = (f"chance {vr[1]:.0%} &times; pays x{f_['pay_up' if vr[0] == 'UP' else 'pay_dn']:.2f} = {vr[2]:+.0%} "
                       f"expected &middot; gap {abs(f_['dev_tick']) / f_['vol60']:.1f} x volatility" if vr else "chance x payout")
            elif self.model in OB_MODELS:
                ob_ = rec.get("ob") or {}
                p_ = (ob_.get("p") or {}).get(OB_ANSWER[self.model])
                sub = ((f"P(UP) {p_:.3f}" if isinstance(p_, float) else "P(UP) ?")
                       + (f" &middot; fit {ob_.get('v')}" if self.model in OB_FITTED else
                          f" &middot; {len((ob_.get('hk') or {}).get('per') or {})} of 6 order books"))
                if self.model == "ob_steady":
                    sub = _steady_text(ob_.get("life"))
            elif self.model in ("prob", "ens"):
                key = {"prob": "prob", "ens": "prob_ens"}[self.model]
                mg = ProbModel.side.__defaults__[0]
                sub = (f"P(UP) {rec[key]:.2f} (calls outside {0.5 - mg:.2f}-{0.5 + mg:.2f})"
                       if isinstance(rec.get(key), float) else "P(UP) models " + ("ready" if self.prob.model else "not fitted yet"))
            else:
                sub = f"HawkesSign P(up) {p.get('sign_p_up', 0):.2f}"
        elif self.model == "fade_hawkes" and not self.pred.ready:
            sub = f"warming up: order-book data {self.pred.data_span():.0f} / 300 s"
        elif self.model in OB_MODELS and rec.get("skip"):
            sub = html.escape(str(rec["skip"]))
        elif status == BETTING:
            sub = f"call {lead_s:g} s before bets close"
        else:
            sub = "no call this round"

        # ---- the order books: the five markets' best-price reading, its age, and what it was at the decision ----
        ob_card = self._ob_card()

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
            exact = (_pool_pay(rec, call, stake) if rec.get("by") in POOL_BOOKED and result == "WIN" and not lv.get("ok")
                     else None)                       # thin's and value's paper books: the final pools with our stake in them
            if exact:
                note = f"booked x{exact:.2f} at the final pools on ${amt(stake)} &rarr; {usd(stake * (exact - 1))}"
            elif result == "WIN" and gain is not None:
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
                xpay = 1 + x.get("pnl", 0.0) / base
                if x["result"] == "WIN" and xpay > 1:      # as the P&L counts it: our own stake lowers the payout
                    xpay = ((x.get("by") in POOL_BOOKED and _pool_pay(x, x["call"], xstake))  # thin, value: the final pools,
                            or _diluted(xpay, xstake, x.get("pools"), x["call"]))            # as its ladder books it
                money = (f'{xstake * (xpay - 1):+.2f} <span class="tag ptag">${xstake:g}</span>'
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
                  + ("compensate " + "-".join(f"{x * c.auto_stake_usd:g}" for x in self._real_ladder()) + " ON"
                     if self.ladder_on() else "flat" + (" (ladder waits for the gate)" if len(c.auto_real_ladder) > 1 else ""))
                  + '</div>'
                  f'<div class="mute">real losses in a row <b>{self.real_run}</b> / {c.auto_max_real_run} &middot; '
                  f'orders today {self.attempts_today}{f" / {c.auto_max_real_bets_day}" if c.auto_max_real_bets_day else ""} &middot; '
                  f'budget left ${self.budget_left():.2f}</div>'
                  f'<div class="mute">campaign {usd(self.campaign_pnl)} (stop -${c.auto_campaign_loss_usd:g}) &middot; '
                  f'balance {f"${self.balance_usd:,.2f}" if self.balance_usd is not None else "&ndash;"} &middot; '
                  f'clock {"bc " + f"{clk["offset"]:+.2f}s" if clk["ok"] else "exchange (" + e(clk["why"] or "?") + ")"}</div>'
                  + (f'<div class="mute">shadow: would-bet {self.shadow_attempts_today} today &middot; run {self.shadow_run} '
                     f'&middot; longest {self.shadow_longest_today} &middot; stops {self.shadow_stops_today}</div>'
                     if c.auto_shadow and not self.armed else ""))
        pnl_card = (
            f'<div class="c pnlc"><div class="ct"><span>P&amp;L &middot; {plabel}</span><span class="mute">since {since}</span></div>'
            f'<div class="two mute"><span>paper P&amp;L</span><span>bets &middot; won</span></div>'
            f'<div class="two"><b class="pnl {"pos" if pnl >= 0 else "neg"}">{usd(pnl)}</b>'
            f'<b class="pnl">{bets} &middot; {won}</b></div>'
            f'<div class="mute">before our stake lowers the payout {usd(lad.get("pnl_shown", 0.0))} &middot; flat $1{usd(lad.get("flat", 0.0))}'
            + (f' &middot; won {won / bets:.1%}' if bets else "") + '</div>'
            f'<div class="ph" style="margin-top:3px">next bet ${amt(nxt_after)} &middot; losses in a row {step} &middot; '
            f'run lost ${amt(run_lost)} &middot; longest {longest}</div>'
            f'<div class="mute">{real} &middot; this run {usd(self.arm_live_pnl)} ({self.arm_orders} orders) &middot; '
            f'real today {usd(self.day_live_pnl)} (limit -${self.cfg.auto_daily_loss_usd:g})</div>'
            f'{safety}'
            f'<div class="sq" title="our last calls, newest left">{strip}</div></div>')

        # ---- MODELS: every model's record (flat $1 per call, all logged rounds) ----
        def model_row(name, label):
            n, ok = s.get(f"{name}_called", 0), s.get(f"{name}_correct", 0)
            mark = " &#9664;" if self.model == name else ""
            return row(label + mark, f"{ok / n:.1%} of {n} &middot; {s.get(f'{name}_pnl', 0.0):+.0f}" if n else "&ndash;")

        mix_n, mix_ok = s.get("mix_n", 0), s.get("mix_ok", 0)
        models_card = (
            '<div class="c modc"><div class="ct"><span>MODELS</span><span class="mute">right &middot; flat $1 P&amp;L</span></div>'
            + "".join(model_row(name, {"trees_mid": "Trees, medium", "overshoot_book": "Overshoot+book",
                                       "fade_fair": "Fade, new price", "ob_logit": "OB logistic", "ob_lgbm": "OB LightGBM",
                                       "ob_xgb": "OB XGBoost", "ob_deeplob": "OB DeepLOB"}.get(name, label))
                      for name, label in self.MODELS.items())
            + row("our price wrong", f"{(mix_n - mix_ok) / mix_n:.1%} of {mix_n}" if mix_n else "&ndash;") + '</div>')
        lab_card = lab_panel_card(read_lab_status(self.cfg.auto_log.parent))
        return (f'{PANEL_CSS}<div class="p"><div class="hd"><span>BC BOT &middot; ROUND #{rid or "&ndash;"}</span>{mode}</div>'
                f'{round_card}{ob_card}{bet_card}{results_card}{pnl_card}{lab_card}{models_card}</div>')

    def curve_data(self) -> dict:
        """What the page needs to draw our price on bc.game's chart: our last 75 s, the last ticks (the page finds
        the chart's last tick among them by price), and the time of the newest tick."""
        now = self.bc_now()
        ticks = [[t, p] for t, p in list(self.bc_ticks)[-40:]]
        rounds = [{"start": rec["start"], "end": rec["end"], "win": rec.get("win"), "tie": rec.get("tie", False)}
                  for rec in list(self.recs.values()) if rec.get("start") and rec.get("end", 0) >= now - 65]
        r = self.round or {}                      # the round the page shows now, even before it has a record
        if r.get("priceStartTime") and r.get("priceEndTime"):
            cur = {"start": r["priceStartTime"] / 1000, "end": r["priceEndTime"] / 1000, "win": None, "tie": False}
            if not any(abs(x["start"] - cur["start"]) < 0.01 for x in rounds):
                rounds.append(cur)
        return {"on": True, "our": self.our.series(now - 75, now), "ticks": ticks,
                "t_last": ticks[-1][0] if ticks else now, "rounds": rounds}

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
            "fair": self._fair_series(cut, now),      # the separate new price (None: no recipe)
            "fair_info": self.fair_info(),
            "fair_problem": self.fair_problem,        # a recipe file is there and was refused at the start: why
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
            "our_vs_bc": {f"{p}{k}": self.stats.get(f"{p}{k}", 0) for p in ("our", "mix", "fair") for k in
                          ("_n", "_ok", "_flat", "_small_n", "_small_ok", "_mid_n", "_mid_ok", "_big_n", "_big_ok")},
        }

    PNL_RANGES = {"1h": 3600.0, "6h": 21600.0}        # the graph's ranges by the clock; "today" and "all" besides

    def pnl_curve(self, rng: str = "all", model: str | None = None, max_pts: int = 1200) -> dict:
        """The page's P&L graph: one model's paper bets (the calling model unless another is named) as a running
        total, and the real orders of the same time as a second one; both start at 0 at the left edge.  The paper
        bets are the ones its row in the Models table counts, at the same amounts: with "all" (since the row
        began) the line ends at the row's P&L.  rng: "1h", "6h", "today" (since this PC's midnight, like the losing
        runs) or "all".  The real orders are every real order of that time as the bot booked it, whichever model
        placed them (the round log does not name the calling model)."""
        name = model if model in self.MODELS or model == "pool" else self.model
        now = time.time()
        ts, ps = self.curves.get(name) or ((), ())
        if rng in self.PNL_RANGES:
            cut = now - self.PNL_RANGES[rng]
        elif rng == "today":
            lt = time.localtime(now)
            cut = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        else:
            rng, cut = "all", (ts[0] - 1.0 if len(ts) else now)
        i = bisect.bisect_left(ts, cut)
        real = sorted(p for p in self.real_pts if p[0] >= cut)
        return {"model": name, "range": rng, "from": round(cut, 1), "now": round(max(now, cut), 1),
                "paper": _running(ts[i:], ps[i:], max_pts),
                "real": _running([p[0] for p in real], [p[1] for p in real], max_pts)}

    def _fair_series(self, cut: float, now: float) -> list | None:
        """The new price's curve for the chart.  An error in it must not take the rest of the chart's answer along."""
        if self.fair is None:
            return None
        try:
            return self.fair.series(cut, now)
        except Exception as e:                                # noqa: BLE001
            if not self._fair_warned:
                self._fair_warned = True
                log.warning("new price curve: %r (said once)", e)
            return []

    def fair_info(self) -> dict | None:
        """The separate new price for the page: its recipe, its feeds, and how it did against our price on the rounds
        both have (same Up/Down as bc.game).  None while there is no recipe."""
        r = self.fair_recipe
        if not r:
            return None
        s = self.stats
        return {"version": r["version"], "registered": r["registered"], "fitted_utc": r.get("fitted_utc"),
                "level_window": r["level_window"],
                # live: the socket is open (as for our price's chips); fresh: it has a price a snapshot would use
                "mix": {k: {"label": self.fair_feeds[k].name, "weight": w, "lag": lag,
                            "live": bool(self.fair_feeds[k].connected),
                            "fresh": self.fair_feeds[k].age() <= STALE_SNAPSHOT} for k, (w, lag) in r["mix"].items()},
                "same_rounds": {"n": s.get("cmp_n", 0), "new_ok": s.get("cmp_fair_ok", 0),
                                "current_ok": s.get("cmp_mix_ok", 0)}}

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
            "startup_problem": getattr(self, "startup_problem", None),
            # ---- real money: the state machine and every counter it is judged by ----
            "state": self.state, "stop_reason": self.stop_reason, "tech_stop": self.tech_stop,
            "stop_text": self.stop_text(), "stopped_at": self.stopped_at or None,
            "armed_at": self.armed_at or None, "rearm_grant": self.rearm_grant, "rearms": self.rearms,
            "real_run": self.real_run, "max_real_run": c.auto_max_real_run,
            "real_run_longest_today": self.real_run_longest_today,
            "attempts_today": self.attempts_today, "max_real_bets_day": c.auto_max_real_bets_day,
            "today_live_pnl": round(self.day_live_pnl, 4), "daily_loss_usd": c.auto_daily_loss_usd,
            "arm_live_pnl": round(self.arm_live_pnl, 4), "arm_orders": self.arm_orders,
            "budget_left": round(self.budget_left(), 4),
            "campaign_pnl": round(self.campaign_pnl, 4), "campaign_since": self.campaign_since or None,
            "run_reset_at": self.run_reset_at or None,
            "campaign_loss_usd": c.auto_campaign_loss_usd, "arm_ttl_h": c.auto_arm_ttl_h,
            "balance": self.balance, "balance_currency": self.balance_currency, "balance_usd": self.balance_usd,
            "balance_age": round(time.time() - self.balance_at, 1) if self.balance_at else None,
            "balance_account": self.balance_account, "dialog": self.dialog,
            "clock": self.clock.dict(), "clock_src": self.clock_src(),
            "pending": list(self.pending_real), "unknown": self.unknown, "last_order": self._last_order,
            "events": list(self.events), "gate": self.gate_status(), "break_even": self.BREAK_EVEN,
            "live_allowed": c.live_allowed, "tos_risk_accepted": c.tos_risk_accepted,
            "real_ladder": list(self._real_ladder()), "ladder_on": self.ladder_on(), "next_real_stake": self.stake(),
            "max_stake_usd": self.stake_cap(), "stake_usd": c.auto_stake_usd,
            "ladder_forced": self.ladder_forced,
            "shadow": {"on": c.auto_shadow, "run": self.shadow_run, "attempts_today": self.shadow_attempts_today,
                       "longest_today": self.shadow_longest_today, "stops_today": self.shadow_stops_today,
                       "last": rec.get("shadow")},
            "dq": (rec.get("feat") or {}).get("dq"),
            "progression": self.progression, "next_stake": self.stake(), "ladders": self.ladders,
            "since": self.since, "max_usd": c.max_usd, "ladder_cap": c.auto_ladder_max_usd or None,
            "model": self.model,
            # each model's losing runs over all logged rounds: {"lengths": {run length: how many}, "current", ...}
            "runs": self.runs,
            "runs_today": self.runs_today,
            "runs_days": {d: self.runs_days[d] for d in sorted(self.runs_days)[-7:]},
            "feed": {"connected": self.feed.connected, "msgs": self.feed.msgs, "data_s": round(self.pred.data_span()),
                     "events": len(self.feed.ev_time), "mid": self.feed.mid},
            "models_ready": self.pred.ready, "models": self.pred.info,
            "accuracy": round(s["correct"] / s["called"], 4) if s["called"] else None,
            "sign_accuracy": round(s["sign_correct"] / s["sign_called"], 4) if s["sign_called"] else None,
            "bitfinex_agrees_with_bc": round(s["bfx_agree"] / s["bfx_n"], 4) if s["bfx_n"] else None,
            "records": {name: {"called": s.get(f"{name}_called", 0), "correct": s.get(f"{name}_correct", 0),
                               "pnl": s.get(f"{name}_pnl", 0.0)}
                        for name in (*self.MODELS, "pool")},
            "ob": {**self.ob_info, "counted_version": self.ob_v},
            "prob_model": self.prob.info(), "fair_price": self.fair_info(),
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
                f"orders today {self.attempts_today}{f'/{c.auto_max_real_bets_day}' if c.auto_max_real_bets_day else ''}, real today {self.day_live_pnl:+.2f}, "
                f"budget left ${self.budget_left():.2f} | campaign {self.campaign_pnl:+.2f} (stop -${c.auto_campaign_loss_usd:g}) | "
                f"balance {f'${self.balance_usd:.2f}' if self.balance_usd is not None else '?'} "
                f"({self.balance_account or '?'}) | clock {'bc ' + f'{clk["offset"]:+.2f}s' if clk['ok'] else 'exchange: ' + str(clk['why'])} | "
                f"gate {'OPEN' if gate['open'] else 'closed'}: {gate['text']} | "
                f"LIVE_ALLOWED={int(c.live_allowed)} TOS_RISK_ACCEPTED={int(c.tos_risk_accepted)} shadow={int(c.auto_shadow)}")
