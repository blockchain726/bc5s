"""Tap analyzer (paper only): bc.game's Tap game for BTC/USD.

Tap: the page shows a grid of price bands (rows: $10 for BTC/USD) x time windows (columns: 2 s).  A tap on a cell wins
if bc.game's BTC/USD price touches that band during that window, and pays the cell's multiplier.  The multipliers are
bc.game's (Detrade's) own model, computed in the page by a web worker (ported below): the log price makes one normal
step every 0.5 s with a fixed per-symbol volatility, P(hit) = the chance a step lands in the band during the window,
and multiplier = 0.97 / P(hit) (3 % for the house), at least x1.02, at most the symbol's maxOdds (x8 for BTC/USD).

The real price doesn't have to move like that.  This module keeps a Tap tab open on BTC/USD in the bot's own Edge and,
without placing any order:
  * rebuilds the grid every column (2 s) with bc.game's multipliers, our P(hit) from the price's real recent
    volatility, and the expected value EV = our P(hit) x multiplier
  * taps every cell virtually and settles it against the real price ticks: the real hit rate and return of each kind
    of cell (how many columns ahead, how many bands from the price) - does any kind pay more than it costs?
  * paper-taps $1 on the best-EV cell each column when its EV is at least EV_MIN
  * records every player's BTC/USD taps and results from the page's feed: their real win rate vs the multiplier, and
    a check of our port (our multiplier for their tap vs the one they got) and of our hit rule (vs their result)
"""
import asyncio
import json
import logging
import math
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from .auto import OVERLAY_JS, PANEL_CSS               # the same left-bar panel as on the Up/Down page
from .tap_live import TapLive

log = logging.getLogger(__name__)

TAP_URL = "https://bc.game/trading/tap"
SYMBOL = "BTC/USD"
TICK_CMD = "/kline/BTC-USD/ticker"
# bc.game's settings for BTC/USD (read again from the page's tapConfig/list when it loads)
DEFAULT_CFG = {"timeIncrement": 2, "priceIncrement": 10.0, "volatility": 6.628e-05, "maxOdds": 8.0,
               "orderGridInterval": 2}
STEP_MS = 500                 # the model's step
N_POINTS = 100                # the model's price grid
FEE = 0.03                    # multiplier = (1 - FEE) / P(hit)
COLS = 6                      # columns evaluated from the next one (k = 0 .. 5)
ROWS = 6                      # bands either side of the price's band (r = -6 .. +6)
PAYOUT_FALLBACK = 0.68        # bc.game's payout / the grid label when there is no live book to measure it from
EV_MIN = 1.35                 # paper-tap the best cell only when its measured EV is at least this: the same bar a
                              # live tap must clear (TAP_MIN_EV 1.15 / the assumed 0.85 payout), so paper rehearses live
VOL_WINDOW_S = 600            # real volatility from this many seconds of ticks
EV_BUCKETS = (0.0, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, float("inf"))
ODDS_BUCKETS = (1.0, 1.5, 2.0, 3.0, 5.0, 7.99, float("inf"))
STEPS_USD = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)   # band sizes bc.game uses; the one on screen is inferred
ODDS_READ_MS = 1000           # bc.game's multiplier labels drawn within this many ms are the grid on screen now
CALIB_MIN_N = 1000
REGIME_FACTOR = {("calm", "hot"): 1.20, ("calm", "cold"): 1.04, ("wild", "hot"): 0.94, ("wild", "cold"): 0.76}
# (wild+cold lost at the server's ~70% payout in history and tonight; the user chose to keep tapping in it at a
#  small stake with TAP_MIN_EV at break-even, to see every order go out)
REGIME_RANGE_BANDS = 4.0      # 60 s price range in band units: below = calm
REGIME_HOT = 0.45             # trailing hit rate of the last 60 settled current-band taps: at least = hot
TRAILING_PRIOR_N = 30         # the long-run rate counts as this many taps against the trailing 60 (shrinkage)            # virtual taps of a cell kind before its realized hit rate calibrates the model
STEP_CONFIRM_S = 10          # a band size read off the labels that differs from the settings must hold this long


# ---------- candidate pickers, tested side by side on paper: each its own book and its own colour on the page ----
def _range_under(ctx, bands: float) -> bool:
    rb = ctx["regime"].get("range_bands")
    return rb is not None and rb < bands


def _calm(ctx) -> bool:
    return _range_under(ctx, REGIME_RANGE_BANDS)


def _score_label(c, ctx):
    return c["ev_emp"]                                    # measured hit rate x the grid label (regime factor in)


def _score_paid(c, ctx):
    return c["ev_emp"] * ctx["pay"](c["k"])               # ... x what bc.game's server really pays of the label


MODELS = (
    {"name": "label", "color": "#facc15", "bar": None, "score": _score_label,
     "desc": "the live picker: best EV at the grid label, the live bar (feeds the paper book)"},
    {"name": "paid", "color": "#a3e635", "bar": 1.35, "score": _score_paid,
     "desc": "best EV at the paid odds (hit rate x label x payout) >= 1.35"},
    {"name": "paid-lo", "color": "#f472b6", "bar": 1.15, "score": _score_paid,
     "desc": "EV at the paid odds >= 1.15: more taps, a thinner edge"},
    {"name": "paid-calm", "color": "#22d3ee", "bar": 1.35,
     "score": lambda c, x: _score_paid(c, x) if _calm(x) else None,
     "desc": "paid >= 1.35, only while the 60 s range is under 4 bands"},
    {"name": "paid-far", "color": "#fb923c", "bar": 1.35,
     "score": lambda c, x: _score_paid(c, x) if c["k"] >= 3 else None,
     "desc": "paid >= 1.35, three or more columns ahead (two ahead pays least)"},
    {"name": "calm-label", "color": "#e879f9", "bar": None,
     "score": lambda c, x: _score_label(c, x) if _calm(x) else None,
     "desc": "the live picker, only while the 60 s range is under 4 bands"},
    {"name": "big-label", "color": "#60a5fa", "bar": 1.35,
     "score": lambda c, x: _score_paid(c, x) if c["odds"] >= 3.5 else None,
     "desc": "paid >= 1.35, only cells whose label is 3.5 or more"},
    # measured 2026-11-02 on 3,500 label picks: at $5 bands the picks lose 2-6% per $ while the 60 s range is under
    # 6 bands and 21-52% above it; at $2 bands they lose at every range
    {"name": "paid-r6", "color": "#2dd4bf", "bar": 1.35,
     "score": lambda c, x: _score_paid(c, x) if _range_under(x, 6.0) else None,
     "desc": "paid >= 1.35, only while the 60 s range is under 6 bands"},
    {"name": "label-r6", "color": "#c084fc", "bar": None,
     "score": lambda c, x: _score_label(c, x) if _range_under(x, 6.0) else None,
     "desc": "the live picker, only while the 60 s range is under 6 bands"},
)
STATS_VERSION = 3             # bumping it resets the EV and paper tables (they depend on how EV is computed)
MIN_AHEAD_MS = 4300           # players never tap a window starting < 4.04 s ahead (0 of 133): ours need 4.3 s

# Where is a time / a price on bc.game's grid?  The grid is a canvas, so the page is watched while it draws its axis
# labels (times "03:06:12" along the bottom, prices "78,010.00" on the right): their screen positions give the
# time -> x and price -> y lines our boxes are drawn with.
GEOM_HOOK_JS = r"""(() => {
  if (window.__tapHook) return;
  window.__tapHook = true;
  const g = window.__tapGeom = {times: {}, prices: {}, rects: new Map(), odds: [], texts: new Map()};
  const orig = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (text, x, y, ...rest) {
    try {
      const s = String(text).trim();
      const kind = /^\d{2}:\d{2}:\d{2}$/.test(s) ? 't' : /^\d{1,3}(,\d{3})*\.\d+$/.test(s) ? 'p'
                 : /^(x\d+(\.\d+)?|\d+(\.\d+)?x)$/i.test(s) ? 'o' : null;
      if (kind || s.length <= 16) {
        const m = this.getTransform(), c = this.canvas, r = c.getBoundingClientRect();
        const sx = r.width / c.width, sy = r.height / c.height;
        const px = m.a * x + m.c * y + m.e, py = m.b * x + m.d * y + m.f;
        const vx = r.left + px * sx, vy = r.top + py * sy, now = performance.now();
        if (kind === 'o') {                      // a cell's multiplier: keep the text box's top-left corner
          const w = this.measureText(s).width * m.a * sx;
          const fs = parseFloat((/(\d+(\.\d+)?)px/.exec(this.font) || [0, 12])[1]) * m.d * sy;
          const al = this.textAlign, bl = this.textBaseline;
          const left = vx - (al === 'center' ? w / 2 : (al === 'right' || al === 'end') ? w : 0);
          const top = vy - (bl === 'middle' ? fs / 2 : (bl === 'top' || bl === 'hanging') ? 0 : fs * 0.8);
          g.odds.push([parseFloat(s.replace(/x/ig, '')), left, top, now, Math.round(w), Math.round(fs)]);
          if (g.odds.length > 6000) g.odds.splice(0, 3000);
        } else if (kind) {
          (kind === 't' ? g.times : g.prices)[s] = [kind === 't' ? vx : vy, now, c];
        } else {                                 // other short texts, to see what the page draws (debug)
          g.texts.set(s, [Math.round(vx), Math.round(vy), Math.round(now)]);
          if (g.texts.size > 300) g.texts.delete(g.texts.keys().next().value);
        }
        g.rects.set(c, [r.left, r.top, r.width, r.height]);
      }
    } catch (e) {}
    return orig.call(this, text, x, y, ...rest);
  };
})()"""

# bc.game's multipliers as drawn on its grid right now: each label -> (time, price) through the same axis fits the
# boxes use, shifted a third of a cell right and down so the point is inside the label's cell whatever the text
# alignment; the spacing of the labels gives the band size on screen (bc.game changes it with volatility).
READ_JS = r"""(d) => {
  const g = window.__tapGeom, now = performance.now();
  if (!g) return {err: 'no hook'};
  const fresh = o => Object.entries(o).filter(([, v]) => now - v[1] < 1500);
  const T = fresh(g.times), P = fresh(g.prices);
  if (T.length < 2 || P.length < 2) return {err: 'labels ' + T.length + '/' + P.length};
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
    return {f: v => my + b * (v - mx), inv: s => b ? mx + (s - my) / b : NaN};
  };
  const X = fit(tx), Y = fit(py);
  let tmax = -1;
  for (const o of g.odds) if (now - o[3] < d.window_ms && o[3] > tmax) tmax = o[3];
  const seen = new Map();                                   // the newest frame only: the chart scrolls
  for (const o of g.odds) if (tmax - o[3] < 40) seen.set(Math.round(o[1]) + ',' + Math.round(o[2]), o);
  const items = [...seen.values()];
  const xs = [...new Set(items.map(o => Math.round(o[1])))].sort((a, b) => a - b);
  const ys = [...new Set(items.map(o => Math.round(o[2])))].sort((a, b) => a - b);
  const pitch = arr => {
    const dd = [];
    for (let i = 1; i < arr.length; i++) if (arr[i] - arr[i - 1] > 8) dd.push(arr[i] - arr[i - 1]);
    dd.sort((a, b) => a - b);
    return dd.length ? dd[Math.floor(dd.length / 2)] : 0;
  };
  const pxs = pitch(xs), pys = pitch(ys);
  const out = items.map(o => ({odds: o[0], x: Math.round(o[1]), y: Math.round(o[2]), w: o[4], h: o[5],
                               t: X.inv(o[1] + 0.35 * pxs), price: Y.inv(o[2] + 0.35 * pys)}));
  return {n: out.length, px: pxs, py: pys, step_usd: pys ? Math.abs(Y.inv(pys) - Y.inv(0)) : null,
          col_ms: pxs ? Math.abs(X.inv(pxs) - X.inv(0)) : null, items: out,
          texts: d.debug ? [...g.texts.entries()].map(([s, v]) => [s, v[0], v[1], Math.round(now - v[2])]) : null};
}"""


DRAW_JS = r"""(d) => {
  let host = document.getElementById('tap-cells');
  if (!host) {
    host = document.createElement('div');
    host.id = 'tap-cells';
    host.attachShadow({mode: 'open'});
    document.body.appendChild(host);
  }
  host.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;z-index:2147483646;pointer-events:none;';
  const g = window.__tapGeom, now = performance.now();
  if (!g) { host.shadowRoot.innerHTML = ''; return 'no hook'; }
  const fresh = o => Object.entries(o).filter(([, v]) => now - v[1] < 1500);
  const T = fresh(g.times), P = fresh(g.prices);
  if (T.length < 2 || P.length < 2) { host.shadowRoot.innerHTML = ''; return 'labels ' + T.length + '/' + P.length; }
  const ref = new Date(d.now);                         // the page labels ticks with the local time of their server ms
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
  const X = fit(tx), Y = fit(py), rT = g.rects.get(T[0][1][2]), rP = g.rects.get(P[0][1][2]);
  if (!rT || !rP) return 'no canvas';
  const L = Math.min(rT[0], rP[0]), Tp = Math.min(rT[1], rP[1]);   // the chart: the canvases the labels are on
  const W = Math.max(rT[0] + rT[2], rP[0] + rP[2]) - L, H = Math.max(rT[1] + rT[3], rP[1] + rP[3]) - Tp;
  let h = `<style>
    .wrap{position:fixed;overflow:hidden}
    .cell{position:absolute;box-sizing:border-box;border:1px solid rgba(143,153,165,.25)}
    .cell span{position:absolute;right:2px;bottom:1px;font:9px Consolas,monospace;color:#c3ccd8;opacity:.85}
    .good{border:1.5px solid rgba(34,210,122,.85)}.good span{color:#22d27a;opacity:1;font-weight:700}
    .off{opacity:.45}
    .pick{border:3px solid #facc15;box-shadow:0 0 8px rgba(250,204,21,.7)}
    .pick span{color:#facc15;opacity:1;font-weight:700;font-size:10px;background:rgba(11,14,19,.75);padding:0 2px}
    .mpick{border:2px solid #fff}
    .mpick span{color:inherit;opacity:1;font-weight:700;font-size:9px;background:rgba(11,14,19,.75);padding:0 2px;left:2px;right:auto;top:1px;bottom:auto}
    .hit{border:3px solid #22d27a;background:rgba(34,210,122,.18)}.miss{border:3px solid #ff5468;background:rgba(255,84,104,.14)}
    .hit span{color:#22d27a;font-weight:700;font-size:10px;opacity:1}.miss span{color:#ff5468;font-weight:700;font-size:10px;opacity:1}
  </style><div class="wrap" style="left:${L}px;top:${Tp}px;width:${W}px;height:${H}px">`;
  for (const c of d.cells) {
    const x0 = X(c.t0) - L, x1 = X(c.t1) - L, y0 = Y(c.hi) - Tp, y1 = Y(c.lo) - Tp;
    if (x1 < 0 || x0 > W || Math.max(y0, y1) < 0 || Math.min(y0, y1) > H) continue;
    const i = c.inset || 0, extra = c.color ? `border-color:${c.color};color:${c.color};` : '';
    h += `<div class="cell ${c.cls}" style="left:${(x0 + i).toFixed(1)}px;top:${(Math.min(y0, y1) + i).toFixed(1)}px;` +
         `width:${(x1 - x0 - 2 * i).toFixed(1)}px;height:${(Math.abs(y1 - y0) - 2 * i).toFixed(1)}px;${extra}"><span>${c.label}</span></div>`;
  }
  host.shadowRoot.innerHTML = h + '</div>';
  return 'ok';
}"""

SET_SYMBOL_JS = r"""(sym) => {
  const k = Object.keys(localStorage).find(k => /^trading-.*\.system$/.test(k));
  if (!k) return 'no store';
  const s = JSON.parse(localStorage.getItem(k));
  if (!s.state || !s.state.tap) return 'no tap state';
  if (s.state.tap.symbol === sym) return 'already';
  s.state.tap.symbol = sym;
  localStorage.setItem(k, JSON.stringify(s));
  return 'set';
}"""


# ---------- bc.game's multiplier model (port of the page's tap worker) ----------

def column(price: float, now_ms: float, t0_ms: float, gap_ms: float, vol: float):
    """One column's random walk (shared by all its cells): transition matrix and price grid."""
    step_end = max(1, math.floor((t0_ms + gap_ms - now_ms) / STEP_MS))
    y0 = math.log(price)
    y_min = y0 - 4 * vol * math.sqrt(step_end)
    dy = 8 * vol * math.sqrt(step_end) / (N_POINTS - 1)
    ys = y_min + dy * np.arange(N_POINTS)
    z = (ys[None, :] - ys[:, None]) / vol
    dens = np.exp(-0.5 * z * z)
    return dens / dens.sum(axis=1, keepdims=True), ys, y0, y_min, dy


def p_hit(col, now_ms: float, lo: float, hi: float, t0_ms: float, t1_ms: float) -> float:
    """P(the walk lands in [lo, hi] at a step between t0 and t1)."""
    trans, ys, y0, y_min, dy = col
    step_start = max(1, math.floor((t0_ms - now_ms) / STEP_MS))
    step_end = math.floor((t1_ms - now_ms) / STEP_MS)
    p = np.zeros(N_POINTS)
    p[min(N_POINTS - 1, max(0, math.floor((y0 - y_min) / dy)))] = 1.0
    band = (ys >= math.log(lo)) & (ys <= math.log(hi))
    for t in range(1, step_end + 1):
        p = p @ trans
        if t >= step_start:
            p[band] = 0.0
    return float(1.0 - p.sum())


def odds(p: float, max_odds: float) -> float:
    """bc.game's multiplier for a P(hit): the page's clamps, then capped at the symbol's maxOdds."""
    p = max(0.00009, min(0.999, p))
    return min(max(1.02, (1 - FEE) / p), max_odds)


def _bucket(v: float, edges) -> str:
    for a, b in zip(edges, edges[1:]):
        if a <= v < b:
            return f"{a:g}-{b:g}" if b != float("inf") else f"{a:g}+"
    return "?"


def touched(ticks: list, lo: float, hi: float, t0: float, t1: float) -> tuple[bool, bool]:
    """Did the price touch [lo, hi] between t0 and t1 (ms)?  (tick rule: a tick inside the band; cross rule: also two
    ticks in a row on opposite sides of it)."""
    tick = cross = False
    prev = None
    for t, p in ticks:
        if t > t1:
            break
        if t >= t0:
            if lo <= p <= hi:
                return True, True
            if prev is not None and (prev < lo < hi < p or p < lo < hi < prev):
                cross = True
        prev = p
    return tick, cross


class TapAnalyzer:
    def __init__(self, trader, log_dir: Path):
        self.trader = trader
        self.page = None
        self.cfg = dict(DEFAULT_CFG)
        self.cfg_from_page = False
        self.ticks: deque = deque(maxlen=4000)          # (server ms, price) of the Tap page's BTC/USD feed
        self._tick_local = 0.0
        self.connected = False
        self.grid: dict = {}                            # the last snapshot, for the dashboard
        self.pending: list = []                         # virtual taps waiting for their window to end
        self.paper_pending: list = []
        self.stats_path = log_dir / "tap_stats.json"
        self.band_hits: deque = deque(maxlen=200)         # (t1, hit) of settled current-band virtual taps
        self.regime = {"name": "unknown", "factor": 1.0, "trailing": None, "range_bands": None}
        self.paper_path = log_dir / "tap_paper.jsonl"
        self.models_path = log_dir / "tap_models.jsonl"   # settled picks of the candidate pickers (not the paper book)
        self.model_hour: dict = {}                         # model -> deque of (settled at, pnl) for the last hour
        self.public_path = log_dir / "tap_public.jsonl"
        self.stats = {"since": time.time(), "cells": {}, "ev": {}, "paper": {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0},
                      "paper_strict": {"bets": 0, "wins": 0, "pnl": 0.0}, "models": {},
                      "public": {"n": 0, "won": 0, "ret": 0.0, "odds": 0.0, "checked": 0, "agree_tick": 0,
                                 "agree_cross": 0, "odds_checked": 0, "odds_err": 0.0, "by_odds": {}}}
        self.paper_recent: deque = deque(maxlen=12)
        self.public_recent: deque = deque(maxlen=12)
        self.odds_ratio: deque = deque(maxlen=500)      # real taps' multiplier / ours for the same tap
        self._seen_public: set = set()
        self._col = None
        self._task = None
        self._saved = 0.0
        self.vol_real = None
        self.bc_odds: dict = {}                         # bc.game's multipliers read off its grid: (t0, lo) -> odds
        self.label_hist: dict = {}                      # the same, kept 10 min: (t0, lo) -> (odds, read at ms); a
                                                        # player's tap on a labelled cell measures the server's payout
        self.bc_read: dict = {"n": 0, "err": "not read yet", "at": 0.0}
        self.step_inferred = None                       # the band size on screen (bc.game changes it)
        self._step_candidate, self._step_since = None, 0.0
        self.calib: dict = {}                           # (k, r) -> realized hit rate / model P(hit)
        self._debug = True
        self.debug_texts = None
        self.sym_cfg: dict = {}                         # the symbol's raw settings (orderStatus, allowTradeTime, ...)
        self._reload_at = 0.0
        self._load()
        self.live = TapLive(self, log_dir)
        if getattr(self, "_rebook", False):
            self._rebook_paper()                           # the saved paper book was priced at the grid label

    # ---------- the Tap tab ----------

    async def start(self) -> None:
        await self._open()
        self._task = asyncio.create_task(self._loop())
        log.info("Tap analyzer on (%s, paper only): grid, virtual taps and paper taps -> http://127.0.0.1:8765/tap",
                 SYMBOL)

    async def _open(self) -> None:
        ctx = self.trader.page.context
        self._tick_local = time.time()                    # the "no price" timer starts now, not at 0
        old = [pg for pg in ctx.pages if "/trading/tap" in pg.url]   # a Tap tab left by an earlier run: reuse it
        for pg in old[1:]:
            await pg.close()
        self.page = old[0] if old else await ctx.new_page()
        self.page.on("websocket", self._on_ws)
        self.page.on("response", lambda r: asyncio.ensure_future(self._on_response(r)))
        await self.page.add_init_script(GEOM_HOOK_JS)       # before the page draws: where times / prices are on screen
        await self.page.goto(TAP_URL, wait_until="domcontentloaded")
        for _ in range(20):                               # the page creates its settings store on first load
            res = await self.page.evaluate(SET_SYMBOL_JS, SYMBOL)
            if res in ("set", "already"):
                break
            await asyncio.sleep(1)
        if res == "set":
            await self.page.reload(wait_until="domcontentloaded")
        try:                                              # keep the Up/Down page in front (the Tap tab runs behind it)
            await self.trader.page.bring_to_front()
        except Exception:
            pass
        log.info("Tap tab open on %s (%s)", SYMBOL, res)

    def _on_ws(self, ws) -> None:
        if "detrade" in ws.url:
            ws.on("framereceived", self._on_frame)

    def _on_frame(self, payload) -> None:
        try:
            raw = payload if isinstance(payload, (bytes, bytearray)) else payload.encode("latin-1")
            m = json.loads(zlib.decompress(raw).decode() if raw[:1] == b"x" else raw.decode())
        except Exception:
            return
        cmd, r = m.get("cmd"), m.get("resp")
        if cmd == TICK_CMD and isinstance(r, dict):
            try:
                self.ticks.append((int(r["t"]), float(r["p"])))
                self._tick_local = time.time()
            except (KeyError, TypeError, ValueError):
                pass
        elif cmd == "/tapOrder/end" and isinstance(r, dict):
            try:
                self.live.on_end(r)                       # one of our live / demo taps?
            except Exception as e:
                log.warning("Tap live settle: %r", e)
            if r.get("symbol") == SYMBOL:
                self._public(r)

    async def _on_response(self, r) -> None:
        path = r.url.split("?")[0]
        if path.endswith("/api/transaction/symbol/list") and "type=5" in r.url:   # Tap's symbols (5 = TapTrading)
            try:
                for c in (await r.json()).get("data") or []:
                    if c.get("symbol") == SYMBOL and "orderStatus" in c:
                        new = {k: c.get(k) for k in ("orderStatus", "allowTradeTime", "onlineStatus")}
                        if any(self.sym_cfg.get(k) != v for k, v in new.items()):
                            log.info("Tap %s: orderStatus %s, allowTradeTime %s, onlineStatus %s (%s)", SYMBOL,
                                     new["orderStatus"], new["allowTradeTime"], new["onlineStatus"],
                                     r.url.split("?")[-1][:60])
                        self.sym_cfg.update(new)
            except Exception as e:
                log.warning("Tap symbols: %r", e)
            return
        if not path.endswith("tapConfig/list"):
            return
        try:
            for c in (await r.json()).get("data") or []:
                if c.get("symbol") == SYMBOL:
                    self.sym_cfg.update({k: v for k, v in c.items() if not isinstance(v, (list, dict))})
                    self.cfg = {k: c.get(k, DEFAULT_CFG[k]) for k in DEFAULT_CFG}
                    self.cfg["orderGridInterval"] = self.cfg["orderGridInterval"] or 0
                    self.cfg_from_page = True
                    log.info("Tap %s settings from the page: %s", SYMBOL, self.cfg)
        except Exception as e:
            log.warning("Tap config: %r", e)

    # ---------- loop: a snapshot at each new column, settle what has ended ----------

    async def _loop(self) -> None:
        n = 0
        while True:
            await asyncio.sleep(0.1)
            n += 1
            try:
                self.connected = bool(self.ticks) and time.time() - self._tick_local < 10
                if self.ticks:
                    gap = self.cfg["timeIncrement"] * 1000
                    t_last, _ = self.ticks[-1]
                    col = int(t_last // gap)
                    if col != self._col:
                        self._col = col
                        await self._snapshot()
                        if self.live.mode != "off":
                            asyncio.ensure_future(self.live.consider())
                    self._settle()
                    self.live.check_stale()
                if n % 5 == 0 and self.page is not None and not self.page.is_closed():
                    await self._draw()                        # our cells on bc.game's grid + the left-bar panel
                if n % 100 == 0:                              # every ~10 s: the tab is still alive and fed
                    if self.page is None or self.page.is_closed():
                        await self._open()
                    elif time.time() - self._tick_local > 30:
                        log.warning("Tap tab: no price for 30 s, reloading")
                        self._tick_local = time.time()
                        await self.page.reload(wait_until="domcontentloaded")
                    elif (self.sym_cfg and not self.sym_cfg.get("orderStatus") and self.market_open()
                          and time.time() - self._reload_at > 120):   # the opening time has passed: fresh settings
                        log.info("Tap %s: opening time passed, reloading for fresh settings", SYMBOL)
                        self._reload_at = time.time()
                        await self.page.reload(wait_until="domcontentloaded")
                if time.time() - self._saved > 60:
                    self._save()
            except Exception as e:
                log.warning("Tap analyzer: %r", e)

    @staticmethod
    def _ms(v) -> float | None:
        """A time from the settings as epoch ms (they may come as ms, s or an ISO string)."""
        if v in (None, "", 0):
            return None
        try:
            x = float(v)
            return x if x > 1e11 else x * 1000
        except (TypeError, ValueError):
            pass
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000
        except ValueError:
            return None

    def market_open(self) -> bool | None:
        """True / False from the symbol's settings (orderStatus; a closed market reopens at allowTradeTime),
        None when they haven't been read."""
        c = self.sym_cfg
        if not c or "orderStatus" not in c:
            return None
        if c.get("orderStatus"):
            return True
        at = self._ms(c.get("allowTradeTime"))
        return bool(at) and time.time() * 1000 >= at

    def touched_cell(self, lo: float, hi: float, t0: float, t1: float):
        return touched(list(self.ticks), lo, hi, t0, t1)

    def _real_vol(self) -> float | None:
        """Std of the price's 0.5 s log returns over the last VOL_WINDOW_S seconds, robust to spikes and feed gaps:
        a grid point more than 3 s after its last tick is skipped, and returns are winsorized at 6 MADs."""
        rows = list(self.ticks)
        if len(rows) < 50:
            return None
        t = np.array([r[0] for r in rows], float)
        v = np.array([r[1] for r in rows], float)
        grid = np.arange(max(t[0], t[-1] - VOL_WINDOW_S * 1000), t[-1], STEP_MS)
        if len(grid) < 120:
            return None
        idx = np.searchsorted(t, grid, side="right") - 1
        px, fresh = v[idx], (grid - t[idx]) < 3000
        r = np.diff(np.log(px))[fresh[1:] & fresh[:-1]]
        if len(r) < 120:
            return None
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
        if mad > 0:
            r = np.clip(r, -6 * mad, 6 * mad)
        return float(r.std())

    async def _read_bc(self) -> dict | None:
        """bc.game's multipliers as drawn on its grid now (see READ_JS), or None when they can't be read."""
        if self.page is None or self.page.is_closed() or not self.ticks:
            return None
        try:
            d = await self.page.evaluate(READ_JS, {"now": self.ticks[-1][0], "window_ms": ODDS_READ_MS,
                                                   "debug": self._debug})
        except Exception as e:
            self.bc_read = {"n": 0, "err": f"{type(e).__name__}", "at": time.time()}
            return None
        if not d or d.get("err"):
            self.bc_read = {"n": 0, "err": (d or {}).get("err", "no answer"), "at": time.time()}
            return None
        if d.get("texts") is not None:
            self.debug_texts = d["texts"]
        return d

    def ev_bar(self) -> float:
        """The EV (measured hit rate x label) a cell must clear: the live tap's bar at the server's real payout
        (TAP_MIN_EV / the measured server-to-label ratio, or the assumed one before it is measured).  Paper taps
        use the same bar, so the paper boxes on the page show exactly where a real tap goes."""
        lv = getattr(self, "live", None)
        if lv is None:
            return EV_MIN
        return round(lv.min_ev / (lv.ratio() or lv.ratio_boot), 3)

    def _regime(self, step: float) -> dict:
        """The regime now: the trailing hit rate of the last 60 settled current-band virtual taps and the 60 s price
        range in band units -> a factor on the measured hit rates (REGIME_FACTOR).  1.0 while unknown."""
        last = list(self.band_hits)[-60:]
        trailing = sum(h for _, h in last) / len(last) if len(last) >= 30 else None
        ticks = [p for t, p in self.ticks if t >= self.ticks[-1][0] - 60000] if self.ticks else []
        rng = (max(ticks) - min(ticks)) / step if len(ticks) >= 20 and step else None
        if trailing is None or rng is None:
            return {"trailing": trailing, "range_bands": rng, "name": "unknown", "factor": 1.0}
        name = ("calm" if rng < REGIME_RANGE_BANDS else "wild", "hot" if trailing >= REGIME_HOT else "cold")
        return {"trailing": round(trailing, 3), "range_bands": round(rng, 2), "name": "+".join(name),
                "factor": REGIME_FACTOR[name]}

    def _shrunk(self, p_long: float, step: float) -> float:
        """The hit rate to trust now: the long-run measured rate times the regime factor, but never above the blend
        of that rate (weight TRAILING_PRIOR_N) with the trailing rate of the last 60 settled current-band taps.  A
        night like the reopen (trailing 23% against a 55% history) is then priced from what is happening."""
        est = p_long * self.regime["factor"]
        tr = self.regime.get("trailing")
        if tr is not None:
            n = min(60, len(self.band_hits))
            est = min(est, (TRAILING_PRIOR_N * p_long + n * tr) / (TRAILING_PRIOR_N + n))
        return est

    def _hit_rates(self, step: float) -> dict:
        """(k, r) -> the realized hit rate of that cell kind at THIS band size, from the virtual taps settled on the
        real price (n >= CALIB_MIN_N).  This is what the picker trusts: not the model, not a corrected model."""
        out = {}
        for key, s in self.stats.get("cells_step", {}).items():
            st, k, r = key.split(",")
            if float(st) == step and s["n"] >= CALIB_MIN_N:
                out[(int(k), int(r))] = s["hit"] / s["n"]
        for key, s in self.stats.get("cells", {}).items():      # the current band only: its rate is alike at
            k, r = map(int, key.split(","))                   # every band size (47-60%), so a band size not yet
            if r == 0 and (k, 0) not in out and s["n"] >= CALIB_MIN_N:   # measured borrows the all-band rate;
                out[(k, 0)] = s["hit"] / s["n"]               # far cells never do (their rates differ by band)
        return out

    def _calibration(self, step: float) -> dict:
        """(k, r) -> realized hit rate / the model's P(hit), from the virtual taps settled so far at THIS band size
        (a $2-band ratio does not carry over to $5 bands: it made far cells look good).  Clipped to [0.2, 1.5]."""
        out = {}
        for key, s in self.stats.get("cells_step", {}).items():
            st, k, r = key.split(",")
            if float(st) == step and s["n"] >= CALIB_MIN_N and s["po"] > 0:
                out[(int(k), int(r))] = max(0.2, min(1.5, s["hit"] / s["po"]))
        return out

    async def _snapshot(self) -> None:
        now, price = self.ticks[-1]
        gap, step_cfg = self.cfg["timeIncrement"] * 1000, self.cfg["priceIncrement"]
        self.vol_real = self._real_vol()
        bc = await self._read_bc()
        step = step_cfg
        if bc and bc.get("step_usd") and bc.get("n", 0) >= 12:        # the band size actually on screen
            s_inf = min(STEPS_USD, key=lambda s: abs(s - bc["step_usd"]))
            if abs(s_inf - bc["step_usd"]) <= 0.25 * s_inf:
                # a band size other than the settings' is accepted only once it has been read the same way for
                # STEP_CONFIRM_S: single frames read during a redraw gave $1 / $5 for a second or two
                if s_inf != self._step_candidate:
                    self._step_candidate, self._step_since = s_inf, time.time()
                if s_inf == step_cfg or time.time() - self._step_since >= STEP_CONFIRM_S:
                    if s_inf != self.step_inferred:
                        log.info("Tap grid on screen: $%g bands (settings say $%g), %.0f ms columns, %d multipliers read",
                                 s_inf, step_cfg, bc.get("col_ms") or 0, bc["n"])
                    self.step_inferred = s_inf
                    step = s_inf
                else:
                    step = self.step_inferred or step_cfg
        bc_odds = {}
        if bc:
            for it in bc["items"]:
                t, p = it.get("t"), it.get("price")
                if t is None or p is None or t != t or p != p:
                    continue
                bc_odds[(math.floor(t / gap) * gap, round(math.floor(p / step) * step, 2))] = it["odds"]
            self.bc_read = {"n": len(bc_odds), "labels": bc["n"], "step_usd": bc.get("step_usd"),
                            "col_ms": bc.get("col_ms"), "px": bc.get("px"), "py": bc.get("py"), "err": None,
                            "at": time.time(),
                            "items": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in it.items()}
                                      for it in bc["items"][:120]]}
        self.bc_odds = bc_odds
        for key, o in bc_odds.items():                   # the first label seen for a cell is the one a player
            self.label_hist.setdefault(key, (o, now))    # tapping early would have seen; later ones update
            if self.label_hist[key][0] != o:
                self.label_hist[key] = (o, now)
        if len(self.label_hist) > 3000:
            self.label_hist = {k: v for k, v in self.label_hist.items() if k[0] >= now - 600000}
        self.calib = self._calibration(step)
        self.regime = self._regime(step)
        self.hits = {kr: self._shrunk(p, step) for kr, p in self._hit_rates(step).items()}
        grid = await asyncio.to_thread(self._grid, now, price, gap, step, self.cfg["volatility"],
                                       self.vol_real or self.cfg["volatility"], self.cfg["maxOdds"], bc_odds,
                                       self.calib, self.hits)
        labeled = sorted({t0 for (t0, _lo) in bc_odds})
        grid["bc_cols"] = labeled
        tappable = {k for k, t0 in enumerate(grid["cols"]) if t0 in labeled and t0 - now >= MIN_AHEAD_MS}
        if not tappable:
            k_min = int(self.cfg.get("orderGridInterval") or 0)
            tappable = {k for k in range(k_min, k_min + 3) if grid["cols"][k] - now >= MIN_AHEAD_MS}
        for c in grid["cells"]:
            c["tappable"] = c["k"] in tappable
            self.pending.append(dict(c, created=now))
        grid["pick"] = None
        grid["picks"] = {}
        grid["ev_bar"] = self.ev_bar()
        self._pick_models(grid, now, price)
        self.grid = grid

    def _pick_models(self, grid: dict, now: float, price: float) -> None:
        """Every candidate picker (MODELS) places at most one paper tap per column: its best cell by its own score
        when that clears its bar.  "label" is the live picker and feeds the paper book; the others only their own
        books.  Only cell kinds measured at this band size, at bc.game's own label, one per column, and (with live
        taps on) only when the live tap is free to click it, so each box shows exactly where a real tap would go."""
        lv = getattr(self, "live", None)
        if lv is not None and lv.mode != "off" and lv._why_not(now) is not None:
            return
        ctx = {"regime": self.regime, "pay": lv.payout if lv is not None else (lambda k: PAYOUT_FALLBACK)}
        taken: dict = {}
        for c in self.paper_pending:
            taken.setdefault(c.get("model", "label"), set()).add(c["t0"])
        for m in MODELS:
            best, best_s = None, None
            for c in grid["cells"]:
                if not c["tappable"] or c["ev_emp"] is None:
                    continue
                s = m["score"](c, ctx)
                if s is not None and (best_s is None or s > best_s):
                    best, best_s = c, s
            bar = m["bar"] if m["bar"] is not None else grid["ev_bar"]
            if best is None or best_s < bar or best["odds_src"] != "bc" or best["t0"] in taken.get(m["name"], ()):
                continue
            grid["picks"][m["name"]] = (best["k"], best["r"])
            if m["name"] == "label":
                grid["pick"] = (best["k"], best["r"])
            self.paper_pending.append(dict(best, created=now, price=price, model=m["name"], score=round(best_s, 4)))

    def models_summary(self) -> list:
        """The candidate pickers side by side: colour, rule, book (wins at bc.game's payout), the last hour, and
        what each is picking now."""
        now = time.time()
        picks = (self.grid or {}).get("picks") or {}
        out = []
        for m in MODELS:
            b = (self.stats.get("models") or {}).get(m["name"]) or {"bets": 0, "wins": 0, "pnl": 0.0,
                                                                     "pnl_label": 0.0, "since": None}
            dq = self.model_hour.get(m["name"])
            while dq and dq[0][0] < now - 3600:
                dq.popleft()
            out.append({"name": m["name"], "color": m["color"], "desc": m["desc"],
                        "bar": m["bar"] if m["bar"] is not None else self.ev_bar(),
                        **{k: b.get(k) for k in ("bets", "wins", "pnl", "pnl_label", "since")},
                        "hour": round(sum(p for _, p in dq), 4) if dq else 0.0, "hour_n": len(dq) if dq else 0,
                        "open": sum(1 for c in self.paper_pending if c.get("model", "label") == m["name"]),
                        "pick": picks.get(m["name"])})
        return out

    def reset_models(self) -> None:
        self.stats["models"] = {}
        self.model_hour = {}
        log.info("Tap candidate pickers: books reset")

    @staticmethod
    def _grid(now, price, gap, step, vol_cfg, vol_real, max_odds, bc_odds=None, calib=None, hits=None) -> dict:
        bc_odds, calib, hits = bc_odds or {}, calib or {}, hits or {}
        first = (now - now % gap) + gap                       # the next column's start (server ms)
        base = math.floor(price / step) * step                # the band the price is in
        cols = [first + k * gap for k in range(COLS)]
        rows = [base + r * step for r in range(-ROWS, ROWS + 1)]
        cells = []
        for k, t0 in enumerate(cols):
            cm, cr = column(price, now, t0, gap, vol_cfg), column(price, now, t0, gap, vol_real)
            for r, lo in zip(range(-ROWS, ROWS + 1), rows):
                pm = p_hit(cm, now, lo, lo + step, t0, t0 + gap)
                po = p_hit(cr, now, lo, lo + step, t0, t0 + gap)
                o_port = odds(pm, max_odds)
                o_bc = bc_odds.get((t0, round(lo, 2)))
                if o_bc and o_port and (o_bc > 2.2 * o_port or o_bc < o_port / 2.2):
                    o_bc = None                               # a label that can't be this cell's (read across a band
                o = o_bc if o_bc else o_port                  # line while the price moved): treat as unlabelled
                po_cal = min(0.999, po * calib.get((k, r), 1.0))
                cells.append({"k": k, "r": r, "t0": t0, "t1": t0 + gap, "lo": lo, "hi": lo + step, "step": step,
                              "odds": round(o, 3), "odds_port": round(o_port, 3), "odds_src": "bc" if o_bc else "port",
                              "pm": round(pm, 5), "po": round(po, 5), "po_cal": round(po_cal, 5),
                              "ev": round(po_cal * o, 4), "cal": (k, r) in calib,
                              "p_emp": round(hits[(k, r)], 4) if (k, r) in hits else None,      # measured hit rate
                              "ev_emp": round(hits[(k, r)] * o, 4) if (k, r) in hits else None})  # x the label
        return {"now": now, "price": price, "cols": cols, "rows": rows, "cells": cells, "step": step,
                "bc_cells": sum(1 for c in cells if c["odds_src"] == "bc")}

    def _settle(self) -> None:
        if not self.ticks:
            return
        t_last = self.ticks[-1][0]
        due = [c for c in self.pending if c["t1"] + 1000 < t_last]
        pdue = [c for c in self.paper_pending if c["t1"] + 1000 < t_last]
        if not due and not pdue:
            return
        self.pending = [c for c in self.pending if c["t1"] + 1000 >= t_last]
        self.paper_pending = [c for c in self.paper_pending if c["t1"] + 1000 >= t_last]
        ticks = list(self.ticks)
        for c in due:
            hit, hit_x = touched(ticks, c["lo"], c["hi"], c["t0"], c["t1"])
            if c["r"] == 0 and c.get("odds_src") == "bc":
                self.band_hits.append((c["t1"], hit))
            for key, table in ((f'{c["k"]},{c["r"]}', "cells"), (_bucket(c["ev"], EV_BUCKETS), "ev"),
                               (f'{c.get("step", c["hi"] - c["lo"]):g},{c["k"]},{c["r"]}', "cells_step")):
                s = self.stats.setdefault(table, {}).setdefault(key, {"n": 0, "hit": 0, "hit_x": 0, "odds": 0.0,
                                                                      "ret": 0.0, "pm": 0.0, "po": 0.0})
                s["n"] += 1
                s["hit"] += hit
                s["hit_x"] += hit_x
                s["odds"] += c["odds"]
                s["ret"] += (c["odds"] if hit else 0.0) - 1.0
                s["pm"] += c["pm"]
                s["po"] += c["po"]
        lv = getattr(self, "live", None)
        for c in pdue:
            hit, _ = touched(ticks, c["lo"], c["hi"], c["t0"], c["t1"])
            model = c.get("model", "label")
            pay = lv.payout(c["k"]) if lv else PAYOUT_FALLBACK        # bc.game pays this fraction of the grid label
            pnl = round((c["odds"] * pay - 1.0) if hit else -1.0, 4)  # what the server would have paid
            pnl_label = round((c["odds"] - 1.0) if hit else -1.0, 4)  # what the grid label promised (the old P&L)
            mb = self.stats.setdefault("models", {}).setdefault(
                model, {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0, "since": time.time()})
            mb["bets"] += 1
            mb["wins"] += hit
            mb["pnl"] = round(mb["pnl"] + pnl, 4)
            mb["pnl_label"] = round(mb["pnl_label"] + pnl_label, 4)
            self.model_hour.setdefault(model, deque()).append((time.time(), pnl))
            if model != "label":                                       # a candidate: its own log, not the paper book
                rec = {"t": c["created"], "model": model, "k": c["k"], "r": c["r"], "lo": c["lo"], "hi": c["hi"],
                       "t0": c["t0"], "odds": c["odds"], "ev_emp": c.get("ev_emp"), "score": c.get("score"),
                       "price": c["price"], "hit": hit, "pnl": pnl, "pnl_label": pnl_label, "pay": round(pay, 3)}
                with open(self.models_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
            strict = (c.get("ev_emp") or c["ev"]) * pay >= (lv.min_ev if lv else EV_MIN)   # clears the bar at the
            p = self.stats["paper"]                                                        # paid odds too
            p["bets"] += 1
            p["wins"] += hit
            p["pnl"] = round(p["pnl"] + pnl, 4)
            p["pnl_label"] = round(p.get("pnl_label", 0.0) + pnl_label, 4)
            if strict:
                ps = self.stats.setdefault("paper_strict", {"bets": 0, "wins": 0, "pnl": 0.0})
                ps["bets"] += 1
                ps["wins"] += hit
                ps["pnl"] = round(ps["pnl"] + pnl, 4)
            rec = {"t": c["created"], "k": c["k"], "r": c["r"], "lo": c["lo"], "hi": c["hi"], "t0": c["t0"],
                   "odds": c["odds"], "odds_src": c.get("odds_src"), "odds_port": c.get("odds_port"),
                   "po": c["po"], "po_cal": c.get("po_cal"), "ev": c["ev"], "ev_emp": c.get("ev_emp"),
                   "price": c["price"], "hit": hit, "pnl": pnl, "pnl_label": pnl_label, "pay": round(pay, 3),
                   "strict": strict, "v": STATS_VERSION}
            self.paper_recent.append(rec)
            with open(self.paper_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _rebook_paper(self) -> None:
        """Re-price the saved paper book at bc.game's payout: its last `bets` settled picks from the paper log, a win
        at label x payout instead of the label (the book used to count what the grid promised, not what is paid)."""
        n = int(self.stats["paper"].get("bets") or 0)
        try:
            lines = self.paper_path.read_text(encoding="utf-8").splitlines()[-n:] if n else []
        except OSError:
            lines = []
        lv = getattr(self, "live", None)
        book = {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0}
        strict = {"bets": 0, "wins": 0, "pnl": 0.0}
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("hit") is None or not r.get("odds"):
                continue
            hit = bool(r["hit"])
            pay = lv.payout(r.get("k")) if lv else PAYOUT_FALLBACK
            pnl = (r["odds"] * pay - 1.0) if hit else -1.0
            book["bets"] += 1
            book["wins"] += hit
            book["pnl"] = round(book["pnl"] + pnl, 4)
            book["pnl_label"] = round(book["pnl_label"] + ((r["odds"] - 1.0) if hit else -1.0), 4)
            if (r.get("ev_emp") or r.get("ev") or 0) * pay >= (lv.min_ev if lv else EV_MIN):
                strict["bets"] += 1
                strict["wins"] += hit
                strict["pnl"] = round(strict["pnl"] + pnl, 4)
        self._rebook = False
        if book["bets"]:
            log.info("Tap paper book re-priced at bc.game's payout: %d picks, %+.2f at the grid label -> %+.2f",
                     book["bets"], book["pnl_label"], book["pnl"])
            self.stats["paper"] = book
            self.stats["paper_strict"] = strict

    # ---------- drawing on bc.game's Tap page ----------

    async def _draw(self) -> None:
        try:
            await self.page.evaluate(DRAW_JS, self.overlay_data())
            await self.page.evaluate(OVERLAY_JS, self.panel_html())
        except Exception:
            pass                                              # the page is reloading; next time

    def overlay_data(self) -> dict:
        """Our boxes for bc.game's grid: every evaluated cell with our EV (green when >= 1, dimmed when it can't be
        tapped now), the paper tap (yellow), and the last paper taps with HIT / MISS."""
        g, gap = self.grid, self.cfg["timeIncrement"] * 1000
        cells = []
        pick = tuple(g["pick"]) if g and g.get("pick") else None
        for c in (g or {}).get("cells", []):
            if pick == (c["k"], c["r"]):
                continue
            ev = c.get("ev_emp")                                 # the measured EV the picker uses (None: unmeasured)
            bar = g.get("ev_bar") or EV_MIN
            cls = ("good" if ev is not None and ev >= bar else "") + ("" if c.get("tappable") else " off")
            cells.append({"t0": c["t0"], "t1": c["t1"], "lo": c["lo"], "hi": c["hi"], "cls": cls,
                          "label": f"{ev:.2f}" if ev is not None else "·"})
        colors = {m["name"]: m["color"] for m in MODELS}
        order = {m["name"]: i for i, m in enumerate(MODELS)}
        for c in self.paper_pending:                          # paper taps still waiting for their window: the paper
            model = c.get("model", "label")                   # book's in yellow, each candidate picker in its colour
            if model == "label":                              # (nested outlines when they pick the same cell)
                cells.append({"t0": c["t0"], "t1": c["t1"], "lo": c["lo"], "hi": c["hi"], "cls": "pick",
                              "label": f"PAPER $1 x{c['odds']:.2f} ({c.get('odds_src', 'port')})"})
            else:
                cells.append({"t0": c["t0"], "t1": c["t1"], "lo": c["lo"], "hi": c["hi"], "cls": "mpick",
                              "color": colors.get(model, "#fff"), "inset": 3 * order.get(model, 1), "label": model})
        for r in list(self.paper_recent)[-8:]:                # settled ones, while still on screen
            cells.append({"t0": r["t0"], "t1": r["t0"] + gap, "lo": r["lo"], "hi": r["hi"],
                          "cls": "hit" if r["hit"] else "miss",
                          "label": f"HIT {r['pnl']:+.2f}" if r["hit"] else "MISS -1"})
        return {"now": self.ticks[-1][0] if self.ticks else time.time() * 1000, "cells": cells}

    def panel_html(self) -> str:
        """The left-bar panel on the Tap page, in the same card style as the Up/Down page's."""
        p, pub = self.stats["paper"], self.stats["public"]
        price = self.ticks[-1][1] if self.ticks else None

        def usd(v):
            return f'{"+" if v >= 0 else "-"}${abs(v):,.2f}'

        def row(k, v):
            return f'<div class="row"><span>{k}</span><span>{v}</span></div>'

        pick = next((c for c in reversed(self.paper_pending) if c.get("model", "label") == "label"), None)
        if pick:
            nxt = (f'<div class="big up" style="font-size:20px">+{pick["k"]} col &middot; {pick["r"]:+d} band</div>'
                   f'<div class="mute">x{pick["odds"]:.2f} &middot; our P(hit) {pick["po"]:.0%} &middot; EV {pick["ev"]:.2f} '
                   f'&middot; waiting for its window</div>')
        else:
            best = max((c for c in (self.grid or {}).get("cells", []) if c.get("tappable")),
                       key=lambda c: c["ev"], default=None)
            nxt = ('<div class="big mute" style="font-size:20px">NO TAP</div><div class="mute">'
                   + (f'best cell EV {best.get("ev_emp") or 0:.2f} &lt; {self.ev_bar()} this column' if best else 'waiting for the grid')
                   + '</div>')
        wr = f' &middot; won {p["wins"] / p["bets"]:.1%}' if p["bets"] else ""
        lv = self.live
        live_card = ""
        if lv.mode != "off":
            b = lv.book()
            stop = lv.st["stopped"]
            live_card = (f'<div class="c"><div class="ct"><span>LIVE TAPS &middot; {lv.mode.upper()}</span>'
                         f'<span class="mute">${lv.stake:g} / tap &middot; budget ${lv.budget:g}</span></div>'
                         f'<div class="two"><b class="pnl {"pos" if b["pnl"] >= 0 else "neg"}">{usd(b["pnl"])}</b>'
                         f'<b class="pnl">{b["bets"]} &middot; {b["wins"]}</b></div>'
                         f'<div class="mute">server / label x{lv.ratio() or 0:.2f} &middot; open {len(lv.open_orders())}'
                         f' &middot; losses in a row {b["run"]}</div>'
                         f'<div class="mute">{"STOPPED: " + stop["why"] if stop else (lv.last_reason or "")}</div></div>')
        n_open = sum(1 for c in self.paper_pending if c.get("model", "label") == "label")
        rows_m = "".join(
            f'<div class="row"><span><i style="display:inline-block;width:9px;height:9px;border:2px solid {m["color"]};'
            f'margin-right:5px;vertical-align:middle"></i>{m["name"]}</span>'
            f'<span><b class="pnl {"pos" if (m["pnl"] or 0) >= 0 else "neg"}">{usd(m["pnl"] or 0)}</b> '
            f'<span class="mute">{m["bets"] or 0}t {m["wins"] or 0}w &middot; 1h {m["hour"]:+.1f}</span></span></div>'
            for m in self.models_summary())
        models_card = (f'<div class="c"><div class="ct"><span>PICKERS ON TRIAL</span>'
                       f'<span class="mute">$1 paper each &middot; wins at the paid odds</span></div>{rows_m}</div>')
        open_row = (f'<div class="mute">open: {n_open} tap{"s" if n_open != 1 else ""} &middot; ${n_open:.0f} at risk '
                    f'&middot; wallet view {usd(p["pnl"] - n_open)}</div>' if n_open else "")
        results = "".join(
            f'<div class="res {"w" if r["hit"] else "l"}"><span class="sd">+{r["k"]}c {r["r"]:+d}b</span>'
            f'<b>{"HIT" if r["hit"] else "MISS"}</b><span class="m">{r["pnl"]:+.2f} '
            f'<span class="tag ptag">x{r["odds"]:.2f}</span></span></div>'
            for r in list(self.paper_recent)[::-1][:6])
        strip = "".join(f'<i class="{"W" if r["hit"] else "L"}"></i>' for r in list(self.paper_recent)[::-1])
        cfg_usd = self.cfg["volatility"] * price if price else None
        real_usd = self.vol_real * price if price and self.vol_real else None
        ratio = real_usd / cfg_usd if real_usd and cfg_usd else None
        vol_words = ("" if ratio is None else
                     "price moves MORE than bc.game assumes: far cells under-priced, near cells over-priced" if ratio > 1.05
                     else "price moves LESS than bc.game assumes: near cells under-priced, far cells over-priced"
                     if ratio < 0.95 else "about as bc.game assumes")
        br = self.bc_read
        if br.get("n") and self.step_inferred:
            src_row = row("multipliers", f'<b class="up">bc.game&#39;s grid</b> ({br["n"]} cells read, '
                                         f'${self.step_inferred:g} bands)')
        else:
            src_row = row("multipliers", f'<b class="down">our port</b> (grid not read: {br.get("err") or "no labels"})')
        kinds = []
        for key, s in self.stats["cells"].items():
            if s["n"] >= 300:
                k, r = map(int, key.split(","))
                kinds.append((s["ret"] / s["n"], k, r, s["n"]))
        kinds.sort(reverse=True)
        best_kinds = "".join(row(f"+{k} col {r:+d} band", f'<b class="{"up" if ret >= 0 else "down"}">{ret:+.0%}</b> of {n}')
                             for ret, k, r, n in kinds[:4]) or row("need more taps", "&ndash;")
        n_virtual = sum(s["n"] for s in self.stats["cells"].values())
        return (
            f'{PANEL_CSS}<div class="p"><div class="hd"><span>BC BOT &middot; TAP {SYMBOL}</span>'
            + ('<span class="mode live">LIVE REAL</span>' if self.live.mode == "real" else
               '<span class="mode paused">DEMO</span>' if self.live.mode == "demo" else '<span class="mode paper">PAPER</span>')
            + '</div>'
            + live_card + f'<div class="c"><div class="ct"><span>NEXT TAP</span><span class="mute">$1 when EV &ge; {self.ev_bar()} (the live bar); wins booked at what bc.game pays (x{self.live.payout(3):.2f} of the label)</span></div>{nxt}</div>'
            f'<div class="c"><div class="ct"><span>PAPER P&amp;L</span><span class="mute">taps &middot; won</span></div>'
            f'<div class="two"><b class="pnl {"pos" if p["pnl"] >= 0 else "neg"}">{usd(p["pnl"])}</b>'
            f'<b class="pnl">{p["bets"]} &middot; {p["wins"]}</b></div><div class="mute">$1 per tap{wr} &middot; at the grid label {usd(p.get("pnl_label", 0.0))}</div>{open_row}'
            + (f'<div class="sq">{strip}</div>' if strip else "") + '</div>' + models_card
            + (f'<div class="c"><div class="ct"><span>LAST TAPS</span><span class="mute">cell &middot; result</span></div>'
               f'{results}</div>' if results else "")
            + f'<div class="c"><div class="ct"><span>VOLATILITY</span><span class="mute">per 0.5 s</span></div>'
            + row("bc.game's model", f"${cfg_usd:.2f}" if cfg_usd else "&ndash;")
            + row("real (10 min)", f"${real_usd:.2f}" if real_usd else "&ndash;")
            + (row("real / model", f'<b class="{"down" if ratio > 1 else "up"}">{ratio:.0%}</b>') if ratio else "")
            + (f'<div class="mute">{vol_words}</div>' if vol_words else "") + src_row + '</div>'
            + f'<div class="c"><div class="ct"><span>VIRTUAL TAPS</span><span class="mute">{n_virtual:,} settled</span></div>'
            f'<div class="mute">best kinds of cell (real return per $1):</div>{best_kinds}</div>'
            + f'<div class="c"><div class="ct"><span>OTHER PLAYERS</span><span class="mute">{SYMBOL}</span></div>'
            + row("taps &middot; won", f'{pub["n"]} &middot; {pub["won"]}')
            + row("their return per $1", f'{pub["ret"] / pub["n"]:+.0%}' if pub["n"] else "&ndash;") + '</div></div>')

    # ---------- other players' taps ----------

    def _public(self, r: dict) -> None:
        oid = r.get("id")
        if not oid or oid in self._seen_public:
            return
        self._seen_public.add(oid)
        try:
            o = float(r["odds"])
            lo, hi = float(r["lowerPriceLimit"]), float(r["upperPriceLimit"])
            t0, t1 = float(r["lowerTimeLimit"]), float(r["upperTimeLimit"])
            sp, st = float(r["startPrice"]), float(r["startTime"])
        except (KeyError, TypeError, ValueError):
            return
        won = float(r.get("profit") or 0) > 0
        s = self.stats["public"]
        s["n"] += 1
        s["won"] += won
        s["odds"] += o
        s["ret"] += (o if won else 0.0) - 1.0
        b = s["by_odds"].setdefault(_bucket(o, ODDS_BUCKETS), {"n": 0, "won": 0, "odds": 0.0})
        b["n"] += 1
        b["won"] += won
        b["odds"] += o
        rec = {"id": oid, "odds": o, "lo": lo, "hi": hi, "t0": t0, "t1": t1, "start_price": sp, "start_time": st,
               "won": won, "amount": r.get("amount"), "profit": r.get("profit"), "currency": r.get("currency"),
               "end_price": r.get("endPrice"), "end_time": r.get("endTime"), "status": r.get("status"),
               "bc_odds_read": (self.label_hist.get((math.floor(t0 / (self.cfg["timeIncrement"] * 1000))
                                                    * self.cfg["timeIncrement"] * 1000, round(lo, 2))) or (None,))[0]}
        ticks = list(self.ticks)
        if ticks and ticks[0][0] <= t0 and ticks[-1][0] >= min(t1, float(r.get("endTime") or t1)):
            hit, hit_x = touched(ticks, lo, hi, t0, t1)              # our hit rule vs their real result
            s["checked"] += 1
            s["agree_tick"] += hit == won
            s["agree_cross"] += hit_x == won
            rec.update(our_hit=hit, our_hit_cross=hit_x)
        try:                                                          # our port of the multiplier vs theirs
            gap = t1 - t0
            mine = odds(p_hit(column(sp, st, t0, gap, self.cfg["volatility"]), st, lo, hi, t0, t1),
                        self.cfg["maxOdds"])
            s["odds_checked"] += 1
            s["odds_err"] += abs(mine - o)
            rec["our_odds"] = round(mine, 3)
            self.odds_ratio.append(o / mine)                          # (on STONKS: theirs ~4 % above ours)
        except (ValueError, ZeroDivisionError):
            pass
        rec["ahead_ms"] = t0 - st
        labeled = (self.grid or {}).get("bc_cols") or []
        if labeled:                                           # their window vs the first column we saw labelled
            rec["labeled_first_ahead_ms"] = labeled[0] - st
            rec["k_vs_labeled"] = int(round((t0 - labeled[0]) / (self.cfg["timeIncrement"] * 1000)))
        self.public_recent.append(rec)
        with open(self.public_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ---------- persistence and state ----------

    def _load(self) -> None:
        try:
            saved = json.loads(self.stats_path.read_text(encoding="utf-8"))
            for k in self.stats:
                if k in saved:
                    self.stats[k] = saved[k] if not isinstance(self.stats[k], dict) else {**self.stats[k], **saved[k]}
            self.stats["cells_step"] = saved.get("cells_step", {})
            if saved.get("v") != STATS_VERSION:            # EV and paper P&L depend on how EV is computed: start over
                log.info("Tap stats: EV / paper tables reset (version %s -> %s); cell hit rates kept",
                         saved.get("v"), STATS_VERSION)
                self.stats["ev"] = {}
                self.stats["paper"] = {"bets": 0, "wins": 0, "pnl": 0.0, "pnl_label": 0.0}
                self.stats["paper_strict"] = {"bets": 0, "wins": 0, "pnl": 0.0}
                self.paper_recent.clear()
            elif self.stats["paper"].get("bets") and "pnl_label" not in (saved.get("paper") or {}):
                if hasattr(self, "live"):                  # an old book priced at the grid label: re-price its picks
                    self._rebook_paper()
                else:
                    self._rebook = True
        except (OSError, ValueError):
            pass
        self.stats["v"] = STATS_VERSION
        try:
            for line in self.paper_path.read_text(encoding="utf-8").splitlines()[-12:]:
                r = json.loads(line)
                if r.get("v") == STATS_VERSION:
                    self.paper_recent.append(r)
        except (OSError, ValueError):
            pass
        try:                                                  # the multiplier check from the logged public taps
            for line in self.public_path.read_text(encoding="utf-8").splitlines()[-500:]:
                r = json.loads(line)
                if r.get("our_odds"):
                    self.odds_ratio.append(r["odds"] / r["our_odds"])
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        self._saved = time.time()
        try:
            self.stats_path.write_text(json.dumps(self.stats), encoding="utf-8")
        except OSError as e:
            log.warning("Tap stats: %r", e)

    def state(self) -> dict:
        last = self.ticks[-1] if self.ticks else None
        price = last[1] if last else None
        cfg_vol, real_vol = self.cfg["volatility"], self.vol_real

        def table(name):
            out = []
            for key, s in self.stats[name].items():
                n = s["n"] or 1
                out.append({"key": key, "n": s["n"], "hit": s["hit"] / n, "hit_x": s["hit_x"] / n,
                            "odds": s["odds"] / n, "ret": s["ret"] / n, "pm": s["pm"] / n, "po": s["po"] / n})
            return out

        pub = self.stats["public"]
        return {
            "symbol": SYMBOL, "connected": self.connected, "price": price, "tick_age": time.time() - self._tick_local,
            "cfg": self.cfg, "cfg_from_page": self.cfg_from_page,
            "vol_cfg": cfg_vol, "vol_real": real_vol,
            "vol_cfg_usd": cfg_vol * price if price else None, "vol_real_usd": real_vol * price if price and real_vol else None,
            "grid": self.grid, "cells": table("cells"), "ev": table("ev"), "since": self.stats["since"],
            "bc_read": self.bc_read, "step_inferred": self.step_inferred,
            "calib": {f"{k},{r}": round(v, 3) for (k, r), v in self.calib.items()},
            "bc_odds": {f"{int(t0)},{lo:g}": o for (t0, lo), o in self.bc_odds.items()},
            "debug_texts": self.debug_texts,
            "sym_cfg": self.sym_cfg, "market_open": self.market_open(), "live": self.live.summary(),
            "regime": self.regime,
            "models": self.models_summary(),
            "paper": {**self.stats["paper"], "recent": list(self.paper_recent)[::-1], "ev_min": self.ev_bar(),
                      "open": sum(1 for c in self.paper_pending if c.get("model", "label") == "label"), "payout": self.live.payout_table(), "min_ev": self.live.min_ev,
                      "strict": self.stats.get("paper_strict")},
            "public": {**pub, "recent": list(self.public_recent)[::-1],
                       "odds_ratio": float(np.median(self.odds_ratio)) if self.odds_ratio else None},
        }
