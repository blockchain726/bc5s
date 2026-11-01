"""Local HTTP API so other programs (signal scripts, webhooks, curl) can place trades.

  GET  /status                          -> current balance / round state
  POST /trade  {"side": "up", "amount": 5}  -> a manual order (only while armed; counted like every real order)
  GET  /auto                            -> auto mode: accuracy, paper P&L, last call, models, the real-money state
  POST /auto   {"on": true, "model": "mhh", "progression": 1.5}   -> auto mode, calling model, paper progression
  POST /auto   {"arm": true, "ack_run": 0, "day": "2026-10-30", "force": false}  -> allow real orders (typed ack)
  POST /auto   {"live": false} | {"kill": true} | {"clear": "unknown", "outcome": "none"}
  GET  /                                -> decision dashboard (the same panel the bot draws on the BC page)
  GET  /tap, /tap/state                 -> Tap analyzer (paper only): bc.game's Tap grid for BTC/USD, our EV,
                                           virtual and paper taps, other players' results

Anything that touches real money (POST /trade; arm, live, kill, clear in POST /auto) needs API_TOKEN from .env
(Authorization: Bearer <token> or ?token=); without a token set, those are refused.  The paper settings (on,
model, progression) and every GET are open on the local address.
"""
import json
import logging

from aiohttp import web

from .auto import AutoTrader
from .config import Config
from .tap_page import TAP_DASHBOARD
from .trader import TradeError, Trader, status_dict

log = logging.getLogger(__name__)


def build_app(trader: Trader, auto: AutoTrader, cfg: Config, tap=None) -> web.Application:
    MONEY_KEYS = ("arm", "live", "kill", "clear")

    def money_auth(request: web.Request) -> web.Response | None:
        """The token check for anything that can arm, disarm or spend: refused without a configured token."""
        if not cfg.api_token:
            return web.json_response({"ok": False, "error": "refused: set API_TOKEN in .env and pass it as "
                                                             "Authorization: Bearer <token> or ?token="}, status=403)
        given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if given != cfg.api_token and request.query.get("token") != cfg.api_token:
            return web.json_response({"ok": False, "error": "unauthorized"}, status=403)
        return None

    @web.middleware
    async def auth(request: web.Request, handler):
        if cfg.api_token and request.method == "POST":       # with a token set, every POST must carry it
            denied = money_auth(request)
            if denied is not None:
                return denied
        return await handler(request)

    async def status(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, **status_dict(await trader.status())})

    async def trade(request: web.Request) -> web.Response:
        denied = money_auth(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
            side = str(body["side"])
            amount = float(body.get("amount", cfg.default_amount))
        except (ValueError, KeyError, TypeError):
            return web.json_response({"ok": False, "error": 'expected JSON {"side": "up"|"dn", "amount": 5}'}, status=400)
        try:
            result = await auto.manual_order(side, amount, who="api")
        except TradeError as e:
            log.warning("API trade refused: %s", e)
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        log.info("API trade placed: %s %g -> order %s", result["side"].upper(), amount, result["order_id"])
        return web.json_response(result)

    async def auto_get(_: web.Request) -> web.Response:
        return web.json_response(auto.summary(), dumps=lambda o: json.dumps(o, default=float))

    async def auto_post(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if any(k in body for k in MONEY_KEYS):
                denied = money_auth(request)
                if denied is not None:
                    return denied
            if "on" in body:
                await auto.start() if body["on"] else auto.stop()
            if "model" in body:
                auto.set_model(str(body["model"]))
            if "progression" in body:
                auto.set_progression(float(body["progression"]))
            if body.get("kill"):
                auto.kill(who="api")
            if body.get("arm"):
                auto.arm(int(body.get("ack_run", -1)), str(body.get("day", "")), who="api",
                         force=bool(body.get("force")))
            if "live" in body:
                if body["live"]:
                    raise TradeError('arming is {"arm": true, "ack_run": <losses in a row>, "day": "YYYY-MM-DD"}')
                auto.disarm("live off", who="api")
            if body.get("clear"):
                auto.clear(str(body["clear"]), body.get("outcome"), who="api")
        except TradeError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        except (ValueError, TypeError):
            return web.json_response({"ok": False, "error": 'expected JSON {"on": true, "live": false}'}, status=400)
        return await auto_get(request)

    async def panel(_: web.Request) -> web.Response:
        return web.Response(text=auto.panel_html(), content_type="text/html")

    async def prices(request: web.Request) -> web.Response:
        window = min(300.0, max(10.0, float(request.query.get("s", 60))))
        return web.json_response(auto.prices(window))

    async def dashboard(_: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD, content_type="text/html")

    async def tap_page(_: web.Request) -> web.Response:
        return web.Response(text=TAP_DASHBOARD, content_type="text/html")

    async def tap_state(_: web.Request) -> web.Response:
        if tap is None:
            return web.json_response({"error": "the Tap analyzer is not running"})
        return web.json_response(tap.state())

    app = web.Application(middlewares=[auth])
    app.add_routes([web.get("/", dashboard), web.get("/status", status), web.post("/trade", trade),
                    web.get("/auto", auto_get), web.post("/auto", auto_post), web.get("/auto/panel", panel),
                    web.get("/auto/prices", prices), web.get("/tap", tap_page), web.get("/tap/state", tap_state)])
    return app


DASHBOARD = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>BC bot</title>
<style>
body{margin:0;min-height:100vh;display:flex;align-items:flex-start;gap:24px;padding:16px;box-sizing:border-box;
     background:#0b0e13;font:13px Consolas,monospace;color:#d6dde6}
#panel{position:sticky;top:16px;flex:none;zoom:1.3}#panel .p{max-height:none;overflow:visible}
#panel .hd{background:none}
main{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;gap:20px}
.bar{display:flex;gap:18px;flex-wrap:wrap;justify-content:center}.grp{display:flex;gap:4px;align-items:center}
.grp span{color:#8793a2;margin-right:4px}button{font:inherit;color:#d6dde6;background:#1b2029;border:1px solid #2c3440;
     border-radius:6px;padding:5px 10px;cursor:pointer}button.on{background:#2f6fd6;border-color:#2f6fd6;color:#fff}
button.liveb.on{background:#b4232f;border-color:#b4232f}#msg{color:#f0b429;min-height:16px}
</style></head>
<body>
<aside id="panel"></aside>
<main>
<div class="bar">
  <div class="grp"><span>calls by</span><button data-model="sign">HawkesSign</button><button data-model="mhh">MHH flow</button><button data-model="agree" title="bet only when HawkesSign and MHH call the same side">Sign+MHH agree</button><button data-model="mom" title="the direction of our price over the last 5 s">Momentum 5s</button><button data-model="book" title="bet with Binance's 20-level book, only when it is strongly lopsided">Strong book</button><button data-model="bookmom" title="only when the strong book and the last 5 s of price point the same way">Book + trend</button><button data-model="revert" title="quiet market: bet that bc.game's price undoes its last 2-second move">Quiet reversal</button><button data-model="overshoot" title="bc.game ran ahead of the market: bet the snap-back">Overshoot</button><button data-model="prob" title="P(UP) fitted on all logged rounds; calls only when it is at least 56% one way">P(UP) model</button><button data-model="ens" title="average of the logistic and LightGBM P(UP); calls only when it is at least 56% one way">P(UP) ensemble</button></div>
  <div class="grp"><span>paper ladder after a loss</span><button data-prog="1">flat $1</button><button data-prog="1.5">x1.5</button><button data-prog="2">x2 (MHH)</button></div>
  <div class="grp"><span>real orders</span><button data-live="0" title="disarm (arming is done in the terminal: live arm run=N day=YYYY-MM-DD)">paper / disarm</button></div>
  <div class="grp"><a href="/tap" style="color:#5aa9ff;text-decoration:none;border:1px solid #2c3440;border-radius:6px;padding:5px 10px">Tap analyzer &rarr;</a></div>
</div>
<div id="safety" style="color:#8793a2"></div>
<div id="msg"></div>
<div style="width:min(1100px,100%)">
  <div style="display:flex;justify-content:space-between;color:#8793a2;margin-bottom:4px">
    <span>BTC price, last 60 s &middot; <span style="color:#facc15">bc.game</span> vs <span style="color:#22d3ee">our price</span>
      = weighted mix of 6 exchange prices (mostly perpetual futures), each read ~0.15 s earlier (bc.game follows
      them) and moved to bc.game's level</span>
    <span id="mixinfo"></span></div>
  <canvas id="chart" style="width:100%;height:300px;background:#11141a;border:1px solid #2c3440;border-radius:8px"></canvas>
  <div style="color:#8793a2;margin:6px 0 2px">difference: bc.game minus our price ($)</div>
  <canvas id="diff" style="width:100%;height:90px;background:#11141a;border:1px solid #2c3440;border-radius:8px"></canvas>
  <div id="legend" style="margin-top:6px;display:flex;gap:18px;flex-wrap:wrap"></div>
  <div id="ourbox" style="margin-top:10px;padding:8px 12px;background:#11141a;border:1px solid #2c3440;border-radius:8px"></div>
  <div id="runs" style="margin-top:10px;padding:8px 12px;background:#11141a;border:1px solid #2c3440;border-radius:8px"></div>
</div>
</main>
<script>
const q = location.search;
async function post(body) {
  const r = await fetch('/auto' + q, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json(); document.getElementById('msg').textContent = d.error || ''; state(d);
}
// ---- losing runs: how many losses in a row, how often, and what a x2 ladder pays at each length ----
const MODEL_NAMES = {sign: 'HawkesSign', time: 'HawkesTime', mhh: 'MHH flow', mhh_hi: 'MHH confident', agree: 'Sign+MHH agree', mom: 'Momentum 5s', book: 'Strong book', bookmom: 'Book + trend', momvol: 'Momentum, volatile only', revert: 'Quiet reversal', overshoot: 'Overshoot', overshoot_book: 'Overshoot + book', prob: 'P(UP) model (logistic, fitted on the log, calls at |P-0.5| >= 0.06)', gbm: 'P(UP) trees (LightGBM)', ens: 'P(UP) ensemble (average of both)', pool: 'Pool: thin side when it pays > x2 (scored only)'};
function every(hours, n) {
  if (!n) return '–';
  const h = hours / n;
  return h < 1 ? `~${Math.max(1, Math.round(h * 60))} min` : `~${h.toFixed(1)} h`;
}
function poolPay(stake) {     // page payout x1.94 after our stake joins a ~$102 side of a ~$204 pool
  return 1.94 * (204 + stake) / (102 + stake) * 102 / 204;
}
function runsBox(d) {
  const runs = d.runs || {}, cur = runs[d.model] || {lengths: {}, current: 0, bets: 0};
  const names = Object.keys(MODEL_NAMES).filter(k => runs[k]);
  if (!names.length) return;
  let maxN = 1;
  for (const k of names) {
    for (const n of Object.keys(runs[k].lengths)) maxN = Math.max(maxN, Number(n));
    maxN = Math.max(maxN, runs[k].current);
  }
  const hours = Math.max(1e-6, (cur.last - cur.first) / 3600);
  const atLeast = n => Object.entries(cur.lengths).reduce((s, [k, v]) => s + (Number(k) >= n ? v : 0), 0);
  const th = s => `<th style="padding:2px 8px;text-align:right;font-weight:normal;color:#8793a2">${s}</th>`;
  const td = (s, st = '') => `<td style="padding:2px 8px;text-align:right;${st}">${s}</td>`;
  let rows = '', bet = 1, lost = 0, gaveUp = 0;
  for (let n = 1; n <= maxN; n++) {
    lost += bet; bet *= 2;                              // the x2 ladder through n losses (back to $1 only
    if (d.ladder_cap && bet > d.ladder_cap) { bet = 1; gaveUp++; }   // if a paper give-up cap is set)
    const net = bet * (poolPay(bet) - 1) - lost;
    rows += '<tr>' + td(n, 'color:#8793a2') + names.map(k => {
      const c = runs[k].lengths[n] || 0, now = runs[k].current === n;
      return td(c + (now ? ' <span style="color:#f59e0b">+now</span>' : ''),
                k === d.model ? 'color:#5aa9ff;font-weight:bold' : '');
    }).join('') + td(every(hours, atLeast(n)), 'color:#5aa9ff') +
      td('$' + bet + (gaveUp ? ` <span style="color:#8793a2">(gave up ${gaveUp === 1 ? 'once' : gaveUp + 'x'})</span>` : '')) +
      td('-$' + lost, 'color:#ff5468') +
      td((net >= 0 ? '+' : '') + net.toFixed(2), `color:${net >= 0 ? '#22d27a' : '#ff5468'}`) + '</tr>';
  }
  const longest = Math.max(0, ...Object.keys(cur.lengths).map(Number), cur.current);
  const lad = (d.ladders || {})[d.model] || {};
  document.getElementById('runs').innerHTML =
    `<b>Losses in a row</b> <span style="color:#8793a2">(how many runs of each length, all logged rounds; ` +
    `a run ends at the next win)</span>` +
    `<table style="border-collapse:collapse;margin-top:6px;font-size:13px"><tr>${th('losses in a row')}` +
    names.map(k => th(MODEL_NAMES[k] + (k === d.model ? ' &#9664;' : ''))).join('') +
    th(`${MODEL_NAMES[d.model] || d.model}: that many or more`) + th('x2: bet after them') +
    th('x2: lost so far') + th('x2: net when it then wins') + `</tr>${rows}</table>` +
    `<div style="margin-top:6px;color:#8793a2">${MODEL_NAMES[d.model] || d.model}: ${cur.bets} bets over ` +
    `${hours.toFixed(1)} h &middot; longest run ${longest} &middot; now ${cur.current} in a row. ` +
    `x2 net assumes ~$102 on each side of the pool, so a big bet pays itself back: x1.94 becomes ~x1.7 at $32. ` +
    (d.ladder_cap ? `The paper x2 ladder gives up after losing the $${2 ** Math.floor(Math.log2(d.ladder_cap))} bet.`
                  : `The paper x2 ladder never gives up: it doubles until a win. Real orders are always flat ` +
                    `$${d.next_stake || 1} and stop at ${d.max_real_run} losses in a row.`) + `</div>` +
    (d.progression === 2 && lad.staked ? `<div style="margin-top:4px">paper x2 since ` +
      `${new Date(d.since * 1000).toLocaleTimeString()}: <b style="color:${lad.pnl >= 0 ? '#22d27a' : '#ff5468'}">` +
      `${lad.pnl >= 0 ? '+' : ''}${lad.pnl.toFixed(2)}</b> pool-adjusted ` +
      `<span style="color:#8793a2">(at page payouts ${(lad.pnl_shown || 0) >= 0 ? '+' : ''}${(lad.pnl_shown || 0).toFixed(2)}, ` +
      `as MHH counts it) &middot; biggest bet $${lad.biggest} &middot; gave up ${lad.gave_up || 0}</span></div>` : '');
}
function safetyLine(d) {
  const g = d.gate || {}, clk = d.clock || {};
  const st = d.state === 'ARMED' ? '<b style="color:#ff5468">LIVE (armed)</b>' :
             d.state === 'STOPPED' ? `<b style="color:#f59e0b">STOPPED</b> (${d.stop_reason || ''})` : 'paper';
  document.getElementById('safety').innerHTML =
    `real orders: ${st} &middot; real losses in a row <b>${d.real_run}</b>/${d.max_real_run} (only a real WIN resets it) ` +
    `&middot; orders today ${d.attempts_today}/${d.max_real_bets_day} &middot; budget left $${(d.budget_left || 0).toFixed(2)} ` +
    `&middot; campaign ${(d.campaign_pnl || 0) >= 0 ? '+' : ''}${(d.campaign_pnl || 0).toFixed(2)} ` +
    `&middot; balance ${d.balance_usd != null ? '$' + d.balance_usd.toFixed(2) : '–'} ` +
    `&middot; clock ${clk.ok ? 'bc ' + (clk.offset >= 0 ? '+' : '') + clk.offset + 's' : 'exchange (' + (clk.why || '?') + ')'} ` +
    `&middot; gate ${g.open ? '<b style="color:#22d27a">open</b>' : 'closed'}: ${g.text || ''}` +
    (d.shadow && d.shadow.on && d.state !== 'ARMED' ? ` &middot; shadow: would-bet ${d.shadow.attempts_today} today, run ${d.shadow.run}, stops ${d.shadow.stops_today}` : '');
}
function state(d) {
  if (!d || d.error) return;
  runsBox(d); safetyLine(d);
  document.querySelectorAll('[data-model]').forEach(b => b.classList.toggle('on', b.dataset.model === d.model));
  document.querySelectorAll('[data-prog]').forEach(b => b.classList.toggle('on', Number(b.dataset.prog) === d.progression));
  document.querySelectorAll('[data-live]').forEach(b => b.classList.toggle('on', !d.live_wanted));
}
document.querySelectorAll('[data-model]').forEach(b => b.onclick = () => post({model: b.dataset.model}));
document.querySelectorAll('[data-prog]').forEach(b => b.onclick = () => post({progression: Number(b.dataset.prog)}));
document.querySelectorAll('[data-live]').forEach(b => b.onclick = () => post({live: false}));
async function refresh() {
  try { document.getElementById('panel').innerHTML = await (await fetch('/auto/panel' + q)).text(); }
  catch (e) { document.getElementById('panel').textContent = 'bot not reachable'; }
}
async function refreshState() { try { state(await (await fetch('/auto' + q)).json()); } catch (e) {} }
refresh(); refreshState(); setInterval(refresh, 500); setInterval(refreshState, 2000);

// ---- price chart: bc.game vs our price (the bot's weighted, delayed exchange mix) ----
function priceAt(arr, t) {                  // last value at or before t
  let lo = 0, hi = arr.length - 1, v = null;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][0] <= t) { v = arr[m][1]; lo = m + 1; } else hi = m - 1; }
  return v;
}
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
function ourPrice(d) {
  const line = d.our || [];
  return {line, at: t => priceAt(line, t)};
}
async function drawChart() {
  let d; try { d = await (await fetch('/auto/prices?s=60' + (q ? '&' + q.slice(1) : ''))).json(); } catch (e) { return; }
  const cv = document.getElementById('chart'), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 70, R = 12, T = 12, B = 22, x1 = d.now, x0 = x1 - 60;
  const bc = d.bc || [], our = ourPrice(d);
  const lines = [{pts: our.line, color: '#22d3ee', w: 1.6}, {pts: bc, color: '#facc15', w: 2.2}];
  // how different: bc.game minus our price at each bc.game tick, and 1 s moves in the same direction
  const diffs = bc.map(p => { const o = our.at(p[0]); return o == null ? null : [p[0], p[1] - o]; }).filter(x => x);
  let same = 0, moves = 0;
  for (let t = x1 - 59; t <= x1; t += 1) {
    const b0 = priceAt(bc, t - 1), b1 = priceAt(bc, t), o0 = our.at(t - 1), o1 = our.at(t);
    if ([b0, b1, o0, o1].includes(null) || b1 === b0 || o1 === o0) continue;
    moves++; same += Math.sign(b1 - b0) === Math.sign(o1 - o0);
  }
  const lastBc = bc.length ? bc[bc.length - 1][1] : null, lastOur = our.line.length ? our.line[our.line.length - 1][1] : null;
  const absd = diffs.map(x => Math.abs(x[1]));
  const f = (v, n = 2) => v == null ? '–' : v.toFixed(n);
  document.getElementById('legend').innerHTML = [
    `<span style="color:#facc15">&#9644; bc.game ${f(lastBc)}</span>`,
    `<span style="color:#22d3ee">&#9644; our price ${f(lastOur)}</span>`,
    `<span>now ${diffs.length ? (diffs[diffs.length - 1][1] >= 0 ? '+' : '') + f(diffs[diffs.length - 1][1]) : '–'}</span>`,
    `<span>average |difference| ${f(mean(absd))} &middot; max ${absd.length ? f(Math.max(...absd)) : '–'}</span>`,
    `<span>1 s moves in the same direction ${moves ? Math.round(100 * same / moves) + '%' : '–'} (${moves})</span>`,
  ].join('');
  document.getElementById('mixinfo').innerHTML = Object.entries(d.mix || {}).map(([k, m]) =>
    `<span style="color:${m.live ? '#8793a2' : '#ff5468'}">${m.label || k} ${Math.round(100 * m.weight)}%</span>`).join(' &middot; ');
  drawDiff(diffs, x0, x1);
  const o = d.our_vs_bc || {};
  const wr = (n, ok) => n ? `<b style="color:#f59e0b">${n - ok}</b> of ${n} = <b style="color:#f59e0b">${Math.round(100 * (n - ok) / n)}%</b>` : '–';
  const row = (p, title) => `<div style="margin-top:4px"><b>${title}</b>: wrong ${wr(o[p + '_n'], o[p + '_ok'])} rounds ` +
    `<span style="color:#8793a2">&middot; moved under $1: ${wr(o[p + '_small_n'], o[p + '_small_ok'])} &middot; ` +
    `$1-5: ${wr(o[p + '_mid_n'], o[p + '_mid_ok'])} &middot; over $5: ${wr(o[p + '_big_n'], o[p + '_big_ok'])} &middot; ` +
    `flat (no answer): ${o[p + '_flat'] || 0}</span></div>`;
  document.getElementById('ourbox').innerHTML =
    `<b>How often our price gave the WRONG Up/Down</b> (bc.game ended one way, our price the other)` +
    row('mix', 'NEW mix (futures, counted since this restart)') + row('our', 'old: Binance + Bitfinex') +
    `<div style="color:#8793a2;margin-top:4px">recording test (8 min it was not fitted on): same Up/Down as ` +
    `bc.game 80% (previous mix 75%); moves under $1 are a coin flip, over $5 match 98%</div>`;
  const ys = lines.flatMap(l => l.pts.filter(p => p[0] >= x0).map(p => p[1]));
  const r = d.round || {}; if (r.start_price) ys.push(r.start_price);
  if (ys.length < 2) return;
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0 || 1) * 0.12; y0 -= pad; y1 += pad;
  const X = t => L + (t - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  g.font = '11px Consolas, monospace'; g.fillStyle = '#8793a2'; g.strokeStyle = '#1f2630'; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.fillText(v.toFixed(1), 4, y + 4); }
  for (let s = 0; s <= 60; s += 10) { const x = X(x1 - s); g.fillText(s ? `-${s}s` : 'now', x - 12, H - 6); }
  for (const rd of (d.rounds || [])) {               // each round's 5 s window: gray while running,
    if (!rd.start || !rd.end || rd.end < x0) continue; // then green (UP) / red (DOWN) with a line at its end
    const rgb = rd.tie ? '148,163,184' : rd.win === 'UP' ? '34,210,122' : rd.win === 'DOWN' ? '255,84,104' : null;
    g.fillStyle = rgb ? `rgba(${rgb},0.16)` : 'rgba(148,163,184,0.13)';
    g.fillRect(Math.max(L, X(rd.start)), T, X(rd.end) - Math.max(L, X(rd.start)), H - T - B);
    if (rgb && X(rd.end) > L) {
      g.strokeStyle = `rgb(${rgb})`; g.lineWidth = 2.5; g.beginPath();
      g.moveTo(X(rd.end), T); g.lineTo(X(rd.end), H - B); g.stroke(); g.lineWidth = 1;
      g.fillStyle = `rgb(${rgb})`; g.font = 'bold 12px Consolas, monospace';
      g.fillText(rd.tie ? 'TIE' : rd.win === 'UP' ? '▲ UP' : '▼ DOWN', X(rd.end) + 5, T + 26);
      if (!rd.tie && rd.our_move != null) {          // what our price said for the same window
        const ours = rd.our_move > 0 ? 'UP' : rd.our_move < 0 ? 'DOWN' : null;
        const wrong = ours && ours !== rd.win;
        g.fillStyle = wrong ? '#f59e0b' : '#8793a2'; g.font = (wrong ? 'bold ' : '') + '11px Consolas, monospace';
        g.fillText(ours ? `ours ${ours === 'UP' ? '▲' : '▼'} ${ours} ${wrong ? '✗' : '✓'}` : 'ours: flat',
                   X(rd.end) + 5, T + 42);
      }
      g.font = '11px Consolas, monospace';
    }
  }
  if (r.cutoff && r.cutoff > x0) { g.strokeStyle = '#fbbf24'; g.setLineDash([4, 3]); g.beginPath();
    g.moveTo(X(r.cutoff), T); g.lineTo(X(r.cutoff), H - B); g.stroke(); g.setLineDash([]);
    g.fillStyle = '#fbbf24'; g.fillText('cutoff', X(r.cutoff) + 4, T + 10); }
  if (r.start_price) { const y = Y(r.start_price); g.strokeStyle = 'rgba(250,204,21,0.6)'; g.setLineDash([6, 4]);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.setLineDash([]);
    g.fillStyle = '#facc15'; g.fillText('start ' + r.start_price.toFixed(2), L + 4, y - 4); }
  for (const l of lines) {                           // our price first, bc.game on top
    const pts = l.pts.filter(p => p[0] >= x0 - 2); if (pts.length < 2) continue;
    g.strokeStyle = l.color; g.lineWidth = l.w; g.beginPath();
    pts.forEach((p, i) => { const x = Math.max(L, X(p[0])), y = Y(p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.lineTo(X(x1), Y(pts[pts.length - 1][1])); g.stroke();
  }
}
function drawDiff(diffs, x0, x1) {                    // bars: bc.game minus our price, at each bc.game tick
  const cv = document.getElementById('diff'), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 70, R = 12, T = 8, B = 8;
  const m = Math.max(1, ...diffs.map(x => Math.abs(x[1])));
  const X = t => L + (t - x0) / (x1 - x0) * (W - L - R), Y = v => T + (m - v) / (2 * m) * (H - T - B);
  g.font = '11px Consolas, monospace'; g.fillStyle = '#8793a2';
  g.fillText('+' + m.toFixed(1), 4, T + 8); g.fillText('0', 4, Y(0) + 4); g.fillText('-' + m.toFixed(1), 4, H - B);
  g.strokeStyle = '#2c3440'; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(W - R, Y(0)); g.stroke();
  for (const [t, v] of diffs) {
    if (t < x0) continue;
    g.fillStyle = v >= 0 ? 'rgba(34,210,122,0.8)' : 'rgba(255,84,104,0.8)';
    const x = X(t), y = Y(v), y0 = Y(0);
    g.fillRect(x - 1.5, Math.min(y, y0), 3, Math.abs(y - y0));
  }
}
drawChart(); setInterval(drawChart, 500);
</script></body></html>"""


async def start(trader: Trader, auto: AutoTrader, cfg: Config, tap=None) -> web.AppRunner:
    runner = web.AppRunner(build_app(trader, auto, cfg, tap), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, cfg.api_host, cfg.api_port).start()
    log.info("HTTP API on http://%s:%s  (POST /trade, GET /status, GET|POST /auto)", cfg.api_host, cfg.api_port)
    return runner
