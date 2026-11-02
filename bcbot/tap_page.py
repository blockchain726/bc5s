"""The Tap analyzer's page (GET /tap on the local API; data from GET /tap/state)."""

TAP_DASHBOARD = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>BC bot · Tap</title>
<style>
body{margin:0;padding:16px;background:#0b0e13;font:13px Consolas,monospace;color:#d6dde6}
a{color:#5aa9ff;text-decoration:none}.mute{color:#8793a2}.up{color:#22d27a}.down{color:#ff5468}.warn{color:#facc15}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.title{font-weight:700;letter-spacing:.6px}.tag{padding:1px 7px;border-radius:4px;background:#2c3440;font-weight:700}
.wrap{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:14px}
@media (max-width:1100px){.wrap{grid-template-columns:1fr}}
.c{background:#11141a;border:1px solid #2c3440;border-radius:9px;padding:10px 12px;margin-bottom:14px}
.ct{display:flex;justify-content:space-between;font-weight:700;font-size:12px;letter-spacing:.6px;margin-bottom:6px}
table{border-collapse:collapse;width:100%}th,td{padding:3px 6px;text-align:right;white-space:nowrap}
th{color:#8793a2;font-weight:normal}td.l,th.l{text-align:left}
.grid td{text-align:center;border:1px solid #1b2029;padding:4px 2px;line-height:1.15}
.grid td.px{text-align:right;color:#8793a2;border:none;padding-right:8px}.grid .now td.px{color:#facc15}
.grid .cell b{display:block;font-size:13px}.grid .cell span{font-size:10px}
.grid .off{opacity:.45}.grid .pick{outline:2px solid #facc15;outline-offset:-2px}
.big{font:700 22px Consolas,monospace}.row{display:flex;justify-content:space-between;gap:8px}
.res{display:flex;gap:8px;align-items:center;padding:3px 7px;border-radius:5px;margin-top:3px;font-weight:700}
.res.w{background:rgba(34,210,122,.16);color:#22d27a}.res.l{background:rgba(255,84,104,.16);color:#ff5468}
.res .m{margin-left:auto}.note{color:#8793a2;font-size:11px;margin-top:6px;line-height:1.4}
</style></head><body>
<div class="top">
  <div><a href="/">&larr; Up/Down dashboard</a></div>
  <div class="title">TAP ANALYZER &middot; <span id="sym">BTC/USD</span> <span class="tag" id="modetag">PAPER</span></div>
  <div id="status" class="mute"></div>
</div>
<div id="info" class="c"></div>
<div class="c" style="border-color:#b4232f">
  <div class="ct"><span>LIVE TAPS</span><span class="mute">real taps on bc.game's Tap game; every click here needs the API token</span></div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0">
    <span class="mute">mode</span>
    <button data-tapmode="off">OFF</button><button data-tapmode="demo" title="the page's DEMO account: real server multipliers, no money">DEMO</button>
    <button data-tapmode="real" class="liveb" title="account money">REAL</button>
    <button id="tapclear" title="clear a technical stop (rejections, an unanswered order); a budget stop stays">clear stop</button>
    <span class="mute" style="margin-left:10px">stake $</span><input id="tapstake" type="number" step="0.1" min="0.1" max="5" style="width:60px;font:inherit;color:#d6dde6;background:#1b2029;border:1px solid #2c3440;border-radius:6px;padding:4px 6px">
    <button id="setstake">set</button>
    <input type="password" id="tok" placeholder="API token (from .env)" title="the API_TOKEN from .env; kept in this browser" style="margin-left:10px;font:inherit;color:#d6dde6;background:#1b2029;border:1px solid #2c3440;border-radius:6px;padding:4px 8px;width:150px">
    <span id="tokst"></span></div>
  <div id="livest" class="mute"></div>
  <div id="liveres" style="font-weight:bold;min-height:16px"></div>
  <div id="livebook"></div>
</div>
<style>button{font:inherit;color:#d6dde6;background:#1b2029;border:1px solid #2c3440;border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:#2f6fd6;border-color:#2f6fd6;color:#fff}button.liveb.on{background:#b4232f;border-color:#b4232f}</style>
<div class="wrap">
  <div>
    <div class="c"><div class="ct"><span>THE GRID NOW</span><span class="mute">bc.game's multiplier &middot; our EV
      (green = our EV above 1) &middot; yellow box = paper tap</span></div>
      <table class="grid" id="grid"></table>
      <div class="note">Rows: price bands ($10). Columns: the 2 s windows starting next; dimmed ones can't be tapped now.
      A tap wins if the price touches its band inside its window. Multiplier = bc.game's model (0.97 / its P(hit),
      max &times;8). EV = the MEASURED hit rate of that cell kind at this band size (virtual taps settled on the
      real price, 1,000+ each) &times; the multiplier; cell kinds without enough measurements are never picked.</div></div>
    <div class="c"><div class="ct"><span>EVERY CELL TAPPED VIRTUALLY</span><span class="mute">real return per $1
      &middot; settled on the real price</span></div>
      <table id="cells"></table>
      <div class="note">Each column, every cell is "tapped" on paper at bc.game's multiplier and settled when its
      window ends: return = (hits &times; multiplier) / taps &minus; 1. Green = that kind of cell paid more than it cost.
      Rows: bands above (+) / below (&minus;) the price's band; columns: how many windows ahead.</div></div>
  </div>
  <div>
    <div class="c" id="paper"></div>
    <div class="c" id="models"></div>
    <div class="c"><div class="ct"><span>DOES OUR EV PREDICT IT?</span><span class="mute">virtual taps by our EV</span></div>
      <table id="ev"></table></div>
    <div class="c" id="public"></div>
  </div>
</div>
<script>
const q = location.search;
const pct = (v, d = 1) => v == null || isNaN(v) ? '–' : (100 * v).toFixed(d) + '%';
const sgn = (v, d = 1) => v == null || isNaN(v) ? '–' : (v >= 0 ? '+' : '') + (100 * v).toFixed(d) + '%';
const usd = v => v == null ? '–' : (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(2);
const tm = ms => new Date(ms).toLocaleTimeString('en-GB');
function shade(ret, n) {             // green for a positive return, red for negative; faint when few samples
  if (!n) return '';
  const a = Math.min(0.55, Math.abs(ret) * 1.2) * Math.min(1, n / 200) + 0.05;
  return `background:rgba(${ret >= 0 ? '34,210,122' : '255,84,104'},${a.toFixed(2)})`;
}
function grid(g) {
  const el = document.getElementById('grid');
  if (!g || !g.cells) { el.innerHTML = '<tr><td class="mute">waiting for the first column…</td></tr>'; return; }
  const ks = [...new Set(g.cells.map(c => c.k))].sort((a, b) => a - b);
  const rs = [...new Set(g.cells.map(c => c.r))].sort((a, b) => b - a);
  const at = {}; g.cells.forEach(c => at[c.k + ',' + c.r] = c);
  let h = '<tr><td class="px"></td>' + ks.map(k => {
    const c = at[k + ',0'];
    return `<th>${tm(c.t0)}<br><span class="mute">${c.tappable ? 'tap' : '–'}</span></th>`;
  }).join('') + '</tr>';
  for (const r of rs) {
    const c0 = at[ks[0] + ',' + r];
    h += `<tr class="${r === 0 ? 'now' : ''}"><td class="px">${c0.lo.toFixed(0)}${r === 0 ? ' ◀' : ''}</td>` + ks.map(k => {
      const c = at[k + ',' + r];
      const pick = g.pick && g.pick[0] === k && g.pick[1] === r;
      const dots = Object.entries(g.picks || {}).filter(([n, p]) => p && p[0] === k && p[1] === r && n !== 'label')
        .map(([n]) => `<i title="${n}" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin:0 1px;background:${MODEL_COLORS[n] || '#fff'}"></i>`).join('');
      const bg = c.ev >= 1 ? `rgba(34,210,122,${Math.min(0.6, (c.ev - 1) * 2 + 0.12).toFixed(2)})`
                           : `rgba(255,84,104,${Math.min(0.35, (1 - c.ev) * 0.5).toFixed(2)})`;
      return `<td class="cell ${c.tappable ? '' : 'off'} ${pick ? 'pick' : ''}" style="background:${bg}">` +
             `<b>×${c.odds.toFixed(2)}</b><span>EV ${c.ev.toFixed(2)}</span>${dots ? '<div>' + dots + '</div>' : ''}</td>`;
    }).join('') + '</tr>';
  }
  el.innerHTML = h;
}
function cells(rows) {
  const el = document.getElementById('cells');
  const at = {}; rows.forEach(s => at[s.key] = s);
  const ks = [0, 1, 2, 3, 4, 5], rs = [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6];
  let h = '<tr><th class="l">band</th>' + ks.map(k => `<th>+${k} col</th>`).join('') + '</tr>';
  for (const r of rs) {
    h += `<tr><td class="l mute">${r > 0 ? '+' + r : r}${r === 0 ? ' (price)' : ''}</td>` + ks.map(k => {
      const s = at[k + ',' + r];
      if (!s) return '<td class="mute">–</td>';
      return `<td style="${shade(s.ret, s.n)}" title="${s.n} taps · hit ${pct(s.hit)} · avg ×${s.odds.toFixed(2)} · bc.game P ${pct(s.pm)} · ours ${pct(s.po)}">${sgn(s.ret, 0)}</td>`;
    }).join('') + '</tr>';
  }
  const n = rows.reduce((a, s) => a + s.n, 0);
  el.innerHTML = h + `<tr><td class="l mute" colspan="7">${n.toLocaleString()} virtual taps settled (hover a cell for details)</td></tr>`;
}
function ev(rows) {
  const order = r => parseFloat(r.key);
  rows.sort((a, b) => order(a) - order(b));
  document.getElementById('ev').innerHTML = '<tr><th class="l">our EV</th><th>taps</th><th>hit</th><th>our P</th>' +
    '<th>bc.game P</th><th>avg ×</th><th>real return</th></tr>' + rows.map(s =>
    `<tr><td class="l">${s.key}</td><td>${s.n.toLocaleString()}</td><td>${pct(s.hit)}</td><td>${pct(s.po)}</td>` +
    `<td>${pct(s.pm)}</td><td>×${s.odds.toFixed(2)}</td><td style="${shade(s.ret, s.n)}">${sgn(s.ret)}</td></tr>`).join('');
}
const MODEL_COLORS = {};
function models(ms) {
  if (!ms.length) return;
  ms.forEach(m => MODEL_COLORS[m.name] = m.color);
  const since = ms.map(m => m.since).filter(Boolean);
  const t0 = since.length ? new Date(Math.min(...since) * 1000).toLocaleTimeString() : '–';
  document.getElementById('models').innerHTML =
    `<div class="ct"><span>PICKERS ON TRIAL</span><span class="mute">$1 paper per tap each, wins at bc.game's payout, since ${t0} &middot; <a href="#" id="mreset" style="color:#5aa9ff">reset books</a></span></div>` +
    `<table><tr><th style="text-align:left">picker</th><th>taps</th><th>won</th><th>P&amp;L</th><th>per $</th><th>last 1 h</th><th>now</th></tr>` +
    ms.map(m => `<tr><td style="text-align:left"><i style="display:inline-block;width:10px;height:10px;border:2px solid ${m.color};margin-right:6px;vertical-align:middle"></i><b>${m.name}</b><div class="mute" style="font-size:10px">${m.desc}</div></td>` +
      `<td>${m.bets || 0}</td><td>${m.bets ? pct(m.wins / m.bets) : '–'}</td><td class="${(m.pnl || 0) >= 0 ? 'up' : 'down'}">${usd(m.pnl || 0)}</td>` +
      `<td>${m.bets ? ((100 * (m.pnl || 0)) / m.bets).toFixed(0) + '%' : '–'}</td><td class="${(m.hour || 0) >= 0 ? 'up' : 'down'}">${usd(m.hour || 0)} <span class="mute">(${m.hour_n || 0})</span></td>` +
      `<td>${m.pick ? `+${m.pick[0]}c ${m.pick[1] >= 0 ? '+' : ''}${m.pick[1]}b` : (m.open ? 'open ' + m.open : '–')}</td></tr>`).join('') + `</table>`;
  document.getElementById('mreset').onclick = async e => { e.preventDefault(); if (confirm('Reset the trial books?')) await post({tap_models_reset: 1}); };
}
function paper(p) {
  const wr = p.bets ? p.wins / p.bets : null;
  const pay = p.payout || {};
  const payTxt = ['2', '3', '4'].filter(k => pay[k] != null).map(k => `+${k}c &times;${pay[k].toFixed(2)}`).join(' ');
  const s = p.strict || {};
  document.getElementById('paper').innerHTML =
    `<div class="ct"><span>PAPER TAPS</span><span class="mute">$1 on the best-EV tappable cell when EV &ge; ${p.ev_min}; a win pays what bc.game's server really pays (${payTxt} of the label, measured on our real orders)</span></div>` +
    `<div class="row"><span class="mute">P&amp;L at bc.game's payout</span><span class="mute">taps &middot; won</span></div>` +
    `<div class="row"><span class="big ${p.pnl >= 0 ? 'up' : 'down'}">${usd(p.pnl)}</span><span class="big">${p.bets} &middot; ${p.wins}</span></div>` +
    `<div class="mute">won ${pct(wr)} &middot; at the grid label it would read ${usd(p.pnl_label || 0)} (the old P&amp;L)${p.open ? ` &middot; open ${p.open} tap${p.open > 1 ? 's' : ''} ($${p.open} at risk)` : ''}</div>` +
    `<div class="mute">picks that also clear the bar at the paid odds (EV &times; payout &ge; ${p.min_ev}): <b class="${(s.pnl || 0) >= 0 ? 'up' : 'down'}">${usd(s.pnl || 0)}</b> on ${s.bets || 0} taps, won ${s.wins || 0}</div>` +
    (p.recent || []).slice(0, 8).map(r => `<div class="res ${r.hit ? 'w' : 'l'}"><span>+${r.k} col ${r.r >= 0 ? '+' : ''}${r.r} band</span>` +
      `<span>×${r.odds.toFixed(2)}${r.pay ? ` &rarr; ×${(r.odds * r.pay).toFixed(2)}` : ''}</span><span class="mute">EV ${r.ev.toFixed(2)}</span><b>${r.hit ? 'HIT' : 'MISS'}</b>` +
      `<span class="m">${usd(r.pnl)}</span></div>`).join('');
}
function pub(p) {
  const by = Object.entries(p.by_odds || {}).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
  document.getElementById('public').innerHTML =
    `<div class="ct"><span>OTHER PLAYERS' BTC/USD TAPS</span><span class="mute">from the page's live feed</span></div>` +
    `<div class="row"><span class="mute">taps</span><span>${p.n}</span></div>` +
    `<div class="row"><span class="mute">won</span><span>${pct(p.n ? p.won / p.n : null)}</span></div>` +
    `<div class="row"><span class="mute">average multiplier</span><span>×${p.n ? (p.odds / p.n).toFixed(2) : '–'}</span></div>` +
    `<div class="row"><span class="mute">players' return per $1</span><span>${sgn(p.n ? p.ret / p.n : null)}</span></div>` +
    `<div class="row"><span class="mute">our hit rule agrees with their result</span><span>${pct(p.checked ? p.agree_tick / p.checked : null)} ` +
    `<span class="mute">(cross rule ${pct(p.checked ? p.agree_cross / p.checked : null)}, ${p.checked})</span></span></div>` +
    `<div class="row"><span class="mute">their multiplier vs our port</span><span>${p.odds_ratio ? sgn(p.odds_ratio - 1) : '–'}</span></div>` +
    (by.length ? '<table style="margin-top:6px"><tr><th class="l">multiplier</th><th>taps</th><th>won</th><th>fair (0.97/×)</th></tr>' +
      by.map(([k, b]) => `<tr><td class="l">${k}</td><td>${b.n}</td><td>${pct(b.won / b.n)}</td><td>${pct(0.97 / (b.odds / b.n))}</td></tr>`).join('') + '</table>' : '');
}
// ---- live taps: the token (shared with the Up/Down dashboard), mode and stake ----
const tokBox = document.getElementById('tok');
const cleanTok = v => (v || '').trim().replace(/^API_TOKEN *= */i, '').replace(/^["']|["']$/g, '').trim();
tokBox.value = localStorage.apiToken || '';
function tokStatus(ok, text) { document.getElementById('tokst').innerHTML = ` <b class="${ok ? 'up' : 'down'}">${text}</b>`; }
async function checkTok() {
  const tok = cleanTok(tokBox.value);
  if (!tok) { tokStatus(false, '<- paste the token'); return false; }
  try {
    const r = await fetch('/auto' + q, {method: 'POST', headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + tok}, body: '{}'});
    if (r.ok) localStorage.apiToken = tok;
    tokStatus(r.ok, r.ok ? 'token OK' : 'token WRONG'); return r.ok;
  } catch (e) { tokStatus(false, 'bot not reachable'); return false; }
}
tokBox.addEventListener('change', checkTok); tokBox.addEventListener('paste', () => setTimeout(checkTok, 50));
if (tokBox.value) checkTok(); else tokStatus(false, '<- paste the token');
function result(ok, text) { const el = document.getElementById('liveres'); el.className = ok ? 'up' : 'down'; el.textContent = text; }
async function post(body) {
  const headers = {'Content-Type': 'application/json'};
  const tok = cleanTok(tokBox.value) || cleanTok(localStorage.apiToken);
  if (tok) headers['Authorization'] = 'Bearer ' + tok;
  const r = await fetch('/auto' + q, {method: 'POST', headers, body: JSON.stringify(body)});
  const d = await r.json();
  return {ok: !d.error, err: d.error ? d.error + (r.status === 401 || r.status === 403 ? ' (paste the API_TOKEN from .env in the token box)' : '') : '', d};
}
document.querySelectorAll('[data-tapmode]').forEach(b => b.onclick = async () => {
  const mode = b.dataset.tapmode;
  if (mode === 'real' && !confirm(`REAL taps with account money: $${document.getElementById('tapstake').value || '?'} per tap on the best cell when its EV is high enough.\nStops at the budget, pauses after a run of losses. Continue?`)) return;
  const res = await post({tap_live: mode});
  result(res.ok, res.ok ? `Tap live ${mode.toUpperCase()}` : 'not changed: ' + res.err);
});
document.getElementById('tapclear').onclick = async () => {
  const res = await post({tap_clear: true});
  result(res.ok, res.ok ? 'Tap stop cleared' : 'not cleared: ' + res.err);
};
document.getElementById('setstake').onclick = async () => {
  const res = await post({tap_stake: parseFloat(document.getElementById('tapstake').value)});
  result(res.ok, res.ok ? `stake $${res.d.tap_live.stake} per tap` : 'not changed: ' + res.err);
};
function live(l) {
  if (!l) return;
  document.querySelectorAll('[data-tapmode]').forEach(b => b.classList.toggle('on', b.dataset.tapmode === l.mode));
  const tag = document.getElementById('modetag');
  tag.textContent = l.mode === 'real' ? 'LIVE · REAL MONEY' : l.mode === 'demo' ? 'DEMO ACCOUNT' : 'PAPER';
  tag.style.background = l.mode === 'real' ? '#b4232f' : l.mode === 'demo' ? '#b7791f' : '#2c3440';
  const sb = document.getElementById('tapstake'); if (document.activeElement !== sb) sb.value = l.stake;
  const b = l.mode === 'real' ? l.real : l.demo;
  document.getElementById('livest').innerHTML =
    `mode <b>${l.mode.toUpperCase()}</b> &middot; stake <b>$${l.stake}</b> &middot; budget -$${l.budget} &middot; min EV ${l.min_ev} &middot; ` +
    `server/label ratio ${l.ratio ? '<b>' + l.ratio.toFixed(2) + '</b> (' + l.ratios_n + ' orders)' : `<b class="warn">assumed ${l.ratio_boot}</b> until ${l.ratio_min_n} orders (${l.ratios_n} so far)`} &middot; ` +
    `open taps ${l.open}/${l.max_open}` + (l.paused_s ? ` &middot; <b class="warn">paused ${l.paused_s}s</b>` : '') +
    (l.stopped ? ` &middot; <b class="down">STOPPED: ${l.stopped.why}</b>` : '') +
    `<div>why no tap now: <span class="warn">${l.last_reason || '–'}</span></div>`;
  document.getElementById('livebook').innerHTML = b ?
    `<div class="row" style="margin-top:4px"><span class="mute">${l.mode === 'real' ? 'REAL' : 'demo'} taps &middot; won</span><span><b>${b.n || 0}</b> &middot; ${b.won || 0}</span></div>` +
    `<div class="row"><span class="mute">P&amp;L</span><span class="big ${(b.pnl || 0) >= 0 ? 'up' : 'down'}">${usd(b.pnl || 0)}</span></div>` +
    (l.recent || []).slice(0, 6).map(r => `<div class="res ${r.result === 'WIN' ? 'w' : r.result ? 'l' : ''}"><span>${r.mode || ''} $${r.usd} &times;${r.odds ? r.odds.toFixed(2) : '?'} (label ${r.label ? r.label.toFixed(2) : '?'})</span><b>${r.result || 'open'}</b><span class="m">${r.pnl != null ? usd(r.pnl) : ''}</span></div>`).join('') : '';
}
async function tick() {
  let d; try { d = await (await fetch('/tap/state' + q)).json(); } catch (e) { document.getElementById('status').textContent = 'bot not reachable'; return; }
  if (d.error) { document.getElementById('status').textContent = d.error; return; }
  live(d.live);
  document.getElementById('status').innerHTML = (d.connected ? '<span class="up">● live</span>' : '<span class="down">● no price</span>') +
    ` &middot; ${d.price ? d.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '–'}`;
  const c = d.cfg || {};
  const ratio = d.vol_real && d.vol_cfg ? d.vol_real / d.vol_cfg : null;
  document.getElementById('info').innerHTML =
    `bc.game's settings: <b>${c.timeIncrement} s</b> columns &middot; <b>$${c.priceIncrement}</b> rows &middot; max <b>×${c.maxOdds}</b> &middot; ` +
    `its model's price step <b>$${d.vol_cfg_usd ? d.vol_cfg_usd.toFixed(2) : '–'}</b> per 0.5 s` +
    `${d.cfg_from_page ? '' : ' <span class="warn">(defaults: page settings not read yet)</span>'}` +
    (d.bc_read && d.bc_read.n ? ` &middot; multipliers <b class="up">read from bc.game's grid</b> (${d.bc_read.n} cells, $${d.step_inferred} bands on screen)`
                              : ` &middot; multipliers <b class="warn">from our port</b> (grid not read: ${d.bc_read ? d.bc_read.err : '–'})`) +
    ` &nbsp;|&nbsp; the real price (last 10 min): <b>$${d.vol_real_usd ? d.vol_real_usd.toFixed(2) : '–'}</b> per 0.5 s` +
    (ratio ? ` = <b class="${ratio < 1 ? 'up' : 'down'}">${(100 * ratio).toFixed(0)}%</b> of bc.game's` : '') +
    `<div class="note">Below 100%: the price moves less than bc.game's model assumes, so cells near the price get hit more often ` +
    `than their multiplier prices in, far cells less often. Statistics since ${new Date(d.since * 1000).toLocaleString()}.</div>`;
  const rg = d.regime || {};
  document.getElementById('info').innerHTML += `<div style="margin-top:4px">regime now: <b>${rg.name || '?'}</b> &middot; last 60 current-band taps hit ${rg.trailing != null ? Math.round(100 * rg.trailing) + '%' : '–'} &middot; 60 s price range ${rg.range_bands != null ? rg.range_bands + ' bands' : '–'} &middot; hit-rate factor <b>${rg.factor}</b> <span class="mute">(calm+hot 1.20 &middot; calm+cold 1.04 &middot; wild+hot 0.94 &middot; wild+cold 0.76, from 70k measured taps)</span></div>`;
  models(d.models || []); grid(d.grid); cells(d.cells || []); ev(d.ev || []); paper(d.paper || {}); pub(d.public || {});
}
tick(); setInterval(tick, 1000);
</script></body></html>"""
