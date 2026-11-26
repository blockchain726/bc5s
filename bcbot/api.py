"""Local HTTP API so other programs (signal scripts, webhooks, curl) can place trades.

  GET  /status                          -> current balance / round state
  POST /trade  {"side": "up", "amount": 5}  -> a manual order (only while armed; counted like every real order)
  GET  /auto                            -> auto mode: accuracy, paper P&L, last call, models, the real-money state
  POST /auto   {"on": true, "model": "fade", "progression": 1.5}   -> auto mode, calling model, paper progression
  POST /auto   {"arm": true, "ack_run": 0, "day": "2026-10-30", "force": false, "ladder": false}  -> allow real orders
               (typed ack; "ladder": true with "force" uses the compensate ladder although the gate is closed)
  POST /auto   {"live": false} | {"kill": true} | {"clear": "unknown", "outcome": "none"}
               {"reset_run": true, "ack_run": 6, "reason": "new model"}  -> the run of real losses back to 0 (logged)
               {"max_real_run": 6} | {"daily_loss_usd": 30}  -> the operator's real-money limits: the
               losses-in-a-row stop (with x2 also the length of the stake series) and the day's loss
               {"tap_live": "off|demo|real"} | {"tap_stake": 0.2}  -> live taps on the Tap game (token; see /tap)
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
from .lab_display import read_status as read_lab_status
from .tap_page import TAP_DASHBOARD
from .trader import TradeError, Trader, status_dict

log = logging.getLogger(__name__)


def build_app(trader: Trader, auto: AutoTrader, cfg: Config, tap=None) -> web.Application:
    MONEY_KEYS = ("arm", "live", "kill", "clear", "reset_run", "max_real_run", "daily_loss_usd", "ladder", "tap_live", "tap_stake", "tap_clear",
                  "tap_live_model")

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
        if getattr(auto, "startup_problem", None):
            return web.json_response({"ok": False, "error": auto.startup_problem}, status=409)
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
                auto.set_progression(float(body["progression"]), who="api")
            if body.get("kill"):
                auto.kill(who="api")
            if body.get("arm"):
                if getattr(auto, "startup_problem", None):
                    raise TradeError(auto.startup_problem)
                auto.arm(int(body.get("ack_run", -1)), str(body.get("day", "")), who="api",
                         force=bool(body.get("force")), ladder=bool(body.get("ladder")))
            if "live" in body:
                if body["live"]:
                    raise TradeError('arming is {"arm": true, "ack_run": <losses in a row>, "day": "YYYY-MM-DD"}')
                auto.disarm("live off", who="api")
            if body.get("clear"):
                auto.clear(str(body["clear"]), body.get("outcome"), who="api")
            if body.get("reset_run"):
                auto.reset_run(int(body.get("ack_run", -1)), str(body.get("reason", "")), who="api")
            for key in auto.LIMITS:                                  # the operator's own real-money limits
                if key in body:
                    auto.set_limit(key, body[key], who="api")
            if "ladder" in body and not body.get("arm"):             # switch the ladder while armed
                auto.set_ladder(bool(body["ladder"]), who="api")
            if body.get("stats_reset"):                                           # 'P&L -> 0': the shown record
                auto.reset_stats(who="api")
            if body.get("tap_models_reset"):                                      # the trial pickers' paper books
                if tap is None:
                    raise TradeError("the Tap analyzer is not running (start without --no-tap)")
                tap.reset_models()
                return web.json_response({"ok": True, "models": tap.models_summary()})
            if "tap_live_model" in body:                                          # which pickers the live tap follows
                if tap is None:
                    raise TradeError("the Tap analyzer is not running (start without --no-tap)")
                try:
                    msg = tap.live.set_live_model(str(body["tap_live_model"]), who="api")
                except ValueError as e:
                    raise TradeError(str(e))
                tap.live_rank_at = 0.0                                           # re-rank at the next snapshot
                return web.json_response({"ok": True, "message": msg})
            if "tap_live" in body or "tap_stake" in body or body.get("tap_clear"):   # live taps (Tap page)
                if tap is None:
                    raise TradeError("the Tap analyzer is not running (start without --no-tap)")
                try:
                    if "tap_stake" in body:
                        tap.live.set_stake(float(body["tap_stake"]), who="api")
                    if body.get("tap_clear"):
                        tap.live.clear()                              # a technical stop (not a budget stop)
                    if "tap_live" in body:
                        tap.live.set_mode(str(body["tap_live"]), who="api")
                except ValueError as e:
                    raise TradeError(str(e))
                return web.json_response({"ok": True, "tap_live": tap.live.summary()})
        except TradeError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        except (ValueError, TypeError):
            return web.json_response({"ok": False, "error": 'expected JSON {"on": true, "live": false}'}, status=400)
        return await auto_get(request)

    async def panel(_: web.Request) -> web.Response:
        return web.Response(text=auto.panel_html(), content_type="text/html")

    async def prices(request: web.Request) -> web.Response:
        window = min(300.0, max(10.0, float(request.query.get("s", 60))))
        return web.json_response(auto.prices(window, exchanges=request.query.get("ex") == "1"))

    async def pnl(request: web.Request) -> web.Response:
        """The P&L graph of the page: the calling model's paper bets and the real orders of the same time."""
        return web.json_response(auto.pnl_curve(request.query.get("r", "all"), request.query.get("model")))

    async def model_lab(_: web.Request) -> web.Response:
        return web.json_response(read_lab_status(cfg.auto_log.parent))

    async def dashboard(_: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD, content_type="text/html")

    async def tap_page(_: web.Request) -> web.Response:
        return web.Response(text=TAP_DASHBOARD, content_type="text/html")

    async def tap_state(_: web.Request) -> web.Response:
        if tap is None:
            return web.json_response({"error": "the Tap analyzer is not running"})
        try:
            return web.json_response(tap.state())
        except Exception as e:                    # its tab opens a little after the Up/Down page
            return web.json_response({"error": f"the Tap analyzer is starting ({type(e).__name__})"})

    app = web.Application(middlewares=[auth])
    app.add_routes([web.get("/", dashboard), web.get("/status", status), web.post("/trade", trade),
                    web.get("/auto", auto_get), web.post("/auto", auto_post), web.get("/auto/panel", panel),
                    web.get("/auto/prices", prices), web.get("/auto/pnl", pnl), web.get("/auto/model-lab", model_lab),
                    web.get("/tap", tap_page), web.get("/tap/state", tap_state)])
    return app


DASHBOARD = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>BC Up/Down</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0a0d12;--card:#10141b;--line:#1f2733;--text:#e6ebf2;--mute:#7d8896;--dim:#39424f;--accent:#3b82f6;
      --up:#22d27a;--down:#ff5468;--warn:#f59e0b}
*{box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#2c3644 transparent}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 Inter,"Segoe UI",system-ui,sans-serif;
     font-variant-numeric:tabular-nums}
.top{display:flex;align-items:center;gap:16px;height:52px;padding:0 24px;border-bottom:1px solid var(--line);
     background:#0d1117;position:sticky;top:0;z-index:5}
.brand{font-weight:700;font-size:15px}.brand span{color:var(--mute);font-weight:500}
.nav{display:flex;gap:4px}.nav a{color:var(--mute);text-decoration:none;padding:6px 12px;border-radius:6px;font-weight:600}
.nav a.on{color:var(--text);background:#1a2130}.nav a:hover{color:var(--text)}
.sp{flex:1}.mute{color:var(--mute)}.dim{color:var(--dim)}
.badge{font-size:11px;font-weight:700;letter-spacing:.6px;padding:3px 9px;border-radius:999px;background:#1a2130;color:var(--mute)}
.badge.live{background:#b4232f;color:#fff}.badge.stop{background:#7a4d0b;color:#ffd58a}
input,select,button{font:inherit;color:var(--text)}
input[type=password],select{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:6px 9px;outline:none}
input[type=password]:focus,select:focus{border-color:var(--accent)}
#tok{width:130px}
.tokst{font-size:12px;font-weight:600;display:inline-flex;align-items:center;gap:6px}
.tokst:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--down)}
.tokst.ok{color:var(--mute)}.tokst.ok:before{background:var(--up)}.tokst.bad{color:var(--down)}
.wrap{max-width:1560px;margin:0 auto;padding:20px 24px 40px;display:flex;flex-direction:column;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.kpi .l{color:var(--mute);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.7px}
.kpi .v{font-size:22px;font-weight:700;margin:2px 0 1px;white-space:nowrap}
.kpi .s{color:var(--mute);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:10px 16px}
.fld{display:flex;align-items:center;gap:8px}.fld>span,.chk{color:var(--mute);font-size:12px;font-weight:600}
.chk{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:#0d1117;border:0;border-right:1px solid var(--line);padding:6px 12px;cursor:pointer;color:var(--mute)}
.seg button:last-child{border-right:0}.seg button.on{background:var(--accent);color:#fff}
.btn{background:#161c26;border:1px solid var(--line);border-radius:6px;padding:6px 14px;cursor:pointer;font-weight:600;
     list-style:none}.btn:hover{border-color:#34404f}.btn::-webkit-details-marker{display:none}
.btn.danger{background:#b4232f;border-color:#b4232f;color:#fff}.btn.danger:disabled{opacity:.9;cursor:default}
.menu{position:relative}.menu>div{position:absolute;right:0;top:36px;z-index:4;background:#161c26;border:1px solid var(--line);
     border-radius:8px;padding:6px;display:flex;flex-direction:column;gap:2px;min-width:180px;box-shadow:0 8px 24px rgba(0,0,0,.5)}
.menu>div button{background:none;border:0;text-align:left;padding:7px 10px;border-radius:5px;cursor:pointer}
.menu>div button:hover{background:#222b39}
#msg{color:var(--warn);font-weight:600;padding:8px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
#msg:empty{display:none}
.seg.mode button{font-weight:700;padding:6px 18px}.seg.mode button.on{background:#2a3444;color:#fff}
.seg.mode button.livebtn.on{background:#b4232f;color:#fff}
.dlg{position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center}
.dlg[hidden]{display:none}.dlg .box{background:#141a23;border:1px solid var(--line);border-radius:12px;padding:20px 22px;
     width:min(520px,92vw);box-shadow:0 20px 60px rgba(0,0,0,.6)}.dlg h2{font-size:16px;margin-bottom:12px}
.dlg input{width:100%;margin-top:8px;background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:7px 9px;outline:none}
.dlg input[hidden]{display:none}.dlg .acts{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
.tbl.kv td{white-space:normal}.tbl.kv td:first-child{color:var(--mute);white-space:nowrap}.tbl.kv tr:first-child td{border-top:0}
.grid{display:grid;grid-template-columns:auto minmax(0,1fr);gap:16px;align-items:start}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
#panel{position:sticky;top:68px;zoom:1.2}#panel .p{max-height:none;overflow:visible}#panel .hd{background:none}
#panel .pnlc,#panel .modc{display:none}#panel .c{box-shadow:none;border-width:1px;border-color:var(--line)}
.ch{display:flex;align-items:center;gap:14px;margin-bottom:10px;flex-wrap:wrap}
h2{margin:0;font-size:14px;font-weight:700}.cap{color:var(--mute);font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mute);font-size:12px}.legend b{color:var(--text);font-weight:600}
.legend i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
canvas{display:block;width:100%;border-radius:6px;background:#0c1016}#chart{height:320px}#diff{height:80px}
.tbl{width:100%;border-collapse:collapse}.tbl th{color:var(--mute);font-size:11px;font-weight:600;text-transform:uppercase;
     letter-spacing:.6px;text-align:right;padding:6px 10px;white-space:nowrap}
.tbl td{text-align:right;padding:6px 10px;border-top:1px solid var(--line);white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left}.tbl tr.sel td{background:rgba(59,130,246,.09)}
.tbl tr.sel td:first-child{box-shadow:inset 3px 0 var(--accent)}.tbl td.h{color:var(--mute)}
.tbl.runs td,.tbl.runs th{padding:4px 7px}.tbl.runs .amb{font-size:11px}.tbl.runs td{border-top:0}.tbl.runs tr.ft td{border-top:1px solid var(--line)}
.tbl.runs tr.sec td{color:var(--mute);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;
     padding-top:12px;text-align:left}
#pnl{height:230px;cursor:crosshair}.pnlwrap{position:relative}
.tip{position:absolute;top:10px;z-index:3;pointer-events:none;background:#161c26;border:1px solid var(--line);border-radius:8px;
     padding:7px 10px;font-size:12px;white-space:nowrap;box-shadow:0 8px 24px rgba(0,0,0,.5)}.tip[hidden]{display:none}
.tip .h{color:var(--mute);margin-bottom:4px}.tip .r{display:flex;align-items:center;gap:8px}
.tip .r i{display:inline-block;width:12px;height:2px;border-radius:2px}.tip .r b{min-width:62px}.tip .r span{color:var(--mute)}
.btn.on{border-color:var(--accent);color:#fff}
.calling{font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--accent);margin-left:8px}
.red{color:var(--down)}.amb{color:var(--warn)}
details.card>summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px}
details.card>summary::-webkit-details-marker{display:none}details.card>summary:after{content:"+";color:var(--mute);margin-left:auto}
details.card[open]>summary:after{content:"−"}details.card[open]>summary{margin-bottom:10px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.chips span{border:1px solid var(--line);border-radius:999px;
     padding:2px 10px;font-size:12px;color:var(--mute)}.chips span.off{color:var(--down);border-color:var(--down)}
@media(max-width:1200px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){.grid{grid-template-columns:1fr}#panel{position:static}}
</style></head>
<body>
<header class="top">
  <div class="brand">BC.GAME <span>bot</span></div>
  <nav class="nav"><a class="on" href="/">Up/Down</a><a href="/tap">Tap</a><a href="#market-memory">Market Memory</a></nav>
  <span id="realst"></span>
  <div class="sp"></div>
  <span id="clock" class="cap"></span>
  <input type="password" id="tok" placeholder="API token" title="API_TOKEN from .env; kept in this browser only">
</header>
<div class="wrap">
<section class="card" id="startupnotice" hidden style="color:var(--warn)"></section>
<section class="kpis" id="kpis"></section>
<section class="card toolbar">
  <label class="fld"><span>Model</span><select id="modelsel">
    <optgroup label="Chance x payout"><option value="value">Value bet (chance x payout)</option></optgroup>
    <optgroup label="Snap-back"><option value="fade">Gap fade</option><option value="fade_strict">Gap fade strict</option><option value="fade_flow">Gap fade + flow</option><option value="fade_flow2">Gap fade + flow, 2x</option><option value="fade_hawkes">Gap fade + Hawkes</option><option value="fade_fair">Gap fade (new price)</option><option value="overshoot">Overshoot</option><option value="overshoot_rel">Overshoot scaled</option>
      <option value="overshoot_book">Overshoot + book</option><option value="revert">Quiet reversal</option></optgroup>
    <optgroup label="Fitted on the log"><option value="prob">P(UP)</option><option value="ens">P(UP) ensemble</option>
      <option value="trees_mid">Trees mid</option></optgroup>
    <optgroup label="Payout, no prediction"><option value="thin">Thin side (pays 2.10+)</option></optgroup>
    <optgroup label="Order books, six markets"><option value="ob_logit">Logistic regression (order books)</option><option value="ob_lgbm">LightGBM (order books)</option>
      <option value="ob_xgb">XGBoost (order books)</option><option value="ob_deeplob">DeepLOB (order books)</option><option value="sign">HawkesSign (six markets)</option><option value="ob_steady">Steady order book (strong 5 s+)</option></optgroup></select></label>
  <div class="fld" title="Paper and LIVE use the same rule"><span>Stake after a loss</span><div class="seg"><button data-prog="1">Flat</button><button data-prog="1.5">&times;1.5</button><button data-prog="2">&times;2</button></div></div>
  <div class="sp"></div>
  <label class="fld" title="Real orders stop at this many real losses in a row (needs the token)"><span>Stop after</span><select id="maxrun">
    <option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option>10</option></select><span>losses in a row</span></label>
  <label class="fld" title="No more real orders today once this much is lost today (needs the token)"><span>Day loss limit</span><select id="dayloss">
    <option value="8">$8</option><option value="15">$15</option><option value="30">$30</option><option value="60">$60</option><option value="100">$100</option><option value="150">$150</option><option value="250">$250</option></select></label>
  <div class="seg mode"><button data-live="0" title="Paper only: no real orders">Paper</button><button data-arm="1" class="livebtn" title="Real orders by the selected model; a confirmation shows the stake and every stop">LIVE</button></div>
  <button class="btn" data-clear="1" style="display:none;border-color:var(--warn);color:var(--warn)" title="Real orders are stopped for a technical reason. Check bc.game, then clear the stop here (needs the token).">Clear stop</button>
  <details class="menu"><summary class="btn">Reset</summary><div>
    <button data-reset="1" title="Set the run of real losses back to 0 (written to the events log)">Real loss run &rarr; 0</button>
    <button id="statsreset" title="Accuracy and P&amp;L count from now; the round log and the gate's record are kept">Paper P&amp;L &rarr; 0</button></div></details>
</section>
<div id="msg"></div>
<div id="dlg" class="dlg" hidden><div class="box"><h2 id="dlgt"></h2><div id="dlgb"></div><input id="dlgi" type="text">
  <div class="acts"><button class="btn" id="dlgno">Cancel</button><button class="btn" id="dlgok">OK</button></div></div></div>
<div class="grid">
<aside id="panel"></aside>
<div class="col">
  <section class="card">
    <div class="ch"><h2>BTC price</h2><div id="legend" class="legend"></div><div class="sp"></div>
      <label class="chk" title="Each exchange's price as the mix uses it"><input type="checkbox" id="showex"> Exchanges</label>
      <select id="win"><option value="30">30 s</option><option value="60">60 s</option><option value="120">2 min</option><option value="300">5 min</option></select></div>
    <canvas id="chart"></canvas>
    <div class="ch" style="margin:12px 0 6px"><span class="cap">Gap: bc.game minus our price ($)</span><div class="sp"></div><span id="gapinfo" class="cap"></span></div>
    <canvas id="diff"></canvas>
  </section>
  <section class="card">
    <div class="ch"><h2 id="pnlt">P&amp;L of the current model</h2><span class="cap" id="pnlcap"></span><div class="sp"></div>
      <div class="seg" id="pnlrange"><button data-r="1h" title="The last hour">1 h</button><button data-r="6h" title="The last 6 hours">6 h</button><button data-r="today" title="Since 00:00 on this PC">Today</button><button data-r="all" title="Since this model's row in the Models table began: the Paper line ends at the row's P&amp;L">All</button></div>
      <button class="btn" id="runsbtn" title="Every model's losing runs today, by length">Losing runs</button></div>
    <div id="pnlinfo" class="legend" style="margin-bottom:8px"></div>
    <div class="pnlwrap"><canvas id="pnl"></canvas><div id="pnltip" class="tip" hidden></div></div>
    <div id="runswrap" hidden>
      <div class="ch" style="margin:16px 0 10px"><h2>Losing runs today</h2><span class="cap">paper &middot; since 00:00 &middot; red = 5 or more</span></div>
      <div id="runs" style="overflow-x:auto"></div>
    </div>
  </section>
  <section class="card">
    <div class="ch"><h2>Models</h2><span class="cap">paper &middot; every model at a flat $1 &middot; since the last reset</span></div>
    <div id="models" style="overflow-x:auto"></div>
    <div id="obinfo" class="cap" style="margin-top:10px"></div>
  </section>
  <section class="card" id="market-memory" style="scroll-margin-top:68px">
    <div class="ch"><h2>Market Memory</h2><span class="badge">PAPER RESEARCH</span><div class="sp"></div><span id="labstate" class="cap">Loading</span></div>
    <div id="labmodels" style="overflow-x:auto"></div>
    <div id="lablatest" class="cap" style="margin-top:10px"></div>
    <div id="labtraining" class="cap" style="margin-top:6px"></div>
    <div class="cap" style="margin-top:6px">Live forward results, flat $1. Records stay with each version across retraining and paper resets. Runs automatically in paper research.</div>
  </section>
  <details class="card">
    <summary><h2>Price feed</h2><span class="cap">our price against bc.game's result</span></summary>
    <div id="mixinfo" class="chips"></div>
    <div id="ourbox" style="overflow-x:auto"></div>
  </details>
</div>
</div>
</div>
<script>
const q = location.search;
const tokBox = document.getElementById('tok');
const cleanTok = v => (v || '').trim().replace(/^API_TOKEN *= */i, '').replace(/^["']|["']$/g, '').trim();
tokBox.value = localStorage.apiToken || '';
function tokStatus(ok, text) {
  let el = document.getElementById('tokst');
  if (!el) { el = document.createElement('span'); el.id = 'tokst'; tokBox.after(el); }
  el.className = 'tokst ' + (ok ? 'ok' : 'bad'); el.textContent = text;
}
async function checkTok() {                       // an empty POST: accepted only with the right token, changes nothing
  const tok = cleanTok(tokBox.value);
  if (!tok) { tokStatus(false, 'No token'); return false; }
  try {
    const r = await fetch('/auto' + q, {method: 'POST', headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + tok}, body: '{}'});
    const ok = r.ok;
    if (ok) localStorage.apiToken = tok;
    tokStatus(ok, ok ? 'Token OK' : 'Wrong token');
    return ok;
  } catch (e) { tokStatus(false, 'Offline'); return false; }
}
tokBox.addEventListener('change', checkTok);
tokBox.addEventListener('paste', () => setTimeout(checkTok, 50));
if (tokBox.value) checkTok(); else tokStatus(false, 'No token');
async function post(body) {
  const headers = {'Content-Type': 'application/json'};
  const tok = cleanTok(tokBox.value) || cleanTok(localStorage.apiToken);
  if (tok) headers['Authorization'] = 'Bearer ' + tok;
  const r = await fetch('/auto' + q, {method: 'POST', headers, body: JSON.stringify(body)});
  const d = await r.json();
  const err = d.error ? d.error + (r.status === 401 || r.status === 403 ? '  (the token box: paste the text after API_TOKEN= in .env)' : '') : '';
  const m = document.getElementById('msg'); m.textContent = err; m.style.color = err ? '#ff5468' : '';
  if (!d.error) state(d);
  return {ok: !d.error, err, d};
}
function result(ok, text) {                      // a visible result line under the buttons
  const m = document.getElementById('msg'); m.style.color = ok ? '#22d27a' : '#ff5468'; m.textContent = text;
}
const localDay = () => { const n = new Date(); return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, '0')}-${String(n.getDate()).padStart(2, '0')}`; };
function ask(o) {                                // an in-page dialog (browser pop-ups are blocked in some viewers)
  return new Promise(done => {
    const el = id => document.getElementById(id);
    el('dlgt').textContent = o.title; el('dlgb').innerHTML = o.html || '';
    el('dlgi').hidden = o.input == null; el('dlgi').value = o.input || '';
    el('dlgok').textContent = o.ok || 'OK'; el('dlgok').className = 'btn' + (o.danger ? ' danger' : '');
    el('dlgno').hidden = !!o.info;
    el('dlg').hidden = false;
    const close = v => { el('dlg').hidden = true; done(v); };
    el('dlgok').onclick = () => close(o.input != null ? el('dlgi').value : true);
    el('dlgno').onclick = () => close(null);
  });
}
const kv = rows => '<table class="tbl kv">' + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>';
async function clearClick(d) {                   // a technical stop: the owner checks bc.game, then clears it here
  if (!d || !d.state) { try { d = await (await fetch('/auto' + q)).json(); } catch (e) { result(false, 'Bot not reachable'); return false; } }
  if (!d.tech_stop) return true;
  const unknown = d.tech_stop === 'unknown';
  const ans = await ask({title: 'Real orders are stopped', ok: 'Clear stop', input: unknown ? 'none' : null, html:
    `<div style="white-space:normal">${esc(d.stop_text || d.stop_reason || d.tech_stop)}</div>` +
    `<div class="cap" style="margin-top:10px">The bot sees now: wallet ${esc(d.balance_currency || '?')}, balance $${(d.balance_usd == null ? 0 : d.balance_usd).toFixed(2)}, account ${esc(d.balance_account || '?')}.</div>` +
    `<div class="cap" style="margin-top:6px">Clearing does not switch real orders on. It only lets you press LIVE again.</div>` +
    (unknown ? `<div class="cap" style="margin-top:10px">What does the bet history on bc.game show for that order? Type won, lost or none (no order was placed):</div>` : '')});
  if (ans == null || ans === false) return false;
  const res = await post(unknown ? {clear: d.tech_stop, outcome: String(ans).trim().toLowerCase()} : {clear: d.tech_stop});
  result(res.ok, res.ok ? 'Stop cleared. Real orders are still off (Paper): press LIVE to switch them on.' : 'Not cleared: ' + res.err);
  if (!res.ok) await ask({title: 'Stop not cleared', html: esc(res.err), info: true});
  return res.ok;
}
async function armClick() {
  let d; try { d = await (await fetch('/auto' + q)).json(); } catch (e) { result(false, 'Bot not reachable'); return; }
  if (d.state === 'ARMED') return;
  if (d.tech_stop) {                             // LIVE while a technical stop is in force: clear it first, then go on
    if (!await clearClick(d)) return;
    try { d = await (await fetch('/auto' + q)).json(); } catch (e) { result(false, 'Bot not reachable'); return; }
  }
  const lad = false, g = d.gate || {};
  const series = (d.real_ladder || [1]).map(x => '$' + +(x * (d.stake_usd || 1)).toFixed(2));
  const stake = d.ladder_on ? `next $${d.next_real_stake} &middot; series ${series.join(' / ')} (x${d.progression} after a loss, back to ${series[0]} after a win)` : 'flat $' + (d.next_real_stake || 1);
  const go = await ask({title: 'Switch to LIVE: real orders', ok: 'Go LIVE', danger: true, html: kv([
    ['Model', esc(LABEL[d.model] || d.model)], ['Stake', stake],
    ...(d.model === 'value' && d.ladder_on ? [['Note', 'Value bet was measured at a flat $1. It wins about half its bets: runs of six losses come about 3 times a day (the longest on the log: 9). A bigger flat stake did better than a series at the same worst drop.']] : []),
    ...(d.model === 'thin' && d.ladder_on ? [['Note', 'Thin side was measured at a flat $1. Your own stake goes into the thin pool: $32 into a $105 pool is paid about x1.9 instead of x2.2, and a doubling series keeps about a third of the return per $1.']] : []),
    ['Real losses in a row', `${d.real_run} / ${d.max_real_run}` + (d.real_run >= d.max_real_run ? ' (past the limit: ONE order, a loss stops it again)' : '')],
    ['Budget left today', '$' + (d.budget_left || 0).toFixed(2)], ['Orders today', d.attempts_today + (d.max_real_bets_day ? ' / ' + d.max_real_bets_day : ' (no daily limit)')],
    ['Campaign', (d.campaign_pnl || 0).toFixed(2)], ['Gate', g.open ? 'open' : 'closed: no proven edge yet (your decision, logged)']]) +
    `<div class="cap" style="margin-top:8px">Stops: ${d.max_real_run} losses in a row, $${d.daily_loss_usd} lost in a day, $${d.campaign_loss_usd} lost in all${d.arm_ttl_h ? ', ' + d.arm_ttl_h + ' hours after switching on' : ''}.</div>`});
  if (!go) return;
  const res = await post({arm: true, ack_run: d.real_run, day: localDay(), force: true, ladder: lad});
  result(res.ok, res.ok ? `LIVE: real orders are on (${res.d.ladder_on ? 'ladder' : 'flat $' + (res.d.next_real_stake || 1)})` : 'Not switched to LIVE: ' + res.err);
  if (!res.ok) await ask({title: 'Not switched to LIVE', html: esc(res.err), info: true});
}
async function paperClick() {
  const res = await post({live: false});
  result(res.ok, res.ok ? 'Paper: real orders are off' : 'Not changed: ' + res.err);
}
async function resetClick() {
  let d; try { d = await (await fetch('/auto' + q)).json(); } catch (e) { result(false, 'Bot not reachable'); return; }
  const reason = await ask({title: `Set the run of real losses (now ${d.real_run}) back to 0?`, html: '<div class="cap">Reason (goes into the events log):</div>', input: 'new model', ok: 'Reset'});
  if (reason == null) return;
  const res = await post({reset_run: true, ack_run: d.real_run, reason: 'dashboard: ' + reason});
  result(res.ok, res.ok ? 'Real losses in a row set to 0' : 'Not reset: ' + res.err);
}
document.querySelector('[data-arm]').onclick = armClick;
document.querySelector('[data-clear]').onclick = () => clearClick();
document.querySelector('[data-live]').onclick = paperClick;
const LIMIT_BOX = {maxrun: 'max_real_run', dayloss: 'daily_loss_usd'};   // the operator's own limits
for (const [id, key] of Object.entries(LIMIT_BOX)) document.getElementById(id).onchange = async e => {
  const res = await post({[key]: Number(e.target.value)});
  result(res.ok, res.ok ? `Saved: stop after ${res.d.max_real_run} losses in a row (largest stake $${res.d.max_stake_usd}), day loss limit $${res.d.daily_loss_usd}` : 'Not changed: ' + res.err);
  e.target.blur();
};
document.querySelector('[data-reset]').onclick = resetClick;
document.getElementById('statsreset').onclick = async () => {
  if (!await ask({title: 'Count paper accuracy and P&L from now?', html: '<div class="cap">The round log, the gate record and the real-loss counters are kept.</div>', ok: 'Reset'})) return;
  const res = await post({stats_reset: true});
  result(res.ok, res.ok ? 'Paper P&L counts from now' : 'Not reset: ' + res.err);
};
// ---- the page's own boxes: the figures on top, every model's record, the losing runs ----
const MODEL_NAMES = {ob_steady: 'Steady order book: follows the order books only when their reading has lasted. The reading is the size at the best bid against the best ask of five markets of our price mixed by their share (Bybit, OKX and Binance futures, OKX spot, Coinbase; not Bitfinex futures, whose best price is thin), averaged over the last second. A bet when 75% or more of that size is on one side AND the reading has been on that side for 5 s or more, on that side. On the recording a young strong reading (under 2 s) was right 49.9% of 6,095 times on bc.game, one of 5 s or more 54.1% of 9,000; on the earlier rounds of the bot 53.0% of 627: not proven, this row is its forward test. About 13% of rounds. In the bot since 2026-11-18 00:20 UTC. It can make the calls: choose it in the Model list', ob_logit: 'Logistic regression on the order books (10 levels a side) and the trades of the six markets of our price, each market alone and mixed by its share (Bitfinex futures 43.3%, Bybit futures 17.2%, OKX futures 16.0%, OKX spot 11.4%, Binance futures 6.8%, Coinbase 5.3%): book imbalance, buy / sell pressure, spread, recent moves, plus the gap; from the fit of 19 Nov 2026 on also how long the best-price reading has been on its side (101 inputs instead of 85). Fitted again as the recording grows (3 / 24 / 72 / 168 h); each new fit starts its record again. Bets in every round, on the side its P(UP) is on. It can make the calls: choose it in the Model list', ob_lgbm: 'LightGBM on the order books (10 levels a side) and the trades of the six markets of our price, each market alone and mixed by its share (Bitfinex futures 43.3%, Bybit futures 17.2%, OKX futures 16.0%, OKX spot 11.4%, Binance futures 6.8%, Coinbase 5.3%): book imbalance, buy / sell pressure, spread, recent moves, plus the gap; from the fit of 19 Nov 2026 on also how long the best-price reading has been on its side (101 inputs instead of 85). Fitted again as the recording grows (3 / 24 / 72 / 168 h); each new fit starts its record again. Bets in every round, on the side its P(UP) is on. It can make the calls: choose it in the Model list', ob_xgb: 'XGBoost on the order books (10 levels a side) and the trades of the six markets of our price, each market alone and mixed by its share (Bitfinex futures 43.3%, Bybit futures 17.2%, OKX futures 16.0%, OKX spot 11.4%, Binance futures 6.8%, Coinbase 5.3%): book imbalance, buy / sell pressure, spread, recent moves, plus the gap; from the fit of 19 Nov 2026 on also how long the best-price reading has been on its side (101 inputs instead of 85). Fitted again as the recording grows (3 / 24 / 72 / 168 h); each new fit starts its record again. Bets in every round, on the side its P(UP) is on. It can make the calls: choose it in the Model list', ob_deeplob: 'DeepLOB: a neural network (convolutions, an inception block and an LSTM) on the raw order books of the six markets, the last 10 s of each, 10 levels a side, with the trades and the gap. Fitted again with the other order-book models as the recording grows; it needs days of books to learn anything. Bets in every round, on the side its P(UP) is on. It can make the calls: choose it in the Model list', sign: 'HawkesSign on the order books of all six markets of our price. Until 2026-11-16 15:36 UTC it read one book, Bitfinex spot, which is not one of the six; its P&L and history were started again with the change, so every number in its row is from the six-market version. One Hawkes model per market, on its price moves and the buy / sell pressure at its best prices over the last 300 s, fitted again every 20 s. At the decision each model is run forward 200 times over the round, and the six moves are mixed by the share of each market in our price (Bitfinex futures 43.3%, Bybit futures 17.2%, OKX futures 16.0%, OKX spot 11.4%, Binance futures 6.8%, Coinbase 5.3%): P(UP) is the share of runs in which the mixed move is up. Bets in every round. Its record is not started again by a new fit of the other order-book models. It can make the calls: choose it in the Model list',value: 'Value bet: the chance of each side (from the gap: the bet against a gap of 2 / 3 / 4 x volatility wins about 53 / 56 / 59%) times what the page pays on it; a bet when that is 1.05 or more. With no gap it bets a side that pays 2.12+, with a gap of 2 x from 1.99, 3 x from 1.88. Fitted on the rounds up to 2026-11-11 20:05 UTC; on the 35 h after: 817 bets, +9.0% per $1 at a flat $1 (a test on the log: this table counts only its bets since it runs in the bot, 2026-11-16 02:28 UTC). Not safe: about half the bets lose. Judge it by its P&L: it breaks even at 47-50% wins', fade: 'Gap fade (bc.game at least 1 x volatility from the market)', fade_strict: 'Gap fade strict (at least 2.45 x volatility: fewer bets, higher win rate)', fade_hawkes: 'Gap fade + Hawkes: the Gap fade bet, placed only when the Hawkes order-book forecast says the same side. On the log its agreement did not change the win rate (53.8% against 53.1%)', fade_flow2: 'Gap fade + flow, 2x: the Gap fade bet when the gap is at least 2 x volatility AND 2 of the 3 futures had order flow on the same side. Fewer bets (about 230 a day), higher win rate on the log (57.6%), long losing runs about once a day', fade_flow: 'Gap fade + flow: the Gap fade bet, placed only when at least 2 of the 3 futures (Binance, OKX, Bybit) had order flow on the same side over the last 3 s', fade_fair: 'Gap fade against the new price (on paper until the new price is confirmed)', thin: 'Thin side: the side with the smaller pool when the page pays 2.10 or more on it. No prediction. Measured at a flat $1: your own stake goes into the thin pool, so a doubling series keeps about a third of the return per $1. Judge it by its P&L: it breaks even near 45.5% wins', overshoot_rel: 'Overshoot, scaled by volatility', trees_mid: 'Trees, medium', revert: 'Quiet reversal', overshoot: 'Overshoot', overshoot_book: 'Overshoot + book', prob: 'P(UP) model (logistic, fitted on the log, calls at |P-0.5| >= 0.06)', ens: 'P(UP) ensemble (the average of the logistic model and a LightGBM model on the same inputs)', pool: 'Pool: thin side when it pays > x2 (scored only)'};
const SHORT = {value: 'Value bet', ob_logit: 'OB logistic', ob_lgbm: 'OB LightGBM', ob_xgb: 'OB XGBoost', ob_deeplob: 'OB DeepLOB', sign: 'HawkesSign', ob_steady: 'Steady book', fade: 'Gap fade', fade_strict: 'Fade strict', fade_flow: 'Fade + flow', fade_flow2: 'Fade+flow 2x', fade_hawkes: 'Fade + Hawkes', fade_fair: 'Fade new price', thin: 'Thin side', overshoot: 'Overshoot', overshoot_rel: 'Oversh. scaled', overshoot_book: 'Oversh.+book', prob: 'P(UP)', ens: 'P(UP) ens.',
  trees_mid: 'Trees mid', revert: 'Quiet rev.', pool: 'Pool'};   // the order of the columns
const LABEL = {value: 'Value bet (chance x payout)', ob_logit: 'Logistic regression (order books)', ob_lgbm: 'LightGBM (order books)', ob_xgb: 'XGBoost (order books)', ob_deeplob: 'DeepLOB (order books)', sign: 'HawkesSign (six markets)', ob_steady: 'Steady order book (strong 5 s+)', fade: 'Gap fade', fade_strict: 'Gap fade strict', fade_flow: 'Gap fade + flow', fade_flow2: 'Gap fade + flow, 2x', fade_hawkes: 'Gap fade + Hawkes', fade_fair: 'Gap fade (new price)', thin: 'Thin side (pays 2.10+)', overshoot: 'Overshoot', overshoot_rel: 'Overshoot scaled', overshoot_book: 'Overshoot + book', prob: 'P(UP)', ens: 'P(UP) ensemble',
  trees_mid: 'Trees mid', revert: 'Quiet reversal', pool: 'Pool'};
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const usd = v => (v >= 0 ? '+' : '&minus;') + '$' + Math.abs(v || 0).toFixed(2);
const tone = v => v > 0 ? 'var(--up)' : v < 0 ? 'var(--down)' : 'var(--text)';
const longestOf = r => r ? Math.max(0, r.current, ...Object.keys(r.lengths).map(Number)) : null;
function kpis(d) {
  const s = d.stats || {}, g = d.gate || {}, clk = d.clock || {};
  const armed = d.state === 'ARMED', stopped = d.state === 'STOPPED';
  const lad = (d.ladders || {})[d.model] || {};            // the calling model's paper bets since the model / stake rule was set
  const stakeName = d.progression === 1 ? 'flat' : 'x' + d.progression + ' ladder';
  const since = d.since ? new Date(d.since * 1000).toTimeString().slice(0, 5) : 'start';
  const lastAD = (d.events || []).filter(e => e.event === 'arm' || e.event === 'disarm').pop();   // a restart that did not resume the arming
  const ended = !armed && !stopped && lastAD && lastAD.event === 'disarm' && lastAD.who === 'restart' ? lastAD.reason || 'see the log' : '';
  const tile = (label, value, sub, color, title) => `<div class="kpi" title="${esc(title || '')}"><div class="l">${label}</div>` +
    `<div class="v" style="color:${color || 'var(--text)'}">${value}</div><div class="s">${sub}</div></div>`;
  document.getElementById('kpis').innerHTML =
    tile('Paper P&amp;L', usd(lad.pnl || 0), `${lad.won || 0} of ${lad.bets || 0} won &middot; ${stakeName} &middot; since ${since}`, tone(lad.pnl || 0),
         `The calling model's paper bets at the stakes shown under Last results, with the payout lowered by our own stake in the pool (the same number as the panel). Before that: ${(lad.pnl_shown || 0).toFixed(2)}. ` +
         `The same bets at a flat $1: ${(lad.flat || 0).toFixed(2)}. Starts again when the model or the stake rule changes.`) +
    tile('Real orders', armed ? 'Live' : stopped ? 'Stopped' : 'Off',
         stopped ? (d.tech_stop ? `<span style="white-space:normal;color:var(--warn)">${esc(d.stop_reason || d.tech_stop)}. Press Clear stop.</span>` : esc(d.stop_reason || '')) : ended ? `<span style="white-space:normal;color:var(--warn)">not resumed after the restart: ${esc(ended)}. Switch to LIVE again if you want it.</span>`
           : `next $${d.next_real_stake} &middot; this run ${usd(d.arm_live_pnl)} &middot; ${d.arm_orders || 0} orders`,
         armed ? 'var(--down)' : stopped ? 'var(--warn)' : '', d.stop_text || d.stop_reason || ended) +
    tile('Real loss run', `${d.real_run} <span class="mute">/ ${d.max_real_run}</span>`, 'real losses in a row',
         d.real_run >= d.max_real_run ? 'var(--down)' : d.real_run ? 'var(--warn)' : '', 'A real win or Reset sets it back to 0') +
    tile('Orders today', d.attempts_today + (d.max_real_bets_day ? ` <span class="mute">/ ${d.max_real_bets_day}</span>` : ''),
         `real today ${usd(d.today_live_pnl)} &middot; budget left $${(d.budget_left || 0).toFixed(2)}`, '', 'The day stops when the budget is used up') +
    tile('Campaign', usd(d.campaign_pnl), `balance ${d.balance_usd != null ? '$' + d.balance_usd.toFixed(2) : '&ndash;'}`, tone(d.campaign_pnl)) +
    tile('Go-live gate', g.open ? 'Open' : 'Closed',
         g.n ? `${(100 * g.p).toFixed(1)}% of ${g.n} &middot; needs ${g.min_n}` : `needs ${g.min_n || 3000} calls`, g.open ? 'var(--up)' : '', g.text);
  document.getElementById('clock').textContent = clk.ok ? `clock ${clk.offset >= 0 ? '+' : ''}${clk.offset} s` : 'clock: exchange';
}
function modelsTable(d) {
  const rec = d.records || {}, s = d.stats || {}, runs = d.runs_today || {}, be = 100 * (d.break_even || 0.5155);
  const byPnl = k => k === 'thin' || k === 'value';   // paid well above x1.94: the 51.55% line is not their break-even
  const rate = (ok, n, plain) => {              // plain: thin is paid about x2.2, the 51.55% line is not its break-even
    if (!n) return '<span class="dim">&ndash;</span>';
    const p = 100 * ok / n, c = plain ? 'var(--text)' : n >= 30 && p > be ? 'var(--up)' : n >= 30 && p < 100 - be ? 'var(--down)' : 'var(--text)';
    return `<span style="color:${c}">${p.toFixed(1)}%</span>`;
  };
  const num = n => n ? n : '<span class="dim">&ndash;</span>';
  const rows = Object.keys(LABEL).filter(k => rec[k]).map(k => {
    const r = rec[k], t = runs[k], on = k === d.model, lg = longestOf(t);
    const fn = s[k + '_fwd_called'] || 0, fok = s[k + '_fwd_correct'] || 0;
    return `<tr class="${on ? 'sel' : ''}"><td title="${esc(MODEL_NAMES[k] || '')}">${LABEL[k]}${on ? '<span class="calling">calling</span>' : ''}</td>` +
      `<td>${rate(r.correct, r.called, byPnl(k))}</td><td>${num(r.called)}</td>` +
      `<td style="color:${r.called ? tone(r.pnl) : ''}">${r.called ? usd(r.pnl) : '<span class="dim">&ndash;</span>'}</td>` +
      `<td>${rate(fok, fn, byPnl(k))}</td><td>${num(fn)}</td>` +
      `<td class="${lg >= 5 ? 'red' : ''}">${lg == null ? '<span class="dim">&ndash;</span>' : lg}</td>` +
      `<td class="${t && t.current ? 'amb' : ''}">${t && t.current ? t.current : '<span class="dim">&ndash;</span>'}</td></tr>`;
  }).join('');
  const ob = d.ob || {};
  document.getElementById('obinfo').innerHTML = 'Order-book models (the six rows under Value bet): ' + (
    ob.ok === undefined ? 'no answer yet from the order-book service' :
    !ob.ok ? `<span style="color:var(--down)">the order-book service does not answer</span> (${esc(ob.error || '')}): no bets from them` :
    `${ob.markets_ok} of ${ob.markets} order books live &middot; ${(ob.hours || 0).toFixed(1)} h recorded &middot; ` +
    (ob.version ? `fit ${ob.version} in place, made on ${(ob.trained_h || 0).toFixed(1)} h` + (ob.inputs ? ` with ${ob.inputs} inputs` : '') : 'not fitted yet') +
    ` &middot; next fit at ${ob.next_fit_h} h${ob.training ? ' (fitting now)' : ''}` +
    (ob.level ? (ob.level.share >= 1 ? ' &middot; they bet in every round' :
                 ` &middot; they bet in the ${Math.round(100 * ob.level.share)}% of rounds they are surest of`) : '') +
    ((ob.models || []).includes('ob_deeplob') ? '' : ' &middot; DeepLOB not fitted yet') +
    (ob.hawkes ? ` &middot; HawkesSign on ${(ob.hawkes.markets || []).length} of ${ob.markets} books` +
                 (ob.hawkes.late ? ` (${ob.hawkes.late} answers too late)` : '') : '') +
    (ob.error ? ` &middot; <span style="color:var(--down)">${esc(ob.error)}</span>` : ''));
  document.getElementById('models').innerHTML =
    `<table class="tbl"><tr><th>Model</th><th>Win rate</th><th>Bets</th><th>P&amp;L</th>` +
    `<th title="Since the model was last changed: the record the go-live gate counts">All-time win rate</th><th>All-time bets</th>` +
    `<th>Max losses in a row today</th><th>Losses in a row now</th></tr>${rows}</table>`;
}
async function loadModelLab() {
  const status = document.getElementById('labstate'), box = document.getElementById('labmodels');
  try {
    const response = await fetch('/auto/model-lab', {cache: 'no-store'});
    if (!response.ok) throw new Error('unavailable');
    const d = await response.json();
    if (!d.available) {
      status.textContent = 'Not available';
      box.innerHTML = '<span class="cap">Waiting for Market Memory paper research.</span>';
      document.getElementById('lablatest').textContent = '';
      document.getElementById('labtraining').textContent = '';
      return;
    }
    status.textContent = d.fresh ? 'Running' : 'Not updating';
    status.style.color = d.fresh ? 'var(--up)' : 'var(--warn)';
    const pct = v => Number.isFinite(v) ? (100*v).toFixed(1) + '%' : '&ndash;';
    const records = d.results || {}, latest = (d.last_prediction || {}).models || {};
    const rows = (d.models || []).map((m, i) => {
      const r = records[m.id] || {}, s = r.selected || {}, all = r.all || {}, p = latest[m.id];
      const role = m.role === 'incumbent' ? 'Current' : 'Challenger';
      const side = p && p.selected ? (p.p >= .5 ? 'UP' : 'DOWN') : p ? 'Skip' : '&ndash;';
      const ci = s.wilson95 && s.calls ? pct(s.wilson95[0]) + ' to ' + pct(s.wilson95[1]) : '&ndash;';
      return `<tr><td title="${esc(m.id)}"><b>Market Memory v${i+1}</b><span class="calling">${role}</span></td>` +
        `<td>${s.calls ? pct(s.accuracy) : '&ndash;'}</td><td>${s.wins || 0} / ${s.calls || 0}</td>` +
        `<td>${ci}</td><td style="color:${s.calls ? tone(s.net_flat_1 || 0) : ''}">${s.calls ? usd(s.net_flat_1 || 0) : '&ndash;'}</td>` +
        `<td>${all.calls ? pct(all.accuracy) : '&ndash;'} <span class="cap">(${all.calls || 0})</span></td>` +
        `<td>${side}${p ? ' <span class="cap">P(UP) ' + pct(p.p) + '</span>' : ''}</td></tr>`;
    }).join('');
    box.innerHTML = '<table class="tbl"><tr><th>Model</th><th>Win rate</th><th>Won / bets</th><th>Approx. 95% range</th><th>Paper P&amp;L</th><th>All-round accuracy</th><th>Last decision</th></tr>' + rows + '</table>';
    const next = d.next_fit_not_before_utc ? new Date(d.next_fit_not_before_utc).toLocaleString() : '';
    document.getElementById('labtraining').textContent = d.training_pid ? 'Training a new challenger now.' :
      `Retrains every ${d.refit_hours || 6} hours after ${d.refit_minimum_new_settlements || 300} new settled rounds.` + (next ? ' Next eligible fit: ' + next + '.' : '');
    document.getElementById('lablatest').textContent = d.last_error || d.last_skip ||
      (d.last_prediction ? 'Last prediction: ' + new Date(d.last_prediction.at_utc).toLocaleTimeString() : 'Waiting for the next round.');
  } catch (_) {
    status.textContent = 'Unavailable';
    box.innerHTML = '<span class="cap">Market Memory results could not be loaded.</span>';
  }
}
loadModelLab(); setInterval(loadModelLab, 5000);

function runsBox(d) {                    // losing runs of every model: today by length, then each earlier day's longest
  const runs = d.runs_today || {}, days = d.runs_days || {};
  const names = Object.keys(SHORT).filter(k => runs[k] && runs[k].bets)
    .sort((x, y) => (y === d.model) - (x === d.model));          // the calling model first
  const box = document.getElementById('runs');
  if (!names.length) { box.innerHTML = '<span class="cap">No settled calls yet today.</span>'; return; }
  let maxN = 1;
  for (const k of names) maxN = Math.max(maxN, longestOf(runs[k]));
  const dot = '<span class="dim">&middot;</span>';
  let rows = '';
  for (let n = 1; n <= maxN; n++) {
    rows += `<tr><td class="h">${n}</td>` + names.map(k => {
      const c = runs[k].lengths[n] || 0, now = runs[k].current === n;
      return `<td class="${n >= 5 && (c || now) ? 'red' : ''}">${c || (now ? '' : dot)}${now ? ' <span class="amb">now</span>' : ''}</td>`;
    }).join('') + '</tr>';
  }
  const foot = (label, f, cls) => `<tr class="ft"><td class="h">${label}</td>` +
    names.map(k => { const v = f(k); return `<td class="${cls ? cls(v) : ''}">${v == null ? dot : v}</td>`; }).join('') + '</tr>';
  const hot = v => v >= 5 ? 'red' : '';
  const dayRows = Object.keys(days).sort().reverse().map(day =>
    foot(day.slice(5), k => { const x = (days[day] || {})[k]; return x ? x.longest : null; }, hot)).join('');
  box.innerHTML =
    `<table class="tbl runs"><tr><th>In a row</th>${names.map(k => `<th style="${k === d.model ? 'color:var(--accent)' : ''}">${SHORT[k]}</th>`).join('')}</tr>` +
    rows + foot('Calls', k => runs[k].bets) + foot('Longest', k => longestOf(runs[k]), hot) +
    (dayRows ? `<tr class="sec"><td colspan="${names.length + 1}">Longest run on earlier days</td></tr>` + dayRows : '') +
    `</table>`;
}
// ---- the current model's P&L: its paper bets (its row in the Models table) and the real orders of the same time ----
const PNL_C = {paper: '#3987e5', real: '#d95926'}, PNL_NAME = {paper: 'Paper, flat $1', real: 'Real money'};
const PNL_RANGES = {'1h': 'the last hour', '6h': 'the last 6 hours', today: 'today, since 00:00', all: ''};
let pnlRange = PNL_RANGES[localStorage.pnlRange] !== undefined ? localStorage.pnlRange : 'all';
let pnlData = null, pnlKey = '', pnlSeq = 0, pnlX = null, pnlLive = false;
const hm = (t, sec) => new Date(t * 1000).toTimeString().slice(0, sec ? 8 : 5);
const dayOf = t => new Date(t * 1000).toLocaleDateString('en-GB', {day: 'numeric', month: 'short'});
const money = v => (v > 0 ? '+' : v < 0 ? '−' : '') + '$' + Math.abs(v).toFixed(2);       // plain text (usd() is HTML)
async function loadPnl() {
  const my = ++pnlSeq;
  try {
    const d = await (await fetch('/auto/pnl?r=' + pnlRange + (q ? '&' + q.slice(1) : ''))).json();
    if (my === pnlSeq && d && d.paper) { pnlData = d; drawPnl(); }
  } catch (e) {}
}
function pnlState(d) {                   // a settled round, another model or another range: read the lines again
  pnlLive = d.state === 'ARMED';
  const key = [d.model, pnlRange, (d.stats || {}).rounds, ((d.records || {})[d.model] || {}).called, d.state].join('|');
  if (key !== pnlKey) { pnlKey = key; loadPnl(); }
}
function lastAt(s, t) {                  // the index of a line's last bet at or before t (-1: none yet)
  let lo = 0, hi = s.t.length - 1, k = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (s.t[m] <= t) { k = m; lo = m + 1; } else hi = m - 1; }
  return k;
}
function niceStep(span, n) {             // 1, 2 or 5 times a power of ten
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
function drawPnl() {
  const d = pnlData; if (!d) return;
  const P = d.paper, R = d.real, lines = [{k: 'paper', s: P}].concat(R.n ? [{k: 'real', s: R}] : []);
  document.getElementById('pnlt').textContent = 'P&L · ' + (LABEL[d.model] || d.model);
  document.getElementById('pnlcap').textContent = (d.range === 'all' ? (P.n ? 'since ' + dayOf(d.from) + ' ' + hm(d.from) + ', when its row began' : 'no bets yet')
    : PNL_RANGES[d.range]) + ' · both lines start at 0';
  const key = k => `<i style="background:${PNL_C[k]}"></i>`, pct = (a, n) => n ? ` (${(100 * a / n).toFixed(1)}%)` : '';
  document.getElementById('pnlinfo').innerHTML =
    `<span title="Every bet of this model at a flat $1, as its row in the Models table counts it. It is not money.">${key('paper')}${PNL_NAME.paper} ` +
    `<b style="color:${tone(P.pnl)}">${usd(P.pnl)}</b> &middot; ${P.won} of ${P.n} bets won${pct(P.won, P.n)}</span>` +
    `<span title="Your real orders in the same time, as the bot booked them (whichever model placed them)">${key('real')}${PNL_NAME.real} ` +
    (R.n ? `<b style="color:${tone(R.pnl)}">${usd(R.pnl)}</b> &middot; ${R.won} of ${R.n} real orders won${pct(R.won, R.n)}`
         : `<b>$0.00</b> &middot; no real order in this time`) + (pnlLive ? ' &middot; <b style="color:var(--down)">LIVE now</b>' : '') + `</span>`;
  const cv = document.getElementById('pnl'), dpr = window.devicePixelRatio || 1, tip = document.getElementById('pnltip');
  const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 58, RP = 70, T = 14, B = 26;
  let x0 = d.from, x1 = d.now, lo = 0, hi = 0;
  for (const {s} of lines) { if (s.t.length) x1 = Math.max(x1, s.t[s.t.length - 1]); for (const v of s.v) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (x1 - x0 < 60) x0 = x1 - 60;
  g.font = '11px Consolas, monospace';
  if (!P.n && !R.n) {
    g.fillStyle = '#7d8896'; g.textAlign = 'center'; g.fillText('No bet by this model in this time', W / 2, H / 2); g.textAlign = 'left';
    tip.hidden = true; return;
  }
  const step = niceStep(Math.max(hi - lo, 2), 4);
  let y0 = Math.floor(lo / step) * step, y1 = Math.ceil(hi / step) * step;
  if (y1 === y0) { y0 -= step; y1 += step; }
  const X = t => L + (t - x0) / (x1 - x0) * (W - L - RP), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  for (let i = 0, n = Math.round((y1 - y0) / step); i <= n; i++) {             // $ lines; the 0 line a step brighter
    const v = y0 + i * step, y = Math.round(Y(v)) + 0.5, zero = Math.abs(v) < step / 1000;
    g.strokeStyle = zero ? '#39424f' : '#1a212b'; g.lineWidth = 1; g.beginPath(); g.moveTo(L, y); g.lineTo(W - RP, y); g.stroke();
    g.fillStyle = '#7d8896'; g.textAlign = 'right';
    g.fillText(zero ? '$0' : (v < 0 ? '−$' : '+$') + Math.abs(v).toFixed(step < 1 ? 1 : 0), L - 8, y + 4);
  }
  const span = x1 - x0, tz = new Date().getTimezoneOffset() * 60;               // time marks on this PC's clock
  const tstep = [60, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800].find(s => span / s <= 7) || 604800;
  g.textAlign = 'center'; g.fillStyle = '#7d8896';
  for (let lt = Math.ceil((x0 - tz) / tstep) * tstep; lt <= x1 - tz; lt += tstep) {
    const t = lt + tz, x = X(t), midnight = lt % 86400 === 0;
    if (x < L + 16 || x > W - RP - 4) continue;
    g.fillText(tstep >= 86400 ? dayOf(t) : midnight ? dayOf(t) : hm(t), x, H - 8);
  }
  g.textAlign = 'left';
  const ends = [];
  for (const {k, s} of lines) {                       // a line holds its value until the next bet
    g.strokeStyle = PNL_C[k]; g.lineWidth = 2; g.lineJoin = 'round'; g.lineCap = 'round'; g.beginPath();
    let last = 0; g.moveTo(X(x0), Y(0));
    for (let i = 0; i < s.t.length; i++) { const x = Math.max(L, X(s.t[i])); g.lineTo(x, Y(last)); last = s.v[i]; g.lineTo(x, Y(last)); }
    g.lineTo(X(x1), Y(last)); g.stroke();
    ends.push({k, v: last, y: Y(last)});
  }
  const dot = (x, y, c) => { g.fillStyle = '#0c1016'; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill(); g.fillStyle = c; g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill(); };
  const apart = ends.length < 2 || Math.abs(ends[0].y - ends[1].y) >= 15;     // two end values too close: the line above has them
  for (const e of ends) {
    dot(X(x1), e.y, PNL_C[e.k]);
    if (apart) { g.fillStyle = '#e6ebf2'; g.font = 'bold 11px Consolas, monospace'; g.fillText(money(e.v), X(x1) + 10, e.y + 4); g.font = '11px Consolas, monospace'; }
  }
  if (pnlX == null || pnlX < L || pnlX > W - RP) { tip.hidden = true; return; }
  let t = x0 + (pnlX - L) / (W - L - RP) * (x1 - x0), best = null;             // the pointer: the nearest bet's time
  for (const {s} of lines) for (const i of [lastAt(s, t), lastAt(s, t) + 1])
    if (i >= 0 && i < s.t.length && s.t[i] >= x0 && (best == null || Math.abs(s.t[i] - t) < Math.abs(best - t))) best = s.t[i];
  if (best != null && Math.abs(X(best) - pnlX) <= 24) t = best;
  const x = Math.round(X(t)) + 0.5;
  g.strokeStyle = '#39424f'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, T); g.lineTo(x, H - B); g.stroke();
  tip.textContent = '';
  const head = document.createElement('div'); head.className = 'h';
  head.textContent = (span > 86400 || d.range === 'all' ? dayOf(t) + ' ' : '') + hm(t, true); tip.appendChild(head);
  for (const {k, s} of lines) {
    const i = lastAt(s, t), v = i < 0 ? 0 : s.v[i];
    dot(x, Y(v), PNL_C[k]);
    const row = document.createElement('div'); row.className = 'r';
    const sw = document.createElement('i'); sw.style.background = PNL_C[k];
    const val = document.createElement('b'); val.textContent = money(v);
    const nm = document.createElement('span'); nm.textContent = PNL_NAME[k];
    row.append(sw, val, nm); tip.appendChild(row);
  }
  tip.hidden = false;
  tip.style.left = (x > W / 2 ? x - tip.offsetWidth - 14 : x + 14) + 'px';
}
(() => {
  const cv = document.getElementById('pnl'), seg = [...document.querySelectorAll('#pnlrange button')];
  const mark = () => seg.forEach(b => b.classList.toggle('on', b.dataset.r === pnlRange));
  seg.forEach(b => b.onclick = () => { pnlRange = b.dataset.r; localStorage.pnlRange = pnlRange; mark(); pnlKey = ''; loadPnl(); });
  mark();
  cv.addEventListener('pointermove', e => { pnlX = e.clientX - cv.getBoundingClientRect().left; drawPnl(); });
  cv.addEventListener('pointerleave', () => { pnlX = null; drawPnl(); });
  window.addEventListener('resize', drawPnl);
  setInterval(loadPnl, 60000);
  const btn = document.getElementById('runsbtn'), wrap = document.getElementById('runswrap');   // the losing runs: on a click
  const show = on => { wrap.hidden = !on; btn.classList.toggle('on', on); btn.textContent = on ? 'Hide losing runs' : 'Losing runs'; localStorage.showRuns = on ? '1' : '0'; };
  btn.onclick = () => show(wrap.hidden);
  show(localStorage.showRuns === '1');
})();
function state(d) {
  if (!d || d.error) return;
  const startup = document.getElementById('startupnotice');
  startup.hidden = !d.startup_problem;
  startup.textContent = d.startup_problem || '';
  kpis(d); modelsTable(d); runsBox(d); pnlState(d);
  const sel = document.getElementById('modelsel');
  if (document.activeElement !== sel) sel.value = d.model;
  document.querySelectorAll('[data-prog]').forEach(b => b.classList.toggle('on', Number(b.dataset.prog) === d.progression));
  const armed = d.state === 'ARMED', ab = document.querySelector('[data-arm]');
  ab.disabled = !!d.startup_problem;
  ab.classList.toggle('on', armed);
  ab.textContent = armed ? 'LIVE · $' + (d.next_real_stake || d.next_stake || 1) + (d.ladder_on ? ' ladder' : '') : 'LIVE';
  document.querySelector('[data-live]').classList.toggle('on', !armed);
  document.querySelector('[data-clear]').style.display = d.tech_stop ? '' : 'none';
  document.getElementById('realst').innerHTML = armed ? '<span class="badge live">LIVE</span>' :
    d.state === 'STOPPED' ? '<span class="badge stop">STOPPED</span>' : '<span class="badge">PAPER</span>';
  for (const [id, key] of Object.entries(LIMIT_BOX)) {
    const box = document.getElementById(id), v = d[key];
    if (document.activeElement === box || v == null) continue;
    if (![...box.options].some(o => Number(o.value) === v)) box.add(new Option((id === 'maxrun' ? '' : '$') + v, v));
    box.value = v;
  }
}
document.getElementById('modelsel').onchange = e => post({model: e.target.value});
document.querySelectorAll('.menu button').forEach(b => b.addEventListener('click', () => b.closest('details').open = false));
document.querySelectorAll('[data-prog]').forEach(b => b.onclick = async () => {
  const res = await post({progression: Number(b.dataset.prog)});         // allowed while LIVE: say what the next real stake is
  if (res.ok && res.d.state === 'ARMED') result(true, `Stake rule changed while LIVE: the next real order is $${res.d.next_real_stake}` +
    (res.d.ladder_on ? ` (x${res.d.progression} after each loss; ${res.d.real_run} in a row now)` : ' (flat)'));
});
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
const EX_COLORS = {bitfinex_perp: '#c084fc', bybit_perp: '#f472b6', okx_perp: '#fb923c', okx: '#a3e635',
                   binance_perp: '#60a5fa', coinbase: '#94a3b8'};
const winSel = document.getElementById('win'), showEx = document.getElementById('showex');
winSel.value = localStorage.chartWin || '60'; showEx.checked = localStorage.chartEx === '1';
winSel.onchange = () => { localStorage.chartWin = winSel.value; drawChart(); };
showEx.onchange = () => { localStorage.chartEx = showEx.checked ? '1' : '0'; drawChart(); };
async function drawChart() {
  const win = Number(winSel.value) || 60, ex = showEx.checked;
  let d; try { d = await (await fetch(`/auto/prices?s=${win}&ex=${ex ? 1 : 0}` + (q ? '&' + q.slice(1) : ''))).json(); } catch (e) { return; }
  const cv = document.getElementById('chart'), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 70, R = 12, T = 12, B = 22, x1 = d.now, x0 = x1 - win;
  const bc = d.bc || [], our = ourPrice(d), exs = (ex && d.exchanges) || {};
  const fair = d.fair || [], fi = d.fair_info;              // the separate new price (drawn only when it has a recipe)
  const lines = Object.entries(exs).map(([k, pts]) => ({pts, color: EX_COLORS[k] || '#8793a2', w: 0.9, dash: [3, 2]}))
    .concat(fair.length ? [{pts: fair, color: '#34d399', w: 1.6}] : [],
            [{pts: our.line, color: '#22d3ee', w: 1.6}, {pts: bc, color: '#facc15', w: 2.2}]);
  // how different: bc.game minus our price at each bc.game tick, and 1 s moves in the same direction
  const diffs = bc.map(p => { const o = our.at(p[0]); return o == null ? null : [p[0], p[1] - o]; }).filter(x => x);
  let same = 0, moves = 0;
  for (let t = x1 - win + 1; t <= x1; t += 1) {
    const b0 = priceAt(bc, t - 1), b1 = priceAt(bc, t), o0 = our.at(t - 1), o1 = our.at(t);
    if ([b0, b1, o0, o1].includes(null) || b1 === b0 || o1 === o0) continue;
    moves++; same += Math.sign(b1 - b0) === Math.sign(o1 - o0);
  }
  const lastBc = bc.length ? bc[bc.length - 1][1] : null, lastOur = our.line.length ? our.line[our.line.length - 1][1] : null;
  const absd = diffs.map(x => Math.abs(x[1]));
  const f = (v, n = 2) => v == null ? '–' : v.toFixed(n);
  document.getElementById('legend').innerHTML = [
    `<span><i style="background:#facc15"></i>bc.game <b>${f(lastBc)}</b></span>`,
    `<span><i style="background:#22d3ee"></i>our price <b>${f(lastOur)}</b></span>`,
    fair.length ? `<span title="The separate new price, on paper"><i style="background:#34d399"></i>new price <b>${f(fair[fair.length - 1][1])}</b></span>` : '',
  ].concat(Object.keys(exs).map(k => { const m = (d.mix || {})[k] || {}, p = exs[k];
    return `<span><i style="background:${EX_COLORS[k] || '#8793a2'}"></i>${m.label || k} <b>${p.length ? f(p[p.length - 1][1]) : '–'}</b></span>`; })).join('');
  document.getElementById('gapinfo').innerHTML =
    `now <b style="color:var(--text)">${diffs.length ? (diffs[diffs.length - 1][1] >= 0 ? '+' : '') + f(diffs[diffs.length - 1][1]) : '–'}</b>` +
    ` &middot; average ${f(mean(absd))} &middot; max ${absd.length ? f(Math.max(...absd)) : '–'}` +
    ` &middot; same direction ${moves ? Math.round(100 * same / moves) + '%' : '–'}` +
    (fair.length && bc.length && priceAt(fair, bc[bc.length - 1][0]) != null
      ? ` &middot; new price gap now <b style="color:#34d399">${(g => (g >= 0 ? '+' : '') + f(g))(lastBc - priceAt(fair, bc[bc.length - 1][0]))}</b>` : '');
  document.getElementById('mixinfo').innerHTML = Object.entries(d.mix || {}).map(([k, m]) =>
    `<span class="${m.live ? '' : 'off'}">${m.label || k} ${Math.round(100 * m.weight)}%</span>`).join('');
  drawDiff(diffs, x0, x1);
  const o = d.our_vs_bc || {};
  const wr = (n, ok) => n ? `${Math.round(100 * (n - ok) / n)}% <span class="mute">of ${n}</span>` : '<span class="dim">–</span>';
  const row = (p, title) => `<tr><td>${title}</td><td>${wr(o[p + '_n'], o[p + '_ok'])}</td><td>${wr(o[p + '_small_n'], o[p + '_small_ok'])}</td>` +
    `<td>${wr(o[p + '_mid_n'], o[p + '_mid_ok'])}</td><td>${wr(o[p + '_big_n'], o[p + '_big_ok'])}</td><td>${o[p + '_flat'] || 0}</td></tr>`;
  document.getElementById('ourbox').innerHTML =
    `<table class="tbl"><tr><th>Wrong Up/Down</th><th>All rounds</th><th>Move under $1</th><th>$1 to $5</th><th>Over $5</th><th>Flat</th></tr>` +
    row('mix', 'Current mix (6 feeds)') + row('our', 'Old: Binance + Bitfinex') +
    (fi ? row('fair', 'New price (on paper, recipe ' + fi.version + ')') : '') + `</table>` +
    (fi ? `<div class="cap" style="margin-top:8px">New price: ` + Object.values(fi.mix).map(m => { const bad = !m.live ? ' (feed down)' : m.fresh === false ? ' (no price now)' : ''; return `<span${bad ? ' style="color:var(--down)"' : ''}>${m.label} ${Math.round(100 * m.weight)}%${bad}</span>`; }).join(', ') +
          (fi.same_rounds.n ? `. On the same ${fi.same_rounds.n} rounds it gave bc.game's result ${Math.round(100 * fi.same_rounds.new_ok / fi.same_rounds.n)}% of the time, the current price ${Math.round(100 * fi.same_rounds.current_ok / fi.same_rounds.n)}%.` : '. No settled round yet.') + `</div>`
        : `<div class="cap" style="margin-top:8px">New price: ` + (d.fair_problem
            ? `the recipe file is there but was not accepted when the bot started (${esc(d.fair_problem)}).`
            : `no recipe yet.`) + `</div>`);
  const ys = lines.flatMap(l => l.pts.filter(p => p[0] >= x0).map(p => p[1]));
  const r = d.round || {}; if (r.start_price) ys.push(r.start_price);
  if (ys.length < 2) return;
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0 || 1) * 0.12; y0 -= pad; y1 += pad;
  const X = t => L + (t - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  g.font = '11px Consolas, monospace'; g.fillStyle = '#8793a2'; g.strokeStyle = '#1f2630'; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.fillText(v.toFixed(1), 4, y + 4); }
  const tick = win >= 300 ? 60 : win >= 120 ? 20 : win >= 60 ? 10 : 5;
  for (let s = 0; s <= win; s += tick) { const x = X(x1 - s); g.fillText(s ? `-${s}s` : 'now', x - 12, H - 6); }
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
  for (const l of lines) {                           // the exchanges (if shown), then our price, bc.game on top
    const pts = l.pts.filter(p => p[0] >= x0 - 2); if (pts.length < 2) continue;
    g.strokeStyle = l.color; g.lineWidth = l.w; g.setLineDash(l.dash || []); g.beginPath();
    pts.forEach((p, i) => { const x = Math.max(L, X(p[0])), y = Y(p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.lineTo(X(x1), Y(pts[pts.length - 1][1])); g.stroke(); g.setLineDash([]);
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
