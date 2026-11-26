"""The order-book service for the Up/Down bot's order-book models.  Its own process, so that reading six order books
never slows the bot down:
  1. reads the order books (10 levels a side) and the trades of the six markets of our price, over their public
     websockets, and writes one row per market every 100 ms into <folder>/<market>_<YYYYMMDD_HH>.bin (bcbot/obfeat.py)
  2. answers the bot at each decision:  GET http://127.0.0.1:8766/ob?gap=<bc.game - our price, $>&vol=<vol60, $>
     with every fitted model's P(UP) on the inputs of that instant
  3. fits the models again when enough has been recorded (tools/train_ob.py at 3 / 24 / 72 / 168 h, then weekly)
  4. gives each model's betting level with the answer (Level; F.BET_SHARE is 1: a bet in every round)
  6. answers "ob_steady" (bcbot/obfeat.py, steady_call): the side of the five markets' best-price reading when it is
     strong and 5 s or older, with the reading, its age and the price's move since in the answer's "life"
  5. runs HawkesSign on the six order books (bcbot/hawkes6.py): one Hawkes model per market, fitted again every 20 s
     on its last 300 s in worker processes, and at each question run forward over the round's window and mixed by
     the markets' shares in our price (the answer's ob_hawkes; a and b in the question say when the window is)
  7. GET /life: the five markets' best-price reading and its age right now, for the dashboard's order-book card
It never sends anything to bc.game or to the bot.  Run it with the research Python (it needs xgboost):
    logs\\research-venv\\Scripts\\python.exe tools\\ob_server.py [folder] [port]      (default logs\\obrec, 8766)
A file named STOP in the folder ends it."""
import asyncio
import glob
import heapq
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bcbot import hawkes6 as H  # noqa: E402
from bcbot import obfeat as F  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "obrec"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
RING = 400                                   # rows kept in memory per market (40 s)
# hours recorded at which the models are fitted again (then every 168 h).  Each fit starts the four records on the
# dashboard again, so there are few of them: the 12 h fit was taken out on 2026-11-16 (the owner saw four restarts
# in four hours and about 10 bets a row)
MILESTONES = (3.0, 24.0, 72.0, 168.0)
RETRY_S = 3600.0                             # after a fit that failed
# the six-book HawkesSign (bcbot/hawkes6.py)
HK_RING = 3200                               # rows kept per market for it (320 s: it fits on the last 300 s)
HK_REFIT_S = 20.0                            # every market's model is fitted again this often (a fit is about 1 s)
HK_MAX_AGE_S = 120.0                         # a model older than this is not used (its market went silent)
HK_WAIT_S = 0.5                              # an answer waits this long for the simulations (the bot waits 0.8 s)
HK_A, HK_B = 2.6, 7.6                        # the round's window from the question, when the bot does not say


class Market:
    def __init__(self, name, unit=1.0):
        self.name, self.unit = name, unit          # unit: BTC per size unit of the exchange
        self.bids, self.asks = {}, {}               # price -> size (exchange units)
        self.ready = False
        self.book_at = 0.0
        self.buy = self.sell = 0.0
        self.nb = self.ns = 0
        self.ring = np.zeros(RING, dtype=F.ROW)
        self.n = 0                                  # rows written so far
        self.hk = np.full((HK_RING, len(H.COLS)), np.nan)   # the last 320 s of what the Hawkes model reads (H.COLS)
        self.hk_n = 0

    def reset(self):
        self.bids, self.asks, self.ready = {}, {}, False

    def set(self, side, price, size):
        book = self.bids if side == "b" else self.asks
        if size > 0:
            book[price] = size
        else:
            book.pop(price, None)

    def touched(self):
        self.book_at = time.time()

    def trade(self, is_buy, size):
        if is_buy:
            self.buy += size * self.unit; self.nb += 1
        else:
            self.sell += size * self.unit; self.ns += 1

    def row(self, now):
        r = np.zeros((), dtype=F.ROW)
        r["t"] = now
        for k in ("bp", "bs", "ap", "as_"):
            r[k] = np.nan
        r["bid"] = r["ask"] = np.nan
        if self.ready and self.bids and self.asks:
            b = heapq.nlargest(F.LEVELS, self.bids)
            a = heapq.nsmallest(F.LEVELS, self.asks)
            r["bid"], r["ask"] = b[0], a[0]
            r["bp"][:len(b)] = [b[0] - p for p in b]
            r["bs"][:len(b)] = [self.bids[p] * self.unit for p in b]
            r["ap"][:len(a)] = [p - a[0] for p in a]
            r["as_"][:len(a)] = [self.asks[p] * self.unit for p in a]
        r["buy"], r["sell"], r["nb"], r["ns"] = self.buy, self.sell, self.nb, self.ns
        r["age"] = min(now - self.book_at, 9999.0) if self.book_at else 9999.0
        self.buy = self.sell = 0.0
        self.nb = self.ns = 0
        self.ring[self.n % RING] = r
        self.n += 1
        self.hk[self.hk_n % HK_RING] = (now, r["bid"], r["ask"], r["bs"][0], r["as_"][0], r["age"])
        self.hk_n += 1
        return r

    def hk_recent(self, n=HK_RING):
        """The Hawkes model's columns of the last n rows, oldest first (a copy: it goes to a worker process)."""
        i = self.hk_n % HK_RING
        a = self.hk[:self.hk_n] if self.hk_n <= HK_RING else np.concatenate([self.hk[i:], self.hk[:i]])
        return a[-n:].copy()

    def recent(self):
        """The rows in memory, oldest first."""
        if self.n <= RING:
            return self.ring[:self.n]
        i = self.n % RING
        return np.concatenate([self.ring[i:], self.ring[:i]])

    def known(self, now):
        return bool(self.ready and self.bids and self.asks and now - self.book_at <= F.MAX_AGE)


M = {"bitfinex_perp": Market("bitfinex_perp"), "bybit_perp": Market("bybit_perp"), "okx_perp": Market("okx_perp", 0.01),
     "okx": Market("okx"), "binance_perp": Market("binance_perp"), "coinbase": Market("coinbase")}


# ---------------------------------------------------------------- parsers (one per connection)
def binance(state, d):
    x = d.get("data") or d
    m = M["binance_perp"]
    if x.get("e") == "depthUpdate":                    # partial depth: the whole top 20 every 100 ms
        m.bids = {float(p): float(q) for p, q in x["b"]}
        m.asks = {float(p): float(q) for p, q in x["a"]}
        m.ready = True
        m.touched()
    elif x.get("e") == "aggTrade":
        m.trade(not x["m"], float(x["q"]))             # m: the buyer is the maker = a taker SOLD


def bybit(state, d):
    topic = d.get("topic") or ""
    m = M["bybit_perp"]
    if topic.startswith("orderbook"):
        x = d["data"]
        if d.get("type") == "snapshot":
            m.reset()
        for p, q in x.get("b") or []:
            m.set("b", float(p), float(q))
        for p, q in x.get("a") or []:
            m.set("a", float(p), float(q))
        m.ready = True
        m.touched()
    elif topic.startswith("publicTrade"):
        for t in d["data"]:
            m.trade(t["S"] == "Buy", float(t["v"]))


def okx(state, d):
    arg = d.get("arg") or {}
    m = M.get({"BTC-USDT-SWAP": "okx_perp", "BTC-USDT": "okx"}.get(arg.get("instId")))
    if m is None or "data" not in d:
        return
    if arg.get("channel") == "books":
        if d.get("action") == "snapshot":
            m.reset()
        for x in d["data"]:
            for row in x.get("bids") or []:
                m.set("b", float(row[0]), float(row[1]))
            for row in x.get("asks") or []:
                m.set("a", float(row[0]), float(row[1]))
        m.ready = True
        m.touched()
    elif arg.get("channel") == "trades":
        for t in d["data"]:
            m.trade(t["side"] == "buy", float(t["sz"]))


def bitfinex(state, d):
    m = M["bitfinex_perp"]
    if isinstance(d, dict):
        if d.get("event") == "subscribed":
            state[d["chanId"]] = d["channel"]
            if d["channel"] == "book":
                m.reset()
        return
    kind = state.get(d[0])
    if kind == "book":
        x = d[1]
        if x == "hb":
            m.touched()
            return
        rows = x if x and isinstance(x[0], list) else [x]
        for price, count, amount in rows:
            if count == 0:
                (m.bids if amount > 0 else m.asks).pop(float(price), None)
            elif amount > 0:
                m.bids[float(price)] = float(amount)
            else:
                m.asks[float(price)] = -float(amount)
        m.ready = True
        m.touched()
    elif kind == "trades" and len(d) >= 3 and d[1] == "te":       # te = executed (tu repeats it)
        m.trade(d[2][2] > 0, abs(float(d[2][2])))


def coinbase(state, d):
    m = M["coinbase"]
    ch = d.get("channel")
    if ch == "l2_data":
        for ev in d.get("events") or []:
            if ev.get("type") == "snapshot":
                m.reset()
            for u in ev.get("updates") or []:
                m.set("b" if u["side"] == "bid" else "a", float(u["price_level"]), float(u["new_quantity"]))
            m.ready = True
        m.touched()
    elif ch == "market_trades":
        for ev in d.get("events") or []:
            if ev.get("type") != "update":                          # the snapshot is old trades
                continue
            for t in ev.get("trades") or []:
                # Coinbase names the MAKER's side (checked against the book: 74% of "SELL" trades print at the ask):
                # a trade marked SELL lifted a resting sell order = a taker BOUGHT
                m.trade(t["side"] == "SELL", float(t["size"]))


CONNS = {
    # Binance futures serve the book on /public and trades on /market (the old address sends no aggTrade any more)
    "binance_perp": ("wss://fstream.binance.com/public/stream?streams=btcusdt@depth20@100ms", [], binance, ["binance_perp"]),
    "binance_perp_trades": ("wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade", [], binance, []),
    "bybit_perp": ("wss://stream.bybit.com/v5/public/linear",
                   [{"op": "subscribe", "args": ["orderbook.50.BTCUSDT", "publicTrade.BTCUSDT"]}], bybit, ["bybit_perp"]),
    "okx": ("wss://ws.okx.com:8443/ws/v5/public",
            [{"op": "subscribe", "args": [{"channel": "books", "instId": "BTC-USDT-SWAP"}, {"channel": "trades", "instId": "BTC-USDT-SWAP"},
                                          {"channel": "books", "instId": "BTC-USDT"}, {"channel": "trades", "instId": "BTC-USDT"}]}],
            okx, ["okx_perp", "okx"]),
    "bitfinex_perp": ("wss://api-pub.bitfinex.com/ws/2",
                      [{"event": "subscribe", "channel": "book", "symbol": "tBTCF0:USTF0", "prec": "P0", "freq": "F0", "len": "25"},
                       {"event": "subscribe", "channel": "trades", "symbol": "tBTCF0:USTF0"}], bitfinex, ["bitfinex_perp"]),
    "coinbase": ("wss://advanced-trade-ws.coinbase.com",
                 [{"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "level2"},
                  {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "market_trades"}], coinbase, ["coinbase"]),
}
CONN = {k: {"connects": 0, "errors": 0, "last_error": ""} for k in CONNS}


async def feed(name):
    url, subs, parse, markets = CONNS[name]
    wait = 1.0
    while True:
        state = {}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
                    for m in subs:
                        await ws.send_json(m)
                    CONN[name]["connects"] += 1
                    wait = 1.0
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            break
                        try:
                            parse(state, json.loads(msg.data))
                        except (KeyError, IndexError, TypeError, ValueError) as e:
                            CONN[name]["errors"] += 1
                            CONN[name]["last_error"] = repr(e)[:120]
        except Exception as e:                                      # noqa: BLE001 - reconnect whatever it was
            CONN[name]["last_error"] = repr(e)[:120]
        for k in markets:                                           # the book is unknown until the next snapshot
            M[k].reset()
        await asyncio.sleep(wait)
        wait = min(wait * 2, 30.0)


# ---------------------------------------------------------------- the fitted models
class Models:
    """The model set in <folder>/models/current.json (written by tools/train_ob.py), loaded again when it changes."""

    def __init__(self, folder: Path):
        self.folder = folder / "models"
        self.meta, self.mtime, self.error = {}, 0.0, None
        self.fn = {}                                  # model name -> X -> P(UP)
        self.dl = None                                # DeepLOB reads the books themselves (tools/deeplob.py)

    def load(self) -> None:
        path = self.folder / "current.json"
        try:
            mt = path.stat().st_mtime
        except OSError:
            return
        if mt == self.mtime:
            return
        self.mtime = mt
        try:
            meta = json.loads(path.read_text())
            if tuple(meta["names"]) not in (F.NAMES, F.NAMES2):      # the 85 of the fits before 2026-11-17, or with
                raise ValueError("fitted on another list of inputs than this service builds: fit again")   # the lifetime inputs
            n_in = len(meta["names"])                                # each model reads the inputs it was fitted on
            fn, dl = {}, None
            for name, rel in meta["files"].items():
                p = str(self.folder / rel)
                if name == "ob_logit":
                    m = json.loads(Path(p).read_text())
                    fn[name] = (lambda X, m=m: F.logit_predict(m, X))
                elif name == "ob_lgbm":
                    import lightgbm as lgb
                    b = lgb.Booster(model_file=p)
                    fn[name] = (lambda X, b=b, n=n_in: F.tree_predict(b.predict, X[:, :n]))
                elif name == "ob_xgb":
                    import xgboost as xgb
                    b = xgb.Booster(model_file=p)
                    fn[name] = (lambda X, b=b, xgb=xgb, n=n_in: F.tree_predict(lambda Z: b.predict(xgb.DMatrix(Z)), X[:, :n]))
                elif name == "ob_deeplob":
                    import deeplob
                    dl = deeplob.Predictor(p)
            self.meta, self.fn, self.dl, self.error = meta, fn, dl, None
            print(time.strftime("%H:%M:%S"), f"models version {meta['version']} loaded ({', '.join(self.names())}; fitted on {meta.get('hours')} h)", flush=True)
        except Exception as e:                                       # noqa: BLE001 - reported on the dashboard
            self.error = repr(e)[:160]
            print(time.strftime("%H:%M:%S"), "models not loaded:", self.error, flush=True)

    def names(self) -> list:
        return sorted(self.fn) + (["ob_deeplob"] if self.dl else [])

    def predict(self, X, recent: dict, now: float, gap: float, vol: float) -> dict:
        out = {}
        if self.dl is not None:
            try:
                p = self.dl(recent, now, gap, vol, X[:1, MIX5_AT])
                if p is not None and p == p:
                    out["ob_deeplob"] = round(p, 5)
            except Exception as e:                                   # noqa: BLE001
                self.error = f"ob_deeplob: {e!r}"[:160]
        for name, fn in self.fn.items():
            try:
                p = float(fn(X)[0])
                if p == p:
                    out[name] = round(p, 5)
            except Exception as e:                                   # noqa: BLE001
                self.error = f"{name}: {e!r}"[:160]
        return out


class Level:
    """Each model's betting level: how far from 50% its P(UP) must be for a bet (F.live_level).  With F.BET_SHARE 1
    (the owner's choice) that is every round.  With a share under 1 it follows the model's own answers to the bot, so
    that the model bets in that share of the rounds; they are kept in models/level.json, so a restart of this
    service does not start it again.  A new fit starts it again from the level that fit gave."""

    def __init__(self, folder: Path):
        self.path = folder / "models" / "level.json"
        self.v, self.past = 0, {}

    def roll(self, v: int) -> None:
        if v == self.v:
            return
        self.v, self.past = v, {}
        try:
            d = json.loads(self.path.read_text())
            if d.get("version") == v:
                self.past = {k: [float(x) for x in a][-F.LEVEL_N:] for k, a in (d.get("past") or {}).items()}
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def thr(self, fit_thr: dict) -> dict:
        return {k: round(F.live_level(self.past.get(k, ()), t), 5) for k, t in fit_thr.items()
                if isinstance(t, (int, float))}

    def add(self, p: dict) -> None:
        for k, x in p.items():
            a = self.past.setdefault(k, [])
            a.append(round(abs(x - 0.5), 5))
            del a[:-F.LEVEL_N]
        try:
            tmp = self.path.with_name("level.json.tmp")
            tmp.write_text(json.dumps({"version": self.v, "past": self.past}))
            os.replace(tmp, self.path)
        except OSError:
            pass


# ---------------------------------------------------------------- HawkesSign on the six books
HK = {"models": {}, "fit_pool": None, "sim_pool": None, "cycle_s": None, "late": 0, "errors": 0, "last_error": "",
      "last": None}


def hk_preload(now: float) -> None:
    """The Hawkes models need 300 s of rows.  After a restart they come from the recording, not from waiting."""
    for k, m in M.items():
        try:
            files = sorted(glob.glob(str(OUT / f"{k}_20*.bin")))[-2:]
            if not files:
                continue
            a = np.concatenate([np.fromfile(f, dtype=F.ROW) for f in files])
            a = a[a["t"] >= now - HK_RING * F.STEP][-HK_RING:]
            m.hk[:len(a)] = np.column_stack([a["t"], a["bid"], a["ask"], a["bs"][:, 0], a["as_"][:, 0], a["age"]])
            m.hk_n = len(a)
        except (OSError, ValueError) as e:
            print(time.strftime("%H:%M:%S"), f"Hawkes: no rows from the recording for {k}: {e!r}", flush=True)


def hk_pools() -> None:
    for key, n in (("fit_pool", 3), ("sim_pool", 1)):
        if HK[key] is not None:
            HK[key].shutdown(wait=False, cancel_futures=True)
        HK[key] = ProcessPoolExecutor(n)
    HK["sim_pool"].submit(H.predict, {}, {}, 0.0, 1.0, 2.0)          # starts the worker now: its first answer took
                                                                     # over 0.5 s (a new process) and came too late


async def hawkes_loop() -> None:
    """Fits each market's Hawkes model again and again on its last 300 s, in worker processes: a fit is a second of
    pure Python and must not hold up the recording."""
    loop = asyncio.get_running_loop()
    while True:
        t0 = time.time()
        try:
            futs = {k: loop.run_in_executor(HK["fit_pool"], H.fit_market, m.hk_recent(), t0) for k, m in M.items()}
            for k, f in futs.items():
                r = await f
                if r is not None:
                    HK["models"][k] = r
            HK["cycle_s"] = round(time.time() - t0, 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:                                       # noqa: BLE001 - a dead worker: new pools, go on
            HK["errors"] += 1
            HK["last_error"] = f"fit: {e!r}"[:160]
            hk_pools()
        now = time.time()
        for k in [k for k, r in HK["models"].items() if now - r["at"] > HK_MAX_AGE_S]:
            del HK["models"][k]
        await asyncio.sleep(max(1.0, HK_REFIT_S - (time.time() - t0)))


async def hawkes_answer(now: float, a: float, b: float) -> dict | None:
    """The six-book HawkesSign for the window that starts a seconds and ends b seconds from now (H.predict, in the
    simulation worker).  None: no models yet, or the answer was not back in HK_WAIT_S."""
    models = dict(HK["models"])
    if not models or HK["sim_pool"] is None:
        return None
    try:
        cols = {k: M[k].hk_recent(int(H.STATE_S / F.STEP) + 20) for k in models}
        fut = asyncio.get_running_loop().run_in_executor(HK["sim_pool"], H.predict, models, cols, now, a, b)
        r = await asyncio.wait_for(fut, HK_WAIT_S)
    except asyncio.TimeoutError:
        HK["late"] += 1
        return None
    except Exception as e:                                           # noqa: BLE001
        HK["errors"] += 1
        HK["last_error"] = f"answer: {e!r}"[:160]
        return None
    r["age_s"] = round(now - min(m["at"] for m in models.values()), 1)      # the oldest fit used
    HK["last"] = {"at": round(now, 1), "p": r["p"], "ms": r["ms"], "wait_ms": round((time.time() - now) * 1000)}
    return r


def hk_status(now: float) -> dict:
    ms = HK["models"]
    return {"markets": [k for k in M if k in ms], "share": round(sum(F.SHARES[k] for k in ms), 3),
            "age_s": round(now - min(m["at"] for m in ms.values()), 1) if ms else None,
            "events": {k: m["events"] for k, m in ms.items()}, "cycle_s": HK["cycle_s"], "late": HK["late"],
            "errors": HK["errors"], "last_error": HK["last_error"], "last": HK["last"]}


MIX5_AT = [F.NAMES2.index(n) for n in F.MIX5]       # DeepLOB's four lifetime inputs in the input row
MODELS = Models(OUT)
LEVEL = Level(OUT)
STATE = {"started": time.time(), "hours": 0.0, "training": False, "last_fit_try": 0.0, "last_fit": None, "rows": 0}


def hours_recorded() -> float:
    size = sum(os.path.getsize(f) for f in glob.glob(str(OUT / "binance_perp_20*.bin")))
    return size / F.ROW.itemsize * F.STEP / 3600


def next_fit(trained_h: float) -> float:
    for m in MILESTONES:
        if m > trained_h + 1e-6:
            return m
    return (trained_h // 168 + 1) * 168


def fitted_h() -> float:
    """Hours of recording the fit in place was started at: what the next fit is counted from.  Its usable hours are a
    little less (the ends of the recording), which made the 3 h fit run a second time at 4 h on 2026-11-16."""
    return max(float(MODELS.meta.get("hours") or 0.0), float(MODELS.meta.get("recorded_h") or 0.0))


def status(now: float) -> dict:
    trained = float(MODELS.meta.get("hours") or 0.0)
    LEVEL.roll(MODELS.meta.get("version", 0))
    return {"hours": round(STATE["hours"], 2), "trained_h": round(trained, 2), "next_fit_h": next_fit(fitted_h()),
            "training": STATE["training"], "version": MODELS.meta.get("version", 0),
            "registered": MODELS.meta.get("registered"),
            "models": MODELS.names() + (["ob_hawkes"] if HK["models"] else []) + ["ob_steady"], "hawkes": hk_status(now),
            "inputs": len(MODELS.meta.get("names") or ()),
            "level": {"share": F.BET_SHARE, "thr": LEVEL.thr(MODELS.meta.get("thr") or {}),
                      "answers": {k: len(a) for k, a in LEVEL.past.items()}},
            "markets_ok": sum(m.known(now) for m in M.values()), "markets": len(M),
            "error": MODELS.error, "last_fit": STATE["last_fit"], "up_s": round(now - STATE["started"])}


async def train() -> None:
    STATE["training"], STATE["last_fit_try"] = True, time.time()
    try:
        with open(OUT / "train.log", "ab") as log:
            log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} fit at {STATE['hours']:.1f} h recorded\n".encode())
            log.flush()
            proc = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "tools" / "train_ob.py"), str(OUT),
                                                        stdout=log, stderr=log, cwd=str(ROOT),
                                                        env={**os.environ, "OB_RECORDED_H": f"{STATE['hours']:.3f}"})
            code = await proc.wait()
        STATE["last_fit"] = {"at": time.time(), "ok": code == 0, "code": code}
        print(time.strftime("%H:%M:%S"), "fit ended with code", code, flush=True)
    except Exception as e:                                           # noqa: BLE001
        STATE["last_fit"] = {"at": time.time(), "ok": False, "error": repr(e)[:120]}
    finally:
        STATE["training"] = False
        MODELS.load()


async def handle_ob(request: web.Request) -> web.Response:
    now = time.time()

    def num(key):
        try:
            return float(request.query[key])
        except (KeyError, ValueError):
            return float("nan")
    gap, vol, a, b = num("gap"), num("vol"), num("a"), num("b")
    if not 0 <= a < b <= 60:                        # the bot says when the round's window is (seconds from now)
        a, b = HK_A, HK_B
    hk_task = asyncio.ensure_future(hawkes_answer(now, a, b))        # runs in its worker while the others answer
    st = status(now)                                # also moves LEVEL to the version in place
    out = {"t": round(now, 3), "v": MODELS.meta.get("version", 0), "p": {}, "thr": st["level"]["thr"],
           "n_ok": sum(m.known(now) for m in M.values()), "status": st}
    recent = {k: m.recent() for k, m in M.items()}
    if MODELS.fn and vol == vol and vol > 0:
        X = F.build2(recent, [now], [gap], [vol])
        out["p"] = MODELS.predict(X, recent, now, gap, vol)
        if request.query.get("full"):
            out["inputs"] = {n: (None if x != x else round(float(x), 4)) for n, x in zip(F.NAMES2, X[0])}
        elif not request.query.get("probe"):        # the bot's question (one a round): it counts for the level, but
            LEVEL.add(out["p"])                     # only for the NEXT answers: this one's level was read above
    try:                                            # "ob_steady": the five markets' best-price reading and its age
        r, age, move, share = (float(x) for x in F.life_mixed(recent, [now])[0])
        if r == r and age == age:
            call = F.steady_call(r, age)            # 50% = no bet: the reading is weak or young
            out["p"]["ob_steady"] = round(0.5 + F.STEADY_P * (1 if call == "UP" else -1 if call == "DOWN" else 0), 5)
            out["life"] = {"r": round(r, 4), "age": round(age, 1), "move": None if move != move else round(move, 2),
                           "share": round(share, 3)}
    except Exception as e:                          # noqa: BLE001 - the other answers still go out
        MODELS.error = f"ob_steady: {e!r}"[:160]
    out["thr"]["ob_steady"] = F.MIN_THR
    hk = await hk_task
    out["thr"]["ob_hawkes"] = F.MIN_THR             # HawkesSign bets in every round (50% exactly: no bet)
    if hk:
        if hk["p"] is not None:
            out["p"]["ob_hawkes"] = hk["p"]
        out["hk"] = {"per": hk["per"], "share": hk["share"], "ms": hk["ms"], "age_s": hk["age_s"]}
    return web.json_response(out)


def life_now(now: float) -> dict:
    """The five markets' best-price reading at this instant, as "ob_steady" and the lifetime inputs read it
    (F.life_mixed): its size (-1 all asks .. +1 all bids), its age, the price's move since it took its side, the
    share of the five that is known, and each market's own reading.  For the dashboard's order-book card, which
    asks once a second: nothing is counted, kept or simulated here (GET /ob runs the models and the Hawkes paths)."""
    def num(x, places):
        return None if x != x else round(float(x), places)
    recent = {k: M[k].recent() for k in F.LIFE_MARKETS}
    r, age, move, share = F.life_mixed(recent, [now])[0]
    per = {k: num(F.life_market(a, [now])[0][0], 3) for k, a in recent.items()}
    return {"t": round(now, 3), "r": num(r, 4), "age": num(age, 1), "move": num(move, 2), "share": num(share, 3),
            "per": per}


async def handle_life(request: web.Request) -> web.Response:
    return web.json_response(life_now(time.time()))


async def handle_status(request: web.Request) -> web.Response:
    now = time.time()
    return web.json_response({**status(now), "connections": CONN,
                              "books": {k: {"known": m.known(now), "rows": m.n} for k, m in M.items()}})


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ob_meta.json").write_text(json.dumps({"levels": F.LEVELS, "step_s": F.STEP, "dtype": F.ROW.descr,
                                                  "markets": list(M), "started": time.time()}, default=str))
    app = web.Application()
    app.add_routes([web.get("/ob", handle_ob), web.get("/status", handle_status), web.get("/life", handle_life)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()            # a second copy fails here: the port is taken
    tasks = [asyncio.create_task(feed(k)) for k in CONNS]
    hk_preload(time.time())
    hk_pools()
    tasks.append(asyncio.create_task(hawkes_loop()))
    MODELS.load()
    STATE["hours"] = hours_recorded()
    print(time.strftime("%H:%M:%S"), f"order-book service on 127.0.0.1:{PORT}, folder {OUT}, {STATE['hours']:.2f} h recorded so far", flush=True)
    files, hour = {}, None
    t_slow = t_say = time.time()
    nxt = (time.time() // F.STEP + 1) * F.STEP
    try:
        while not (OUT / "STOP").exists():
            await asyncio.sleep(max(0.0, nxt - time.time()))
            now = time.time()
            nxt = max(nxt + F.STEP, (now // F.STEP + 1) * F.STEP)   # never try to catch up after a stall
            h = time.strftime("%Y%m%d_%H", time.gmtime(now))
            if h != hour:
                for fh in files.values():
                    fh.close()
                files = {k: open(OUT / f"{k}_{h}.bin", "ab") for k in M}
                hour = h
            for k, m in M.items():
                files[k].write(m.row(now).tobytes())
            STATE["rows"] += 1
            if STATE["rows"] % 10 == 0:
                for fh in files.values():
                    fh.flush()
            if now - t_slow >= 30:
                t_slow = now
                STATE["hours"] = hours_recorded()
                MODELS.load()
                due = STATE["hours"] >= next_fit(fitted_h())
                if due and not STATE["training"] and now - STATE["last_fit_try"] >= RETRY_S:
                    asyncio.create_task(train())
            if now - t_say >= 1800:
                t_say = now
                print(time.strftime("%H:%M:%S"), json.dumps(status(now)),
                      {k: v for k, v in CONN.items() if v["errors"] or v["connects"] > 1}, flush=True)
    finally:
        for fh in files.values():
            fh.close()
        for t in tasks:
            t.cancel()
        for key in ("fit_pool", "sim_pool"):
            if HK[key] is not None:
                HK[key].shutdown(wait=False, cancel_futures=True)
        await runner.cleanup()
    print("stopped (STOP file)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
