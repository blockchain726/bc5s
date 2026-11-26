"""Records the order books (10 levels a side) and the trades of three more markets, in the order-book service's row
format (bcbot/obfeat.py ROW, one row per market every 100 ms): Bybit spot, Deribit BTC-PERPETUAL and Bitfinex spot.

Why these three: bc.game's price follows them most (together about half of its 0.5 s moves, measured on 2026-11-18)
and none of them is one of the six markets the order-book service reads.  This is ONLY a recording: no model reads
it yet.  It is its own process with its own folder, so the service, its fits and the files it writes are not touched:

    logs\\research-venv\\Scripts\\python.exe tools\\record_books.py [folder]        (default logs\\obrec\\extra)

Files <folder>/<market>_<YYYYMMDD_HH>.bin (markets: bybit, deribit_perp, bitfinex: the names bcbot/ourprice.py uses);
a file named STOP in the folder ends it.  It never sends anything to bc.game, to the bot or to the service.
Sizes are in BTC (Deribit quotes its perpetual in USD: size / price)."""
import asyncio
import heapq
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcbot import obfeat as F  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "obrec" / "extra"


class Market:
    """One market's book and trades between two rows; row() is the order-book service's (tools/ob_server.py)."""

    def __init__(self, name):
        self.name = name
        self.bids, self.asks = {}, {}               # price -> size in BTC
        self.ready = False
        self.book_at = 0.0
        self.buy = self.sell = 0.0
        self.nb = self.ns = 0
        self.rows = self.known_rows = 0

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
            self.buy += size; self.nb += 1
        else:
            self.sell += size; self.ns += 1

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
            r["bs"][:len(b)] = [self.bids[p] for p in b]
            r["ap"][:len(a)] = [p - a[0] for p in a]
            r["as_"][:len(a)] = [self.asks[p] for p in a]
        r["buy"], r["sell"], r["nb"], r["ns"] = self.buy, self.sell, self.nb, self.ns
        r["age"] = min(now - self.book_at, 9999.0) if self.book_at else 9999.0
        self.buy = self.sell = 0.0
        self.nb = self.ns = 0
        self.rows += 1
        self.known_rows += int(bool(self.ready and self.bids and self.asks and r["age"] <= F.MAX_AGE))
        return r


M = {"bybit": Market("bybit"), "deribit_perp": Market("deribit_perp"), "bitfinex": Market("bitfinex")}


# ---------------------------------------------------------------- parsers (one per connection)
def bybit(state, d):
    topic = d.get("topic") or ""
    m = M["bybit"]
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
            m.trade(t["S"] == "Buy", float(t["v"]))        # S: the taker's side


def deribit(state, d):
    p = d.get("params") or {}
    ch, x = p.get("channel") or "", p.get("data")
    m = M["deribit_perp"]
    if ch.startswith("book."):                             # the whole top 10 every 100 ms, [price, USD]
        m.bids = {float(r[-2]): float(r[-1]) / float(r[-2]) for r in x["bids"] if float(r[-1]) > 0}
        m.asks = {float(r[-2]): float(r[-1]) / float(r[-2]) for r in x["asks"] if float(r[-1]) > 0}
        m.ready = True
        m.touched()
    elif ch.startswith("trades."):
        for t in x:
            m.trade(t["direction"] == "buy", float(t["amount"]) / float(t["price"]))   # direction: the taker's


def bitfinex(state, d):
    m = M["bitfinex"]
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


CONNS = {
    "bybit": ("wss://stream.bybit.com/v5/public/spot",
              [{"op": "subscribe", "args": ["orderbook.50.BTCUSDT", "publicTrade.BTCUSDT"]}], bybit, ["bybit"]),
    "deribit_perp": ("wss://www.deribit.com/ws/api/v2",
                     [{"jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
                       "params": {"channels": ["book.BTC-PERPETUAL.none.10.100ms", "trades.BTC-PERPETUAL.100ms"]}}],
                     deribit, ["deribit_perp"]),
    "bitfinex": ("wss://api-pub.bitfinex.com/ws/2",
                 [{"event": "subscribe", "channel": "book", "symbol": "tBTCUSD", "prec": "P0", "freq": "F0", "len": "25"},
                  {"event": "subscribe", "channel": "trades", "symbol": "tBTCUSD"}], bitfinex, ["bitfinex"]),
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
                        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                            CONN[name]["errors"] += 1
                            CONN[name]["last_error"] = repr(e)[:120]
        except Exception as e:                                      # noqa: BLE001 - reconnect whatever it was
            CONN[name]["last_error"] = repr(e)[:120]
        for k in markets:                                           # the book is unknown until the next snapshot
            M[k].reset()
        await asyncio.sleep(wait)
        wait = min(wait * 2, 30.0)


def status() -> dict:
    return {"rows": max(m.rows for m in M.values()),
            "known": {k: round(m.known_rows / m.rows, 4) if m.rows else None for k, m in M.items()},
            "conn": {k: v for k, v in CONN.items() if v["errors"] or v["connects"] != 1}}


async def main(seconds: float = 0.0):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "books_meta.json").write_text(json.dumps({"levels": F.LEVELS, "step_s": F.STEP, "dtype": F.ROW.descr,
                                                     "markets": list(M), "started": time.time()}, default=str))
    tasks = [asyncio.create_task(feed(k)) for k in CONNS]
    print(time.strftime("%H:%M:%S"), f"recording the books of {', '.join(M)} into {OUT}", flush=True)
    files, hour = {}, None
    t_start = t_say = time.time()
    nxt = (time.time() // F.STEP + 1) * F.STEP
    n = 0
    try:
        while not (OUT / "STOP").exists() and not (seconds and time.time() - t_start >= seconds):
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
            n += 1
            if n % 10 == 0:
                for fh in files.values():
                    fh.flush()
            if now - t_say >= 1800:
                t_say = now
                print(time.strftime("%H:%M:%S"), json.dumps(status()), flush=True)
    finally:
        for fh in files.values():
            fh.close()
        for t in tasks:
            t.cancel()
    print(time.strftime("%H:%M:%S"), "stopped", json.dumps(status()), flush=True)


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0))
