"""Our estimate of bc.game's BTC price from live exchange prices.

A weighted mix of exchanges' best bid/ask mids, each read a little earlier than now because bc.game's price follows
them with a short delay.  Weights and delays: fitted on a side-by-side recording of bc.game and the exchanges
(see MIX below).  bc.game's price changes on every tick (every ~0.55 s) even when the exchanges stand still, so no
exchange-based price can match it tick for tick: over 5 s, moves under $1 are a coin flip, moves over $5 match ~98%.

Clocks: bc.game's ticks carry bc.game's server time, the feeds are stamped with this PC's clock, and this PC's clock
can be off by a second or more.  Feeds whose messages carry the exchange's own time measure that difference live
(MidFeed.offset), so the delays below are real delays behind the exchanges, whatever this PC's clock says.

Level: each exchange is moved to bc.game's level by its median gap to bc.game over the last GAP_WINDOW seconds.
A level gap can't change a move, so Up/Down results (move()) don't use the gaps at all.
"""
import asyncio
import bisect
import json
import logging
import math
import statistics
import time
from collections import deque
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

# exchange -> (weight, delay in seconds: bc.game's price at time t follows this exchange's price at t - delay).
# Fitted 2026-10-29 on a 20-minute recording (bc.game + 5 spot + 4 perpetual feeds): weights by least squares on
# bc.game's 5 s moves over the first 12 minutes, scored on the last 8: Up/Down same as bc.game 80.1% (the previous
# 5-exchange spot mix: 75.2%), average |our price - bc.game| $2.07 (was $3.61).  bc.game follows every exchange by
# ~0.15 s; the perpetual futures prices track it best.  bc.game moves ~10% more than the mix.
MIX = {"bitfinex_perp": (0.433, 0.15), "bybit_perp": (0.172, 0.15), "okx_perp": (0.160, 0.15),
       "okx": (0.114, 0.15), "binance_perp": (0.068, 0.15), "coinbase": (0.053, 0.15)}
MIX_VERSION = 3             # logged with each round's mix move, so the wrong-rate counts only this mix
                            # (3, 2026-11-01: Bitfinex read DEPTH_BTC deep instead of its top of book)
DEPTH_BTC = 0.5             # Bitfinex: mid of the average prices to buy / sell this much (depth_mid)
GAP_WINDOW = 60.0           # s of bc.game ticks used for each exchange's level gap
STALE = 120.0               # these feeds only send when the price changes, so silence means "unchanged";
                            # for the CHART an exchange is left out only after this long without any update
STALE_SNAPSHOT = 5.0        # for a feature snapshot or a decision, a price older than this is unknown (None):
                            # a feed that is silent for 5 s is down, and a dead price must never look fresh
BOOK_STALE = 3.0            # order flow (OFI) is unknown when the feed's book hasn't updated for this long
MIN_WEIGHT = 0.75           # a snapshot's mix is unknown when feeds carrying more than 25% of the weight are stale
# The separate "new price" (2026-11-11): a second mix with its own weights, its own delay per exchange and, if it
# needs them, more exchanges.  Our price above was fitted to MATCH bc.game's move in the same 5 s (and reads every
# exchange 0.15 s late to look like it); the Gap fade needs a price that says where bc.game is GOING, so this one is
# fitted so that bc.game minus it predicts bc.game's move in the round's window.  Its recipe is a file, so that a
# refit needs no code change; while the file is absent there is no new price and the model on it stays silent.
# Nothing else reads it: our price, its inputs and every model on them are unchanged until the owner switches.
FAIR_FILE = Path(__file__).with_name("fair_price.json")


class MidFeed:
    """Best bid/ask mid of one exchange: rows of (this PC's time, mid), one per 50 ms (the latest price in it).
    If the messages carry the exchange's time, also this PC's clock minus the exchange's clock (offset())."""

    def __init__(self, name: str, url: str, subscribe: list, parse):
        self.name, self.url, self.subscribe, self.parse = name, url, subscribe, parse
        self.mids: deque = deque(maxlen=12000)
        self.offsets: deque = deque(maxlen=240)        # this PC's time - exchange time (incl. network delay)
        self.ofi_events: deque = deque(maxlen=20000)   # (this PC's time, order-flow imbalance of one book update)
        self.connected = False
        self.last_msg_at = 0.0                         # this PC's time of the last price message
        self.last_book_at = 0.0                        # ... of the last best bid/ask with sizes (for the OFI)
        self._state: dict = {}
        self._off_at = 0.0
        self._top = None

    def age(self) -> float:
        return time.time() - self.last_msg_at if self.last_msg_at else 999.0

    def book_age(self) -> float:
        return time.time() - self.last_book_at if self.last_book_at else 999.0

    def put(self, mid, ts: float | None = None) -> None:
        if not mid:
            return
        now = time.time()
        self.last_msg_at = now
        if ts and now - self._off_at >= 0.5:
            self.offsets.append(now - ts)
            self._off_at = now
        if self.mids and now // 0.05 == self.mids[-1][0] // 0.05:
            self.mids[-1] = (now, float(mid))      # one row per 50 ms: the latest price in it, not the first
        else:
            self.mids.append((now, float(mid)))

    def book(self, bid: float, bid_qty: float, ask: float, ask_qty: float, ts: float | None = None) -> None:
        """A new best bid / ask with sizes: the mid, and the order-flow imbalance of the change (Cont, Kukanov &
        Stoikov 2014): bids added or raised and asks removed count as buying pressure, the reverse as selling."""
        self.last_book_at = time.time()
        if self._top is not None:
            pb, pbq, pa, paq = self._top
            e = ((bid_qty if bid >= pb else 0.0) - (pbq if bid <= pb else 0.0)
                 - (ask_qty if ask <= pa else 0.0) + (paq if ask >= pa else 0.0))
            if e:
                self.ofi_events.append((self.last_book_at, e))
        self._top = (bid, bid_qty, ask, ask_qty)
        self.put((bid + ask) / 2, ts)

    def ofi(self, seconds: float, max_age: float = BOOK_STALE) -> float | None:
        """Order-flow imbalance summed over the last `seconds` (in this exchange's size units); None when the
        book hasn't updated for `max_age` s (a silent feed is unknown flow, not balanced flow)."""
        if self.book_age() > max_age:
            return None
        cut, total = time.time() - seconds, 0.0
        for t, e in reversed(self.ofi_events):
            if t < cut:
                break
            total += e
        return total

    def offset(self) -> float | None:
        return statistics.median(self.offsets) if len(self.offsets) >= 5 else None

    async def run(self) -> None:
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self.url, heartbeat=20, max_msg_size=0) as ws:
                        for m in self.subscribe:
                            await ws.send_json(m)
                        self.connected = True
                        log.info("%s BTC best bid/ask connected", self.name)
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            try:
                                self.parse(self, json.loads(msg.data))
                            except (KeyError, IndexError, TypeError, ValueError):
                                pass
            except Exception as e:
                log.warning("%s feed: %r", self.name, e)
            self.connected = False
            await asyncio.sleep(3)


def _binance(f, d):                     # bookTicker (b/B bid price/size, a/A ask); futures messages carry the time E
    f.book(float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]), d["E"] / 1000 if "E" in d else None)


def _okx(f, d):
    for row in d.get("data") or []:
        b, a = row["bids"][0], row["asks"][0]
        f.book(float(b[0]), float(b[1]), float(a[0]), float(a[1]), int(row["ts"]) / 1000)


def _bybit(f, d):
    data = d.get("data") or {}
    if data.get("b"):
        f._state["b"] = (float(data["b"][0][0]), float(data["b"][0][1]))
    if data.get("a"):
        f._state["a"] = (float(data["a"][0][0]), float(data["a"][0][1]))
    if "b" in f._state and "a" in f._state:
        (b, bq), (a, aq) = f._state["b"], f._state["a"]
        f.book(b, bq, a, aq, d["ts"] / 1000 if "ts" in d else None)


def _coinbase(f, d):
    if d.get("channel") == "ticker":
        for ev in d.get("events") or []:
            for t in ev.get("tickers") or []:
                if t.get("best_bid") and t.get("best_ask"):
                    f.put((float(t["best_bid"]) + float(t["best_ask"])) / 2)


def depth_mid(book: dict, size: float = None, levels: int = 25):
    """The mid of the average prices to buy and to sell `size` BTC through the best `levels` price levels.
    None if a side is thinner than that.  Unlike the top-of-book mid it does not jump when one small order at the
    top comes and goes."""
    size = DEPTH_BTC if size is None else size
    out = []
    for side, best_first in (("b", True), ("a", False)):
        left, cost = size, 0.0
        for p in sorted(book[side], reverse=best_first)[:levels]:
            take = min(left, book[side][p])
            cost += take * p
            left -= take
            if left <= 1e-12:
                break
        if left > 1e-12:
            return None
        out.append(cost / size)
    return (out[0] + out[1]) / 2


def _bitfinex(f, d):
    if isinstance(d, dict) or len(d) < 2 or d[1] == "hb":
        return
    snapshot = bool(d[1]) and isinstance(d[1][0], list)
    if snapshot:                                  # a (re)subscription: start the book again
        f._state["book"] = {"b": {}, "a": {}}
    book = f._state.setdefault("book", {"b": {}, "a": {}})
    rows = d[1] if snapshot else [d[1]]
    for p, c, a in rows:
        if c == 0:
            (book["b"] if a == 1 else book["a"]).pop(p, None)
        else:
            (book["b"] if a > 0 else book["a"])[p] = abs(a)
    if book["b"] and book["a"]:
        mid = depth_mid(book)
        f.put(mid if mid is not None else (max(book["b"]) + min(book["a"])) / 2)


def _kraken(f, d):
    if d.get("channel") == "ticker":
        for row in d.get("data") or []:
            f.book(float(row["bid"]), float(row["bid_qty"]), float(row["ask"]), float(row["ask_qty"]))


def _bitstamp(f, d):
    if d.get("event") == "data":
        x = d["data"]
        b, a = x["bids"][0], x["asks"][0]
        f.book(float(b[0]), float(b[1]), float(a[0]), float(a[1]), int(x["microtimestamp"]) / 1e6)


def _deribit(f, d):
    x = (d.get("params") or {}).get("data")
    if x:
        f.book(float(x["best_bid_price"]), float(x["best_bid_amount"]), float(x["best_ask_price"]),
               float(x["best_ask_amount"]), x["timestamp"] / 1000)


def _hyper(f, d):
    if d.get("channel") == "bbo":
        x = d["data"]
        b, a = x["bbo"]
        if b and a:
            f.book(float(b["px"]), float(b["sz"]), float(a["px"]), float(a["sz"]), x["time"] / 1000)


def _okx_sub(inst):
    return [{"op": "subscribe", "args": [{"channel": "bbo-tbt", "instId": inst}]}]


def _bitfinex_sub(sym):
    return [{"event": "subscribe", "channel": "book", "symbol": sym, "prec": "P0", "freq": "F0", "len": "25"}]


FEEDS = {   # name -> (label, websocket, subscribe messages, parser); "_perp" = USDT perpetual futures
    "binance": ("Binance", "wss://stream.binance.com:9443/ws/btcusdt@bookTicker", [], _binance),
    "okx": ("OKX", "wss://ws.okx.com:8443/ws/v5/public", _okx_sub("BTC-USDT"), _okx),
    "bybit": ("Bybit", "wss://stream.bybit.com/v5/public/spot",
              [{"op": "subscribe", "args": ["orderbook.1.BTCUSDT"]}], _bybit),
    "coinbase": ("Coinbase", "wss://advanced-trade-ws.coinbase.com",
                 [{"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"}], _coinbase),
    "bitfinex": ("Bitfinex", "wss://api-pub.bitfinex.com/ws/2", _bitfinex_sub("tBTCUSD"), _bitfinex),
    "binance_perp": ("Binance perp", "wss://fstream.binance.com/ws/btcusdt@bookTicker", [], _binance),
    "okx_perp": ("OKX perp", "wss://ws.okx.com:8443/ws/v5/public", _okx_sub("BTC-USDT-SWAP"), _okx),
    "bybit_perp": ("Bybit perp", "wss://stream.bybit.com/v5/public/linear",
                   [{"op": "subscribe", "args": ["orderbook.1.BTCUSDT"]}], _bybit),
    "bitfinex_perp": ("Bitfinex perp", "wss://api-pub.bitfinex.com/ws/2", _bitfinex_sub("tBTCF0:USTF0"), _bitfinex),
    # only the new price may use these four (they are connected only when its recipe names them)
    "kraken": ("Kraken", "wss://ws.kraken.com/v2",
               [{"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD"], "event_trigger": "bbo"}}], _kraken),
    "bitstamp": ("Bitstamp", "wss://ws.bitstamp.net",
                 [{"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}}], _bitstamp),
    "deribit_perp": ("Deribit perp", "wss://www.deribit.com/ws/api/v2",
                     [{"jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
                       "params": {"channels": ["quote.BTC-PERPETUAL"]}}], _deribit),
    "hyper_perp": ("Hyperliquid perp", "wss://api.hyperliquid.xyz/ws",
                   [{"method": "subscribe", "subscription": {"type": "bbo", "coin": "BTC"}}], _hyper),
}


def make_feeds(names=None) -> dict:
    """The exchange feeds the mix needs (or the named ones)."""
    return {k: MidFeed(*FEEDS[k]) for k in (names or MIX)}


def _check_fair_recipe(raw) -> dict:
    """The recipe as the bot uses it, or a ValueError that says what is wrong with the file."""
    def num(v, what):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what} is not a number")
        return float(v)

    if not isinstance(raw, dict) or not isinstance(raw.get("mix"), dict) or not raw["mix"]:
        raise ValueError("no exchanges ('mix')")
    mix = {}
    for k, v in raw["mix"].items():
        if k not in FEEDS:
            raise ValueError(f"unknown exchange {str(k)[:30]!r}")
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"{k} needs [weight, delay]")
        w, lag = num(v[0], f"the weight of {k}"), num(v[1], f"the delay of {k}")
        if not 0 < w <= 1e6:
            raise ValueError(f"the weight of {k} must be above 0")
        if not 0 <= lag <= 2:
            raise ValueError(f"the delay of {k} must be 0 to 2 s")
        mix[k] = (w, lag)
    window = num(raw.get("level_window", GAP_WINDOW), "level_window")
    if not 10 <= window <= 600:
        raise ValueError("level_window must be 10 to 600 s")
    version = raw.get("version")
    if type(version) is not int or version < 1:
        raise ValueError("version must be a whole number from 1")
    registered = num(raw.get("registered"), "registered")
    if not 1.6e9 < registered < 4e9:                # the bot formats it as a time: milliseconds would break that
        raise ValueError("registered must be a time in seconds")
    total = sum(w for w, _ in mix.values())
    fitted = raw.get("fitted_utc")                  # a note for people: only short text is passed on (the bot's
    return {"version": version, "registered": registered, "level_window": window,       # answers must stay JSON)
            "fitted_utc": fitted[:40] if isinstance(fitted, str) else None,
            "mix": {k: (round(w / total, 6), lag) for k, (w, lag) in mix.items()}}


def read_fair_recipe(path=None) -> tuple:
    """The new price's recipe from bcbot/fair_price.json and, when the file is there but not usable, why:
    (recipe, None), (None, None) when there is no file, (None, reason) when it was refused.  Never raises: a bad
    file must not keep the bot from starting.  The file:
    {"version": 1, "registered": <epoch s its paper record counts from>, "level_window": 60,
     "mix": {"<exchange in FEEDS>": [weight, delay in s], ...}}.  The weights are scaled to sum to 1."""
    p = Path(path or FAIR_FILE)
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))     # -sig: Notepad and PowerShell 5.1 write a mark first
    except FileNotFoundError:
        return None, None
    except Exception as e:                                      # noqa: BLE001
        log.warning("new-price recipe %s not readable: %r", p.name, e)
        return None, f"the file can't be read ({type(e).__name__})"
    try:
        return _check_fair_recipe(raw), None
    except Exception as e:                                      # noqa: BLE001
        log.warning("new-price recipe %s ignored: %r", p.name, e)
        return None, str(e)[:120] if isinstance(e, ValueError) and str(e) else type(e).__name__


def load_fair_recipe(path=None) -> dict | None:
    """The new price's recipe, or None when there is none or it is not usable (read_fair_recipe says which)."""
    return read_fair_recipe(path)[0]


def _at(times: list, values: list, t: float, max_age: float = STALE):
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > max_age:
        return None
    return values[i]


class OurPrice:
    """The weighted, delayed mix.  `feeds` maps exchange name -> MidFeed; times in and out are bc.game server time.
    `mix` ({exchange: (weight, delay)}) and `gap_window` default to our price's (MIX, GAP_WINDOW); the separate new
    price is the same class with its own recipe and, as `clock`, our price's clock(): an exchange that sends no time
    of its own is then read at the same instant by both (and by the recording the recipe is fitted on), whichever
    exchanges the recipe names."""

    def __init__(self, feeds: dict, bc_ticks: deque, mix: dict | None = None, gap_window: float | None = None,
                 clock=None):
        self.feeds, self.bc_ticks = feeds, bc_ticks
        self._clock = clock
        self.mix = dict(mix) if mix else MIX
        self.gap_window = gap_window or GAP_WINDOW
        self._gaps: dict = {}
        self._gaps_at = 0.0

    def clock(self) -> float:
        """This PC's clock minus the exchanges' clocks (median over the feeds that measure it; 0 until known)."""
        offs = [o for f in self.feeds.values() if (o := f.offset()) is not None]
        own = statistics.median(offs) if offs else 0.0
        if self._clock is not None:                 # our price's clock; while that one is not known yet (0.0), the
            return self._clock() or own             # clock this price's own exchanges measure
        return own

    def server_now(self) -> float:
        return time.time() - self.clock()

    def _snapshot(self):
        """Per exchange: (local times, mids, the local time shift for a server time)."""
        clock = self.clock()
        snap = {}
        for k, f in self.feeds.items():
            rows = list(f.mids)
            off = f.offset()
            snap[k] = ([r[0] for r in rows], [r[1] for r in rows], clock if off is None else off)
        return snap

    @staticmethod
    def _price(snap, k, lag, t, max_age: float = STALE):
        """Exchange k's price at server time t - lag, as it was known here (None if older than max_age)."""
        if k not in snap:
            return None
        ts, vs, off = snap[k]
        return _at(ts, vs, t - lag + off, max_age)

    def gaps(self, snap=None) -> dict:
        """Each exchange's median gap to bc.game over the last GAP_WINDOW seconds (recomputed every 5 s)."""
        now = time.time()
        if now - self._gaps_at < 5 and self._gaps:
            return self._gaps
        snap = snap or self._snapshot()
        ticks = list(self.bc_ticks)
        t_end = ticks[-1][0] if ticks else 0
        bc = [r for r in ticks if r[0] >= t_end - self.gap_window]
        gaps = {}
        for k, (w, lag) in self.mix.items():
            diffs = sorted(v - p for t, p in bc if (v := self._price(snap, k, lag, t)) is not None)
            if diffs:
                gaps[k] = diffs[len(diffs) // 2]
        self._gaps, self._gaps_at = gaps, now
        return gaps

    def value(self, t: float, snap=None, max_age: float = STALE, min_weight: float = 0.0):
        """Our price at server time t, at bc.game's level.  For a decision or a feature snapshot pass
        max_age=STALE_SNAPSHOT and min_weight=MIN_WEIGHT: then it is None when too much of the mix is stale."""
        snap = snap or self._snapshot()
        gaps = self.gaps(snap)
        num = den = 0.0
        for k, (w, lag) in self.mix.items():
            if k not in gaps:
                continue
            v = self._price(snap, k, lag, t, max_age)
            if v is not None:
                num += w * (v - gaps[k]); den += w
        return num / den if den and den >= min_weight else None

    def stale(self, t: float, snap=None, max_age: float = STALE_SNAPSHOT) -> list:
        """The feeds with no price within max_age of server time t (their names), for a snapshot's dq block."""
        snap = snap or self._snapshot()
        return [k for k, (w, lag) in self.mix.items() if self._price(snap, k, lag, t, max_age) is None]

    def now_value(self):
        """Our price now, at bc.game's level; cheap (reads only the newest rows of each feed), for the panel."""
        t, clock, gaps = self.server_now(), self.clock(), self.gaps()
        num = den = 0.0
        for k, (w, lag) in self.mix.items():
            f = self.feeds.get(k)
            if f is None or k not in gaps:
                continue
            off = f.offset()
            target = t - lag + (clock if off is None else off)
            for ts, mid in reversed(f.mids):
                if ts <= target:
                    if target - ts <= STALE:
                        num += w * (mid - gaps[k]); den += w
                    break
        return num / den if den else None

    def move(self, t0: float, t1: float):
        """Our price's change from t0 to t1 (no level gaps needed): the Up/Down it gives for a round."""
        snap = self._snapshot()
        num = den = 0.0
        for k, (w, lag) in self.mix.items():
            a, b = self._price(snap, k, lag, t0), self._price(snap, k, lag, t1)
            if a is not None and b is not None:
                num += w * (b - a); den += w
        return round(num / den, 3) if den else None

    def series(self, t0: float, t1: float, step: float = 0.2) -> list:
        snap = self._snapshot()
        if not self.gaps(snap):                 # no level for any exchange: no value at any time (an empty table is
            return []                           # not kept, so every point below would work it out again)
        out, t = [], t0
        while t <= t1:
            v = self.value(t, snap)
            if v is not None:
                out.append([round(t, 2), round(v, 3)])
            t += step
        return out

    def exchange_series(self, t0: float, t1: float, step: float = 0.2) -> dict:
        """Each exchange's own curve as the mix sees it (delayed by its lag, moved to bc.game's level by its gap),
        for the chart: exchange name -> [[server time, price]].  Shows which exchange moves our price."""
        snap = self._snapshot()
        gaps = self.gaps(snap)
        out = {}
        for k, (w, lag) in self.mix.items():
            if k not in gaps:
                continue
            pts, t = [], t0
            while t <= t1:
                v = self._price(snap, k, lag, t)
                if v is not None:
                    pts.append([round(t, 2), round(v - gaps[k], 3)])
                t += step
            out[k] = pts
        return out
