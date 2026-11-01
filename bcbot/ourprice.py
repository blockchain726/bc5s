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
import statistics
import time
from collections import deque

import aiohttp

log = logging.getLogger(__name__)

# exchange -> (weight, delay in seconds: bc.game's price at time t follows this exchange's price at t - delay).
# Fitted 2026-10-29 on a 20-minute recording (bc.game + 5 spot + 4 perpetual feeds): weights by least squares on
# bc.game's 5 s moves over the first 12 minutes, scored on the last 8: Up/Down same as bc.game 80.1% (the previous
# 5-exchange spot mix: 75.2%), average |our price - bc.game| $2.07 (was $3.61).  bc.game follows every exchange by
# ~0.15 s; the perpetual futures prices track it best.  bc.game moves ~10% more than the mix.
MIX = {"bitfinex_perp": (0.433, 0.15), "bybit_perp": (0.172, 0.15), "okx_perp": (0.160, 0.15),
       "okx": (0.114, 0.15), "binance_perp": (0.068, 0.15), "coinbase": (0.053, 0.15)}
MIX_VERSION = 2             # logged with each round's mix move, so the wrong-rate counts only this mix
GAP_WINDOW = 60.0           # s of bc.game ticks used for each exchange's level gap
STALE = 120.0               # these feeds only send when the price changes, so silence means "unchanged";
                            # for the CHART an exchange is left out only after this long without any update
STALE_SNAPSHOT = 5.0        # for a feature snapshot or a decision, a price older than this is unknown (None):
                            # a feed that is silent for 5 s is down, and a dead price must never look fresh
BOOK_STALE = 3.0            # order flow (OFI) is unknown when the feed's book hasn't updated for this long
MIN_WEIGHT = 0.75           # a snapshot's mix is unknown when feeds carrying more than 25% of the weight are stale


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


def _bitfinex(f, d):
    if isinstance(d, dict) or len(d) < 2 or d[1] == "hb":
        return
    book = f._state.setdefault("book", {"b": {}, "a": {}})
    rows = d[1] if d[1] and isinstance(d[1][0], list) else [d[1]]
    for p, c, a in rows:
        if c == 0:
            (book["b"] if a == 1 else book["a"]).pop(p, None)
        else:
            (book["b"] if a > 0 else book["a"])[p] = abs(a)
    if book["b"] and book["a"]:
        f.put((max(book["b"]) + min(book["a"])) / 2)


def _okx_sub(inst):
    return [{"op": "subscribe", "args": [{"channel": "bbo-tbt", "instId": inst}]}]


def _bitfinex_sub(sym):
    return [{"event": "subscribe", "channel": "book", "symbol": sym, "prec": "P0", "freq": "F0", "len": "1"}]


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
}


def make_feeds(names=None) -> dict:
    """The exchange feeds the mix needs (or the named ones)."""
    return {k: MidFeed(*FEEDS[k]) for k in (names or MIX)}


def _at(times: list, values: list, t: float, max_age: float = STALE):
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > max_age:
        return None
    return values[i]


class OurPrice:
    """The weighted, delayed mix.  `feeds` maps exchange name -> MidFeed; times in and out are bc.game server time."""

    def __init__(self, feeds: dict, bc_ticks: deque):
        self.feeds, self.bc_ticks = feeds, bc_ticks
        self._gaps: dict = {}
        self._gaps_at = 0.0

    def clock(self) -> float:
        """This PC's clock minus the exchanges' clocks (median over the feeds that measure it; 0 until known)."""
        offs = [o for f in self.feeds.values() if (o := f.offset()) is not None]
        return statistics.median(offs) if offs else 0.0

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
        bc = [r for r in ticks if r[0] >= t_end - GAP_WINDOW]
        gaps = {}
        for k, (w, lag) in MIX.items():
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
        for k, (w, lag) in MIX.items():
            if k not in gaps:
                continue
            v = self._price(snap, k, lag, t, max_age)
            if v is not None:
                num += w * (v - gaps[k]); den += w
        return num / den if den and den >= min_weight else None

    def stale(self, t: float, snap=None, max_age: float = STALE_SNAPSHOT) -> list:
        """The feeds with no price within max_age of server time t (their names), for a snapshot's dq block."""
        snap = snap or self._snapshot()
        return [k for k, (w, lag) in MIX.items() if self._price(snap, k, lag, t, max_age) is None]

    def now_value(self):
        """Our price now, at bc.game's level; cheap (reads only the newest rows of each feed), for the panel."""
        t, clock, gaps = self.server_now(), self.clock(), self.gaps()
        num = den = 0.0
        for k, (w, lag) in MIX.items():
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
        for k, (w, lag) in MIX.items():
            a, b = self._price(snap, k, lag, t0), self._price(snap, k, lag, t1)
            if a is not None and b is not None:
                num += w * (b - a); den += w
        return round(num / den, 3) if den else None

    def series(self, t0: float, t1: float, step: float = 0.2) -> list:
        snap = self._snapshot()
        out, t = [], t0
        while t <= t1:
            v = self.value(t, snap)
            if v is not None:
                out.append([round(t, 2), round(v, 3)])
            t += step
        return out
