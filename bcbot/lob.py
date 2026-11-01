"""Bitfinex BTC/USD limit order book -> LOB event streams, as in Raffaelli et al. (2026), Sect. 2.

Event types (an update that changes the mid-price is a price event; otherwise level-1 volume changes count):
  HawkesTime types:  Ec (mid-price change), El+ (L1 liquidity increase), El- (L1 liquidity decrease)
  HawkesSign types:  Ec+ (mid up), Ec- (mid down), Es (sell pressure), Eb (buy pressure)
Base Imbalance (eq. 1), with level l = 5 (Table 4):
  BI = ((Pbid1 - Pbid_l) - (Pask_l - Pask1)) / ((Pbid1 - Pbid_l) + (Pask_l - Pask1))
Returns R_k = (P_{k+1} - P_k) / P_k between consecutive mid-price changes, stamped at the change that realises them.
Times are Bitfinex server seconds (the TIMESTAMP flag), the same epoch clock as the round times of bc.game.
"""
import asyncio
import json
import logging
import time
from collections import deque

import aiohttp

log = logging.getLogger(__name__)

WS_URL = "wss://api-pub.bitfinex.com/ws/2"
BI_LEVEL = 5
KEEP_S = 2400                      # 40 min: the COE trains on the last 30 min (Sect. 4.2)

EC, EL_UP, EL_DN = 0, 1, 2         # HawkesTime event types
EC_UP, EC_DN, ES, EB = 0, 1, 2, 3  # HawkesSign event types


class LobFeed:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ev_time: deque = deque()     # (t, type) over {Ec, El+, El-}
        self.ev_sign: deque = deque()     # (t, type) over {Ec+, Ec-, Es, Eb}
        self.bi: deque = deque()          # (t, BI) whenever BI changes (zero-order hold input)
        self.ret: deque = deque()         # (t_{k+1}, R_k) realised at each mid-price change
        self.mids: deque = deque()        # (t, mid) at each mid-price change
        self.mid = 0.0
        self.connected = False
        self.msgs = 0
        self.reconnects = 0
        self.offset = 0.0                 # Bitfinex server time - local time, seconds (median of recent messages)
        self._offsets: deque = deque(maxlen=200)
        self.last_msg_at = 0.0            # local time of the last book message
        self._prev = None                 # (best bid, best ask, bid vol 1, ask vol 1)
        self._last_bi = None

    def now(self) -> float:
        """Current time on the Bitfinex server clock."""
        return time.time() + self.offset

    def age(self) -> float:
        """Seconds since the last book message (local clock)."""
        return time.time() - self.last_msg_at if self.last_msg_at else 999.0

    def mid_at(self, t: float) -> float | None:
        """Last mid-price at or before t."""
        best = None
        for tm, m in reversed(self.mids):
            if tm <= t:
                return m
            best = m
        return best

    # ---------- book maintenance ----------

    def _apply(self, price: float, count: int, amount: float) -> None:
        side = self.bids if amount > 0 else self.asks
        if count == 0:
            (self.bids if amount == 1 else self.asks).pop(price, None)
        else:
            side[price] = abs(amount)

    def _bi(self) -> float:
        b = sorted(self.bids, reverse=True)[:BI_LEVEL]
        a = sorted(self.asks)[:BI_LEVEL]
        if len(b) < BI_LEVEL or len(a) < BI_LEVEL:
            return 0.0
        db, da = b[0] - b[-1], a[-1] - a[0]
        return (db - da) / (db + da) if db + da > 0 else 0.0

    def _event(self, t: float) -> None:
        if not self.bids or not self.asks:
            return
        bb, ba = max(self.bids), min(self.asks)
        bv, av = self.bids[bb], self.asks[ba]
        mid = (bb + ba) / 2
        bi = self._bi()
        if bi != self._last_bi:
            self.bi.append((t, bi))
            self._last_bi = bi
        prev, self._prev = self._prev, (bb, ba, bv, av)
        if prev is None:
            self.mid = mid
            return
        pbb, pba, pbv, pav = prev
        pmid = (pbb + pba) / 2
        if mid != pmid:
            self.ev_time.append((t, EC))
            self.ev_sign.append((t, EC_UP if mid > pmid else EC_DN))
            self.ret.append((t, (mid - pmid) / pmid))
            self.mids.append((t, mid))
        elif bb == pbb and ba == pba:
            dbid, dask = bv - pbv, av - pav
            if dbid > 0 or dask > 0:
                self.ev_time.append((t, EL_UP))
            elif dbid < 0 or dask < 0:
                self.ev_time.append((t, EL_DN))
            if dask > 0 or dbid < 0:
                self.ev_sign.append((t, ES))
            elif dbid > 0 or dask < 0:
                self.ev_sign.append((t, EB))
        self.mid = mid

    def _prune(self, t: float) -> None:
        for q in (self.ev_time, self.ev_sign, self.bi, self.ret, self.mids):
            while q and q[0][0] < t - KEEP_S:
                q.popleft()

    # ---------- websocket ----------

    async def run(self) -> None:
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(WS_URL, heartbeat=20) as ws:
                        await ws.send_json({"event": "conf", "flags": 32768})      # server timestamps
                        await ws.send_json({"event": "subscribe", "channel": "book", "symbol": "tBTCUSD",
                                            "prec": "P0", "freq": "F0", "len": "25"})
                        self.connected = True
                        log.info("Bitfinex BTC/USD book connected")
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            self._on_message(json.loads(msg.data))
            except Exception as e:
                log.warning("Bitfinex book: %r", e)
            self.connected = False
            self.reconnects += 1
            self._prev = None
            self._offsets.clear()         # the clock is re-measured on the new connection
            await asyncio.sleep(2)

    def _on_message(self, d) -> None:
        if isinstance(d, dict):
            if d.get("event") == "info" and d.get("code") == 20051:
                raise ConnectionError("Bitfinex asked to reconnect")
            return
        if len(d) < 2 or d[1] == "hb":
            return
        stamped = len(d) > 2 and isinstance(d[2], (int, float))
        payload, ts = d[1], (d[2] / 1000.0 if stamped else self.now())
        self.msgs += 1
        self.last_msg_at = time.time()
        if stamped:                       # a median: late messages (never early ones) don't drag the clock
            self._offsets.append(ts - self.last_msg_at)
            if len(self._offsets) >= 5:
                self.offset = sorted(self._offsets)[len(self._offsets) // 2]
            elif self.msgs == 1:
                self.offset = ts - self.last_msg_at
        if payload and isinstance(payload[0], list):          # snapshot
            self.bids.clear(); self.asks.clear()
            for p, c, a in payload:
                self._apply(p, c, a)
            self._prev = None
            self._event(ts)
        else:
            self._apply(*payload)
            self._event(ts)
        if self.msgs % 500 == 0:
            self._prune(ts)
