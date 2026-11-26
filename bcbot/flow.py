"""Binance BTCUSDT feeds for the Up/Down bot: the 1 s klines with taker-buy volume (the taker-flow inputs) and the
best bid/ask mid (a price series), and the top-20 order book imbalance (the panel's OB card and the book inputs).

Until 2026-11-11 this file also held the MHH bot's logistic model on that flow ("mhh", "mhh_hi", "agree"); it lost
(49.5% of 2,985 calls since the last 'P&L -> 0', 49.4% of 13,878 on all logged rounds) and was removed with its model
file (models/flow5s_logit.json).
"""
import asyncio
import json
import logging
import time
from collections import deque

import aiohttp

log = logging.getLogger(__name__)

STREAM_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1s/btcusdt@bookTicker"
MID_EVERY_S = 0.2        # bookTicker mid sampled at most this often (for the price chart)


class BinanceFlow:
    def __init__(self) -> None:
        self.ring: deque = deque(maxlen=600)     # (sec, close, volume, taker-buy volume) of closed seconds
        self.mids: deque = deque(maxlen=3000)    # (local time, best bid/ask mid), for the price chart
        self.last_closed_sec = 0
        self.connected = False
        self.reconnects = 0

    async def run(self) -> None:
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(STREAM_WS, heartbeat=20) as ws:
                        self.connected = True
                        log.info("Binance BTCUSDT 1 s klines + best bid/ask connected")
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            d = json.loads(msg.data).get("data") or {}
                            if "b" in d and "a" in d and "k" not in d:        # bookTicker
                                now, mid = time.time(), (float(d["b"]) + float(d["a"])) / 2
                                if self.mids and now // MID_EVERY_S == self.mids[-1][0] // MID_EVERY_S:
                                    self.mids[-1] = (now, mid)      # one row per 0.2 s: the latest price in it
                                else:
                                    self.mids.append((now, mid))
                                continue
                            k = d.get("k") or {}
                            if k.get("x"):                                   # closed second
                                sec = int(k["t"]) // 1000
                                if sec > self.last_closed_sec:
                                    self.ring.append((sec, float(k["c"]), float(k["v"]), float(k["V"])))
                                    self.last_closed_sec = sec
            except Exception as e:
                log.warning("Binance stream: %r", e)
            self.connected = False
            self.reconnects += 1
            await asyncio.sleep(2)

    def age(self) -> float:
        return time.time() - (self.last_closed_sec + 1) if self.last_closed_sec else 999.0


DEPTH_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"


class BinanceBook:
    """Binance BTCUSDT top-20 order book imbalance, as the MHH dashboard's "OB · top 20 levels" card shows it:
    (bid qty - ask qty) / (bid qty + ask qty) over the top 1, 5 and 20 levels, +1 = all bids, -1 = all asks."""

    def __init__(self) -> None:
        self.obi1 = self.obi5 = self.obi20 = 0.0
        self.obi_w = 0.0                          # depth-weighted imbalance: level k weighted 1/(1+k), 20 levels
        self.micro_off = 0.0                      # microprice minus mid ($): size-weighted best quotes
        self.spread = 0.0                         # best ask minus best bid ($)
        self.hist: deque = deque(maxlen=240)      # (sec, obi20) once a second, for the trend line
        self.connected = False
        self.at = 0.0                             # local time of the last book update

    def age(self) -> float:
        return time.time() - self.at if self.at else 999.0

    async def run(self) -> None:
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(DEPTH_WS, heartbeat=20) as ws:
                        self.connected = True
                        log.info("Binance BTCUSDT top-20 book connected")
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                break
                            d = json.loads(msg.data)
                            bids, asks = d.get("bids") or [], d.get("asks") or []
                            if bids and asks:
                                self._update(bids, asks)
            except Exception as e:
                log.warning("Binance book stream: %r", e)
            self.connected = False
            await asyncio.sleep(2)

    def _update(self, bids: list, asks: list) -> None:
        def imb(n):
            b, a = sum(float(q) for _, q in bids[:n]), sum(float(q) for _, q in asks[:n])
            return (b - a) / (b + a) if b + a > 0 else 0.0
        self.obi1, self.obi5, self.obi20 = imb(1), imb(5), imb(20)
        wb = sum(float(q) / (1 + k) for k, (_, q) in enumerate(bids[:20]))
        wa = sum(float(q) / (1 + k) for k, (_, q) in enumerate(asks[:20]))
        self.obi_w = (wb - wa) / (wb + wa) if wb + wa > 0 else 0.0
        try:
            pb, qb = float(bids[0][0]), float(bids[0][1])
            pa, qa = float(asks[0][0]), float(asks[0][1])
            if qb + qa > 0:
                self.micro_off = (pb * qa + pa * qb) / (qb + qa) - (pb + pa) / 2
            self.spread = pa - pb
        except (ValueError, IndexError):
            pass
        self.at = time.time()
        sec = int(self.at)
        if not self.hist or self.hist[-1][0] != sec:
            self.hist.append((sec, round(self.obi20, 4)))
