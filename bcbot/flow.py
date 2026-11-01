"""The MHH bot's BC 5 s model, ported as-is (mhh-btc-15min-bot pm/bc_updown.py: BinanceSpot, features_from_arrays,
Predictor): a logistic regression on Binance BTCUSDT order flow, P(UP over the 5 s window starting lead_s from now).

Features from the last 30 closed 1 s klines, for w in 1, 3, 5, 10, 30 seconds:
  ret_w = sign(close now - close w s ago)
  obi_w = (2 * taker-buy volume - volume) / volume over the last w s, clipped to [-1, 1]
The model file (models/flow5s_logit.json) is MHH's own: fitted on 8 days of Binance data, lead 2 s, horizon 5 s.
"""
import asyncio
import json
import logging
import math
import time
from collections import deque
from pathlib import Path

import aiohttp
import numpy as np

log = logging.getLogger(__name__)

STREAM_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1s/btcusdt@bookTicker"
MID_EVERY_S = 0.2        # bookTicker mid sampled at most this often (for the price chart)
WINDOWS = (1, 3, 5, 10, 30)
MAX_AGE_S = 2.5          # MHH's max_binance_age_s: the last closed second must be this fresh


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

    def arrays(self, need: int = 31):
        """Close, volume, taker-buy volume of the last `need` contiguous closed seconds, or None on a gap."""
        if len(self.ring) < need:
            return None
        rows = list(self.ring)[-need:]
        if rows[-1][0] - rows[0][0] != need - 1:
            return None
        a = np.array(rows, float)
        return a[:, 1], a[:, 2], a[:, 3]

    def age(self) -> float:
        return time.time() - (self.last_closed_sec + 1) if self.last_closed_sec else 999.0


DEPTH_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"


class BinanceBook:
    """Binance BTCUSDT top-20 order book imbalance, as the MHH dashboard's "OB · top 20 levels" card shows it:
    (bid qty - ask qty) / (bid qty + ask qty) over the top 1, 5 and 20 levels, +1 = all bids, -1 = all asks."""

    def __init__(self) -> None:
        self.obi1 = self.obi5 = self.obi20 = 0.0
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
        self.at = time.time()
        sec = int(self.at)
        if not self.hist or self.hist[-1][0] != sec:
            self.hist.append((sec, round(self.obi20, 4)))


def features_from_arrays(c: np.ndarray, v: np.ndarray, tb: np.ndarray) -> dict:
    f = {}
    for w in WINDOWS:
        f[f"ret{w}"] = float(np.sign(c[-1] - c[-1 - w]))
        vv, tt = float(v[-w:].sum()), float(tb[-w:].sum())
        f[f"obi{w}"] = float(np.clip((2 * tt - vv) / max(vv, 1e-9), -1, 1))
    return f


class FlowLogit:
    def __init__(self, path: Path):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        self.features = m["features"]
        self.coef = np.array(m["coef"], float)
        self.intercept = float(m["intercept"])
        self.lead_ms = int(float(m.get("lead_s", 2)) * 1000)
        self.q75 = float((m.get("conf_thresholds") or {}).get("q75", 0.0))

    def predict(self, flow: BinanceFlow) -> dict:
        """{"p_up", "conf", "call"} or {"error": why}."""
        arrays = flow.arrays(31)
        if arrays is None:
            return {"error": "Binance flow: not 31 contiguous seconds yet"}
        if flow.age() > MAX_AGE_S:
            return {"error": f"Binance flow stale ({flow.age():.1f}s)"}
        f = features_from_arrays(*arrays)
        z = self.intercept + float(np.dot(self.coef, [f[k] for k in self.features]))
        p = 1.0 / (1.0 + math.exp(-z))
        return {"p_up": round(p, 4), "conf": round(abs(p - 0.5), 4), "call": "UP" if p >= 0.5 else "DOWN",
                "age_s": round(flow.age(), 2)}
