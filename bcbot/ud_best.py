"""History-trained BC five-second direction models and causal order-book inputs.

The same feature builder serves historical fitting and prospective inference.
Only decision snapshots and books received by decision_pc are inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import obfeat as F
from .ud_model import fit_logistic, logits, sigmoid

SCHEMA = 1
MARKETS = tuple(F.SHARES)
DOLLARS = ("dev_tick", "dev_chg1", "dev_1s", "dev_3s", "tick_chg",
           "bc_mom05", "bc_mom1", "bc_mom2", "bc_mom5", "bc_mom10",
           "our_mom1", "our_mom2", "our_mom5", "our_mom10", "basis_chg10", "dev_fair")
RATIOS = ("obi1", "obi5", "obi20", "obi_w", "obi20_chg5", "taker1", "taker3", "taker5", "taker10", "dev_z60")
FLOWS = tuple(f"ofi_{k}_{w}s" for k in ("binance_perp", "bybit_perp", "okx_perp") for w in (1, 3))
CONTEXT = tuple("ctx_" + k for k in DOLLARS + RATIOS + FLOWS) + (
    "ctx_gap_dead1", "ctx_gap_dead2", "ctx_gap_quiet", "ctx_pool_imb",
    "ctx_log_vol", "ctx_tick_age", "ctx_gap_std", "ctx_fair_missing")
CONTEXT_UNSIGNED = {"ctx_log_vol", "ctx_tick_age", "ctx_gap_std", "ctx_fair_missing"}
DYNAMIC = ("ofi025", "ofi1", "ofi3", "avgimb05", "avgimb1", "avgimb3", "dimb1",
           "micro", "depthflow1", "trade_depth1", "trade_depth3", "activity1",
           "ret025", "ret05", "spreadchange1", "logdepth",
           "life_avg", "life_age_signed", "life_response")
DYNAMIC_UNSIGNED = {"activity1", "spreadchange1", "logdepth"}
PER = F.PER + DYNAMIC
MIX = tuple("mix_" + k for k in PER) + tuple("mix5_" + k for k in PER) + ("breadth", "dispersion")
NAMES = CONTEXT + tuple(k + "_" + f for k in MARKETS for f in PER) + MIX
UNSIGNED = CONTEXT_UNSIGNED | {k + "_" + f for k in MARKETS for f in ("spread",) + tuple(DYNAMIC_UNSIGNED)}
UNSIGNED |= {prefix + f for prefix in ("mix_", "mix5_") for f in ("spread",) + tuple(DYNAMIC_UNSIGNED)} | {"dispersion"}
SIGN = np.array([1.0 if n in UNSIGNED else -1.0 for n in NAMES])


def number(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return float(x)
    return None


def valid_snapshot(f):
    if not isinstance(f, dict) or f.get("error"):
        return False
    d, v, at = (number(f.get(k)) for k in ("dev_tick", "vol60", "at"))
    age = number(f.get("tick_age"))
    return d is not None and v is not None and v >= .05 and at is not None and age is not None and 0 <= age <= 1.5


def context(f):
    v = max(float(f["vol60"]), .25)
    def get(k):
        x = number(f.get(k))
        return np.nan if x is None else x
    out = [np.clip(get(k) / v, -12, 12) for k in DOLLARS]
    out += [np.clip(get(k), -12, 12) for k in RATIOS]
    out += [np.arcsinh(get(k)) for k in FLOWS]
    g = np.clip(f["dev_tick"] / v, -12, 12)
    up, dn = number(f.get("pool_up")), number(f.get("pool_dn"))
    pi = (up - dn) / (up + dn) if up is not None and dn is not None and up + dn > 0 and not f.get("pools_prev") else np.nan
    out += [np.sign(g) * max(abs(g) - 1, 0), np.sign(g) * max(abs(g) - 2, 0),
            g / math.sqrt(1 + v), pi, math.log(v), f["tick_age"],
            get("dev_std60") / v, float(number(f.get("dev_fair")) is None)]
    return np.asarray(out, dtype=float)


def _window_sum(t, values, decisions, window):
    values = np.asarray(values, dtype=float)
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(values))]
    hi = np.searchsorted(t, decisions, side="right")
    lo = np.searchsorted(t, decisions - window, side="right")
    return cs[hi] - cs[lo]


def market_features(a, decisions, vol):
    """Features use only rows at or before each decision, including rolling sums."""
    decisions, vol = np.asarray(decisions), np.asarray(vol)
    if not len(a):
        return np.full((len(decisions), len(PER)), np.nan)
    t = a["t"]
    i = np.searchsorted(t, decisions, side="right") - 1
    ix = np.clip(i, 0, len(a) - 1)
    ok = ((i >= 0) & (decisions - t[ix] <= F.MAX_LAG) & (a["age"][ix] <= F.MAX_AGE)
          & np.isfinite(a["bid"][ix]) & np.isfinite(a["ask"][ix]) & (a["ask"][ix] >= a["bid"][ix]))
    b, s = a["bs"][:, 0].astype(float), a["as_"][:, 0].astype(float)
    depth = b + s
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(a["bid"]) & np.isfinite(a["ask"]) & (a["age"] <= F.MAX_AGE)
    imb = np.where(valid, (b - s) / np.where(depth > 0, depth, 1), np.nan)
    mid = (a["bid"] + a["ask"]) / 2
    spread = a["ask"] - a["bid"]
    ofi = np.zeros(len(a))
    df = np.zeros(len(a))
    good_pair = valid[1:] & valid[:-1] & (np.diff(t) <= F.MAX_LAG)
    # Cont-Kukanov-Stoikov signed changes at the best bid/ask, normalized by depth.
    e = ((a["bid"][1:] >= a["bid"][:-1]) * b[1:]
         - (a["bid"][1:] <= a["bid"][:-1]) * b[:-1]
         - (a["ask"][1:] <= a["ask"][:-1]) * s[1:]
         + (a["ask"][1:] >= a["ask"][:-1]) * s[:-1])
    ofi[1:] = np.where(good_pair, e / np.maximum((depth[1:] + depth[:-1]) / 2, 1e-8), 0)
    df[1:] = np.where(good_pair, ((b[1:] - b[:-1]) - (s[1:] - s[:-1])) / np.maximum(depth[:-1], 1e-8), 0)
    col = {}
    for label, w in (("025", .25), ("1", 1.), ("3", 3.)):
        col["ofi" + label] = np.clip(_window_sum(t, ofi, decisions, w), -20, 20)
    for label, w in (("05", .5), ("1", 1.), ("3", 3.)):
        count = _window_sum(t, np.isfinite(imb), decisions, w)
        col["avgimb" + label] = np.where(count >= max(2, w / F.STEP / 2),
                                                        _window_sum(t, imb, decisions, w) / np.maximum(count, 1), np.nan)
    def past(w):
        j = np.searchsorted(t, decisions - w, side="right") - 1
        jc = np.clip(j, 0, len(a) - 1)
        known = (j >= 0) & (decisions - w - t[jc] <= F.MAX_LAG) & valid[jc]
        return jc, known
    j, known = past(1)
    col["dimb1"] = np.where(known, imb[ix] - imb[j], np.nan)
    col["micro"] = spread[ix] * imb[ix] / (2 * vol)
    col["depthflow1"] = np.clip(_window_sum(t, df, decisions, 1), -20, 20)
    for w in (1, 3):
        traded = _window_sum(t, a["buy"].astype(float) - a["sell"].astype(float), decisions, w)
        col[f"trade_depth{w}"] = np.arcsinh(traded / np.maximum(depth[ix], 1e-8))
    col["activity1"] = np.log1p(_window_sum(t, a["nb"].astype(float) + a["ns"].astype(float), decisions, 1))
    for label, w in (("025", .25), ("05", .5)):
        j, known = past(w)
        col["ret" + label] = np.where(known, (mid[ix] - mid[j]) / vol, np.nan)
    j, known = past(1)
    col["spreadchange1"] = np.where(known, (spread[ix] - spread[j]) / vol, np.nan)
    col["logdepth"] = np.log1p(np.maximum(depth[ix], 0))
    life = F.life_market(a, decisions)
    col["life_avg"] = life[:, 0]
    col["life_age_signed"] = np.sign(life[:, 0]) * life[:, 1] / 10
    col["life_response"] = life[:, 2] / vol
    dyn = np.column_stack([col[k] for k in DYNAMIC])
    dyn[~ok] = np.nan
    base = F.market_features(a, decisions, vol)
    base[~ok] = np.nan
    return np.column_stack([base, dyn])


def assemble(contexts, per):
    n = len(contexts)
    blocks = [per.get(k, np.full((n, len(PER)), np.nan)) for k in MARKETS]
    mix = []
    for weights in ([F.SHARES[k] for k in MARKETS], [0 if k == "bitfinex_perp" else 1 for k in MARKETS]):
        cube = np.stack(blocks, axis=1)
        w = np.asarray(weights)[None, :, None] * np.isfinite(cube)
        den = w.sum(axis=1)
        mix.append(np.where(den >= sum(weights) * .5,
                            np.nansum(cube * w, axis=1) / np.maximum(den, 1e-8), np.nan))
    values = np.column_stack([p[:, PER.index("imb1")] for p in blocks])
    count = np.isfinite(values).sum(axis=1)
    breadth = np.nansum(np.sign(values), axis=1) / np.maximum(count, 1)
    mean = np.nansum(values, axis=1) / np.maximum(count, 1)
    dispersion = np.sqrt(np.nansum((values - mean[:, None]) ** 2, axis=1) / np.maximum(count, 1))
    breadth[count < 3] = np.nan
    dispersion[count < 3] = np.nan
    return np.column_stack([contexts] + blocks + mix + [breadth, dispersion])


def build(snapshots, books, decision_pc):
    vol = np.array([max(float(f["vol60"]), .25) for f in snapshots])
    return assemble(np.array([context(f) for f in snapshots]),
                    {k: market_features(books.get(k, np.zeros(0, dtype=F.ROW)), decision_pc, vol) for k in MARKETS})


def indices(group):
    if group == "gap":
        return [CONTEXT.index(n) for n in ("ctx_dev_tick", "ctx_gap_dead1", "ctx_gap_dead2")]
    if group == "context":
        return list(range(len(CONTEXT)))
    if group == "book":
        return list(range(len(CONTEXT))) + [i for i, n in enumerate(NAMES) if i >= len(CONTEXT) and any(n.endswith("_" + f) for f in F.PER)]
    if group == "dynamic":
        return list(range(len(CONTEXT))) + [i for i, n in enumerate(NAMES) if n.startswith(("mix_", "mix5_")) or n in ("breadth", "dispersion")]
    if group == "all":
        return list(range(len(NAMES)))
    raise ValueError(group)


def raw_predict(payload, X):
    ix = payload["indices"]
    A = X[:, ix]
    sign = SIGN[ix]
    if payload["kind"] == "ridge":
        return sigmoid(logits(payload["fit"], np.nan_to_num(A)))
    if payload["kind"] == "lgbm":
        import lightgbm as lgb
        model = payload.get("_model")
        if model is None:
            model = payload["_model"] = lgb.Booster(model_str=payload["model"])
        a, b = model.predict(A, num_threads=2), model.predict(A * sign, num_threads=2)
    elif payload["kind"] == "xgb":
        import xgboost as xgb
        model = payload.get("_model")
        if model is None:
            model = payload["_model"] = xgb.Booster(params={"nthread": 2})
            model.load_model(bytearray(payload["model"], "utf-8"))
        a, b = model.predict(xgb.DMatrix(A)), model.predict(xgb.DMatrix(A * sign))
    else:
        raise ValueError("unknown model kind")
    return (a + 1 - b) / 2


def calibrated(payload, X, temperature):
    p = np.clip(raw_predict(payload, X), 1e-8, 1 - 1e-8)
    return sigmoid(temperature * np.log(p / (1 - p)))


class BestUDModel:
    """Standalone predictor: supplies probabilities and calls; never places orders."""
    def __init__(self, artifact):
        if artifact.get("schema") != SCHEMA or artifact.get("features") != list(NAMES):
            raise ValueError("incompatible history-model feature schema")
        if artifact.get("purpose") != "paper":
            raise ValueError("expected a paper model artifact")
        for key in ("all", "selective"):
            policy = artifact["policies"][key]
            if not 0 <= policy["temperature"] <= 2 or not .5 <= policy["threshold"] < 1:
                raise ValueError("invalid policy calibration")
            ix = policy["model"]["indices"]
            if not ix or len(set(ix)) != len(ix) or any(i < 0 or i >= len(NAMES) for i in ix):
                raise ValueError("invalid policy feature indices")
        self.artifact = artifact

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_matrix(self, X, policy="selective"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(NAMES):
            raise ValueError("wrong feature dimensions")
        p = self.artifact["policies"][policy]
        return calibrated(p["model"], X, p["temperature"])

    def predict(self, snapshot, books, decision_pc, policy="selective"):
        if not valid_snapshot(snapshot):
            return None
        # A book model refuses to turn missing books into an apparent prediction.
        p = self.artifact["policies"][policy]
        if p["group"] not in ("gap", "context"):
            coverage = sum(F.SHARES[k] for k in MARKETS if len(books.get(k, ())) and
                           np.isfinite(market_features(books[k], [decision_pc], [max(snapshot["vol60"], .25)])[0, 1]))
            if coverage < F.MIN_SHARE:
                return None
        X = build([snapshot], books, [decision_pc])
        prob = float(self.predict_matrix(X, policy)[0])
        selected = abs(prob - .5) > 1e-10 and max(prob, 1 - prob) >= p["threshold"]
        return {"p_up": prob, "side": "UP" if prob > .5 else "DOWN" if prob < .5 else None,
                "selected": selected, "policy": policy, "purpose": "paper"}


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
