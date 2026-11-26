"""Frozen, paper-only BTC direction classifier. No trading or auto-mode integration.

The training tool records a chronological evaluation and exports one JSON artifact.
A direction is a prediction; 'selected' only means the fixed probability threshold
was met. It is never a claim that the model's edge has been established.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 1
MIN_LEAD = 1.5
MAX_LEAD = 4.5
MIN_VOL = 0.05
THRESHOLD = 0.56
CORE = ("dev_tick", "dev_chg1", "bc_mom1", "bc_mom2", "bc_mom5",
        "our_mom1", "our_mom2", "our_mom5")
FLOW = ("obi1", "obi5", "obi20", "obi20_chg5", "taker1", "taker3", "taker5",
        "ofi_binance_perp_1s", "ofi_bybit_perp_1s", "ofi_okx_perp_1s")
FAMILIES = ("gap", "raw", "normalized", "flow")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def feature_names(family):
    if family not in FAMILIES:
        raise ValueError(f"unknown feature family: {family}")
    if family == "gap":
        return ["dev_tick/vol60"]
    names = [f"{k}/vol60" if family != "raw" else k for k in CORE]
    if family == "flow":
        names += list(FLOW)
    return names


def vector(feat, family):
    """Use only the decision snapshot; missing optional inputs are neutral.

    Signed features and a zero intercept make UP/DOWN predictions symmetric.
    Raw price changes are clipped too, using a much wider dollar bound.
    """
    if not isinstance(feat, dict) or feat.get("error"):
        return None
    if not (feat.get("dq") or {}).get("ok"):
        return None
    d, v, m = (number(feat.get(k)) for k in ("dev_tick", "vol60", "bc_mom2"))
    lead, age = number(feat.get("lead_s")), number(feat.get("tick_age"))
    if d is None or v is None or m is None or v < MIN_VOL:
        return None
    if lead is None or not MIN_LEAD <= lead <= MAX_LEAD:
        return None
    if age is None or not 0 <= age <= 1.0:
        return None
    if family == "gap":
        return np.array([max(-12.0, min(12.0, d / v))])
    feature_names(family)  # validate before building a vector
    divisor, cap = (1.0, 100.0) if family == "raw" else (v, 12.0)
    x = [max(-cap, min(cap, (number(feat.get(k)) or 0.0) / divisor)) for k in CORE]
    if family == "flow":
        x += [max(-1e6, min(1e6, number(feat.get(k)) or 0.0)) for k in FLOW]
    return np.array(x, dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_logistic(X, y, penalty):
    """Ridge logistic fit. RMS scaling is fitted only on the supplied train set."""
    scale = np.sqrt(np.mean(X * X, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    Z = np.clip(X / scale, -8.0, 8.0)
    weights = np.zeros(X.shape[1])
    eye = np.eye(len(weights))
    for _ in range(60):
        p = sigmoid(Z @ weights)
        gradient = Z.T @ (p - y) + penalty * weights
        hessian = (Z * (p * (1.0 - p))[:, None]).T @ Z + penalty * eye
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.isfinite(weights).all():
        raise ValueError("non-finite logistic fit")
    return {"scale": scale.tolist(), "weights": weights.tolist()}


def logits(fit, X):
    return np.clip(X / np.asarray(fit["scale"]), -8.0, 8.0) @ np.asarray(fit["weights"])


class UDDirectionModel:
    def __init__(self, artifact):
        if artifact.get("schema") != SCHEMA or artifact.get("purpose") != "paper":
            raise ValueError("expected a paper direction-model artifact")
        self.artifact = artifact
        self.family = artifact["family"]
        if artifact["features"] != feature_names(self.family):
            raise ValueError("artifact feature order differs from code")
        self.fit = artifact["fit"]
        scales = np.asarray(self.fit["scale"], dtype=float)
        weights = np.asarray(self.fit["weights"], dtype=float)
        if scales.shape != (len(artifact["features"]),) or weights.shape != scales.shape:
            raise ValueError("artifact dimensions differ")
        if not np.isfinite(scales).all() or not np.isfinite(weights).all() or np.any(scales <= 0):
            raise ValueError("invalid model parameters")
        self.temperature = float(artifact["temperature"])
        self.threshold = float(artifact["threshold"])
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("invalid calibration temperature")
        if not 0.5 < self.threshold < 1:
            raise ValueError("invalid selection threshold")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, feat):
        x = vector(feat, self.family)
        if x is None:
            return None
        p = float(sigmoid(self.temperature * logits(self.fit, x[None, :]))[0])
        side = "UP" if p >= 0.5 else "DOWN"
        confidence = max(p, 1.0 - p)
        return {"side": side, "p_up": p, "confidence": confidence,
                "selected": confidence >= self.threshold, "purpose": "paper"}
