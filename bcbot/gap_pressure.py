"""Gap Pressure: hierarchical probability tables for BC five-second direction.

Learns how changing liquidity alters a gap-reversion prior. No decision trees.
The saved tables, scaling, calibration and call threshold define a frozen model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import ud_best as B
from .ud_model import sigmoid

SCHEMA = 1
GAP_EDGES = np.array([.5, 1., 1.5, 2., 2.5, 3., 4., 6.])
COARSE_EDGES = np.array([1., 2., 3.])
PRESSURE_EDGES = np.array([-.6, -.15, .15, .6])
RESPONSE_EDGES = np.array([-.5, .5])
RAW_NAMES = ("mix5_ofi025", "mix5_ofi1", "mix5_ofi3", "mix5_trade_depth1")


def extract(X):
    """Only quantities known at the logged decision; missing books remain unknown."""
    X = np.asarray(X, dtype=float)
    return (X[:, B.NAMES.index("ctx_dev_tick")],
            X[:, [B.NAMES.index(n) for n in RAW_NAMES]],
            X[:, B.NAMES.index("mix5_ret1")])


def pressure(raw, scales, recipe):
    q = np.tanh(raw / np.asarray(scales))
    if recipe == "queue":
        return q[:, 1]
    if recipe == "turn":
        return .5*q[:, 0] + .5*q[:, 1] - .25*q[:, 2]
    if recipe == "blend":
        return .5*q[:, 1] + .3*q[:, 3] + .2*q[:, 0]
    raise ValueError("unknown pressure recipe")


def states(gap, q, response):
    # Orient every observation to a possible bet against the current price gap.
    # At zero gap use the pressure direction; at zero gap and pressure no direction exists.
    direction = np.where(gap < 0, 1., np.where(gap > 0, -1., np.sign(q)))
    g = np.searchsorted(GAP_EDGES, np.abs(gap), side="right")
    coarse = np.searchsorted(COARSE_EDGES, np.abs(gap), side="right")
    qb = np.searchsorted(PRESSURE_EDGES, direction*q, side="right")
    rb = np.searchsorted(RESPONSE_EDGES, direction*response, side="right")
    known = np.isfinite(gap) & np.isfinite(q) & np.isfinite(response) & (direction != 0)
    return direction, g, coarse*5 + qb, (coarse*5 + qb)*3 + rb, known


def monotonic(values, weights):
    """Weighted pool-adjacent-violators estimate of a monotone gap prior."""
    groups = []
    for i, (v, w) in enumerate(zip(values, weights)):
        groups.append([i, i+1, float(v*w), float(w)])
        while len(groups) > 1 and groups[-2][2]/groups[-2][3] > groups[-1][2]/groups[-1][3]:
            b = groups.pop()
            a = groups.pop()
            groups.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
    out = np.empty(len(values))
    for lo, hi, total, weight in groups:
        out[lo:hi] = total/weight
    return out


def table(keys, target, weights, length):
    n = np.bincount(keys, weights=weights, minlength=length)
    w = np.bincount(keys, weights=weights*target, minlength=length)
    count = np.bincount(keys, minlength=length)
    return {"effective_count": n.tolist(), "effective_wins": w.tolist(), "rows": count.tolist()}


def fit(X, y_up, times, train, have_books, spec):
    """Fit only supplied past rows; the trainer purges settlement boundaries."""
    gap, raw, response = extract(X)
    train = np.asarray(train, dtype=bool)
    book_train = train & have_books & np.isfinite(raw).all(axis=1) & np.isfinite(response)
    if book_train.sum() < 100:
        raise ValueError("not enough past book observations")
    scales = np.quantile(np.abs(raw[book_train]), .75, axis=0)
    scales = np.maximum(scales, 1e-6)
    q = pressure(raw, scales, spec["recipe"])
    direction, gb, qk, rk, known = states(gap, q, response)
    # All earlier snapshots contribute to the gap prior, including pre-book history.
    gap_train = train & np.isfinite(gap) & (gap != 0)
    gap_target = np.where(gap < 0, y_up, 1-y_up)
    end = float(np.max(times[train]))
    if spec["half_life_h"]:
        weights = np.exp2(-np.maximum(end-times, 0)/(spec["half_life_h"]*3600))
    else:
        weights = np.ones(len(times))
    prior_table = table(gb[gap_train], gap_target[gap_train], weights[gap_train], len(GAP_EDGES)+1)
    n = np.asarray(prior_table["effective_count"])
    w = np.asarray(prior_table["effective_wins"])
    prior = monotonic((w+15)/(n+30), n+30)
    book_train &= known
    aligned_target = np.where(direction > 0, y_up, 1-y_up)
    return {"spec": spec, "scales": scales.tolist(), "gap_prior": prior.tolist(),
            "gap_table": prior_table,
            "pressure_table": table(qk[book_train], aligned_target[book_train], weights[book_train], 20),
            "response_table": table(rk[book_train], aligned_target[book_train], weights[book_train], 60),
            "training_rows": int(train.sum()), "training_book_rows": int(book_train.sum())}


def raw_predict(fitted, X, gap_only=False):
    gap, raw, response = extract(X)
    q = pressure(raw, fitted["scales"], fitted["spec"]["recipe"])
    direction, gb, qk, rk, known = states(gap, q, response)
    p = np.asarray(fitted["gap_prior"])[gb]
    if not gap_only:
        strength = fitted["spec"]["strength"]
        t = fitted["pressure_table"]
        n, w = np.asarray(t["effective_count"])[qk], np.asarray(t["effective_wins"])[qk]
        p = (w + strength*p)/(n+strength)
        if fitted["spec"]["depth"] == 3:
            t = fitted["response_table"]
            n, w = np.asarray(t["effective_count"])[rk], np.asarray(t["effective_wins"])[rk]
            p = (w + strength*p)/(n+strength)
        p[~known] = np.nan
    else:
        direction = np.where(gap < 0, 1., np.where(gap > 0, -1., 0.))
    out = np.where(direction > 0, p, 1-p)
    out[direction == 0] = .5
    out[~np.isfinite(gap)] = np.nan
    return out


def calibrated(fitted, X, temperature, gap_only=False):
    p = np.clip(raw_predict(fitted, X, gap_only), 1e-8, 1-1e-8)
    return sigmoid(temperature*np.log(p/(1-p)))


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class GapPressureModel:
    def __init__(self, artifact):
        if artifact.get("schema") != SCHEMA or artifact.get("algorithm") != "gap_pressure":
            raise ValueError("incompatible Gap Pressure artifact")
        if artifact.get("feature_names") != list(B.NAMES) or artifact.get("purpose") != "paper":
            raise ValueError("incompatible features or purpose")
        if not .001 <= artifact["temperature"] <= 2 or not .5 < artifact["threshold"] < 1:
            raise ValueError("invalid calibration")
        fitted = artifact["fitted"]
        if len(fitted["gap_prior"]) != 9 or len(fitted["scales"]) != 4 or min(fitted["scales"]) <= 0:
            raise ValueError("invalid probability-table dimensions")
        if not np.isfinite(fitted["gap_prior"]).all() or not all(0 <= p <= 1 for p in fitted["gap_prior"]):
            raise ValueError("invalid gap probabilities")
        self.artifact = artifact

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_matrix(self, X):
        return calibrated(self.artifact["fitted"], X, self.artifact["temperature"])

    def predict(self, snapshot, books, decision_pc):
        if not B.valid_snapshot(snapshot):
            return None
        X = B.build([snapshot], books, [decision_pc])
        p = float(self.predict_matrix(X)[0])
        if not np.isfinite(p):
            return None
        confidence = max(p, 1-p)
        selected = abs(p-.5) > 1e-10 and confidence >= self.artifact["threshold"]
        return {"p_up": p, "side": "UP" if p>.5 else "DOWN" if p<.5 else None,
                "selected": selected, "confidence": confidence, "purpose": "paper",
                "model": "gap_pressure"}
