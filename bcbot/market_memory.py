"""Market Memory: recent, similar order-book states correct a gap prior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import gap_pressure as G
from . import ud_best as B
from .ud_model import sigmoid

FEATURES = ("ctx_dev_tick", "ctx_dev_chg1", "ctx_bc_mom1", "ctx_our_mom1",
            "mix5_ofi025", "mix5_ofi1", "mix5_ofi3", "mix5_trade_depth1",
            "mix5_avgimb1", "mix5_ret1", "ctx_log_vol", "dispersion")
IDX = [B.NAMES.index(k) for k in FEATURES]
SIGNS = B.SIGN[IDX]
WEIGHTS = np.array([1.5, .5, .5, .5, .5, .75, .5, .5, .5, .5, .75, .5])


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def gap_probability(prior, X):
    gap = X[:, B.NAMES.index("ctx_dev_tick")]
    b = np.searchsorted(G.GAP_EDGES, np.abs(gap), side="right")
    p = np.asarray(prior)[b]
    p = np.where(gap < 0, p, 1-p)
    p[gap == 0] = .5
    p[~np.isfinite(gap)] = np.nan
    return p


def fit(X, y, times, train, have, spec):
    """All labels, normalization, states and residuals come from train only."""
    train = np.asarray(train, dtype=bool)
    asof = float(np.max(times[train]))
    g = X[:, B.NAMES.index("ctx_dev_tick")]
    w = np.exp2(-np.maximum(asof-times, 0)/(96*3600))
    m = train & np.isfinite(g) & (g != 0)
    bins = np.searchsorted(G.GAP_EDGES, np.abs(g[m]), side="right")
    n = np.bincount(bins, weights=w[m], minlength=9)
    win = np.bincount(bins, weights=w[m]*np.where(g[m] < 0, y[m], 1-y[m]), minlength=9)
    prior = G.monotonic((win+15)/(n+30), n+30)
    m = train & have & np.isfinite(X[:, IDX]).all(axis=1) & (times >= asof-168*3600)
    if m.sum() < 300:
        raise ValueError("at least 300 past states with known books are required")
    A = X[m][:, IDX]
    center = np.where(SIGNS > 0, np.median(A, axis=0), 0.)
    scale = np.maximum(np.quantile(np.abs(A-center), .75, axis=0), .05)
    Z = np.clip((A-center)/scale, -6, 6)*WEIGHTS
    residual = y[m]-gap_probability(prior, X[m])
    recency = np.exp2(-np.maximum(asof-times[m], 0)/(spec["half_life_h"]*3600))
    payload = {"prior": prior.tolist(), "center": center.tolist(), "scale": scale.tolist(),
               "spec": spec, "training_states": int(m.sum()), "trained_asof": asof}
    arrays = {"Z": np.vstack([Z, Z*SIGNS]), "residual": np.r_[residual, -residual],
              "recency": np.r_[recency, recency]}
    return MarketMemory(payload, arrays)


class MarketMemory:
    def __init__(self, fitted, arrays):
        self.fitted = fitted
        self.arrays = arrays
        self.tree = cKDTree(arrays["Z"])
        self.temperature = 1.
        self.threshold = .50000001

    def raw(self, X):
        X = np.asarray(X, dtype=float)
        p = gap_probability(self.fitted["prior"], X)
        known = np.isfinite(X[:, IDX]).all(axis=1)
        if not known.any():
            return np.full(len(X), np.nan)
        z = np.clip((X[known][:, IDX]-self.fitted["center"])/self.fitted["scale"], -6, 6)*WEIGHTS
        k = min(self.fitted["spec"]["neighbors"], len(self.arrays["Z"]))
        distance, indices = self.tree.query(z, k=k, workers=1)
        if k == 1:
            distance, indices = distance[:, None], indices[:, None]
        radius = np.maximum(distance[:, -1:], .1)
        w = np.exp(-(distance/radius)**2)*self.arrays["recency"][indices]
        den = np.maximum(w.sum(axis=1), 1e-12)
        correction = (w*self.arrays["residual"][indices]).sum(axis=1)/den
        effective = den**2/np.maximum((w*w).sum(axis=1), 1e-12)
        correction *= effective/(effective+50)
        p[known] = np.clip(p[known]+self.fitted["spec"]["blend"]*correction, .01, .99)
        p[~known] = np.nan
        return p

    def predict_matrix(self, X):
        p = np.clip(self.raw(X), 1e-8, 1-1e-8)
        return sigmoid(self.temperature*np.log(p/(1-p)))

    def save(self, folder, metadata):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/"memory.npz", **self.arrays)
        artifact = {"schema": 1, "algorithm": "market_memory", "purpose": "paper",
                    "features": list(FEATURES), "fitted": self.fitted,
                    "temperature": self.temperature, "threshold": self.threshold,
                    "source_sha256": code_hash(), "feature_sha256": B.code_hash(),
                    "memory_sha256": hashlib.sha256((folder/"memory.npz").read_bytes()).hexdigest(),
                    **metadata}
        (folder/"model.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return artifact

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        a = json.loads((folder/"model.json").read_text(encoding="utf-8"))
        if a.get("algorithm") != "market_memory" or a.get("purpose") != "paper" or a.get("features") != list(FEATURES):
            raise ValueError("incompatible Market Memory artifact")
        if a["source_sha256"] != code_hash() or a["feature_sha256"] != B.code_hash():
            raise ValueError("frozen model code changed")
        if a["memory_sha256"] != hashlib.sha256((folder/"memory.npz").read_bytes()).hexdigest():
            raise ValueError("frozen market states changed")
        with np.load(folder/"memory.npz") as z:
            arrays = {k:z[k] for k in z.files}
        model = cls(a["fitted"], arrays)
        model.temperature, model.threshold = a["temperature"], a["threshold"]
        return model
