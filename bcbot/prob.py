"""The "prob" model: P(UP) for a round from the decision-time features, fitted on the logged rounds.

A regularized logistic regression (numpy only) on a FIXED set of inputs: the 22 raw decision-time features below plus
six combinations that encode "bc.game's index ran ahead of the market" (its last move and its gap from our exchange
price point the same way), which is the one effect that held out of sample.  It is refitted from logs/auto_rounds.jsonl
at start and every REFIT_S, always on rounds that are already settled, so a prediction is never made by a model that
saw that round.  The call: UP when P(UP) >= 0.5 + MARGIN, DOWN when <= 0.5 - MARGIN, else none.

Walk-forward on 3,446 logged rounds (2026-10-31 06:47 -> 2026-11-01 14:01, each block of 400 predicted by a fit on
everything before it): 57.7% [53-62] of 395 calls at MARGIN 0.06 (12.6 calls/h); 54.3% of 915 at 0.04; 52.8% of 1,935
at 0.02.  The margin was the best of seven looked at, so expect less forward.  Fixed 2026-11-01.
"""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FEATURES = ["bc_mom1", "bc_mom2", "bc_mom5", "bc_mom10", "dev_tick", "dev_1s", "dev_3s", "dev_chg1", "tick_chg",
            "our_mom1", "our_mom2", "our_mom5", "our_mom10", "vol60", "obi5", "obi20", "obi20_chg5", "taker3", "taker5",
            "ofi_bybit_perp_1s", "ofi_binance_perp_1s", "basis_chg10"]
REQUIRED = ("bc_mom2", "dev_tick", "vol60", "obi20")      # without these the round is not predicted
MARGIN = 0.06
REFIT_S = 3600
MIN_ROWS = 800
LAMBDAS = (3, 10, 30, 100, 300, 1000)
QUIET = 1.92


def _num(f: dict, k: str):
    v = f.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def row(feat: dict) -> np.ndarray | None:
    """The model's input vector for one round's decision-time features, or None when a required one is missing."""
    if not isinstance(feat, dict) or feat.get("error") or any(_num(feat, k) is None for k in REQUIRED):
        return None
    x = np.array([_num(feat, k) or 0.0 for k in FEATURES])
    m, d, v, o = x[FEATURES.index("bc_mom2")], x[FEATURES.index("dev_tick")], x[FEATURES.index("vol60")], x[FEATURES.index("obi20")]
    m1, d1 = x[FEATURES.index("bc_mom1")], x[FEATURES.index("dev_1s")]
    sm, quiet = np.sign(m), 1.0 if v < QUIET else 0.0
    against = 1.0 if abs(o) >= 0.5 and np.sign(o) != sm else 0.0
    inter = [sm * min(abs(d), 10), sm * min(abs(d), 10) * quiet, sm * (1.0 if np.sign(d) == sm else 0.0) * min(abs(m), 5),
             np.sign(m1) * min(abs(d1), 10), -o * sm * (1.0 if abs(o) >= 0.5 else 0.0), sm * min(abs(d), 10) * against]
    return np.concatenate([x, inter])


def load_rows(path: Path, before: float | None = None):
    """(X, y, last cutoff) from the round log: settled, non-tie rounds with features, in time order."""
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("win") not in ("UP", "DOWN") or r.get("tie") or not r.get("cutoff"):
                    continue
                if before is not None and r["cutoff"] >= before:
                    continue
                x = row(r.get("feat"))
                if x is not None:
                    rows.append((r["cutoff"], x, 1.0 if r["win"] == "UP" else 0.0))
    except OSError:
        pass
    rows.sort(key=lambda t: t[0])
    if not rows:
        return None, None, 0.0
    return np.array([x for _, x, _ in rows]), np.array([y for _, _, y in rows]), rows[-1][0]


def _fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), np.clip((X - mu) / sd, -5, 5)])
    w = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y)
        Rg = np.eye(len(w)) * lam
        Rg[0, 0] = 0
        H = (Z * (p * (1 - p))[:, None]).T @ Z + Rg
        step = np.linalg.solve(H, g + Rg @ w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def _predict(model, X):
    mu, sd, w = model
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), np.clip((X - mu) / sd, -5, 5)]) @ w)))


class ProbModel:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.model = None
        self.n_rows = 0
        self.lam = None
        self.fitted_at = 0.0
        self.trained_to = 0.0
        self.val_acc = None

    def fit(self, before: float | None = None) -> dict:
        """Blocking (run it in a thread).  The regularization is picked on the last 20% of the rows, then the model
        is refitted on all of them."""
        X, y, last = load_rows(self.log_path, before)
        if X is None or len(y) < MIN_ROWS:
            return {"rows": 0 if X is None else len(y), "fitted": False}
        cut = int(len(y) * 0.8)
        best = None
        for lam in LAMBDAS:
            m = _fit(X[:cut], y[:cut], lam)
            acc = float(((_predict(m, X[cut:]) > 0.5) == (y[cut:] > 0.5)).mean())
            if best is None or acc > best[0]:
                best = (acc, lam)
        self.model, self.lam, self.val_acc = _fit(X, y, best[1]), best[1], round(best[0], 4)
        self.n_rows, self.trained_to = len(y), last
        import time
        self.fitted_at = time.time()
        log.info("P(UP) model fitted on %d rounds (L2 %s, last-20%% accuracy %.1f%%)", len(y), best[1], 100 * best[0])
        return {"rows": len(y), "fitted": True, "lam": best[1], "val_acc": self.val_acc}

    def predict(self, feat: dict) -> float | None:
        if self.model is None:
            return None
        x = row(feat)
        if x is None:
            return None
        return float(_predict(self.model, x[None, :])[0])

    @staticmethod
    def side(p: float | None, margin: float = MARGIN) -> str | None:
        if p is None:
            return None
        return "UP" if p >= 0.5 + margin else "DOWN" if p <= 0.5 - margin else None

    def info(self) -> dict:
        return {"rows": self.n_rows, "lam": self.lam, "val_acc": self.val_acc, "fitted_at": self.fitted_at,
                "margin": MARGIN, "ready": self.model is not None}
