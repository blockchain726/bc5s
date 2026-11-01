"""Continuous-time Output Error model from Base Imbalance to returns (Raffaelli et al. 2026, Sect. 3.2, eq. 9):

  R(t) = B(p) / A(p) BI(t) + e(t),   A(p) = p^2 + a1 p + a2,   B(p) = b0 p + b1     (na = 2, nb = 1, Sect. 4.2)

The paper estimates it with SRIVC (MATLAB CONTSID).  Here the same output-error criterion - squared error between the
model's simulated output and the observed returns R_k at the mid-price changes - is minimised directly with
scipy.optimize.least_squares.  BI is held constant between book updates (zero-order hold, as in the paper) and the
continuous-time model is discretised exactly for that hold on a 20 ms grid, so the parameters stay continuous-time.
Stability is enforced through a1 = exp(p1), a2 = exp(p2).
"""
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import BadCoefficients, cont2discrete, lfilter

warnings.filterwarnings("ignore", category=BadCoefficients)     # b0 ~ 0 is a valid fit, not an error

GRID = 0.02            # s


def _grid_input(bi_t: np.ndarray, bi_v: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(t0, t1, GRID)
    idx = np.searchsorted(bi_t, grid, side="right") - 1
    u = np.where(idx >= 0, bi_v[np.clip(idx, 0, None)], 0.0)
    return grid, u


class COE:
    def __init__(self, a1: float, a2: float, b0: float, b1: float, scale: float, rmse: float = float("nan")):
        self.a1, self.a2, self.b0, self.b1, self.scale, self.rmse = a1, a2, b0, b1, scale, rmse

    @property
    def gain(self) -> float:
        """Steady-state gain b1/a2: the sign of the BI -> return relation (negative in Fig. 3)."""
        return self.b1 / self.a2 / self.scale

    def _filter(self):
        bd, ad, _ = cont2discrete(([self.b0, self.b1], [1.0, self.a1, self.a2]), GRID, method="zoh")
        return np.ravel(bd), np.ravel(ad)

    def simulate(self, u: np.ndarray) -> np.ndarray:
        bd, ad = self._filter()
        return lfilter(bd, ad, u) / self.scale

    @classmethod
    def fit(cls, bi_t, bi_v, r_t, r_v, t0: float, t1: float) -> "COE":
        bi_t, bi_v, r_t, r_v = map(np.asarray, (bi_t, bi_v, r_t, r_v))
        grid, u = _grid_input(bi_t, bi_v, t0, t1)
        sel = (r_t >= t0 + 5) & (r_t < t1)                 # skip the first seconds of the transient
        # output sampled just before each mid-price change: the model sees the book up to the change, not after it
        k = np.clip(np.searchsorted(grid, r_t[sel], side="left") - 1, 0, len(grid) - 1)
        scale = 1.0 / (np.std(r_v[sel]) + 1e-12)
        y = r_v[sel] * scale

        def resid(p):
            a1, a2, b0, b1 = np.exp(p[0]), np.exp(p[1]), p[2], p[3]
            bd, ad, _ = cont2discrete(([b0, b1], [1.0, a1, a2]), GRID, method="zoh")
            return lfilter(np.ravel(bd), np.ravel(ad), u)[k] - y

        slope = np.polyfit(u[k], y, 1)[0] if len(k) > 10 else -1.0
        best = None
        for a1, a2 in ((2.0, 1.0), (10.0, 25.0), (0.5, 0.1)):     # a few starts: the criterion is not convex
            r = least_squares(resid, [np.log(a1), np.log(a2), 0.0, slope * a2], method="trf", max_nfev=300)
            if best is None or r.cost < best.cost:
                best = r
        p = best.x
        return cls(np.exp(p[0]), np.exp(p[1]), p[2], p[3], scale, float(np.sqrt(2 * best.cost / max(len(y), 1))))

    def forecast(self, bi_t, bi_v, now: float, offsets: np.ndarray, history: float = 120.0) -> np.ndarray:
        """Model output R_hat at now + offsets, simulating the last `history` seconds with the observed BI and
        then holding the current BI constant (zero-order hold), as HawkesTime does with BI_k up to t_hat_{k+1}."""
        bi_t, bi_v = np.asarray(bi_t), np.asarray(bi_v)
        horizon = float(np.max(offsets)) + GRID
        grid, u = _grid_input(bi_t, bi_v, now - history, now)
        cur = u[-1] if len(u) else 0.0
        steps = int(np.ceil(horizon / GRID))
        y = self.simulate(np.concatenate([u, np.full(steps, cur)]))
        idx = len(u) - 1 + np.clip(np.ceil(np.asarray(offsets) / GRID).astype(int), 0, steps)
        return y[idx]
