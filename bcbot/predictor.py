"""HawkesTime / HawkesSign forecasts for one bc.game Up/Down round (Raffaelli et al. 2026, Sect. 3.3).

The paper forecasts the sign of the NEXT mid-price change.  A bc.game round is decided by the BTC price move over a
5 s window that starts when betting closes, so both of the paper's pipelines are run forward over that window:

  HawkesTime  MHP over {Ec, El+, El-} simulates (Ogata thinning) when the mid-price changes inside the window; the COE
              model gives the return R_hat(t) at each of those times with the current BI held (zero-order hold).
              Expected window return = mean over simulations of the sum of R_hat at the window's simulated changes.
              Call = its sign.  The paper's own next-change forecast (t_hat_{k+1}, sign R_hat) is reported too.
  HawkesSign  MHP over {Ec+, Ec-, Es, Eb}: P(UP) = share of simulations with more up than down changes in the window.

Training follows the paper: MHPs on the last 300 s (Table 3), refitted every round; the COE on the last 30 min
(Sect. 4.2), refitted every 5 min, starting once 5 min of data exist.
"""
import logging
import time

import numpy as np

from .coe import COE, GRID
from .hawkes import MHP
from .lob import EC, EC_DN, EC_UP, LobFeed

log = logging.getLogger(__name__)

MHP_TRAIN_S = 300
COE_TRAIN_S = 1800
COE_MIN_S = 300
COE_REFIT_S = 300
N_SIMS = 200


def _arr(q) -> tuple[np.ndarray, np.ndarray]:
    # list(deque) copies in one C call, so the feed (event-loop thread) can't change it while a worker thread reads
    a = np.array(list(q), float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


class HawkesPredictor:
    def __init__(self, feed: LobFeed, seed: int = 0):
        self.feed = feed
        self.rng = np.random.default_rng(seed)
        self.mhp_time: MHP | None = None
        self.mhp_sign: MHP | None = None
        self.coe: COE | None = None
        self.coe_at = 0.0
        self.fitted_at = 0.0
        self.info: dict = {}

    def data_span(self) -> float:
        return self.feed.now() - self.feed.ev_time[0][0] if self.feed.ev_time else 0.0

    @property
    def ready(self) -> bool:
        return self.mhp_time is not None and self.mhp_sign is not None and self.coe is not None

    def refit(self) -> dict:
        """Blocking (run it in a thread).  Returns timing / parameter info."""
        now = self.feed.now()
        span = self.data_span()
        if span < MHP_TRAIN_S:
            return {"warming_up": f"{span:.0f}/{MHP_TRAIN_S} s of order-book data"}
        t0 = time.perf_counter()
        t, ty = _arr(self.feed.ev_time)
        self.mhp_time = MHP.fit(t, ty, 3, now - MHP_TRAIN_S, now)
        ts, tys = _arr(self.feed.ev_sign)
        self.mhp_sign = MHP.fit(ts, tys, 4, now - MHP_TRAIN_S, now)
        t_mhp = time.perf_counter() - t0
        if self.coe is None or now - self.coe_at >= COE_REFIT_S:
            win = min(COE_TRAIN_S, span)
            if win >= COE_MIN_S:
                t1 = time.perf_counter()
                bt, bv = _arr(self.feed.bi)
                rt, rv = _arr(self.feed.ret)
                self.coe = COE.fit(bt, bv, rt, rv, now - win, now)
                self.coe_at = now
                self.info["coe"] = {"window_s": round(win), "a1": round(self.coe.a1, 4), "a2": round(self.coe.a2, 4),
                                    "b0": round(self.coe.b0, 4), "b1": round(self.coe.b1, 4),
                                    "gain": self.coe.gain, "rmse": round(self.coe.rmse, 4),
                                    "fit_s": round(time.perf_counter() - t1, 2)}
        self.fitted_at = now
        self.info["mhp"] = {"fit_s": round(t_mhp, 2), "events_300s": int(((t >= now - MHP_TRAIN_S)).sum()),
                            "mu_time": np.round(self.mhp_time.mu, 3).tolist(),
                            "branching_time": np.round(self.mhp_time.branching, 3).tolist()}
        return self.info

    def predict(self, a: float, b: float) -> dict:
        """Blocking.  Forecast for the window that starts `a` seconds from now and ends `b` seconds from now (the
        caller measures both on bc.game's clock; this feed's own clock is only used for its event history)."""
        if not self.ready:
            return {"ready": False}
        t_calc = time.perf_counter()
        feed = self.feed
        now = feed.now()
        lead = a
        a, b = max(a, 0.0), max(b, 0.0)

        # HawkesTime: MHP timing + COE returns
        t, ty = _arr(feed.ev_time)
        bt, bv = _arr(feed.bi)
        E = self.mhp_time.excitation(t, ty, now)
        offsets = np.arange(0.0, b + 2 * GRID, GRID)
        rhat = self.coe.forecast(bt, bv, now, offsets)
        sums, counts, first = [], [], []
        for _ in range(N_SIMS):
            ev = [s for s, j in self.mhp_time.simulate(E, b, self.rng) if j == EC]
            first.append(ev[0] if ev else np.nan)
            in_win = [s for s in ev if s >= a]
            counts.append(len(in_win))
            sums.append(float(sum(rhat[min(int(s / GRID), len(rhat) - 1)] for s in in_win)))
        exp_ret = float(np.mean(sums))
        t_next = float(np.nanmedian(first)) if not np.all(np.isnan(first)) else float("nan")
        r_next = float(rhat[min(int(t_next / GRID), len(rhat) - 1)]) if t_next == t_next else 0.0

        # HawkesSign: directional MHP
        ts, tys = _arr(feed.ev_sign)
        Es = self.mhp_sign.excitation(ts, tys, now)
        ups = net_next = 0.0
        for _ in range(N_SIMS):
            ev = self.mhp_sign.simulate(Es, b, self.rng)
            moves = [(s, j) for s, j in ev if j in (EC_UP, EC_DN)]
            if moves:
                net_next += 1.0 if moves[0][1] == EC_UP else 0.0
            n_up = sum(1 for s, j in moves if s >= a and j == EC_UP)
            n_dn = sum(1 for s, j in moves if s >= a and j == EC_DN)
            ups += 1.0 if n_up > n_dn else 0.5 if n_up == n_dn else 0.0
        return {
            "ready": True,
            "exp_ret": exp_ret,                         # HawkesTime expected window return
            "time_call": "UP" if exp_ret > 0 else "DOWN" if exp_ret < 0 else None,
            "n_changes": float(np.mean(counts)),        # expected mid-price changes in the window
            "t_next": t_next, "r_next": r_next,         # the paper's next-change forecast
            "sign_p_up": ups / N_SIMS,                  # HawkesSign P(UP over the window)
            "sign_p_next_up": net_next / N_SIMS,
            "bi": float(bv[-1]) if len(bv) else 0.0,
            "lead_s": round(lead, 3),                   # window start minus the decision time (bc.game's clock)
            "calc_ms": round((time.perf_counter() - t_calc) * 1000),
        }
