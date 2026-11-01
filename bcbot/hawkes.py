"""Multivariate Hawkes process with exponential kernels (Raffaelli et al. 2026, Sect. 3.1).

  lambda_j(t) = mu_j + sum_i sum_{t_k^i < t} alpha_ji exp(-beta_ji (t - t_k^i))                    (eq. 5)

Fit: the least-squares contrast of Bacry et al. (2020), eqs. (6)-(7),
  LS = (1/T) sum_j int_0^T lambda_j^2 dt - (2/T) sum_j sum_{t_k^j} lambda_j(t_k^j)
plus an l1 penalty of 1e-3 on alpha, minimised by FISTA with mu, alpha >= 0, and the decays beta_ji chosen by the
greedy search of Algorithm 1 over a fixed grid.  For a fixed beta the contrast is a convex quadratic in (mu, alpha),
and it separates by target type j, so each row is solved on its own.  With exponential kernels every integral is
exact (no time grid): between two consecutive events each kernel sum decays as a plain exponential.
Prediction: Ogata (1981) thinning from the current excitation state (events of the last 150 s, Sect. 3.3).
"""
import numpy as np

BETA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)     # 1/s
L1_PENALTY = 1e-3
STATE_WINDOW = 150.0                                   # s of history used for the intensities (Sect. 3.3)


def _sufficient_stats(t: np.ndarray, typ: np.ndarray, J: int, betas: np.ndarray, T0: float, T1: float):
    """Exact integrals for every (source type i, decay beta) kernel sum g_a(t) = sum_{t_k^i < t} exp(-beta (t - t_k^i)):
    I[a] = int g_a, Q[a, b] = int g_a g_b over [T0, T1], and B[j, a] = sum over events of type j of g_a(t_k-)."""
    Z = len(betas)
    M = J * Z
    bet = np.tile(betas, J)                       # a = i * Z + z
    src = np.repeat(np.arange(J), Z)
    bsum = bet[:, None] + bet[None, :]
    G = np.zeros(M)
    I = np.zeros(M)
    Q = np.zeros((M, M))
    B = np.zeros((J, M))
    last = T0
    for tk, jk in zip(np.append(t, T1), np.append(typ, -1)):
        d = tk - last
        if d > 0:
            I += G * (1 - np.exp(-bet * d)) / bet
            Q += np.outer(G, G) * (1 - np.exp(-bsum * d)) / bsum
            G = G * np.exp(-bet * d)
            last = tk
        if jk < 0:
            break
        B[jk] += G                                # left limit: the event does not excite itself
        G = G + (src == jk)
    return I, Q, B


def _fista(A: np.ndarray, c: np.ndarray, pen: np.ndarray, iters: int = 400) -> np.ndarray:
    """min x'Ax - 2c'x + pen'x  s.t. x >= 0   (accelerated projected gradient)."""
    L = 2 * np.linalg.eigvalsh(A).max() + 1e-12
    x = y = np.maximum(np.linalg.lstsq(A + 1e-9 * np.eye(len(c)), c, rcond=None)[0], 0)
    tk = 1.0
    for _ in range(iters):
        x_new = np.maximum(y - (2 * A @ y - 2 * c + pen) / L, 0)
        t_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = x_new + (tk - 1) / t_new * (x_new - x)
        x, tk = x_new, t_new
    return x


class MHP:
    def __init__(self, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        self.mu, self.alpha, self.beta = mu, alpha, beta
        self.J = len(mu)

    @property
    def branching(self) -> np.ndarray:
        """rho_ji = alpha_ji / beta_ji (Fig. 8)."""
        return self.alpha / self.beta

    @classmethod
    def fit(cls, t: np.ndarray, typ: np.ndarray, J: int, T0: float, T1: float,
            beta_grid=BETA_GRID, pen: float = L1_PENALTY) -> "MHP":
        betas = np.asarray(beta_grid, float)
        Z, T = len(betas), T1 - T0
        sel = (t >= T0) & (t < T1)
        t, typ = t[sel], typ[sel].astype(int)
        I, Q, B = _sufficient_stats(t, typ, J, betas, T0, T1)
        counts = np.bincount(typ, minlength=J)
        mu, alpha, beta = np.zeros(J), np.zeros((J, J)), np.full((J, J), betas[0])

        for j in range(J):
            def solve(zs):
                idx = [i * Z + z for i, z in enumerate(zs)]
                A = np.empty((J + 1, J + 1))
                A[0, 0] = T
                A[0, 1:] = A[1:, 0] = I[idx]
                A[1:, 1:] = Q[np.ix_(idx, idx)]
                A /= T
                c = np.concatenate(([counts[j]], B[j, idx])) / T
                p = np.concatenate(([0.0], np.full(J, pen)))
                x = _fista(A, c, p)
                return x, float(x @ A @ x - 2 * c @ x + p @ x)

            zs = [0] * J                                  # Algorithm 1: greedy over beta_ji in the grid
            best_x, best_c = solve(zs)
            improved = True
            while improved:
                improved = False
                for i in range(J):
                    for z in range(Z):
                        if z == zs[i]:
                            continue
                        trial = zs.copy(); trial[i] = z
                        x, cost = solve(trial)
                        if cost < best_c - 1e-12:
                            best_x, best_c, zs, improved = x, cost, trial, True
            mu[j], alpha[j] = best_x[0], best_x[1:]
            beta[j] = betas[zs]
        return cls(np.maximum(mu, 1e-9), alpha, beta)

    def excitation(self, t: np.ndarray, typ: np.ndarray, now: float, window: float = STATE_WINDOW) -> np.ndarray:
        """E[j, i] = sum over events of type i in [now - window, now) of exp(-beta_ji (now - t_k))."""
        sel = (t >= now - window) & (t < now)
        t, typ = t[sel], typ[sel].astype(int)
        E = np.zeros((self.J, self.J))
        for i in range(self.J):
            ti = t[typ == i]
            if len(ti):
                E[:, i] = np.exp(-np.outer(self.beta[:, i], now - ti)).sum(axis=1)
        return E

    def intensity(self, E: np.ndarray) -> np.ndarray:
        return self.mu + (self.alpha * E).sum(axis=1)

    def simulate(self, E: np.ndarray, horizon: float, rng: np.random.Generator, stop_type: int | None = None,
                 max_events: int = 5000) -> list[tuple[float, int]]:
        """Ogata thinning from excitation state E: events (offset from now, type) up to `horizon`
        (or up to the first event of `stop_type`)."""
        E = E.copy()
        lam = self.intensity(E)
        bound, s, out = lam.sum(), 0.0, []
        while len(out) < max_events:
            w = rng.exponential(1.0 / bound)
            s += w
            if s >= horizon:
                break
            E *= np.exp(-self.beta * w)
            lam = self.intensity(E)
            total = lam.sum()
            if rng.random() * bound <= total:              # accept
                j = int(rng.choice(self.J, p=lam / total))
                out.append((s, j))
                E[:, j] += 1.0
                if stop_type is not None and j == stop_type:
                    break
                bound = self.intensity(E).sum()
            else:
                bound = total                              # intensities only decay until the next event
        return out
