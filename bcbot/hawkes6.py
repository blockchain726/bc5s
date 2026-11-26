"""HawkesSign on the order books of ALL six markets of our price (the owner, 2026-11-16: "we need to consider all OB
for these platforms since bc price follows them").  The bot's own HawkesSign (predictor.py, lob.py) reads one book,
Bitfinex spot, which is not one of the six.

Per market, from its rows (tools/ob_server.py writes one every 100 ms), the paper's four event types (lob.py):
    Ec+ / Ec-   the mid-price is higher / lower than in the row before
    Eb / Es     the best prices stayed, and the sizes at them show buy / sell pressure
One Hawkes model per market (hawkes.MHP: the paper's fit, on the last 300 s), fitted again all the time.  At a
decision each market's model is run forward N times over the round's window; a run gives that market's move as
(up moves - down moves) x its usual step in $.  Run n's six moves are mixed by the markets' shares in our price
(obfeat.SHARES):  P(UP) = the share of runs in which the mixed move is up.

Numpy only (the order-book service runs it in worker processes).  The markets are simulated one by one: what one
market's move does to another market is not in the model."""
import time

import numpy as np

from .hawkes import MHP
from .lob import EB, EC_DN, EC_UP, ES
from .obfeat import MAX_AGE, MIN_SHARE, SHARES

TRAIN_S = 300.0                # the fit: the last 300 s (the paper's Table 3, as predictor.py)
STATE_S = 150.0                # the intensities at the decision: the events of the last 150 s
N_SIMS = 200
MAX_GAP = 0.5                  # two rows further apart than this are not compared (the recording stalled)
MIN_EVENTS = 40                # a market with fewer events in the 300 s gets no model
EPS = 1e-9                     # $: a smaller change of the mid-price is rounding, not a move
COLS = ("t", "bid", "ask", "bs", "as_", "age")      # what is kept of a row: time, best prices, sizes at them, book age


def events(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One market's rows in time order, (rows, len(COLS)) with NaN prices where the book is not known ->
    the events: time, type, and the size of the mid-price move in $ (0 for a pressure event)."""
    if len(c) < 2:
        return np.zeros(0), np.zeros(0, int), np.zeros(0)
    t, bid, ask, bs, as_, age = c.T
    ok = ~np.isnan(bid) & ~np.isnan(ask) & (age <= MAX_AGE)
    pair = ok[1:] & ok[:-1] & (np.diff(t) <= MAX_GAP)
    dm = np.diff((bid + ask) / 2)
    same = (np.diff(bid) == 0) & (np.diff(ask) == 0)
    dbid, dask = np.diff(bs), np.diff(as_)
    typ = np.full(len(t) - 1, -1)
    typ[pair & same & ((dbid > 0) | (dask < 0))] = EB
    typ[pair & same & ((dask > 0) | (dbid < 0))] = ES            # written second: lob.py asks for sell pressure first
    typ[pair & (dm > EPS)] = EC_UP                               # EPS: a bid down and an ask up by the same amount
    typ[pair & (dm < -EPS)] = EC_DN                              # leave the mid where it was, give or take rounding
    sel = typ >= 0
    price = (typ[sel] == EC_UP) | (typ[sel] == EC_DN)
    return t[1:][sel], typ[sel], np.where(price, np.abs(dm[sel]), 0.0)


def fit_market(c: np.ndarray, now: float) -> dict | None:
    """One market's model on its last TRAIN_S seconds of rows.  None: too few events (a silent or unknown book)."""
    t0 = time.perf_counter()
    c = c[c[:, 0] >= now - TRAIN_S - 1]
    t, typ, step = events(c)
    sel = (t >= now - TRAIN_S) & (t < now)
    if int(sel.sum()) < MIN_EVENTS:
        return None
    m = MHP.fit(t[sel], typ[sel], 4, now - TRAIN_S, now)
    moves = step[sel][step[sel] > 0]
    return {"mu": m.mu, "alpha": m.alpha, "beta": m.beta, "step": float(moves.mean()) if len(moves) else 0.0,
            "at": now, "events": int(sel.sum()), "moves": int(len(moves)), "fit_s": round(time.perf_counter() - t0, 3)}


def simulate_net(m: dict, E0: np.ndarray, a: float, b: float, n: int, rng: np.random.Generator,
                 max_steps: int = 4000) -> np.ndarray:
    """n runs of the model at once (Ogata thinning, as MHP.simulate does one run), from the excitation state E0 at
    the decision until b seconds after it.  Per run: up moves minus down moves from a seconds on."""
    mu, alpha, beta = m["mu"], m["alpha"], m["beta"]
    J = len(mu)
    E = np.broadcast_to(E0, (n, J, J)).copy()
    s = np.zeros(n)
    net = np.zeros(n)
    alive = np.ones(n, bool)
    bound = (mu + (alpha * E).sum(axis=2)).sum(axis=1)
    for _ in range(max_steps):
        if not alive.any():
            break
        w = rng.exponential(1.0, n) / bound
        s += w
        alive &= s < b
        E *= np.exp(-beta[None] * w[:, None, None])
        lam = mu + (alpha * E).sum(axis=2)
        total = lam.sum(axis=1)
        acc = alive & (rng.random(n) * bound <= total)          # the candidate is an event
        j = np.minimum((np.cumsum(lam, axis=1) < (rng.random(n) * total)[:, None]).sum(axis=1), J - 1)
        i = np.nonzero(acc)[0]
        E[i, :, j[i]] += 1.0
        net += (acc & (s >= a)) * ((j == EC_UP) * 1.0 - (j == EC_DN) * 1.0)
        bound = np.where(acc, (mu + (alpha * E).sum(axis=2)).sum(axis=1), total)   # only decays until the next event
    return net


def _p_up(x: np.ndarray) -> float:
    return float((x > 0).mean() + 0.5 * (x == 0).mean())


def predict(models: dict, cols: dict, now: float, a: float, b: float, n: int = N_SIMS, seed=None) -> dict:
    """models: {market: fit_market's result}; cols: {market: its rows of the last STATE_S seconds or more}.
    The window starts a seconds and ends b seconds after now.  -> p: P(UP) of the mixed move (None when the markets
    with a model make up less than half of our price), per: each market's own P(UP), share: the share covered."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    a, b = max(float(a), 0.0), max(float(b), 0.1)
    mixed, share, per = np.zeros(n), 0.0, {}
    for k, w in SHARES.items():
        m, c = models.get(k), cols.get(k)
        if m is None or c is None or not len(c):
            continue
        t, typ, _ = events(c[c[:, 0] >= now - STATE_S - 1])
        E0 = MHP(m["mu"], m["alpha"], m["beta"]).excitation(t, typ, now, STATE_S)
        net = simulate_net(m, E0, a, b, n, rng)
        per[k] = round(_p_up(net), 4)
        mixed += w * m["step"] * net
        share += w
    return {"p": round(_p_up(mixed), 5) if share >= MIN_SHARE else None, "per": per, "share": round(share, 3),
            "ms": round((time.perf_counter() - t0) * 1000)}
