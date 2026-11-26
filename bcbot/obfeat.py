"""Order-book inputs of the six markets of "our price", for the order-book models (logistic regression, LightGBM,
XGBoost, later DeepLOB).  Numpy only: the same code builds the inputs for the fit (from the recording) and for the
live call (from the last seconds), so a model sees at the decision exactly what it was fitted on.

The recording (tools/ob_server.py): one row per market every 100 ms, ROW below.  Per market, at a decision:
    spread   best ask - best bid, in units of vol60                         (not mirrored: the same for UP and DOWN)
    imb1 / imb5 / imb10   (bid size - ask size) / (bid size + ask size) over the best 1 / 5 / 10 levels
    imbw     the same over 10 levels, level k weighted 1 / (1 + k)
    press1 / press3 / press10   (bought - sold) / (bought + sold) by takers over the last 1 / 3 / 10 s (0: no trade)
    ret1 / ret3 / ret10   the market's own mid-price move over the last 1 / 3 / 10 s, in units of vol60
    dimb5    imb5 now minus imb5 3 s ago
then the same inputs mixed over the six markets by their share in our price (SHARES), and from the bot:
    gap      bc.game's price minus our price, in units of vol60 (the Gap fade's input)
    vol      log of vol60 ($)
"""
import numpy as np

LEVELS, STEP = 10, 0.1
ROW = np.dtype([("t", "f8"), ("bid", "f8"), ("ask", "f8"), ("bp", "f4", LEVELS), ("bs", "f4", LEVELS),
                ("ap", "f4", LEVELS), ("as_", "f4", LEVELS), ("buy", "f4"), ("sell", "f4"), ("nb", "f4"),
                ("ns", "f4"), ("age", "f4")])
# the markets and their share in our price (MIX in ourprice.py; a test keeps the two equal)
SHARES = {"bitfinex_perp": 0.433, "bybit_perp": 0.172, "okx_perp": 0.160, "okx": 0.114, "binance_perp": 0.068,
          "coinbase": 0.053}
PER = ("spread", "imb1", "imb5", "imb10", "imbw", "press1", "press3", "press10", "ret1", "ret3", "ret10", "dimb5")
UNSIGNED = ("spread", "vol")
MAX_LAG, MAX_AGE = 0.5, 3.0          # a row older than this at the decision, or a book silent this long: unknown
MIN_SHARE = 0.5                      # a mixed input needs markets with at least this share of our price
NAMES = tuple(f"{k}_{f}" for k in SHARES for f in PER) + tuple(f"mix_{f}" for f in PER if f not in UNSIGNED) + ("gap", "vol")
SIGN = np.array([1.0 if n.rsplit("_", 1)[-1] in UNSIGNED or n == "vol" else -1.0 for n in NAMES])   # the mirror image
OB_FITTED = ("ob_logit", "ob_lgbm", "ob_xgb", "ob_deeplob")     # fitted by tools/train_ob.py: a new fit = a new record
# HawkesSign on the six order books (bcbot/hawkes6.py) is "ob_hawkes" in the service's answer and the model "sign" in
# the bot (since 2026-11-16 the bot's HawkesSign IS the six-market one).  It is fitted again every 20 s on the last
# 300 s, so it has no fit version and its record is never started again by one.
# "ob_steady" (2026-11-17, the owner's "the order book has a lifetime"): a rule, not a fit, so like HawkesSign it
# keeps one record.  See steady_call below.
OB_ANSWER = {**{m: m for m in OB_FITTED}, "sign": "ob_hawkes", "ob_steady": "ob_steady"}   # the bot's model -> its key in the service's answer
OB_MODELS = tuple(OB_ANSWER)                                    # the models whose call comes from the service's answer
# The share of the rounds each order-book model bets in.  1 = every round, on the side its P(UP) is on: the owner's
# choice on 2026-11-16 ("I was thinking that will bet every round"; it was 0.25, the rounds a model is surest of).
# Under 1 the level follows the model's own last LEVEL_N answers (live_level).
BET_SHARE = 1.0
MIN_THR = 0.00001                    # an answer of exactly 50% (the service rounds to five places) is no bet
LEVEL_N, LEVEL_MIN = 240, 40         # under 1: the level follows the model's last 240 answers (2 h of rounds), from 40 on
_W = 1.0 / (1.0 + np.arange(LEVELS))


def _imb(bs, as_, n=None, w=None):
    b, a = (bs[:, :n], as_[:, :n]) if n else (bs, as_)
    if w is not None:
        b, a = b * w, a * w
    b, a = b.sum(axis=1), a.sum(axis=1)
    tot = b + a
    return np.where(tot > 0, (b - a) / np.where(tot > 0, tot, 1.0), np.nan)


def market_features(a: np.ndarray, t_dec, vol) -> np.ndarray:
    """One market's inputs at each decision.  a: its ROW rows in time order; t_dec: this PC's time of each decision;
    vol: vol60 in $ at each decision.  (decisions, len(PER)) with NaN where the book is not known."""
    t_dec, vol = np.atleast_1d(np.asarray(t_dec, dtype=float)), np.atleast_1d(np.asarray(vol, dtype=float))
    out = np.full((len(t_dec), len(PER)), np.nan)
    if not len(a):
        return out
    t = a["t"]
    i = np.searchsorted(t, t_dec, side="right") - 1
    ic = np.clip(i, 0, len(a) - 1)
    r = a[ic]
    ok = (i >= 0) & (t_dec - r["t"] <= MAX_LAG) & (r["age"] <= MAX_AGE) & ~np.isnan(r["bid"]) & ~np.isnan(r["ask"])
    v = np.where(vol > 0, vol, np.nan)
    mid = (r["bid"] + r["ask"]) / 2
    bs, as_ = np.nan_to_num(r["bs"].astype(float)), np.nan_to_num(r["as_"].astype(float))
    imb5 = _imb(bs, as_, 5)
    col = {"spread": (r["ask"] - r["bid"]) / v, "imb1": _imb(bs, as_, 1), "imb5": imb5, "imb10": _imb(bs, as_),
           "imbw": _imb(bs, as_, w=_W)}
    cb, cs = np.cumsum(a["buy"].astype(float)), np.cumsum(a["sell"].astype(float))

    def back(w):                                   # the row at or before (decision - w): index, and whether it is known
        j = np.searchsorted(t, t_dec - w, side="right") - 1
        return j, np.clip(j, 0, len(a) - 1)
    for w in (1, 3, 10):
        j, jc = back(w)
        b = cb[ic] - np.where(j >= 0, cb[jc], 0.0)
        s = cs[ic] - np.where(j >= 0, cs[jc], 0.0)
        col[f"press{w}"] = np.where(b + s > 0, (b - s) / np.where(b + s > 0, b + s, 1.0), 0.0)
        p = a[jc]
        known = (j >= 0) & (t_dec - w - p["t"] <= MAX_LAG) & ~np.isnan(p["bid"])
        col[f"ret{w}"] = np.where(known, (mid - (p["bid"] + p["ask"]) / 2) / v, np.nan)
        if w == 3:
            was = _imb(np.nan_to_num(p["bs"].astype(float)), np.nan_to_num(p["as_"].astype(float)), 5)
            col["dimb5"] = np.where(known, imb5 - was, np.nan)
    for c, name in enumerate(PER):
        out[:, c] = np.where(ok, col[name], np.nan)
    return out


def build(markets: dict, t_dec, gap, vol) -> np.ndarray:
    """The whole input row (NAMES) at each decision.  markets: {name: ROW rows}; gap: bc.game minus our price in $
    (None / NaN: unknown); vol: vol60 in $."""
    t_dec = np.atleast_1d(np.asarray(t_dec, dtype=float))
    vol = np.atleast_1d(np.asarray(vol, dtype=float))
    per = {k: market_features(markets.get(k, np.zeros(0, dtype=ROW)), t_dec, vol) for k in SHARES}
    return assemble(per, gap, vol)


def assemble(per: dict, gap, vol) -> np.ndarray:
    """The input rows from each market's own inputs (market_features), the gap and vol60."""
    vol = np.atleast_1d(np.asarray(vol, dtype=float))
    gap = np.atleast_1d(np.asarray(gap, dtype=float))
    n = len(vol)
    cols = [per[k] for k in SHARES]
    mixed = []
    for c, f in enumerate(PER):
        if f in UNSIGNED:
            continue
        num, den = np.zeros(n), np.zeros(n)
        for k, share in SHARES.items():
            x = per[k][:, c]
            known = ~np.isnan(x)
            num += np.where(known, share * x, 0.0)
            den += np.where(known, share, 0.0)
        mixed.append(np.where(den >= MIN_SHARE, num / np.where(den > 0, den, 1.0), np.nan))
    v = np.where(vol > 0, vol, np.nan)
    return np.column_stack(cols + [np.column_stack(mixed), gap / v, np.log(v)])


def mirror(X: np.ndarray) -> np.ndarray:
    """The same market seen upside down: every signed input changes sign, UP becomes DOWN.  (85 columns: the inputs
    of the fits before 2026-11-17; with the lifetime inputs: NAMES2.)"""
    return X * (SIGN if X.shape[1] == len(SIGN) else SIGN2)


# ---------------------------------------------------------------- the lifetime of a reading (2026-11-17)
# The owner: "the current OB has a lifetime, but we do not calc this and follow the current OB only".  Measured on
# 66,721 decisions of the recording (11-16 05:14 .. 11-17 23:19 UTC): a reading (the best-price imbalance of five
# markets by share, averaged over 1 s) is on the same side 2.6 s later, when the round's window opens, in 63%; one
# under 1 s old in 57%, one 5-10 s old in 69%, one of 20 s and more in 79%.  Following a STRONG reading (75%+ of the
# size on one side) on bc.game's result: under 2 s old 49.9% of 6,095, 2-5 s 51.9%, 5 s and older 54.1% of 9,000
# (both halves 54.1%).  Not confirmed on the bot's own rounds (53.0% of 627).  So: (1) the paper model "ob_steady"
# follows a strong reading of 5 s or more; (2) the fitted models get the reading's age as inputs (LIFE_NAMES).
LIFE_MARKETS = tuple(k for k in SHARES if k != "bitfinex_perp")    # the reading that was measured (Bitfinex perp's
#                                                                    best price is thin: spread about $12)
LIFE_AVG_S = 1.0                     # a reading: the best-price imbalance averaged over the last second
LIFE_CAP_S = 30.0                    # its age is counted up to this (the service keeps 40 s of rows in memory)
LIFE_PER = ("age1", "old1")                          # per market: side x age / 10, reading x age / 10
LIFE_MIX = ("avg1", "age1", "old1", "resp1")         # the five mixed: the reading, the same two, and the price's move
#                                                      since the reading took its side, in units of vol60
LIFE_NAMES = tuple(f"{k}_{f}" for k in SHARES for f in LIFE_PER) + tuple(f"mix5_{f}" for f in LIFE_MIX)
NAMES2 = NAMES + LIFE_NAMES          # what a fit reads from 2026-11-17 on (NAMES: the 85 of the fits before)
SIGN2 = np.concatenate([SIGN, -np.ones(len(LIFE_NAMES))])
STEADY_MIN, STEADY_AGE_S = 0.50, 5.0  # "ob_steady": 75%+ of the best-price size on one side, for 5 s or more
STEADY_P = 0.04                      # its answer is 50% +- this (the 54% measured), 50% = no bet


SLIM = np.dtype([("t", "f8"), ("bid", "f8"), ("ask", "f8"), ("bs", "f4", (1,)), ("as_", "f4", (1,)), ("age", "f4")])
MIX5 = tuple(f"mix5_{f}" for f in LIFE_MIX)          # the four that DeepLOB reads next to the gap and vol60


def slim(a: np.ndarray) -> np.ndarray:
    """What the lifetime inputs read of a market's rows (36 bytes a row instead of 204): the fit holds all the
    markets' rows at once for the mixed reading."""
    out = np.zeros(len(a), dtype=SLIM)
    for f in ("t", "bid", "ask", "age"):
        out[f] = a[f]
    out["bs"][:, 0], out["as_"][:, 0] = a["bs"][:, 0], a["as_"][:, 0]
    return out


def _best(a: np.ndarray):
    """Per row of one market: the best-price imbalance (NaN: the book is not known) and the mid price."""
    ok = ~np.isnan(a["bid"]) & ~np.isnan(a["ask"]) & (a["age"] <= MAX_AGE)
    b, s = np.nan_to_num(a["bs"][:, 0].astype(float)), np.nan_to_num(a["as_"][:, 0].astype(float))
    tot = b + s
    return (np.where(ok & (tot > 0), (b - s) / np.where(tot > 0, tot, 1.0), np.nan),
            np.where(ok, (a["bid"] + a["ask"]) / 2, np.nan))


def _life(t: np.ndarray, x: np.ndarray, dp: np.ndarray):
    """A reading and its life along a timeline.  t: the rows' times; x: the imbalance per row (NaN: unknown); dp: the
    price's move from the row before to this row in $ (0: unknown).  Per row: the reading (the mean of x over the
    last LIFE_AVG_S; NaN when under half of those rows are known), its age in s (how long it has been on its present
    side without a break, LIFE_CAP_S at most; NaN without a reading) and the price's move in $ since it took that
    side (LIFE_CAP_S back at most).  Only the last LIFE_CAP_S + LIFE_AVG_S of rows matter for a row's values, so the
    fit (the whole recording) and the live answer (the last 40 s) get the same numbers."""
    n = len(t)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    cn = np.concatenate([[0], np.cumsum(~np.isnan(x))])
    i1 = np.arange(n) + 1
    i0 = np.searchsorted(t, t - LIFE_AVG_S, side="right")
    cnt = cn[i1] - cn[i0]
    r = np.where(cnt >= max(int(round(LIFE_AVG_S / STEP)) // 2, 1), (cs[i1] - cs[i0]) / np.where(cnt > 0, cnt, 1), np.nan)
    s = np.sign(np.nan_to_num(r))
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = (s[1:] != s[:-1]) | (np.diff(t) > MAX_LAG)
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    back = np.maximum(start, np.searchsorted(t, t - LIFE_CAP_S, side="left"))
    age = np.where(s != 0, np.minimum(t - t[start], LIFE_CAP_S), np.nan)
    P = np.cumsum(dp)
    return r, age, np.where(s != 0, P - P[back], np.nan)


def _step_moves(mid: np.ndarray) -> np.ndarray:
    d = np.zeros(len(mid))
    if len(mid) > 1:
        d[1:] = mid[1:] - mid[:-1]
    return np.nan_to_num(d)


def _at(t: np.ndarray, t_dec: np.ndarray):
    """The row at or before each decision, and whether it is fresh enough."""
    i = np.searchsorted(t, t_dec, side="right") - 1
    ic = np.clip(i, 0, max(len(t) - 1, 0))
    return ic, (i >= 0) & (t_dec - t[ic] <= MAX_LAG) if len(t) else np.zeros(len(t_dec), dtype=bool)


def life_market(a: np.ndarray, t_dec) -> np.ndarray:
    """One market's own reading at each decision: columns reading, age (s), the price's move ($) since the reading
    took its side.  NaN where there is none."""
    t_dec = np.atleast_1d(np.asarray(t_dec, dtype=float))
    out = np.full((len(t_dec), 3), np.nan)
    if not len(a):
        return out
    x, mid = _best(a)
    r, age, move = _life(a["t"], x, _step_moves(mid))
    ic, ok = _at(a["t"], t_dec)
    for c, v in enumerate((r, age, move)):
        out[:, c] = np.where(ok, v[ic], np.nan)
    return out


def life_mixed(markets: dict, t_dec) -> np.ndarray:
    """The reading of the five markets mixed by their share (LIFE_MARKETS) at each decision: columns reading, age
    (s), the mixed price's move ($) since the reading took its side, and the share of the five that was known.
    The service writes one row per market at the same instant, so the markets are joined on the rows' times."""
    t_dec = np.atleast_1d(np.asarray(t_dec, dtype=float))
    out = np.full((len(t_dec), 4), np.nan)
    rows = {k: markets.get(k) for k in LIFE_MARKETS}
    have = [k for k in LIFE_MARKETS if rows[k] is not None and len(rows[k])]
    if not have:
        return out
    t = rows[max(have, key=lambda k: len(rows[k]))]["t"]
    num, den, dnum, dden = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    for k in have:
        a = rows[k]
        x, mid = _best(a)
        j = np.searchsorted(a["t"], t, side="right") - 1
        jc = np.clip(j, 0, len(a) - 1)
        same = (j >= 0) & (t - a["t"][jc] <= STEP / 2)              # this market's row of the same instant
        x, mid = np.where(same, x[jc], np.nan), np.where(same, mid[jc], np.nan)
        w, known = SHARES[k], ~np.isnan(x)
        num += np.where(known, w * x, 0.0); den += np.where(known, w, 0.0)
        d = np.full(len(t), np.nan)
        if len(t) > 1:
            d[1:] = mid[1:] - mid[:-1]
        moved = ~np.isnan(d)
        dnum += np.where(moved, w * d, 0.0); dden += np.where(moved, w, 0.0)
    total = sum(SHARES[k] for k in LIFE_MARKETS)
    x = np.where(den >= total / 2, num / np.where(den > 0, den, 1.0), np.nan)
    r, age, move = _life(t, x, np.where(dden > 0, dnum / np.where(dden > 0, dden, 1.0), 0.0))
    ic, ok = _at(t, t_dec)
    for c, v in enumerate((r, age, move, den / total)):
        out[:, c] = np.where(ok, v[ic], np.nan)
    return out


def life_inputs(markets: dict, t_dec, vol) -> np.ndarray:
    """The inputs LIFE_NAMES at each decision (all of them change sign in the mirror image)."""
    t_dec = np.atleast_1d(np.asarray(t_dec, dtype=float))
    vol = np.atleast_1d(np.asarray(vol, dtype=float))
    v = np.where(vol > 0, vol, np.nan)
    cols = []
    for k in SHARES:
        m = life_market(markets.get(k, np.zeros(0, dtype=ROW)), t_dec)
        cols += [np.sign(m[:, 0]) * m[:, 1] / 10, m[:, 0] * m[:, 1] / 10]
    m = life_mixed(markets, t_dec)
    cols += [m[:, 0], np.sign(m[:, 0]) * m[:, 1] / 10, m[:, 0] * m[:, 1] / 10, m[:, 2] / v]
    return np.column_stack(cols)


def build2(markets: dict, t_dec, gap, vol) -> np.ndarray:
    """The input row NAMES2 at each decision: build()'s 85 and then the lifetime inputs."""
    return np.column_stack([build(markets, t_dec, gap, vol), life_inputs(markets, t_dec, vol)])


def steady_call(reading, age) -> str | None:
    """The "ob_steady" model (paper, 2026-11-17): follow the five markets' best-price reading when it is strong
    (STEADY_MIN: 75% of the size on one side) and has been on that side for STEADY_AGE_S or more.  A young reading
    is gone before the round's window opens (a coin on bc.game's result); an old one was right 54.1% of 9,000
    made-up decisions and 53.0% of 627 of the bot's rounds: its forward record on the page is what counts."""
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in (reading, age)):
        return None
    if abs(reading) < STEADY_MIN or age < STEADY_AGE_S:
        return None
    return "UP" if reading > 0 else "DOWN"


def side(p, thr) -> str | None:
    """A model's call from its P(UP): UP / DOWN when it is at least `thr` away from 50%, else no call."""
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in (p, thr)):
        return None
    return "UP" if p >= 0.5 + thr else "DOWN" if p <= 0.5 - thr else None


def live_level(past, fit_thr, share: float = BET_SHARE, n_min: int = LEVEL_MIN, min_thr: float = MIN_THR):
    """How far from 50% a model's P(UP) must be for a bet.  share 1 (BET_SHARE now): every round, so only an answer of
    exactly 50% is no bet.  Under 1: past is |P(UP) - 50%| of its EARLIER answers to the bot (the fit that is in
    place, newest last); from n_min answers on, the level is the one the surest `share` of them reached, so the model
    bets in about that share of the rounds whatever the spread of its answers; before that, the level the fit gave.
    Measured 2026-11-16: the fit's level alone (taken from models fitted on parts of the data) let LightGBM and
    XGBoost bet in 5-8% of the rounds and DeepLOB in 54%, where 25% was meant."""
    if share >= 1:
        return min_thr
    if len(past) >= n_min:
        return max(float(np.quantile(np.asarray(past, dtype=float), 1 - share)), min_thr)
    return fit_thr


# ---------------------------------------------------------------- the fitted models (used by the service and the fit)
def logit_predict(m: dict, X: np.ndarray) -> np.ndarray:
    """Logistic regression on the signed inputs, no intercept (UP and DOWN are mirror images)."""
    idx = np.array(m["idx"])
    z = np.clip(np.nan_to_num(X[:, idx] / np.array(m["sd"])), -6, 6) @ np.array(m["w"])
    return 1 / (1 + np.exp(-z))


def tree_predict(predict, X: np.ndarray) -> np.ndarray:
    """A tree model's P(UP), averaged with its answer to the mirror image (so that it is symmetric)."""
    return 0.5 * (predict(X) + 1 - predict(mirror(X)))
