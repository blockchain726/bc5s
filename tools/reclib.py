"""The exchanges as THE BOT reads them, rebuilt from a recording made by record.py: for fitting the new price.

lib.py places every exchange row at its exchange time, whether or not it had reached this PC at the decision
(Hyperliquid's rows arrive ~0.45 s after their own time, Binance perp's ~0.34 s, a bc.game tick ~0.15-0.25 s after
its own), so at a small delay it sees more than the bot has.  This loader keeps this PC's arrival times and reads a
price exactly as bcbot/ourprice.py does at a decision made at PC time D:

    price of exchange k for bc.game server time t, delay lag
        = the last row that arrived at or before  min(D, t - lag + off_k)      (None when older than max_age)
    off_k = this PC's clock minus the exchange's clock: the feed's own measurement (O rows), or, for a feed whose
            messages carry no time, the bot's clock estimate (C rows).  The bot's new price uses the same.
    level of k = the median of (price_k at each bc.game tick - bc.game) over the last `window` s of ticks
    value = sum of w_k x (price_k - level_k) / sum of w_k        (None when under 75% of the weight has a price)
    gap   = bc.game's last tick - value at that tick's time      (the bot's dev_tick / dev_fair)

    import sys; sys.path.insert(0, r"<this folder>"); import lib2 as L
    R = L.load(path)                       # one rec_*.csv (or a frozen copy of its first part)
    R.bc_t, R.bc_p                         # bc.game ticks: server time, price
    R.names                                # the exchanges recorded
    x = R.parts(D, tick_t, bc, lags)       # per exchange: bc - (price_k - level_k), NaN where unknown; see parts()
    R.gap(D, tick_t, bc, mix)              # the gap for a recipe {exchange: (weight, delay)}
    R.decisions(step)                      # made-up decision instants on the recording (for more samples than rounds)
    L.rounds(path_to_auto_rounds_jsonl, R) # the bot's real rounds inside the recording, with their logged snapshots

check():  the bot's own dev_tick, logged with every round, against gap(...) with the bot's present recipe.
"""
import json
import os
from array import array

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIX_V3 = {"bitfinex_perp": (0.433, 0.15), "bybit_perp": (0.172, 0.15), "okx_perp": (0.160, 0.15),
          "okx": (0.114, 0.15), "binance_perp": (0.068, 0.15), "coinbase": (0.053, 0.15)}
STALE, STALE_SNAPSHOT, MIN_WEIGHT, GAP_WINDOW = 120.0, 5.0, 0.75, 60.0
CUT_LEAD, WINDOW_S = 2.93, 5.0       # the bot decides this long before the betting cutoff (median of 26,000 logged
#                                      rounds: 2.93-2.94 s); bc.game's window is the 5 s after the cutoff


def _step(table, t):
    """The last value of a (time, value) table at or before each t (the first value before the table starts)."""
    i = np.clip(np.searchsorted(table[:, 0], t, side="right") - 1, 0, len(table) - 1)
    return table[i, 1]


class Rec:
    def __init__(self):
        self.ex, self.off = {}, {}
        self.bc_t = self.bc_p = self.clock = self.bc_off = None

    @property
    def names(self):
        return sorted(self.ex)

    def off_at(self, k, D):
        """This PC's clock minus exchange k's, as known at PC time D."""
        table = self.off.get(k)
        if table is None or not len(table):
            table = self.clock
        return _step(table, np.asarray(D, dtype=float)) if len(table) else np.zeros_like(np.asarray(D, dtype=float))

    def price(self, k, t, lag, D, max_age=STALE_SNAPSHOT):
        """Exchange k's price for server time(s) t and delay lag, as the bot knows it at PC time(s) D."""
        ts, vs = self.ex[k]
        t, D = np.asarray(t, dtype=float), np.asarray(D, dtype=float)
        target = t - lag + self.off_at(k, D)
        i = np.searchsorted(ts, np.minimum(target, D), side="right") - 1
        ok = (i >= 0) & (target - ts[np.clip(i, 0, len(ts) - 1)] <= max_age)
        return np.where(ok, vs[np.clip(i, 0, len(vs) - 1)], np.nan)

    def last_tick(self, D):
        """Index of the last bc.game tick that had reached this PC at PC time D (-1: none)."""
        D = np.asarray(D, dtype=float)
        return np.searchsorted(self.bc_t, D - _step(self.bc_off, D), side="right") - 1

    def level(self, k, lag, D, tick_i, window=GAP_WINDOW, min_ticks=1):
        """Exchange k's level against bc.game at each decision: the median, as the bot takes it (the upper of two
        middle values), of price_k - bc.game over the ticks of the last `window` s up to tick tick_i."""
        D, tick_i = np.atleast_1d(np.asarray(D, dtype=float)), np.atleast_1d(np.asarray(tick_i))
        out = np.full(len(D), np.nan)
        lo = np.searchsorted(self.bc_t, self.bc_t[np.clip(tick_i, 0, None)] - window, side="left")
        for j in range(len(D)):
            if tick_i[j] < 0:
                continue
            sl = slice(lo[j], tick_i[j] + 1)
            d = self.price(k, self.bc_t[sl], lag, D[j], STALE) - self.bc_p[sl]
            d = np.sort(d[~np.isnan(d)])
            if len(d) >= min_ticks:
                out[j] = d[len(d) // 2]
        return out

    def parts(self, D, tick_i, lags, window=GAP_WINDOW, names=None):
        """For each exchange k (and its delay lags[k]): bc.game's last tick minus (price_k - level_k) at that tick's
        time, i.e. the gap a recipe of that exchange alone would show.  {k: array}, NaN where the bot has no price.
        A recipe's gap is the weighted mean of these over the exchanges that have one (gap())."""
        D, tick_i = np.atleast_1d(np.asarray(D, dtype=float)), np.atleast_1d(np.asarray(tick_i))
        t, bc = self.bc_t[np.clip(tick_i, 0, None)], self.bc_p[np.clip(tick_i, 0, None)]
        out = {}
        for k in (names or self.names):
            lag = lags[k] if isinstance(lags, dict) else lags
            v = self.price(k, t, lag, D) - self.level(k, lag, D, tick_i, window)
            out[k] = np.where(tick_i >= 0, bc - v, np.nan)
        return out

    def gap(self, D, tick_i, mix, window=GAP_WINDOW, parts=None):
        """bc.game's gap from the price a recipe {exchange: (weight, delay)} gives, at each decision (NaN: no value)."""
        parts = parts or self.parts(D, tick_i, {k: lag for k, (w, lag) in mix.items()}, window, list(mix))
        total = sum(w for w, _ in mix.values())
        num = den = 0.0
        for k, (w, lag) in mix.items():
            ok = ~np.isnan(parts[k])
            num = num + np.where(ok, w * parts[k], 0.0)
            den = den + np.where(ok, w, 0.0)
        return np.where(den >= MIN_WEIGHT * total, num / np.where(den > 0, den, 1.0), np.nan)

    def vol60(self, tick_i):
        """The bot's vol60 at each decision: the spread of bc.game's 0.5 s log moves over the last minute, in $."""
        tick_i = np.atleast_1d(np.asarray(tick_i))
        out = np.full(len(tick_i), np.nan)
        for j, i in enumerate(tick_i):
            if i < 20:
                continue
            now = self.bc_t[i]                      # (the bot's "now" is a little after the tick; the grid moves with it)
            g = np.searchsorted(self.bc_t, now - 60 + 0.5 * np.arange(121), side="right") - 1
            p = np.where(g >= 0, self.bc_p[np.clip(g, 0, None)], np.nan)
            r = np.log(p[1:] / p[:-1])
            r = r[~np.isnan(r)]
            if len(r) > 20:
                out[j] = float(np.std(r)) * self.bc_p[i]
        return out

    def decisions(self, step=1.0, lead=(2.52, 7.55), tick_age=0.27, margin=75.0, cut_lead=None):
        """Made-up decisions every `step` s of the recording, shaped like the bot's: read `tick_age` s after a bc.game
        tick.  Returns D (PC time), tick_i, bc.game's move over the round's window (NaN where a tick is missing) and
        the decision's time on bc.game's clock.

        The window, with cut_lead (CUT_LEAD: what the fits use since 2026-11-25): bc.game's own rule.  Its first
        price is the last tick at or before the betting cutoff, its last price the last tick at or before cutoff + 5 s
        (7,250 logged rounds against the recorded ticks: 99.9%, and both prices are ticks of the feed the bot reads in
        100.00%); the cutoff is cut_lead s after the decision.

        Without cut_lead (the fits before, and the frozen forward test in logs/obrec/fwd): the last tick at or
        before decision + lead[0] and + lead[1].  Those two numbers are the median TIMES of the real first and last
        tick, but bc_move() takes the last tick AT OR BEFORE a time, so the first price was one tick (0.53 s) early in
        32-75% of the decisions and the answer differed from bc.game's rule in 6-13% of them.  Measured on the same
        decisions: the bet against a gap of 2 x vol60 wins 0.1-0.6 points more by that earlier window (it is a
        little nearer to the decision), a bet in every round the same."""
        last = lead[1] if cut_lead is None else cut_lead + WINDOW_S
        t0, t1 = self.bc_t[0] + margin, self.bc_t[-1] - last - 1
        at = np.arange(t0, t1, step)                                   # server time of the decision
        tick_i = np.searchsorted(self.bc_t, at - tick_age, side="right") - 1
        at = self.bc_t[tick_i] + tick_age
        D = at + _step(self.bc_off, at)
        move = self.bc_move(at + lead[0], at + lead[1]) if cut_lead is None else self.bc_move(at + cut_lead, at + last)
        fresh = at - self.bc_t[tick_i] <= 1.0
        return D[fresh], tick_i[fresh], move[fresh], at[fresh]

    def bc_move(self, t0, t1, max_age=1.5):
        """bc.game's price change between server times t0 and t1 (the last tick at or before each)."""
        def at(t):
            i = np.searchsorted(self.bc_t, t, side="right") - 1
            ok = (i >= 0) & (t - self.bc_t[np.clip(i, 0, None)] <= max_age)
            return np.where(ok, self.bc_p[np.clip(i, 0, None)], np.nan)
        return at(np.asarray(t1, dtype=float)) - at(np.asarray(t0, dtype=float))


def load(path, until=None):
    """Read a recording (rows after PC / server time `until` are left out)."""
    M, O, B, C = {}, {}, [], []
    with open(path, encoding="ascii", errors="ignore") as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            try:
                if p[0] == "M" and len(p) == 4:
                    a, b = float(p[2]), float(p[3])
                    m = M.get(p[1])
                    if m is None:
                        m = M[p[1]] = (array("d"), array("d"))
                    m[0].append(a); m[1].append(b)
                elif p[0] == "O" and len(p) == 4:
                    O.setdefault(p[1], []).append((float(p[2]), float(p[3])))
                elif p[0] == "B" and len(p) == 3:
                    B.append((float(p[1]), float(p[2])))
                elif p[0] == "C" and len(p) == 5 and p[2] not in ("None", "") and p[3] not in ("None", ""):
                    if float(p[2]) != 0.0:                 # 0.0: the bot had just started and did not know its clock yet
                        C.append((float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                continue                                   # a line cut off at the end of the file
    R = Rec()
    B = np.array(sorted(set(B)))
    C = np.array(C)
    if until:
        B, C = B[B[:, 0] <= until], C[C[:, 0] <= until]
    R.bc_t, R.bc_p = B[:, 0].copy(), B[:, 1].copy()
    R.clock, R.bc_off = C[:, [0, 1]].copy(), C[:, [0, 2]].copy()
    for k, (mt, mv) in M.items():
        a = np.column_stack([np.frombuffer(mt, dtype=np.float64), np.frombuffer(mv, dtype=np.float64)])
        a = a[np.argsort(a[:, 0], kind="stable")]
        if until:
            a = a[a[:, 0] <= until]
        R.ex[k] = (a[:, 0].copy(), a[:, 1].copy())
        if k in O:
            o = np.array(O[k])
            R.off[k] = o[o[:, 0] <= until] if until else o
    return R


def rounds(log_path, R, snapshots=("feat", "feat_mid", "feat_late")):
    """The bot's real rounds inside the recording: one row per round and snapshot that has a logged gap.
    Columns (arrays): rid, snap (0 / 1 / 2), D (PC time of the snapshot), tick_i (its bc.game tick in R), bc, dev_tick
    (the bot's own gap from our price), vol60, lead (s to the cutoff), move (end price - start price), up (1 / 0),
    tie, cutoff."""
    t_lo, t_hi = R.bc_t[0] + 75, R.bc_t[-1]
    rows = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"win"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not (t_lo <= (r.get("cutoff") or 0) and (r.get("end") or 0) <= t_hi) or r.get("win") not in ("UP", "DOWN"):
                continue
            vol = (r.get("feat") or {}).get("vol60")
            for s, name in enumerate(snapshots):
                f = r.get(name) or {}
                if f.get("dev_tick") is None or f.get("tick_t") is None or not (f.get("clock") or {}).get("offset"):
                    continue
                i = int(np.searchsorted(R.bc_t, f["tick_t"] + 0.0006, side="right") - 1)
                if i < 0 or abs(R.bc_t[i] - f["tick_t"]) > 0.002:
                    continue                                # the tick is not in the recording (a gap in it)
                rows.append((r["id"], s, f["at"] + f["clock"]["offset"], i, f["bc"], f["dev_tick"],
                             f.get("vol60") or vol or np.nan, f.get("lead_s") or np.nan,
                             (r.get("end_price") or np.nan) - (r.get("start_price") or np.nan),
                             1.0 if r["win"] == "UP" else 0.0, 1.0 if r.get("tie") else 0.0, r["cutoff"]))
    a = np.array(rows, dtype=float)
    keys = ("rid", "snap", "D", "tick_i", "bc", "dev_tick", "vol60", "lead", "move", "up", "tie", "cutoff")
    out = {k: a[:, j] for j, k in enumerate(keys)}
    out["tick_i"] = out["tick_i"].astype(int)
    out["snap"] = out["snap"].astype(int)
    return out


def check(path, log_path):
    R = load(path)
    span = R.bc_t[-1] - R.bc_t[0]
    print(f"recording: {span / 3600:.2f} h, {len(R.bc_t)} bc.game ticks, {len(R.names)} exchanges")
    for k in R.names:
        t = R.ex[k][0]
        o = R.off.get(k)
        print(f"  {k:14s} {len(t) / max(1.0, t[-1] - t[0]):5.1f} rows/s   clock offset "
              + (f"{np.median(o[:, 1]):+.3f} (own)" if o is not None and len(o) else f"{np.median(R.clock[:, 1]):+.3f} (the bot's)"))
    print(f"  bc.game tick delay to this PC: median {np.median(R.bc_off[:, 1]):+.3f} s")
    q = rounds(log_path, R)
    mine = R.gap(q["D"], q["tick_i"], MIX_V3)
    ok = ~np.isnan(mine)
    d = mine[ok] - q["dev_tick"][ok]
    print(f"the bot's logged gap against the gap rebuilt here (recipe v3): {ok.sum()} snapshots of "
          f"{len(set(q['rid']))} rounds, {(~ok).sum()} without a rebuilt value")
    print(f"  difference: median {np.median(d):+.3f}, median |d| {np.median(np.abs(d)):.3f}, "
          f"90% within {np.percentile(np.abs(d), 90):.3f}, 99% within {np.percentile(np.abs(d), 99):.3f} $; "
          f"correlation {np.corrcoef(mine[ok], q['dev_tick'][ok])[0, 1]:.4f}")
    for s, name in enumerate(("decision (2.9 s)", "mid (2.2 s)", "late (1.35 s)")):
        m = ok & (q["snap"] == s)
        if not m.any():
            continue
        vol = q["vol60"][m]
        a, b = q["dev_tick"][m], mine[m]
        ca, cb = np.abs(a) >= vol, np.abs(b) >= vol
        same = (ca == cb) & (~ca | (np.sign(a) == np.sign(b)))
        print(f"  {name:17s} n {m.sum():5d}  median |d| {np.median(np.abs(b - a)):.3f}  Gap fade calls: the bot {ca.sum()}, "
              f"rebuilt {cb.sum()}, the same decision in {100 * same.mean():.1f}% of snapshots")
    return R, q


if __name__ == "__main__":
    import sys
    check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else r"D:\Workspace\BC BTC\logs\auto_rounds.jsonl")
