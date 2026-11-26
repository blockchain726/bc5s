"""DeepLOB on the order books of the six markets of our price (after Zhang, Zohren & Roberts 2019, "DeepLOB: Deep
convolutional neural networks for limit order books"), for the bot's "ob_deeplob" model.  Needs torch.

Input at a decision: the last T = 100 rows (10 s, one every 100 ms) of each market's book as recorded by
tools/ob_server.py: per level the ask's distance from the mid (in 1/10,000 of the price) and its size, the bid's
distance and its size (10 levels = 40 numbers a row, the six markets as six channels); per row and market the mid's
distance from its value at the decision (in units of vol60) and what takers bought and sold; and the bot's gap and
vol60.  The network: three convolution blocks (distance with size, ask with bid, the ten levels), an inception block
over time, an LSTM, and one output: P(UP) for bc.game's price over the round's window.
UP and DOWN are mirror images (asks <-> bids, bought <-> sold, moves and the gap change sign): half of each training
batch is mirrored, and the answer is the average of the book's and its mirror image's.
Fit (fit()): the same split as the other order-book models; inside the fitting part the last fifth (60 s left out)
says after how many passes to stop and gives the betting level; the model put in place is then fitted on everything
for that many passes."""
import time

import numpy as np
import torch
from torch import nn

from bcbot import obfeat as F

T = 100                               # rows of 100 ms
LOB = 4 * F.LEVELS
MARKETS = tuple(F.SHARES)
MAX_EPOCHS, BATCH = 8, 256
SWAP = np.arange(LOB).reshape(F.LEVELS, 4)[:, [2, 3, 0, 1]].reshape(-1)       # asks <-> bids


def raw(a: np.ndarray):
    """A market's rows as the network reads them, before scaling: (rows, 40) book, (rows, 2) bought / sold, mid, known."""
    mid = (a["bid"] + a["ask"]) / 2
    half = (a["ask"] - a["bid"]) / 2
    scale = (mid * 1e-4)[:, None]
    lob = np.empty((len(a), LOB), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        lob[:, 0::4] = (half[:, None] + a["ap"]) / scale
        lob[:, 1::4] = np.log1p(a["as_"])
        lob[:, 2::4] = (half[:, None] + a["bp"]) / scale
        lob[:, 3::4] = np.log1p(a["bs"])
        trades = np.column_stack([np.log1p(a["buy"]), np.log1p(a["sell"])]).astype(np.float32)
    known = ~np.isnan(mid) & (a["age"] <= F.MAX_AGE)
    return lob, trades, mid, known


def norm_of(lob, trades, known) -> dict:
    """Scaling from the known rows: asks and bids share one mean and spread per level, so the mirror image is exact."""
    x = lob[known]
    d = np.concatenate([x[:, 0::4], x[:, 2::4]])
    s = np.concatenate([x[:, 1::4], x[:, 3::4]])
    t = trades[known]
    f = lambda v: [float(z) for z in v]
    return {"d_mu": f(np.nanmean(d, axis=0)), "d_sd": f(np.nanstd(d, axis=0) + 1e-6), "s_mu": f(np.nanmean(s, axis=0)),
            "s_sd": f(np.nanstd(s, axis=0) + 1e-6), "t_mu": float(t.mean()), "t_sd": float(t.std() + 1e-6)}


def scaled(lob, trades, n: dict):
    out = np.empty_like(lob)
    for j, (mu, sd) in ((0, ("d_mu", "d_sd")), (1, ("s_mu", "s_sd")), (2, ("d_mu", "d_sd")), (3, ("s_mu", "s_sd"))):
        out[:, j::4] = (lob[:, j::4] - np.array(n[mu], dtype=np.float32)) / np.array(n[sd], dtype=np.float32)
    return np.clip(np.nan_to_num(out), -8, 8), np.clip((trades - n["t_mu"]) / n["t_sd"], -8, 8).astype(np.float32)


class Books:
    """The six markets' rows, scaled, and the windows cut from them."""

    def __init__(self):
        self.t, self.lob, self.trades, self.mid, self.known, self.norm = {}, {}, {}, {}, {}, {}
        self.side = None            # more inputs next to the gap and vol60, one row per decision (signed: they change
        #                             sign in the mirror image): the lifetime of the best-price reading, F.MIX5

    def add(self, k: str, a: np.ndarray, norm: dict | None = None, until: float | None = None) -> None:
        lob, trades, mid, known = raw(a)
        if norm is None:                                 # scaling from the rows up to `until` (the fitting part)
            sel = known & (a["t"] <= until) if until else known
            norm = norm_of(lob, trades, sel if sel.any() else known)
        self.norm[k] = norm
        self.lob[k], self.trades[k] = scaled(lob, trades, norm)
        self.t[k], self.mid[k], self.known[k] = a["t"].copy(), mid, known

    def rows(self, D) -> tuple:
        """For decisions at PC times D: each market's row index, and which decisions have a full fresh window."""
        D = np.atleast_1d(np.asarray(D, dtype=float))
        idx, ok = {}, np.ones(len(D), dtype=bool)
        for k in MARKETS:
            t = self.t[k]
            i = np.searchsorted(t, D, side="right") - 1
            ic = np.clip(i, T - 1, max(len(t) - 1, T - 1))
            if len(t) < T:
                return {k: np.zeros(len(D), dtype=int) for k in MARKETS}, np.zeros(len(D), dtype=bool)
            ok &= (i >= T - 1) & (D - t[ic] <= F.MAX_LAG) & self.known[k][ic] & (t[ic] - t[ic - T + 1] <= (T - 1) * F.STEP + 2.0)
            idx[k] = ic
        return idx, ok

    def batch(self, idx: dict, sel, gap, vol, vol_mu: float):
        """Windows for the decisions `sel`: X (n, markets, T, 40), E (n, T, 3 x markets), S (n, 2)."""
        off = np.arange(-T + 1, 1)
        v = np.asarray(vol, dtype=float)[sel]
        X = np.empty((len(sel), len(MARKETS), T, LOB), dtype=np.float32)
        E = np.empty((len(sel), T, 3 * len(MARKETS)), dtype=np.float32)
        for c, k in enumerate(MARKETS):
            r = idx[k][sel][:, None] + off
            X[:, c] = self.lob[k][r]
            m = self.mid[k][r]
            E[:, :, 3 * c] = np.clip(np.nan_to_num((m - m[:, -1:]) / v[:, None]), -10, 10)
            E[:, :, 3 * c + 1:3 * c + 3] = self.trades[k][r]
        g = np.clip(np.nan_to_num(np.asarray(gap, dtype=float)[sel] / v), -6, 6)
        S = np.column_stack([g, np.log(v) - vol_mu]).astype(np.float32)
        if self.side is not None:
            S = np.column_stack([S, np.clip(np.nan_to_num(np.asarray(self.side, dtype=float)[sel]), -6, 6)]).astype(np.float32)
        return X, E, S


def mirror(X, E, S):
    """The same books upside down (torch tensors)."""
    E2 = E.clone()
    E2[:, :, 0::3] = -E[:, :, 0::3]
    E2[:, :, 1::3], E2[:, :, 2::3] = E[:, :, 2::3], E[:, :, 1::3]
    S2 = S.clone()
    S2[:, 0] = -S[:, 0]
    S2[:, 2:] = -S[:, 2:]                                # the lifetime inputs (column 1 is vol60: the same both ways)
    return X[..., torch.as_tensor(SWAP, device=X.device)], E2, S2


class Net(nn.Module):
    def __init__(self, markets: int = len(MARKETS), ch: int = 16, hidden: int = 32, n_side: int = 2):
        super().__init__()

        def block(cin, kernel, stride):
            return nn.Sequential(nn.Conv2d(cin, ch, kernel, stride=stride), nn.LeakyReLU(0.01), nn.BatchNorm2d(ch),
                                 nn.Conv2d(ch, ch, (4, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(ch))

        def branch(k):
            return nn.Sequential(nn.Conv2d(ch, ch, 1), nn.LeakyReLU(0.01),
                                 nn.Conv2d(ch, ch, (k, 1), padding="same"), nn.LeakyReLU(0.01), nn.BatchNorm2d(ch))
        self.c1 = block(markets, (1, 2), (1, 2))            # a level's distance with its size
        self.c2 = block(ch, (1, 2), (1, 2))                 # the ask with the bid
        self.c3 = block(ch, (1, F.LEVELS), 1)               # the ten levels
        self.i3, self.i5 = branch(3), branch(5)
        self.ip = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), nn.Conv2d(ch, ch, 1), nn.LeakyReLU(0.01),
                                nn.BatchNorm2d(ch))
        self.lstm = nn.LSTM(3 * ch + 3 * markets, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(hidden + n_side, 1)
        # The gap and vol60 go straight into this layer.  Their two weights start at 0, so that what the model does
        # with the gap is what the fit taught it.  With torch's random start (+0.118 for the gap under this file's
        # seed) fits 2-4 kept +0.04 .. +0.11 after their 27-110 steps: DeepLOB bet WITH the gap in every big-gap
        # round (415 of 415, 2026-11-16), which nothing in its data said (gap against result: -0.005).
        # The same for the lifetime inputs that come after them (n_side 6 from 2026-11-17).
        with torch.no_grad():
            self.out.weight[:, -n_side:] = 0.0

    def forward(self, x, e, s):
        h = self.c3(self.c2(self.c1(x)))                                              # (n, ch, T, 1)
        h = torch.cat([self.i3(h), self.i5(h), self.ip(h)], dim=1).squeeze(3).transpose(1, 2)      # (n, T, 3 ch)
        o, _ = self.lstm(torch.cat([h, e], dim=2))
        return self.out(torch.cat([self.drop(o[:, -1]), s], dim=1)).squeeze(1)


def _tensors(parts, device):
    return [torch.as_tensor(p, device=device) for p in parts]


@torch.no_grad()
def p_up(model, books: Books, idx, sel, gap, vol, vol_mu, device) -> np.ndarray:
    """P(UP) for the decisions `sel`: the average of the book's answer and its mirror image's."""
    model.eval()
    out = []
    for i in range(0, len(sel), 1024):
        X, E, S = _tensors(books.batch(idx, sel[i:i + 1024], gap, vol, vol_mu), device)
        a = torch.sigmoid(model(X, E, S))
        b = torch.sigmoid(model(*mirror(X, E, S)))
        out.append((0.5 * (a + 1 - b)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _train(books, idx, rows, y, gap, vol, vol_mu, epochs, device, seed, after_epoch=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(n_side=_n_side(books)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(rows)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 8:
                continue
            X, E, S = _tensors(books.batch(idx, b, gap, vol, vol_mu), device)
            t = torch.as_tensor(y[b], dtype=torch.float32, device=device)
            flip = torch.rand(len(b), device=device) < 0.5              # half the batch upside down
            Xm, Em, Sm = mirror(X, E, S)
            X, E, S = torch.where(flip[:, None, None, None], Xm, X), torch.where(flip[:, None, None], Em, E), torch.where(flip[:, None], Sm, S)
            t = torch.where(flip, 1 - t, t)
            opt.zero_grad()
            loss_fn(model(X, E, S), t).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if after_epoch:
            after_epoch(ep + 1, model)
    return model


def _n_side(books) -> int:
    return 2 + (0 if books.side is None else np.asarray(books.side).shape[1])


def _logloss(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def fit(books: Books, D, y, gap, vol, at, fit_m, test_m, bet_share, min_thr, say, block_range, auc, extra=None):
    """Fit as described above.  D, y, gap ($), vol ($), at: one entry per usable decision of the other models;
    fit_m / test_m: their masks; extra: more signed inputs per decision, read next to the gap and vol60 (the
    lifetime inputs F.MIX5).  Returns (what to save, betting level, hold-out result)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    books.side = extra
    idx, ok = books.rows(D)
    fit_rows, test_rows = np.flatnonzero(fit_m & ok), np.flatnonzero(test_m & ok)
    if len(fit_rows) < 100 or not len(test_rows):       # said plainly (fit 6: an IndexError, and an empty row)
        raise ValueError(f"too few decisions with their books: {len(fit_rows)} in the fitting part, {len(test_rows)} "
                         "in the test part")
    vol_mu = float(np.mean(np.log(vol[fit_m & ok])))
    cut = at[fit_rows][int(len(fit_rows) * 0.8)]        # the last fifth of the DECISIONS (a pause in the recording
    #                                                     must not move it: tools/fitsplit.py)
    tr, va = fit_rows[at[fit_rows] < cut - 30], fit_rows[at[fit_rows] > cut + 30]
    say(f"  ob_deeplob on {device}: {len(tr)} decisions to learn from, {len(va)} to choose the stop, {len(test_rows)} test; "
        f"windows of {T} rows x {len(MARKETS)} markets x {LOB} numbers")
    best = {"loss": 9.0, "epoch": 1, "state": None, "p": None}
    t0 = time.time()

    def look(ep, model):
        p = p_up(model, books, idx, va, gap, vol, vol_mu, device)
        loss = _logloss(p, y[va])
        if loss < best["loss"]:
            best.update(loss=loss, epoch=ep, p=p, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        say(f"    pass {ep}: held-out log loss {loss:.4f} (a coin 0.6931)  {time.time() - t0:.0f} s")
    _train(books, idx, tr, y, gap, vol, vol_mu, MAX_EPOCHS, device, 7, look)
    thr = max(float(np.quantile(np.abs(best["p"] - 0.5), 1 - bet_share)), min_thr)
    model = Net(n_side=_n_side(books)).to(device)
    model.load_state_dict(best["state"])
    p = p_up(model, books, idx, test_rows, gap, vol, vol_mu, device)
    bet = np.abs(p - 0.5) >= thr
    right = (p > 0.5) == (y[test_rows] == 1)
    lo, hi = block_range(right, bet, at[test_rows])
    hold = {"bets": int(bet.sum()), "won": float(right[bet].mean()) if bet.any() else None, "range": [lo, hi],
            "auc": auc(p, y[test_rows]), "epochs": best["epoch"]}
    say(f"  ob_deeplob passes {best['epoch']}  bets when P(UP) is {100 * thr:.2f} points from 50%  |  test: {int(bet.sum())} bets, "
        f"{100 * right[bet].mean() if bet.any() else float('nan'):.1f}% right  (95% {100 * lo:.1f} .. {100 * hi:.1f}), AUC {hold['auc']:.3f}")
    final = _train(books, idx, np.flatnonzero(ok), y, gap, vol, vol_mu, best["epoch"], device, 11)
    payload = {"state": {k: v.cpu() for k, v in final.state_dict().items()}, "norm": books.norm, "vol_mu": vol_mu, "T": T,
               "markets": list(MARKETS), "n_side": _n_side(books)}
    return payload, round(thr, 5), hold


class Predictor:
    """The fitted network for the live answer (CPU)."""

    def __init__(self, path: str):
        d = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(d["markets"]) != MARKETS or d["T"] != T:
            raise ValueError("DeepLOB was fitted on other markets or another window than this service builds")
        self.norm, self.vol_mu = d["norm"], d["vol_mu"]
        self.n_side = int(d.get("n_side", 2))             # 2: a fit from before the lifetime inputs
        self.model = Net(n_side=self.n_side)
        self.model.load_state_dict(d["state"])
        self.model.eval()
        torch.set_num_threads(2)

    def __call__(self, recent: dict, now: float, gap: float, vol: float, extra=None):
        """P(UP) from each market's last rows (ROW arrays, oldest first), or None when a book is not known.
        extra: the lifetime inputs of this decision (1 x 4), read only by a fit that has them."""
        if not (vol == vol and vol > 0):
            return None
        books = Books()
        for k in MARKETS:
            a = recent.get(k)
            if a is None or len(a) < T:
                return None
            books.add(k, a[-T - 5:], self.norm[k])
        idx, ok = books.rows([now])
        if not ok[0]:
            return None
        if self.n_side > 2:
            books.side = np.zeros((1, self.n_side - 2)) if extra is None else np.asarray(extra, dtype=float).reshape(1, -1)
        p = p_up(self.model, books, idx, np.array([0]), [gap if gap == gap else 0.0], [vol], self.vol_mu, "cpu")
        return float(p[0])
