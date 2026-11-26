"""Fit the order-book models (logistic regression, LightGBM, XGBoost, and DeepLOB when torch is installed:
tools/deeplob.py) on what tools/ob_server.py and record.py recorded, and put them in place for the service.
    python tools/train_ob.py [folder] [--dry] [--no-deeplob]
    python tools/train_ob.py [folder] --add-deeplob    DeepLOB alone, added to the version in place: the other three
        models, the version number and so the rows' records stay (for a fit that said "ob_deeplob NOT fitted").  It
        reads the last OB_DL_HOURS hours of the recording (default 72; 24 h took 2 GB of memory and 12 minutes).

Rows: a made-up decision every 2 s of the recording, read as the bot reads (tools/reclib.py: only what had reached
this PC), with bc.game's move over the round's window (opens 2.52 s, closes 7.55 s after the decision) as the answer.
Inputs: bcbot/obfeat.py (order-book imbalance, takers' buy / sell pressure, spread and recent moves of the six
markets of our price, each alone and mixed by its share; the gap and vol60; since 2026-11-17 the lifetime of the
best-price reading: F.LIFE_NAMES.  OB_NO_LIFE=1 fits without them, for a comparison with --dry).
Protocol: the first two thirds of the DECISIONS are for fitting, 120 s are left out, the last third is the test
(counted in decisions, not in hours on the clock: tools/fitsplit.py says why).  The
number of trees and each model's betting level (F.BET_SHARE of the decisions, the ones it is surest of; now 1: every
decision, so the test result is its plain hit rate) are chosen inside the fitting part, on 4 time blocks held out in turn (60 s left out on each side).  The test result is written next to
the models; the models put in place are then fitted on everything, with those settings.
Decisions 2 s apart share most of their 5 s result: the counts are not independent bets."""
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402

import fitsplit as S  # noqa: E402
import reclib as L  # noqa: E402
from bcbot import obfeat as F  # noqa: E402

try:
    import deeplob as DL  # noqa: E402
    DL_WHY = None
except Exception as e:                                               # noqa: BLE001 - no torch: the other three go on
    DL, DL_WHY = None, repr(e)[:160]
if "--no-deeplob" in sys.argv:
    DL, DL_WHY = None, "switched off (--no-deeplob)"
DL_HOURS = float(os.environ.get("OB_DL_HOURS") or 72.0)   # DeepLOB reads the raw books: at most the last 72 h of the
#                                                           recording (memory)
ADD_DL = "--add-deeplob" in sys.argv                      # DeepLOB alone, added to the version in place

args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args[0]) if args else ROOT / "logs" / "obrec"
DRY = "--dry" in sys.argv
MIN_HOURS, MAX_HOURS = float(os.environ.get("OB_MIN_HOURS", 0.5)), 168.0     # under half an hour there is nothing to fit
STEP, EMBARGO, FOLD_GAP = 2.0, 120.0, 60.0
BET_SHARE = F.BET_SHARE              # each model bets on this share of decisions (1 = all of them)
MIN_THR = F.MIN_THR                  # ... and never nearer to 50% than this (a model that learned nothing says 50%)
# the level written here is only the start: the service moves it with the model's live answers (F.live_level)
REC_H = float(os.environ.get("OB_RECORDED_H") or 0.0)   # hours on disk when the service started this fit
LIFE = not os.environ.get("OB_NO_LIFE")                 # with the lifetime inputs (F.NAMES2); without: the 85 (F.NAMES)
NAMES = F.NAMES2 if LIFE else F.NAMES
TREES = (25, 50, 100, 200, 400)
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.7,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=6)
XGB = dict(objective="binary:logistic", eta=0.03, max_depth=4, min_child_weight=75, subsample=0.7, colsample_bytree=0.7,
           reg_lambda=10.0, tree_method="hist", seed=7, nthread=6, verbosity=0)
t0 = time.time()


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


ADD_TO = None
if ADD_DL:                                                           # asked before an hour of reading and fitting
    try:
        ADD_TO = json.loads((OUT / "models" / "current.json").read_text())
    except (OSError, ValueError):
        say("no version in place to add DeepLOB to: fit all the models first"); sys.exit(1)
    if DL is None:
        say("DeepLOB cannot be fitted here:", DL_WHY); sys.exit(1)
    if list(ADD_TO.get("names") or ()) != list(NAMES):
        say("the version in place was fitted on another list of inputs: fit all the models again"); sys.exit(1)
    if "ob_deeplob" in (ADD_TO.get("files") or {}):
        say(f"version {ADD_TO.get('version')} has its DeepLOB already: nothing to add"); sys.exit(1)
    say(f"DeepLOB alone, for version {ADD_TO.get('version')}, on the last {DL_HOURS:g} h of the recording")


# ---------------------------------------------------------------- the decisions and their answers
D, Y, AT, VOL, GAP = [], [], [], [], []
PATHS = sorted(glob.glob(str(OUT / "rec_*.csv")))
NEED = int(DL_HOURS * 3600 / STEP) if ADD_DL else 0                  # DeepLOB alone: the newest recordings are enough
for path in (reversed(PATHS) if ADD_DL else PATHS):
    if ADD_DL and sum(len(x) for x in D) >= NEED:
        break
    try:
        R = L.load(path)
        d, tick_i, y, at = R.decisions(step=STEP, cut_lead=L.CUT_LEAD)   # the answer by bc.game's own rule
    except Exception as e:                                          # noqa: BLE001 - a recording too short to read
        say("skipped", os.path.basename(path), repr(e)[:100])
        continue
    cut = at >= at.max() - MAX_HOURS * 3600 if len(at) else at > 0
    d, tick_i, y, at = d[cut], tick_i[cut], y[cut], at[cut]
    D.append(d); Y.append(y); AT.append(at)
    VOL.append(R.vol60(tick_i)); GAP.append(R.gap(d, tick_i, L.MIX_V3))
    say(f"{os.path.basename(path)}: {(R.bc_t[-1] - R.bc_t[0]) / 3600:.2f} h, {len(d)} decisions")
    del R
if not D:
    say("no recording of bc.game's price yet (rec_*.csv)"); sys.exit(2)
D, Y, AT, VOL, GAP = (np.concatenate(x) for x in (D, Y, AT, VOL, GAP))
order = np.argsort(D)
D, Y, AT, VOL, GAP = D[order], Y[order], AT[order], VOL[order], GAP[order]
if ADD_DL:                                                           # only the decisions DeepLOB has books for
    keep = slice(max(0, len(D) - int(DL_HOURS * 3600 / STEP)), None)
    D, Y, AT, VOL, GAP = D[keep], Y[keep], AT[keep], VOL[keep], GAP[keep]

per, slim = {}, {}
BOOKS = DL.Books() if DL else None
# every cut counts recorded decisions, not hours on the clock (a pause in the recording: tools/fitsplit.py)
fit_until = S.time_at_share(D, 2 / 3) - 60                           # DeepLOB's scaling: from the fitting part only
dl_from = S.last_hours(D, DL_HOURS, STEP) - 60                       # ... and its books: the last DL_HOURS recorded
for k in F.SHARES:
    a = S.read_rows(S.hour_files(glob.glob(str(OUT / f"{k}_20*.bin")), D.min() - 60), F.ROW, D.min() - 60)
    per[k] = F.market_features(a, D, VOL)
    if LIFE:
        slim[k] = F.slim(a)
    say(f"{k}: {len(a)} rows, known at {100 * np.mean(~np.isnan(per[k][:, 1])):.1f}% of the decisions")
    if BOOKS is not None:
        BOOKS.add(k, S.rows_since(a, dl_from), until=fit_until)
    del a
X = F.assemble(per, GAP, VOL)
if LIFE:                                                             # the lifetime of the best-price reading
    X = np.column_stack([X, F.life_inputs(slim, D, VOL)])
    say(f"lifetime inputs: the five markets' reading known at {100 * np.mean(~np.isnan(X[:, NAMES.index('mix5_avg1')])):.1f}% of the decisions")
del slim
known_share = sum(s * ~np.isnan(per[k][:, 1]) for k, s in F.SHARES.items())
use = ~np.isnan(Y) & (Y != 0) & (VOL > 0) & ~np.isnan(GAP) & (known_share >= F.MIN_SHARE)
X, y, at = X[use], (Y[use] > 0).astype(float), AT[use]
D_u, GAP_u, VOL_u = D[use], GAP[use], VOL[use]
hours = len(y) * STEP / 3600
say(f"usable decisions {len(y)} = {hours:.2f} h (unchanged price: {int((Y == 0).sum())}); UP {100 * y.mean():.1f}%")
if hours < MIN_HOURS:
    say(f"not enough yet: {hours:.2f} h of {MIN_HOURS} h"); sys.exit(2)

cut = S.time_at_share(at, 2 / 3)
fit, test = at < cut - EMBARGO / 2, at > cut + EMBARGO / 2
edges = S.block_edges(at[fit], 4)
FOLDS = [(fit & ((at < edges[i] - FOLD_GAP) | (at >= edges[i + 1] + FOLD_GAP)), fit & (at >= edges[i]) & (at < edges[i + 1]))
         for i in range(4)]
SIGNED = np.flatnonzero((F.SIGN2 if LIFE else F.SIGN) < 0)


# ---------------------------------------------------------------- the three models
def fit_logit(A, t, n=None, l2=20.0):
    Z = A[:, SIGNED]
    sd = np.nanstd(Z, axis=0) + 1e-9
    Z = np.clip(np.nan_to_num(Z / sd), -6, 6)
    w = np.zeros(Z.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Z @ w))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w)), Z.T @ (p - t) + l2 * w)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return {"idx": SIGNED.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def fit_lgb(A, t, n):
    return lgb.train(LGB, lgb.Dataset(np.vstack([A, F.mirror(A)]), np.concatenate([t, 1 - t])), num_boost_round=n)


def fit_xgb(A, t, n):
    return xgb.train(XGB, xgb.DMatrix(np.vstack([A, F.mirror(A)]), np.concatenate([t, 1 - t])), num_boost_round=n)


KINDS = {
    "ob_logit": (fit_logit, lambda m, A, n=None: F.logit_predict(m, A), (None,)),
    "ob_lgbm": (fit_lgb, lambda m, A, n=None: F.tree_predict(lambda Z: m.predict(Z, num_iteration=n), A), TREES),
    "ob_xgb": (fit_xgb, lambda m, A, n=None: F.tree_predict(
        lambda Z: m.predict(xgb.DMatrix(Z), iteration_range=(0, n or m.num_boosted_rounds())), A), TREES),
}


def logloss(p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def auc(p, t):
    o = np.argsort(p, kind="stable")
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def block_range(right, bet, idx_at, block=300.0, n_boot=1000, seed=3):
    """95% range of the hit rate of the bets, resampling whole blocks of time."""
    rng = np.random.default_rng(seed)
    b = ((idx_at - idx_at.min()) // block).astype(int)
    groups = [np.flatnonzero(b == u) for u in np.unique(b)]
    out = []
    for _ in range(n_boot):
        s = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if bet[s].any():
            out.append(right[s][bet[s]].mean())
    return tuple(np.percentile(out, [2.5, 97.5])) if out else (float("nan"), float("nan"))


META = {"thr": {}, "trees": {}, "holdout": {}, "files": {}}
FINAL = {}
say(f"fitting part {int(fit.sum())} decisions, test part {int(test.sum())}; inputs {X.shape[1]}")
g = X[:, F.NAMES.index("gap")]
fade = np.abs(g) >= 1.0
fade_right = (g < 0) == (y == 1)
lo, hi = block_range(fade_right[test], fade[test], at[test])
META["holdout"]["gap_fade"] = {"bets": int((fade & test).sum()), "won": float(fade_right[fade & test].mean()), "range": [lo, hi]}
say(f"  the Gap fade on the test part: {100 * fade_right[fade & test].mean():.1f}% of {int((fade & test).sum())}  (95% {100 * lo:.1f} .. {100 * hi:.1f})")
for name, (fit_fn, pred_fn, grid) in ({} if ADD_DL else KINDS).items():
    oof = {n: np.full(len(y), np.nan) for n in grid}
    for f_m, held in FOLDS:
        m = fit_fn(X[f_m], y[f_m], max(TREES) if grid[0] else None)
        for n in grid:
            oof[n][held] = pred_fn(m, X[held], n)
    seen = ~np.isnan(oof[grid[0]])
    best = min(grid, key=lambda n: logloss(oof[n][seen], y[seen]))
    thr = max(float(np.quantile(np.abs(oof[best][seen] - 0.5), 1 - BET_SHARE)), MIN_THR)
    m = fit_fn(X[fit], y[fit], best)
    p = pred_fn(m, X[test])
    bet = np.abs(p - 0.5) >= thr
    right = (p > 0.5) == (y[test] == 1)
    lo, hi = block_range(right, bet, at[test])
    META["thr"][name], META["trees"][name] = round(thr, 5), best
    META["holdout"][name] = {"bets": int(bet.sum()), "won": float(right[bet].mean()) if bet.any() else None,
                             "range": [lo, hi], "auc": auc(p, y[test]), "auc_fit": auc(oof[best][seen], y[seen])}
    say(f"  {name:9s} trees {best}  bets when P(UP) is {100 * thr:.2f} points from 50%  |  test: {int(bet.sum())} bets, "
        f"{100 * right[bet].mean() if bet.any() else float('nan'):.1f}% right  (95% {100 * lo:.1f} .. {100 * hi:.1f}), AUC {auc(p, y[test]):.3f} "
        f"(held-out blocks of the fitting part {auc(oof[best][seen], y[seen]):.3f})")
    FINAL[name] = fit_fn(X, y, best)                                 # the model put in place: fitted on everything
DL_PAYLOAD = None
if DL is None:
    say("  ob_deeplob NOT fitted:", DL_WHY)
else:
    try:
        DL_PAYLOAD, thr, hold = DL.fit(BOOKS, D_u, y, GAP_u, VOL_u, at, fit, test, BET_SHARE, MIN_THR, say, block_range, auc,
                                       extra=X[:, [NAMES.index(n) for n in F.MIX5]] if LIFE else None)
        META["thr"]["ob_deeplob"], META["holdout"]["ob_deeplob"] = thr, hold
    except Exception as e:                                           # noqa: BLE001 - the other three are still put in place
        say("  ob_deeplob NOT fitted:", repr(e)[:300])
if ADD_DL and DL_PAYLOAD is None:
    say("nothing added"); sys.exit(1)
if DRY:
    say("dry run: nothing written"); sys.exit(0)
if ADD_DL:                                           # into the version in place: its number, and so the rows, stay
    import torch
    mdir = OUT / "models"
    meta = json.loads((mdir / "current.json").read_text())           # read again: a fit may have ended meanwhile
    if meta.get("version") != ADD_TO.get("version") or "ob_deeplob" in meta.get("files", {}):
        say("another fit was put in place meanwhile: nothing added"); sys.exit(1)
    rel = f"v{meta['version']}/deeplob.pt"
    torch.save(DL_PAYLOAD, str(mdir / rel))
    meta["files"]["ob_deeplob"] = rel
    meta["thr"]["ob_deeplob"], meta["holdout"]["ob_deeplob"] = META["thr"]["ob_deeplob"], META["holdout"]["ob_deeplob"]
    meta["deeplob_added"] = {"at": time.time(), "hours": round(hours, 2), "decisions": int(len(y))}
    tmp = mdir / "current.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, mdir / "current.json")               # the service loads it within 30 s (its file time changed)
    say(f"DeepLOB added to version {meta['version']} ({hours:.1f} h, {len(y)} decisions), {time.time() - t0:.0f} s")
    sys.exit(0)

# ---------------------------------------------------------------- put in place
mdir = OUT / "models"
mdir.mkdir(exist_ok=True)
try:
    version = int(json.loads((mdir / "current.json").read_text())["version"]) + 1
except (OSError, ValueError, KeyError):
    version = 1
vdir = mdir / f"v{version}"
vdir.mkdir(exist_ok=True)
(vdir / "logit.json").write_text(json.dumps(FINAL["ob_logit"]))
FINAL["ob_lgbm"].save_model(str(vdir / "lgbm.txt"))
FINAL["ob_xgb"].save_model(str(vdir / "xgb.json"))
files = {"ob_logit": f"v{version}/logit.json", "ob_lgbm": f"v{version}/lgbm.txt", "ob_xgb": f"v{version}/xgb.json"}
if DL_PAYLOAD is not None:
    import torch
    torch.save(DL_PAYLOAD, str(vdir / "deeplob.pt"))
    files["ob_deeplob"] = f"v{version}/deeplob.pt"
META.update(version=version, registered=time.time(), hours=round(hours, 2), decisions=int(len(y)), names=list(NAMES),
            files=files, bet_share=BET_SHARE, recorded=[float(at[0]), float(at[-1])],
            recorded_h=round(max(hours, REC_H), 3))   # what the service counts the next fit from (hours: usable only)
tmp = mdir / "current.json.tmp"
tmp.write_text(json.dumps(META, indent=1))
os.replace(tmp, mdir / "current.json")
say(f"version {version} in place ({hours:.1f} h, {len(y)} decisions), {time.time() - t0:.0f} s")
