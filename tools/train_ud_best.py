"""Fit, compare, export and prospectively score BC's history model.

Run with logs/research-venv/Scripts/python.exe -B tools/train_ud_best.py.
Independent artifacts only: no bot settings, running models or orders are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Leave CPU capacity for the running bot.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcbot import ud_best as B
from bcbot import obfeat as F
from bcbot.ud_model import fit_logistic, sigmoid

SEED, PURGE, SHARE = 181126, 60.0, .25
PROTOCOL = {
    "target": "actual BC end_price > start_price; one observed decision per real settled round",
    "inputs": "decision snapshot and books received by its PC timestamp; no end/start prices or final pools",
    "split": "earliest 60% book-covered rounds development, next 20% calibration, final 20% historical test",
    "cv": "three expanding past-only folds; 60 seconds purged by settlement time; trailing training block calibrates each fold",
    "selection_all": "pooled chronological-CV accuracy, then lower log loss",
    "selection_selective": "largest pooled Wilson95 lower accuracy; >=200 calls and >=10% coverage; target top 25%",
    "calibration": "logit temperature in [0.001,2], fit only before the scored block",
    "payout": "flat hypothetical $1: win 0.95 * opposite_final_pool / (own_final_pool + 1); loss -1",
    "inference": "UP/DOWN symmetric; no bias, mirrored trees; no prediction with insufficient books",
    "caveat": "history was examined previously; historical holdout is not prospective proof",
    "threads": 2, "seed": SEED,
}


def say(*args):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S UTC"), *args, flush=True)


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def freeze_rows(path, size=None):
    data = Path(path).read_bytes()
    if size is not None:
        if len(data) < size:
            raise ValueError("round log is shorter than its frozen snapshot")
        data = data[:size]
    data = data[:data.rfind(b"\n") + 1]
    rows, seen, excluded = [], set(), Counter()
    for line in data.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            excluded["invalid_json"] += 1
            continue
        if r.get("id") in seen:
            excluded["duplicate_id"] += 1
            continue
        seen.add(r.get("id"))
        f = r.get("feat")
        if r.get("win") not in ("UP", "DOWN") or r.get("tie") or not B.valid_snapshot(f):
            excluded["invalid_or_tie"] += 1
            continue
        at, start, end, sp, ep = (B.number(v) for v in (f["at"], r.get("start"), r.get("end"), r.get("start_price"), r.get("end_price")))
        if None in (at, start, end, sp, ep) or not 1.5 <= start - at <= 4.5 or not 4 <= end - start <= 6:
            excluded["invalid_timing"] += 1
            continue
        if sp == ep or (ep > sp) != (r["win"] == "UP"):
            excluded["inconsistent_label"] += 1
            continue
        r["_at"], r["_end"], r["_y"] = at, end, int(ep > sp)
        off = B.number((f.get("clock") or {}).get("offset"))
        r["_pc"] = at + off if off is not None else np.nan
        rows.append(r)
    rows.sort(key=lambda r: (r["_at"], str(r["id"])))
    return rows, {"snapshot_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "rows": len(rows), "excluded": dict(excluded)}


def read_books(folder, key, lo, hi):
    folder = Path(folder).resolve()
    parts, manifest = [], []
    for path in sorted(Path(folder).glob(key + "_20*.bin")):
        count = path.stat().st_size // F.ROW.itemsize
        if not count:
            continue
        a = np.memmap(path, dtype=F.ROW, mode="r", shape=(count,))
        if float(a["t"][-1]) < lo or float(a["t"][0]) > hi:
            del a
            continue
        i, j = np.searchsorted(a["t"], [lo, hi], side="left")
        j = min(j + 1, count)
        part = np.array(a[i:j], copy=True)
        digest = hashlib.sha256(part.tobytes()).hexdigest()
        parts.append(part)
        manifest.append({"file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "rows_read": len(part), "first_row": int(i),
                         "sha256_of_rows": digest})
        del a
    a = np.concatenate(parts) if parts else np.zeros(0, dtype=F.ROW)
    if len(a):
        order = np.argsort(a["t"], kind="stable")
        a = a[order]
        # One observation at a PC timestamp; keep the last received state.
        a = a[np.r_[np.diff(a["t"]) > 0, True]]
    return a, manifest


def dataset(rows, folder):
    C = np.array([B.context(r["feat"]) for r in rows])
    D = np.array([r["_pc"] for r in rows])
    vol = np.array([max(r["feat"]["vol60"], .25) for r in rows])
    finite = np.isfinite(D)
    per, manifests = {}, []
    lo, hi = float(D[finite].min() - 45), float(D[finite].max())
    for key in B.MARKETS:
        a, manifest = read_books(folder, key, lo, hi)
        say("reconstructing", key, len(a), "book rows")
        values = np.full((len(rows), len(B.PER)), np.nan)
        if len(a):
            values[finite] = B.market_features(a, D[finite], vol[finite])
        per[key] = values
        manifests.extend(manifest)
        del a
    X = B.assemble(C, per)
    coverage = sum(F.SHARES[k] * np.isfinite(per[k][:, B.PER.index("imb1")]) for k in B.MARKETS)
    return X, coverage >= F.MIN_SHARE, manifests


def loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def temperature(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    z = np.log(p / (1 - p))
    fn = lambda t: loss(y, sigmoid(t * z)) + .002 * (t - 1) ** 2
    opt = minimize_scalar(fn, bounds=(0., 2.), method="bounded")
    return max(.001, float(min((0., 1., 2., float(opt.x)), key=fn)))


def threshold(p, share=SHARE):
    c = np.maximum(p, 1 - p)
    return float(min(max(np.quantile(c, 1 - share), .50000001), .999999))


def wilson(k, n):
    if not n:
        return [0., 1.]
    p, z = k / n, 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [c-h, c+h]


def block_ci(rows, values):
    groups = defaultdict(lambda: [0., 0.])
    for r, v in zip(rows, values):
        if v is not None:
            g = groups[int(r["_at"] // 3600)]
            g[0] += float(v)
            g[1] += 1
    if len(groups) < 5:
        return None
    b = np.array(list(groups.values()))
    rng = np.random.default_rng(SEED)
    draws = b[rng.integers(0, len(b), (3000, len(b)))].sum(axis=1)
    return np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()


def score(rows, p, level=.5):
    p = np.asarray(p)
    choose = np.isfinite(p) & (np.abs(p - .5) > 1e-10) & (np.maximum(p, 1-p) >= level)
    ix = np.flatnonzero(choose)
    if not len(ix):
        return {"calls": 0, "coverage": 0., "accuracy": None, "wilson95": [0., 1.], "roi": None}
    selected = [rows[i] for i in ix]
    wins = [int((p[i] > .5) == bool(rows[i]["_y"])) for i in ix]
    money = []
    for i, won in zip(ix, wins):
        pools = rows[i].get("pools")
        if not isinstance(pools, list) or len(pools) != 2 or any(B.number(v) is None or v <= 0 for v in pools):
            money.append(None)
        else:
            own, other = pools if p[i] > .5 else pools[::-1]
            money.append(.95 * other / (own + 1) if won else -1.)
    known = [v for v in money if v is not None]
    days = defaultdict(lambda: [0, 0])
    for r, won in zip(selected, wins):
        d = days[utc(r["_at"])[:10]]
        d[0] += 1
        d[1] += won
    # Observation time is the number of observed rounds at the game's ~120 rounds/h.
    return {"calls": len(ix), "wins": sum(wins), "coverage": len(ix)/len(rows),
            "calls_per_observed_hour": len(ix)/len(rows)*120,
            "accuracy": float(np.mean(wins)), "wilson95": wilson(sum(wins), len(ix)),
            "accuracy_hour95": block_ci(selected, wins),
            "payout_rows": len(known), "roi": float(np.mean(known)) if known else None,
            "net_flat_1": float(sum(known)), "roi_hour95": block_ci(selected, money),
            "by_utc_day": {k: {"calls": v[0], "wins": v[1]} for k, v in sorted(days.items())}}


def candidates():
    out = []
    for group, penalties in (("gap", (10, 100)), ("context", (100, 1000)),
                             ("book", (1000,)), ("dynamic", (100, 1000)), ("all", (1000,))):
        for penalty in penalties:
            out.append({"name": f"ridge_{group}_{penalty}", "kind": "ridge", "group": group, "penalty": penalty})
    for group in ("context", "book", "dynamic", "all"):
        for rounds in (80, 240):
            out.append({"name": f"lgbm_{group}_{rounds}", "kind": "lgbm", "group": group, "rounds": rounds})
    for group in ("dynamic", "all"):
        for rounds in (80, 240):
            out.append({"name": f"xgb_{group}_{rounds}", "kind": "xgb", "group": group, "rounds": rounds})
    return out


def fit(spec, X, y, mask):
    idx = B.indices(spec["group"])
    if spec["kind"] == "ridge":
        idx = [i for i in idx if B.SIGN[i] < 0]
    A, yy = X[mask][:, idx], y[mask]
    payload = {"kind": spec["kind"], "indices": idx}
    if spec["kind"] == "ridge":
        payload["fit"] = fit_logistic(np.nan_to_num(A), yy, spec["penalty"])
    else:
        M = np.vstack([A, A * B.SIGN[idx]])
        target = np.r_[yy, 1-yy]
        if spec["kind"] == "lgbm":
            import lightgbm as lgb
            m = lgb.train(dict(objective="binary", learning_rate=.025, num_leaves=5,
                               min_data_in_leaf=120, lambda_l2=30., feature_fraction=.85,
                               bagging_fraction=.85, bagging_freq=1, verbosity=-1,
                               seed=SEED, num_threads=2, deterministic=True, force_col_wise=True),
                          lgb.Dataset(M, target), num_boost_round=spec["rounds"])
            payload["model"] = m.model_to_string()
        else:
            import xgboost as xgb
            m = xgb.train(dict(objective="binary:logistic", eta=.025, max_depth=2,
                               min_child_weight=50, reg_lambda=30., subsample=.85,
                               colsample_bytree=.85, tree_method="hist", nthread=2, seed=SEED),
                          xgb.DMatrix(M, label=target), num_boost_round=spec["rounds"])
            payload["model"] = m.save_raw(raw_format="json").decode("utf-8")
    return payload


def prior_mask(rows, boundary, book_mask, spec):
    out = np.array([r["_end"] + PURGE < boundary for r in rows])
    return out if spec["group"] in ("gap", "context") else out & book_mask


def fold_indices(rows, book_indices):
    edges = [int(len(book_indices)*f) for f in (.40, .60, .80, 1.)]
    for lo, hi in zip(edges, edges[1:]):
        validation = book_indices[lo:hi]
        prior = book_indices[:lo]
        cal_start = max(int(len(prior)*.8), 1)
        cal = prior[cal_start:]
        cal = cal[np.array([rows[i]["_end"] + PURGE < rows[validation[0]]["_at"] for i in cal])]
        yield cal, validation


def cross_validate(rows, X, book_mask, dev_indices, specs):
    y = np.array([r["_y"] for r in rows], dtype=float)
    output = []
    for spec in specs:
        val_rows, all_p, selected_p, folds = [], [], [], []
        for cal, val in fold_indices(rows, dev_indices):
            mask = prior_mask(rows, rows[cal[0]]["_at"], book_mask, spec)
            payload = fit(spec, X, y, mask)
            cp = B.raw_predict(payload, X[cal])
            temp = temperature(y[cal], cp)
            vp = B.calibrated(payload, X[val], temp)
            level = threshold(B.calibrated(payload, X[cal], temp))
            selected = np.where(np.maximum(vp, 1-vp) >= level, vp, np.nan)
            rr = [rows[i] for i in val]
            folds.append({"train_rows": int(mask.sum()), "train_last_end": max(rows[i]["_end"] for i in np.flatnonzero(mask)),
                          "calibration_first_at": rows[cal[0]]["_at"],
                          "calibration_last_end": rows[cal[-1]]["_end"], "validation_first_at": rows[val[0]]["_at"],
                          "all": score(rr, vp), "selected": score(rr, selected), "temperature": temp, "threshold": level})
            val_rows.extend(rr)
            all_p.extend(vp.tolist())
            selected_p.extend(selected.tolist())
        result = {"spec": spec, "all": score(val_rows, all_p), "selected": score(val_rows, selected_p),
                  "log_loss": loss(np.array([r["_y"] for r in val_rows]), np.asarray(all_p)), "folds": folds}
        output.append(result)
        say("CV", spec["name"], "all", round(100*result["all"]["accuracy"], 2) if result["all"]["accuracy"] is not None else None,
            "selected", round(100*result["selected"]["accuracy"], 2) if result["selected"]["accuracy"] is not None else None,
            "calls", result["selected"]["calls"], "coverage", round(result["selected"]["coverage"], 3))
    return output


def choose(cv, policy):
    eligible = cv
    if policy == "selective":
        eligible = [r for r in cv if r["selected"]["calls"] >= 200 and .10 <= r["selected"]["coverage"] <= .60]
        if not eligible:
            raise ValueError("no candidate meets the predeclared selective coverage requirement")
        return max(eligible, key=lambda r: (r["selected"]["wilson95"][0], -r["log_loss"]))
    return max((r for r in eligible if r["all"]["accuracy"] is not None and r["all"]["coverage"] >= .95), key=lambda r: (r["all"]["accuracy"], -r["log_loss"]))


def make_policy(spec, rows, X, book_mask, cal, all_mode):
    y = np.array([r["_y"] for r in rows], dtype=float)
    payload = fit(spec, X, y, prior_mask(rows, rows[cal[0]]["_at"], book_mask, spec))
    cp = B.raw_predict(payload, X[cal])
    temp = temperature(y[cal], cp)
    level = .5 if all_mode else threshold(B.calibrated(payload, X[cal], temp))
    return {"candidate": spec["name"], "group": spec["group"], "model": payload,
            "temperature": temp, "threshold": level, "target_share": 1. if all_mode else SHARE,
            "trained_before": rows[cal[0]]["_at"] - PURGE,
            "calibrated_through": rows[cal[-1]]["_end"]}


def clean_models(value):
    if isinstance(value, dict):
        return {k: clean_models(v) for k, v in value.items() if k != "_model"}
    if isinstance(value, list):
        return [clean_models(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_models(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def report(result):
    pct = lambda v: "-" if v is None else f"{100*v:.2f}%"
    lines = ["# History-trained BC five-second model", "", f"Created {result['created_utc']}.",
             "", "Models are selected by chronological development validation, before historical test labels are scored.",
             "This history has been researched before. The test below is historical, not a prospective profitability claim.",
             "", "## Data", "", f"- {result['data']['rows']} usable real rounds; {result['book_rows']} with recorded books.",
             f"- {len(B.NAMES)} causal features: logged context, six individual books, two market mixtures and liquidity changes.",
             "- One decision snapshot per real round; actual BC start/end settlement labels.",
             "- Final pools are used only to estimate return, never as model features.",
             "- The three extra books have too little earlier training history and are excluded from this experiment.",
             "", "## Selected policies", "",
             "| Policy | Chosen candidate | CV win rate | Test calls | Test coverage | Test win rate | Test paper ROI |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("all", "selective"):
        test, cv = result["test"][name], result["chosen"][name]
        lines.append(f"| {name} | {cv['spec']['name']} | {pct(cv[name]['accuracy'] if name == 'all' else cv['selected']['accuracy'])} | {test['calls']} | {pct(test['coverage'])} | {pct(test['accuracy'])} | {pct(test['roi'])} |")
    lines += ["", "## Same-period baselines", "",
              "| Baseline | Calls | Win rate | Paper ROI |", "|---|---:|---:|---:|"]
    for name, s in result["baselines"].items():
        lines.append(f"| {name} | {s['calls']} | {pct(s['accuracy'])} | {pct(s['roi'])} |")
    lines += ["", "## Validation limits", "",
              "The entire final test block was excluded from candidate selection, calibration and thresholds. "
              "Tree count and features were chosen in three expanding past-only folds with 60-second embargoes based on settlement time. "
              "At each fold, a later part of past data calibrated probabilities and selected the confidence threshold.",
              "Calibration and thresholds can change prospective coverage. Counts are actual rounds, not overlapping synthetic samples.",
              "Paper returns include a hypothetical $1 added to the chosen final pool and the 5% fee; fills are unverified.",
              "Hourly-block intervals and daily breakdowns are in evaluation.json. No 60% claim is made unless supported by future outcomes.",
              "", "## Artifacts and use", "",
              "evaluated.json contains the exact models scored on the test. model.json refits the chosen settings on "
              "the frozen history preceding the final calibration block, then calibrates on that final block. "
              "Its prospective record starts 60 seconds after export; historical test scores are for evaluated.json.",
              "", "Load with bcbot.ud_best.BestUDModel.load. predict(snapshot, books, decision_pc, policy='selective') "
              "returns a direction, probability and selected flag. policy='all' supplies the every-round model.",
              "", "Prospective scoring:", "", "~~~powershell",
              f"logs\\research-venv\\Scripts\\python.exe -B tools\\train_ud_best.py --score-forward --out {result['output_relative']}",
              "~~~", "", "This experiment does not change the running bot or place orders.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "logs/auto_rounds.jsonl")
    parser.add_argument("--books", type=Path, default=ROOT / "logs/obrec")
    parser.add_argument("--out", type=Path, default=ROOT / "models/ud_best_20261118")
    parser.add_argument("--score-forward", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an incomplete frozen experiment from its saved dataset")
    args = parser.parse_args()
    if args.score_forward:
        artifact = json.loads((args.out / "model.json").read_text(encoding="utf-8"))
        if artifact["code_sha256"] != B.code_hash():
            raise ValueError("inference code changed since model export")
        _, meta = freeze_rows(args.log, artifact["data"]["snapshot_bytes"])
        if meta["sha256"] != artifact["data"]["sha256"]:
            raise ValueError("frozen training history changed")
        rows, _ = freeze_rows(args.log)
        rows = [r for r in rows if r["_at"] >= artifact["forward_after"]]
        if not rows:
            say("No prospective rounds yet; record begins", utc(artifact["forward_after"]))
            return
        X, have, _ = dataset(rows, args.books)
        model = B.BestUDModel(artifact)
        result = {}
        for name, policy in artifact["policies"].items():
            p = model.predict_matrix(X, name)
            if policy["group"] not in ("gap", "context"):
                p[~have] = np.nan
            result[name] = score(rows, p, policy["threshold"])
        print(json.dumps(result, indent=2))
        return
    if args.out.exists() and not args.resume:
        raise FileExistsError("choose a new output directory; frozen experiments are never overwritten")
    if args.resume:
        if (args.out / "model.json").exists():
            raise FileExistsError("completed frozen experiments cannot be resumed or overwritten")
        saved = json.loads((args.out / "data.json").read_text(encoding="utf-8"))
        rows, metadata = freeze_rows(args.log, saved["data"]["snapshot_bytes"])
        if metadata != saved["data"] or [r["id"] for r in rows] != saved["ids"]:
            raise ValueError("frozen history differs from saved dataset")
        with np.load(args.out / "dataset.npz") as cached:
            X, have = cached["X"], cached["have"]
        if X.shape != (len(rows), len(B.NAMES)) or have.shape != (len(rows),):
            raise ValueError("cached dataset dimensions differ")
        manifest = saved["manifest"]
        say("resuming frozen dataset", len(rows), "rounds")
    else:
        args.out.mkdir(parents=True)
        rows, metadata = freeze_rows(args.log)
        say("frozen", metadata["rows"], "real rounds", metadata["snapshot_bytes"], "bytes")
        X, have, manifest = dataset(rows, args.books)
        np.savez_compressed(args.out / "dataset.npz", X=X, have=have)
        write_json(args.out / "data.json", {"data": metadata, "manifest": manifest, "ids": [r["id"] for r in rows]})
    indices = np.flatnonzero(have)
    if len(indices) < 3000:
        raise ValueError("at least 3000 real book-covered rounds required")
    a, b = int(len(indices)*.6), int(len(indices)*.8)
    dev, cal, test = indices[:a], indices[a:b], indices[b:]
    dev = dev[np.array([rows[i]["_end"] + PURGE < rows[cal[0]]["_at"] for i in dev])]
    cal = cal[np.array([rows[i]["_end"] + PURGE < rows[test[0]]["_at"] for i in cal])]
    say("split development/calibration/test", len(dev), len(cal), len(test), "features", X.shape[1])
    write_json(args.out / "protocol.json", {**PROTOCOL, "data": metadata,
               "test_first_at": rows[test[0]]["_at"], "test_last_at": rows[test[-1]]["_at"]})
    if args.resume and (args.out / "cv.json").exists():
        cv = json.loads((args.out / "cv.json").read_text(encoding="utf-8"))
        if [c["spec"] for c in cv] != candidates():
            raise ValueError("candidate definitions changed since frozen validation")
        say("using frozen chronological CV results")
    else:
        cv = cross_validate(rows, X, have, dev, candidates())
        write_json(args.out / "cv.json", cv)
    chosen = {name: choose(cv, name) for name in ("all", "selective")}
    # Decisions above are frozen before any test probabilities or labels are scored.
    policies = {name: make_policy(c["spec"], rows, X, have, cal, name == "all") for name, c in chosen.items()}
    evaluated = {"schema": B.SCHEMA, "purpose": "paper", "features": list(B.NAMES),
                 "policies": policies, "data": metadata, "code_sha256": B.code_hash(), "protocol": PROTOCOL}
    write_json(args.out / "evaluated.json", evaluated)
    model = B.BestUDModel(evaluated)
    test_rows = [rows[i] for i in test]
    scores = {name: score(test_rows, model.predict_matrix(X[test], name), p["threshold"]) for name, p in policies.items()}
    baselines = {"always_up": score(test_rows, np.full(len(test), .500000001))}
    for name in ("ob_logit", "ob_lgbm", "ob_xgb", "ob_deeplob", "ob_hawkes", "ob_steady"):
        p = np.array([((r.get("ob") or {}).get("p") or {}).get(name, np.nan) for r in test_rows], dtype=float)
        baselines[name] = score(test_rows, p)
    for name in ("gap_from_2x", "gap_strict"):
        calls = []
        for r in test_rows:
            f = r["feat"]
            d, v = f["dev_tick"], f["vol60"]
            required = 2*v if name == "gap_from_2x" else 2.45*max(v, 1.25)
            calls.append(("DOWN" if d > 0 else "UP") if d != 0 and abs(d) >= required else None)
        p = np.array([.75 if c == "UP" else .25 if c == "DOWN" else np.nan for c in calls])
        baselines[name] = score(test_rows, p)
    now = time.time()
    # Refit settings chosen above; keep the final 20% as past-only calibration.
    production_cal = indices[b:]
    production_policies = {name: make_policy(c["spec"], rows, X, have, production_cal, name == "all") for name, c in chosen.items()}
    production = {**evaluated, "policies": production_policies, "created_utc": utc(now),
                  "forward_after": max(now, rows[-1]["_end"]) + 60,
                  "historical_evaluation": "evaluated.json; production parameters were refitted after evaluation"}
    write_json(args.out / "model.json", production)
    result = {"created_utc": utc(now), "protocol": PROTOCOL, "data": metadata, "book_rows": int(have.sum()),
              "chosen": chosen, "test": scores, "baselines": baselines,
              "splits": {name: {"n": len(ix), "first_utc": utc(rows[ix[0]]["_at"]), "last_utc": utc(rows[ix[-1]]["_at"])}
                         for name, ix in (("development", dev), ("calibration", cal), ("test", test))},
              "output_relative": str(args.out.relative_to(ROOT)), "code_sha256": B.code_hash(),
              "forward_after": production["forward_after"]}
    write_json(args.out / "evaluation.json", result)
    (args.out / "report.md").write_text(report(result), encoding="utf-8")
    for name, s in scores.items():
        say("HISTORICAL TEST", name, policies[name]["candidate"], json.dumps(s))
    say("EXPORTED", args.out)


if __name__ == "__main__":
    main()
