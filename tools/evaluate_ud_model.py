"""Reproducible offline evaluation and prospective paper scoring for Up/Down.

No browser, API, credentials, bot settings, or order placement are accessed.
Run with --train once; --score-forward evaluates new rounds with the frozen model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcbot.ud_model import (FAMILIES, THRESHOLD, UDDirectionModel, feature_names,
                           fit_logistic, logits, number, sigmoid, vector)

PROTOCOL = {
    "train_fraction": 0.65, "calibration_fraction": 0.15, "test_fraction": 0.20,
    "families": list(FAMILIES), "penalties": [10.0, 100.0, 1000.0],
    "cv_folds": 3, "purge_seconds": 60.0,
    "selection": "chronological CV log loss; one standard error; prefer fewer features then stronger ridge",
    "calibration": "one nonnegative temperature, [0, 2], fitted on calibration block only",
    "primary_threshold": THRESHOLD, "bootstrap_hour_blocks": 1,
    "bootstrap_samples": 5000, "seed": 726,
    "target": "BC end_price > start_price over the recorded approximately 5-second window",
    "payout": "1 + 0.9415 * opposite_final_pool / (own_final_pool + 1), flat hypothetical $1",
    "evidence_rule": "prospective only: >=3000 selected calls, >=7 UTC dates, Wilson95 lower >51.55%, hourly-block ROI95 lower >0",
}
PAY_K = 0.9415


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, snapshot_bytes=None):
    data = Path(path).read_bytes()
    if snapshot_bytes is not None:
        if len(data) < snapshot_bytes:
            raise ValueError("log is shorter than the requested frozen snapshot")
        data = data[:snapshot_bytes]
    data = data[:data.rfind(b"\n") + 1]
    reasons = Counter()
    unique = {}
    for line in data.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            reasons["invalid_json"] += 1
            continue
        if not isinstance(r, dict) or "id" not in r:
            reasons["invalid_record"] += 1
            continue
        if r["id"] in unique:
            reasons["duplicate_id"] += 1
            continue
        unique[r["id"]] = r
    rows = []
    for r in unique.values():
        if r.get("v", 0) < 3:
            reasons["old_log_version"] += 1
            continue
        if r.get("win") not in ("UP", "DOWN") or r.get("tie"):
            reasons["unsettled_or_tie"] += 1
            continue
        f = r.get("feat")
        if vector(f, "gap") is None:
            reasons["missing_stale_or_invalid_features"] += 1
            continue
        at, start, end, cutoff = (number(x) for x in (f.get("at"), r.get("start"), r.get("end"), r.get("cutoff")))
        if None in (at, start, end, cutoff) or not 1.5 <= start - at <= 4.5 or not 4 <= end - start <= 6:
            reasons["incorrect_timing"] += 1
            continue
        sp, ep = number(r.get("start_price")), number(r.get("end_price"))
        if sp is None or ep is None or sp == ep or (ep > sp) != (r["win"] == "UP"):
            reasons["inconsistent_label"] += 1
            continue
        r["_at"], r["_end"], r["_y"] = at, end, int(r["win"] == "UP")
        rows.append(r)
    rows.sort(key=lambda r: (r["_at"], str(r["id"])))
    return rows, {"snapshot_bytes": len(data), "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                  "raw_unique_records": len(unique), "usable_rows": len(rows), "excluded": dict(reasons)}


def matrix(rows, family):
    return np.array([vector(r["feat"], family) for r in rows]), np.array([r["_y"] for r in rows], dtype=float)


def log_loss(y, p):
    p = np.clip(np.asarray(p), 1e-10, 1 - 1e-10)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def purged_before(rows, next_rows):
    return [r for r in rows if r["_end"] + PROTOCOL["purge_seconds"] < next_rows[0]["_at"]]


def choose_model(dev):
    scores = []
    n = len(dev)
    # Three nonoverlapping validation blocks in the latter half of development.
    bounds = [int(n * f) for f in (0.50, 2 / 3, 5 / 6, 1.0)]
    for family in FAMILIES:
        for penalty in PROTOCOL["penalties"]:
            losses = []
            for lo, hi in zip(bounds, bounds[1:]):
                val = dev[lo:hi]
                train = purged_before(dev[:lo], val)
                X, y = matrix(train, family)
                V, vy = matrix(val, family)
                fit = fit_logistic(X, y, penalty)
                losses.append(log_loss(vy, sigmoid(logits(fit, V))))
            scores.append({"family": family, "penalty": penalty, "features": len(feature_names(family)),
                           "fold_losses": losses, "mean_loss": float(np.mean(losses)),
                           "se": float(np.std(losses, ddof=1) / math.sqrt(len(losses)))})
    best = min(scores, key=lambda s: s["mean_loss"])
    eligible = [s for s in scores if s["mean_loss"] <= best["mean_loss"] + best["se"]]
    preference = {"gap": 0, "normalized": 1, "raw": 2, "flow": 3}
    chosen = min(eligible, key=lambda s: (s["features"], -s["penalty"], preference[s["family"]], s["mean_loss"]))
    return chosen, sorted(scores, key=lambda s: s["mean_loss"])


def calibrate(y, z):
    def loss(temperature):
        return log_loss(y, sigmoid(temperature * z)) + 0.001 * (temperature - 1) ** 2
    fit = minimize_scalar(loss, bounds=(0.0, 2.0), method="bounded",
                          options={"xatol": 1e-7})
    candidates = [0.0, 1.0, 2.0, float(fit.x)]
    return min(candidates, key=loss)


def wilson(wins, n):
    if not n:
        return None
    z = 1.96
    p, den = wins / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def payout(r, side, final=True):
    pools = r.get("pools") if final else [r["feat"].get("pool_up"), r["feat"].get("pool_dn")]
    if not isinstance(pools, (tuple, list)) or len(pools) != 2:
        return None
    up, dn = map(number, pools)
    if up is None or dn is None or up <= 0 or dn <= 0:
        return None
    own, other = (up, dn) if side else (dn, up)
    return 1.0 + PAY_K * other / (own + 1.0)


def block_interval(rows, values):
    """Resample whole UTC hour blocks, preserving dependence inside each block."""
    blocks = defaultdict(list)
    for r, value in zip(rows, values):
        if value is not None:
            blocks[int(r["_at"] // 3600)].append(float(value))
    if len(blocks) < 5:
        return {"blocks": len(blocks), "ci95": None}
    totals = np.array([sum(v) for v in blocks.values()])
    counts = np.array([len(v) for v in blocks.values()])
    rng = np.random.default_rng(PROTOCOL["seed"])
    draws = rng.integers(0, len(totals), size=(PROTOCOL["bootstrap_samples"], len(totals)))
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return {"blocks": len(blocks), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def score_sides(rows, sides):
    selected = [(r, int(s)) for r, s in zip(rows, sides) if s is not None]
    if not selected:
        return {"calls": 0, "wins": 0, "accuracy": None, "roi": None, "roi_hour_block": {"blocks": 0, "ci95": None}}
    rs, ss = zip(*selected)
    won = [int(s == r["_y"]) for r, s in selected]
    values, snapshots = [], []
    for (r, side), win in zip(selected, won):
        final, snap = payout(r, side), payout(r, side, final=False)
        values.append(None if final is None else final - 1.0 if win else -1.0)
        snapshots.append(None if snap is None else snap - 1.0 if win else -1.0)
    valid_values = [v for v in values if v is not None]
    valid_snap = [v for v in snapshots if v is not None]
    days = defaultdict(lambda: [0, 0, 0.0, 0])
    for (r, _), win, value in zip(selected, won, values):
        day = utc(r["_at"])[:10]
        days[day][0] += 1
        days[day][1] += win
        if value is not None:
            days[day][2] += value
            days[day][3] += 1
    return {"calls": len(selected), "wins": sum(won), "accuracy": float(np.mean(won)),
            "wilson95": wilson(sum(won), len(won)),
            "accuracy_hour_block": block_interval(rs, won),
            "payout_rows": len(valid_values),
            "roi": float(np.mean(valid_values)) if valid_values else None,
            "net_per_1_dollar": float(sum(valid_values)) if valid_values else None,
            "snapshot_roi": float(np.mean(valid_snap)) if valid_snap else None,
            "roi_hour_block": block_interval(rs, values),
            "by_utc_day": {day: {"calls": a, "wins": b, "net": c, "payout_rows": d}
                           for day, (a, b, c, d) in sorted(days.items())}}


def probabilities(rows, p, threshold=THRESHOLD):
    p = np.asarray(p)
    y = np.array([r["_y"] for r in rows])
    confidence = np.maximum(p, 1 - p)
    direction = (p >= 0.5).astype(int)
    selected = [int(s) if c >= threshold else None for s, c in zip(direction, confidence)]
    calibration = []
    for lo, hi in zip([0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65], [0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 1.000001]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            calibration.append({"low": lo, "high": min(hi, 1.0), "n": int(mask.sum()),
                                "mean_confidence": float(confidence[mask].mean()),
                                "accuracy": float((direction[mask] == y[mask]).mean())})
    return {"log_loss": log_loss(y, p), "brier": float(np.mean((p - y) ** 2)),
            "all_directions": score_sides(rows, direction.tolist()),
            "selected": score_sides(rows, selected), "calibration": calibration,
            "threshold": threshold}


def baseline_sides(rows, name):
    sides = []
    for r in rows:
        f = r["feat"]
        d, m, v = (f[k] for k in ("dev_tick", "bc_mom2", "vol60"))
        side = None
        if name == "fade" and abs(d) >= v and d != 0:
            side = int(d < 0)
        elif name == "overshoot" and v < 1.92 and abs(m) > 0.5 and abs(d) > 2 and d * m > 0:
            side = int(m < 0)
        elif name == "momentum":
            mom = number(f.get("our_mom5"))
            side = int(mom > 0) if mom is not None and mom != 0 else None
        elif name in ("logged_prob", "logged_gbm", "logged_ens"):
            key = {"logged_prob": "prob", "logged_gbm": "prob_gbm", "logged_ens": "prob_ens"}[name]
            p = number(r.get(key))
            if p is not None and max(p, 1 - p) >= THRESHOLD:
                side = int(p >= 0.5)
        sides.append(side)
    return sides


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def interval(value):
    return "-" if value is None else f"{100 * value[0]:.2f}% to {100 * value[1]:.2f}%"


def report_text(result, model_path):
    candidate = result["test"]["selected"]
    lines = [
        "# BTC Up/Down model evaluation",
        "",
        f"Created {result['created_utc']}. Research and paper scoring only.",
        "",
        f"Verdict: **{result['verdict']}**. A confidence threshold is a selection rule, not a validated accuracy claim.",
        "",
        "## Protocol",
        "",
        "The input log was frozen before fitting. The earliest 65% of usable rows form development; "
        "the next 15% calibrate probabilities; the latest 20% form the historical holdout, which never selects features, regularization or calibration. "
        "Three chronological development folds select the feature family and ridge strength by log loss, "
        "using the one-standard-error rule. Sixty seconds are purged before each boundary. "
        "Only decision snapshots with valid data quality, a fresh tick, 1.5-4.5 seconds to the actual window "
        "and a consistent approximately five-second settlement label qualify.",
        "",
        "This corpus has already been examined for earlier strategy research. It is a holdout for this "
        "new fit and selection procedure, not a never-seen dataset or a prospective profitability proof.",
        "",
        "Payout selection never reads future pools. Reported return is estimated afterward at the final "
        "recorded pools, including dilution by a hypothetical flat $1 stake. Missing pools are excluded "
        "from return estimates and counted separately. These are paper returns without verified fills.",
        "",
        "## Frozen data and model",
        "",
        f"- Snapshot: {result['data']['snapshot_bytes']} bytes; SHA256 {result['data']['snapshot_sha256']}.",
        f"- Usable rows: {result['data']['usable_rows']}; exclusions: {json.dumps(result['data']['excluded'])}.",
        f"- Model: {result['chosen']['family']}; ridge {result['chosen']['penalty']:g}; "
        f"temperature {result['temperature']:.4f}; fixed selected-call threshold {THRESHOLD:.2f}.",
        f"- Model artifact: {model_path.as_posix()}; SHA256 {result['model_sha256']}.",
    ]
    for name, bounds in result["splits"].items():
        lines.append(f"- {name}: {bounds['n']} rows, {bounds['first_utc']} through {bounds['last_utc']} (UTC).")
    lines += ["", "## Development comparison", "",
              "| Feature family | Best CV log loss | Ridge |",
              "|---|---:|---:|"]
    for family in FAMILIES:
        best = min((score for score in result["development_cv"] if score["family"] == family),
                   key=lambda score: score["mean_loss"])
        lines.append(f"| {family} | {best['mean_loss']:.6f} | {best['penalty']:g} |")
    lines += [
        "",
        "The one-standard-error rule chose the model before the final holdout was scored. "
        "Volatility normalization did not improve development performance in this experiment.",
        "",
        "## Historical holdout",
        "",
        "| Model / rule | Calls | Accuracy | Approx. Wilson 95% | Final-pool return per $1 | Hour-block return 95% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparisons = {"candidate, every direction": result["test"]["all_directions"],
                   "candidate, selected": candidate, **result["baselines"]}
    for name, score in comparisons.items():
        lines.append(f"| {name} | {score['calls']} | {percent(score.get('accuracy'))} | "
                     f"{interval(score.get('wilson95'))} | {percent(score.get('roi'))} | "
                     f"{interval(score.get('roi_hour_block', {}).get('ci95'))} |")
    lines += [
        "",
        f"Candidate holdout log loss: {result['test']['log_loss']:.6f}; "
        f"Brier score: {result['test']['brier']:.6f}. A constant 50% forecast has log loss "
        f"{math.log(2):.6f} and Brier score 0.25.",
        "",
        f"Selected-call coverage: {candidate['calls']}/{result['splits']['holdout']['n']}. "
        f"Final-pool payout coverage: {candidate.get('payout_rows', 0)}/{candidate['calls']}. "
        f"Decision-pool return: {percent(candidate.get('snapshot_roi'))}.",
        "",
        "Wilson intervals assume independent calls and are only approximate. Hour-block bootstrap intervals "
        "retain within-hour dependence but are also uncertain with few independent dates. Baseline rules "
        "were researched previously and are descriptive comparisons. No threshold or feature was revised "
        "after inspecting this holdout.",
        "",
        "## Confidence calibration on holdout",
        "",
        "| Predicted direction confidence | Calls | Mean predicted | Observed correct |",
        "|---|---:|---:|---:|",
    ]
    for b in result["test"]["calibration"]:
        lines.append(f"| {percent(b['low'])}-{percent(b['high'])} | {b['n']} | "
                     f"{percent(b['mean_confidence'])} | {percent(b['accuracy'])} |")
    lines += [
        "",
        "## Prospective paper test",
        "",
        "The exported model remains frozen. The training tool's --score-forward mode evaluates only "
        "decision snapshots captured at least 60 seconds after the model was frozen. It checks the "
        "recorded model-code hash before scoring. It does not place orders or modify the running bot.",
        "",
        "A positive historical result remains unproven. The prospective evidence rule requires at least "
        "3,000 selected calls across at least seven UTC dates, a Wilson 95% lower bound above 51.55%, "
        "and a positive lower bound for hourly-block return. Optional stopping, missing data, payout "
        "changes and execution still need review before treating an edge as established.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        f".venv\\Scripts\\python.exe -B tools\\evaluate_ud_model.py --train --snapshot-bytes {result['data']['snapshot_bytes']} "
        "--model models/ud_direction_replay.json --evaluation models/ud_direction_replay_evaluation.json "
        "--report docs/ud-model-replay.md",
        ".venv\\Scripts\\python.exe -B tools\\evaluate_ud_model.py --score-forward",
        "```",
        "",
        "Full development-fold scores, per-date returns and calibration statistics are in the JSON evaluation.",
        "",
    ]
    return "\n".join(lines)


def check_outputs(args):
    paths = [args.model.resolve(), args.evaluation.resolve(), args.report.resolve()]
    if len(set(paths)) != 3:
        raise ValueError("model, evaluation and report must use distinct paths")
    if not args.replace and any(path.exists() for path in paths):
        raise FileExistsError("experiment outputs already exist; use separate paths to reproduce, "
                              "or --replace to explicitly register a new experiment")


def verify_snapshot(path, metadata):
    with Path(path).open("rb") as handle:
        prefix = handle.read(metadata["snapshot_bytes"])
    if hashlib.sha256(prefix).hexdigest() != metadata["snapshot_sha256"]:
        raise ValueError("the frozen input-log prefix was changed or removed")


def train(args):
    check_outputs(args)
    rows, data = load_rows(args.log, args.snapshot_bytes)
    if len(rows) < 3000:
        raise ValueError("at least 3000 usable historical rows are required")
    n = len(rows)
    a, b = int(n * 0.65), int(n * 0.80)
    dev, calibration, test = rows[:a], rows[a:b], rows[b:]
    dev = purged_before(dev, calibration)
    calibration = purged_before(calibration, test)
    chosen, cv = choose_model(dev)
    X, y = matrix(dev, chosen["family"])
    C, cy = matrix(calibration, chosen["family"])
    fit = fit_logistic(X, y, chosen["penalty"])
    temperature = calibrate(cy, logits(fit, C))
    # All model/threshold/calibration choices above are finalized before test labels are scored.
    T, _ = matrix(test, chosen["family"])
    p = sigmoid(temperature * logits(fit, T))
    evaluated = probabilities(test, p)
    comparisons = {name: score_sides(test, baseline_sides(test, name))
                   for name in ("momentum", "fade", "overshoot", "logged_prob", "logged_gbm", "logged_ens")}
    selected = evaluated["selected"]
    roi_ci = selected.get("roi_hour_block", {}).get("ci95")
    suggestive = (selected["calls"] >= 100 and selected.get("roi", -1) is not None
                  and selected.get("roi", -1) > 0 and roi_ci is not None and roi_ci[0] > 0)
    verdict = "historically promising; prospective edge unproven" if suggestive else "no demonstrated confident edge"
    now = time.time()
    artifact = {"schema": 1, "purpose": "paper", "family": chosen["family"],
                "features": feature_names(chosen["family"]), "fit": fit,
                "temperature": temperature, "threshold": THRESHOLD,
                "validation_status": verdict, "live_ready": False,
                "created_utc": utc(now), "forward_after": max(now, rows[-1]["_end"]) + 60.0,
                "data": data, "protocol": PROTOCOL,
                "code_sha256": hash_file(ROOT / "bcbot" / "ud_model.py"),
                "evaluator_sha256": hash_file(Path(__file__))}
    UDDirectionModel(artifact)  # validate before writing
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = {"created_utc": utc(now), "verdict": verdict, "data": data, "protocol": PROTOCOL,
              "chosen": chosen, "development_cv": cv, "temperature": temperature,
              "calibration_uncalibrated_log_loss": log_loss(cy, sigmoid(logits(fit, C))),
              "calibration_calibrated_log_loss": log_loss(cy, sigmoid(temperature * logits(fit, C))),
              "splits": {name: {"n": len(rs), "first_utc": utc(rs[0]["_at"]), "last_utc": utc(rs[-1]["_at"])}
                         for name, rs in (("development", dev), ("calibration", calibration), ("holdout", test))},
              "test": evaluated, "baselines": comparisons, "model_sha256": hash_file(args.model)}
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(result, args.model), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "chosen": chosen, "temperature": temperature,
                      "splits": result["splits"], "holdout": evaluated, "baselines": comparisons,
                      "model": str(args.model), "report": str(args.report)}, indent=2))
    return result


def score_forward(args):
    model = UDDirectionModel.load(args.model)
    artifact = model.artifact
    if artifact["code_sha256"] != hash_file(ROOT / "bcbot" / "ud_model.py"):
        raise ValueError("model code changed since freezing; a new registration is required")
    if artifact["evaluator_sha256"] != hash_file(Path(__file__)):
        raise ValueError("evaluation code changed since freezing; a new registration is required")
    verify_snapshot(args.log, artifact["data"])
    rows, _ = load_rows(args.log)
    rows = [r for r in rows if r["_at"] >= artifact["forward_after"]]
    output = {"model_sha256": hash_file(args.model), "forward_after_utc": utc(artifact["forward_after"]),
              "usable_forward_rows": len(rows), "status": "waiting for forward data"}
    if rows:
        p = np.array([model.predict(r["feat"])["p_up"] for r in rows])
        output["metrics"] = probabilities(rows, p, model.threshold)
        s = output["metrics"]["selected"]
        ci = s.get("roi_hour_block", {}).get("ci95")
        passed = (s["calls"] >= 3000 and len(s.get("by_utc_day", {})) >= 7
                  and s["wilson95"][0] > 0.5155 and ci is not None and ci[0] > 0)
        output["status"] = "evidence threshold met; review required" if passed else "edge remains unproven"
    print(json.dumps(output, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--train", action="store_true")
    actions.add_argument("--score-forward", action="store_true")
    actions.add_argument("--predict-features", type=Path, help="JSON file containing one decision feature object")
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "auto_rounds.jsonl")
    parser.add_argument("--snapshot-bytes", type=int)
    parser.add_argument("--replace", action="store_true", help="explicitly replace experiment outputs and reset registration")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ud_direction_v1.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "models" / "ud_direction_v1_evaluation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "ud-model-evaluation-2026-11-11.md")
    args = parser.parse_args()
    if args.train:
        train(args)
    elif args.score_forward:
        score_forward(args)
    else:
        result = UDDirectionModel.load(args.model).predict(json.loads(args.predict_features.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
