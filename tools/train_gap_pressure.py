"""Discover and export the new Gap Pressure probability-table model.

Uses the earlier frozen feature dataset for all model selection. Later real
rounds, after that snapshot and an embargo, are evaluated only after selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcbot import gap_pressure as G
from bcbot import ud_best as B
from tools.train_ud_best import freeze_rows, dataset, score, temperature, utc

PURGE, SHARE = 60., .20
PROTOCOL = {
    "algorithm": "hierarchical probability tables: gap prior, signed changing queue pressure, price response",
    "candidates": "3 pressure recipes x 2 shrinkage strengths x 2 hierarchy depths x 2 time decays",
    "selection": "highest pooled chronological CV Wilson95 lower win rate; >=200 calls; coverage 8%-50%",
    "cv": "3 expanding past-only folds; settlement+60 seconds before calibration and validation",
    "call_policy": "confidence level from past calibration block, targeting 20% of recorded book-covered rounds",
    "test": "real rounds appended after the earlier frozen log, with a 60-second settlement embargo",
    "caveat": "historical corpus has been examined before; this is not a never-seen or prospective proof",
    "payout": "flat $1 added to final chosen pool, 5% fee; final pools score return only",
}


def say(*args):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S UTC"), *args, flush=True)


def specs():
    return [{"recipe": recipe, "strength": strength, "depth": depth, "half_life_h": half_life}
            for recipe in ("queue", "turn", "blend") for strength in (50., 150.)
            for depth in (2, 3) for half_life in (0., 72.)]


def training_mask(rows, boundary):
    return np.array([r["_end"] + PURGE < boundary for r in rows])


def calibrate(fitted, X, y, indices):
    raw = G.raw_predict(fitted, X[indices])
    known = np.isfinite(raw)
    if known.sum() < 100:
        raise ValueError("at least 100 past calibration observations required")
    temp = temperature(y[indices][known], raw[known])
    p = G.calibrated(fitted, X[indices][known], temp)
    level = min(max(float(np.quantile(np.maximum(p, 1-p), 1-SHARE)), .50000001), .999999)
    return temp, level


def folds(rows, have):
    books = np.flatnonzero(have)
    edges = [int(len(books)*f) for f in (.5, 2/3, 5/6, 1.)]
    for lo, hi in zip(edges, edges[1:]):
        val = books[lo:hi]
        prior = books[:lo]
        cal = prior[int(len(prior)*.8):]
        cal = cal[np.array([rows[i]["_end"] + PURGE < rows[val[0]]["_at"] for i in cal])]
        yield training_mask(rows, rows[cal[0]]["_at"]), cal, val


def validate(rows, X, have):
    y = np.array([r["_y"] for r in rows], dtype=float)
    at = np.array([r["_at"] for r in rows])
    output = []
    for spec in specs():
        rr, predictions, selected, per_fold = [], [], [], []
        for train, cal, val in folds(rows, have):
            fitted = G.fit(X, y, at, train, have, spec)
            temp, level = calibrate(fitted, X, y, cal)
            p = G.calibrated(fitted, X[val], temp)
            chosen = np.where(np.maximum(p, 1-p) >= level, p, np.nan)
            rs = [rows[i] for i in val]
            per_fold.append({"train_last_end": max(rows[i]["_end"] for i in np.flatnonzero(train)),
                             "cal_first_at": rows[cal[0]]["_at"], "cal_last_end": rows[cal[-1]]["_end"],
                             "val_first_at": rows[val[0]]["_at"], "temperature": temp,
                             "threshold": level, "all": score(rs, p), "selected": score(rs, chosen)})
            rr.extend(rs); predictions.extend(p.tolist()); selected.extend(chosen.tolist())
        result = {"spec": spec, "all": score(rr, predictions), "selected": score(rr, selected), "folds": per_fold}
        output.append(result)
        s = result["selected"]
        say("CV", json.dumps(spec), "wins", s.get("wins",0), "/", s["calls"],
            "win%", round(100*s["accuracy"], 2) if s["accuracy"] is not None else None)
    return output


def choose(cv):
    eligible = [r for r in cv if r["selected"]["calls"] >= 200 and .08 <= r["selected"]["coverage"] <= .50]
    if not eligible:
        raise ValueError("no setting met the predeclared evidence and coverage requirements")
    return max(eligible, key=lambda r:(r["selected"]["wilson95"][0], r["selected"]["accuracy"]))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def boundaries(rows, have):
    books = np.flatnonzero(have)
    cal = books[int(len(books)*.8):]
    return training_mask(rows, rows[cal[0]]["_at"]), cal


def percent(x):
    return "-" if x is None else f"{100*x:.2f}%"


def report(result):
    s, a = result["test"]["selected"], result["test"]["all"]
    chosen = result["chosen"]
    lines = ["# Gap Pressure: a new BC five-second model", "",
             "This model learns hierarchical conditional probability tables. It does not use XGBoost or LightGBM.",
             "", "## Decision", "",
             "1. Estimate the chance that BC's price gap reverses, conditional on its size.",
             "2. Update that chance using the direction and change of liquidity at the best bid and ask.",
             "3. Where selected, update it again from the price's response to that pressure.",
             "4. Apply probability calibration and the confidence threshold learned on an earlier calibration block.",
             "",
             "A gap can be overridden: strong opposing pressure can make the learned probability favour continuation. "
             "The model can abstain, and missing book data never becomes an invented reading.",
             "", "## Learned settings", "", f"- {json.dumps(chosen['spec'])}",
             f"- Chronological CV: {chosen['selected']['wins']}/{chosen['selected']['calls']} selected calls, "
             f"{percent(chosen['selected']['accuracy'])} correct.",
             f"- Target selection share: {percent(SHARE)}; future realized frequency can differ.",
             "", "## Later historical test", "",
             f"Period: {result['test_period']['first_utc']} through {result['test_period']['last_utc']}.",
             f"These rounds were appended after the frozen training snapshot, and did not choose settings or thresholds. "
             "The broader history has been examined previously, so this is a historical test, not prospective proof.",
             "", "| Policy | Wins / calls | Win rate | Approximate 95% range | Paper ROI |",
             "|---|---:|---:|---:|---:|"]
    for name, value in (("selected",s),("all probabilities",a),("gap-prior ablation",result["test"]["gap_prior"])):
        bounds=value.get("wilson95",[0,1])
        lines.append(f"| {name} | {value.get('wins',0)}/{value['calls']} | {percent(value['accuracy'])} | "
                     f"{percent(bounds[0])} to {percent(bounds[1])} | {percent(value['roi'])} |")
    lines += ["", f"Selected coverage: {percent(s['coverage'])}. Counts are actual BC rounds, not overlapping synthetic windows.",
              "The gap-prior ablation removes both order-book updates; it has its own calibration and selection threshold "
              "from the same past calibration observations. It is diagnostic, not a replacement chosen after this test.",
              "Hourly-block intervals, all CV settings, frozen-data hashes, and per-day counts are saved in JSON.",
              "", "## Inference", "",
              "Load bcbot.gap_pressure.GapPressureModel.load(model.json), then call "
              "predict(decision_snapshot, six_market_book_rows, decision_pc). The result includes P(UP), side, selected and confidence.",
              "The exported model is exactly the one evaluated above. It has not been refitted on the later test.",
              "", "~~~powershell",
              "logs\\research-venv\\Scripts\\python.exe -B tools\\train_gap_pressure.py --score-forward",
              "~~~", "", "No live bot settings or orders are changed. A 60% future win rate is not established.", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"models/ud_best_20261118")
    parser.add_argument("--out",type=Path,default=ROOT/"models/gap_pressure_20261118")
    parser.add_argument("--log",type=Path,default=ROOT/"logs/auto_rounds.jsonl")
    parser.add_argument("--books",type=Path,default=ROOT/"logs/obrec")
    parser.add_argument("--score-forward",action="store_true")
    args=parser.parse_args()
    if args.score_forward:
        a=json.loads((args.out/"model.json").read_text(encoding="utf-8"))
        if a["source_sha256"]!=G.source_hash() or a["feature_sha256"]!=B.code_hash():
            raise ValueError("model or feature code changed after export")
        _,meta=freeze_rows(args.log,a["training_data"]["snapshot_bytes"])
        if meta["sha256"]!=a["training_data"]["sha256"]:
            raise ValueError("frozen training history was changed")
        rows,_=freeze_rows(args.log)
        rows=[r for r in rows if r["_at"]>=a["forward_after"]]
        if not rows:
            say("No prospective rounds yet; starts",utc(a["forward_after"]))
            return
        X,have,_=dataset(rows,args.books)
        p=G.GapPressureModel(a).predict_matrix(X)
        p[~have]=np.nan
        print(json.dumps({"selected":score(rows,p,a["threshold"]),"all":score(rows,p)},indent=2))
        return
    if args.out.exists():
        raise FileExistsError("choose a new output directory; frozen models are never overwritten")
    saved=json.loads((args.source/"data.json").read_text(encoding="utf-8"))
    rows,meta=freeze_rows(args.log,saved["data"]["snapshot_bytes"])
    if meta!=saved["data"] or [r["id"] for r in rows]!=saved["ids"]:
        raise ValueError("frozen source log or feature row order changed")
    with np.load(args.source/"dataset.npz") as z:
        X,have=z["X"],z["have"]
    if X.shape!=(len(rows),len(B.NAMES)):
        raise ValueError("feature schema differs from frozen dataset")
    # Freeze the later test period before any fitting or search, without scoring its labels.
    current,current_meta=freeze_rows(args.log)
    later=[r for r in current if r["_at"]>rows[-1]["_end"]+PURGE]
    if len(later)<300:
        raise ValueError("at least 300 later real rounds required for the historical test")
    args.out.mkdir(parents=True)
    write(args.out/"protocol.json",{**PROTOCOL,"training_data":meta,"test_data":current_meta,
          "test_first_at":later[0]["_at"],"test_last_at":later[-1]["_at"]})
    say("OLD FROZEN DATA",len(rows),"real rounds;",int(have.sum()),"with books")
    say("RESERVED LATER DATA",len(later),"rounds",utc(later[0]["_at"]),utc(later[-1]["_at"]))
    cv=validate(rows,X,have)
    write(args.out/"cv.json",cv)
    winner=choose(cv)
    write(args.out/"chosen.json",winner)
    train,cal=boundaries(rows,have)
    y=np.array([r["_y"] for r in rows],dtype=float)
    times=np.array([r["_at"] for r in rows])
    fitted=G.fit(X,y,times,train,have,winner["spec"])
    temp,level=calibrate(fitted,X,y,cal)
    raw_gap=G.raw_predict(fitted,X[cal],gap_only=True)
    gap_temp=temperature(y[cal],raw_gap)
    gap_p=G.calibrated(fitted,X[cal],gap_temp,gap_only=True)
    gap_level=min(max(float(np.quantile(np.maximum(gap_p,1-gap_p),1-SHARE)),.50000001),.999999)
    say("SETTINGS FROZEN",json.dumps(winner["spec"]),"before later test")
    TX,test_have,manifest=dataset(later,args.books)
    p=G.calibrated(fitted,TX,temp)
    p[~test_have]=np.nan
    gp=G.calibrated(fitted,TX,gap_temp,gap_only=True)
    gp[~test_have]=np.nan
    scores={"selected":score(later,p,level),"all":score(later,p),"gap_prior":score(later,gp,gap_level)}
    now=time.time()
    artifact={"schema":G.SCHEMA,"algorithm":"gap_pressure","purpose":"paper",
              "feature_names":list(B.NAMES),"fitted":fitted,"temperature":temp,"threshold":level,
              "target_share":SHARE,"training_data":meta,"calibrated_through":rows[cal[-1]]["_end"],
              "created_utc":utc(now),"forward_after":max(now,later[-1]["_end"])+60,
              "source_sha256":G.source_hash(),"feature_sha256":B.code_hash(),"protocol":PROTOCOL}
    G.GapPressureModel(artifact)
    result={"created_utc":utc(now),"chosen":winner,"training_data":meta,"later_data":current_meta,
            "test_period":{"first_utc":utc(later[0]["_at"]),"last_utc":utc(later[-1]["_at"]),"rounds":len(later)},
            "test":scores,"temperature":temp,"threshold":level,"protocol":PROTOCOL,
            "forward_after":artifact["forward_after"]}
    write(args.out/"model.json",artifact)
    write(args.out/"evaluation.json",result)
    write(args.out/"test_manifest.json",manifest)
    (args.out/"report.md").write_text(report(result),encoding="utf-8")
    say("NEW MODEL EXPORTED",args.out)
    for key,s in scores.items():say("LATER TEST",key,json.dumps(s))


if __name__=="__main__":
    main()
