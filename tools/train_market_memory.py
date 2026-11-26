"""Past-only research fits for Market Memory; consumes settled paper observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bcbot import market_memory as M, ud_best as B
from tools.train_ud_best import freeze_rows,dataset,score,temperature,utc
from tools.atomic_files import atomic_json,atomic_write


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def past(rows,at):
    return np.array([r["_end"]+60<at for r in rows])


def prepare(cache,live_db=None):
    """Extend a verified feature cache; live observations replace approximate historical states."""
    rows,meta=freeze_rows(ROOT/"logs/auto_rounds.jsonl")
    source=cache if (cache/"data.json").exists() else ROOT/"models/ud_best_20261118"
    saved=json.loads((source/"data.json").read_text(encoding="utf-8"))
    if saved.get("feature_sha256",B.code_hash())!=B.code_hash():
        raise ValueError("cached feature code changed; use a new cache")
    original,oldmeta=freeze_rows(ROOT/"logs/auto_rounds.jsonl",saved["data"]["snapshot_bytes"])
    if oldmeta!=saved["data"] or [r["id"] for r in original]!=saved["ids"]:
        raise ValueError("historical cache changed")
    with np.load(source/saved.get("file","dataset.npz")) as z:
        X,have=z["X"],z["have"]
    if X.shape!=(len(original),len(B.NAMES)):
        raise ValueError("feature schema changed")
    if len(rows)>len(original):
        added=rows[len(original):]
        fresh,known,_=dataset(added,ROOT/"logs/obrec")
        X,have=np.vstack([X,fresh]),np.r_[have,known]
    cache.mkdir(parents=True,exist_ok=True)
    cache_name="dataset_"+meta["sha256"][:16]+".npz"
    atomic_write(cache/cache_name,lambda stream:np.savez_compressed(stream,X=X,have=have))
    atomic_json(cache/"data.json",{"data":meta,"ids":[r["id"] for r in rows],
                                 "feature_sha256":B.code_hash(),"file":cache_name})
    live_count=0
    live_provenance=[]
    if live_db and Path(live_db).exists():
        db=sqlite3.connect(Path(live_db).resolve().as_uri()+"?mode=ro",uri=True)
        db.row_factory=sqlite3.Row
        live=db.execute("""SELECT o.*,s.winner,s.tie,s.pools FROM observations o
          JOIN settlements s ON o.key=s.key WHERE o.committed IS NOT NULL AND s.tie=0 ORDER BY o.at""").fetchall()
        db.close()
        by_cutoff={f"{r['cutoff']:.3f}":i for i,r in enumerate(rows)}
        extra_rows,extra_x=[],[]
        for a in live:
            r={"id":"paper:"+a["key"],"cutoff":a["cutoff"],"start":a["start"],"end":a["end"],
               "feat":json.loads(a["features"]),"_at":a["at"],"_end":a["end"],
               "_y":int(a["winner"]=="UP"),"pools":json.loads(a["pools"]),"_pc":a["captured"]}
            x=np.array(json.loads(a["vector"]),dtype=float)
            if x.shape!=(len(B.NAMES),) or not np.isfinite(x[M.IDX]).all():
                continue
            i=by_cutoff.get(a["key"])
            if i is not None:
                rows[i],X[i],have[i]=r,x,True
            else:
                extra_rows.append(r);extra_x.append(x)
            live_count+=1
            live_provenance.append([a["key"],a["winner"],a["vector"]])
        if extra_rows:
            rows+=extra_rows;X=np.vstack([X,extra_x]);have=np.r_[have,np.ones(len(extra_rows),dtype=bool)]
        order=np.argsort([r["_at"] for r in rows],kind="stable")
        rows=[rows[i] for i in order];X,have=X[order],have[order]
    live_hash=hashlib.sha256(json.dumps(live_provenance,separators=(",",":")).encode()).hexdigest()
    return rows,X,have,{**meta,"settled_live_rows":live_count,"live_rows_sha256":live_hash}


def calibrate(model,X,y,cal):
    p=model.raw(X[cal]);valid=np.isfinite(p)
    model.temperature=temperature(y[cal][valid],p[valid])
    cp=model.predict_matrix(X[cal])[valid]
    model.threshold=min(max(float(np.quantile(np.maximum(cp,1-cp),.75)),.50000001),.999999)


def run(out,cache,live_db):
    if out.exists():
        raise FileExistsError("model versions are immutable; choose a new folder")
    out.mkdir(parents=True)
    rows,X,have,meta=prepare(cache,live_db)
    y=np.array([r["_y"] for r in rows],dtype=float)
    times=np.array([r["_at"] for r in rows])
    have &= np.isfinite(X[:,M.IDX]).all(axis=1)
    books=np.flatnonzero(have)
    if len(books)<3000:
        raise ValueError("need 3000 known past market states")
    # Reserve the latest fifth; no candidate selection uses these labels.
    split=int(.8*len(books));dev,test=books[:split],books[split:]
    dev=dev[np.array([rows[i]["_end"]+60<rows[test[0]]["_at"] for i in dev])]
    candidates=[{"neighbors":k,"half_life_h":h,"blend":a}
                for k in (96,384) for h in (24.,96.) for a in (.25,.75)]
    results=[]
    edges=[int(len(dev)*f) for f in (.5,2/3,5/6,1.)]
    for spec in candidates:
        rr,ps,selected,folds=[],[],[],[]
        for lo,hi in zip(edges,edges[1:]):
            val=dev[lo:hi];before=dev[:lo];cal=before[int(.8*len(before)):]
            cal=cal[np.array([rows[i]["_end"]+60<rows[val[0]]["_at"] for i in cal])]
            train=past(rows,rows[cal[0]]["_at"])
            model=M.fit(X,y,times,train,have,spec);calibrate(model,X,y,cal)
            p=model.predict_matrix(X[val]);s=np.where(np.maximum(p,1-p)>=model.threshold,p,np.nan)
            rs=[rows[i] for i in val]
            folds.append({"train_end":max(rows[i]["_end"] for i in np.flatnonzero(train)),
                          "cal_first":rows[cal[0]]["_at"],"cal_end":rows[cal[-1]]["_end"],
                          "val_first":rows[val[0]]["_at"],"selected":score(rs,s)})
            rr.extend(rs);ps.extend(p);selected.extend(s)
        result={"spec":spec,"all":score(rr,ps),"selected":score(rr,selected),"folds":folds}
        results.append(result)
        print("CV",spec,result["selected"]["calls"],result["selected"]["accuracy"],flush=True)
    eligible=[r for r in results if r["selected"]["calls"]>=200 and .10<=r["selected"]["coverage"]<=.60]
    if not eligible:
        raise ValueError("no candidate met forward-development coverage requirements")
    winner=max(eligible,key=lambda r:(r["selected"]["wilson95"][0],r["selected"]["accuracy"]))
    cal=dev[int(.8*len(dev)):]
    train=past(rows,rows[cal[0]]["_at"])
    evaluated=M.fit(X,y,times,train,have,winner["spec"]);calibrate(evaluated,X,y,cal)
    p=evaluated.predict_matrix(X[test]);testrows=[rows[i] for i in test]
    test_scores={"all":score(testrows,p),"selected":score(testrows,p,evaluated.threshold)}
    write(out/"evaluation.json",{"winner":winner,"cv":results,"test":test_scores,"data":meta,
          "test_first_utc":utc(testrows[0]["_at"]),"test_last_utc":utc(testrows[-1]["_at"]),
          "caveat":"previously examined historical corpus; only the live ledger is prospective evidence"})
    # Keep a short recent calibration block so six-hour updates can learn new
    # live states promptly rather than withholding the latest fifth forever.
    cal=books[-min(480,len(books)//5):]
    model=M.fit(X,y,times,past(rows,rows[cal[0]]["_at"]),have,winner["spec"])
    calibrate(model,X,y,cal)
    created=time.time()
    model.save(out,{"created_pc":created,"created_utc":utc(created),"data":meta,
               "trained_through":rows[cal[-1]]["_end"],"forward_after":created+60,
               "historical_test":test_scores,"target_share":.25,"calibration_rows":len(cal)})
    print("EXPORTED",out,"TEST",json.dumps(test_scores),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,default=ROOT/"logs/model_lab/history")
    p.add_argument("--live-db",type=Path,default=ROOT/"logs/model_lab/paper.sqlite3")
    a=p.parse_args();run(a.out.resolve(),a.cache.resolve(),a.live_db.resolve())


if __name__=="__main__":
    main()
