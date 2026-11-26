"""Paper-only live model research: predict, settle, report, and refit every six hours.

Reads the existing local GET API and recorder files. It has no order-placement
code, never changes the trading bot, and never backfills prospective predictions.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import aiohttp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bcbot import market_memory as M, ud_best as B, obfeat as F
from bcbot.paper_lab import PaperLedger,promotion_evidence
from tools.train_ud_best import score,utc
from tools.atomic_files import atomic_json


def at(series,t):
    if not series:
        return None
    j=int(np.searchsorted([p[0] for p in series],t,side="right")-1)
    return float(series[j][1]) if j>=0 else None


def snapshot(d):
    """Rebuild causal decision features from the bot's public price curves.

    The curve is sampled at 200 ms, so this initial history/live input difference
    is recorded; subsequent fits use these exact captured vectors and outcomes.
    """
    now=float(d["now"])
    bc=[p for p in d.get("bc") or [] if p[0]<=now]
    our=[p for p in d.get("our") or [] if p[0]<=now]
    if len(bc)<30 or len(our)<30:
        return None
    tick_t,price=bc[-1]
    own=at(our,tick_t);oldbc=at(bc,tick_t-1);oldour=at(our,tick_t-1)
    current_our=at(our,now);b1=at(bc,now-1);o1=at(our,now-1)
    if None in (own,oldbc,oldour,current_our,b1,o1):
        return None
    grid=[at(bc,now-60+.5*i) for i in range(121)]
    rets=[math.log(b/a) for a,b in zip(grid,grid[1:]) if a and b]
    if len(rets)<80:
        return None
    f={"at":now,"dev_tick":float(price-own),"dev_chg1":float(price-own-oldbc+oldour),
       "vol60":float(np.std(rets)*price),"bc_mom1":float(price-b1),
       "our_mom1":float(current_our-o1),"tick_age":float(now-tick_t),
       "source":"sampled_api_curve","clock":d.get("clock_bc") or {}}
    return f if B.valid_snapshot(f) else None


def recent_books(folder,decision_pc):
    out={}
    hours={datetime.fromtimestamp(t,timezone.utc).strftime("%Y%m%d_%H") for t in (decision_pc,decision_pc-45)}
    for name in B.MARKETS:
        parts=[]
        for hour in sorted(hours):
            path=folder/f"{name}_{hour}.bin"
            if not path.exists():
                continue
            count=path.stat().st_size//F.ROW.itemsize
            if not count:
                continue
            a=np.memmap(path,dtype=F.ROW,mode="r",shape=(count,))
            lo,hi=np.searchsorted(a["t"],[decision_pc-42,decision_pc],side="right")
            parts.append(np.array(a[lo:hi],copy=True));del a
        out[name]=np.concatenate(parts) if parts else np.zeros(0,dtype=F.ROW)
    return out


class RoundTail:
    def __init__(self,path):
        self.path=path;self.position=0;self.buffer=b""

    def read(self):
        if not self.path.exists():
            return []
        if self.path.stat().st_size<self.position:
            self.position=0;self.buffer=b""
        with self.path.open("rb") as f:
            f.seek(self.position);data=self.buffer+f.read();self.position=f.tell()
        lines=data.split(b"\n");self.buffer=lines.pop()
        out=[]
        for line in lines:
            try:out.append(json.loads(line))
            except ValueError:continue
        return out


def metric_records(records):
    rs=[];p=[];selected=[]
    ties=sum(bool(r["tie"]) for r in records)
    for r in records:
        if r["tie"]:
            continue
        rs.append({"_at":r["at"],"_y":int(r["winner"]=="UP"),"pools":json.loads(r["pools"]),"cutoff":r["cutoff"]})
        p.append(r["p"]);selected.append(r["p"] if r["selected"] else np.nan)
    if not rs:
        return {"all":{"calls":0,"accuracy":None},"selected":{"calls":0,"accuracy":None},"ties":ties}
    return {"all":score(rs,p),"selected":score(rs,selected),"ties":ties}


class Lab:
    def __init__(self,args):
        self.args=args;self.root=args.folder.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.ledger=PaperLedger(self.root/"paper.sqlite3")
        self.tail=RoundTail(ROOT/"logs/auto_rounds.jsonl")
        self.models={};self.started=time.time();self.job=None;self.job_log=None
        self.state={"last_fit":0.,"last_fit_settlements":0,"checks":{}}
        if (self.root/"control.json").exists():
            self.state.update(json.loads((self.root/"control.json").read_text()))
        self.last_error=None;self.last_prediction=None;self.last_skip=None
        self.load_models()

    def load_models(self):
        folders=[self.args.bootstrap.resolve()]+sorted((self.root/"versions").glob("*"))
        for folder in folders:
            file=folder/"model.json"
            if not file.exists():continue
            a=json.loads(file.read_text(encoding="utf-8"))
            if a.get("forward_after",0)>time.time():continue
            name="memory_"+datetime.fromtimestamp(a["created_pc"],timezone.utc).strftime("%Y%m%d_%H%M%S")
            if name in self.models:continue
            try:model=M.MarketMemory.load(folder)
            except Exception as e:
                self.last_error="model load: "+str(e);continue
            self.ledger.register(name,folder,a["created_pc"],a["trained_through"])
            self.models[name]=model
            self.state["last_fit"]=max(self.state["last_fit"],a["created_pc"])
        active={m["id"] for m in self.ledger.active()}
        self.models={k:v for k,v in self.models.items() if k in active}

    def settlements(self):
        for record in self.tail.read():
            self.ledger.settle(record)

    def maybe_train(self):
        if self.job is not None:
            code=self.job.poll()
            if code is None:return
            self.ledger.event("training_finished",exit_code=code)
            if self.job_log:self.job_log.close()
            self.job=self.job_log=None
            self.load_models()
            self.state["training_error"]="training failed; see training.log" if code!=0 else None
        count=self.ledger.counts()["settlements"]
        if time.time()-self.state["last_fit"]<self.args.refit_hours*3600:
            return
        if self.models and count-self.state["last_fit_settlements"]<300:
            return
        version=self.root/"versions"/datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cmd=[str(ROOT/"logs/research-venv/Scripts/python.exe"),"-B",str(ROOT/"tools/train_market_memory.py"),
             "--out",str(version),"--cache",str(self.root/"history"),"--live-db",str(self.ledger.path)]
        self.job_log=(self.root/"training.log").open("a",encoding="utf-8")
        flags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=="nt" else 0
        self.job=subprocess.Popen(cmd,cwd=ROOT,stdout=self.job_log,stderr=subprocess.STDOUT,creationflags=flags)
        self.state["last_fit"]=time.time();self.state["last_fit_settlements"]=count
        self.ledger.event("training_started",pid=self.job.pid,version=str(version))

    def capture(self,d):
        r=d.get("round") or {};now=float(d.get("now") or 0)
        if not all(r.get(k) for k in ("cutoff","start","end")) or r.get("status")!="betting":return
        left=min(r["cutoff"],r["start"])-now
        if not 1.6<=left<=3.1 or self.ledger.has_round(r["cutoff"]):return
        f=snapshot(d);offset=(d.get("clock_bc") or {}).get("offset")
        if f is None or not isinstance(offset,(int,float)):
            self.last_skip="price history, tick or clock unavailable";return
        decision_pc=now+offset
        if abs(time.time()-decision_pc)>.75:
            self.last_skip="API snapshot too old";return
        books=recent_books(self.args.books.resolve(),decision_pc)
        X=B.build([f],books,[decision_pc])
        if not np.isfinite(X[0,M.IDX]).all():
            self.last_skip="waiting for fresh recorded books";return
        predictions={}
        for name,model in self.models.items():
            p=float(model.predict_matrix(X)[0])
            if np.isfinite(p):
                predictions[name]={"p":p,"selected":abs(p-.5)>1e-10 and max(p,1-p)>=model.threshold}
        if not predictions:return
        observed=time.time()
        o={"cutoff":r["cutoff"],"start":r["start"],"end":r["end"],"at":f["at"],
           "captured":observed,"offset":offset,"features":f,
           "vector":[float(v) if np.isfinite(v) else None for v in X[0]]}
        if self.ledger.record(o,predictions):
            self.last_prediction={"at_utc":utc(observed),"cutoff":r["cutoff"],"lead_s":left,"models":predictions}
            self.last_skip=None

    def report(self):
        active=self.ledger.active();allmodels=[dict(r) for r in self.ledger.db.execute("SELECT * FROM models ORDER BY created")]
        metrics={m["id"]:metric_records(self.ledger.records(m["id"])) for m in allmodels}
        incumbent=next((m for m in active if m["role"]=="incumbent"),None)
        if incumbent:
            for candidate in active:
                if candidate["role"]!="challenger":continue
                c=self.ledger.records(candidate["id"]);b=self.ledger.records(incumbent["id"])
                ev=promotion_evidence(c,b)
                checkpoint=min(ev.get("calls",[0,0]))//500
                key=candidate["id"]+":"+incumbent["id"]
                if checkpoint>self.state["checks"].get(key,0):
                    self.state["checks"][key]=checkpoint
                    self.ledger.event("promotion_checkpoint",candidate=candidate["id"],incumbent=incumbent["id"],evidence=ev)
                    roi=metrics[candidate["id"]]["selected"].get("roi_hour95")
                    if ev["promote"] and roi is not None and roi[0]>0:
                        self.ledger.promote(candidate["id"])
        # Old model records stay in SQLite when a model leaves the active comparison.
        if len(active)>16:
            for m in active:
                if m["role"]=="challenger" and time.time()-m["created"]>72*3600:
                    self.ledger.db.execute("UPDATE models SET role='retired' WHERE id=?",(m["id"],));self.ledger.db.commit()
                    self.ledger.event("paper_retirement",model=m["id"]);break
        result={"updated_utc":utc(time.time()),"pid":os.getpid(),"mode":"paper",
                "refit_hours":self.args.refit_hours,"refit_minimum_new_settlements":300,
                "counts":self.ledger.counts(),"models":self.ledger.active(),"results":metrics,
                "last_prediction":self.last_prediction,"last_skip":self.last_skip,
                "last_error":self.last_error or self.state.get("training_error"),
                "training_error":self.state.get("training_error"),
                "training_pid":self.job.pid if self.job and self.job.poll() is None else None,
                "next_fit_not_before_utc":utc(self.state["last_fit"]+self.args.refit_hours*3600),
                "input_note":"API curve is sampled at 200 ms; future fits use exact captured vectors",
                "promotion_note":"500 selected calls in both models, 24 hours, 99% block accuracy gain, log-loss improvement and positive return interval; fixed checkpoints; paper only"}
        atomic_json(self.root/"status.json",result);atomic_json(self.root/"control.json",self.state)
        lines=["# Live paper model results","",f"Updated {result['updated_utc']}. Paper only.","",
               "Every prediction is saved before the round starts. Records remain separate by model version and survive refits and restarts.","",
               "| Model version | Role | Selected wins / calls | Win rate | Approx. 95% | Paper ROI | All-round win rate |",
               "|---|---|---:|---:|---:|---:|---:|"]
        pct=lambda v:"-" if v is None else f"{100*v:.2f}%"
        for m in allmodels:
            s=metrics[m["id"]]["selected"];a=metrics[m["id"]]["all"];ci=s.get("wilson95")
            lines.append(f"| {m['id']} | {m['role']} | {s.get('wins',0)}/{s['calls']} | {pct(s['accuracy'])} | "
                         f"{pct(ci[0])+' to '+pct(ci[1]) if ci else '-'} | {pct(s.get('roi'))} | {pct(a['accuracy'])} |")
        lines += ["",f"Retraining: every {self.args.refit_hours:g} hours after at least 300 new settled observations.",
                  "A challenger is paper-tested against the incumbent. No model change here places or arms real orders.",
                  "Initial historical results are in each version's evaluation.json and are excluded from this table.",
                  "The initial live inputs use sampled API curves. New fits replace historical proxy states with the exact captured vectors.",
                  "Promotion is a monitoring rule, not a proof against adaptive selection or future losses.",
                  f"Latest skipped-input reason: {self.last_skip or '-'}.",""]
        (self.root/"results.md").write_text("\n".join(lines),encoding="utf-8")

    def report_safely(self):
        # The error handler must not crash on the same locked file that caused
        # the original report to fail. Retry publication on the next interval.
        try:
            self.report()
        except OSError as e:
            self.last_error=f"report failed: {type(e).__name__}: {str(e)[:180]}"

    async def run(self):
        last_report=0
        timeout=aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            while not (self.root/"STOP").exists():
                try:
                    self.settlements()
                    if time.time()-last_report>=15:
                        self.load_models();self.maybe_train();self.report_safely();last_report=time.time()
                    async with http.get(self.args.api.rstrip("/")+"/auto/prices?s=80") as response:
                        response.raise_for_status();d=await response.json()
                    if self.models:self.capture(d)
                    self.last_error=None
                    r=d.get("round") or {};left=min(r.get("start",0),r.get("cutoff",0))-d.get("now",0)
                    await asyncio.sleep(.15 if 1.3<left<3.3 else .7)
                except Exception as e:
                    self.last_error=f"{type(e).__name__}: {str(e)[:180]}"
                    if time.time()-last_report>=15:self.report_safely();last_report=time.time()
                    await asyncio.sleep(1)
        self.report_safely();self.ledger.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,default=ROOT/"logs/model_lab")
    p.add_argument("--books",type=Path,default=ROOT/"logs/obrec")
    p.add_argument("--bootstrap",type=Path,default=ROOT/"models/market_memory_20261118")
    p.add_argument("--api",default="http://127.0.0.1:8765")
    p.add_argument("--refit-hours",type=float,default=6.)
    p.add_argument("--status",action="store_true")
    a=p.parse_args()
    if a.status:
        print((a.folder/"status.json").read_text(encoding="utf-8"));return
    a.folder.mkdir(parents=True,exist_ok=True)
    lock=(a.folder/"run.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):lock.write(b"0");lock.flush()
    lock.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        lock.close();raise SystemExit("a model-lab process already owns this folder")
    try:
        asyncio.run(Lab(a).run())
    finally:
        lock.close()


if __name__=="__main__":main()
