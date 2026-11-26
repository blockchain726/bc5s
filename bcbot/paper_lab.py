"""Durable, pre-result paper predictions and never-reset model-version records."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

import numpy as np


class PaperLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS models (
          id TEXT PRIMARY KEY, folder TEXT NOT NULL, created REAL NOT NULL,
          trained_through REAL NOT NULL, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (
          key TEXT PRIMARY KEY, cutoff REAL NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
          at REAL NOT NULL, captured REAL NOT NULL, committed REAL,
          offset REAL NOT NULL, features TEXT NOT NULL, vector TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
          key TEXT NOT NULL REFERENCES observations(key), model TEXT NOT NULL REFERENCES models(id),
          p REAL NOT NULL, selected INTEGER NOT NULL, PRIMARY KEY(key, model));
        CREATE TABLE IF NOT EXISTS settlements (
          key TEXT PRIMARY KEY, winner TEXT NOT NULL, tie INTEGER NOT NULL,
          pools TEXT, settled REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events (at REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
        """)
        self.db.commit()

    @staticmethod
    def key(cutoff):
        return f"{float(cutoff):.3f}"

    def event(self, kind, **body):
        self.db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), kind, json.dumps(body)))
        self.db.commit()

    def register(self, model_id, folder, created, trained_through):
        role = "challenger" if self.db.execute("SELECT 1 FROM models WHERE role='incumbent'").fetchone() else "incumbent"
        self.db.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?)",
                        (model_id, str(Path(folder).resolve()), created, trained_through, role))
        self.db.commit()

    def active(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM models WHERE role!='retired' ORDER BY created")]

    def has_round(self, cutoff):
        return self.db.execute("SELECT 1 FROM observations WHERE key=?", (self.key(cutoff),)).fetchone() is not None

    def record(self, observation, predictions, now=None):
        injected_clock = now is not None
        now = time.time() if now is None else now
        o = observation
        deadline = min(o["start"], o["cutoff"])+o["offset"]
        if o["captured"] >= deadline or now >= deadline:
            return False
        for name, p in predictions.items():
            m = self.db.execute("SELECT * FROM models WHERE id=?", (name,)).fetchone()
            if m is None or m["created"] > o["captured"] or m["trained_through"]+60 >= o["at"]:
                raise ValueError("model was not available and trained strictly before this decision")
            if not np.isfinite(p["p"]) or not 0 <= p["p"] <= 1:
                raise ValueError("invalid probability")
        key = self.key(o["cutoff"])
        try:
            self.db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,NULL,?,?,?)",
                            (key,o["cutoff"],o["start"],o["end"],o["at"],o["captured"],o["offset"],
                             json.dumps(o["features"],allow_nan=False),json.dumps(o["vector"],allow_nan=False)))
            self.db.executemany("INSERT INTO predictions VALUES (?,?,?,?)",
                                [(key,k,p["p"],int(p["selected"])) for k,p in predictions.items()])
            self.db.commit()
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False
        # Only predictions whose durable transaction completed before the window count.
        committed = now if injected_clock else time.time()
        if committed >= deadline:
            self.event("late_prediction",key=key)
            return False
        self.db.execute("UPDATE observations SET committed=? WHERE key=?", (committed,key))
        self.db.commit()
        return True

    def settle(self, record):
        if record.get("win") not in ("UP","DOWN") or not record.get("cutoff"):
            return False
        key = self.key(record["cutoff"])
        o = self.db.execute("SELECT committed,offset FROM observations WHERE key=?",(key,)).fetchone()
        if o is None or o["committed"] is None:
            return False
        actual_start = record.get("start")
        if actual_start and o["committed"] >= actual_start+o["offset"]:
            self.event("actual_start_preceded_prediction",key=key)
            return False
        cur = self.db.execute("INSERT OR IGNORE INTO settlements VALUES (?,?,?,?,?)",
                              (key,record["win"],int(bool(record.get("tie"))),
                               json.dumps(record.get("pools")),time.time()))
        self.db.commit()
        return cur.rowcount > 0

    def records(self, model_id):
        return [dict(r) for r in self.db.execute("""
          SELECT o.key,o.at,o.cutoff,p.p,p.selected,s.winner,s.tie,s.pools
          FROM predictions p JOIN observations o ON p.key=o.key
          JOIN settlements s ON s.key=o.key
          WHERE p.model=? AND o.committed IS NOT NULL ORDER BY o.at
        """,(model_id,))]

    def training_observations(self):
        return [dict(r) for r in self.db.execute("""
          SELECT o.*,s.winner,s.tie,s.pools FROM observations o JOIN settlements s ON o.key=s.key
          WHERE o.committed IS NOT NULL AND s.tie=0 ORDER BY o.at
        """)]

    def counts(self):
        return {name:self.db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("models","observations","predictions","settlements")}

    def promote(self, model_id):
        with self.db:
            self.db.execute("UPDATE models SET role='challenger' WHERE role='incumbent'")
            self.db.execute("UPDATE models SET role='incumbent' WHERE id=?",(model_id,))
        self.event("paper_promotion",model=model_id)

    def close(self):
        self.db.close()


def promotion_evidence(challenger, incumbent, minimum=500):
    """Fixed checkpoints only; compare the same forward rounds in whole-hour blocks."""
    c = {r["key"]:r for r in challenger if not r["tie"]}
    both = [(c[r["key"]],r) for r in incumbent if r["key"] in c and not r["tie"]]
    counts = [sum(bool(a[j]["selected"]) for a in both) for j in (0,1)]
    if min(counts,default=0) < minimum:
        return {"promote":False,"reason":"collecting forward selected calls","calls":counts}
    groups = {}
    for a,b in both:
        h = int(a["at"]//3600)
        g = groups.setdefault(h,np.zeros(4))
        for j,r in enumerate((a,b)):
            if r["selected"]:
                g[2*j] += (r["p"]>.5)==(r["winner"]=="UP")
                g[2*j+1] += 1
    if len(groups) < 24:
        return {"promote":False,"reason":"need 24 distinct forward hours","hours":len(groups)}
    a = np.array(list(groups.values()))
    rng = np.random.default_rng(181126)
    sampled = a[rng.integers(0,len(a),(5000,len(a)))].sum(axis=1)
    difference = sampled[:,0]/np.maximum(sampled[:,1],1)-sampled[:,2]/np.maximum(sampled[:,3],1)
    interval = np.quantile(difference,[.005,.995]).tolist()
    def logloss(j):
        p = np.clip([r[j]["p"] for r in both],1e-8,1-1e-8)
        y = np.array([r[j]["winner"]=="UP" for r in both])
        return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    gain = logloss(1)-logloss(0)
    return {"promote":interval[0]>0 and gain>0,"calls":counts,"hours":len(groups),
            "accuracy_gain_99":interval,"logloss_gain":gain,
            "reason":"forward comparison only; paper promotion is not live-trading approval"}
