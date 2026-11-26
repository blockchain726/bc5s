"""Prospective logging, chronology, model persistence and live input tests."""
import json
import time

import numpy as np
import pytest

from bcbot import market_memory as M,ud_best as B
from bcbot.paper_lab import PaperLedger,promotion_evidence
from tools.model_lab import snapshot,RoundTail,metric_records


def observation():
    return {"cutoff":1003.,"start":1003.,"end":1008.,"at":1000.,"captured":1001.,
            "offset":0.,"features":{"at":1000.},"vector":[0.]}


def ledger(tmp_path):
    a=PaperLedger(tmp_path/"paper.sqlite3")
    a.register("one",tmp_path,0.,100.)
    return a


def test_predictions_must_precede_window_and_training_must_precede_prediction(tmp_path):
    a=ledger(tmp_path);o=observation();p={"one":{"p":.7,"selected":True}}
    assert not a.record(o,p,now=1004.)
    a.register("future",tmp_path,1002.,100.)
    with pytest.raises(ValueError):a.record(o,{"future":{"p":.7,"selected":True}},now=1001.)
    a.register("label_leak",tmp_path,0.,950.)
    with pytest.raises(ValueError):a.record(o,{"label_leak":{"p":.7,"selected":True}},now=1001.)
    assert a.counts()["observations"]==0
    a.close()


def test_predictions_and_results_are_deduplicated_and_survive_restart(tmp_path):
    a=ledger(tmp_path);o=observation();p={"one":{"p":.7,"selected":True}}
    assert a.record(o,p,now=1001.)
    assert not a.record(o,p,now=1001.)
    result={"cutoff":1003.,"start":1003.,"win":"UP","pools":[100.,100.],"tie":False}
    assert a.settle(result) and not a.settle(result)
    a.close();a=PaperLedger(tmp_path/"paper.sqlite3")
    records=a.records("one")
    assert len(records)==1 and records[0]["p"]==.7
    assert metric_records(records)["selected"]["wins"]==1
    a.register("two",tmp_path,2000.,1500.)
    assert len(a.records("one"))==1
    a.close()


def test_slow_disk_commit_is_excluded_from_prospective_results(tmp_path,monkeypatch):
    import bcbot.paper_lab as module
    a=ledger(tmp_path)
    readings=iter([1001.,1004.,1004.])
    monkeypatch.setattr(module.time,"time",lambda:next(readings))
    assert not a.record(observation(),{"one":{"p":.7,"selected":True}})
    assert not a.settle({"cutoff":1003.,"start":1003.,"win":"UP"})
    assert not a.training_observations()
    a.close()


def test_periodic_training_requires_both_time_and_new_settlements(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tools.model_lab as module
    obj=module.Lab.__new__(module.Lab)
    obj.root=tmp_path;obj.args=SimpleNamespace(refit_hours=6.)
    obj.job=obj.job_log=None;obj.models={"one":object()}
    obj.state={"last_fit":module.time.time()-7*3600,"last_fit_settlements":0}
    counts={"settlements":299};events=[];started=[]
    obj.ledger=SimpleNamespace(path=tmp_path/"db",counts=lambda:counts,event=lambda *a,**k:events.append((a,k)))
    monkeypatch.setattr(module.subprocess,"Popen",lambda *a,**k:started.append(a) or SimpleNamespace(pid=999))
    obj.maybe_train();assert not started
    counts["settlements"]=300;obj.maybe_train()
    assert len(started)==1 and obj.state["last_fit_settlements"]==300
    obj.job_log.close()


def test_failed_training_stays_visible_after_successful_data_collection(tmp_path):
    from types import SimpleNamespace
    from tools.model_lab import Lab

    obj=Lab.__new__(Lab)
    obj.root=tmp_path;obj.args=SimpleNamespace(refit_hours=6.)
    obj.job=SimpleNamespace(poll=lambda:1);obj.job_log=None;obj.models={}
    obj.state={"last_fit":time.time(),"last_fit_settlements":0,"checks":{}}
    obj.ledger=PaperLedger(tmp_path/"paper.sqlite3")
    obj.load_models=lambda:None
    obj.last_error=obj.last_prediction=obj.last_skip=None
    try:
        obj.maybe_train()
        # A healthy subsequent GET clears collection errors; training is separate.
        obj.last_error=None
        obj.report()
        saved=json.loads((tmp_path/"status.json").read_text())
        assert saved["last_error"]==saved["training_error"]=="training failed; see training.log"
        assert json.loads((tmp_path/"control.json").read_text())["training_error"]
        obj.job=SimpleNamespace(poll=lambda:0)
        obj.maybe_train();obj.report()
        assert json.loads((tmp_path/"status.json").read_text())["training_error"] is None
    finally:
        obj.ledger.close()


def test_report_lock_does_not_crash_collection_or_its_error_handler(tmp_path,monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import tools.model_lab as module

    obj=module.Lab.__new__(module.Lab)
    obj.root=tmp_path;obj.args=SimpleNamespace(api="http://unused")
    obj.last_error=None;obj.models={}
    obj.settlements=obj.load_models=obj.maybe_train=lambda:None
    obj.ledger=SimpleNamespace(close=lambda:None)
    attempts=[]

    def locked():
        attempts.append(1)
        raise PermissionError(13,"locked status.json")

    class Http:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def get(self,*args):raise ConnectionError("bot is stopped")

    async def stop_after_iteration(delay):
        (tmp_path/"STOP").touch()

    obj.report=locked
    monkeypatch.setattr(module.aiohttp,"ClientSession",lambda **kw:Http())
    monkeypatch.setattr(module.asyncio,"sleep",stop_after_iteration)
    asyncio.run(obj.run())
    assert len(attempts)>=2 and "report failed" in obj.last_error


def test_later_actual_start_correction_cannot_make_late_prediction_valid(tmp_path):
    a=ledger(tmp_path)
    assert a.record(observation(),{"one":{"p":.7,"selected":True}},now=1001.)
    assert not a.settle({"cutoff":1003.,"start":1000.5,"win":"UP"})
    assert not a.records("one")
    a.close()


def test_promotion_requires_enough_new_calls_and_distinct_hours():
    c=[];b=[]
    for i in range(600):
        r={"key":str(i),"at":i*180.,"tie":0,"winner":"UP","selected":1}
        c.append({**r,"p":.7});b.append({**r,"p":.3})
    assert not promotion_evidence(c[:100],b[:100])["promote"]
    assert promotion_evidence(c,b)["promote"]
    assert not promotion_evidence(b,c)["promote"]


def test_live_snapshot_does_not_use_future_ticks_or_result_fields():
    t=np.arange(1200.)*.1+900
    series=[[float(x),float(100+2*np.sin(x))] for x in t]
    d={"now":1010.,"bc":series,"our":[[x,p-.1] for x,p in series],"clock_bc":{"offset":0.}}
    a=snapshot(d)
    d["bc"]+=[[1011.,1e6]];d["our"]+=[[1011.,1.]];d["win"]="DOWN"
    assert snapshot(d)==a and a is not None


def test_log_tail_keeps_partial_json_line_until_complete(tmp_path):
    path=tmp_path/"rounds.jsonl";path.write_bytes(b'{"a":1}\n{"a":')
    tail=RoundTail(path)
    assert tail.read()==[{"a":1}]
    with path.open("ab") as f:f.write(b'2}\n')
    assert tail.read()==[{"a":2}]
    assert tail.read()==[]


def planted():
    rng=np.random.default_rng(181126);n=3000
    X=np.full((n,len(B.NAMES)),np.nan);q=rng.normal(size=n)
    for i in M.IDX:X[:,i]=q+rng.normal(size=n)*.15
    X[:,B.NAMES.index("ctx_dev_tick")]=rng.choice([-1.,1.],n)*rng.uniform(.2,1.2,n)
    X[:,B.NAMES.index("ctx_log_vol")]=1.
    X[:,B.NAMES.index("dispersion")]=.3
    y=(q+rng.normal(size=n)*.2>0).astype(float)
    times=np.arange(n)*30.;train=np.arange(n)<2400
    return X,y,times,train


def test_memory_learns_states_without_future_labels_and_reloads(tmp_path):
    X,y,t,train=planted();have=np.ones(len(y),dtype=bool)
    spec={"neighbors":96,"half_life_h":96.,"blend":.75}
    model=M.fit(X,y,t,train,have,spec)
    p=model.predict_matrix(X[2400:])
    assert ((p>.5)==y[2400:]).mean()>.75
    np.testing.assert_allclose(model.predict_matrix(X[2400:]*B.SIGN)+p,1.,atol=1e-8)
    changed=y.copy();changed[~train]=1-changed[~train]
    np.testing.assert_array_equal(M.fit(X,changed,t,train,have,spec).predict_matrix(X[2400:]),p)
    model.save(tmp_path/"model",{"created_pc":0.,"trained_through":0.})
    np.testing.assert_array_equal(M.MarketMemory.load(tmp_path/"model").predict_matrix(X[2400:]),p)
    with (tmp_path/"model/memory.npz").open("ab") as f:f.write(b"changed")
    with pytest.raises(ValueError):M.MarketMemory.load(tmp_path/"model")


def test_retraining_consumes_exact_settled_live_vectors_and_extends_cache(tmp_path,monkeypatch):
    import tools.train_market_memory as trainer
    source=tmp_path/"models/ud_best_20261118";source.mkdir(parents=True)
    rows=[{"id":str(i),"cutoff":1003.+30*i,"_at":1000.+30*i,"_end":1008.+30*i,"_y":1} for i in range(3)]
    old={"snapshot_bytes":2,"sha256":"a"*64};fresh={"snapshot_bytes":3,"sha256":"b"*64}
    (source/"data.json").write_text(json.dumps({"data":old,"ids":["0","1"]}),encoding="utf-8")
    np.savez(source/"dataset.npz",X=np.zeros((2,len(B.NAMES))),have=np.ones(2,dtype=bool))
    monkeypatch.setattr(trainer,"ROOT",tmp_path)
    monkeypatch.setattr(trainer,"freeze_rows",lambda path,size=None:
                        ([dict(r) for r in rows[:2]],old) if size==2 else ([dict(r) for r in rows],fresh))
    monkeypatch.setattr(trainer,"dataset",lambda rr,path:(np.zeros((len(rr),len(B.NAMES))),np.ones(len(rr),dtype=bool),[]))
    db=ledger(tmp_path)
    o=observation();o["vector"]=[3.14]*len(B.NAMES)
    assert db.record(o,{"one":{"p":.6,"selected":True}},now=1001.)
    assert db.settle({"cutoff":1003.,"start":1003.,"win":"DOWN","pools":[100.,100.]})
    db.close()
    rr,X,have,meta=trainer.prepare(tmp_path/"cache",tmp_path/"paper.sqlite3")
    assert rr[0]["_y"]==0 and rr[0]["id"].startswith("paper:")
    assert np.all(X[0]==3.14) and meta["settled_live_rows"]==1
    manifest=json.loads((tmp_path/"cache/data.json").read_text())
    assert (tmp_path/"cache"/manifest["file"]).exists()
    rr2,X2,_,meta2=trainer.prepare(tmp_path/"cache",tmp_path/"paper.sqlite3")
    np.testing.assert_array_equal(X2,X)
    assert meta2["live_rows_sha256"]==meta["live_rows_sha256"]
