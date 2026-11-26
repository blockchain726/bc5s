"""Tests for the new hierarchical probability-table model."""
import json

import numpy as np
import pytest

from bcbot import gap_pressure as G
from bcbot import ud_best as B
from tools.train_gap_pressure import training_mask


def planted(n=6000):
    rng=np.random.default_rng(18)
    X=np.full((n,len(B.NAMES)),np.nan)
    gap=rng.choice([-1.,1.],n)*rng.uniform(.3,1.3,n)
    q=rng.normal(size=n)
    X[:,B.NAMES.index("ctx_dev_tick")]=gap
    for name in G.RAW_NAMES:
        X[:,B.NAMES.index(name)]=q
    X[:,B.NAMES.index("mix5_ret1")]=rng.normal(size=n)*.2
    y=(q+rng.normal(size=n)*.25>0).astype(float)
    times=np.arange(n)*30.
    train=np.arange(n)<4000
    spec={"recipe":"queue","strength":50.,"depth":3,"half_life_h":0.}
    fitted=G.fit(X,y,times,train,np.ones(n,dtype=bool),spec)
    return X,y,fitted


def artifact(fitted):
    return {"schema":G.SCHEMA,"algorithm":"gap_pressure","purpose":"paper",
            "feature_names":list(B.NAMES),"fitted":fitted,"temperature":1.,"threshold":.55}


def test_learns_conditional_pressure_and_can_override_the_gap():
    X,y,m=planted()
    p=G.raw_predict(m,X[4000:])
    assert ((p>.5)==y[4000:]).mean()>.8
    gap_only=G.raw_predict(m,X[4000:],gap_only=True)
    assert ((gap_only>.5)==y[4000:]).mean()<.6
    assert np.mean((p>.5)!=(gap_only>.5))>.2


def test_up_down_symmetry_and_missing_book_abstention():
    X,_,m=planted()
    p=G.raw_predict(m,X[4000:])
    mirrored=G.raw_predict(m,X[4000:]*B.SIGN)
    np.testing.assert_allclose(p+mirrored,1.,atol=1e-12)
    X[0,B.NAMES.index("mix5_ofi1")]=np.nan
    assert np.isnan(G.raw_predict(m,X[:1])[0])


def test_saved_tables_produce_identical_probabilities(tmp_path):
    X,_,m=planted()
    one=G.GapPressureModel(artifact(m))
    path=tmp_path/"model.json"
    path.write_text(json.dumps(artifact(m)),encoding="utf-8")
    two=G.GapPressureModel.load(path)
    np.testing.assert_array_equal(one.predict_matrix(X[-30:]),two.predict_matrix(X[-30:]))


def test_future_labels_cannot_change_a_fit_on_past_rows():
    X,y,_=planted()
    times=np.arange(len(y))*30.
    train=np.arange(len(y))<4000
    spec={"recipe":"turn","strength":150.,"depth":2,"half_life_h":72.}
    a=G.fit(X,y,times,train,np.ones(len(y),dtype=bool),spec)
    changed=y.copy()
    changed[~train]=1-changed[~train]
    b=G.fit(X,changed,times,train,np.ones(len(y),dtype=bool),spec)
    assert a==b


def test_settlement_purge_excludes_future_and_unsettled_labels():
    rows=[{"_end":90.},{"_end":140.},{"_end":190.}]
    np.testing.assert_array_equal(training_mask(rows,201.),[True,True,False])


def test_monotonic_gap_prior_pools_noisy_adjacent_cells():
    values=G.monotonic([.5,.6,.55,.7],[100,100,100,100])
    np.testing.assert_allclose(values,[.5,.575,.575,.7])
    assert (np.diff(values)>=0).all()


def test_rejects_wrong_feature_schema_and_calibration():
    _,_,m=planted()
    for change in ({"schema":2},{"feature_names":[]},{"temperature":3.},{"threshold":.5},{"purpose":"live"}):
        with pytest.raises(ValueError):
            G.GapPressureModel({**artifact(m),**change})


def test_book_replay_accepts_relative_recording_folder(tmp_path, monkeypatch):
    from bcbot import obfeat as F
    from tools.train_ud_best import read_books
    folder=tmp_path/"books"
    folder.mkdir()
    a=np.zeros(4,dtype=F.ROW)
    a["t"]=[100.,100.1,100.2,100.3]
    a.tofile(folder/"bybit_perp_20261118_12.bin")
    monkeypatch.chdir(tmp_path)
    loaded,manifest=read_books("books","bybit_perp",100.,100.3)
    np.testing.assert_array_equal(loaded["t"],a["t"])
    assert manifest[0]["rows_read"]==4
