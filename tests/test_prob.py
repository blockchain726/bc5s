"""The P(UP) model: its input row, the margin rule, and a fit on a synthetic log where one feature decides the result."""
import json
import random

import numpy as np

from bcbot.prob import ProbModel, FEATURES, MARGIN, row


def test_row_needs_the_required_features():
    assert row({}) is None
    assert row({"bc_mom2": 1.0, "dev_tick": 2.0, "vol60": 1.0}) is None          # obi20 missing
    x = row({"bc_mom2": 1.0, "dev_tick": 2.0, "vol60": 1.0, "obi20": -0.6})
    assert x is not None and len(x) == len(FEATURES) + 6
    assert x[len(FEATURES)] == 2.0                                                # sign(mom2) * |dev| = +2


def test_side_margin():
    assert ProbModel.side(None) is None
    assert ProbModel.side(0.5 + MARGIN) == "UP"
    assert ProbModel.side(0.5 - MARGIN) == "DOWN"
    assert ProbModel.side(0.5 + MARGIN / 2) is None


def test_fit_learns_a_planted_effect(tmp_path):
    rnd = random.Random(3)
    rows = []
    for i in range(1500):
        dev = rnd.gauss(0, 3)
        mom = rnd.gauss(0, 1)
        up = (dev < 0) if rnd.random() < 0.8 else (dev >= 0)   # a big positive gap -> DOWN next, 80% of the time
        feat = {k: rnd.gauss(0, 1) for k in FEATURES}
        feat.update(bc_mom2=mom, dev_tick=dev, vol60=1.0, obi20=rnd.uniform(-1, 1))
        rows.append({"cutoff": 1e9 + 30 * i, "win": "UP" if up else "DOWN", "feat": feat})
    path = tmp_path / "rounds.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    m = ProbModel(path)
    info = m.fit()
    assert info["fitted"] and info["rows"] == 1500
    p_hi = m.predict({**rows[0]["feat"], "dev_tick": 6.0})
    p_lo = m.predict({**rows[0]["feat"], "dev_tick": -6.0})
    assert p_hi < 0.5 - MARGIN and p_lo > 0.5 + MARGIN
    assert ProbModel.side(p_hi) == "DOWN" and ProbModel.side(p_lo) == "UP"
    # nothing to predict without the required features
    assert m.predict({"bc_mom2": 1.0}) is None
