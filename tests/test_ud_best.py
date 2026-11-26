"""Causal inputs, chronological boundaries and inference round-trip checks."""
import json

import numpy as np
import pytest

from bcbot import obfeat as F
from bcbot import ud_best as B
from tools.train_ud_best import freeze_rows, fold_indices, prior_mask, score


def snapshot(at=1000., **updates):
    f = {"at": at, "dev_tick": 2., "vol60": 2., "tick_age": .1,
         "bc_mom1": .2, "our_mom1": .1, "pool_up": 100., "pool_dn": 110.}
    f.update(updates)
    return f


def books():
    a = np.zeros(500, dtype=F.ROW)
    a["t"] = 960 + np.arange(500) * .1
    a["bid"] = 100 + np.floor(np.arange(500)/20)*.01
    a["ask"] = a["bid"] + .02
    a["bs"] = 3 + np.sin(np.arange(500)[:, None]/10)
    a["as_"] = 2 + np.cos(np.arange(500)[:, None]/10)
    a["buy"] = .1
    a["sell"] = .05
    a["nb"] = 2
    a["ns"] = 1
    a["age"] = .02
    return a


def record(i):
    at = 1000. + i*30
    return {"id": i, "feat": snapshot(at), "win": "UP", "tie": False,
            "start": at + 2.7, "end": at + 7.7, "start_price": 100., "end_price": 101.,
            "_at": at, "_end": at+7.7, "_y": 1, "pools": [100., 100.]}


def artifact():
    ix = B.indices("gap")
    policy = {"group": "gap", "temperature": 1., "threshold": .52,
              "model": {"kind": "ridge", "indices": ix,
                        "fit": {"scale": [1.]*len(ix), "weights": [-.2]*len(ix)}}}
    return {"schema": B.SCHEMA, "purpose": "paper", "features": list(B.NAMES),
            "policies": {"all": policy, "selective": policy}}


def test_future_books_cannot_change_a_decision():
    a = books()
    first = B.market_features(a, [1000.], [2.])
    changed = a.copy()
    future = changed["t"] > 1000.
    changed["bid"][future] += 1000
    changed["ask"][future] += 1000
    changed["bs"][future] *= 100
    changed["buy"][future] = 100000
    np.testing.assert_allclose(first, B.market_features(changed, [1000.], [2.]), equal_nan=True)
    np.testing.assert_allclose(first, B.market_features(a[a["t"] <= 1000.], [1000.], [2.]), equal_nan=True)


def test_market_mirror_flips_direction_but_preserves_liquidity_and_spread():
    a = books()
    reflected = a.copy()
    reflected["bid"], reflected["ask"] = -a["ask"], -a["bid"]
    reflected["bs"], reflected["as_"] = a["as_"], a["bs"]
    reflected["buy"], reflected["sell"] = a["sell"], a["buy"]
    reflected["nb"], reflected["ns"] = a["ns"], a["nb"]
    x = B.market_features(a, [1000.], [2.])
    y = B.market_features(reflected, [1000.], [2.])
    signs = np.array([1 if k in {"spread"} | B.DYNAMIC_UNSIGNED else -1 for k in B.PER])
    np.testing.assert_allclose(y, x*signs, atol=1e-7, equal_nan=True)


def test_predictions_round_trip_and_ignore_settlement_fields(tmp_path):
    model = B.BestUDModel(artifact())
    f = snapshot()
    one = model.predict(f, {}, 1000.)
    f.update(win="DOWN", start_price=1000000, end_price=1, pools=[1, 1000000])
    assert model.predict(f, {}, 1000.) == one
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact()), encoding="utf-8")
    assert B.BestUDModel.load(path).predict(f, {}, 1000.) == one
    assert one["side"] == "DOWN" and one["selected"]


def test_book_policy_refuses_missing_or_stale_books():
    a = artifact()
    a["policies"]["selective"]["group"] = "all"
    m = B.BestUDModel(a)
    assert m.predict(snapshot(), {}, 1000.) is None
    stale = books()
    stale["age"] = 100
    assert m.predict(snapshot(), {k: stale for k in B.MARKETS}, 1000.) is None
    assert m.predict(snapshot(tick_age=10.), {}, 1000.) is None


def test_expanding_folds_and_training_masks_never_see_future_labels():
    rows = [record(i) for i in range(200)]
    have = np.ones(len(rows), dtype=bool)
    for cal, val in fold_indices(rows, np.arange(len(rows))):
        assert rows[cal[-1]]["_end"] + 60 < rows[val[0]]["_at"]
        mask = prior_mask(rows, rows[cal[0]]["_at"], have, {"group": "all"})
        assert max(r["_end"] for r, yes in zip(rows, mask) if yes) + 60 < rows[cal[0]]["_at"]
        assert not mask[val].any()


def test_loader_freezes_history_and_rejects_future_snapshot_and_wrong_label(tmp_path):
    path = tmp_path / "rounds.jsonl"
    rs = [record(0), record(1), record(2)]
    rs[1]["feat"]["at"] = rs[1]["start"] + 1
    rs[2]["end_price"] = 99
    path.write_text("".join(json.dumps(r)+"\n" for r in rs), encoding="utf-8")
    size = path.stat().st_size
    rows, meta = freeze_rows(path)
    assert [r["id"] for r in rows] == [0]
    assert meta["excluded"] == {"invalid_timing": 1, "inconsistent_label": 1}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record(3))+"\n")
    frozen, frozen_meta = freeze_rows(path, size)
    assert frozen == rows and frozen_meta == meta


def test_return_uses_fee_and_own_stake_and_keeps_missing_payout_visible():
    rs = [record(0), record(1)]
    rs[1]["pools"] = None
    s = score(rs, [.6, .4])
    assert s["calls"] == 2 and s["wins"] == 1 and s["payout_rows"] == 1
    assert s["roi"] == pytest.approx(.95*100/101)
    assert score(rs, [.5, .5])["calls"] == 0
