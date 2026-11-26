"""Tests for the isolated direction model and its evaluation boundaries."""
import json

import numpy as np
import pytest

from bcbot.ud_model import UDDirectionModel, feature_names, fit_logistic, logits, sigmoid, vector
from tools.evaluate_ud_model import (block_interval, calibrate, load_rows, payout,
                                     probabilities, purged_before, score_sides)


def features(**updates):
    f = {"dev_tick": 2.0, "vol60": 1.0, "bc_mom2": 0.5, "bc_mom1": 0.2,
         "dq": {"ok": True}, "lead_s": 2.9, "tick_age": 0.1}
    f.update(updates)
    return f


def artifact():
    return {"schema": 1, "purpose": "paper", "family": "gap",
            "features": feature_names("gap"),
            "fit": {"scale": [1.0], "weights": [-0.5]},
            "temperature": 0.8, "threshold": 0.56}


def record(i, **updates):
    at = 1e9 + 45 * i
    f = features(at=at)
    r = {"id": i, "v": 3, "win": "UP", "tie": False, "feat": f,
         "start": at + 2.9, "end": at + 7.9, "cutoff": at + 3.0,
         "start_price": 100.0, "end_price": 101.0,
         "pools": [100.0, 100.0], "_at": at, "_end": at + 7.9, "_y": 1}
    r.update(updates)
    return r


@pytest.mark.parametrize("change", [
    {"dev_tick": float("nan")}, {"vol60": float("inf")}, {"vol60": 0},
    {"vol60": -1}, {"dev_tick": True}, {"dq": {"ok": False}},
    {"lead_s": 1.49}, {"lead_s": 4.51}, {"tick_age": 1.01}, {"tick_age": -0.1},
])
def test_invalid_or_stale_data_never_produces_a_direction(change):
    assert UDDirectionModel(artifact()).predict(features(**change)) is None


def test_normalization_is_invariant_to_dollar_scale():
    f = features(dev_chg1=1.0, our_mom2=-2.0)
    bigger = {k: (v * 10 if k in ("dev_tick", "vol60", "bc_mom2", "bc_mom1",
                                  "dev_chg1", "our_mom2") else v) for k, v in f.items()}
    np.testing.assert_allclose(vector(f, "normalized"), vector(bigger, "normalized"))
    assert np.isfinite(vector(features(), "flow")).all()


def test_model_is_symmetric_and_survives_serialization(tmp_path):
    model = UDDirectionModel(artifact())
    positive = model.predict(features(dev_tick=2.0))
    negative = model.predict(features(dev_tick=-2.0))
    assert positive["side"] == "DOWN" and negative["side"] == "UP"
    assert positive["p_up"] + negative["p_up"] == pytest.approx(1.0)
    assert positive["selected"] and positive["purpose"] == "paper"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact()), encoding="utf-8")
    assert UDDirectionModel.load(path).predict(features()) == positive


def test_artifact_rejects_incompatible_or_invalid_parameters():
    for change in ({"schema": 2}, {"purpose": "live"}, {"features": ["wrong"]},
                   {"temperature": float("nan")}, {"threshold": 0.5},
                   {"fit": {"scale": [0.0], "weights": [1.0]}}):
        with pytest.raises(ValueError):
            UDDirectionModel({**artifact(), **change})


def test_logistic_fit_learns_a_planted_effect_on_future_data():
    rng = np.random.default_rng(726)
    X = rng.normal(size=(3000, 1))
    y = (rng.uniform(size=3000) < sigmoid(-1.5 * X[:, 0])).astype(float)
    fit = fit_logistic(X[:2000], y[:2000], 10.0)
    predictions = sigmoid(logits(fit, X[2000:]))
    assert np.mean((predictions >= 0.5) == y[2000:]) > 0.70
    assert fit["weights"][0] < 0


def test_calibration_reduces_known_overconfidence():
    rng = np.random.default_rng(10)
    z = rng.normal(size=4000) * 2
    y = (rng.uniform(size=4000) < sigmoid(0.25 * z)).astype(float)
    temperature = calibrate(y, z)
    assert 0.15 < temperature < 0.40


def test_purge_uses_settlement_time_not_just_decision_time():
    earlier = [record(i) for i in range(6)]
    later = [record(6)]
    purged = purged_before(earlier, later)
    assert [r["id"] for r in purged] == [0, 1, 2, 3, 4]
    assert all(r["_end"] + 60 < later[0]["_at"] for r in purged)


def test_loader_freezes_bytes_and_excludes_bad_timing_labels_and_duplicates(tmp_path):
    path = tmp_path / "rounds.jsonl"
    rows = [record(0), record(1, end_price=99.0), record(2, start=1e9 + 90 + 0.5)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    size = path.stat().st_size
    first, meta = load_rows(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record(3)) + "\n")
    frozen, frozen_meta = load_rows(path, size)
    assert [r["id"] for r in first] == [0]
    assert first == frozen and meta == frozen_meta
    assert meta["excluded"]["inconsistent_label"] == 1
    assert meta["excluded"]["incorrect_timing"] == 1
    fresh, _ = load_rows(path)
    assert [r["id"] for r in fresh] == [0, 3]


def test_payout_uses_final_pool_only_for_scoring_and_includes_own_stake():
    r = record(0, pools=[120.0, 80.0])
    r["feat"].update(pool_up=10.0, pool_dn=90.0)
    assert payout(r, True) == pytest.approx(1 + 0.9415 * 80 / 121)
    assert payout(r, True, final=False) == pytest.approx(1 + 0.9415 * 90 / 11)
    r["pools"] = None
    assert payout(r, True) is None


def test_score_does_not_fabricate_a_payout_for_missing_pools():
    rows = [record(0), record(1, pools=None)]
    result = score_sides(rows, [1, 0])
    assert result["calls"] == 2 and result["wins"] == 1
    assert result["payout_rows"] == 1
    assert result["roi"] == pytest.approx(0.9415 * 100 / 101)


def test_future_pool_changes_cannot_change_predictions_or_selection():
    rows = [record(i) for i in range(20)]
    model = UDDirectionModel(artifact())
    before = [model.predict(r["feat"]) for r in rows]
    for r in rows:
        r["pools"] = [1e6, 1.0]
        r["win"] = "DOWN"
    assert [model.predict(r["feat"]) for r in rows] == before


def test_block_bootstrap_is_reproducible_and_uses_distinct_hour_blocks():
    rows = [record(i * 80) for i in range(20)]
    values = [1.0, -1.0] * 10
    a = block_interval(rows, values)
    assert a == block_interval(rows, values)
    assert a["blocks"] == 20 and a["ci95"][0] < 0 < a["ci95"][1]
    assert block_interval(rows[:4], values[:4])["ci95"] is None


def test_probability_threshold_can_select_no_calls():
    rs = [record(i) for i in range(10)]
    stats = probabilities(rs, np.full(10, 0.52))
    assert stats["selected"]["calls"] == 0
    assert stats["selected"]["roi"] is None
    assert stats["all_directions"]["calls"] == 10


def test_existing_frozen_outputs_require_explicit_replacement(tmp_path):
    from types import SimpleNamespace
    from tools.evaluate_ud_model import check_outputs
    args = SimpleNamespace(model=tmp_path / "model.json", evaluation=tmp_path / "eval.json",
                           report=tmp_path / "report.md", replace=False)
    args.model.write_text("keep this frozen artifact", encoding="utf-8")
    with pytest.raises(FileExistsError):
        check_outputs(args)
    assert args.model.read_text(encoding="utf-8") == "keep this frozen artifact"
    args.replace = True
    check_outputs(args)
    args.report = args.model
    with pytest.raises(ValueError):
        check_outputs(args)


def test_forward_scoring_checks_snapshot_and_excludes_pre_registration_rows(tmp_path, capsys):
    from types import SimpleNamespace
    import tools.evaluate_ud_model as evaluation
    log = tmp_path / "rounds.jsonl"
    log.write_text(json.dumps(record(0)) + "\n", encoding="utf-8")
    _, metadata = load_rows(log)
    frozen = {**artifact(), "data": metadata, "forward_after": record(2)["_at"],
              "code_sha256": evaluation.hash_file(evaluation.ROOT / "bcbot" / "ud_model.py"),
              "evaluator_sha256": evaluation.hash_file(evaluation.__file__)}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(frozen), encoding="utf-8")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record(1)) + "\n")
        handle.write(json.dumps(record(2)) + "\n")
    evaluation.score_forward(SimpleNamespace(log=log, model=model_path))
    result = json.loads(capsys.readouterr().out)
    assert result["usable_forward_rows"] == 1
    assert result["status"] == "edge remains unproven"
    changed = log.read_text(encoding="utf-8").replace('"id": 0', '"id": 9', 1)
    log.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        evaluation.score_forward(SimpleNamespace(log=log, model=model_path))
