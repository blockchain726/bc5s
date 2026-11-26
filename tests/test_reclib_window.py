"""The answer a fit learns must be the round's result as bc.game decides it.  Found 2026-11-25: bc.game's first
price is the last tick at or before the betting cutoff and its last price the last tick at or before cutoff + 5 s
(7,250 logged rounds against the recorded ticks), but the made-up decisions of the fits read the last tick at or
before decision + 2.52 s / + 7.55 s: the first price was one tick early in 32-75% of them."""
import importlib.util

import numpy as np


def _lib():
    spec = importlib.util.spec_from_file_location("reclib_under_test", "tools/reclib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(L, spacing=0.529, n=2000, seed=4):
    rng = np.random.default_rng(seed)
    R = L.Rec()
    R.bc_t = 1000.0 + spacing * np.arange(n) + rng.uniform(-0.01, 0.01, n)
    R.bc_p = 60000.0 + np.cumsum(rng.normal(0, 2.0, n))
    R.clock = np.array([[0.0, 0.08]])
    R.bc_off = np.array([[0.0, 0.08]])
    return R


def test_the_window_is_bc_games_rule():
    L = _lib()
    R = _rec(L)
    d, tick_i, move, at = R.decisions(step=2.0, cut_lead=L.CUT_LEAD)
    assert len(at) > 300
    cutoff = at + L.CUT_LEAD
    i0 = np.searchsorted(R.bc_t, cutoff, side="right") - 1            # the last tick at or before the cutoff
    i1 = np.searchsorted(R.bc_t, cutoff + 5.0, side="right") - 1      # ... and at or before cutoff + 5 s
    assert np.allclose(move, R.bc_p[i1] - R.bc_p[i0])
    assert np.all(R.bc_t[i0] <= cutoff) and np.all(R.bc_t[i0 + 1] > cutoff)
    assert np.allclose(at - R.bc_t[tick_i], 0.27)                     # the decision itself is where it was


def test_the_two_fixed_leads_took_an_earlier_tick():
    """What was wrong: with ticks 0.529 s apart the old window's first tick is one tick before bc.game's."""
    L = _lib()
    R = _rec(L)
    _, _, old, at = R.decisions(step=2.0)
    _, _, new, at2 = R.decisions(step=2.0, cut_lead=L.CUT_LEAD)
    n = min(len(at), len(at2))
    assert np.allclose(at[:n], at2[:n])
    j0 = np.searchsorted(R.bc_t, at[:n] + 2.52, side="right") - 1
    i0 = np.searchsorted(R.bc_t, at[:n] + L.CUT_LEAD, side="right") - 1
    assert np.mean(j0 == i0 - 1) > 0.9
    assert np.mean(np.sign(old[:n]) != np.sign(new[:n])) > 0.03       # and the answer differs in part of them


def test_without_cut_lead_nothing_changes():
    """The frozen forward test (logs/obrec/fwd) was fitted with the two leads: its decisions must stay as they were."""
    L = _lib()
    R = _rec(L)
    d, tick_i, move, at = R.decisions(step=2.0)
    assert np.allclose(move, R.bc_move(at + 2.52, at + 7.55), equal_nan=True)


def test_the_fit_uses_the_rule():
    src = open("tools/train_ob.py", encoding="utf-8").read()
    assert "R.decisions(step=STEP, cut_lead=L.CUT_LEAD)" in src
