"""The order-book fits cut the recording by the number of recorded decisions, not by hours on the clock.  Fit 6
(2026-11-24 04:14 UTC) was the first fit after the recording had been stopped for 4.9 days: "two thirds of the time"
fell into the pause, the test part was the 2.75 h after it, and DeepLOB's "last 72 h" held no decision of the fitting
part: "ob_deeplob NOT fitted: IndexError", and its row on the page stayed empty."""
import importlib.util

import numpy as np

from bcbot import obfeat as F


def _mod():
    spec = importlib.util.spec_from_file_location("fitsplit_under_test", "tools/fitsplit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP = 2.0
PAUSE = 4.9 * 86400


def recording():
    """69.3 h recorded, a pause of 4.9 days, then 2.75 h: the recording fit 6 was made on."""
    a = 1_000_000.0 + np.arange(int(69.3 * 1800)) * STEP
    b = a[-1] + PAUSE + np.arange(int(2.75 * 1800)) * STEP
    return np.concatenate([a, b]), len(a), len(b)


def test_two_thirds_of_the_decisions_not_of_the_clock():
    S = _mod()
    at, n_a, n_b = recording()
    by_clock = at[0] + (at[-1] - at[0]) * 2 / 3                       # the old cut: inside the pause
    assert at[n_a - 1] < by_clock < at[n_a] and (at > by_clock).sum() == n_b       # the test part was the 2.75 h
    cut = S.time_at_share(at, 2 / 3)
    assert abs((at < cut).sum() / len(at) - 2 / 3) < 1e-3 and cut < at[n_a - 1]    # now a third of the decisions
    assert S.time_at_share(at, 0.0) == at[0] and S.time_at_share(at, 1.0) == at[-1]


def test_deeplobs_72_hours_are_recorded_hours():
    S = _mod()
    at, n_a, n_b = recording()
    old = at.max() - 72 * 3600                                        # the old start: inside the pause
    assert (at >= old).sum() == n_b                                   # ... so it had the 2.75 h only
    since = S.last_hours(at, 72.0, STEP)
    assert (at >= since).sum() == 72 * 1800 and since < S.time_at_share(at, 2 / 3)  # books for the fitting part too
    assert S.last_hours(at[:100], 72.0, STEP) == at[0]                # a short recording: all of it


def test_the_blocks_hold_the_same_number_of_decisions():
    S = _mod()
    at, n_a, n_b = recording()
    e = S.block_edges(at, 4)
    counts = [int(((at >= e[i]) & (at < e[i + 1])).sum()) for i in range(4)]
    assert sum(counts) == len(at) and max(counts) - min(counts) <= 2
    by_clock = np.linspace(at[0], at[-1] + 1e-6, 5)                   # the old blocks: two of the four nearly empty
    assert min(int(((at >= by_clock[i]) & (at < by_clock[i + 1])).sum()) for i in range(4)) == 0


def test_only_the_files_and_rows_that_are_needed_are_read(tmp_path):
    S = _mod()
    t0 = 1794806400.0                                                 # 2026-11-16 05:20 UTC
    names = []
    for h, hour in enumerate(("20261116_05", "20261116_06", "20261124_01")):
        a = np.zeros(10, dtype=F.ROW)
        a["t"] = (t0 if h < 2 else t0 + 8 * 86400) + h * 3600 + np.arange(10)
        a["bid"] = h
        p = tmp_path / f"okx_{hour}.bin"
        a.tofile(p)
        names.append(str(p))
    since = t0 + 3600 + 4.5                                           # in the second file
    files = S.hour_files(list(reversed(names)), since)
    assert files == names[1:]                                         # the first hour ends before it; in time order
    got = S.read_rows(files, F.ROW, since)
    assert len(got) == 15 and got["t"][0] == t0 + 3600 + 5 and list(np.unique(got["bid"])) == [1.0, 2.0]
    assert len(S.read_rows([], F.ROW, since)) == 0
    assert S.hour_files([str(tmp_path / "notes.bin")], since) == [str(tmp_path / "notes.bin")]
    a = np.zeros(6, dtype=F.ROW)
    a["t"] = [5, 1, 2, 9, 3, 7]                                       # not in time order: still the right rows
    assert sorted(S.rows_since(a, 4.0)["t"]) == [5, 7, 9] and len(S.rows_since(a[:0], 4.0)) == 0
    a["t"] = np.arange(6)
    assert S.rows_since(a, 2.0).base is a and S.rows_since(a, -1.0) is a          # no copy
