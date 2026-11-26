"""Where the order-book fits cut the recording (tools/train_ob.py).  The cuts count recorded decisions, not hours on
the clock.  The recording was stopped for 4.9 days (2026-11-19 02:50 .. 11-24 01:27 UTC), and "two thirds of the
time" then fell into the pause: the test part of fit 6 was the 2.75 h after it instead of a third, and DeepLOB's "last
72 h" held no decision of the fitting part, so DeepLOB was not fitted (IndexError) and its row stayed empty."""
import calendar
import os
import time

import numpy as np


def time_at_share(at, share: float) -> float:
    """The time by which `share` of the decisions had been made (`at`: their times, ascending)."""
    at = np.asarray(at, dtype=float)
    return float(at[min(max(int(len(at) * share), 0), len(at) - 1)])


def block_edges(at, parts: int) -> np.ndarray:
    """parts + 1 times that cut the decisions (`at` ascending) into blocks of the same number of decisions; a block is
    edge[i] <= t < edge[i + 1], and the last edge lies just after the last decision."""
    at = np.asarray(at, dtype=float)
    e = at[np.linspace(0, len(at) - 1, parts + 1).round().astype(int)].copy()
    e[-1] += 1e-6
    return e


def last_hours(at, hours: float, step: float) -> float:
    """The time from which the last `hours` of RECORDED decisions (one every `step` s) begin."""
    at = np.asarray(at, dtype=float)
    return float(at[max(0, len(at) - int(hours * 3600 / step))])


def hour_files(paths, since: float) -> list:
    """Of the hourly book files (<market>_<YYYYMMDD>_<HH>.bin, UTC) those that can hold a row from `since` on, in
    the order of time.  A file with another name is kept."""
    out = []
    for p in sorted(paths):
        try:
            start = calendar.timegm(time.strptime(os.path.basename(p)[-15:-4], "%Y%m%d_%H"))
        except ValueError:
            start = None
        if start is None or start + 3600 >= since:
            out.append(p)
    return out


def rows_since(a: np.ndarray, since: float) -> np.ndarray:
    """The rows with t >= since.  Without a copy when the rows are in the order of time (they are: the service writes
    them one after the other): a week of one market is 1.2 GB."""
    if not len(a):
        return a
    t = a["t"]
    if np.all(t[1:] >= t[:-1]):
        return a if t[0] >= since else a[np.searchsorted(t, since):]
    return a[t >= since]


def read_rows(files, dtype, since: float) -> np.ndarray:
    """The rows of the files, from `since` on, read into one array (no list of parts next to it)."""
    n = [os.path.getsize(f) // dtype.itemsize for f in files]
    a = np.empty(sum(n), dtype=dtype)
    i = 0
    for f, m in zip(files, n):
        got = np.fromfile(f, dtype=dtype, count=m)
        a[i:i + len(got)] = got
        i += len(got)
    return rows_since(a[:i], since)
