"""Leaderboard: every model's honest record, ranked.  For models fixed on 2026-10-31 or later only the rounds after
they were fixed count (their forward record); older models use all logged rounds.  Reads the running bot's API when
it is up, else the round log.   Run: .venv\\Scripts\\python.exe tools\\leaderboard.py"""
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bcbot.auto import AutoTrader, MODEL_REGISTERED  # noqa: E402

LABELS = {**AutoTrader.MODELS, "mhh_hi": "MHH confident", "pool": "Pool thin side"}
BREAK_EVEN = 0.5155


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def stats_from_api():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/auto", timeout=5) as r:
            return json.load(r)["stats"], "the running bot"
    except Exception:
        return None, None


def leaderboard(stats, now=None):
    now = now or time.time()
    rows = []
    for name, label in LABELS.items():
        fwd = name in MODEL_REGISTERED
        key = f"{name}_fwd" if fwd else name
        n, k = stats.get(f"{key}_called", 0), stats.get(f"{key}_correct", 0)
        if not n:
            continue
        p, lo, hi = wilson(k, n)
        hours = max((now - MODEL_REGISTERED[name]) / 3600, 0.01) if fwd else None
        rows.append({"model": name, "label": label, "won": k, "calls": n, "p": p, "lo": lo, "hi": hi,
                     "scope": f"since fixed, {hours:.0f} h" if fwd else "all logged rounds",
                     "calls_h": n / hours if fwd else None, "edge_proven": lo > BREAK_EVEN and n >= AutoTrader.GATE_MIN_N})
    rows.sort(key=lambda r: (r["lo"], r["p"]), reverse=True)        # by the lower end of the 95% range
    return rows


def render(rows, source):
    out = [f"LEADERBOARD ({source}; ranked by the lower end of the 95% range; break-even {100*BREAK_EVEN:.2f}%)",
           f"{'model':18} | {'won/calls':>11} | {'win%':>6} | {'95% range':>11} | {'calls/h':>7} | scope"]
    for r in rows:
        ch = f"{r['calls_h']:7.1f}" if r["calls_h"] else "      -"
        flag = " <- proven" if r["edge_proven"] else ""
        out.append(f"{r['label']:18} | {r['won']:5d}/{r['calls']:5d} | {100*r['p']:5.1f}% | {100*r['lo']:4.1f}-{100*r['hi']:4.1f} | {ch} | {r['scope']}{flag}")
    return "\n".join(out)


if __name__ == "__main__":
    stats, source = stats_from_api()
    if stats is None:
        print("the bot's API is not up; start the bot (run.bat --auto) and try again")
        sys.exit(1)
    print(render(leaderboard(stats), source))
