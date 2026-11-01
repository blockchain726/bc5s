"""Our price: the mix is the weighted average of the exchanges' level-adjusted curves the chart shows."""
from collections import deque

from bcbot.ourprice import MIX, MidFeed, OurPrice


def _feeds(t0):
    feeds = {}
    for i, k in enumerate(MIX):
        f = MidFeed(k, "", [], None)
        for j in range(200):                      # 20 s of prices at 0.1 s, each exchange at its own level
            f.mids.append((t0 + j * 0.1, 70000.0 + 100 * i + (j // 10)))
        feeds[k] = f
    return feeds


def test_exchange_series_matches_the_mix():
    t0 = 1_700_000_000.0
    feeds = _feeds(t0)
    ticks = deque((t0 + j * 0.5, 70003.0 + (j // 2)) for j in range(40))      # bc.game at its own level
    our = OurPrice(feeds, ticks)
    a, b = t0 + 5, t0 + 15
    ex = our.exchange_series(a, b, step=0.5)
    assert set(ex) == set(MIX)
    mix = dict((t, v) for t, v in our.series(a, b, step=0.5))
    for t, v in mix.items():
        w = sum(MIX[k][0] for k in ex)
        want = sum(MIX[k][0] * dict((p[0], p[1]) for p in ex[k])[t] for k in ex) / w
        assert abs(v - want) < 1e-6
    # every exchange is moved to bc.game's level: within a few dollars of bc.game, not $100-500 away
    for k, pts in ex.items():
        assert pts and all(abs(p[1] - 70000) < 30 for p in pts), k
