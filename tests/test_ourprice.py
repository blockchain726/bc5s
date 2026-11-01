"""Our price: the mix is the weighted average of the exchanges' level-adjusted curves the chart shows."""
from collections import deque

from bcbot.ourprice import MIX, MidFeed, OurPrice, _bitfinex, depth_mid


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


def test_depth_mid_ignores_one_small_top_order():
    book = {"b": {100.0: 0.01, 99.0: 5.0}, "a": {101.0: 5.0}}
    before = depth_mid(book, 0.5)                 # bids 0.01 @100 + 0.49 @99, asks 0.5 @101
    assert abs(before - ((0.01 * 100 + 0.49 * 99) / 0.5 + 101) / 2) < 1e-9
    book["b"][100.9] = 0.02                       # a tiny bid at the top: the top-of-book mid jumps $0.45,
    assert (100.9 + 101) / 2 - (100 + 101) / 2 > 0.4
    assert abs(depth_mid(book, 0.5) - before) < 0.05     # the 0.5 BTC mid moves about 4 cents
    assert depth_mid({"b": {100.0: 0.1}, "a": {101.0: 5.0}}, 0.5) is None     # too thin: unknown


def test_bitfinex_parser_resets_on_snapshot():
    f = MidFeed("bitfinex_perp", "", [], None)
    _bitfinex(f, [1, [[100.0, 1, 1.0], [110.0, 1, -1.0]]])          # snapshot: bid 1 @100, ask 1 @110
    assert f.mids[-1][1] == 105.0
    _bitfinex(f, [1, [100.0, 0, 1]])                                 # the bid is removed: one side empty
    _bitfinex(f, [1, [[98.0, 1, 2.0], [108.0, 1, -2.0]]])            # a new snapshot replaces the old book
    assert f.mids[-1][1] == 103.0
