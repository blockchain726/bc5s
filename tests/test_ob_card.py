"""The order-book card of the Up/Down page (the owner, 2026-11-18: "ok, i want it"): the five markets' best-price
reading and its age, asked from the order-book service once a second (GET /life), instead of Binance spot's top 20
levels, a market that is not one of the six and that bc.game's price does not follow.  It is shown only."""
import asyncio
import json
import time

import pytest

from bcbot import obfeat as F
from bcbot.auto import OB_LIFE_STALE_S, AutoTrader
from tests.test_obmodels import _Http, rows
from tests.test_safety import FakeTrader, make_cfg
from tests.test_steady import _Req, _service

LIFE = {"t": 1.0, "r": 0.56, "age": 7.2, "move": 1.5, "share": 1.0,
        "per": {"bybit_perp": 0.7, "okx_perp": 0.6, "okx": -0.2, "binance_perp": 0.4, "coinbase": 0.9}}


def bot(tmp_path, url="http://127.0.0.1:1/ob"):
    return AutoTrader(FakeTrader(), make_cfg(tmp_path, auto_ob_url=url, auto_stake_usd=1.0))


def test_the_service_says_what_the_books_show_right_now(tmp_path, monkeypatch):
    S = _service(tmp_path, monkeypatch)
    now = time.time()
    for k, m in S.M.items():
        a = rows(now, 200, **({"bid_size": 1.0, "ask_size": 3.0} if k in ("okx", "bitfinex_perp") else
                              {"bid_size": 3.0, "ask_size": 1.0}))
        m.ring[:200], m.n, m.ready, m.book_at = a, 200, True, now
        m.bids, m.asks = {100.0: 1.0}, {100.2: 1.0}
    d = S.life_now(now)
    mixed = 0.5 * (0.172 + 0.160 - 0.114 + 0.068 + 0.053) / 0.567          # OKX spot the other way; Bitfinex perp not read
    assert d["r"] == pytest.approx(mixed, abs=1e-4) and 19.0 < d["age"] < 20.0 and d["share"] == 1.0
    assert d["per"] == {"bybit_perp": 0.5, "okx_perp": 0.5, "okx": -0.5, "binance_perp": 0.5, "coinbase": 0.5}
    asked = json.loads(asyncio.run(S.handle_ob(_Req(gap=1.0, vol=2.0, probe=1))).text)["life"]
    assert {k: d[k] for k in ("r", "age", "move", "share")} == asked     # the reading a round is decided on
    assert json.loads(asyncio.run(S.handle_life(_Req())).text)["r"] == d["r"]
    for m in S.M.values():
        m.n = 0                                                       # just started: no rows yet
    d = S.life_now(now)
    assert d["r"] is None and d["age"] is None and set(d["per"].values()) == {None}
    assert 'web.get("/life", handle_life)' in open("tools/ob_server.py", encoding="utf-8").read()


def test_the_card_shows_the_reading_and_its_age(tmp_path):
    a = bot(tmp_path)
    html = a.panel_html()
    assert "ORDER BOOKS &middot; 5 MARKETS" in html and "the order-book service does not answer" in html
    assert "TOP 20 LEVELS" not in html                                # Binance spot's book is no longer shown
    a._ob_life_take(LIFE)
    html = a.panel_html()
    assert "78% UP<small>for 7.2 s</small>" in html and 'class="c obup"' in html
    assert '<div class="obw up">mostly buyers &middot; 4 of 5 markets</div>' in html      # 4 of the 5 on that side
    assert "<span>at decision</span><span>&ndash;</span>" in html and 'class="spark"' not in html
    a._ob_life_take({"r": -0.3, "age": 30.0, "per": {"okx": -0.3, "coinbase": None}})
    html = a.panel_html()
    assert "65% DOWN<small>for 30 s+</small>" in html and 'class="c obdn"' in html
    assert "more sellers &middot; 1 of 1 markets</div>" in html and 'class="spark"' in html   # two points: a line
    a._ob_life_take({"r": 0.1, "age": 0.4, "per": {}})
    html = a.panel_html()
    assert "55% UP<small>for 0.4 s</small>" in html and '<div class="obw ">balanced</div>' in html
    assert "obup" not in html.split("ORDER BOOKS")[1].split("</div>")[0]
    assert len(a.ob_life_hist) == 3 and a.recs == {} and a.stats.get("called", 0) == 0      # shown only
    a._ob_life_take({"r": 0.195, "age": 1.2, "per": {}})              # the words go by the number that is shown
    assert "60% UP<small>for 1.2 s</small>" in a.panel_html() and '<div class="obw up">more buyers</div>' in a.panel_html()


def test_the_card_says_what_the_books_showed_at_the_decision_and_at_the_start(tmp_path):
    """The owner's case: "the OB was showing Buying, but when the bet round is starting that has already ended"."""
    a = bot(tmp_path)
    a.on = True
    a.bc_now = lambda: 1000.0                                         # bc.game's clock; the graph's times are the PC's
    now = time.time()
    rec = {"id": 5, "cutoff": 1002.9, "start": 1002.6, "end": 1007.6, "feat": {"dev_tick": 3.5, "vol60": 2.0},
           "ob": {"v": 5, "p": {}, "thr": {}, "life": {"r": 0.4, "age": 12.0, "move": 0.5, "share": 1.0}}}
    a.round, a.recs[5] = {"id": 5}, rec
    a._ob_life_take(LIFE, at=now - 2.0)
    html = a.panel_html()
    assert "<span>at decision</span><span>70% UP for 12.0 s</span>" in html
    assert "at start</span><span>&ndash;</span>" in html             # the window opens in 2.6 s
    a.bc_now = lambda: 1004.0                                         # 1.4 s after the window opened
    a._ob_life_take({"r": -0.3, "age": 0.6, "per": {}}, at=now - 1.0)                # the first reading from then on
    a._ob_life_take({"r": -0.5, "age": 1.5, "per": {}}, at=now - 0.1)
    html = a.panel_html()
    assert "at start</span><span>65% DOWN &middot; <b>turned</b></span>" in html and "75% DOWN<small>for 1.5 s" in html
    rec["ob"]["life"] = {"r": -0.22, "age": 2.1, "move": 0.0, "share": 1.0}          # it was DOWN at the decision too
    html = a.panel_html()
    assert "<span>at decision</span><span>61% DOWN for 2.1 s</span>" in html and "at start</span><span>65% DOWN</span>" in html
    a.round, a.recs[6] = {"id": 6}, {"id": 6, "cutoff": 1032.9}       # the next round, not decided yet: the rows stay
    assert "<span>at decision</span><span>61% DOWN for 2.1 s</span>" in a.panel_html()
    a.bc_now = lambda: 1100.0                                         # no reading was taken when that window opened
    assert "at start</span><span>&ndash;</span>" in a.panel_html()
    del rec["ob"]["life"]
    html = a.panel_html()
    assert "<span>at decision</span><span>&ndash;</span>" in html and "at start</span><span>&ndash;</span>" in html


def test_no_reading_is_said_plainly(tmp_path):
    a = bot(tmp_path)
    a._ob_life_take(LIFE, at=time.time() - OB_LIFE_STALE_S - 1)       # the service went silent
    html = a.panel_html()
    assert "the order-book service does not answer" in html and "78% UP" not in html
    a._ob_life_take({"r": None, "age": None, "per": {"okx": None}})   # it answers, but the books are not known
    assert "the books are not known right now" in a.panel_html() and len(a.ob_life_hist) == 1
    a._ob_life_take("not an answer")
    assert a.ob_life["r"] is None
    off = bot(tmp_path, url="")
    off._ob_life_take(LIFE)
    assert "the order-book service is switched off" in off.panel_html()


def test_the_bot_asks_the_service_for_the_reading(tmp_path):
    a = bot(tmp_path)
    a._life_http = _Http(LIFE)
    asyncio.run(a._ob_life_fetch())
    assert a._life_http.asked == [("http://127.0.0.1:1/life", None)]
    assert a.ob_life["r"] == 0.56 and a.ob_life["age"] == 7.2 and a.ob_life["per"]["okx"] == -0.2
    assert a._ob_http is None                                         # never on the connection of the round's question
    at = a.ob_life["at"]
    a._life_http = _Http(fail=OSError("connection refused"))
    asyncio.run(a._ob_life_fetch())                                   # no answer: nothing raised, the last one stays
    assert a.ob_life["at"] == at and len(a.ob_life_hist) == 1
    assert F.LIFE_MARKETS == tuple(LIFE["per"])
