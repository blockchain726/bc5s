"""tools/record_books.py: the recorder of three more order books (Bybit spot, Deribit perpetual, Bitfinex spot).
Its rows must mean what the order-book service's rows mean, and it must stay out of the service's folder."""
import asyncio
import importlib.util
import sys

import numpy as np
import pytest

from bcbot import obfeat as F
from bcbot.ourprice import FEEDS
from tests.test_steady import _service


def _recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record_books.py", str(tmp_path / "extra")])
    spec = importlib.util.spec_from_file_location("record_books_under_test", "tools/record_books.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_three_markets_none_of_them_one_of_the_services_six(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)
    assert list(R.M) == ["bybit", "deribit_perp", "bitfinex"]
    assert not set(R.M) & set(F.SHARES) and set(R.M) <= set(FEEDS)        # the names our price's feeds use
    assert R.OUT == tmp_path / "extra"
    monkeypatch.setattr(sys, "argv", ["record_books.py"])                 # by default: its own folder under the
    spec = importlib.util.spec_from_file_location("record_books_default", "tools/record_books.py")   # service's
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUT.name == "extra" and mod.OUT.parent.name == "obrec"


def test_a_row_is_the_services_row(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)
    S = _service(tmp_path, monkeypatch)
    mine, theirs = R.Market("x"), S.Market("x")
    for m in (mine, theirs):
        for i in range(14):                                               # more than 10 levels: the best 10 are kept
            m.set("b", 100.0 - 0.5 * i, 1.0 + i)
            m.set("a", 100.5 + 0.5 * i, 2.0 + i)
        m.set("b", 99.5, 0.0)                                             # a level taken away
        m.ready = True
        m.book_at = 1000.0
        m.trade(True, 0.25); m.trade(True, 0.5); m.trade(False, 1.0)
    a, b = mine.row(1000.2), theirs.row(1000.2)
    for name in F.ROW.names:
        assert np.array_equal(a[name], b[name], equal_nan=True), name
    assert (a["bid"], a["ask"]) == (100.0, 100.5) and a["bp"][1] == pytest.approx(1.0) and a["bs"][0] == 1.0
    assert (a["buy"], a["sell"], a["nb"], a["ns"]) == (0.75, 1.0, 2, 1) and a["age"] == pytest.approx(0.2)
    again = mine.row(1000.3)                                              # trades are counted once
    assert (again["buy"], again["sell"], again["nb"]) == (0.0, 0.0, 0)
    empty = R.Market("y").row(5.0)                                        # no book yet: unknown, never zero
    assert np.isnan(empty["bid"]) and np.isnan(empty["bs"]).all() and empty["age"] == 9999.0


def test_bybit_spot_book_and_trades(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)
    R.bybit({}, {"topic": "orderbook.50.BTCUSDT", "type": "snapshot",
                 "data": {"b": [["64000.1", "0.5"], ["64000.0", "1.5"]], "a": [["64000.2", "0.25"], ["64000.4", "2"]]}})
    R.bybit({}, {"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"b": [["64000.1", "0"]], "a": [["64000.3", "1"]]}})
    R.bybit({}, {"topic": "publicTrade.BTCUSDT", "data": [{"S": "Buy", "v": "0.01"}, {"S": "Sell", "v": "0.03"}]})
    r = R.M["bybit"].row(R.M["bybit"].book_at + 0.05)
    assert (r["bid"], r["ask"]) == (64000.0, 64000.2) and r["bs"][0] == 1.5 and r["as_"][1] == 1.0
    assert r["buy"] == pytest.approx(0.01) and r["sell"] == pytest.approx(0.03)
    R.bybit({}, {"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "data": {"b": [["63990", "1"]], "a": [["63991", "1"]]}})
    assert list(R.M["bybit"].bids) == [63990.0]                           # a new snapshot starts the book again


def test_deribit_sizes_are_turned_into_btc(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)
    R.deribit({}, {"params": {"channel": "book.BTC-PERPETUAL.none.10.100ms",
                              "data": {"bids": [[64000.0, 32000.0], [63999.5, 6400.0]], "asks": [[64000.5, 12800.1], [64001.0, 0.0]]}}})
    R.deribit({}, {"params": {"channel": "trades.BTC-PERPETUAL.100ms",
                              "data": [{"direction": "buy", "amount": 6400.0, "price": 64000.0},
                                       {"direction": "sell", "amount": 640.0, "price": 64000.0}]}})
    R.deribit({}, {"jsonrpc": "2.0", "id": 1, "result": ["book.BTC-PERPETUAL.none.10.100ms"]})   # the answer to subscribe
    r = R.M["deribit_perp"].row(R.M["deribit_perp"].book_at + 0.05)
    assert (r["bid"], r["ask"]) == (64000.0, 64000.5)
    assert r["bs"][0] == pytest.approx(0.5) and r["bs"][1] == pytest.approx(6400 / 63999.5, rel=1e-6)
    assert r["as_"][0] == pytest.approx(12800.1 / 64000.5, rel=1e-6) and np.isnan(r["as_"][1])   # a level of 0: not there
    assert r["buy"] == pytest.approx(0.1) and r["sell"] == pytest.approx(0.01)


def test_bitfinex_spot_book_and_trades(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)
    st = {}
    R.bitfinex(st, {"event": "subscribed", "channel": "book", "chanId": 7})
    R.bitfinex(st, {"event": "subscribed", "channel": "trades", "chanId": 9})
    R.bitfinex(st, [7, [[64000.0, 2, 1.5], [63999.0, 1, 0.5], [64006.0, 1, -0.75]]])     # snapshot: + bids, - asks
    R.bitfinex(st, [7, [63999.0, 0, 1]])                                                # count 0: the level is gone
    R.bitfinex(st, [7, [64005.0, 1, -0.2]])
    R.bitfinex(st, [9, "te", [1, 1794990000000, 0.3, 64005.0]])                         # a taker bought 0.3
    R.bitfinex(st, [9, "tu", [1, 1794990000000, 0.3, 64005.0]])                         # the same trade again: not counted
    R.bitfinex(st, [9, "te", [2, 1794990000001, -0.1, 64000.0]])
    r = R.M["bitfinex"].row(R.M["bitfinex"].book_at + 0.05)
    assert (r["bid"], r["ask"]) == (64000.0, 64005.0) and r["bs"][0] == 1.5 and np.isnan(r["bs"][1])
    assert r["as_"][0] == pytest.approx(0.2) and r["ap"][1] == pytest.approx(1.0)
    assert r["buy"] == pytest.approx(0.3) and r["sell"] == pytest.approx(0.1)


def test_it_writes_hourly_files_in_its_own_folder_and_stops(tmp_path, monkeypatch):
    R = _recorder(tmp_path, monkeypatch)

    async def no_feed(name):                                              # no network in a test
        await asyncio.sleep(3600)

    monkeypatch.setattr(R, "feed", no_feed)
    asyncio.run(R.main(0.45))
    files = sorted(p.name for p in (tmp_path / "extra").glob("*.bin"))
    assert [f.split("_20")[0] for f in files] == ["bitfinex", "bybit", "deribit_perp"]
    a = np.fromfile(tmp_path / "extra" / files[0], dtype=F.ROW)
    assert 3 <= len(a) <= 6 and np.isnan(a["bid"]).all() and (np.diff(a["t"]) > 0.05).all()
    assert (tmp_path / "extra" / "books_meta.json").exists() and not list(tmp_path.glob("*.bin"))
