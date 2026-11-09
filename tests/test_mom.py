"""The 'mom' model: the sign of our exchange-mix 5 s momentum at the decision snapshot (Up/Down bot only)."""
from bcbot.auto import AutoTrader, _mom_call


def test_mom_call_sign():
    assert _mom_call({"our_mom5": 1.25}) == "UP"
    assert _mom_call({"our_mom5": -0.4}) == "DOWN"


def test_mom_call_no_signal():
    assert _mom_call({"our_mom5": 0}) is None
    assert _mom_call({"our_mom5": None}) is None
    assert _mom_call({}) is None
    assert _mom_call(None) is None
    assert _mom_call({"error": "boom", "our_mom5": 2.0}) is None


def test_mom_is_a_caller():
    assert "mom" in AutoTrader.CALLERS and "mom" in AutoTrader.MODELS


def test_book_call_strong_only():
    from bcbot.auto import _book_call, BOOK_MIN
    assert BOOK_MIN == 0.7
    assert _book_call({"obi20": 0.85}) == "UP"
    assert _book_call({"obi20": -0.7}) == "DOWN"
    assert _book_call({"obi20": 0.69}) is None
    assert _book_call({"obi20": -0.2}) is None
    assert _book_call({}) is None and _book_call(None) is None
    assert "book" in AutoTrader.CALLERS


def test_bookmom_needs_book_and_trend_to_agree():
    from bcbot.auto import _bookmom_call
    assert _bookmom_call({"obi20": 0.8, "our_mom5": 1.2}) == "UP"
    assert _bookmom_call({"obi20": -0.9, "our_mom5": -0.5}) == "DOWN"
    assert _bookmom_call({"obi20": 0.8, "our_mom5": -1.2}) is None      # buying book, falling price: skip
    assert _bookmom_call({"obi20": -0.8, "our_mom5": 0.3}) is None
    assert _bookmom_call({"obi20": 0.5, "our_mom5": 2.0}) is None       # book not strong enough
    assert _bookmom_call({"obi20": 0.8, "our_mom5": 0}) is None
    assert "bookmom" in AutoTrader.CALLERS


def test_momvol_only_when_volatile():
    from bcbot.auto import _momvol_call
    assert _momvol_call({"vol60": 3.0, "our_mom5": 1.0}) == "UP"
    assert _momvol_call({"vol60": 3.0, "our_mom5": -1.0}) == "DOWN"
    assert _momvol_call({"vol60": 1.0, "our_mom5": 1.0}) is None
    assert _momvol_call({"our_mom5": 1.0}) is None


def test_revert_quiet_only():
    from bcbot.auto import _revert_call
    assert _revert_call({"vol60": 1.0, "bc_mom2": 1.2}) == "DOWN"
    assert _revert_call({"vol60": 1.0, "bc_mom2": -0.8}) == "UP"
    assert _revert_call({"vol60": 1.0, "bc_mom2": 0.3}) is None       # move too small
    assert _revert_call({"vol60": 3.0, "bc_mom2": 1.2}) is None       # busy market
    assert _revert_call({}) is None


def test_overshoot_rules():
    from bcbot.auto import _overshoot_call
    # bc.game moved up $1 and sits $3 above the exchanges, quiet market: bet DOWN
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 3.0, "vol60": 1.0, "obi20": 0.0}) == "DOWN"
    assert _overshoot_call({"bc_mom2": -1.0, "dev_tick": -3.0, "vol60": 1.0, "obi20": 0.0}) == "UP"
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": -3.0, "vol60": 1.0}) is None    # exchanges moved too
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 3.0, "vol60": 3.0}) is None     # busy market
    # with the book: any market, book against the move
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 1.5, "vol60": 3.0, "obi20": -0.6}, with_book=True) == "DOWN"
    assert _overshoot_call({"bc_mom2": 1.0, "dev_tick": 1.5, "vol60": 3.0, "obi20": 0.6}, with_book=True) is None


def test_runs_today_counts_only_rounds_since_midnight(tmp_path):
    import time
    from tests.test_safety import make_cfg, FakeTrader
    from bcbot.auto import AutoTrader
    a = AutoTrader(FakeTrader(), make_cfg(tmp_path))
    now = time.time()
    lt = time.localtime(now)
    day0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    for won in (False, False, False, True):              # yesterday: a run of 3
        a._run_step("overshoot", won, day0 - 3600)
    for won in (False, True, False, False):              # today: a run of 1, then 2 still running
        a._run_step("overshoot", won, day0 + 60)
    assert a.runs["overshoot"]["lengths"] == {3: 1, 1: 1} and a.runs["overshoot"]["current"] == 2
    td = a.runs_today["overshoot"]
    assert td["lengths"] == {1: 1} and td["current"] == 2 and td["bets"] == 4


def test_jump_follows_a_big_10s_move():
    from bcbot.auto import _jump_call
    assert _jump_call({"our_mom10": 12.0}) == "UP"
    assert _jump_call({"our_mom10": -18.2}) == "DOWN"
    assert _jump_call({"our_mom10": 9.9}) is None
    assert _jump_call({}) is None
    assert _jump_call({"our_mom10": 15.0, "error": "x"}) is None


def test_trees_mid_is_symmetric_and_only_in_medium_markets(monkeypatch):
    import bcbot.auto as a

    class Booster:
        def predict(self, X):                   # P(UP) from the first signed feature (dev), mirrored rows flip it
            return [0.5 + 0.1 * (1 if row[0] > 0 else -1) for row in X]
    order = list(a.UD_SIGNED) + list(a.UD_UNSIGNED) + ["pay_lean", "pool_total", "abs_dev", "abs_mom2", "same_way"]
    monkeypatch.setattr(a, "_ud_booster", lambda: {"booster": Booster(), "features": order})
    f = {"dev": 2.0, "vol60": 2.0, "pay_up": 1.94, "pay_dn": 1.94}
    assert a._treesmid_call(f) == "UP"                          # 0.6 x 1.94 = 1.16 >= 1.03
    assert a._treesmid_call(dict(f, dev=-2.0)) == "DOWN"        # the mirror image
    assert a._treesmid_call(dict(f, vol60=1.0)) is None         # calm
    assert a._treesmid_call(dict(f, vol60=3.5)) is None         # busy
    assert a._treesmid_call(dict(f, pay_up=1.5, pay_dn=1.5)) is None   # 0.6 x 1.5 = 0.9: no value at these payouts

