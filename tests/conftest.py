"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_new_price_recipe(tmp_path, monkeypatch):
    """Tests run without the new price's recipe unless they write one: the repo may hold a fitted
    bcbot/fair_price.json, and a test must not depend on it.  A test that wants one writes tmp_path/fair_price.json."""
    import bcbot.ourprice as op
    from bcbot.auto import MODEL_REGISTERED
    monkeypatch.setattr(op, "FAIR_FILE", tmp_path / "fair_price.json")
    monkeypatch.setenv("AUTO_OB_URL", "")                # no test asks the order-book service of this PC
    yield
    MODEL_REGISTERED.pop("fade_fair", None)              # set when a trader loads a recipe
