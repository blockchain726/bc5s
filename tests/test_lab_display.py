"""Market Memory is visible in both dashboards without changing trading state."""
import asyncio
from datetime import datetime,timezone
import json
from types import SimpleNamespace

from aiohttp.test_utils import TestClient,TestServer

from bcbot.api import build_app,DASHBOARD
from bcbot.lab_display import read_status,panel_card


def status():
    return {"updated_utc":"2026-11-18T20:00:00+00:00","refit_hours":6,
            "models":[{"id":"version_one","role":"incumbent"},{"id":"version_two","role":"challenger"}],
            "results":{"version_one":{"selected":{"calls":10,"wins":6,"accuracy":.6,"net_flat_1":1.4}},
                       "version_two":{"selected":{"calls":4,"wins":0,"accuracy":0.,"net_flat_1":-4.}}}}


def write_status(tmp_path,value):
    p=tmp_path/"model_lab/status.json";p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(value),encoding="utf-8")


def test_status_reports_missing_bad_and_stale_data_without_breaking_dashboard(tmp_path):
    assert not read_status(tmp_path)["available"]
    write_status(tmp_path,status())
    now=datetime(2026,11,18,20,0,10,tzinfo=timezone.utc).timestamp()
    d=read_status(tmp_path,now)
    assert d["available"] and d["fresh"] and d["age_s"]==10
    assert not read_status(tmp_path,now+100)["fresh"]
    (tmp_path/"model_lab/status.json").write_text("partial json",encoding="utf-8")
    assert not read_status(tmp_path)["available"]


def test_panel_shows_market_memory_both_versions_zero_wins_and_paper_profit(tmp_path):
    write_status(tmp_path,status())
    d=read_status(tmp_path,datetime(2026,11,18,20,0,10,tzinfo=timezone.utc).timestamp())
    text=panel_card(d)
    assert "MARKET MEMORY" in text and "Current" in text and "Challenger" in text
    assert "60.0% &middot; 10" in text and "0.0% &middot; 4" in text
    assert "+1.40" in text and "-4.00" in text and "Retrains every 6 h" in text
    assert "Not updating" in panel_card({**d,"fresh":False})
    assert "Waiting" in panel_card({"available":False})


def test_research_display_escapes_version_names():
    d={"available":True,"fresh":True,"models":[{"id":"\"><script>bad()</script>","role":"incumbent"}]}
    text=panel_card(d)
    assert "<script>" not in text and "&lt;script&gt;" in text


def test_read_only_api_exposes_forward_records_without_auth_or_trading_actions(tmp_path):
    write_status(tmp_path,status())
    cfg=SimpleNamespace(api_token="",auto_log=tmp_path/"auto_rounds.jsonl")
    async def check():
        app=build_app(None,None,cfg)
        async with TestClient(TestServer(app)) as client:
            response=await client.get("/auto/model-lab")
            assert response.status==200
            data=await response.json()
            assert data["models"][0]["id"]=="version_one"
            assert data["results"]["version_one"]["selected"]["calls"]==10
            assert (await client.post("/auto/model-lab",json={})).status==405
    asyncio.run(check())


def test_web_dashboard_has_visible_research_card_and_periodic_refresh():
    assert 'id="market-memory"' in DASHBOARD and 'href="#market-memory"' in DASHBOARD
    assert "fetch('/auto/model-lab'" in DASHBOARD
    assert "setInterval(loadModelLab, 5000)" in DASHBOARD
    assert "Paper P&amp;L" in DASHBOARD and "Next eligible fit:" in DASHBOARD


def test_bc_page_panel_includes_research_without_changing_the_calling_model(tmp_path):
    from bcbot.auto import AutoTrader
    from tests.test_safety import FakeTrader,make_cfg
    write_status(tmp_path,status())
    auto=AutoTrader(FakeTrader(),make_cfg(tmp_path,auto_model="ob_xgb"))
    before=(auto.model,auto.armed,dict(auto.stats))
    text=auto.panel_html()
    assert "MARKET MEMORY" in text and "60.0% &middot; 10" in text and "Challenger" in text
    assert (auto.model,auto.armed,auto.stats)==before
