"""The local results remain available while the trading site cannot connect."""
import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient,TestServer

from main import connect_trading
from bcbot.api import build_app
from bcbot.auto import AutoTrader


def test_region_restriction_waits_without_login_reload_or_trading():
    calls=[]
    async def evaluate(script):
        calls.append("read_page");return True
    async def forbidden():
        raise AssertionError("restricted page must not trigger login or trading")
    async def sleep(seconds):
        assert seconds==30
        raise asyncio.CancelledError()
    trader=SimpleNamespace(page=SimpleNamespace(evaluate=evaluate),ensure_ready=forbidden,login=forbidden)
    auto=SimpleNamespace(start=forbidden)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(connect_trading(trader,auto,None,SimpleNamespace(no_login=False,auto=True),sleep))
    assert calls==["read_page"] and "region restriction" in auto.startup_problem


def test_ready_site_starts_normally_and_clears_dashboard_message():
    calls=[]
    async def evaluate(script):return False
    async def ensure():calls.append("ready")
    async def login():calls.append("login")
    async def loaded():
        calls.append("loaded");return SimpleNamespace(line=lambda:"ready")
    async def start():calls.append("start")
    trader=SimpleNamespace(page=SimpleNamespace(evaluate=evaluate),ensure_ready=ensure,login=login,wait_loaded=loaded)
    auto=SimpleNamespace(start=start)
    asyncio.run(connect_trading(trader,auto,None,SimpleNamespace(no_login=False,auto=True)))
    assert calls==["ready","login","loaded","start"] and auto.startup_problem is None


def test_history_is_restored_once_before_connection_attempts():
    auto=AutoTrader.__new__(AutoTrader);calls=[]
    auto._load_state=lambda:calls.append("state")
    auto._load_log=lambda:calls.append("history")
    auto._check_start=lambda:calls.append("checks")
    auto.restore_history();auto.restore_history()
    assert calls==["state","history","checks"]


def test_uninitialized_connection_cannot_arm_or_place_but_can_disarm(tmp_path):
    calls=[]
    auto=SimpleNamespace(startup_problem="Trading site is unavailable",LIMITS=(),
                         disarm=lambda *a,**k:calls.append("disarm"),summary=lambda:{"state":"PAPER"})
    cfg=SimpleNamespace(api_token="test-only-token",auto_log=tmp_path/"auto_rounds.jsonl")
    async def check():
        async with TestClient(TestServer(build_app(None,auto,cfg))) as client:
            headers={"Authorization":"Bearer test-only-token"}
            for path,body in (("/trade",{"side":"up","amount":1}),("/auto",{"arm":True})):
                response=await client.post(path,json=body,headers=headers)
                assert response.status==409 and "unavailable" in (await response.json())["error"]
            response=await client.post("/auto",json={"live":False},headers=headers)
            assert response.status==200 and calls==["disarm"]
    asyncio.run(check())
