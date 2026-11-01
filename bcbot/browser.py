"""Launch a normal Edge/Chrome window with a dedicated profile and attach Playwright over CDP.

A browser launched by Playwright itself carries automation flags, and the bc.game trading widget
refuses to load under them. A regular browser process with --remote-debugging-port loads fine,
keeps the login session in its profile, and survives restarts of this bot.
"""
import asyncio
import logging
import os
import subprocess
import urllib.request

from playwright.async_api import Page, Playwright

from .config import Config

log = logging.getLogger(__name__)

TRADE_URL = "https://bc.game/trading/up-down"


def _cdp_alive(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1)
        return True
    except OSError:
        return False


async def _launch(cfg: Config) -> None:
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    args = [
        cfg.browser_path,
        f"--remote-debugging-port={cfg.cdp_port}",
        f"--user-data-dir={cfg.profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1440,900",
        # keep the page running at full speed when the window is behind others or minimised; otherwise Edge
        # throttles it and a click can reach bc.game after betting has closed
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        TRADE_URL,
    ]
    # Detach so Ctrl+C in this console doesn't kill the browser (and the login session with it).
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(args, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if _cdp_alive(cfg.cdp_port):
            return
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Browser did not open debugging port {cfg.cdp_port}")


async def open_trade_page(pw: Playwright, cfg: Config) -> Page:
    if _cdp_alive(cfg.cdp_port):
        log.info("Attaching to browser already running on port %s", cfg.cdp_port)
    else:
        log.info("Launching %s (profile: %s)", cfg.browser_path, cfg.profile_dir)
        await _launch(cfg)

    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cfg.cdp_port}")
    ctx = browser.contexts[0]
    page = next((p for p in ctx.pages if "/trading/up-down" in p.url), None)
    if page is None:
        page = next((p for p in ctx.pages if "bc.game" in p.url), None) or await ctx.new_page()
        await page.goto(TRADE_URL, wait_until="domcontentloaded")
    await page.bring_to_front()
    return page
