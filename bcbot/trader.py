"""Drives the bc.game Up/Down trading panel: login, status, and placing UP / DOWN orders."""
import asyncio
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from .browser import TRADE_URL
from .config import Config

log = logging.getLogger(__name__)

SIGNIN_URL = "https://bc.game/login/signin"

SEL_DIRECTION = "#up-down-direction-buttons"
SEL_AMOUNT = "#available-balance-input input"
SEL_HEADER_SIGNIN = "#track-header_signin"
SEL_PASSWORD_TAB = "#track-tab_click-signin_method-1"
SEL_SIGNIN_SUBMIT = "#track-signin_submit"
ORDER_API = "/updown/order/create"

SIDES = {"up": "up", "u": "up", "dn": "down", "down": "down", "d": "down"}
BUTTON_LABEL = {"up": "Up", "down": "Down"}

STATE_JS = r"""
() => {
  const txt = el => el ? el.innerText.replace(/\s+/g, ' ').trim() : '';
  const btns = [...document.querySelectorAll('#up-down-direction-buttons button')];
  const box = document.querySelector('#available-balance-input');
  const signin = document.querySelector('#track-header_signin');
  const modal = [...document.querySelectorAll('.detrade-modal')].find(m => m.getBoundingClientRect().width > 0);
  return {
    ready: btns.length === 2,
    content: txt(document.querySelector('#up-down-content')),
    panel: box ? txt(box.parentElement) : '',
    wallet: txt(document.querySelector('#track-header_currency_select')),
    modal: modal ? txt(modal).slice(0, 60) : null,
    signinVisible: !!signin && signin.getBoundingClientRect().width > 0,
    upEnabled: !!btns[0] && !btns[0].disabled,
    downEnabled: !!btns[1] && !btns[1].disabled,
    rooms: [...document.querySelectorAll('button')]
      .map(b => ({text: txt(b), active: b.classList.contains('bg-colorful12')}))
      .filter(r => /^\d+s \$[\d.,]+-[\d.,]+$/.test(r.text)),
  };
}
"""

ROOM_RE = re.compile(r"^(\d+)s \$([\d.,]+)-([\d.,]+)$")


class TradeError(Exception):
    pass


class TooLate(TradeError):
    """The order could not be sent early enough; nothing was clicked."""


class OrderUnknown(TradeError):
    """The button was clicked but no order response came back: the order MAY have been placed.  The caller must
    treat it as placed (and lost) until the site's order history or the balance says otherwise."""


@dataclass
class Room:
    period: str
    lo: float
    hi: float
    active: bool

    @property
    def label(self) -> str:
        return f"{self.period}s ${self.lo:g}-{self.hi:g}"


@dataclass
class Status:
    logged_in: bool
    account: str           # "REAL" or "DEMO"
    balance: float | None
    currency: str          # currency the amount box is in, e.g. USDT or NOKFIAT
    phase: str             # "open" (orders accepted), "locked", "settling", "loading"
    seconds_left: int | None
    up_payout: int | None  # percent, e.g. 225 means 2.25x
    down_payout: int | None
    price: float | None
    room: str | None
    dialog: str | None     # a site dialog covering the panel (must be closed by hand)

    def line(self) -> str:
        bal = f"{self.balance:g} {self.currency}" if self.balance is not None else "?"
        secs = f" {self.seconds_left}s left" if self.phase == "open" and self.seconds_left is not None else ""
        text = (f"[{self.account}] balance {bal} | room {self.room or '?'} | {self.phase}{secs} | "
                f"UP {self.up_payout or '?'}%  DN {self.down_payout or '?'}% | BTC {self.price or '?'}")
        if self.dialog:
            text += f"\n  !! Dialog open in browser: '{self.dialog}...' - close it there before trading"
        return text


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _search(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _fmt_amount(amount: float) -> str:
    return f"{amount:.8f}".rstrip("0").rstrip(".")


class Trader:
    def __init__(self, page: Page, cfg: Config, reopen=None):
        self.page = page
        self.cfg = cfg
        self._reopen = reopen                # async () -> Page: relaunch / re-attach the browser window
        self.page_listeners: list = []       # async (page) -> None, called after the page was replaced
        self._lock = asyncio.Lock()
        self._reopen_lock = asyncio.Lock()
        self._prices: dict = {}
        self._prices_at = float("-inf")
        self.last_rtt: float | None = None   # round trip of the last site API call made from the page (seconds)

    # ---------- page state ----------

    def page_gone(self) -> bool:
        browser = self.page.context.browser
        return self.page.is_closed() or (browser is not None and not browser.is_connected())

    async def _recover(self) -> None:
        """The window (or the whole browser) was closed: open it again; the login is kept in the profile."""
        async with self._reopen_lock:
            if not self.page_gone() or self._reopen is None:
                return
            log.warning("Browser window was closed: reopening it")
            self.page = await self._reopen()
            for listener in self.page_listeners:
                await listener(self.page)

    async def widget_crashed(self) -> bool:
        """The Up/Down panel replaced by "Something went wrong! Retry / Reload" (e.g. after a wallet currency switch)."""
        try:
            return await self.page.evaluate(
                "() => !document.querySelector('#up-down-direction-buttons') && "
                "!!document.body && document.body.innerText.includes('Something went wrong')")
        except Exception:                    # page still loading / navigating: not a crash
            return False

    async def ensure_ready(self) -> None:
        if self.page_gone():
            await self._recover()
        if "/trading/up-down" not in self.page.url:
            await self.page.goto(TRADE_URL, wait_until="domcontentloaded")
        elif await self.widget_crashed():
            log.warning("Trading panel shows 'Something went wrong': reloading the page")
            await self.page.reload(wait_until="domcontentloaded")
        buttons = self.page.locator(f"{SEL_DIRECTION} button").first
        try:
            await buttons.wait_for(timeout=60_000)
        except PlaywrightTimeout:
            log.warning("Trading panel did not load; reloading page")
            await self.page.reload(wait_until="domcontentloaded")
            await buttons.wait_for(timeout=60_000)

    async def _raw_state(self) -> dict:
        return await self.page.evaluate(STATE_JS)

    async def status(self, light: bool = False) -> Status:
        """The panel's state.  light=True only reads the page (no reload / navigation), for the decision path."""
        if not light:
            await self.ensure_ready()
        s = await self._raw_state()
        content, panel = s["content"], s["panel"]
        # Demo shows "Available 100.05 USDT(Demo)" under the box; a real account shows its balance
        # in the header wallet ("NOK 0.00 Deposit").
        bal = _search(r"Available ([\d.,]+)", panel) or _search(r"(\d[\d.,]*)", s["wallet"])
        if bal and not (0 <= _num(bal) < 1e6):     # a layout change gave a number that is not a balance
            bal = None
        if s["upEnabled"] or s["downEnabled"]:
            phase = "open"
        elif "No More Orders" in content:
            phase = "locked"
        elif "Money Distributed" in content:
            phase = "settling"
        else:
            phase = "loading"
        secs, up, dn, price = (_search(p, content) for p in
                               # wide layout "Up Wins 187%", narrow layout "Up Wins $0.99 ($1.87) 187%"
                               (r"(\d+) Sec", r"Up Wins[^%]*?(\d+)%", r"Down Wins[^%]*?(\d+)%", r"Live ([\d,]+\.\d+)"))
        active = next((r["text"] for r in s["rooms"] if r["active"]), None)
        return Status(
            logged_in=not s["signinVisible"],
            account="DEMO" if "(Demo)" in panel or s["signinVisible"] else "REAL",
            balance=_num(bal) if bal else None,
            currency=_search(r"Amount \(([^)]+)\)", panel) or "",
            phase=phase,
            seconds_left=int(secs) if secs else None,
            up_payout=int(up) if up else None,
            down_payout=int(dn) if dn else None,
            price=_num(price) if price else None,
            room=active,
            dialog=s["modal"],
        )

    async def wait_loaded(self, timeout: float = 20) -> Status:
        """Status once the panel has its live data (balance and price arrive a few seconds after render)."""
        deadline = time.monotonic() + timeout
        while True:
            s = await self.status()
            if (s.price is not None and s.balance is not None) or time.monotonic() > deadline:
                return s
            await asyncio.sleep(0.5)

    async def usd_rate(self, currency: str) -> float:
        """USD value of 1 unit of `currency`, from bc.game's own price table (cached 60s)."""
        if time.monotonic() - self._prices_at > 60:
            t0 = time.monotonic()
            body = await self.page.evaluate(
                "fetch('/api/game/support/system/conf/usdPrice').then(r => r.json())")
            self.last_rtt = round(time.monotonic() - t0, 3)    # a free probe of the site's round trip
            self._prices = body.get("data") or {}
            self._prices_at = time.monotonic()
        price = self._prices.get(currency)
        if price is None:
            raise TradeError(f"No USD price for currency {currency!r}; can't check room limits")
        return float(price)

    # ---------- login ----------

    async def login(self) -> None:
        await self.ensure_ready()
        if not await self.page.locator(SEL_HEADER_SIGNIN).is_visible():
            log.info("Already logged in")
            return
        if not (self.cfg.email and self.cfg.password):
            raise TradeError("Not logged in and BC_EMAIL / BC_PASS are not set. Log in manually in the browser window.")

        log.info("Logging in as %s", self.cfg.email)
        page = self.page
        await page.goto(SIGNIN_URL, wait_until="domcontentloaded")
        password = page.locator("input[type=password]")
        await password.wait_for(timeout=60_000)
        await page.locator(SEL_PASSWORD_TAB).click()
        email = password.locator("xpath=preceding::input[not(@type='hidden')][1]")
        await email.fill("")
        await email.press_sequentially(self.cfg.email, delay=40)
        await password.fill("")
        await password.press_sequentially(self.cfg.password, delay=40)
        await page.locator(SEL_SIGNIN_SUBMIT).click()

        log.info("Login submitted. If a captcha or verification code appears, complete it in the browser window (5 min).")
        try:
            await page.locator(SEL_HEADER_SIGNIN).wait_for(state="hidden", timeout=300_000)
        except PlaywrightTimeout:
            raise TradeError("Login did not complete within 5 minutes")
        await page.goto(TRADE_URL, wait_until="domcontentloaded")
        await self.ensure_ready()
        if await page.locator(SEL_HEADER_SIGNIN).is_visible():
            raise TradeError("Login failed: still signed out. Check the browser window.")
        log.info("Logged in")

    # ---------- trading ----------

    async def _rooms(self) -> list[Room]:
        rooms = []
        for r in (await self._raw_state())["rooms"]:
            if m := ROOM_RE.match(r["text"]):
                rooms.append(Room(m.group(1), _num(m.group(2)), _num(m.group(3)), r["active"]))
        return rooms

    async def _select_room(self, usd: float) -> None:
        """Rooms are priced in USD ("5s $1-50"); pick one that accepts this order's USD value."""
        rooms = await self._rooms()
        if not rooms:
            return
        fits = [r for r in rooms if r.lo <= usd <= r.hi]
        if not fits:
            ranges = ", ".join(f"${r.lo:g}-{r.hi:g}" for r in rooms)
            raise TradeError(f"Order value ${usd:.2f} is outside every room ({ranges})")
        if any(r.active for r in fits):
            return
        target = fits[0]
        log.info("Switching to room %s", target.label)
        # Button text may render as "5s$1-50" (textContent) or "5s $1-50" (innerText).
        pattern = re.compile(rf"^\s*{target.period}s\s*\$\s*{target.lo:g}\s*-\s*{target.hi:g}\s*$")
        await self.page.locator("button").filter(has_text=pattern).first.click()
        for _ in range(20):
            if any(r.active and r.lo == target.lo for r in await self._rooms()):
                return
            await asyncio.sleep(0.25)
        raise TradeError(f"Could not switch to room {target.label}")

    async def _set_amount(self, amount: float) -> None:
        box = self.page.locator(SEL_AMOUNT)
        await box.fill(_fmt_amount(amount))
        value = await box.input_value()
        try:
            ok = abs(float(value) - amount) < 1e-9
        except ValueError:
            ok = False
        if not ok:
            raise TradeError(f"Amount box shows {value!r}, expected {_fmt_amount(amount)}")

    @staticmethod
    def _usd_amount(usd: float, rate: float) -> float:
        """USD -> account currency with bc.game's price table: rounded up, +0.5% for non-dollar currencies so BC's
        own rate never puts it under a room minimum, at most 12 characters (the box's maxlength)."""
        amount = usd / rate * (1.005 if abs(rate - 1) > 0.01 else 1.0)
        decimals = max(0, min(8, 12 - len(str(int(amount))) - 1))
        return math.ceil(amount * 10 ** decimals - 1e-9) / 10 ** decimals

    async def buy_usd(self, side: str, usd: float, wait_window: bool = True) -> dict:
        """buy() for a stake given in USD."""
        rate = await self.usd_rate((await self.status()).currency)
        return await self.buy(side, self._usd_amount(usd, rate), wait_window)

    async def prepare_usd(self, usd: float) -> None:
        """Select the room and fill the amount box ahead of time, so click_prepared() only has to click."""
        async with self._lock:
            st = await self.status(light=True)
            if st.dialog:
                raise TradeError(f"A dialog is open in the browser ('{st.dialog}...'). Close it there first.")
            if usd > self.cfg.max_usd:
                raise TradeError(f"${usd:g} exceeds MAX_USD {self.cfg.max_usd:g} (see .env)")
            amount = self._usd_amount(usd, await self.usd_rate(st.currency))
            await self._select_room(usd)
            await self._set_amount(amount)
            self._prepared = (amount, usd, st)

    @staticmethod
    def _late(deadline: float | None, now) -> float | None:
        """Seconds past `deadline` (positive = too late), on the clock `now` gives (a callable; bc.game's clock
        for automatic orders, so a PC clock that is off by a second never sends an order late)."""
        if deadline is None:
            return None
        t = now() if now is not None else time.time()
        return None if t is None else t - deadline

    async def click_prepared(self, side: str, deadline: float | None = None, now=None) -> dict:
        """The fast path for automatic calls: re-check the prepared amount and click.  Fails at once when betting
        is closed, and raises TooLate instead of clicking after `deadline` (seconds on the clock `now` gives)."""
        side = SIDES.get(side.strip().lower())
        if side is None:
            raise TradeError("Side must be 'up' or 'dn'")
        if getattr(self, "_prepared", None) is None:
            raise TradeError("Nothing prepared")
        async with self._lock:
            amount, usd, st = self._prepared
            if await self.page.locator(SEL_AMOUNT).input_value() != _fmt_amount(amount):
                await self._set_amount(amount)
            late = self._late(deadline, now)
            if late is not None and late > 0:
                raise TooLate(f"too late to click ({late:.2f}s past the deadline)")
            return await self._submit(side, amount, usd, st)

    async def buy(self, side: str, amount: float, wait_window: bool = True, deadline: float | None = None,
                  now=None) -> dict:
        """Place an order.  wait_window=False fails at once when betting is closed instead of waiting for the
        next round (an automatic call belongs to one specific round).  With a `deadline` (on the clock `now`
        gives) it raises TooLate instead of clicking after it."""
        side = SIDES.get(side.strip().lower())
        if side is None:
            raise TradeError("Side must be 'up' or 'dn'")
        if amount <= 0:
            raise TradeError("Amount must be positive")

        def check_deadline():
            late = self._late(deadline, now)
            if late is not None and late > 0:
                raise TooLate(f"too late to click ({late:.2f}s past the deadline)")

        async with self._lock:
            check_deadline()
            st = await self.status(light=deadline is not None)
            if st.dialog:
                raise TradeError(f"A dialog is open in the browser ('{st.dialog}...'). Close it there first.")
            usd = amount * await self.usd_rate(st.currency)
            if usd > self.cfg.max_usd:
                raise TradeError(f"{amount:g} {st.currency} (${usd:.2f}) exceeds MAX_USD {self.cfg.max_usd:g} (see .env)")
            await self._select_room(usd)
            await self._set_amount(amount)

            button = self.page.locator(SEL_DIRECTION).get_by_role("button", name=BUTTON_LABEL[side], exact=True)
            wait_until = time.monotonic() + self.cfg.window_wait
            if not await button.is_enabled():
                if not wait_window:
                    raise TradeError("Betting window is closed")
                log.info("Round is closed; waiting for the next betting window...")
            while not await button.is_enabled():
                if time.monotonic() > wait_until:
                    raise TradeError(f"Betting window did not open within {self.cfg.window_wait:g}s")
                check_deadline()
                await asyncio.sleep(0.2)
            # Re-check: the amount box is reset when a room changes or the page re-renders.
            if await self.page.locator(SEL_AMOUNT).input_value() != _fmt_amount(amount):
                await self._set_amount(amount)
            check_deadline()
            return await self._submit(side, amount, usd, await self.status(light=True))

    async def _submit(self, side: str, amount: float, usd: float, before: Status) -> dict:
        """Click Up / Down and read the site's order response (caller holds the lock)."""
        button = self.page.locator(SEL_DIRECTION).get_by_role("button", name=BUTTON_LABEL[side], exact=True)
        if not await button.is_enabled():
            raise TradeError("Betting window is closed")
        sent = time.time()
        try:
            async with self.page.expect_response(
                lambda r: ORDER_API in r.url and r.request.method == "POST", timeout=10_000
            ) as info:
                # force: skip Playwright's "stable" check, which waits for animation frames (paused while the
                # window is in the background); the button was just checked to be enabled
                await button.click(force=True, timeout=3000)
            resp = await info.value
        except PlaywrightTimeout:
            # the click happened; whether an order went out is unknown (a message on the page, a slow network)
            self._record({"ok": None, "unknown": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "sent": round(sent, 3), "side": side, "amount": amount, "currency": before.currency,
                          "usd": round(usd, 4), "account": before.account, "msg": "no order response in 10 s"})
            raise OrderUnknown("Clicked, but no order response came within 10 s: the order may or may not have "
                               "been placed. Check the site's order history before trading again.")
        acked = time.time()

        try:
            body = await resp.json()
        except Exception:
            body = {"code": None, "msg": (await resp.text())[:300]}
        data = body.get("data") or {}
        result = {
            "ok": resp.ok and body.get("code") == 0,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sent": round(sent, 3),
            "acked": round(acked, 3),
            "rtt": round(acked - sent, 3),
            "side": side,
            "amount": amount,
            "currency": data.get("currency", before.currency),
            "usd": round(float(data.get("usdAmount") or usd), 4),
            "account": before.account,
            "payout_pct": before.up_payout if side == "up" else before.down_payout,
            "order_id": data.get("id"),
            "round_id": data.get("roundId"),
            "msg": body.get("msg"),
            "code": body.get("code"),
        }
        self._record(result)
        if not result["ok"]:
            raise TradeError(f"Order rejected: {result['msg']} (code {result['code']}, HTTP {resp.status})")
        return result

    def _record(self, result: dict) -> None:
        self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cfg.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


def status_dict(s: Status) -> dict:
    return asdict(s)
