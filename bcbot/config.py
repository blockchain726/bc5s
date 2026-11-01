import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


@dataclass(frozen=True)
class Config:
    email: str
    password: str
    browser_path: str
    cdp_port: int
    profile_dir: Path
    api_host: str
    api_port: int
    api_token: str               # required for every POST to the local API (arming, orders); GET stays open
    default_amount: float
    max_usd: float               # the most one manual order may be worth (real stakes above ~$5 dilute the payout)
    window_wait: float
    log_file: Path
    auto_model: str              # which model makes the calls: "sign" (HawkesSign), "mhh" (the MHH bot's
                                 # Binance-flow logit), or "agree" (only rounds where HawkesSign and MHH call the
                                 # same side); HawkesTime ("time") is scored but can't call
    flow_model: Path             # the MHH model file
    auto_lead_ms: int            # decide this long before betting closes (on bc.game's clock)
    auto_stake_usd: float        # the real base stake per round
    auto_real_ladder: tuple      # AUTO_REAL_LADDER, e.g. "1,2": real stake = base x the step for the current run of
                                 # real losses ("compensate"), ONLY while the go-live gate is open; "1" = flat
    auto_progression: float      # paper ladders: next stake = previous x this after a loss, base after a win (1 = flat)
    auto_ladder_max_usd: float   # paper ladder: give up (back to the base) when the next stake would pass this;
                                 # 0 = never give up, keep raising until a win
    auto_daily_loss_usd: float   # real-money loss allowed per calendar day, checked BEFORE each order
    auto_log: Path
    # ---- real-money safety (see docs/decision-review-2026-10-30.md) ----
    live_allowed: bool           # LIVE_ALLOWED=1: the user allows real orders at all (default: no)
    tos_risk_accepted: bool      # TOS_RISK_ACCEPTED=1: the user accepts that bc.game's terms forbid bots
    auto_max_real_run: int       # real losses in a row at which the bot stops and disarms (only a real WIN resets it)
    auto_rearm_cooloff_min: float  # minutes after a losses-in-a-row stop before a re-arm is accepted
    auto_arm_ttl_h: float        # an arming expires after this many hours (the operator must be present)
    auto_max_stake_usd: float    # the most the automatic path may stake per round
    auto_min_bankroll_stakes: int  # no real order unless the balance is at least this many stakes
    auto_max_real_bets_day: int  # real order attempts per calendar day (placed, rejected and unknown all count)
    auto_campaign_loss_usd: float  # cumulative real loss since the first arming at which live is over for good
    auto_send_margin_s: float    # never click later than this before the cutoff, on bc.game's clock
    auto_balance_max_age_s: float  # the balance must have been read this recently before an order
    auto_reject_stop: int        # rejected orders within 10 minutes that stop real orders (needs 'live clear')
    auto_min_lead_s: float       # a round whose window starts sooner / later than this is skipped as odd timing
    auto_max_lead_s: float
    auto_late_snapshot_s: float  # the second feature snapshot (paper research), this long before the cutoff
    auto_shadow: bool            # run every real-order check on paper and log what the bot would have done
    auto_state_file: Path        # the real-loss counter and campaign, kept across restarts
    auto_events: Path            # audit trail of every arm / stop / order
    auto_kill_file: Path         # create this file to disarm at once (deleting it does not re-arm)


def _find_browser() -> str:
    if path := os.getenv("BROWSER_PATH", "").strip():
        return path
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("Edge/Chrome not found. Set BROWSER_PATH in .env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, "").strip() or default)
    return p if p.is_absolute() else ROOT / p


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def load() -> Config:
    return Config(
        email=os.getenv("BC_EMAIL", "").strip(),
        password=os.getenv("BC_PASS", "").strip(),
        browser_path=_find_browser(),
        cdp_port=int(os.getenv("CDP_PORT", "9222")),
        profile_dir=_path("PROFILE_DIR", "browser-profile"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8765")),
        api_token=os.getenv("API_TOKEN", "").strip(),
        default_amount=float(os.getenv("DEFAULT_AMOUNT", "1")),
        max_usd=float(os.getenv("MAX_USD", "5")),
        window_wait=float(os.getenv("WINDOW_WAIT", "45")),
        log_file=_path("TRADE_LOG", "logs/trades.jsonl"),
        auto_model=os.getenv("AUTO_MODEL", "sign").strip().lower(),
        flow_model=_path("FLOW_MODEL", "models/flow5s_logit.json"),
        auto_lead_ms=int(os.getenv("AUTO_LEAD_MS", "3000")),
        auto_stake_usd=float(os.getenv("AUTO_STAKE_USD", "1")),
        auto_real_ladder=tuple(float(x) for x in (os.getenv("AUTO_REAL_LADDER") or "1").replace(" ", "").split(",")
                               if x and float(x) > 0) or (1.0,),
        auto_progression=float(os.getenv("AUTO_PROGRESSION", "1")),
        auto_ladder_max_usd=float(os.getenv("AUTO_LADDER_MAX_USD", "0")),
        auto_daily_loss_usd=float(os.getenv("AUTO_DAILY_LOSS_USD", "8")),
        auto_log=_path("AUTO_LOG", "logs/auto_rounds.jsonl"),
        live_allowed=_flag("LIVE_ALLOWED"),
        tos_risk_accepted=_flag("TOS_RISK_ACCEPTED"),
        auto_max_real_run=int(os.getenv("AUTO_MAX_REAL_RUN", "5")),
        auto_rearm_cooloff_min=float(os.getenv("AUTO_REARM_COOLOFF_MIN", "15")),
        auto_arm_ttl_h=float(os.getenv("AUTO_ARM_TTL_H", "4")),
        auto_max_stake_usd=float(os.getenv("AUTO_MAX_STAKE_USD", "5")),
        auto_min_bankroll_stakes=int(os.getenv("AUTO_MIN_BANKROLL_STAKES", "20")),
        auto_max_real_bets_day=int(os.getenv("AUTO_MAX_REAL_BETS_DAY", "60")),
        auto_campaign_loss_usd=float(os.getenv("AUTO_CAMPAIGN_LOSS_USD", "40")),
        auto_send_margin_s=float(os.getenv("AUTO_SEND_MARGIN_S", "1.5")),
        auto_balance_max_age_s=float(os.getenv("AUTO_BALANCE_MAX_AGE_S", "20")),
        auto_reject_stop=int(os.getenv("AUTO_REJECT_STOP", "2")),
        auto_min_lead_s=float(os.getenv("AUTO_MIN_LEAD_S", "1.5")),
        auto_max_lead_s=float(os.getenv("AUTO_MAX_LEAD_S", "5")),
        auto_late_snapshot_s=float(os.getenv("AUTO_LATE_SNAPSHOT_S", "1.4")),
        auto_shadow=_flag("AUTO_SHADOW", "1"),
        auto_state_file=_path("AUTO_STATE_FILE", "logs/live_state.json"),
        auto_events=_path("AUTO_EVENTS", "logs/live_events.jsonl"),
        auto_kill_file=_path("AUTO_KILL_FILE", "logs/KILL"),
    )
