# Running the bot on another PC

Everything the bot needs is in this repository except three things that must never be in a public repo and are
therefore ignored by git: the `.env` file (login, password, API token), the browser profile, and the logs. The
measured Tap hit rates live in the logs too, and the Tap picker is empty without them, so copy that one file.

## 1. On the new PC

```powershell
git clone https://github.com/blockchain726/bc5s.git "BC BTC"
cd "BC BTC"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m playwright install msedge     # only if Edge is not installed; the bot uses the system Edge
```

Python 3.11 or newer and Microsoft Edge are required. Windows is assumed (the supervisor is PowerShell).

## 2. Copy by hand, NOT through git

From the old PC's project folder to the same places on the new one:

| File | Why |
|---|---|
| `.env` | login, password, `API_TOKEN`, and every setting you changed (model, ladder, Tap stake, Tap live mode) |
| `logs\tap_stats.json` | the measured hit rate of every Tap cell kind (70k+ virtual taps); without it the Tap picker places nothing for about an hour per band size |
| `logs\tap_live_state.json` | optional: the Tap live book, the measured server payout ratios, the stake set from the page |
| `logs\auto_rounds.jsonl` | optional, large: every logged round; the P(UP) model and the forward records are rebuilt from it |
| `logs\live_state.json` | optional: the real-money state (run of losses, campaign) |

Use a USB stick or any private channel. Do not commit or paste these anywhere.

## 3. First start

```powershell
run.bat --auto
```

The first run opens Edge with remote debugging, logs in with `BC_EMAIL` / `BC_PASS` (complete a captcha or
email code in the window if one appears), and keeps the session in `browser-profile\`. bc.game then shows its
"Risk warning" dialog: click **I acknowledge** yourself. Check http://127.0.0.1:8765/ (Up/Down) and
http://127.0.0.1:8765/tap (Tap) are up.

## 4. Keep it running

```powershell
powershell -File tools\supervise.ps1
```

restarts the bot 15 s after any exit and writes to `logs\supervisor.log`. `tools\supervise_tap.ps1` does the same for the Tap bot alone (no Up/Down auto mode). An arming survives a restart while its
4 hours last (`AUTO_RESUME_ARM=1`), the Tap tab opens 25 s after the Up/Down page (`TAP_START_DELAY_S`), and the
Tap stake and mode set from the page are kept.

## 5. Arming and the Tap

- Up/Down dashboard: paste `API_TOKEN` from `.env` into the token box once, then **ARM real** (tick **compensate
  ladder** first for the $1 / $2 ladder). `losses -> 0` resets the run of real losses.
- Tap page: the same token box, then **REAL** at the stake in the box. **OFF** ends it; `clear stop` clears a
  technical stop; the budget stop (`TAP_BUDGET_USD`) does not clear.

## Not to run twice

Do not run the bot on two PCs with the same bc.game account at the same time: both would place orders on one
balance, and the Up/Down bot's loss counter and the Tap book would each see only half of what happened.
