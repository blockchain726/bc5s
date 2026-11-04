# Working on this bot

Two bots in one process (`main.py --auto`): the 5 s Up/Down bot (`bcbot/auto.py`) and the Tap bot (`bcbot/tap.py`,
`bcbot/tap_live.py`), attached over CDP (port 9222) to a normally started Edge. Dashboards: http://127.0.0.1:8765/
(Up/Down) and http://127.0.0.1:8765/tap. `docs/run-on-another-pc.md` says how to set a machine up.

## Rules the owner set (keep them)

- Secrets only in `.env` (`BC_EMAIL`, `BC_PASS`, `API_TOKEN`): never print them, never commit them. The repo is public.
- Real money is the owner's decision. Arming the Up/Down bot, switching Tap live to REAL, resetting the run of real
  losses and clearing stops are done from the dashboards with the API token. Do not try to do them another way.
- Never stop the Tap live bot for a technical reason. Every former stop rule (two rejections, an unanswered order,
  an order booked in the next column, an unsettled order, a losing run) was turned into a warning or a wait. The
  only stop is the budget (`TAP_BUDGET_USD`). Do not add stop rules back.
- A restart must not disarm the Up/Down bot (`AUTO_RESUME_ARM=1`), and the Tap tab opens 25 s after the Up/Down page
  (`TAP_START_DELAY_S`). Still: before a restart check `GET /auto` for a pending real order, and restart sparingly:
  bc.game refuses game launches that come too fast ("request_too_fast").
- The Tap stake and mode set from the page are kept in `logs/tap_live_state.json` and beat `.env` on restart.
- The Tap page and the Up/Down work are separate: a change to one must not touch the other.

## Facts measured (do not re-derive)

- Up/Down: every model is ~50% on bc.game's result except the snap-back family (`overshoot`, `prob`, `ens`), ~58-61%
  on a few hundred forward calls; the gate (3,000 forward calls, Wilson lower bound > 51.55%) is still closed.
- Tap: the band the price is in (r = 0, 2-4 columns ahead) hits 47-60% of the time at every band size; cells two or
  more bands away hit 2-17%. The picker rates cells by the measured hit rate of the cell kind at the current band
  size (1,000+ virtual taps) times the label, with a regime factor and a trailing-rate cap.
- bc.game's server pays about 65% of the multiplier shown on the grid (measured on 400+ real orders; 61% two
  columns ahead, 66% further out). The owner chose to price taps at the label anyway (`TAP_USE_RATIO=0`).
- Paper picks and real orders correspond one to one: one tap per column, a paper box only when the live tap is free.
  Paper wins are booked at bc.game's measured payout (0.64 / 0.68 / 0.70 of the label two / three / four columns
  ahead, `TapLive.payout`); the P&L at the label is shown alongside. Until 2026-11-02 the book counted the label,
  which is why paper looked profitable (+$588 per $1 on 1,523 picks) while the same taps lost real money (-$42).
- The bot's own settlement of a cell agrees with bc.game's result in 99% of real orders.
- The live tap follows a selection of the trial pickers (`TAP_LIVE_MODEL`, page-settable, kept in the Tap state):
  `auto` = every hour the 3 pickers with the best return per $ over the last 12 h (30+ taps each; `rank_live_set`),
  else a list of names. Each snapshot the live pick is the best-ranked followed picker's pick for a free column; the
  paper book (`LIVE_MODEL = "live"`) records exactly those, so paper stays one-to-one with real orders.
  `TapLive._choose` has no rule of its own. The `mid` rule (current band, price mid-band, 2 s move under a quarter
  band, 60 s range under 4 bands, paid EV >= 1) is the best single picker so far; the old label picker (best EV at
  the grid label >= 1.35) lost ~8% per $ at the payout and was removed on 2026-11-02 at the owner's request.
  Backtest of the selection (11-02/03): 3 h and 6 h windows chase noise; 12 h by per $ works; never rank by total.
- `bcbot/tap_lr.json` is a fitted conditional hit model (logistic regression on the research record; refit with
  the scratch research script's `--export`); the `lr` / `lr-calm` trial pickers use it. Absent file: they idle.
- `bcbot/tap_gbm.txt` + `tap_gbm.json` are a LightGBM hit model for the current and adjacent bands (refit with the
  scratch `tapresearch/research.py --export`); the `gbm` (EV at the paid odds >= 1.2) and `gbm-safe` (>= 1.3) trial
  pickers use it. Walk-forward 11-02..11-04: gbm-safe +44% per $ on 3,788 taps, 9 of 9 four-hour blocks positive;
  the strongest features are the distance to the cell's edge, the move toward the cell, the 60 s range and the
  real/configured volatility. Time of day adds nothing out of sample.
- Candidate pickers (`MODELS`) run side by side on paper, each with its own book, colour and box on the page.
  Their books are in `tap_stats.json` ("models") and `logs/tap_models.jsonl`; the page features `TAP_SHOW_MODEL`.

## Commands

- Tests: `.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` (62 tests). Check syntax without a pipe
  before committing (`python -c "import ast, sys; ast.parse(open(f).read())"`); a piped check hides the exit code.
- Keep running: `powershell -File tools\supervise.ps1` (log in `logs\supervisor.log`).
- Git: commit identity `blockchain726 <blockchain726@users.noreply.github.com>`; branch `micro-ensemble`, kept equal
  to `main`.
