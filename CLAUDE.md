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
- The bot's own settlement of a cell agrees with bc.game's result in 99% of real orders.

## Commands

- Tests: `.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` (50 tests). Check syntax without a pipe
  before committing (`python -c "import ast, sys; ast.parse(open(f).read())"`); a piped check hides the exit code.
- Keep running: `powershell -File tools\supervise.ps1` (log in `logs\supervisor.log`).
- Git: commit identity `blockchain726 <blockchain726@users.noreply.github.com>`; branch `micro-ensemble`, kept equal
  to `main`.
