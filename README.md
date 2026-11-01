# BC.Game Up/Down bot

Opens https://bc.game/trading/up-down in a real Edge window, logs in, and buys **UP** or **DOWN** when you tell it to,
either by typing a command or by calling a local HTTP API.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then fill in BC_EMAIL / BC_PASS
```

## Run

```powershell
run.bat                 # browser + login + command prompt + HTTP API
run.bat --no-repl       # HTTP API only (run in background)
run.bat --no-login      # logged out: uses the site's demo balance, good for testing
```

The first run logs in with `BC_EMAIL` / `BC_PASS`. If a captcha or email code appears, complete it in the browser
window. The session is kept in `browser-profile/`, so later runs skip the login. Quitting the bot leaves the browser
open, and the next start attaches to it again.

## Commands

```
up 5        buy UP for 5      (only while armed, see "Real money"; counted like every real order)
dn 5        buy DOWN for 5
up          repeat with the last amount
status      balance, round phase, seconds left, payouts
live        the real-money state: losses in a row, orders today, budget, campaign, clock, gate
```

Amounts are in the account's **trading currency**, shown by `status` and on the site as "Amount (XXX)".
Change it with the wallet selector in the site header (e.g. USDT). The bot converts each order to USD to
pick the right room (`$1-50`, `$50-100`, `$100-200`) and refuses anything above `MAX_USD` (default $5: bigger
stakes dilute the pari-mutuel payout).

A manual order goes into the open round only and is refused when betting is closed or the bot is not armed.

## Auto mode (Hawkes models)

Based on Raffaelli, Cestari, Marazzina & Formentin (2026), *Forecasting Bitcoin price movements using multivariate
Hawkes processes and limit order book data*, Decisions in Economics and Finance.

```powershell
run.bat --auto          # or type "auto on" in the running bot
```

- `bcbot/lob.py` reads Bitfinex's live BTC/USD order book, which is the paper's data source. It turns each update into the paper's event types (mid-price change, level-1 liquidity up or down, buy or sell pressure) and computes the Base Imbalance (BI, level 5).
- `bcbot/hawkes.py` is the multivariate Hawkes process: exponential kernels, the least-squares fit with an l1 penalty, FISTA, the greedy decay search of Algorithm 1, and Ogata thinning.
- `bcbot/coe.py` is the continuous-time output-error model from BI to the return (na=2, nb=1), fitted by output-error least squares instead of MATLAB's CONTSID.
- `bcbot/predictor.py` covers the bc.game round. The paper forecasts the *next* mid-price change, but a round is the BTC move over the 5 s after betting closes. So the Hawkes model simulates when mid-price changes happen inside that window, and the COE model gives the return at each one. The sign of their sum is the call (HawkesTime). HawkesSign's P(UP) is logged next to it.
- `bcbot/auto.py` follows bc.game's own round feed in the page. It retrains when each betting window opens, makes the call `AUTO_LEAD_MS` before cutoff, and scores it against bc.game's result. Every round goes to `logs/auto_rounds.jsonl`.

- `bcbot/flow.py` is the MHH bot's BC model, ported as-is: a logistic regression on Binance BTCUSDT 1 s order flow (price direction and taker buy/sell imbalance over 1/3/5/10/30 s), using MHH's own fitted file `models/flow5s_logit.json`. It predicts at its own 2 s lead.

- `mom` (added 2026-10-31) is the direction of our exchange-mix price over the last 5 s, read at the decision instant (~2.9 s before the cutoff). It is the only signal that came out ahead of HawkesSign against bc.game's own results (52.6% vs 50.6% on the same rounds); see "Why the win rate is ~50%" in `docs/decision-review-2026-10-30.md`. It was fixed before any forward data, so its record from now on is the real test.

- `book` bets with Binance's 20-level order book, only when it is strongly lopsided (|imbalance| ≥ 0.7). `bookmom` bets only when that strong book and the last 5 s of our price point the same way, and skips every other round. When they disagree (a buying book with a falling price) it was a coin flip (49% UP), so neither side is bet. A 30-feature logistic regression scored 49.4% out of sample on the same ~2,000 rounds, which is why these simple rules are used instead of a fitted model. All of them were chosen on the logged rounds; only their forward record counts.

- `revert`, `overshoot`, `overshoot_book` (2026-11-01) bet that bc.game's own index snaps back after running ahead of the market: in a quiet market, or when its last 2 s move left it $1–2 past our exchange price in that direction (optionally with the Binance book against the move). These held on rounds they were not picked from (57–59% on the unseen 40%); see the review document.
- `prob` (2026-11-01) is the statistical decision: a logistic model in `bcbot/prob.py`, refitted every hour on all settled rounds in the log, giving P(UP) for each round from 22 decision-time features plus the "ran ahead of the market" combinations. It calls only when P(UP) is at least 56% one way (margin 0.06, fixed). Walk-forward on 3,446 logged rounds: 57.7% of 395 calls at that margin, 54.3% of 915 at 0.04. Every round also logs 60 s deviation statistics (`dev_mean60`, `dev_std60`, `dev_z60`) for the next version.

- `gbm` and `ens` (branch `micro-ensemble`, 2026-11-01): a LightGBM learner on the same fixed inputs, and the ensemble (the average of the logistic and LightGBM P(UP)); walk-forward they scored 56.2% of 276 and 56.5% of 260 at margin 0.06 against 57.9% of 399 for the logistic alone, so the ensemble is run as a forward test, not because it was better on the log. `lightgbm` is optional: without it `gbm`/`ens` stay silent. The Binance book now also gives the microprice offset, a depth-weighted imbalance and the spread, logged with every round next to `bc_mom05` and `dev_z60` for the next model version.

Branches: `bc-hawkes-signal` is the code as it stood on 2026-11-01 (tag `v0.1.0`, the Hawkes / rule models and the logistic P(UP) model); `micro-ensemble` adds the trees, the ensemble and the microstructure features.

All models are scored on every round. `AUTO_MODEL=sign|mhh|agree|mom|book|bookmom|momvol|revert|overshoot|overshoot_book|prob|gbm|ens` (or `model ...` in the terminal) picks which one makes the calls, and so which one would place live orders. Models fixed on 2026-10-31 or later count only rounds after they were fixed towards the go-live gate.

- `pool` is scored too but never calls: the thin side of the pari-mutuel pool whenever the page pays more than x2.0 on it. The crowd's imbalance carried no information in 1,996 logged rounds, so that is a coin paid above even money, worth about a cent per $1 after your own stake dilutes the pool. Its record (at the diluted payout) is on the dashboard next to the models; weeks of rounds are needed to tell whether it is really above zero.

Training follows the paper: the Hawkes models on the last 300 s, retrained every round, and the COE on the last 30 min, retrained every 5 min. The first calls come after about 5 minutes of order-book data.

`auto` shows the results so far: accuracy, HawkesSign accuracy, paper P&L at the payouts the page shows, and how often Bitfinex moved the same way as bc.game's price.

**Paper first.** `auto on` only records calls. With bc.game's fee, a call has to be right more than 1/payout of the time (51.55% at x1.94, 51.8% once your own stake dilutes the pool) just to break even; on 3,000+ logged rounds every model is at 48.5-50.6%. The papers' 67% is for the *next tick* on Bitfinex's own book, without fees; see `docs/decision-review-2026-10-30.md` for the full review.

## Real money

Real orders are **off at every start** and the bot never switches them on by itself.

1. Put `LIVE_ALLOWED=1` and `TOS_RISK_ACCEPTED=1` in `.env` (bc.game's User Agreement lists automated play as prohibited; that risk is yours) and, for the HTTP API / dashboard, an `API_TOKEN`.
2. In the terminal: `live arm run=<losses in a row> day=<YYYY-MM-DD>` - you must type the current run of real losses and today's date (`live` shows them). While the gate is closed (the calling model's record does not beat the break-even) you also have to add `force`.
3. Each call is then placed as a flat `AUTO_STAKE_USD` order, after a checklist run right before the click on bc.game's own clock: run of losses, no order pending, orders today, today's budget, campaign loss, balance, clock, time to the cutoff, data quality, prepared order. Refusals are logged with the round.

The bot **stops and disarms** at `AUTO_MAX_REAL_RUN` (5) real losses in a row - the count is rebuilt from the log at every start and only a real WIN resets it - at `-AUTO_DAILY_LOSS_USD` (8) for the day (checked *before* each order), after `AUTO_MAX_REAL_BETS_DAY` (60) orders in a day, at `-AUTO_CAMPAIGN_LOSS_USD` (40) in all, after 4 hours, on a rejected / unknown / misbooked order, on a balance that does not match the booked result, and when the file `logs/KILL` exists. It never resumes: you re-arm with the command above, and beyond 5 losses in a row each re-arm allows **one** real order (a loss stops it again). Technical stops need `live clear <reason>` first. Everything is written to `logs/live_events.jsonl`.

Progressions (`stakes 1.5|2`) only drive the paper ladders; real orders are always flat. `AUTO_SHADOW=1` (default) runs the whole checklist on paper while disarmed and shows what the bot would have done.

## Tap analyzer (paper only)

`run.bat --auto` also opens bc.game's Tap game (`/trading/tap`, BTC/USD) in a second tab and analyses it without
placing orders: http://127.0.0.1:8765/tap, plus our boxes and a left-bar panel on the page itself (`--no-tap` skips it).

- The grid is a canvas, so the analyzer watches what the page draws. The axis labels give the time → x and price → y
  lines; the multiplier labels ("8.0x") give **bc.game's own multiplier for every cell** and the band size on screen
  (bc.game switches between $5, $10 and $20 bands with volatility). Our port of its pricing model is only the fallback.
- Every cell is tapped virtually each column and settled on the real price ticks, per kind of cell and per band size.
  Realized hit rate / the model's P(hit) calibrates the model once a cell kind has `CALIB_MIN_N` taps, and
  EV = calibrated P(hit) × bc.game's multiplier.
- A $1 paper tap goes on the best tappable cell (the columns bc.game labels) when its EV is at least `EV_MIN` (1.05)
  and the multiplier was read from the page. P&L is net per tap (hit: multiplier − 1, miss: −1); the card also shows
  the open taps and a wallet view with their stakes taken out.
- Other players' BTC/USD taps are logged from the page's feed with their result, profit, and the multiplier we read
  for their cell, which checks our reading of the grid.

Findings so far: bc.game's pricing overestimates P(hit) for cells away from the price (they return −40 % to −90 %
per $1 tapped); only the current-price band is near break-even. The earlier paper profit came from our port's
multipliers being too high (x8 where the page showed x5–6) whenever bc.game's volatility setting had moved since the
page was loaded. Logs: `logs/tap_stats.json`, `logs/tap_paper.jsonl`, `logs/tap_public.jsonl`.

## HTTP API

```powershell
curl http://127.0.0.1:8765/status
curl -X POST http://127.0.0.1:8765/trade -H "Content-Type: application/json" -d "{\"side\":\"up\",\"amount\":5}"
```

Auto mode: `GET /auto` gives the accuracy, paper P&L, last call, model parameters and the real-money state (run of losses, budget, clock, gate, last events). `POST /auto` with `{"on": true}`, `{"model": "mhh"}` or `{"progression": 1.5}` changes the paper settings; `{"arm": true, "ack_run": 0, "day": "2026-10-30"}`, `{"live": false}`, `{"kill": true}` and `{"clear": "unknown", "outcome": "none"}` arm, disarm, kill and clear, and need the token.

A successful trade returns `{"ok": true, "order_id": ..., "round_id": ..., ...}`. A failure returns `{"ok": false, "error": "..."}`.
`POST /trade` and everything that arms or disarms require `API_TOKEN` from `.env` (`Authorization: Bearer <token>` or `?token=`); without one they are refused.

Every order attempt is written to `logs/trades.jsonl`.

## How it works

- Edge is started as a normal browser with `--remote-debugging-port`, and Playwright attaches over CDP. A browser
  launched by Playwright itself fails to load the trading widget.
- An order counts as placed only when the site's `updown/order/create` call answers `code: 0`. Any other
  answer (e.g. `Account balance lack`) is reported as a failure.
- If a site dialog covers the panel, the bot refuses to trade and tells you to close it in the browser.

## Note

bc.game's risk-warning dialog states that automated scripts are prohibited and may lead to trading restrictions or
account suspension. Using this bot is at the account holder's own risk.
