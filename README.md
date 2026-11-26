# BC.Game Up/Down bot

Opens https://bc.game/trading/up-down in a real Edge window, logs in, and buys **UP** or **DOWN** when you tell it to,
either by typing a command or by calling a local HTTP API.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then fill in BC_EMAIL / BC_PASS
```

To run it on another PC, see `docs/run-on-another-pc.md` (what to clone and the three private files to copy by hand).

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
- `bcbot/predictor.py` covers the bc.game round. The paper forecasts the *next* mid-price change, but a round is the BTC move over the 5 s after betting closes. So the Hawkes model simulates when mid-price changes happen inside that window, and the COE model gives the return at each one. HawkesSign's P(UP) over the window is the call when the model is `sign`. The sign of the summed returns (HawkesTime) is still computed and logged, but no longer scored.
- `bcbot/auto.py` follows bc.game's own round feed in the page. It retrains when each betting window opens, makes the call `AUTO_LEAD_MS` before cutoff, and scores it against bc.game's result. Every round goes to `logs/auto_rounds.jsonl`.

- `bcbot/flow.py` holds the Binance BTCUSDT feeds: 1 s klines with taker-buy volume (the taker-flow inputs), the best bid/ask mid, and the top-20 order book imbalance.

- Removed on 2026-11-11, after losing at a flat $1 on every call (the dashboard's record since 2026-11-10 15:26 UTC): `mom` (our price's 5 s direction, 49.0% of 3,065 calls), `momvol`, `jump`, `book` and `bookmom` (Binance's order book, 49.1% of 1,646), `mhh` / `mhh_hi` / `agree` (the MHH bot's logistic model on Binance order flow, 49.5% of 2,985) and `time` (HawkesTime, 50.4% of 2,943). Their forward records since each was fixed say the same: 48.4–51.0% on 1,505–13,882 calls each. Betting against them is no better (49–52%): on bc.game's result the trend, the order book and the order flow carry no signal either way. Their code is in git before that date.

- `revert`, `overshoot`, `overshoot_book` (2026-11-01) bet that bc.game's own index snaps back after running ahead of the market: in a quiet market, or when its last 2 s move left it $1–2 past our exchange price in that direction (optionally with the Binance book against the move). These held on rounds they were not picked from (57–59% on the unseen 40%); see the review document.
- `prob` (2026-11-01) is the statistical decision: a logistic model in `bcbot/prob.py`, refitted every hour on all settled rounds in the log, giving P(UP) for each round from 22 decision-time features plus the "ran ahead of the market" combinations. It calls only when P(UP) is at least 56% one way (margin 0.06, fixed). Walk-forward on 3,446 logged rounds: 57.7% of 395 calls at that margin, 54.3% of 915 at 0.04. Every round also logs 60 s deviation statistics (`dev_mean60`, `dev_std60`, `dev_z60`) for the next version.

- `ens` (branch `micro-ensemble`, 2026-11-01; `gbm`, the LightGBM model by itself, was removed on 2026-11-16 at the owner's request: it hardly ever bet, 99 bets in two weeks): a LightGBM learner on the same fixed inputs, and the ensemble (the average of the logistic and LightGBM P(UP)); walk-forward they scored 56.2% of 276 and 56.5% of 260 at margin 0.06 against 57.9% of 399 for the logistic alone, so the ensemble is run as a forward test, not because it was better on the log. `lightgbm` is optional: without it `ens` stays silent. The Binance book now also gives the microprice offset, a depth-weighted imbalance and the spread, logged with every round next to `bc_mom05` and `dev_z60` for the next model version.

Branches: `bc-hawkes-signal` is the code as it stood on 2026-11-01 (tag `v0.1.0`, the Hawkes / rule models and the logistic P(UP) model); `micro-ensemble` adds the trees, the ensemble and the microstructure features.

All models are scored on every round. `AUTO_MODEL=value|fade|fade_strict|fade_flow|fade_flow2|fade_hawkes|fade_fair|overshoot|overshoot_rel|overshoot_book|revert|prob|ens|trees_mid|thin|sign` (or `model ...` in the terminal, or the dashboard) picks which one makes the calls, and so which one would place live orders. Models fixed on 2026-10-31 or later count only rounds after they were fixed towards the go-live gate.

- The order-book models (2026-11-16, paper; rows of the Models table, and selectable in the page's Model list under
  "Order books, six markets": the chosen one's call is made when the service's answer is in): logistic
  regression, LightGBM, XGBoost and DeepLOB (`tools/deeplob.py`, needs torch), on the order books and trades of the
  six markets of our price, each market alone and mixed by its share. They run in their own process, the
  order-book service (`powershell -File tools\supervise_ob.ps1`, research Python in `logs\research-venv`): it reads
  the six books, records them, fits the models again as the recording grows (3 / 24 / 72 / 168 h) and answers
  the bot at each decision. Each model bets in every round, on the side its P(UP) is on (about 120 bets an hour;
  `BET_SHARE` in `bcbot/obfeat.py`: under 1 it bets only that share of the rounds, the ones it is surest of, measured
  on its own last 240 answers). Each new fit starts the four records again. HawkesSign (`sign`,
  "HawkesSign (six markets)", `bcbot/hawkes6.py`) comes from the same service since 2026-11-16 15:36 UTC: the
  HawkesSign method on all six books instead of Bitfinex spot alone (its record was started again with the change):
  one Hawkes model per market on its last 300 s (price moves and the buy / sell pressure at the best prices),
  fitted again every 20 s in the service's worker processes, run forward 200 times over the round's window and
  mixed by the markets' shares in our price. Its record is not started again by a fit of the other four. The
  forecast on Bitfinex spot's book (`bcbot/predictor.py`) is still made every round: `fade_hawkes` uses it.
  Without the service the rows stay empty and the page says so.
- `value` (2026-11-15, paper; "Value bet") bets the side whose chance x payout is 1.05 or more. The chance comes from the only input that predicts anything here, bc.game's gap from the market (the bet against a gap of 2 / 3 / 4 x the volatility wins about 53 / 56 / 59%; under 1 x it is a coin), with a small correction for the pools; the payout is what the page shows at the decision. So with no gap it bets a side that pays x2.12 or more, with a gap of 2 x from x1.99, 3 x from x1.88, 4 x from x1.79. Fitted and chosen on the rounds up to 2026-11-11 20:05 UTC only (walk-forward by day: 1,391 bets, +$142.72 at a flat $1); on the 4,115 rounds after it (35 h): 817 bets, 23 an hour, 51.7% won, +$73.84 = +9.0% per $1 paid at the final pools, against +$43.84 for `thin`, +$37.01 for the Gap fade from 2 x and +$19.18 for the Gap fade on the same rounds. It is not safe: about half its bets lose, runs of six losses come about 3 times a day (longest on the log: 9), the worst drop at a flat $1 was $24, and a bigger stake is paid less ($5: +6.9% per $1, $10: +4.4%). A bigger flat stake beat every stake series at the same worst drop. Its record counts the rounds after 2026-11-11 20:05 UTC; judge it by its P&L (it is paid x2.0-2.1, so break-even is 47-50% wins, not 51.55%).
- `thin` (2026-11-11, paper) makes no prediction: it takes the side the page pays x2.10 or more on at the decision (the smaller pool). On 11,140 logged rounds that side won as often as the other, and a flat $1 on it returned +14.4% per $1 on 470 bets paid at the final pools (a fair coin at those payouts: +9.9%; expect +7 to +10%). It is small (about 5 bets an hour, $8-12 a day at $1), it needs someone else to keep overloading one pool, and it only works at small flat stakes: your own stake goes into the thin pool, so a doubling series keeps about a third of the return per $1 and ends 470 bets with a loss about one time in three. No real win on a side paying x2.10+ has been checked against the balance yet. Its record counts rounds after 2026-11-11 20:05 UTC and is booked from each round's final pools. When `thin` itself makes the calls, the calling record's P&L, `pnl` in the round log and a real order's booked result are at the page's payout at the call, as for every model, so they differ from its own row by cents.
- `fade_flow` (2026-11-12, paper) is the Gap fade's bet placed only when a second opinion agrees: at least 2 of the 3 perpetual futures (Binance, OKX, Bybit) had order flow on the bet's side over the last 3 s. Of 29 second opinions measured on the Gap fade's 5,250 logged bets (HawkesSign, the P(UP) models, trend, order book, payout) it was the only one that moved the win rate: 55.7% of 1,952 against 52.1% of the bets it leaves out, in both halves of the log. It keeps about 37% of the bets and does not make losing runs rare. Picked after looking, so only its record after 2026-11-12 05:12 UTC counts.
- `fade_flow2` (2026-11-12, paper) is the answer to "how do we get fewer long losing runs": the Gap fade's bet only when the gap is at least 2 x the volatility and 2 of the 3 futures had order flow on its side. Losses are independent, so runs per day = bets per day x p x (1 - p)^k; this rule raises p (57.6% of 802 on the log, 57.7% on the older days and 57.0% of 93 on the recent ones) and bets about 230 times on a full day, which gives about 1.8 runs of five losses and 0.8 runs of six a day against 15 and 7 for the plain Gap fade. It does not make them vanish. Only its record after 2026-11-12 counts.
- `fade_hawkes` (2026-11-12, paper, the owner's rule) is the Gap fade's bet placed only when HawkesSign's forecast is on the same side. On the log that agreement did not change the win rate (53.8% of 2,463 when it agreed, 53.1% of 2,435 when it did not), so it is there to be watched next to the others. It makes no call while the Hawkes models warm up (about 5 minutes after a start).
- `fade_fair` (2026-11-11, paper) is the Gap fade against the separate new price (`bcbot/fair_price.json`); silent while that file is absent.
- `pool` is scored too but never calls: the thin side of the pari-mutuel pool whenever the page pays more than x2.0 on it. The crowd's imbalance carried no information in 1,996 logged rounds, so that is a coin paid above even money, worth about a cent per $1 after your own stake dilutes the pool. Its record (at the diluted payout) is on the dashboard next to the models; weeks of rounds are needed to tell whether it is really above zero.

Training follows the paper: the Hawkes models on the last 300 s, retrained every round, and the COE on the last 30 min, retrained every 5 min. The first calls come after about 5 minutes of order-book data.

`auto` shows the results so far: accuracy, HawkesSign accuracy, paper P&L at the payouts the page shows, and how often Bitfinex moved the same way as bc.game's price.

**Paper first.** `auto on` only records calls. With bc.game's fee, a call has to be right more than 1/payout of the time (51.55% at x1.94, 51.8% once your own stake dilutes the pool) just to break even; on 3,000+ logged rounds every model is at 48.5-50.6%. The papers' 67% is for the *next tick* on Bitfinex's own book, without fees; see `docs/decision-review-2026-10-30.md` for the full review.

## Real money

Real orders are **off at every start** and the bot never switches them on by itself.

1. Put `LIVE_ALLOWED=1` and `TOS_RISK_ACCEPTED=1` in `.env` (bc.game's User Agreement lists automated play as prohibited; that risk is yours) and, for the HTTP API / dashboard, an `API_TOKEN`.
2. In the terminal: `live arm run=<losses in a row> day=<YYYY-MM-DD>` - you must type the current run of real losses and today's date (`live` shows them). While the gate is closed (the calling model's record does not beat the break-even) you also have to add `force`.
3. Each call is then placed as a flat `AUTO_STAKE_USD` order, after a checklist run right before the click on bc.game's own clock: run of losses, no order pending, orders today, today's budget, campaign loss, balance, clock, time to the cutoff, data quality, prepared order. Refusals are logged with the round.

The bot **stops and disarms** at `AUTO_MAX_REAL_RUN` (5) real losses in a row - the count is rebuilt from the log at every start and only a real WIN resets it - at `-AUTO_DAILY_LOSS_USD` (8) for the day (checked *before* each order), after `AUTO_MAX_REAL_BETS_DAY` (60) orders in a day, at `-AUTO_CAMPAIGN_LOSS_USD` (40) in all, after `AUTO_ARM_TTL_H` hours if that is set (off by default since 2026-11-12), on a rejected / unknown / misbooked order, on a balance that does not match the booked result, and when the file `logs/KILL` exists. It never resumes: you re-arm with the command above, and beyond 5 losses in a row each re-arm allows **one** real order (a loss stops it again). `live arm ... force ladder` (or `"ladder": true` in the API body) uses the real compensate ladder (`AUTO_REAL_LADDER`, e.g. 1,2) for that arming although the gate is closed; it is logged and ends with the arming. `live reset run=<n> <reason>` (or `{"reset_run": true, "ack_run": n, "reason": "..."}` with the token) sets the run back to 0 by hand, for example for a new model: it is your decision, written to the events log, kept across restarts, and it leaves today's budget, the order count and the campaign loss as they are. Technical stops need `live clear <reason>` first. Everything is written to `logs/live_events.jsonl`.

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

Auto mode: `GET /auto` gives the accuracy, paper P&L, last call, model parameters and the real-money state (run of losses, budget, clock, gate, last events). `POST /auto` with `{"on": true}`, `{"model": "fade"}` or `{"progression": 1.5}` changes the paper settings; `{"arm": true, "ack_run": 0, "day": "2026-10-30"}`, `{"live": false}`, `{"kill": true}` and `{"clear": "unknown", "outcome": "none"}` arm, disarm, kill and clear, and need the token.

Our price (`bcbot/ourprice.py`) is a weighted average of six exchanges' bid/ask mids (Bitfinex, Bybit and OKX perpetuals, OKX spot, Binance perpetual, Coinbase), each read 0.15 s earlier and moved to bc.game's level by its median gap over the last 60 s. Bitfinex is read 0.5 BTC deep in its order book (the mid of the average prices to buy and to sell 0.5 BTC), because its thin top of book made our price jump about $5 for a fraction of a second. A 26-minute recording of 11 exchanges' quotes and trades next to bc.game (2026-11-01) found no better recipe: bc.game follows quotes, not trades, and the rest of the gap is bc.game's own wander.

The dashboard's "real orders" row arms (with the compensate ladder as an option), disarms and resets the run of real losses; paste `API_TOKEN` from `.env` into its token box once (it stays in that browser). Each arming shows a confirmation with the stake and every stop. The dashboard at http://127.0.0.1:8765/ draws bc.game's price (yellow) and our price (cyan) on one chart, with each round's window and result, and the bc.game-minus-ours strip under it. The window selector gives 60 s, 2 min or 5 min; "show the 6 exchanges" adds each exchange's own curve as the mix uses it (delayed 0.15 s and moved to bc.game's level), so you can see which exchange moves our price and which one lags. `GET /auto/prices?s=120&ex=1` returns the same data. With `AUTO_PAGE_CURVE=1` (default) the bot also draws our price as a cyan line on bc.game's own chart in the browser, aligned to the chart's price axis and to its last tick (the chart shows 60 s before it and 30 s after).

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

## Offline direction-model experiment (2026-11-11)

The isolated experiment in `tools/evaluate_ud_model.py` compares 12 logistic configurations, including volatility-normalized price and order-flow inputs. It uses chronological development folds, a separate calibration block, and a final historical holdout. The selected model scored **49.49% on 1,865 held-out rounds**, with an estimated **-4.19% return per dollar** at the final recorded pools and a flat hypothetical $1 stake. Calls above the fixed 56% confidence threshold won 5 of 15: this experiment did **not** establish a confident edge.

See [the evaluation report](docs/ud-model-evaluation-2026-11-11.md) for the protocol, comparisons and uncertainty. The frozen artifact is `models/ud_direction_v1.json`; it is paper only and is separate from the bot's calling model.

To score newly logged rounds after the model's registration without refitting or placing orders:

```powershell
.venv\Scripts\python.exe -B tools\evaluate_ud_model.py --score-forward
```

The evaluator checks the original log prefix and model/evaluator code hashes. Reproduction commands in the report use separate output files; replacing an existing experiment requires explicit `--replace`.
