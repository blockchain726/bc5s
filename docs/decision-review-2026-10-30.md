# BC 5 s bot: decision-system review and the "no more than 5 losses in a row" question

Date: 2026-10-30. Scope: `bcbot/` (auto.py, trader.py, predictor/lob/hawkes/coe, flow.py, ourprice.py, api.py),
the logs (`logs/auto_rounds.jsonl` 3,750+ rounds, `logs/trades.jsonl` 179 order attempts) and the sister MHH bot's
recordings (6,894 rounds, 364,574 bc.game ticks, 1.1 M Binance book rows), plus the two papers the models come from.

How it was done: two passes of independent review agents. Pass 1 (11 agents) read the code and the data from eleven
angles (pipeline, models, features, streaks, money, selectivity, index noise, game rules, literature, streak math,
synthetic index). Pass 2 (14 agents) produced three competing designs (safety, engineering, edge), had two judges
score them, synthesised one plan, and then sent seven adversarial verifiers at the plan's key claims with their own
code on fresh snapshots; a completeness critic looked for what was missing. Everything below that carries a number
was re-derived by at least two agents from the raw logs; where they disagreed, the verifier's number is used.

---

## 1. The short answer

**You cannot get "at most 5 consecutive losses" from a better prediction. You can only get it from a stop rule.**

* Every model the bot has (HawkesSign, HawkesTime, the MHH flow logit, "agree") wins **48.5-50.6%** of bc.game's
  rounds. HawkesSign: 1,664 / 3,293 = 50.5% (95% CI 48.8-52.2%). The break-even at the x1.94 payout is **51.55%**
  (51.8% once your own $1 dilutes the pool). The 155 real bets won 48.4% and lost **$13.29** (-$8.48 at flat $1).
* The win/loss sequence is a fair coin's: runs of 6+ losses 23 observed vs 24.3 expected; P(loss | 1, 2, 3, 4, 5
  previous losses) = 49.2 / 48.0 / 46.1 / 47.5 / 45.2%, none different from the base rate; autocorrelation within
  +/-0.02. **Nothing in the past rounds predicts the next one, so no "resume after N paper wins", cool-down,
  thinning or confidence filter can make a 6th loss rarer.** At 50% a run of 5 starts every ~62 bets (about 30
  minutes of continuous play) and a run of 6 about once an hour; even a 70% model would meet a 6-run every ~16 h.
* The old code did not even keep the promise it made. Its counter (`loss_streak`) was reset to 0 at every "live
  on", at every paper-win resume and at every restart, so the **real** sequence had no cap at all: the log holds one
  real run of **6** (10-28 16:06-16:09 UTC) and three runs of **5**, one of them stretched over 11.4 hours across a
  balance stop and a manual "live on". Simulated days under the old rules exceeded 5 consecutive real losses on
  68-77% of days.

What was built instead (section 4): **one persistent counter of real losses in a row, rebuilt from the log at every
start, reset only by a real WIN. At 5 the bot stops and disarms. It never resumes by itself.** Re-arming is a typed
command that must name the current run; beyond 5 each re-arm allows exactly one real order, and a loss stops it
again (the counter then says 6, 7, ...). The honest guarantee is therefore: **the bot alone can never place the 6th
bet of a losing run; only you can, one at a time, knowing the count.**

---

## 2. What the data say about the edge (verified)

| Question | Finding |
|---|---|
| Does any model beat the coin on bc.game's result? | No. HawkesSign 50.5% [48.8, 52.2]; HawkesTime 48.6%; MHH 49.5%; "agree" 49.5% (n = 1,487); the sister bot's own HawkesSign 51.5% [49.8, 53.2]. All CIs include 50%, none clears 51.55%. |
| Is there a subset (hour, volatility, confidence, pools, previous results, agreement) that does? | No. 714 subset tests, 0 survive multiple-testing correction. The best cells (HawkesSign 12-15h CDT 55% of 436; HawkesSign when `lead_s >= 2.97 s` 54.2% of 1,313) are 1 of 500+ tests with no physical reason and fade on the most recent 20% of rounds. |
| Do the pools / payouts carry information? | No. Bigger pool wins 50.8% +/- 0.9%; the higher-payout side wins 49.0% [47.3, 50.8]; pools are pre-seeded ~$100-110 a side. |
| "Bet against bc.game's deviation from the market" (our price vs bc.game)? | 50.1% [48.5, 51.8] at the bot's 2.5 s lead (n = 3,474 rounds). At a 1.2 s lead and \|dev\| >= $4: 53.0% [50.5, 55.6] on 1,478 rounds, 52.6% at 1.5 s - a candidate, not an edge, and not reachable: orders must be sent >= 1.2 s before the cutoff and the click path's p95 overhead is 0.55 s. The 56.5% figure came from one 35-minute sample; a second 40-minute sample read causally gave 49.5-50.4%. |
| Does exchange information help at the decision lead? | No. Binance L1 imbalance, 1 s momentum, OFI: 48-52% at every lead >= 0.5 s before the window; 61% at -0.5 s and 73% at -1.0 s (inside the window, where you can no longer bet). A clock error of +2.75 s would be needed to make a 2.5 s lead informative; the measured error is <= 0.9 s. |
| What would perfect knowledge of the exchanges' 5 s move give? | 79.5-85.8% on bc.game's result - that is the ceiling, because bc.game's index moves on its own ~every 0.55 s. |
| Do the papers say 70% at 5 s? | No. Raffaelli et al. 2026: 0.67 (HawkesTime) / 0.55 (HawkesSign) is the sign of the **next** mid-price change on Bitfinex's own book (median 0.215 s ahead, changes only, no costs; a 5 s window spans ~23 such changes). Rustin 2026: 5-**minute** forward returns on 1-minute Binance bars, win rates 44-49%, one 48-trade rule at 55% (CI 41-68%). Neither mentions a 5-second horizon or any 70%. |
| HawkesTime | Equals -sign(BI) in 91.8% of rounds and scores 48.6%: it is now scored but can no longer make calls. |
| Progressions (x1.5, x2 "compensate") | Lose faster: with b = 0.94 and pari-mutuel dilution ($32 into ~$110 a side pays x1.72, $256 pays x1.26) a x2 ladder's 4th bet only returns the stake and the 5th/6th lose $2-8; every ladder's EV per bet is below a flat stake of the same average size (Jensen), and at the measured 49-53% they lose 1.6-6x more per bet than flat. Kelly f* = p - q/b is negative below 51.8%. |
| The 13 "Balance not enough" rejections | All were ladder stakes of 3.38 USDT / 4.0 BCD; the stake was never compared with the balance. |

Realistic prize if something is ever found: 52-56% on a subset, i.e. 1-5 cents per dollar; proving 53% needs
~8,000-9,000 forward calls; 52% is practically unprovable (14% power). The most likely verdict after three weeks of
paper is "no tradable edge".

---

## 3. Defects found in the decision system (all fixed in section 4 unless marked "later")

1. Real-loss counter reset at resume (`auto.py:622`), at "live on" (`:266`) and at every restart (`:170`, never rebuilt).
2. Daily loss limit checked **after** the loss (overshoot by one stake); the balance was parsed but never compared with
   the stake; the "since live on" -$8 limit reset at every "live on".
3. The slow order path (`trader.buy`) had no deadline; three clocks (PC, Bitfinex EMA, exchange median) timed a round
   with the PC drifting -0.05 s/h and jumping ~1 s once a day; 21 rounds had leads of -103..+29 s, 3 of them with
   real orders; replayed rounds after a reload were predicted and scored.
4. The 10 s "Clicked, but no order was sent" timeout hid a possibly placed order from every counter; a misbooked order
   (`round_id != rid`) only produced a warning.
5. Manual orders (REPL `up`/`dn`, `POST /trade`) bypassed every counter; `POST /auto {live: true}` needed no token
   (`.env` has none); the dashboard LIVE button was a browser `confirm()`.
6. Feature snapshot taken after the awaited Hawkes predict (lead median 2.0 s, half the rounds below 2.0 s in busy
   hours); `feat.dev` paired a tick up to 0.55 s old with the mix at "now" (sd $1.77 of alignment noise on a $4 signal).
7. Stale feeds frozen at their last value or 0.0 (STALE = 120 s on a feed with 43% of the weight; OFI 0.0 when silent;
   book imbalance never stale).
8. 82 WIN records with pnl -1.0 (payout 0 at write); the window start was taken as the cutoff while the real window
   starts 0.27 s before it and lasts 4.95 s.
9. `AUTO_MIN_Z` confidence gate: z is a scale, not a confidence (49.3% at z >= 2).
10. **later**: the Hawkes predict runs 1.0-1.2 s (p95 1.8 s) in busy hours under the GIL, so it now finishes less than
    1.5 s before the cutoff in 17-27% of busy-hour rounds - those rounds get no real order (fail-closed); a worker
    process / vectorised thinning is the fix.

---

## 4. What changed in the code (2026-10-30)

New: `bcbot/clock.py` (BcClock), `tests/test_clock.py`, `tests/test_safety.py`, `.env.example`,
`logs/live_state.json` (the counter), `logs/live_events.jsonl` (audit trail), `logs/KILL` (kill file).

**Real money (auto.py, config.py, api.py, main.py)**

* States `DISARMED` (every start) / `ARMED` / `STOPPED`. Arming: `live arm run=<losses in a row> day=<YYYY-MM-DD>`
  in the terminal, or `POST /auto {"arm": true, "ack_run": n, "day": "..."}` with the token. It needs
  `LIVE_ALLOWED=1` and `TOS_RISK_ACCEPTED=1` in `.env`, the exact current run and date, no technical stop, the
  15-minute cool-off after a run stop, a balance of >= 20 stakes, budget and attempts left, and the gate (section 5)
  open - or the word `force`, which is logged. An arming expires after 4 h.
* `real_run`: consecutive real losses, rebuilt at every start from `auto_rounds.jsonl` in time order and cross-checked
  against `trades.jsonl` (an order the round log never saw = unknown outcome -> stopped until you clear it) and
  against the events file (an order intent with no outcome = the process died at the click). Reset by a real WIN
  only. At `AUTO_MAX_REAL_RUN` (5): hard stop + disarm. Beyond it a re-arm permits one order (`rearm_grant`).
* Pre-order checklist `_may_place`, run right before the click on every route (automatic, manual, shadow): armed and
  not expired, no kill file, no pending or unknown real order, run limit, attempts today < 60, stake <= today's
  budget (-$8/day, checked **before** the order), campaign loss > -$40, stake <= $5, balance read <= 20 s ago and
  account REAL and >= 20 stakes, no page dialog, bc.game clock ok, >= 1.5 s to the cutoff on bc.game's clock,
  the open round is this round, no reload/reconnect in 60 s, no rejection in the previous round, forecast lead in
  [1.5, 5] s, data quality ok, order prepared for this round. Any refusal is logged in the record; no retry.
* Order path: the deadline is passed into `click_prepared` and `buy` and compared on bc.game's clock; no slow-path
  fallback for automatic orders; the balance is re-read right before the click; the intent is written to the
  events file before the click and the outcome after it; a 10 s timeout raises `OrderUnknown` -> counted as a loss,
  hard stop until `live clear unknown won|lost|none`; a misbooked order is moved to the round it was booked in and
  stops the bot; two rejections within 10 minutes stop it; ~5 s after every real settle the page balance is compared
  with the booked result (mismatch -> stop); a wallet-currency change stops it.
* Stakes: real orders are always the flat `AUTO_STAKE_USD`; `stakes x1.5|x2` now only drives the paper ladders.
  `MAX_USD` default lowered 50 -> 5.
* Kill: create `logs/KILL` (checked every second), `live off`, `POST /auto {"kill": true}`. Deleting the file does
  not re-arm.
* Manual orders (`up 5`, `POST /trade`) go through the same checklist and counters and are refused unless armed.
* API: anything that arms, disarms or places orders needs `API_TOKEN`; the dashboard's LIVE button is gone (paper /
  disarm stays); the HawkesTime button is gone.
* Removed: `AUTO_MAX_LOSS_STREAK`, `AUTO_RESUME_WINS`, `AUTO_MAX_LOSS_USD`, `AUTO_MIN_Z`, the pause/resume, `live on`.

**Timing and measurement**

* One clock: `BcClock` = median of (arrival - server stamp) over the last 60 s of bc.game's own price ticks; ok when
  >= 20 ticks, last tick <= 1 s old and the 20 s / 60 s medians agree within 0.3 s; a PC clock step is declared only
  when the disagreement persists 10 s **and** the exchange clocks moved the same way. All triggers (Hawkes at
  -3.0 s, MHH at -2.0 s, the snapshots, the deadline) and log stamps use it; a round whose window starts outside
  [1.5, 5] s from the trigger is skipped as odd timing; betting frames whose cutoff is past are ignored.
* `predict(a, b)` takes the window as offsets from now (no Bitfinex clock in the timing path); the Bitfinex clock
  offset is a median instead of an EMA.
* Snapshot A (`feat`) is taken synchronously at the decision instant **before** the Hawkes predict, which now runs as
  a task (the loop no longer waits for it); snapshot B (`feat_late`) at 1.4 s before the cutoff. Both carry the
  tick-aligned deviation `dev_tick = bc(t_tick) - our(t_tick)`, `dev_chg1`, the clock block and a `dq` block
  (ok / stale feeds / ages). Stale = None, never a frozen or zero value (mix: 5 s window and >= 75% of the weight;
  OFI: book updated within 3 s; book imbalance: 2 s; taker flow: 2.5 s).
* At settle the window is overwritten with the result frame's `priceStartTime` / `priceEndTime`; records carry
  `v: 3`.
* Shadow mode (`AUTO_SHADOW=1`, default): while disarmed, every call still runs the whole checklist and logs
  `shadow: {ok, reason}`; the panel shows would-bets, the shadow run and how many times the 5-loss stop would have
  fired today. This is the data that says whether the machine behaves before any money is at risk.
* Dashboard / panel: state, real losses in a row, orders today, budget left, campaign P&L, balance, clock source and
  offset, gate status, shadow counters; `GET /auto` exposes all of it plus the last 10 audit events.

**Verified**: 18 unit tests (clock sign in both directions, deadline on bc.game's clock, late-tick robustness, step
detection; counter rebuilt from the log in time order, ties and paper rounds ignored, stop at 5 survives a restart,
exact acknowledgement, env flags, one-order re-arm with a loss stopping again and a win resetting, paper results
never touching the real run, every checklist rule, shadow checks while disarmed, unknown orders blocking until
cleared and the clearing surviving a restart, the kill file, manual orders refused when disarmed). A replay of the
real log gives `real_run = 2` (the last two real orders lost), 155 orders, -$13.29, runs {1:20, 2:8+current,
3:3, 4:3, 5:3, 6:1} - identical to the independent count.

**Not done yet (next phase)**: the frozen forward gate (`gate.py`: a rule frozen with a config hash, judged only at
3,000 / 6,000 / 10,000 forward calls), the research registry and daily monitor, the Hawkes worker process, Bitfinex
checksum/sequence checks and feed persistence across restarts. Until the gate exists, `gate_status()` shows the
calling model's record over all logged rounds against the break-even, and arming while it is closed needs `force`.

---

## 5. What to expect

* With the rule set (stop at 5, -$8/day pre-checked, 60 orders/day, flat $1) **and** today's 50-51% models, the bot
  places about 30-45 real orders a day in steady state (most days start with you re-arming at run >= 5, and that
  single order loses on ~46% of those days - a 6th consecutive loss decided by you, not by the bot). Expected
  P&L per day: **-$0.9 / -$0.35 / +$0.34 / +$3.6 at a true 50 / 51 / 52 / 55%**; worst 5% of days about -$7.5;
  no day can lose more than $8; a no-edge campaign hits the -$40 stop in about a month.
* The caps cost upside only if an edge exists: a true 55% model would earn ~$190/day betting every round versus
  ~$3.6/day under the rules. That is the price of the "<= 5" promise.
* Your balance is $50.97: the bankroll floor is 20 stakes ($20 at $1), so there is room for ~30 losing dollars before
  the floor refuses orders. Keep only what you are prepared to lose on the account: bc.game's User Agreement
  (2.1.13) lists automated gameplay as prohibited and 8.1/8.2 let it withhold winnings; no rule here reduces that
  risk, the audit trail only documents what happened. `TOS_RISK_ACCEPTED=1` is your explicit decision.
* Timing is a hard limit: the information in exchange data appears only inside the 5 s window, while orders must be
  sent >= 1.2 s (now 1.5 s) before the cutoff. Fixing the clock makes every measurement honest but does not add
  accuracy by itself.
* Daylight saving changes on 2026-11-01: the daily caps use the local calendar day, so that day is one hour longer.

---

## 6. The research plan (paper only, pre-registered)

1. **Clock audit (1-2 days)**: with the new log fields, the Hawkes trigger must land at 3.00 +/- 0.10 s before the
   cutoff across a daily PC clock step; 0 rounds with a lead outside [1.5, 5] s reaching a call; clock not-ok <= 2%.
2. **Shadow week (7 days, no money)**: zero invariant violations (a shadow run > 5, an order while one is pending, a
   stake above the budget, a click after the deadline, an unknown order, a balance change without an order, a
   counter mismatch after a restart); the shadow stop frequency should match the simulation (a 5-run stop on ~56%
   of days, ~38 would-bets a day).
3. **Order-path timing without orders**: the site's GET round trip (`trader.last_rtt`) p95 < 0.5 s keeps the 1.5 s
   margin; it may later be set to 1.2 s + p95, never below 1.2 s.
4. **Frozen forward test of HawkesSign as it is**: open only if >= 54.8% at 3,000 calls, 53.2% at 6,000 or 52.6% at
   10,000, with both halves >= 51.5% and paper P&L > 0; abandoned below 50.5% at 3,000 or 51.2% at 6,000. Expected:
   futility within ~1.4 days.
5. **Late tick-aligned deviation** on our own 6-exchange mix at the 1.4 s snapshot: bet against `dev_tick` in band A
   (2 <= |dev| < 8 $, and a >= 8 $ band) and band B (|dev| >= 3 x vol60); family of two, same looks; paired
   comparison of the 3.0 s vs 1.4 s snapshots on identical rounds. Expected: +1-2 points over 50%, below break-even.
6. **Where the Hawkes information is lost**: score HawkesSign on its own Bitfinex per-window target next to bc.game's
   result; if it is ~55% on its target and ~51% on bc.game, the index destroys it and the Hawkes stack is retired to
   research.
7. **HawkesSign 12-15h CDT** as one pre-registered test over its next ~450 calls (needs >= 55.4% to mean anything;
   expected to fail). Register hours in UTC.
8. **Programme futility**: if after 21 days no registered rule shows >= 52.5% on >= 1,500 qualifying calls, the
   verdict is "no tradable edge" and the bot stays paper-only.

Rejected (with the numbers that killed them): resume after N paper wins or any reset other than a real WIN; a
rolling or re-evaluated go-live gate (false-opens 31-93% of no-edge campaigns); progressions, Kelly, scaling the
stake down to the balance; skipping on the clock offset's size (would block 20% of rounds); a live decision at
1.5 s with a 1.2 s send; the payout/EV gate as an edge; any cell chosen from past data as a rule; HawkesTime as a
caller; shuffled cross-validation on the feature rounds; a 70% target anywhere.

---

## 7. The Tap game ("pick multiple points for a safe profit")

Not safe, and probably negative. On our own multiplier port some own-band cells show +4-10% EV, but the real
multipliers bc.game shows for BTC are about 21% lower than our port (the server-side volatility setting changes),
which turns those cells negative. Picking several cells per column adds their (negative) EVs; it does not reduce the
variance of the sum enough to matter. Until the real multipliers are logged next to ours for a few thousand columns,
treat every Tap "edge" as unverified.

---

## 7b. Why the win rate is ~50% (follow-up, 2026-10-31)

Measured on 48 h of Binance's public BTCUSDT 1 s bars (2026-10-29 04:18 to 10-31 04:21 UTC) lined up with the logged
bc.game rounds of the same hours (decision 2.2 s before the cutoff; bc.game's window starts at the cutoff, median
0.000 s):

| Link | Rate |
|---|---|
| Binance 3 s momentum -> Binance's own move over bc.game's window | 56.5% [54.6, 58.5] (n = 2,504) |
| Binance's window move has the sign of bc.game's result | 77.8% (best alignment +0.5 s: 78.5%, $-move correlation 0.88) |
| Binance 3 s momentum -> bc.game's result | 50.9% [49.1, 52.7] (n = 3,068) |
| Our exchange-mix 5 s momentum (`our_mom5`, snapshot A) -> bc.game's result | 52.6% [50.3, 54.8] (n = 1,853) |
| HawkesSign on the same rounds | 50.6% [48.8, 52.5] |
| Break-even at x1.94 | 51.55% |

1. A 5 s BTC move is only slightly predictable at the time a bet has to be placed: on Binance's own price, 3 s
   momentum gives 56-59% for windows starting 0-3 s after the decision (both last-trade and 1 s VWAP prices; a
   linear model on 1/3/5/10/30/60 s returns and taker flow adds nothing: 57.8% at a 1 s gap, 55.6% at 3 s). It only
   becomes strong once the window has started (65-75% when 1-3 s of the window are known), when betting is closed.
2. bc.game settles on its own blended index, which has Binance's sign in only ~78% of windows (at 1 s resolution
   Binance is flat in ~18% of them). That dilutes 56.5% to ~51-52% on bc.game's result.
3. The models in use do not capture even that: HawkesSign forecasts the next Bitfinex tick (~0.2 s), not a 5 s
   window starting ~2 s later.

Consequence: the momentum model `mom` (the sign of `our_mom5` at snapshot A) was added as a caller and made the
paper caller. It was the best of ~15 signals looked at on the logged rounds, so its 52.6% is optimistic; it is
fixed from now on and only its forward record counts (the go-live gate needs >= 3,000 calls with the Wilson lower
bound above 51.55%). Scripts: `scratchpad\winrate\lead_decay.py`, `vwap_check.py`, `chain.py`, `align.py`.

Follow-up the same night (live test, order book, parameters, compensate):

- Live test of `mom` (flat $1): 4 wins / 9 losses, -$5.24; the stop at 5 real losses in a row worked (state file
  write now retries when Windows has the file locked).
- Order book vs price trend (2,000 rounds, decision snapshot): strong Binance 20-level book AND the 5 s price trend
  the same way -> 55.4% UP / 53.0% DOWN; a strong book against the trend (e.g. buying book, falling price) -> 49% UP,
  a coin flip, so it is skipped, not bet the other way. Models `book` (|obi20| >= 0.7) and `bookmom` (book and
  trend agree) added; `bookmom` is the paper caller.
- A 30-feature logistic regression (book, flows, momentum, deviations, interactions), walk-forward: 49.4% out of
  sample (1,190 rounds) - too few rounds for a fitted model. Walk-forward re-tuning of 67 simple rules (parameters
  picked on earlier rounds only): 52.2% [48.2, 56.1] on 613 later rounds; it picks "strong book + trend agree"
  every time, and the fixed rule scores 54.0% [50.1, 57.8] on the same rounds (but it was chosen on all data).
- Compensate, simulated under the real rules (stop at 5, -$8/day): at 50% every ladder loses ~$1.75/day; at 54% the
  1-2 ladder makes +$4.52/day vs flat +$3.95 with the same worst days; 1-2-4 adds days that hit -$8 (4-7%).
  Implemented as `AUTO_REAL_LADDER=1,2`: the real stake follows the persistent loss counter, and is used ONLY while
  the go-live gate is open; for `mom`, `book`, `bookmom` the gate counts only rounds after they were fixed
  (`MODEL_REGISTERED`), so the ladder cannot start on in-sample numbers.

Forward results after ~20 h (2026-11-01): `mom` 50.9% on 2,393 calls, `bookmom` 48.8% on 781, `book` 47.7% on 1,414:
every rule chosen on the logged rounds fell back to a coin flip on new rounds.

Why, in one number (7,656 rounds): bc.game's 5 s move is under $1 in 31% of rounds and under $2 in 46% (median
$2.36). In those rounds even our 6-exchange price measured DURING the window has bc.game's sign only 55% (< $1) and
69% ($1-3); above $6 it is 98%. So about half the rounds are settled by bc.game's own index wobble, which nothing
outside bc.game can see; momentum there is 50.4-50.7%, on moves above $6 it is 53-55%. Our side's payout is not
worse than the other's (x1.943 vs x1.942, break-even 51.47%). Momentum by the last minute's volatility: 49.9% calm,
51.9% normal, 53.4% [50.8, 56.0] in the most volatile third -> model `momvol` (momentum only when vol60 >= 2.8),
fixed 2026-11-01 07:09 UTC, scored forward.

Context study (5,445 rounds; rules picked on the first 60%, scored on the last 40%): market (vol60 terciles) x book
(obi20 +-0.5) x the price move over the last 2 s. With OUR price's move every picked rule failed on the test part
(44-50%); with bc.game's OWN price move the picked rules held (52-59%). Underneath is one simple effect: in a quiet
market (vol60 < 1.64) bc.game's index undoes its last 2-second move - betting against it won 54.1% of 603 / 54.1% of
828 (|move| > $0.5), 54.0% / 56.0% (> $1), 54.9% / 57.7% (> $2) on the two parts; no effect in busy markets. The
won and lost rounds of `mom` had the same book, volatility and move sizes (nothing separates them). Model `revert`
(quiet market, |bc_mom2| > $0.5, bet the other way), fixed 2026-11-01 13:30 UTC, made the paper caller; 53.0% of
1,451 on the whole log.

Overshoot search (576 reversal rules: market, which bc.game move 1/2/5 s and its size, whether bc.game sits ahead of
our exchange price in the same direction (dev_tick), book with/against; picked on the first 60% of 4,995 rounds,
judged on the last 40%): 27 of 339 rules with enough calls reached >= 55% on BOTH parts (about 2 expected by chance;
the rules overlap, so this is suggestive, not a proof). They share one effect: bc.game's index ran ahead of the
market (it moved and sits ahead of the exchanges in that direction) and snaps back. Fixed 2026-11-01 13:35 UTC:
`overshoot` (vol60 < 1.92, |bc_mom2| > $0.5, dev_tick same way > $2): 59.6% of 240 -> 57.1% of 226, 58.3% of 470
on the whole log, ~10 calls/h - the paper caller; `overshoot_book` (any market, |bc_mom2| > $0.5, dev > $1, Binance
book |obi20| >= 0.5 against the move): 60.3% of 199 -> 55.7% of 149, 58.3% of 348.

Statistical decision (`prob`, bcbot/prob.py, fixed 2026-11-01 14:04 UTC): a regularized logistic regression on 22
decision-time features + 6 "ran ahead of the market" combinations, refitted hourly on all settled rounds, calling
only when |P(UP) - 0.5| >= 0.06.  Walk-forward (blocks of 400 rounds, each predicted by a fit on everything before
it; 3,446 rounds): 52.4% on all rounds, 52.8% of 1,935 at margin 0.02, 54.3% of 915 at 0.04, 57.7% [53-62] of 395
at 0.06 (12.6 calls/h), 55.8% of 190 at 0.08.  The largest weight by far is dev_tick (bc.game's gap from the exchanges).
On the same rounds the hard rules: overshoot 59.0% of 393 (partly in-sample), overshoot+book 57.3% of 232, quiet
reversal 53.3% of 1,198, momentum 50.6%.  So the model equals the best rule and replaces the stack of filters with
one dial (margin <-> calls/h).  The paper caller since 14:04 UTC.  Also logged from now on: dev_mean60, dev_std60,
dev_z60 (the gap's z-score against the last minute), for the next model version.

---

## 8. Where the evidence is

Review outputs (per-agent reports, scripts, outputs) under
`C:\Users\Administrator\AppData\Local\Temp\claude\d--Workspace-BC-BTC\9301e543-83c4-459c-8ec4-f9174b863842\scratchpad\`:
`wf1\digest2\*.md` (pass 1, 11 reports), `wf2\digest\design-*.md`, `judge-*.md`, `synthesize-plan.md`,
`verify-*.md` (7 verdicts), `critic.md`, and the verifiers' scripts in `wf2\verify-*\`. Paper texts:
`s10203-026-00570-z.txt` (Raffaelli et al.), `ssrn-6444920.txt` (Rustin).
