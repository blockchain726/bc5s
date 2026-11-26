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
- Up/Down real-money limits as the owner set them on 2026-11-10: no daily order count (`AUTO_MAX_REAL_BETS_DAY=0`),
  the losses-in-a-row stop is chosen on the dashboard ("Stop after", kept in `logs/auto_settings.json`, beats `.env`),
  and real orders follow the page's stake rule like paper (Flat / x1.5 / x2 on the run of real losses; no gate
  needed and no separate stake cap: the series is as long as the losses-in-a-row stop, e.g. 6 -> $1..$32). The
  owner insisted on both after being shown the loss-run table. "Day loss limit" is set on the dashboard too
  (`AutoTrader.LIMITS`). The campaign limit stays. The 4 h arming limit is off (the owner's decision,
  2026-11-12: `AUTO_ARM_TTL_H` defaults to 0, LIVE stays on until Paper is pressed or a stop is hit, also across
  restarts); do not turn it back on unless the owner asks.
- The Up/Down stake rule (Flat / x1.5 / x2) can be changed while real orders are on (the owner's decision,
  2026-11-12): the next real stake follows the new rule at the present loss run, event `stake_rule`. The calling
  model still cannot change while LIVE.
- The Up/Down "balance must be at least 20 stakes" rule is off (the owner's decision, 2026-11-12, after a real order
  was refused at a balance of $18.90): `AUTO_MIN_BANKROLL_STAKES` defaults to 0, and a real order only needs the
  balance to cover its own stake. Do not turn it back on unless the owner asks.
- An Up/Down technical stop (`tech_stop`: `currency`, `unknown`, `rejected`, `misbooked`, `mismatch`) is cleared by
  the owner on the page: the **Clear stop** button (shown only while one is in force), or LIVE, which opens the same
  dialog first (2026-11-17: the owner pressed LIVE and read "stopped: None. Check the site, then 'live clear
  currency'": a stop of 2026-11-12 17:09 UTC, "wallet currency changed from USDT to BCD", whose reason was lost
  when Paper was pressed, and the page had no way to clear it; `live clear` only exists in the console, which the
  supervisor does not have). A refusal now says what happened, when, and what to check (`AutoTrader.stop_text`,
  `TECH_STOPS`); Paper pressed during such a stop leaves it STOPPED with its reason. Clearing never arms. Do not
  clear a stop for the owner and do not make one clear itself.
- Two Up/Down real-order checks were removed by the owner on 2026-11-17, LIVE on XGBoost ("Remove this"); do not
  add them back: (1) a late bc.game tick no longer holds a real order back. The clock check only asks for a clock
  that can time the order (`BcClock.usable`: 20+ ticks in the last minute, one in the last 20 s, the 20 s and 60 s
  medians within 0.3 s; `bc_now()` uses the same), and "data quality" no longer counts `bc_ticks` (`dq.held_by`;
  a stale exchange feed, a disconnected Bitfinex book or an unknown our price still hold an order back, about
  0.3% of rounds; `dq.ok` and `dq.stale` are logged as before). (2) No 45 s pause after an order bc.game refused.
  Still there, and named to the owner: `AUTO_REJECT_STOP` (2 refusals within 10 minutes stop real orders until
  Clear stop).
- The Tap page and the Up/Down work are separate: a change to one must not touch the other.
- The new price (2026-11-11) is separate from our price until the owner says to switch: our price (`MIX` in
  `bcbot/ourprice.py`), every input computed from it and every model on them must stay as they are. The new price
  has its own recipe file, its own feeds, its own logged fields and one paper model (`fade_fair`).

## Facts measured (do not re-derive)

- Up/Down: every model is ~50% on bc.game's result except the snap-back family (`overshoot`, `prob`, `ens`), ~58-61%
  on a few hundred forward calls; the gate (3,000 forward calls, Wilson lower bound > 51.55%) is still closed.
- Up/Down research 2026-11-09 (7,242 rounds, final payout): following a move or the order book is about break-even
  (book + 5 s move 52.5%, busy markets 48.5% when following the move). Paper models `jump` (our 10 s move >= $10, 53.4%
  on the last 40%) and `trees_mid` (`bcbot/ud_trees.txt`, medium activity only, 55.9% of 771 walk-forward bets) were added
  to widen coverage beyond Overshoot; neither calls unless the owner selects it.
- Up/Down 2026-11-10 (11,534 rounds): Overshoot's fixed $ thresholds made no call in 370 rounds on 11-09 (median
  vol60 3.72). Paper model `overshoot_rel` (bet against `dev_tick` when it is >= 2 x vol60, any market: 56% of 1,021
  then 58% of 732, 2.6 times Overshoot's calls) was added, and `jump` / `trees_mid` now really call when selected.
  The same bet read at the late snapshot (1.35 s) is as good or better than at 2.9 s; the latest real order that
  got through was sent 1.56 s before the cutoff (round trip 0.92 s median), so a third paper snapshot `feat_mid`
  (`AUTO_MID_SNAPSHOT_S`, 2.2 s) is logged to measure a later decision. The `prob` model is overconfident forward
  (says 59%, gets 56%); its inputs are raw $ and not yet scaled by vol60.
- Up/Down 2026-11-10, later: Overshoot called once in 5 hours, so the paper caller is now `fade` (bet against
  `dev_tick` when it is >= 1 x vol60, `FADE_K`): about a quarter of all rounds, 54.3% of 3,831 on the log and
  51.9-59.3% on each of 6 days. Thin edge, multiplier chosen on the whole log: only its forward record counts.
  No extra condition (book, quiet / busy, bc.game's last move, the late snapshot) helped on both halves. The
  dashboard's losing-run table counts calendar days from the log and is not cut by 'P&L -> 0'.
- Up/Down 2026-11-11 (11,221 rounds): a search for a model that bets often and rarely loses 5-6 times in a row.
  Four independent searches (fitted models incl. meta-labelling, bc.game's price mechanics, losing runs, agreement
  of signals and payouts; 2,218 variants chosen on 10-30..11-01, each result re-run by a second reviewer, no leak
  found) gave 12 candidates; none beat the plain Gap fade on 11-02 / 11-09..11-11. Every refinement that gained 3-5
  points on the first days lost them later. Measured: the gap is the only input with signal (57% of it is still
  there at the cutoff, only ~8% comes back through bc.game's price inside the window); following our own move is
  48-51% at every size; pools and payouts say nothing; the fade's wins and losses are independent (autocorrelation
  +0.016; after 3 / 4 / 5 losses the next bet wins 48 / 60 / 53%), so no pause, skip or trailing-record rule
  shortens losing runs (80 variants: raising the multiplier was better every time). Runs of 6 losses per 1,000 bets
  are p x (1-p)^6: 5.3 at 53.7%, 3.6 at 56.9%, and rare only near 70%, which nothing here reaches. What is left is
  paper model `fade_strict` (the fade from 2.45 x max(vol60, $1.25)): 61.2% of 719, then 56.9% of 253 (6% of rounds,
  95% range 50.4-63.2%); the $1.25 floor did not bind on the later days. On 11-11 every gap rule was 49-53%; the
  later days are 2-3 times busier than 10-31 / 11-01 (median vol60 3.1-3.7 against 1.1-1.4), untested as a filter.
- Up/Down 2026-11-11, later: the owner had the strongly losing models removed: `mom`, `momvol`, `jump`, `book`,
  `bookmom`, `mhh`, `mhh_hi`, `agree`, `time` (`REMOVED_MODELS` in `bcbot/auto.py`; at a flat $1 since the last 'P&L -> 0', 26 h:
  48.0-50.8% on 814-3,065 calls each, -$21 to -$147; forward records since each was fixed: 48.4-51.0% on 1,505-13,882
  calls; inverted they are 49-52%, so there is nothing to improve: do not add them back). The MHH logit and `models/flow5s_logit.json` went with them; the Binance feeds in `bcbot/flow.py`
  stay (taker-flow and book inputs, the OB card). Left: `fade`, `fade_strict`, `overshoot`, `overshoot_rel`,
  `overshoot_book`, `revert`, `prob`, `gbm`, `ens`, `trees_mid`, `sign`, and `pool` (scored only). Added the same day, on paper: `fade_fair`, `thin`; on 11-12 `fade_flow`, `fade_flow2` and `fade_hawkes` (the owner asked for the Hawkes
  version: the Gap fade's bet when HawkesSign agrees; on the log 53.8% of 2,463 against 53.1% of 2,435 when it does
  not, so no gain is expected; its call waits for the Hawkes forecast). A chosen model
  that no longer exists falls back to `sign`. The one exception to the resume rule: real orders resume only with the
  model that was armed (the state file names it), so a fallback never starts real orders by itself. For `fade_fair`
  the model is its recipe: an arming on it ends at a restart that loads another recipe version or none (the state
  file keeps `fair_v` and a fingerprint of the recipe, so a file changed under the same version ends it too), and it cannot be armed while there is no recipe. So before a refit is put in place: if
  `GET /auto` shows ARMED with `fade_fair`, tell the owner that the restart will end that arming.
- Our price, measured 2026-11-11: its move in the window gives bc.game's result in 79.3% of 4,980 rounds (move
  under $1: 56%, $1-2: 66%, $2-5: 82%, over $5: 96-99.7%); the recipe was fitted on 12 minutes (2026-10-29), one
  0.15 s delay for every exchange, 43% on Bitfinex perp. It was fitted to MATCH bc.game's move, not to predict it:
  inside the window only ~8% of the decision-time gap comes back through bc.game's price, the rest is our price
  catching up (more so on busy days, when the Gap fade is 49-53%). The new price is to be fitted so that bc.game
  minus it predicts bc.game's move in the window (per-exchange weights and delays, 13 exchanges recorded). Mechanism:
  `bcbot/fair_price.json` (`read_fair_recipe`; absent = no new price, `fade_fair` silent; a refused file is reported
  with its reason on the page: weights must be above 0, `registered` in seconds, `version` a whole number),
  `AutoTrader.fair` (the same `OurPrice` class with its own mix, keeping time by our price's clock so that an
  exchange without a time stamp is read at the same instant as in the recording), logged as `feat.fair` / `feat.dev_fair` / `feat.fair_v` and `fair_move`,
  drawn green on the chart, compared with our price on the same rounds (`fair_info`). A refit = a new `version` and
  `registered` time: the record of `fade_fair` starts again. Recorder and loader: scratch `ud4/record.py`, `lib.py`
  (the bot's own price is rebuilt from the raw rows to a median $0.13). For a fit: use only exchange rows that had
  reached this PC at the decision (Hyperliquid and Binance perp arrive 0.35-0.46 s after their own time, a bc.game
  tick 0.15 s after its own; at delay 0 the loader otherwise sees up to 0.3 s more than the bot has), read the gap at
  the last bc.game tick as `_features` does, and judge the result on the bot's own logged `dev_fair`.
- Up/Down 2026-11-11, evening (11,140 rounds, every bet paid at the round's final pools): following the trend (bc.game's
  or our price's last 1-10 s, any size or big moves only) wins 47.7-51.4% and returns -0.3% to -7.6% per $1; the order
  book (Binance 1 / 5 / 20 levels) 49.9-50.5%, -2.3% to -3.5%; order flow 49.6-51.4%, -0.5% to -3.9%; book and trend
  agreeing 49.2-49.9%. Do not build models on them again. The only positive line is the PAYOUT: the game is
  pari-mutuel, the side with the smaller pool wins as often as the other (49.7%) and pays more. Paper model `thin`
  (`_thin_call`: the side the snapshot's pools pay >= 2.10 on; `THIN_MIN_PAY`): 470 bets, 52.1% wins, +14.4% per $1
  (a fair coin at those payouts +9.9%; at 48.5% wins +6.6%; break-even 45.5%); 2.00-2.10 loses 3.9%. Checked by five
  agents: holds when the threshold is chosen on one half and scored on the other (+12.9%), is not the Gap fade in
  disguise, the pools hardly move after the decision. Limits: ~5 bets an hour (1 to 12 by day: one or a few bettors
  overload one side by $20-50 and can stop), $8-12 a day at $1 flat, the best stake is about $9, $5 flat has a 42%
  chance of halving a $50 balance, and the x2 stake rule cuts it to +3% (our stake dilutes the thin pool: $32 into
  $105 is paid x1.93). bc.game really pays 1 + 0.95 x other / own at the FINAL pools with our stake inside
  (`POOL_FEE_K`; 9 real wins by balance move, within 0.007; `live_pnl` in the log is the bot's own booking, not a
  payment), but all nine were near-even pools: no real win at x2.10+ has been checked, so a cap there is not ruled
  out (order 1549405330546701, 2026-10-30 06:28 UTC, should show about x2.27 in the bet history). Settled on 2026-11-12 16:50-16:59 UTC: the owner went LIVE on `thin` and the first four real wins were credited x2.200 / x2.080 / x2.170 / x2.100 against x2.195 / x2.081 / x2.173 / x2.095 from the final pools: no cap. `thin` is booked
  from each round's final pools (`_pool_pay`; it makes no call on a snapshot that only has the previous round's
  pools, `pools_prev`), its record counts rounds after 2026-11-11 20:05 UTC, and the gate
  (win rate above 51.55%) does not fit it: judge it by its P&L. About 500 forward bets are needed.
- Up/Down 2026-11-12: the owner switched LIVE on Gap fade with x2 at 04:02 UTC; 51 real orders, 28 won (55%), and
  one run of six losses (1+2+4+8+16+32) took the balance from $85 to $22; the same 51 orders at a flat $1 made +$2.35.
  The campaign limit (-$40) stopped it; it only checks that the limit is already passed, not that the next stake
  fits (the $32 order went out with $24 of it left). The owner then had the campaign started again (05:19 UTC: `campaign_since` 0 in the state file while the bot was stopped, event `campaign_reset`; the next arming starts a new one). Gap fade's runs on the log (5,250 bets, 53.4%): 5 losses in a
  row 11.4 times per 1,000 bets, 6 in a row 6.1, 7 in a row 1.9 (15-16 / 6-11 / 1-5 on a full day); the next bet
  after k losses wins 52-55% for k = 0..4, 46.7% of 60 after 5, 68.8% of 32 after 6: no memory. The owner's idea,
  measured: bet only when a second opinion agrees. Of 29 (HawkesSign, P(UP) models, trend, order book, order flow,
  payout) only the futures' order flow moved the win rate: paper model `fade_flow` (`_fade_flow_call`: the fade's
  bet when 2+ of the 3 perps' `ofi_*_3s` are on its side) 55.7% of 1,952 against 52.1% of 3,298 left out, both halves
  of the log; 37% of the bets; runs of 5+ still 9.7 per 1,000. The gap still being there 1.35 s before the cutoff
  also separates (54.1% of 4,610 against 48.0% of 569) but is too late for a real order. Forward record only.
- Up/Down 2026-11-12, fewer long losing runs for the Gap fade (5,277 logged bets; `ud4/runs_study.py`): runs per day
  = bets per day x p x (1 - p)^k, so only the win rate of the bets taken and their number matter. Both known levers
  add up: gap >= 2 x vol 56.1%, flow 2+ of 3 55.7%, both 57.6% of 802 (older days 57.7%, recent 57.0% of 93; 233
  bets on a full day; expected 1.8 runs of 5+ and 0.8 of 6+ a day against 15.3 and 7.1 for the plain fade): paper
  model `fade_flow2`. Flow 3 of 3 with gap >= 1.5 x: 58.0% of 721; flow 2+ with HawkesSign agreeing: 57.5% of 946. A
  logistic score on 10 inputs did not carry over (fit on the older days, best-scored 30% of the recent days 51.3%
  against 52.1% for all). Stake rules on the same bets, paid at the final pools: plain Gap fade flat $1 +$211
  (worst drop $59); x2 never giving up -$127 (worst drop $827, stakes to $512); x2 giving up after 4 losses +$287
  (drop $162); x1.5 giving up after 5 +$294 (drop $88). Gap fade + flow flat $1 +$167 with a worst drop of $18.
- Up/Down 2026-11-15 (the bot stopped since 11-13 06:39 UTC; 14,895 logged rounds with pools; scratch `ud4/best1.py`
  .. `best5.py`): the owner asked for the best model from the log. Money at a flat $1 paid at the final pools;
  everything fitted and chosen on the rounds up to 2026-11-11 20:05 UTC (OLD), then scored on the 4,115 rounds after
  (NEW, 35 h). Only two things pay on both parts: the gap (the bet against it wins 49-51% under 1.5 x vol60 and
  53-58% from 1.5 x to 4 x; the Gap fade from 2 x: +8.7% per $1 of 2,065, then +9.6% of 386) and the payout (Thin
  side: +14.8% of 464, then +5.9% of 738). Paper model `value` ("Value bet", `_value_call`) puts both on one scale:
  chance of a side = logistic(0.1154 x the gap beyond 1 x vol60, up to 4 x, against the gap; 0.0831 x log of the
  pools' ratio), a bet when chance x payout >= 1.05 (`VALUE_MIN_EV`; no gap: from x2.12, gap 2 / 3 / 4 x: from
  x1.99 / 1.88 / 1.79). Shape and threshold were picked walk-forward by day inside OLD (4 shapes x 7 thresholds,
  largest profit / sqrt(bets): 1,391 bets, +$142.72). On NEW: 817 bets, 23 an hour, 51.7% won, +$73.84 = +9.0% per
  $1 (Thin side +$43.84, Gap fade from 2 x +$37.01, Gap fade +$19.18, Gap fade + flow -$12.77 on the same rounds;
  11-11 +$13.93, 11-12 +$57.15, 11-13 +$2.75); all 28 shape / threshold pairs were positive on NEW (+3% to +15%).
  Not safe: about half the bets lose; runs of 6+ about 3 a day, longest 9, worst drop $24 at a flat $1; stake $5
  +6.9% per $1, $10 +4.4% (our stake fills the pool). A bigger flat stake beat every series at the same worst drop
  (all rounds, in-sample for OLD: flat $2 +$502, drop $49; x1.5 giving up after 5 +$374, drop $51; x2 after 6 +$423,
  drop $207). Flat $1 from a $20 balance: 5% of random 7-day draws of the log's 4-hour blocks end under $1 (14.5%
  with half the edge); from $50: 0% (1.3%). `value` is booked from the final pools like `thin` (`POOL_BOOKED`); its
  record counts rounds after 2026-11-11 20:05 UTC (out of sample, but seen once before the rule went into the bot:
  truly forward rounds start when the bot runs again); the win-rate gate does not fit it (paid x2.0-2.1, break-even
  47-50%): judge it by P&L. About 500 forward bets tell +9% per $1 from zero, about 2,000 tell +4.5%. The owner
  selects it on the page (group "Chance x payout"). Since 2026-11-16 02:28 UTC the bot runs on Paper with `value` as
  the paper caller (the owner asked for the paper run; nothing was armed).
- Up/Down 2026-11-15, the exchanges as SEPARATE model inputs (outside advice the owner passed on; scratch
  `ud4/sep_feeds.py` on `ud4/fit2`, the 24 h recording, a decision every 2 s; fitted on the first 16 h, 120 s left
  out, scored once on the last 8 h, ranges from whole 300 s blocks): NO gain. Logistic regression with signed weights
  and LightGBM, on the six exchanges of our price or on all 13, with or without each exchange's own 0.3 s / 1 s move,
  betting as often as the Gap fade: 51.5-53.0% of 3,000-4,300 bets against the Gap fade's 52.7% of 3,427 (-1.1 to
  +0.4 points, every 95% range covers 0); over all decisions AUC 0.515-0.522 against 0.524 for our price's gap alone.
  Which exchange moved first 0-0.1 s ago is gone by the time an order can be sent. Unchanged prices do not happen: 0
  ties in 19,380 logged rounds. Collected: best bid / ask of the exchanges (Bitfinex 25 levels), Binance spot's top
  20 levels and buy / sell volume per second, the three perps' order flow from their best bid / ask; NOT collected:
  single trades, deeper books of the other exchanges. The round log keeps no price per exchange (only `our`, `fair`
  and `ofi_*`); only the recorder does. Do not test separate feeds again on prices alone. The recording's bets are
  decisions 2 s apart that share their 5 s result (14,360 in 8 h against about 957 real rounds): the Gap fade's
  52.7% there has a block range of 50.6-54.7%, break-even inside it.
- Up/Down 2026-11-16, the same reviewer's follow-up, on real rounds (one bet per round, flat $1 at the final pools;
  scratch `ud4/advisor2.py`): do the pressure inputs add to the Gap fade? Fitted on OLD (up to 11-11 20:05 UTC), each
  model keeps the half of the fade's bets it scores best, scored on the 1,420 fade bets after: gap size alone +7.9%
  per $1 of 422; + the perps' order flow +8.2% of 475 (+0.3, 95% -6.1 .. +6.6); + Binance book and trades +4.2%;
  + all +4.0%; LightGBM on the same inputs 0.0% to +3.5%. `fade_flow`'s rule turned round: kept / left out 55.8% /
  52.0% on OLD, 49.6% of 337 / 52.9% of 1,083 after. So NO pressure input adds to the gap. Per-$1 ranges after OLD:
  Gap fade +1.4% (-3.8 .. +6.5), from 2 x +9.6% (-0.1 .. +19.3), Thin side +5.9% (-1.9 .. +13.7), Value bet +9.0%
  (+1.8 .. +16.3). Real-order attempts in the log: 598, placed 543 (90.8%), too late 23 (12 refused by bc.game as
  closed, 11 not clicked), balance too small 12, held back by a check 20: paper counts every bet as placed.
- Value bet's forward test, fixed on 2026-11-16 BEFORE its results (the reviewer's advice: no stopping when the
  result looks good): the rule as in the code (`_value_call`, weights and `VALUE_MIN_EV` unchanged), rounds from
  2026-11-16 02:28 UTC, flat $1 paid at the final pools, the bets whose `shadow.ok` is true. Checkpoints: the first
  500 bets (21-35 h at 14-23 bets an hour) and the first 2,000. Its per-bet spread is about 1.06, so the 95% range
  is +-9.3% per $1 at 500 and +-4.6% at 2,000. Reading: at 500, under 0% = the log's +9% is not confirmed, over
  +9.3% = an edge of zero is unlikely, between = no verdict; at 2,000, over +4.6% = an edge is shown, under 0% =
  drop it. Results before a checkpoint are reported but decide nothing. Do not change the rule, its threshold or
  the bot's code during the run (the supervisor restarts the bot after a crash and would load changed code). The
  one change since, at the owner's request (2026-11-16): the scored-only order-book model rows; `_value_call` and
  the caller's path are untouched. Since 2026-11-16 06:07 UTC the page counts `value` only from the time it went
  into the bot (`MODEL_REGISTERED["value"]` = 1794796080, 2026-11-16 02:28 UTC): before, the Models table showed 612
  / 872 bets, of which 557 / 817 were the rule replayed on older logged rounds, and the owner took them for bets
  the bot had made ("that is wrong, we run it from today"). The end of its fit is now `VALUE_FITTED_UNTIL`
  (2026-11-11 20:05 UTC): the scratch scripts that split OLD / NEW by `MODEL_REGISTERED["value"]` (`best5.py`,
  `best6.py`, `advisor2.py`, `dump_rounds.py`) must use it instead. A model's row must show only bets made while
  it ran in the bot; a test on the log belongs in the answer, not in the table.
- What paper cannot see about real orders (2026-11-16, scratch `ud4/executable.py`): every paper bet carries the
  real-order checks' answer (`shadow`: ok / reason / `lead_s`), taken at the decision, 2.9 s before the cutoff. It
  does not go through the last step (balance read, then the click before cutoff - 1.5 s) or bc.game's own answer:
  since log version 3, 11 of 420 real attempts (2.6%) passed the checks and were still too late (10 not clicked in
  time, 1 refused by bc.game as closed; the 11 refusals of 10-28 / 29 were the old timing). The 22 late orders:
  11 would have won; vol60 median 2.02 against 2.14 for placed orders, 9% against 10% in fast markets (vol60 over
  4): no sign that they cluster in fast moves, on few cases. All real orders placed so far: 271 of 543 won (49.9%).
- Up/Down 2026-11-16, order books and the model comparison the owner asked for (logistic regression / LightGBM /
  XGBoost / DeepLOB on book imbalance, buy / sell pressure, spread and recent returns of ALL six markets of our
  price at their shares). The bot reads only the best bid / ask with sizes from the six (Bitfinex: 25 levels, used
  for the 0.5 BTC mid) and a 20-level book from Binance SPOT, which is not in our price; no book or trade history
  per market was ever stored. So the books are recorded since 2026-11-16 05:14 UTC (10 levels a side and the
  takers' buy / sell volume of the six markets every 100 ms, `logs/obrec/<market>_<day>_<hour>.bin`, about 1 GB a
  day; `logs/obrec/ob_check.py <folder>` shows what is in it), and `logs/obrec/record.py` (PID in `pids.txt`, 7
  days) records the prices of 13 exchanges, bc.game's ticks and the clocks (what `tools/reclib.py` reads). Research
  Python with xgboost: `logs/research-venv` (the bot's `.venv` is untouched; torch is the CPU build, the PC's
  RTX 5060 8 GB is not used). Binance futures now
  serve the book on `fstream.binance.com/public/...` and `aggTrade` on `/market/...` (the old address sends no
  aggTrade); Coinbase's `market_trades` name the MAKER's side; Bitfinex perp, 43.3% of our price, is thin (spread
  about $12, 10-15 trades a minute). First comparison, on the round log's inputs only (scratch `ud4/dump_rounds.py`,
  `models3.py`; fitted on OLD, each model bets the quarter of rounds it is surest of, 4,450 rounds after OLD, flat
  $1 at the final pools): book + pressure + spread + returns WITHOUT the gap: logistic 49.0% of 987 (-4.3% per $1),
  LightGBM 50.4% of 671 (-1.5%), XGBoost 49.1% of 674 (-3.9%), AUC 0.495-0.501, a coin. WITH the gap: logistic
  -5.0% of 782, LightGBM +3.2% of 689, XGBoost +8.0% of 415 (-1.3 .. +17.4); the gap alone in a one-input logistic
  +7.9% of 653 (+0.4 .. +15.3); Gap fade from 2 x +8.8% of 400, Value bet +8.8% of 862. No model beats the gap rule.
- Up/Down 2026-11-16, the order-book models on the dashboard (the owner wanted them as rows of the Models table,
  not only as an offline test): `ob_logit`, `ob_lgbm`, `ob_xgb`, `ob_deeplob` (`OB_MODELS` in `bcbot/obfeat.py`).
  SCORED ONLY, like `pool`: not in `MODELS` / `CALLERS`, so they cannot be selected or armed. Parts:
  `tools/ob_server.py` (its own process under `tools/supervise_ob.ps1`, research Python, port 8766: reads the six
  books and trades, writes the recording, answers `GET /ob?gap=&vol=` with every fitted model's P(UP), `GET /status`,
  and runs `tools/train_ob.py` when 3 / 24 / 72 / 168 h are recorded, then weekly; a file `logs/obrec/STOP`
  ends it), `bcbot/obfeat.py` (the 85 inputs, the same code for the fit and the live answer: per market spread,
  imbalance at 1 / 5 / 10 levels, buy / sell pressure over 1 / 3 / 10 s, its own moves, each market alone and mixed
  by its share in our price, plus the gap and vol60), `tools/train_ob.py` (a decision every 2 s of the recording,
  first two thirds / 120 s gap / last third, trees and the betting level chosen inside the fitting part; each model
  bet the quarter of decisions it is surest of until 12:44 UTC that day, every decision since; the hold-out result is in
  `logs/obrec/models/current.json`, the models in place are fitted on everything). The bot asks the service at each
  decision without waiting for it (`_ob_fetch`, `AUTO_OB_URL`, empty = off) and logs `rec["ob"]` (version, P(UP),
  betting level); each new fit starts the four records again (`_ob_roll`); the page shows the service's state under
  the Models table (`ob` in `GET /auto`). The bot was restarted with this code on 2026-11-16 05:43 UTC (the
  owner's OK; one round missed; Value bet's record went on unchanged). Fit 1 was made by hand at 05:57 UTC on only
  0.7 h (the owner did not want to wait for the 3 h mark to see bets: `python tools/train_ob.py logs/obrec` with the
  research Python; the script refuses under 0.5 h): it is noise, each model bets in about 15% of rounds; the fits
  that mean something start at 24 h. DeepLOB (`tools/deeplob.py`, added the same day because the owner asked why
  its row was "sleeping"): six markets as six channels, the last 100 rows (10 s) of each book, 10 levels a side
  (ask distance, ask size, bid distance, bid size), three convolution blocks, an inception block over time, an LSTM,
  plus each market's mid move, the takers' buys and sells, the gap and vol60; half of every batch mirrored. It is
  fitted by `tools/train_ob.py` with the other three at every fit (when torch is importable; at most the last 72 h
  of books), the last fifth of the fitting part chooses the number of passes and the betting level, and the service
  answers with it on CPU (10-35 ms for all four). torch is the CPU build (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`): the CUDA build is a 3 GB download at 0.7 MB/s here, and CPU is enough
  (about 2 ms per decision and pass: 24 h of books about 10 minutes, 72 h about an hour). Fit 2 (all four, on 1.3 h)
  went in at 06:33 UTC: noise like fit 1.
- Up/Down 2026-11-16, the order-book rows bet too rarely (the owner at 10:05 UTC: "for 4hr around, we did bet around
  10 for new models? OMG why?"). Two faults, both fixed in the SERVICE (the bot was not touched or restarted): (1) the
  rows start again at every fit and there were four fits in four hours (two by hand, the 3 h fit, and the 3 h fit a
  SECOND time at 4 h, because the service counted the next fit from the fit's usable hours, 2.95, which is under the
  3 h mark: it now counts from `recorded_h`, the hours on disk when the fit was started, `fitted_h()`); the 12 h fit
  was taken out as well (`MILESTONES` 3 / 24 / 72 / 168 h), so fit 4 (made on 4.0 h) stays until the 24 h fit of
  2026-11-17 about 05:15 UTC. (2) The betting level from the fit did not hold live: it is the 75th percentile of
  answers of models fitted on PARTS of the data, and the model put in place (fitted on everything) answers nearer to
  50%, or, for DeepLOB, with another spread at every fit. Since fit 4: logistic 17% of rounds, LightGBM 7.5%, XGBoost
  4.7%, DeepLOB 54%, where 25% was meant. Now the service sets the level itself (`Level` in `tools/ob_server.py`,
  `live_level` in `bcbot/obfeat.py`): the level that the surest quarter of the model's own last 240 answers to the
  bot reached (from 40 answers on; before that the fit's level; never under half a point; `logs/obrec/models/
  level.json` keeps the answers across a service restart; a new fit starts it again). The bot takes the level from
  each answer (`rec["ob"]["thr"]`), so it needed no change. A question with `&probe=1` or `&full=1` does not count
  as an answer: use one of them for a check by hand. That gave 19-33% of 27 rounds for each model.
- Up/Down 2026-11-16, 12:44 UTC: the order-book models bet in EVERY round (the owner: "why this models are betting
  so small? I was thinking that will bet every round"; the surest-quarter rule was mine, never theirs).
  `BET_SHARE = 1.0` in `bcbot/obfeat.py`: `live_level` then gives `MIN_THR` (0.00001: only an answer of exactly 50%
  is no bet), the fit's hold-out result becomes the plain hit rate over all test decisions, and the mechanism above
  only works again if the share is put under 1. Do not put it under 1 unless the owner asks. About 120 bets an hour
  a row; a row at 50% loses about 3% per $1 (the fee), about $3.60 an hour on paper. The four rows kept their bets
  (fit 4: the first 87-140 bets of each were made under the quarter rule). Every P(UP) is in the log
  (`rec["ob"]["p"]`), so "how did the surest quarter do" can still be answered from it. The bot was restarted at
  12:45 UTC for the page's texts (status line: "they bet in every round", no more "DeepLOB from 72 h"; Paper, no
  order open, Value bet's record went on). The owner counts bets: do not let a refit or a level leave a row near
  empty again.
- Up/Down 2026-11-16, HawkesSign on the SIX order books (the owner, on hearing that `sign` reads only Bitfinex spot,
  which is not in our price: "we need to consider all OB for these platforms since bc price follows them"). New
  scored-only row `ob_hawkes` ("HawkesSign (six order books)", `bcbot/hawkes6.py`; `sign` itself is unchanged): per
  market the paper's four event types from the service's 100 ms rows (mid up / down, buy / sell pressure at the best
  prices), one `MHP` per market on its last 300 s, fitted again every 20 s (`hawkes_loop` in `tools/ob_server.py`,
  3 worker processes, about 1 s a market; after a restart the 300 s come from the recording), and at each question
  every market is run forward 200 times at once (`simulate_net`, about 0.12 s for the six, in a fourth worker
  process; the answer waits 0.5 s at most, `late` in `GET /status`) and the six moves are mixed by `SHARES` in $
  (up minus down moves x the market's mean step). It is in `OB_MODELS` but not in `OB_FITTED`: no fit version, its
  record is not started again by a fit (`_ob_roll`), a bet in every round. The bot sends the round's window (`a`,
  `b`) with its question and logs each market's own P(UP) (`rec["ob"]["hk"]`). Bot restarted 14:28 UTC (Paper, no
  open order). Replayed on the recorded books at the bot's 1,063 logged rounds of 05:20-14:20 UTC (scratch
  `ud4/hk6_backtest.py`, `hk6_report.py`; flat $1 at the final pools): 48.4% of 1,052 (95% 45.4 .. 51.4), -5.5% per
  $1; first / second half 48.5 / 48.3%; only 10+ points from 50%: 51.4% of 465; each market alone 46.9-50.8%;
  `sign` on the same rounds 50.7% of 1,000 (-1.9%); the Gap fade 50.8% of 264. A coin, like every order-book input
  before it. Known limits: the markets are simulated one by one (no market leads another in it), and it says UP in
  57% of the rounds (the paper's rule counts a row with more size on both sides as sell pressure). The owner was
  told the replay's result. Do not tune it on these rounds.
- Up/Down 2026-11-16, 15:05 UTC: the five order-book models can be SELECTED (the owner sent a picture of the Model
  list: "I don't see the new models Logistic, lightgbm, xgboost, deeplob here"). What the paragraphs above say about
  "scored only: not in `MODELS` / `CALLERS`, cannot be selected or armed" no longer holds: all five are in
  `AutoTrader.MODELS` (so in `CALLERS`, with a paper stake book each) and in the page's list (group "Order books,
  six markets"). The chosen one's call is made in `_ob_fetch` as soon as the service's answer is in (0.04-0.5 s
  after the decision); no answer in 0.8 s, a model missing from the answer or `AUTO_OB_URL` empty = no bet, and the
  round's `skip` says which (`_ob_no_call`; shown on the BET card). Nothing was added to the order checks: arming
  one of them is the owner's decision like any model (the gate is closed for all five: its text names the fit's
  rounds). A new fit replaces a fitted model while it calls: its record and paper stake book start again
  (`_ob_roll`), and if real orders are on, the event `ob_fit` is written; there is no stop. Value bet's forward test
  counts "bets whose `shadow.ok` is true", and `shadow` is only written for the calling model: if the owner selects
  another model, Value bet's row goes on counting but the test as fixed has a hole for that time (the owner was
  told). Bot restarted 15:05 UTC (Paper, no open order; `value` still the caller).
- Up/Down 2026-11-16, 15:36 UTC: HawkesSign (`sign`) IS the six-market model, and its record was started again (the
  owner: "reset pnl and history for hawkes sign bot, we are doing it from 6 markets again so we need to monitor from
  the first again"; for them there is one Hawkes sign bot, so the separate row of the paragraph above was the wrong
  shape). `sign` is now called and scored from the order-book service's answer (`OB_ANSWER` in `bcbot/obfeat.py`:
  the bot's model -> its key in the answer, `sign` -> `ob_hawkes`; `OB_MODELS` = the four fitted ones + `sign`); the
  row and the list entry `ob_hawkes` are gone (one row, "HawkesSign (six markets)", with the order-book rows and in
  the Model list's group "Order books, six markets"). `SIGN_SIX_SINCE` = 1794843360 (15:36 UTC) is
  `MODEL_REGISTERED["sign"]`: no earlier round counts for it in any number (row, all-time, losing runs, the paper
  stake book), which is the reset; nothing was deleted from the log. The 62+ bets the six-market version made as
  `ob_hawkes` from 14:28 to 15:36 UTC are not in the row either (the owner asked for zero). The old forecast on
  Bitfinex spot's book (`bcbot/predictor.py`, `rec["pred"]`) is still made and logged every round, and `fade_hawkes`
  ("Gap fade + Hawkes") still uses THAT one (`_hawkes`): the owner did not ask to change it. Its last record as
  HawkesSign: 9,639 of 19,127 all-time (50.4%), -$119.12 on 4,217 bets since the last 'P&L -> 0'. `sign` needs the
  order-book service now (no answer = no bet, the round says why), and it is still the model a removed one falls
  back to. Bot restarted 15:30 UTC (Paper, no open order; the owner had selected `ob_xgb` on the page before, it
  stayed selected and makes the paper calls).
- Up/Down 2026-11-16, about 21:00 UTC, "any idea to increase Win Rate for new models?" (scratch `ud4/ob_ideas.py`,
  `ob_ideas2.py`; round log only, nothing changed). A bet in EVERY round has a ceiling: against the gap in every
  round is 51.1% of 17,070 logged rounds (-0.5% per $1); under 1.5 x vol60 (3 rounds of 4) it is 49.5-50.5%. The
  win rate only rises with fewer bets: the half of rounds with the biggest gap 52.3%, the quarter (from 1.45 x)
  54.4%, 15% (from 2 x) 55.6%, 10% (from 2.5 x) 56.8%. The four fitted models on fit 4 (4 h of books; 1,260
  rounds, each +-2.8 points): logistic 47.5%, LightGBM 51.8%, XGBoost 50.7%, DeepLOB 50.4%; first / second half
  50.3 / 44.6, 52.9 / 50.6, 52.7 / 48.7, 51.9 / 49.0. No way of combining them helped: their average 49.5%, 4 of 4
  on one side 48.2% of 338, each model's surest quarter 48.3 / 51.7 / 54.3 / 47.3% of 315, side by P x payout
  47.1-49.6%. DeepLOB bet WITH the gap (P(UP) 0.41 at -4 x vol60, 0.60 at +4 x; in all 291
  rounds with a gap of 1 x or more it bet against the Gap fade): NOT learned, a fault of its start weights (next
  paragraph). The other three hardly use the gap (answers 0.7-3 points from 50%). The same every-round bet read later, 9,644
  rounds with all three snapshots: 50.3% at 2.9 s, 50.9% at 2.1 s, 51.5% at 1.3 s; from 1.5 x vol60 it does not
  improve (53.9 / 53.2 / 52.7%). Offered to the owner, not built: fit the gap's effect on the bot's logged rounds
  and let the order-book inputs only add to it (an offset in the fit and the answer), 14 inputs instead of 85.
  The outside reviewer's corrections, all accepted: 51.1% is what the gap rule DID, not a ceiling ("nothing tested
  so far has shown 55% betting every round"); logistic's row IS below 50% as a single test (46.6% of 1,297 at
  00:05 UTC, 95% 43.9-49.4; no reason to turn it round); the idea's gain is unproven until scored on rounds that
  come after it; a rule that uses the payout is judged by net return (side by P x payout against the plain side,
  net per $1 on fit 4's 1,260 rounds: logistic -6.4 / -7.9%, LightGBM +0.1 / +0.8%, XGBoost 0.0 / -1.2%, DeepLOB
  -3.4 / -2.0%, each +-5.5: no difference; this is not the frozen Value bet).
- Up/Down 2026-11-17, 00:07 UTC: the reviewer's checks and its experiment (scratch `ud4/fwd_check.py`,
  `fwd_check2.py`; `logs/obrec/fwd/`). CHECKS of the fit against the bot's own log, 2,189 rounds: the fit's gap has
  the bot's sign in 99.6% of the rounds with a gap of 1 x vol60 or more (correlation 0.904, the same Gap-fade
  decision in 96.0%), its vol60 / the bot's 1.000 (correlation 0.998), its label = the round's logged result in
  100.0% of 2,181; a round's window opens 2.66 s and closes 7.66 s after the snapshot (the fit: 2.52 / 7.55); the
  service reads the books 0.10 s after the snapshot (fit 4's logistic replayed from the recorded books: the same
  side as logged in 91.9% of 1,613 at +0.10 s, median difference 0.00000; DeepLOB 93.6%). So no fault in label,
  sign, scaling or timing. What made DeepLOB bet with the gap: the gap and vol60 go straight into its last layer,
  whose weights start at random (+0.118 for the gap under the file's seed) and a fit of 27-110 steps hardly moves
  them (fits 2 / 3 / 4: +0.110 / +0.041 / +0.099 = +2.5 points of P(UP) per 1 x vol60 in fit 4). Its data said
  nothing of the kind: in fit 4's four hours the bet against the gap was 50.0% of 1,597 made-up decisions
  (gap against result: -0.005). FIXED in `tools/deeplob.py` (the two weights start at 0; the fit of 05:15 UTC is
  the first with it; the service needed no restart, it only loads weights). Both directions on the 415 rounds of
  fit 4 with a gap of 1 x or more: DeepLOB as it bet 48.4% (-6.0% per $1), turned round = the Gap fade 51.6%
  (+0.3%), ranges 43.6-53.2 and 46.8-56.4: these rounds do not tell the directions apart, the reviewer was right.
  By 4 hours of the recording the bet against a gap of 1 x or more was 50.0 / 52.5 / 51.7 / 48.2 / 49.5% of
  1,100-1,700 made-up decisions each: a weak day for the gap.
  THE FORWARD TEST "does the order book add to the gap?", fixed before its rounds: five models frozen in
  `logs/obrec/fwd/frozen.json` (sha256 7f2c7944...b890e1; `fit.py` refuses to write it again; `score.py` checks
  the LightGBM files' hashes): `gap` (one-input logistic on gap / vol60 clipped at 4, b = 0.0667 from the 15,231
  logged rounds BEFORE the recording: 1.67 / 2.50 / 3.33 points from 50% at 1 / 1.5 / 2 x vol60; side always
  against the gap), `s_logit` (the gap's logit as a fixed offset + logistic on the 11 mixed book inputs),
  `f_logit` (+ every market's own: 77), `s_lgbm` (the gap's logit as LightGBM's starting score + the 11 mixed
  inputs, gap, vol60; 25 trees), `f_lgbm` (all 85; 25 trees). Book parts fitted on 33,307 made-up decisions (18.7
  h, to 2026-11-16 23:58 UTC). Forward rounds: the bot's own, cutoff from 2026-11-17 00:07 UTC (1794874020), one
  bet a round, flat $1 at the final pools, all models on the same rounds (books of half of our price known),
  inputs replayed from `logs/obrec/*.bin` at the snapshot + 0.10 s with the bot's logged gap and vol60. Run:
  `logs\research-venv\Scripts\python.exe logs\obrec\fwd\score.py`. It prints every-round win rate and net per $1,
  the same at fixed levels (as sure as the gap model at 1 / 1.5 / 2 x vol60; each model's own surest 25 / 15%
  level, fixed from the rounds before), each model against the gap alone on the rounds where their sides differ,
  small against full, and the dashboard's rows on the same rounds. Checkpoints: 3,000 rounds (about 2026-11-18
  01:00 UTC) and 10,000 (about 11-20 11:30 UTC). Main comparison: `s_logit` against `gap`, every round; with a
  third of the sides differing its 95% range is about +-2.1 points at 3,000 and +-1.2 at 10,000. Reading at
  10,000: range all above 0 = the books add; all below +1 point = not worth building; else no verdict. Results
  before a checkpoint decide nothing; do not refit, re-freeze or change `fwdlib.py`'s inputs during the run. What
  the fitting showed first (NOT proof; the last third, 731 real rounds 17:44-23:57 UTC, models fitted on the two
  thirds before): gap alone 51.7%, +0.9% per $1; s_logit 47.9%, f_logit 48.0%, s_lgbm 48.7%, f_lgbm 50.8%;
  against the gap alone -3.8 / -3.7 / -3.0 / -1.0 points, every range covers 0. So the offered change ("teach
  the models the gap, 14 inputs") is NOT built into the service: it waits for this test.
- Up/Down 2026-11-17, HawkesSign checked against its paper (the owner: "the essay was good, but we may integrated
  wrong"; scratch `ud4/hk_review.py`, `hk_review_report.py`: 610 logged rounds of 11-16 05:19 .. 11-17 14:55 UTC,
  each market's model fitted as the service does; nothing in the bot was changed). The paper (Raffaelli, Cestari,
  Marazzina, Formentin 2026, Decisions in Economics and Finance, 10.1007/s10203-026-00570-z) forecasts the sign of
  the NEXT mid-price change of one book (Bitfinex BTC/USD, tick by tick, no costs, no delay). Its strong model is
  HawkesTime (the book-imbalance model, 0.67); HawkesSign is "a modest improvement over the benchmark" (0.55) and
  its trading line stays near zero. OUR CODE DOES THE PAPER'S JOB: the next move of a market 56.0% of 3,461
  (53-61% by market; 59.4% read from the intensities without simulating). It fades with the distance: each
  market's own move over 0-0.5 s 60.0%, 0-1 s 58.4%, 0-2.6 s 56.2%, 2.6-7.6 s 51.4%, the bot's window 50.7%; the
  mixed forecast 52.1% of 584 on our price's move in the window and 49.1% of 601 on bc.game's result (the row:
  50.9% of 2,380). At the window the forecast is the side with more up or down moves in the last 300 s in 85.6% of
  cases (a 5-minute trend; it said DOWN in 61% of these rounds), and Bitfinex perp (share x step 1.35, the book it
  reads worst: 53.1% next move, 50.0% at the window) is on the mixed side in 82%. NOT the cause: the pressure rule
  (4-26% of the pressure rows show both pressures and count as sell; a rule by the net size change gave 55.7% next
  move, 50.1% at the window) and Bitfinex perp (without it 47.7% on bc.game). The plain size imbalance at the best
  price beats HawkesSign on a market's own move at every distance (65.2% next move; 72.1 / 68.7 / 63.8 / 53.7 /
  53.5%) and is still a coin on bc.game (mixed by share 49.7%, without Bitfinex perp 50.7%); it is already an
  input of the fitted order-book models. So nothing is broken, and no change of the Hawkes model gives an edge at
  this distance: do not tune HawkesSign on these rounds and do not build variants of it.
- Up/Down 2026-11-17, "why did XGBoost bet DOWN when the chart shows it going up?" (round of 16:04:19 UTC, +$67
  on bc.game): the chart was looked at 15 s later. Decision 16:04:16.5 (prices flat for 6 s, gap -$0.25), last
  moment for an order 16:04:18.0 (still flat), the jump began at 16:04:19.3-19.5, as betting closed. XGBoost's
  answer was 48.5% (Coinbase's book almost all asks: -1.3 points; Bybit's book swinging: -0.8). The fitted models'
  answers are small leans: XGBoost is within 3 points of 50% in 1,144 of 1,195 bets of fit 5 (LightGBM in all), so
  with a bet in every round a single bet says little. Their inputs look at the books now and at the last 1 / 3 /
  10 s, nothing longer. By market speed, fit 5 so far: vol60 of 7+ XGBoost 48.3% of 60, LightGBM 39.0% of 59;
  calmer 54.2% / 54.6% of about 1,135 (too few fast rounds to conclude).
- Up/Down 2026-11-17, 16:03 UTC: the owner went LIVE on XGBoost (flat $1) and asked why real orders were missed.
  Of the first 24 rounds 11 got their real order (5 won, 6 lost). The 13 others: 5 "the order-book service did not
  answer" (no bet at all), 2 "too late to click" (the click would have come after cutoff - 1.5 s), 1 refused by
  bc.game as closed and 1 "an order was rejected in the previous round" (the 45 s pause after a rejection), 2 "data
  quality: bc_ticks" and 1 "bc.game clock not ok (last tick 1.1 s ago)" (bc.game's ticks stale at the decision),
  1 "order not prepared for this round" (the round the arming fell in). CAUSE of most: the bot's own Hawkes forecast
  (`bcbot/predictor.py`, Bitfinex spot; only `fade_hawkes` calls with it) is pure-Python simulation in a thread,
  median 0.72 s, and started at the decision: it holds the interpreter while the order-book answer, the balance
  read and the click have to happen. The service answers in 0.10 s when asked from outside the bot (24 probes, max
  0.13 s); inside the bot the answer was there after 0.38 s (median), 0.75 s (90%), and not within 0.8 s in 5.6% of
  2,570 rounds; real orders went out either at once or right after the forecast ended (2.0-2.1 s before the
  cutoff). FIX (in the code, 139 tests; LOADED ONLY AT THE NEXT BOT START, which was not made: the bot was ARMED
  and a restart of a live bot needs the owner's word): unless that forecast makes the call itself it starts 0.15 s
  after the last moment for an order and the late snapshot (`HAWKES_AFTER_S`, about 1.25 s before the cutoff) and
  is made as of the decision (`HawkesPredictor.predict(a, b, at)`: the same events, so `rec["pred"]` and the
  `fade_hawkes` row mean what they did). After the restart: check that `decided_lead_s` of an order-book caller is
  about 2.8 s and that "did not answer" is rare. The two stale-tick checks and the pause after a rejection were
  removed the same hour at the owner's word (see the rules above); they load with the same restart.
  Loaded at the restart of 16:29 UTC (the owner chose "Restart now"; the arming resumed by itself). The 12 rounds
  after it: the answer is there after 0.12 s (median, max 0.14 s), the call is made 2.75-2.87 s before the cutoff,
  9 real orders in the 10 rounds past the first minute (the first two: "page reloaded or round feed reconnected
  less than 60 s ago", the check after any restart). WHAT WAS LEFT: the balance read right before the click also
  fetched bc.game's price table whenever its copy was older than 60 s, i.e. in every second round, and that
  round trip takes 0.6-1.6 s: of 34 real attempts five were "too late to click" for it (1.30-1.64 s) and eleven
  went out 0.6-0.8 s later than they could. Fix: that read uses the table read with the round's preparation
  (`RATE_KEEP_S`, `_refresh_balance(quick=True)`, `Trader.usd_rate(currency, max_age)`). Loaded at the restart
  of 16:44 UTC (the owner chose it; the arming resumed at 16:44:24) and NOT yet seen on a real order: the owner
  pressed Paper at 16:45:36 UTC, inside the 60 s after a restart in which no real order goes out. On paper since:
  the answer after 0.10-0.15 s, the call 2.76-2.90 s before the cutoff. Real orders of that LIVE run (16:03-16:45
  UTC, XGBoost, flat $1): 42, won 19, lost 23, -$5.47 (balance $40.50 -> $35.09); 23 of them in the 15 minutes
  between the two restarts, against 11 in the first 24 rounds.
- Up/Down 2026-11-17, the owner's idea "the order book showed DOWN for a moment, we bet DOWN, a second later it
  showed UP: a steadier reading would have predicted it" (scratch `ud4/ob_stability.py`; the 1,254 rounds of fit 5,
  models replayed from the recorded books; each number +-2.8). The observation is right: a model's side is another
  one 1 s later in 23.5% of the rounds (XGBoost; LightGBM 21.9%), and a market's raw imbalance changes side within
  1 s in 23-25% (1, 5 or 10 levels alike). But a steadier reading did not predict better: XGBoost asked 4 .. 0 s
  before the decision 51.2-52.6%, at +0.5 s 54.1%, at +1 s 51.6%; its average answer over the last 1 / 2 / 4 s
  53.2 / 52.9 / 51.4% against 52.3% for the single one (LightGBM 52.6 / 51.8 / 51.3 against 53.8%); rounds where
  the side held for 2 s 53.1% of 684 against 51.4% of 570 (LightGBM 53.8 / 53.9). The raw imbalance mixed by share,
  on bc.game: one instant 51.0%, the last 1 s averaged 52.6%, 3 s 50.8%, 10 / 30 / 60 s 50.6-51.0%; on our price's
  move in the window 52.8 / 54.7 / 52.1 / 51.6 / 49.5%. So what the book says is short-lived: older than about a
  second it is a coin, and averaging over more seconds makes it worse. Only the 1 s average is not worse than one
  instant (untested as a model input; adding inputs changes `F.NAMES`, which needs a new fit at once and breaks
  the frozen forward test's scorer unless they are appended and the scorer reads the first 85).
- Up/Down 2026-11-17, "as I know we can increase the win rate till 60% around for 5-8 s prediction" (the owner;
  scratch `ud4/winrate_levers.py`). That 60% is real for a market's OWN price counted FROM THE DECISION, and we
  reproduce it with one input, the size imbalance at the best price (610 rounds x 6 markets): next 0.5 s 72.1%,
  1 s 68.7%, 2.6 s 63.8%, decision to the round's end (7.6 s) 56.9%. The bet is another question twice over:
  (a) it pays on the round's window only, which opens 2.6 s after the decision: 53.6% (the 37% of the move that
  falls before the window is the part that can be predicted); (b) the result is bc.game's price: the same reading
  (1 s average, five markets without Bitfinex perp, 3,917 rounds, +-1.6) gives 53.5% on our price's move in the
  window and 49.9% on bc.game's result. Taken LATER: 2.4 s before the cutoff 50.7%, 1.9 s 50.3%, 1.5 s (the last
  moment for an order) 51.6%, 1.0 s 51.9%, 0.5 s 52.2%, 0.1 s 52.5% on bc.game (our price: 53.7 / 53.3 / 54.2 /
  55.3 / 56.2 / 57.3%); following our price's last second: 51.4% at 2.9 s .. 52.8% at 0.1 s; both agreeing at
  0.1 s 53.9%. So even a decision at the last instant is 52.5-53.9% on bc.game: its own wobble (about $2, half
  the rounds are settled by less) hides what the books predict. How late bc.game accepts an order is NOT known:
  the bot never sent later than 1.50 s before the cutoff (430 accepted orders: latest 1.50 s, 5% 1.87 s, median
  2.47 s); the refusals as "closed" were clicks that took 2 s to leave the page, not late decisions. Said to the
  owner: no way to 60% in every round is in sight; above 55% only the big-gap rounds (10-15% of rounds).
- Up/Down 2026-11-17, 23:00 UTC, "PnL is showing wrong for XGBoost because I already lost 7 usd" (the owner, LIVE
  on XGBoost a second time, 22:35-23:03 UTC; scratch `ud4/row_vs_real.py`, `row_vs_real2.py`). No number was wrong:
  a Models row is PAPER, every round since the row's start (fit 5, 05:22 UTC), and it had fallen from +$79.22 at
  the arming to +$71.57 on the owner's picture. The owner reads the calling model's row as their real money: the
  page does not show real orders next to it (offered, not built). Every real order was on the row's side and booked
  like it; paid by the final pools the 90 real orders of the day are -$14.97 against -$15.03 booked. XGBoost's row
  by period: paper before the first LIVE 54.0% of 1,187 (+$60.06); LIVE 1 (16:03-16:45 UTC) 44.6% of 65, real
  19 of 42, -$5.47; paper 53.0% of 349; paper after the bot's restart 58.2% of 122; LIVE 2 39.6% of 53, real 20 of
  48, -$9.56. So 42% of 118 while LIVE against 54.1% of 1,658 on paper. Not the real orders (the 28 rounds of the
  LIVE time without one: 11 won), not the timing (question 2.75 s before the cutoff against 2.77 s), six books in
  every round; LightGBM in the same rounds 47.7 / 45.3%, the bet against the gap 47.7 / 62.3%. By the hour the row
  runs from 43.0% (14:00 UTC, 107 paper bets) to 59.5%, spread 4.4 points, what chance alone gives at 110 bets an
  hour. LIVE 1 was a fast market (vol60 median 5.9 against 2.9). The row as a whole: 53.3% of 1,776 (95% 51.0 ..
  55.6; break-even about 51.5% at these payouts): its edge is not shown, and both LIVE runs began after a good
  stretch on paper.
- Up/Down 2026-11-17, 23:20 UTC, the owner's idea "the current OB has a lifetime, we do not calc this and follow
  the current OB only; OB at 78% UP while BTC goes flat = the OB will turn to selling, BTC will fall", and, with a
  picture (the page's OB card +39.5% while the price fell, XGBoost UP at 50.8%, lost): "we must not follow the
  short term one" (scratch `ud4/ob_lifetime.py`: 66,721 made-up decisions 2 s apart on the recording of 11-16
  05:14 .. 11-17 23:19 UTC with bc.game's result from its recorded ticks, ranges from whole 300 s blocks; then the
  bot's 4,385 rounds at the final pools). Reading = size imbalance at the best price, five markets by share
  (no Bitfinex perp), averaged over 1 s; age = how long it has been on its present side; strong = the strongest
  third (75%+ on one side). (1) A LIFETIME IS REAL AND SHORT: a reading is on the same side 1 s later in 76.6%,
  2.6 s later (the window opens) 63.2%, 7.6 s later 55.2%; under 1 s old: 72.8 / 56.9 / 51.5%; 5-10 s old: 81.3 /
  68.9 / 58.1%; 20 s and older: 89.7 / 79.3 / 68.9%. 44% of all readings are under 2 s old. No input of the fitted
  models says how old a reading is. (2) THE REVERSAL IS NOT THERE: strong, 5 s or older and the price flat (it is
  almost never against: a best-price reading ends when the price moves against it) stays on its side 2.6 s later
  in 81.5% (other strong readings 71.3%), following it wins 64.8% on the exchanges' price to the round's end,
  51.9% of 1,132 on bc.game (halves 54.5 / 49.4%). A strong reading with the last 10 s of the price clearly
  AGAINST it (the picture's case, 6.8% of strong readings): following the BOOK wins 62.7% on the price, 53.4% of
  1,507 (+-2.7) on bc.game. On the bot's rounds the bet AGAINST the book in the owner's case won 54.8% of 93 /
  52.3% of 88 / 65.0% of 80 (1 / 5 / 10 levels; +-10: says nothing). (3) OLD READINGS ARE WORTH MORE: following a
  STRONG reading on bc.game by its age: under 2 s 49.9% of 6,095 (+-1.3), 2-5 s 51.9% of 7,095, 5 s and older
  54.1% of 9,000 (+-1.2; halves 54.1 / 54.1; 13.5% of decisions), 10 s and older 54.3% of 3,888; 5 levels 49.7 /
  51.8 / 53.8 / 53.1%, 10 levels 50.3 / 51.4 / 53.4 / 54.0%, each market alone on its own book 50.4 / 51.4 / 52.6
  / 53.4%; on the exchanges' price in the window 53.7 / 56.8 / 61.3 / 63.5%. NOT confirmed on the bot's own
  rounds: strong, by age 52.2% of 410 / 55.4% of 482 / 53.0% of 627 (+-3.9; net +2.2% per $1 +-7.6); all strong
  readings 53.5% of 1,519 (+3.8% per $1, +-4.9); every round 50.0% of 4,385 (-2.9%). Both uses were built the
  same night (next paragraph). In any study of made-up decisions leave out the repeats (8,416 of 75,137 here):
  `Rec.decisions` snaps to the last bc.game tick, so a pause in the recorder's ticks repeats one decision, and a
  NaN result must not be cast to a number.
- Up/Down 2026-11-18, 00:30 UTC, the lifetime of a reading in the bot (the owner: "let's do both as you
  recommend"). All of it is in `bcbot/obfeat.py` (`_life`, `life_market`, `life_mixed`, `life_inputs`, `build2`,
  `steady_call`): a reading = the best-price imbalance averaged over the last 1 s (`LIFE_AVG_S`), its age = how
  long it has been on its present side (a hole of more than 0.5 s in the rows ends it), counted up to 30 s
  (`LIFE_CAP_S`; the service keeps 40 s of rows, so the fit on the whole recording and the live answer get the
  same numbers: a test), the mixed reading = five markets by share (`LIFE_MARKETS`, no Bitfinex perp, half of
  their share must be known). (1) PAPER MODEL `ob_steady` ("Steady order book (strong 5 s+)", selectable, group
  "Order books, six markets"): the reading's side when 75%+ of the size is on it (`STEADY_MIN` 0.50) and it is 5 s
  or older (`STEADY_AGE_S`). The service answers it as P(UP) 0.54 / 0.46 / 0.50 (`STEADY_P`; 0.50 = no bet) with
  the reading in the answer's `life` (r, age, move in $, share), which the bot logs in EVERY round
  (`rec["ob"]["life"]`), so other levels can be read from the log later. It is in `OB_ANSWER` / `OB_MODELS`, not
  in `OB_FITTED`: a new fit does not start its record again. `STEADY_SINCE` = 1794961200 (2026-11-18 00:20 UTC)
  is its `MODEL_REGISTERED`. When it is the caller and does not bet, the round says what the books showed ("no
  steady reading: the books 61% DOWN for 2.1 s (a bet needs 75% for 5 s)"). About 13% of rounds, 16 bets an hour.
  Its forward record is the test: about 500 bets (31 h) give +-4.4 points, 1,500 bets (4 days) +-2.5; the log says
  54.1% / 53.0%, break-even is about 51.5%. Do not change its two levels on its own rounds. (2) INPUTS OF THE
  FITTED MODELS: `F.LIFE_NAMES` (16: per market `age1` = side x age / 10 and `old1` = reading x age / 10; the five
  mixed `mix5_avg1`, `mix5_age1`, `mix5_old1`, `mix5_resp1` = the price's move since the reading took its side, in
  vol60), appended: `F.NAMES2` (101) = `F.NAMES` (the 85, unchanged, as are `F.build`, `F.SIGN` and so the frozen
  forward test) + `LIFE_NAMES`. `tools/train_ob.py` fits on `NAMES2` (`OB_NO_LIFE=1`: the 85, for a comparison
  with `--dry`); DeepLOB reads the four mixed ones next to the gap and vol60 (`n_side` 6, start weights 0); the
  service gives each fit the inputs it was fitted on (`Models.load`: the names must be `NAMES` or `NAMES2`), so
  fit 5 (85 inputs) answers as before and the 72 h fit (about 2026-11-19 05:20 UTC) is the first with 101; the
  page's status line says how many inputs the fit in place has. Dry fits on the 38.2 h recorded so far, the last
  third as the test (21,780 made-up decisions, each +-0.8): 85 inputs logistic 51.6% / LightGBM 50.9% / XGBoost
  51.1%; 101 inputs 51.5% / 51.5% / 51.4% (AUC 0.524 / 0.512 / 0.515 against 0.523 / 0.514 / 0.515): NO gain that
  can be told from chance, as expected for a bet in every round (the effect sits in 13% of the decisions: +4
  points there is half a point over all). A small test fit with all four models and the service's loader on both
  kinds of fit was run in a scratch folder. Order-book service restarted 00:29:48-00:29:55 UTC (7 s without
  books), bot 00:30:13 UTC (Paper, XGBoost the caller, no open order; one round without a call).
- Up/Down 2026-11-18, 02:40 UTC, the page's P&L graph (the owner sent a picture of the Losing runs table: "let's
  show this if I click graph button and let's show up current model PnL"; they were LIVE on XGBoost). Card "P&L ·
  <calling model>" between the price chart and the Models table: a blue line = the model's paper bets as a running
  total, the SAME bets and amounts as its row in the Models table (with "All" the line ends at the row's P&L;
  compared with the running bot: 2,190 bets, +$72.39 in both), an orange line = every real order of the same time
  as the bot booked it (`live_pnl`; whichever model placed it: the round log does not name the calling model),
  both from 0 at the left edge. Ranges 1 h / 6 h / Today / All (kept in the browser); the line above the graph
  gives bets / won / $ for paper and for real money and says "LIVE now"; the pointer shows both values at a time.
  The Losing runs table is in that card, hidden until the "Losing runs" button is clicked (kept in the browser).
  Parts: `AutoTrader.curves` (per model, filled in `_count` where the row's P&L is added up; emptied by 'P&L -> 0'
  and, for the fitted order-book models, by a new fit), `AutoTrader.real_pts` (not cut by 'P&L -> 0'),
  `pnl_curve()`, `GET /auto/pnl?r=1h|6h|today|all[&model=]`, `tests/test_pnl_graph.py`. It answers the owner's
  earlier confusion (the calling row read as real money) on the page itself. How a page change is checked while
  the bot is LIVE (a script error in the page would leave the owner without the Paper button): scratch
  `ud4/preview_pnl.py` serves the new page on port 8799, passes the bot's own `GET /auto`, `/auto/panel` and
  `/auto/prices` through, refuses every POST, and answers `/auto/pnl` from a second AutoTrader on a COPY of the
  round log; `ud4/shot_pnl.py` opens it in a separate headless Edge (Playwright in the bot's `.venv`,
  `channel="msedge"`, its own temporary profile: not the bot's browser), reports script errors and takes pictures.
  Loaded at the bot restart of 2026-11-18 08:04 UTC (the owner chose "Restart now" in a question dialog, but
  answered it 5.5 hours after it was asked: the state read before the question was old, so read `GET /auto` again
  right before any restart). By then real orders were off: LIVE 3 (XGBoost, flat $1, 2026-11-18 00:36-04:26 UTC)
  made 448 real orders, 227 won (50.7%), -$7.31, and ended at the losses-in-a-row stop (9); the owner pressed
  Paper after it. Real orders since the first LIVE of 2026-11-17: 538, 266 won (49.4%), -$22.34 (balance $40.50
  -> $19.93); XGBoost's paper row over the same 26.7 h: 52.4% of 2,853, +$50.31, most of it made before the first
  real order. After a restart the page says STOPPED "9 real losses in a row" where it said Paper before: every
  start does that while the run of real losses is at the limit (`_check_start`). It is not a technical stop and
  needs no Clear stop: LIVE then allows ONE order (a loss stops it again), or the owner first sets the run back
  with Reset -> "Real loss run -> 0".
- Up/Down 2026-11-18, 08:30 UTC, the owner's case "current OB was showing Buying, but when the bet round is
  starting that has already ended and will change into Selling soon, so the final result ended up with DN; we are
  predicting wrong for these cases" (scratch `ud4/ob_end.py`: the five-market best-price reading of `ob_steady`,
  83,182 made-up decisions 2 s apart on the recordings of 11-16 05:14 .. 11-18 08:29 UTC, ranges from whole 300 s
  blocks; then the bot's 5,479 rounds at the final pools; nothing in the bot was changed).
  (1) THE CASE IS REAL AND COMMON: the reading is on the other side when the window opens (2.6 s after the
  decision) in 36.4% of decisions (clear readings, 60%+ on one side: 33.1%; strong, 75%+: 27.8%; strong and 5 s or
  older: 21.6%), and on the other side over the window as a whole in 41.1%. On the bot's rounds 36.8%. XGBoost
  (fit 5) bets on the reading's side in 70.4% of its rounds (84.2% when the reading is strong), so "our prediction
  follows the OB" is true.
  (2) WE DO WIN LESS THEN, BUT IT IS NOT "DOWN": following the reading on the bot's rounds wins 52.0% of 3,463 when
  it lasted to the window (+0.8% per $1) and 47.4% of 2,014 when it had ended (-7.8%); strong readings 54.2% of
  1,431 (+4.9%, +-5.0) and 49.2% of 559; XGBoost's own bets 53.7% of 1,875 (+4.0%) and 49.9% of 1,026 (-2.7%).
  On the exchanges' own price in the window the difference is large (lasted 60.8%, ended 43.2% for the follower),
  on bc.game it is 52.2% against 49.6%. So even KNOWING that a reading will last gives 52-54% on bc.game: the
  ceiling of this whole idea.
  (3) IT CAN PARTLY BE SEEN AT THE DECISION. Still there at the window (clear readings, 66.9% overall): by age
  under 1 s 59.8%, 10 s and more 77.8%; the book in the last 100 ms already on the other side 46.3% (7.3% of
  cases), very strong 75.8%; fading fast over 1 s 59.9%, steady 71.5%; 3 of 5 markets agreeing 61.3%, all five
  69.7%; no flip in 10 s 77.8%, six or more 58.3%; calm market 80.3%, fast 61.0%; the deeper book agrees in 97%
  (no sign). A LightGBM on 16 such signs (first 60% of the time, tested on the 32,719 decisions after 11-17 12:00
  UTC) tells lasting from ending readings with AUC 0.644: its lowest fifth is still there in 47.0%, its highest
  in 78.9%.
  (4) IT DOES NOT TURN INTO A BET ON bc.game. The owner's rule (a clear reading the model says will end: bet
  AGAINST it): 50.2% of 2,944 (+-1.9), a coin; following by the model's fifths 49.8 / 51.3 / 50.9 / 51.7 / 53.0%;
  follow a clear reading it says will last 53.0% of 6,991 (+-1.5) on the made-up decisions but 49.3% of 481 (+-4.5)
  on the bot's rounds; XGBoost without the rounds of the lowest fifth 52.0% of 1,712 against 51.7% of 2,137 with
  them; the same 16 signs fitted straight on bc.game's result: AUC 0.502. Steady order book's forward row at
  08:31 UTC: 47.3% of 169, -$14.73 (+-7.5: too few). So: no rule here; do not add these signs to the fits (they
  say nothing about bc.game's result) and do not build a "will it last" row unless the owner asks. The page's OB
  card is Binance SPOT's top 20 levels (`obi20`): not one of the six markets, a market bc.game's price does not
  follow (0% in the recalculation), above 71% on one side in half of the rounds; following it at the decision wins
  51.3% of 5,467. Offered: show the six-market reading and its age on that card instead.
- The extra order-book recording, since 2026-11-18 08:27 UTC (the owner: "record", after the recalculation of the
  weights): `tools/record_books.py` under `tools/supervise_books.ps1` writes the books (10 levels a side) and the
  trades of Bybit spot, Deribit BTC-PERPETUAL and Bitfinex spot (`bybit`, `deribit_perp`, `bitfinex`: the markets
  bc.game's price follows most, none of them one of the service's six) every 100 ms in the service's row format
  (`F.ROW`; Deribit's USD sizes turned into BTC) into `logs/obrec/extra/<market>_<day>_<hour>.bin`, about 0.5 GB
  a day. Its own process and folder: the service, its fits, the frozen forward test and `F.SHARES` are untouched,
  and NOTHING reads the recording yet. `tools\start_all.ps1` starts it with the rest; a file
  `logs\obrec\extra\STOP` ends it; log `logs\obrec\extra\record_books.log`; `tests/test_record_books.py` (its row
  equals the service's row for the same book). Do not stop it unasked. A fit on nine books needs the owner's word
  and at least a day of these files.
- Up/Down 2026-11-18, 09:59 UTC, the page's order-book card (the owner, after the study above: "ok, i want it").
  The card in the left panel no longer shows Binance SPOT's top 20 levels. It shows what the order-book models'
  markets show: the best-price reading of five of the six by share, 1 s average (`F.life_mixed`, the same numbers
  as `ob_steady` and the lifetime inputs), as "78% UP for 7.2 s" (the share of the size on that side; an age up to
  30 s), how many of the five markets are on that side, the last 2 minutes as a line, and two rows for the last
  round that was decided: "at decision" (`rec["ob"]["life"]`) and "at start" (the first reading after that round's
  window opened, with "turned" when it is on the other side than at the decision: the owner's case, now seen round
  by round). Words and colour go by the number shown (60%+ coloured, 75%+ "mostly"). Parts: `GET /life` on the
  service (`life_now` in `tools/ob_server.py`: no models, no Hawkes paths, 3-5 ms), asked by the bot once a second
  on its own connection (`_ob_life_loop`, `_ob_life_fetch`, `_ob_life_take`, `ob_life`, `ob_life_hist`,
  `OB_LIFE_EVERY_S`), drawn by `_ob_card`; no answer for 3.5 s = "the order-book service does not answer". SHOWN
  ONLY: nothing of it is logged, scored or used for a call (a round is decided on the reading in the answer to
  the bot's question, as before). Binance spot's book feed (`self.book`) still runs: `obi20` and the other logged
  inputs are unchanged. `tests/test_ob_card.py`. Do not poll `GET /ob` for a display: every question starts the
  Hawkes paths in the one simulation worker and can make the round's own answer late. Loaded: the service
  restarted 09:57:58-09:58:12 UTC (14.6 s without books; the round of 09:58:17 was answered from ONE book, `n_ok`
  1, and the four fitted rows made a paper bet on it), the bot 09:58:41-09:59:05 UTC (Paper, STOPPED "9 real
  losses in a row" as before, no order open; one round without a call: no vol60 yet). Checked first on a scratch
  copy of the service (its own folder, port 18766) behind a preview page (scratch `ud4/preview_ob.py`,
  `shot_ob.py`), then on the real page: the panel is 230 px wide, about 34 characters a line.
- State 2026-11-18 .. 11-24 (found on 2026-11-24 01:25 UTC, when the owner wrote "let's try to test the bot").
  After the card the owner went LIVE a fourth time (XGBoost, flat $1, 2026-11-18 11:48-15:23 UTC): 418 real orders,
  213 won, -$16.46; it ended at the campaign limit (-$40.32 of -$40) and Paper was pressed. All real orders since
  the first LIVE of 11-17: 956, 479 won (50.1%), -$38.80; balance $2.97. Everything was stopped at once on
  2026-11-19 02:50 UTC (STOP files in `logs/obrec`, `logs/obrec/extra` and `logs/model_lab`; not by these
  sessions) and the PC restarted on 11-23 19:53 UTC. Started again on 2026-11-24 01:27 UTC with the start-all
  script, after reading `logs/live_state.json` (state DISARMED: an ARMED state would resume real orders at a
  start, so read it before any start): Paper, XGBoost the caller, first call 01:28 UTC, the question 2.6-2.8 s
  before the cutoff, six books, fit 5 (85 inputs), 69.3 h recorded, so the 72 h fit (the first with 101 inputs)
  comes after about 2.7 h of running. No round was logged for 4 days 22.6 h: every forward test has that hole.
  Rows at the start: Value bet 50.3% of 1,360 (+$78.45), logistic 51.3% of 5,055 (-$24.41), LightGBM 52.9% of
  5,051 (+$127.68), XGBoost 52.6% of 5,055 (+$96.92), DeepLOB 50.7% of 5,018 (-$81.44), HawkesSign 50.7% of 6,204
  (-$129.58), Steady order book 49.8% of 432 (-$17.63). A real-money test is refused while the campaign is at its
  limit ("campaign loss -40.32 reached -$40" in every round's `shadow`): starting a new campaign is the owner's
  decision, like LIVE.
- Up/Down 2026-11-24, the Pool row (the owner: "i dont understand Pool mode, that shows 50% win rate, but how PnL
  can be 186usd over?"; scratch `ud4/pool_row.py`, round log only). The number is right. `pool` (`_pool_call`,
  scored only) bets the side with the smaller pool when the page pays more than x2.00 on it (`POOL_MIN_PAY`), about
  45 bets an hour, and is booked at the page's payout with our $1 put into that pool (`_diluted`). Since the last
  'P&L -> 0' (2026-11-12 06:44 UTC): 4,991 bets, 2,493 won (49.9%), an average win pays x2.076, so +$185.69 =
  +3.7% per $1 (break-even at that payout 48.2%; the models that bet both sides are paid about x1.94 and need
  51.5%). Paid as bc.game pays, at the final pools: +$178.80 = +3.6% per $1 (about +0.7 .. +6.5). By the payout
  shown: 2.00-2.10 50.8% of 3,105, +3.4% per $1 (the log of 11-11 had this band at -3.9%); 2.10-2.30 48.5% of
  1,795, +3.2%; 2.30 and more 49.5% of 91, +15.6%. First / second half of the bets +5.7% / +1.4%. The same idea
  as `thin` (from x2.10) and part of `value`; nothing was changed for it.
- Up/Down 2026-11-24, fit 6 had no DeepLOB (seen on the owner's picture: "DeepLOB not fitted yet"; its row made no
  bet from 04:14 to 16:08 UTC). Fit 6 (the 72 h fit, 04:12 UTC, 101 inputs) was the first after the 4.9-day pause in
  the recording, and `tools/train_ob.py` cut by hours on the clock: "two thirds of the time" fell into the pause, so
  the test part was the 2.75 h after it (4,908 decisions, not a third) and DeepLOB's "last 72 h" held no decision
  of the fitting part: `ob_deeplob NOT fitted: IndexError`. The three other models of fit 6 are sound (fitted on
  all 66.8 usable hours; only their hold-out line is from 2.75 h). FIXED: every cut counts recorded DECISIONS
  (`tools/fitsplit.py`: the 2/3 split, the four blocks inside the fitting part, DeepLOB's last 72 h and its last
  fifth; the book files are read without a second copy and only from the hour that is needed;
  `tests/test_fitsplit.py`), and DeepLOB says plainly when it has too few decisions. New:
  `python tools/train_ob.py logs/obrec --add-deeplob` (research Python) fits DeepLOB alone and adds it to the
  version in place: the version number stays, so the other rows keep their bets and the next fit is still at
  168 h; the service loads it within 30 s without a restart (`current.json`'s file time). Run by hand at 15:55 UTC
  with `OB_DL_HOURS=24` (the last 24 recorded hours; 2 GB of memory, 12 minutes at low priority): loaded 16:08:24
  UTC; hold-out 49.5% of 14,065 made-up decisions, AUC 0.496 (a coin, as before). NOT 72 h: the PC had 5 GB of
  memory free and 61 of 65 GB committed (the owner's two Android emulators, browsers), and a 72 h DeepLOB fit needs
  about 4 GB. The 168 h fit (about 84 h of running from here) will try 72 h again: if a fit says "ob_deeplob NOT
  fitted", run `--add-deeplob` (it refuses when the version in place has one). Look at `logs/obrec/train.log`
  after every fit: a fit that "ends with code 0" can still have left a model out. After the load the row bet from
  the round of 16:10 UTC on, and logistic, LightGBM and XGBoost kept their 1,407 bets. DeepLOB answers only when
  all six books are known (`n_ok` 6), as before. Rounds missed around it: two at 15:58-15:59 UTC (during the
  fit, which ran at low priority on 8 threads) and three at 16:08-16:10 (after it; bc.game's results came 50-70 s
  late, the service knew 4-5 of 6 books for three rounds, the Tap tab reloaded twice at 16:13 "no price for
  30 s"): a stall of the connection as far as the log shows; the fit's load is not ruled out for the first one.
- The bot was STOPPED from 2026-11-25 06:55 to 09:32 UTC at the owner's word ("stop bot, that is totally wrong, we
  must find out other model"; it was on Paper, LightGBM calling, no order open; supervisor and bot ended by hand,
  Edge left open) and STARTED AGAIN at 09:32 UTC at their word, after the integration audit ("run bot and test
  it"): `tools\start_all.ps1` after reading `logs/live_state.json` (DISARMED, no stop); Paper, `ob_lgbm` still the
  caller (the owner's last choice on the page; not changed for them), balance $2.97, every real-order check says
  "campaign loss -40.32 reached -$40". Tested on the first 31 rounds: none missed, the call 2.80 s before the
  cutoff, six books known in every round, the answer in the bot after 0.14 s, fit 6's answers rebuilt from the
  recorded books to four places in 29 of 29, bc.game's start / end prices = recorded ticks in 30 of 30, no real
  order, the price recorder gets bc.game's ticks again, the Tap page and the model lab's card answer. A stop of
  the bot is: end the supervisor (`tools\supervise.ps1`) first, then the bot's python; while it is stopped NO
  bc.game price, round or pool is recorded (the ticks reach the price recorder through `GET /auto/prices`). Do
  not stop or start it unasked. STOPPED AGAIN on 2026-11-26 09:44:54 UTC at the owner's word ("stop bot", after
  asking why Value bet earns at 50% wins; no reason given): Paper, `ob_lgbm` the caller, no order open, state file
  DISARMED; last logged round 09:44:50 UTC. Supervisor and bot ended by hand; Edge, the order-book service (fit 6,
  125.6 h recorded, next fit at 168 h), both recorders and the model lab were left running. It ran 24 h 12 min.
  Rows at the stop (paper): Value bet 50.4% of 2,006 (+$115.69), LightGBM 51.1% of 3,511 (-$59.90), XGBoost 51.7%
  of 3,510 (-$21.71), logistic 51.0% of 3,509 (-$59.79), Overshoot, scaled 55.4% of 1,050 (+$80.97), Pool 49.7% of
  6,162 (+$194.93), Thin side 48.7% of 2,452 (+$104.03). Value bet takes the side with less money in 93% of its
  bets (a win pays x2.10 at the final pools; scratch `ud4/value_side.py`); LightGBM since 11-16 on the side with
  less money +$189 of 5,613 bets, on the other -$174 of 6,711, 51-52% won on both. Do not start it unasked.
- Up/Down 2026-11-25, "which model then?" (scratch `ud4/next_model.py`, `next_model2.py`; 26,153 logged rounds with
  pools, every rule one bet a round, flat $1 paid at the final pools; A = up to 11-11 20:05 UTC, 10,780 rounds; B =
  to 11-16 02:28, 4,115; C = after, 11,258 rounds on which nothing was chosen). (1) The every-round order-book
  rows are coins on C: logistic 50.5% of 10,675 (-2.2% per $1), LightGBM 51.7% (+0.1%), XGBoost 51.5% (-0.3%),
  DeepLOB 50.9% of 9,161 (-1.2%), HawkesSign 50.5% of 9,544 (-2.2%), each +-1.8; the bet against the gap in every
  round 51.0% (-1.0%). (2) The bet against a gap of 2 x vol60 or more (`overshoot_rel`, "Overshoot, scaled" on the
  page, fixed 2026-11-10) is the one rule above break-even in every part: 55.9% of 2,065 (+8.7% per $1) / 56.2% of
  386 (+9.6%) / 55.6% of 773 (+7.5%, +-6.8); after it was fixed 647 of 1,159 = 55.8% (+-2.9), +8.2% per $1. On C:
  8.2 bets an hour, longest losing run 7, worst drop $13.91 at $1. By gap size (A / B / C): under 1.5 x 49-51%,
  1.5-2 x 52.6 / 53.5 / 52.7%, then FLAT at about 56% (2-2.5 x 53.6 / 55.4 / 56.2; 2.5-3 x 57.9 / 55.3 / 54.2;
  3-4 x 54.7 / 58.0 / 56.2; 4 x and more 58.2 / 57.8 / 54.8): a bigger gap does not reach 60%. From 1.5 x: 54.8 /
  55.0 / 54.1% (+4.8% of 1,559 on C). Calm, middle and fast markets all positive (C: +10.7 / +7.4 / +4.3%); a gap
  under $3 is weak (A: 52.2% of 714). With "our side pays 1.90 or more": 55.5% of 1,656 (+8.9%) / 58.3% of 247
  (+16.9%) / 55.6% of 511 (+10.1%, +-8.6; 5.4 bets an hour): the bets left out (pay under 1.90) made +2.4% on C; a
  small gain inside the noise. A second look 0.7 s later (gap still 1.5 x or more at 2.2 s: 56.8% of 511, +9.2%,
  gone: 53.2% of 111) and the rule read only at 2.2 s (56.6% of 791) are no better than chance allows.
  (3) Payout rules on C: Pool +2.4% of 6,594 (49.7% won; it now bets in 59% of rounds), Thin side +3.6% of 1,868
  (48.5%), Value bet 50.4% of 1,921, +5.8% (+-4.7; A / B: +10.0 / +9.0%; 20 bets an hour, longest losing run 9,
  worst drop $32); Value bet read at 2.2 s +8.2% of 2,043. Value bet gives a 2 x gap a chance of only 52.9% (its
  fitted weight), where the log says 56%: it leaves out most strong-gap rounds under x1.99. (4) Money: pools are
  about $100 a side (the smaller one: median $107), so a stake lowers its own payout. Measured on C (95 h, scratch
  `ud4/ceiling.py`, our stake inside the final pools; $ an hour / worst drop): the 2 x rule $1 +0.61 / $14, $5
  +2.32 / $71, $10 +2.94 / $144, $20 -0.10; Value bet $1 +1.18 / $32, $5 +3.80 / $181, $10 +2.84 / $407; the 2 x
  rule and, where it is silent, Value bet (2,239 bets, 23.6 an hour, 51.5% won): $1 +1.45 / $27, $2 +2.66 / $57, $5
  +4.91 / $158, $10 +4.41 / $368, $20 -10.21. So about $5 an hour is the most this game paid on paper, at $5
  stakes (the owner on 2026-11-25: "all models are bad, not sure what could be solution"). Told to the owner:
  no model here wins 60%; the 2 x gap rule (56%) is what the data supports; it was never run with real money
  (the LIVE of 11-12 was the 1 x fade with x2 stakes; the four later ones XGBoost in every round).
- Up/Down 2026-11-25, the integration audit (the owner: "keep checking the bot history, i believe we have big issue
  on integration"; scratch `ud4/audit1.py` .. `audit6.py`: the round log, the price recordings, the recorded books;
  the bot stayed stopped). EVERY LINK HOLDS. (1) Label: bc.game's winner is the direction of its own start and end
  price in 29,667 of 29,667 rounds. (2) The settling prices are ticks of the feed the bot reads: `priceStartTime` /
  `priceEndTime` are recorded ticks in 99.9% of 7,250 rounds and their prices are bc.game's start / end price in
  100.00%. bc.game's RULE: the first price is the last tick at or before the betting cutoff, the last price the
  last tick at or before cutoff + 5 s (99.9-100%; ticks are 0.53-0.55 s apart, so the window opens 0.27 s before
  the cutoff and 2.66 s after the snapshot at the median, the same in every period of the log). (3) Decision: the
  logged bc price is that tick's in 100%; the logged gap against the gap rebuilt from the recording: correlation
  0.95-0.97, median difference $0.23-0.27, the same 2 x decision in 98%, never opposite sides (the recorder has its
  own connections). (4) The models' answers: fit 6's logistic / LightGBM / XGBoost rebuilt from the recorded books
  with the FIT's code (the whole recording, not the service's 40 s of memory) give the logged P(UP) to four places
  in 98.6-98.7% of 2,981 rounds, the lifetime reading and its age in 98.7-98.9%; the service reads the books 0-0.1 s
  after the snapshot. (5) Real orders, all 1,493: the side sent is the call in every one, bc.game booked every one in
  the round decided, and the wallet moved as the final pools pay to 2 cents in 99.4% of 1,233 pairs of consecutive
  orders (the rest: the owner's own money moves of 11-12 17:06-17:08 UTC and 11-02). `_pool_pay(rec, side, stake)`
  takes a real order of ours out of the pool before it puts `stake` in: pass the stake, not 0. (6) Live as logged =
  replayed = the fit's own test part: 50-52%.
  THREE SMALL MISMATCHES, none of which moves a result. (a) The fit's window was one tick early: `Rec.decisions`
  read the last tick at or before decision + 2.52 s / + 7.55 s (the median TIMES of the real first and last tick),
  so its first price was one tick before bc.game's in 32-75% of the made-up decisions and its answer differed from
  bc.game's rule in 6-13%; by that window the 2 x gap rule wins 0.1-0.6 points more, a bet in every round the same.
  FIXED for the next fit: `R.decisions(step=STEP, cut_lead=L.CUT_LEAD)` (2.93 s) in `tools/train_ob.py`; without
  `cut_lead` the old window stays (the frozen forward test's `fwdlib.py`); `tests/test_reclib_window.py`. Dry fit
  with it on all 92 h (`--dry --no-deeplob`, last third as the test, 55,154 made-up decisions): logistic 50.3%,
  LightGBM 50.1%, XGBoost 50.1% (AUC 0.501-0.506), the 1 x Gap fade 52.7% of 14,725. (b) In 14-15% of decisions the
  bot's bc.game tick is one tick older than bc.game had sent (older than 1.2 s in 1.5-2%); in those rounds every
  exchange feed is 0.05 s older too (95%: 0.7-1.0 s against 0.35-0.5), so it is this PC or the bot's own loop, not
  bc.game. The 2 x gap rule there: 55.3% of 313 against 55.9% of 2,914: no cost seen, nothing changed. (c) The
  fit's gap is the recorder's, not the bot's (see 3).
  A FIT WITHOUT ANY MISMATCH is no better: the same kinds of model fitted on the bot's own 10,947 rounds (real
  result, logged gap and vol60, books at the snapshot; walk-forward, 8,208 test rounds, flat $1 at the final pools):
  logistic 49.8%, LightGBM 50.0-50.4%, the gap alone 51.2%; the live rows on the same rounds 50.9 / 51.7 / 51.9 /
  51.0%; the gap from 2 x 56.0% of 629 (+8.0% per $1).
  NINE BOOKS are no better than six (Bitfinex spot, Bybit spot, Deribit perp from `logs/obrec/extra`, 44.7 h,
  80,518 made-up decisions by bc.game's rule, the last 40% as the test): the gap alone 52.0% (AUC 0.526), six books
  + gap 50.7%, the three new + gap 50.9%, nine + gap 50.9%, nine without the gap 50.8%; following each market's own
  book 49.5-50.1% on bc.game's result. So no fit on nine books: do not offer it again.
  The frozen forward test at 8,744 rounds (its checkpoint is 10,000: this decides nothing): the gap alone 51.1%,
  `s_logit` 50.8% (-0.35 points against the gap, -1.61 .. +0.90), `s_lgbm` +0.70 (-0.18 .. +1.58), `f_lgbm` +0.98
  (-0.11 .. +2.08); the gap model at its 2 x level 56.3% of 662 (+8.8% per $1). Not explained: on made-up decisions
  the 2 x gap rule wins 53% (10,900 decisions 2 s apart) where the real rounds give 56% of 3,224; inside the ranges.
  Told to the owner: the integration is right, the models read a 5 s result that the books do not predict.
- The paper model lab ("Market Memory"; built on 2026-11-18 by other work, not in these sessions; read
  `docs/model-lab.md`): `tools/model_lab.py` under `tools/supervise_model_lab.ps1` (research Python; folder
  `logs/model_lab`: `status.json`, `results.md`, `paper.sqlite3`; a file `STOP` there ends it), the models
  `bcbot/market_memory.py`, `gap_pressure.py`, `ud_best.py` with `models/*_20261118/`, a "Market Memory" section
  on the page and a card in the left panel (`GET /auto/model-lab`, `bcbot/lab_display.py`), a fifth block in the
  start-all script, and a start in `main.py` that brings the dashboard up before the game page (`startup_problem`
  in `GET /auto`: "Connecting to BC.GAME", or bc.game's region notice). It places no orders and does not change
  the calling model. Its record when it was stopped: current model 87 of 192 (45.3%), challenger 115 of 236
  (48.7%). Its retraining of 2026-11-19 01:21 UTC failed (PermissionError on
  `logs/model_lab/history/data.json.tmp`); not looked into.
- Up/Down 2026-11-16: the owner had `gbm` ("P(UP) trees") removed as well ("that is bet 5 times, I don't need such
  dead bot": 5 bets since the last 'P&L -> 0', 99 in two weeks). It is in `REMOVED_MODELS`; its P(UP) is still
  computed and logged (`prob_gbm`) because `ens` is the average of it and the logistic. Do not add it back, and do
  not offer the owner models that hardly ever bet. The bot was restarted for it at 05:50 UTC.
- New price, first fit 2026-11-11 (2.2 h of recording; three fits, a chance-level run, each recipe recomputed from
  the raw rows): bc.game's tick-to-tick move is 71-73% explained by exchanges read 0-0.1 s back (Bitfinex spot 29%,
  Deribit perp 16%, Bybit perp 13%, Coinbase 11%, Bybit spot 10%, OKX spot 9%; nothing from 0.15 s up; Hyperliquid
  arrives too late to use); its own part is a wobble of about $2. Recipe version 1 (`bcbot/fair_price.json`: those
  six, level window 30 s) brings ~30% of the gap back through bc.game's own price inside the window, against 12-17%
  for our price (p 0.02-0.05), but its Gap fade could NOT be told from our price's: out of fold 52.6% against 50.3%,
  held-out part 55.0% of 389 against 55.3% of 660, and it calls ~40% less often. The regression and direct-search
  recipes were worse on the held-out part. A search on meaningless outcomes reports 53.7% out of fold (95th
  percentile 56.5%), so nothing cleared the chance level. It runs on paper because the owner wanted to see it, not
  because it was shown to be better. To see a 2-point gain: a 24 h recording (test = the last third); the page's real
  rounds would need 2.5-4 days. Fit tools (scratch `ud4`): `lib2.py` (reads the recording as the bot does: same
  Gap-fade decision as the bot's own log in 97% of snapshots), `fit_common.py`, `fit_workflow.js`. A refit must
  raise `version`.
- New price, the 24 h refit 2026-11-12 (recording 11-11 17:44 .. 11-12 17:44 UTC, 7.96 million rows; scratch
  `ud4/refit24.py` on `ud4/fit2`, `ud4/fair_oos.py`): NO recipe improves the Gap fade. Recipe 1 on the 21.7 h after
  its fit (77,754 decisions, one a second): Gap fade 53.7% of 19,101 on it, 53.6% of 25,850 on our price, difference
  +0.1 points (95% -0.6 .. +0.9). The refit tried 20 recipes (bc.game's own 2 s move regressed on the exchanges:
  R2 0.86, Bitfinex spot 35%, Deribit perp 21%, OKX 16%, Binance perp 9%, Bitfinex perp 8%, Bybit perp 8%;
  regressions of the round's move on the gaps; recipe 1's weights; level windows 10-120 s): all 53.7-55.0% on the
  first 16 h (our price 53.2%); the best one (+1.8 points there) gave 52.8% of 2,581 on the last 8 h against our
  price's 52.7% of 3,427 (+0.1, 95% -1.6 .. +1.8). 'Both prices show the gap' was 54.5% against 52.6% over the
  21.7 h but 52.4% against 52.9% on those last 8 h. So the price is not what limits the Gap fade (bc.game's own
  wobble is about $2 against a window move of $11-14): do not spend more work on recipes, do not switch the
  existing models to the new price, and no version 2 was loaded; recipe 1 stays on paper as it is.
- Our price, every exchange's weight and delay recalculated 2026-11-18 (the owner: "we need to recalc for all
  parameters of platform weight to make sure our btc price"; scratch `ud4/reweigh.py` on the two price recordings,
  50.4 h, 13 exchanges, 306,048 bc.game ticks: fitted on the first 60% of the time, chosen on the next 20%, the
  last 20% looked at once; `ud4/price_bets.py` on the round log). Nothing in the bot was changed.
  (1) THE WEIGHTS IN USE ARE NOT WHAT bc.game FOLLOWS. bc.game's 0.5 s move regressed on all 13 (weights not below
  0, delays 0-0.5 s): Bybit spot 17%, Bitfinex spot 17%, Deribit perp 15%, Bybit perp 14%, OKX spot 13%, Bitfinex
  perp 6.5%, Binance perp 6%, OKX perp 6%, Coinbase 3%, Bitstamp 3%; over 2-4.5 s Bitfinex spot 28-32%, Bybit perp
  17-20%, Bybit spot 15-16%; delays 0.02-0.15 s. Bitfinex perp, 43.3% of our price, gets 5-10% in every fit and in
  each of six 8-hour parts; three exchanges our price does not read (Bitfinex spot, Bybit spot, Deribit perp) carry
  about half. Single weights are not pinned down (between the 8-hour parts they move by 5-10 points: the exchanges
  move together), and no recipe matches bc.game tick for tick: the exchanges explain 42-70% of its 0.5 s moves and
  93% of its 4.5 s moves, the rest is its own wobble.
  (2) A MORE EXACT PRICE EXISTS AND IS ALREADY IN THE BOT. On the test part the refit of all 13 and recipe 1 (the
  "new price", fitted 11-11 on 2.2 h) are equal, and both beat our price: explained share of bc.game's 0.5 / 2 /
  5 s moves 46.5 / 71.2 / 81.7% and 45.5 / 71.8 / 81.7% against 37.4 / 66.6 / 79.2%; the same Up/Down over 5 s
  85.7 / 85.9% against 82.7% of 7,580 windows; median gap at a decision $1.54 / $1.52 against $1.98. Refitting only
  the six exchanges in use gains nothing (82.8%). On the bot's own log (9,883 rounds 11-11 20:29 .. 11-18 08:15
  UTC, all after recipe 1's fit) the round's Up/Down from our price is right in 84.2%, from the new price in 87.6%
  (+3.5 points, 95% +2.8 .. +4.1, better on each of 6 days; move under $1: 53.4 / 57.3%, $1-5: 74.6 / 82.2%, over
  $5: 96.5 / 97.7%). So recipe 1 needs no version 2.
  (3) THE BETS ARE WORSE ON THE MORE EXACT PRICE (the same 9,860 rounds, flat $1 at the final pools, both rules
  fixed before them): Value bet on our price 1,831 bets, +6.9% per $1 (+2.2 .. +11.5), on the new price 1,681 bets,
  +4.0% (-1.5 .. +9.3), difference -$58.60 (95% -105.9 .. -12.9); Gap fade from 2 x +8.0% of 820 against +1.4% of
  563, difference -$57.99 (-108.0 .. -8.9); from 1 x +0.2 / +0.3%; from 3 x +10.7% of 247 / +10.1% of 157. On the
  recording's made-up decisions the bet against the gap wins 52-53% with every recipe. So do NOT switch the
  models' price (`MIX`) to make it "right": the gap from THIS price is what pays (measured, not explained), and a
  switch would also start every gap model's record again and end Value bet's forward test. Told to the owner.
  Offered, not built: the order-book models mix six books by `MIX`'s shares (Bitfinex perp 43.3%), while the
  books bc.game follows most (Bitfinex spot, Bybit spot, Deribit perp) are not recorded at all.
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
  `TapLive._choose` has no rule of its own. No live pick while the last minute's range is 10 bands or more
  (`LIVE_MAX_RANGE60`): a market filter on the live selection, not a stop. The `mid` rule (current band, price mid-band, 2 s move under a quarter
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
- Tap 2026-11-10 (156,799 paper taps): the +25-35% per $ books are mostly one day (11-03); after 11-04 23:12 UTC
  `gbm-safe` made +17% per $ (1,245 taps), the live paper book +6% (3,525), and every picker lost on 11-09. Cells
  several pickers choose do far better than cells one chooses (4 of 5: +29% per $; one: -19%), `lr` loses under a
  score of 1.3 and `gbm` loses in the adjacent bands: paper pickers `both`, `vote4`, `lr-hi`, `gbm-r0` test that.
  Not done: the hit-rate table never forgets (all-time counts), and `gbm` / `lr` are still fitted on 11-02..11-04.
- Candidate pickers (`MODELS`) run side by side on paper, each with its own book, colour and box on the page.
  Their books are in `tap_stats.json` ("models") and `logs/tap_models.jsonl`; the page features `TAP_SHOW_MODEL`.

## Commands

- Audit 2026-11-25: `docs/bc5s-audit-2026-11-25.md`. Main bot still stopped at the
  owner's word; do not start it unasked. Model-lab scheduled fits all failed on
  Windows sharing locks (six attempts; two November 18 versions still active),
  and its report error handler could crash on the same lock again (194 supervisor
  retries in the log). Patched private temporary files + bounded replacement
  retries in `tools/atomic_files.py`, cache publication in `train_market_memory.py`,
  safe reporting and persistent training errors in `model_lab.py`. These source
  fixes have NOT been loaded into the running lab. No model artifact, feature or
  trading setting changed. 214 tests pass; a scratch preparation of the real cache
  succeeded (25,181 rows, 4,077 exact settled live vectors). Last logged 24 h:
  LightGBM 50.51% of 2,663 (-$75.38 flat $1 at final pools), XGBoost 50.90% of 2,662
  (-$54.10); 0 winner-label mismatches in fit 6's 2,984 non-tie rounds and 0
  LightGBM call-direction mismatches in 3,179 calls after the last model selection.

- Tests: `.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` (218 tests on 2026-11-25: 152 of this work, the others from `tests/test_ud_model.py`, the model lab's five files and `tests/test_atomic_files.py`). Check syntax without a pipe
  before committing (`python -c "import ast, sys; ast.parse(open(f).read())"`); a piped check hides the exit code.
- After a PC restart: double-click `start-all.bat` (`tools\start_all.ps1`): the bot under its supervisor, the
  order-book service under its supervisor, the price recorder (`logs\obrec\record.py`, 7 days) and the recorder of
  the three extra order books (`tools\supervise_books.ps1`), and the paper model lab
  (`tools\supervise_model_lab.ps1`), each only when it is not running yet; it never arms. `run.bat` alone starts the bot WITHOUT `--auto` (no calls, no paper bets) and
  neither of the other two: on 2026-11-17 the PC restarted at 15:16 UTC, the owner started `run.bat`, and "the 5 s
  bot is not working" (no automatic mode, no order-book answers). Nothing here starts by itself after a restart.
- Keep running: `powershell -File tools\supervise.ps1` (log in `logs\supervisor.log`).
- The order-book service (needed only by the order-book model rows): `powershell -File tools\supervise_ob.ps1`
  (log in `logs\obrec\ob_server.log`; state: `GET http://127.0.0.1:8766/status`). It needs `logs\research-venv`
  (`python -m venv logs\research-venv`, then `pip install numpy scipy aiohttp lightgbm xgboost` and, for DeepLOB,
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`). To load new service code: create
  `logs\obrec\STOP`, wait for the loop to end, delete the file, start `tools\supervise_ob.ps1` again.
- Git: commit identity `blockchain726 <blockchain726@users.noreply.github.com>`; branch `micro-ensemble`, kept equal
  to `main`.
