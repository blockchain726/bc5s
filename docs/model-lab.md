# Continuous paper model research

The independent model lab records predictions before BC's five-second window,
settles them against BC's actual results, and trains a new challenger every six
hours after at least 300 additional observed rounds settle. It has no order
placement code and does not change the trading bot's model or real-money state.

Run `powershell -File tools\supervise_model_lab.ps1`, or use `start-all.bat`.
Results are in `logs/model_lab/results.md`; `status.json` includes the latest
prediction, missing-input reason, training status and next fit time. Run
`logs\research-venv\Scripts\python.exe -B tools\model_lab.py --status` for JSON.
The Up/Down dashboard has a **Market Memory** section (also linked in its top
navigation), showing current and challenger forward records, latest decisions,
paper P&L and the training schedule. It refreshes every five seconds. The same
paper results appear in the BC page sidebar. `GET /auto/model-lab` supplies the
read-only display. The research models run automatically; this display does not
change the bot's real-order model selector.
The local dashboard starts independently of the game connection. If the website
is unavailable, a connection notice appears while saved research results remain
readable. A region restriction is reported as-is; no login or reload loop is
attempted on that page. Existing saved stops, orders and model records are kept.
Create `logs/model_lab/STOP` to stop this research process. Its ledger survives
stops, refits and restarts. Removing STOP allows its next explicit start.

Market Memory finds similar historical states using BC's gap, short price moves,
signed queue changes at 0.25/1/3 seconds, taker pressure, book imbalance, price
response, volatility and disagreement between markets. A time-weighted nearest
state estimate corrects a smoothed gap prior. Eight combinations of memory size,
recency and correction strength are compared using expanding chronological folds,
with settlement-based 60-second embargoes. A separate past block calibrates
probabilities and the selection level, targeting about 25% of observed rounds.
Periodic production fits reserve at most 480 recent observed rounds (about four
hours) for calibration, so older newly settled live states can enter the next
six-hour memory fit. The exact live-vector/label input hash is saved per version.

The first fit's historical selected-call result was 139/258 (53.88%), with an
approximate 95% range of 47.78%-59.86%. Its estimated flat-$1 return was +4.73%,
with a wide interval including losses. This is previously examined history,
not prospective evidence or an established 60% edge. Full fit details are in
`models/market_memory_20261118/evaluation.json`.

Only newly captured predictions appear in the live results table. SQLite stores
the captured feature vector, model version, probability, selection flag, durable
commit time and later settlement. A prediction cannot be inserted after the
window or assigned to a model trained on that outcome. Ties do not count as wins
or losses. Duplicate observations and settlements are ignored. No results are
reconstructed after the fact and labelled as prospective.

The collector reads the existing local GET price API and recorder files; it does
not poll the Hawkes forecast endpoint or restart the bot. Initially its price
inputs come from the API's sampled 200 ms curves, while historical snapshots are
more precise. This input difference is recorded. Subsequent fits replace the
historical proxy for a captured round with its exact stored live feature vector
and settled result. Decisions with stale or missing books are skipped.

Model versions remain frozen. A six-hour refit becomes a paper challenger, not an
automatic replacement of a real-money strategy. A paper promotion is considered
at fixed 500-selected-call checkpoints and requires both models to have at least
500 selected forward calls, 24 distinct forward hours, a positive 99% hourly-block
interval for accuracy improvement, improved all-round log loss, and a positive
hourly-block return interval. These are conservative monitoring rules; repeated
comparisons, distribution changes and execution still limit conclusions.

Each version's model, state memory and evaluation remain in its own folder.
Historical test figures are separate from prospective results, and retiring a
challenger never deletes its ledger. CPU-intensive training runs in a separate
process with two computational threads and lower Windows priority. Public model
files contain no credentials.
