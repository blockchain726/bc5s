# BTC Up/Down model evaluation

Created 2026-11-11T15:19:45+00:00. Research and paper scoring only.

Verdict: **no demonstrated confident edge**. A confidence threshold is a selection rule, not a validated accuracy claim.

## Protocol

The input log was frozen before fitting. The earliest 65% of usable rows form development; the next 15% calibrate probabilities; the latest 20% form the historical holdout, which never selects features, regularization or calibration. Three chronological development folds select the feature family and ridge strength by log loss, using the one-standard-error rule. Sixty seconds are purged before each boundary. Only decision snapshots with valid data quality, a fresh tick, 1.5-4.5 seconds to the actual window and a consistent approximately five-second settlement label qualify.

This corpus has already been examined for earlier strategy research. It is a holdout for this new fit and selection procedure, not a never-seen dataset or a prospective profitability proof.

Payout selection never reads future pools. Reported return is estimated afterward at the final recorded pools, including dilution by a hypothetical flat $1 stake. Missing pools are excluded from return estimates and counted separately. These are paper returns without verified fills.

## Frozen data and model

- Snapshot: 36560608 bytes; SHA256 98e5ba27d91d24ee242e63ba19b85722fe48c400f4d8a4336aa0b2db9021a3a7.
- Usable rows: 9324; exclusions: {"old_log_version": 3929, "missing_stale_or_invalid_features": 1254}.
- Model: raw; ridge 100; temperature 0.3431; fixed selected-call threshold 0.56.
- Model artifact: D:/Workspace/BC BTC/models/ud_direction_v1.json; SHA256 23b3a7b18960989c33245f9eaebf2bb4b5e63dbb2b282ffd2201dc42912be553.
- development: 6058 rows, 2026-10-30T16:29:51+00:00 through 2026-11-02T04:42:46+00:00 (UTC).
- calibration: 1397 rows, 2026-11-02T04:44:15+00:00 through 2026-11-10T23:00:38+00:00 (UTC).
- holdout: 1865 rows, 2026-11-10T23:02:06+00:00 through 2026-11-11T15:09:40+00:00 (UTC).

## Development comparison

| Feature family | Best CV log loss | Ridge |
|---|---:|---:|
| gap | 0.691213 | 100 |
| raw | 0.690461 | 10 |
| normalized | 0.691146 | 1000 |
| flow | 0.691427 | 1000 |

The one-standard-error rule chose the model before the final holdout was scored. Volatility normalization did not improve development performance in this experiment.

## Historical holdout

| Model / rule | Calls | Accuracy | Approx. Wilson 95% | Final-pool return per $1 | Hour-block return 95% |
|---|---:|---:|---:|---:|---:|
| candidate, every direction | 1865 | 49.49% | 47.22% to 51.76% | -4.19% | -7.82% to -0.43% |
| candidate, selected | 15 | 33.33% | 15.18% to 58.29% | -34.20% | -76.68% to 6.57% |
| momentum | 1865 | 48.20% | 45.94% to 50.47% | -6.55% | -11.52% to -1.79% |
| fade | 712 | 50.42% | 46.76% to 54.08% | -2.36% | -9.85% to 5.33% |
| overshoot | 35 | 37.14% | 23.17% to 53.66% | -29.83% | -57.03% to -1.20% |
| logged_prob | 270 | 50.37% | 44.44% to 56.29% | -2.43% | -16.42% to 11.81% |
| logged_gbm | 0 | - | - | - | - |
| logged_ens | 75 | 45.33% | 34.57% to 56.55% | -11.98% | -37.17% to 19.76% |

Candidate holdout log loss: 0.693969; Brier score: 0.250408. A constant 50% forecast has log loss 0.693147 and Brier score 0.25.

Selected-call coverage: 15/1865. Final-pool payout coverage: 15/15. Decision-pool return: -33.90%.

Wilson intervals assume independent calls and are only approximate. Hour-block bootstrap intervals retain within-hour dependence but are also uncertain with few independent dates. Baseline rules were researched previously and are descriptive comparisons. No threshold or feature was revised after inspecting this holdout.

## Confidence calibration on holdout

| Predicted direction confidence | Calls | Mean predicted | Observed correct |
|---|---:|---:|---:|
| 50.00%-52.00% | 1465 | 50.79% | 49.01% |
| 52.00%-54.00% | 315 | 52.75% | 51.43% |
| 54.00%-56.00% | 70 | 54.72% | 54.29% |
| 56.00%-58.00% | 12 | 56.89% | 33.33% |
| 58.00%-60.00% | 2 | 58.35% | 50.00% |
| 60.00%-65.00% | 1 | 61.27% | 0.00% |

## Prospective paper test

The exported model remains frozen. The training tool's --score-forward mode evaluates only decision snapshots captured at least 60 seconds after the model was frozen. It checks the recorded model-code hash before scoring. It does not place orders or modify the running bot.

A positive historical result remains unproven. The prospective evidence rule requires at least 3,000 selected calls across at least seven UTC dates, a Wilson 95% lower bound above 51.55%, and a positive lower bound for hourly-block return. Optional stopping, missing data, payout changes and execution still need review before treating an edge as established.

## Reproduce

```powershell
.venv\Scripts\python.exe -B tools\evaluate_ud_model.py --train --snapshot-bytes 36560608 --model models/ud_direction_replay.json --evaluation models/ud_direction_replay_evaluation.json --report docs/ud-model-replay.md
.venv\Scripts\python.exe -B tools\evaluate_ud_model.py --score-forward
```

Full development-fold scores, per-date returns and calibration statistics are in the JSON evaluation.
