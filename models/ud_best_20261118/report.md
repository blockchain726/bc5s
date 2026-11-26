# History-trained BC five-second model

Created 2026-11-18T11:52:44+00:00.

Models are selected by chronological development validation, before historical test labels are scored.
This history has been researched before. The test below is historical, not a prospective profitability claim.

## Data

- 20205 usable real rounds; 5681 with recorded books.
- 290 causal features: logged context, six individual books, two market mixtures and liquidity changes.
- One decision snapshot per real round; actual BC start/end settlement labels.
- Final pools are used only to estimate return, never as model features.
- The three extra books have too little earlier training history and are excluded from this experiment.

## Selected policies

| Policy | Chosen candidate | CV win rate | Test calls | Test coverage | Test win rate | Test paper ROI |
|---|---|---:|---:|---:|---:|---:|
| all | lgbm_book_240 | 52.74% | 1137 | 100.00% | 51.36% | -0.39% |
| selective | xgb_all_80 | 54.09% | 357 | 31.40% | 52.94% | 2.26% |

## Same-period baselines

| Baseline | Calls | Win rate | Paper ROI |
|---|---:|---:|---:|
| always_up | 1137 | 45.56% | -9.39% |
| ob_logit | 1137 | 53.39% | 3.34% |
| ob_lgbm | 1137 | 53.30% | 3.10% |
| ob_xgb | 1137 | 53.47% | 3.46% |
| ob_deeplob | 1134 | 52.47% | 1.28% |
| ob_hawkes | 1127 | 51.11% | -1.62% |
| ob_steady | 198 | 49.49% | -4.36% |
| gap_from_2x | 130 | 59.23% | 13.59% |
| gap_strict | 76 | 65.79% | 26.72% |

## Validation limits

The entire final test block was excluded from candidate selection, calibration and thresholds. Tree count and features were chosen in three expanding past-only folds with 60-second embargoes based on settlement time. At each fold, a later part of past data calibrated probabilities and selected the confidence threshold.
Calibration and thresholds can change prospective coverage. Counts are actual rounds, not overlapping synthetic samples.
Paper returns include a hypothetical $1 added to the chosen final pool and the 5% fee; fills are unverified.
Hourly-block intervals and daily breakdowns are in evaluation.json. No 60% claim is made unless supported by future outcomes.

## Artifacts and use

evaluated.json contains the exact models scored on the test. model.json refits the chosen settings on the frozen history preceding the final calibration block, then calibrates on that final block. Its prospective record starts 60 seconds after export; historical test scores are for evaluated.json.

Load with bcbot.ud_best.BestUDModel.load. predict(snapshot, books, decision_pc, policy='selective') returns a direction, probability and selected flag. policy='all' supplies the every-round model.

Prospective scoring:

~~~powershell
logs\research-venv\Scripts\python.exe -B tools\train_ud_best.py --score-forward --out models\ud_best_20261118
~~~

This experiment does not change the running bot or place orders.
