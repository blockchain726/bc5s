# Gap Pressure: a new BC five-second model

This model learns hierarchical conditional probability tables. It does not use XGBoost or LightGBM.

## Decision

1. Estimate the chance that BC's price gap reverses, conditional on its size.
2. Update that chance using the direction and change of liquidity at the best bid and ask.
3. Where selected, update it again from the price's response to that pressure.
4. Apply probability calibration and the confidence threshold learned on an earlier calibration block.

A gap can be overridden: strong opposing pressure can make the learned probability favour continuation. The model can abstain, and missing book data never becomes an invented reading.

## Learned settings

- {"recipe": "queue", "strength": 150.0, "depth": 3, "half_life_h": 72.0}
- Chronological CV: 362/637 selected calls, 56.83% correct.
- Target selection share: 20.00%; future realized frequency can differ.

## Later historical test

Period: 2026-11-18T11:51:51+00:00 through 2026-11-18T18:12:36+00:00.
These rounds were appended after the frozen training snapshot, and did not choose settings or thresholds. The broader history has been examined previously, so this is a historical test, not prospective proof.

| Policy | Wins / calls | Win rate | Approximate 95% range | Paper ROI |
|---|---:|---:|---:|---:|
| selected | 113/221 | 51.13% | 44.58% to 57.65% | -1.29% |
| all probabilities | 365/695 | 52.52% | 48.80% to 56.21% | 2.14% |
| gap-prior ablation | 45/79 | 56.96% | 45.97% to 67.31% | 10.76% |

Selected coverage: 31.80%. Counts are actual BC rounds, not overlapping synthetic windows.
The gap-prior ablation removes both order-book updates; it has its own calibration and selection threshold from the same past calibration observations. It is diagnostic, not a replacement chosen after this test.
Hourly-block intervals, all CV settings, frozen-data hashes, and per-day counts are saved in JSON.

## Inference

Load bcbot.gap_pressure.GapPressureModel.load(model.json), then call predict(decision_snapshot, six_market_book_rows, decision_pc). The result includes P(UP), side, selected and confidence.
The exported model is exactly the one evaluated above. It has not been refitted on the later test.

~~~powershell
logs\research-venv\Scripts\python.exe -B tools\train_gap_pressure.py --score-forward
~~~

No live bot settings or orders are changed. A 60% future win rate is not established.
