# Independent model and data assessment

## Verdict

The official submission remains the selected `starter_trend` model. The data is
clean and complete, the causal/serialization contract passes, and no candidate
tested here provides a sufficiently reliable improvement to justify replacing
the preserved submission pickle.

This is an evidence-based non-improvement result, not a claim that the strategy
has a durable market edge. In the 2023 development check, the strategy returned
0.9% with Sharpe 0.139 versus buy-and-hold's 16.1% and Sharpe 1.190. The 2024
diagnostic was better (8.2% and Sharpe 0.967 on the fixed eight-stock panel),
but that is still too little evidence for deployment confidence.

## Initial assessment

- Official CSV: 85,544 rows, 34 tickers, 2015-01-02 through 2024-12-31.
- Every ticker has 2,516 rows.
- Null cells, duplicate rows, duplicate ticker/date keys, invalid prices,
  negative volume, and OHLC ordering violations: zero.
- The supplied harness, causal-prefix tests, deterministic bounds checks, and
  fresh-process cloudpickle check all pass (`8/8` tests).
- The official report correctly identifies that 2023 and 2024 are development
  diagnostics, not untouched tests after being inspected.

## Data improvements

No source rows were edited, fabricated, imputed, or removed. Because the audit
found no supported correction, preserving the official bytes is the least
biased data-quality decision. The audit and candidate tables are recoverable in
`research_artifacts/`; the original CSV and official pickle remain unchanged.

## Training and evaluation changes

The existing workflow already uses chronological folds, trains only through the
fold cutoff, and keeps future labels out of feature construction. I evaluated
three additional low-complexity causal rules: 10/80, 30/120, and a majority
ensemble of 10/80, 20/100, and 30/120. Selection was scored only on 2019–2022;
2023 and 2024 were reported diagnostically.

The 30/120 rule scored higher on the predeclared validation folds (selection
score about 1.110 versus 1.036 for 20/100, using sample standard deviation), but it underperformed the official
rule in 2023. The ensemble was more balanced across later years but did not
beat the official predeclared selection score. This is a classic small-sample
model-selection risk, so it was not promoted.

## Saved artifacts

- Official model: `Signal_Lab_submission.pkl` (preserved; not overwritten).
- Official reports and validation tables: existing `report.json`,
  `validation_results.csv`, `model_selection.csv`.
- New audit: `research_artifacts/research_audit.json`.
- New per-year comparisons: `research_artifacts/candidate_metrics.csv`.
- New candidate summary: `research_artifacts/candidate_summary.csv`.
- Reproduction script: `research_audit.py`.

## Remaining weaknesses and recommendations

The primary limitation is not data cleanliness but weak evidence of an edge:
the system is a small set of technical trend rules, validation years are
correlated, the stock universe is fixed, and the harness omits slippage,
borrowing costs, interest, and dividends. The next defensible improvement would
be a truly unseen evaluation on organizer secret rounds, followed by a
pre-registered walk-forward study over more assets and realistic costs. Do not
use the inspected 2023/2024 outcomes to claim an untouched final test.
