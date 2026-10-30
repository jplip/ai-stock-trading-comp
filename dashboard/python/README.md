# Python companion

Start with the browser app in the parent folder. This optional script is for a teammate who knows Python or for adapting the model once the club releases its rules. It creates generic signals, not a ready-to-submit competition file.

## Run

In Terminal, move into this `python` folder, then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py /path/to/prices.csv --output /path/to/results
```

No data is downloaded and no API key is needed. The only third-party dependency is NumPy. Run this with `--help` to see options.

## Data

Use **one stock**, 180–5,000 rows, oldest first, with unique ISO dates. Required columns are `Date`, `Open`, and `Close`; `Volume` is optional. If `Symbol` or `Ticker` columns are present, every row must identify the same stock and both columns must agree. Other columns are ignored. Duplicate headers and extra/missing fields are rejected. Do not mix an adjusted Close with unadjusted Open: all prices must use the same split-adjustment basis.

```csv
Date,Open,Close,Volume
2025-01-02,100.00,101.20,1500000
2025-01-03,101.40,100.80,1800000
2025-01-06,100.60,102.10,1700000
```

Those three rows illustrate the format; a working input needs at least 180 rows. Every row represents one trading bar. Defaults assume daily bars and 252 periods per year. Change `--periods-per-year` for another frequency.

## What it does

1. Calculates 11 features from price/volume observations available by each bar's close.
2. Predicts whether **the next open to the following open** return is positive. This deliberately leaves time to act after the signal.
3. Reserves chronological 60% / 20% / 20% portions for initial fitting, validation, and testing. Two labeled observations before each boundary are purged.
4. Fits custom regularized logistic regression, a shallow bootstrap random forest, and nearest neighbors. Standardization uses only the fitting data. Validation log loss determines the blend weights.
5. Refits on pre-test data and measures the frozen ensemble on the final 20%. A separate refit using all matured labels produces the latest generic signal.
6. Converts the score into a stock weight: `clip((score - 0.50) / 0.15, 0, 1)`. This is a fixed long/cash rule; it is not tuned against test performance.

A score of 0.575 produces a 50% stock / 50% cash target. These scores **are not calibrated probabilities**. The latest signal has not been evaluated on unseen future prices.

## Outputs

| File | Purpose |
|---|---|
| `predictions.csv` | Held-out signal dates, assumed trade dates, component scores, target weights, and realized returns. |
| `report.json` | Split counts, validation weights, classification metrics, after-cost equity curves, and comparison with buy and hold. |
| `latest_signal.json` | Most recent generic next-open target weight; an adapter will be needed for the event simulator. |

Both strategies pay a default 10-basis-point one-way fee per full portfolio turnover, including entry, weight changes, and final liquidation. For example, moving from 0% to 50% stock costs 5 basis points of portfolio value. Exposure drifts with price changes before each subsequent rebalance. The simulation approximates cost from the requested change in exposure, pays it first, rebalances remaining equity to the target, and earns the next open-to-open return. It omits taxes, variable spreads, market impact, and interest on cash.

## Explain it in 30 seconds

The model reads recent momentum, volatility, intraday moves, and optional volume. Three different methods make a score, and their validation results determine how much each contributes. A higher combined score increases stock exposure, with the rest held as cash. The final historical segment is reserved for testing, with realistic timing and explicit transaction costs. Results can be weak or negative; inspect the comparison with buy and hold and the worst drawdown.

The Python and browser versions are separate educational implementations. Their numerical predictions may differ. Do not compare synthetic demo returns as evidence of real trading ability, and do not change settings repeatedly after inspecting the held-out results while still calling them an untouched test.

## Concepts behind the safeguards

- [scikit-learn: time-series splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: leakage and preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [scikit-learn: ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)

These links explain the ideas. This script implements its models directly with NumPy and does not call scikit-learn.

## Verify the companion

Run `python test_run.py` in this folder. The retained tests cover data validation, future-data invariance, drift-aware transaction costs, final liquidation, and the complete pipeline on a minimum-sized single-class dataset. Fixtures and generated reports stay in a temporary directory.
