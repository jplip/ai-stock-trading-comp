#!/usr/bin/env python3
"""One-stock, chronological ML experiment. Requires NumPy; makes no network calls."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "distance_from_mean_5", "distance_from_mean_20", "volatility_20",
    "distance_from_high_20", "intraday_return", "volume_vs_mean_20",
]
EPS = 1e-8


def read_prices(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        normalized = [name.strip().lower() for name in headers]
        if any(not name for name in normalized):
            raise ValueError("CSV headers must not be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("CSV has duplicate headers (case insensitive).")
        names = dict(zip(normalized, headers))
        if not {"date", "open", "close"}.issubset(names):
            raise ValueError("CSV requires Date, Open and Close headers (case insensitive).")
        rows, previous, expected_symbol = [], None, None
        for line, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV line {line}: extra fields; expected {len(headers)} fields.")
            if any(value is None for value in raw.values()):
                raise ValueError(f"CSV line {line}: missing fields; expected {len(headers)} fields.")
            if not any(str(value or "").strip() for value in raw.values()):
                continue
            stamp = (raw.get(names["date"]) or "").strip()
            try:
                parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                prices = {key: float(raw[names[key]]) for key in ("open", "close")}
                if any(not math.isfinite(value) or value <= 0 for value in prices.values()):
                    raise ValueError("prices must be positive and finite")
                volume_text = (raw.get(names.get("volume", "")) or "").strip()
                volume = float(volume_text) if volume_text else 0.0
                if not math.isfinite(volume) or volume < 0:
                    raise ValueError("volume must be finite and nonnegative")
                if previous is not None and parsed <= previous:
                    raise ValueError("dates must be unique and strictly increasing")
                symbols = [raw[names[key]].strip().upper() for key in ("symbol", "ticker") if key in names]
                if symbols:
                    if any(not symbol for symbol in symbols):
                        raise ValueError("Symbol/Ticker values must not be empty when those columns are present")
                    if len(set(symbols)) > 1:
                        raise ValueError("contradictory Symbol and Ticker values in the same row")
                    if expected_symbol is not None and symbols[0] != expected_symbol:
                        raise ValueError("multiple stocks found in Symbol/Ticker; use one stock per CSV")
                    expected_symbol = symbols[0]
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
            rows.append({"date": stamp, **prices, "volume": volume})
            previous = parsed
    if not 180 <= len(rows) <= 5000:
        raise ValueError("Use 180 to 5,000 chronological bars for one stock.")
    return rows, "volume" in names


def make_features(rows, use_volume):
    close = np.array([row["close"] for row in rows], dtype=float)
    opens = np.array([row["open"] for row in rows], dtype=float)
    volumes = np.array([row["volume"] for row in rows], dtype=float)
    features = []
    for i in range(20, len(rows)):
        window = close[i - 19:i + 1]
        changes = np.diff(np.log(close[i - 20:i + 1]))
        average_volume = volumes[i - 19:i + 1].mean()
        features.append([
            *[close[i] / close[i - lag] - 1 for lag in (1, 3, 5, 10, 20)],
            close[i] / close[i - 4:i + 1].mean() - 1,
            close[i] / window.mean() - 1,
            changes.std(), close[i] / window.max() - 1,
            close[i] / opens[i] - 1,
            volumes[i] / average_volume - 1 if use_volume and average_volume > 0 else 0,
        ])
    x = np.asarray(features, dtype=float)
    # Signal after close i; enter at open i+1; mark/exit at open i+2.
    indices = np.arange(20, len(rows) - 2)
    returns = opens[indices + 2] / opens[indices + 1] - 1
    return x[:-2], (returns > 0).astype(float), returns, indices, x[-1:]


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -35, 35)))


class Logistic:
    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.intercept = float(np.log((y.mean() + .01) / (1.01 - y.mean())))
        for _ in range(600):
            error = sigmoid(x @ self.weights + self.intercept) - y
            self.weights -= .07 * (x.T @ error / len(x) + .035 * self.weights)
            self.intercept -= .07 * error.mean()
        return self

    def predict(self, x):
        return sigmoid(x @ self.weights + self.intercept)


class SmallForest:
    """Bootstrap trees with random feature subsets and shallow Gini splits."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def fit(self, x, y):
        self.trees = []
        for _ in range(32):
            sample = self.rng.integers(0, len(y), len(y))
            self.trees.append(self._grow(x[sample], y[sample], 0))
        return self

    def _grow(self, x, y, depth):
        probability = float((y.sum() + 2) / (len(y) + 4))
        if depth == 3 or len(y) < 24 or y.min() == y.max():
            return probability
        best = None
        for feature in self.rng.choice(x.shape[1], max(1, int(math.sqrt(x.shape[1]))), replace=False):
            for threshold in np.unique(np.quantile(x[:, feature], [.15, .3, .5, .7, .85])):
                left = x[:, feature] <= threshold
                nl = int(left.sum())
                if min(nl, len(y) - nl) < 8:
                    continue
                pl, pr = y[left].mean(), y[~left].mean()
                impurity = nl * pl * (1 - pl) + (len(y) - nl) * pr * (1 - pr)
                if best is None or impurity < best[0]:
                    best = (impurity, int(feature), float(threshold), left)
        if best is None:
            return probability
        _, feature, threshold, left = best
        return (feature, threshold, self._grow(x[left], y[left], depth + 1),
                self._grow(x[~left], y[~left], depth + 1))

    def _predict_tree(self, tree, x):
        if isinstance(tree, float):
            return np.full(len(x), tree)
        feature, threshold, left_tree, right_tree = tree
        left = x[:, feature] <= threshold
        result = np.empty(len(x))
        if left.any():
            result[left] = self._predict_tree(left_tree, x[left])
        if (~left).any():
            result[~left] = self._predict_tree(right_tree, x[~left])
        return result

    def predict(self, x):
        return np.mean([self._predict_tree(tree, x) for tree in self.trees], axis=0)


class Neighbors:
    def fit(self, x, y):
        self.x, self.y = x, y
        self.k = min(len(y), 31)
        return self

    def predict(self, x):
        results = []
        for start in range(0, len(x), 64):
            batch = x[start:start + 64]
            distance = ((batch[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argpartition(distance, self.k - 1, axis=1)[:, :self.k]
            results.extend((self.y[nearest].sum(axis=1) + 2) / (self.k + 4))
        return np.asarray(results)


class Ensemble:
    def fit(self, x, y, seed=42):
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < EPS] = 1
        normalized = self.transform(x)
        self.models = [Logistic().fit(normalized, y), SmallForest(seed).fit(normalized, y),
                       Neighbors().fit(normalized, y)]
        return self

    def transform(self, x):
        return np.clip((x - self.mean) / self.scale, -8, 8)

    def predict(self, x):
        normalized = self.transform(x)
        return np.column_stack([model.predict(normalized) for model in self.models])


def log_loss(y, probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def allocation(probability):
    # A fixed policy, not optimized on the validation or test trading returns.
    return np.clip((probability - .5) / .15, 0, 1)


def simulate(weights, returns, fee, annual_periods):
    equity, previous, changes, turnover = [1.0], 0.0, [], 0.0
    for weight, change in zip(weights, returns):
        traded = abs(float(weight) - previous)
        growth = (1 - fee * traded) * (1 + float(weight) * float(change))
        equity.append(equity[-1] * growth)
        changes.append(growth - 1)
        turnover += traded
        # Stock/cash proportions drift before the next rebalance.
        previous = float(weight) * (1 + float(change)) / (1 + float(weight) * float(change))
    # Liquidation occurs at the last evaluated open and is included in metrics.
    final_fee = fee * previous
    equity[-1] *= 1 - final_fee
    changes[-1] = (1 + changes[-1]) * (1 - final_fee) - 1
    turnover += previous
    curve = np.asarray(equity)
    changes = np.asarray(changes)
    volatility = float(changes.std())
    return {
        "total_return": float(curve[-1] - 1),
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
        "annualized_sharpe_zero_risk_free": float(changes.mean() / volatility * math.sqrt(annual_periods)) if volatility > EPS else None,
        "total_turnover": float(turnover),
        "average_exposure": float(np.mean(weights)),
        "equity_curve": [float(value) for value in curve],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="One stock, 180–5,000 rows, Date/Open/Close and optional Volume")
    parser.add_argument("--output", required=True, help="Directory for generated reports and predictions")
    parser.add_argument("--cost-bps", type=float, default=10, help="One-way cost per full portfolio turnover (default 10 bps)")
    parser.add_argument("--periods-per-year", type=float, default=252, help="252 assumes daily trading bars")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not math.isfinite(args.cost_bps) or not 0 <= args.cost_bps <= 1000:
        parser.error("--cost-bps must be between 0 and 1000")
    if not math.isfinite(args.periods_per_year) or args.periods_per_year <= 0:
        parser.error("--periods-per-year must be positive")
    try:
        rows, use_volume = read_prices(args.csv)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x, y, returns, indices, latest_x = make_features(rows, use_volume)
        if not all(np.isfinite(value).all() for value in (x, returns, latest_x)):
            raise ValueError("Price data creates non-finite features or returns.")
    except (OSError, ValueError, FloatingPointError) as error:
        parser.error(str(error))
    n = len(y)
    split_a, split_b = int(n * .6), int(n * .8)
    training = np.arange(split_a - 2)
    validation = np.arange(split_a, split_b - 2)
    testing = np.arange(split_b, n)
    initial = Ensemble().fit(x[training], y[training], args.seed)
    validation_p = initial.predict(x[validation])
    losses = np.array([log_loss(y[validation], validation_p[:, i]) for i in range(3)])
    blend_weights = (1 / np.maximum(losses, .05))
    blend_weights /= blend_weights.sum()
    # Refit only on pre-test observations; hold fitted models fixed during test.
    refit = np.arange(split_b - 2)
    evaluated = Ensemble().fit(x[refit], y[refit], args.seed)
    test_components = evaluated.predict(x[testing])
    probability = test_components @ blend_weights
    weights = allocation(probability)
    fee = args.cost_bps / 10000
    strategy = simulate(weights, returns[testing], fee, args.periods_per_year)
    benchmark = simulate(np.ones(len(testing)), returns[testing], fee, args.periods_per_year)
    # The latest signal may use every label matured by the latest observed close.
    latest_model = Ensemble().fit(x, y, args.seed)
    latest_components = latest_model.predict(latest_x)[0]
    latest_probability = float(latest_components @ blend_weights)
    latest_weight = float(allocation(latest_probability))
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_date", "entry_open_date", "exit_open_date", "probability_up_score",
                         "target_weight", "realized_open_to_open_return", "actual_up",
                         "logistic_score", "forest_score", "knn_score"])
        for j, sample in enumerate(testing):
            i = indices[sample]
            writer.writerow([rows[i]["date"], rows[i + 1]["date"], rows[i + 2]["date"],
                             float(probability[j]), float(weights[j]), float(returns[sample]),
                             int(y[sample]), *[float(v) for v in test_components[j]]])
    names = ["logistic", "shallow_random_forest", "knn"]
    latest = {
        "signal_observed_through": rows[-1]["date"],
        "execution_assumption": "After the final observed close, rebalance at the next bar's open; hold until the following open.",
        "probability_up_score": latest_probability,
        "calibrated_probability": False,
        "target_weight": latest_weight,
        "action": "cash" if latest_weight == 0 else "long",
        "component_scores": dict(zip(names, [float(v) for v in latest_components])),
        "official_submission_format": False,
        "scope": "Educational generic one-stock signal; no orders placed.",
    }
    report = {
        "input_file": str(Path(args.csv).resolve()),
        "input_bars": len(rows), "labeled_samples": n,
        "feature_names": FEATURE_NAMES,
        "volume_feature_enabled": use_volume,
        "target": "Open[i+2] > Open[i+1], predicted after Close[i]",
        "splits": {"initial_training_samples": len(training), "validation_samples": len(validation),
                   "test_samples": len(testing), "pre_test_refit_samples": len(refit),
                   "purged_samples_per_boundary": 2,
                   "first_test_signal_date": rows[indices[testing[0]]]["date"],
                   "last_test_exit_date": rows[indices[testing[-1]] + 2]["date"]},
        "ensemble_weights_from_validation_only": dict(zip(names, [float(v) for v in blend_weights])),
        "validation_component_log_loss": dict(zip(names, [float(v) for v in losses])),
        "test_accuracy": float(np.mean((probability >= .5) == y[testing])),
        "test_log_loss": log_loss(y[testing], probability),
        "test_up_fraction": float(y[testing].mean()),
        "training_majority_class_accuracy_baseline": float(np.mean(y[testing] == (y[refit].mean() >= .5))),
        "cost_bps_one_way": args.cost_bps,
        "annualization_periods": args.periods_per_year,
        "test_strategy": strategy, "test_buy_and_hold": benchmark,
        "policy": "Long/cash only: target weight = clip((score - 0.50) / 0.15, 0, 1). No leverage or shorting.",
        "limitations": ["Custom educational models; scores are not calibrated probabilities.",
                        "One chronological split, one asset, and a short test window cannot establish predictive value.",
                        "Execution at next open is an assumption, not guaranteed fills.",
                        "Use consistently split-adjusted OHLC data; costs are a fixed approximation.",
                        "Latest model is refitted after evaluation and its signal is not a held-out result.",
                        "Competition rules, horizon, ranking and submission adapter remain unknown."],
    }
    for name, value in [("report.json", report), ("latest_signal.json", latest)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Test bars: {len(testing)} | strategy: {strategy['total_return']:+.2%} | buy & hold: {benchmark['total_return']:+.2%}")
    print(f"Latest score: {latest_probability:.3f} | next-open target stock weight: {latest_weight:.1%}")
    print(f"Reports: {out.resolve()}")


if __name__ == "__main__":
    main()
