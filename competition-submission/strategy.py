"""Small causal models for the published SDSU trading harness.

Training learns from next-open to following-open returns. Prediction reads only
the supplied history, identifies no ticker, and never fits or changes state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


OHLCV = ("Open", "High", "Low", "Close", "Volume")
FEATURE_NAMES = (
    "return_1", "return_5", "return_20", "return_60", "return_120",
    "trend_20", "trend_100", "volatility_20", "volatility_60",
    "daily_range", "relative_volume_20", "opening_gap", "intraday_return",
    "close_location",
)
MIN_HISTORY = 121


@dataclass(frozen=True)
class StrategyConfig:
    """Fixed choices; select them using chronological training-period folds."""
    target_vol: float = 0.12
    max_long: float = 0.80
    max_short: float = 0.25
    ml_weight: float = 0.30
    ml_strength: float = 3.0
    trend_bias: float = 0.30
    trend_strength: float = 0.90
    exposure_step: float = 0.05
    deadband: float = 0.05
    ridge_alpha: float = 100.0
    n_estimators: int = 40
    max_depth: int = 4
    min_samples_leaf: int = 80
    random_state: int = 42


def _config(value=None):
    if isinstance(value, StrategyConfig):
        result = value
    else:
        supplied = dict(value or {})
        unknown = set(supplied) - set(StrategyConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown strategy config: {sorted(unknown)}")
        result = StrategyConfig(**supplied)
    for name, number in asdict(result).items():
        if not isinstance(number, (float, int)) or not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
    if not (0 < result.target_vol <= 1 and 0 <= result.max_long <= 1 and 0 <= result.max_short <= 1):
        raise ValueError("Volatility target must be positive; long/short caps must be in [0, 1]")
    if not (0 <= result.ml_weight <= 1 and result.ml_strength >= 0):
        raise ValueError("ML weight must be in [0, 1] and its strength nonnegative")
    if not (0 <= result.exposure_step <= .5 and 0 <= result.deadband < 1):
        raise ValueError("Invalid exposure step or deadband")
    if result.ridge_alpha <= 0:
        raise ValueError("Ridge regularization must be positive")
    for name in ("n_estimators", "max_depth", "min_samples_leaf", "random_state"):
        if not isinstance(getattr(result, name), int):
            raise ValueError(f"{name} must be an integer")
    if not (1 <= result.n_estimators <= 100 and 1 <= result.max_depth <= 8 and result.min_samples_leaf >= 10):
        raise ValueError("Use a small forest with positive depth and at least 10 rows per leaf")
    return result


def _rolling(values, window, statistic="mean"):
    """Each result uses its own exact window, independent of earlier rows."""
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1:] = windows.std(axis=1, ddof=1) if statistic == "std" else windows.mean(axis=1)
    return out


def make_features(frame):
    """Causal batch features for fitting; the final row matches tail(252).

    No price levels, ticker identifiers, dates, or future values are features.
    The maximum lag is 120; no expanding statistics or recursive averages occur.
    """
    close = frame["Close"].to_numpy(dtype=float)
    opening = frame["Open"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    volume = frame["Volume"].to_numpy(dtype=float)
    n = len(frame)
    columns = []
    for lag in (1, 5, 20, 60, 120):
        values = np.full(n, np.nan)
        if n > lag:
            values[lag:] = close[lag:] / close[:-lag] - 1
        columns.append(values)
    r1 = columns[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns.extend((close / _rolling(close, 20) - 1,
                        close / _rolling(close, 100) - 1,
                        _rolling(r1, 20, "std"), _rolling(r1, 60, "std"),
                        (high - low) / close))
        average_volume = _rolling(volume, 20)
        relative_volume = np.divide(volume, average_volume, out=np.ones(n), where=average_volume > 0) - 1
        gap = np.full(n, np.nan)
        if n > 1:
            gap[1:] = opening[1:] / close[:-1] - 1
        location = np.divide(close - low, high - low, out=np.full(n, .5), where=high > low)
        columns.extend((relative_volume, gap, close / opening - 1, location))
    return pd.DataFrame(np.column_stack(columns), index=frame.index, columns=FEATURE_NAMES)


def _latest_features(history):
    """Fast last-row equivalent of make_features; at most 121 rows are read."""
    values = history.iloc[-MIN_HISTORY:].loc[:, list(OHLCV)].to_numpy(dtype=float)
    if values.shape[0] < MIN_HISTORY or not np.isfinite(values).all():
        return None
    opening, high, low, close, volume = values.T
    if np.any(values[:, :4] <= 0) or np.any(volume < 0):
        return None
    returns = close[1:] / close[:-1] - 1
    volume_mean = volume[-20:].mean()
    width = high[-1] - low[-1]
    x = np.array([
        *[close[-1] / close[-1 - lag] - 1 for lag in (1, 5, 20, 60, 120)],
        close[-1] / close[-20:].mean() - 1,
        close[-1] / close[-100:].mean() - 1,
        returns[-20:].std(ddof=1), returns[-60:].std(ddof=1),
        width / close[-1], volume[-1] / volume_mean - 1 if volume_mean > 0 else 0,
        opening[-1] / close[-2] - 1, close[-1] / opening[-1] - 1,
        (close[-1] - low[-1]) / width if width > 0 else .5,
    ])
    return x if np.isfinite(x).all() else None


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FrozenStrategy:
    kind: str
    config: StrategyConfig
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    trees: tuple
    training_rows: int
    training_start: str | None
    training_end: str | None
    training_label_end: str | None
    stocks_used: int

    def predict(self, history):
        """Return a deterministic Python float exposure using prior data only."""
        if self.kind in ("starter_trend", "defensive_trend"):
            if len(history) < 100:
                return 0.0
            close = history["Close"].iloc[-100:].to_numpy(dtype=float)
            if not np.isfinite(close).all() or np.any(close <= 0):
                return 0.0
            fast, slow = float(close[-20:].mean()), float(close.mean())
            bullish = fast > slow
            if self.kind == "starter_trend":
                # Exactly the provided starter rule: full long above, cash below.
                return 1.0 if bullish else 0.0
            if not bullish or close[-1] <= slow:
                return 0.0
            daily = close[-20:] / close[-21:-1] - 1
            annual_vol = float(daily.std(ddof=1) * np.sqrt(252))
            tier = .75 if annual_vol < .35 else .50
            # Fixed quarter-size tiers avoid small daily risk-target adjustments.
            return float(np.floor(min(tier, self.config.max_long) / .25) * .25)
        if len(history) < MIN_HISTORY:
            return 0.0
        x = _latest_features(history)
        if x is None:
            return 0.0
        config = self.config
        sigma20, sigma60 = max(x[7], .003), max(x[8], .003)
        # Slow trend changes less often than a daily up/down prediction.
        trend_z = (.50 * x[6] / (sigma60 * np.sqrt(100))
                   + .35 * x[3] / (sigma60 * np.sqrt(60))
                   + .15 * x[5] / (sigma20 * np.sqrt(20)))
        trend = float(np.clip(config.trend_bias + config.trend_strength * np.tanh(1.2 * trend_z), -1, 1))
        score = trend
        if self.kind != "trend":
            z = np.clip((x - self.means) / self.scales, -8, 8)
            if self.kind == "ridge":
                predicted = float(self.intercept + z @ self.coefficients)
            else:
                predictions = []
                for feature, threshold, left, right, leaf_value in self.trees:
                    node = 0
                    while left[node] >= 0:
                        node = left[node] if z[feature[node]] <= threshold[node] else right[node]
                    predictions.append(leaf_value[node])
                predicted = float(np.mean(predictions))
            ml_score = float(np.tanh(config.ml_strength * predicted))
            score = (1 - config.ml_weight) * trend + config.ml_weight * ml_score
        annual_vol = max(sigma20 * np.sqrt(252), .06)
        risk_scale = min(1.0, config.target_vol / annual_vol)
        position = float(np.clip(score * risk_scale, -config.max_short, config.max_long))
        if abs(position) < config.deadband:
            return 0.0
        if config.exposure_step:
            position = float(np.round(position / config.exposure_step) * config.exposure_step)
        return float(np.clip(position, -config.max_short, config.max_long))

    def describe(self):
        rule = self.kind in ("trend", "starter_trend", "defensive_trend")
        simple_rule = self.kind in ("starter_trend", "defensive_trend")
        return {
            "kind": self.kind,
            "features": list(FEATURE_NAMES),
            "target": "No fitted prediction target; fixed trading rule" if rule else "Open[t+2] / Open[t+1] - 1, divided by trailing 20-day close-return volatility",
            "timing": "history through close t; position executed at open t+1; marked at open t+2",
            "minimum_history": 100 if simple_rule else MIN_HISTORY,
            "learned_parameters": not rule,
            "training_rows": self.training_rows,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "training_label_end": self.training_label_end,
            "stocks_used": self.stocks_used,
            "config": asdict(self.config),
            "standardized_coefficients": dict(zip(FEATURE_NAMES, self.coefficients.tolist())) if self.kind == "ridge" else {},
            "explanation": ("Exact starter rule: 20-day average above 100-day average means 100% long; otherwise cash. No fitted parameters."
                            if self.kind == "starter_trend" else
                            "Fixed defensive rule: 20-day average and latest close above the 100-day average; 75% long below 35% annual volatility, 50% otherwise, subject to the configured cap and quarter-size tiers; cash when the trend fails. No fitted parameters."
                            if self.kind == "defensive_trend" else
                            "Fixed smooth trend with volatility sizing and exposure bins; no fitted parameters."
                            if self.kind == "trend" else
                            "Slow price trend blended with a small fitted return model, trailing-volatility sizing, asymmetric long/short caps, and exposure bins. No state or fitting in predict."),
        }


def fit_strategy(training_frames, kind="ridge", config=None):
    """Fit using ONLY the caller-supplied training slice; never fetch data.

    Each stock is processed independently. Both future opens of every training
    target must exist inside that stock's supplied slice. Normalization means
    and scales are learned from these eligible training examples only.
    """
    config = _config(config)
    if kind == "extra":
        kind = "extra_trees"
    rule = kind in ("trend", "starter_trend", "defensive_trend")
    if kind not in ("trend", "ridge", "extra_trees", "starter_trend", "defensive_trend"):
        raise ValueError("kind must be trend, starter_trend, defensive_trend, ridge, or extra_trees")
    frames = list(training_frames.values()) if isinstance(training_frames, Mapping) else list(training_frames)
    xs, ys, starts, ends, label_ends = [], [], [], [], []
    for frame in frames:
        if len(frame) < MIN_HISTORY + 2:
            continue
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Training histories must have ascending unique dates")
        raw = frame.loc[:, list(OHLCV)].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or np.any(raw[:, :4] <= 0) or np.any(raw[:, 4] < 0):
            raise ValueError("Training OHLCV values must be finite with positive prices and nonnegative volume")
        features = make_features(frame)
        opening = frame["Open"].to_numpy(dtype=float)
        target = np.full(len(frame), np.nan)
        target[:-2] = opening[2:] / opening[1:-1] - 1
        values = features.to_numpy()
        usable = np.isfinite(values).all(axis=1) & np.isfinite(target)
        indices = np.flatnonzero(usable)
        if not len(indices):
            continue
        xs.append(values[usable])
        # Vol normalization pools unnamed stocks without rewarding high-vol ones.
        ys.append(np.clip(target[usable] / np.maximum(values[usable, 7], .003), -5, 5))
        starts.append(str(frame.index[indices[0]].date()))
        ends.append(str(frame.index[indices[-1]].date()))
        label_ends.append(str(frame.index[indices[-1] + 2].date()))
    if not xs:
        if not rule:
            raise ValueError("Need at least 123 bars per training stock for causal features and labels")
        x = np.empty((0, len(FEATURE_NAMES)))
        y = np.empty(0)
    else:
        x, y = np.concatenate(xs), np.concatenate(ys)
    if not rule and len(x) < 120:
        raise ValueError("Need at least 120 eligible training examples")
    means = x.mean(axis=0) if len(x) else np.zeros(len(FEATURE_NAMES))
    scales = x.std(axis=0) if len(x) else np.ones(len(FEATURE_NAMES))
    scales = np.where(scales > 1e-12, scales, 1.0)
    z = np.clip((x - means) / scales, -8, 8)
    coefficients = np.zeros(len(FEATURE_NAMES))
    intercept = 0.0
    trees = []
    if kind == "ridge":
        intercept = float(y.mean())
        centered = z - z.mean(axis=0)
        coefficients = np.linalg.solve(centered.T @ centered + config.ridge_alpha * np.eye(z.shape[1]), centered.T @ (y - intercept))
        # Clipping standardized inputs means their mean need not be exactly zero.
        intercept = float(y.mean() - z.mean(axis=0) @ coefficients)
    elif kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=config.n_estimators, max_depth=config.max_depth,
                                    min_samples_leaf=config.min_samples_leaf, max_features=.75,
                                    random_state=config.random_state, n_jobs=1)
        model.fit(z, y)
        for estimator in model.estimators_:
            t = estimator.tree_
            arrays = (t.feature.copy(), t.threshold.copy(), t.children_left.copy(), t.children_right.copy(), t.value[:, 0, 0].copy())
            for a in arrays:
                a.flags.writeable = False
            trees.append(arrays)
    return FrozenStrategy(kind, config, _readonly(means), _readonly(scales), _readonly(coefficients),
                          intercept, tuple(trees), len(x), min(starts) if starts else None,
                          max(ends) if ends else None, max(label_ends) if label_ends else None, len(xs))
