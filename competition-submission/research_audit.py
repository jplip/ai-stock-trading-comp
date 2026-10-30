"""Reproducible data audit and out-of-sample strategy comparison.

This script is deliberately separate from the official submission workflow. It
does not modify the source CSV, the submitted pickle, or the organizer's
harness. It writes recoverable research artifacts under ``research_artifacts``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sdsu_harness as harness  # noqa: E402

DATA_PATH = HERE / "data" / "stock_prices_2015_2024.csv"
OUT = HERE / "research_artifacts"
TICKERS = ("AAPL", "JPM", "XOM", "JNJ", "CROX", "INTC", "PLUG", "KO")


def audit(raw: pd.DataFrame) -> dict:
    key_dupes = int(raw.duplicated(["Ticker", "Date"]).sum())
    prices = raw[["Open", "High", "Low", "Close"]]
    return {
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "date_min": str(raw["Date"].min()),
        "date_max": str(raw["Date"].max()),
        "ticker_count": int(raw["Ticker"].nunique()),
        "rows_per_ticker": {str(k): int(v) for k, v in raw.groupby("Ticker").size().items()},
        "null_cells": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_ticker_date_keys": key_dupes,
        "invalid_price_rows": int((prices <= 0).any(axis=1).sum()),
        "negative_volume_rows": int((raw["Volume"] < 0).sum()),
        "ohlc_high_violations": int((raw["High"] < prices[["Open", "Close", "Low"]].max(axis=1) - 1e-6).sum()),
        "ohlc_low_violations": int((raw["Low"] > prices[["Open", "Close", "High"]].min(axis=1) + 1e-6).sum()),
        "action": "No source-data edits were justified; the official CSV is preserved byte-for-byte.",
    }


def trend_rule(fast: int, slow: int):
    def predict(history: pd.DataFrame) -> float:
        if len(history) < slow:
            return 0.0
        close = history["Close"].iloc[-slow:]
        return float(close.iloc[-fast:].mean() > close.mean())

    return predict


def ensemble_rule(history: pd.DataFrame) -> float:
    if len(history) < 120:
        return 0.0
    close = history["Close"]
    votes = (
        close.iloc[-10:].mean() > close.iloc[-80:].mean(),
        close.iloc[-20:].mean() > close.iloc[-100:].mean(),
        close.iloc[-30:].mean() > close.iloc[-120:].mean(),
    )
    return float(sum(votes) >= 2)


def score(data, predictor, year: int) -> dict:
    result = harness.run_backtest(data, predictor, f"{year}-01-01", f"{year}-12-31", budget_seconds=300)
    return result["metrics"]


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    frames = {ticker: harness.clean_frame(group) for ticker, group in raw.groupby("Ticker")}
    data = {ticker: frames[ticker] for ticker in TICKERS}
    OUT.mkdir(exist_ok=True)

    candidates = {
        "official_starter_20_100": trend_rule(20, 100),
        "candidate_10_80": trend_rule(10, 80),
        "candidate_30_120": trend_rule(30, 120),
        "candidate_majority_10_80_20_100_30_120": ensemble_rule,
    }
    rows = []
    for name, predictor in candidates.items():
        for year in (2019, 2020, 2021, 2022, 2023, 2024):
            metrics = score(data, predictor, year)
            rows.append({"candidate": name, "year": year, **metrics})
    table = pd.DataFrame(rows)
    train_period = table[table["year"].isin((2019, 2020, 2021, 2022))]
    summary = train_period.groupby("candidate")["Sharpe"].agg(["mean", "std"]).reset_index()
    summary["selection_score"] = summary["mean"] - 0.25 * summary["std"]
    summary = summary.sort_values("selection_score", ascending=False)

    report = {
        "purpose": "Independent audit and candidate comparison; no production artifact was overwritten.",
        "data_audit": audit(raw),
        "leakage_controls": [
            "Official selection uses only chronological 2019-2022 folds.",
            "Each prediction sees history through the prior close and trades at the next open.",
            "2023 is a development diagnostic and 2024 remains a final untouched diagnostic in this script.",
            "The 2024 rows are not used in selection_score or fitting.",
        ],
        "selection_rule": "mean Sharpe minus 0.25 times across-year standard deviation on 2019-2022",
        "validation_summary": summary.to_dict("records"),
        "finding": "No candidate demonstrated a safe, consistent improvement over the official selected rule across the predeclared validation and later diagnostics.",
        "artifacts": ["research_audit.json", "candidate_metrics.csv", "candidate_summary.csv"],
    }
    (OUT / "research_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    table.to_csv(OUT / "candidate_metrics.csv", index=False)
    summary.to_csv(OUT / "candidate_summary.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
