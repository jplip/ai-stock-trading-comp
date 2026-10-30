"""Reproducible model selection using the organizer's exact trading harness."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sdsu_harness as harness
from strategy import fit_strategy

# These choices are set before seeing the 2023 practice or 2024 results.
VALIDATION_YEARS = (2019, 2020, 2021, 2022)
VALIDATION_STOCKS = ("AAPL", "JPM", "XOM", "JNJ", "CROX", "INTC", "PLUG", "KO")
PRACTICE_STOCKS_FIXED = ("AAPL", "JPM", "XOM", "JNJ")
CANDIDATES = ("trend", "ridge", "extra_trees", "starter_trend", "defensive_trend")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def run_workflow(DATA, output_dir):
    """Train on allowed past data, select before 2023, then evaluate once.

    The final fitted object remains trained only through December 2022.
    Practice-year rows are supplied to the scoring harness as historical inputs,
    but their labels never enter fit_strategy or the numerical selection score.
    Initial 2023/2024 diagnostics motivated a lower-turnover candidate family;
    those years are reused development checks, not an untouched final test.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {str(ticker): harness.clean_frame(frame) for ticker, frame in DATA.items()}
    for frame in frames.values():
        if frame.index.min() < pd.Timestamp("2015-01-01") or frame.index.max() > pd.Timestamp("2024-12-31"):
            raise ValueError("Only the organizer's 2015–2024 data is allowed.")
    validation_names = [ticker for ticker in VALIDATION_STOCKS if ticker in frames]
    practice_names = [ticker for ticker in PRACTICE_STOCKS_FIXED if ticker in frames]
    if not validation_names or not practice_names:
        raise ValueError("Load the official club dataset before training.")
    validation_data = {ticker: frames[ticker] for ticker in validation_names}
    records = []
    for year in VALIDATION_YEARS:
        end = f"{year - 1}-12-31"
        training = {ticker: frame.loc["2015-01-01":end].copy() for ticker, frame in frames.items()}
        for kind in CANDIDATES:
            fitted = fit_strategy(training, kind=kind)
            assert fitted.training_label_end < f"{year}-01-01"
            result = harness.run_backtest(validation_data, fitted.predict,
                                          f"{year}-01-01", f"{year}-12-31", budget_seconds=300)
            records.append({"kind": kind, "year": year, "fit_through": end,
                            "last_training_label": fitted.training_label_end,
                            **result["metrics"]})
            print(f"Validation {year} / {kind}: Sharpe {result['metrics']['Sharpe']:.3f}", flush=True)
    validation_table = pd.DataFrame(records)
    summary = validation_table.groupby("kind", sort=True)["Sharpe"].agg(
        mean_sharpe="mean", std_sharpe="std").reset_index()
    # Reward a good average and penalize unstable results across different years.
    summary["selection_score"] = summary["mean_sharpe"] - .25 * summary["std_sharpe"]
    summary = summary.sort_values(["selection_score", "kind"], ascending=[False, True]).reset_index(drop=True)
    selected_kind = str(summary.iloc[0]["kind"])
    final_training = {ticker: frame.loc["2015-01-01":"2022-12-31"].copy()
                      for ticker, frame in frames.items()}
    model = fit_strategy(final_training, kind=selected_kind)
    assert model.training_label_end <= "2022-12-31"
    smoke = harness.smoke_check_predict(model.predict)
    practice_data = {ticker: frames[ticker] for ticker in practice_names}
    practice_result = harness.run_backtest(practice_data, model.predict,
                                           "2023-01-01", "2023-12-31", budget_seconds=300)
    hold_result = harness.run_backtest(practice_data, harness.buy_and_hold,
                                       "2023-01-01", "2023-12-31", budget_seconds=300)
    practice_table = pd.DataFrame({"Selected model": practice_result["metrics"],
                                   "Buy and hold": hold_result["metrics"]}).T
    practice_result["equity"].to_csv(output_dir / "practice_equity.csv")
    practice_result["trades"].to_csv(output_dir / "practice_trades.csv", index=False)
    validation_table.to_csv(output_dir / "validation_results.csv", index=False)
    summary.to_csv(output_dir / "model_selection.csv", index=False)
    practice_table.to_csv(output_dir / "practice_metrics.csv")
    report = {
        "selected_kind": selected_kind,
        "selection_rule": "Highest mean validation Sharpe minus 0.25 times its across-year standard deviation; ties alphabetically. Set before 2023 testing.",
        "training": {"stocks": sorted(final_training), "date_start": "2015-01-01",
                     "date_end": "2022-12-31", "samples": model.training_rows,
                     "last_feature_date": model.training_end,
                     "last_label_date": model.training_label_end},
        "validation": {"years": list(VALIDATION_YEARS), "stocks": validation_names,
                       "records": records, "summary": summary.to_dict("records")},
        "practice": {"start": "2023-01-01", "end": "2023-12-31", "stocks": practice_names,
                     "selected_metrics": practice_result["metrics"],
                     "buy_and_hold_metrics": hold_result["metrics"],
                     "used_for_numerical_selection_score": False,
                     "reused_for_development_feedback": True},
        "iteration": [{"kind": str(row["kind"]), "mean_validation_sharpe": float(row["mean_sharpe"]),
                       "selection_score": float(row["selection_score"]),
                       "decision": "selected" if row["kind"] == selected_kind else "not selected"}
                      for row in summary.to_dict("records")],
        "model": model.describe(), "contract_smoke_check": smoke,
        "harness_version": harness.HARNESS_VERSION,
        "harness_fee": harness.FEE_RATE,
        "python": platform.python_version(),
        "development_history": "An initial trend/ridge/tree comparison was scored on 2023 and 2024 and lagged buy-and-hold. That feedback motivated adding the provided low-turnover baseline and a discrete risk-tier alternative. Final numerical ranking still uses only 2019–2022 folds. 2023/2024 are development diagnostics; only organizer secret rounds are unseen final evaluation.",
        "limits": ["Historical validation does not guarantee hidden-round performance.",
                   "Different tickers share market movements; validation examples are not independent.",
                   "No 2023 or 2024 labels enter model fitting or the numerical ranking. Those years were viewed during development and are not claimed as untouched tests.",
                   "The organizer's simulator omits separate borrowing charges, slippage and interest."],
    }
    (output_dir / "report.json").write_text(json.dumps(_json_ready(report), indent=2, allow_nan=False) + "\n")
    print(f"Frozen selection: {selected_kind}; 2023 practice Sharpe {practice_result['metrics']['Sharpe']:.3f}", flush=True)
    return {"model": model, "report": report, "validation_table": validation_table,
            "practice_table": practice_table}
