#!/usr/bin/env python3
"""Run with `python test_run.py`; all fixtures/reports stay in a temporary folder."""

import csv
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np
import run as lab


class SignalLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(123)
        self.rows = []
        price = 100.0
        for i in range(240):
            opening = price * math.exp(float(rng.normal(0, .003)))
            price = opening * math.exp(float(rng.normal(.0002, .012)))
            self.rows.append([str(dt.date(2024, 1, 1) + dt.timedelta(days=i)), opening, price, 100000 + i])

    def write(self, rows=None, headers=None, name="prices.csv"):
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers or ["Date", "Open", "Close", "Volume"])
            writer.writerows(self.rows if rows is None else rows)
        return path

    def test_import_schema_and_labels(self):
        rows, volume = lab.read_prices(self.write())
        x, y, returns, indices, latest = lab.make_features(rows, volume)
        self.assertEqual(x.shape, (218, 11))
        self.assertEqual(latest.shape, (1, 11))
        i = indices[0]
        self.assertAlmostEqual(returns[0], rows[i + 2]["open"] / rows[i + 1]["open"] - 1)
        self.assertEqual(y[0], float(returns[0] > 0))
        self.assertTrue(np.isfinite(x).all())

    def test_flat_buy_hold_costs_and_cash(self):
        result = lab.simulate(np.ones(5), np.zeros(5), .001, 252)
        self.assertAlmostEqual(result["total_return"], .999 ** 2 - 1)
        self.assertEqual(result["total_turnover"], 2)
        cash = lab.simulate(np.zeros(5), np.zeros(5), .001, 252)
        self.assertEqual(cash["total_return"], 0)
        self.assertIsNone(cash["annualized_sharpe_zero_risk_free"])

    def test_drift_costs_and_final_liquidation(self):
        result = lab.simulate(np.array([.5, .5]), np.array([.1, 0]), .01, 252)
        drifted = .5 * 1.1 / 1.05
        expected = (.995 * 1.05) * (1 - .01 * abs(.5 - drifted)) * .995 - 1
        self.assertAlmostEqual(result["total_return"], expected)
        self.assertAlmostEqual(result["total_turnover"], .5 + abs(.5 - drifted) + .5)

    def test_no_future_feature_or_prediction_influence(self):
        rows, volume = lab.read_prices(self.write())
        x, y, _, _, _ = lab.make_features(rows, volume)
        changed = [dict(row) for row in rows]
        for row in changed[210:]:
            row["open"] *= 1.8
            row["close"] *= 2.2
        x2, y2, _, _, _ = lab.make_features(changed, volume)
        # Fit strictly earlier data; future data must not change earlier features,
        # labels, or the predictions obtained by repeating the same experiment.
        np.testing.assert_array_equal(x[:188], x2[:188])
        np.testing.assert_array_equal(y[:188], y2[:188])
        model = lab.Ensemble().fit(x[:128], y[:128], 42)
        model2 = lab.Ensemble().fit(x2[:128], y2[:128], 42)
        np.testing.assert_array_equal(model.predict(x[160:188]), model2.predict(x2[160:188]))

    def test_duplicate_headers_reject(self):
        with self.assertRaisesRegex(ValueError, "duplicate headers"):
            lab.read_prices(self.write(headers=["Date", "Open", "Close", " open "]))

    def test_extra_and_missing_fields_reject(self):
        for altered, expected in [(self.rows[0] + ["extra"], "extra fields"), (self.rows[0][:-1], "missing fields")]:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    lab.read_prices(self.write(rows=[altered] + self.rows[1:]))

    def test_multiple_tickers_reject_even_with_unique_dates(self):
        rows = [row + ["ABC" if i < 100 else "XYZ"] for i, row in enumerate(self.rows)]
        with self.assertRaisesRegex(ValueError, "multiple stocks"):
            lab.read_prices(self.write(rows, ["Date", "Open", "Close", "Volume", "Ticker"]))

    def test_contradictory_symbol_ticker_reject(self):
        rows = [row + ["ABC", "XYZ"] for row in self.rows]
        with self.assertRaisesRegex(ValueError, "contradictory Symbol and Ticker"):
            lab.read_prices(self.write(rows, ["Date", "Open", "Close", "Volume", "Symbol", "Ticker"]))

    def test_single_symbol_case_variants_accept(self):
        rows = [row + ["abc" if i % 2 else "ABC", " ABC "] for i, row in enumerate(self.rows)]
        accepted, _ = lab.read_prices(self.write(rows, ["Date", "Open", "Close", "Volume", "Symbol", "Ticker"]))
        self.assertEqual(len(accepted), len(rows))

    def test_bad_dates_reject(self):
        for stamp, message in [(self.rows[0][0], "strictly increasing"), ("bad-date", "Invalid isoformat")]:
            with self.subTest(stamp=stamp):
                rows = [list(row) for row in self.rows]
                rows[1][0] = stamp
                with self.assertRaisesRegex(ValueError, message):
                    lab.read_prices(self.write(rows))

    def test_end_to_end_single_class_minimum_input(self):
        rows = [[row[0], 100, 100] for row in self.rows[:180]]
        path = self.write(rows, ["Date", "Open", "Close"])
        output = self.directory / "results"
        completed = subprocess.run([sys.executable, str(Path(lab.__file__)), str(path), "--output", str(output)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((output / "report.json").read_text())
        signal = json.loads((output / "latest_signal.json").read_text())
        self.assertEqual(report["splits"]["test_samples"], 32)
        self.assertEqual(report["test_strategy"]["total_return"], 0)
        self.assertAlmostEqual(report["test_buy_and_hold"]["total_return"], .999 ** 2 - 1)
        self.assertFalse(signal["official_submission_format"])
        self.assertFalse(signal["calibrated_probability"])
        self.assertTrue((output / "predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
