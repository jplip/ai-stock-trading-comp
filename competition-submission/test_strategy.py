"""Contract and causal-timing checks; run with python test_strategy.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cloudpickle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy
import sdsu_harness as harness


def synthetic_frame(seed=82, rows=500):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, rows)))
    opening = close * np.exp(rng.normal(0, .003, rows))
    return pd.DataFrame({
        "Open": opening, "High": np.maximum(opening, close) * 1.01,
        "Low": np.minimum(opening, close) * .99, "Close": close,
        "Volume": rng.integers(100_000, 2_000_000, rows).astype(float),
    }, index=pd.bdate_range("2018-01-02", periods=rows, name="Date"))


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_frame()
        cls.training = cls.frame.iloc[:350].copy()
        cls.models = {kind: strategy.fit_strategy({"unnamed": cls.training}, kind)
                      for kind in ("trend", "ridge", "extra_trees", "starter_trend", "defensive_trend")}

    def test_batch_fast_and_tail_features_match_exactly(self):
        for end in (252, 350, 500):
            history = self.frame.iloc[:end]
            full = strategy.make_features(history).iloc[-1].to_numpy()
            tail = strategy.make_features(history.tail(252)).iloc[-1].to_numpy()
            np.testing.assert_array_equal(full, tail)
            np.testing.assert_array_equal(full, strategy._latest_features(history))

    def test_future_changes_cannot_change_prefix_features_or_predictions(self):
        changed = self.frame.copy()
        changed.iloc[400:, :4] *= 1.8
        changed.iloc[400:, 4] *= 3
        pd.testing.assert_frame_equal(strategy.make_features(self.frame).iloc[:400],
                                      strategy.make_features(changed).iloc[:400])
        for model in self.models.values():
            for end in (252, 330, 400):
                self.assertEqual(model.predict(self.frame.iloc[:end]),
                                 model.predict(changed.iloc[:end]))

    def test_short_flat_deterministic_bounds_and_no_input_mutation(self):
        flat = self.frame.copy()
        flat.loc[:, ["Open", "High", "Low", "Close"]] = 100.0
        flat.loc[:, "Volume"] = 0.0
        for model in self.models.values():
            for history in (self.frame, flat, self.frame.iloc[:5], self.frame.iloc[:120]):
                before = history.copy(deep=True)
                value = model.predict(history)
                self.assertIs(type(value), float)
                self.assertTrue(np.isfinite(value) and -1 <= value <= 1)
                self.assertEqual(value, model.predict(history.copy(deep=True)))
                pd.testing.assert_frame_equal(history, before)
                if len(history) < model.describe()["minimum_history"]:
                    self.assertEqual(value, 0.0)

    def test_training_labels_end_inside_only_the_provided_slice(self):
        cutoff = self.training.index[-1]
        changed = self.frame.copy()
        changed.iloc[350:, :4] *= 2
        for kind, model in self.models.items():
            details = model.describe()
            self.assertLessEqual(pd.Timestamp(details["training_label_end"]), cutoff)
            self.assertEqual(pd.Timestamp(details["training_label_end"]), cutoff)
            self.assertLess(pd.Timestamp(details["training_end"]), cutoff)
            again = strategy.fit_strategy({"other name": changed.loc[:cutoff]}, kind)
            np.testing.assert_array_equal(model.means, again.means)
            np.testing.assert_array_equal(model.coefficients, again.coefficients)
            self.assertEqual(model.predict(self.frame), again.predict(self.frame))

    def test_ticker_metadata_and_extra_columns_are_not_inputs(self):
        decorated = self.frame.copy()
        decorated["Ticker"] = "arbitrary ticker"
        decorated.attrs["ticker"] = "different ticker"
        for kind, model in self.models.items():
            self.assertEqual(model.predict(self.frame), model.predict(decorated))
            again = strategy.fit_strategy({"renamed ticker": decorated.iloc[:350]}, kind)
            self.assertEqual(model.predict(self.frame), again.predict(self.frame))
            self.assertTrue(all("ticker" not in name.lower() for name in model.describe()["features"]))

    def test_official_harness_smoke_all_candidates(self):
        for kind, model in self.models.items():
            with self.subTest(kind=kind):
                self.assertTrue(harness.smoke_check_predict(model.predict)["passed"])

    def test_low_churn_rules_match_starter_and_fixed_defensive_tiers(self):
        rising = self.frame.copy()
        rising["Close"] = np.linspace(100, 150, len(rising))
        falling = rising.copy()
        falling["Close"] = rising["Close"].to_numpy()[::-1]
        for history in (self.frame, rising, falling):
            c = history["Close"]
            expected = 1.0 if c.iloc[-20:].mean() > c.iloc[-100:].mean() else 0.0
            self.assertEqual(self.models["starter_trend"].predict(history), expected)
        self.assertEqual(self.models["defensive_trend"].predict(rising), .75)
        self.assertEqual(self.models["defensive_trend"].predict(falling), 0.0)
        volatile = rising.copy()
        volatile["Close"] += np.where(np.arange(len(volatile)) % 2, 4, -4)
        self.assertEqual(self.models["defensive_trend"].predict(volatile), .50)
        self.assertFalse(self.models["defensive_trend"].describe()["learned_parameters"])

    def test_cloudpickle_by_value_runs_in_fresh_isolated_process(self):
        cloudpickle.register_pickle_by_value(strategy)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bundle.pkl"
                payload = {"predict": self.models["extra_trees"].predict,
                           "history": self.frame, "expected": self.models["extra_trees"].predict(self.frame)}
                path.write_bytes(cloudpickle.dumps(payload))
                code = ("import cloudpickle,sys,json; b=cloudpickle.load(open(sys.argv[1],'rb')); "
                        "p=b['predict'](b['history']); assert type(p) is float; "
                        "assert -1<=p<=1 and abs(p-b['expected'])<1e-12; print(json.dumps({'position':p}))")
                result = subprocess.run([sys.executable, "-I", "-c", code, str(path)],
                                        cwd=directory, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout)["position"], float)
        finally:
            cloudpickle.unregister_pickle_by_value(strategy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
