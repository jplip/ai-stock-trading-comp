"""Readable equivalent of the selected strategy in the saved submission."""
import numpy as np


def predict(history):
    if len(history) < 100:
        return 0.0
    close = history["Close"].iloc[-100:].to_numpy(dtype=float)
    if not np.isfinite(close).all() or np.any(close <= 0):
        return 0.0
    recent_average = float(close[-20:].mean())
    longer_average = float(close.mean())
    return 1.0 if recent_average > longer_average else 0.0
