"""SDSU AI Club trading harness v1.0.

Share this exact file with the organizer-side builder. The member notebook embeds
the same bytes. This is a deterministic backtest, not a security sandbox.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import pandas as pd

HARNESS_VERSION = 'sdsu-trading-harness-v1.0'
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
FEE_RATE = 0.001
WARMUP_BARS = 252
TRADING_DAYS_PER_YEAR = 252


class ContractError(ValueError):
    """The submitted prediction function broke the published contract."""


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a daily OHLCV frame without filling missing prices."""
    result = frame.copy()
    if 'Date' in result.columns:
        result = result.set_index('Date')
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize()
    result.index.name = 'Date'
    if result.index.has_duplicates:
        raise ValueError('Duplicate dates: one stock must have one row per day.')
    missing = set(OHLCV) - set(result.columns)
    if missing:
        raise ValueError(f'Missing OHLCV columns: {sorted(missing)}')
    result = result[OHLCV].sort_index().astype(float)
    result.attrs = {}
    if len(result) == 0 or not np.isfinite(result.to_numpy()).all():
        raise ValueError('Data must contain finite OHLCV values; inspect missing rows.')
    if (result[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        raise ValueError('Prices must be positive.')
    if (result['Volume'] < 0).any():
        raise ValueError('Volume cannot be negative.')
    if (result['High'] + 1e-6 < result[['Open', 'Close', 'Low']].max(axis=1)).any():
        raise ValueError('A High value is below another price in the same bar.')
    if (result['Low'] - 1e-6 > result[['Open', 'Close', 'High']].min(axis=1)).any():
        raise ValueError('A Low value is above another price in the same bar.')
    return result


def checked_position(result) -> float:
    if isinstance(result, (bool, np.bool_)) or not isinstance(result, (float, np.floating)):
        raise ContractError('predict(history) must return a float, e.g. return float(0.5).')
    result = float(result)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ContractError(f'Position {result!r} is not a finite float between -1 and 1.')
    return result


def smoke_check_predict(predict: Callable) -> dict:
    """Check unseen dummy OHLCV input; this data is never used for training."""
    rng = np.random.default_rng(812)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .008, WARMUP_BARS)))
    history = pd.DataFrame({
        'Open': close * .999,
        'High': close * 1.01,
        'Low': close * .99,
        'Close': close,
        'Volume': rng.integers(100_000, 1_000_000, WARMUP_BARS).astype(float),
    }, index=pd.bdate_range('2020-01-02', periods=WARMUP_BARS, name='Date'))
    before = history.copy(deep=True)
    start = time.perf_counter()
    first = checked_position(predict(history))
    elapsed = time.perf_counter() - start
    if not history.equals(before):
        raise ContractError('predict() modified its input. Work on a copy instead.')
    second = checked_position(predict(before.copy(deep=True)))
    if abs(first - second) > 1e-10:
        raise ContractError('Identical histories must produce identical positions.')
    flat = before.copy()
    flat[['Open', 'High', 'Low', 'Close']] = 100.0
    checked_position(predict(flat))
    return {'dummy_position': first, 'seconds_per_call': round(elapsed, 4), 'passed': True}


def _rebalance(cash: float, shares: float, price: float, target: float, fee: float):
    """Solve for target exposure as a fraction of equity AFTER transaction fees."""
    equity = cash + shares * price
    if equity <= 0:
        raise ContractError('A stock allocation became insolvent (equity <= 0).')
    gap = target * equity - shares * price
    denominator = 1.0 + target * fee if gap >= 0 else 1.0 - target * fee
    notional = gap / denominator
    cost = fee * abs(notional)
    cash -= notional + cost
    shares += notional / price
    after = cash + shares * price
    if after <= 0:
        raise ContractError('Transaction fees exhausted a stock allocation.')
    if abs(shares * price) > after * (1 + 1e-9):
        raise ContractError('Gross exposure exceeded the no-leverage limit.')
    return cash, shares, cost, abs(notional)


def performance_metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    sigma = float(daily.std(ddof=1)) if len(daily) >= 2 else 0.0
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * daily.mean() / sigma) if sigma > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        'Sharpe': sharpe,
        'Total return': float(equity.iloc[-1] / equity.iloc[0] - 1),
        'Max drawdown': float(drawdown.min()),
        'Final equity': float(equity.iloc[-1]),
        'Intervals': int(len(daily)),
    }


def run_backtest(data: dict, predict: Callable, start: str, end: str,
                 initial_capital: float = 10_000.0, fee_rate: float = FEE_RATE,
                 warmup_bars: int = WARMUP_BARS, budget_seconds: float = 120.0) -> dict:
    """Observe prior close -> trade next open -> mark at the following open.

    `start` and `end` inclusively select execution/mark dates. The final selected
    open liquidates all holdings; it is not an additional prediction date.
    Each stock starts with an equal independent allocation. Allocations are not
    rebalanced across stocks. No interest, dividends, borrow charges, or slippage
    beyond the specified fee are separately credited/debited.
    """
    if not data or initial_capital <= 0 or not 0 <= fee_rate < .1:
        raise ValueError('Supply stock data, positive capital, and a valid fee rate.')
    frames = {str(name): clean_frame(frame) for name, frame in data.items()}
    dates = None
    for frame in frames.values():
        dates = frame.index if dates is None else dates.intersection(frame.index)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].sort_values()
    if len(dates) < 3:
        raise ValueError('Choose a window containing at least three common trading dates.')
    deadline = time.perf_counter() + budget_seconds
    allocation = initial_capital / len(frames)
    curves, trades = {}, []
    total_fees = 0.0
    for name, frame in frames.items():
        first = frame.index.get_loc(dates[0])
        if first < warmup_bars:
            raise ValueError(f'{name}: need {warmup_bars} earlier bars before the window starts.')
        history_start = first - warmup_bars
        cash, shares = float(allocation), 0.0
        values = [allocation]
        for k, date in enumerate(dates[:-1]):
            if time.perf_counter() > deadline:
                raise TimeoutError('Local backtest budget exceeded. Simplify predict(); never train inside it.')
            at = frame.index.get_loc(date)
            history = frame.iloc[history_start:at][OHLCV].copy(deep=True)
            history.attrs = {}
            position = checked_position(predict(history))
            if time.perf_counter() > deadline:
                raise TimeoutError('Local backtest budget exceeded during a prediction.')
            price = float(frame.at[date, 'Open'])
            cash, shares, cost, turnover = _rebalance(cash, shares, price, position, fee_rate)
            total_fees += cost
            if turnover > 1e-8:
                trades.append({'Stock': name, 'Execution date': date,
                               'History through': history.index[-1], 'Target': position,
                               'Notional traded': turnover, 'Fee': cost})
            marked = cash + shares * float(frame.at[dates[k + 1], 'Open'])
            if marked <= 0:
                raise ContractError(f'{name}: a stock allocation became insolvent. Reduce exposure or shorting.')
            values.append(marked)
        exit_price = float(frame.at[dates[-1], 'Open'])
        cash, shares, cost, turnover = _rebalance(cash, shares, exit_price, 0.0, fee_rate)
        total_fees += cost
        values[-1] = cash
        if turnover > 1e-8:
            trades.append({'Stock': name, 'Execution date': dates[-1],
                           'History through': pd.NaT, 'Target': 0.0,
                           'Notional traded': turnover, 'Fee': cost})
        curves[name] = pd.Series(values, index=dates)
    by_stock = pd.DataFrame(curves)
    equity = by_stock.sum(axis=1).rename('Portfolio equity')
    metrics = performance_metrics(equity)
    metrics['Fees paid'] = total_fees
    metrics['Trades'] = len(trades)
    return {'equity': equity, 'daily_returns': equity.pct_change().dropna(),
            'by_stock': by_stock, 'trades': pd.DataFrame(trades), 'metrics': metrics,
            'harness_version': HARNESS_VERSION}


def buy_and_hold(history: pd.DataFrame) -> float:
    return 1.0
