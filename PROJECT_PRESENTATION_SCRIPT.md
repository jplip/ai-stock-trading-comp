# Signal Lab presentation script

“Our project is an auditable stock-trading decision system. It reads daily open, high, low, close, and volume data and returns a position between minus one and plus one. The goal is a reproducible strategy that respects the competition’s timing and risk contract.”

“We used the organizer’s official 2015–2024 data for 34 stocks. We checked missing values, duplicate ticker-date rows, invalid prices, and OHLC violations. The harness gives the strategy history through the previous close, executes at the next open, marks at the following open, and applies a 0.1 percent transaction fee.”

“We compared five approaches: a smooth trend rule, Ridge regression, Extra Trees, the official starter trend rule, and a defensive trend rule. The learned approaches use lagged returns, trend, volatility, range, volume, gaps, and close location. Every feature is causal.”

“To reduce look-ahead bias, we used chronological validation folds for 2019 through 2022. Each fold was trained only on data before that year. We ranked candidates by mean Sharpe minus one quarter of the cross-year Sharpe standard deviation, so unstable results were penalized.”

“The selected model was the simple 20/100-day trend rule. If the 20-day average is above the 100-day average, it is fully long; otherwise it stays in cash. It has no learned coefficients, no hidden state, and is easy to explain and reproduce.”

“On the saved 2023 practice check across AAPL, JPM, XOM, and JNJ, the selected model returned about 0.88 percent with a Sharpe of 0.14. Buy-and-hold returned about 16.07 percent with a Sharpe of 1.19. The honest conclusion is that our selected strategy did not beat the benchmark in that period.”

“The project completes the full pipeline: data validation, causal feature construction, walk-forward model comparison, frozen strategy selection, official-harness backtesting, contract checks, metrics, equity and trade exports, a serialized submission, and an audit report. The secret competition rounds remain unseen, so we do not present them as results.”

“The main value is the discipline of the experiment: clear timing, explicit leakage controls, reproducible artifacts, and an honest account of the result.”
