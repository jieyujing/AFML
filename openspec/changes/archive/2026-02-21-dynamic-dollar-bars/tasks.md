## 1. Numba Core Extension
- [x] 1.1 In `afmlkit/bar/logic.py`, copy `_dollar_bar_indexer` as a base and modify it into `_dynamic_dollar_bar_indexer` which accepts `thresholds: NDArray[np.float64]` instead of `threshold: float`.

## 2. Dynamic Bar Builder
- [x] 2.1 In `afmlkit/bar/kit.py`, implement `DynamicDollarBarKit(BarBuilderBase)`.
- [x] 2.2 Inside `DynamicDollarBarKit`, compute daily dollar volume aggregates, calculate the EWMA, and map these dynamic thresholds back onto the original tick `timestamps` to create the `thresholds_array`.
- [x] 2.3 Connect `DynamicDollarBarKit._comp_bar_close()` to the new `_dynamic_dollar_bar_indexer`.

## 3. Evaluation and Scripting
- [x] 3.1 Create `scripts/dynamic_dollar_bars_run.py`.
- [x] 3.2 Add data loading logic (Polars/Pandas from `.parquet`).
- [x] 3.3 Add loop iterating through target frequencies [4, 6, 10, 20, 50].
- [x] 3.4 For each target frequency, instantiate `DynamicDollarBarKit`, build bars, compute log returns.
- [x] 3.5 Use `scipy.stats.jarque_bera` and `statsmodels` (or `pd.Series.autocorr`) to calculate the evaluation metrics (lowest JB is better, low autocorrelation is better).
- [x] 3.6 Automatically select the best performing frequency strategy.

## 4. Visualization
- [x] 4.1 Plot the JB scores vs. target frequencies as a bar/line chart.
- [x] 4.2 Use `matplotlib` to plot the Close price series for the optimal Dollar Bars generated.
