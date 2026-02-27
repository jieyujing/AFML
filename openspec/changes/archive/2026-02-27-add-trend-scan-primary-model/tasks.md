## 1. Core Algorithmic Implementation (Numba)

- [x] 1.1 Create `afmlkit/features/trend_scan.py` (or corresponding module path) structure.
- [x] 1.2 Implement the Numba `@njit` core backend function `_trend_scan_core` that processes a 1D price array and a list of event indices, backwards scans through windows `L`, computes OLS t-statistics, handles zero-variance edge cases, and returns the maximum t-value, its side, and the optimal window length.
- [x] 1.3 Implement the Python frontend wrapper `trend_scan_labels(price_series, t_events, L_windows)` that aligns Pandas indices, converts data to fast NumPy arrays for the Numba kernel, and formats the output into a DataFrame with `t1`, `t_value`, and `side` columns.

## 2. CUSUM Pipeline Adjustments

- [x] 2.1 Update `scripts/cusum_filtering.py` (and related CUSUM functions if necessary) to export discrete timestamps (`t_events`) cleanly so they can be injected into the downstream Trend Scan. Ensure time causality validation.

## 3. Triple-Barrier & Meta-Labeling Integration

- [x] 3.1 Refactor the Triple-Barrier labeling invocation in the main quantitative pipeline to consume the dynamically generated `side` array from the Trend Scan DataFrame.
- [x] 3.2 Implement a mechanism to pass the absolute `t_value` outputted by Trend Scan as `sample_weight` parameters or raw material for sample weight calculation in the Meta-Model training phase.

## 4. Verification & Testing

- [x] 4.1 Write a test script or validation routine verifying that the Numba-accelerated Trend Scan output matches a pure-Pandas/Statsmodels baseline Implementation (to assure mathematical correctness).
- [x] 4.2 Verify through debugging or log outputs that no look-ahead bias is introduced (i.e. each event's backward window does not touch future data).
