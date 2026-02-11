## 1. Stationarity Enforcement

- [x] 1.1 Create `src/afml/stationarity.py` to implement fractional differentiation and stationarity checking logic.
    - Implement `frac_diff_ffd(series, d, thres=1e-5)` if not present.
    - Implement `get_min_d(series, max_d=1.0, step_size=0.1, p_val_thres=0.05)` that finds the minimum `d` passing ADF test.
- [x] 1.2 Update `src/afml_polars_pipeline.py` to import and use `get_min_d` for automatic stationarity detection before feature generation.

## 2. Statistical Verification (DSR)

- [x] 2.1 Create `src/afml/metrics.py` to implement performance metrics.
    - Implement `sharpe_ratio(returns, risk_free=0, periods=252)`.
    - Implement `probabilistic_sharpe_ratio(observed_sr, benchmark_sr, skew, kurtosis, n_obs)`.
    - Implement `deflated_sharpe_ratio(observed_sr, skew, kurtosis, n_obs, n_trials)`.
- [x] 2.2 Update `src/afml_polars_pipeline.py` to compute and report DSR after backtesting.

## 3. Pipeline Orchestration

- [x] 3.1 Refactor `src/afml_polars_pipeline.py` to support `auto_stationarity` flag (default True).
- [x] 3.2 Ensure the pipeline passes the dynamically calculated `d` to the feature engineering step.
- [x] 3.3 Add proper logging of all intermediate steps (optimal d found, DSR probability, etc.).
- [x] 3.4 Add final "Acceptance" check: If DSR Probability > 0.95 (or user defined threshold), mark as PASSED, else FAILED.
