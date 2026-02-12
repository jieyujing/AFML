## 1. JB Statistics Module

- [x] 1.1 Add `scipy.stats` import to `src/afml/visualization.py`
- [x] 1.2 Create helper function `_compute_jb_statistics(returns: np.ndarray) -> dict` that computes JB stat, p-value, skewness, kurtosis

## 2. Enhanced plot_bar_stats Method

- [x] 2.1 Modify `plot_bar_stats()` signature to accept optional `time_bars_df: pl.DataFrame = None` parameter
- [x] 2.2 Add JB statistics computation for dollar bars using helper function
- [x] 2.3 Add JB statistics computation for time bars if parameter provided
- [x] 2.4 Create statistics comparison table using matplotlib table or text
- [x] 2.5 Add AFML compliance conclusion text based on JB comparison
- [x] 2.6 Update method to return JB statistics dictionary

## 3. Pipeline Integration

- [x] 3.1 Update `src/afml_polars_pipeline.py` to compute and print JB statistics for dollar bars and time bars
- [x] 3.2 Print JB statistics to console in addition to visualization

## 4. Testing and Verification

- [x] 4.1 Create unit tests in `tests/test_jb_statistics.py` for `_compute_jb_statistics()` helper
- [x] 4.2 Test `plot_bar_stats()` with and without time bars comparison
- [x] 4.3 Run end-to-end pipeline to verify JB visualization displays correctly
