# Progress Tracking

## Completed Work
- **[2026-02-11] AFML Pipeline Implementation**: 
    - Implemented `src/afml/stationarity.py` for automatic fractional differentiation selection (min `d`).
    - Implemented `src/afml/metrics.py` for Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).
    - Refactored `src/afml_polars_pipeline.py` into an end-to-end AFML pipeline with stationarity enforcement and statistical verification.
    - Archived change `afml-pipeline-script`.
    - Synced `afml-pipeline` capability to main specs.
- **[2026-02-11] AFML Pipeline Execution (T9999)**:
    - Successfully executed `src/afml_polars_pipeline.py` on `T9999.CCFX-2020-1-1-To-2026-02-11-1m.csv`.
    - Generated 5,932 dollar bars from 380,100 raw ticks.
    - Achieved DSR of 1.0000 (Pass) on the labeled events (4,290 samples).
    - Saved all artifacts to `data/` directory.
- **[2026-02-11] Visual Analysis Components**:
    - Implemented `src/afml/visualization.py` with comprehensive plotting for AFML artifacts.
    - Integrated visualization into `src/afml_polars_pipeline.py` with `--visualize` flag.
    - Verified plots for stationarity search, triple barrier events, meta-labeling performance, and equity curves.
    - Updated README with visualization details.
- **[2026-02-11] Project Documentation (README)**:
    - Created comprehensive `README.md` with AFML methodology overview, project architecture, and quick start guide.
    - Standardized project positioning as a high-performance AFML Quant Factory using Polars.
- **[2026-02-12] Polars Optimization & File Cleanup**:
    - Completed full migration to Polars-first architecture.
    - Simplified library structure by moving Polars implementations to the root package (`afml.*`).
    - Standardized names (e.g., `DollarBarsProcessor` now points to Polars version by default).
    - Removed legacy pandas-based files and the `polars/` subdirectory.
    - Eliminated pandas dependencies from `stationarity.py` and `metrics.py`.
- **[2026-02-12] Environment & Configuration**:
    - Updated `.gitignore` to ignore `data/`, `visual_analysis/`, and `*.csv` files to keep the repository clean.
    - Moved `examples/afml_polars_pipeline.py` to the root directory as the primary entry point.
- **[2026-02-14] Pipeline Optimization**:
    - Switched all intermediate data storage from CSV to Parquet in `afml_polars_pipeline.py` for significant performance improvement on large datasets.
    - Updated default input/output paths to use `.parquet` extension.
- **[2026-02-14] Performance Audit & Optimization**:
    - `stationarity.py`: Replaced Python for-loop FFD convolution with `np.convolve` — verified **19.5x speedup**.
    - `labeling.py`: Vectorized `_get_vertical_barrier` (eliminated per-row `.row(idx, named=True)`), pre-allocated label arrays, replaced `events.to_dicts()` with direct column extraction.
    - `sample_weights.py`: Replaced O(N²) Python `_compute_uniqueness` loop with vectorized NumPy offset-broadcasting — 10K events in 12ms. Replaced `_compute_decay_weights` loop with `np.power`.
    - `features.py`: Batched all rolling window `with_columns` calls into single expression list.
    - `afml_polars_pipeline.py`: Eliminated stationarity double-computation (`get_min_d` + `get_stationarity_search_history` → single `get_stationarity_search_history`), cached `collect_schema().names()` (was called 3x), fixed `run_step_load` to use `sink_parquet` and removed massive DataFrame from result dict, fixed `run_step_bars` LazyFrame `.columns` compatibility.
- **[2026-02-14] Chunked Dollar Bars (OOM Fix for 1.5B rows)**:
    - Root cause: `--step bars` eagerly loaded 1.57B rows via `pl.read_parquet()` + global `sort()` + `cum_sum()` — impossible to stream.
    - Added `DollarBarsProcessor.transform_chunked()`: processes data in 50M-row vectorized chunks with running `cum_offset`, uses `group_by("bar_id")` within each chunk, then re-aggregates split bars across chunk boundaries.
    - Updated `run_step_bars` to always `scan_parquet` (lazy) and route to `transform_chunked()` for parquet files.
    - Successfully generated 4,553 dollar bars from 1,573,743,538 rows without OOM (32 chunks, ~5 min).

## Current Status
- **AFML Implementation**: Core pipeline is now rigorous, verified, and fully optimized for Polars.
- **Architecture**: Cleaned up legacy files, library is now Polars-native.
- **Verification**: Pipeline parity preserved after cleanup.

## Next Steps
- Implement proper Out-Of-Sample (OOS) testing and Backtesting paths (Chapter 16).
- Add Structural Break features (Chapter 17).
- Expand feature engineering with more microstructural features.
