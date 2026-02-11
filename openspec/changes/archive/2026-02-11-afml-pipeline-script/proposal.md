## Why

To rigorously validate quantitative strategies, we need a standardized pipeline that strictly adheres to the Advances in Financial Machine Learning (AFML) methodology. This pipeline must handle data ingestion, stationarity checks, labeling, feature engineering, modeling, and final verification with Deflated Sharpe Ratio (DSR), ensuring no look-ahead bias or overfitting.

## What Changes

We will implement a robust AFML pipeline script that accepts a data file and parameters.
Key improvements over ad-hoc scripts:
1.  **Stationarity Enforcement**: Automatically find the minimum fractional differentiation `d` such that the series is stationary (p-value < 0.05).
2.  **Strict Labeling**: Apply Triple-Barrier Method with vertical barriers and profit-taking/stop-loss limits.
3.  **Sample Weights**: Calculate uniqueness and decay to handle overlapping labels.
4.  **Purged CV**: Use Purged K-Fold Cross-Validation with embargo to prevent leakage.
5.  **Meta-Labeling**: Train a primary model (side) and a meta-model (size/confidence).
6.  **Bet Sizing**: Size positions based on meta-model probabilities.
7.  **Verification**: Compute Deflated Sharpe Ratio (DSR) to statistically validate the strategy.

## Capabilities

### New Capabilities
- `afml-pipeline`: End-to-end execution of the AFML workflow including data processing, labeling, modeling, and statistical verification.

### Modified Capabilities
- `feature-engineering`: Enhanced to support dynamic fractional differentiation selection (min `d`).

## Impact

- **New Script**: `src/run_afml_pipeline.py` (or updated `src/afml_polars_pipeline.py`).
- **Dependencies**: `polars`, `numpy`, `scikit-learn`, `mlfinlab` (if available) or internal `afml` package.
