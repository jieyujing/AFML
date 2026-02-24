# Implementation Tasks

## 1. Initialize Pipeline Script
- [x] Create Python script `scripts/cusum_filtering.py`.
- [x] Import necessary libraries: `pandas`, `numpy`, and `afmlkit` modules.
- [x] Add data loading logic to read `outputs/dollar_bars/dollar_bars_freq20.csv`.

## 2. Implement Volatility & Threshold Calculation
- [x] Extract the `close` price column.
- [x] Calculate log returns of the close price.
- [x] Compute the exponentially weighted standard deviation ($\sigma_t$) with a lookback logic (e.g., span=100) using `afmlkit.feature.core.volatility.ewms`.
- [x] Pre-process the NaNs in the calculated volatility array (using `bfill` or fallback logic) to ensure numeric stability.
- [x] Create the dynamic threshold sequence arrays by multiplying volatility by $k=2.0$.

## 3. Executive CUSUM Filtering
- [x] Extract the close prices as a continuous `np.float64` array.
- [x] Execute `afmlkit.sampling.filters.cusum_filter(close_prices, dynamic_threshold)`.
- [x] Receive the returned array of integer `event_indices`.

## 4. Data Extraction and Saving
- [x] Extract the corresponding sub-sampled rows from the original Dollar Bars DataFrame using `event_indices`.
- [x] Log the compression ratio (number of events / sample size) to evaluate filtering aggressiveness.
- [x] Save the resulting filtered dataset to an output CSV (e.g., `outputs/dollar_bars/cusum_sampled_bars.csv`).
