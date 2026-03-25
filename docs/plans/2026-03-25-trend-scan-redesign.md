# Trend Scan Redesign (Forward-Looking Labels)

## Overview

The current implementation of `trend_scan_labels` in `afmlkit/feature/core/trend_scan.py` operates in a **backward-looking** manner. It is used as a Primary Model (signal generator) to provide the `side` for Meta-Labeling. However, this deviates from the strict definition in Advances in Financial Machine Learning (AFML) / Machine Learning for Asset Managers (MLAM), where Trend Scanning is proposed as a **forward-looking labeling method**.

This document outlines the design to refactor `trend_scan_labels` to align perfectly with the theoretical definition: scanning future windows $[t, t+L]$ to determine the optimal trend and returning the future timestamp `t1` where the trend optimally formed.

## Core Algorithm Refactoring (Numba Kernel)

The underlying `@njit` kernel `_trend_scan_core` will be modified:

1.  **Scanning Direction**: Instead of looking back from `idx - L + 1` to `idx`, it will look forward from `idx` to `idx + L - 1` (inclusive, representing a window of length `L`).
2.  **Boundary Handling**:
    *   If `idx + L > len(prices)`, the window extends beyond the available price series. We will skip such windows.
    *   If no valid windows (length $\ge 3$) can be formed for a given event, the event will receive neutral/null outputs (e.g., `best_L = 0`, `t_value = 0`, `side = 0`).

## Python Frontend Interface Changes

The wrapper function `trend_scan_labels` will be updated to format the output appropriately for downstream tasks (like PurgedKFold and SampleWeights):

1.  **`t1` Column Conversion**: The original Numba kernel returned the length of the optimal window `best_L` in the `t1` column. This will be transformed into an actual **Datetime** representing the end of the optimal window.
    *   Formula: `t1[i] = price_series.index[event_idx[i] + best_L[i] - 1]`
    *   This is critical for establishing the "lifespan" of the label.
2.  **Handling Edge Cases**: Events too close to the end of the series might not have any valid forward windows. These will be dropped from the final output DataFrame, or their `t1` will be set to `NaT` (we'll opt for dropping or returning neutral fields based on the core's default output, likely dropping invalid `best_L == 0` rows).

## Testing Updates

The existing test suite `tests/features/test_trend_scan.py` will require significant updates:
*   Tests verifying backward causality will need to be inverted to verify forward causality.
*   Assertions checking boundary conditions at the beginning of the series will move to the end of the series.
*   The type and content of the `t1` column must be asserted as `pd.DatetimeIndex` / `pd.Timestamp` instead of integers.

## Implementation Steps

1.  Modify `afmlkit/feature/core/trend_scan.py`: Update Numba kernel mathematically and index-wise. Update Pandas frontend to map `t1` to timestamps.
2.  Modify `tests/features/test_trend_scan.py`: Rewrite tests to reflect the new forward-looking behavior.
3.  Modify `scripts/cusum_filtering.py`: Update the example script to reflect the new nature of `trend_scan_labels`. Since it is now a *Labeling* method, its usage in Meta-Labeling as a *Primary Signal* is theoretically invalid (it would introduce look-ahead bias if used as a feature). We will discuss and implement the correct integration if needed, or simply update it to show how it's used as a pure label generation step.

## Note on Usage
By converting this to a forward-looking label, **it can no longer be used as a causal feature or primary signal** in real-time prediction pipelines without introducing look-ahead bias. It is strictly for creating the target `y` for training models.
