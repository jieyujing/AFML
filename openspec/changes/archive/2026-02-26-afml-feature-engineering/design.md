## Context

Following the standard Advances in Financial Machine Learning (AFML) pipeline, we have successfully generated Dollar Bars, sampled informative events using a dynamic CUSUM filter, and computed Triple Barrier Method (TBM) labels alongside sample weights (uniqueness and return attribution). The next step is Feature Engineering. We need to construct a robust feature matrix $X$ that machine learning models can use to predict the TBM labels $y$, while strictly avoiding lookahead bias and balancing the stationarity-memory trade-off.

## Goals / Non-Goals

**Goals:**
- Design a modular feature extraction pipeline utilizing `afmlkit.feature.kit`.
- Implement Fractional Differentiation with an automated search for the optimal differencing order $d$ (using the Augmented Dickey-Fuller test).
- Implement rolling volatility and basic momentum/structural features (e.g., Log Returns, moving averages).
- Ensure absolutely pure, lookahead-free alignment of the continuous feature matrix $X$ with the discrete sampled event timestamps $t_{events}$.

**Non-Goals:**
- Machine learning model backtesting or Meta-labeling (this will be a subsequent pipeline step).
- Deep learning feature extractors (sticking to standard structural quantitative features first).

## Decisions

1. **Fractional Differentiation Implementation:**
   - *Decision:* Use Fixed-Width Window Fractional Differentiation (FFD) from `afmlkit.feature.core.frac_diff` rather than standard expanding window.
   - *Rationale:* FFD maintains constant memory over time and prevents the weights from being dependent on the length of the data series, making the features strictly consistent.
   - *Alternative:* Standard difference. *Why Rejected:* Completely destroys memory. Expanding window frac diff. *Why Rejected:* The memory weight varies across time, potentially shifting feature distributions.

2. **Optimal $d$ Search (Stationarity vs. Memory):**
   - *Decision:* Implement an optimization step that iterators over $d \in [0, 1]$, applies FFD, and runs an Augmented Dickey-Fuller (ADF) test. It stops at the lowest $d$ where the ADF p-value falls below 5% (or the critical value threshold).
   - *Rationale:* Ensures we reach stationarity while retaining the mathematically maximum possible amount of original price memory.

3. **Computing Framework:**
   - *Decision:* Directly leverage `afmlkit.feature.kit.FeatureKit` or wrap discrete functions from `afmlkit.feature.core` (like `ewms` for volatility).
   - *Rationale:* Reusing established, Numba-accelerated functions guarantees performance and consistency with the rest of the toolkit.

4. **Event Alignment Strategy:**
   - *Decision:* Compute all structural features (moving averages, volatility, frac diff) on the *entire* sequence of Dollar Bars. Only *after* computation, slice the resulting DataFrame using the exact DatetimeIndex of the CUSUM events (`sampled_df.index`).
   - *Rationale:* Preserves continuous moving averages and correctly initializes lag windows. If we sliced first and computed later, the windows would incorrectly aggregate over massive chronological gaps. Avoids lookahead bias because all calculations are strictly causal. 

## Risks / Trade-offs

- **[Risk] Heavy Computational Load for Frac Diff**
  - *Mitigation:* We will truncate the binomial series weights at a predefined threshold (e.g., $1e-4$) to keep the FFD window finite and relatively small, leveraging Numba/vectorized operations where possible.
  
- **[Risk] Index Mismatches during Alignment**
  - *Mitigation:* Explicitly verify that the Dollar Bars index and the `cusum_sampled_bars` index share the exact same format before performing `.loc[]` lookups.

- **[Risk] Memory Loss due to Over-Differencing**
  - *Mitigation:* The optimal $d$ search routine will start from $d=0$ and increment slowly (e.g., steps of 0.1 or 0.05), carefully monitoring the ADF statistic.
