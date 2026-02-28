---
name: afml-quant-factory
description: A comprehensive framework for developing, testing, and verifying financial machine learning strategies based on the principles of Advances in Financial Machine Learning (AFML) and Machine Learning for Asset Managers (MLAM). Use this skill when the user wants to develop quantitative strategies, process financial data, check stationarity, label data with triple-barrier method, perform feature importance analysis, or conduct backtesting with Deflated Sharpe Ratio (DSR) verification. It enforces rigorous statistical validation to avoid false discoveries and overfitting.
---

# AFML Quant Factory

This skill provides a standardized, industrial-grade workflow for quantitative finance strategy development, following the methodologies of Marcos López de Prado.

## Core Philosophy

1.  **Stationarity is Non-Negotiable**: Financial data must be stationary (memory-preserving FracDiff) before any modeling.
2.  **No Peeking**: Strict prevention of look-ahead bias through Purged K-Fold Cross-Validation.
3.  **Honest Backtesting**: Use Deflated Sharpe Ratio (DSR) to account for multiple testing and selection bias.
4.  **Meta-Labeling**: Separate the decision of "side" (long/short) from "size" (bet sizing).

## Workflow Decision Tree

### Phase 1: Data Engineering

**Goal**: Transform raw market data into information-rich, stationary features.

1.  **Sampling**:
    *   **Check**: Are you using Time Bars?
    *   **Action**: If YES -> STOP. Switch to **Dollar Bars** or **Volume Bars** to recover normality and synchronize with information flow.
    *   **Event Trigger**: Apply **CUSUM Filter** (Cumulative Sum) to trigger sampling. Only record data when price deviation hits a threshold to capture "significant events" and reduce noise.
    *   **Frequency Selection (Best Practice)**: When evaluating Dollar Bars (IID test):
        *   **Independence First**: Prioritize **Low Autocorrelation** over the Jarque-Bera (JB) absolute value. Independent samples (AC1 ≈ 0) are critical for preventing CV leakage.
        *   **Sample Size Sensitivity**: JB statistics scale with $N$. A larger $N$ (more bars) naturally leads to higher JB. Don't let high JB scare you away from high-frequency bars if the autocorrelation is near-zero and labels look balanced.
        *   **Optimal Frequency**: Often around 20-50 bars/day for high-volume assets (e.g., BTCUSDT). Aim for a balance: enough samples for ML training vs. preserving IID properties.
2.  **Stationarity**:
    *   **Check**: Run Augmented Dickey-Fuller (ADF) test. Is p-value < 0.05?
    *   **Action**: If NO -> Apply **Fractional Differentiation (FracDiff)**. Find minimum `d` such that p < 0.05 while maximizing memory preservation. *Never use integer differencing (d=1).*
    *   **🚨 Execution Pitfall (Variance Collapse)**: FracDiff inherently compresses the absolute variance of the series (方差坍缩). If you apply a standard log-returns-based CUSUM filter immediately *after* FracDiff without standardizing the threshold, you will suffer from "Scale Mismatch" (量纲错位). The threshold (elephant) will be far too large for the compressed FracDiff diffs (ants), leading to near-zero event sampling.
    *   **Solution**: Compute the CUSUM diffs identically to the scale of the threshold. If your dynamic threshold is the EWMS of log returns, ensure your input to CUSUM represents the unit-matched percent change space. Do not hardcode `np.log()` inside low-level CUSUM algorithms; pass pre-calculated diffs exactly matching the threshold scale.

### Phase 1.5: Primary Model (Side Determination)

**Goal**: Determine the directional bias (Long/Short) for each CUSUM event using a statistically rigorous, parameter-free method.

1.  **Method**: **Trend Scanning** (MLAM Ch.3.5).
    *   For each CUSUM event at time $t_i$, backward-scan over multiple window lengths $L = [L_1, L_2, \ldots, L_k]$.
    *   Fit OLS regression $y = a + bx$ in each window, compute the slope's t-statistic.
    *   Select the window $L^*$ that maximizes $|t_{\text{value}}|$ — the most statistically significant trend.
    *   **Output**: `side = sign(t_value)` (+1 = Long, -1 = Short) and `|t_value|` as confidence score.
2.  **Why Trend Scan over Fixed-Window Momentum?**
    *   **No hyperparameter sensitivity**: No fixed look-back period to tune. The algorithm self-selects the optimal window.
    *   **White-box logic**: Pure OLS — fully interpretable, no black-box model risk.
    *   **Dual output**: Side + confidence ($|t_{\text{value}}|$) feeds directly into Meta-Labeling sample weights.
3.  **Implementation**: `afmlkit.feature.core.trend_scan.trend_scan_labels()`.
    *   **Performance**: Numba `@njit` compiled, pure math OLS (no `statsmodels`). Handles 4700+ events × 10 windows in <1 second.
    *   **No look-ahead bias**: Strictly backward-looking windows. Verified by tests.
    *   **Zero-variance guard**: Flat price segments → `t_value=0, side=0` (filtered out before Meta-Labeling).
4.  **🚨 Anti-Pattern**: Do NOT use dual moving average crossover, fixed-period momentum, or any method requiring a manually chosen look-back window as your Primary Model. These exhibit catastrophic parameter fragility in volatile markets.

### Phase 2: Labeling & Weighting

**Goal**: Scientifically define "success" and handle data overlap.

1.  **Labeling**:
    *   **Method**: Use **Triple-Barrier Method**.
    *   **Configuration**: Set upper barrier (take profit), lower barrier (stop loss), and vertical barrier (time expiration). First touch determines the label.
2.  **Sample Weights**:
    *   **Check**: Are labels overlapping (common in finance)?
    *   **Action**: Calculate **Average Uniqueness**. Down-weight samples with low uniqueness to prevent the model from overfitting to redundant regimes.

### Phase 3: Modeling & Feature Selection

**Goal**: Train models that generalize, not memorize.

1.  **Cross-Validation**:
    *   **Method**: **Purged K-Fold CV**.
    *   **Action**: Enforce an **Embargo** period between training and test sets to eliminate leakage from serial correlation.
2.  **Feature Importance**:
    *   **Method**: **Clustered MDA (Mean Decrease Accuracy/Log-loss)**.
    *   **Why Not Vanilla MDA?**: Financial features exhibit extreme multicollinearity (e.g., `vol_ewm_10`, `vol_ewm_50`, `vol_ewm_100` are highly correlated). Vanilla MDA suffers from the **Substitution Effect** — important features get low scores because correlated features "substitute" for them. Clustered MDA groups correlated features into clusters and permutes entire clusters together.
    *   **Step 1 — Feature Clustering**: Compute distance matrix $D = \sqrt{0.5 \times (1 - \rho)}$ → Hierarchical clustering (Ward linkage) → Auto-select optimal k via Silhouette Score.
    *   **Step 2 — Clustered MDA**: For each cluster, permute all its features together on the OOS fold, measure drop in negative log-loss.
    *   **Scoring**: Use **Log-loss with `sample_weight`** (not accuracy). Probability calibration quality matters for downstream Bet Sizing / Meta-labeling.
    *   **Action**: Discard clusters with MDA ≤ 0 (permuting them *improves* the model — they inject noise).
    *   **🚨 Execution Pitfall (The Masking Effect)**: When using tree-based models (Random Forest or Bagging) for financial feature importance, NEVER use `max_features='sqrt'` or `'auto'`. **Force `max_features=1`**. This forces the trees to evaluate "weak" features independently, preventing dominant features (like market momentum) from "masking" or drowning out subtle but valuable signals (like volatility or microstructure regimes).
    *   **Implementation**: `afmlkit.importance.clustering.cluster_features()` + `afmlkit.importance.mda.clustered_mda()`. See `scripts/feature_importance_analysis.py` for end-to-end pipeline.
    *   **🔍 Practical Insight**: In BTCUSDT Dollar Bar analysis, `log_return` + `corr_pv_10` cluster dominated importance (0.2047), while momentum/trend indicators were secondary (0.0286). ATR/EMA price-level features had *negative* importance — indicating they added noise. This validates the AFML principle that raw returns and microstructure features often outperform traditional technical indicators.

### Phase 4: Strategy & Verification

**Goal**: Deploy only strategies that are statistically significant.

1.  **Strategy Structure**:
    *   **Architecture**: **Meta-Labeling**.
        *   *Primary Model*: **Trend Scan** — determines side (Long/Short) with self-adaptive window selection. Outputs `side` and `|t_value|` confidence score. See Phase 1.5.
        *   *Secondary (Meta) Model*: Determines confidence (Probability of Primary Model being correct). Uses `|t_value|` as sample weight multiplier.
    *   **Pipeline**: CUSUM events → `trend_scan_labels()` → inject `side` into `TBMLabel(is_meta=True)` → normalize `|t_value|` × `avg_uniqueness` → Meta-Model training.
    *   **Implementation**: See `scripts/cusum_filtering.py` for the complete end-to-end pipeline and `scripts/meta_model_training.py` for model training details.
    *   **Bet Sizing / Risk Controls**: 
        *   **Probability Calibration**: Use **Reliability Curves** (Calibration Curve) to verify that the Meta-Model's predicted probability $P$ accurately reflects the true win rate. A "self-aware" model is critical for Kelly-style sizing.
        *   **Bet Size Distribution**: Monitor the histogram of bet sizes. A healthy Meta-Model should ideally show a high concentration near 0 (filtering out noise) and selective high-conviction peaks, rather than constant "all-in" signals.
        *   **Fold Variance**: Monitor performance across **Purged CV Folds**. High variance in ROC-AUC or Log-loss across folds suggests non-stationarity or severe overfitting to specific regimes.
    *   **Allocation**: For multi-asset portfolios, use **Hierarchical Risk Parity (HRP)** instead of Mean-Variance (MV). HRP exploits graph theory and clustering, avoids covariance inversion, and is robust to noise.
2.  **Final Acceptance**:
    *   **Metric**: **Deflated Sharpe Ratio (DSR)**.
    *   **Threshold**: DSR Probability > 0.95.
    *   **Input**: Track number of trials, return skewness/kurtosis, and track record length. *Reject strategies with high Sharpe but low DSR (lucky outcomes).*

## References

*   **Concepts**: See `references/glossary.md` for definitions of AFML terms (FracDiff, Triple-Barrier, etc.).
*   **Implementation**: See `references/implementation_guide.md` for Python code snippets and library usage (MlFinLab).
