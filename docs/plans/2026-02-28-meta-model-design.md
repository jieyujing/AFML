# AFML Meta Model Implementation Plan

## 1. Overview
Implement the Meta Model (Secondary Model) for the AFML pipeline. This model takes features and the primary model's directions (Trend Scan) to predict the probability of the primary model's trading signal being correct.

## 2. Core Architecture: Bagging Classifier
Instead of a standard `RandomForestClassifier`, we will implement a custom bagging ensemble to correctly handle non-IID financial data:
- **Base Estimator**: `DecisionTreeClassifier(criterion='entropy', max_features='auto', class_weight='balanced')`
- **Ensemble Wrapper**: `BaggingClassifier(base_estimator=clf, n_estimators=1000, max_samples=avgU_mean)`
- **`max_samples` Constraints**: The `max_samples` will be set to the mean of `avg_uniqueness` across the training set to prevent "pseudo-bagging" and tree homogenization caused by overlapping labels safely. 

## 3. Data Inputs
- Feature Matrix: `outputs/dollar_bars/feature_matrix.csv`
- Essential Columns:
  - Input Features `X`: `['log_return', 'vol_ewm_10', 'vol_ewm_50', 'vol_ewm_100', 'vol_parkinson', 'vol_atr_14', 'vol_bb_pct_b_20', 'trend_variance_ratio_20', 'ema_short', 'ema_long', 'log_dist_ema_short', 'log_dist_ema_long', 'ema_diff', 'rsi_14', 'mom_roc_10', 'mom_stoch_k_14', 'corr_pv_10', 'liq_amihud', 'vol_rel_20', 'ffd_log_price']` 
  - Target `y`: `bin` (Meta-label: 0 or 1)
  - CV Timing: `timestamp` (as index) & `t1` (event touch time)
  - Weights: `trend_weighted_uniqueness` (Combination of avg_uniqueness and trend_scan confidence).

## 4. Feature Selection & Evaluation (Clustered MDA)
- Use `afmlkit.importance.clustering.cluster_features` to compute distance matrices ($D = \sqrt{0.5 \times (1 - \rho)}$) and perform hierarchical clustering (Ward linkage).
- Evaluate feature cluster importance using `afmlkit.importance.mda.clustered_mda`.
- Scoring Metric: `neg_log_loss` initialized with `sample_weight=trend_weighted_uniqueness`.

## 5. Cross Validation (PurgedKFold)
- Evaluate and process the MDA with `PurgedKFold(n_splits=5, t1=t1_series, embargo_pct=0.01)`.
- Purging removes training samples overlapping test evaluations.
- Embargo adds a gap period to prevent serial correlation spillovers.

## 6. Output Delivery
- Trained Model persistence (if required)
- `clustered_mda_results.csv` detailing MDA stats.
- Visualizations:
  1. `feature_distance_heatmap.png`
  2. `feature_dendrogram.png`
  3. `clustered_mda_importance.png`
