## Context

The AFML pipeline relies on the Meta-Labeling architecture to correctly size positions based on the primary model's confidence. The primary model (Trend Scan) outputs the direction (Long/Short) along with a basic confidence score. However, to correctly map this to position sizing, we need a secondary Meta-Model to learn the probability that the primary model's prediction is correct (`bin = 1` vs `bin = 0`). Because financial data is highly non-IID, featuring overlapping labels, serial correlation, and extreme multicollinearity among technical features, we cannot just plug the data into a standard off-the-shelf classifier. We must implement proper controls like constrained Bagging, Clustered Mean Decrease Accuracy (MDA) for importance ranking, and Purged K-Fold Cross Validation.

## Goals / Non-Goals

**Goals:**
- Construct a standalone script/pipeline for training the Meta-Model based on existing generated feature matrices.
- Implement a `BaggingClassifier` constrained by data uniqueness to mitigate "pseudo-bagging" where correlated samples saturate all weak learners.
- Perform robust feature selection using Clustered MDA.
- Safely cross-validate the model using `PurgedKFold` to remove look-ahead and correlation leakage.
- Output results for the strategy review process, including the MDA results CSV and visualization plots (heatmap, dendrogram, importance metrics, confusion matrix, ROC/PR curves, probability calibration curve, bet size distribution, purged CV fold plot).

**Non-Goals:**
- Modifying the Trend Scan primary model logic or CUSUM filter.
- Live execution or production serving of the trained model (restricted to offline training/evaluation for now).
- Exhaustive hyperparameter tuning (this design focuses on correct architectural implementation first).

## Decisions

1. **Model Architecture: Uniqueness-Constrained Bagging**
   - **Decision**: Avoid applying `RandomForestClassifier` directly. Instead, manually wrap a `DecisionTreeClassifier` configured with `class_weight='balanced'` and `max_features=1` (to avoid masking effects) within a `BaggingClassifier`.
   - **Rationale**: Overlapping events cause the same data to appear in multiple samples. Using `BaggingClassifier(max_samples=avgU_mean)` where `avgU_mean` is the global mean of `avg_uniqueness` ensures that each bagged estimator receives roughly independent data, fighting off tree homogenization and overconfidence bias.
   - **Alternative Considered**: Standard Random Forest or Gradient Boosting. Rejected due to their tendency to overfit noisy, highly correlated labels without uniqueness constraints.

2. **Feature Selection: Clustered MDA**
   - **Decision**: Use `afmlkit.importance` to compute feature distance matrices (`D = sqrt(0.5 * (1 - rho))`), perform Ward linkage clustering, and compute Clustered MDA.
   - **Rationale**: Financial features (e.g., `vol_ewm_10`, `vol_ewm_50`) are highly collinear. Vanilla MDA suffers from the "Substitution Effect". Permuting entire clusters of correlated features provides true importance rankings that aren't undermined by redundant variables.
   
3. **Evaluation Metric: Log-Loss with Sample Weights**
   - **Decision**: Use `neg_log_loss` initialized with `sample_weight=trend_weighted_uniqueness` when scoring MDA.
   - **Rationale**: We care about probability calibration (how precise is our probability estimate) far more than binary accuracy, as the output probability feeds directly into bet sizing algorithms down the line.

4. **Cross-Validation Strategy: Purged K-Fold**
   - **Decision**: Utilize `PurgedKFold(n_splits=5, t1=t1_series, embargo_pct=0.01)`.
   - **Rationale**: Standard K-Fold leaks due to temporal overlap. Purging removes train samples spanning test sets, and Embargo enforces a post-test period freeze to prevent right-to-left correlation spillage.

## Risks / Trade-offs

- **Risk: Excessive Feature Pruning constraints `max_samples` severely** → If `avg_uniqueness` is extremely low, `max_samples` might be too small, causing underfitting in base estimators.
  - **Mitigation**: Monitor the distribution of `avg_uniqueness`. If it's critically low, tune the event-driven sampling threshold upstream (CUSUM filter parameters) before running the Meta-Model training.
- **Risk: Clustered MDA Computational Overheads** → Re-running PurgedKFold on clustered features can be computationally expensive on massive feature matrices.
  - **Mitigation**: Start with moderate `n_jobs` in cross-validation and ensure the feature distance matrix is computed upstream efficiently in `afmlkit.importance.clustering`.
