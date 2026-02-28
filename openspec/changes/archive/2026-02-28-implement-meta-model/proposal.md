## Why
The AFML pipeline needs a Meta Model (Secondary Model) to evaluate the probability of the primary model's (Trend Scan) trading signals being correct. This step is crucial in the Meta-Labeling architecture to separate the decision of "side" (long/short) from "size" (bet sizing). Implementing this completes the core machine learning pipeline, enabling statistically rigorous strategy evaluation and position sizing.

## What Changes
- Implement a custom bagging ensemble (`BaggingClassifier` with `DecisionTreeClassifier` as base estimator) designed for financial data, setting `max_samples` to the mean of `avg_uniqueness` to prevent over-sampling of overlapping labels.
- Construct the data input pipeline to consume the feature matrix (`outputs/dollar_bars/feature_matrix.csv`), target labels (`bin`), timing data (`timestamp`, `t1`), and sample weights (`trend_weighted_uniqueness`).
- Integrate Clustered Mean Decrease Accuracy (MDA) for robust feature selection, scoring with `neg_log_loss` initialized with sample weights.
- Integrate `PurgedKFold` cross-validation (with embargo) into the feature importance and model evaluation process to prevent look-ahead bias and serial correlation leakage.
- Generate standard output deliverables including the trained model, MDA results CSV, and visualization plots (heatmap, dendrogram, importance bar chart, confusion matrix, ROC/PR curves, probability calibration curve, bet size distribution, purged CV fold plot).

## Capabilities

### New Capabilities
- `meta-model-pipeline`: The complete secondary model training, feature selection (Clustered MDA), and evaluation (PurgedKFold) workflow.

### Modified Capabilities

## Impact
- **Strategy Pipeline**: Completes the end-to-end workflow by adding the Meta-Model layer on top of the existing Trend Scan primary model.
- **afmlkit Integration**: Deeply integrates with existing `afmlkit.importance` (clustering, MDA) and cross-validation components.
- **Artifacts**: Will generate new persistent outputs (`clustered_mda_results.csv` and visualization images) for strategy verification.
