## ADDED Requirements

### Requirement: Meta-Model Data Ingestion
The meta-model pipeline SHALL consume the features, labels, and event timing records from the upstream processes, utilizing `trend_weighted_uniqueness` as the sample weight for training and evaluation.

#### Scenario: Ingesting training inputs
- **WHEN** the Meta-Model training pipeline starts
- **THEN** it reads the feature matrix dataset, extracting `X` (features), `y` (target `bin`), `timestamp`, `t1`, and `trend_weighted_uniqueness` correctly.

### Requirement: Uniqueness-Constrained Bagging Classifier
The Meta-Model SHALL use a `BaggingClassifier` constrained by data uniqueness to mitigate pseudo-bagging effects caused by overlapping labels in financial data.

#### Scenario: Sub-sample size constraint configuration
- **WHEN** configuring the `BaggingClassifier`
- **THEN** the `max_samples` parameter MUST be set to the mean of the training set's `avg_uniqueness` value.

#### Scenario: Base Estimator parameters
- **WHEN** initializing the base estimator for the Bagging Classifier
- **THEN** it MUST be a `DecisionTreeClassifier` configured with `class_weight='balanced'` and an appropriate feature subsampling strategy set to `max_features=1` to prevent masking effects.

### Requirement: Clustered MDA Feature Selection
The pipeline SHALL employ Clustered Mean Decrease Accuracy (MDA) to evaluate the importance of feature clusters rather than individual features, effectively combating multicollinearity.

#### Scenario: Feature cluster generation
- **WHEN** running feature importance analysis
- **THEN** the pipeline calculates the distance matrix ($D = \sqrt{0.5 \times (1 - \rho)}$) and uses Ward linkage to cluster features.

#### Scenario: MDA evaluation scoring
- **WHEN** assessing the drop in performance from feature cluster permutation
- **THEN** it MUST use `neg_log_loss` as the scoring metric and apply `sample_weight=trend_weighted_uniqueness` to appropriately weight the evaluations.

### Requirement: Cross-Validation with Purging and Embargo
The pipeline SHALL use `PurgedKFold` to prevent look-ahead bias and serial correlation leakage during both feature importance analysis and model evaluation.

#### Scenario: K-Fold initialization
- **WHEN** evaluating the model or running MDA across folds
- **THEN** it MUST partition data using `PurgedKFold` initialized with the event boundaries `t1_series` and an appropriate `embargo_pct` (e.g., 0.01).

### Requirement: Deliverable Generation
The pipeline SHALL generate persistent outputs documenting the model's performance, structural relationships, and feature importance rankings.

#### Scenario: Saving visual and tabular results
- **WHEN** the training and evaluation complete successfully
- **THEN** it MUST save `clustered_mda_results.csv`, and visualization plots including a feature distance heatmap, a feature dendrogram, a clustered MDA importance bar chart, a confusion matrix, ROC/PR curves, a probability calibration curve, a bet size distribution histogram, and a purged CV fold variance plot.
