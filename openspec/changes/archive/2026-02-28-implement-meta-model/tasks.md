## 1. Data Ingestion Pipeline

- [x] 1.1 Create the python script skeleton for the meta-model pipeline (e.g., `scripts/meta_model_training.py`).
- [x] 1.2 Load the dollar bars feature matrix dataset (`outputs/dollar_bars/feature_matrix.csv`).
- [x] 1.3 Extract feature data `X`, labels `y` (the `bin` column), and temporal data (`timestamp`, `t1`).
- [x] 1.4 Extract or compute the `trend_weighted_uniqueness` to be used for model sample weights.
- [x] 1.5 Handle missing values and ensure feature formats are compatible with scikit-learn tree models.

## 2. Meta-Model Integration (Bagging Classifier)

- [x] 2.1 Initialize the base estimator: `DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_features=1)` (force 1 feature to prevent masking effects).
- [x] 2.2 Calculate `avgU_mean` (the global mean of `avg_uniqueness` across the training dataset).
- [x] 2.3 Initialize the `BaggingClassifier` configured with the base estimator and `max_samples=avgU_mean` to constrain over-sampling.

## 3. Clustered Feature Selection (MDA)

- [x] 3.1 Compute the feature distance matrix based on Pearson correlation ($D = \sqrt{0.5 \times (1 - \rho)}$) via `afmlkit.importance.clustering.cluster_features`.
- [x] 3.2 Execute Ward linkage hierarchical clustering to group collinear features.
- [x] 3.3 Configure the evaluation metric as `neg_log_loss`, incorporating `sample_weight=trend_weighted_uniqueness`.
- [x] 3.4 Integrate and run `afmlkit.importance.mda.clustered_mda` to compute the importance scores for each cluster.

## 4. Cross-Validation (PurgedKFold)

- [x] 4.1 Initialize `PurgedKFold` taking `t1_series` and applying `embargo_pct=0.01`.
- [x] 4.2 Wrap the training and Clustered MDA evaluation inside the `PurgedKFold` cross-validation loops to ensure strict temporal separation.

## 5. Artifact Generation & Deliverables

- [x] 5.1 Save the result of the MDA evaluation to `outputs/models/clustered_mda_results.csv`.
- [x] 5.2 Generate and save the Feature Distance Heatmap plot (`outputs/models/feature_distance_heatmap.png`).
- [x] 5.3 Generate and save the Hierarchical Clustering Dendrogram plot (`outputs/models/feature_dendrogram.png`).
- [x] 5.4 Generate and save the Clustered MDA Importance Bar Chart (`outputs/models/clustered_mda_importance.png`).
- [x] 5.5 Generate and save the Meta-Model Confusion Matrix plot (`outputs/models/confusion_matrix.png`).
- [x] 5.6 Generate and save the ROC Curve and Precision-Recall Curve plots (`outputs/models/roc_pr_curves.png`).
- [x] 5.7 Generate and save the Reliability / Probability Calibration Curve plot (`outputs/models/probability_calibration.png`).
- [x] 5.8 Generate and save the Bet Size (Position) Distribution Histogram (`outputs/models/bet_size_distribution.png`).
- [x] 5.9 Generate and save the Purged CV Fold Variance Plot (`outputs/models/purged_cv_scores.png`).
- [x] 5.10 Optional: Serialize and save the final trained `BaggingClassifier` model (`outputs/models/meta_model.pkl`).
