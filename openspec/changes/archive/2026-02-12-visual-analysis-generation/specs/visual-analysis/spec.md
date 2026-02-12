## ADDED Requirements

### Requirement: Automated Pipeline Visualization
The pipeline MUST generate a set of standard AFML visualizations for each execution step and save them to a `visual_analysis/` directory.

#### Scenario: Successful Pipeline Execution
- **WHEN** the `src/afml_polars_pipeline.py` is executed with a valid dataset.
- **THEN** it should produce at least 8 visualization artifacts in the `visual_analysis/` folder, covering Bars, Labels, Features, Weights, CV, Meta-Labeling, and Performance.

### Requirement: Specific Plot Types
The following plot types MUST be supported:
1. **Bar Statistics**: Count and frequency of generated bars.
2. **Triple Barrier Sample**: A price plot showing 3-5 events with their TP/SL/Vertical barriers.
3. **Label Distribution**: Balance of 1, 0, -1 labels.
4. **Feature Heatmap**: Correlation between generated features.
5. **Stationarity Analysis**: ADF p-value vs Differentiation `d`.
6. **Purged CV Timeline**: Visual representation of training/test periods with embargoes.
7. **Meta-Model Performance**: ROC Curve and Confusion Matrix.
8. **Equity Curve**: Cumulative returns of the strategy.

#### Scenario: Data Volume Handling
- **WHEN** the dataset is very large (>1M rows).
- **THEN** time-series plots MUST use downsampling or focal windows to remain readable and performant.
