# Polars Meta-Labeling Pipeline

## ADDED Requirements

### Requirement: Pipeline Initialization
The PolarsMetaLabelingPipeline SHALL be initialized with primary and meta model configurations.

#### Scenario: Initialize with default models
- **WHEN** creating a new PolarsMetaLabelingPipeline without arguments
- **THEN** the pipeline SHALL use default RandomForest classifiers

#### Scenario: Initialize with custom models
- **WHEN** creating a PolarsMetaLabelingPipeline with primary_model and meta_model
- **THEN** the pipeline SHALL store these model configurations

### Requirement: Fit Method
The fit method SHALL train both primary and meta models using Polars-optimized data.

#### Scenario: Fit primary model
- **WHEN** calling fit() with features and labels
- **THEN** the primary model SHALL be trained first on all labels

#### Scenario: Fit meta model
- **WHEN** primary model is fitted
- **THEN** the meta model SHALL be trained on primary model predictions

### Requirement: Predict Method
The predict method SHALL generate meta-labeled predictions.

#### Scenario: Generate binary predictions
- **WHEN** calling predict() after fitting
- **THEN** the output SHALL contain binary predictions (-1, 1)

#### Scenario: Generate probability predictions
- **WHEN** calling predict_proba() after fitting
- **THEN** the output SHALL contain probability estimates

### Requirement: Memory Efficiency
The pipeline SHALL use Polars for efficient data processing.

#### Scenario: Memory efficient training
- **WHEN** processing 50k+ samples
- **THEN** memory consumption SHALL be reduced compared to pandas equivalent

### Requirement: LazyFrame Support
The pipeline SHALL support LazyFrame for large-scale training.

#### Scenario: Lazy data processing
- **WHEN** input is a LazyFrame and lazy=True
- **THEN** data processing SHALL be deferred until necessary

### Requirement: sklearn Compatibility
The pipeline SHALL be compatible with sklearn interfaces.

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in sklearn Pipeline
- **THEN** the pipeline SHALL integrate seamlessly

#### Scenario: Compatible with sklearn metrics
- **THEN** predictions SHALL work with sklearn metrics for evaluation

### Requirement: Performance Metrics
The pipeline SHALL compute relevant performance metrics.

#### Scenario: Calculate Sharpe ratio
- **WHEN** calling get_sharpe_ratio()
- **THEN** the method SHALL return the annualized Sharpe ratio

#### Scenario: Calculate return metrics
- **WHEN** calling get_return_metrics()
- **THEN** the method SHALL return cumulative return and other metrics
