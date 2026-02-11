# Polars Purged K-Fold Cross Validation

## ADDED Requirements

### Requirement: CV Initialization
The PolarsPurgedKFoldCV SHALL be initialized with configurable cross-validation parameters.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new PolarsPurgedKFoldCV without arguments
- **THEN** the CV SHALL use n_splits=5, embargo=0.01

#### Scenario: Initialize with custom parameters
- **WHEN** creating a PolarsPurgedKFoldCV with n_splits=10, embargo=0.02
- **THEN** the CV SHALL use these values for splits and embargo

### Requirement: Split Method
The split method SHALL generate train/test indices while purging overlaps.

#### Scenario: Generate purged splits
- **WHEN** calling split() with features and labels
- **THEN** the output SHALL contain train/test index pairs with purged overlaps

#### Scenario: Apply embargo period
- **WHEN** calling split() with embargo=0.01
- **THEN** test samples at the end SHALL be excluded from training

### Requirement: Memory Efficiency
The CV SHALL use Polars for efficient index computation.

#### Scenario: Memory efficient split generation
- **WHEN** generating splits for 100k+ samples
- **THEN** memory consumption SHALL be significantly less than pandas equivalent

### Requirement: Polars Integration
The CV SHALL accept and return Polars DataFrames and Series.

#### Scenario: Accept Polars DataFrame
- **WHEN** input is a Polars DataFrame
- **THEN** split indices SHALL be compatible with Polars indexing

#### Scenario: Return compatible indices
- **WHEN** generating splits
- **THEN** indices SHALL work with Polars DataFrame/Series for filtering

### Requirement: sklearn Compatibility
The CV SHALL be compatible with sklearn's cross-validation interface.

#### Scenario: Compatible with cross_val_score
- **WHEN** used with sklearn's cross_val_score
- **THEN** the CV SHALL work seamlessly

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in sklearn Pipeline
- **THEN** the CV SHALL integrate properly

### Requirement: No Information Leakage
The CV SHALL prevent data leakage through purging and embargo.

#### Scenario: No overlap between train and test
- **WHEN** generating splits
- **THEN** train indices SHALL NOT overlap with test indices in time

#### Scenario: Embargo prevents leakage
- **WHEN** embargo=0.01
- **THEN** samples within embargo period after test set SHALL NOT be in training set
