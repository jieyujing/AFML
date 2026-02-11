# Polars Sample Weight Calculator

## ADDED Requirements

### Requirement: Calculator Initialization
The PolarsSampleWeightCalculator SHALL be initialized with configurable decay parameters.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new PolarsSampleWeightCalculator without arguments
- **THEN** the calculator SHALL use decay=0.9

#### Scenario: Initialize with custom parameters
- **WHEN** creating a PolarsSampleWeightCalculator with decay=0.8
- **THEN** the calculator SHALL use this decay value

### Requirement: Fit Method
The fit method SHALL compute uniqueness and concurrency metrics.

#### Scenario: Fit calculates uniqueness
- **WHEN** calling fit() with events and labels
- **THEN** the calculator SHALL compute average uniqueness for each label

### Requirement: Transform Method
The transform method SHALL generate sample weights using Polars.

#### Scenario: Generate weights with decay
- **WHEN** calling fit_transform() with decay=0.9
- **THEN** output weights SHALL apply time decay factor of 0.9

#### Scenario: Generate weights with custom lambda
- **WHEN** calling fit_transform() with lambda_param=0.95
- **THEN** weights SHALL use this lambda for exponential decay

### Requirement: Concurrency Calculation
The calculator SHALL compute sample overlap/concurrency using Polars operations.

#### Scenario: Concurrency computed efficiently
- **WHEN** computing concurrency for overlapping events
- **THEN** operations SHALL use Polars for efficient set operations

### Requirement: Memory Efficiency
The calculator SHALL use Polars to handle large sample sets efficiently.

#### Scenario: Memory usage reduced by 10x
- **WHEN** processing 100k+ samples with concurrency
- **THEN** memory consumption SHALL be at least 10x less than pandas equivalent

### Requirement: LazyFrame Support
The calculator SHALL support LazyFrame for batch weight computation.

#### Scenario: Lazy weight computation
- **WHEN** input is a LazyFrame and lazy=True
- **THEN** weight computation SHALL be deferred until .collect() is called

### Requirement: sklearn Compatibility
The calculator SHALL be compatible with sklearn sample_weight parameter.

#### Scenario: Compatible with sklearn
- **THEN** weights SHALL be returned as numpy array compatible with sklearn estimators
