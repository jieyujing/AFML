# Polars Feature Engineer

## ADDED Requirements

### Requirement: Feature Engineer Initialization
The PolarsFeatureEngineer SHALL be initialized with configurable rolling window parameters.

#### Scenario: Initialize with default windows
- **WHEN** creating a new PolarsFeatureEngineer without arguments
- **THEN** the engineer SHALL use windows=[5, 10, 20, 30, 50]

#### Scenario: Initialize with custom windows
- **WHEN** creating a PolarsFeatureEngineer with windows=[10, 20, 40]
- **THEN** the engineer SHALL use these windows for feature computation

### Requirement: Fit Method
The fit method SHALL prepare internal state for feature computation.

#### Scenario: Fit stores metadata
- **WHEN** calling fit() with a Polars DataFrame
- **THEN** the engineer SHALL store column information and parameters

### Requirement: Transform Method
The transform method SHALL generate features using Polars operations.

#### Scenario: Generate Alpha158 features
- **WHEN** calling transform() on price data
- **THEN** the output SHALL contain all Alpha158 benchmark features

#### Scenario: Generate FFD features
- **WHEN** calling transform() with ffd_mode=True
- **THEN** the output SHALL contain Fractionally Differentiated features preserving memory

### Requirement: Rolling Window Operations
The engineer SHALL efficiently compute rolling window features using Polars.

#### Scenario: Rolling statistics computed in parallel
- **WHEN** computing rolling mean, std, min, max
- **THEN** operations SHALL utilize multiple CPU cores via Polars

#### Scenario: Rolling features for multiple windows
- **WHEN** using windows=[5, 10, 20, 30, 50]
- **THEN** features SHALL be computed for all windows efficiently

### Requirement: Memory Efficiency
The engineer SHALL use Polars to minimize memory during feature computation.

#### Scenario: Memory usage reduced by 5x
- **WHEN** processing 100k+ rows with 50+ features
- **THEN** memory consumption SHALL be at least 5x less than pandas equivalent

### Requirement: LazyFrame Support
The engineer SHALL support LazyFrame for large-scale feature computation.

#### Scenario: Lazy feature computation
- **WHEN** input is a LazyFrame and lazy=True
- **THEN** feature computation SHALL be deferred until .collect() is called

### Requirement: sklearn Compatibility
The engineer SHALL implement fit_transform for sklearn Pipeline compatibility.

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in a sklearn Pipeline
- **THEN** the engineer SHALL work with other transformers seamlessly
