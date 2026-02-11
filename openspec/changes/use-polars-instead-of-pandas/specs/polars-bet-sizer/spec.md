# Polars Bet Sizer

## ADDED Requirements

### Requirement: Bet Sizer Initialization
The PolarsBetSizer SHALL be initialized with configurable bet sizing parameters.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new PolarsBetSizer without arguments
- **THEN** the sizer SHALL use default probability threshold of 0.5

#### Scenario: Initialize with custom parameters
- **WHEN** creating a PolarsBetSizer with threshold=0.6, quantity=100
- **THEN** the sizer SHALL store these values for position sizing

### Requirement: Bet Size from Probability
The sizer SHALL calculate bet sizes based on predicted probabilities.

#### Scenario: Size bets using EDF
- **WHEN** calling bet_size_probability() with predicted probabilities
- **THEN** the output SHALL contain bet sizes derived from the Empirical Distribution Function

#### Scenario: Apply dynamic scaling
- **WHEN** calling bet_size_probability() with max_size parameter
- **THEN** bet sizes SHALL be capped at max_size

### Requirement: Get Bet Size Method
The sizer SHALL provide bet size recommendations.

#### Scenario: Get discrete bet size
- **WHEN** calling get_bet_size() with a continuous size
- **THEN** the output SHALL be discretized to the nearest tradable quantity

#### Scenario: Handle insufficient capital
- **WHEN** requesting size exceeds available capital
- **THEN** the sizer SHALL return the maximum affordable position

### Requirement: Memory Efficiency
The sizer SHALL use Polars for efficient probability operations.

#### Scenario: Memory efficient EDF computation
- **WHEN** computing EDF for 100k+ probabilities
- **THEN** memory consumption SHALL be reduced compared to pandas equivalent

### Requirement: LazyFrame Support
The sizer SHALL support LazyFrame for batch position sizing.

#### Scenario: Lazy bet sizing
- **WHEN** input is a LazyFrame and lazy=True
- **THEN** bet size computation SHALL be deferred until .collect() is called

### Requirement: sklearn Compatibility
The sizer SHALL return numpy arrays compatible with sklearn workflows.

#### Scenario: Compatible with sklearn
- **WHEN** used in sklearn Pipeline
- **THEN** the sizer SHALL work with other transformers

#### Scenario: Compatible with model predictions
- **THEN** input SHALL accept numpy arrays from sklearn model predictions

### Requirement: Accuracy Scoring
The sizer SHALL compute accuracy metrics for bet sizing strategy.

#### Scenario: Calculate accuracy score
- **WHEN** calling score_accuracy() with predictions and actuals
- **THEN** the output SHALL return the accuracy of the sizing strategy

#### Scenario: Calculate custom metrics
- **WHEN** calling get_metrics() with predictions and actuals
- **THEN** the output SHALL return multiple performance metrics
