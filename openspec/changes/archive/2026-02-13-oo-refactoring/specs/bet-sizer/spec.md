# Bet Sizer

## ADDED Requirements

### Requirement: Processor Initialization
The BetSizer SHALL be initialized with configurable step size.

#### Scenario: Initialize with default step size
- **WHEN** creating a new BetSizer without arguments
- **THEN** the sizer SHALL use step_size=0.0 (continuous sizing)

#### Scenario: Initialize with discrete step size
- **WHEN** creating a BetSizer with step_size=0.25
- **THEN** the sizer SHALL discretize bet sizes to 0.25 increments

### Requirement: Calculate Method
The calculate method SHALL compute bet sizes from probabilities.

#### Scenario: Calculate bet size from probability
- **WHEN** calling calculate() with probability series
- **THEN** the output SHALL be a Series of bet sizes between 0 and 1

#### Scenario: Apply sign from predictions
- **WHEN** pred_series is provided
- **THEN** the output SHALL include direction (-1, 0, 1)

### Requirement: Probability Mapping
The sizer SHALL map probabilities to bet sizes using CDF.

#### Scenario: Probability mapping formula
- **WHEN** computing bet size
- **THEN** the sizer SHALL use: size = 2 * CDF(z) - 1 where z = (p - 0.5) / sqrt(p * (1-p))

#### Scenario: Clip negative sizes to zero
- **WHEN** computed size is negative
- **THEN** the sizer SHALL set that size to 0

### Requirement: Active Signal Averaging
The sizer SHALL average concurrent signals when average_active=True.

#### Scenario: Average concurrent signals
- **WHEN** average_active=True and events overlap
- **THEN** the output SHALL be scaled by average uniqueness

### Requirement: sklearn Compatibility
The sizer SHALL be compatible with sklearn sample_weight parameter.

#### Scenario: Compatible with model.fit()
- **WHEN** using bet sizes as sample weights
- **THEN** the sizes SHALL directly pass to model.fit(sample_weight=sizes)
