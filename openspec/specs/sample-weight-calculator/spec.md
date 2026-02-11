# Sample Weight Calculator

## Purpose

TBD

## Requirements

### Requirement: Processor Initialization
The SampleWeightCalculator SHALL be initialized with configurable decay parameter.

#### Scenario: Initialize with default decay
- **WHEN** creating a new SampleWeightCalculator without arguments
- **THEN** the calculator SHALL use decay=0.9

#### Scenario: Initialize with custom decay
- **WHEN** creating a SampleWeightCalculator with decay=0.95
- **THEN** the calculator SHALL store this decay value

### Requirement: Fit Method
The fit method SHALL compute concurrency and uniqueness metrics.

#### Scenario: Fit calculates concurrency
- **WHEN** calling fit() with labeled events DataFrame
- **THEN** the calculator SHALL compute concurrency for each timestamp

#### Scenario: Fit calculates uniqueness
- **WHEN** calling fit() with labeled events
- **THEN** the calculator SHALL compute average uniqueness for each event

### Requirement: Transform Method
The transform method SHALL add weight columns to the DataFrame.

#### Scenario: Add sample_weight column
- **WHEN** calling transform()
- **THEN** the output DataFrame SHALL contain a 'sample_weight' column

#### Scenario: Add avg_uniqueness column
- **WHEN** calling transform()
- **THEN** the output DataFrame SHALL contain an 'avg_uniqueness' column

### Requirement: Weight Calculation
The weights SHALL combine return attribution with uniqueness.

#### Scenario: Weight formula
- **WHEN** computing weights
- **THEN** weight SHALL equal return * uniqueness * decay_factor

### Requirement: sklearn Compatibility
The calculator SHALL implement fit_transform for sklearn Pipeline compatibility.

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in a sklearn Pipeline
- **THEN** the calculator SHALL work with sample_weight parameter in model.fit()
