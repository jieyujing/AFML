# Feature Engineer

## Purpose

TBD

## Requirements

### Requirement: Processor Initialization
The FeatureEngineer SHALL be initialized with configurable window parameters.

#### Scenario: Initialize with default windows
- **WHEN** creating a new FeatureEngineer without arguments
- **THEN** the engineer SHALL use windows=[5, 10, 20, 30, 50]

#### Scenario: Initialize with custom windows
- **WHEN** creating a FeatureEngineer with windows=[10, 20, 40]
- **THEN** the engineer SHALL use these custom windows for all calculations

### Requirement: Fit Method
The fit method SHALL optionally perform feature selection based on importance.

#### Scenario: Fit without feature selection
- **WHEN** calling fit() with labels
- **THEN** the engineer SHALL compute feature statistics without filtering

#### Scenario: Fit with feature selection
- **WHEN** calling fit() with labels and a minimum importance threshold
- **THEN** the engineer SHALL store a list of selected feature names

### Requirement: Transform Method
The transform method SHALL generate all configured features.

#### Scenario: Generate Alpha158 features
- **WHEN** calling transform()
- **THEN** the output SHALL contain ROC and returns features

#### Scenario: Generate FFD features
- **WHEN** calling transform()
- **THEN** the output SHALL contain fractionally differentiated features

#### Scenario: Generate market regime features
- **WHEN** calling transform()
- **THEN** the output SHALL contain volatility, serial correlation, and entropy features

### Requirement: sklearn Compatibility
The engineer SHALL implement fit_transform for sklearn Pipeline compatibility.

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in a sklearn Pipeline
- **THEN** the engineer SHALL work with cross-validation splits

### Requirement: Feature Names
The engineer SHALL provide a method to get feature names.

#### Scenario: Get feature names
- **WHEN** calling get_feature_names()
- **THEN** the output SHALL return a list of all generated feature column names
