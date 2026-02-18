## ADDED Requirements

### Requirement: Test fixtures return Polars DataFrames
All test fixtures in tests/test_*.py files SHALL return Polars DataFrame or Polars Series instead of pandas types.

#### Scenario: Sample data fixture returns Polars DataFrame
- **WHEN** a test requests a fixture like `sample_dollar_bars`
- **THEN** the fixture returns a `pl.DataFrame` with correct schema
- **AND** the fixture does NOT return `pd.DataFrame`

#### Scenario: Sample events fixture returns Polars DataFrame
- **WHEN** a test requests a fixture like `sample_events`
- **THEN** the fixture returns a `pl.DataFrame` with columns: datetime, close, t1, label, ret

### Requirement: Processors have expected attributes
All processor classes SHALL have the attributes that tests expect.

#### Scenario: FeatureEngineer has ffd_check_stationarity
- **WHEN** `FeatureEngineer()` is instantiated
- **THEN** `engineer.ffd_check_stationarity` exists as an attribute

#### Scenario: SampleWeightCalculator has concurrency_ and uniqueness_
- **WHEN** `SampleWeightCalculator().fit(events)` is called
- **THEN** `calculator.concurrency_` exists after fit
- **AND** `calculator.uniqueness_` exists after fit

#### Scenario: BetSizer has step_size
- **WHEN** `BetSizer()` is instantiated
- **THEN** `sizer.step_size` exists as an attribute

#### Scenario: MetaLabelingPipeline has n_splits
- **WHEN** `MetaLabelingPipeline()` is instantiated
- **THEN** `pipeline.n_splits` exists as an attribute

### Requirement: Return column names match test expectations
Transform methods SHALL return DataFrames with expected column names.

#### Scenario: SampleWeightCalculator returns sample_weight column
- **WHEN** `calculator.transform(events)` is called
- **THEN** result DataFrame has column named `sample_weight`

#### Scenario: SampleWeightCalculator returns avg_uniqueness column
- **WHEN** `calculator.transform(events)` is called
- **THEN** result DataFrame has column named `avg_uniqueness`

### Requirement: Transform returns Polars types
Transform methods SHALL return Polars DataFrame, not pandas.

#### Scenario: FeatureEngineer transform returns Polars DataFrame
- **WHEN** `engineer.transform(df)` is called
- **THEN** `isinstance(result, pl.DataFrame)` returns True

#### Scenario: TripleBarrierLabeler fit accepts pandas input
- **WHEN** `labeler.fit(pandas_series)` is called with pandas input
- **THEN** the method executes without AttributeError
- **AND** volatility_ is computed correctly
