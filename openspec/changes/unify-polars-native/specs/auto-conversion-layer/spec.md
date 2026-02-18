## ADDED Requirements

### Requirement: TripleBarrierLabeler converts pandas to Polars
The fit() method SHALL automatically convert pandas input to Polars.

#### Scenario: fit() receives pandas Series
- **WHEN** `labeler.fit(pandas_series)` is called
- **THEN** the pandas Series is converted to Polars internally
- **AND** `.log()` method is called on Polars Series (not pandas)

#### Scenario: fit() receives pandas DataFrame
- **WHEN** `labeler.fit(pandas_dataframe)` is called
- **THEN** the DataFrame is converted to Polars
- **AND** the 'close' column is extracted as Polars Series

#### Scenario: fit() receives Polars input (no-op)
- **WHEN** `labeler.fit(polars_series)` is called with Polars input
- **THEN** the input is used directly without conversion

### Requirement: All processors handle mixed input types
Processor fit() methods SHALL accept both pandas and Polars types.

#### Scenario: Any processor accepts pandas DataFrame
- **WHEN** `processor.fit(pandas_df)` is called
- **THEN** input is converted to Polars automatically
- **AND** internal processing uses Polars

#### Scenario: Any processor accepts Polars DataFrame (no-op)
- **WHEN** `processor.fit(polars_df)` is called
- **THEN** input is used directly without unnecessary conversion
