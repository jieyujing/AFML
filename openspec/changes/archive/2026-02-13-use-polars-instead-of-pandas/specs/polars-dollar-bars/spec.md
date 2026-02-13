# Polars Dollar Bars Processor

## ADDED Requirements

### Requirement: Processor Initialization
The PolarsDollarBarsProcessor SHALL be initialized with configurable parameters for daily target bars and EMA span.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new PolarsDollarBarsProcessor without arguments
- **THEN** the processor SHALL use daily_target=4 and ema_span=20

#### Scenario: Initialize with custom parameters
- **WHEN** creating a new PolarsDollarBarsProcessor with daily_target=8 and ema_span=30
- **THEN** the processor SHALL store these values for later use

### Requirement: Fit Method
The fit method SHALL calculate internal threshold parameters from the input data using Polars LazyFrame optimization.

#### Scenario: Fit calculates threshold with Polars
- **WHEN** calling fit() with a Polars DataFrame containing price data
- **THEN** the processor SHALL compute and store the dollar threshold efficiently using Polars operations

### Requirement: Transform Method
The transform method SHALL generate dollar bars from raw tick data using Polars for memory efficiency.

#### Scenario: Generate fixed dollar bars
- **WHEN** calling fit_transform() with daily_target=4
- **THEN** the output SHALL contain approximately 4 bars per trading day

#### Scenario: Generate dynamic dollar bars
- **WHEN** calling fit_transform() with ema_span=20
- **THEN** the threshold SHALL update daily based on EMA of volume

### Requirement: Lazy Evaluation Support
The processor SHALL support LazyFrame for processing large datasets without loading into memory.

#### Scenario: Use lazy mode for large datasets
- **WHEN** creating a processor with lazy=True and calling fit_transform()
- **THEN** the processor SHALL return a LazyFrame instead of executing immediately

#### Scenario: Collect lazy results
- **WHEN** calling .collect() on the LazyFrame result
- **THEN** the processor SHALL execute the computation and return a DataFrame

### Requirement: sklearn Compatibility
The processor SHALL implement fit_transform for sklearn Pipeline compatibility.

#### Scenario: Compatible with sklearn Pipeline
- **WHEN** used in a sklearn Pipeline
- **THEN** the processor SHALL work with StandardScaler and other transformers

### Requirement: Memory Efficiency
The processor SHALL use Polars' zero-copy operations to minimize memory usage.

#### Scenario: Memory usage reduced by 50%+
- **WHEN** processing a dataset with 1M+ rows
- **THEN** the memory consumption SHALL be at least 50% less than pandas equivalent
