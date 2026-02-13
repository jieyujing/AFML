# Polars Triple Barrier Labeler

## ADDED Requirements

### Requirement: Labeler Initialization
The PolarsTripleBarrierLabeler SHALL be initialized with configurable parameters for profit/loss limits and vertical barrier.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new PolarsTripleBarrierLabeler without arguments
- **THEN** the labeler SHALL use pt_sl=[1.0, 1.0] and vertical_barrier_bars=12

#### Scenario: Initialize with custom parameters
- **WHEN** creating a PolarsTripleBarrierLabeler with pt_sl=[2.0, 1.0] and vertical_barrier_bars=24
- **THEN** the labeler SHALL store these values for later use

### Requirement: Fit Method
The fit method SHALL calculate volatility and other parameters from price data using Polars.

#### Scenario: Fit calculates volatility with Polars
- **WHEN** calling fit() with a Polars Series containing close prices
- **THEN** the labeler SHALL compute and store volatility-based thresholds

### Requirement: Label Method
The label method SHALL generate triple barrier labels using Polars operations.

#### Scenario: Generate labels for events
- **WHEN** calling label() with close prices and events
- **THEN** the output SHALL contain labels (-1, 0, 1) for each event

#### Scenario: Labels include vertical barrier
- **WHEN** calling label() with vertical_barrier_bars=12
- **THEN** events without TP/SL triggers SHALL be closed after 12 bars

### Requirement: CUSUM Events
The labeler SHALL support CUSUM events detection using Polars.

#### Scenario: Detect CUSUM events
- **WHEN** calling get_cusum_events() with a threshold
- **THEN** the method SHALL return a Polars Series of CUSUM event indices

### Requirement: Memory Efficiency
The labeler SHALL use Polars' optimized operations to reduce memory footprint.

#### Scenario: Memory usage reduced by 3x
- **WHEN** processing 1M+ price bars
- **THEN** memory consumption SHALL be at least 3x less than pandas equivalent

### Requirement: LazyFrame Support
The labeler SHALL support LazyFrame for large-scale batch processing.

#### Scenario: Process with lazy evaluation
- **WHEN** input is a LazyFrame and lazy=True
- **THEN** operations SHALL be deferred until .collect() is called
