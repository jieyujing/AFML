# Triple Barrier Labeler

## Purpose

TBD

## Requirements

### Requirement: Processor Initialization
The TripleBarrierLabeler SHALL be initialized with configurable barrier parameters.

#### Scenario: Initialize with default parameters
- **WHEN** creating a new TripleBarrierLabeler without arguments
- **THEN** the labeler SHALL use pt_sl=[1.0, 1.0], vertical_barrier_bars=12, min_ret=0.001

#### Scenario: Initialize with custom barriers
- **WHEN** creating a TripleBarrierLabeler with pt_sl=[2.0, 1.0] and vertical_barrier_bars=20
- **THEN** the labeler SHALL store these barrier configurations

### Requirement: Fit Method
The fit method SHALL compute volatility-based barrier width from price data.

#### Scenario: Fit calculates volatility
- **WHEN** calling fit() with close prices
- **THEN** the labeler SHALL compute and store volatility estimates

### Requirement: Label Method
The label method SHALL apply triple barrier and generate labels.

#### Scenario: Apply triple barrier
- **WHEN** calling label() with event timestamps
- **THEN** the output SHALL contain columns: t1, trgt, side, ret, label

#### Scenario: Label values are correct
- **WHEN** generating labels
- **THEN** label SHALL be 1 for profit-taking, -1 for stop-loss, 0 for timeout

#### Scenario: Vertical barrier limits holding period
- **WHEN** vertical_barrier_bars=12
- **THEN** no event SHALL exceed 12 bars in holding period

### Requirement: Event Filtering
The labeler SHALL filter events by minimum return threshold.

#### Scenario: Filter low-return events
- **WHEN** min_ret=0.001 and an event has return < 0.001
- **THEN** that event SHALL be excluded from output
