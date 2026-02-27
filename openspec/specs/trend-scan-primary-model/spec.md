# Specification: Trend Scan Primary Model

## ADDED Requirements

### Requirement: Backward-looking multiple window T-value scanning
The system SHALL perform ordinary least squares (OLS) regression over a configurable set of historical lookback windows (e.g., `L = [10, 20, ..., 100]`) starting strictly backward from each given event timestamp to avoid lookahead bias.

#### Scenario: Best window selection based on statistical significance
- **WHEN** multiple nested historical price vectors are extracted for an event 
- **THEN** the system calculates the t-statistic for each regression slope and selects the window with the maximal absolute t-value

### Requirement: Highly performant Numba integration
The Trend Scan process SHALL be implemented using pure NumPy arrays and `@njit` compilation to process thousands of event timestamps exponentially faster than pure Python loops.

#### Scenario: Large dataset processing
- **WHEN** evaluating large tick or bar datasets over an extended historical period
- **THEN** the system limits object instantiations and executes O(N×L) mathematical routines inside Numba

### Requirement: Structure of returned signal DataFrames
The Trend Scan module SHALL return a pandas DataFrame aligned with the input `t_events` indices.

#### Scenario: Exporting side and confidence scores
- **WHEN** the module provides its final output
- **THEN** the output contains columns `t1` (the selected optimal window), `t_value` (the maximum absolute t-statistic value in float), and `side` (1 or -1 representing the sign of the t-statistic)
