## ADDED Requirements

### Requirement: Fractional differentiation transformation
The system SHALL provide a utility to transform a non-stationary price series into a stationary fractional differentiated series using the Fixed-Width Window (FFD) method.

#### Scenario: FFD transformation of price features
- **WHEN** the `compute_frac_diff` pipeline is called with parameter $d$ and a truncation threshold $\tau$
- **THEN** the system applies standard binomial weights to the log price series to calculate fractionally differenced values exactly avoiding extending memory beyond $\tau$.

### Requirement: Auto-optimization of difference order
The system SHALL search and identify the lowest possible differencing order $d \in [0, 1]$ that passes the Augmented Dickey-Fuller (ADF) test for stationarity within a user-defined confidence level.

#### Scenario: Search for optimal stationary d
- **WHEN** the user invokes the optimal $d$ search routine on a continuous dollar bar price series and sets a critical value threshold (e.g., 5%)
- **THEN** it iteratively applies FFD increasing $d$ iteratively (e.g., steps of 0.05) and returns the first $d$ that meets the ADF stationarity criteria, retaining maximum history.
