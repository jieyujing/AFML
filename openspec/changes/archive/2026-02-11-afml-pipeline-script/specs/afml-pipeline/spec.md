## ADDED Requirements

### Requirement: Stationarity Enforcement
The pipeline MUST ensure that all input time series are stationary before feature engineering or modeling. This is achieved by finding the minimum fractional differentiation order $d$ such that the series passes the Augmented Dickey-Fuller (ADF) test with a p-value < 0.05.

#### Scenario: Optimal d found
- **WHEN** the p-value of the ADF test for a fractional differentiation order `d` (e.g., 0.4) is < 0.05
- **THEN** the pipeline uses this `d` for feature generation and logs it.

#### Scenario: No d found
- **WHEN** no fractional `d` < 1.0 yields a p-value < 0.05
- **THEN** the pipeline defaults to integer differencing (`d=1.0`) and issues a warning.

### Requirement: Deflated Sharpe Ratio (DSR) Verification
The pipeline MUST compute the Deflated Sharpe Ratio calculation to validate the strategy's performance, accounting for non-normality and track record length.

#### Scenario: Strategy Validation
- **WHEN** the backtest is complete
- **THEN** calculate the Probabilistic Sharpe Ratio (PSR) using the observed Sharpe Ratio, skewness, kurtosis, and track record length.
- **THEN** report the probability that the true Sharpe Ratio is positive (PSR > 0.95?).
- **THEN** if possible, estimate DSR based on the number of trials (default to 1 trial if not available).

### Requirement: Pipeline Orchestration
The main script `src/afml_polars_pipeline.py` MUST provide a CLI interface to execute the full AFML workflow end-to-end, accepting data file path and parameters.

#### Scenario: Single File Execution
- **WHEN** the user runs `python src/afml_polars_pipeline.py data.csv`
- **THEN** the script executes all steps: Load -> Stationarity Check -> Dollar Bars -> Labeling -> Weights -> Features -> CV -> Meta-Labeling -> Bet Sizing -> DSR.
- **THEN** it saves intermediate artifacts to `data/` and prints a summary report.
