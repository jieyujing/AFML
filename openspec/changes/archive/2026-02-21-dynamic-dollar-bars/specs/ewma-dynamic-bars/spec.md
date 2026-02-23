## ADDED Requirements

### Requirement: Dynamic Dollar Bar Generation MUST scale dynamically

The system MUST be able to generate dollar bars using an Exponentially Weighted Moving Average (EWMA) dynamic threshold based on historical daily dollar volumes, calibrated to target a specific daily frequency.

#### Scenario: Generate Bars for Target Frequencies
- **WHEN** the user provides raw trade data and a list of target frequencies (e.g., 4, 6, 10, 20, 50 bars per day)
- **THEN** the script calculates the EWMA dynamic threshold for each target frequency
- **AND** generates the corresponding Dollar Bars matching those dynamic threshold constraints.

### Requirement: Statistical Evaluation MUST perform tests

The generated bars MUST be evaluated using statistical tests to determine the "best" bar frequency, specifically aiming to recover normality and reduce autocorrelation.

#### Scenario: Select the Optimal Bar Frequency
- **WHEN** multiple sets of Dollar Bars are generated for different target frequencies
- **THEN** the system calculates the Jarque-Bera (JB) test statistic and first-order autocorrelation for each set's log returns
- **AND** selects the frequency that produces the lowest JB statistic (closest to normal distribution) and lowest autocorrelation.

### Requirement: System MUST plot the results

The selection process and the resulting optimal bars MUST be accompanied by clear data visualizations.

#### Scenario: Plot Optimal Dollar Bars
- **WHEN** the optimal Dollar Bar set is identified
- **THEN** the system generates a visualization showing the comparison of JB test scores across frequencies
- **AND** plots the time series of the optimal Dollar Bars' prices alongside the dynamic EWMA threshold to illustrate how the sampling adapts over time.
