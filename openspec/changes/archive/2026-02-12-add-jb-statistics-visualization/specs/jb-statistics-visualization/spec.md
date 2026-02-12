## ADDED Requirements

### Requirement: JB Statistics Display in Bar Statistics Visualization
The visualization module MUST display Jarque-Bera (JB) statistics alongside bar statistics to validate that Dollar Bars meet AFML normality requirements.

#### Scenario: JB Statistics Computed
- **WHEN** `AFMLVisualizer.plot_bar_stats()` is called with a valid dollar bars DataFrame
- **THEN** the method SHALL compute the following statistics from log returns:
  - Jarque-Bera statistic using `scipy.stats.jarque_bera()`
  - JB p-value for normality testing
  - Skewness using `scipy.stats.skew()`
  - Kurtosis using `scipy.stats.kurtosis(fisher=False)` (Pearson kurtosis, normal=3)

#### Scenario: JB Statistics Displayed
- **WHEN** JB statistics are computed successfully
- **THEN** the method SHALL display these statistics in a formatted table below the existing plots:
  - JB Statistic value (rounded to 2 decimal places)
  - p-value (rounded to 4 decimal places)
  - Skewness (rounded to 4 decimal places)
  - Kurtosis (rounded to 4 decimal places)
  - Normality Conclusion: "Pass" if p-value > 0.05, "Fail" otherwise

#### Scenario: Return JB Statistics as Dictionary
- **WHEN** `plot_bar_stats()` completes
- **THEN** the method SHALL return a dictionary containing:
  ```python
  {
      "jb_stat": float,
      "p_value": float,
      "skewness": float,
      "kurtosis": float,
      "is_normal": bool,
      "n_samples": int
  }
  ```

#### Scenario: Comparison with Time Bars
- **WHEN** an optional `time_bars_df` parameter is provided to `plot_bar_stats()`
- **THEN** the method SHALL compute JB statistics for both dollar bars and time bars
- **THEN** the method SHALL display a comparison table showing:
  | Metric | Dollar Bars | Time Bars |
  |-------|-------------|-----------|
  | JB Stat | X.XX | Y.YY |
  | p-value | 0.XXXX | 0.ZZZZ |
  | Skewness | 0.XXXX | 0.YYYY |
  | Kurtosis | 3.XXXX | 3.YYYY |
  | Normality | Pass/Fail | Pass/Fail |

### Requirement: Visual Conclusion for AFML Compliance
The visualization module MUST provide a clear visual conclusion about whether Dollar Bars meet AFML normality standards.

#### Scenario: AFML Compliance Displayed
- **WHEN** Dollar Bars JB statistics are computed
- **THEN** the method SHALL display an AFML compliance conclusion:
  - "✓ Dollar Bars reduce JB statistic: [dollar JB] < [time JB]" if dollar JB < time JB
  - "✗ Dollar Bars do not improve normality" if dollar JB >= time JB
  - The comparison is only displayed when time bars data is provided

#### Scenario: Statistical Moments Reference
- **WHEN** JB statistics are displayed
- **THEN** the visualization SHALL include a reference showing normal distribution targets:
  - Skewness = 0.0
  - Kurtosis = 3.0 (Pearson definition)
  - JB Statistic = 0 for perfectly normal distribution
