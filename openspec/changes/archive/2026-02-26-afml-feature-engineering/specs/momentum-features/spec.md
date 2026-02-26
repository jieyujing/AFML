## ADDED Requirements

### Requirement: Momentum metrics calculation
The system SHALL compute structural event-based momentum metrics, including consecutive returns, moving averages (SMA/EMA) crossovers, and relative strength metrics based purely on Dollar Bars.

#### Scenario: Appending MACD or EMA crossover distances
- **WHEN** the `FeatureKit` is requested to add basic technical features
- **THEN** it generates moving average fields (e.g., Short EMA, Long EMA) and evaluates the logarithmic return distance between current price and mean moving averages.
