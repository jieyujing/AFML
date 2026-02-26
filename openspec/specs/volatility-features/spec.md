# volatility-features Specification

## Purpose
TBD - created by archiving change afml-feature-engineering. Update Purpose after archive.
## Requirements
### Requirement: Rolling volatility feature generation
The system SHALL provide a method to compute rolling and exponential moving volatility measurements over configurable observation windows, ensuring computations act on continuous bar properties.

#### Scenario: Continuous volatility pipeline
- **WHEN** the `FeatureKit` is requested to compute volatility features for specific windows (e.g., spans of [10, 50, 100])
- **THEN** an exponentially weighted standard deviation (EWMS) of log returns over these spans is attached to the Dollar Bars DataFrame as new columns without nan lookahead values.

