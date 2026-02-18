# log_price_ffc_features Specification

## Purpose
TBD - created by archiving change add-log-price-ffd-features. Update Purpose after archive.
## Requirements
### Requirement: Cumulative Volume FFD Feature

The system SHALL generate fractional differentiated feature from cumulative volume.

#### Scenario: Compute cumulative volume

- **GIVEN** a DataFrame with `volume` column
- **WHEN** `FeatureEngineer` is initialized with `ffd_d > 0`
- **THEN** a new column `cum_volume_ffd` MUST be added
- **AND** the value MUST equal `FFD(cumsum(volume))`

#### Scenario: Feature names in output

- **GIVEN** a DataFrame with OHLCV columns
- **WHEN** `FeatureEngineer.fit_transform()` is called
- **THEN** the output MUST contain both existing features AND new features:
  - `log_close_ffd`, `log_open_ffd`, `log_high_ffd`, `log_low_ffd`
  - `cum_volume_ffd`

