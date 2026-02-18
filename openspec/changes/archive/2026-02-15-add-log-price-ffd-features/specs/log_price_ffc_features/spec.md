# Specification: Log-Price FFD Features

## ADDED Requirements

### Requirement: Log-Price FFD Features

The system SHALL generate fractional differentiated (FFD) features from logarithmic prices of OHLC data.

#### Scenario: Compute log price for close

- **GIVEN** a DataFrame with `close` column containing positive prices
- **WHEN** `FeatureEngineer` is initialized with `ffd_d > 0`
- **THEN** a new column `log_close_ffd` MUST be added
- **AND** the value MUST equal `log(close)` after FFD transformation with parameter `d`

#### Scenario: Compute log price for open

- **GIVEN** a DataFrame with `open` column containing positive prices
- **WHEN** `FeatureEngineer` is initialized with `ffd_d > 0`
- **THEN** a new column `log_open_ffd` MUST be added
- **AND** the value MUST equal `log(open)` after FFD transformation

#### Scenario: Compute log price for high

- **GIVEN** a DataFrame with `high` column containing positive prices
- **WHEN** `FeatureEngineer` is initialized with `ffd_d > 0`
- **THEN** a new column `log_high_ffd` MUST be added

#### Scenario: Compute log price for low

- **GIVEN** a DataFrame with `low` column containing positive prices
- **WHEN** `FeatureEngineer` is initialized with `ffd_d > 0`
- **THEN** a new column `log_low_ffd` MUST be added

---

# Specification: Cumulative Volume FFD Feature

## ADDED Requirements

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
