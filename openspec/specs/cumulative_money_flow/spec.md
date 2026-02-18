# cumulative_money_flow Specification

## Purpose
TBD - created by archiving change add-cumulative-money-flow. Update Purpose after archive.
## Requirements
### Requirement: Cumulative Money Flow

The system SHALL compute cumulative money flow from trade data.

#### Scenario: Compute signed amount

- **GIVEN** a trade record with `amount` and `is_buyer_maker` columns
- **WHEN** `is_buyer_maker = False` (buyer initiated trade)
- **THEN** the signed amount MUST equal `+amount` (positive for inflow)

#### Scenario: Compute signed amount for seller

- **GIVEN** a trade record with `amount` and `is_buyer_maker` columns
- **WHEN** `is_buyer_maker = True` (seller initiated trade)
- **THEN** the signed amount MUST equal `-amount` (negative for outflow)

#### Scenario: Compute cumulative money flow

- **GIVEN** a sorted DataFrame of trade records with signed amounts
- **WHEN** the data is processed
- **THEN** a new column `cum_money_flow` MUST be added
- **AND** it MUST equal the cumulative sum of signed amounts

#### Scenario: Field presence in output

- **GIVEN** the raw parquet data after processing
- **WHEN** loading the data
- **THEN** the output MUST contain the column `cum_money_flow`

