# Specification: Dollar Imbalance Feature

## ADDED Requirements

### Requirement: Dollar Imbalance per Bar

The system SHALL compute dollar imbalance for each dollar bar.

#### Scenario: Aggregate buyer amount

- **GIVEN** a group of trades forming a dollar bar
- **WHEN** the bar contains trades where `is_buyer_maker = False` (buyer initiated)
- **THEN** the column `buyer_amount` MUST equal the sum of amounts for buyer-initiated trades

#### Scenario: Aggregate seller amount

- **GIVEN** a group of trades forming a dollar bar
- **WHEN** the bar contains trades where `is_buyer_maker = True` (seller initiated)
- **THEN** the column `seller_amount` MUST equal the sum of amounts for seller-initiated trades

#### Scenario: Compute dollar imbalance

- **GIVEN** a dollar bar with `buyer_amount` and `seller_amount`
- **THEN** the column `dollar_imbalance` MUST equal `buyer_amount - seller_amount`
- **AND** positive values indicate net buying pressure
- **AND** negative values indicate net selling pressure

#### Scenario: Fields present in output

- **GIVEN** the dollar bars after processing
- **WHEN** the input data contains `is_buyer_maker` column
- **THEN** the output MUST contain: `buyer_amount`, `seller_amount`, `dollar_imbalance`

#### Scenario: Graceful handling when is_buyer_maker not present

- **GIVEN** the dollar bars after processing
- **WHEN** the input data does NOT contain `is_buyer_maker` column
- **THEN** the new imbalance columns MUST be omitted from output (backward compatible)
