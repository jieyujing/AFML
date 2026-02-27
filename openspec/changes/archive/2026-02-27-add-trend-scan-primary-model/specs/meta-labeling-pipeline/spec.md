## ADDED Requirements

### Requirement: Meta-Labeling Dependency on Primary `Side`
The Triple-Barrier Method labeling logic MUST enforce the inclusion of the primary `side` (1 for Buy, -1 for Sell) provided by the Trend Scan model to determine the outcome.

#### Scenario: Dynamic Triple-Barrier application
- **WHEN** the primary model indicates a `Buy` or `Sell` side array mapping with events
- **THEN** the Profit Target (Pt) and Stop Loss (Sl) threshold logic conforms to the initial orientation before hitting barriers

### Requirement: Converting Trend Confidence mapped as Sample Weights
The Meta-Labeling system SHALL be capable of ingesting the dynamically generated confidence scores (`t-value` magnitude) from Trend Scan to modify the training sample weights prior to feeding into the Meta-Model classifiers.

#### Scenario: High conviction trend weighting
- **WHEN** the Primary Model exhibits an exceptionally high absolute `t-value` for a specific event
- **THEN** the subsequent sample weight mapping mechanism scales up this event for Meta-Labeling training
