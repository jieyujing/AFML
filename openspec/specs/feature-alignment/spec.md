# feature-alignment Specification

## Purpose
TBD - created by archiving change afml-feature-engineering. Update Purpose after archive.
## Requirements
### Requirement: Event time causal feature alignment
The system SHALL extract and align features computed on the continuous DataFrame exactly to the discrete event times specified by CUSUM filtering, avoiding any forward-looking interpolation.

#### Scenario: Alignment with TBM labels
- **WHEN** the user calls the alignment step providing a generated `full_features_df` and the existing CUSUM event `labels_df` 
- **THEN** the module joins the features sequentially against the label dataframe using exact time `DatetimeIndex` matches, preserving TBM labels (`bin`, `t1`) and weights (`avg_uniqueness`, `return_attribution`).

### Requirement: NaN purging
The system SHALL purge any aligned rows containing NaN entries originating from window or lag initialization in the feature generation process without losing label coherence.

#### Scenario: Dropping initialization lag
- **WHEN** the resulting joined DataFrame contains feature rows evaluated with incomplete data early in the series
- **THEN** the system purges those rows and outputs a clean dataset comprising only valid, non-null features coupled with valid labels.

