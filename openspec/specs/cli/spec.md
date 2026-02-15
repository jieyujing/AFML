# cli Specification

## Purpose
TBD - created by archiving change refactor-pipeline-modular-steps. Update Purpose after archive.
## Requirements
### Requirement: Individual Step Parameters

Each step MUST accept its specific parameters while sharing common options.

#### Scenario: Dollar bars step parameters

- **GIVEN** running `--step bars --daily-target 20`
- **WHEN** processing dollar bars
- **THEN** uses `daily_target=20` for threshold calculation

#### Scenario: Labels step parameters

- **GIVEN** running `--step labels --pt-sl 2.0 1.0 --vertical-barrier 24`
- **WHEN** applying triple barrier labels
- **THEN** uses PT=2.0, SL=1.0, vertical barrier=24 bars

#### Scenario: Features step parameters

- **GIVEN** running `--step features --windows 5 10 20 --ffd-d 0.6`
- **WHEN** generating features
- **THEN** uses windows [5,10,20] and ffd_d=0.6

