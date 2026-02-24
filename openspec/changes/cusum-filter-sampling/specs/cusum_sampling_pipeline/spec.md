# Specification: CUSUM Sampling Pipeline

## ADDED Requirements

### Requirement: Calculate Rolling Volatility and Dynamic Threshold
- **GIVEN** a dataset of Dollar Bars containing a 'close' column mapped from `outputs/dollar_bars/dollar_bars_freq20.csv`
- **WHEN** the sampling pipeline is executed
- **THEN** it must calculate the log returns of the 'close' price
- **AND** calculate an exponentially weighted standard deviation ($\sigma_t$) with a lookback window (e.g., window=100)
- **AND** generate a dynamic threshold sequence $h_t = 2 \times \sigma_t$.
- **AND** address early missing `NaN` values through backfilling to avoid errors in Numba JIT engines.

### Requirement: Execute CUSUM Filter
- **GIVEN** the original price sequence and the dynamically calculated threshold sequence $h_t$
- **WHEN** passed to the event detector
- **THEN** it must use `afmlkit.sampling.filters.cusum_filter` to identify significant price deviations
- **AND** return an array of integer indices where the cumulative sum of log returns exceeded $h_t$.

### Requirement: Persist Sampled Data
- **GIVEN** the resulting event indices from the filter
- **WHEN** writing the outputs
- **THEN** it must map the indices back to the original DataFrame to create a subset DataFrame
- **AND** display the compression ratio (Sampled length / Original length)
- **AND** save the output subset to disk at `outputs/features/cusum_sampled_bars.csv` (or an appropriate directory).

## CHANGED Requirements
None

## REMOVED Requirements
None
