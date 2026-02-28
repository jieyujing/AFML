## Why

The AFML pipeline currently generates continuous probability forecasts $P(bin=1)$ from the Meta-Model, but lacks a mechanism to translate these abstract probabilities into actionable, real-world portfolio allocations. To avoid high transaction costs, transaction friction, and extreme position flipping in noisy financial data, we need a rigorous Bet Sizing module that dynamically maps probabilities to positions, aggregates concurrent active signals properly, and enforces configurable step-size limits.

## What Changes

- Implement a standard normal CDF mapping function to convert raw Meta-Model probabilities and primary model sides into continuous theoretical bet sizes $m \in [-1, 1]$.
- Add a safety bound mechanism `np.clip(p, 1e-5, 1-1e-5)` to prevent `ZeroDivisionError` during Z-score calculations from overconfident tree models.
- Build a concurrent sequence aggregator to compute the average target position across all overlapping signals whose true exit time ($t_{exit}$) has not yet occurred.
- Introduce a step-size discretization gate `round(m / step) * step` to filter out minor, untradeable fluctuations.

## Capabilities

### New Capabilities
- `bet-sizing-pipeline`: A new end-to-end module that computes base bet sizes from probabilities, averages tracking of active concurrent bets based on real exit bounds, and applies step-size discretization to generate actionable allocation targets.

### Modified Capabilities
- None.

## Impact

- **New Modules:** Adds a dedicated `afmlkit.label.bet_size` module (or similar) encapsulating the logic.
- **Dependencies:** Relies on pandas/numpy for calculations and relies on outputs derived from the primary model and the Meta-Model training processes.
- **Downstream Strategy Evaluation:** Directly impacts how backtests parse model predictions, translating pure classification logs into actual traded capital paths.
