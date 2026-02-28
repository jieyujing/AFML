## Context

Following the Meta-Model training, we need to convert the continuous probability forecasts $P(bin=1)$ into actionable, discrete position sizes ($[-1, 1]$). The financial markets are dominated by noise, overlapping signals, and transaction costs. Thus, our Bet Sizing module must intelligently scale positions, aggregate concurrent signals, and discretize allocations to avoid excessive trading.

## Goals / Non-Goals

**Goals:**
* Implement AFML Chapter 10's concurrent bet sizing methodology.
* Calculate continuous base bet sizes using standard normal CDF mappings.
* Implement robust active bet tracking accounting for actual exit times (not just `t1`).
* Discretize the final aggregated bet size using a configurable step size to minimize friction and prevent constant micro-rebalancing.
* Handle edge cases, notably probability bounds causing ZeroDivisionErrors.

**Non-Goals:**
* Portfolio-level allocation (e.g., HRP). This module focuses on single-instrument position sizing based on event-driven signals.
* Execution logic (actually sending orders to an exchange).

## Decisions

1. **Probability Safety Bounding**
   - **Decision**: Force clip probabilities using `np.clip(p_i, 1e-5, 1 - 1e-5)` before z-score calculation.
   - **Rationale**: Tree models can output absolute 0.0 or 1.0. This would cause a `ZeroDivisionError` when computing $z_i = \frac{p - 0.5}{\sqrt{p(1 - p)}}$. Clipping ensures the script never crashes on extreme predictions.
   - **Alternative Considered**: Dropping events with p=0 or p=1. Rejected because extreme confidence signals are valuable and dropping them distorts the backtest.

2. **Base Size Mapping via Standard Normal CDF**
   - **Decision**: Convert probability to a theoretically sound continuous size via $m_i = 2 \cdot \Phi(z_i) - 1$, then apply the primary model's sign: $s_i = m_i \cdot side_i$.
   - **Rationale**: This is the formally validated method in AFML Ch.10, offering a smooth, S-curve transition that scales size non-linearly with confidence.

3. **Concurrency Aggregation leveraging True Event Duration**
   - **Decision**: The active state of an event is tracked using $t_{start, i} \le t \le t_{exit, i}$, where $t_{exit}$ is the *actual* resolution time (upper/lower barrier touch, or vertical expiration), and the aggregated size is the mean of active bets: $S_t = \frac{1}{|Active(t)|} \sum_{k \in Active(t)} s_k$.
   - **Rationale**: Financial series feature heavy overlapping events. Using the actual exit time ensures we only average bets that are genuinely still "alive" in the market.

4. **Step-Size Discretization (Friction Filter)**
   - **Decision**: Apply `round(S_t / step_size) * step_size` (e.g., `step_size = 0.1` or `0.05`) to the final aggregated and theoretical size.
   - **Rationale**: Continuous rebalancing burns capital through transaction costs and slippage. Discretization acts as a noise filter, ensuring trades only occur when conviction shifts meaningfully.

## Risks / Trade-offs

- **Risk: Discretization Delay** → A `step_size` that is too large might cause the model to ignore valid, smaller signal shifts until it's too late.
  - **Mitigation**: Expose `step_size` as a configurable parameter allowing empirical optimization for specific asset liquidity and fee structures.
- **Risk: Memory Exhaustion on Massive Overlaps** → Tracking concurrent state across huge tick data might be slow if implemented iteratively in Python.
  - **Mitigation**: Implement the active signal aggregation using vectorized Pandas operations or Numba acceleration if necessary.
