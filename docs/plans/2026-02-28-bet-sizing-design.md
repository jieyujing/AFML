# Bet Sizing Module Design

## Context
Following the Meta-Model training, we need to convert the continuous probability forecasts $P(bin=1)$ into actionable, discrete position sizes ($[-1, 1]$). The financial markets are dominated by noise, overlapping signals, and transaction costs. Thus, our Bet Sizing module must intelligently scale positions, aggregate concurrent signals, and discretize allocations to avoid excessive trading.

## Goals & Non-Goals
**Goals:**
* Implement AFML Chapter 10's concurrent bet sizing methodology.
* Calculate continuous base bet sizes using standard normal CDF mappings.
* Implement robust active bet tracking accounting for actual exit times (not just `t1`).
* Discretize the final aggregated bet size using a configurable step size to minimize friction and prevent constant micro-rebalancing.
* Handle edge cases, notably probability bounds causing ZeroDivisionErrors.

**Non-Goals:**
* Portfolio-level allocation (e.g., HRP). This module focuses on single-instrument position sizing based on event-driven signals.
* Execution logic (actually sending orders to an exchange).

## Architecture

### Phase A: Base Probabilistic Bet Size (Continuous)
For each new event signal $i$:
1. **Safety Bound:** Probabilities must be clipped to prevent infinite z-scores and zero division.
   ```python
   p_safe = np.clip(p_i, 1e-5, 1 - 1e-5)
   ```
2. **Test Statistic:** Calculate the standardized z-score.
   $$z_i = \frac{p_{safe} - 0.5}{\sqrt{p_{safe}(1 - p_{safe})}}$$
3. **CDF Mapping:** Map to a $[0, 1]$ magnitude using the standard normal CDF ($\Phi$).
   $$m_i = 2 \cdot \Phi(z_i) - 1$$
4. **Directionality:** Multiply by the primary model's predicted side ($side_i \in \{-1, 1\}$).
   $$s_i = m_i \cdot side_i$$

### Phase B: Signal Aggregation (Concurrency)
Financial signals often overlap. The target position at any time $t$ is the average of all *currently active* signals.
1. **Active Definition:** A signal $i$ is active at time $t$ if $t_{start, i} \le t \le t_{exit, i}$.
   *   *Crucial Detail:* $t_{exit, i}$ is the **actual** resolution time of the event (either touching the upper barrier, lower barrier, or the vertical time expiration $t1$). It is NOT merely the scheduled $t1$.
2. **Averaging:**
   $$S_t = \frac{1}{|Active(t)|} \sum_{k \in Active(t)} s_k$$
   *(If $Active(t)$ is empty, $S_t = 0$.)*

### Phase C: Constrained Discretization
To prevent micro-adjustments and excessive transaction costs (slippage/fees):
1. **Step Size:** Introduce a configurable `step_size` (e.g., `0.1`).
2. **Rounding:** Discretize the continuous average $S_t$.
   $$S_{final, t} = \text{round}(S_t / step\_size) \cdot step\_size$$
3. **Action Trigger:** An order is only emitted if $S_{final, t}$ differs from the current holding.

## Implementation Plan
1. Create `afmlkit/label/bet_size.py` to house the core Numba-accelerated/Pandas logic.
2. Develop a `BetSizer` class to handle streaming or batch processing of signals.
3. Write extensive unit tests ensuring edge cases (P=1.0, P=0.0) are caught and concurrent averaging behaves correctly over known event series.
4. Integrate with `meta_model_training.py` or a new standalone script to generate historical state transitions from backtest outputs.
