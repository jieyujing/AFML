## ADDED Requirements

### Requirement: Base Probability Mapping
The pipeline SHALL securely map raw probabilities from the Meta-Model to continuous target allocations using the standardized z-score methodology, preventing zero-division crashes.

#### Scenario: Normal Probability Mapping
- **WHEN** the Meta-Model issues a probability forecast $P$ and the primary model issues a side index $side$
- **THEN** it MUST bounds-clip the probability to $[1e-5, 1 - 1e-5]$
- **THEN** it MUST calculate the standardized $z$-score and map it to a continuous base size $m_i \in [0, 1]$ via the standard normal cumulative distribution function
- **THEN** it MUST apply the primary model's sign to emit a base signal $s_i = m_i \times side$

### Requirement: Concurrent Signal Tracking
The pipeline SHALL aggregate overlapping event signals by averaging the active theoretical positions to formulate the current continuous target portfolio allocation.

#### Scenario: Determining Active Status
- **WHEN** evaluating which signals are actively influencing current allocation at time $t$
- **THEN** it MUST identify all past signals whose calculated real exit time ($t_{exit}$, based on hitting barrier boundaries) is $\ge t$

#### Scenario: Computing Target Average
- **WHEN** a set of $N$ active signals exists
- **THEN** it MUST compute the mean of their base theoretical sizes $s_k$ to determine $S_t$
- **THEN** it MUST output $S_t = 0$ if no active signals are present

### Requirement: Actionable Size Discretization
The pipeline SHALL enforce a configurable filter step size to minimize excessive small trades due to continuous probability drifting.

#### Scenario: Discretizing the Aggregated Signal
- **WHEN** the pipeline generates the continuous aggregated target position $S_t$
- **THEN** it MUST apply the rounding formula: $S_{final, t} = \text{round}(S_t / step\_size) \times step\_size$

#### Scenario: Change Notification
- **WHEN** evaluating a newly discretized step size $S_{final, t}$ against the previous value
- **THEN** it MUST ONLY trigger a rebalance or output an order magnitude if the new $S_{final, t}$ value differs from the previous holding, suppressing noise.
