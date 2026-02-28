## 1. Core Logic Functions (Phase A & C)

- [x] 1.1 Implement the probability mapping function (safety clip, z-score, standard normal CDF, side multiplication)
- [x] 1.2 Implement the step-size discretization function and change-trigger logic
- [x] 1.3 Write unit tests to verify ZeroDivisionError prevention with $p=1.0$ and $p=0.0$
- [x] 1.4 Write unit tests for step-size discretization and filtering logic

## 2. Active Bet Aggregation (Phase B)

- [x] 2.1 Implement the active signal tracker that inputs signals, times, and true exit times to define $Active(t)$
- [x] 2.2 Implement the sequential/vectorized averaging engine to compute $S_t$ and output sizes over time
- [x] 2.3 Write unit tests for signal overlap and averaging with known event sequences and staggered $t_{exit}$

## 3. Module Integration

- [x] 3.1 Create `afmlkit/label/bet_size.py` combining all functions into a unified `get_concurrent_bet_sizes` high-level API or class
- [x] 3.2 Update module `__init__.py` to export the new bet sizing functions
- [x] 3.3 Ensure the logic is properly documented with standard docstrings (Parameters, Returns)

## 4. Pipeline Execution Script

- [x] 4.1 Create `scripts/bet_sizing_pipeline.py` (or integrate into an execution test script) that loads data and tests the full flow
- [x] 4.2 Test the full pipeline output visually and statistically (e.g. plot generated bet sizes over time)
