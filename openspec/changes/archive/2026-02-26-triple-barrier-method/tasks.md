## 1. Environment & Serialization Setup

- [x] 1.1 Extract core variables (`close`, `events`, `pt`, `sl`) from Pandas structures to plain NumPy arrays.
- [x] 1.2 Implement timestamp conversion (casting `pd.Timestamp` and DatetimeIndex into explicit `int64` nanosecond representations) in the preprocessing step before crossing the Numba boundary.

## 2. Implement Core Numba Iterator (`triple_barrier`)

- [x] 2.1 Establish a uniprocessing `@njit` (no parallel) block for the `triple_barrier` loop logic in `afmlkit.label.tbm`.
- [x] 2.2 Implement the Barrier Deactivation logic (swap 0 or NaN limit multipliers to float infinity, enforcing silent ignored bypasses) before entering the event scanning loop.
- [x] 2.3 Implement the forward-scanning loop that checks the condition `price_diff >= pt` or `price_diff <= -sl` at each incoming timestamp up to the `t1` boundary limit.
- [x] 2.4 Incorporate Edge Case Handlers where the loop reaches array boundaries identically without hitting `t1`, `pt` or `sl`, and mark `t1_touch` explicitly as `NaN`.

## 3. Implement Meta-Labeling Schema Integration

- [x] 3.1 Incorporate passing the primary model's `side` vector into the lowest numerical layer.
- [x] 3.2 Add conditional path logic reflecting the side multiplier flip (if `side == -1`, standard falling prices hit `pt` instead of `sl`).
- [x] 3.3 Create secondary label binning transformation mapping 3-class outcomes back into positive `1` (win) and `0` (loss/timeout).

## 4. Integration, Consolidation & Output Verification

- [x] 4.1 Implement upstream collection functions inside `afmlkit.label.kit` that retrieves the pure numeric scalar arrays returned by the Numba engine.
- [x] 4.2 Reconstruct returning objects mapping Numba's structured scalar results back into a unified Pandas DataFrame scheme with `t1`, `ret`, and `bin` fields.
- [x] 4.3 Develop rigorous test suites mimicking edge cases (unreachable large `pt`/`sl`, `NaN` limits, early cutoffs, Short Trade profit hits) comparing actual outcomes against expected specs.
