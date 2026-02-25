## ADDED Requirements

### Requirement: TBM Core Numba Iterator
The system SHALL implement a pure state Numba iteration that processes chronological touches of the barriers and identifies the first touched barrier (pt, sl, or t1) without any Pandas-based slicing loops.

#### Scenario: Normal touch on profit taking
- **WHEN** price reaches the upper threshold (pt) calculated from dynamic volatility before any sl curve or t1 barrier is touched.
- **THEN** returns a touch time equal to the timestamp where it exceeded the pt line and a final label corresponding to profit taking.

#### Scenario: Normal touch on stop loss
- **WHEN** price hits the lower threshold (sl) before pt or t1.
- **THEN** returns a touch time equal to the timestamp where it dropped below sl and a final label corresponding to stop loss.

#### Scenario: Timeout via vertical barrier
- **WHEN** the chronological index hits or exceeds the `t1` vertical barrier timestamp before the price series reaches either the `pt` or `sl` limits.
- **THEN** returns the `t1` timestamp as the touch time and a label value corresponding to timeout.

### Requirement: Barrier Deactivation
The system MUST silently ignore and discard consideration of horizontal barriers if given 0 or NaN as multiplier config limits.

#### Scenario: Zero stop-loss parameter
- **WHEN** the `sl` parameter passed is 0 or NaN.
- **THEN** the system SHALL treat the lower barrier as unreachably low (e.g. negative infinity) forcing eventual t1 timeout or pt touch instead.

### Requirement: Edge Case Out Of Bounds Safety
The system SHALL prevent lookahead bias when iterating to the boundary of the provided series without touching any barrier.

#### Scenario: Data ends before t1 or a price hit
- **WHEN** evaluating the last available events but the price history array ends before ever triggering `t1`, `pt`, or `sl`.
- **THEN** the function MUST return NaN for the touch time and appropriate marker for the label to allow the Pandas caller layer to drop these unresolved incomplete records via `dropna()`.

### Requirement: Unified Output Schema
The system SHALL reconstruct a final resulting structured format that correctly associates the timestamp of initialization, the timestamp of the first touch, the exact realized return, and the final classification outcome for transparent hand-offs.

#### Scenario: Full extraction post-Numba
- **WHEN** the lowest-level Numba calculation successfully computes arrays of first touches and yields.
- **THEN** it outputs a unified DataFrame containing index (event initiation), `t1` (the actual timestamp of first hit), `ret` (the continuous realized return up to `t1`), and `bin` (label).
