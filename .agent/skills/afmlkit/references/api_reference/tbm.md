# API Reference: tbm.py

**Language**: Python

**Source**: `label\tbm.py`

---

## Functions

### triple_barrier(timestamps: NDArray[np.int64], close: NDArray[np.float64], event_idxs: NDArray[np.int64], targets: NDArray[np.float64], horizontal_barriers: Tuple[float, float], vertical_barrier: float, min_close_time_sec: float, side: Optional[NDArray[np.int8]], min_ret: float) → Tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]

Implements the Triple Barrier Method (TBM) for labeling financial data based on
Advances in Financial Machine Learning, Chapter 3.

:param timestamps: The timestamps in nanoseconds for the close prices series.
:param close: The close prices of the asset.
:param event_idxs: The nanosecond timestamps of the events, e.g. acquired from the cusum filter. (subset of timestamps)
:param targets: Log-return targets for the events, e.g. acquired from a moving volatility estimator. Length must matchevent_idxs.
:param horizontal_barriers: The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
    This setup determines the width of the horizontal barriers. If you want to disable the barriers, set it to np.inf or -np.inf.
:param vertical_barrier: The temporal barrier in seconds. Set it to np.inf to disable the vertical barrier.
:param min_close_time_sec: The minimum open time in seconds (useful when raw tick data is used). This prevents closing the event prematurely before the minimum open time is reached.
:param side: Optional array indicating the side of the event (-1 for sell, 1 for buy) for meta labeling. Length must match event_idxs. None for side predication.
:param min_ret: The minimum target value for meta-labeling. If the return is below this value, the label will be 0, otherwise 1.
:returns: A tuple of 4 elements containing:

    - The label (-1, 1) for side prediction (barriers should be symmetric); If side is provided, the meta-labels are (0, 1)
    - The first barrier touch index,
    - The return,
    - Maximum return-barrier ratio during the search describing how close the path came to a horizontal barrier.
      This can be used to weight samples. If a barrier is hit, the ratio is 1.0, otherwise it is less than 1.0 – or np.nan if barriers are disabled)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| close | NDArray[np.float64] | - | - |
| event_idxs | NDArray[np.int64] | - | - |
| targets | NDArray[np.float64] | - | - |
| horizontal_barriers | Tuple[float, float] | - | - |
| vertical_barrier | float | - | - |
| min_close_time_sec | float | - | - |
| side | Optional[NDArray[np.int8]] | - | - |
| min_ret | float | - | - |

**Returns**: `Tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]`


