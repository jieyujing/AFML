# API Reference: filters.py

**Language**: Python

**Source**: `sampling\filters.py`

---

## Functions

### cusum_filter(raw_time_series: NDArray[np.float64], threshold: NDArray) → NDArray[np.int64]

Apply the CUSUM filter to detect events based on the cumulative sum of log returns.

:param raw_time_series: Array of price series.
:param threshold: Threshold values for event detection.
    - If array has 1 element, a constant threshold is used.
    - If multiple elements, it must be of the same length as `raw_time_series`.
:returns: Indices where events occurred. These indices correspond to positions in `raw_time_series`.

.. note::
    This function implements the Symmetric CUSUM Filter, which is designed to detect a shift
    in the mean value of a measured quantity away from a target value. It identifies events
    when the cumulative sum of log returns exceeds a specified threshold.

    This implementation follows the methodology outlined in:

    - Lopez de Prado, Marcos. "Advances in Financial Machine Learning." Wiley, 2018. Snippet 2.4, page 39.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| raw_time_series | NDArray[np.float64] | - | - |
| threshold | NDArray | - | - |

**Returns**: `NDArray[np.int64]`



### z_score_peak_filter(y: NDArray[np.float64], window: int, threshold: float = 3) → NDArray[np.int64]

Implement a z-score peak detection filter.

:param y: The input time series data of at least length `window + 2`.
:param window: The window parameter for the moving window (number of observations to use for mean and standard deviation calculations).
:param threshold: The z-score threshold for detecting peaks.
:returns: The indices of the events (peaks) in the input time series data `y`.

.. note::
    This function implements a z-score based peak detection algorithm suitable for real-time data.
    It is optimized for performance using Numba's JIT compilation.

.. seealso::
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray[np.float64] | - | - |
| window | int | - | - |
| threshold | float | 3 | - |

**Returns**: `NDArray[np.int64]`


