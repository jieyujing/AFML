# API Reference: ma.py

**Language**: Python

**Source**: `feature\core\ma.py`

---

## Functions

### ewma(y: NDArray, span: int) → NDArray[np.float64]

Exponentially weighted moving average (EWMA) of a one-dimensional numpy array.
Calculates the equivalent of `pandas.DataFrame.ewm(...).mean()` with `adjust=True`.

By using this weighting scheme, the function provides a more accurate and unbiased estimate of the EWMA,
especially in the early stages of the data series.

:param y: A one-dimensional numpy array of floats.
:param span: The decay window, or 'span'. Determines how many past points meaningfully impact the EWMA value.
:returns: The EWMA vector, same length and shape as `y`.
:raises ValueError: If `span` is less than 1.

.. note::
    This function adjusts for small sample sizes by dividing by the cumulative weight.
    For more information, see: https://terbe.dev/blog/posts/exponentially-weighted-moving-average

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray | - | - |
| span | int | - | - |

**Returns**: `NDArray[np.float64]`



### sma(array: NDArray[np.float64], window: int) → NDArray[np.float64]

Calculate Simple Moving Average (SMA) with Numba for better performance.

:param array: np.array, input array to calculate SMA
:param window: int, window for the rolling average
:return: np.array, SMA values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| array | NDArray[np.float64] | - | - |
| window | int | - | - |

**Returns**: `NDArray[np.float64]`


