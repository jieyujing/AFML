# API Reference: cusum.py

**Language**: Python

**Source**: `feature\core\structural_break\cusum.py`

---

## Functions

### _comp_max_s_nt(y: NDArray, t: int, sigma_sq_t: float) → Tuple[float, float, float, float]

Compute the maximum S_n values and critical values for upward and downward movements.

:param y: Array of log price series.
:param t: Current time index.
:param sigma_sq_t: Estimated variance at time t.
:returns: A tuple containing:
    - max_s_n_value_up: Maximum test statistic for upward movements.
    - max_s_n_value_down: Maximum test statistic for downward movements.
    - max_s_n_critical_value_up: Critical value for the upward test statistic.
    - max_s_n_critical_value_down: Critical value for the downward test statistic.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray | - | - |
| t | int | - | - |
| sigma_sq_t | float | - | - |

**Returns**: `Tuple[float, float, float, float]`



### cusum_test_developing(y: NDArray, warmup_period: int = 30) → Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

Perform the Chu-Stinchcombe-White CUSUM Test on Levels.

This implementation follows Homm and Breitung (2011), but the one-sided tests
are used to detect upward or downward structural breaks, providing directionality.

:param y: Array of price series (log prices).
:param warmup_period: Number of initial observations kept for std warmup, by default 30.
:returns: A tuple containing:
    - s_n_t_values_up: Test statistic values for upward movements (one-sided test).
    - s_n_t_values_down: Test statistic values for downward movements (one-sided test).
    - critical_values_up: Critical values for the upward test statistics.
    - critical_values_down: Critical values for the downward test statistics.

.. note::
    The test detects deviations from the random walk hypothesis in the time series.
    A significant test statistic indicates a structural break in the time series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray | - | - |
| warmup_period | int | 30 | - |

**Returns**: `Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`



### cusum_test_last(y: NDArray) → Tuple[float, float, float, float]

Perform the Chu-Stinchcombe-White CUSUM Test on Levels for the last observation.

This implementation follows Homm and Breitung (2011), but the one-sided tests
are used to detect upward or downward structural breaks, providing directionality.

:param y: Array of price series (log prices).
:returns: A tuple containing:
    - max_s_n_value_up: Test statistic value for upward movement (one-sided test) at the last observation.
    - max_s_n_value_down: Test statistic value for downward movement (one-sided test) at the last observation.
    - max_s_n_critical_value_up: Critical value for the upward test statistic at the last observation.
    - max_s_n_critical_value_down: Critical value for the downward test statistic at the last observation.

.. note::
    A significant test statistic at the last observation indicates a recent structural break
    in the time series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray | - | - |

**Returns**: `Tuple[float, float, float, float]`



### cusum_test_rolling(close_prices: NDArray, window_size: int = 1000, warmup_period: int = 30) → Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

Perform the Chu-Stinchcombe-White CUSUM Test on Levels over a rolling window.

This implementation follows Homm and Breitung (2011), where the one-sided tests
are used to detect upward or downward structural breaks, providing directionality.

:param close_prices: Array of close price series.
:param window_size: Size of the rolling window, by default 1000.
:param warmup_period: Minimum number of observations before the first statistic is calculated, by default 30.
:returns: A tuple containing:
    - snt_up: Test statistic values for upward movements (one-sided test) over the rolling window.
    - snt_down: Test statistic values for downward movements (one-sided test) over the rolling window.
    - critical_values_up: Critical values for the upward test statistics over the rolling window.
    - critical_values_down: Critical values for the downward test statistics over the rolling window.

.. note::
    This test detects deviations from the random walk hypothesis in the time series over a moving window.
    It helps identify periods of structural breaks in the time series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close_prices | NDArray | - | - |
| window_size | int | 1000 | - |
| warmup_period | int | 30 | - |

**Returns**: `Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`


