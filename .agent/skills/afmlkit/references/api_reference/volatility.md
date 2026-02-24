# API Reference: volatility.py

**Language**: Python

**Source**: `feature\core\volatility.py`

---

## Functions

### ewms(y: NDArray[np.float64], span: int) → NDArray[np.float64]

Calculates the Exponentially Weighted Moving Standard Deviation (EWM_STD) of a one-dimensional numpy array.
Equivalent to `pandas.Series.ewm(...).std()` with `adjust=True` and `bias=False`.

:param y: A one-dimensional numpy array of floats.
:param span: The decay window, or 'span'.
:returns: The EWM standard deviation vector, same length and shape as `y`.

.. note::
    This function adjusts for small sample sizes by dividing by the cumulative weight minus the sum of squared weights
    divided by the cumulative weight, matching the behavior of `adjust=True` and `bias=False` in pandas.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| y | NDArray[np.float64] | - | - |
| span | int | - | - |

**Returns**: `NDArray[np.float64]`



### ewmst_mean0(timestamps: NDArray[np.int64], y: NDArray[np.float64], half_life: float, sigma_floor: float = 1e-12) → NDArray[np.float64]

Unbiased EWMA std-dev with time-decay half-life fo a zero-mean series)

σ_t² = U_t / V_t  with
  U_t = α_t * y_t² + (1-α_t) * U_{t-1}
  V_t = α_t       + (1-α_t) * V_{t-1}

where α_t = 1 - exp(-Δt / half_life), Δt in seconds
and y_t is your return at timestamp[t].

:param timestamps: 1D array of event times in nanoseconds.
:param y:          1D array of floats (e.g. lagged returns).
:param half_life:  Decay half-life in seconds.
:param sigma_floor: Minimum σ to enforce stability.
:returns:          EWMA standard deviation array.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| y | NDArray[np.float64] | - | - |
| half_life | float | - | - |
| sigma_floor | float | 1e-12 | - |

**Returns**: `NDArray[np.float64]`



### ewmst(timestamps: NDArray[np.int64], y: NDArray[np.float64], half_life: float, sigma_floor: float = 1e-12) → NDArray[np.float64]

Unbiased time-decay EWMA std-dev (adjust=True, bias=False semantics).

Maintains:
  V  = sum of weights
  V2 = sum of squared weights
  Sy  = EWMA sum of y
  Syy = EWMA sum of y^2

Then
  mean_t   = Sy / V
  ewma_y2  = Syy / V
  var_raw  = ewma_y2 - mean_t^2
  denom    = V - V2 / V
  var_t    = var_raw * (V / denom)
  σ_t      = sqrt(max(var_t, 0))

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| y | NDArray[np.float64] | - | - |
| half_life | float | - | - |
| sigma_floor | float | 1e-12 | - |

**Returns**: `NDArray[np.float64]`



### true_range(high: NDArray, low: NDArray, close: NDArray) → NDArray

Calculate True Range using Numba.

:param high: np.array, high prices
:param low: np.array, low prices
:param close: np.array, close prices
:return: np.array, True Range values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | NDArray | - | - |
| low | NDArray | - | - |
| close | NDArray | - | - |

**Returns**: `NDArray`



### realized_vol(r: NDArray[np.float64], window: int, is_sample: bool) → NDArray[np.float64]

Calculate realized volatility from return input using Numba.

:param r: np.array of returns
:param window: int, number of samples for volatility calculation
:param is_sample: bool, if True uses (n-1) divisor for sample standard deviation, else uses n for population
:return: np.array, realized volatility values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| r | NDArray[np.float64] | - | - |
| window | int | - | - |
| is_sample | bool | - | - |

**Returns**: `NDArray[np.float64]`



### bollinger_percent_b(close: NDArray[np.float64], window: int, num_std: float) → NDArray[np.float64]

Calculate Bollinger Percent B indicator.

Bollinger Percent B shows where the price is in relation to the Bollinger Bands.
Values range typically between 0 and 1, where:
- Values above 1 indicate price is above the upper band
- Values below 0 indicate price is below the lower band
- Value of 0.5 indicates price is at the middle band (SMA)

:param close: Array of close prices
:param window: Lookback window for calculations
:param num_std: Number of standard deviations for bands
:return: Array of Bollinger Percent B values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close | NDArray[np.float64] | - | - |
| window | int | - | - |
| num_std | float | - | - |

**Returns**: `NDArray[np.float64]`



### parkinson_range(high: NDArray[np.float64], low: NDArray[np.float64]) → NDArray[np.float64]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | NDArray[np.float64] | - | - |
| low | NDArray[np.float64] | - | - |

**Returns**: `NDArray[np.float64]`



### atr(high: NDArray[np.float64], low: NDArray[np.float64], close: NDArray[np.float64], window: int, ema_based: bool = False, normalize: bool = False) → NDArray[np.float64]

Calculate Average True Range (ATR).

ATR is a measure of market volatility showing how much an asset price moves, on average,
during a given time period. ATR can be calculated using either a simple moving average
or an exponential moving average method.

:param high: np.array, high prices
:param low: np.array, low prices
:param close: np.array, close prices
:param window: int, lookback period
:param ema_based: bool, if True uses EMA calculation, if False uses SMA calculation
:param normalize: bool, if True normalizes ATR by mid price (avg of high and low)
:return: np.array, ATR values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| high | NDArray[np.float64] | - | - |
| low | NDArray[np.float64] | - | - |
| close | NDArray[np.float64] | - | - |
| window | int | - | - |
| ema_based | bool | False | - |
| normalize | bool | False | - |

**Returns**: `NDArray[np.float64]`



### rolling_variance_nb(series: NDArray[np.float64], window: int, ddof: int = 1, min_periods: int = 1) → NDArray[np.float64]

Calculate rolling variance of a series with NaN handling.

:param series: Input array
:param window: Window size for rolling calculation
:param ddof: Delta degrees of freedom (1 for sample variance, 0 for population variance)
:param min_periods: Minimum number of valid observations required to calculate result
:return: Array of rolling variances, same length as input series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| series | NDArray[np.float64] | - | - |
| window | int | - | - |
| ddof | int | 1 | - |
| min_periods | int | 1 | - |

**Returns**: `NDArray[np.float64]`



### variance_ratio_1_4_core(price: NDArray[np.float64], window: int, ddof: int, ret_type: str) → NDArray[np.float64]

Calculate the variance ratio: var(1-bar return) / var(4×1-bar return).

This ratio helps detect microstructure noise vs trending. Values closer to 0.25 suggest
a random walk (efficient market), while values significantly different from 0.25 suggest
either mean-reversion (<0.25) or momentum/trending (>0.25).

:param price: Input price array
:param window: Window size for variance calculation
:param ddof: Delta degrees of freedom for variance
:param ret_type: Type of returns to use: "simple" or "log"
:return: Array of variance ratios, same length as input price

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price | NDArray[np.float64] | - | - |
| window | int | - | - |
| ddof | int | - | - |
| ret_type | str | - | - |

**Returns**: `NDArray[np.float64]`


