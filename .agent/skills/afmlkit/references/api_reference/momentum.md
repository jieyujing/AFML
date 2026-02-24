# API Reference: momentum.py

**Language**: Python

**Source**: `feature\core\momentum.py`

---

## Functions

### roc(price: NDArray, period: int) → NDArray

Calculate the Rate of Change (ROC) feature.

:param price: np.array, an array of prices.
:param period: int, the period over which to calculate ROC.
:return: np.array, ROC values.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price | NDArray | - | - |
| period | int | - | - |

**Returns**: `NDArray`



### rsi_wilder(close: NDArray[np.float64], window: int) → NDArray[np.float64]

Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.
:param close: A one-dimensional numpy array of closing prices.
:param window: The number of periods to use for the RSI calculation, default is 14.
:return: A one-dimensional numpy array of RSI values, same length as `close`.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close | NDArray[np.float64] | - | - |
| window | int | - | - |

**Returns**: `NDArray[np.float64]`



### stoch_k(close: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64], length: int) → NDArray[np.float64]

Calculate the Stochastic Oscillator %K value.

:param close: A one-dimensional numpy array of closing prices
:param low: A one-dimensional numpy array of low prices
:param high: A one-dimensional numpy array of high prices
:param length: The lookback period for the stochastic calculation
:return: A one-dimensional numpy array of %K values, same length as input arrays

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| close | NDArray[np.float64] | - | - |
| low | NDArray[np.float64] | - | - |
| high | NDArray[np.float64] | - | - |
| length | int | - | - |

**Returns**: `NDArray[np.float64]`


