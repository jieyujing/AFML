# API Reference: transforms.py

**Language**: Python

**Source**: `feature\transforms.py`

---

## Classes

### Identity

Returns the identity transform of a selected column

**Inherits from**: BaseTransform

#### Methods

##### __init__(self, input_col: str = 'close')

Identity transform that returns the input column as is.

:param input_col: If DataFrame is passed, this is the column name to return.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_col | str | 'close' | - |


##### __call__(self, df: pd.DataFrame) → pd.Series

Returns the selected col as a series.

:param df: Input DataFrame.
:param backend: Ignored for this transform.
:return: The input DataFrame with the specified column.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| df | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _validate_input(self, x: pd.DataFrame) → bool

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `bool`


##### output_name(self) → str

Returns the name of the output column.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`




### Lag

Implements lagged values of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, periods: int = 1, input_col: str = 'close')

Compute lagged values over the specified number of periods.

:param input_col: If DataFrame is passed, this is the column name to compute lags on.
:param periods: The lag period.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| periods | int | 1 | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ReturnT

Calculates the lagged returns of a time series using a specified period defined in seconds.
Works for irregular time series too.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: pd.Timedelta = pd.Timedelta(seconds=1e-06), is_log: bool = False, input_col: str = 'close')

Compute lagged returns over the specified time window. Works for unregular time series too.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param window: Period in for the lagged returns. Set it to a small value (e.g. 1e-6) for 1 sample lag.
:param is_log: If True, compute log returns. Otherwise, compute simple returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | pd.Timedelta | pd.Timedelta(seconds=1e-06) | - |
| is_log | bool | False | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### Return

Implements lagged return

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, periods: int = 1, input_col: str = 'close', is_log: bool = False)

Compute lagged returns over the specified number of periods.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param periods: The lag period.
:param is_log: If True, compute log returns. Otherwise, compute simple returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| periods | int | 1 | - |
| input_col | str | 'close' | - |
| is_log | bool | False | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ROC

Computes the Rate of Change (ROC) of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, periods: int, input_col: str = 'close')

Compute the Rate of Change (ROC) of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param periods: The lag periods for the ROC calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| periods | int | - | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### PctChange

Computes the percentage change of a time series using a specified lag.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, input_col: str = 'close')

Calculate the percentage change of a signal with a specified lag.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param periods: The lag period.
:return: The percentage change of the signal.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### RSIWilder

Computes the Relative Strength Index (RSI) of a time series using Wilder's smoothing method.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 14, input_col: str = 'close')

Compute the Relative Strength Index (RSI) of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute RSI on.
:param window: Window size for the RSI calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 14 | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### StochK

Computes the Stochastic Oscillator %K of a time series.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, length: int = 14, input_cols: list[str] = None)

Compute the Stochastic Oscillator %K of a time series.

:param input_cols: If DataFrame is passed, this is the column names to compute %K on. Default is ["high", "low", "close"].
:param length: The lookback period for the stochastic calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| length | int | 14 | - |
| input_cols | list[str] | None | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### EWMST

Computes the exponentially weighted moving standard deviation of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, half_life: pd.Timedelta, input_col: str = 'y')

Compute the exponentially weighted moving standard deviation.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param half_life_sec: Period for the moving standard deviation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| half_life | pd.Timedelta | - | - |
| input_col | str | 'y' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### ZScore

Computes the z-score of a time series using a rolling window.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, input_col: str, ddof: int = 0)

Compute the z-score of a time series using a rolling window.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param window: Window size for the rolling calculation.
:param ddof: Delta degrees of freedom for standard deviation calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| input_col | str | - | - |
| ddof | int | 0 | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### BurstRatio

Computes the burst ratio of a time series using a rolling window.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, input_col: str)

Compute the burst ratio of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param window: Window size for the rolling calculation.
:return: series of burst ratios (same size as input)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| input_col | str | - | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### VWAPDistance

Computes the distance of the current price from the VWAP (Volume Weighted Average Price).

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, periods: int, is_log: bool = False, input_cols: str = None)

Calculate the distance of the current price from the VWAP (Volume Weighted Average Price).

:param input_cols: close and volume columns. Default is ["close", "volume"] when None.
:param periods: Number of periods to calculate VWAP.
:param is_log: If True, calculate log distance.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| periods | int | - | - |
| is_log | bool | False | - |
| input_cols | str | None | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### TimeCues

Computes time-based features for a time series.

**Inherits from**: SIMOTransform

#### Methods

##### __init__(self, input_col: str = 'close')

Compute time-based features for a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → tuple[pd.Series, ...]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, ...]`


##### output_name(self)

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### RealizedVolatility

Computes the realized volatility of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, is_sample = False, input_col: str = 'ret')

Compute the realised volatility of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param is_sample: If True, use sample standard deviation. If False, use population standard deviation.
:param window: Window size for the rolling calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| is_sample | None | False | - |
| input_col | str | 'ret' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### BollingerPercentB

Computes the Bollinger Percent B of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, num_std: float = 2.0, input_col: str = 'close')

Compute the Bollinger Percent B of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param num_std: Number of standard deviations for the Bollinger Bands.
:param window: Window size for the rolling calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| num_std | float | 2.0 | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ParkinsonRange

Computes the Parkinson range of a time series.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, input_cols = None)

Compute the Parkinson range of a time series.

:param input_cols: High and Low columns. If None defaults to ["high", "low"].

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_cols | None | None | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### SMA

Computes the Simple Moving Average (SMA) of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, input_col: str = 'x')

Compute the Simple Moving Average (SMA) of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param window: Window size for the rolling calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| input_col | str | 'x' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### EWMA

Computes the Exponentially Weighted Moving Average (EWMA) of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, span: int, input_col: str = None)

Compute the Exponentially Weighted Moving Average (EWMA) of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param span: The decay window, or 'span'. Determines how many past points meaningfully impact the EWMA value.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| span | int | - | - |
| input_col | str | None | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### FlowAcceleration

Computes the Flow Acceleration of a time series.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int, recent_periods, input_col: str = 'volume')

Compute the Flow Acceleration of a time series.

:param input_col: If DataFrame is passed, this is the column name to compute returns on.
:param window: Window size for the rolling calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |
| recent_periods | None | - | - |
| input_col | str | 'volume' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### CUSUMTest

Computes the CUSUM test statistics for structural breaks in time series.

Features include:
- Break indicators (snt - critical values): Positive when a break is detected
- Flag features: Binary indicator when a break is detected (1 when break just fired, else 0)
- Score features: Magnitude of the break, clipped to ±10 σ_noise
- Age features: Number of bars since the last break, capped at a maximum value

**Inherits from**: SIMOTransform

#### Methods

##### __init__(self, window_size: int = 50, warmup_period: int = 30, max_age: int = 144, input_col: str = 'close')

Compute the CUSUM test statistics for structural breaks in time series.

:param input_col: If DataFrame is passed, this is the column name to compute the CUSUM test on.
:param window_size: Size of the rolling window for CUSUM test, by default 50.
:param warmup_period: Minimum number of observations before the first statistic is calculated, by default 30.
:param max_age: Maximum age to track since last break (in bars), by default 144 (12h in 5-min bars).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window_size | int | 50 | - |
| warmup_period | int | 30 | - |
| max_age | int | 144 | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → tuple[pd.Series, ...]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, ...]`


##### output_name(self)

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### ATR

Computes the Average True Range (ATR) of price data.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, window: int = 14, ema_based: bool = False, normalize: bool = False, input_cols: list[str] = None)

Compute the Average True Range (ATR) of price data.

:param window: int, lookback period for ATR calculation, default is 14
:param ema_based: bool, if True uses EMA calculation, if False uses SMA calculation
:param normalize: bool, if True normalizes ATR by mid price (avg of high and low)
:param input_cols: list of column names for [high, low, close], defaults to ["high", "low", "close"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 14 | - |
| ema_based | bool | False | - |
| normalize | bool | False | - |
| input_cols | list[str] | None | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### PriceVolumeCorrelation

Calculates the rolling Pearson correlation coefficient between price returns and volume.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, window: int = 8, input_cols: list[str] = None)

Compute the rolling correlation between price returns and volume.

:param window: int, lookback period for correlation calculation, default is 8
:param input_cols: list of column names for [close, volume], defaults to ["close", "volume"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 8 | - |
| input_cols | list[str] | None | - |


##### _pd(self, x)

Pandas implementation of price-volume correlation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

Numba implementation of price-volume correlation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### VPIN

Calculates the VPIN (Volume-synchronized Probability of Informed Trading) metric.
VPIN measures the fraction of signed volume imbalance to total volume in a rolling window.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, window: int = 32, input_cols: list[str] = None)

Compute the VPIN metric over a specified window.

:param window: int, lookback period for VPIN calculation, default is 32
:param input_cols: list of column names for [volume_buy, volume_sell], defaults to ["volume_buy", "volume_sell"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 32 | - |
| input_cols | list[str] | None | - |


##### _pd(self, x)

Pandas implementation of VPIN calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

Numba implementation of VPIN calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### VarianceRatio14

Computes the Variance Ratio of 1-bar returns to 4-bar returns: var(1-bar) / var(4×1-bar).

This is a useful metric to detect microstructure noise vs trending behavior.
For random walks, the ratio should be close to 0.25 (1/4).
Values < 0.25 suggest mean reversion, while values > 0.25 suggest trending/momentum.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 32, input_col: str = 'close', ret_type: str = 'log', ddof: int = 0)

Compute the variance ratio var(1-bar return) / var(4×1-bar return)

:param window: Window size for variance calculation, default is 32
:param input_col: Column to compute the ratio on, default is "close"
:param ret_type: Type of returns, "simple" or "log", default is "log"
:param ddof: Delta degrees of freedom for variance calculation, default is 0 (sample variance)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 32 | - |
| input_col | str | 'close' | - |
| ret_type | str | 'log' | - |
| ddof | int | 0 | - |


##### _pd(self, x)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### KurtosisTransform

Computes the rolling excess kurtosis of returns.

Excess kurtosis measures the "tailedness" of a distribution compared to a normal distribution.
Positive values indicate fat tails (more extreme values than normal distribution).
Useful for identifying fat-tail regimes (liquidations) that can hurt naive swing trading strategies.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 32, input_col: str = 'ret1')

Compute the rolling excess kurtosis of returns.

:param window: Window size for rolling kurtosis calculation, default is 32
:param input_col: Input column to compute kurtosis on, expected to be returns, default is "ret1"

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 32 | - |
| input_col | str | 'ret1' | - |


##### _pd(self, x)

Pandas implementation of rolling excess kurtosis

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Numba implementation would be more complex - falling back to pandas for now

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### TrendSlope

Computes the OLS slope of ln(close) over a specified window and converts it to an angle in degrees.

This is useful as a trend indicator where the angle represents how steep the trend is.
Positive angles indicate uptrend, negative angles indicate downtrend, and the magnitude
represents the steepness of the trend.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 24, input_col: str = 'close')

Compute the OLS slope of ln(close) over a specified window and convert to an angle in degrees.

:param window: Window size for the rolling OLS calculation, default is 24
:param input_col: Input column to compute slope on, default is "close"

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 24 | - |
| input_col | str | 'close' | - |


##### _pd(self, x)

Pandas implementation of trend slope calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Numba implementation would be more complex - falling back to pandas for now

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ADX

Computes the Average Directional Index (ADX) of price data.

ADX measures the strength of a trend (regardless of direction) on a scale from 0 to 100.
Values below 20 indicate a weak trend, above 25 indicate a strong trend.

This implementation uses Wilder's smoothing method for calculations.

**Inherits from**: MISOTransform

#### Methods

##### __init__(self, length: int = 14, input_cols: list[str] = None)

Compute the Average Directional Index (ADX) of price data.

:param length: Period for ADX calculation, default is 14
:param input_cols: List of column names for [high, low, close], defaults to ["high", "low", "close"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| length | int | 14 | - |
| input_cols | list[str] | None | - |


##### _pd(self, x)

Pandas implementation of ADX calculation (falls back to numba)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | None | - | - |


##### _nb(self, x: pd.DataFrame) → pd.Series

Numba implementation of ADX calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`




### MeanReversionZScore

Calculates the z-score of price relative to its simple moving average.
Formula: (close - SMA_window)/std_window
Used as a mean-reversion filter to identify potential mean-reversion opportunities.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 48, input_col: str = 'close')

Calculate the z-score of price relative to its simple moving average.

:param window: The window size for SMA and standard deviation calculation
:param input_col: If DataFrame is passed, this is the column name to compute z-score on

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 48 | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### DailyGap

Calculates the overnight UTC gap between the close price at 00:00 and the previous day's close at 23:45.
Formula: (close_{00:00} - close_{23:45_prev}) / close_{23:45_prev}

This assumes the input data is in 15-minute intervals and is UTC-aligned.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, input_col: str = 'close')

Calculate the overnight (UTC) gap in price.

:param input_col: If DataFrame is passed, this is the column name to compute the gap on

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ORBBreak

Detects Opening Range Breakout (ORB) signals within a UTC day.

An ORB occurs when the price breaks above the high or below the low of the first hour of trading.
The transform returns two signals: a long signal (1 when price breaks above opening range high,
otherwise 0) and a short signal (1 when price breaks below opening range low, otherwise 0).

This implementation assumes the input data is in 15-minute intervals and is UTC-aligned.
The opening range is defined as the first 4 bars (first hour) of each UTC day.

**Inherits from**: MIMOTransform

#### Methods

##### __init__(self, input_cols: list[str] = None)

Calculate Opening Range Breakout signals

:param input_cols: List of column names for [high, low, close], defaults to ["high", "low", "close"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_cols | list[str] | None | - |


##### _pd(self, x: pd.DataFrame) → tuple[pd.Series, pd.Series]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, pd.Series]`


##### _nb(self, x: pd.DataFrame) → tuple[pd.Series, ...]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, ...]`


##### output_name(self)

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### BarRate

Calculates the rate of bars (number of bars divided by time window) in a specified time window.

This is useful for:
- Detecting rare "flurries" of activity (multiple jumps in short periods)
- Distinguishing between normal and super-quiet market regimes
- Identifying periods of unusual market activity

For example:
- rate_6m: bars in last 6 min ÷ 360 s - Flags the rare flurries (2-3 jumps in a few minutes)
- rate_30m: CUSUM bars in last 30 min ÷ 1800 s - Separates "normal" from "super-quiet" regimes

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: pd.Timedelta, input_col: str = 'close')

Calculate the rate of bars in a specified time window.

:param window_sec: Time window size in seconds
:param input_col: Input column to use (only needed for timestamp extraction)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | pd.Timedelta | - | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of bar rate calculation.

:param x: Input DataFrame with DatetimeIndex
:return: Series containing bar rates

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation for now

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### CandleShape

Computes various candle shape metrics to characterize price action.

Features include:
- wick_up_ratio: Ratio of upper wick to total candle range
- wick_dn_ratio: Ratio of lower wick to total candle range
- body_ratio: Ratio of candle body to total candle range
- vwap_drift: Percentage difference between VWAP and open price

**Inherits from**: MIMOTransform

#### Methods

##### __init__(self, input_cols: list[str] = None)

Calculate various candle shape metrics.

:param input_cols: List of column names for [open, high, low, close, vwap],
                   defaults to ["open", "high", "low", "close", "vwap"]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_cols | list[str] | None | - |


##### _pd(self, x: pd.DataFrame) → tuple[pd.Series, ...]

Pandas implementation of candle shape metrics.

:param x: Input DataFrame with OHLCV data
:return: Tuple of Series containing the calculated metrics

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, ...]`


##### _nb(self, x: pd.DataFrame) → tuple[pd.Series, ...]

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `tuple[pd.Series, ...]`


##### output_name(self)

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### HurstExponent

Computes the Hurst exponent of a time series using the aggregated variance method.

The Hurst exponent (H) is a measure of the long-term memory of a time series:
- H > 0.5 indicates a persistent/trending price path
- H = 0.5 indicates a random walk (Brownian motion)
- H < 0.5 indicates a mean-reverting/noisy series

This implementation uses the aggregated variance method to estimate H.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 24, input_col: str = 'ret1')

Compute the Hurst exponent using the aggregated variance method.

:param window: The rolling window size to compute the Hurst exponent
:param input_col: The input column to compute the Hurst exponent on (typically returns)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 24 | - |
| input_col | str | 'ret1' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of Hurst exponent via aggregated variance method

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _hurst_aggregated_variance(series)

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| series | None | - | - |


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### ApproximateEntropy

Computes the approximate entropy (ApEn) of a time series.

Approximate entropy measures the complexity or irregularity of a time series:
- High ApEn values indicate high irregularity/unpredictability
- Low ApEn values indicate regularity/structure/predictability

This implementation uses the antropy package for calculating ApEn.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 24, m: int = 2, tolerance: float = 0.2, input_col: str = 'ret1')

Compute the approximate entropy of a time series.

:param window: The rolling window size for ApEn calculation
:param m: Embedding dimension (pattern length), default is 2
:param tolerance: Tolerance parameter, default is 0.2 (will be multiplied by std of window)
:param input_col: Input column to compute ApEn on (typically returns)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 24 | - |
| m | int | 2 | - |
| tolerance | float | 0.2 | - |
| input_col | str | 'ret1' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of approximate entropy calculation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### BarDurationEWMA

Computes the Exponentially Weighted Moving Average (EWMA) of bar durations.

This transform calculates the time difference between consecutive bars and then
applies an exponential moving average to these durations. It's useful for:
- Identifying periods of high/low trading activity
- Normalizing other features based on time flow
- Detecting regime changes in market microstructure

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, span: int = 20, input_col: str = 'close')

Compute the EWMA of bar durations.

:param span: The span parameter for the EWM calculation, default is 20
            (equivalent to alpha=2/(span+1))
:param input_col: Input column to use (only needed for timestamp extraction)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| span | int | 20 | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of bar duration EWMA.

:param x: Input DataFrame with DatetimeIndex
:return: Series containing EWMA of bar durations

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### BarDuration

This transform calculates the time difference between consecutive bars in seconds.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, periods = 1, input_col: str = 'close')

Compute the EWMA of bar durations.

:param input_col: Input column to use (only needed for timestamp extraction)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| periods | None | 1 | - |
| input_col | str | 'close' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of bar duration EWMA.

:param x: Input DataFrame with DatetimeIndex
:return: Series containing EWMA of bar durations

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### BiPowerVariation

Computes the bi-power variation (BV) of a return series.

Bi-power variation is used to estimate the integrated variance in the presence of jumps.
It is calculated as the sum of the products of consecutive absolute returns,
multiplied by a correction factor.

This is useful for:
- Separating continuous and jump components of volatility
- Creating jump-robust volatility estimators
- Identifying the presence of jumps when compared to realized volatility

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, window: int = 12, input_col: str = 'ret1')

Compute the bi-power variation of a return series.

:param window: The window size for the calculation (12 means 12 consecutive returns ≈ 60 minutes on 5-min grid)
:param input_col: Input column containing returns to compute BV on

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | 12 | - |
| input_col | str | 'ret1' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of bi-power variation.

:param x: Input DataFrame with return data
:return: Series containing bi-power variation values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

Fall back to pandas implementation

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`




### DirRunLen

Counts consecutive same-sign returns until just before the current bar.

The streak resets when the sign changes or when a return is 0.
The count indicates the length of the streak of consecutive returns with the same sign.

**Inherits from**: SISOTransform

#### Methods

##### __init__(self, input_col: str = 'ret1')

Initialize the directional run length transform.

:param input_col: Input column containing returns

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| input_col | str | 'ret1' | - |


##### _pd(self, x: pd.DataFrame) → pd.Series

Pandas implementation of directional run length.

:param x: Input DataFrame with return data
:return: Series containing directional run length values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `pd.Series`


##### _nb(self, x: Union[pd.DataFrame, pd.Series]) → pd.Series

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | Union[pd.DataFrame, pd.Series] | - | - |

**Returns**: `pd.Series`


##### numba_core(x: np.ndarray) → np.ndarray

Numba implementation of directional run length calculation.

:param x: Input array of returns
:return: Array containing directional run lengths

**Decorators**: `@staticmethod`, `@njit(nogil=True)`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| x | np.ndarray | - | - |

**Returns**: `np.ndarray`




### ExternalFunction

Wrap an external Python callable (by object or import path) as a Transform.

This enables integrating third-party libraries (e.g., NumPy, TA-Lib) into the
Feature/FeatureKit pipelines while preserving consistent input/output handling
and enabling serialization.

Notes:

- For multiple outputs (functions returning a tuple/list), you must provide
  `output_cols` with matching length.
- If `pass_numpy=True`, the callable receives NumPy arrays instead of pandas
  Series, which improves compatibility/performance for libraries expecting
  ndarrays (e.g., TA-Lib).

**Inherits from**: BaseTransform

#### Methods

##### __init__(self, func: Union[str, Callable], input_cols: Union[str, Sequence[str]], output_cols: Union[str, Sequence[str], None] = None)

Initialize the ExternalFunction transform.
This transform wraps an external Python function or import path to be used
as a Transform in the Feature/FeatureKit pipeline.

:param func: String import path or callable object to wrap.
:param input_cols: Columns required as input for the function.
:param output_cols: Columns produced by the function.
:param args: Additional positional arguments to pass to the function.
:param kwargs: Additional keyword arguments to pass to the function.
:param pass_numpy: If True, the function receives NumPy arrays instead of pandas Series.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| func | Union[str, Callable] | - | - |
| input_cols | Union[str, Sequence[str]] | - | - |
| output_cols | Union[str, Sequence[str], None] | None | - |


##### output_name(self) → str | list[str]

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str | list[str]`


##### _validate_input(self, x: pd.DataFrame) → bool

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `bool`


##### _resolve_func(self) → Callable

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `Callable`


##### __call__(self, x: pd.DataFrame) → Union[pd.Series, tuple[pd.Series, ...]]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| x | pd.DataFrame | - | - |

**Returns**: `Union[pd.Series, tuple[pd.Series, ...]]`



