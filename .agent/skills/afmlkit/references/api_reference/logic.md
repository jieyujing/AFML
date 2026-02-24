# API Reference: logic.py

**Language**: Python

**Source**: `bar\logic.py`

---

## Functions

### _time_bar_indexer(timestamps: NDArray[np.int64], interval_seconds: float) → Tuple[NDArray[np.int64], NDArray[np.int64]]

Determine the time bar open indices in the raw trades timestamp array.

:param timestamps: Raw sorted trade timestamps in nanoseconds.
:param interval_seconds: Length of the time bar in seconds.
:returns: A tuple of:
    - bar_close_ts: Timestamps at which each bar closes.
    - bar_close_indices: Indices in the trade data corresponding to bar closings.

.. note::
    The first bar is aligned to the ceiling of the first timestamp, ensuring consistent bar boundaries.
    Duplicate indices may occur if a bar interval contains no trades (empty bars).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| interval_seconds | float | - | - |

**Returns**: `Tuple[NDArray[np.int64], NDArray[np.int64]]`



### _tick_bar_indexer(timestamps: NDArray[np.int64], threshold: int) → NumbaList

Determine the tick bar open indices in the raw trades timestamp array.

:param timestamps: Raw trade timestamps.
:param threshold: The tick count threshold for opening a new bar.
:returns: close_indices: Timestamps at which each tick bar opens.

.. note::
    The first trade is always the start of a bar.
    A new bar is opened every time the tick count reaches the specified threshold.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| threshold | int | - | - |

**Returns**: `NumbaList`



### _volume_bar_indexer(volumes: NDArray[np.float64], threshold: float) → NumbaList

Determine the volume bar open indices using cumulative volume.
:param volumes: Trade volumes.
:param threshold: Volume bucket threshold for opening a new bar.
:returns: close_indices: Timestamps at which each volume bar opens.

.. note::
    The first trade is always the start of a bar.
    A new bar is opened when the cumulative trade volume meets or exceeds the threshold.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| volumes | NDArray[np.float64] | - | - |
| threshold | float | - | - |

**Returns**: `NumbaList`



### _dollar_bar_indexer(prices: NDArray[np.int64], volumes: NDArray[np.float64], threshold: float) → NumbaList

Determine the dollar bar open indices using cumulative dollar value.

:param prices: Trade prices.
:param volumes: Trade volumes.
:param threshold: Dollar value threshold for opening a new bar.
:returns: close_indices: Timestamps at which each dollar bar opens.

.. note::
    The first trade is always the start of a bar.
    A new bar is opened when the cumulative dollar value (price × volume) meets or exceeds the threshold.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| prices | NDArray[np.int64] | - | - |
| volumes | NDArray[np.float64] | - | - |
| threshold | float | - | - |

**Returns**: `NumbaList`



### _dynamic_dollar_bar_indexer(prices: NDArray[np.int64], volumes: NDArray[np.float64], thresholds: NDArray[np.float64]) → NumbaList

Determine the dollar bar open indices using cumulative dollar value with a dynamic threshold.

:param prices: Trade prices.
:param volumes: Trade volumes.
:param thresholds: Array of dollar value thresholds for opening a new bar, aligned with prices.
:returns: close_indices: Timestamps at which each dollar bar opens.

.. note::
    The first trade is always the start of a bar.
    A new bar is opened when the cumulative dollar value (price × volume) meets or exceeds the current threshold.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| prices | NDArray[np.int64] | - | - |
| volumes | NDArray[np.float64] | - | - |
| thresholds | NDArray[np.float64] | - | - |

**Returns**: `NumbaList`



### _cusum_bar_indexer(timestamps: NDArray[np.int64], prices: NDArray[np.float64], sigma: NDArray[np.float64], sigma_floor: float, sigma_mult: float) → NumbaList

Determine CUSUM bar open indices using a symmetric CUSUM filter
on successive price changes (López de Prado, 2018).

A new bar starts whenever the cumulative sum of price changes
exceeds +sigma*lambda or –sigma*lambda.

:param timestamps: timestamps of the trades.
:param prices: Trade prices.
:param sigma: Threshold vector for CUSUM (e.g. calculated EWMS volatility or constant).
:param sigma_floor: Minimum value for sigma to avoid division by zero.
:param sigma_mult: sigma multiplier for the CUSUM filter (threshold will be lambda_mult*sigma).
:returns: close_indices

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| prices | NDArray[np.float64] | - | - |
| sigma | NDArray[np.float64] | - | - |
| sigma_floor | float | - | - |
| sigma_mult | float | - | - |

**Returns**: `NumbaList`



### _imbalance_bar_indexer(timestamps: NDArray[np.int64], prices: NDArray[np.float64], volumes: NDArray[np.float64], threshold: float) → Tuple[NDArray[np.int64], NDArray[np.int64]]

Determine the imbalance bar open indices based on cumulative imbalance.

:param timestamps: Raw trade timestamps.
:param prices: Trade prices.
:param volumes: Trade volumes.
:param threshold: Imbalance threshold for opening a new bar.
:returns: A tuple of open timestamps and indices for imbalance bars.
:raises NotImplementedError: Always raised as this function is not yet implemented.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| prices | NDArray[np.float64] | - | - |
| volumes | NDArray[np.float64] | - | - |
| threshold | float | - | - |

**Returns**: `Tuple[NDArray[np.int64], NDArray[np.int64]]`



### _run_bar_indexer(timestamps: NDArray[np.int64], prices: NDArray[np.float64], volumes: NDArray[np.float64], threshold: float) → Tuple[NDArray[np.int64], NDArray[np.int64]]

Determine the run bar open indices using cumulative run activity.

:param timestamps: Raw trade timestamps.
:param prices: Trade prices.
:param volumes: Trade volumes.
:param threshold: Run threshold for opening a new bar.
:returns: A tuple of open timestamps and indices for run bars.
:raises NotImplementedError: Always raised as this function is not yet implemented.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| timestamps | NDArray[np.int64] | - | - |
| prices | NDArray[np.float64] | - | - |
| volumes | NDArray[np.float64] | - | - |
| threshold | float | - | - |

**Returns**: `Tuple[NDArray[np.int64], NDArray[np.int64]]`


