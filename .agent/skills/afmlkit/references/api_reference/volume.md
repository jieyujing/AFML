# API Reference: volume.py

**Language**: Python

**Source**: `feature\core\volume.py`

---

## Classes

### VolumePro

Encapsulates numba functions for smoother calling and parameter setting.

**Inherits from**: (none)

#### Methods

##### __init__(self, window_size: pd.Timedelta, n_bins: int = 27, va_pct: float = 68.34)

Initialize the Volume Profile calculator with the given parameters.

:param window_size: Size of the rolling windows.
:param n_bins: Number of bins for price level bucketing.
:param va_pct: Value area percentage.
:note:
    This sets the rolling window size, the bin size for price level bucketing, and the value area percentage
    used for determining the high and low value areas (HVA and LVA).
    Default values are window_size_sec=1800, n_bins=27, va_pct=68.34.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window_size | pd.Timedelta | - | - |
| n_bins | int | 27 | - |
| va_pct | float | 68.34 | - |


##### reset_parameters(self, window_size_sec: int = None, n_bins: int = None, va_pct: float = None)

Reset the parameters of the Volume Profile calculator.

:param window_size_sec: Optional new window size in seconds. If None, the existing value is retained.
:param n_bins: Optional number of bins for price level bucketing. If None, the existing value is retained.
:param va_pct: Optional new value area percentage. If None, the existing value is retained.
:note:
    This method allows dynamic reconfiguration of the rolling window size, price bin size, or value area percentage
    for the Volume Profile calculations. Any parameter left as None will retain its prior value.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window_size_sec | int | None | - |
| n_bins | int | None | - |
| va_pct | float | None | - |


##### compute(self, bars: pd.DataFrame, fp_data: FootprintData) → tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Compute the volume profile parameters (POC, HVA, LVA) in a rolling window fashion.

:param bars: DataFrame containing dynamic bars (must include `high` and `low` columns).
:param fp_data: FootprintData object with price levels and volume information.
:returns: Tuple of POC, HVA, and LVA prices, and volume percentage above POC, (as NumPy arrays)
:raises AssertionError: If `bars` and `fp_data` have different lengths.
:note:
    The computation is performed in a rolling window fashion, using the set window size, bin size, and value area percentage.
    The bars DataFrame must contain 'high' and 'low' columns, and the footprint data must be aligned in length.
    Returned arrays represent the Point of Control (POC), High Value Area (HVA), and Low Value Area (LVA) prices for each bar.
    The computation replaces starting zeros with NaN to indicate insufficient data at the window start.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| bars | pd.DataFrame | - | - |
| fp_data | FootprintData | - | - |

**Returns**: `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`


##### compute_range(self, bars: pd.DataFrame, fp_data: FootprintData, start: Union[str, int, pd.Timestamp], end: Union[str, int, pd.Timestamp]) → tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Compute the volume profile (POC, HVA, LVA) in a rolling window fashion for a given time range.

:param bars: DataFrame containing dynamic bars with `high` and `low`.
:param fp_data: FootprintData object containing volume profiles.
:param start: Start timestamp for slicing (str, int, or pd.Timestamp).
:param end: End timestamp for slicing (same type as `start`).
:returns: Tuple of bar timestamps, POC, HVA, and LVA prices.
:raises AssertionError: If `bars` and `fp_data` lengths differ or `start` and `end` are of different types.
:note:
    This method computes the rolling window volume profile (POC, HVA, LVA) for a specified time range.
    The range is set by the `start` and `end` timestamps, which must be of the same type.
    The method internally adjusts the start timestamp for window warm-up and aligns bar and footprint data by timestamp.
    Returns arrays for the bar timestamps and the computed POC, HVA, and LVA prices over the specified range.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| bars | pd.DataFrame | - | - |
| fp_data | FootprintData | - | - |
| start | Union[str, int, pd.Timestamp] | - | - |
| end | Union[str, int, pd.Timestamp] | - | - |

**Returns**: `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`




## Functions

### aggregate_footprint(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray, price_levels: list[np.ndarray], buy_volumes: list[np.ndarray], sell_volumes: list[np.ndarray], start_ts: int, end_ts: int, price_tick: float) → tuple[np.ndarray, np.ndarray, np.ndarray]

Aggregate the volume footprint data within a specified time window to calculate the POC, HVA, and LVA.

:param ts: Array of timestamps (int64) for each bar in nanoseconds.
:param highs: High prices for each bar.
:param lows: Low prices for each bar.
:param price_levels: List of arrays of price levels for each bar.
:param buy_volumes: List of arrays of buy volumes for each price level.
:param sell_volumes: List of arrays of sell volumes for each price level.
:param start_ts: Start timestamp of the aggregation window.
:param end_ts: End timestamp of the aggregation window.
:param price_tick: Tick size used to discretize prices.
:returns: Tuple of:
    - complete_price_levels: Aggregated price levels in integer tick units.
    - aligned_buy_volumes: Aggregated buy volumes aligned to tick grid.
    - aligned_sell_volumes: Aggregated sell volumes aligned to tick grid.
:raises AssertionError: If input lists/arrays are not aligned in length or empty.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ts | np.ndarray | - | - |
| highs | np.ndarray | - | - |
| lows | np.ndarray | - | - |
| price_levels | list[np.ndarray] | - | - |
| buy_volumes | list[np.ndarray] | - | - |
| sell_volumes | list[np.ndarray] | - | - |
| start_ts | int | - | - |
| end_ts | int | - | - |
| price_tick | float | - | - |

**Returns**: `tuple[np.ndarray, np.ndarray, np.ndarray]`



### bucket_price_levels(all_price_levels: np.ndarray, total_volumes: np.ndarray, n_bins: int) → tuple[np.ndarray, np.ndarray]

Bucket the price levels and associated volumes using a fixed-size bin to reduce noise.

:param all_price_levels: Array of all price levels.
:param total_volumes: Corresponding total volumes.
:param n_bins: number of bins to create for bucketing the price levels.
:returns: Tuple of:
    - binned_price_levels: Midpoints of each bucket.
    - binned_volumes: Aggregated volumes per bucket.
:raises AssertionError: If input arrays are empty or of mismatched length.
:raises ValueError: If a price level falls outside defined bins.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| all_price_levels | np.ndarray | - | - |
| total_volumes | np.ndarray | - | - |
| n_bins | int | - | - |

**Returns**: `tuple[np.ndarray, np.ndarray]`



### comp_poc_hva_lva(price_levels: np.ndarray, volumes: np.ndarray, va_pct = 68.34) → tuple[int, int, int]

Calculate the POC (Point of Control), HVA (High Value Area), and LVA (Low Value Area)
for a given volume profile.

:param price_levels: Price levels (sorted ascending).
:param volumes: Corresponding volumes.
:param va_pct: Value area percentage (default 68.34).
:returns: Tuple of:
    - poc_price: Price with the highest volume.
    - hva_price: High value area bound.
    - lva_price: Low value area bound.
:raises AssertionError: If inputs are empty or mismatched in length.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price_levels | np.ndarray | - | - |
| volumes | np.ndarray | - | - |
| va_pct | None | 68.34 | - |

**Returns**: `tuple[int, int, int]`



### calc_volume_percentage_above_poc(price_levels: np.ndarray, volumes: np.ndarray, poc_price: int) → float

Calculate the percentage of volume above the Point of Control (POC) price level.

:param price_levels: Array of price levels.
:param volumes: Corresponding volumes.
:param poc_price: The Point of Control price level.
:returns: Percentage of volume above POC (0-1 range).

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price_levels | np.ndarray | - | - |
| volumes | np.ndarray | - | - |
| poc_price | int | - | - |

**Returns**: `float`



### volume_profile_rolling(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray, price_levels: list[np.ndarray], buy_volumes: list[np.ndarray], sell_volumes: list[np.ndarray], window_size_sec: float, n_bins: int = None, price_tick: float = None, va_pct: float = 68.34) → tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Compute rolling volume profiles over a fixed-width time window.

:param ts: Nanosecond timestamps per bar.
:param highs: High prices per bar.
:param lows: Low prices per bar.
:param price_levels: List of price levels per bar.
:param buy_volumes: List of buy volumes per bar.
:param sell_volumes: List of sell volumes per bar.
:param window_size_sec: Width of the rolling time window.
:param n_bins: Optional number of bins for bucketing price levels.
:param price_tick: Price tick size for discretization.
:param va_pct: Value area percentage.
:returns: Tuple of POC, HVA, LVA, and vp_pct_abv_poc price series aligned to input bars.
:raises AssertionError: If input arrays are empty or misaligned in length.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ts | np.ndarray | - | - |
| highs | np.ndarray | - | - |
| lows | np.ndarray | - | - |
| price_levels | list[np.ndarray] | - | - |
| buy_volumes | list[np.ndarray] | - | - |
| sell_volumes | list[np.ndarray] | - | - |
| window_size_sec | float | - | - |
| n_bins | int | None | - |
| price_tick | float | None | - |
| va_pct | float | 68.34 | - |

**Returns**: `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`



### trim_trailing_zeros(price_levels: np.ndarray, volumes: np.ndarray) → tuple[np.ndarray, np.ndarray]

Trim trailing zero volumes from a price-level volume profile.

:param price_levels: Array of price levels.
:param volumes: Corresponding volumes.
:returns: Trimmed price levels and volumes.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| price_levels | np.ndarray | - | - |
| volumes | np.ndarray | - | - |

**Returns**: `tuple[np.ndarray, np.ndarray]`



### volume_profile_developing(ts: np.ndarray, highs: np.ndarray, lows: np.ndarray, price_levels: list[np.ndarray], buy_volumes: list[np.ndarray], sell_volumes: list[np.ndarray], start_ts: int, end_ts: int, n_bins: int = None, price_tick: float = None, va_pct: float = 68.34) → tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Compute a developing volume profile between two timestamps using cumulative aggregation.

:param ts: Nanosecond timestamps per bar.
:param highs: High prices per bar.
:param lows: Low prices per bar.
:param price_levels: List of price levels per bar.
:param buy_volumes: List of buy volumes per bar.
:param sell_volumes: List of sell volumes per bar.
:param start_ts: Start time in nanoseconds.
:param end_ts: End time in nanoseconds.
:param n_bins: Optional number of bins for bucketing price levels.
:param price_tick: Tick size for price bucketing.
:param va_pct: Value area percentage.
:returns: Tuple of timestamps, POC, HVA, and LVA series for the range.
:raises AssertionError: If input arrays are empty or misaligned in length.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ts | np.ndarray | - | - |
| highs | np.ndarray | - | - |
| lows | np.ndarray | - | - |
| price_levels | list[np.ndarray] | - | - |
| buy_volumes | list[np.ndarray] | - | - |
| sell_volumes | list[np.ndarray] | - | - |
| start_ts | int | - | - |
| end_ts | int | - | - |
| n_bins | int | None | - |
| price_tick | float | None | - |
| va_pct | float | 68.34 | - |

**Returns**: `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`



### comp_flow_acceleration(volumes: NDArray[np.float64], window: int, recent_periods: int) → NDArray[np.float64]

Calculate flow acceleration using Numba

:param volumes: volumes
:param window: window size (eg. 20)
:param recent_periods: Most recent periods to consider for acceleration calculation
:return:

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| volumes | NDArray[np.float64] | - | - |
| window | int | - | - |
| recent_periods | int | - | - |

**Returns**: `NDArray[np.float64]`



### vpin(volume_buy, volume_sell, window)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| volume_buy | None | - | - |
| volume_sell | None | - | - |
| window | None | - | - |

**Returns**: (none)


