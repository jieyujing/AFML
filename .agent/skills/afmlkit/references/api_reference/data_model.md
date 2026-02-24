# API Reference: data_model.py

**Language**: Python

**Source**: `bar\data_model.py`

---

## Classes

### TradesData

Preprocessor class for raw trades data, designed for efficient bar building and financial analysis.

This class handles standardization of column names, timestamp conversion, trade merging, side inference,
and data validation for consistent processing across different data sources. It serves as the primary
data preparation component for high-frequency trading analysis and bar construction workflows.

In high-frequency trading data, raw trades often come in various formats with inconsistent timestamps,
split trades at the same price level, missing side information, and data integrity issues. This class
addresses these challenges by providing a robust preprocessing pipeline that:

- **Normalizes timestamps** to nanosecond precision for consistent temporal analysis
- **Merges split trades** that occur at identical timestamps and price levels to reduce noise
- **Infers trade sides** (buyer/seller initiated) when not explicitly provided
- **Validates data integrity** by detecting trade ID discontinuities and temporal inconsistencies
- **Provides efficient storage** via compressed HDF5 format with monthly partitioning
- **Enables time-slice queries** with multiprocessing support for large datasets

The preprocessing pipeline follows these steps when ``preprocess=True``:

1. **Timestamp Conversion**: Convert to nanosecond precision from various units (s, ms, μs, ns)
2. **Data Sorting**: Sort by trade ID first to detect gaps, then by timestamp for chronological order
3. **Trade Merging**: Aggregate trades with identical timestamps and prices.

4. **Resolution Processing**: Optionally round timestamps to specified resolution (e.g., millisecond)
5. **Side Inference**: Determine trade direction from price movements when side data is unavailable

The class supports **monthly HDF5 partitioning** for efficient storage and retrieval of large datasets.
Each month's data is stored under ``/trades/YYYY-MM`` with accompanying metadata for fast range queries.
This approach enables handling multi-terabyte datasets while maintaining query performance.

**Data Integrity Monitoring**: The class tracks discontinuities in trade IDs and timestamps, computing
missing data percentages and flagging potential data quality issues. This is crucial for ensuring
reliable downstream analysis.

.. tip::
    For optimal performance with large datasets (>10Gb trades), enable preprocessing and use HDF5 storage
    with compression. The class automatically handles memory-efficient processing via chunking and
    can leverage multiprocessing for data loading operations.

.. note::
    Trade side inference uses price tick rule and other heuristics when explicit side information
    is unavailable. For critical applications, prefer data sources with explicit buyer/seller flags.

Args:
    ts (NDArray): Array of timestamps in various units (s, ms, μs, ns). Must be numeric and monotonic.
    px (NDArray): Array of trade prices as floating-point values.
    qty (NDArray): Array of trade quantities/amounts as floating-point values.
    id (NDArray, optional): Array of unique trade identifiers for data validation. Required if ``preprocess=True``.
    is_buyer_maker (NDArray, optional): Boolean array indicating buyer-maker status (True if buyer is maker).
        If provided, used for accurate side determination.
    side (NDArray, optional): Pre-computed trade side array (-1: sell, 1: buy). Used when loading from HDF5.
    dt_index (pd.DatetimeIndex, optional): Pre-computed datetime index. If None, created from timestamps.
    timestamp_unit (str, optional): Explicit timestamp unit ('s', 'ms', 'us', 'ns'). Auto-inferred if None.
    preprocess (bool, optional): Enable full preprocessing pipeline. Default: False.
    proc_res (str, optional): Target timestamp resolution for rounding ('ms', 'us'). Default: None (no rounding).
    name (str, optional): Instance name for logging purposes. Default: None.

Raises:
    TypeError: If input arrays are not numpy ndarrays or have incompatible types.
    ValueError: If required columns are missing, timestamp format is invalid, or preprocessing fails.

Examples:
    Basic usage with preprocessing:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from afmlkit.bar.data_model import TradesData
    >>> # Raw trades data
    >>> timestamps = np.array([1609459200000, 1609459201000, 1609459202000])  # ms
    >>> prices = np.array([100.0, 100.5, 99.8])
    >>> quantities = np.array([1.5, 2.0, 0.8])
    >>> trade_ids = np.array([1001, 1002, 1003])
    >>>
    >>> # Create TradesData with preprocessing
    >>> trades = TradesData(timestamps, prices, quantities, trade_ids,
    ...                     timestamp_unit='ms', preprocess=True, name='BTCUSD')
    >>> print(f"Processed {len(trades.data)} trades")
    Processed 3 trades

    Loading from HDF5 with time filtering:

    >>> # doctest: +SKIP
    >>> # Load specific time range with multiprocessing
    >>> trades = TradesData.load_trades_h5('trades.h5',
    ...                                     start_time='2021-01-01',
    ...                                     end_time='2021-01-31',
    ...                                     enable_multiprocessing=True)
    >>> trades.set_view_range('2021-01-15', '2021-01-20')
    >>> subset = trades.data  # Only data in view range

See Also:
    :class:`afmlkit.bar.base.BarBuilderBase`: Uses TradesData for constructing various bar types.
    :func:`afmlkit.bar.utils.merge_split_trades`: Core function for trade aggregation.
    :func:`afmlkit.bar.utils.comp_trade_side_vector`: Trade side inference algorithm.

References:
    .. _`HDF5 for High-Frequency Trading`: https://www.hdfgroup.org/
    .. _`Market Microstructure in Practice`: https://www.cambridge.org/core/books/market-microstructure-in-practice/

**Inherits from**: (none)

#### Methods

##### __init__(self, ts: NDArray, px: NDArray, qty: NDArray, id: NDArray = None)

Initialize the TradesData with raw trades data.

:param ts: array of timestamps
:param px: array of prices
:param qty: array of quantity or amount of trades
:param id: array of trades id
:param is_buyer_maker: Optional Array of side info: True if buyer maker, False otherwise. If None side information will be inferred from data.
:param side: Optional Array Market order side information (-1: sell, 1: buy) [needed when loading from HDF5 store].
:param dt_index: Optional DatetimeIndex for the trades data. If provided, it will be used as the index.  [needed when loading from HDF5 store].
:param timestamp_unit: (Optional) timestamp unit (e.g., 'ms', 'us', 'ns'); inferred if None.
:param proc_res: (Optional) processing resolution for timestamps (e.g., 'ms' cuts us to ms resolution).
:param preprocess: If True, runs the preprocessing pipeline (sorting, merging split trades etc...)
:param name: Optional name for the trades data instance (logging purposes).
:raises ValueError: If required columns are missing or timestamp format is invalid.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| ts | NDArray | - | - |
| px | NDArray | - | - |
| qty | NDArray | - | - |
| id | NDArray | None | - |


##### start_date(self)

Get the start date of the trades data.

:return: Start date as a pandas Timestamp.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### end_date(self)

Get the end date of the trades data.

:return: End date as a pandas Timestamp.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### set_view_range(self, start: pd.Timestamp | str, end: pd.Timestamp | str)

Set the active view range for data access, enabling efficient time-slice analysis.

:param start: Start timestamp for the view range. Accepts string or pd.Timestamp.
:param end: End timestamp for the view range. Accepts string or pd.Timestamp.
:raises ValueError: If start timestamp is not before end pd.timestamp.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| start | pd.Timestamp | str | - | - |
| end | pd.Timestamp | str | - | - |


##### data(self) → pd.DataFrame

Get the processed trades data as a DataFrame, respecting the active view range.

Returns the full dataset if no view range is set, or a time-filtered subset otherwise.
The DataFrame includes columns: timestamp, price, amount, and optionally side.

:return: DataFrame containing trades data with datetime index.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `pd.DataFrame`


##### orig_timestamp_unit(self) → str

Get the timestamp unit used for processing.

:return: Timestamp unit string.

**Decorators**: `@property`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`


##### _validate_data(self)

Check for gaps in trade IDs

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _sort_trades(self) → None

Sort trades by timestamp to ensure correct order for processing.
Also performs data integrity checks by identifying discontinuities in trade IDs.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `None`


##### _merge_trades(self)

Merge trades that occur at the same timestamp and price level.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _convert_timestamps_to_ns(self)

Convert timestamps to nanosecond representation.
:raises ValueError: If timestamp format is invalid.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _apply_timestamp_resolution(self, proc_res: Optional[str]) → None

Apply processing resolution to timestamps if specified.

:param proc_res: Target processing resolution for timestamps.
:raises ValueError: If processing resolution is invalid.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| proc_res | Optional[str] | - | - |

**Returns**: `None`


##### _add_trade_side_info(self) → None

Extract trade side information from the trades data.

:returns: None - modifies the trades DataFrame in place to include a 'side' column.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `None`


##### _infer_timestamp_unit(self) → str

Infer the unit of timestamps in the trades data if not explicitly provided.
:return: Inferred or provided timestamp unit.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`


##### save_h5(self, filepath: str) → str

Persist trades data to HDF5 format with monthly partitioning and compression.

Stores data under ``/trades/YYYY-MM`` groups for efficient range queries. Each month
includes metadata for fast discovery and data integrity information when available.

:param filepath: Destination HDF5 file path. Parent directories created automatically.
:param month_key: Override automatic monthly key derivation (format: "YYYY-MM").
:param complib: Compression library ("blosc:lz4", "blosc:zstd", "zlib"). Default: "blosc:lz4".
:param complevel: Compression level (0-9). Higher values increase compression ratio. Default: 1.
:param mode: File access mode ("a" for append, "w" for overwrite). Default: "a".
:param chunksize: Row chunk size for writing large datasets. Default: 1,000,000.
:param overwrite_month: Prompt for confirmation when overwriting existing monthly data. Default: True.
:returns: Full HDF5 key path used for storage (e.g., "/trades/2021-03").
:raises ValueError: If user declines to overwrite existing data or if data format is invalid.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| filepath | str | - | - |

**Returns**: `str`


##### _keys_for_timerange(cls, store: pd.HDFStore, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) → list[str]

Internal helper – determine which monthly groups intersect the
*[start, end]* interval by consulting the per‑group metadata.

**Decorators**: `@classmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| cls | None | - | - |
| store | pd.HDFStore | - | - |
| start | Optional[pd.Timestamp] | - | - |
| end | Optional[pd.Timestamp] | - | - |

**Returns**: `list[str]`


##### load_trades_h5(cls, filepath: str) → 'TradesData'

Load trades from HDF5 storage with optional multiprocessing and time filtering.

Supports three loading modes:

    1. **Single month**: Load specific monthly partition using ``key`` parameter
    2. **Time range**: Auto-discover monthly groups intersecting ``[start_time, end_time]``
    3. **Filtered month**: Combine ``key`` with time range for constrained loading

Multiprocessing is automatically enabled for loading multiple monthly groups,
significantly improving performance for large time ranges.

:param filepath: Path to HDF5 file containing trades data.
:param key: Specific monthly key to load (e.g., "2021-03"). If None, uses time range discovery.
:param start_time: Start time for filtering (string or Timestamp). None for no start limit.
:param end_time: End time for filtering (string or Timestamp). None for no end limit.
:param n_workers: Number of worker processes. If None, uses CPU count - 1.
:param enable_multiprocessing: Enable parallel loading for multiple groups. Default: True.
:param min_groups_for_mp: Minimum groups required to trigger multiprocessing. Default: 2.
:returns: TradesData instance with loaded and concatenated data.
:raises KeyError: If specified key doesn't exist or no groups match the time range.
:raises ValueError: If no data is successfully loaded from any group.

Examples:
    Load specific month:

    >>> # doctest: +SKIP
    >>> trades = TradesData.load_trades_h5('data.h5', key='2021-03')

    Load time range with multiprocessing:

    >>> # doctest: +SKIP
    >>> trades = TradesData.load_trades_h5('data.h5',
    ...                                     start_time='2021-01-01',
    ...                                     end_time='2021-12-31',
    ...                                     n_workers=4)

**Decorators**: `@classmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| cls | None | - | - |
| filepath | str | - | - |

**Returns**: `'TradesData'`




### FootprintData

Container for dynamic memory footprint calculations including trade volumes, price levels, and imbalance information.

:param bar_timestamps: Timestamps of each bar in nanoseconds.
:param price_tick: Price tick size.
:param price_levels: Array of price levels per bar.
:param buy_volumes: Buy volumes per price level.
:param sell_volumes: Sell volumes per price level.
:param buy_ticks: Number of buy ticks per price level.
:param sell_ticks: Number of sell ticks per price level.
:param buy_imbalances: Buy imbalance flags per price level.
:param sell_imbalances: Sell imbalance flags per price level.
:param cot_price_levels: Optional Commitment of Traders price levels.
:param sell_imbalances_sum: Optional total sell imbalance counts per bar.
:param buy_imbalances_sum: Optional total buy imbalance counts per bar.
:param imb_max_run_signed: Optional longest signed imbalance run for each bar.
:param vp_skew: Optional volume profile skew for each bar (positive = buy pressure above VWAP).
:param vp_gini: Optional volume profile Gini coefficient for each bar (0 = concentrated, →1 = even).

**Inherits from**: (none)

#### Methods

##### __post_init__(self)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### __len__(self) → int

Return the number of bars in the data.
:returns: Number of bars.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `int`


##### __repr__(self) → str

Generate a summary string representation for debugging.
:returns: Formatted string summary.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`


##### __getitem__(self, key) → 'FootprintData'

Support slicing or indexing of the footprint data.
:param key: Slice, integer index, or datetime range.
:returns: New FootprintData object with selected range.
:raises TypeError: If key is not a supported type.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| key | None | - | - |

**Returns**: `'FootprintData'`


##### from_numba(cls, data: Tuple, price_tick: float) → 'FootprintData'

Create a FootprintData object from Numba-based output.
:param data: Output tuple from comp_bar_footprint.
:param price_tick: Tick size for price levels.
:returns: A validated FootprintData instance.
:raises ValueError: If data length is inconsistent.

**Decorators**: `@classmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| cls | None | - | - |
| data | Tuple | - | - |
| price_tick | float | - | - |

**Returns**: `'FootprintData'`


##### get_df(self)

Convert the footprint data into a pandas DataFrame.
:returns: A DataFrame with structured footprint information.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### cast_to_numba_list(self)

Convert internal arrays to NumbaList for JIT-compatible processing.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### cast_to_numpy(self)

Convert internal lists to NumPy arrays for general-purpose processing.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### memory_usage(self)

Calculate the approximate memory usage of this object in MB.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### is_valid(self) → bool

Check if all internal arrays are consistent.
:returns: True if valid, False otherwise.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `bool`




## Functions

### _load_single_h5_group(args: Tuple[str, str, Optional[str]]) → pd.DataFrame

Helper function to load a single HDF5 group in a separate process.

:param args: Tuple of (filepath, h5_key, where_clause)
:returns: DataFrame with the loaded data

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| args | Tuple[str, str, Optional[str]] | - | - |

**Returns**: `pd.DataFrame`



### _is_notebook_environment() → bool

Detect if we're running in a Jupyter notebook environment.

:returns: True if in notebook, False otherwise

**Returns**: `bool`


