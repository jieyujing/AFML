# API Reference: base_bars.py

**Language**: Python

**Source**: `data_structures/base_bars.py`

---

## Classes

### BaseBars

Abstract base class which contains the structure which is shared between the various standard and information
driven bars. There are some methods contained in here that would only be applicable to information bars but
they are included here so as to avoid a complicated nested class structure.

**Inherits from**: ABC

#### Methods

##### __init__(self, metric: str, batch_size: int = 20000000.0)

Constructor

:param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
:param batch_size: (int) Number of rows to read in from the csv, per batch.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| metric | str | - | - |
| batch_size | int | 20000000.0 | - |


##### batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None) → Union[pd.DataFrame, None]

Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in the form of a DataFrame.
The csv file or DataFrame must have only 3 columns: date_time, price, & volume.

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                        raw tick data  in the format[date_time, price, volume]
:param verbose: (bool) Flag whether to print message on each processed batch or not
:param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
:param output_path: (bool) Path to results file, if to_csv = True

:return: (pd.DataFrame or None) Financial data structure

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |
| verbose | bool | True | - |
| to_csv | bool | False | - |
| output_path | Optional[str] | None | - |

**Returns**: `Union[pd.DataFrame, None]`


##### _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) → Generator[pd.DataFrame, None, None]

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame
                        containing raw tick data in the format[date_time, price, volume]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |

**Returns**: `Generator[pd.DataFrame, None, None]`


##### _read_first_row(self, file_path: str)

:param file_path: (str) Path to the csv file containing raw tick data in the format[date_time, price, volume]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| file_path | str | - | - |


##### run(self, data: Union[list, tuple, pd.DataFrame]) → list

Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.
The List, Tuple, or DataFrame must have only 3 attrs: date_time, price, & volume.

:param data: (list, tuple, or pd.DataFrame) Dict or ndarray containing raw tick data in the format[date_time, price, volume]

:return: (list) Financial data structure

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | Union[list, tuple, pd.DataFrame] | - | - |

**Returns**: `list`


##### _extract_bars(self, data: pd.DataFrame) → list

This method is required by all the bar types and is used to create the desired bars.

:param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
:return: (list) Bars built using the current batch.

**Decorators**: `@abstractmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | pd.DataFrame | - | - |

**Returns**: `list`


##### _reset_cache(self)

This method is required by all the bar types. It describes how cache should be reset
when new bar is sampled.

**Decorators**: `@abstractmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _assert_csv(test_batch: pd.DataFrame)

Tests that the csv file read has the format: date_time, price, and volume.
If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

:param test_batch: (pd.DataFrame) The first row of the dataset.

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| test_batch | pd.DataFrame | - | - |


##### _update_high_low(self, price: float) → Union[float, float]

Update the high and low prices using the current price.

:param price: (float) Current price
:return: (tuple) Updated high and low prices

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |

**Returns**: `Union[float, float]`


##### _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) → None

Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, volume,
cum_buy_volume, cum_ticks, cum_dollar_value.
These bars are appended to list_bars, which is later used to construct the final bars DataFrame.

:param date_time: (str) Timestamp of the bar
:param price: (float) The current price
:param high_price: (float) Highest price in the period
:param low_price: (float) Lowest price in the period
:param list_bars: (list) List to which we append the bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| date_time | str | - | - |
| price | float | - | - |
| high_price | float | - | - |
| low_price | float | - | - |
| list_bars | list | - | - |

**Returns**: `None`


##### _apply_tick_rule(self, price: float) → int

Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.

:param price: (float) Price at time t
:return: (int) The signed tick

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |

**Returns**: `int`


##### _get_imbalance(self, price: float, signed_tick: int, volume: float) → float

Advances in Financial Machine Learning, page 29.

Get the imbalance at a point in time, denoted as Theta_t

:param price: (float) Price at t
:param signed_tick: (int) signed tick, using the tick rule
:param volume: (float) Volume traded at t
:return: (float) Imbalance at time t

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |
| signed_tick | int | - | - |
| volume | float | - | - |

**Returns**: `float`




### BaseImbalanceBars

Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic

**Inherits from**: BaseBars

#### Methods

##### __init__(self, metric: str, batch_size: int, expected_imbalance_window: int, exp_num_ticks_init: int, analyse_thresholds: bool)

Constructor

:param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
:param batch_size: (int) Number of rows to read in from the csv, per batch.
:param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
:param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                 For Const Imbalance Bars expected number of ticks equals expected number of ticks init
:param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a
                                  form of Pandas DataFrame

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| metric | str | - | - |
| batch_size | int | - | - |
| expected_imbalance_window | int | - | - |
| exp_num_ticks_init | int | - | - |
| analyse_thresholds | bool | - | - |


##### _reset_cache(self)

Implementation of abstract method _reset_cache for imbalance bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _extract_bars(self, data: Tuple[dict, pd.DataFrame]) → list

For loop which compiles the various imbalance bars: dollar, volume, or tick.

:param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
:return: (list) Bars built using the current batch.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | Tuple[dict, pd.DataFrame] | - | - |

**Returns**: `list`


##### _get_expected_imbalance(self, window: int)

Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
:param window: (int) EWMA window for calculation
:return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| window | int | - | - |


##### _get_exp_num_ticks(self)

Abstract method which updates expected number of ticks when new run bar is formed

**Decorators**: `@abstractmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




### BaseRunBars

Base class for Run Bars (EMA and Const) which implements run bars calculation logic

**Inherits from**: BaseBars

#### Methods

##### __init__(self, metric: str, batch_size: int, num_prev_bars: int, expected_imbalance_window: int, exp_num_ticks_init: int, analyse_thresholds: bool)

Constructor

:param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
:param batch_size: (int) Number of rows to read in from the csv, per batch.
:param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
:param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                 For Const Imbalance Bars expected number of ticks equals expected number of ticks init
:param analyse_thresholds: (bool) Flag to return thresholds values (thetas, exp_num_ticks, exp_runs) in Pandas DataFrame

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| metric | str | - | - |
| batch_size | int | - | - |
| num_prev_bars | int | - | - |
| expected_imbalance_window | int | - | - |
| exp_num_ticks_init | int | - | - |
| analyse_thresholds | bool | - | - |


##### _reset_cache(self)

Implementation of abstract method _reset_cache for imbalance bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _extract_bars(self, data: Tuple[list, np.ndarray]) → list

For loop which compiles the various run bars: dollar, volume, or tick.

:param data: (list or np.ndarray) Contains 3 columns - date_time, price, and volume.
:return: (list) of bars built using the current batch.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | Tuple[list, np.ndarray] | - | - |

**Returns**: `list`


##### _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False)

Advances in Financial Machine Learning, page 29.

Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.

:param array: (list) of imbalances
:param window: (int) EWMA window for calculation
:parawm warm_up: (bool) flag of whether warm up period passed
:return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| array | list | - | - |
| window | int | - | - |
| warm_up | bool | False | - |


##### _get_exp_num_ticks(self)

Abstract method which updates expected number of ticks when new imbalance bar is formed

**Decorators**: `@abstractmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




## Functions

### _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) → list

Splits df into chunks of chunksize

:param df: (pd.DataFrame) Dataframe to split
:param chunksize: (int) Number of rows in chunk
:return: (list) Chunks (pd.DataFrames)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| df | pd.DataFrame | - | - |
| chunksize | int | - | - |

**Returns**: `list`


