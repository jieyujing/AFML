# API Reference: feature_generator.py

**Language**: Python

**Source**: `microstructural_features/feature_generator.py`

---

## Classes

### MicrostructuralFeaturesGenerator

Class which is used to generate inter-bar features when bars are already compressed.

:param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                           in the format[date_time, price, volume]
:param tick_num_series: (pd.Series) Series of tick number where bar was formed.
:param batch_size: (int) Number of rows to read in from the csv, per batch.
:param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
:param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages

**Inherits from**: (none)

#### Methods

##### __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 20000000.0, volume_encoding: dict = None, pct_encoding: dict = None)

Constructor

:param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                           in the format[date_time, price, volume]
:param tick_num_series: (pd.Series) Series of tick number where bar was formed.
:param batch_size: (int) Number of rows to read in from the csv, per batch.
:param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
:param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| trades_input | (str, pd.DataFrame) | - | - |
| tick_num_series | pd.Series | - | - |
| batch_size | int | 20000000.0 | - |
| volume_encoding | dict | None | - |
| pct_encoding | dict | None | - |


##### get_features(self, verbose = True, to_csv = False, output_path = None)

Reads a csv file of ticks or pd.DataFrame in batches and then constructs corresponding microstructural intra-bar features:
average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, tick/volume/pct Shannon, Lempel-Ziv,
Plug-in entropies if corresponding mapping dictionaries are provided (self.volume_encoding, self.pct_encoding).
The csv file must have only 3 columns: date_time, price, & volume.

:param verbose: (bool) Flag whether to print message on each processed batch or not
:param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
:param output_path: (bool) Path to results file, if to_csv = True
:return: (DataFrame or None) Microstructural features for bar index

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| verbose | None | True | - |
| to_csv | None | False | - |
| output_path | None | None | - |


##### _reset_cache(self)

Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
calculated

:return: None

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _extract_bars(self, data)

For loop which calculates features for formed bars using trades data

:param data: (tuple) Contains 3 columns - date_time, price, and volume.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | None | - | - |


##### _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) → list

Calculate inter-bar features: lambdas, entropies, avg_tick_size, vwap

:param date_time: (pd.Timestamp) When bar was formed
:param list_bars: (list) Previously formed bars
:return: (list) Inter-bar features

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| date_time | pd.Timestamp | - | - |
| list_bars | list | - | - |

**Returns**: `list`


##### _apply_tick_rule(self, price: float) → int

Advances in Financial Machine Learning, page 29.

Applies the tick rule

:param price: (float) Price at time t
:return: (int) The signed tick

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |

**Returns**: `int`


##### _get_price_diff(self, price: float) → float

Get price difference between ticks

:param price: (float) Price at time t
:return: (float) Price difference

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |

**Returns**: `float`


##### _get_log_ret(self, price: float) → float

Get log return between ticks

:param price: (float) Price at time t
:return: (float) Log return

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| price | float | - | - |

**Returns**: `float`


##### _assert_csv(test_batch)

Tests that the csv file read has the format: date_time, price, and volume.
If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

:param test_batch: (pd.DataFrame) the first row of the dataset.
:return: (None)

**Decorators**: `@staticmethod`

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| test_batch | None | - | - |



