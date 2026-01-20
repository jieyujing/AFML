# API Reference: time_data_structures.py

**Language**: Python

**Source**: `data_structures/time_data_structures.py`

---

## Classes

### TimeBars

Contains all of the logic to construct the time bars. This class shouldn't be used directly.
Use get_time_bars instead

**Inherits from**: BaseBars

#### Methods

##### __init__(self, resolution: str, num_units: int, batch_size: int = 20000000)

Constructor

:param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']
:param num_units: (int) Number of days, minutes, etc.
:param batch_size: (int) Number of rows to read in from the csv, per batch

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| resolution | str | - | - |
| num_units | int | - | - |
| batch_size | int | 20000000 | - |


##### _reset_cache(self)

Implementation of abstract method _reset_cache for time bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _extract_bars(self, data: Union[list, tuple, np.ndarray]) → list

For loop which compiles time bars.
We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

:param data: (tuple) Contains 3 columns - date_time, price, and volume.
:return: (list) Extracted bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| data | Union[list, tuple, np.ndarray] | - | - |

**Returns**: `list`




## Functions

### get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = 'D', num_units: int = 1, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None)

Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                        in the format[date_time, price, volume]
:param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
:param num_units: (int) Number of resolution units (3 days for example, 2 hours)
:param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
:param verbose: (int) Print out batch numbers (True or False)
:param to_csv: (bool) Save bars to csv after every batch run (True or False)
:param output_path: (str) Path to csv file, if to_csv is True
:return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |
| resolution | str | 'D' | - |
| num_units | int | 1 | - |
| batch_size | int | 20000000 | - |
| verbose | bool | True | - |
| to_csv | bool | False | - |
| output_path | Optional[str] | None | - |

**Returns**: (none)


