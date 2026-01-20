# API Reference: standard_data_structures.py

**Language**: Python

**Source**: `data_structures/standard_data_structures.py`

---

## Classes

### StandardBars

Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
We have added functions to the package such as get_dollar_bars which will create an instance of this
class and then construct the standard bars, to return to the user.

This is because we wanted to simplify the logic as much as possible, for the end user.

**Inherits from**: BaseBars

#### Methods

##### __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000)

Constructor

:param metric: (str) Type of run bar to create. Example: "dollar_run"
:param threshold: (int) Threshold at which to sample
:param batch_size: (int) Number of rows to read in from the csv, per batch

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| metric | str | - | - |
| threshold | int | 50000 | - |
| batch_size | int | 20000000 | - |


##### _reset_cache(self)

Implementation of abstract method _reset_cache for standard bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### _extract_bars(self, data: Union[list, tuple, np.ndarray]) → list

For loop which compiles the various bars: dollar, volume, or tick.
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

### get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None)

Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
properties.

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                        in the format[date_time, price, volume]
:param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                  If a series is given, then at each sampling time the closest previous threshold is used.
                  (Values in the series can only be at times when the threshold is changed, not for every observation)
:param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
:param verbose: (bool) Print out batch numbers (True or False)
:param to_csv: (bool) Save bars to csv after every batch run (True or False)
:param output_path: (str) Path to csv file, if to_csv is True
:return: (pd.DataFrame) Dataframe of dollar bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |
| threshold | Union[float, pd.Series] | 70000000 | - |
| batch_size | int | 20000000 | - |
| verbose | bool | True | - |
| to_csv | bool | False | - |
| output_path | Optional[str] | None | - |

**Returns**: (none)



### get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None)

Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                        in the format[date_time, price, volume]
:param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                  If a series is given, then at each sampling time the closest previous threshold is used.
                  (Values in the series can only be at times when the threshold is changed, not for every observation)
:param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
:param verbose: (bool) Print out batch numbers (True or False)
:param to_csv: (bool) Save bars to csv after every batch run (True or False)
:param output_path: (str) Path to csv file, if to_csv is True
:return: (pd.DataFrame) Dataframe of volume bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |
| threshold | Union[float, pd.Series] | 70000000 | - |
| batch_size | int | 20000000 | - |
| verbose | bool | True | - |
| to_csv | bool | False | - |
| output_path | Optional[str] | None | - |

**Returns**: (none)



### get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000, batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None)

Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

:param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                         in the format[date_time, price, volume]
:param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                  If a series is given, then at each sampling time the closest previous threshold is used.
                  (Values in the series can only be at times when the threshold is changed, not for every observation)
:param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
:param verbose: (bool) Print out batch numbers (True or False)
:param to_csv: (bool) Save bars to csv after every batch run (True or False)
:param output_path: (str) Path to csv file, if to_csv is True
:return: (pd.DataFrame) Dataframe of volume bars

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| file_path_or_df | Union[str, Iterable[str], pd.DataFrame] | - | - |
| threshold | Union[float, pd.Series] | 70000000 | - |
| batch_size | int | 20000000 | - |
| verbose | bool | True | - |
| to_csv | bool | False | - |
| output_path | Optional[str] | None | - |

**Returns**: (none)


