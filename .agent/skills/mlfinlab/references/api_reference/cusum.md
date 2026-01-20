# API Reference: cusum.py

**Language**: Python

**Source**: `structural_breaks/cusum.py`

---

## Functions

### _get_values_diff(test_type, series, index, ind)

Gets the difference between two values given a test type.
:param test_type: (str) Type of the test ['one_sided', 'two_sided']
:param series: (pd.Series) Series of values
:param index: (pd.Index) primary index
:param ind: (pd.Index) secondary index
:return: (float) Difference between 2 values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| test_type | None | - | - |
| series | None | - | - |
| index | None | - | - |
| ind | None | - | - |

**Returns**: (none)



### _get_s_n_for_t(series: pd.Series, test_type: str, molecule: list) → pd.Series

Get maximum S_n_t value for each value from molecule for Chu-Stinchcombe-White test

:param series: (pd.Series) Series to get statistics for
:param test_type: (str): Two-sided or one-sided test
:param molecule: (list) Indices to get test statistics for
:return: (pd.Series) Statistics

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| series | pd.Series | - | - |
| test_type | str | - | - |
| molecule | list | - | - |

**Returns**: `pd.Series`



### get_chu_stinchcombe_white_statistics(series: pd.Series, test_type: str = 'one_sided', num_threads: int = 8, verbose: bool = True) → pd.Series

Multithread Chu-Stinchcombe-White test implementation, p.251

:param series: (pd.Series) Series to get statistics for
:param test_type: (str): Two-sided or one-sided test
:param num_threads: (int) Number of cores
:param verbose: (bool) Flag to report progress on asynch jobs
:return: (pd.Series) Statistics

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| series | pd.Series | - | - |
| test_type | str | 'one_sided' | - |
| num_threads | int | 8 | - |
| verbose | bool | True | - |

**Returns**: `pd.Series`


