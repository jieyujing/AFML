# Test Example Extraction Report

**Total Examples**: 614  
**High Value Examples** (confidence > 0.7): 614  
**Average Complexity**: 0.31  

## Examples by Category

- **instantiation**: 264
- **method_call**: 350

## Examples by Language

- **Python**: 614

## Extracted Examples

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(len(gaps_diff_no_backward.unique()) == len(roll_dates) + 1)
self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:64*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_diff_no_backward.iloc[0] == 0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)
self.assertTrue(gaps_diff_no_backward.iloc[0] == 0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:65*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_diff_no_backward.iloc[-1] == -1.75)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(gaps_diff_no_backward.iloc[0] == 0)
self.assertTrue(gaps_diff_no_backward.iloc[-1] == -1.75)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:68*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_diff_with_backward.iloc[0] == 1.75)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(gaps_diff_no_backward.iloc[-1] == -1.75)
self.assertTrue(gaps_diff_with_backward.iloc[0] == 1.75)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:69*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_diff_with_backward.iloc[-1] == 0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(gaps_diff_with_backward.iloc[0] == 1.75)
self.assertTrue(gaps_diff_with_backward.iloc[-1] == 0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:70*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_rel_no_backward.iloc[0] == 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(gaps_diff_with_backward.iloc[-1] == 0)
self.assertTrue(gaps_rel_no_backward.iloc[0] == 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:71*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(abs(gaps_rel_no_backward.iloc[-1] - 0.999294) < 1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(gaps_rel_no_backward.iloc[0] == 1)
self.assertTrue(abs(gaps_rel_no_backward.iloc[-1] - 0.999294) < 1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:74*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(abs(gaps_rel_with_backward.iloc[0] - 1 / 0.999294) < 1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(abs(gaps_rel_no_backward.iloc[-1] - 0.999294) < 1e-06)
self.assertTrue(abs(gaps_rel_with_backward.iloc[0] - 1 / 0.999294) < 1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:75*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(gaps_rel_with_backward.iloc[-1] == 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])
self.open_df.rename(columns={'spx': 'open'}, inplace=True)
self.close_df.rename(columns={'spx': 'close'}, inplace=True)

self.assertTrue(abs(gaps_rel_with_backward.iloc[0] - 1 / 0.999294) < 1e-06)
self.assertTrue(gaps_rel_with_backward.iloc[-1] == 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:76*

### test_futures_roll

**Category**: method_call  
**Description**: Tests get_futures_roll function  
**Expected**: self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(len(gaps_diff_no_backward.unique()) == len(roll_dates) + 1)
self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_futures_roll.py:64*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.shape == csv_trick_series_100.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.shape == csv_trick_series_4.shape)
self.assertTrue(in_memory_trick_series.shape == csv_trick_series_100.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:55*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.shape == csv_trick_series_all.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.shape == csv_trick_series_100.shape)
self.assertTrue(in_memory_trick_series.shape == csv_trick_series_all.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:56*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(abs(in_memory_trick_series.iloc[20] - 0.9933502) < 1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.shape == csv_trick_series_all.shape)
self.assertTrue(abs(in_memory_trick_series.iloc[20] - 0.9933502) < 1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:57*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.iloc[0] == 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(abs(in_memory_trick_series.iloc[20] - 0.9933502) < 1e-06)
self.assertTrue(in_memory_trick_series.iloc[0] == 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:60*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(csv_trick_series_4.iloc[0] == 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.iloc[0] == 1.0)
self.assertTrue(csv_trick_series_4.iloc[0] == 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:63*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(csv_trick_series_100.iloc[0] == 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(csv_trick_series_4.iloc[0] == 1.0)
self.assertTrue(csv_trick_series_100.iloc[0] == 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:64*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(csv_trick_series_all.iloc[0] == 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(csv_trick_series_100.iloc[0] == 1.0)
self.assertTrue(csv_trick_series_all.iloc[0] == 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:65*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_4.iloc[-1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(csv_trick_series_all.iloc[0] == 1.0)
self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_4.iloc[-1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:66*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_100.iloc[-1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_4.iloc[-1])
self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_100.iloc[-1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:69*

### test_etf_trick_costs_defined

**Category**: method_call  
**Description**: Tests in-memory and csv ETF trick implementation, when costs_df is defined  
**Expected**: self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_all.iloc[-1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
path = project_path + '/test_data'
self.open_df_path = '{}/open_df.csv'.format(path)
self.close_df_path = '{}/close_df.csv'.format(path)
self.alloc_df_path = '{}/alloc_df.csv'.format(path)
self.costs_df_path = '{}/costs_df.csv'.format(path)
self.rates_df_path = '{}/rates_df.csv'.format(path)
self.open_df = pd.read_csv(self.open_df_path, index_col=0, parse_dates=[0])
self.close_df = pd.read_csv(self.close_df_path, index_col=0, parse_dates=[0])
self.alloc_df = pd.read_csv(self.alloc_df_path, index_col=0, parse_dates=[0])
self.costs_df = pd.read_csv(self.costs_df_path, index_col=0, parse_dates=[0])
self.rates_df = pd.read_csv(self.rates_df_path, index_col=0, parse_dates=[0])

self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_100.iloc[-1])
self.assertTrue(in_memory_trick_series.iloc[-1] == csv_trick_series_all.iloc[-1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_etf_trick.py:70*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_frame_equal(test2_actual, test2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_series_equal(test1_actual, test1, check_names=False)
pd.testing.assert_frame_equal(test2_actual, test2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:47*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_frame_equal(test3_actual, test3)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test2_actual, test2)
pd.testing.assert_frame_equal(test3_actual, test3)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:48*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_frame_equal(test4_actual, test4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test3_actual, test3)
pd.testing.assert_frame_equal(test4_actual, test4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:49*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_series_equal(test5_actual, test5, check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test4_actual, test4)
pd.testing.assert_series_equal(test5_actual, test5, check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:50*

### test_dynamic_threshold

**Category**: method_call  
**Description**: Tests for when threshold is a pd.Series rather than a constant.  
**Expected**: pd.testing.assert_series_equal(test7_actual, test7, check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test6_actual, test6)
pd.testing.assert_series_equal(test7_actual, test7, check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:66*

### test_with_standardization

**Category**: method_call  
**Description**: Test cases with standardization, with constant and dynamic threshold.  
**Expected**: pd.testing.assert_frame_equal(test9_actual, test9)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test8_actual, test8)
pd.testing.assert_frame_equal(test9_actual, test9)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:92*

### test_with_standardization

**Category**: method_call  
**Description**: Test cases with standardization, with constant and dynamic threshold.  
**Expected**: pd.testing.assert_frame_equal(test10_actual, test10)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test9_actual, test9)
pd.testing.assert_frame_equal(test10_actual, test10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:93*

### test_resample

**Category**: method_call  
**Description**: Tests for when a resample period is used.  
**Expected**: pd.testing.assert_frame_equal(test12_actual, test12)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test11_actual, test11)
pd.testing.assert_frame_equal(test12_actual, test12)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:118*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_frame_equal(test2_actual, test2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test1_actual, test1, check_names=False)
pd.testing.assert_frame_equal(test2_actual, test2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:47*

### test_basic

**Category**: method_call  
**Description**: Tests for basic case, constant threshold and no standardization, lag.  
**Expected**: pd.testing.assert_frame_equal(test3_actual, test3)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test2_actual, test2)
pd.testing.assert_frame_equal(test3_actual, test3)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_fixed_time_horizon.py:48*

### test_daily_volatility

**Category**: method_call  
**Description**: Daily vol as implemented here matches the code in the book.
Although I have reservations, example: no minimum value is set in the EWM.
Thus it returns values for volatility before there are even enough data points.  
**Expected**: self.assertTrue(daily_vol.shape[0] == 960)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(daily_vol[-1] == 0.008968238932170641)
self.assertTrue(daily_vol.shape[0] == 960)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:39*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(triple_barrier_events.iloc[0, 1] == 0.010166261175903357)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(triple_barrier_events.shape == (8, 4))
self.assertTrue(triple_barrier_events.iloc[0, 1] == 0.010166261175903357)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:113*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(triple_barrier_events.iloc[-1, 1] == 0.006455887663302871)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(triple_barrier_events.iloc[0, 1] == 0.010166261175903357)
self.assertTrue(triple_barrier_events.iloc[-1, 1] == 0.006455887663302871)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:116*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(np.all(triple_barrier_events.index == cusum_events[1:]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(triple_barrier_events.iloc[-1, 1] == 0.006455887663302871)
self.assertTrue(np.all(triple_barrier_events.index == cusum_events[1:]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:117*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(np.all(meta_labeled_events['trgt'] == triple_barrier_events['trgt']))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(np.all(meta_labeled_events['t1'] == triple_barrier_events['t1']))
self.assertTrue(np.all(meta_labeled_events['trgt'] == triple_barrier_events['trgt']))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:136*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(meta_labeled_events.shape == (8, 5))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(np.all(meta_labeled_events['trgt'] == triple_barrier_events['trgt']))
self.assertTrue(meta_labeled_events.shape == (8, 5))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:137*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue(no_vertical_events.shape == (8, 4))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(np.all(triple_barrier_events['trgt'] == no_vertical_events['trgt']))
self.assertTrue(no_vertical_events.shape == (8, 4))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:155*

### test_triple_barrier_events

**Category**: method_call  
**Description**: Assert that the different version of triple barrier labeling match our expected output.
Assert that trgts are the same for all 3 methods.  
**Expected**: self.assertTrue((no_vertical_events['t1'] != triple_barrier_events['t1']).sum() == 2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(no_vertical_events.shape == (8, 4))
self.assertTrue((no_vertical_events['t1'] != triple_barrier_events['t1']).sum() == 2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:156*

### test_triple_barrier_labeling

**Category**: method_call  
**Description**: Assert that meta labeling as well as standard labeling works. Also check that if a vertical barrier is
reached, then a 0 class label is assigned (in the case of standard labeling).  
**Expected**: self.assertTrue(triple_labels.shape == (8, 4))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue((condition1 & condition2 == triple_labels['bin']).all())
self.assertTrue(triple_labels.shape == (8, 4))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:202*

### test_pt_sl_levels_triple_barrier_events

**Category**: method_call  
**Description**: Previously a bug was introduced by not multiplying the target by the profit taking / stop loss multiple. This
meant that the get_bins function would not return the correct label. Example: if take profit was set to 1000,
it would ignore this multiple and use only the target value. This meant that if we set a very large pt value
(so high that it would never be hit before the vertical barrier is reached), it would ignore the multiple and
only use the target value (it would signal that price reached the pt barrier). This meant that vertical barriers
were incorrectly labeled.

This also meant that irrespective of the pt_sl levels set, the labels would always be the same.  
**Expected**: self.assertTrue(np.all(labels_small[0:5] != labels_large[0:5]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(np.all(labels_no_ones < 1))
self.assertTrue(np.all(labels_small[0:5] != labels_large[0:5]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labels.py:272*

### test_simulate_covariance

**Category**: method_call  
**Description**: Test the deriving an empirical vector of means and an empirical covariance matrix.  
**Expected**: np.testing.assert_almost_equal(cov_mat, cov_empir, decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(mu_empir.flatten(), mu_vec.flatten(), decimal=2)
np.testing.assert_almost_equal(cov_mat, cov_empir, decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:42*

### test_simulate_covariance

**Category**: method_call  
**Description**: Test the deriving an empirical vector of means and an empirical covariance matrix.  
**Expected**: np.testing.assert_almost_equal(cov_mat, cov_empir_shr, decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(mu_empir_shr.flatten(), mu_vec.flatten(), decimal=2)
np.testing.assert_almost_equal(cov_mat, cov_empir_shr, decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:47*

### test_cluster_kmeans_base

**Category**: method_call  
**Description**: Test the finding of the optimal partition of clusters using K-Means algorithm.  
**Expected**: np.testing.assert_almost_equal(np.array(corr), np.array(expected_corr), decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

self.assertTrue(clusters == expected_clust)
np.testing.assert_almost_equal(np.array(corr), np.array(expected_corr), decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:80*

### test_cluster_kmeans_base

**Category**: method_call  
**Description**: Test the finding of the optimal partition of clusters using K-Means algorithm.  
**Expected**: np.testing.assert_almost_equal(np.array(silh_coef), np.array(expected_silh_coef), decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(np.array(corr), np.array(expected_corr), decimal=4)
np.testing.assert_almost_equal(np.array(silh_coef), np.array(expected_silh_coef), decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:81*

### test_cluster_kmeans_base

**Category**: method_call  
**Description**: Test the finding of the optimal partition of clusters using K-Means algorithm.  
**Expected**: np.testing.assert_almost_equal(np.array(corr), np.array(corr_no_max), decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(np.array(silh_coef), np.array(expected_silh_coef), decimal=4)
np.testing.assert_almost_equal(np.array(corr), np.array(corr_no_max), decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:82*

### test_allocate_cvo

**Category**: method_call  
**Description**: Test the estimates of the Convex Optimization Solution (CVO).  
**Expected**: np.testing.assert_almost_equal(w_cvo, w_cvo_mu, decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(w_cvo, w_expected, decimal=4)
np.testing.assert_almost_equal(w_cvo, w_cvo_mu, decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:112*

### test_allocate_nco

**Category**: method_call  
**Description**: Test the estimates the optimal allocation using the (NCO) algorithm  
**Expected**: np.testing.assert_almost_equal(w_nco, w_nco_mu, decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(w_nco, w_expected, decimal=4)
np.testing.assert_almost_equal(w_nco, w_nco_mu, decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:144*

### test_allocate_mcos

**Category**: method_call  
**Description**: Test the estimates of the optimal allocation using the Monte Carlo optimization selection  
**Expected**: np.testing.assert_almost_equal(np.array(w_nco), np.array(w_nco_expected), decimal=6)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(np.array(w_cvo), np.array(w_cvo_expected), decimal=6)
np.testing.assert_almost_equal(np.array(w_nco), np.array(w_nco_expected), decimal=6)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:209*

### test_allocate_mcos

**Category**: method_call  
**Description**: Test the estimates of the optimal allocation using the Monte Carlo optimization selection  
**Expected**: np.testing.assert_almost_equal(np.array(w_cvo_sr), np.array(w_cvo_sr_expected), decimal=6)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(np.array(w_nco), np.array(w_nco_expected), decimal=6)
np.testing.assert_almost_equal(np.array(w_cvo_sr), np.array(w_cvo_sr_expected), decimal=6)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:210*

### test_allocate_mcos

**Category**: method_call  
**Description**: Test the estimates of the optimal allocation using the Monte Carlo optimization selection  
**Expected**: np.testing.assert_almost_equal(np.array(w_nco_sr), np.array(w_nco_sr_expected), decimal=6)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize\n        '

np.testing.assert_almost_equal(np.array(w_cvo_sr), np.array(w_cvo_sr_expected), decimal=6)
np.testing.assert_almost_equal(np.array(w_nco_sr), np.array(w_nco_sr_expected), decimal=6)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_nco.py:212*

### test_mp_pdf

**Category**: method_call  
**Description**: Test the deriving of pdf of the Marcenko-Pastur distribution.  
**Expected**: self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertAlmostEqual(pdf_mp.index[0], 0.03056, delta=0.0001)
self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:49*

### test_mp_pdf

**Category**: method_call  
**Description**: Test the deriving of pdf of the Marcenko-Pastur distribution.  
**Expected**: self.assertTrue(pdf_mp.values[1] > pdf_mp.values[2] > pdf_mp.values[3])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=0.0001)
self.assertTrue(pdf_mp.values[1] > pdf_mp.values[2] > pdf_mp.values[3])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:50*

### test_fit_kde

**Category**: method_call  
**Description**: Test the kernel fitting to a series of observations.  
**Expected**: self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:70*

### test_fit_kde

**Category**: method_call  
**Description**: Test the kernel fitting to a series of observations.  
**Expected**: self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:71*

### test_fit_kde

**Category**: method_call  
**Description**: Test the kernel fitting to a series of observations.  
**Expected**: self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:72*

### test_fit_kde

**Category**: method_call  
**Description**: Test the kernel fitting to a series of observations.  
**Expected**: self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertEqual(pdf_kde[0.1], pdf_kde_default[0.1])
self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:77*

### test_find_max_eval

**Category**: method_call  
**Description**: Test the search for maximum random eigenvalue.  
**Expected**: self.assertAlmostEqual(var, 0.82702, delta=1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

self.assertAlmostEqual(maximum_eigen, 2.41011, delta=1e-05)
self.assertAlmostEqual(var, 0.82702, delta=1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:115*

### test_get_pca

**Category**: method_call  
**Description**: Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.  
**Expected**: np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

np.testing.assert_almost_equal(eigenvalues, expected_eigenvalues, decimal=4)
np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:190*

### test_denoise_covariance

**Category**: method_call  
**Description**: Test the shrinkage the eigenvalues associated with noise.  
**Expected**: np.testing.assert_almost_equal(cov_matrix_denoised_alt, expected_cov_alt, decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

np.testing.assert_almost_equal(cov_matrix_denoised, expected_cov, decimal=4)
np.testing.assert_almost_equal(cov_matrix_denoised_alt, expected_cov_alt, decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:326*

### test_denoise_covariance

**Category**: method_call  
**Description**: Test the shrinkage the eigenvalues associated with noise.  
**Expected**: np.testing.assert_almost_equal(cov_matrix_detoned, expected_cov_detoned, decimal=4)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and get the test data\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns = ret_est.calculate_returns(self.data)

np.testing.assert_almost_equal(cov_matrix_denoised_alt, expected_cov_alt, decimal=4)
np.testing.assert_almost_equal(cov_matrix_detoned, expected_cov_detoned, decimal=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_estimators.py:329*

### test_ewma

**Category**: method_call  
**Description**: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res[0] == price_arr[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(ewma_res.shape == price_arr.shape)
self.assertTrue(ewma_res[0] == price_arr[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:37*

### test_ewma

**Category**: method_call  
**Description**: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(abs(ewma_res[1] - 1100.0) < 1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(ewma_res[0] == price_arr[0])
self.assertTrue(abs(ewma_res[1] - 1100.0) < 1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:39*

### test_ewma

**Category**: method_call  
**Description**: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res[0] == price_arr[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(ewma_res.shape == price_arr.shape)
self.assertTrue(ewma_res[0] == price_arr[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:37*

### test_ewma

**Category**: method_call  
**Description**: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(abs(ewma_res[1] - 1100.0) < 1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(ewma_res[0] == price_arr[0])
self.assertTrue(abs(ewma_res[1] - 1100.0) < 1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:39*

### test_correlations

**Category**: method_call  
**Description**: Test correlation based coefficients: angular (abs, square), distance correlation.  
**Expected**: self.assertAlmostEqual(abs_angular_dist, 0.6703, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(angular_dist, 0.67, delta=0.01)
self.assertAlmostEqual(abs_angular_dist, 0.6703, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:42*

### test_correlations

**Category**: method_call  
**Description**: Test correlation based coefficients: angular (abs, square), distance correlation.  
**Expected**: self.assertAlmostEqual(sq_angular_dist, 0.7, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(abs_angular_dist, 0.6703, delta=0.01)
self.assertAlmostEqual(sq_angular_dist, 0.7, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:43*

### test_correlations

**Category**: method_call  
**Description**: Test correlation based coefficients: angular (abs, square), distance correlation.  
**Expected**: self.assertAlmostEqual(dist_corr, 0.529, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(sq_angular_dist, 0.7, delta=0.01)
self.assertAlmostEqual(dist_corr, 0.529, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:44*

### test_information_metrics

**Category**: method_call  
**Description**: Test mutual info, information variability metrics.  
**Expected**: self.assertAlmostEqual(mut_info_norm, 0.64, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(mut_info, 0.522, delta=0.01)
self.assertAlmostEqual(mut_info_norm, 0.64, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:59*

### test_information_metrics

**Category**: method_call  
**Description**: Test mutual info, information variability metrics.  
**Expected**: self.assertAlmostEqual(mut_info_bins, 0.626, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(mut_info_norm, 0.64, delta=0.01)
self.assertAlmostEqual(mut_info_bins, 0.626, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:60*

### test_information_metrics

**Category**: method_call  
**Description**: Test mutual info, information variability metrics.  
**Expected**: self.assertAlmostEqual(info_var_norm, 0.7316, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(info_var, 1.4256, delta=0.01)
self.assertAlmostEqual(info_var_norm, 0.7316, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:68*

### test_information_metrics

**Category**: method_call  
**Description**: Test mutual info, information variability metrics.  
**Expected**: self.assertAlmostEqual(info_var_bins, 1.418, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertAlmostEqual(info_var_norm, 0.7316, delta=0.01)
self.assertAlmostEqual(info_var_bins, 1.418, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:69*

### test_number_of_bins

**Category**: method_call  
**Description**: Test get_optimal_number_of_bins functions.  
**Expected**: self.assertEqual(n_bins_x_y, 9)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertEqual(n_bins_x, 15)
self.assertEqual(n_bins_x_y, 9)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:80*

### test_codependence_matrix

**Category**: method_call  
**Description**: Test the get_dependence_matrix and get_distance_matrix function  
**Expected**: self.assertEqual(mi_matrix.shape[0], self.X_matrix.shape[1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertEqual(vi_matrix.shape[0], self.X_matrix.shape[1])
self.assertEqual(mi_matrix.shape[0], self.X_matrix.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:104*

### test_codependence_matrix

**Category**: method_call  
**Description**: Test the get_dependence_matrix and get_distance_matrix function  
**Expected**: self.assertEqual(corr_matrix.shape[0], self.X_matrix.shape[1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
state = np.random.RandomState(42)
self.x = state.normal(size=1000)
self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
self.y_2 = abs(self.x) + state.normal(size=1000) / 5
self.X_matrix, _ = get_classification_data(6, 2, 2, 100, sigma=0)

self.assertEqual(mi_matrix.shape[0], self.X_matrix.shape[1])
self.assertEqual(corr_matrix.shape[0], self.X_matrix.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_codependence.py:105*

### test_get_linkage_corr

**Category**: method_call  
**Description**: Testing the creation of a linkage object from empirical correlation matrix and tree graph  
**Expected**: np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dendrogram_alt)), decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)), decimal=2)
np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dendrogram_alt)), decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:68*

### test_update_dist

**Category**: method_call  
**Description**: Testing the update of the general distance matrix to take the new clusters into account  
**Expected**: np.testing.assert_almost_equal(np.array(dist_new_alt), np.array(dist_expected), decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

np.testing.assert_almost_equal(np.array(dist_new), np.array(dist_expected), decimal=2)
np.testing.assert_almost_equal(np.array(dist_new_alt), np.array(dist_expected), decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:132*

### test_get_linkage_corr

**Category**: method_call  
**Description**: Testing the creation of a linkage object from empirical correlation matrix and tree graph  
**Expected**: np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dendrogram_alt)), decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)), decimal=2)
np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dendrogram_alt)), decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:68*

### test_update_dist

**Category**: method_call  
**Description**: Testing the update of the general distance matrix to take the new clusters into account  
**Expected**: np.testing.assert_almost_equal(np.array(dist_new_alt), np.array(dist_expected), decimal=2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
np.testing.assert_almost_equal(np.array(dist_new), np.array(dist_expected), decimal=2)
np.testing.assert_almost_equal(np.array(dist_new_alt), np.array(dist_expected), decimal=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:132*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertEqual(self.data.shape[0], yz_vol.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], gm_vol.shape[0])
self.assertEqual(self.data.shape[0], yz_vol.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:35*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertEqual(self.data.shape[0], park_vol.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], yz_vol.shape[0])
self.assertEqual(self.data.shape[0], park_vol.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:36*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], park_vol.shape[0])
self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:37*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)
self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:39*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)
self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:40*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertEqual(self.data.shape[0], yz_vol.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(self.data.shape[0], gm_vol.shape[0])
self.assertEqual(self.data.shape[0], yz_vol.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:35*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertEqual(self.data.shape[0], park_vol.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(self.data.shape[0], yz_vol.shape[0])
self.assertEqual(self.data.shape[0], park_vol.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:36*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(self.data.shape[0], park_vol.shape[0])
self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:37*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-06)
self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:39*

### test_vol_features

**Category**: method_call  
**Description**: Test volatility estimators  
**Expected**: self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-06)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-06)
self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-06)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_volatility_features.py:40*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db3.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db1.shape == db2.shape)
self.assertTrue(db1.shape == db3.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:53*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db4.shape == db1.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db1.shape == db3.shape)
self.assertTrue(db4.shape == db1.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:54*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db2.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db4.shape == db1.shape)
self.assertTrue(np.all(db1.values == db2.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:55*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db3.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(np.all(db1.values == db2.values))
self.assertTrue(np.all(db1.values == db3.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:58*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db4.values == db1.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(np.all(db1.values == db3.values))
self.assertTrue(np.all(db4.values == db1.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:59*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'open'] == 1205)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(np.all(db4.values == db1.values))
self.assertTrue(db1.loc[0, 'open'] == 1205)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:60*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'high'] == 1904.75)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db1.loc[0, 'open'] == 1205)
self.assertTrue(db1.loc[0, 'high'] == 1904.75)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:63*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'low'] == 1005.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db1.loc[0, 'high'] == 1904.75)
self.assertTrue(db1.loc[0, 'low'] == 1005.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:64*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'close'] == 1304.5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(db1.loc[0, 'low'] == 1005.0)
self.assertTrue(db1.loc[0, 'close'] == 1304.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:65*

### test_dollar_bars

**Category**: method_call  
**Description**: Tests the dollar bars implementation.  
**Expected**: self.assertTrue(np.all(df_constant.values == db1.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

self.assertTrue(df_constant.shape == db1.shape)
self.assertTrue(np.all(df_constant.values == db1.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_standard_data_structures.py:74*

### test_vol_adj_ret

**Category**: method_call  
**Description**: Tests for getting the correct volatility adjusted return.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

pd.testing.assert_frame_equal(test1, simple_returns)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:54*

### test_vol_adj_ret

**Category**: method_call  
**Description**: Tests for getting the correct volatility adjusted return.  
**Expected**: np.testing.assert_array_almost_equal(test3.iloc[4:5, 3:13], np.array([[-0.88220253, -0.05699642, -0.41151834, 0.22209753, -0.26852039, -0.41058931, -0.72457246, -0.45304492, -1.62358571, -0.74637241]]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

pd.testing.assert_frame_equal(test2, test2_actual)
np.testing.assert_array_almost_equal(test3.iloc[4:5, 3:13], np.array([[-0.88220253, -0.05699642, -0.41151834, 0.22209753, -0.26852039, -0.41058931, -0.72457246, -0.45304492, -1.62358571, -0.74637241]]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:55*

### test_extract_tail_sets

**Category**: method_call  
**Description**: Tests for extracting the tail set in one row, including positive and negative class.  
**Expected**: self.assertEqual(test5, test5_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

pd.testing.assert_series_equal(test4, test4_actual)
self.assertEqual(test5, test5_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:74*

### test_extract_tail_sets

**Category**: method_call  
**Description**: Tests for extracting the tail set in one row, including positive and negative class.  
**Expected**: self.assertEqual(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

self.assertEqual(test5, test5_actual)
self.assertEqual(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:75*

### test_overall

**Category**: method_call  
**Description**: Tests the overall output of the tail set labels.  
**Expected**: self.assertEqual(test7_neg[1], ['EWU', 'XLK', 'DIA'])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

self.assertEqual(test7_pos[1], ['XLU', 'EPP', 'FXI'])
self.assertEqual(test7_neg[1], ['EWU', 'XLK', 'DIA'])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:85*

### test_vol_adj_ret

**Category**: method_call  
**Description**: Tests for getting the correct volatility adjusted return.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test1, simple_returns)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:54*

### test_vol_adj_ret

**Category**: method_call  
**Description**: Tests for getting the correct volatility adjusted return.  
**Expected**: np.testing.assert_array_almost_equal(test3.iloc[4:5, 3:13], np.array([[-0.88220253, -0.05699642, -0.41151834, 0.22209753, -0.26852039, -0.41058931, -0.72457246, -0.45304492, -1.62358571, -0.74637241]]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test2, test2_actual)
np.testing.assert_array_almost_equal(test3.iloc[4:5, 3:13], np.array([[-0.88220253, -0.05699642, -0.41151834, 0.22209753, -0.26852039, -0.41058931, -0.72457246, -0.45304492, -1.62358571, -0.74637241]]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:55*

### test_extract_tail_sets

**Category**: method_call  
**Description**: Tests for extracting the tail set in one row, including positive and negative class.  
**Expected**: self.assertEqual(test5, test5_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test4, test4_actual)
self.assertEqual(test5, test5_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:74*

### test_extract_tail_sets

**Category**: method_call  
**Description**: Tests for extracting the tail set in one row, including positive and negative class.  
**Expected**: self.assertEqual(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(test5, test5_actual)
self.assertEqual(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:75*

### test_overall

**Category**: method_call  
**Description**: Tests the overall output of the tail set labels.  
**Expected**: self.assertEqual(test7_neg[1], ['EWU', 'XLK', 'DIA'])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(test7_pos[1], ['XLU', 'EPP', 'FXI'])
self.assertEqual(test7_neg[1], ['EWU', 'XLK', 'DIA'])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_tail_sets.py:85*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertEqual(self.data.shape[0], roll_impact.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], roll_measure.shape[0])
self.assertEqual(self.data.shape[0], roll_impact.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:48*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertEqual(self.data.shape[0], corwin_schultz.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], roll_impact.shape[0])
self.assertEqual(self.data.shape[0], corwin_schultz.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:49*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertEqual(self.data.shape[0], bekker_parkinson.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], corwin_schultz.shape[0])
self.assertEqual(self.data.shape[0], bekker_parkinson.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:50*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_measure.max(), 7.1584, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.shape[0], bekker_parkinson.shape[0])
self.assertAlmostEqual(roll_measure.max(), 7.1584, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:51*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_measure.mean(), 2.341, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_measure.max(), 7.1584, delta=0.0001)
self.assertAlmostEqual(roll_measure.mean(), 2.341, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:54*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_measure[25], 1.176, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_measure.mean(), 2.341, delta=0.001)
self.assertAlmostEqual(roll_measure[25], 1.176, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:55*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_impact.max(), 1.022e-07, delta=1e-07)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_measure[25], 1.176, delta=0.001)
self.assertAlmostEqual(roll_impact.max(), 1.022e-07, delta=1e-07)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:56*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_impact.mean(), 3.3445e-08, delta=1e-07)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_impact.max(), 1.022e-07, delta=1e-07)
self.assertAlmostEqual(roll_impact.mean(), 3.3445e-08, delta=1e-07)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:58*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(roll_impact[25], 1.6807e-08, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_impact.mean(), 3.3445e-08, delta=1e-07)
self.assertAlmostEqual(roll_impact[25], 1.6807e-08, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:59*

### test_first_generation

**Category**: method_call  
**Description**: Test first generation intra-bar features  
**Expected**: self.assertAlmostEqual(corwin_schultz.max(), 0.01652, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.trades_path = project_path + '/test_data/tick_data.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
self.data.index = pd.to_datetime(self.data.index)

self.assertAlmostEqual(roll_impact[25], 1.6807e-08, delta=0.0001)
self.assertAlmostEqual(corwin_schultz.max(), 0.01652, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_microstructural_features.py:60*

### test_quasi_diagnalization

**Category**: method_call  
**Description**: Test the quasi-diagnalisation step of HERC algorithm.  
**Expected**: assert herc.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

herc.allocate(asset_prices=self.data, linkage='single', optimal_num_clusters=5, asset_names=self.data.columns)
assert herc.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:125*

### test_quasi_diagnalization

**Category**: method_call  
**Description**: Test the quasi-diagnalisation step of HERC algorithm.  
**Expected**: assert herc.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]  
**Confidence**: 0.85  
**Tags**: unittest  

```python
herc.allocate(asset_prices=self.data, linkage='single', optimal_num_clusters=5, asset_names=self.data.columns)
assert herc.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:125*

### test_orthogonal_features

**Category**: method_call  
**Description**: Test orthogonal features: PCA features, importance vs PCA importance analysis  
**Expected**: self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-07)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(np.mean(pca_features[:, 2]), 0, delta=1e-07)
self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-07)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:49*

### test_orthogonal_features

**Category**: method_call  
**Description**: Test orthogonal features: PCA features, importance vs PCA importance analysis  
**Expected**: self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-07)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-07)
self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-07)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:50*

### test_orthogonal_features

**Category**: method_call  
**Description**: Test orthogonal features: PCA features, importance vs PCA importance analysis  
**Expected**: self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.3813, delta=0.2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-07)
self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.3813, delta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:51*

### test_orthogonal_features

**Category**: method_call  
**Description**: Test orthogonal features: PCA features, importance vs PCA importance analysis  
**Expected**: self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.0255, delta=0.2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.3813, delta=0.2)
self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.0255, delta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:54*

### test_orthogonal_features

**Category**: method_call  
**Description**: Test orthogonal features: PCA features, importance vs PCA importance analysis  
**Expected**: self.assertAlmostEqual(np.std(pca_features[:, 4]), 1.0011, delta=0.2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.0255, delta=0.2)
self.assertAlmostEqual(np.std(pca_features[:, 4]), 1.0011, delta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:55*

### test_feature_importance

**Category**: method_call  
**Description**: Test features importance: MDI, MDA, SFI and plot function  
**Expected**: self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], 0.46835, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(mdi_feat_imp['mean'].sum(), 1, delta=0.001)
self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], 0.46835, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:116*

### test_feature_importance

**Category**: method_call  
**Description**: Test features importance: MDI, MDA, SFI and plot function  
**Expected**: self.assertAlmostEqual(mdi_feat_imp.loc['I_0', 'mean'], 0.08214, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], 0.46835, delta=0.01)
self.assertAlmostEqual(mdi_feat_imp.loc['I_0', 'mean'], 0.08214, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:121*

### test_feature_importance

**Category**: method_call  
**Description**: Test features importance: MDI, MDA, SFI and plot function  
**Expected**: self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], 0.06511, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(mdi_feat_imp.loc['I_0', 'mean'], 0.08214, delta=0.01)
self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], 0.06511, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:122*

### test_feature_importance

**Category**: method_call  
**Description**: Test features importance: MDI, MDA, SFI and plot function  
**Expected**: self.assertAlmostEqual(mdi_feat_imp.loc['N_0', 'mean'], 0.02229, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], 0.06511, delta=0.01)
self.assertAlmostEqual(mdi_feat_imp.loc['N_0', 'mean'], 0.02229, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:124*

### test_feature_importance

**Category**: method_call  
**Description**: Test features importance: MDI, MDA, SFI and plot function  
**Expected**: self.assertAlmostEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], 0.65522, delta=0.1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Generate X, y data sets and fit a RF\n        '
self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100, oob_score=True, random_state=1)
self.fit_clf = self.bag_clf.fit(self.X, self.y)
self.cv_gen = KFold(n_splits=3)

self.assertAlmostEqual(mdi_feat_imp.loc['N_0', 'mean'], 0.02229, delta=0.01)
self.assertAlmostEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], 0.65522, delta=0.1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_importance.py:126*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

self.assertEqual(tr_scan_labels.shape[0], len(t_events))
self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:36*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))
self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:39*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)
self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:42*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.shape[0] - tr_scan_labels.dropna().shape[0], 19)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)
self.assertEqual(tr_scan_labels.shape[0] - tr_scan_labels.dropna().shape[0], 19)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:43*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertTrue((tr_scan_labels.t1 > tr_scan_labels.index).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

tr_scan_labels.dropna(inplace=True)
self.assertTrue((tr_scan_labels.t1 > tr_scan_labels.index).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:48*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertTrue((tr_scan_labels == tr_scan_labels_none).all().all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

tr_scan_labels_none.dropna(inplace=True)
self.assertTrue((tr_scan_labels == tr_scan_labels_none).all().all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:59*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(tr_scan_labels.shape[0], len(t_events))
self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:36*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))
self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:39*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)
self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:42*

### test_trend_scanning_labels

**Category**: method_call  
**Description**: Test trend scanning labels  
**Expected**: self.assertEqual(tr_scan_labels.shape[0] - tr_scan_labels.dropna().shape[0], 19)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)
self.assertEqual(tr_scan_labels.shape[0] - tr_scan_labels.dropna().shape[0], 19)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_trend_scanning_labels.py:43*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][0], 0.465, delta=0.01)
self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:38*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)
self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:39*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][3], 0.348, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)
self.assertAlmostEqual(haircuts[0][3], 0.348, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:40*

### test_profit_hurdle

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(p_values[1], 0.702, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(p_values[0], 0.365, delta=0.01)
self.assertAlmostEqual(p_values[1], 0.702, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:62*

### test_profit_hurdle

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(p_values[2], 0.687, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(p_values[1], 0.702, delta=0.01)
self.assertAlmostEqual(p_values[2], 0.687, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:63*

### test_profit_hurdle

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(p_values[3], 0.62, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(p_values[2], 0.687, delta=0.01)
self.assertAlmostEqual(p_values[3], 0.62, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:64*

### test_profit_hurdle

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(p_values[4], 0.694, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(p_values[3], 0.62, delta=0.01)
self.assertAlmostEqual(p_values[4], 0.694, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:65*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][0], 0.465, delta=0.01)
self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:38*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][1], 0.409, delta=0.01)
self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:39*

### test_haircut_sharpe_ratios_simple_input

**Category**: method_call  
**Description**: Test the calculation of haircuts with simple inputs  
**Expected**: self.assertAlmostEqual(haircuts[0][3], 0.348, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(haircuts[0][2], 0.174, delta=0.01)
self.assertAlmostEqual(haircuts[0][3], 0.348, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtests.py:40*

### test_get_weights

**Category**: method_call  
**Description**: get_weights as implemented here matches the code in the book (Snippet 5.1).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is same as the requested length  
**Expected**: self.assertTrue(weights.shape[0] == number_ele)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(weights[-1] == 1.0)
self.assertTrue(weights.shape[0] == number_ele)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:41*

### test_get_weights_ffd

**Category**: method_call  
**Description**: get_weights_ffd as implemented here matches the code in the book (Snippet 5.2).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is equal to 12  
**Expected**: self.assertTrue(weights.shape[0] == 12)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(weights[-1] == 1.0)
self.assertTrue(weights.shape[0] == 12)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:61*

### test_plot_min_ffd

**Category**: method_call  
**Description**: Assert that the plot for min ffd is correct,

Testing is based on the correlation between the original series (d=0)
and the differentiated series.  
**Expected**: np.testing.assert_allclose(correlation, expected_correlation)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

print(type(plot))
np.testing.assert_allclose(correlation, expected_correlation)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:117*

### test_get_weights

**Category**: method_call  
**Description**: get_weights as implemented here matches the code in the book (Snippet 5.1).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is same as the requested length  
**Expected**: self.assertTrue(weights.shape[0] == number_ele)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(weights[-1] == 1.0)
self.assertTrue(weights.shape[0] == number_ele)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:41*

### test_get_weights_ffd

**Category**: method_call  
**Description**: get_weights_ffd as implemented here matches the code in the book (Snippet 5.2).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is equal to 12  
**Expected**: self.assertTrue(weights.shape[0] == 12)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(weights[-1] == 1.0)
self.assertTrue(weights.shape[0] == 12)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:61*

### test_plot_min_ffd

**Category**: method_call  
**Description**: Assert that the plot for min ffd is correct,

Testing is based on the correlation between the original series (d=0)
and the differentiated series.  
**Expected**: np.testing.assert_allclose(correlation, expected_correlation)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
print(type(plot))
np.testing.assert_allclose(correlation, expected_correlation)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:117*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(z_score_events.shape[0] == 68)
self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:70*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])
self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:71*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertEqual(self.data.loc[z_score_events[25], 'close'], 2009.5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)
self.assertEqual(self.data.loc[z_score_events[25], 'close'], 2009.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:72*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(z_score_events.shape[0] == 68)
self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:70*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(z_score_events.shape[0] == z_score_events_timestamp_false.shape[0])
self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:71*

### test_z_score_filter

**Category**: method_call  
**Description**: Test Z-score filter  
**Expected**: self.assertEqual(self.data.loc[z_score_events[25], 'close'], 2009.5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(self.data.loc[z_score_events[0], 'close'], 2037.25)
self.assertEqual(self.data.loc[z_score_events[25], 'close'], 2009.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:72*

### test_small_set

**Category**: method_call  
**Description**: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:42*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical labels with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:87*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical labels with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test7, test6_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test7, test6_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:88*

### test_small_set

**Category**: method_call  
**Description**: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:42*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical labels with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:87*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical labels with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test7, test6_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test7, test6_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:88*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.data, _ = load_breast_cancer(return_X_y=True)

self.assertGreaterEqual(len(clusters.keys()), 5)
self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:42*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.data, _ = load_breast_cancer(return_X_y=True)

self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))
self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:43*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([5, 6, 7, 25, 26, 27], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.data, _ = load_breast_cancer(return_X_y=True)

self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))
self.assertTrue(self._check_if_in_cluster([5, 6, 7, 25, 26, 27], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:44*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertGreaterEqual(len(clusters.keys()), 5)
self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:42*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))
self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:43*

### test_get_onc_clusters

**Category**: method_call  
**Description**: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertTrue(self._check_if_in_cluster([5, 6, 7, 25, 26, 27], clusters))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))
self.assertTrue(self._check_if_in_cluster([5, 6, 7, 25, 26, 27], clusters))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:44*

### test_ret_attribution

**Category**: method_call  
**Description**: Assert that return attribution length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(abs(ret_weights.iloc[0] - 0.781807) <= 100000.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(ret_weights.shape[0] == non_nan_meta_labels.shape[0])
self.assertTrue(abs(ret_weights.iloc[0] - 0.781807) <= 100000.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:52*

### test_ret_attribution

**Category**: method_call  
**Description**: Assert that return attribution length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(abs(ret_weights.iloc[3] - 1.627944) <= 100000.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(abs(ret_weights.iloc[0] - 0.781807) <= 100000.0)
self.assertTrue(abs(ret_weights.iloc[3] - 1.627944) <= 100000.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:53*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(standard_decay.shape == neg_decay.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(standard_decay.shape == no_decay.shape)
self.assertTrue(standard_decay.shape == neg_decay.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:67*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(standard_decay.shape == converge_decay.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(standard_decay.shape == neg_decay.shape)
self.assertTrue(standard_decay.shape == converge_decay.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:68*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(standard_decay.shape == pos_decay.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(standard_decay.shape == converge_decay.shape)
self.assertTrue(standard_decay.shape == pos_decay.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:69*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(standard_decay.iloc[-1] == 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(standard_decay.shape == pos_decay.shape)
self.assertTrue(standard_decay.iloc[-1] == 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:70*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(abs(standard_decay.iloc[0] - 0.582191) <= 100000.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(standard_decay.iloc[-1] == 1.0)
self.assertTrue(abs(standard_decay.iloc[0] - 0.582191) <= 100000.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:72*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(no_decay.values.all() == 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(abs(standard_decay.iloc[0] - 0.582191) <= 100000.0)
self.assertTrue(no_decay.values.all() == 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:73*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(neg_decay[neg_decay == 0].shape[0] == 3)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(no_decay.values.all() == 1)
self.assertTrue(neg_decay[neg_decay == 0].shape[0] == 3)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:75*

### test_time_decay_weights

**Category**: method_call  
**Description**: Assert that time decay weights length equals triple barrier length, check particular values  
**Expected**: self.assertTrue(pos_decay.iloc[0] == pos_decay.max())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
cusum_events = cusum_filter(self.data['close'], threshold=0.02)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=2)
self.data['side'] = 1
self.meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[4, 4], target=daily_vol, min_ret=0.005, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)

self.assertTrue(neg_decay[neg_decay == 0].shape[0] == 3)
self.assertTrue(pos_decay.iloc[0] == pos_decay.max())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sample_weights.py:78*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:50*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))
pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:51*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test4, test3_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test4, test3_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:52*

### test_series

**Category**: method_call  
**Description**: Verifies raw returns for a series for simple/logarithmic returns, with numerical/binary labels.  
**Expected**: pd.testing.assert_series_equal(test5, test5_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_series_equal(test4, test4_actual, check_less_precise=True)
pd.testing.assert_series_equal(test5, test5_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:68*

### test_series

**Category**: method_call  
**Description**: Verifies raw returns for a series for simple/logarithmic returns, with numerical/binary labels.  
**Expected**: pd.testing.assert_series_equal(test6, test5_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_series_equal(test5, test5_actual, check_less_precise=True)
pd.testing.assert_series_equal(test6, test5_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:69*

### test_resample

**Category**: method_call  
**Description**: Tests that resampling works correctly.  
**Expected**: pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:96*

### test_resample

**Category**: method_call  
**Description**: Tests that resampling works correctly.  
**Expected**: pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx5 = self.data[:5].index
self.col5 = self.data.iloc[:, 0:5].columns

pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:97*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:50*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))
pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:51*

### test_dataframe

**Category**: method_call  
**Description**: Verifies raw returns for a DataFrame.  
**Expected**: pd.testing.assert_frame_equal(test4, test3_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test4, test3_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_raw_return.py:52*

### test_transform_data

**Category**: method_call  
**Description**: Tests that the transform_data method gives the correct 10 by 10 matrix.  
**Expected**: pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/close_df.csv'
self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

pd.testing.assert_frame_equal(test1, test1_actual)
pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:79*

### test_transform_data

**Category**: method_call  
**Description**: Tests that the transform_data method gives the correct 10 by 10 matrix.  
**Expected**: self.assertTrue(test2.sum(axis=0).values.all() == np.array([1.0] * 10).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/close_df.csv'
self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)
self.assertTrue(test2.sum(axis=0).values.all() == np.array([1.0] * 10).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:80*

### test_threshold

**Category**: method_call  
**Description**: Tests for when threshold is desired.  
**Expected**: pd.testing.assert_series_equal(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/close_df.csv'
self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

pd.testing.assert_series_equal(test5, test5_actual)
pd.testing.assert_series_equal(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:149*

### test_template_init

**Category**: method_call  
**Description**: Checks that other templates are given correctly.  
**Expected**: pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/close_df.csv'
self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

pd.testing.assert_frame_equal(test7.template, leigh_bear)
pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:182*

### test_template_init

**Category**: method_call  
**Description**: Checks that other templates are given correctly.  
**Expected**: pd.testing.assert_frame_equal(test9.template, cervelloroyo_bull)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/close_df.csv'
self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)
pd.testing.assert_frame_equal(test9.template, cervelloroyo_bull)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:183*

### test_transform_data

**Category**: method_call  
**Description**: Tests that the transform_data method gives the correct 10 by 10 matrix.  
**Expected**: pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test1, test1_actual)
pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:79*

### test_transform_data

**Category**: method_call  
**Description**: Tests that the transform_data method gives the correct 10 by 10 matrix.  
**Expected**: self.assertTrue(test2.sum(axis=0).values.all() == np.array([1.0] * 10).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)
self.assertTrue(test2.sum(axis=0).values.all() == np.array([1.0] * 10).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:80*

### test_threshold

**Category**: method_call  
**Description**: Tests for when threshold is desired.  
**Expected**: pd.testing.assert_series_equal(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test5, test5_actual)
pd.testing.assert_series_equal(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:149*

### test_template_init

**Category**: method_call  
**Description**: Checks that other templates are given correctly.  
**Expected**: pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test7.template, leigh_bear)
pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:182*

### test_template_init

**Category**: method_call  
**Description**: Checks that other templates are given correctly.  
**Expected**: pd.testing.assert_frame_equal(test9.template, cervelloroyo_bull)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)
pd.testing.assert_frame_equal(test9.template, cervelloroyo_bull)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_matrix_flags.py:183*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db2.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.shape, (2, 10))
self.assertTrue(db1.shape == db2.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:52*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db3.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db2.shape)
self.assertTrue(db1.shape == db3.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:55*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db4.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db3.shape)
self.assertTrue(db1.shape == db4.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:56*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db2.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db4.shape)
self.assertTrue(np.all(db1.values == db2.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:57*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db3.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db2.values))
self.assertTrue(np.all(db1.values == db3.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:60*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db4.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db3.values))
self.assertTrue(np.all(db1.values == db4.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:61*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db4.values))
self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:62*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:64*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'open'], 1306.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))
self.assertEqual(db1.loc[0, 'open'], 1306.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:65*

### test_ema_run_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA run dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'high'], 1306.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.loc[0, 'open'], 1306.0)
self.assertEqual(db1.loc[0, 'high'], 1306.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_run_data_structures.py:68*

### test_timing_of_flattening_and_flips

**Category**: method_call  
**Description**: Check that moments of flips and flattenings are picked correctly and
that last is added  
**Expected**: self.assertTrue(flattenings_and_flips_last.sort_values().equals(flattenings_and_flips.sort_values()))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertTrue(test_flat_flip.sort_values().equals(flattenings_and_flips.sort_values()))
self.assertTrue(flattenings_and_flips_last.sort_values().equals(flattenings_and_flips.sort_values()))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:76*

### test_average_holding_period

**Category**: method_call  
**Description**: Check average holding period calculation  
**Expected**: self.assertTrue(np.isnan(nan_average_holding))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertAlmostEqual(average_holding, 2, delta=0.0001)
self.assertTrue(np.isnan(nan_average_holding))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:88*

### test_bets_concentration

**Category**: method_call  
**Description**: Check if concentration is balanced and correctly calculated  
**Expected**: self.assertAlmostEqual(positive_concentration, 2.0111445, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertAlmostEqual(positive_concentration, negative_concentration, delta=1e-05)
self.assertAlmostEqual(positive_concentration, 2.0111445, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:101*

### test_all_bets_concentration

**Category**: method_call  
**Description**: Check if concentration is nan when not enough observations, also values
testing  
**Expected**: self.assertTrue(np.isnan(positive_returns_concentration[2]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertTrue(np.isnan(positive_returns_concentration[1]))
self.assertTrue(np.isnan(positive_returns_concentration[2]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:118*

### test_all_bets_concentration

**Category**: method_call  
**Description**: Check if concentration is nan when not enough observations, also values
testing  
**Expected**: self.assertAlmostEqual(all_returns_concentration[0], 0.0014938, delta=1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertTrue(np.isnan(positive_returns_concentration[2]))
self.assertAlmostEqual(all_returns_concentration[0], 0.0014938, delta=1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:121*

### test_all_bets_concentration

**Category**: method_call  
**Description**: Check if concentration is nan when not enough observations, also values
testing  
**Expected**: self.assertAlmostEqual(all_returns_concentration[1], 0.0016261, delta=1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertAlmostEqual(all_returns_concentration[0], 0.0014938, delta=1e-05)
self.assertAlmostEqual(all_returns_concentration[1], 0.0016261, delta=1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:122*

### test_all_bets_concentration

**Category**: method_call  
**Description**: Check if concentration is nan when not enough observations, also values
testing  
**Expected**: self.assertAlmostEqual(all_returns_concentration[2], 0.0195998, delta=1e-05)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertAlmostEqual(all_returns_concentration[1], 0.0016261, delta=1e-05)
self.assertAlmostEqual(all_returns_concentration[2], 0.0195998, delta=1e-05)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:124*

### test_drawdown_and_time_under_water

**Category**: method_call  
**Description**: Check if drawdowns and time under water calculated correctly for
dollar and non-dollar test sets.  
**Expected**: self.assertTrue(list(time_under_water) == list(time_under_water_dol))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertTrue(list(drawdown_dol) == [20.0, 30.0, 10.0])
self.assertTrue(list(time_under_water) == list(time_under_water_dol))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:140*

### test_drawdown_and_time_under_water

**Category**: method_call  
**Description**: Check if drawdowns and time under water calculated correctly for
dollar and non-dollar test sets.  
**Expected**: self.assertAlmostEqual(time_under_water[0], 0.010951, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertTrue(list(time_under_water) == list(time_under_water_dol))
self.assertAlmostEqual(time_under_water[0], 0.010951, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:141*

### test_drawdown_and_time_under_water

**Category**: method_call  
**Description**: Check if drawdowns and time under water calculated correctly for
dollar and non-dollar test sets.  
**Expected**: self.assertAlmostEqual(time_under_water[1], 0.008213, delta=0.0001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the data for tests.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/dollar_bar_sample.csv'
self.logret = pd.read_csv(data_path, index_col='date_time')
self.logret.index = pd.to_datetime(self.logret.index)
self.logret = np.log(self.logret['close']).diff()[1:]
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
cumulated_ret = np.cumprod(1 + normal_ret)
self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
self.hold_positions = pd.Series(data=hold_positions, index=dates)
self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

self.assertAlmostEqual(time_under_water[0], 0.010951, delta=0.0001)
self.assertAlmostEqual(time_under_water[1], 0.008213, delta=0.0001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_backtest_statistics.py:142*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db2.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.shape, (624, 10))
self.assertTrue(db1.shape == db2.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:50*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db3.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db2.shape)
self.assertTrue(db1.shape == db3.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:54*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(db1.shape == db4.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db3.shape)
self.assertTrue(db1.shape == db4.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:55*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db2.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(db1.shape == db4.shape)
self.assertTrue(np.all(db1.values == db2.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:56*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db3.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db2.values))
self.assertTrue(np.all(db1.values == db3.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:59*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db4.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db3.values))
self.assertTrue(np.all(db1.values == db4.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:60*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'open'], 1306.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertTrue(np.all(db1.values == db4.values))
self.assertEqual(db1.loc[0, 'open'], 1306.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:61*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'high'], 1306.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.loc[0, 'open'], 1306.0)
self.assertEqual(db1.loc[0, 'high'], 1306.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:64*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'low'], 1304.25)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.loc[0, 'high'], 1306.0)
self.assertEqual(db1.loc[0, 'low'], 1304.25)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:65*

### test_ema_imbalance_dollar_bars

**Category**: method_call  
**Description**: Tests the EMA imbalance dollar bars implementation.  
**Expected**: self.assertEqual(db1.loc[0, 'close'], 1304.5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

self.assertEqual(db1.loc[0, 'low'], 1304.25)
self.assertEqual(db1.loc[0, 'close'], 1304.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_imbalance_data_structures.py:66*

### test_num_concurrent_events

**Category**: method_call  
**Description**: Assert that number of concurent events have are available for all labels and equal to particular values  
**Expected**: self.assertTrue(num_conc_events.value_counts()[0] == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(num_conc_events[self.samples_info_sets.index].shape[0] == self.samples_info_sets.shape[0])
self.assertTrue(num_conc_events.value_counts()[0] == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:46*

### test_num_concurrent_events

**Category**: method_call  
**Description**: Assert that number of concurent events have are available for all labels and equal to particular values  
**Expected**: self.assertTrue(num_conc_events.value_counts()[1] == 11)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(num_conc_events.value_counts()[0] == 5)
self.assertTrue(num_conc_events.value_counts()[1] == 11)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:47*

### test_num_concurrent_events

**Category**: method_call  
**Description**: Assert that number of concurent events have are available for all labels and equal to particular values  
**Expected**: self.assertTrue(num_conc_events.value_counts()[2] == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(num_conc_events.value_counts()[1] == 11)
self.assertTrue(num_conc_events.value_counts()[2] == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:48*

### test_num_concurrent_events

**Category**: method_call  
**Description**: Assert that number of concurent events have are available for all labels and equal to particular values  
**Expected**: self.assertTrue(num_conc_events.value_counts()[3] == 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(num_conc_events.value_counts()[2] == 5)
self.assertTrue(num_conc_events.value_counts()[3] == 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:49*

### test_get_av_uniqueness

**Category**: method_call  
**Description**: Assert that average event uniqueness is available for all labels and equals to particular values  
**Expected**: self.assertAlmostEqual(av_un['tW'].iloc[0], 0.66, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(av_un.shape[0] == self.samples_info_sets.shape[0])
self.assertAlmostEqual(av_un['tW'].iloc[0], 0.66, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:59*

### test_get_av_uniqueness

**Category**: method_call  
**Description**: Assert that average event uniqueness is available for all labels and equals to particular values  
**Expected**: self.assertAlmostEqual(av_un['tW'].iloc[2], 0.83, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertAlmostEqual(av_un['tW'].iloc[0], 0.66, delta=0.01)
self.assertAlmostEqual(av_un['tW'].iloc[2], 0.83, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:60*

### test_get_av_uniqueness

**Category**: method_call  
**Description**: Assert that average event uniqueness is available for all labels and equals to particular values  
**Expected**: self.assertAlmostEqual(av_un['tW'].iloc[5], 0.44, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertAlmostEqual(av_un['tW'].iloc[2], 0.83, delta=0.01)
self.assertAlmostEqual(av_un['tW'].iloc[5], 0.44, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:61*

### test_get_av_uniqueness

**Category**: method_call  
**Description**: Assert that average event uniqueness is available for all labels and equals to particular values  
**Expected**: self.assertAlmostEqual(av_un['tW'].iloc[-1], 1.0, delta=0.01)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertAlmostEqual(av_un['tW'].iloc[5], 0.44, delta=0.01)
self.assertAlmostEqual(av_un['tW'].iloc[-1], 1.0, delta=0.01)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:62*

### test_seq_bootstrap

**Category**: method_call  
**Description**: Test sequential bootstrapping length, indicator matrix length and NaN checks  
**Expected**: self.assertTrue(ind_mat.shape == (22, 8))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(bool((ind_mat_book_implementation.values == ind_mat).all()) is True)
self.assertTrue(ind_mat.shape == (22, 8))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:81*

### test_seq_bootstrap

**Category**: method_call  
**Description**: Test sequential bootstrapping length, indicator matrix length and NaN checks  
**Expected**: self.assertTrue(bool((ind_mat[:3, 0] == np.ones(3)).all()) is True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set samples_info_sets (t1), price bars\n        '
self.price_bars = pd.Series(index=pd.date_range(start='1/1/2018', end='1/8/2018', freq='H'), dtype='float64')
self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

self.assertTrue(ind_mat.shape == (22, 8))
self.assertTrue(bool((ind_mat[:3, 0] == np.ones(3)).all()) is True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sampling.py:83*

### test_m2n_constructor

**Category**: method_call  
**Description**: Tests that the constructor of the M2N class executes properly.  
**Expected**: self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(m2n_test.moments, moments_test)
self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:25*

### test_m2n_constructor

**Category**: method_call  
**Description**: Tests that the constructor of the M2N class executes properly.  
**Expected**: self.assertEqual(m2n_test.parameters, [0, 0, 0, 0, 0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])
self.assertEqual(m2n_test.parameters, [0, 0, 0, 0, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:26*

### test_m2n_constructor

**Category**: method_call  
**Description**: Tests that the constructor of the M2N class executes properly.  
**Expected**: self.assertEqual(m2n_test.error, sum([moments_test[i] ** 2 for i in range(len(moments_test))]))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(m2n_test.parameters, [0, 0, 0, 0, 0])
self.assertEqual(m2n_test.error, sum([moments_test[i] ** 2 for i in range(len(moments_test))]))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:27*

### test_get_moments

**Category**: method_call  
**Description**: Tests the 'get_moments' method of the M2N class.  
**Expected**: self.assertEqual(test_mmnts, m2n_test.new_moments)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.get_moments(test_params, return_result=False)
self.assertEqual(test_mmnts, m2n_test.new_moments)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:51*

### test_fit_variant_1

**Category**: method_call  
**Description**: Tests the 'fit' method of the M2N class, using variant 1.  
**Expected**: self.assertTrue(len(m2n_test.parameters) == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.fit(mu_2_test)
self.assertTrue(len(m2n_test.parameters) == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:232*

### test_fit_variant_2

**Category**: method_call  
**Description**: Tests the 'fit' method of the M2N class, using variant 2.  
**Expected**: self.assertTrue(len(m2n_test.parameters) == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.fit(mu_2_test)
self.assertTrue(len(m2n_test.parameters) == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:247*

### test_fit_success_via_error

**Category**: method_call  
**Description**: Tests that the 'fit' method successfully exits due to a low error being reached.  
**Expected**: self.assertTrue(len(m2n_test.parameters) == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.fit(mu_2_test)
self.assertTrue(len(m2n_test.parameters) == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:278*

### test_fit_success_via_epsilon

**Category**: method_call  
**Description**: Tests that the 'fit' method successfully exits due to p_1 converging.  
**Expected**: self.assertTrue(len(m2n_test.parameters) == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.fit(mu_2_test)
self.assertTrue(len(m2n_test.parameters) == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:295*

### test_fit_success_via_max_iter

**Category**: method_call  
**Description**: Tests that the 'fit' method successfully exits due to the maximum number of iterations being reached.  
**Expected**: self.assertTrue(len(m2n_test.parameters) == 5)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
m2n_test.fit(mu_2=mu_2_test)
self.assertTrue(len(m2n_test.parameters) == 5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:312*

### test_m2n_constructor

**Category**: method_call  
**Description**: Tests that the constructor of the M2N class executes properly.  
**Expected**: self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(m2n_test.moments, moments_test)
self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ef3m.py:25*

### test_get_feature_clusters

**Category**: method_call  
**Description**: Test get_feature_clusters arguments  
**Expected**: self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Create X, y datasets\n        '
self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

self.assertEqual(len(clustered_subsets), 2)
self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:47*

### test_get_feature_clusters

**Category**: method_call  
**Description**: Test get_feature_clusters arguments  
**Expected**: self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Create X, y datasets\n        '
self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)
self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:49*

### test_get_feature_clusters

**Category**: method_call  
**Description**: Test get_feature_clusters arguments  
**Expected**: self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(len(clustered_subsets), 2)
self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:47*

### test_get_feature_clusters

**Category**: method_call  
**Description**: Test get_feature_clusters arguments  
**Expected**: self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)
self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:49*

### test_bet_size_reserve_return_params

**Category**: method_call  
**Description**: Tests for successful execution of 'bet_size_reserve' using return_parameters=True.
Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
random numbers.  
**Expected**: self.assertTrue(events_active.equals(eval_events))  
**Confidence**: 0.85  
**Tags**: mock, unittest  

```python
self.assertEqual(test_params, eval_params)
self.assertTrue(events_active.equals(eval_events))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:244*

### test_bet_size_reserve_return_params

**Category**: method_call  
**Description**: Tests for successful execution of 'bet_size_reserve' using return_parameters=True.
Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
random numbers.  
**Expected**: self.assertTrue(events_active.equals(eval_events))  
**Confidence**: 0.85  
**Tags**: mock, unittest  

```python
# Setup
# Fixtures: mock_likely_parameters

self.assertEqual(test_params, eval_params)
self.assertTrue(events_active.equals(eval_events))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:244*

### test_get_signal

**Category**: method_call  
**Description**: Tests calculating the bet size from probability.  
**Expected**: self.assertEqual(0, len(return_val))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

self.assertIsInstance(return_val, pd.Series)
self.assertEqual(0, len(return_val))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:89*

### test_inv_price_power

**Category**: method_call  
**Description**: Test for the successful execution of 'inv_price_sigmoid'.  
**Expected**: self.assertEqual(f_i_pow, inv_price_power(f_i_pow, w_pow, 0.0))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(inv_p_pow, inv_price_power(f_i_pow, w_pow, m_pow), 7)
self.assertEqual(f_i_pow, inv_price_power(f_i_pow, w_pow, 0.0))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:235*

### test_get_signal

**Category**: method_call  
**Description**: Tests calculating the bet size from probability.  
**Expected**: self.assertEqual(0, len(return_val))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertIsInstance(return_val, pd.Series)
self.assertEqual(0, len(return_val))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:89*

### test_inv_price_power

**Category**: method_call  
**Description**: Test for the successful execution of 'inv_price_sigmoid'.  
**Expected**: self.assertEqual(f_i_pow, inv_price_power(f_i_pow, w_pow, 0.0))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertAlmostEqual(inv_p_pow, inv_price_power(f_i_pow, w_pow, m_pow), 7)
self.assertEqual(f_i_pow, inv_price_power(f_i_pow, w_pow, 0.0))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:235*

### test_hrp_long_short

**Category**: method_call  
**Description**: Test the Long Short Portfolio via side_weights Serries 1 for Long, -1 for Short (index=asset names)  
**Expected**: self.assertAlmostEqual(np.sum(weights), 0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

self.assertEqual(len(weights) - self.data.shape[1], 0)
self.assertAlmostEqual(np.sum(weights), 0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:50*

### test_quasi_diagnalization

**Category**: method_call  
**Description**: Test the quasi-diagnalisation step of HRP algorithm.  
**Expected**: assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:73*

### test_hrp_with_input_as_distance_matrix

**Category**: method_call  
**Description**: Test HRP when passing a distance matrix as input.  
**Expected**: self.assertTrue(len(weights) == self.data.shape[1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

self.assertTrue((weights >= 0).all())
self.assertTrue(len(weights) == self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:148*

### test_hrp_with_input_as_distance_matrix

**Category**: method_call  
**Description**: Test HRP when passing a distance matrix as input.  
**Expected**: self.assertAlmostEqual(np.sum(weights), 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

self.assertTrue(len(weights) == self.data.shape[1])
self.assertAlmostEqual(np.sum(weights), 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:149*

### test_hrp_with_linkage_method

**Category**: method_call  
**Description**: Test HRP when passing a custom linkage method.  
**Expected**: self.assertTrue(len(weights) == self.data.shape[1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

self.assertTrue((weights >= 0).all())
self.assertTrue(len(weights) == self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:162*

### test_hrp_with_linkage_method

**Category**: method_call  
**Description**: Test HRP when passing a custom linkage method.  
**Expected**: self.assertAlmostEqual(np.sum(weights), 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

self.assertTrue(len(weights) == self.data.shape[1])
self.assertAlmostEqual(np.sum(weights), 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:163*

### test_hrp_long_short

**Category**: method_call  
**Description**: Test the Long Short Portfolio via side_weights Serries 1 for Long, -1 for Short (index=asset names)  
**Expected**: self.assertAlmostEqual(np.sum(weights), 0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(len(weights) - self.data.shape[1], 0)
self.assertAlmostEqual(np.sum(weights), 0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:50*

### test_quasi_diagnalization

**Category**: method_call  
**Description**: Test the quasi-diagnalisation step of HRP algorithm.  
**Expected**: assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]  
**Confidence**: 0.85  
**Tags**: unittest  

```python
hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:73*

### test_hrp_with_input_as_distance_matrix

**Category**: method_call  
**Description**: Test HRP when passing a distance matrix as input.  
**Expected**: self.assertTrue(len(weights) == self.data.shape[1])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue((weights >= 0).all())
self.assertTrue(len(weights) == self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:148*

### test_hrp_with_input_as_distance_matrix

**Category**: method_call  
**Description**: Test HRP when passing a distance matrix as input.  
**Expected**: self.assertAlmostEqual(np.sum(weights), 1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue(len(weights) == self.data.shape[1])
self.assertAlmostEqual(np.sum(weights), 1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_hrp.py:149*

### test_sb_bagging_not_tree_base_estimator

**Category**: method_call  
**Description**: Test SB Bagging with non-tree base estimator (KNN)  
**Expected**: self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

sb_clf.fit(self.X_train, self.y_train_clf)
self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:142*

### test_sb_bagging_float_max_samples_warm_start_true

**Category**: method_call  
**Description**: Test SB Bagging with warm start = True and float max_samples  
**Expected**: self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))
self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:200*

### test_sb_classifier

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
test oos predictions values  
**Expected**: self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

sb_clf.fit(self.X_train, self.y_train_clf)
self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:278*

### test_sb_classifier

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
test oos predictions values  
**Expected**: self.assertEqual(sb_precision, 1.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

self.assertAlmostEqual(sb_accuracy, 0.66, delta=0.2)
self.assertEqual(sb_precision, 1.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:288*

### test_sb_classifier

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
test oos predictions values  
**Expected**: self.assertAlmostEqual(sb_roc_auc, 0.59, delta=0.2)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

self.assertEqual(sb_precision, 1.0)
self.assertAlmostEqual(sb_roc_auc, 0.59, delta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:289*

### test_sb_regressor

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Regressor  
**Expected**: self.assertTrue((sb_reg.X_time_index == self.X_train.index).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

self.assertTrue(self.X_train.index.isin(sb_reg.timestamp_int_index_mapping.index).all())
self.assertTrue((sb_reg.X_time_index == self.X_train.index).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:314*

### test_sb_regressor

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Regressor  
**Expected**: self.assertAlmostEqual(mae_sb_reg, 0.29, delta=0.1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data and get triple barrier events, generate features\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)
self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20, center=False).mean()
self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50, center=False).mean()
self.data['side'] = np.nan
long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
self.data.loc[long_signals, 'side'] = 1
self.data.loc[short_signals, 'side'] = -1
self.data['side'] = self.data['side'].shift(1)
daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
cusum_events = cusum_filter(self.data['close'], threshold=0.005)
vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=2)
meta_labeled_events = get_events(close=self.data['close'], t_events=cusum_events, pt_sl=[1, 4], target=daily_vol, min_ret=5e-05, num_threads=3, vertical_barrier_times=vertical_barriers, side_prediction=self.data['side'], verbose=False)
meta_labeled_events.dropna(inplace=True)
labels = get_bins(meta_labeled_events, self.data['close'])
ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)
unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)
X = self.data.loc[labels.index,].iloc[unique_samples].dropna()
labels = labels.loc[X.index, :]
X.loc[labels.index, 'y'] = labels.bin
for index, value in X.y.iteritems():
    X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
    X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
    X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
    X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
    X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)
features = ['label_prob_0.6', 'label_prob_0.2']
for prob in [0.5, 0.3, 0.2, 0.1]:
    for window in [2, 5, 10]:
        X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(window=window).mean()
        features.append('label_prob_{}_sma_{}'.format(prob, window))
X.dropna(inplace=True)
y = X.pop('y')
self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4, random_state=1, shuffle=False)
self.y_train_reg = 1 + self.y_train_clf
self.y_test_reg = 1 + self.y_test_clf
self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
self.price_bars_trim = self.data[(self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

self.assertAlmostEqual(mse_sb_reg, 0.16, delta=0.1)
self.assertAlmostEqual(mae_sb_reg, 0.29, delta=0.1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:321*

### test_sb_bagging_not_tree_base_estimator

**Category**: method_call  
**Description**: Test SB Bagging with non-tree base estimator (KNN)  
**Expected**: self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
sb_clf.fit(self.X_train, self.y_train_clf)
self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:142*

### test_sb_bagging_float_max_samples_warm_start_true

**Category**: method_call  
**Description**: Test SB Bagging with warm start = True and float max_samples  
**Expected**: self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
sb_clf.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)))
self.assertTrue((sb_clf.predict(self.X_train)[:10] == np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])).all)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:200*

### test_sb_classifier

**Category**: method_call  
**Description**: Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
test oos predictions values  
**Expected**: self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
sb_clf.fit(self.X_train, self.y_train_clf)
self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_sb_bagging.py:278*

### test_load_tick_sample

**Category**: method_call  
**Description**: Test load_tick_sample function.  
**Expected**: self.assertTrue('Price' in tick_sample_df.columns)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(tick_sample_df.shape[0], 100)
self.assertTrue('Price' in tick_sample_df.columns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_datasets.py:20*

### test_load_tick_sample

**Category**: method_call  
**Description**: Test load_tick_sample function.  
**Expected**: self.assertTrue('Volume' in tick_sample_df.columns)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue('Price' in tick_sample_df.columns)
self.assertTrue('Volume' in tick_sample_df.columns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_datasets.py:21*

### test_load_tick_sample

**Category**: method_call  
**Description**: Test load_tick_sample function.  
**Expected**: self.assertTrue('Price' in tick_sample_df.columns)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(tick_sample_df.shape[0], 100)
self.assertTrue('Price' in tick_sample_df.columns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_datasets.py:20*

### test_load_tick_sample

**Category**: method_call  
**Description**: Test load_tick_sample function.  
**Expected**: self.assertTrue('Volume' in tick_sample_df.columns)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue('Price' in tick_sample_df.columns)
self.assertTrue('Volume' in tick_sample_df.columns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_datasets.py:21*

### test_get_train_times_1

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train STARTS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:74*

### test_get_train_times_2

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train ENDS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:94*

### test_get_train_times_3

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train ENVELOPES test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:119*

### test_get_train_times_1

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train STARTS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:74*

### test_get_train_times_2

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train ENDS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:94*

### test_get_train_times_3

**Category**: method_call  
**Description**: Tests the get_train_times method for the case where the train ENVELOPES test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.log(f'train_times=\n{train_times_ok}')
self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:119*

### test_cla_efficient_frontier

**Category**: method_call  
**Description**: Test the calculation of the efficient frontier solution.  
**Expected**: assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and len(cla.efficient_frontier_sigma) == len(cla.weights.values)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla.allocate(asset_prices=self.data, solution='efficient_frontier', asset_names=self.data.columns)
assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and len(cla.efficient_frontier_sigma) == len(cla.weights.values)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:105*

### test_expected_returns_equals_means

**Category**: method_call  
**Description**: Test for condition when expected returns equal the mean value.  
**Expected**: assert cla.expected_returns[-1, 0] == 1e-05  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla._initialise(asset_prices=data, resample_by='B', expected_asset_returns=None, covariance_matrix=None)
assert cla.expected_returns[-1, 0] == 1e-05
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:152*

### test_flag_true_for_purge_num_err

**Category**: method_call  
**Description**: Test whether the flag becomes True in the purge num error function.  
**Expected**: assert not cla.weights  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla._purge_num_err(tol=1)
assert not cla.weights
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:219*

### test_cla_efficient_frontier

**Category**: method_call  
**Description**: Test the calculation of the efficient frontier solution.  
**Expected**: assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and len(cla.efficient_frontier_sigma) == len(cla.weights.values)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
cla.allocate(asset_prices=self.data, solution='efficient_frontier', asset_names=self.data.columns)
assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and len(cla.efficient_frontier_sigma) == len(cla.weights.values)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:105*

### test_expected_returns_equals_means

**Category**: method_call  
**Description**: Test for condition when expected returns equal the mean value.  
**Expected**: assert cla.expected_returns[-1, 0] == 1e-05  
**Confidence**: 0.85  
**Tags**: unittest  

```python
cla._initialise(asset_prices=data, resample_by='B', expected_asset_returns=None, covariance_matrix=None)
assert cla.expected_returns[-1, 0] == 1e-05
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:152*

### test_flag_true_for_purge_num_err

**Category**: method_call  
**Description**: Test whether the flag becomes True in the purge num error function.  
**Expected**: assert not cla.weights  
**Confidence**: 0.85  
**Tags**: unittest  

```python
cla._purge_num_err(tol=1)
assert not cla.weights
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:219*

### test_chow_test

**Category**: method_call  
**Description**: Test get_chow_type_stat function  
**Expected**: self.assertAlmostEqual(stats.max(), 0.179, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertEqual(log_prices.shape[0] - min_length * 2, stats.shape[0])
self.assertAlmostEqual(stats.max(), 0.179, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:40*

### test_chow_test

**Category**: method_call  
**Description**: Test get_chow_type_stat function  
**Expected**: self.assertAlmostEqual(stats.mean(), -0.653, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertAlmostEqual(stats.max(), 0.179, delta=0.001)
self.assertAlmostEqual(stats.mean(), -0.653, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:41*

### test_chow_test

**Category**: method_call  
**Description**: Test get_chow_type_stat function  
**Expected**: self.assertAlmostEqual(stats[3], -0.6649, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertAlmostEqual(stats.mean(), -0.653, delta=0.001)
self.assertAlmostEqual(stats[3], -0.6649, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:42*

### test_chu_stinchcombe_value_diff_function

**Category**: method_call  
**Description**: Test the values diff hidden function.  
**Expected**: self.assertEqual(1, two_sided_diff)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertEqual(-1, one_sided_diff)
self.assertEqual(1, two_sided_diff)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:52*

### test_chu_stinchcombe_value_diff_function

**Category**: method_call  
**Description**: Test the values diff hidden function.  
**Expected**: self.assertRaises(ValueError, _get_values_diff, test_type='rubbish', series=pd.Series([1, 2, 3, 4, 5]), index=0, ind=1)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertEqual(1, two_sided_diff)
self.assertRaises(ValueError, _get_values_diff, test_type='rubbish', series=pd.Series([1, 2, 3, 4, 5]), index=0, ind=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:53*

### test_chu_stinchcombe_white_test

**Category**: method_call  
**Description**: Test get_chu_stinchcombe_white_statistics function  
**Expected**: self.assertEqual(log_prices.shape[0] - 2, two_sided_test.shape[0])  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertEqual(log_prices.shape[0] - 2, one_sided_test.shape[0])
self.assertEqual(log_prices.shape[0] - 2, two_sided_test.shape[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:67*

### test_chu_stinchcombe_white_test

**Category**: method_call  
**Description**: Test get_chu_stinchcombe_white_statistics function  
**Expected**: self.assertAlmostEqual(one_sided_test.critical_value.max(), 3.265, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertEqual(log_prices.shape[0] - 2, two_sided_test.shape[0])
self.assertAlmostEqual(one_sided_test.critical_value.max(), 3.265, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:68*

### test_chu_stinchcombe_white_test

**Category**: method_call  
**Description**: Test get_chu_stinchcombe_white_statistics function  
**Expected**: self.assertAlmostEqual(one_sided_test.critical_value.mean(), 2.7809, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertAlmostEqual(one_sided_test.critical_value.max(), 3.265, delta=0.001)
self.assertAlmostEqual(one_sided_test.critical_value.mean(), 2.7809, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:70*

### test_chu_stinchcombe_white_test

**Category**: method_call  
**Description**: Test get_chu_stinchcombe_white_statistics function  
**Expected**: self.assertAlmostEqual(one_sided_test.critical_value[20], 2.4466, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertAlmostEqual(one_sided_test.critical_value.mean(), 2.7809, delta=0.001)
self.assertAlmostEqual(one_sided_test.critical_value[20], 2.4466, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:71*

### test_chu_stinchcombe_white_test

**Category**: method_call  
**Description**: Test get_chu_stinchcombe_white_statistics function  
**Expected**: self.assertAlmostEqual(one_sided_test.stat.max(), 3729.001, delta=0.001)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

self.assertAlmostEqual(one_sided_test.critical_value[20], 2.4466, delta=0.001)
self.assertAlmostEqual(one_sided_test.stat.max(), 3729.001, delta=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_structural_breaks.py:72*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.shape == db2.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertEqual(db1.shape[0], 1)
self.assertTrue(db1.shape == db2.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:41*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.shape == db3.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.shape == db2.shape)
self.assertTrue(db1.shape == db3.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:42*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.shape == db4.shape)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.shape == db3.shape)
self.assertTrue(db1.shape == db4.shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:43*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db2.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.shape == db4.shape)
self.assertTrue(np.all(db1.values == db2.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:44*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db3.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(np.all(db1.values == db2.values))
self.assertTrue(np.all(db1.values == db3.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:47*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(np.all(db1.values == db4.values))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(np.all(db1.values == db3.values))
self.assertTrue(np.all(db1.values == db4.values))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:48*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'open'] == 1200.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(np.all(db1.values == db4.values))
self.assertTrue(db1.loc[0, 'open'] == 1200.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:49*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'high'] == 1249.75)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.loc[0, 'open'] == 1200.0)
self.assertTrue(db1.loc[0, 'high'] == 1249.75)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:52*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'low'] == 1200.0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.loc[0, 'high'] == 1249.75)
self.assertTrue(db1.loc[0, 'low'] == 1200.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:53*

### test_day_bars

**Category**: method_call  
**Description**: Tests the seconds bars implementation.  
**Expected**: self.assertTrue(db1.loc[0, 'close'] == 1249.75)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data_time_bars.csv'

self.assertTrue(db1.loc[0, 'low'] == 1200.0)
self.assertTrue(db1.loc[0, 'close'] == 1249.75)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_time_data_structures.py:54*

### test_basic

**Category**: method_call  
**Description**: Tests for the basic case where the benchmark is a constant.  
**Expected**: pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx10 = self.data[:10].index

pd.testing.assert_series_equal(test1, test1_actual, check_names=False)
pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:45*

### test_basic

**Category**: method_call  
**Description**: Tests for the basic case where the benchmark is a constant.  
**Expected**: pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx10 = self.data[:10].index

pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:46*

### test_given_benchmark

**Category**: method_call  
**Description**: Tests comparing value to a dynamic benchmark.  
**Expected**: pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx10 = self.data[:10].index

pd.testing.assert_series_equal(test4, test4_actual)
pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:68*

### test_given_benchmark

**Category**: method_call  
**Description**: Tests comparing value to a dynamic benchmark.  
**Expected**: pd.testing.assert_series_equal(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx10 = self.data[:10].index

pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
pd.testing.assert_series_equal(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:69*

### test_resample

**Category**: method_call  
**Description**: Tests for when resampling is used.  
**Expected**: pd.testing.assert_frame_equal(test8, test8_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
self.idx10 = self.data[:10].index

pd.testing.assert_frame_equal(test7, test7b)
pd.testing.assert_frame_equal(test8, test8_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:88*

### test_basic

**Category**: method_call  
**Description**: Tests for the basic case where the benchmark is a constant.  
**Expected**: pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test1, test1_actual, check_names=False)
pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:45*

### test_basic

**Category**: method_call  
**Description**: Tests for the basic case where the benchmark is a constant.  
**Expected**: pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:46*

### test_given_benchmark

**Category**: method_call  
**Description**: Tests comparing value to a dynamic benchmark.  
**Expected**: pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test4, test4_actual)
pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:68*

### test_given_benchmark

**Category**: method_call  
**Description**: Tests comparing value to a dynamic benchmark.  
**Expected**: pd.testing.assert_series_equal(test6, test6_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
pd.testing.assert_series_equal(test6, test6_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:69*

### test_resample

**Category**: method_call  
**Description**: Tests for when resampling is used.  
**Expected**: pd.testing.assert_frame_equal(test8, test8_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test7, test7b)
pd.testing.assert_frame_equal(test8, test8_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_vs_benchmark.py:88*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertEqual(len(cv_gen.backtest_paths), cv_gen.n_splits - 1)
self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:50*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[0][1]['train'] == train_splits[0]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())
self.assertTrue((cv_gen.backtest_paths[0][1]['train'] == train_splits[0]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:51*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[1][0]['train'] == train_splits[1]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertTrue((cv_gen.backtest_paths[0][1]['train'] == train_splits[0]).all())
self.assertTrue((cv_gen.backtest_paths[1][0]['train'] == train_splits[1]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:52*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[1][2]['train'] == train_splits[5]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertTrue((cv_gen.backtest_paths[1][0]['train'] == train_splits[1]).all())
self.assertTrue((cv_gen.backtest_paths[1][2]['train'] == train_splits[5]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:53*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[-1][-1]['train'] == train_splits[-1]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertTrue((cv_gen.backtest_paths[1][2]['train'] == train_splits[5]).all())
self.assertTrue((cv_gen.backtest_paths[-1][-1]['train'] == train_splits[-1]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:54*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[-1][-2]['train'] == train_splits[-1]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertTrue((cv_gen.backtest_paths[-1][-1]['train'] == train_splits[-1]).all())
self.assertTrue((cv_gen.backtest_paths[-1][-2]['train'] == train_splits[-1]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:55*

### test_purge_and_embargo

**Category**: method_call  
**Description**: Test that purging and embargo works correctly  
**Expected**: self.assertEqual(len(train_splits_no[1]) - len(train_splits[1]), 10)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertEqual(len(train_splits_no[0]) - len(train_splits[0]), 5)
self.assertEqual(len(train_splits_no[1]) - len(train_splits[1]), 10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:82*

### test_purge_and_embargo

**Category**: method_call  
**Description**: Test that purging and embargo works correctly  
**Expected**: self.assertEqual(len(train_splits_no[-1]) - len(train_splits[-1]), 0)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Test Combinatorial Purged CV class\n        '
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'), data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'))

self.assertEqual(len(train_splits_no[1]) - len(train_splits[1]), 10)
self.assertEqual(len(train_splits_no[-1]) - len(train_splits[-1]), 0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:83*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertEqual(len(cv_gen.backtest_paths), cv_gen.n_splits - 1)
self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:50*

### test_test_times

**Category**: method_call  
**Description**: Tests the Combinatorial CV with various number of train, test splits  
**Expected**: self.assertTrue((cv_gen.backtest_paths[0][1]['train'] == train_splits[0]).all())  
**Confidence**: 0.85  
**Tags**: unittest  

```python
self.assertTrue((cv_gen.backtest_paths[0][0]['train'] == train_splits[0]).all())
self.assertTrue((cv_gen.backtest_paths[0][1]['train'] == train_splits[0]).all())
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_combinatorial_cross_validation.py:51*

### test_basic

**Category**: method_call  
**Description**: Test basic case for a small set with manually inputted results, with numerical and categorical outputs, with
no resampling or forward looking labels.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:42*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test3, excess_over_median(subset1, lag=False))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test3.drop(datetime.strptime('2008-01-21', '%Y-%m-%d'), inplace=True)
pd.testing.assert_frame_equal(test3, excess_over_median(subset1, lag=False))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:56*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test4, test4_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:80*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test6, test5_actual.apply(np.sign), check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test6, test5_actual.apply(np.sign), check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:81*

### test_forward

**Category**: method_call  
**Description**: Tests with lagged returns.  
**Expected**: pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign), check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign), check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:105*

### test_forward

**Category**: method_call  
**Description**: Tests with lagged returns.  
**Expected**: pd.testing.assert_frame_equal(test9, test9_actual, check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign), check_less_precise=True)
pd.testing.assert_frame_equal(test9, test9_actual, check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:106*

### test_forward

**Category**: method_call  
**Description**: Tests with lagged returns.  
**Expected**: pd.testing.assert_frame_equal(test10, test9_actual.apply(np.sign), check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

pd.testing.assert_frame_equal(test9, test9_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test10, test9_actual.apply(np.sign), check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:107*

### test_nan

**Category**: method_call  
**Description**: Tests to check that NaN values in prices get ignored.  
**Expected**: pd.testing.assert_frame_equal(test11, excess_over_median(subset), check_less_precise=True)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test11.drop('nan', axis=1, inplace=True)
pd.testing.assert_frame_equal(test11, excess_over_median(subset), check_less_precise=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:118*

### test_basic

**Category**: method_call  
**Description**: Test basic case for a small set with manually inputted results, with numerical and categorical outputs, with
no resampling or forward looking labels.  
**Expected**: pd.testing.assert_frame_equal(test2, test2_actual)  
**Confidence**: 0.85  
**Tags**: unittest  

```python
pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
pd.testing.assert_frame_equal(test2, test2_actual)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:42*

### test_resample_period

**Category**: method_call  
**Description**: Test numerical and categorical with a resample period.  
**Expected**: pd.testing.assert_frame_equal(test3, excess_over_median(subset1, lag=False))  
**Confidence**: 0.85  
**Tags**: unittest  

```python
test3.drop(datetime.strptime('2008-01-21', '%Y-%m-%d'), inplace=True)
pd.testing.assert_frame_equal(test3, excess_over_median(subset1, lag=False))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_median.py:56*

### test_up_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of UP weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(up1.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:39*

### test_up_uniform_solution

**Category**: instantiation  
**Description**: Instantiate UP: Tests UP with uniform capital allocation.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up3 = UP(2, weighted='uniform')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:61*

### test_up_uniform_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests UP with uniform capital allocation.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(up3.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:65*

### test_up_top_k_solution

**Category**: instantiation  
**Description**: Instantiate UP: Tests UP with top-k experts capital allocation.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up4 = UP(5, weighted='top-k', k=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:78*

### test_up_top_k_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests UP with top-k experts capital allocation.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(up4.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:82*

### test_up_wrong_method

**Category**: instantiation  
**Description**: Instantiate UP: Tests ValueError if the method is not 'hist_performance', 'uniform', or 'top-k'.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up5 = UP(5, weighted='random', k=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:95*

### test_up_recalculate_solution

**Category**: instantiation  
**Description**: Instantiate UP: Tests recalculate method in UP.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up6 = UP(3, weighted='top-k', k=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:105*

### test_up_recalculate_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests recalculate method in UP.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(up6.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:111*

### test_up_recalculate_error

**Category**: instantiation  
**Description**: Instantiate UP: Tests ValueError if k is greater number of experts for recalculate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up7 = UP(3, weighted='top-k', k=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:124*

### test_up_recalculate1_error

**Category**: instantiation  
**Description**: Instantiate UP: Tests ValueError if k is not an integer for recalculate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

up8 = UP(3, weighted='top-k', k=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_universal_portfolio.py:136*

### test_pamr_solution

**Category**: instantiation  
**Description**: Instantiate PAMR: Test the calculation of passive aggressive mean reversion with the original optimization
method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr = PAMR(optimization_method=0, epsilon=0.5, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:36*

### test_pamr_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of passive aggressive mean reversion with the original optimization
method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(pamr.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:40*

### test_pamr1_solution

**Category**: instantiation  
**Description**: Instantiate PAMR: Test the calculation of passive aggressive mean reversion with PAMR-1 optimization method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr1 = PAMR(optimization_method=1, epsilon=0.5, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:53*

### test_pamr1_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of passive aggressive mean reversion with PAMR-1 optimization method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(pamr1.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:57*

### test_pamr2_solution

**Category**: instantiation  
**Description**: Instantiate PAMR: Test the calculation of passive aggressive mean reversion with the PAMR-2 optimization method  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr2 = PAMR(optimization_method=2, epsilon=0.5, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:70*

### test_pamr2_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of passive aggressive mean reversion with the PAMR-2 optimization method  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(pamr2.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:74*

### test_pamr_epsilon_error

**Category**: instantiation  
**Description**: Instantiate PAMR: Tests ValueError if epsilon is less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr3 = PAMR(optimization_method=2, epsilon=-1, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:87*

### test_pamr_agg_error

**Category**: instantiation  
**Description**: Instantiate PAMR: Tests ValueError if aggressiveness is less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr4 = PAMR(optimization_method=2, epsilon=0.5, agg=-5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:97*

### test_pamr_method_error

**Category**: instantiation  
**Description**: Instantiate PAMR: Tests ValueError if optimization method is not 0, 1, or 2.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

pamr5 = PAMR(optimization_method=5, epsilon=0.5, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:107*

### test_pamr_solution

**Category**: instantiation  
**Description**: Instantiate PAMR: Test the calculation of passive aggressive mean reversion with the original optimization
method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
pamr = PAMR(optimization_method=0, epsilon=0.5, agg=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_passive_aggressive_mean_reversion.py:36*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate read_csv: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

test_sample = pd.read_csv(self.path)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:32*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate array: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

price_arr = np.array(test_sample.Price.values, dtype=float)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:33*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate ewma: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/tick_data.csv'

ewma_res = ewma(price_arr, window=20)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:34*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate read_csv: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
test_sample = pd.read_csv(self.path)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:32*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate array: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
price_arr = np.array(test_sample.Price.values, dtype=float)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:33*

### test_ewma

**Category**: instantiation  
**Description**: Instantiate ewma: Tests the imbalance dollar bars implementation.  
**Expected**: self.assertTrue(ewma_res.shape == price_arr.shape)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
ewma_res = ewma(price_arr, window=20)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fast_ewma.py:34*

### test_mu_solution

**Category**: instantiation  
**Description**: Instantiate EG: Test calculation of exponential gradient weights with multiplicative update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

multiplicative_update = EG(update_rule='MU')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:34*

### test_mu_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of exponential gradient weights with multiplicative update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(multiplicative_update.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:37*

### test_gp_solution

**Category**: instantiation  
**Description**: Instantiate EG: Test calculation of exponential gradient weights with gradient projection update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

gradient_projection = EG(update_rule='GP')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:49*

### test_gp_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of exponential gradient weights with gradient projection update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(gradient_projection.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:52*

### test_em_solution

**Category**: instantiation  
**Description**: Instantiate EG: Test calculation of exponential gradient weights with expectation maximization update rule.  
**Expected**: expectation_maximization.allocate(self.data, resample_by='M')  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

expectation_maximization = EG(update_rule='EM')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:64*

### test_em_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of exponential gradient weights with expectation maximization update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(expectation_maximization.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:67*

### test_wrong_update

**Category**: instantiation  
**Description**: Instantiate EG: Tests ValueError if the passing update rule is not correct.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

expectation_maximization = EG(update_rule='SS')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:79*

### test_mu_solution

**Category**: instantiation  
**Description**: Instantiate EG: Test calculation of exponential gradient weights with multiplicative update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
multiplicative_update = EG(update_rule='MU')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:34*

### test_mu_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of exponential gradient weights with multiplicative update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(multiplicative_update.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:37*

### test_gp_solution

**Category**: instantiation  
**Description**: Instantiate EG: Test calculation of exponential gradient weights with gradient projection update rule.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
gradient_projection = EG(update_rule='GP')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_exponential_gradient.py:49*

### test_fcorn_k_solution

**Category**: instantiation  
**Description**: Instantiate FCORNK: Test the calculation of FCORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k = FCORNK(window=1, rho=1, lambd=1, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:34*

### test_fcorn_k_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of FCORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(fcorn_k.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:38*

### test_fcorn_k_window_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k1 = FCORNK(window=2.5, rho=2, lambd=1, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:51*

### test_fcorn_k_window_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k2 = FCORNK(window=0, rho=2, lambd=1, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:57*

### test_fcorn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k3 = FCORNK(window=2, rho=2.5, lambd=1, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:67*

### test_fcorn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k4 = FCORNK(window=2, rho=0, lambd=1, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:73*

### test_fcorn_k_lambd_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if lambd is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k5 = FCORNK(window=2, rho=2, lambd=1.5, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:83*

### test_fcorn_k_lambd_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if lambd is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k6 = FCORNK(window=2, rho=2, lambd=0, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:89*

### test_fcorn_k_k_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if k is not an integer of greater than window * rho * lambd  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k7 = FCORNK(window=2, rho=2, lambd=2, k=16)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:99*

### test_fcorn_k_k_error

**Category**: instantiation  
**Description**: Instantiate FCORNK: Tests ValueError if k is not an integer of greater than window * rho * lambd  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn_k8 = FCORNK(window=2, rho=2, lambd=2, k=1.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning_k.py:105*

### test_get_linkage_corr

**Category**: instantiation  
**Description**: Instantiate array: Testing the creation of a linkage object from empirical correlation matrix and tree graph  
**Expected**: np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)), decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

dend_expected = np.array([(1, 4, 0.16853455, 2), (5, 0, 0.22227236, 3), (3, 6, 0.26530006, 4), (7, 2, 0.76268129, 5)])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:50*

### test_get_linkage_corr

**Category**: instantiation  
**Description**: Instantiate _get_linkage_corr: Testing the creation of a linkage object from empirical correlation matrix and tree graph  
**Expected**: np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)), decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

dendrogram = tic._get_linkage_corr(etf_classification_tree, etf_corr)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:61*

### test_get_linkage_corr

**Category**: instantiation  
**Description**: Instantiate _get_linkage_corr: Testing the creation of a linkage object from empirical correlation matrix and tree graph  
**Expected**: np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)), decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

dendrogram_alt = tic._get_linkage_corr(etf_classification_tree_alt, etf_corr)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:64*

### test_link_clusters

**Category**: instantiation  
**Description**: Instantiate array: Testing the transformation of linkage object from local linkage to global linkage  
**Expected**: np.testing.assert_almost_equal(link_new, link_expected, decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

lnk1 = np.array([[1, 3, 0.10526126, 2], [4, 2, 0.23105119, 3], [0, 5, 0.40104189, 4]])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:85*

### test_link_clusters

**Category**: instantiation  
**Description**: Instantiate array: Testing the transformation of linkage object from local linkage to global linkage  
**Expected**: np.testing.assert_almost_equal(link_new, link_expected, decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

link_expected = np.array([[1, 4, 0.10526126, 2], [5, 3, 0.23105119, 3], [0, 6, 0.40104189, 4]])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:92*

### test_link_clusters

**Category**: instantiation  
**Description**: Instantiate _link_clusters: Testing the transformation of linkage object from local linkage to global linkage  
**Expected**: np.testing.assert_almost_equal(link_new, link_expected, decimal=2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Initialize and load data\n        '
project_path = os.path.dirname(__file__)
price_data_path = project_path + '/test_data/stock_prices.csv'
self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col='Date')
ret_est = ReturnsEstimators()
self.returns_data = ret_est.calculate_returns(self.price_data)
classification_tree_path = project_path + '/test_data/classification_tree.csv'
self.classification_tree = pd.read_csv(classification_tree_path)

link_new = tic._link_clusters(lnk0, lnk1, items0, items1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_tic.py:97*

### test_plotting_efficient_frontier

**Category**: instantiation  
**Description**: Instantiate calculate_mean_historical_returns: Test the plotting of the efficient frontier.  
**Expected**: assert plot.axes.xaxis.label._text == 'Volatility'  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:148*

### test_plotting_efficient_frontier

**Category**: instantiation  
**Description**: Instantiate plot_efficient_frontier: Test the plotting of the efficient frontier.  
**Expected**: assert plot.axes.xaxis.label._text == 'Volatility'  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

plot = mvo.plot_efficient_frontier(covariance=covariance, max_return=1.0, expected_asset_returns=expected_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:151*

### test_exception_in_plotting_efficient_frontier

**Category**: instantiation  
**Description**: Instantiate calculate_mean_historical_returns: Test raising of exception when plotting the efficient frontier.  
**Expected**: assert len(plot._A) == 41  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:165*

### test_exception_in_plotting_efficient_frontier

**Category**: instantiation  
**Description**: Instantiate plot_efficient_frontier: Test raising of exception when plotting the efficient frontier.  
**Expected**: assert len(plot._A) == 41  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

plot = mvo.plot_efficient_frontier(covariance=covariance, max_return=1.0, expected_asset_returns=expected_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:168*

### test_mvo_with_input_as_returns_and_covariance

**Category**: instantiation  
**Description**: Instantiate calculate_mean_historical_returns: Test MVO when we pass expected returns and covariance matrix as input.  
**Expected**: mvo.allocate(covariance_matrix=covariance, expected_asset_returns=expected_returns, asset_names=self.data.columns)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:180*

### test_mvo_with_exponential_returns

**Category**: instantiation  
**Description**: Instantiate MeanVarianceOptimisation: Test the calculation of inverse-variance portfolio weights.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

mvo = MeanVarianceOptimisation(calculate_expected_returns='exponential')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:309*

### test_unknown_returns_calculation

**Category**: instantiation  
**Description**: Instantiate MeanVarianceOptimisation: Test ValueError on passing unknown returns calculation string.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

mvo = MeanVarianceOptimisation(calculate_expected_returns='unknown_returns')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:323*

### test_no_asset_names_by_passing_cov

**Category**: instantiation  
**Description**: Instantiate calculate_exponential_historical_returns: Test MVO when not supplying a list of asset names but passing covariance matrix as input  
**Expected**: mvo.allocate(expected_asset_returns=expected_returns, covariance_matrix=covariance)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_returns = ReturnsEstimators().calculate_exponential_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:486*

### test_valuerror_with_no_asset_names

**Category**: instantiation  
**Description**: Instantiate calculate_mean_historical_returns: Test ValueError when not supplying a list of asset names and no other input  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:502*

### test_plotting_efficient_frontier

**Category**: instantiation  
**Description**: Instantiate calculate_mean_historical_returns: Test the plotting of the efficient frontier.  
**Expected**: assert plot.axes.xaxis.label._text == 'Volatility'  
**Confidence**: 0.80  
**Tags**: unittest  

```python
expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_mean_variance.py:148*

### test_herc_with_input_as_returns

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when passing asset returns dataframe as input.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:166*

### test_herc_with_asset_returns_as_none

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when asset returns are not required for calculating the weights.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:179*

### test_herc_with_input_as_covariance_matrix

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when passing a covariance matrix as input.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:195*

### test_no_asset_names_with_asset_returns

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when not supplying a list of asset names and when the user passes asset_returns.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:233*

### test_value_error_with_no_asset_names

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test ValueError when not supplying a list of asset names and no other input  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:248*

### test_dendrogram_plot

**Category**: instantiation  
**Description**: Instantiate plot_clusters: Test if dendrogram plot object is correctly rendered.  
**Expected**: assert dendrogram.get('icoord')  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

dendrogram = herc.plot_clusters(assets=self.data.columns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:259*

### test_herc_with_input_as_returns

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when passing asset returns dataframe as input.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:166*

### test_herc_with_asset_returns_as_none

**Category**: instantiation  
**Description**: Instantiate calculate_returns: Test HERC when asset returns are not required for calculating the weights.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
returns = ReturnsEstimators().calculate_returns(asset_prices=self.data)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_herc.py:179*

### test_default_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of CRP weights with default settings.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:38*

### test_given_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate zeros: Tests the calculation of constant rebalanced portfolio weights with weights given initially.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weights = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:54*

### test_given_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of constant rebalanced portfolio weights with weights given initially.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:62*

### test_given_allocate_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate zeros: Test calculation of constant rebalanced portfolio weights with weights given in allocate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weights = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:78*

### test_given_allocate_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of constant rebalanced portfolio weights with weights given in allocate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:86*

### test_default_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of CRP weights with default settings.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:38*

### test_given_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate zeros: Tests the calculation of constant rebalanced portfolio weights with weights given initially.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
weights = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:54*

### test_given_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of constant rebalanced portfolio weights with weights given initially.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:62*

### test_given_allocate_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate zeros: Test calculation of constant rebalanced portfolio weights with weights given in allocate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
weights = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:78*

### test_given_allocate_weights_crp_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of constant rebalanced portfolio weights with weights given in allocate.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(crp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_constant_rebalanced_portfolio.py:86*

### test_variance_calculation

**Category**: instantiation  
**Description**: Instantiate calculate_variance: Test the calculation of variance.  
**Expected**: assert isinstance(variance, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

variance = RiskMetrics().calculate_variance(self.data.cov(), weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:27*

### test_value_at_risk_calculation

**Category**: instantiation  
**Description**: Instantiate calculate_value_at_risk: Test the calculation of value at risk.  
**Expected**: assert isinstance(value_at_risk, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

value_at_risk = RiskMetrics().calculate_value_at_risk(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:36*

### test_expected_shortfall_calculation

**Category**: instantiation  
**Description**: Instantiate calculate_expected_shortfall: Test the calculation of expected shortfall.  
**Expected**: assert isinstance(expected_shortfall, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_shortfall = RiskMetrics().calculate_expected_shortfall(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:45*

### test_conditional_drawdown_calculation

**Category**: instantiation  
**Description**: Instantiate calculate_conditional_drawdown_risk: Test the calculation of conditional drawdown at risk.  
**Expected**: assert isinstance(conditional_drawdown, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

conditional_drawdown = RiskMetrics().calculate_conditional_drawdown_risk(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:54*

### test_value_at_risk_for_dataframe

**Category**: instantiation  
**Description**: Instantiate DataFrame: Test the calculation of value at risk.  
**Expected**: assert isinstance(value_at_risk, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

test_returns = pd.DataFrame(self.data.iloc[:, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:62*

### test_value_at_risk_for_dataframe

**Category**: instantiation  
**Description**: Instantiate calculate_value_at_risk: Test the calculation of value at risk.  
**Expected**: assert isinstance(value_at_risk, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

value_at_risk = RiskMetrics().calculate_value_at_risk(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:63*

### test_expected_shortfall_for_dataframe

**Category**: instantiation  
**Description**: Instantiate DataFrame: Test the calculation of expected shortfall.  
**Expected**: assert isinstance(expected_shortfall, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

test_returns = pd.DataFrame(self.data.iloc[:, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:71*

### test_expected_shortfall_for_dataframe

**Category**: instantiation  
**Description**: Instantiate calculate_expected_shortfall: Test the calculation of expected shortfall.  
**Expected**: assert isinstance(expected_shortfall, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

expected_shortfall = RiskMetrics().calculate_expected_shortfall(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:72*

### test_conditional_drawdown_for_dataframe

**Category**: instantiation  
**Description**: Instantiate DataFrame: Test the calculation of conditional drawdown at risk.  
**Expected**: assert isinstance(conditional_drawdown, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

test_returns = pd.DataFrame(self.data.iloc[:, 0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:80*

### test_conditional_drawdown_for_dataframe

**Category**: instantiation  
**Description**: Instantiate calculate_conditional_drawdown_risk: Test the calculation of conditional drawdown at risk.  
**Expected**: assert isinstance(conditional_drawdown, float)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

conditional_drawdown = RiskMetrics().calculate_conditional_drawdown_risk(test_returns)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_risk_metrics.py:81*

### test_cwmr_solution

**Category**: instantiation  
**Description**: Instantiate CWMR: Test the calculation of CWMR with the original method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr = CWMR(confidence=0.5, epsilon=0.5, method='var')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:34*

### test_cwmr_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CWMR with the original method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(cwmr.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:38*

### test_cwmr_sd_solution

**Category**: instantiation  
**Description**: Instantiate CWMR: Test the calculation of CWMR with the second method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr = CWMR(confidence=0.5, epsilon=0.5, method='sd')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:51*

### test_cwmr_sd_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CWMR with the second method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(cwmr.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:55*

### test_cwmr_epsilon_error

**Category**: instantiation  
**Description**: Instantiate CWMR: Tests ValueError if epsilon is greater than 1 or less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr1 = CWMR(confidence=0.5, epsilon=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:68*

### test_cwmr_epsilon_error

**Category**: instantiation  
**Description**: Instantiate CWMR: Tests ValueError if epsilon is greater than 1 or less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr2 = CWMR(confidence=0.5, epsilon=-1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:74*

### test_cwmr_confidence_error

**Category**: instantiation  
**Description**: Instantiate CWMR: Tests ValueError if confidence is greater than 1 or less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr3 = CWMR(confidence=2, epsilon=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:84*

### test_cwmr_confidence_error

**Category**: instantiation  
**Description**: Instantiate CWMR: Tests ValueError if confidence is greater than 1 or less than 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr4 = CWMR(confidence=-1, epsilon=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:90*

### test_cwmr_method_error

**Category**: instantiation  
**Description**: Instantiate CWMR: Tests ValueError if method is not 'sd' or 'var'.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

cwmr5 = CWMR(confidence=0.5, epsilon=0.5, method='normal')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:100*

### test_cwmr_weights_solution

**Category**: instantiation  
**Description**: Instantiate zeros: Test the calculation of CWMR with given weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weight = np.zeros(self.data.iloc[0].shape)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_confidence_weighted_mean_reversion.py:110*

### test_get_weights

**Category**: instantiation  
**Description**: Instantiate get_weights: get_weights as implemented here matches the code in the book (Snippet 5.1).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is same as the requested length  
**Expected**: self.assertTrue(weights[-1] == 1.0)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

weights = fracdiff.get_weights(diff_amt, size=number_ele)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:38*

### test_get_weights_ffd

**Category**: instantiation  
**Description**: Instantiate get_weights_ffd: get_weights_ffd as implemented here matches the code in the book (Snippet 5.2).
We test:
1. if the first element of the weights vector is 1.0
2. The length of the weights vector is equal to 12  
**Expected**: self.assertTrue(weights[-1] == 1.0)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

weights = fracdiff.get_weights_ffd(diff_amt, thresh=thresh, lim=number_ele)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:58*

### test_frac_diff

**Category**: instantiation  
**Description**: Instantiate frac_diff: Assert that for any positive real number d,
1. Length of the output is the same as the length of the input
2. First element is NaN  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

fd_series = fracdiff.frac_diff(data_series, diff_amt=diff_amt)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:75*

### test_frac_diff_ffd

**Category**: instantiation  
**Description**: Instantiate frac_diff_ffd: Assert that for any positive real number d,
1. Length of the output is the same as the length of the input
2. First element is NaN  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

fd_series = fracdiff.frac_diff_ffd(data_series, diff_amt=diff_amt)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fracdiff.py:88*

### test_cusum_filter

**Category**: instantiation  
**Description**: Instantiate cusum_filter: Assert that the CUSUM filter works as expected.
Checks that all the events generated by different threshold values meet the requirements of the filter.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

cusum_events = cusum_filter(self.data['close'], threshold=threshold, time_stamps=timestamp)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:37*

### test_cusum_filter

**Category**: instantiation  
**Description**: Instantiate get_loc: Assert that the CUSUM filter works as expected.
Checks that all the events generated by different threshold values meet the requirements of the filter.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

event_1 = self.data.index.get_loc(cusum_events[i - 1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:40*

### test_cusum_filter

**Category**: instantiation  
**Description**: Instantiate get_loc: Assert that the CUSUM filter works as expected.
Checks that all the events generated by different threshold values meet the requirements of the filter.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

event_2 = self.data.index.get_loc(cusum_events[i])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:41*

### test_cusum_filter

**Category**: instantiation  
**Description**: Instantiate log: Assert that the CUSUM filter works as expected.
Checks that all the events generated by different threshold values meet the requirements of the filter.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/dollar_bar_sample.csv'
self.data = pd.read_csv(self.path, index_col='date_time')
self.data.index = pd.to_datetime(self.data.index)

last = np.log(date_range[-1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_filters.py:44*

### test_small_set

**Category**: instantiation  
**Description**: Instantiate excess_over_mean: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test1 = excess_over_mean(subset)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:31*

### test_small_set

**Category**: instantiation  
**Description**: Instantiate excess_over_mean: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test2 = excess_over_mean(subset, binary=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:32*

### test_small_set

**Category**: instantiation  
**Description**: Instantiate DataFrame: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test1_actual = pd.DataFrame([(0.005666, -0.006157, 4.411e-05, 0.0004476), (-0.011169, -0.000684, 0.018588, -0.006734), (0.006871, 0.000411, -0.000643, -0.006639), (-0.003687, -0.002863, 0.004815, 0.001735), (np.nan, np.nan, np.nan, np.nan)], columns=self.data[cols].iloc[0:5].columns, index=self.data[cols].iloc[0:5].index)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:33*

### test_small_set

**Category**: instantiation  
**Description**: Instantiate apply: Check for a small set with manually inputted results, with numerical and categorical outputs.  
**Expected**: pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample data.\n        '
project_path = os.path.dirname(__file__)
self.path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(self.path, index_col='Date')
self.data.index = pd.to_datetime(self.data.index)

test2_actual = test1_actual.apply(np.sign)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_labeling_over_mean.py:39*

### test_get_onc_clusters

**Category**: instantiation  
**Description**: Instantiate get_onc_clusters: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertGreaterEqual(len(clusters.keys()), 5)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.data, _ = load_breast_cancer(return_X_y=True)

_, clusters, _ = get_onc_clusters(pd.DataFrame(self.data).corr(), repeat=50)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:41*

### test_get_onc_clusters

**Category**: instantiation  
**Description**: Instantiate get_onc_clusters: Test get_onc_clusters function on Breast Cancer data set from sklearn  
**Expected**: self.assertGreaterEqual(len(clusters.keys()), 5)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
_, clusters, _ = get_onc_clusters(pd.DataFrame(self.data).corr(), repeat=50)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_onc.py:41*

### test_olps_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of OLPS weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(olps.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:40*

### test_olps_weight

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that the user inputted weights have matching dimensions as the data's dimensions
and ValueError if the user inputted weights do not sum to one.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:63*

### test_user_weight

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that users can input their own weights for OLPS.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:95*

### test_user_weight

**Category**: instantiation  
**Description**: Instantiate array: Tests that users can input their own weights for OLPS.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(olps5.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:102*

### test_normalize

**Category**: instantiation  
**Description**: Instantiate ones: Tests that weights sum to 1.  
**Expected**: np.testing.assert_almost_equal(normalized_weight, random_weight / 3)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

random_weight = np.ones(3)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:132*

### test_normalize

**Category**: instantiation  
**Description**: Instantiate _normalize: Tests that weights sum to 1.  
**Expected**: np.testing.assert_almost_equal(normalized_weight, random_weight / 3)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

normalized_weight = olps7._normalize(random_weight)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:134*

### test_simplex_projection

**Category**: instantiation  
**Description**: Instantiate _simplex_projection: Tests edge cases where the inputted weights already satisfy the simplex requirements.  
**Expected**: np.testing.assert_almost_equal(weights, simplex_weights)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

simplex_weights = olps8._simplex_projection(weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:149*

### test_simplex_all_negatives

**Category**: instantiation  
**Description**: Instantiate array: Tests case where negative weights have to be projected onto the simplex.  
**Expected**: np.testing.assert_almost_equal(olps10._simplex_projection(neg_weight), np.array([0.5, 0.5]))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

neg_weight = np.array([-1e+21, -1e+21])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:171*

### test_olps_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of OLPS weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(olps.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:40*

### test_olps_weight

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that the user inputted weights have matching dimensions as the data's dimensions
and ValueError if the user inputted weights do not sum to one.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_portfolio_selection.py:63*

### test_buy_and_hold_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of buy and hold weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(bah.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:38*

### test_buy_two_assets

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that weights are changing for a portfolio of two assets.  
**Expected**: np.testing.assert_almost_equal(last_bah1_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:51*

### test_buy_two_assets

**Category**: instantiation  
**Description**: Instantiate array: Tests that weights are changing for a portfolio of two assets.  
**Expected**: np.testing.assert_almost_equal(last_bah1_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:60*

### test_buy_five_assets

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that weights are changing for a portfolio of five assets.  
**Expected**: np.testing.assert_almost_equal(last_bah2_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:73*

### test_buy_five_assets

**Category**: instantiation  
**Description**: Instantiate array: Tests that weights are changing for a portfolio of five assets.  
**Expected**: np.testing.assert_almost_equal(last_bah2_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:82*

### test_buy_and_hold_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of buy and hold weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(bah.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:38*

### test_buy_two_assets

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that weights are changing for a portfolio of two assets.  
**Expected**: np.testing.assert_almost_equal(last_bah1_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:51*

### test_buy_two_assets

**Category**: instantiation  
**Description**: Instantiate array: Tests that weights are changing for a portfolio of two assets.  
**Expected**: np.testing.assert_almost_equal(last_bah1_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:60*

### test_buy_five_assets

**Category**: instantiation  
**Description**: Instantiate zeros: Tests that weights are changing for a portfolio of five assets.  
**Expected**: np.testing.assert_almost_equal(last_bah2_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
weight = np.zeros(self.data.shape[1])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:73*

### test_buy_five_assets

**Category**: instantiation  
**Description**: Instantiate array: Tests that weights are changing for a portfolio of five assets.  
**Expected**: np.testing.assert_almost_equal(last_bah2_weight, norm_new_weight)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_buy_and_hold.py:82*

### test_ftl_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of follow the leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_leader.py:38*

### test_ftl_first_weight

**Category**: instantiation  
**Description**: Instantiate array: Tests that the weights calculated for the first time period is uniform.  
**Expected**: np.testing.assert_almost_equal(uniform_weight, all_weights[0])  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftl1.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_leader.py:53*

### test_ftl_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of follow the leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(ftl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_leader.py:38*

### test_ftl_first_weight

**Category**: instantiation  
**Description**: Instantiate array: Tests that the weights calculated for the first time period is uniform.  
**Expected**: np.testing.assert_almost_equal(uniform_weight, all_weights[0])  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(ftl1.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_leader.py:53*

### test_scorn_k_solution

**Category**: instantiation  
**Description**: Instantiate SCORNK: Test the calculation of SCORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k = SCORNK(window=2, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:35*

### test_scorn_k_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of SCORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(scorn_k.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:39*

### test_scorn_k_window_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k1 = SCORNK(window=2.5, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:52*

### test_scorn_k_window_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k2 = SCORNK(window=0, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:58*

### test_scorn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k3 = SCORNK(window=2, rho=2.5, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:68*

### test_scorn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k4 = SCORNK(window=2, rho=0, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:74*

### test_scorn_k_k_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k5 = SCORNK(window=2, rho=2, k=5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:84*

### test_scorn_k_k_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k6 = SCORNK(window=2, rho=2, k=1.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:90*

### test_scorn_k_k_error

**Category**: instantiation  
**Description**: Instantiate SCORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn_k7 = SCORNK(window=2, rho=2, k=0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:96*

### test_scorn_k_solution

**Category**: instantiation  
**Description**: Instantiate SCORNK: Test the calculation of SCORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
scorn_k = SCORNK(window=2, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning_k.py:35*

### test_corn_u_solution

**Category**: instantiation  
**Description**: Instantiate CORNU: Test the calculation of CORN-U.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_u = CORNU(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:33*

### test_corn_u_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CORN-U.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(corn_u.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:37*

### test_corn_u_window_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_u1 = CORNU(window=2.5, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:50*

### test_corn_u_window_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_u2 = CORNU(window=0, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:56*

### test_corn_u_rho_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_u3 = CORNU(window=2, rho=-2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:66*

### test_corn_u_rho_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_u4 = CORNU(window=2, rho=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:72*

### test_corn_u_solution

**Category**: instantiation  
**Description**: Instantiate CORNU: Test the calculation of CORN-U.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn_u = CORNU(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:33*

### test_corn_u_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CORN-U.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(corn_u.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:37*

### test_corn_u_window_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn_u1 = CORNU(window=2.5, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:50*

### test_corn_u_window_error

**Category**: instantiation  
**Description**: Instantiate CORNU: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn_u2 = CORNU(window=0, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_uniform.py:56*

### test_corn_k_solution

**Category**: instantiation  
**Description**: Instantiate CORNK: Test the calculation of CORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k = CORNK(window=2, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:33*

### test_corn_k_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(corn_k.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:37*

### test_corn_k_window_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k1 = CORNK(window=2.5, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:50*

### test_corn_k_window_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k2 = CORNK(window=0, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:56*

### test_corn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k3 = CORNK(window=2, rho=2.5, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:66*

### test_corn_k_rho_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if rho is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k4 = CORNK(window=2, rho=0, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:72*

### test_corn_k_k_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k5 = CORNK(window=2, rho=2, k=5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:82*

### test_corn_k_k_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k6 = CORNK(window=2, rho=2, k=1.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:88*

### test_corn_k_k_error

**Category**: instantiation  
**Description**: Instantiate CORNK: Tests ValueError if k is greater than window * rho, greater than 1, or an integer.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn_k7 = CORNK(window=2, rho=2, k=0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:94*

### test_corn_k_solution

**Category**: instantiation  
**Description**: Instantiate CORNK: Test the calculation of CORN-K.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn_k = CORNK(window=2, rho=2, k=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning_k.py:33*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Create X, y datasets\n        '
self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

clustered_subsets = get_feature_clusters(self.X, dependence_metric='information_variation', distance_metric='angular', linkage_method='single', n_clusters=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:27*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Create X, y datasets\n        '
self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

clustered_subsets_ha = get_feature_clusters(self.X, dependence_metric='linear', distance_metric='angular', linkage_method='single', n_clusters=None, critical_threshold=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:31*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Create X, y datasets\n        '
self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

clustered_subsets_distance = get_feature_clusters(self.X, dependence_metric='linear', distance_metric=None, linkage_method=None, n_clusters=None, critical_threshold=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:41*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
clustered_subsets = get_feature_clusters(self.X, dependence_metric='information_variation', distance_metric='angular', linkage_method='single', n_clusters=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:27*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
clustered_subsets_ha = get_feature_clusters(self.X, dependence_metric='linear', distance_metric='angular', linkage_method='single', n_clusters=None, critical_threshold=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:31*

### test_get_feature_clusters

**Category**: instantiation  
**Description**: Instantiate get_feature_clusters: Test get_feature_clusters arguments  
**Expected**: self.assertEqual(len(clustered_subsets), 2)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
clustered_subsets_distance = get_feature_clusters(self.X, dependence_metric='linear', distance_metric=None, linkage_method=None, n_clusters=None, critical_threshold=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_feature_clusters.py:41*

### test_bet_size_probability_default

**Category**: instantiation  
**Description**: Instantiate array: Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:31*

### test_bet_size_probability_default

**Category**: instantiation  
**Description**: Instantiate array: Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
shift_dt = np.array([dt.timedelta(days=0.5 * i + 1) for i in range(5)])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:32*

### test_bet_size_probability_default

**Category**: instantiation  
**Description**: Instantiate DataFrame: Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
events_test = pd.DataFrame(data=[[0.55, 1], [0.7, -1], [0.95, 1], [0.65, -1], [0.85, 1]], columns=['prob', 'side'], index=dates_test)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:34*

### test_bet_size_probability_default

**Category**: instantiation  
**Description**: Instantiate get_signal: Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
signal_0 = get_signal(events_test['prob'], 2, events_test['side'])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:39*

### test_bet_size_probability_default

**Category**: instantiation  
**Description**: Instantiate join: Tests for successful execution using the default arguments of 'bet_size_probability', which are:
 average_active = False
 step_size = 0.0  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
df_signal_0 = signal_0.to_frame('signal').join(events_test['t1'], how='left')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:40*

### test_bet_size_probability_avg_active

**Category**: instantiation  
**Description**: Instantiate array: Tests for successful execution of 'bet_size_probability' with 'average_active' set to True.  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events=events_test, prob=events_test['prob'], num_classes=2, pred=events_test['side'], average_active=True)))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:50*

### test_bet_size_probability_avg_active

**Category**: instantiation  
**Description**: Instantiate array: Tests for successful execution of 'bet_size_probability' with 'average_active' set to True.  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events=events_test, prob=events_test['prob'], num_classes=2, pred=events_test['side'], average_active=True)))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
shift_dt = np.array([dt.timedelta(days=0.5 * i + 1) for i in range(5)])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:51*

### test_bet_size_probability_avg_active

**Category**: instantiation  
**Description**: Instantiate DataFrame: Tests for successful execution of 'bet_size_probability' with 'average_active' set to True.  
**Expected**: self.assertTrue(signal_1.equals(bet_size_probability(events=events_test, prob=events_test['prob'], num_classes=2, pred=events_test['side'], average_active=True)))  
**Confidence**: 0.80  
**Tags**: unittest  

```python
events_test = pd.DataFrame(data=[[0.55, 1], [0.7, -1], [0.95, 1], [0.65, -1], [0.85, 1]], columns=['prob', 'side'], index=dates_test)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_bet_sizing.py:53*

### test_bcrp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of best constant rebalanced portfolio weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(bcrp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_constant_rebalanced_portfolio.py:39*

### test_bcrp_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests the calculation of best constant rebalanced portfolio weights.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(bcrp.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_constant_rebalanced_portfolio.py:39*

### test_get_signal

**Category**: instantiation  
**Description**: Instantiate get_signal: Tests calculating the bet size from probability.  
**Expected**: self.assertEqual(self.bet_size.equals(test_bet_size_1), True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

test_bet_size_1 = get_signal(prob=self.prob, num_classes=2, pred=self.side)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:79*

### test_get_signal

**Category**: instantiation  
**Description**: Instantiate get_signal: Tests calculating the bet size from probability.  
**Expected**: self.assertEqual(self.bet_size.abs().equals(test_bet_size_2), True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

test_bet_size_2 = get_signal(prob=self.prob, num_classes=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:83*

### test_get_signal

**Category**: instantiation  
**Description**: Instantiate DataFrame: Tests calculating the bet size from probability.  
**Expected**: self.assertIsInstance(return_val, pd.Series)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

df_empty = pd.DataFrame({'a': []})
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:87*

### test_get_signal

**Category**: instantiation  
**Description**: Instantiate get_signal: Tests calculating the bet size from probability.  
**Expected**: self.assertIsInstance(return_val, pd.Series)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

return_val = get_signal(df_empty, 2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:88*

### test_avg_active_signals

**Category**: instantiation  
**Description**: Instantiate avg_active_signals: Tests the avg_active_signals function. Also implicitly tests the
molecular multiprocessing function mp_avg_active_signals.  
**Expected**: self.assertEqual(self.avg_active.equals(test_avg_active), True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

test_avg_active = avg_active_signals(self.events_2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:97*

### test_mp_avg_active_signals

**Category**: instantiation  
**Description**: Instantiate mp_avg_active_signals: An explicit test of the mp_avg_active_signals subroutine.  
**Expected**: self.assertEqual(self.avg_active.equals(test_mp_avg_active), True)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets up the data to be used for the following tests.\n        '
prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
side_arr = np.array([1, 1, -1, 1, -1, 1])
dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(6)])
shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
dates_shifted = dates + shift_dt
z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal / 0.1, 0) * 0.1])
self.prob = pd.Series(data=prob_arr, index=dates)
self.side = pd.Series(data=side_arr, index=dates)
self.t_1 = pd.Series(data=dates_shifted, index=dates)
self.bet_size = pd.Series(data=m_signal, index=dates)
self.bet_size_d = pd.Series(data=m_discrete, index=dates)
self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
self.events_2 = self.events.copy()
self.events_2['signal'] = self.bet_size
t_p = set(self.events_2['t1'].to_numpy())
t_p = t_p.union(self.events_2.index.to_numpy())
t_p = list(t_p)
t_p.sort()
self.t_pnts = t_p.copy()
avg_list = []
for t_i in t_p:
    avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

test_mp_avg_active = mp_avg_active_signals(self.events_2, self.t_pnts)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_ch10_snippets.py:104*

### test_corn_solution

**Category**: instantiation  
**Description**: Instantiate CORN: Test the calculation of CORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn = CORN(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:34*

### test_corn_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(corn.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:38*

### test_corn_window_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn1 = CORN(window=2.5, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:51*

### test_corn_window_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn2 = CORN(window=0, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:57*

### test_corn_rho_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn3 = CORN(window=2, rho=-2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:67*

### test_corn_rho_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

corn4 = CORN(window=2, rho=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:73*

### test_corn_solution

**Category**: instantiation  
**Description**: Instantiate CORN: Test the calculation of CORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn = CORN(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:34*

### test_corn_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of CORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(corn.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:38*

### test_corn_window_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn1 = CORN(window=2.5, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:51*

### test_corn_window_error

**Category**: instantiation  
**Description**: Instantiate CORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
corn2 = CORN(window=0, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_correlation_driven_nonparametric_learning.py:57*

### test_get_train_times_1

**Category**: instantiation  
**Description**: Instantiate Series: Tests the get_train_times method for the case where the train STARTS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

test_times = pd.Series(index=pd.date_range(start='2019-01-01 00:01:00', periods=1, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=1, freq='T'))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:62*

### test_get_train_times_1

**Category**: instantiation  
**Description**: Instantiate ml_get_train_times: Tests the get_train_times method for the case where the train STARTS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

train_times_ret = ml_get_train_times(self.info_sets, test_times)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:67*

### test_get_train_times_1

**Category**: instantiation  
**Description**: Instantiate Series: Tests the get_train_times method for the case where the train STARTS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

train_times_ok = pd.Series(index=pd.date_range(start='2019-01-01 00:03:00', end='2019-01-01 00:09:00', freq='T'), data=pd.date_range(start='2019-01-01 00:05:00', end='2019-01-01 00:11:00', freq='T'))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:70*

### test_get_train_times_2

**Category**: instantiation  
**Description**: Instantiate Series: Tests the get_train_times method for the case where the train ENDS within test.  
**Expected**: self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        This is how the observations dataset looks like\n        2019-01-01 00:00:00   2019-01-01 00:02:00\n        2019-01-01 00:01:00   2019-01-01 00:03:00\n        2019-01-01 00:02:00   2019-01-01 00:04:00\n        2019-01-01 00:03:00   2019-01-01 00:05:00\n        2019-01-01 00:04:00   2019-01-01 00:06:00\n        2019-01-01 00:05:00   2019-01-01 00:07:00\n        2019-01-01 00:06:00   2019-01-01 00:08:00\n        2019-01-01 00:07:00   2019-01-01 00:09:00\n        2019-01-01 00:08:00   2019-01-01 00:10:00\n        2019-01-01 00:09:00   2019-01-01 00:11:00\n        '
pwd_path = os.path.dirname(__file__)
self.log(f'pwd_path= {pwd_path}')
self.info_sets = pd.Series(index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'), data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'))

test_times = pd.Series(index=pd.date_range(start='2019-01-01 00:08:00', periods=1, freq='T'), data=pd.date_range(start='2019-01-01 00:11:00', periods=1, freq='T'))
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cross_validation.py:82*

### test_cla_with_mean_returns

**Category**: instantiation  
**Description**: Instantiate CriticalLineAlgorithm: Test the calculation of CLA turning points using mean returns.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla = CriticalLineAlgorithm(weight_bounds=(0, 1), calculate_expected_returns='mean')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:36*

### test_cla_with_weight_bounds_as_lists

**Category**: instantiation  
**Description**: Instantiate CriticalLineAlgorithm: Test the calculation of CLA turning points when we pass the weight bounds as a list
instead of just lower and upper bound value.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla = CriticalLineAlgorithm(weight_bounds=([0] * self.data.shape[1], [1] * self.data.shape[1]), calculate_expected_returns='mean')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:51*

### test_cla_with_exponential_returns

**Category**: instantiation  
**Description**: Instantiate CriticalLineAlgorithm: Test the calculation of CLA turning points using exponential returns  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla = CriticalLineAlgorithm(weight_bounds=(0, 1), calculate_expected_returns='exponential')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:66*

### test_cla_max_sharpe

**Category**: instantiation  
**Description**: Instantiate CriticalLineAlgorithm: Test the calculation of maximum sharpe ratio weights.  
**Expected**: assert (weights >= 0).all()  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date')

cla = CriticalLineAlgorithm(weight_bounds=(0, 1), calculate_expected_returns='mean')
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_cla.py:80*

### test_ftrl_solution

**Category**: instantiation  
**Description**: Instantiate FTRL: Test calculation of follow the regularized leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

ftrl = FTRL(beta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:34*

### test_ftrl_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:37*

### test_beta_0_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader for beta value of 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:52*

### test_beta_1_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader for beta value of 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:67*

### test_beta_10_solution

**Category**: instantiation  
**Description**: Instantiate FTRL: Test calculation of follow the regularized leader for beta value of 10.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

ftrl = FTRL(beta=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:79*

### test_beta_10_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader for beta value of 10.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:82*

### test_ftrl_solution

**Category**: instantiation  
**Description**: Instantiate FTRL: Test calculation of follow the regularized leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
ftrl = FTRL(beta=0.2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:34*

### test_ftrl_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:37*

### test_beta_0_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader for beta value of 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:52*

### test_beta_1_solution

**Category**: instantiation  
**Description**: Instantiate array: Test calculation of follow the regularized leader for beta value of 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(ftrl.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_follow_the_regularized_leader.py:67*

### test_scorn_solution

**Category**: instantiation  
**Description**: Instantiate SCORN: Test the calculation of SCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

scorn = SCORN(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning.py:35*

### test_scorn_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of SCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(scorn.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning.py:39*

### test_scorn_solution

**Category**: instantiation  
**Description**: Instantiate SCORN: Test the calculation of SCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
scorn = SCORN(window=2, rho=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning.py:35*

### test_scorn_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of SCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(scorn.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_symmetric_correlation_driven_nonparametric_learning.py:39*

### test_classification_fingerpint

**Category**: instantiation  
**Description**: Instantiate load_breast_cancer: Test model fingerprint values (linear, non-linear, pairwise) for classification model.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.X, self.y = load_boston(return_X_y=True)
self.X = pd.DataFrame(self.X[:100])
self.y = pd.Series(self.y[:100])
self.reg_rf = RandomForestRegressor(n_estimators=10, random_state=42)
self.reg_linear = LinearRegression(fit_intercept=True, normalize=False)
self.reg_rf.fit(self.X, self.y)
self.reg_linear.fit(self.X, self.y)
self.reg_fingerprint = RegressionModelFingerprint()

X, y = load_breast_cancer(return_X_y=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fingerpint.py:129*

### test_classification_fingerpint

**Category**: instantiation  
**Description**: Instantiate RandomForestClassifier: Test model fingerprint values (linear, non-linear, pairwise) for classification model.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Set the file path for the sample dollar bars data.\n        '
self.X, self.y = load_boston(return_X_y=True)
self.X = pd.DataFrame(self.X[:100])
self.y = pd.Series(self.y[:100])
self.reg_rf = RandomForestRegressor(n_estimators=10, random_state=42)
self.reg_linear = LinearRegression(fit_intercept=True, normalize=False)
self.reg_rf.fit(self.X, self.y)
self.reg_linear.fit(self.X, self.y)
self.reg_fingerprint = RegressionModelFingerprint()

clf = RandomForestClassifier(n_estimators=10, random_state=42)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fingerpint.py:131*

### test_classification_fingerpint

**Category**: instantiation  
**Description**: Instantiate load_breast_cancer: Test model fingerprint values (linear, non-linear, pairwise) for classification model.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
X, y = load_breast_cancer(return_X_y=True)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fingerpint.py:129*

### test_classification_fingerpint

**Category**: instantiation  
**Description**: Instantiate RandomForestClassifier: Test model fingerprint values (linear, non-linear, pairwise) for classification model.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
clf = RandomForestClassifier(n_estimators=10, random_state=42)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_fingerpint.py:131*

### test_fcorn_solution

**Category**: instantiation  
**Description**: Instantiate FCORN: Test the calculation of FCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn = FCORN(window=2, rho=0.5, lambd=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:35*

### test_fcorn_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of FCORN.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(fcorn.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:39*

### test_fcorn_window_error

**Category**: instantiation  
**Description**: Instantiate FCORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn1 = FCORN(window=2.5, rho=0.5, lambd=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:52*

### test_fcorn_window_error

**Category**: instantiation  
**Description**: Instantiate FCORN: Tests ValueError if window is not an integer or less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn2 = FCORN(window=0, rho=0.5, lambd=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:58*

### test_fcorn_rho_error

**Category**: instantiation  
**Description**: Instantiate FCORN: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn3 = FCORN(window=2, rho=-2, lambd=4)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:68*

### test_fcorn_rho_error

**Category**: instantiation  
**Description**: Instantiate FCORN: Tests ValueError if rho is less than -1 or more than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn4 = FCORN(window=2, rho=2, lambd=8)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:74*

### test_sigmoid

**Category**: instantiation  
**Description**: Instantiate FCORN: Tests Sigmoid Calculation.  
**Expected**: np.testing.assert_almost_equal(sig, 0.5)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn5 = FCORN(window=1, rho=0.5, lambd=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:86*

### test_sigmoid

**Category**: instantiation  
**Description**: Instantiate _sigmoid: Tests Sigmoid Calculation.  
**Expected**: np.testing.assert_almost_equal(sig, 0.5)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

sig = fcorn5._sigmoid(0.0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:87*

### test_fcorn_solution1

**Category**: instantiation  
**Description**: Instantiate FCORN: Test the calculation of FCORN for edge case that activation function is 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

fcorn6 = FCORN(window=1, rho=0.5, lambd=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:95*

### test_fcorn_solution1

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of FCORN for edge case that activation function is 0.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(fcorn6.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_functional_correlation_driven_nonparametric_learning.py:99*

### test_rmr_solution

**Category**: instantiation  
**Description**: Instantiate RMR: Test the calculation of RMR with the original method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:34*

### test_rmr_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of RMR with the original method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(rmr.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:38*

### test_rmr_epsilon_error

**Category**: instantiation  
**Description**: Instantiate RMR: Tests ValueError if epsilon is greater than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr1 = RMR(epsilon=0.5, n_iteration=10, window=3, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:51*

### test_rmr_n_iteration_error

**Category**: instantiation  
**Description**: Instantiate RMR: Tests ValueError if n_iteration is not an integer or less than 2.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr2 = RMR(epsilon=1.2, n_iteration=1.5, window=3, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:61*

### test_rmr_n_iteration_error

**Category**: instantiation  
**Description**: Instantiate RMR: Tests ValueError if n_iteration is not an integer or less than 2.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr3 = RMR(epsilon=1.2, n_iteration=1, window=3, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:67*

### test_rmr_window_error

**Category**: instantiation  
**Description**: Instantiate RMR: Tests ValueError if window is not an integer or less than 2.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr4 = RMR(epsilon=1.2, n_iteration=4, window=3.5, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:77*

### test_rmr_window_error

**Category**: instantiation  
**Description**: Instantiate RMR: Tests ValueError if window is not an integer or less than 2.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr5 = RMR(epsilon=1.2, n_iteration=4, window=1, tau=0.001)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:83*

### test_rmr_break_solution

**Category**: instantiation  
**Description**: Instantiate RMR: Test the calculation of RMR with the break case in _calc_median.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr6 = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.9)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:93*

### test_rmr_break_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of RMR with the break case in _calc_median.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(rmr6.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:97*

### test_rmr_transform_non_mu

**Category**: instantiation  
**Description**: Instantiate RMR: Tests edge case for _transform non_mu edge case.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

rmr7 = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.9)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_robust_median_reversion.py:112*

### test_best_stock_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests that the weights sum to 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(beststock.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:38*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate array: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

price_diff = np.array(self.data.iloc[-1] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:55*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate argmax: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

idx_price_diff = np.argmax(price_diff)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:57*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate argmax: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

idx_best_stock = np.argmax(beststock_weight)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:61*

### test_best_stock_solution

**Category**: instantiation  
**Description**: Instantiate array: Tests that the weights sum to 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
all_weights = np.array(beststock.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:38*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate array: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
price_diff = np.array(self.data.iloc[-1] / self.data.iloc[0])
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:55*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate argmax: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
idx_price_diff = np.argmax(price_diff)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:57*

### test_best_performing

**Category**: instantiation  
**Description**: Instantiate argmax: Tests that returning weights indicate the best performing asset.  
**Expected**: np.testing.assert_equal(idx_best_stock, idx_price_diff)  
**Confidence**: 0.80  
**Tags**: unittest  

```python
idx_best_stock = np.argmax(beststock_weight)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_best_stock.py:61*

### test_olmar_solution

**Category**: instantiation  
**Description**: Instantiate OLMAR: Test the calculation of online moving average reversion with the original reversion method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar = OLMAR(reversion_method=1, epsilon=1, window=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:34*

### test_olmar_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of online moving average reversion with the original reversion method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(olmar.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:38*

### test_olmar1_solution

**Category**: instantiation  
**Description**: Instantiate OLMAR: Test the calculation of online moving average reversion with the second reversion method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar1 = OLMAR(reversion_method=2, epsilon=10, alpha=0.5)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:51*

### test_olmar1_solution

**Category**: instantiation  
**Description**: Instantiate array: Test the calculation of online moving average reversion with the second reversion method.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

all_weights = np.array(olmar1.all_weights)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:55*

### test_olmar_epsilon_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests ValueError if epsilon is below than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar2 = OLMAR(reversion_method=1, epsilon=0, window=10)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:68*

### test_olmar_window_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests ValueError if reversion method is 1 and window is less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar3 = OLMAR(reversion_method=1, epsilon=2, window=0)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:78*

### test_olmar_alpha_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests ValueError if reversion method is 2 and alpha is greater than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar4 = OLMAR(reversion_method=2, epsilon=2, alpha=2)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:88*

### test_olmar_alpha1_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests ValueError if reversion method is 2 and alpha is less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar5 = OLMAR(reversion_method=2, epsilon=2, alpha=-1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:98*

### test_olmar_method_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests ValueError if reversion method is 2 and alpha is less than 1.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar6 = OLMAR(reversion_method=4, epsilon=2, alpha=-1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:108*

### test_olmar_edge_case_error

**Category**: instantiation  
**Description**: Instantiate OLMAR: Tests that lambd returns 0 if predicted change is mean change.  
**Confidence**: 0.80  
**Tags**: unittest  

```python
# Setup
'\n        Sets the file path for the tick data csv.\n        '
project_path = os.path.dirname(__file__)
data_path = project_path + '/test_data/stock_prices.csv'
self.data = pd.read_csv(data_path, parse_dates=True, index_col='Date').dropna(axis=1)

olmar7 = OLMAR(reversion_method=1, epsilon=2, window=1)
```

*Source: /Users/link/Documents/Skill_Seekers/src/mlfinlab/tests/test_online_moving_average_reversion.py:118*

