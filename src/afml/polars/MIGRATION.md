# Pandas to Polars Migration Guide

This guide provides a comprehensive reference for migrating from pandas to Polars in the AFML codebase.

## Why Migrate to Polars?

| Aspect | Pandas | Polars |
|--------|--------|--------|
| Memory Usage | High (3-10x more) | Low (zero-copy when possible) |
| Execution Speed | Single-threaded | Multi-threaded (2-5x faster) |
| Lazy Evaluation | Not supported | Native support |
| API Consistency | Inconsistent | Uniform |

## Key Differences

### 1. Creation

**Pandas:**
```python
import pandas as pd

# From dict
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# From CSV
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
```

**Polars:**
```python
import polars as pl

# From dict
df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# From CSV
df = pl.read_csv('data.csv', has_header=True)
# Note: Polars handles dates automatically
```

### 2. Column Selection

**Pandas:**
```python
# Single column (returns Series)
df['close']

# Multiple columns
df[['close', 'volume']]

# With conditions
df[df['close'] > 100]
```

**Polars:**
```python
# Single column (returns Series)
df['close']

# Multiple columns
df.select(['close', 'volume'])

# With conditions (returns new DataFrame)
df.filter(pl.col('close') > 100)
```

### 3. Column Operations

**Pandas:**
```python
# Add column
df['new_col'] = df['close'] * 2

# Modify column
df['close'] = df['close'].pct_change()

# Drop column
df = df.drop('volume', axis=1)
```

**Polars:**
```python
# Add column
df = df.with_columns((pl.col('close') * 2).alias('new_col'))

# Modify column
df = df.with_columns(pl.col('close').pct_change())

# Drop column
df = df.drop('volume')
```

### 4. GroupBy Operations

**Pandas:**
```python
df.groupby('symbol').agg({
    'close': ['mean', 'std'],
    'volume': 'sum'
})
```

**Polars:**
```python
df.group_by('symbol').agg([
    pl.col('close').mean().alias('close_mean'),
    pl.col('close').std().alias('close_std'),
    pl.col('volume').sum().alias('volume_sum')
])
```

### 5. Rolling Windows

**Pandas:**
```python
df['close'].rolling(window=20).mean()
df['close'].ewm(span=20).mean()
```

**Polars:**
```python
df.with_columns(
    pl.col('close').rolling_mean(window_size=20).alias('close_rolling_mean'),
    pl.col('close').ewm_mean(span=20).alias('close_ewm_mean')
)
```

### 6. Date/Time Operations

**Pandas:**
```python
df.index = pd.to_datetime(df.index)
df['year'] = df.index.year
df['dayofweek'] = df.index.dayofweek
```

**Polars:**
```python
# Polars handles datetime parsing automatically in most cases
df = df.with_columns(
    pl.col('date').dt.year().alias('year'),
    pl.col('date').dt.weekday().alias('dayofweek')
)
```

## Common Mappings

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Shape | `df.shape` | `df.shape` |
| Columns | `df.columns` | `df.columns` |
| dtypes | `df.dtypes` | `df.dtypes` |
| Head | `df.head()` | `df.head()` |
| Tail | `df.tail()` | `df.tail()` |
| Describe | `df.describe()` | `df.describe()` |
| Info | `df.info()` | `df.describe()` |
| Drop NA | `df.dropna()` | `df.drop_nulls()` |
| Fill NA | `df.fillna(0)` | `df.fill_null(0)` |
| Sort | `df.sort_values('close')` | `df.sort('close')` |
| Unique | `df['col'].unique()` | `df['col'].unique()` |
| Value Counts | `df['col'].value_counts()` | `df['col'].value_counts()` |
| Merge | `pd.merge(df1, df2, on='key')` | `df1.join(df2, on='key')` |
| Concat | `pd.concat([df1, df2])` | `pl.concat([df1, df2])` |
| Pivot | `df.pivot_table()` | `df.pivot()` |
| Melt | `df.melt()` | `df.unpivot()` |

## AFML-Specific Mappings

### Dollar Bars Processing

**Pandas (original):**
```python
def get_dollar_bars(
    df: pd.DataFrame,
    dollar_threshold: float = 1e8,
) -> pd.DataFrame:
    cum_dollar = (df['close'] * df['volume']).cumsum()
    bars = cum_dollar // dollar_threshold
    return df[bars.diff() > 0]
```

**Polars:**
```python
def get_dollar_bars(
    df: pl.DataFrame,
    dollar_threshold: float = 1e8,
    *,
    lazy: bool = False,
) -> pl.DataFrame:
    dollar_volume = (pl.col('close') * pl.col('volume')).cum_sum()
    bar_indices = (dollar_volume / dollar_threshold).floor()
    
    query = df.with_columns(
        bar_indices.alias('bar')
    ).filter(
        pl.col('bar').diff() > 0
    ).drop('bar')
    
    return query.collect() if lazy else query
```

### Triple Barrier Labeling

**Pandas (original):**
```python
def apply_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: List[float],
    vertical_barrier: int,
) -> pd.DataFrame:
    # Implementation details...
    pass
```

**Polars:**
```python
def apply_labels(
    close: pl.Series,
    events: pl.DataFrame,
    pt_sl: List[float],
    vertical_barrier: int,
    *,
    lazy: bool = False,
) -> pl.DataFrame:
    # Use Polars expressions for vectorized operations
    pass
```

## Lazy Evaluation

Polars supports lazy evaluation for optimal performance:

```python
# Eager mode (immediate execution)
df = pl.read_csv('data.csv')

# Lazy mode (deferred execution)
df = pl.scan_csv('data.csv')  # Returns LazyFrame

# Add transformations
result = (
    df.lazy()
    .filter(pl.col('close') > 0)
    .with_columns(pl.col('close').pct_change().alias('returns'))
    .group_by('symbol')
    .agg(pl.col('returns').mean().alias('mean_return'))
    .collect()  # Execute all operations
)
```

### When to Use Lazy Mode

- Processing large datasets (>100MB)
- Multiple chained operations
- When you don't need immediate results
- For optimization across entire pipeline

### When to Use Eager Mode

- Small datasets
- Interactive exploration
- When you need immediate results
- Single operation on data

## Type Hints

Polars uses its own type system:

```python
from polars import DataFrame, Series, LazyFrame

# DataFrame type hint
def process_df(df: DataFrame) -> DataFrame:
    return df.filter(pl.col('close') > 0)

# LazyFrame type hint
def process_lazy(lf: LazyFrame) -> LazyFrame:
    return lf.filter(pl.col('close') > 0)

# Series type hint
def compute_vol(series: Series, window: int = 20) -> Series:
    return series.rolling_std(window_size=window)
```

## Error Handling

Common errors and solutions:

### 1. Type Mismatch
```
Error: TypeError: cannot add column of length 10 to DataFrame of length 100
Fix: Ensure all columns have the same length or use .with_columns() carefully
```

### 2. Expression Not Found
```
Error: ColumnNotFoundError: 'close'
Fix: Use pl.col('close') instead of 'close' in expressions
```

### 3. Schema Mismatch
```
Error: SchemaError: expected type Float64 but found String
Fix: Convert types with .cast(pl.Float64)
```

## Performance Tips

1. **Use LazyFrame** for large datasets
2. **Avoid Python loops** - use vectorized Polars expressions
3. **Reorder columns** - frequently accessed columns first
4. **Use .collect()** at the end of lazy operations
5. **Cache intermediate results** for reuse

## Converting Between Pandas and Polars

```python
import pandas as pd
import polars as pl

# Pandas to Polars
pandas_df = pd.DataFrame({'a': [1, 2, 3]})
polars_df = pl.from_pandas(pandas_df)

# Polars to Pandas
polars_df = pl.DataFrame({'a': [1, 2, 3]})
pandas_df = polars_df.to_pandas()

# In-place conversion
df_polars = df.to_polars()
df_pandas = df.to_pandas()
```

## Debugging Tips

### Print Query Plan
```python
# For LazyFrame
lf = df.lazy()
print(lf.explain())
```

### Check Data Types
```python
print(df.dtypes)
print(df.schema)
```

### Sample Data
```python
df.sample(n=100)  # Random sample
df.head(10)       # First 10 rows
```

## Migration Checklist

- [ ] Add `polars>=1.0.0` to dependencies
- [ ] Import polars instead of pandas where needed
- [ ] Replace `pd.DataFrame` with `pl.DataFrame`
- [ ] Replace `pd.Series` with `pl.Series`
- [ ] Update column selection syntax
- [ ] Convert aggregation operations
- [ ] Update date/time operations
- [ ] Test lazy mode for large datasets
- [ ] Verify output matches pandas version
- [ ] Run performance benchmarks

## Known Differences

1. **Index handling**: Polars doesn't have a default index like pandas
2. **In-place operations**: Polars returns new DataFrames (immutable)
3. **Boolean indexing**: Use `.filter()` instead of boolean masks directly
4. **String operations**: Methods are under `.str` namespace
5. **Date operations**: Methods are under `.dt` namespace

## Resources

- [Polars Documentation](https://docs.pola.rs/)
- [Polars API Reference](https://docs.pola.rs/api/python/stable/reference/)
- [Polars Cheat Sheet](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- [Pandas to Polars Guide](https://pandas.pydata.org/docs/user_guide/cookbook.html)
