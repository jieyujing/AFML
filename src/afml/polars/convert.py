"""
Data Conversion Utilities for AFML Polars.

This module provides conversion functions between Polars and pandas DataFrames/Series,
as well as utilities for handling mixed data types.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar, Union

import pandas as pd
import polars as pl
from polars import DataFrame, LazyFrame, Series

# Type variables for generic conversion
PolarsType = TypeVar("PolarsType", DataFrame, LazyFrame, Series)
PandasType = TypeVar("PandasType", pd.DataFrame, pd.Series)


def to_polars(
    data: Any,
    *,
    lazy: bool = False,
    schema: Optional[Dict[str, Any]] = None,
) -> Union[DataFrame, LazyFrame, Series]:
    """
    Convert various data formats to Polars DataFrame or Series.

    Args:
        data: Input data (pandas DataFrame/Series, dict, list, etc.)
        lazy: Whether to return LazyFrame (default False)
        schema: Optional schema overrides for creation

    Returns:
        Polars DataFrame, LazyFrame, or Series

    Raises:
        TypeError: If data cannot be converted
    """
    if isinstance(data, (DataFrame, LazyFrame, Series)):
        if lazy and isinstance(data, DataFrame):
            return data.lazy()
        elif not lazy and isinstance(data, LazyFrame):
            return data.collect()
        return data

    if isinstance(data, pd.DataFrame):
        result = pl.from_pandas(data)
        if lazy:
            return result.lazy()
        return result

    if isinstance(data, pd.Series):
        return pl.from_pandas(data)

    try:
        if isinstance(data, dict):
            result = DataFrame(data, schema=schema)
        elif isinstance(data, list):
            if schema:
                result = DataFrame(data, schema=schema)
            else:
                result = DataFrame(data)
        elif hasattr(data, "__dataframe__"):
            import pyarrow as pa

            table = pa.table(data)
            result = pl.from_arrow(table)
        else:
            result = DataFrame(data, schema=schema)

        if lazy:
            return result.lazy()
        return result
    except Exception as e:
        raise TypeError(f"Cannot convert {type(data)} to Polars: {e}") from e


def to_pandas(
    data: Union[DataFrame, LazyFrame, Series, Any],
) -> Union[pd.DataFrame, pd.Series]:
    """
    Convert Polars DataFrame/Series to pandas.

    Args:
        data: Polars DataFrame, LazyFrame, Series, or other

    Returns:
        pandas DataFrame or Series

    Raises:
        TypeError: If data cannot be converted
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data

    if isinstance(data, LazyFrame):
        data = data.collect()

    if isinstance(data, DataFrame):
        return data.to_pandas()

    if isinstance(data, Series):
        return data.to_pandas()

    raise TypeError(f"Cannot convert {type(data)} to pandas")


def convert_columns(
    df: Union[DataFrame, LazyFrame],
    column_mapping: Dict[str, str],
    *,
    lazy: bool = False,
) -> Union[DataFrame, LazyFrame]:
    """
    Convert columns between pandas and Polars types.

    Args:
        df: Polars DataFrame or LazyFrame
        column_mapping: Dict of column names to types
        lazy: Whether to use lazy evaluation

    Returns:
        DataFrame with converted columns
    """
    result = df

    for col, dtype in column_mapping.items():
        if dtype == "categorical":
            result = result.with_columns(pl.col(col).cast(pl.Categorical))
        elif dtype == "datetime":
            result = result.with_columns(pl.col(col).str.to_datetime())
        elif dtype == "float":
            result = result.with_columns(pl.col(col).cast(pl.Float64))
        elif dtype == "int":
            result = result.with_columns(pl.col(col).cast(pl.Int64))
        elif dtype == "string":
            result = result.with_columns(pl.col(col).cast(pl.Utf8))

    if lazy:
        return result.lazy()
    return result


def optimize_schema(
    df: Union[DataFrame, LazyFrame],
    *,
    for_gpu: bool = False,
    lazy: bool = False,
) -> Union[DataFrame, LazyFrame]:
    """
    Optimize DataFrame schema for memory efficiency.

    Args:
        df: Polars DataFrame or LazyFrame
        for_gpu: Whether to optimize for GPU usage
        lazy: Whether to use lazy evaluation

    Returns:
        Optimized DataFrame
    """
    result = df

    for col in df.columns:
        col_dtype = df.schema[col]

        if col_dtype == pl.Float64:
            result = result.with_columns(pl.col(col).cast(pl.Float32))
        elif col_dtype == pl.Int64:
            result = result.with_columns(pl.col(col).cast(pl.Int32))

    if lazy:
        return result.lazy()
    return result


def check_schema_compatibility(
    polars_df: DataFrame,
    pandas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check schema compatibility between Polars and pandas DataFrames.

    Args:
        polars_df: Polars DataFrame
        pandas_df: pandas DataFrame

    Returns:
        Dict with compatibility status and details
    """
    result = {
        "compatible": True,
        "column_differences": [],
        "dtype_differences": [],
        "row_count_match": True,
    }

    polars_cols = set(polars_df.columns)
    pandas_cols = set(pandas_df.columns)

    if polars_cols != pandas_cols:
        result["compatible"] = False
        result["column_differences"] = list(
            polars_cols.symmetric_difference(pandas_cols)
        )

    for col in polars_cols.intersection(pandas_cols):
        polars_dtype = polars_df.schema[col]
        pandas_dtype = pandas_df[col].dtype

        if str(polars_dtype) != str(pandas_dtype):
            result["dtype_differences"].append(
                {
                    "column": col,
                    "polars": str(polars_dtype),
                    "pandas": str(pandas_dtype),
                }
            )

    if len(polars_df) != len(pandas_df):
        result["compatible"] = False
        result["row_count_match"] = False

    return result
