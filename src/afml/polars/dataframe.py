"""
Polars DataFrame Utilities for AFML.

This module provides utility functions and classes for working with
Polars DataFrames in the AFML context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import polars as pl
from polars import DataFrame, LazyFrame


class PolarsDataFrameUtils:
    """Utility class for Polars DataFrame operations specific to financial data."""

    @staticmethod
    def ensure_dataframe(
        data: Union[DataFrame, LazyFrame, Any],
        *,
        lazy: bool = False,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Ensure the input is a Polars DataFrame or LazyFrame.

        Args:
            data: Input data (DataFrame, LazyFrame, pandas DataFrame, or other)
            lazy: Whether to return LazyFrame (default False)

        Returns:
            Polars DataFrame or LazyFrame

        Raises:
            TypeError: If data cannot be converted to DataFrame
        """
        if isinstance(data, (DataFrame, LazyFrame)):
            if lazy and isinstance(data, DataFrame):
                return data.lazy()
            elif not lazy and isinstance(data, LazyFrame):
                return data.collect()
            return data

        if hasattr(data, "to_polars"):
            return data.to_polars(lazy=lazy)

        try:
            if isinstance(data, dict):
                result = DataFrame(data)
            else:
                result = DataFrame(data)

            if lazy:
                return result.lazy()
            return result
        except Exception as e:
            raise TypeError(f"Cannot convert {type(data)} to Polars DataFrame: {e}")

    @staticmethod
    def df_from_pandas(
        pandas_df: Any,
        *,
        lazy: bool = False,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Convert pandas DataFrame to Polars DataFrame.

        Args:
            pandas_df: pandas DataFrame to convert
            lazy: Whether to return LazyFrame (default False)

        Returns:
            Polars DataFrame or LazyFrame
        """
        result = pl.from_pandas(pandas_df)

        if lazy:
            return result.lazy()
        return result

    @staticmethod
    def create_from_csv(
        filepath: str,
        *,
        has_header: bool = True,
        schema_overrides: Optional[Dict[str, Any]] = None,
        try_parse_dates: bool = True,
        lazy: bool = False,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Read CSV file into Polars DataFrame.

        Args:
            filepath: Path to CSV file
            has_header: Whether the file has a header row
            schema_overrides: Dict of column names to types
            try_parse_dates: Whether to try parsing date columns
            lazy: Whether to return LazyFrame

        Returns:
            Polars DataFrame or LazyFrame
        """
        read_func = pl.scan_csv if lazy else pl.read_csv

        kwargs: Dict[str, Any] = {
            "has_header": has_header,
            "try_parse_dates": try_parse_dates,
        }

        if schema_overrides:
            kwargs["schema_overrides"] = schema_overrides

        return read_func(filepath, **kwargs)

    @staticmethod
    def create_from_parquet(
        filepath: str,
        *,
        lazy: bool = False,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Read Parquet file into Polars DataFrame.

        Args:
            filepath: Path to Parquet file
            lazy: Whether to return LazyFrame

        Returns:
            Polars DataFrame or LazyFrame
        """
        read_func = pl.scan_parquet if lazy else pl.read_parquet
        return read_func(filepath)

    @staticmethod
    def to_parquet(
        df: Union[DataFrame, LazyFrame],
        filepath: str,
        *,
        compression: str = "zstd",
    ) -> None:
        """
        Write DataFrame to Parquet file.

        Args:
            df: Polars DataFrame or LazyFrame
            filepath: Output file path
            compression: Compression codec ('zstd', 'snappy', 'gzip')
        """
        if isinstance(df, LazyFrame):
            df = df.collect()

        df.write_parquet(filepath, compression=compression)

    @staticmethod
    def select_columns(
        df: Union[DataFrame, LazyFrame],
        columns: List[str],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Select specific columns from DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            columns: List of column names to select

        Returns:
            DataFrame with selected columns
        """
        return df.select(columns)

    @staticmethod
    def drop_columns(
        df: Union[DataFrame, LazyFrame],
        columns: Union[str, List[str]],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Drop specified columns from DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            columns: Column name or list of column names to drop

        Returns:
            DataFrame without specified columns
        """
        if isinstance(columns, str):
            columns = [columns]
        return df.drop(columns)

    @staticmethod
    def rename_columns(
        df: Union[DataFrame, LazyFrame],
        mapping: Dict[str, str],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Rename columns using a mapping dict.

        Args:
            df: Polars DataFrame or LazyFrame
            mapping: Dict of old_name -> new_name

        Returns:
            DataFrame with renamed columns
        """
        return df.rename(mapping)

    @staticmethod
    def filter_by_condition(
        df: Union[DataFrame, LazyFrame],
        condition: pl.Expr,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Filter DataFrame by a Polars expression condition.

        Args:
            df: Polars DataFrame or LazyFrame
            condition: Polars expression for filtering

        Returns:
            Filtered DataFrame
        """
        return df.filter(condition)

    @staticmethod
    def add_column(
        df: Union[DataFrame, LazyFrame],
        name: str,
        value: Union[pl.Expr, pl.Series, List[Any]],
        *,
        overwrite: bool = True,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Add a new column to DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            name: Name of new column
            value: Expression, Series, or list for column values
            overwrite: Whether to overwrite existing column

        Returns:
            DataFrame with new column
        """
        if isinstance(value, (list, pl.Series)):
            value = pl.Series(values=value)

        expr = value if isinstance(value, pl.Expr) else pl.lit(value)

        return df.with_columns(expr.alias(name))

    @staticmethod
    def sort_by_column(
        df: Union[DataFrame, LazyFrame],
        column: str,
        *,
        descending: bool = False,
        nulls_last: bool = True,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Sort DataFrame by column.

        Args:
            df: Polars DataFrame or LazyFrame
            column: Column name to sort by
            descending: Sort in descending order
            nulls_last: Place null values last

        Returns:
            Sorted DataFrame
        """
        return df.sort(column, descending=descending, nulls_last=nulls_last)

    @staticmethod
    def group_by_aggregate(
        df: Union[DataFrame, LazyFrame],
        group_columns: Union[str, List[str]],
        aggregations: Dict[str, Union[str, pl.Expr]],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Group by columns and apply aggregations.

        Args:
            df: Polars DataFrame or LazyFrame
            group_columns: Column name(s) to group by
            aggregations: Dict of output_name -> aggregation (str or Expr)

        Returns:
            Aggregated DataFrame
        """
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        agg_exprs = []
        for out_name, agg in aggregations.items():
            if isinstance(agg, str):
                agg_exprs.append(
                    getattr(
                        pl.col(agg.split(".")[0]),
                        agg.split(".")[-1] if "." in agg else agg,
                    )()
                )
            else:
                agg_exprs.append(agg.alias(out_name))

        return df.group_by(group_columns).agg(agg_exprs)

    @staticmethod
    def get_column_names(df: Union[DataFrame, LazyFrame]) -> List[str]:
        """
        Get list of column names.

        Args:
            df: Polars DataFrame or LazyFrame

        Returns:
            List of column names
        """
        return df.columns

    @staticmethod
    def get_row_count(df: Union[DataFrame, LazyFrame]) -> int:
        """
        Get number of rows in DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame

        Returns:
            Number of rows
        """
        if isinstance(df, LazyFrame):
            return df.select(pl.count()).collect().item()
        return df.height

    @staticmethod
    def get_column_count(df: Union[DataFrame, LazyFrame]) -> int:
        """
        Get number of columns in DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame

        Returns:
            Number of columns
        """
        return df.width

    @staticmethod
    def describe_schema(df: Union[DataFrame, LazyFrame]) -> Dict[str, str]:
        """
        Get DataFrame schema as dict of column names to dtypes.

        Args:
            df: Polars DataFrame or LazyFrame

        Returns:
            Dict mapping column names to dtype strings
        """
        if isinstance(df, LazyFrame):
            schema = df.schema
        else:
            schema = df.schema

        return {col: str(dtype) for col, dtype in schema.items()}

    @staticmethod
    def head(
        df: Union[DataFrame, LazyFrame],
        n: int = 5,
    ) -> DataFrame:
        """
        Get first n rows as DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            n: Number of rows to get

        Returns:
            DataFrame with first n rows
        """
        if isinstance(df, LazyFrame):
            return df.head(n).collect()
        return df.head(n)

    @staticmethod
    def tail(
        df: Union[DataFrame, LazyFrame],
        n: int = 5,
    ) -> DataFrame:
        """
        Get last n rows as DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            n: Number of rows to get

        Returns:
            DataFrame with last n rows
        """
        if isinstance(df, LazyFrame):
            return df.tail(n).collect()
        return df.tail(n)

    @staticmethod
    def sample_rows(
        df: Union[DataFrame, LazyFrame],
        n: int,
        *,
        seed: Optional[int] = None,
    ) -> DataFrame:
        """
        Randomly sample n rows from DataFrame.

        Args:
            df: Polars DataFrame or LazyFrame
            n: Number of rows to sample
            seed: Random seed for reproducibility

        Returns:
            DataFrame with sampled rows
        """
        if isinstance(df, LazyFrame):
            return df.sample(n=n, seed=seed).collect()
        return df.sample(n=n, seed=seed)

    @staticmethod
    def is_empty(df: Union[DataFrame, LazyFrame]) -> bool:
        """
        Check if DataFrame is empty.

        Args:
            df: Polars DataFrame or LazyFrame

        Returns:
            True if DataFrame has no rows
        """
        if isinstance(df, LazyFrame):
            return df.select(pl.count()).collect().item() == 0
        return df.height == 0


def ensure_dataframe(
    data: Union[DataFrame, LazyFrame, Any],
    *,
    lazy: bool = False,
) -> Union[DataFrame, LazyFrame]:
    """
    Ensure the input is a Polars DataFrame or LazyFrame.

    This is a convenience function that wraps PolarsDataFrameUtils.ensure_dataframe.

    Args:
        data: Input data (DataFrame, LazyFrame, pandas DataFrame, or other)
        lazy: Whether to return LazyFrame (default False)

    Returns:
        Polars DataFrame or LazyFrame

    Raises:
        TypeError: If data cannot be converted to DataFrame
    """
    return PolarsDataFrameUtils.ensure_dataframe(data, lazy=lazy)


def df_from_pandas(
    pandas_df: Any,
    *,
    lazy: bool = False,
) -> Union[DataFrame, LazyFrame]:
    """
    Convert pandas DataFrame to Polars DataFrame.

    Args:
        pandas_df: pandas DataFrame to convert
        lazy: Whether to return LazyFrame (default False)

    Returns:
        Polars DataFrame or LazyFrame
    """
    return PolarsDataFrameUtils.df_from_pandas(pandas_df, lazy=lazy)
