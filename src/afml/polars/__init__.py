"""
AFML Polars - High-Performance Financial Machine Learning

This package provides Polars-based implementations of core quantitative
finance algorithms. Polars offers significant performance improvements
over pandas for large-scale financial time series data.

Key Benefits
-------------
- 3-10x memory efficiency improvement
- 2-5x faster execution with multi-threading
- Lazy evaluation for optimal query planning
- Zero-copy operations where possible

Notes
-----
All processors follow sklearn conventions with fit(), transform(), and
fit_transform() methods. Enable lazy evaluation by passing lazy=True
to constructor for large datasets.
"""

from .dataframe import (
    PolarsDataFrameUtils,
    ensure_dataframe,
    df_from_pandas,
)
from .series import (
    PolarsSeriesUtils,
    ensure_series,
    series_from_pandas,
)
from .convert import (
    to_polars,
    to_pandas,
)
from .dollar_bars import PolarsDollarBarsProcessor
from .labeling import PolarsTripleBarrierLabeler
from .features import PolarsFeatureEngineer
from .sample_weights import PolarsSampleWeightCalculator
from .cv import PolarsPurgedKFoldCV
from .meta_labeling import PolarsMetaLabelingPipeline
from .bet_sizing import PolarsBetSizer

__all__ = [
    # DataFrame utilities
    "PolarsDataFrameUtils",
    "ensure_dataframe",
    "df_from_pandas",
    # Series utilities
    "PolarsSeriesUtils",
    "ensure_series",
    "series_from_pandas",
    # Conversion utilities
    "to_polars",
    "to_pandas",
    # Processors
    "PolarsDollarBarsProcessor",
    "PolarsTripleBarrierLabeler",
    "PolarsFeatureEngineer",
    "PolarsSampleWeightCalculator",
    "PolarsPurgedKFoldCV",
    "PolarsMetaLabelingPipeline",
    "PolarsBetSizer",
]
