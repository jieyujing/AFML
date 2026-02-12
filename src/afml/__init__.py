"""
AFML - Advances in Financial Machine Learning (Polars Optimized)

This package provides high-performance, object-oriented implementations of core 
quantitative finance algorithms based on Marcos López de Prado's book, 
optimized for Polars.

Processors
----------
DollarBarsProcessor : Generates dollar bars from tick data (Polars)
TripleBarrierLabeler : Applies triple barrier labeling method (Polars)
FeatureEngineer : Generates Alpha158 and FFD features (Polars)
SampleWeightCalculator : Calculates sample weights (Polars)
BetSizer : Computes bet sizes from probabilities (Polars)
MetaLabelingPipeline : Orchestrates meta-labeling workflow (Polars)
PurgedKFoldCV : Cross-validation with purging and embargo (Polars)

Utilities
---------
to_polars : Convert data to Polars format
to_pandas : Convert data to pandas format
ensure_dataframe : Ensure data is a Polars DataFrame
ensure_series : Ensure data is a Polars Series
"""

from .base import ProcessorMixin, ConfigurableProcessorMixin

# Polars-native implementations (Standard names now point to Polars)
from .dollar_bars import DollarBarsProcessor
from .labeling import TripleBarrierLabeler
from .features import FeatureEngineer
from .sample_weights import SampleWeightCalculator
from .bet_sizing import BetSizer
from .meta_labeling import MetaLabelingPipeline
from .cv import PurgedKFoldCV

# Utilities
from .convert import to_polars, to_pandas
from .dataframe import ensure_dataframe
from .series import ensure_series

# Legacy aliases for backward compatibility (optional, but keep for now if needed by pipeline)
PolarsDollarBarsProcessor = DollarBarsProcessor
PolarsTripleBarrierLabeler = TripleBarrierLabeler
PolarsFeatureEngineer = FeatureEngineer
PolarsSampleWeightCalculator = SampleWeightCalculator
PolarsBetSizer = BetSizer
PolarsMetaLabelingPipeline = MetaLabelingPipeline
PolarsPurgedKFoldCV = PurgedKFoldCV

__all__ = [
    "ProcessorMixin",
    "ConfigurableProcessorMixin",
    "DollarBarsProcessor",
    "TripleBarrierLabeler",
    "FeatureEngineer",
    "SampleWeightCalculator",
    "BetSizer",
    "MetaLabelingPipeline",
    "PurgedKFoldCV",
    "PolarsDollarBarsProcessor",
    "PolarsTripleBarrierLabeler",
    "PolarsFeatureEngineer",
    "PolarsSampleWeightCalculator",
    "PolarsBetSizer",
    "PolarsMetaLabelingPipeline",
    "PolarsPurgedKFoldCV",
    "to_polars",
    "to_pandas",
    "ensure_dataframe",
    "ensure_series",
]
