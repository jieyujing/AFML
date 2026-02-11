"""
AFML - Advances in Financial Machine Learning

This package provides object-oriented implementations of core quantitative
finance algorithms based on Marcos López de Prado's book.

Processors
----------
DollarBarsProcessor : Generates dollar bars from tick data
TripleBarrierLabeler : Applies triple barrier labeling method
FeatureEngineer : Generates Alpha158 and FFD features
SampleWeightCalculator : Calculates sample weights
BetSizer : Computes bet sizes from probabilities
MetaLabelingPipeline : Orchestrates meta-labeling workflow
PurgedKFoldCV : Cross-validation with purging and embargo

Polars Modules
-------------
All modules also available in afml.polars for high-performance
data processing using Polars instead of pandas.
"""

from .base import ProcessorMixin, ConfigurableProcessorMixin

from .dollar_bars import DollarBarsProcessor
from .labeling import TripleBarrierLabeler
from .features import FeatureEngineer
from .sample_weights import SampleWeightCalculator
from .bet_sizing import BetSizer
from .meta_labeling import MetaLabelingPipeline
from .cv import PurgedKFoldCV

# Polars modules (high-performance alternatives)
try:
    from .polars.dollar_bars import PolarsDollarBarsProcessor
    from .polars.labeling import PolarsTripleBarrierLabeler
    from .polars.features import PolarsFeatureEngineer
    from .polars.sample_weights import PolarsSampleWeightCalculator
    from .polars.bet_sizing import PolarsBetSizer
    from .polars.meta_labeling import PolarsMetaLabelingPipeline
    from .polars.cv import PolarsPurgedKFoldCV
    from .polars.convert import to_polars, to_pandas
    from .polars.dataframe import ensure_dataframe
    from .polars.series import ensure_series

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

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
]

if _POLARS_AVAILABLE:
    __all__.extend(
        [
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
    )
