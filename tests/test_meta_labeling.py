"""
Unit tests for MetaLabelingPipeline.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime

from afml import MetaLabelingPipeline


@pytest.fixture
def sample_dataset():
    """Create sample dataset for meta-labeling."""
    np.random.seed(42)
    n_samples = 500

    start_time = datetime(2023, 1, 1)
    times = [start_time for _ in range(n_samples)]

    features = pl.DataFrame(
        {
            "f1": np.random.randn(n_samples),
            "f2": np.random.randn(n_samples),
            "f3": np.random.randn(n_samples),
            "f4": np.random.randn(n_samples),
            "f5": np.random.randn(n_samples),
        }
    )

    labels = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.1, 0.6])
    returns = np.random.randn(n_samples) * 0.001
    sample_weights = np.random.uniform(0.5, 1.0, size=n_samples)

    df = features.with_columns(
        [
            pl.Series("label", labels),
            pl.Series("ret", returns),
            pl.Series("sample_weight", sample_weights),
        ]
    )

    return df, ["f1", "f2", "f3", "f4", "f5"]


class TestMetaLabelingPipeline:
    """Tests for MetaLabelingPipeline class."""

    def test_initialization_default(self):
        """Test default initialization."""
        pipeline = MetaLabelingPipeline()

        assert pipeline.n_splits == 5
        assert pipeline.embargo == 0.01
        assert pipeline.cv_ is None
        assert pipeline.primary_probs_ is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        pipeline = MetaLabelingPipeline(
            primary_model="random_forest",
            meta_model="logistic",
            n_splits=3,
            embargo=0.05,
        )

        assert pipeline.n_splits == 3
        assert pipeline.embargo == 0.05


class TestMetaLabelingPipelineEdgeCases:
    """Edge case tests for MetaLabelingPipeline."""

    def test_unfitted_pipeline(self, sample_dataset):
        """Test that unfitted pipeline raises error."""
        df, features = sample_dataset

        pipeline = MetaLabelingPipeline()

        try:
            pipeline.predict(df.select(features))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
