"""
Unit tests for MetaLabelingPipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from afml import MetaLabelingPipeline


@pytest.fixture
def sample_dataset():
    """Create sample dataset for meta-labeling."""
    np.random.seed(42)
    n_samples = 500  # Increased for better CV splits

    start_time = datetime(2023, 1, 1)
    times = pd.date_range(start=start_time, periods=n_samples, freq="h")

    # Create features
    features = pd.DataFrame(
        {
            "f1": np.random.randn(n_samples),
            "f2": np.random.randn(n_samples),
            "f3": np.random.randn(n_samples),
            "f4": np.random.randn(n_samples),
            "f5": np.random.randn(n_samples),
        },
        index=times,
    )

    # Create labels (-1, 0, 1)
    labels = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.1, 0.6])

    # Create returns
    returns = np.random.randn(n_samples) * 0.001

    # Create sample weights
    sample_weights = np.random.uniform(0.5, 1.0, size=n_samples)

    # Create t1 (vertical barrier times) - more spread out to reduce purging
    t1_times = pd.date_range(start=start_time, periods=n_samples, freq="24h")

    # Combine into dataset
    df = features.copy()
    df["label"] = labels
    df["ret"] = returns
    df["sample_weight"] = sample_weights
    df["t1"] = t1_times

    return df, list(features.columns)


class TestMetaLabelingPipeline:
    """Tests for MetaLabelingPipeline class."""

    def test_initialization_default(self):
        """Test default initialization."""
        pipeline = MetaLabelingPipeline()

        assert pipeline.n_splits == 5
        assert pipeline.embargo == 0.01
        assert pipeline.cv_ is None
        assert pipeline.primary_probs_ is None
        assert pipeline.meta_clf_ is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        primary = RandomForestClassifier(n_estimators=50)
        secondary = RandomForestClassifier(n_estimators=50, max_depth=3)

        pipeline = MetaLabelingPipeline(
            primary_model=primary, secondary_model=secondary, n_splits=3, embargo=0.05
        )

        assert pipeline.n_splits == 3
        assert pipeline.embargo == 0.05

    def test_fit(self, sample_dataset):
        """Test fit method."""
        df, features = sample_dataset
        y = df["label"]

        pipeline = MetaLabelingPipeline(n_splits=3)
        result = pipeline.fit(df, features, y)

        assert result is pipeline
        assert pipeline.cv_ is not None
        assert pipeline.meta_clf_ is not None


class TestMetaLabelingPipelineEdgeCases:
    """Edge case tests for MetaLabelingPipeline."""

    def test_no_sample_weights(self, sample_dataset):
        """Test without sample weights."""
        df, features = sample_dataset
        y = df["label"]

        # Drop sample weights
        df_no_weights = df.drop(columns=["sample_weight"])

        pipeline = MetaLabelingPipeline(n_splits=3)
        pipeline.fit(df_no_weights, features, y)

        # Just verify fit works without sample weights
        assert pipeline.meta_clf_ is not None

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        n = 100  # Increased for better CV splits

        start_time = datetime(2023, 1, 1)
        times = pd.date_range(start=start_time, periods=n, freq="h")

        features = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
            },
            index=times,
        )

        labels = np.random.choice([-1, 0, 1], size=n)
        returns = np.random.randn(n) * 0.001

        # More spread out t1 to reduce purging
        t1_times = pd.date_range(start=start_time, periods=n, freq="24h")

        df = features.copy()
        df["label"] = labels
        df["ret"] = returns
        df["t1"] = t1_times

        pipeline = MetaLabelingPipeline(n_splits=2)
        pipeline.fit(df, ["f1", "f2"], df["label"])

        # Just verify fit works
        assert pipeline.meta_clf_ is not None

    def test_unfitted_pipeline(self, sample_dataset):
        """Test that unfitted pipeline raises error."""
        df, features = sample_dataset

        pipeline = MetaLabelingPipeline()

        # predict on unfitted pipeline should raise
        try:
            pipeline.predict(df[features])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

    def test_binary_labels_only(self, sample_dataset):
        """Test with only binary labels (-1, 1)."""
        df, features = sample_dataset
        # Convert to binary
        y = df["label"].apply(lambda x: 1 if x > 0 else 0)

        pipeline = MetaLabelingPipeline(n_splits=3)
        pipeline.fit(df, features, y)

        # Verify fit works
        assert pipeline.meta_clf_ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
