"""
Tests for PCA features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.pca_features import (
    compute_pca,
    compute_pca_with_standardization,
    transform_with_pca,
    rolling_pca,
    compute_feature_correlation_distance,
    PCATransform,
    RollingPCATransform,
    pca_features
)


class TestComputePCA:
    """Tests for core PCA computation."""

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.randn(n_samples, n_features)

        components, var_ratio, transformed, n_comp = compute_pca(X)

        assert components.shape[0] == n_comp
        assert components.shape[1] == n_features
        assert len(var_ratio) == n_comp
        assert transformed.shape == (n_samples, n_comp)

    def test_variance_explained_decreasing(self):
        """Variance explained should decrease with component index."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        _, var_ratio, _, _ = compute_pca(X)

        for i in range(len(var_ratio) - 1):
            assert var_ratio[i] >= var_ratio[i + 1]

    def test_cumulative_variance_reaches_threshold(self):
        """Auto-selected components should reach variance threshold."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        threshold = 0.95

        _, var_ratio, _, _ = compute_pca(X, variance_threshold=threshold)

        cumulative = np.sum(var_ratio)
        assert cumulative >= threshold - 0.01  # Allow small tolerance

    def test_fixed_n_components(self):
        """Test with fixed number of components."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        _, var_ratio, transformed, n_comp = compute_pca(X, n_components=3)

        assert n_comp == 3
        assert transformed.shape[1] == 3

    def test_components_are_orthogonal(self):
        """Principal components should be orthogonal."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        components, _, _, _ = compute_pca(X)

        # Check orthogonality: V @ V.T should be identity
        VVt = components @ components.T
        identity = np.eye(components.shape[0])
        assert_array_almost_equal(VVt, identity, decimal=6)

    def test_transformed_data_uncorrelated(self):
        """Transformed data should be uncorrelated."""
        np.random.seed(42)
        X = np.random.randn(200, 10)

        _, _, transformed, _ = compute_pca(X)

        # Compute correlation of transformed data
        corr = np.corrcoef(transformed.T)

        # Off-diagonal elements should be near zero
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(corr[i, j]) < 0.1  # Allow some tolerance


class TestComputePCAWithStandardization:
    """Tests for PCA with standardization."""

    def test_standardization_effect(self):
        """Standardization should change results."""
        np.random.seed(42)
        # Create features with very different scales
        X = np.random.randn(100, 3)
        X[:, 0] *= 1000  # First feature much larger
        X[:, 1] *= 0.001  # Second feature much smaller

        _, _, transformed_std, _ = compute_pca_with_standardization(
            X, standardize=True
        )
        _, _, transformed_no_std, _ = compute_pca_with_standardization(
            X, standardize=False
        )

        # Results should be different
        assert not np.allclose(transformed_std, transformed_no_std)

    def test_stats_returned(self):
        """Stats dictionary should be returned."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        _, _, _, stats = compute_pca_with_standardization(X)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'standardized' in stats
        assert len(stats['mean']) == 5
        assert len(stats['std']) == 5


class TestTransformWithPCA:
    """Tests for applying learned PCA to new data."""

    def test_same_data_same_result(self):
        """Applying to same data should give same result."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        components, _, transformed, stats = compute_pca_with_standardization(X)

        new_transformed = transform_with_pca(
            X, components, stats['mean'], stats['std'] if stats['standardized'] else None
        )

        assert_array_almost_equal(transformed, new_transformed, decimal=6)

    def test_new_data_shape(self):
        """New data should have correct output shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_new = np.random.randn(50, 5)

        components, var_ratio, _, n_comp = compute_pca(X_train)

        # Compute mean for new data
        mean = np.mean(X_train, axis=0)
        transformed = transform_with_pca(X_new, components, mean)

        assert transformed.shape == (50, n_comp)


class TestRollingPCA:
    """Tests for rolling PCA."""

    def test_output_shape(self):
        """Output should have correct shape."""
        np.random.seed(42)
        n_samples, n_features = 200, 5
        X = np.random.randn(n_samples, n_features)

        transformed, n_components = rolling_pca(X, window=50)

        # Max components is min(n_features, window)
        max_components = min(n_features, 50)
        assert transformed.shape == (n_samples, max_components)

    def test_first_window_minus_one_are_nan(self):
        """First window-1 rows should be NaN."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        window = 30

        transformed, _ = rolling_pca(X, window=window)

        assert np.all(np.isnan(transformed[:window - 1]))

    def test_n_components_tracking(self):
        """n_components array should track component counts."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        _, n_components = rolling_pca(X, window=30, variance_threshold=0.95)

        # First valid values should have some components
        valid_idx = np.where(n_components > 0)[0]
        assert len(valid_idx) > 0
        assert np.all(n_components[valid_idx] > 0)


class TestFeatureCorrelationDistance:
    """Tests for correlation-based distance matrix."""

    def test_diagonal_zero(self):
        """Diagonal should be zero."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        dist = compute_feature_correlation_distance(X)

        assert_array_almost_equal(np.diag(dist), np.zeros(5), decimal=6)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        dist = compute_feature_correlation_distance(X)

        assert_array_almost_equal(dist, dist.T, decimal=6)

    def test_non_negative(self):
        """All distances should be non-negative."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        dist = compute_feature_correlation_distance(X)

        assert np.all(dist >= 0)

    def test_perfect_correlation_zero_distance(self):
        """Perfectly correlated features should have zero distance."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[:, 1] = X[:, 0]  # Perfect correlation

        dist = compute_feature_correlation_distance(X)

        assert dist[0, 1] < 0.01


class TestPCATransform:
    """Tests for PCATransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        features = np.random.randn(n, 5)
        cols = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']
        return pd.DataFrame(features, columns=cols, index=dates)

    def test_transform_returns_tuple(self, sample_data):
        """Transform should return tuple of Series."""
        transform = PCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            variance_threshold=0.95
        )

        result = transform(sample_data, backend='pd')

        assert isinstance(result, tuple)
        assert all(isinstance(s, pd.Series) for s in result)

    def test_output_names(self, sample_data):
        """Test output name generation."""
        transform = PCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            variance_threshold=0.95,
            output_prefix='pca'
        )

        # Need to call transform first to fit
        _ = transform(sample_data, backend='pd')

        # Check output names
        output_names = transform.output_name
        assert all(name.startswith('pca_') for name in output_names)

    def test_explained_variance_ratio(self, sample_data):
        """Test getting explained variance ratio."""
        transform = PCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            variance_threshold=0.95
        )

        _ = transform(sample_data, backend='pd')
        var_ratio = transform.get_explained_variance_ratio()

        assert len(var_ratio) > 0
        assert np.all(var_ratio > 0)
        assert np.all(var_ratio < 1)

    def test_cumulative_variance(self, sample_data):
        """Test getting cumulative variance."""
        transform = PCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            variance_threshold=0.95
        )

        _ = transform(sample_data, backend='pd')
        cum_var = transform.get_cumulative_variance()

        assert len(cum_var) > 0
        assert cum_var[-1] >= 0.94  # Near threshold
        # Cumulative should be increasing
        assert np.all(np.diff(cum_var) >= -1e-10)

    def test_invalid_column_raises(self, sample_data):
        """Invalid column should raise ValueError."""
        transform = PCATransform(
            input_cols=['nonexistent'],
            variance_threshold=0.95
        )

        with pytest.raises(ValueError, match="not found"):
            transform(sample_data, backend='pd')

    def test_index_preserved(self, sample_data):
        """Output index should match input index."""
        transform = PCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            variance_threshold=0.95
        )

        result = transform(sample_data, backend='pd')

        for s in result:
            assert s.index.equals(sample_data.index)


class TestRollingPCATransform:
    """Tests for RollingPCATransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        features = np.random.randn(n, 5)
        cols = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']
        return pd.DataFrame(features, columns=cols, index=dates)

    def test_transform_returns_tuple(self, sample_data):
        """Transform should return tuple of Series."""
        transform = RollingPCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            window=50,
            variance_threshold=0.95
        )

        result = transform(sample_data, backend='pd')

        assert isinstance(result, tuple)

    def test_first_window_nan(self, sample_data):
        """First window-1 values should be NaN."""
        window = 50
        transform = RollingPCATransform(
            input_cols=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'],
            window=window,
            variance_threshold=0.95
        )

        result = transform(sample_data, backend='pd')

        for s in result:
            assert np.all(np.isnan(s[:window - 1]))


class TestPCAAFeaturesConvenience:
    """Tests for convenience function."""

    def test_returns_dataframe(self):
        """Function should return DataFrame."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        df = pd.DataFrame(
            np.random.randn(200, 5),
            columns=['a', 'b', 'c', 'd', 'e'],
            index=dates
        )

        result = pca_features(df, ['a', 'b', 'c', 'd', 'e'])

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 200

    def test_columns_start_with_pca(self):
        """Column names should start with 'pca_'."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        df = pd.DataFrame(
            np.random.randn(200, 5),
            columns=['a', 'b', 'c', 'd', 'e'],
            index=dates
        )

        result = pca_features(df, ['a', 'b', 'c', 'd', 'e'])

        assert all(col.startswith('pca_') for col in result.columns)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Test with single feature (should return 1 component)."""
        np.random.seed(42)
        X = np.random.randn(100, 1)

        _, var_ratio, transformed, n_comp = compute_pca(X)

        assert n_comp == 1
        assert transformed.shape == (100, 1)

    def test_more_features_than_samples(self):
        """Test with more features than samples."""
        np.random.seed(42)
        X = np.random.randn(50, 100)  # 50 samples, 100 features

        _, _, transformed, n_comp = compute_pca(X)

        # Can only have min(n_samples, n_features) components
        assert n_comp <= 50

    def test_constant_feature(self):
        """Test with constant feature (should handle gracefully)."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 0] = 1.0  # Constant feature

        # Should not raise
        _, _, transformed, _ = compute_pca(X)
        assert transformed.shape[0] == 100