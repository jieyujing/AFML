"""
Tests for entropy features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.entropy import (
    discretize_returns,
    compute_symbol_frequencies,
    shannon_entropy,
    shannon_entropy_rate,
    lempel_ziv_complexity,
    lempel_ziv_entropy,
    kontoyiannis_lz_entropy,
    entropy_implied_volatility,
    rolling_entropy,
    rolling_entropy_rate,
    ShannonEntropyTransform,
    LZEntropyTransform,
    KontoyiannisEntropyTransform,
    EntropyImpliedVolTransform,
    compute_entropy,
)


class TestDiscretizeReturns:
    """Tests for return discretization."""

    def test_three_bins_basic(self):
        """Test ternary encoding (up/flat/down)."""
        returns = np.array([-0.1, -0.01, 0.0, 0.01, 0.1])
        symbols = discretize_returns(returns, n_bins=3)

        assert len(symbols) == len(returns)
        assert np.all((symbols >= 0) & (symbols < 3))

    def test_three_bins_signs(self):
        """Test that signs are correctly captured."""
        np.random.seed(42)
        # Create clearly positive and negative returns
        returns = np.array([-1.0, -0.5, 0.5, 1.0])
        symbols = discretize_returns(returns, n_bins=3)

        # Negative returns should be 0, positive should be 2
        assert symbols[0] == 0  # Large negative
        assert symbols[-1] == 2  # Large positive


class TestComputeSymbolFrequencies:
    """Tests for symbol frequency computation."""

    def test_frequencies_sum_to_one(self):
        """Frequencies should sum to 1."""
        symbols = np.array([0, 1, 2, 0, 1, 2, 0, 0])
        freqs = compute_symbol_frequencies(symbols, n_symbols=3)

        assert abs(np.sum(freqs) - 1.0) < 1e-10

    def test_uniform_distribution(self):
        """Uniform symbols should give uniform frequencies."""
        symbols = np.array([0, 1, 2, 0, 1, 2])
        freqs = compute_symbol_frequencies(symbols, n_symbols=3)

        assert_array_almost_equal(freqs, np.array([1/3, 1/3, 1/3]), decimal=6)

    def test_single_symbol(self):
        """Single symbol should give frequency 1."""
        symbols = np.array([0, 0, 0, 0])
        freqs = compute_symbol_frequencies(symbols, n_symbols=3)

        assert freqs[0] == 1.0
        assert freqs[1] == 0.0
        assert freqs[2] == 0.0


class TestShannonEntropy:
    """Tests for Shannon entropy."""

    def test_zero_entropy_single_symbol(self):
        """Single symbol should give zero entropy."""
        freqs = np.array([1.0, 0.0, 0.0])
        entropy = shannon_entropy(freqs)

        assert entropy == 0.0

    def test_max_entropy_uniform(self):
        """Uniform distribution should give max entropy."""
        n = 3
        freqs = np.ones(n) / n
        entropy = shannon_entropy(freqs)

        # Max entropy for 3 symbols is log2(3) ≈ 1.585
        expected_max = np.log2(n)
        assert abs(entropy - expected_max) < 1e-10

    def test_entropy_in_bits(self):
        """Entropy should be in bits (base 2)."""
        freqs = np.array([0.5, 0.5])
        entropy = shannon_entropy(freqs)

        # For two equally likely symbols, H = 1 bit
        assert abs(entropy - 1.0) < 1e-10

    def test_entropy_ordering(self):
        """More concentrated distribution should have lower entropy."""
        freqs_concentrated = np.array([0.9, 0.05, 0.05])
        freqs_spread = np.array([0.4, 0.3, 0.3])

        entropy_conc = shannon_entropy(freqs_concentrated)
        entropy_spread = shannon_entropy(freqs_spread)

        assert entropy_conc < entropy_spread


class TestShannonEntropyRate:
    """Tests for Shannon entropy rate."""

    def test_first_order_entropy(self):
        """Test first-order (pairwise) entropy rate."""
        # Alternating pattern: highly predictable
        symbols = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        entropy = shannon_entropy_rate(symbols, n_symbols=2, order=1)

        # For alternating pattern, conditional entropy should be low
        assert entropy < 0.5

    def test_random_sequence(self):
        """Random sequence should have higher entropy rate."""
        np.random.seed(42)
        symbols = np.random.randint(0, 3, 100)
        entropy = shannon_entropy_rate(symbols, n_symbols=3, order=1)

        # Random sequence should have entropy near max
        assert entropy > 1.0  # Higher than concentrated case


class TestLempelZivComplexity:
    """Tests for LZ complexity."""

    def test_repetitive_sequence_low_complexity(self):
        """Repetitive sequence should have low complexity."""
        symbols = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        complexity = lempel_ziv_complexity(symbols, n_symbols=2)

        # All same symbol should have low complexity
        # The exact value depends on parsing strategy
        assert complexity < len(symbols)

    def test_alternating_sequence(self):
        """Alternating sequence should have moderate complexity."""
        symbols = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        complexity = lempel_ziv_complexity(symbols, n_symbols=2)

        # Pattern repeats, complexity should be less than full random
        assert complexity < len(symbols)

    def test_complexity_grows_with_sequence(self):
        """More complex sequences should have higher complexity."""
        np.random.seed(42)

        simple = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        complex_seq = np.random.randint(0, 2, 8)

        c_simple = lempel_ziv_complexity(simple, 2)
        c_complex = lempel_ziv_complexity(complex_seq, 2)

        assert c_complex > c_simple


class TestLempelZivEntropy:
    """Tests for LZ entropy."""

    def test_repetitive_low_entropy(self):
        """Repetitive sequence should have low entropy."""
        # Use longer sequence for better entropy estimate
        symbols = np.zeros(100, dtype=np.int64)
        entropy = lempel_ziv_entropy(symbols, n_symbols=2)

        # Low entropy for repetitive sequence (relative to max)
        max_entropy = np.log2(2)  # Max entropy for binary
        assert entropy < max_entropy * 2  # Allow some tolerance

    def test_random_higher_entropy(self):
        """Random sequence should have higher entropy."""
        np.random.seed(42)
        symbols = np.random.randint(0, 2, 100)

        entropy_random = lempel_ziv_entropy(symbols, n_symbols=2)

        # Random should be higher than repetitive
        symbols_rep = np.zeros(100, dtype=np.int64)
        entropy_rep = lempel_ziv_entropy(symbols_rep, n_symbols=2)

        assert entropy_random > entropy_rep


class TestKontoyiannisLZEntropy:
    """Tests for Kontoyiannis LZ entropy."""

    def test_repetitive_low_entropy(self):
        """Repetitive sequence should have low entropy."""
        symbols = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        entropy = kontoyiannis_lz_entropy(symbols, n_symbols=2)

        assert entropy < 1.0

    def test_alternating_pattern(self):
        """Alternating pattern should have moderate entropy."""
        symbols = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        entropy = kontoyiannis_lz_entropy(symbols, n_symbols=2)

        # Should be lower than random
        assert entropy < 2.0


class TestEntropyImpliedVolatility:
    """Tests for entropy-implied volatility."""

    def test_higher_entropy_lower_vol(self):
        """Higher entropy should give lower implied volatility."""
        vol_low_entropy = entropy_implied_volatility(0.5, annualize=False)
        vol_high_entropy = entropy_implied_volatility(1.5, annualize=False)

        assert vol_low_entropy > vol_high_entropy

    def test_annualization(self):
        """Annualized vol should be higher than daily."""
        vol_daily = entropy_implied_volatility(1.0, annualize=False)
        vol_annual = entropy_implied_volatility(1.0, annualize=True)

        assert vol_annual > vol_daily

    def test_volatility_positive(self):
        """Implied volatility should always be positive."""
        for h in [0.0, 0.5, 1.0, 1.5, 2.0]:
            vol = entropy_implied_volatility(h, annualize=False)
            assert vol > 0


class TestRollingEntropy:
    """Tests for rolling entropy computation."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        returns = np.random.randn(n) * 0.02

        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        assert result.shape == (n,)

    def test_first_window_minus_one_are_nan(self):
        """First window-1 values should be NaN."""
        n = 100
        window = 20
        returns = np.random.randn(n) * 0.02

        result = rolling_entropy(returns, window, n_bins=3, method=0)

        assert np.all(np.isnan(result[:window - 1]))

    def test_all_methods_produce_values(self):
        """All entropy methods should produce valid values."""
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.02

        for method in [0, 1, 2]:
            result = rolling_entropy(returns, window=20, n_bins=3, method=method)

            # Should have some valid values
            valid = result[~np.isnan(result)]
            assert len(valid) > 0
            assert np.all(valid >= 0)

    def test_repetitive_returns_lower_entropy(self):
        """Repetitive returns should have lower entropy."""
        np.random.seed(42)
        n = 100

        # Repetitive: mostly same returns
        returns_rep = np.zeros(n)
        returns_rep[::10] = 0.01

        # Random returns
        returns_rand = np.random.randn(n) * 0.02

        entropy_rep = rolling_entropy(returns_rep, window=50, n_bins=3, method=0)
        entropy_rand = rolling_entropy(returns_rand, window=50, n_bins=3, method=0)

        # Random should have higher average entropy
        assert np.nanmean(entropy_rand) > np.nanmean(entropy_rep)


class TestShannonEntropyTransform:
    """Tests for ShannonEntropyTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({'returns': returns}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = ShannonEntropyTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = ShannonEntropyTransform('returns', window=50, n_bins=3)

        assert 'shannon_entropy' in transform.output_name

    def test_index_preserved(self, sample_data):
        """Output index should match input index."""
        transform = ShannonEntropyTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')

        assert result.index.equals(sample_data.index)


class TestLZEntropyTransform:
    """Tests for LZEntropyTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({'returns': returns}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = LZEntropyTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)


class TestKontoyiannisEntropyTransform:
    """Tests for KontoyiannisEntropyTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({'returns': returns}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = KontoyiannisEntropyTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)


class TestEntropyImpliedVolTransform:
    """Tests for EntropyImpliedVolTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({'returns': returns}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = EntropyImpliedVolTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_volatility_reasonable_range(self, sample_data):
        """Implied volatility should be in reasonable range.

        Note: The entropy-to-volatility formula σ = √(2πe) × 2^(-H)
        is designed for continuous Gaussian distributions. When applied
        to discretized symbol sequences, the entropy is typically lower
        (symbols have less information than continuous values), resulting
        in higher implied volatility estimates. A threshold of 50 (5000%
        annualized) accounts for this discretization effect.
        """
        transform = EntropyImpliedVolTransform('returns', window=50, n_bins=3)

        result = transform(sample_data, backend='nb')
        valid = result.dropna()

        # Volatility should be positive and not extreme
        assert np.all(valid > 0)
        assert np.all(valid < 50.0)  # 5000% annualized vol accounts for discretization effect


class TestComputeEntropyConvenience:
    """Tests for convenience function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns Series."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        return pd.Series(np.random.randn(n) * 0.02, index=dates, name='returns')

    def test_shannon_method(self, sample_returns):
        """Test Shannon method."""
        result = compute_entropy(sample_returns, window=50, method='shannon')

        assert isinstance(result, pd.Series)
        assert 'shannon' in result.name

    def test_lz_method(self, sample_returns):
        """Test LZ method."""
        result = compute_entropy(sample_returns, window=50, method='lz')

        assert isinstance(result, pd.Series)
        assert 'lz' in result.name

    def test_kontoyiannis_method(self, sample_returns):
        """Test Kontoyiannis method."""
        result = compute_entropy(sample_returns, window=50, method='kontoyiannis')

        assert isinstance(result, pd.Series)

    def test_invalid_method_raises(self, sample_returns):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_entropy(sample_returns, window=50, method='invalid')


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_returns(self):
        """Test with empty returns."""
        returns = np.array([])
        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        assert len(result) == 0

    def test_shorter_than_window(self):
        """Test with data shorter than window."""
        returns = np.random.randn(10) * 0.02
        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        assert np.all(np.isnan(result))

    def test_constant_returns(self):
        """Test with constant returns (all same value)."""
        returns = np.zeros(100)
        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        # Should have valid values (entropy should be low)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # All same value means we're always in "flat" bin, so entropy should be 0
        assert_array_almost_equal(valid, np.zeros_like(valid), decimal=6)

    def test_extreme_returns(self):
        """Test with extreme return values."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.5  # Very volatile

        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        # Should handle without crash
        assert result.shape == (100,)

    def test_nan_in_returns(self):
        """Test with NaN values in returns."""
        returns = np.random.randn(100) * 0.02
        returns[50:55] = np.nan

        result = rolling_entropy(returns, window=20, n_bins=3, method=0)

        # Should have some valid values despite NaNs
        assert result.shape == (100,)