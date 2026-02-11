"""
Polars Bet Sizer for Financial Machine Learning.

This module implements bet sizing using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from polars import DataFrame, Series



class PolarsBetSizer:
    """
    Bet sizing calculator using discrete probability estimates.

    Features:
    - Bet size from probability estimates using EDF
    - Discretized position sizing
    - Accuracy scoring

    This implementation uses Polars for improved performance on large datasets.

    Attributes:
        threshold: Probability threshold for bet sizing
        quantity: Maximum position quantity
        num_classes: Number of probability classes for discretization

    Example:
        >>> sizer = PolarsBetSizer(threshold=0.5)
        >>> bet_sizes = sizer.get_bet_size(predictions)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        quantity: int = 100,
        num_classes: int = 10,
        *,
        lazy: bool = False,
    ):
        """
        Initialize PolarsBetSizer.

        Args:
            threshold: Probability threshold for bet sizing
            quantity: Maximum position quantity
            num_classes: Number of probability classes for discretization
            lazy: Whether to use lazy evaluation
        """
        self.threshold = threshold
        self.quantity = quantity
        self.num_classes = num_classes
        self.lazy = lazy
        self._is_fitted = False

    def fit(
        self,
        prob_series: Union[Series, DataFrame],
        returns: Union[Series, DataFrame],
        y: Optional[Any] = None,
    ) -> "PolarsBetSizer":
        """
        Fit the bet sizer with probability estimates and returns.

        Args:
            prob_series: Probability estimates
            returns: Actual returns
            y: Ignored

        Returns:
            self
        """
        if isinstance(prob_series, DataFrame):
            prob_series = prob_series["prob"]

        if isinstance(returns, DataFrame):
            returns = returns["return"]

        prob_array = prob_series.to_numpy()
        returns_array = returns.to_numpy()

        self._fitEDF = EmpiricalDistributionFunction(prob_array, returns_array)

        self._is_fitted = True

        return self

    def bet_size_probability(
        self,
        prob_series: Union[Series, DataFrame],
    ) -> Series:
        """
        Calculate bet sizes from probability estimates.

        Args:
            prob_series: Probability estimates

        Returns:
            Series with bet sizes
        """
        if isinstance(prob_series, DataFrame):
            prob_series = prob_series["prob"]

        if not self._is_fitted:
            return Series(values=np.zeros(len(prob_series)))

        prob_array = prob_series.to_numpy()
        bet_sizes = np.zeros_like(prob_array)

        # Vectorized calculation
        # Case 1: p > threshold
        mask_long = prob_array > self.threshold
        if np.any(mask_long):
            cdf_vals = self._fitEDF.get_cdf_vectorized(prob_array[mask_long])
            bet_sizes[mask_long] = self.quantity * cdf_vals

        # Case 2: p < 1 - threshold
        mask_short = prob_array < (1 - self.threshold)
        if np.any(mask_short):
            # Using 1-p for short side sizing logic
            cdf_vals = self._fitEDF.get_cdf_vectorized(1 - prob_array[mask_short])
            bet_sizes[mask_short] = -self.quantity * cdf_vals

        return Series(values=bet_sizes)

    def get_bet_size(
        self,
        prob_series: Union[Series, DataFrame],
        *,
        prices: Optional[Union[Series, np.ndarray]] = None,
    ) -> Series:
        """
        Get discretized bet sizes.

        Args:
            prob_series: Probability estimates
            prices: Optional price series for position sizing

        Returns:
            Series with bet sizes
        """
        raw_sizes = self.bet_size_probability(prob_series)

        if prices is not None:
            if isinstance(prices, Series):
                prices = prices.to_numpy()

            raw_sizes = raw_sizes.to_numpy()
            raw_sizes = raw_sizes / prices
            raw_sizes = raw_sizes * prices[0] if len(prices) > 0 else raw_sizes

        sizes_array = raw_sizes.to_numpy()

        size_ranks = np.argsort(sizes_array)
        size_quantized = np.linspace(0, self.quantity, self.num_classes)

        quantized_sizes = np.zeros(len(sizes_array))
        for i in range(self.num_classes):
            mask = size_ranks == i
            quantized_sizes[mask] = size_quantized[i]

        return Series(values=quantized_sizes)

    def discretize(
        self,
        sizes: Union[Series, np.ndarray],
    ) -> Series:
        """
        Discretize bet sizes into buckets.

        Args:
            sizes: Raw bet sizes

        Returns:
            Series with discretized sizes
        """
        if isinstance(sizes, Series):
            sizes = sizes.to_numpy()

        unique_sizes = np.unique(sizes)
        if len(unique_sizes) <= self.num_classes:
            return Series(values=sizes)

        percentiles = np.linspace(0, 100, self.num_classes + 1)
        thresholds = np.percentile(sizes, percentiles[1:-1])

        discretized = np.zeros(len(sizes))
        for i, threshold in enumerate(thresholds):
            discretized[sizes > threshold] = i + 1

        return Series(values=discretized)

    def score_accuracy(
        self,
        predictions: Union[Series, np.ndarray],
        actual: Union[Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate accuracy scores.

        Args:
            predictions: Predicted values
            actual: Actual values

        Returns:
            Dict with accuracy metrics
        """
        if isinstance(predictions, Series):
            predictions = predictions.to_numpy()
        if isinstance(actual, Series):
            actual = actual.to_numpy()

        accuracy = (predictions == actual).mean()

        return {
            "accuracy": accuracy,
        }

    def get_metrics(
        self,
        returns: Union[Series, np.ndarray],
        bet_sizes: Union[Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            returns: Strategy returns
            bet_sizes: Bet sizes

        Returns:
            Dict with performance metrics
        """
        if isinstance(returns, Series):
            returns = returns.to_numpy()
        if isinstance(bet_sizes, Series):
            bet_sizes = bet_sizes.to_numpy()

        strategy_returns = returns * bet_sizes

        sharpe = self._sharpe_ratio(strategy_returns)
        sortino = self._sortino_ratio(strategy_returns)

        cumulative = np.cumsum(strategy_returns)
        max_dd = self._max_drawdown(cumulative)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "total_return": cumulative[-1] if len(cumulative) > 0 else 0,
            "avg_return": np.mean(strategy_returns),
            "std_return": np.std(strategy_returns),
        }

    def _sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        excess = returns - risk_free
        if np.std(excess) < 1e-10:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    def _sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        return np.mean(returns) / downside_std * np.sqrt(252)

    def _max_drawdown(self, cumulative: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return np.max(drawdown)

    def fit_transform(
        self,
        prob_series: Union[Series, DataFrame],
        returns: Union[Series, DataFrame],
        y: Optional[Any] = None,
    ) -> Series:
        """
        Fit and get bet sizes in one step.

        Args:
            prob_series: Probability estimates
            returns: Actual returns
            y: Ignored

        Returns:
            Bet sizes
        """
        self.fit(prob_series, returns, y)
        return self.bet_size_probability(prob_series)

    def get_bet_sizer_info(self) -> Dict[str, Any]:
        """
        Get information about the bet sizer.

        Returns:
            Dict with configuration
        """
        return {
            "threshold": self.threshold,
            "quantity": self.quantity,
            "num_classes": self.num_classes,
            "lazy": self.lazy,
            "is_fitted": self._is_fitted,
        }


class EmpiricalDistributionFunction:
    """
    Empirical Distribution Function for bet sizing.

    Used to estimate the probability of returns given
    probability estimates from a classifier.
    """

    def __init__(
        self,
        probabilities: np.ndarray,
        returns: np.ndarray,
    ):
        """
        Initialize EDF.

        Args:
            probabilities: Probability estimates
            returns: Corresponding returns
        """
        self.probabilities = probabilities
        self.returns = returns

        sorted_idx = np.argsort(probabilities)
        self.sorted_probs = probabilities[sorted_idx]
        self.sorted_returns = returns[sorted_idx]

        self.n = len(probabilities)

        # Precompute weights and cumulative sums for O(1) lookup
        # Weight formula: 1/(i+1) where i is the rank index
        weights = 1.0 / (np.arange(self.n) + 1)
        self.total_weight = np.sum(weights)

        # Cumulative weighted returns
        # Only sum returns where prob <= x (which corresponds to sorted indices 0..k)
        weighted_returns = weights * np.maximum(0, self.sorted_returns)
        self.cum_weighted_returns = np.cumsum(weighted_returns)

    def get_cdf(self, x: float) -> float:
        """
        Get CDF value at x (single value).
        """
        idx = np.searchsorted(self.sorted_probs, x, side='right') - 1
        if idx < 0:
            return 0.0
        return self.cum_weighted_returns[idx] / self.total_weight

    def get_cdf_vectorized(self, x_array: np.ndarray) -> np.ndarray:
        """
        Get CDF values for an array of x.
        """
        indices = np.searchsorted(self.sorted_probs, x_array, side='right') - 1
        
        counts = np.zeros(len(x_array))
        mask = indices >= 0
        counts[mask] = self.cum_weighted_returns[indices[mask]]
        
        return counts / self.total_weight

    def get_returns_distribution(self) -> Dict[str, float]:
        """
        Get statistics about returns distribution.

        Returns:
            Dict with return statistics
        """
        return {
            "mean": np.mean(self.returns),
            "std": np.std(self.returns),
            "min": np.min(self.returns),
            "max": np.max(self.returns),
            "positive_ratio": (self.returns > 0).mean(),
        }
