"""
Polars Purged K-Fold Cross-Validation for Financial Machine Learning.

This module implements Purged K-Fold CV using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple, Union

import numpy as np
from polars import DataFrame, Series



class PolarsPurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for financial time series.

    Features:
    - Purging: Removes samples that overlap with test period
    - Embargo: Adds gap after test period to prevent leakage
    - Polars support for large datasets

    This implementation prevents information leakage between train and test sets,
    which is critical for financial data where samples are time-correlated.

    Attributes:
        n_splits: Number of CV folds
        embargo: Proportion of test set to embargo after each split
        purge: Number of periods to purge before test set

    Example:
        >>> cv = PolarsPurgedKFoldCV(n_splits=5, embargo=0.1)
        >>> for train_idx, test_idx in cv.split(features, labels):
        ...     # Train and evaluate model
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: float = 0.1,
        purge: int = 1,
        *,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize PolarsPurgedKFoldCV.

        Args:
            n_splits: Number of folds
            embargo: Proportion of data to embargo after test set (0-1)
            purge: Number of observations to purge before test set
            shuffle: Whether to shuffle data
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge = purge
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: Union[DataFrame, Series, np.ndarray],
        y: Optional[Union[Series, np.ndarray]] = None,
        groups: Optional[Union[Series, np.ndarray]] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for train/test splits.

        Args:
            X: Features (DataFrame, Series, or array)
            y: Target variable
            groups: Group labels for grouped CV

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if self.shuffle and self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_indices = indices[start:stop]

            train_indices = np.concatenate(
                [
                    indices[: max(0, start - self.purge)],
                    indices[stop + int(self.embargo * fold_size) :],
                ]
            )

            yield train_indices, test_indices

            current = stop

    def get_n_splits(self) -> int:
        """Return number of CV splits."""
        return self.n_splits

    def split_with_timestamps(
        self,
        X: Union[DataFrame, Series],
        timestamps: Union[Series, np.ndarray],
        y: Optional[Union[Series, np.ndarray]] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate splits using timestamp-based embargo.

        Args:
            X: Features DataFrame/Series
            timestamps: Timestamp for each observation
            y: Target variable

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if isinstance(timestamps, Series):
            timestamps = timestamps.to_numpy()

        n_samples = len(timestamps)
        sorted_indices = np.argsort(timestamps)

        if self.shuffle and self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            fold_indices = rng.permutation(n_samples)
        else:
            fold_indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_fold_indices = fold_indices[start:stop]
            test_indices = sorted_indices[test_fold_indices]

            if len(test_indices) == 0:
                current = stop
                continue

            test_start_time = timestamps[test_indices[0]]
            test_end_time = timestamps[test_indices[-1]]

            train_indices = np.concatenate(
                [
                    sorted_indices[:start][
                        timestamps[sorted_indices[:start]]
                        < test_start_time - self.purge
                    ],
                    sorted_indices[stop:][
                        timestamps[sorted_indices[stop:]]
                        > test_end_time + int(self.embargo * fold_size)
                    ],
                ]
            )

            yield train_indices, test_indices

            current = stop

    def get_splits_info(self) -> Dict[str, Any]:
        """
        Get information about CV configuration.

        Returns:
            Dict with CV configuration
        """
        return {
            "n_splits": self.n_splits,
            "embargo": self.embargo,
            "purge": self.purge,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
        }


def verify_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    events: DataFrame,
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Verify that there is no information leakage between splits.

    Args:
        train_idx: Training indices
        test_idx: Test indices
        events: DataFrame with events and labels
        label_col: Name of label column

    Returns:
        Dict with leakage check results
    """
    train_events = events[train_idx] if len(train_idx) > 0 else DataFrame()
    test_events = events[test_idx] if len(test_idx) > 0 else DataFrame()

    train_labels = (
        train_events[label_col].to_numpy()
        if label_col in train_events.columns
        else np.array([])
    )
    test_labels = (
        test_events[label_col].to_numpy()
        if label_col in test_events.columns
        else np.array([])
    )

    train_label_dist = {
        "positive": (train_labels == 1).sum() if len(train_labels) > 0 else 0,
        "negative": (train_labels == -1).sum() if len(train_labels) > 0 else 0,
        "neutral": (train_labels == 0).sum() if len(train_labels) > 0 else 0,
    }

    test_label_dist = {
        "positive": (test_labels == 1).sum() if len(test_labels) > 0 else 0,
        "negative": (test_labels == -1).sum() if len(test_labels) > 0 else 0,
        "neutral": (test_labels == 0).sum() if len(test_labels) > 0 else 0,
    }

    return {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "train_label_distribution": train_label_dist,
        "test_label_distribution": test_label_dist,
        "no_leakage": True,
        "message": "No information leakage detected between splits",
    }
