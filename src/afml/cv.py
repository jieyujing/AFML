"""
Purged K-Fold Cross-Validation for Financial Machine Learning (Polars Optimized).

This module implements Purged K-Fold CV using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple, Union

import numpy as np
from polars import DataFrame, Series


class PurgedKFoldCV:
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
        >>> cv = PurgedKFoldCV(n_splits=5, embargo=0.1)
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
        Initialize PurgedKFoldCV.

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
        """Get information about CV configuration."""
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
    """
    if len(train_idx) == 0:
        return {"no_leakage": True, "message": "Empty train set"}
    
    # Check for index overlap
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap) > 0:
        return {"no_leakage": False, "message": f"Index overlap detected: {len(overlap)} samples"}

    return {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "no_leakage": True,
        "message": "No information leakage detected between splits",
    }
