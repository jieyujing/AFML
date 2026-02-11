"""
Purged K-Fold Cross Validation for Financial Machine Learning.

This module implements PurgedKFoldCV from AFML Chapter 7,
handling purging and embargoing to prevent information leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Optional, Generator


def get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Find training set indexes given sample info and test range.

    Removes training samples that overlap with test-label intervals.

    Args:
        samples_info_sets: Series with index as sample ID and value as info time (t1)
        test_times: Series with test start and end times

    Returns:
        Series of valid training indices
    """
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        # Purge samples whose t1 (vertical barrier) falls within test range
        # These are samples whose labels would leak into the test period
        t1_in_test = train[(start_ix <= train) & (train <= end_ix)].index.unique()
        train = train.drop(t1_in_test)
    return train


class PurgedKFoldCV(KFold):
    """
    Extended KFold for financial data with label overlap.

    Purges observations overlapping test-label intervals and applies
    embargo to samples immediately following test sets.

    Args:
        n_splits: Number of folds
        samples_info_sets: Series with vertical barrier times (t1)
        embargo: Percentage of data to embargo after test set

    Example:
        >>> cv = PurgedKFoldCV(n_splits=5, samples_info_sets=df['t1'], embargo=0.01)
        >>> for train_idx, test_idx in cv.split(X):
        ...     # Train on purged train set
        ...     # Test on clean test set
    """

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[pd.Series] = None,
        embargo: float = 0.01,
    ):
        """
        Initialize PurgedKFoldCV.

        Args:
            n_splits: Number of CV splits
            samples_info_sets: Series with 't1' (vertical barrier) values
            embargo: Fraction of samples to embargo after test set
        """
        if samples_info_sets is not None and not isinstance(
            samples_info_sets, pd.Series
        ):
            raise ValueError("samples_info_sets must be a pandas Series")

        super(PurgedKFoldCV, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.embargo = embargo

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices.

        Args:
            X: Feature DataFrame
            y: Labels Series
            groups: Ignored

        Yields:
            (train_indices, test_indices) tuples
        """
        if self.samples_info_sets is None:
            raise ValueError("samples_info_sets must be provided")

        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and samples_info_sets must have the same length")

        indices = np.arange(X.shape[0])
        n_embargo = int(X.shape[0] * self.embargo)

        test_ranges = [
            (ix[0], ix[-1] + 1)
            for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            t0 = self.samples_info_sets.index[start_ix]
            t1 = self.samples_info_sets.iloc[end_ix - 1]

            test_times = pd.Series(index=[t0], data=[t1])

            train_times = get_train_times(self.samples_info_sets, test_times)

            train_mask = self.samples_info_sets.index.isin(train_times.index)
            train_indices = indices[train_mask]

            if n_embargo > 0 and end_ix < X.shape[0]:
                embargo_end = min(end_ix + n_embargo, X.shape[0])
                embargo_indices = np.arange(end_ix, embargo_end)
                train_indices = np.setdiff1d(train_indices, embargo_indices)

            yield train_indices, test_indices

    def get_n_splits(
        self, X: pd.DataFrame = None, y: pd.Series = None, groups=None
    ) -> int:
        """Return number of splits."""
        return self.n_splits
