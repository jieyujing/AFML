"""
Purged K-Fold Cross-Validation for financial time series.

Implements the Purged K-Fold CV methodology from *Advances in Financial
Machine Learning* (López de Prado, 2018, Ch. 7).  Prevents information
leakage in two ways:

1. **Purge** – Remove training samples whose event span overlaps the test set.
2. **Embargo** – Skip training samples that fall within a buffer period
   after the test set ends, eliminating serial-correlation leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator compatible with the Sklearn API.

    :param n_splits: Number of folds (default 5).
    :param t1: Series mapping each sample's start time (index) to its
               event end time.  Must have the same index as the feature matrix.
    :param embargo_pct: Fraction of the total samples to use as an embargo
                        window after each test fold (default 0.01 = 1%).

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, t1=events_t1, embargo_pct=0.01)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])

    References
    ----------
    López de Prado, M. "Advances in Financial Machine Learning."
    Wiley, 2018.  Ch. 7, Snippet 7.3.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        embargo_pct: float = 0.01,
    ):
        super().__init__()
        if t1 is None:
            raise ValueError(
                "PurgedKFold requires the `t1` series "
                "(sample_start → event_end mapping)."
            )
        self._n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self._n_splits

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Yields (train_indices, test_indices) tuples for each fold.
        Training indices are purged (overlapping events removed) and
        embargoed (post-test buffer applied).

        :param X: Feature matrix (must share the same index as ``self.t1``).
        :param y: Ignored, present for API compatibility.
        :param groups: Ignored.
        :yields: Tuple of (train_indices, test_indices) as numpy integer arrays.
        """
        # Validate index alignment
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with a DatetimeIndex.")

        # Use integer positions for splitting
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Embargo width in number of samples
        embargo_width = int(n_samples * self.embargo_pct)

        # Create time-ordered folds
        fold_bounds = self._get_fold_bounds(n_samples)

        for fold_start, fold_end in fold_bounds:
            test_indices = indices[fold_start:fold_end]

            # ----- Purge: remove training samples whose events overlap test set -----
            test_start_time = X.index[fold_start]
            test_end_time = X.index[fold_end - 1]  # inclusive end

            train_indices = self._purge(
                X, indices, test_indices, test_start_time, test_end_time
            )

            # ----- Embargo: skip samples right after test set -----
            train_indices = self._embargo(
                train_indices, fold_end, n_samples, embargo_width
            )

            # ----- Diagnostic: print effective training ratio -----
            ratio = len(train_indices) / n_samples
            print(
                f"  [PurgedKFold] Fold test=[{fold_start}:{fold_end}] "
                f"| train_eff={len(train_indices)}/{n_samples} "
                f"({ratio:.1%})"
            )
            if ratio < 0.3:
                print(
                    f"  ⚠ WARNING: training ratio {ratio:.1%} < 30% — "
                    "考虑减少 n_splits 或降低 embargo_pct"
                )

            yield train_indices, test_indices

    # ------------------------------------------------------------------
    # Purge logic
    # ------------------------------------------------------------------
    def _purge(
        self,
        X: pd.DataFrame,
        all_indices: np.ndarray,
        test_indices: np.ndarray,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> np.ndarray:
        """
        Remove training samples whose event span [t_i, t1_i] overlaps
        the test time range [test_start, test_end].

        A training sample *i* is dropped if:
        - Its start time < test_end  **AND**
        - Its event end time (t1[i]) > test_start

        (i.e., any temporal overlap at all.)
        """
        test_set = set(test_indices)
        t1 = self.t1

        keep = []
        for i in all_indices:
            if i in test_set:
                continue
            sample_start = X.index[i]
            sample_end = pd.Timestamp(t1.iloc[i])

            # Check for overlap: [sample_start, sample_end] ∩ [test_start, test_end]
            overlaps = (sample_start < test_end) and (sample_end > test_start)
            # Actually we should check:
            # sample overlaps if NOT (sample_end <= test_start or sample_start >= test_end)
            # i.e. overlap = sample_start < test_end AND sample_end > test_start
            # But be careful: a sample that starts AT test_end is fine (not overlapping)
            if not overlaps:
                keep.append(i)

        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Embargo logic
    # ------------------------------------------------------------------
    @staticmethod
    def _embargo(
        train_indices: np.ndarray,
        fold_end: int,
        n_samples: int,
        embargo_width: int,
    ) -> np.ndarray:
        """
        Remove training samples that fall in the embargo zone
        [fold_end, fold_end + embargo_width).

        This eliminates serial-correlation leakage from samples
        immediately following the test set.
        """
        if embargo_width <= 0:
            return train_indices

        embargo_end = min(fold_end + embargo_width, n_samples)
        embargo_set = set(range(fold_end, embargo_end))
        return np.array(
            [i for i in train_indices if i not in embargo_set],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Fold boundary calculation
    # ------------------------------------------------------------------
    def _get_fold_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        """
        Compute contiguous fold boundaries for time-ordered splits.

        Returns a list of (start, end) tuples where each fold covers
        indices [start, end).
        """
        fold_size = n_samples // self._n_splits
        bounds = []
        for i in range(self._n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self._n_splits - 1 else n_samples
            bounds.append((start, end))
        return bounds
