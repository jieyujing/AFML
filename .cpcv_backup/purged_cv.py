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

import itertools
from scipy.special import comb

class CombinatorialPurgedKFold(BaseCrossValidator):
    def __init__(self, n_groups=6, n_test_groups=2, t1=None, embargo_pct=0.01):
        super().__init__()
        if t1 is None:
            raise ValueError("CombinatorialPurgedKFold requires the `t1` series.")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.embargo_pct = embargo_pct
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(comb(self.n_groups, self.n_test_groups))
        
    def _get_fold_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        fold_size = n_samples // self.n_groups
        bounds = []
        for i in range(self.n_groups):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_groups - 1 else n_samples
            bounds.append((start, end))
        return bounds
        
    def get_sample_groups(self, n_samples: int) -> np.ndarray:
        bounds = self._get_fold_bounds(n_samples)
        groups = np.zeros(n_samples, dtype=int)
        for g, (start, end) in enumerate(bounds):
            groups[start:end] = g
        return groups

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a Dataframe with DatetimeIndex.")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_width = int(n_samples * self.embargo_pct)
        group_bounds = self._get_fold_bounds(n_samples)
        
        group_indices = list(range(self.n_groups))
        combos = list(itertools.combinations(group_indices, self.n_test_groups))
        
        # Purge tool function
        # Instantiate a dummy PurgedKFold just to use its purge/embargo methods easily
        from afmlkit.validation.purged_cv import PurgedKFold
        pkf = PurgedKFold(n_splits=self.n_groups, t1=self.t1, embargo_pct=self.embargo_pct)
        
        for combo in combos:
            test_groups = combo
            train_groups = [g for g in group_indices if g not in test_groups]
            
            test_indices = []
            for g in test_groups:
                s, e = group_bounds[g]
                test_indices.extend(indices[s:e])
            test_indices = np.array(test_indices)
            
            all_train = []
            for g in train_groups:
                s, e = group_bounds[g]
                all_train.extend(indices[s:e])
            all_train = np.array(all_train)
            
            # Purge & Embargo against EACH test group separately
            for g in test_groups:
                ts, te = group_bounds[g]
                test_start_time = X.index[ts]
                test_end_time = X.index[max(0, te - 1)]
                all_train = pkf._purge(X, all_train, test_indices, test_start_time, test_end_time)
                all_train = pkf._embargo(all_train, te, n_samples, embargo_width)
                
            yield all_train, test_indices

    def get_paths(self):
        group_indices = list(range(self.n_groups))
        combos = list(itertools.combinations(group_indices, self.n_test_groups))
        n_paths = int(comb(self.n_groups - 1, self.n_test_groups - 1))
        
        paths = {p: {} for p in range(n_paths)}
        for split_idx, combo in enumerate(combos):
            for group in combo:
                for p in range(n_paths):
                    if group not in paths[p]:
                        paths[p][group] = split_idx
                        break
        return paths
