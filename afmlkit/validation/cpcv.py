import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection._split import BaseCrossValidator

class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold Cross Validation.
    """
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        t1: pd.Series | None = None,
        embargo_pct: float = 0.01,
    ):
        super().__init__()
        if t1 is None:
            raise ValueError("CPCV requires the `t1` series.")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with a DatetimeIndex.")

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo_width = int(n_samples * self.embargo_pct)
        fold_bounds = self._get_fold_bounds(n_samples)

        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))

        for combo_idx, test_folds in enumerate(combinations):
            test_indices = []
            test_ranges = []
            test_folds_indices = [] # keep track of indices per test fold
            
            for f in test_folds:
                start, end = fold_bounds[f]
                fold_idx_arr = indices[start:end]
                test_indices.extend(fold_idx_arr)
                test_folds_indices.append((f, fold_idx_arr))
                
                # The time range of the test fold
                if end > start:
                    t_start = X.index[start]
                    t_end = X.index[end - 1]
                    test_ranges.append((t_start, t_end))

            test_indices = np.array(test_indices)
            test_set = set(test_indices)

            train_indices = []
            embargo_zones = []
            for f in test_folds:
                fold_end = fold_bounds[f][1]
                embargo_zones.append(set(range(fold_end, min(fold_end + embargo_width, n_samples))))
            
            embargo_set = set().union(*embargo_zones)

            for i in indices:
                if i in test_set or i in embargo_set:
                    continue

                sample_start = X.index[i]
                sample_end = pd.Timestamp(self.t1.iloc[i])

                overlaps = False
                for (t_start, t_end) in test_ranges:
                    if (sample_start < t_end) and (sample_end > t_start):
                        overlaps = True
                        break

                if not overlaps:
                    train_indices.append(i)

            train_indices = np.array(train_indices, dtype=np.int64)
            
            yield train_indices, test_indices, test_folds_indices

    def _get_fold_bounds(self, n_samples: int):
        fold_size = n_samples // self.n_splits
        bounds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            bounds.append((start, end))
        return bounds

def generate_cpcv_paths(combinations, n_splits, n_test_splits):
    n_paths = len(combinations) * n_test_splits // n_splits
    path_assignments = {}
    for fold in range(n_splits):
        splits_with_fold = [i for i, combo in enumerate(combinations) if fold in combo]
        for path_idx, split_idx in enumerate(splits_with_fold):
            path_assignments[(split_idx, fold)] = path_idx
    return n_paths, path_assignments
