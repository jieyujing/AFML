import itertools
from collections.abc import Callable, Iterable
from math import comb

import numpy as np
import pandas as pd
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
        (
            self._combinations,
            self._n_paths,
            path_assignments,
        ) = generate_cpcv_paths(self.n_splits, self.n_test_splits)
        self._path_assignments = self._build_path_assignment_lookup(
            self._combinations,
            path_assignments,
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._combinations)

    def get_n_paths(self) -> int:
        return self._n_paths

    def get_path_assignments(self) -> dict[tuple[int, int], int]:
        return dict(self._path_assignments)

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with a DatetimeIndex.")

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo_width = int(n_samples * self.embargo_pct)
        fold_bounds = self._get_fold_bounds(n_samples)

        for test_folds in self._combinations:
            test_indices = []
            test_ranges = []
            test_folds_indices = []  # keep track of indices per test fold
            
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

    def generate_paths(
        self,
        X: pd.DataFrame,
        strategy_func: Callable,
        y=None,
        annualization_factor: float = 252.0,
    ) -> tuple[list[pd.Series], np.ndarray]:
        """
        Generate CPCV return paths and per-path Sharpe ratios.

        :param X: Feature matrix used by the splitter.
        :param strategy_func: Callable that receives
            (train_idx, test_idx, test_folds) and returns
            (is_sharpe, fold_returns).
        :param y: Optional target vector (kept for API compatibility).
        :param annualization_factor: Annualization factor for Sharpe ratio.
        :returns: (returns_paths, sharpe_paths).
        """
        n_paths = self.get_n_paths()
        assignment_lookup = self.get_path_assignments()
        path_returns: list[list[float]] = [[] for _ in range(n_paths)]

        for split_idx, (train_idx, test_idx, test_folds) in enumerate(self.split(X, y=y)):
            _is_sharpe, fold_returns = strategy_func(train_idx, test_idx, test_folds)

            for fold_idx, _ in test_folds:
                fold_path_key = (split_idx, fold_idx)
                if fold_idx not in fold_returns or fold_path_key not in assignment_lookup:
                    continue

                path_idx = assignment_lookup[fold_path_key]
                fold_ret_iter: Iterable[float] = fold_returns[fold_idx]
                path_returns[path_idx].extend(fold_ret_iter)

        returns_paths = [
            pd.Series(path_returns[path_idx], dtype=np.float64)
            for path_idx in range(n_paths)
        ]

        sharpe_paths = np.zeros(n_paths, dtype=np.float64)
        annualization_scale = np.sqrt(float(annualization_factor))
        for path_idx, returns in enumerate(returns_paths):
            if len(returns) < 2:
                continue

            std = float(returns.std(ddof=1))
            if std == 0.0:
                continue

            sharpe_paths[path_idx] = float(returns.mean()) / std * annualization_scale

        return returns_paths, sharpe_paths

    @staticmethod
    def _build_path_assignment_lookup(
        combinations: list[tuple[int, ...]],
        path_assignments: list[int],
    ) -> dict[tuple[int, int], int]:
        assignment_lookup: dict[tuple[int, int], int] = {}
        assignment_idx = 0

        for split_idx, combo in enumerate(combinations):
            for fold in combo:
                assignment_lookup[(split_idx, fold)] = path_assignments[assignment_idx]
                assignment_idx += 1

        if assignment_idx != len(path_assignments):
            raise ValueError("Unexpected CPCV path assignments length.")

        return assignment_lookup

    def _get_fold_bounds(self, n_samples: int):
        fold_size = n_samples // self.n_splits
        bounds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            bounds.append((start, end))
        return bounds


def generate_cpcv_paths(n_splits: int, n_test_splits: int):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not 1 <= n_test_splits < n_splits:
        raise ValueError("n_test_splits must satisfy 1 <= n_test_splits < n_splits.")

    all_combos = list(itertools.combinations(range(n_splits), n_test_splits))
    n_paths = comb(n_splits - 1, n_test_splits - 1)

    assignment_lookup = {}
    for fold in range(n_splits):
        splits_with_fold = [idx for idx, combo in enumerate(all_combos) if fold in combo]
        for path_idx, split_idx in enumerate(splits_with_fold):
            assignment_lookup[(split_idx, fold)] = path_idx

    path_assignments = []
    for split_idx, combo in enumerate(all_combos):
        for fold in combo:
            path_assignments.append(assignment_lookup[(split_idx, fold)])

    return all_combos, n_paths, path_assignments
