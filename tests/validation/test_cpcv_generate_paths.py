import numpy as np
import pandas as pd

from afmlkit.validation.cpcv import CombinatorialPurgedKFold


def _build_cpcv(n_samples: int = 60) -> tuple[CombinatorialPurgedKFold, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=n_samples, freq="min")
    t1 = pd.Series(index + pd.Timedelta(minutes=1), index=index)
    X = pd.DataFrame({"feature": np.arange(n_samples, dtype=np.float64)}, index=index)
    cpcv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_splits=2,
        t1=t1,
        embargo_pct=0.0,
    )
    return cpcv, X


def test_generate_paths_contract_lengths_and_shape():
    cpcv, X = _build_cpcv()

    def strategy_func(train_idx, test_idx, test_folds):
        fold_returns = {
            fold_idx: [fold_idx + 0.1, fold_idx + 0.2, fold_idx + 0.3]
            for fold_idx, _ in test_folds
        }
        return False, fold_returns

    returns_paths, sharpe_paths = cpcv.generate_paths(X, strategy_func)

    assert len(returns_paths) == cpcv.get_n_paths()
    assert sharpe_paths.shape == (cpcv.get_n_paths(),)
    assert all(len(path_returns) == 18 for path_returns in returns_paths)
