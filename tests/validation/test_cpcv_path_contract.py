from math import comb

import pandas as pd
import pytest

from afmlkit.validation.cpcv import CombinatorialPurgedKFold, generate_cpcv_paths


def test_generate_cpcv_paths_contract():
    all_combos, n_paths, path_assignments = generate_cpcv_paths(
        n_splits=6,
        n_test_splits=2,
    )

    assert len(all_combos) == 15
    assert n_paths == comb(5, 1)
    assert len(path_assignments) == 30
    assert set(path_assignments) == {0, 1, 2, 3, 4}


def test_cpcv_exposes_path_metadata():
    index = pd.date_range("2024-01-01", periods=60, freq="min")
    t1 = pd.Series(index + pd.Timedelta(minutes=1), index=index)

    cpcv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_splits=2,
        t1=t1,
    )

    all_combos, _, _ = generate_cpcv_paths(n_splits=6, n_test_splits=2)
    expected_keys = {
        (split_idx, fold)
        for split_idx, combo in enumerate(all_combos)
        for fold in combo
    }

    path_assignments = cpcv.get_path_assignments()
    assert cpcv.get_n_paths() == 5
    assert len(path_assignments) == 30
    assert set(path_assignments.keys()) == expected_keys
    assert set(path_assignments.values()) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("n_splits,n_test_splits", [(6, 6), (6, 7)])
def test_generate_cpcv_paths_invalid_n_test_splits(n_splits, n_test_splits):
    with pytest.raises(ValueError):
        generate_cpcv_paths(n_splits=n_splits, n_test_splits=n_test_splits)
