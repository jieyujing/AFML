import numpy as np
import pytest

from afmlkit.validation import calculate_pbo, generate_cpcv_paths


def test_calculate_pbo_rank_definition_returns_one():
    sharpe_is = np.array(
        [
            [3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    sharpe_oos = np.array(
        [
            [-2.0, -2.0, -2.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    pbo, stats = calculate_pbo(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos, method='rank')

    assert pbo == 1.0
    assert stats['n_strategies'] == 4
    assert stats['n_paths'] == 3
    assert stats['pbo'] == 1.0


def test_calculate_pbo_raises_on_shape_mismatch():
    sharpe_is = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    sharpe_oos = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        calculate_pbo(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos)


def test_validation_exports_calculate_pbo_and_generate_cpcv_paths():
    assert callable(calculate_pbo)
    assert callable(generate_cpcv_paths)


def test_calculate_pbo_legacy_probability_1d_input():
    sharpe_paths = np.array([0.2, -0.1, 0.4, 0.0, 0.3], dtype=np.float64)

    pbo, stats = calculate_pbo(sharpe_paths, method='probability')

    assert 0.0 <= pbo <= 1.0
    assert stats['n_paths'] == sharpe_paths.size
    assert 0.0 <= stats['pbo'] <= 1.0
