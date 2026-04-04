from pathlib import Path
import importlib.util

import numpy as np


def _load_09b_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "09b_cpcv_pbo_validation.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_cpcv_pbo_validation", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_param_grid_has_minimum_candidates():
    module = _load_09b_module()
    param_grid = module.build_param_grid()

    assert isinstance(param_grid, list)
    assert len(param_grid) >= 25


def test_compute_rank_pbo_on_synthetic_matrix():
    module = _load_09b_module()

    sharpe_is = np.array([[2.0, 2.0], [1.0, 1.0]], dtype=np.float64)
    sharpe_oos = np.array([[-1.0, -0.8], [0.1, 0.2]], dtype=np.float64)

    pbo, stats = module.compute_rank_pbo(sharpe_is=sharpe_is, sharpe_oos=sharpe_oos)

    assert pbo == 1.0
    assert stats["n_paths"] == 2
