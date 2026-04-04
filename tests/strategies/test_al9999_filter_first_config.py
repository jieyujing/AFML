import importlib.util
from pathlib import Path


def _load_config_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "config.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_config", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_first_config_exists_and_has_expected_keys():
    config = _load_config_module()
    cfg = config.FILTER_FIRST_CONFIG
    assert "threshold_grid" in cfg
    assert "shrinkage_min" in cfg
    assert "shrinkage_max" in cfg
    assert "execution_guard" in cfg
    assert "side_mode" in cfg
