import importlib.util
from pathlib import Path


def _load_config_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "config.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_config_primary_factory", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_primary_factory_config_exists_and_has_expected_keys():
    config = _load_config_module()
    cfg = config.PRIMARY_FACTORY_CONFIG

    assert "cusum_rates" in cfg
    assert "fast_windows" in cfg
    assert "slow_windows" in cfg
    assert "vertical_bars" in cfg
    assert "top_n_lightweight" in cfg
    assert "top_n_final" in cfg
    assert "score_weights" in cfg
