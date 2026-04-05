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


def test_primary_model_type_supports_rf():
    config = _load_config_module()
    allowed = {"ma", "cusum_direction", "rf"}
    assert config.PRIMARY_MODEL_TYPE in allowed


def test_rf_primary_config_contains_expected_keys():
    config = _load_config_module()
    cfg = config.RF_PRIMARY_CONFIG
    assert cfg["n_estimators"] == 1000
    assert cfg["cv_n_splits"] == 5
    assert cfg["cv_embargo_pct"] == 0.01
    assert cfg["holdout_months"] == 12
    assert cfg["max_samples_method"] in {"avgU", "float"}
    assert isinstance(cfg["feature_prefixes"], list)
    assert cfg["t1_col"] == "exit_ts"
