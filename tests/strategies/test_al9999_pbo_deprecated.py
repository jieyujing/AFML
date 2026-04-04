from pathlib import Path
import importlib.util
import warnings


def _load_09_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "AL9999"
        / "09_pbo_validation.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_pbo_validation", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emit_deprecation_warning_contains_deprecated_message():
    module = _load_09_module()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module.emit_deprecation_warning()

    assert any(
        issubclass(item.category, DeprecationWarning) and "已废弃" in str(item.message)
        for item in caught
    )
