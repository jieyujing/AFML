# Tasks: Add Log-Price and Cumulative Series FFD Features

## 1. Implement Log-Price FFD Features

- [x] 1.1 在 `FeatureEngineer` 类中添加 `_add_log_price_ffd_features()` 方法
- [x] 1.2 对 close, open, high, low 依次取自然对数后应用 FFD
- [x] 1.3 特征命名: `log_close_ffd`, `log_open_ffd`, `log_high_ffd`, `log_low_ffd`
- [x] 1.4 在 `_compute_features()` 中调用新方法

## 2. Implement Cumulative Volume FFD Feature

- [x] 2.1 添加 `_add_cum_volume_ffd_features()` 方法
- [x] 2.2 先计算 `cumsum(volume)`，再应用 FFD
- [x] 2.3 特征命名: `cum_volume_ffd`

## 3. Verify

- [x] 3.1 运行现有测试确保没有破坏原有功能: `uv run pytest tests/test_features.py`
- [x] 3.2 验证新特征出现在输出中
