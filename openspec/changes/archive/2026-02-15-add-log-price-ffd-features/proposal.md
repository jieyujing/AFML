# Proposal: Add Log-Prices and Cumulative Series FFD Features

## Why

当前 `FeatureEngineer` 只对 `close` 价格做 FFD (Fractional Differentiation)。根据 Qlib158 的设计，需要扩展到：

1. **Log-Prices FFD**: 对 OHLC 全部取对数后做 FFD，能捕获过去几十天的价格加权累积（替代仅看今天和昨天的 ROC）
2. **Cumulative Volume FFD**: 累积成交量 + FFD，提取更稳健的资金流信号

FFD 能保留价格记忆（非平稳性），同时让序列变得平稳可建模——这是金融时间序列的核心技术。

## What Changes

在 `src/afml/features.py` 的 `FeatureEngineer` 类中新增：

1. **Log-Price FFD 特征**：
   - `log_close_ffd`, `log_open_ffd`, `log_high_ffd`, `log_low_ffd`
   - 对每个价格取自然对数后，应用 FFD 变换

2. **Cumulative Volume FFD 特征**：
   - `cum_volume_ffd`: 累积成交量 + FFD
   - 如果有 tick sign 数据，可扩展 `cum_tick_sign_ffd`

3. **保留现有功能**：
   - 现有的 `return_1` ~ `return_5` ROC 特征保持不变
   - 现有的 `ffd_close`, `ffd_momentum`, `ffd_volatility`, `ffd_slope` 保持不变

## Capabilities

### New Capabilities
- `log_price_ffd_features`: 为 OHLC 生成对数价格 FFD 特征
- `cum_volume_ffd_feature`: 生成累积成交量 FFD 特征

### Modified Capabilities
- `FeatureEngineer.fit_transform()`: 新增两个特征族

## Impact

- `src/afml/features.py`: 新增 `_add_log_price_ffd_features()` 和 `_add_cum_volume_ffd_features()` 方法
- `tests/test_features.py`: 新增测试用例验证新特征
