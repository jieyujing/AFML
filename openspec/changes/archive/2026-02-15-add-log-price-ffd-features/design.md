# Design: Add Log-Price and Cumulative Series FFD Features

## Context

`FeatureEngineer` 当前只对 `close` 价格做 FFD。根据 Qlib158 的设计，需要扩展到：
1. 对 OHLC 全部取对数后做 FFD
2. 对累积成交量做 FFD

FFD (Fractional Differentiation) 的核心优势：
- 保留价格记忆（非平稳性）
- 同时让序列变得平稳可建模
- 比简单 ROC（只看 1 天差异）能看到更长期的价格累积效应

## Goals / Non-Goals

**Goals:**
- 新增 `log_close_ffd`, `log_open_ffd`, `log_high_ffd`, `log_low_ffd` 特征
- 新增 `cum_volume_ffd` 特征
- 保留现有所有 ROC 和 FFD 特征不变

**Non-Goals:**
- 不修改现有的 `return_1` ~ `return_5` 特征
- 不添加 tick sign 相关的累积特征（当前数据中没有 tick sign 列）
- 不修改 `ffd_d` 的自动搜索逻辑

## Decisions

### Decision 1: Log-Price FFD 实现方式

**选择**: 先取对数，再做 FFD

```python
# 方式 1: log then FFD (chosen)
log_close = pl.col("close").log()
ffd_log_close = _frac_diff(log_close, d)
```

**理由**: 这是 Qlib158 的标准做法。先取对数可以将乘法关系转为加法关系，FFD 后的序列更接近正态分布。

### Decision 2: Cumulative Volume FFD 实现方式

**选择**: 先累积，再 FFD

```python
cum_volume = pl.col("volume").cumsum()
ffd_cum_volume = _frac_diff(cum_volume, d)
```

**理由**: 累积成交量本身是非平稳的（单调递增），FFD 可以提取其中的周期/均值回复特性。

### Decision 3: 特征命名规范

采用 `{原变量名}_{变换类型}_{ffd}` 格式：
- `log_close_ffd` - 对数价格 FFD
- `cum_volume_ffd` - 累积成交量 FFD

## Implementation Notes

1. 在 `_compute_features()` 中添加对新方法的调用
2. 复用现有的 `_frac_diff()` 方法（已存在于 `_add_ffd_features` 中）
3. 确保 `ffd_d <= 0` 时跳过新特征（与现有逻辑一致）
