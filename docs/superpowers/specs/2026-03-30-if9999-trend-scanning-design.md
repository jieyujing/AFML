---
name: IF9999 Phase 3 Trend Scanning Labels
description: Trend Scanning 标签生成设计，作为 Primary Model 输出供 Meta Model 学习
type: project
---

# IF9999 Phase 3: Trend Scanning 标签生成设计

## 概述

Phase 3 实现 Trend Scanning 标签生成，作为 Primary Model 输出。CUSUM 事件点经 Trend Scanning 处理后输出 side（趋势方向）和 t_value（置信度），供后续 Meta Model 学习。

**Why**: Trend Scanning 通过 OLS 统计检验定义"什么是趋势"，比简单规则更科学。t_value 作为样本权重让 Meta Model 更关注高置信度样本。

**How to apply**: 在 IF9999 策略开发中，Phase 3 位于特征工程之后、Meta Labeling 之前。

## 核心决策

| 项目 | 决定 | 理由 |
|------|------|------|
| 定位 | Primary Model 输出 | CUSUM → Trend Scan → side，符合 AFML 方法论 |
| L_windows | `[5, 10, 20, 30, 50]` | 期货波动快，聚焦短期趋势（5-50 Bars ≈ 1-8 天） |
| t_value 用途 | 样本权重 | 高 |t_value| 表示趋势更确定，训练时贡献更大 |
| 脚本位置 | `03_trend_scanning.py` | 独立脚本，清晰的流程分离 |

## 数据流

```
Phase 2 输出:
  → output/features/fracdiff_series.parquet (FracDiff 平稳化序列)
  → output/features/cusum_events.parquet (CUSUM 事件点 DatetimeIndex)

Phase 3 输入:
  fracdiff_series.parquet (价格序列)
  cusum_events.parquet (事件点)

Phase 3 处理:
  trend_scan_labels(price_series, cusum_events, L_windows)

Phase 3 输出:
  → output/features/trend_labels.parquet
```

## 输出格式

DataFrame indexed by CUSUM event timestamps:

| 列 | 类型 | 说明 |
|----|------|------|
| `t1` | Datetime | 趋势窗口结束时间（最优 L 的终点） |
| `t_value` | float64 | OLS 斜率 t 统计量（用于样本权重） |
| `side` | int8 | +1（上涨趋势）/-1（下跌趋势） |

## 新增配置参数

`strategies/IF9999/config.py` 新增:

```python
# ============================================================
# Phase 3: Trend Scanning 参数
# ============================================================

# Trend Scanning 窗口范围（Bars 数量）
TREND_WINDOWS = [5, 10, 20, 30, 50]  # 短期趋势，适合期货
```

## 脚本结构

`03_trend_scanning.py`:

```python
# 1. 加载配置
# 2. 读取 Phase 2 输出（fracdiff_series, cusum_events）
# 3. 调用 trend_scan_labels()
# 4. 保存 trend_labels.parquet
# 5. 生成诊断可视化：
#    - side 分布统计
#    - t_value 分布直方图
#    - 事件时间序列示例（标注趋势窗口）
```

## 可视化输出

| 文件 | 说明 |
|------|------|
| `03_trend_distribution.png` | side 和 t_value 分布 |
| `03_trend_example.png` | 示例事件的趋势窗口可视化 |

## 与现有模块的关系

- **依赖**: `afmlkit.feature.core.trend_scan.trend_scan_labels()`（已实现）
- **无需新架构**: Trend Scanning 核心已在 feature 模块，Phase 3 直接调用

## 后续阶段

Phase 4: Meta-Labeling
- 输入: trend_labels（side 作为 Primary Model 输出）
- 处理: TBM 标签 + Meta Model 训练
- 输出: bet size 信号