# Volume Distribution Factors — FeatureKit Integration Design

**Date:** 2026-04-05
**Status:** Approved

## 1. Overview

将 `strategies/AL9999/volume_distribution.py` 中的核心成交量分布因子集成到 `afmlkit.feature.core`，以 `Transform` 类的形式供 `FeatureKit` 流水线调用。

## 2. Factor Definitions

| ID | Transform 类名 | 核心计算逻辑 | 经济含义 |
|----|---|---|---|
| QIML0514 | `VolEntropyTransform` | 成交额分箱后计算 Shannon 熵 | 成交量分布均匀度（不稳定则熵高） |
| QIML0607 | `VolSkewTransform` | 5min 成交额占比的偏度 | 成交时间分布的偏斜方向 |
| QIML0618 | `VolKurtTransform` | 5min 成交额占比的峰度 | 成交时间分布的尾部厚度 |
| QIML0124 | `VolPeakTransform` | 超均值+1σ 的 bar 数占比 | 放量程度（趋势/知情交易信号） |
| QIML0116 | `VolDiffStdTransform` | (amount/count) 差分的标准化std | 局部成交量峰值变化强度 |

> 注：QIML0618 作为 QIML0607 的配对因子（偏度+峰度）一并加入，共 5 个。

## 3. Architecture

### 3.1 文件位置

```
afmlkit/feature/core/volume_distribution.py   # 新增
```

### 3.2 类层次

```
BaseTransform (afmlkit/feature/base.py)
└── SISOTransform  — 单输入单输出，每个因子一个类
    ├── VolEntropyTransform
    ├── VolSkewTransform
    ├── VolKurtTransform
    ├── VolPeakTransform
    └── VolDiffStdTransform
```

### 3.3 Transform 接口约定

```python
class VolEntropyTransform(SISOTransform):
    """QIML0514: 成交量不稳定性的分散程度（Shannon 熵）"""
    def __init__(self, frequency: str = 'H', n_bins: int = 5, output_name: str = 'vol_entropy'):
        ...

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        # df: OHLCV DataFrame，需包含 ['amount', 'count', 'close']
        # 返回: pd.Series，index = df.index
        ...

    def get_params(self) -> dict: ...
    def set_params(self, **params): ...
```

### 3.4 命名约定

| Transform 类 | output_name 默认值 |
|---|---|
| `VolEntropyTransform` | `vol_entropy` |
| `VolSkewTransform` | `vol_skew` |
| `VolKurtTransform` | `vol_kurt` |
| `VolPeakTransform` | `vol_peak` |
| `VolDiffStdTransform` | `vol_diff_std` |

## 4. Data Flow

```
OHLCV DataFrame (df)
    │
    ▼
Vol{Xxx}Transform._compute(df)
    │  使用 pandas groupby + resample + apply
    │  （保留原 volume_distribution.py 逻辑，不强行 Numba 化）
    ▼
pd.Series (因子值，index = df.index)
    │
    ▼
FeatureKit pipeline 拼接 / 滚动窗口聚合
```

### 输入列要求

| 列名 | 类型 | 说明 |
|---|---|---|
| `code` | str | 品种代码（用于 groupby） |
| `amount` | float | 成交额 |
| `count` | int | 成交笔数 |
| `close` | float | 收盘价 |
| `open`, `high`, `low` | float | OHLC |

### 输出

`pd.Series`，index 与输入 `df` 对齐（按 code + time resample 后的 index）。

## 5. Key Implementation Notes

### 5.1 保留原始计算逻辑

这些因子的核心价值在于 groupby + resample + apply 的语义（按品种分组后按时间窗口聚合），不适合强行改造为纯 Numba JIT 计算。保持 pandas 模式，但用 Transform 封装为统一接口。

### 5.2 参数化 frequency

原实现硬编码 `frequency='H'`，封装时改为构造函数参数，支持 `'5min'`, `'15min'`, `'H'`, `'D'` 等。

### 5.3 滚动平滑

部分因子原实现返回 `factors` 和 `factors_mean`（24 周期 rolling mean）。Transform 默认只返回原始因子值；如需平滑，用户通过 `FeatureKit` 流水线叠加 `RollingMeanTransform`。

### 5.4 与现有 volume.py 的关系

`afmlkit/feature/core/volume.py` 包含 Volume Profile（POC/HVA/LVA）等 Numba 加速的成交量特征。新文件 `volume_distribution.py` 专注于成交量**分布统计**特征（熵、偏度、峰度等），职责不重叠。

## 6. Export

在 `afmlkit/feature/core/__init__.py` 中导出所有新 Transform：
```python
from .volume_distribution import (
    VolEntropyTransform,
    VolSkewTransform,
    VolKurtTransform,
    VolPeakTransform,
    VolDiffStdTransform,
)
```

在 `afmlkit/feature/kit.py` 中从 `.core` 导入并加入 `__all__`。

## 7. Testing Strategy

| 测试文件 | 内容 |
|---|---|
| `tests/feature/test_volume_distribution.py` | 单元测试：每个 Transform 与原 `volume_distribution.py` 结果比对 |

测试数据：使用 `strategies/AL9999/` 目录下的历史 OHLCV 数据片段（pickle/csv）作为 fixture，对齐 `NUMBA_DISABLE_JIT=1` 环境。

## 8. Files to Modify

| 文件 | 操作 |
|---|---|
| `afmlkit/feature/core/volume_distribution.py` | 新增 |
| `afmlkit/feature/core/__init__.py` | 添加 import/export |
| `afmlkit/feature/kit.py` | 添加导入到 `__all__` |
| `tests/feature/test_volume_distribution.py` | 新增 |

## 9. Out of Scope

- 剩余 15+ 个未集成的因子（QIML0212, QIML0301, QIML0401 等）
- Numba JIT 加速改造
- FeatureKit 流水线封装逻辑修改
