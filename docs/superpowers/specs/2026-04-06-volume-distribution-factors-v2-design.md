# Volume Distribution Factors — Remaining 17 Factors Design

**Date:** 2026-04-06
**Status:** Approved

## Overview

继续将 `strategies/AL9999/volume_distribution.py` 中的剩余 17 个因子集成到 `afmlkit.feature.core.volume_distribution`，与已实现的 5 个因子合并为完整的成交量分布特征库。

## Remaining Factors

| ID | Transform 类名 | 核心计算逻辑 | 经济含义 | 备注 |
|----|---|---|---|---|
| QIML0212 | `AmbiguityTransform` | amb=rolling(5d std)^2; fogging=ret[amb>amb.mean]; return fogging_amt_ratio - fogging_vol_ratio | 厌恶模糊性 | 截面计算，单品种退化 |
| QIML0301 | `AmbiguityVolTransform` | 同 amb 但用 volume 均值比 | 模糊厌恶 | |
| QIML0331 | `AmbiguityCountTransform` | 同 amb 但用 count 均值比 | 模糊厌恶 | |
| QIML0401 | `PShapeTransform` | 找 50% 累计成交价的下界，计算到高点的距离 | Volume Profile P 型 | |
| QIML0413 | `BShapeTransform` | 找 50% 累计成交价的下界，计算到低点的距离 | Volume Profile B 型 | |
| QIML0503 | `PShapeDiffTransform` | 最低成交价到收盘价的距离/最低价 | 价格位置 | |
| QIML0629 | `UnanimousBuyingTransform` | (vol_up + vol_down) / (vol_up - vol_down)，alpha>0.5 的 bar | 一致买入强度 | |
| QIML0722 | `VolStdTransform` | 5min 成交额占比的标准差 | 成交量分散度 | |
| QIML0806 | `VolumeVarTransform` | 成交额方差 | 成交量波动性 | |
| QIML0827 | `UnanimousTradingTransform` | alpha>0.5 的 bar 的成交额占比 | 一致交易强度 | |
| QIML0914 | `TailVolumeRatioTransform` | 最后 5 分钟 volume / 总 volume | 尾盘占比 | |
| QIML1014 | `VolumeRatioTransform` | 开盘前5min/最后5min 成交额比 | 开盘尾盘比 | |
| QIML1021 | `VolumeShareTransform` | 开盘前5min + 最后5min 成交额占比 | 首尾占比 | |
| QIML1105 | `AmountQuantileTransform` | 单笔成交额的分位数位置 | 交易规模 | |
| QIML1113 | `TradingIntensityTransform` | vol/amt 与 vol 的 spearman 相关 | 交易强度 | |
| QIML1215 | `TailVolumeRatioVTransform` | 同 QIML0914 但用 volume 非 amount | 尾盘量占比 | |
| QIML1222 | `TVDTransform` | amount/amount.sum() 的标准差 | 换手率均匀度 | |

## Architecture

### File

```
afmlkit/feature/core/volume_distribution.py  # 追加 17 个 Transform
```

### 类层次

```
BaseTransform
└── SISOTransform
    ├── VolEntropyTransform (existing)
    ├── VolSkewTransform (existing)
    ├── VolKurtTransform (existing)
    ├── VolPeakTransform (existing)
    ├── VolDiffStdTransform (existing)
    ├── AmbiguityTransform (new)
    ├── AmbiguityVolTransform (new)
    ├── AmbiguityCountTransform (new)
    ├── PShapeTransform (new)
    ├── BShapeTransform (new)
    ├── PShapeDiffTransform (new)
    ├── UnanimousBuyingTransform (new)
    ├── VolStdTransform (new)
    ├── VolumeVarTransform (new)
    ├── UnanimousTradingTransform (new)
    ├── TailVolumeRatioTransform (new)
    ├── VolumeRatioTransform (new)
    ├── VolumeShareTransform (new)
    ├── AmountQuantileTransform (new)
    ├── TradingIntensityTransform (new)
    ├── TailVolumeRatioVTransform (new)
    └── TVDTransform (new)
```

## Data Flow

- 输入：`pd.DataFrame` 含 `code`, `amount`, `count`, `close`, `open`, `high`, `low`, `volume`
- 输出：`pd.Series`，index 与输入对齐
- 参数化 `frequency`（默认 `'H'`）
- 命名规范：`feat_vol_{因子名}`（如 `feat_vol_unanimous_buying`）

## QIML0212 特殊处理

QIML0212 需要截面计算（同一时间点所有品种归一化），单品种数据退化。

**处理方案：** 在 Transform 中检测是否单品种，若是则计算"历史 amb ratio 的滚动均值"替代截面均值。即：先用时间维度近似截面语义。

## Testing

- 新增 `tests/features/test_volume_distribution.py` 中每个因子一个测试类
- 使用合成 DataFrame fixture 验证输出形状和有效性
- 与 `strategies/AL9999/volume_distribution.py` 原始实现的数值结果比对

## Files to Modify

| 文件 | 操作 |
|---|---|
| `afmlkit/feature/core/volume_distribution.py` | 追加 17 个 Transform |
| `afmlkit/feature/core/__init__.py` | 追加导出 |
| `tests/features/test_volume_distribution.py` | 追加测试 |

## Out of Scope

- Numba JIT 加速（分布统计语义不适合）
- 与现有 `volume.py` 的合并（职责不重叠）
