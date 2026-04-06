# AL9999 tsfresh 特征工程模块设计

## 概述

为 AL9999 策略新增独立的 tsfresh 特征提取模块，在 CUSUM 事件点上提取基于滚动窗口的 tsfresh 特征。

## 架构

```
输入层: close, log_close, volume, open_interest
         │
         ├── raw ──────────────→ tsfresh 16-func ──→ feat_{col}_raw_{func}
         ├── pct_change ───────→ tsfresh 16-func ──→ feat_{col}_pct_{func}
         ├── fracdiff ─────────→ tsfresh 16-func ──→ feat_{col}_fd_{func}  (仅 close, log_close)
         │
         └── zscore_raw ───────→ tsfresh 16-func ──→ feat_{col}_z{window}_{func}
                                                               │
                                                         windows: [10, 20, 40]

事件切片: 每个 CUSUM 事件点，回看 TSFRESH_LOOKBACK bars 提取特征
```

## 输入层

| 列名 | 说明 |
|------|------|
| `close` | 收盘价 |
| `log_close` | 对数收盘价 |
| `volume` | 成交量 |
| `open_interest` | 持仓量 |

## 变换层

### raw

原始序列，无变换。

### pct_change

百分比变化率：`pct_change(periods=1)`，应用于所有 4 列。

### fracdiff

分数差分（FFD），仅应用于 `close` 和 `log_close`。

- 复用 `FRACDIFF_THRES`（默认 `1e-4`）
- 调用 `optimize_d()` 自动搜索最优 d
- 不应用于 `volume` 和 `open_interest`（非平稳性处理意义不同）

### zscore_raw

滚动窗口标准化：`zscore = (x - rolling_mean) / rolling_std`

应用于所有 4 列，窗口：`windows=[10, 20, 40]`

## 事件切片

- 事件来源：`cusum_events.parquet` 中的事件点索引
- 回看窗口：`TSFRESH_LOOKBACK`（默认 20 bars）
- 对每个事件点，提取该事件前 `lookback` 个 bars 的数据作为 tsfresh 输入

## tsfresh 特征函数（16个）

```python
TSFRESH_FEATURES = [
    "mean",
    "median",
    "standard_deviation",
    "skewness",
    "kurtosis",
    "minimum",
    "maximum",
    "abs_energy",
    "mean_change",
    "mean_abs_change",
    "count_above_mean",
    "count_below_mean",
    "first_location_of_maximum",
    "first_location_of_minimum",
    "last_location_of_maximum",
    "last_location_of_minimum",
]
```

## 输出命名

| 变换 | 命名格式 | 示例 |
|------|----------|------|
| raw | `feat_{col}_raw_{func}` | `feat_close_raw_mean` |
| pct_change | `feat_{col}_pct_{func}` | `feat_volume_pct_skewness` |
| fracdiff | `feat_{col}_fd_{func}` | `feat_close_fd_kurtosis` |
| zscore | `feat_{col}_z{window}_{func}` | `feat_close_z20_mean` |

可选添加 `_lb{lookback}` 后缀（如 `feat_close_raw_mean_lb20`），暂不启用。

## 输出文件

```
output/features/tsfresh_features.parquet
```

## 配置项（config.py）

```python
TSFRESH_CONFIG = {
    "enabled": True,
    "lookback": 20,                        # 回看 bars 数量
    "fracdiff_cols": ["close", "log_close"],  # 只对这些列做 fracdiff
    "zscore_windows": [10, 20, 40],
    "features": [
        "mean", "median", "standard_deviation", "skewness", "kurtosis",
        "minimum", "maximum", "abs_energy", "mean_change", "mean_abs_change",
        "count_above_mean", "count_below_mean",
        "first_location_of_maximum", "first_location_of_minimum",
        "last_location_of_maximum", "last_location_of_minimum",
    ],
}
```

## 实现文件

```
strategies/AL9999/02b_tsfresh_feature_engineering.py  # 新增独立脚本
```

## 依赖

- `tsfresh`：需添加到 `pyproject.toml` 依赖
- 复用 `afmlkit.feature.core.frac_diff` 中的 `optimize_d`, `frac_diff_ffd`

## 已知约束

- tsfresh 特征提取在较大数据集上可能较慢，建议设置合理的 `lookback` 值
- fracdiff 仅适用于价格类序列，不适用于 volume/open_interest
