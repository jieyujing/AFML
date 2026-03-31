# Feature Stationarity Analysis & FracDiff Design

## Overview

创建独立脚本对全量特征文件 (`bars_features.parquet`) 进行 ADF 平稳性分析，对非平稳特征应用 FracDiff 处理，输出处理后的特征文件和 ADF 分析报告。

## Problem Statement

当前 `02_feature_engineering.py` 生成的特征文件包含 30+ 列特征，部分特征（如 RSI、ADX 等）可能非平稳，影响后续机器学习模型的效果。需要一个自动化流程：

1. 检测每个特征的平稳性
2. 对非平稳特征应用 FracDiff 使其平稳
3. 保留处理记录供分析

## Design

### Architecture

```
05_feature_stationarity.py (独立脚本)
    ├── 加载 bars_features.parquet
    ├── classify_features() → 分为候选/排除两类
    ├── process_feature_stationarity() → 单列处理核心逻辑
    ├── run_stationarity_analysis() → 批量处理
    └── 输出: ADF 报告 + 处理后特征文件
```

### Components

#### 1. Feature Classification (`classify_features`)

基于列名模式识别需要排除的特征：

| 排除类型 | 列名模式 | 原因 |
|----------|----------|------|
| 收益率类 | `*_roc_*`, `*_return*` | 天然平稳（已差分） |
| 时间特征 | `*_sin_*`, `*_cos_*`, `*_sess*` | 周期编码，无平稳概念 |
| 二值信号 | `*_sig_*`, `*_cross_ma_sig_*` | 离散值，不适用 FracDiff |

返回：
- `candidate_features`: 需要处理的特征列名列表
- `excluded_features`: 被排除的特征列名列表

#### 2. Single Feature Processing (`process_feature_stationarity`)

输入：单个特征 Series + FracDiff 参数

流程：
```
ADF 检验 → p-value < 0.05? 
    Yes → 返回原序列, d=0.0
    No  → optimize_d() → frac_diff_ffd() → 返回处理后序列, d值
```

输出：
- `processed_series`: 处理后的 Series
- `result_dict`: {p_value, optimal_d, is_stationary, is_excluded}

#### 3. Batch Processing (`run_stationarity_analysis`)

输入：`bars_features.parquet` 路径 + FracDiff 配置

流程：
```
加载特征文件
    ↓
classify_features()
    ↓
for each candidate feature:
    process_feature_stationarity()
    ↓
合并所有结果（排除特征保持原样）
    ↓
保存输出文件
```

### Data Flow

```
bars_features.parquet (输入)
    │
    ├─ excluded_features ──────────────────────┐
    │                                          │
    ├─ candidate_features                      │
    │   │                                      │
    │   ├─ stationary (p<0.05) ── 保持原样 ────┤
    │   │                                      │
    │   └─ non-stationary ── FracDiff ────────┤
    │                                          │
    └──────────────────────────────────────────┘
    │
bars_features_fd.parquet (输出)
```

### Output Files

| 文件 | 路径 | 内容 |
|------|------|------|
| ADF 报告 | `FEATURES_DIR/adf_report.csv` | 每列的 p_value, optimal_d, is_stationary, is_excluded |
| 处理后特征 | `FEATURES_DIR/bars_features_fd.parquet` | 原地替换后的特征 DataFrame |

### Configuration

使用 `config.py` 现有参数：
- `FRACDIFF_THRES = 1e-4`
- `FRACDIFF_D_STEP = 0.05`
- `FRACDIFF_MAX_D = 1.0`

### Dependencies

- `afmlkit.feature.core.frac_diff.optimize_d`
- `afmlkit.feature.core.frac_diff.frac_diff_ffd`
- `statsmodels.tsa.stattools.adfuller`
- `strategies.IF9999.config`

## Error Handling

1. **列名无法识别**：默认作为候选特征处理
2. **NaN 过多**：若有效数据少于 10 个，跳过该列并标记 `is_valid=False`
3. **FracDiff 失败**：返回原序列，标记 `fracdiff_success=False`

## Testing

手动验证：
1. 运行脚本生成输出
2. 检查 ADF 报告中 p-value 分布
3. 对比原始和处理后特征的统计特性（均值、方差、相关性）
4. 确认排除的特征未被修改

## Implementation Notes

- 脚本独立运行，不修改现有流程文件
- 函数设计可复用（未来可集成到其他策略）
- 添加进度显示（处理 30+ 列需要时间）