# WebUI 特征工程指南

## 概述

WebUI 特征工程模块实现了与 `scripts/feature_engineering.py` 一致的完整 AFML 特征计算流程。

## 特征列表

### 波动率特征
- `vol_ewm_{span}` - 多窗口 EWM 波动率
- `vol_parkinson` - Parkinson 波动率
- `vol_atr_14` - 平均真实波动幅度
- `vol_bb_pct_b_20` - Bollinger %B
- `trend_variance_ratio_20` - 方差比率

### 动量特征
- `ema_short` / `ema_long` - 短/长期 EMA
- `log_dist_ema_short` / `log_dist_ema_long` - 对数价格距离 EMA
- `ema_diff` - EMA 差值
- `rsi_14` - 相对强弱指标
- `mom_roc_{period}` - 变化率
- `mom_stoch_k_14` - 随机指标 %K
- `corr_pv_10` - 价量相关性

### 流动性特征
- `liq_amihud` - Amihud 非流动性
- `vol_rel_20` - 相对成交量

### 分数阶差分
- `ffd_log_price` - 分数阶差分对数价格 (自动优化 d 参数)

## Alpha158 特征（FFD 改造版）

### 底座序列
- `ffd_log_price` - 分数阶差分对数价格（自动优化 d* 参数）

### 波动率特征
- `ffd_vol_std_{span}` - 滚动标准差
- `ffd_vol_ewm_{span}` - EWM 波动率

### 均线特征
- `ffd_ma_{window}` - 简单移动平均
- `ffd_ema_{window}` - 指数移动平均

### 动量特征
- `ffd_mom` - FFD 序列本身（替代传统收益率动量）

### 量价特征
- `ffd_vwap` - 成交量加权平均价
- `ffd_amount` - 成交金额
- `ffd_amplification` - 振幅

### 时序排序特征
- `ffd_rank_{feature}_{window}` - 滚动百分位排序（单股票时序历史）

## Alpha158 配置示例

```yaml
alpha158:
  enabled: true
  volatility:
    spans: [5, 10, 20]
  ma:
    windows: [5, 10, 20]
  rank:
    enabled: true
    window: 20
```

## 使用流程

1. **数据准备**: 确保已生成 Dollar Bars 数据
2. **特征配置**: 选择特征类别和参数
3. **CUSUM 对齐**: 可选，将特征对齐到事件时间戳
4. **特征计算**: 执行计算并查看摘要
5. **特征预览**: 查看统计、相关性、分布
6. **特征导出**: 导出为 CSV/Parquet/HDF5

## CUSUM 对齐

启用 CUSUM 对齐后，特征将与 `outputs/dollar_bars/cusum_sampled_bars.csv` 中的事件时间戳对齐。

如果 CUSUM 文件不存在，系统将自动降级为连续特征模式。

## 配置示例

```yaml
volatility:
  spans: [10, 50, 100]
momentum:
  rsi_window: 14
  roc_period: 10
fractional_diff:
  enabled: true
  threshold: 0.0001
  d_step: 0.05
cusum:
  align_enabled: true
  path: outputs/dollar_bars/cusum_sampled_bars.csv
```

## API 参考

### compute_all_features

```python
def compute_all_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    cusum_path: Optional[str] = None,
    align_to_cusum: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    计算所有特征的入口函数

    Args:
        df: 输入 DataFrame (必须包含 close 列，可选 high/low/volume)
        config: 特征配置字典
        cusum_path: CUSUM 采样文件路径（如果 align_to_cusum=True）
        align_to_cusum: 是否对齐到 CUSUM 事件

    Returns:
        tuple: (特征矩阵 DataFrame, 元数据字典)
    """
```

### 元数据字段

- `optimal_d`: FFD 最优参数 d
- `aligned_to_cusum`: 是否对齐到 CUSUM 事件
- `rows_before_clean`: 清理前的行数
- `rows_after_clean`: 清理后的行数
- `rows_dropped`: 丢弃的行数
- `final_shape`: 最终形状
- `feature_columns`: 特征列列表
- `label_distribution`: 标签分布（如果有）

---

## Clustered MDA 特征重要性分析

### 为什么使用 Clustered MDA？

传统 MDA（单个特征打乱）存在**替代效应**问题：当两个特征高度相关时，打乱其中一个，另一个可以替代它，导致重要性被低估。

Clustered MDA 的解决方案：
1. 使用 ONC 算法自动识别特征簇
2. 同时打乱整个簇的所有特征
3. 使用 Purged CV 计算样本外 Log-loss 变化

### 解读结果

**条形图：**
- 蓝色条形：重要性 > 0，特征有价值
- 红色条形：重要性 ≤ 0，"毒药"簇

**误差线：**
- 红色误差线表示标准差
- 误差线越短，重要性估计越稳定

**毒药簇警告：**
如果发现红色簇（重要性 ≤ 0），说明这些特征在打乱后模型表现反而变好。这通常意味着：
- 特征包含严重噪音
- 特征导致模型过拟合
- 建议：直接从特征池中移除

### 使用示例

```python
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda

# Step 1: 特征聚类
clusters = cluster_features(X_features)

# Step 2: 计算 Clustered MDA
mda_df = clustered_mda(
    X=X_features,
    y=y_labels,
    clusters=clusters,
    t1=t1_events,
    n_splits=5,
    embargo_pct=0.01,
    n_repeats=10
)

# Step 3: 识别毒药簇
poison = mda_df[mda_df['mean_importance'] <= 0]
print(f"需要移除的簇：{poison['cluster_id'].tolist()}")
```

### WebUI 操作流程

1. 进入"特征分析"页面
2. 选择"3. 特征重要性"步骤
3. 选择"Clustered MDA (推荐)"模式
4. 配置参数：
   - 重复次数：建议 10-20 次
   - CV 折数：建议 5 折
   - Embargo 比例：建议 0.01 (1%)
5. 点击"计算 Clustered MDA"
6. 查看聚类结果和重要性图表
7. 关注毒药簇警告，考虑移除相关特征
