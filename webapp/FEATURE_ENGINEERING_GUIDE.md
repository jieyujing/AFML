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
