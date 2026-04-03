# 特征工程重构分析报告

## 1. 数据概览

| 指标 | 值 |
|------|-----|
| 样本数量 | 4700 |
| 原始特征数 | 29 |  # 减去 label 相关列 |
| 高相关特征对 | 72 |
| 噪声特征数 | 0 |

## 2. 特征重要性分析

### MDI (Random Forest) Top 10

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | vol_ewm_100 | 0.039671 |
| 2 | vol_rel_20 | 0.039023 |
| 3 | vol_ewm_50 | 0.037655 |
| 4 | vol_atr_14 | 0.037042 |
| 5 | ema_diff | 0.036992 |
| 6 | close | 0.036972 |
| 7 | rsi_14 | 0.036829 |
| 8 | ema_short | 0.036542 |
| 9 | vwap | 0.036444 |
| 10 | volume | 0.036323 |

### SFI (Logistic Regression) Top 10

| 排名 | 特征 | AUC |
|------|------|-----|
| 1 | volume | 0.5325 |
| 2 | vol_atr_14 | 0.5324 |
| 3 | trades | 0.5315 |
| 4 | log_dist_ema_long | 0.5301 |
| 5 | rsi_14 | 0.5256 |
| 6 | mom_roc_10 | 0.5191 |
| 7 | vol_rel_20 | 0.5190 |
| 8 | vol_bb_pct_b_20 | 0.5187 |
| 9 | log_dist_ema_short | 0.5171 |
| 10 | mom_stoch_k_14 | 0.5157 |

## 3. 高相关特征对 (|corr| > 0.7)

| 特征 1 | 特征 2 | 相关系数 |
|--------|--------|----------|
| high | vwap | 1.000 |
| low | vwap | 1.000 |
| close | vwap | 1.000 |
| open | vwap | 1.000 |
| high | low | 1.000 |
| open | high | 1.000 |
| high | close | 1.000 |
| open | low | 1.000 |
| low | close | 1.000 |
| ema_short | ema_long | 1.000 |

## 4. 特征推荐

### 4.1 推荐保留的特征 (9 个)

- `corr_pv_10` (MDI=0.0354, SFI=0.4967)
- `ema_diff` (MDI=0.0370, SFI=0.4988)
- `liq_amihud` (MDI=0.0324, SFI=0.5000)
- `log_return` (MDI=0.0340, SFI=0.5021)
- `trend_variance_ratio_20` (MDI=0.0354, SFI=0.4722)
- `vol_atr_14` (MDI=0.0370, SFI=0.5324)
- `vol_ewm_100` (MDI=0.0397, SFI=0.5000)
- `vol_parkinson` (MDI=0.0348, SFI=0.5000)
- `vol_rel_20` (MDI=0.0390, SFI=0.5190)

### 4.2 推荐移除的噪声特征 (0 个)


### 4.3 推荐移除的冗余特征 (19 个)

- `high` (MDI=0.0348)
- `low` (MDI=0.0359)
- `vwap` (MDI=0.0364)
- `open` (MDI=0.0357)
- `ema_long` (MDI=0.0350)
- `ema_short` (MDI=0.0365)
- `vol_ewm_50` (MDI=0.0377)
- `log_dist_ema_long` (MDI=0.0354)
- `mom_stoch_k_14` (MDI=0.0351)
- `log_dist_ema_short` (MDI=0.0352)
- `ffd_log_price` (MDI=0.0336)
- `vol_bb_pct_b_20` (MDI=0.0354)
- `mom_roc_10` (MDI=0.0352)
- `vol_ewm_10` (MDI=0.0363)
- `median_trade_size` (MDI=0.0306)
- `volume` (MDI=0.0363)
- `rsi_14` (MDI=0.0368)
- `trades` (MDI=0.0363)
- `close` (MDI=0.0370)

### 4.4 建议添加的新特征 (5 个)

- `vol_garman_klass`
- `vol_yang_zhang`
- `mom_macd`
- `vol_kyle_lambda`
- `entropy_shannon`

## 5. 重构后的特征配置

### 推荐特征列表

```python
RECOMMENDED_FEATURES = [
    "corr_pv_10",
    "ema_diff",
    "liq_amihud",
    "log_return",
    "trend_variance_ratio_20",
    "vol_atr_14",
    "vol_ewm_100",
    "vol_parkinson",
    "vol_rel_20",
]
```

### 预期效果

- 特征数量：从 29 个减少到 14 个
- 去除共线性：移除 72 对高相关特征
- 去除噪声：移除 0 个低重要性特征
- 新增特征：添加 5 个增强特征


## 6. 后续行动

1. [ ] 审核推荐特征列表
2. [ ] 实现新增特征的计算逻辑
3. [ ] 重新运行特征工程管道
4. [ ] 重新训练 Meta Model
5. [ ] 验证改进效果
