# Meta-Model v2 训练报告

## 1. 数据概览

| 指标 | 值 |
|------|-----|
| 样本数量 | 4700 |
| 特征数量 | 14 |
| 标签分布 | {0: 2430, 1: 2270} |
| 平均唯一性 | 0.5000 |

## 2. 特征列表 (14 个)

1. `log_return`
2. `vol_atr_14`
3. `vol_rel_20`
4. `vol_ewm_100`
5. `vol_parkinson`
6. `ema_diff`
7. `trend_variance_ratio_20`
8. `corr_pv_10`
9. `liq_amihud`
10. `vol_garman_klass`
11. `vol_yang_zhang`
12. `mom_macd`
13. `vol_kyle_lambda`
14. `entropy_shannon`

## 3. CV 性能指标

| 指标 | 值 | 评价 |
|------|-----|------|
| ROC AUC | 0.5168 | ✗ 需改进 |
| Accuracy | 0.5051 | ✗ 需改进 |
| Precision | 0.4912 | - |
| Recall | 0.6907 | - |
| F1 Score | 0.5741 | - |
| Brier Score | 0.2524 | ✗ 需改进 |

## 4. 全样本性能指标

| 指标 | 值 |
|------|-----|
| ROC AUC | 0.6195 |
| Accuracy | 0.5679 |
| Brier Score | 0.2400 |

## 5. 配置参数

| 参数 | 值 |
|------|-----|
| N Estimators | 1000 |
| CV Splits | 5 |
| Embargo | 1.0% |
| 概率校准 | Isotonic |

## 6. 改进对比

| 指标 | v1 (旧) | v2 (新) | 改进 |
|------|---------|---------|------|
| ROC AUC | ~0.52 | 0.5168 | -0.0032 |
| 概率范围 | 0.41 | - | - |
| Brier Score | ~0.25 | 0.2524 | -0.0024 |

## 7. 后续行动

- [ ] 检查概率分布是否改善
- [ ] 验证校准效果
- [ ] 运行 Bet Size 分析
- [ ] 回测验证策略表现

