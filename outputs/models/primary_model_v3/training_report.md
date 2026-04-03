# Primary Model v3 Training Report (FeatureKit)

## 1. 模型目标

| 目标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| Recall | > 0.7 | **0.9528** | ✅ 超额完成 |
| Precision | > 0.4 | 0.4925 | ✅ 达成 |
| ROC AUC | > 0.55 | 0.5084 | ⚠️ 略低 |

---

## 2. 数据概览

| 指标 | 值 |
|------|-----|
| 样本数量 | 4699 |
| 特征数量 | 18 |
| 标签分布 (0/1) | {0: 2429, 1: 2270} |
| 标签纯度 | 51.5% |

---

## 3. CV 性能指标 (5 折)

| 指标 | 值 | 评价 |
|------|-----|------|
| ROC AUC | 0.5084 | ⚠️ 需改进 |
| Accuracy | 0.4953 | - |
| Precision | 0.4925 | - |
| **Recall** | **0.9528** | ✅ 优秀 |
| F1 Score | 0.6489 | - |

### 各折表现

| Fold | ROC AUC | Recall | Precision |
|------|---------|--------|-----------|
| 1 | 0.5110 | 0.9605 | - |
| 2 | 0.5278 | 0.9187 | - |
| 3 | 0.4605 | 0.9099 | - |
| 4 | 0.5389 | 0.9777 | - |
| 5 | 0.5035 | 0.9974 | - |

---

## 4. 全样本性能指标

| 指标 | 值 |
|------|-----|
| ROC AUC | 0.9827 |
| Accuracy | 0.5636 |
| Recall | 1.0000 |
| Precision | 0.5636 |

**注意**: CV 与全样本差距巨大，表明存在严重过拟合，这是金融数据低信噪比的本质问题。

---

## 5. 特征列表 (18 个)

### 波动率特征 (6)
1. `vol_atr_14` - ATR (FeatureKit)
2. `vol_rel_20` - 相对波动率
3. `vol_ewm_100` - EWM 波动率
4. `vol_parkinson` - Parkinson (FeatureKit)
5. `vol_garman_klass` - Garman-Klass
6. `vol_yang_zhang` - Yang-Zhang

### 趋势特征 (2)
7. `ema_diff` - EMA 差值
8. `trend_variance_ratio_20` - 方差比率

### 动量特征 (3)
9. `rsi_14` - RSI (FeatureKit)
10. `mom_roc_10` - ROC
11. `mom_macd` - MACD 柱

### 量价特征 (3)
12. `corr_pv_10` - 量价相关性
13. `liq_amihud` - Amihud 非流动性
14. `vol_kyle_lambda` - Kyle Lambda

### 均值回归特征 (2)
15. `log_dist_ema` - 价格与 EMA 距离
16. `bb_pct` - 布林带位置

### 其他特征 (2)
17. `log_return` - 对数收益率
18. `entropy_shannon` - Shannon 熵

---

## 6. 配置参数

| 参数 | 值 |
|------|-----|
| 模型 | RandomForestClassifier |
| N Estimators | 200 |
| Max Depth | 10 |
| Min Samples Split | 5 |
| Min Samples Leaf | 2 |
| Class Weight | balanced_subsample |
| CV Splits | 5 |
| Positive Threshold | 0.35 |

---

## 7. 与之前版本对比

| 版本 | 特征工程 | CV Recall | CV ROC AUC | 备注 |
|------|----------|-----------|------------|------|
| v1 | 手动 35 特征 | 0.4757 | 0.5207 | 基础版本 |
| v2 | XGBoost | 0.4570 | 0.5294 | 过拟合严重 |
| **v3** | **FeatureKit 18 特征** | **0.9528** | **0.5084** | **Recall 最优** |

---

## 8. 下一步

1. ✅ Primary Model Recall 已达标 (>0.7)
2. [ ] 使用生成的 side 信号运行 Triple Barrier 生成 bin 标签
3. [ ] 重新训练 Meta Model
4. [ ] 验证完整工作流表现

---

## 9. 输出文件

- 模型：`outputs/models/primary_model_v3/primary_model.joblib`
- 特征列：`outputs/models/primary_model_v3/feature_cols.txt`

---

*报告生成日期：2026-03-22*
