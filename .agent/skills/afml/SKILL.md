---
name: afml
description: Industrial-grade workflow for developing, testing, and verifying financial ML strategies based on "Advances in Financial Machine Learning" (AFML) and "Machine Learning for Asset Managers" (MLAM) by Marcos López de Prado. Use when: (1) Building Dollar/Volume/Imbalance bars, (2) Triple-barrier or meta-labeling, (3) Sample weights and uniqueness, (4) Purged/embargoed cross-validation, (5) Feature importance (MDI/MDA/Clustered MDA), (6) Trend scanning, (7) Backtest verification (DSR/PSR/CPCV), (8) HRP portfolios, (9) Fractional differentiation, (10) CUSUM filtering, (11) Market microstructure analysis. Includes causal verification framework with validation metrics and book references. For code-level implementation, see afmlkit skill.
---

# Advances in Financial Machine Learning

Industrial-grade workflow for quantitative finance strategy development.

## Core Philosophy

1.  **Stationarity is Non-Negotiable**: Financial data must be stationary (memory-preserving FracDiff) before any modeling.
2.  **No Peeking**: Strict prevention of look-ahead bias through Purged K-Fold Cross-Validation.
3.  **Honest Backtesting**: Use Deflated Sharpe Ratio (DSR) to account for multiple testing and selection bias.
4.  **Meta-Labeling**: Separate the decision of "side" (long/short) from "size" (bet sizing).

---

# Part I: Causal Verification Framework

> **终极验证框架：因果链 + 问题 + 验证指标**
>
> 每一步都按照**因果关系**串联：前一步如何影响后一步。这是开发量化策略的"超级检查清单"。

---

## 一、数据采样层（核心：消除时间扭曲与捕捉突变）

### 因果逻辑

> 传统的按时间采样（Time Bars）会导致高频交易时段和低频时段的信息量严重不均。
> **如果源头数据就是被扭曲的，后面所有的机器学习模型都会学到错误的规律。**

---

### 1. Dollar Bars / Imbalance Bars

**目标**：消除时间的非均匀性，捕捉买卖力量失衡触发的"信息事件"

**验证指标**：

| 指标 | 目标 | 说明 |
|------|------|------|
| Jarque-Bera (JB) 检验 | 正态性提升 | 越接近正态越好 |
| 自相关函数 (ACF) | 低自相关 | AC1 ≈ 0 最优 |
| Ljung-Box | 无序列相关 | p > 0.05 |
| Hurst Exponent | 接近 0.5 | 随机游走 |

**核心逻辑**：

```
时间 bars → clustering（聚集）
Dollar bars → 信息均匀 → 更接近白噪声
```

**最佳实践**：
- **Independence First**: 优先考虑低自相关，而非 JB 绝对值
- **Sample Size Sensitivity**: JB 统计量随 N 增大，高频数据高 JB 正常
- **Optimal Frequency**: 高交易量资产约 20-50 bars/day

---

### 2. Event-Based Sampling (CUSUM Filter)

**目标**：市场并非一直在提供有用信息。CUSUM 用于检测累计偏离目标的结构性突变，只有当市场真正发生"异动"时才采样

**验证指标**：

| 指标 | 目标 |
|------|------|
| 过滤后样本波动率 | 更加平稳 |
| 事件触发频率 | 合理（非过度密集/稀疏） |

---

## 二、标签层（核心：定义真实可交易的事件）

### 因果逻辑

> 如果强行规定"3天后卖出"，这打破了市场自我演化的因果性。
> **我们需要让数据自己告诉我们"趋势何时结束"。**

---

### 3. Triple Barrier Method (TBM)

**目标**：结合动态止盈、止损和最大持仓时间，让标签反映真实的路径依赖交易结果

**验证指标**：

| 指标 | 目标 |
|------|------|
| 标签分布（±1 / 0） | 是否均衡 |
| 每类样本数量 | 避免极端不平衡 |
| 平均持仓时间 | 合理性检查 |
| Hit Ratio | 统计显著性 |
| Payoff Asymmetry | 风险收益比 |

**配置**：
- Upper barrier（止盈）
- Lower barrier（止损）
- Vertical barrier（时间到期）

---

### 4. Trend Scanning

**目标**：放弃固定时间墙，通过在不同时间窗口内寻找最大 t-value，找出最显著的趋势段

**验证指标**：

| 指标 | 说明 |
|------|------|
| t-value 最大化 | 选择最显著窗口 |
| trend duration 分布 | 趋势长度分布 |
| label stability | 标签稳定性 |

**优势**：
- 无超参数敏感性：自动选择最优窗口
- 白盒逻辑：纯 OLS，完全可解释
- 双输出：side + confidence

---

### 5. Meta-labeling

**目标**：主模型找方向（买/卖），元模型决定是否下注（头寸大小）。将"寻找机会"和"过滤风险"解耦

**验证指标**：

| 指标 | 目标 | 说明 |
|------|------|------|
| Precision | ↑ | 精确率提升 |
| Recall | 略降但不能崩 | 召回率保持 |
| F1-score | ↑ | **金融数据极度不平衡，Accuracy 会骗人，F1 才能真实反映模型找准机会的能力** |
| 策略 Sharpe | ↑ | 风险调整收益 |
| 最大回撤 | ↓ | 风险控制 |

**本质**：

```
primary：找机会
meta：筛机会
```

---

## 三、特征层（核心：保留记忆与剔除噪音）

### 因果逻辑

> 如果直接用价格，数据不平稳，模型会崩溃；
> 如果用简单收益率，数据失去了所有长期记忆（过去的趋势），模型就变成了瞎猜。
> **同时，多余的噪音特征会引发过拟合。**

---

### 6. Fractional Differentiation (FracDiff)

**目标**：在"平稳性（Stationarity）"和"记忆保留（Memory）"之间找到最佳的因果平衡点 $d^*$

**验证指标**：

| 指标 | 目标 |
|------|------|
| ADF test | p < 0.05（刚好通过 5%） |
| KPSS test | 平稳性确认 |
| 与原序列相关性 | 接近 1（最大化信息保留） |
| 自相关 | 不能完全消失 |

**核心**：

```
一阶差分 ❌（信息丢失）
frac diff ✅（保留 alpha）
```

**执行陷阱**：
- FracDiff 会压缩方差，需注意 CUSUM 阈值的"量纲错位"
- 解决方案：确保 CUSUM 输入与阈值在同一尺度空间

---

### 7. Feature Importance (MDI/MDA)

**目标**：找出真正对预测有因果贡献的特征，剔除伪相关或被替代效应掩盖的噪音特征

> ⚠️ **极易被忽略的致命点**

**验证指标**：

| 指标 | 说明 |
|------|------|
| MDI | 基于树分裂的不纯度下降 |
| MDA | 样本外打乱特征后的准确率下降程度 |
| Clustered MDA | 解决共线性特征的替代效应 |

**为什么不用 Vanilla MDA**：
- 金融特征高度共线性
- Vanilla MDA 存在"替代效应"——重要特征因被替代而得分低

**步骤**：
1. **Feature Clustering**: 距离矩阵 $D = \sqrt{0.5 \times (1 - \rho)}$ → 层次聚类
2. **Clustered MDA**: 对整个 cluster 置换，测量 log-loss 下降

---

### 8. PCA

**目标**：消除金融特征之间极高的共线性

**验证指标**：

| 指标 | 说明 |
|------|------|
| explained variance ratio | 解释方差比例 |
| eigenvalue decay | 特征值衰减率 |
| condition number | 条件数 |

---

## 四、样本权重与数据清洗（核心：打破样本重叠导致的非独立性）

### 因果逻辑

> 如果两个标签对应的时间段重叠了，它们就包含了相同的未来信息（非独立同分布）。
> **如果在交叉验证时不清洗它们，模型就会发生严重的"信息泄露（Data Leakage）"，导致回测完美、实盘破产。**

---

### 9. Label Uniqueness & Sample Weights

**目标**：计算每个样本在时间上的重叠度，重叠越多的样本权重越低，让重要且独立的样本主导训练

**验证指标**：

| 指标 | 目标 |
|------|------|
| Uniqueness $\bar{u}_i$ | ∈ (0, 1) |
| 平均 uniqueness | > 0.5 理想 |
| overlap matrix | 可视化重叠 |

---

### 10. Sequential Bootstrap

**目标**：在随机抽样时，动态降低已抽出重叠样本被再次抽中的概率

**验证指标**：

| 指标 | 目标 |
|------|------|
| Bootstrap 后平均 Uniqueness | 显著提升 |
| 模型稳定性 | ↑ |
| Variance | ↓ |

---

### 11. Purging & Embargo

**目标**：在划分训练集/测试集时，删除与测试集在时间上重叠的训练样本（Purging），并在测试集后留出一段空白期（Embargo）以应对序列相关性

**验证指标**：

| 指标 | 目标 |
|------|------|
| Train/Test Overlap | **严格等于 0** |
| performance drop | 合理下降 |
| out-of-sample 稳定性 | ↑ |

---

## 五、模型层（核心：非线性与集成的力量）

### 因果逻辑

> 金融市场极度复杂且信噪比极低，
> **线性模型无法捕捉非线性交互，而单棵决策树极易过拟合。**

---

### 12. Ensemble Methods (Random Forest / Bagging)

**目标**：通过对特征和样本的随机抽样生成大量弱分类器，再进行投票，大幅降低方差（Variance）

**验证指标**：

| 指标 | 说明 |
|------|------|
| Out-of-Bag (OOB) Error | 袋外误差 |
| Recall | 召回率（关键指标） |
| ROC-AUC | 整体判别能力 |

**执行陷阱**：
- 对于特征重要性，**必须设置 `max_features=1`**
- 避免 dominant features "masking" 弱信号

---

### 13. High Recall Model

**目标**：不错过交易机会（Recall 优先）

**验证指标**：

| 指标 | 目标 |
|------|------|
| Recall | ↑（关键） |
| False Negative | ↓ |
| ROC-AUC | 监控 |
| Precision-Recall curve | 分析 |

---

## 六、策略与回测评估层（核心：防范回测过拟合）

### 因果逻辑

> 只要你尝试的参数组合足够多（Multiple Testing），
> 你总能"偶然"撞上一个夏普比率极高的完美策略。
> **我们必须用统计学把这种水分挤干。**

---

### 14. PSR (Probabilistic Sharpe Ratio)

**目标**：针对收益率的非正态性（负偏度、厚尾）和样本长度进行惩罚，计算真实 Sharpe 大于目标的概率

**验证指标**：

| 指标 | 目标 |
|------|------|
| PSR | **> 95%** |
| 收益偏度 | 输入 |
| 收益峰度 | 输入 |
| 样本长度 | 输入 |

---

### 15. DSR (Deflated Sharpe Ratio)

**目标**：惩罚过度参数寻优。你回测的次数越多，DSR 要求的达标门槛就越高

**验证指标**：

| 指标 | 目标 |
|------|------|
| DSR Probability | **> 95%** |
| 试验次数 | 记录 |
| 收益偏度/峰度 | 输入 |
| 回测长度 | 输入 |

**原则**：拒绝高 Sharpe 但低 DSR 的策略（幸运结果）

---

### 16. CPCV (Combinatorial Purged Cross-Validation)

**目标**：生成多条非重叠的历史演化路径，得出一个"Sharpe Ratio 分布"，而不是单一的虚假点估计

**验证指标**：

| 指标 | 说明 |
|------|------|
| Sharpe 分布均值 | OOS 性能 |
| PBO（回测过拟合概率） | 越低越好 |
| Sharpe 分布方差 | 稳定性 |

---

### 17. 策略综合评估

**目标**：是否真实可交易

**验证指标**：

| 指标 | 说明 |
|------|------|
| Sharpe Ratio | 风险调整收益 |
| Sortino Ratio | 下行风险调整 |
| Maximum Drawdown | 最大回撤 |
| Calmar Ratio | 年化收益/最大回撤 |
| Turnover | 换手率 |
| Capacity | 容量 |

---

# Part II: Workflow Decision Tree

> 操作级别的执行指南

## Phase 1: Data Engineering

**Goal**: Transform raw market data into information-rich, stationary features.

1. **Sampling**:
   * **Check**: Are you using Time Bars?
   * **Action**: If YES -> STOP. Switch to **Dollar Bars** or **Volume Bars**.
   * **Event Trigger**: Apply **CUSUM Filter** to trigger sampling.
   * **Frequency Selection**: Prioritize **Low Autocorrelation** over JB absolute value. Aim for 20-50 bars/day.

2. **Stationarity**:
   * **Check**: Run ADF test. Is p-value < 0.05?
   * **Action**: If NO -> Apply **Fractional Differentiation (FracDiff)**. Find minimum `d` such that p < 0.05 while maximizing memory preservation. *Never use integer differencing (d=1).*
   * **🚨 Pitfall (Variance Collapse)**: FracDiff compresses variance. Ensure CUSUM threshold matches the compressed scale.

## Phase 1.5: Primary Model (Side Determination)

**Goal**: Determine the directional bias (Long/Short) for each CUSUM event using a statistically rigorous, parameter-free method.

1. **Method**: **Trend Scanning** (MLAM Ch.3.5).
   * For each event, backward-scan over multiple window lengths.
   * Fit OLS regression in each window, compute t-statistic.
   * Select the window $L^*$ that maximizes $|t_{\text{value}}|$.
   * **Output**: `side = sign(t_value)` (+1 = Long, -1 = Short) and `|t_value|` as confidence score.

2. **Why Trend Scan over Fixed-Window Momentum?**
   * **No hyperparameter sensitivity**: Auto-selects optimal window.
   * **White-box logic**: Pure OLS, fully interpretable.
   * **Dual output**: Side + confidence feeds into Meta-Labeling weights.

3. **🚨 Anti-Pattern**: Do NOT use dual moving average crossover or fixed-period momentum. These exhibit catastrophic parameter fragility.

## Phase 2: Labeling & Weighting

**Goal**: Scientifically define "success" and handle data overlap.

1. **Labeling**: Use **Triple-Barrier Method**. Set upper barrier (take profit), lower barrier (stop loss), and vertical barrier (time expiration).

2. **Sample Weights**: Calculate **Average Uniqueness**. Down-weight samples with low uniqueness.

## Phase 3: Modeling & Feature Selection

**Goal**: Train models that generalize, not memorize.

1. **Cross-Validation**: Use **Purged K-Fold CV** with **Embargo** period.

2. **Feature Importance**: Use **Clustered MDA**.
   * **Step 1**: Feature Clustering via distance matrix $D = \sqrt{0.5 \times (1 - \rho)}$
   * **Step 2**: Permute entire clusters, measure log-loss drop
   * **Pitfall**: Force `max_features=1` in tree-based models to avoid masking effects.

## Phase 4: Strategy & Verification

**Goal**: Deploy only strategies that are statistically significant.

1. **Architecture**: **Meta-Labeling**.
   * *Primary Model*: **Trend Scan** — determines side.
   * *Secondary (Meta) Model*: Determines confidence.

2. **Final Acceptance**:
   * **Metric**: **Deflated Sharpe Ratio (DSR)**.
   * **Threshold**: DSR Probability > 0.95.
   * *Reject strategies with high Sharpe but low DSR.*

---

# Part III: Quick Reference

## Bar Types (Chapter 2)

Use information-driven bars instead of time bars:

| Type | Trigger | Use Case |
|------|---------|----------|
| Tick bars | Every N ticks | High-frequency analysis |
| Volume bars | Every N shares | Volume-sensitive strategies |
| Dollar bars | Every $N traded | **Recommended for ML** |
| Imbalance bars | Volume imbalance | Order flow analysis |

## Feature Importance Methods (Chapter 8)

| Method | Source | Use Case |
|--------|--------|----------|
| MDI | In-sample impurity | Fast, biased toward high-cardinality |
| MDA | OOS permutation | Robust, detects generalization |
| SFI | Single-feature CV | Detects substitution effects |

## Hierarchical Risk Parity (Chapter 16)

Portfolio construction without matrix inversion:
1. Hierarchical clustering on correlation
2. Quasi-diagonalization (reorder by cluster)
3. Recursive bisection (inverse-variance weights)

## Best Practices Summary

### Data Preparation
1. Never use time bars—use volume/dollar bars
2. Apply CUSUM filter for sampling
3. Fractionally differentiate to preserve memory

### Modeling
1. Use triple-barrier labeling for path dependency
2. Apply meta-labeling to separate signal from sizing
3. Always purge and embargo cross-validation

### Backtesting
1. Deflate Sharpe ratio for multiple testing
2. Compute PBO to estimate overfitting
3. Validate with synthetic data

---

# References

## Core Documentation

- **[glossary.md](references/glossary.md)** - AFML term definitions (FracDiff, Triple-Barrier, etc.)
- **[implementation_guide.md](references/implementation_guide.md)** - Python code snippets and mlfinlab usage

## By Topic

**Data Structures & Labeling**:
- `references/part_1.md` - Data analysis fundamentals
- `references/3_form_labels_using_the_triple_barrier_method_with_symmetric.md` - Triple barrier implementation
- `references/1_sample_bars_using_the_cusum_filter_where_y_t_are_absolute_returns_and_h_.md` - CUSUM filtering

**Cross-Validation & Feature Importance**:
- `references/5_the_cv_must_be_purged_and_embargoed_for_the_reasons_explained_in.md` - Purged CV
- `references/1_masking_effects_take_place_when_some_features_are_systematically_ignored.md` - Feature masking

**Backtesting**:
- `references/part_3.md` - Backtesting methodology
- `references/part_3_backtesting.md` - Backtesting deep dive
- `references/1_survivorship_bias_using_as_investment_universe_the_current_one_hence.md` - Bias detection
- `references/chapter_15_understanding_strategy_risk.md` - Risk metrics

**High-Performance Computing**:
- `references/1_chapter_20_multiprocessing_and_vectorization.md` - Parallel processing

**Bibliography**:
- `references/index.md` - Complete structure
- `references/9_bibliography.md` - Academic references

## Related Libraries

- **mlfinlab**: Python implementation of book techniques
- **afmlkit**: Project-specific implementation (see project CLAUDE.md)
- **sklearn**: ML algorithms
- **pandas/numpy**: Data structures

---

**Source**: "Advances in Financial Machine Learning" by Marcos López de Prado (Wiley, 2018)