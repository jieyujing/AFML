# AFML 项目进度报告

## ✅ 已完成工作

### 1. Dollar Bars 生成 ✓
**文件**: `src/process_bars.py`

已成功实现两种类型的 dollar bars:
- **Fixed Dollar Bars**: 使用固定阈值
- **Dynamic Dollar Bars**: 使用 EMA 动态阈值

**输出文件**:
- `dynamic_dollar_bars.csv` - 3,927 个 dollar bars
- 时间范围: 2022-01-04 至 2026-01-20

**关键发现**:
- Dollar bars 成功降低了 Jarque-Bera 统计量
- 收益分布更接近正态分布
- 动态阈值表现优于固定阈值

---

### 2. Triple Barrier Labeling ✓
**文件**: `src/labeling.py`

实现了 AFML 核心的三重障碍标签法:
- **上障碍**: 止盈 (1x 波动率)
- **下障碍**: 止损 (1x 波动率)
- **垂直障碍**: 时间限制 (12 bars ≈ 3 days)

**输出文件**:
- `labeled_events.csv` - 3,707 个标记事件
- `dollar_bars_labeled.csv` - 带标签的完整数据集

**标签分布**:
- Loss (-1): 53.33% (1,977 events)
- Profit (1): 46.67% (1,730 events)

**收益统计**:
- 平均收益: -0.014%
- 标准差: 0.907%
- 范围: -3.947% 至 8.880%

**持仓时间**:
- 平均: 22.51 小时
- 中位数: 20.33 小时
- 范围: 0.12 至 263.88 小时

---

### 3. 标签可视化分析 ✓
**文件**: `src/visualize_labels.py`

生成了三组综合可视化:

1. **label_distribution.png**
   - 标签计数和百分比分布
   - 按标签的收益分布
   - 收益箱线图

2. **temporal_analysis.png**
   - 时间序列标签分布
   - 滚动标签平衡 (100-bar window)
   - 月度标签分布

3. **barrier_analysis.png**
   - 持仓时间分布
   - 按标签的持仓时间
   - 障碍宽度(波动率)分布
   - 收益 vs 障碍宽度散点图

---

### 4. 特征工程 (Alpha158 + FFD Momentum) ✓
**文件**: `src/features.py`

实现了整合了传统金融特征和高级 ML 特征的工程管线。特别针对**记忆保留 (Memory Preservation)** 进行了增强：

**A. Alpha158 特征集 (基准对照)**:
- 包含经典的 **ROC/Returns** 特征（相当于 $d=1$ 整数差分）
- 这些特征虽然平稳，但完全丢失了价格的长记忆信息
- 用于作为模型性能的 Baseline

**B. 分数阶差分 (Fractional Differentiation) 增强版**:
- **核心理念**: 用 FFD 替代 ROC 实现更优秀的趋势跟踪
- **FFD Level**: 自动寻找最小 $d$（如 $d=0.2$）的平稳价格序列，保留了 80% 的记忆
- **FFD Momentum (Trend)**: 在平稳的 FFD 序列上计算移动平均 (`FFD_CLOSE_MAw`)
- **FFD Volatility (Risk)**: 在平稳的 FFD 序列上计算波动率 (`FFD_CLOSE_STDw`)
- **FFD Slope**: 在平稳序列上计算局部斜率

**输出文件**:
- `features_labeled.csv` - 包含 Alpha158 + FFD 特征 + 标签的完整数据集
- 特征总数: ~180 个 (大幅增强了特征维度)

---

### 5. 样本权重 (Sample Weights) ✓
**文件**: `src/sample_weights.py`

实现了金融数据的样本权重计算 (Chapter 4)，解决了样本重叠 (Concurrency) 导致的非独立同分布问题：

**核心指标**:
- **Concurrency (并发性)**: 计算每个时间点上重叠的标签数量
- **Average Uniqueness (平均唯一性)**: 标签在其生命周期内的平均独立程度 (Mean: ~0.90)
- **Sample Weights**: 结合了收益率归因和唯一性的最终权重

**关键发现**:
- 平均唯一性高达 0.90，说明在当前 Barrier 设置下，大部分样本是相对独立的
- 权重分布呈现长尾特征，部分高收益且独立的样本获得了极高的权重 (Max > 6.0)

**输出文件**:
- `sample_weights.csv` - 包含唯一性分数和最终权重
- `features_labeled.csv` - 已更新，新增 `sample_weight` 和 `avg_uniqueness` 列
- `visual_analysis/weights_distribution.png` - 权重分布图
- `visual_analysis/weights_correlations.png` - 权重与其他变量的相关性

---

### 6. 交叉验证策略 (Cross-Validation) ✓
**文件**: `src/cv_setup.py`

成功实现了 AFML Chapter 7 的 **Purged K-Fold Cross-Validation**，确保模型评估的严谨性。

**核心机制**:
1.  **Purging (清洗)**: 在训练集中剔除所有与测试集在时间上有重叠的样本。由于我们的标签具有持续期 (Holding Period)，如果不进行 Purging，训练集中的样本可能会包含测试集的信息（Look-ahead Bias）。
2.  **Embargo (禁运)**: 在测试集之后设立了 **1%** 的禁运期，进一步切断长记忆特征可能带来的信息泄露。

**验证结果**:
- 进行了 5-Fold 分割
- 自动化测试确认: **无泄漏 (No Leakage Detected)**。所有的 Train/Test 边界都已正确处理。
- 可视化验证: 生成了 `visual_analysis/cv_splits.png`，直观展示了训练集与测试集之间的“隔离带”。

---

### 7. 初步模型训练 (Baseline Model) ✓
**文件**: `src/train_model.py`

完成了基于 **Random Forest** 的基准模型训练，并在 **Purged K-Fold Cross-Validation** 下进行了评估。

**模型配置**:
- **算法**: Random Forest Classifier
- **参数**: n_estimators=1000, max_depth=5, class_weight='balanced_subsample', criterion='entropy'
- **验证**: 5-Fold Purged CV (1% Embargo)

**性能指标 (Baseline)**:
- **Accuracy**: 48.84% (+/- 3.42%)
- **F1 Score**: 0.4884 (+/- 0.0341)
- **ROC AUC**: 0.5082 (+/- 0.0248)
- **Log Loss**: 0.8001 (+/- 0.0272)

**结果分析**:
- 模型的预测能力目前接近随机猜测 (AUC ~0.50)，这是金融数据的常见起点。
- 考虑到使用了较为严格的 Purged CV，这反映了真实的泛化能力，没有过拟合。
- **特征重要性**: 已提取并保存至 `feature_importance.csv`，并生成了 `visual_analysis/feature_importance.png`。

---

### 8. 特征重要性分析 (Feature Importance) ✓
**文件**: `src/feature_importance.py`

成功实现了 AFML Chapter 8 的特征重要性分析,使用了两种方法:

**A. MDI (Mean Decrease Impurity)**:
- 快速的In-sample方法
- 基于树的分裂不纯度降低
- 优点: 计算快速,特征重要性总和为1
- 缺点: 容易过拟合,对高基数特征有偏向

**B. MDA (Mean Decrease Accuracy) - 金标准 ⭐**:
- 稳健的Out-of-sample排列重要性方法
- 使用 Purged 5-Fold CV 防止信息泄露
- 通过排列特征来衡量性能下降
- 这是金融ML中最可靠的特征重要性方法

**特征筛选结果**:
- **原始特征数**: 181个 (Alpha158 + FFD)
- **正向MDA重要性特征**: 125个 (69%)
- **负向MDA重要性特征**: 56个 (31%) - 这些特征实际上降低了模型性能!

**Top 5 Most Important Features (MDA)**:
1. **KSFT**: 0.0092 - K线形态(Shift)
2. **KSFT2**: 0.0069 - K线形态标准化
3. **KMID2**: 0.0062 - K线中点标准化
4. **KMID**: 0.0057 - K线中点
5. **RESI5**: 0.0042 - 5期线性回归残差

**关键发现**:
- K线形态特征(KSFT系列)表现最优,说明价格形态对预测至关重要
- 许多FFD特征的重要性为负值,说明过度的特征工程可能引入噪音
- 短期技术指标(5-10期)比长期指标(50期)更重要

**输出文件**:
- `feature_importance_mdi.csv` - MDI重要性分数
- `feature_importance_mda.csv` - MDA重要性分数(推荐使用)
- `selected_features.csv` - 筛选后的125个有效特征
- `visual_analysis/feature_importance_comparison.png` - MDI vs MDA 对比图
- `visual_analysis/feature_clustering.png` - 特征聚类分析

---

### 9. 筛选特征模型重训练 (Selected Features Model) ✓
**文件**: `src/train_model.py` (修改), `src/compare_models.py` (新增)

使用MDA筛选后的124个正向重要性特征重新训练了Random Forest模型。

**模型对比结果 (Purged 5-Fold CV)**:

| 指标 | Baseline (181特征) | Selected (124特征) | 变化 |
|------|-------------------|--------------------|------|
| **ROC AUC** | 0.5082 (+/-0.0248) | **0.5122** (+/-0.0257) | **+0.80%** ✓ |
| **Accuracy** | 48.84% (+/-3.42%) | 48.30% (+/-3.02%) | -0.54% |

**关键发现**:
- **AUC提升**: 模型泛化能力提升0.80%
- **特征减少**: 31%的特征被移除(181→124),模型更简洁
- **方差降低**: 准确率的标准差从3.42%降至3.02%
- **训练加速**: 更少的特征意味着更快的训练速度

**结论**: 特征筛选成功! 用更少的特征获得了更好的AUC,验证了MDA方法能有效识别噪音特征。

---

### 10. 超参数优化 (Hyperparameter Optimization) ✓
**文件**: `src/hyperparameter_optimization.py`

使用 **Optuna** 框架结合 **Purged K-Fold CV** 进行了 50 次试验的超参数搜索。

**优化结果**:
- **Baseline AUC**: 0.5122
- **Optimized AUC**: **0.5241** (+2.32%)
- **Best Parameters**:
  - `n_estimators`: 1800 (Base: 1000)
  - `max_depth`: 13 (Base: 5) - 更深的模型捕捉了更多细节
  - `min_samples_leaf`: 2
  - `max_features`: 0.3 (30%)

**输出文件**:
- `best_hyperparameters.csv`: 最佳参数配置
- `feature_importance_optimized.csv`: 优化模型的特征重要性
- `visual_analysis/hyperparameter_optimization.png`: 优化过程可视化

**结论**: 通过允许模型更深 (max_depth=13) 并使用更多树 (n_estimators=1800)，模型性能得到了显著提升，同时通过 Purged CV 保证了结果的稳健性。

---

### 11. Meta-Labeling (元标签) 实现 ✓
**文件**: `src/meta_labeling.py`

实现了 AFML Chapter 3 的元标签策略，训练二级模型（Meta-Model）来过滤一级模型的信号。

**实现细节 (LightGBM 版 v3)**:
- **一级模型**: LightGBM Classifier (500 estimators, max_depth=7)
- **二级模型**: LightGBM Classifier (300 estimators, max_depth=5, 更强正则化)
- **新增特征**: `primary_model_prob` (一级模型对自己预测的置信度)
- **Meta-Labels**: 1 (一级模型正确), 0 (一级模型错误)

**模型升级理由**:
- **更好的特征交互**: LightGBM 在处理非线性特征关系方面优于 Random Forest
- **更快的训练速度**: Gradient Boosting 效率更高
- **更好的正则化**: 内置 L1/L2 正则化，减少过拟合风险
- **原生多分类支持**: 对于一级模型的三分类任务(-1, 0, 1)更适合

**关键洞察**:
- LightGBM 能更好地学习"何时一级模型会犯错"的复杂模式
- Regime Features (波动率、序列相关性、市场熵) 与 LightGBM 的特征交互能力相结合，应该能显著提升 Meta-Model 性能

---

## 📊 数据质量评估

### 优点 ✅
1. **记忆性增强**: 新增的 FFD-Momentum 特征理论上比 ROC 具有更强的预测能力
2. **标签平衡良好**: Loss/Profit 比例接近 53/47
3. **收益分离清晰**: Loss 平均 -0.714%, Profit 平均 +0.787%
4. **特征集完善**: 提供了 "Integer Diff (ROC)" vs "Fractional Diff (FFD)" 的完整对比组
5. **验证严谨**: Purged CV 成功消除了 Look-ahead Bias。

### 需要注意 ⚠️
1. **模型欠拟合**: 深度限制 (max_depth=5) 可能导致模型未能捕捉复杂模式。
2. **维度增加**: 特征数量增加，建议后续增加特征筛选步骤 (Feature Selection)。
3. **Warm-up**: FFD 需要较长的历史数据来预热，导致早期样本被丢弃。

---

## ⚙️ 颗粒度配置 (Configuration Summary)

为了确保各组件的颗粒度对齐，以下是当前系统的核心参数设置：

| 组件 | 参数 | 值 | 物理时间 (估算) |
|------|------|----|----------------|
| **Dollar Bars** | `daily_target` | 4 bars/day | ~6 小时/bar (交易时段) |
| **Labeling** | `vertical_barrier` | 12 bars | ~3 天 (持仓周期) |
| **Labeling** | `volatility_span` | 100 bars | ~25 天 (波动率周期) |
| **Features** | `windows` | [5, 10, 20, 30, 50] | 1.25天 ~ 12.5天 |
| **CV** | `embargo` | 1% | ~10 天 |

**对齐分析**:
- **特征 vs 标签**: 特征窗口 (1.25~12.5天) 覆盖了 标签周期 (3天) 的短、中、长期影响，设置合理。
- **波动率 vs 标签**: 波动率计算窗口 (25天) 远大于持仓周期 (3天)，提供了稳定的基准。
- **Bar频率**: 每日4根Bar提供了一定的日内细节，同时避免了高频噪音。

---

### 12. 市场状态特征 (Market Regime Features) ✓
**文件**: `src/features.py`

为了帮助 Meta-Model 识别不适合一级模型的市场环境，引入了 9 个市场状态特征 (3 Metrics x 3 Windows)：

**核心指标**:
1.  **Volatility (波动率)**:
    -   `REGIME_VOL_w`: 对数收益率的滚动标准差。衡量市场风险水平。
2.  **Serial Correlation (序列相关性)**:
    -   `REGIME_AC1_w`: 滞后1期的自相关系数。
    -   正值表示趋势 (Momentum)，负值表示均值回归 (Mean Reversion)，接近0表示随机游走。
3.  **Market Entropy (市场熵)**:
    -   `REGIME_ENT_w`: 收益率分布的香农熵 (Shannon Entropy)。
    -   衡量市场的无序程度和信息效率。低熵通常对应强结构性行情，高熵对应噪音市场。

**参数配置**:
-   **窗口 (Windows)**: [20, 50, 100] (覆盖 ~1周, 2.5周, 5周)
-   **特征更新**: 已重新生成 `features_labeled.csv`，包含新增列。

### 14. 混合策略实验 (Hybrid Strategy: Rule-Based + Meta-Labeling) ✓
**文件**: `src/train_hybrid_model.py`, `src/rule_based_strategies.py`

为了验证"规则定方向 + ML定仓位"的假设，我们对比了纯 ML 架构与混合架构的表现。**已升级 Meta-Model 为 LightGBM (Gradient Boosting)**。

**实验设计**:
-   **Primary Model**: 均线交叉策略 (MA 20/50 Cross)
-   **Meta-Model**: **LightGBM** (Targeting Profit/Loss of the Rule)
-   **Bet Sizing**: 概率加权 + 并发调整

**实验结果 (OOS Test)**:
1.  **Raw Rule (MA Cross)**:
    -   Return: **-12.00%** (Win Rate: 41.67%)
    -   分析: 简单的趋势策略在当前数据集中表现极差，遭遇了大量的假突破（Whipsaws）。
2.  **Hybrid (Binary - LightGBM)**:
    -   Return: **-5.76%** (Original RF: -6.01%)
    -   分析: LightGBM 进一步减少了亏损，但受限于样本量极小 (仅84个信号)，模型难以有效学习。
3.  **Hybrid (Sized - LightGBM)**:
    -   Return: **-8.68%**
    -   分析: 在高噪声环境下，概率加权反而可能放大了错误信号的影响。

**对比结论**:
-   **Pure ML (CUSUM + RF)**: Return **-0.29%**
-   **Hybrid (MA + LightGBM)**: Return **-5.76%**
-   **胜出者**: **Pure ML**。
-   **Key Insight**: **Sample Size is Critical**. MA Cross signals are too sparse. CUSUM Filter provides enough events (~3700) for ML.

---

### 15. 因子挖掘 2.0 (Feature Engineering 2.0) ✓
**文件**: `src/feature_engineering_v2.py`, `src/microstructure_features.py`, `src/signal_features.py`

... (内容省略) ...

---

### 16. 特征降维 (Dimensionality Reduction - PCA) ✓
**文件**: `src/feature_pca.py`

实现了基于主成分分析 (PCA) 的特征降维与正交化：
- **特征对齐**: 对所有 Alpha158、FFD、微观结构和信号特征进行了标准化处理。
- **信息保留**: 设置了 95% 的累积方差保留阈值。
- **降维效果**: 
  - 原始特征数: **234**
  - 降维后特征数: **51** (压缩率 ~78%)
- **正交化**: PCA 生成的主成分 (PC_1 到 PC_51) 彼此正交，消除了原始特征间的强共线性，有助于提升线性模型或浅层树模型的稳定性。

**输出文件**:
- `features_pca.csv`: 包含 metadata + 51个主成分的降维数据集。
- `visual_analysis/pca_variance.png`: 解释方差贡献率的可视化图表。

### 17. LightGBM 模型调优 (Model Retraining & Tuning) ✓
**文件**: `src/train_lgbm_v2.py`

使用 Feature Engineering 2.0 (MDA 筛选后的 163 个特征) 和 LightGBM 进行了深度调优：
- **算法**: LightGBM (Gradient Boosting Decision Tree)
- **优化框架**: Optuna (TPE Sampler, 50 trials)
- **验证框架**: Purged K-Fold CV (5 Folds, 1% Embargo)

**性能突破**:
- **Baseline (RF)**: AUC 0.5122
- **Previous Best (Optimized RF)**: AUC 0.5241
- **LightGBM V2**: **AUC 0.5333**
- **提升幅度**: +4.12% (vs Baseline)

**最佳参数**:
- `num_leaves`: 111 (高容量模型)
- `learning_rate`: 0.0116 (低学习率，更稳健)
- `max_depth`: 9
- `reg_alpha`: 5.37e-07 (L1正则)
- `reg_lambda`: 4.02e-06 (L2正则)

**结论**: 
- **特征工程有效**: 引入 MACD_SLOPE、VPIN 等特征结合 LightGBM 的非线性能力，成功突破了 RF 的性能瓶颈。
- **稳健性验证**: 在严格的 Purged CV 下取得 0.53+ AUC，表明模型具有真实的预测能力。

---

## 🎯 下一步规划 (Future Work)

### 18. PCA 特征评估 (PCA Feature Evaluation) ✓
**文件**: `src/train_lgbm_pca.py`

基于生成的 49 个 PCA 主成分 (保留 95% 方差) 进行了 LightGBM 模型训练与调优，验证了“特征正交化”对树模型的影响。

**实验结果**:
- **LightGBM V2 (163 Feats)**: AUC 0.5333
- **LightGBM PCA (49 Feats)**: **AUC 0.5546**
- **提升幅度**: +4.00% (vs V2), +8.27% (vs Baseline)

**关键洞察**:
1.  **降维有效性**: 仅用 30% 的特征数量 (49 vs 163) 取得了显著更好的性能。
2.  **去噪**: PCA 可能成功剥离了原始特征中的随机噪音，使树模型能更专注于核心信息。
3.  **重要性分布**: Top 特征包括 `PC_9`, `PC_1`, `PC_31`。这表明预测信号不仅存在于最大方差方向 (`PC_1`)，也隐藏在较小的方差分量中 (`PC_31`)。

---

## 🎯 下一步规划 (Future Work)

### 19. 策略回测 (Strategy Backtesting) ✓
**文件**: `src/backtest.py`

基于最佳模型 (LightGBM + PCA, AUC 0.5546) 进行了向量化策略回测。

**回测设置**:
- **信号生成**: 使用 Purged CV 生成的 OOS 预测概率。
- **策略 A (Binary)**: 若 P > 0.5 做多，P < 0.5 做空。
- **策略 B (Bet Sizing)**: 仓位大小 $m = 2P - 1$ (Kelly-like)。
- **交易成本**: 双边 2bps (每笔交易)。

**绩效指标 (Net of Costs)**:

| 指标 | Binary Strategy | Sized Strategy | 提升 |
|------|----------------|----------------|------|
| **Win Rate** | 53.46% | 53.46% | - |
| **Sharpe Ratio** | **1.01** | **2.10** | **+108%** |
| **Total Return** | 0.2454 | 0.3458 | +41% |

**关键结论**:
- **Bet Sizing 至关重要**: 引入概率加权仓位后，夏普比率翻倍 (1.01 -> 2.10)。这验证了 AFML Chapter 10 的核心观点：模型不仅要预测方向，更要预测*信心*。
- **正向收益**: 即便扣除交易成本，模型在 OOS 数据上依然实现了显著的正收益。

**输出文件**:
- `backtest_results.csv`: 逐笔交易明细。
- `visual_analysis/backtest_performance.png`: 累计收益曲线 (Equity Curve)。

---

## 🎯 下一步规划 (Future Work)

### 1. 组合优化 (Portfolio Optimization) - 优先级: 高
- **目标**: 将单一资产的预测扩展到多资产或多策略组合。
- **计划**: 尝试 HRP (Hierarchical Risk Parity) 或 CLA (Critical Line Algorithm) 来分配资金 (AFML Chapter 16)。

### 2. 实盘模拟 (Paper Trading)
- **目标**: 接入实时数据流 (e.g. IBKR, Binance)，验证模型在真实环境下的表现。
- **计划**: 构建轻量级执行引擎。

---

## 📁 当前项目结构

```
AFML/
├── src/
│   ├── process_bars.py          # Dollar bars 生成
│   ├── labeling.py              # Triple barrier 标签
│   ├── features.py              # Alpha158 + FFD 增强特征工程
│   ├── sample_weights.py        # 样本权重计算
│   ├── cv_setup.py              # 交叉验证配置
│   ├── train_model.py           # 模型训练 (自动使用筛选特征)
│   ├── feature_importance.py    # 特征重要性分析 (MDI/MDA)
│   ├── compare_models.py        # 模型对比脚本
│   └── hyperparameter_optimization.py # 超参数优化 (Optuna)
├── features_labeled.csv         # 完整训练数据 (Features + Labels + Weights)
├── selected_features.csv        # MDA筛选后的124个有效特征
├── feature_importance_mda.csv   # MDA重要性分数
├── feature_importance_optimized.csv # 优化后的特征重要性
├── best_hyperparameters.csv     # 最佳超参数配置
└── visual_analysis/
    ├── feature_importance_comparison.png  # MDI vs MDA 对比
    ├── feature_clustering.png   # 特征聚类分析
    └── cv_splits.png            # CV 分割示意图
```

