# IF9999 趋势跟踪策略

基于 AFML 方法论的沪深300股指期货趋势跟踪策略开发项目。

## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py     # Dollar Bars 构建与验证
├── 02_feature_engineering.py    # Phase 2: FracDiff + CUSUM
├── 03_trend_scanning.py         # Phase 3: Trend Scanning 标签
├── 04_primary_model_backtest.py # Phase 3.5: Primary Model 回测
├── config.py                    # 配置参数
├── output/
│   ├── bars/                    # 生成的 Bar 数据
│   ├── features/                # 特征和标签输出
│   └── figures/                 # 可视化图表
└── README.md
```

## 快速开始

```bash
# 构建 Dollar Bars 并生成验证图表
uv run python strategies/IF9999/01_dollar_bar_builder.py
```

## Dollar Bars 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TARGET_DAILY_BARS | 6 | 目标每日 Bar 数量（参数优化最优值） |
| EWMA_SPAN | 20 | 动态阈值 EWMA 窗口 |
| CONTRACT_MULTIPLIER | 300 | IF 合约乘数（每点300元） |

## 三刀验证

验证 Dollar Bars 是否更接近 I.I.D. Normal：

1. **独立性（第一刀）**：AC1 ≈ 0，Ljung-Box p > 0.05
2. **同分布（第二刀）**：方差的方差 VoV → 0
3. **正态性（第三刀）**：JB 统计量最低

## 数据源

- **品种**：IF9999.CCFX（沪深300股指期货主力合约）
- **频率**：1 分钟
- **范围**：2020-01-02 至 2026-03-27
- **数据量**：362,161 行

## Phase 2: 特征工程

运行特征工程流程：

```bash
uv run python strategies/IF9999/02_feature_engineering.py
```

### FracDiff 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FRACDIFF_THRES | 1e-4 | FFD 权重截断阈值 |
| FRACDIFF_D_STEP | 0.05 | d 搜索步长 |
| FRACDIFF_MAX_D | 1.0 | d 最大值 |

### CUSUM Filter 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CUSUM_WINDOW | 20 | 动态阈值滚动窗口 |
| CUSUM_MULTIPLIER | 3 | 阈值乘数（控制事件率） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `fracdiff_series.parquet` | FracDiff 平稳化序列 |
| `fracdiff_params.parquet` | 最优 d 值和统计信息 |
| `cusum_events.parquet` | CUSUM 事件点索引 |

## Phase 3: Trend Scanning 标签生成

运行 Trend Scanning 生成 Primary Model 输出：

```bash
uv run python strategies/IF9999/03_trend_scanning.py
```

### Trend Scanning 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TREND_WINDOWS | [5, 10, 20, 30, 50] | 窗口长度范围（Bars） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `trend_labels.parquet` | Trend Scanning 标签（t1, t_value, side） |

### 标签格式

| 列 | 类型 | 说明 |
|----|------|------|
| `t1` | Datetime | 趋势窗口结束时间 |
| `t_value` | float64 | OLS 斜率 t 统计量（样本权重） |
| `side` | int8 | +1（上涨）/ -1（下跌）/ 0（无趋势） |

### 可视化

| 文件 | 说明 |
|------|------|
| `03_trend_distribution.png` | side 和 t_value 分布 |
| `03_trend_example.png` | 趋势窗口示例（高 |t_value| 事件） |

## Phase 3.5: Primary Model 回测验证

运行 Primary Model 回测验证信号有效性：

```bash
uv run python strategies/IF9999/04_primary_model_backtest.py
```

### 回测方法

| 项目 | 说明 |
|------|------|
| 收益计算 | 点数收益 = side × (close_t1 - close_entry) |
| 持仓周期 | 使用 t1 作为平仓时间（Trend Scanning 最优窗口） |
| 分析维度 | 整体统计 + t_value 分位数分析 |

### 统计指标

| 指标 | 说明 |
|------|------|
| 胜率 | 盈利信号比例 |
| 平均收益 | 所有信号平均点数收益 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 总收益 | 所有信号累计点数 |

### t_value 分位数分析

验证假设：高 |t_value| 信号应有更好的表现。

| 分组 | 阈值 |
|------|------|
| Top 10% | |t_value| > 90% 分位数 |
| 10%-50% | |t_value| > 50% 分位数 |
| Bottom 50% | |t_value| ≤ 50% 分位数 |

### 可视化输出

| 文件 | 说明 |
|------|------|
| `04_pnl_distribution.png` | 收益分布直方图 |
| `04_cumulative_pnl.png` | 累积收益曲线 |
| `04_tvalue_vs_pnl.png` | |t_value| vs 收益散点图 |

## Phase 4-6: Primary Models & Meta Model

### 训练脚本

| 脚本 | 说明 |
|------|------|
| `04_lgbm_primary_model.py` | LightGBM Primary Model |
| `04_ma_primary_model.py` | MA Cross Primary Model |
| `04_momentum_primary_model.py` | Momentum Primary Model |
| `04_supertrend_primary_model.py` | SuperTrend Primary Model |
| `04c_compare_primary_models.py` | Primary Model 对比 |
| `06_meta_labels.py` | Meta Labels 生成 |
| `07_meta_model.py` | Meta Model 训练（MDI + MDA） |
| `07b_meta_model_feature_selection.py` | 特征筛选对比实验 |

### Meta Model 架构

```python
# AFML 规范配置
base_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_features=1,              # 消除遮蔽效应
    max_depth=5,
    class_weight="balanced",
)

model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=1000,
    max_samples=0.5,             # 随机子集采样
)
```

### 验证方法

- **PurgedKFold**: 5-fold 跨时间验证，防止数据泄露
- **Embargo**: 5% 禁运期，应对序列相关性

## Phase 7: 特征重要性分析

### MDI vs MDA 核心区别

| 维度 | MDI (Mean Decrease Impurity) | MDA (Mean Decrease Accuracy) |
|------|------------------------------|------------------------------|
| **原理** | 树分裂时特征减少的不纯度 | 置换特征后模型性能下降 |
| **计算时机** | 训练时（in-sample） | 测试时（out-of-sample） |
| **偏置** | 偏向高基数特征 | 无此偏置 |
| **替代效应** | 有（共线特征共享重要性） | Clustered MDA 可解决 |

### 关键经验：MDA 负贡献 ≠ 特征无用

**实验结果**：

| Scheme | Features | F1 | Precision | Recall |
|--------|----------|-----|-----------|--------|
| Full | 39 | 0.619 | 0.580 | 0.667 |
| Aggressive (仅MDA正) | 4 | 0.588 | 0.540 | 0.645 |
| **Balanced (删除C4)** | **26** | **0.633** | **0.590** | **0.688** |

**核心发现**：

1. **Momentum 特征 MDA 负但有价值**
   - MDA = -0.029，但与标签负相关（rsi_7: -0.177, stoch_k_14: -0.188）
   - 模型学到了反向信号，删除反而降低性能

2. **Time/Session 特征应删除**
   - MDA = -0.049 ± 0.506（高方差）
   - 不同 fold 表现极端不一致
   - 删除后 F1 提升 2.3%

3. **Bagging 内置特征筛选**
   - `max_features=1` 每棵树只看 1 个特征
   - 1000 棵树投票自动过滤噪音

### 特征筛选决策框架

```
MDA 负贡献的可能原因:
├─ a) 特征是纯噪音 (特征-标签相关性 ≈ 0) → 删除
├─ b) 特征被反向利用 (特征-标签相关性显著负) → 保留 ✅
└─ c) 特征在组合中有交互价值 → 评估保留
```

**判断流程**：

1. 计算 MDA → 识别负贡献 cluster
2. 检查特征-标签相关性 → 区分噪音 vs 反向信号
3. 检查 MDA 方差 → 高方差特征优先删除
4. 训练对比实验 → 验证筛选效果

### 删除的 C4 特征（13个）

```
feat_adx_14, feat_sin_time, feat_cos_time, feat_sin_dow, feat_cos_dow,
feat_asia_sess, feat_eu_sess, feat_lz_entropy_100,
feat_serial_corr_lag1_20, feat_serial_corr_lag5_20, feat_serial_corr_lag10_20,
feat_ljung_box_20, feat_adf_test_100
```

**删除理由**：MDA 负贡献 + 高方差（std=0.506）

### 保留的特征（26个）

- **C1 (Momentum)**：RSI, ROC, Stoch, VWAP dist 等（反向信号有价值）
- **C2 (Volatility)**：EWM vol, HL vol, Amihud, spread 等
- **C3 (Entropy)**：Shannon, LZ entropy（MDA 正贡献 +0.096）

## 输出文件

### 模型文件

| 文件 | 说明 |
|------|------|
| `output/models/meta_model_balanced.pkl` | 最优 Meta Model（26特征） |
| `output/models/meta_model_balanced_features.txt` | 特征列表 |

### 特征重要性

| 文件 | 说明 |
|------|------|
| `output/features/meta_mda_importance.parquet` | Clustered MDA 结果 |
| `output/figures/07_feature_importance_mdi.png` | MDI 可视化 |
| `output/figures/07_feature_importance_mda.png` | MDA 可视化 |
| `output/figures/07b_feature_selection_comparison.png` | 三方案对比 |

## 后续阶段

- Phase 8: 策略组合与回测
- Phase 9: PBO 验证

## 参考

- Marcos Lopez de Prado, *Advances in Financial Machine Learning*, 2018
- AFML Chapter 8: Feature Importance
- AFML Chapter 3: Meta-Labeling