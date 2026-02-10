# AFML - Advances in Financial Machine Learning

本项目实现了 Marcos López de Prado 所著《Advances in Financial Machine Learning》一书中的核心算法，专注于量化金融领域的机器学习研究。

## 项目概述

AFML 是一个完整的量化金融 ML 管线，包含从数据预处理到模型训练的完整流程。项目采用 Python 语言，使用 pandas、numpy、scikit-learn 等主流数据科学库。

### 核心功能

- **Dollar Bars 生成**：使用固定或动态阈值生成美元Bars，降低市场微观结构噪音
- **三重障碍标签法 (Triple Barrier)**：实现止盈、止损、时间限制三位一体的标签生成
- **特征工程**：整合 Alpha158 基准特征集与分数阶差分 (FFD) 增强特征
- **样本权重计算**：基于唯一性 (Uniqueness) 和时间衰减的权重体系
- **交叉验证**：Purged K-Fold CV，防止前视偏差和信息泄露
- **元标签 (Meta-Labeling)**：二级模型过滤一级模型信号，提升策略质量

## 项目结构

```
AFML/
├── src/                          # 核心源代码
│   ├── process_bars.py          # Dollar Bars 生成
│   ├── labeling.py              # Triple Barrier 标签生成
│   ├── features.py              # 特征工程 (Alpha158 + FFD)
│   ├── sample_weights.py        # 样本权重计算
│   ├── cv_setup.py              # Purged K-Fold 交叉验证
│   ├── train_model.py           # Random Forest 模型训练
│   ├── feature_importance.py    # 特征重要性分析 (MDI/MDA)
│   ├── compare_models.py        # 模型对比
│   ├── hyperparameter_optimization.py # 超参数优化 (Optuna)
│   ├── meta_labeling.py         # 元标签策略
│   ├── bet_sizing.py            # 仓位管理
│   ├── visualize_labels.py      # 标签可视化
│   └── visualize_weights.py     # 权重可视化
├── tests/                       # 测试文件
├── visual_analysis/             # 可视化输出
├── data/                        # 数据文件
├── pyproject.toml              # 项目配置
├── README.md                   # 项目说明
└── AGENTS.md                   # AI 代理开发指南
```

## 快速开始

### 环境配置

```bash
# 安装依赖
uv sync

# 运行完整管线
uv run python src/process_bars.py   # 1. 生成 Dollar Bars
uv run python src/labeling.py         # 2. 生成标签
uv run python src/features.py         # 3. 特征工程
uv run python src/sample_weights.py   # 4. 计算样本权重
uv run python src/meta_labeling.py    # 5. 运行 Meta-Labeling
```

### 核心参数配置

| 组件 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Dollar Bars | `daily_target` | 4 bars/day | 每日目标Bars数量 |
| 标签生成 | `vertical_barrier` | 12 bars | 持仓时间限制 (~3天) |
| 标签生成 | `volatility_span` | 100 bars | 波动率计算窗口 |
| 特征工程 | `windows` | [5,10,20,30,50] | 滚动窗口参数 |
| 交叉验证 | `embargo` | 1% | 禁运期比例 |
| 交叉验证 | `n_splits` | 5 | K-Fold 折数 |

## 数据流

```
原始价格数据
    ↓
process_bars.py (Dollar Bars)
    ↓
labeling.py (Triple Barrier Labels)
    ↓
features.py (Feature Engineering)
    ↓
sample_weights.py (Sample Weights)
    ↓
训练数据 (features_labeled.csv)
    ↓
cv_setup.py (Purged K-Fold)
    ↓
train_model.py / meta_labeling.py (模型训练)
```

## 核心模块说明

### 1. Dollar Bars (process_bars.py)

将原始时间序列转换为美元Bars，相比传统时间Bars具有更好的统计特性。

**输出**: `dynamic_dollar_bars.csv` (~3,927 bars)

### 2. Triple Barrier Labeling (labeling.py)

为每个事件设置三个障碍：
- **上障碍**：止盈位 (1x 波动率)
- **下障碍**：止损位 (1x 波动率)
- **垂直障碍**：时间限制 (12 bars)

**标签分布**: Loss (-1): ~53%, Profit (1): ~47%

**输出**: `labeled_events.csv`, `dollar_bars_labeled.csv`

### 3. 特征工程 (features.py)

**A. Alpha158 基准特征集**:
- ROC/Returns 特征 (整数差分)
- 短期技术指标 (5-50期)

**B. FFD 增强特征** (记忆保留):
- Fractional Differentiated Level: 保留80%价格记忆
- FFD Momentum / Volatility / Slope

**C. 市场状态特征**:
- Volatility (波动率)
- Serial Correlation (序列相关性)
- Market Entropy (市场熵)

**输出**: `features_labeled.csv` (~180+ 特征)

### 4. 样本权重 (sample_weights.py)

计算考虑以下因素的权重：
- **Concurrency**: 样本重叠程度
- **Average Uniqueness**: 生命周期内独立程度
- **Time Decay**: 时间衰减因子

**输出**: `sample_weights.csv`

### 5. 交叉验证 (cv_setup.py)

**Purged K-Fold CV** 机制：
- **Purging**: 剔除训练集中与测试集时间重叠的样本
- **Embargo**: 测试集后设置禁运期，防止信息泄露

**验证结果**: 5-Fold 分割无泄漏

### 6. 模型训练 (train_model.py)

**算法**: Random Forest Classifier

**关键参数**:
- `n_estimators`: 1000-1800
- `max_depth`: 5-13
- `class_weight`: balanced_subsample

**输出**: `feature_importance.csv`

### 7. 特征重要性 (feature_importance.py)

**方法**:
- **MDI** (Mean Decrease Impurity): 快速样本内方法
- **MDA** (Mean Decrease Accuracy): 稳健样本外方法 (推荐)

**Top 5 特征**:
1. KSFT (K线形态)
2. KSFT2 (K线形态标准化)
3. KMID2 (K线中点标准化)
4. KMID (K线中点)
5. RESI5 (5期线性回归残差)

**输出**: `feature_importance_mda.csv`, `selected_features.csv`

### 8. 超参数优化 (hyperparameter_optimization.py)

使用 **Optuna** 框架进行贝叶斯优化。

**优化结果**: AUC 从 0.5122 提升至 0.5241 (+2.32%)

### 9. Meta-Labeling (meta_labeling.py)

二级模型策略：
- **一级模型**: 预测交易方向
- **二级模型**: 过滤一级模型信号

**策略对比**:
| 策略 | 收益率 | Sharpe | 交易次数 |
|------|--------|--------|----------|
| Base (Primary) | -16.35% | -5.02 | 147 |
| Meta (Binary) | -4.65% | -2.96 | 35 |

## 关键发现

### 优点

- Dollar Bars 有效降低 Jarque-Bera 统计量
- 标签分布平衡 (53/47)
- Purged CV 消除前视偏差
- FFD 特征保留价格记忆
- MDA 有效识别噪音特征

### 注意事项

- 模型预测能力接近随机 (AUC ~0.52)
- 需引入市场状态特征改善 Meta-Model
- 短期技术指标优于长期指标

## 依赖环境

- Python >= 3.13
- pandas >= 2.3.3
- numpy >= 2.4.1
- scikit-learn >= 1.5.0
- matplotlib >= 3.10.8
- scipy >= 1.17.0
- seaborn >= 0.13.2
- statsmodels >= 0.14.6

## 开发指南

详见 [AGENTS.md](AGENTS.md)

## 参考文献

- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.

## License

MIT
