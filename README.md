# AFML 量化研发工厂 (AFML Quant Factory)

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-High--Performance-orange.svg)](https://pola.rs/)
[![AFML](https://img.shields.io/badge/Methodology-AFML-red.svg)](https://www.cambridge.org/core/books/advances-in-financial-machine-learning/2A6E941A5C4010174241517565B1705D)

本项目是一个基于 Marcos López de Prado 的经典著作《Advances in Financial Machine Learning (AFML)》构建的高性能量化研发框架。它旨在为量化研究者提供一套遵循金融严谨性、防止过拟合且具备工业级性能的工具链。

## 🌟 核心特性

- **🚀 高性能引擎**: 全面采用 [Polars](https://pola.rs/) 进行数据处理。在保持 Pandas 般便利性的同时，在大规模金融数据处理上实现了数倍乃至数十倍的性能提升。
- **🏗️ 严谨的 AFML 架构**: 严格遵循 Lopez de Prado 提出的金融机器学习方法论，包括但不限于：
  - **金融数据结构**: 支持 Dollar Bars, Volume Bars, Tick Bars 等。
  - **三重障碍法 (Triple-Barrier Method)**: 结合 CUSUM 过滤器和动态波动率调整的标签标注。
  - **平稳性处理**: 自动分式微分 (Fractional Differentiation) 搜索，在保留价格记忆的同时确保统计平稳性。
  - **样本权重**: 基于并发度的唯一性权重 (Uniqueness Weights) 与时间衰减 (Time Decay)。
  - **净化交叉验证**: 提供 Purged & Embargoed K-Fold 交叉验证，彻底消除标签泄漏。
  - **元标签 (Meta-Labeling)**: 采用辅助模型过滤主模型信号，显著提高策略精确度。
- **🛡️ 统计验证 (Anti-Overfitting)**: 内置 Probabilistic Sharpe Ratio (PSR) 和 Deflated Sharpe Ratio (DSR)，有效防范“由于多次检验导致的伪发现”。
- **🧩 模块化设计**:
  - 提供 Scikit-Learn 兼容的面向对象 (OO) 接口。
  - 完善的 Pipeline 编排能力，支持从原始分笔数据到策略验收的全流程自动化。
- **👁️ 可视化分析**: 
  - 支持全流程图表生成 (`--visualize`)，包括 Dollar Bars 统计、Triple Barrier 标注可视化、特征平稳性热图、交叉验证时间轴、元标签性能评估及策略净值曲线。

## 📂 项目结构

```text
├── src/
│   ├── afml/                # 核心库 (Polars Native)
│   │   ├── stationarity.py  # 平稳性检测与 FFD 自动搜索
│   │   ├── metrics.py       # DSR, PSR 等金融指标
│   │   ├── labeling.py      # 三重障碍法标注
│   │   ├── dollar_bars.py   # 金融条柱生成
│   │   └── ...              # 交叉验证、权重计算、元标签等
├── examples/                # 示例脚本 (端到端 pipeline)
├── data/                    # 原始数据与生成的人造数据 (已 Gitignore)
├── tests/                   # 单元测试与集成测试
├── config/                  # 处理器默认配置文件
├── PROGRESS.md              # 项目开发进度追踪
└── AGENTS.md                # 针对 AI 协作的规范文档
```

## 🛠️ 快速开始

### 1. 环境准备

本项目使用 `uv` 进行依赖管理。请确保已安装 `uv`:

```bash
# 推荐使用 uv 同步环境
uv sync
```

### 2. 运行端到端 Pipeline

运行内置的高性能 Pipeline，体验从数据加载、平稳性处理到 DSR 策略验收的全流程。该脚本支持自动处理大规模 Parquet 数据集，并生成全流程可视化报告。

#### A. 标准完整运行 (End-to-End)
自动执行从数据标准化到 DSR 收益验收的 10 个步骤：
```bash
uv run python afml_polars_pipeline.py data/BTCUSDT/parquet_db
```

#### B. 参数寻优 (Dollar Bar 采样优化)
根据 Jarque-Bera 正态性测试搜寻最佳的 `daily_target` 采样频率：
```bash
uv run python afml_polars_pipeline.py --jb-sweep --jb-targets 4 6 8 10 20 50 100 --jb-data-dir data/raw_data_polars.parquet
```

#### C. 分步执行 (Step-by-Step AFML Workflow)
严格按照《Advances in Financial Machine Learning》的标准流程执行：

1. **美元条柱生成 (Sampling)**
   将时间序列转换为信息驱动的 Dollar Bars，恢复正态性：
   ```bash
   uv run python afml_polars_pipeline.py --step bars --daily-target 50
   ```

2. **三重障碍标注 (Labeling)**
   应用 Triple-Barrier Method，定义“盈亏”事件：
   ```bash
   uv run python afml_polars_pipeline.py --step labels --pt-sl 1.0 1.0 --vertical-barrier 10
   ```

3. **特征工程 (Feature Engineering)**
   **[关键步骤]** 生成 Alpha 因子（Alpha158）并进行分式微分 (Fractional Differentiation) 以确保平稳性：
   ```bash
   uv run python afml_polars_pipeline.py --step features --ffd-d 0.5
   ```

4. **样本权重计算 (Sample Weights)**
   **[关键步骤]** 计算标签的平均唯一性 (Average Uniqueness)，解决金融数据样本重叠导致的过拟合问题：
   ```bash
   uv run python afml_polars_pipeline.py --step weights
   ```

5. **建模与交叉验证 (Modeling)**
   使用 Purged K-Fold CV 进行训练，并应用 Embargo 防止数据泄漏：
   ```bash
   uv run python afml_polars_pipeline.py --step model --embargo 0.01
   ```

#### E. 查看完整参数说明
该 Pipeline 包含数十项微调参数（如 CUSUM 阈值、DSR 验收标准等），可以通过以下命令查看完整说明：
```bash
uv run python afml_polars_pipeline.py --help
```

### 3. 使用 OO 接口示例

```python
from afml import DollarBarsProcessor, TripleBarrierLabeler

# 1. 生成 Dollar Bars
processor = DollarBarsProcessor(daily_target=4)
df_bars = processor.fit_transform(raw_df)

# 2. 标注标签
labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
labeler.fit(df_bars["close"])
labels = labeler.label(df_bars["close"], events)
```

## 📈 开发进度

目前项目已全面完成 Polars 迁移与架构简化。

- [x] **Polars Native**: 全库采用 Polars 实现，无冗余代码。
- [x] **平稳性模块**: 已实现自动搜索最小 $d$ 值。
- [x] **策略验证**: 已集成 DSR/PSR 验收机制。
- [ ] **进阶特性**: 计划加入结构性断点检测 (Structural Breaks) 等。

详见 [PROGRESS.md](./PROGRESS.md)。

## 📚 参考资料

- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
- Marcos López de Prado, *Machine Learning for Asset Managers*, Cambridge University Press, 2020.

---

> **注意**: 金融研发具有高风险性，本库仅供学习与研究参考。在使用任何策略进行实盘交易前，请务必进行充分的离线验证。
