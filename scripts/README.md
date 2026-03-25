# AFML Pipeline Execution Guide (AFML 流水线执行指南)

此目录下的脚本按照 *Advances in Financial Machine Learning* 方法论的逻辑顺序组织。

## 核心流水线顺序 (Core Pipeline)

按以下编号顺序运行脚本以构建完整的交易策略：

1.  **`01_data_ingestion.py`** (原 `binance2h5.py`)
    *   **作用**：将原始 CSV/API 数据导入并转换为高性能 HDF5 格式。
2.  **`02_bar_evaluation.py`** (原 `evaluate_dollar_bars.py`)
    *   **作用**：构建金额棒 (Dollar Bars) 并评估其统计特性（IID 性、正态性）。
3.  **`03_feature_engineering.py`** (原 `feature_engineering.py`)
    *   **作用**：计算技术因子、微观结构因子，并构建初始特征矩阵。
4.  **`04_sampling_and_labeling.py`** (原 `cusum_filtering.py`)
    *   **作用**：应用 CUSUM 过滤器进行采样，使用 Trend Scan 生成侧向信号 (Side)，并应用三重屏障标签 (TBM) 及样本权重计算。
5.  **`05_feature_importance_analysis.py`** (原 `feature_importance_analysis.py`)
    *   **作用**：使用聚类 MDA (Clustered MDA) 和 PurgedKFold 进行无偏差特征重要性评估。
6.  **`06_primary_model_training.py`** (原 `primary_model_training.py`)
    *   **作用**：训练主模型。核心目标是**高召回率 (High Recall)**，宁可错报不可漏报。
7.  **`07_meta_model_training.py`** (原 `meta_model_training.py`)
    *   **作用**：训练元模型 (Meta-Labeling)。通过预测主模型信号的正确性，显著提升整体**精度 (Precision)**。
8.  **`08_bet_sizing_pipeline.py`** (原 `bet_sizing_pipeline.py`)
    *   **作用**：将元模型的概率映射为实际交易仓位。
9.  **`09_backtest_performance.py`** (原 `backtest_performance.py`)
    *   **作用**：执行带 PSR (概率夏普比率) 的回测，评估策略在考虑多重测试偏见后的可靠性。

## 辅助与可视化工具 (Utilities)

*   **`util_plotting_lib.py`**：核心绘图函数库（封装了 CUSUM、TBM 等可视化逻辑）。
*   **`10_run_visualizations.py`** (原 `runner_visuals.py`)：用于生成上述所有图表的执行脚本。

---

## 重要注意事项 (Important Notes)

*   **数据流向**：所有脚本的输出建议统一存放在 `outputs/` 目录中。
*   **性能工程**：脚本使用了 Numba 进行 JIT 加速，首次运行会进行编译。
*   **训练/测试隔离**：流水线内置了从 05 到 07 的 PurgedKFold 和 Embargo 逻辑，严禁在这些步骤外手动拆分训练集以防信息泄露。
