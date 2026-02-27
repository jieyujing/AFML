## Why

在构建基于元标签（Meta-Labeling）的量化流水线时，CUSUM 过滤器已经解决了“何时交易（时机触发）”的问题。然而，传统用于决定“交易多空方向（Side）”的主模型（如双均线或固定窗口动量）存在一个致命缺陷：它们高度依赖主观设定的固定时间窗口。这种固定参数在波动的市场中极易失效，导致在震荡市中产生严重的滞后和高频假信号（反复摩擦）。

引入 Trend Scan（趋势扫描法 / 滚动 t 统计量趋势）作为主模型，旨在从根本上消除固定回溯/预测窗口带来的参数敏感性。它通过动态遍历多个不同长度的时间窗口进行 OLS 线性回归，自动选择 t 统计量绝对值最大（统计学上最显著）的窗口作为真实趋势。

**现在做的原因：** 我们的系统需要一个具有“高召回率”且“逻辑绝对白盒”的 Primary Model。Trend Scan 不仅能输出自适应的多空方向（符号 1/-1），还能附带输出趋势的置信度强度（t-value）。这一强度指标可以直接作为下游次级机器学习模型（Secondary ML Model）的样本权重（Sample Weights），实现“主模型找方向+给底气，次模型算概率”的完美 AFML 架构闭环。

## What Changes

- **引入 Trend-Scanning 算法核心逻辑：** 在给定的时间序列上，针对不同的窗口长度集合 L 执行滚动线性回归，提取回归系数的 t 统计量（t-value）。
- **信号方向生成（Side）：** 在 CUSUM 触发的每一个离散事件时间戳（t_events）上，提取使 t 统计量绝对值最大化的那个窗口。取该最大 t 统计量的**符号（Sign）**作为主模型的交易方向信号（Side = 1 为多，-1 为空）。
- **样本权重输出（Strength/Weights）：** 提取最大 t 统计量的绝对值（Absolute t-value），将其标准化后作为当前交易信号的强度置信度，供下游 Meta-Labeling 训练使用。
- **流水线串联：** 将原有系统中的 CUSUM 输出与 Trend Scan 结合，确立 “连续数据计算 -> CUSUM 事件采样 -> 在事件点进行 Trend Scan 计算 -> 传递 Side 矩阵至三重屏障” 的无前视偏差（Lookahead-bias free）数据流。

## Capabilities

### New Capabilities
- `trend-scan-primary-model`: 基于 OLS 滚动线性回归与 t 统计量最大化的动态趋势识别模块，提供无固定窗口参数的交易方向（Side）判定与显著性评分。

### Modified Capabilities
- `cusum-event-sampling`: 接口层调整，支持将过滤生成的离散时间戳直接作为索引，驱动下游的 Trend Scan 在特定时刻进行计算，而非在全量连续数据上计算。
- `meta-labeling-pipeline`: 调整三重屏障打标与模型训练的需求。现在必须强制要求接收 Primary Model 传入的 Side 参数，并且支持接收 Trend Scan 传出的 t-value 转换为机器学习训练用的 sample_weight。

## Impact

- **代码与 API：** 将新增 `features/trend_scan.py` 模块。回测主流程或 `scripts/cusum_filtering.py` 需要重构，剥离旧有的均线/方向判定逻辑，替换为实例化 Trend Scan 模块。
- **系统性能与依赖（高风险）：** Trend Scan 需要在大量事件点上向回/向后穷举多个窗口进行 OLS 回归，计算复杂度极高（O(N×L)）。强烈依赖 afmlkit 底层的 Numba / JIT 向量化加速机制，原生 Pandas/Statsmodels 的实现会导致计算时间呈指数级上升。
- **下游流水线（Downstream）：** Triple-Barrier Method 和 Meta-Labeling 的输入签名将发生变化。下游特征工程矩阵（X）与标签（y）对齐时，必须严格通过 CUSUM 的 t_events 索引进行 `.loc[]` 切片校验，严防时间穿越。
