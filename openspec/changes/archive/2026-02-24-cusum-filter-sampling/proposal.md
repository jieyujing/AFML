# Proposal: CUSUM Filter Sampling

## Motivation
根据 AFML（Advances in Financial Machine Learning）的最佳实践要求，需要对生成的 Dollar Bars 数据（`outputs/dollar_bars/dollar_bars_freq20.csv`）进行 CUSUM 过滤器（CUSUM Filter）采样去噪，从而仅在市场发生结构性价格偏离时（即有效信息到达时）才生成观测样本。此举能有效减少微观结构噪声，提高产生的机器学习特征的信噪比。

## What Changes
我们需要实现一个动态阈值的 CUSUM 采样流程：
1. **数据读取**：加载 `outputs/dollar_bars/dollar_bars_freq20.csv`。
2. **波动率计算**：根据目标特征序列（如收益率或分数阶差分），计算其在过去一个滚动窗口内的标准差 $\sigma$。
3. **阈值设定**：动态设定 CUSUM 过滤器的阈值为 $h = 2 \times \sigma$。
4. **事件过滤**：调用 `afmlkit.sampling.filters.cusum_filter` 对原序列进行事件过滤，获取采样索引。
5. **数据输出**：根据采样索引提取高信息量的子序列，以供后续三重屏障打标签等模型训练阶段使用。

## Capabilities
- `cusum_sampling_pipeline`: 数据读取、滚动波动率计算、动态阈值生成以及利用 CUSUM 方法执行采样过滤的核心业务管道。

## Impact
此变更构建在 `afmlkit` 现有的采样过滤能力之上。它是一个相对独立的数据预处理环节，直接消费现存的 Dollar Bars 文件并产出新的过滤后的事件索引及数据子集。不破坏现有核心功能，仅作为自动化工作流 Pipeline 的一个新增步骤。
