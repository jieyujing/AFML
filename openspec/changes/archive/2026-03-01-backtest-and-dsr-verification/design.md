## Context

目前已生成离散化的头寸信号（Bet Sizes），但存在数值稳定性问题（部分 `-inf`）。我们需要将这些信号映射到实际的资产价格序列（Dollar Bars）上，以评估策略在风险调整后的收益表现。由于研发过程中涉及特征选择和模型调优，必须使用 DSR 来校正多重测试带来的偏误。

## Goals / Non-Goals

**Goals:**
- **稳定性修复**：消除头寸计算过程中的非数值（NaN/Inf）异常。
- **高性能回测**：支持对数收益率空间的向量化 PnL 计算。
- **统计验证**：计算 PSR（Probabilistic Sharpe Ratio）并进而获得 DSR（Deflated Sharpe Ratio）。
- **指标评估**：输出累计收益、最大回撤、夏普比率及 DSR 概率。

**Non-Goals:**
- **高胜率优化**：本阶段只负责验证现有逻辑，不涉及模型重新训练或选币逻辑修改。
- **实盘撮合模拟**：不进行深度订单簿模拟（Match Engine），使用中间价或收盘价进行点对点理想回测。

## Decisions

- **回测颗粒度**：基于 CUSUM 事件点及 Exit Time 构建的时间轴进行收益核算。
- **数值处理逻辑**：在 `get_concurrent_sizes` 中，若 `active_count` 为 0，则强制 `avg_size` 为 0；对于 `active_sum` 中可能产生的 `inf`，在最终输出前进行裁剪（Replace & Clip）。
- **DSR 算法实现**：
    - 使用收益序列的一阶、二阶、三阶和四阶矩（Mean, Std, Skew, Kurtosis）。
    - 记录当前研发流程中的 Trial 数量（预估 N=50）。
    - 计算 $\widehat{SR}_0$（基准夏普）并在相应置信度下折现。

## Risks / Trade-offs

- **[Risk] 回测偏误** → **[Mitigation]** 确保对齐价格数据时使用“下一个 Bar”的价格，避免引入 Look-ahead bias。
- **[Trade-off] 算力消耗** → **[Mitigation]** DSR 计算仅针对最终策略序列进行一次性计算，无需 JIT 加速。
