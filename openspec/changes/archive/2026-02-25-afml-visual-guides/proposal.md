<artifact id="proposal" change="afml-visual-guides">

# Proposal: AFML 核心概念的可视化指南

## Context
在证明和解说基于《Advances in Financial Machine Learning》的量化研究方法论时，传统的图表（如普通的 K 线图和简单的收益率饼图）无法有力反映出它的内在统计学价值。我们需要一套能够直观呈现其优越性（如降低异方差性、滤除震荡、路径依赖和样本重叠惩罚）的可视化展示手段，来辅助研究员做汇报、验证及复盘。

## Proposed Change
我们将使用 Python 的绘图库（如 matplotlib/plotly）创建 4 张核心图表（3 个概念 + 1 个避坑指南）及实现逻辑：
1. **Dollar Bars 频率 vs 传统 Bars**：同一折线图对比 Tick Bars 和 Dollar Bars 的日均生成数量（避免对比 Time Bars），证明 Dollar Bars 能提供稳定的统计特性。
2. **CUSUM Filter 无聊中的高光**：在长周期的价格折线图上，通过打点（Scatter）高亮超出动态阈值（h）的 Sampled Observations，展示去除了多少横盘垃圾时间。
3. **TBM 动态包厢（Triple Barrier Method）**：单次交易局部分开放大图。包含绿线（动态止盈）、红线（动态止损）及虚线（最大持仓时间右墙），用一条穿越的轨迹线高亮首次撞墙的锚点 `(t_i,1)`。
4. **隐藏的坑（并发与样本唯一性验证框架）**：针对重叠样本（Non-IID）的展示。提供并发标签数量（Concurrent Labels）的时间序列，叠加每个样本对应的唯一性权重惩罚曲线（Uniqueness），证明回测不自欺欺人。

## Capabilities
- visual-dollar-bars
- visual-cusum-filter
- visual-tbm-bounds
- visual-sample-uniqueness

## Acceptance Criteria
- [ ] 开发 `visual-dollar-bars`：绘制 Dollar Bars 和 Tick Bars 日生成量对比曲线。
- [ ] 开发 `visual-cusum-filter`：绘制带有 Scatter 的动态 CUSUM 采样点价格曲线及阈值演变图。
- [ ] 开发 `visual-tbm-bounds`：绘制 TBM 局部“三面墙”触碰判定示意图。
- [ ] 开发 `visual-sample-uniqueness`：绘制样本并发数曲线与对应的唯一性衰减权重。

## Impact
提供了一套“硬核”且直观的数据科学可视化指南，它能以最直观的方式说明我们选择 AFML 特处理方案的原因，并在展示研究成果时建立信服力。

</artifact>
