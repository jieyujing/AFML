<artifact id="design" change="afml-visual-guides">

# Technical Design: AFML 可视化指南

## Context & Motivation
依据 `proposal` 中定义的策略，我们需要提供一套标准的图表方法来验证及演示《Advances in Financial Machine Learning》（AFML）中核心概念的价值。本设计文档（Design）重点阐述各可视化模块的技术实现选型、数据流向以及代码结构设计。

## Architecture & Module Structure
我们将新建一个单独的画图模块 `afmlkit.visualization` 或者在 `scripts/` 下创建一个汇总的可视化脚本。为了避免过度封装，我们首先在 `scripts/afml_visual_guides.py` 中编写直观且可复用的画图函数，随后可根据需要并入核心 `afmlkit` 仓库中。

### 采用的图形库
- **Matplotlib** + **Seaborn**: 由于要展示统计学特征和时间序列，`matplotlib` 配合 `seaborn` 的样式能够提供科研级的高质量图片输出。
- Pandas 内置的绘图功能辅助数据预处理。

## Component Details

### 1. Dollar Bars 频率生成图 (`visual_dollar_bars.py`)
- **数据输入**：原始的 Tick 数据或高频原始特征，提取其时间戳生成按天（或按小时）分布的统计。同时也需要计算出的 Dollar Bars 时间戳。
- **计算逻辑**：使用 `pandas.Series.resample('D').count()` 计算不同抽样方式下每天含有多少个 Bar。
- **可视化元素**：
  - 共用的 X 轴：时间（Date）。
  - 双 Y 轴或不同颜色折线：Tick Bars 生成数量走势与 Dollar Bars 生成数量走势。
- **预期效果**：Tick Bars 是剧烈波动的山峰，Dollar Bars 相对是一条平缓的直线。

### 2. CUSUM Filter 采样点散点图 (`visual_cusum_filter.py`)
- **数据输入**：价格时间序列、对数收益率、以及 CUSUM 过滤器触发的索引（事件点）。
- **计算逻辑**：将带有 CUSUM 触发点索引的时间序列与原始价格对齐。
- **可视化元素**：
  - 基础线图：价格走势曲线。
  - 散点（Scatter）：在价格曲线上标记出 CUSUM 触发的点。可以用绿色向上的三角形代表上界突破，红色向下的三角形代表下界突破。
- **预期效果**：震荡行情中基本没点，在出现单边或高波行情时连续打点。

### 3. 三重屏障 TBM "包厢"放大图 (`visual_tbm_bounds.py`)
- **数据输入**：某一次特定交易（TBM 样本点）附近的一段高频价格时间序列，加上 TBM 标记工具跑出的该点的顶部阈值标量、底部阈值标量、最长持有时间 `t3`。
- **可视化元素**：
  - 截取局部价格线 `[t0, t3 + margin]`，其中 `t0` 为建仓点。
  - **红条/绿条曲线**：如果在每个 point 上动态止盈止损在缩放，画出上下两条包络线。
  - **右墙垂直虚线**：在 `t3` (Max Holding Period) 位置画垂直线。
  - **碰撞点**：高亮价格轨迹首次碰到任何一堵墙的点。

### 4. 样本并发与唯一性验证图 (`visual_sample_uniqueness.py`)
- **数据输入**：每一个 TBM 样本的 `t0` 和 `t1`，以及依据 AFML 算法计算出的每个样本的 `Uniqueness (平均衰减权重)`。
- **计算逻辑**：计算在任意时刻 $t$，有多少个开启状态的 bars（并发数 $c_t$）。依据时间点还原并绘制。
- **可视化元素**：
  - 子图 1：时间序列下的并发数 $c_t$。
  - 子图 2：与之相关的样本唯一性散点图（或者直方图，X轴为事件发生的 t0，Y轴为该事件的唯一性权重值）。
- **预期效果**：当并发数飙升时，底下的样本唯一性权重显著下降，证明了去相关性干预的有效。

## Risks & Mitigations
- **数据量过大导致 OOM 或绘图极慢**：
  - **缓解措施**：在 CUSUM 或 Dollar bars 全局对比时，如果刻度是日内分钟级可能点数过多。在作图时需先按照 `每天` 聚合（resample），对于 TBM 只画**单例（Single Case Zoom-in）**。
- **缺乏实际数据**：
  - **缓解措施**：现有的 `outputs/dollar_bars/` 以及相关的 csv 将会被用来驱动这些画图代码。

</artifact>
