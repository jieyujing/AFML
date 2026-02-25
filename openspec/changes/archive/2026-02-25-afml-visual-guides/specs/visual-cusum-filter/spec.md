## ADDED Requirements

### Requirement: 绘制 CUSUM 过滤器的触发点在价格曲线上
系统需要能够直观地渲染一条覆盖较长周期的核心连续数值（如价格、收益率等），并在这些连续线上打点（Scatter）以标示 CUSUM filter 发生跳变的时刻。

#### Scenario: 跳变标记
- **WHEN** 给出基础价格序列和 CUSUM 事件的索引数组
- **THEN** 生成覆盖背景时间的价格折线图，并在给定索引对应的价格值上添加醒目的散点，证明它跳过了长时期的横盘期
