## ADDED Requirements

### Requirement: 绘制 Dollar Bars 和 Tick Bars 的对比图
函数或系统应当提供一键式对比 Tick Bars（基于固定交易笔数）和 Dollar Bars（基于固定交易金额）生成时间频率的方法，展示后者的相对平滑性，并降低异方差性。

#### Scenario: 对比结果呈现
- **WHEN** 研究人员输入同周期的 Tick bars 和 Dollar Bars 时间戳或者触发点数据，并指定时间窗口聚合级别（比如 'D' 每天）
- **THEN** 系统应该生成折线图，同时展示并对比在每一个时段两者的生成数量，证明 Dollar Bars 分布更加平稳
