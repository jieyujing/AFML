<artifact id="tasks" change="afml-visual-guides">

## 1. 基础环境与工程搭建

- [x] 1.1 在 `scripts/` 目录下创建总入口文件 `afml_visual_guides.py`（或在 `afmlkit.visualization` 模块中创建，视具体组织结构而定）。
- [x] 1.2 引入必要的绘图库（`matplotlib.pyplot`, `seaborn`）及数据处理库（`pandas`, `numpy`）。
- [x] 1.3 准备基础的主题和样式配置（例如配置 seaborn 风格，确保科研级图表输出）。

## 2. 核心图表 1：Dollar Bars 与 Tick Bars 对比

- [x] 2.1 实现 `plot_dollar_vs_tick_bars` 函数，支持输入两组时间序列或特征数据。
- [x] 2.2 在该函数中对数据按照指定维度（如每天 'D'）进行 `resample` 汇总。
- [x] 2.3 利用双坐标轴（Twinx）或者同轴不同颜色折线图完成 Tick Bars 与 Dollar Bars 日生成频率走势的可视化呈现。

## 3. 核心图表 2：CUSUM Filter 触发点

- [x] 3.1 实现 `plot_cusum_filter_events` 函数，输入包括价格曲线序列及 CUSUM 生成的触发点索引列表。
- [x] 3.2 绘制底层的价格（或收益率）长周期折线图。
- [x] 3.3 使用 Scatter（如向上/向下对应不同颜色的 marker）将过滤得到的采样点标注在对应发生时间的价格线上。

## 4. 核心图表 3：TBM 动态包厢放大图

- [x] 4.1 实现 `plot_tbm_bounds_single_case` 函数，专为展示单个交易样本的微观视角设计。
- [x] 4.2 计算并绘制一条表示从 `t0` 开始真实价格轨迹的曲线。
- [x] 4.3 渲染三堵墙：动态绘制上界止盈墙（绿线）、下界止损墙（红线）、最大持有期右墙（垂直虚线_t3）。
- [x] 4.4 判定并用特殊标记符号（如 x 叉号或大圆点）高亮标出首次“碰墙”的 `(t_i)` 碰撞点。

## 5. 核心图表 4：样本并发与唯一性验证

- [x] 5.1 实现 `plot_sample_uniqueness_and_concurrency` 函数，输入 TBM label 输出的起止时间数组与对应的 Uniqueness 权重。
- [x] 5.2 根据起止时间轴计算在每一个对应时间点下开启状态的并发数 $c_t$ 数组。
- [x] 5.3 绘制包含双子图的设计：上子图绘制时间序列下的并发数量曲线，下子图对应时间打点展示样本权重的惩罚/衰减（Uniqueness 值），从而直观展示并发增加如何降低权重。

</artifact>
