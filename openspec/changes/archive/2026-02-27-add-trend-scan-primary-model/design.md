## Context

在 AFML 标准量化研发流程中，我们采用 CUSUM 滤波器进行了第一步的数据降噪，提取出了可能存在结构性改变的时间点（$t_{events}$）。当前，需要决定在这些事件发生时的多空方向（Side）。固定周期的动量指标（如双均线）存在参数脆弱性，无法适应多变的市场节奏。Trend-Scanning 方法（Marcos López de Prado, MLAM）通过向回扫描多个不同长度的时间窗口，并利用 t-statistic 最大化来动态识别显著的趋势方向和强度。

由于 $t_{events}$ 可以在几万到几十万个量级，而每个事件可能需要回扫数十个甚至上百个窗口（每个窗口又要执行 OLS 回归计算），这导致了 O(N * L * W) 的极高计算复杂度。原生的 Pandas 或 Statsmodels 无法满足实盘或大规模回测的性能要求，必须进行底层算法级别的优化设计。目前代码库中可能部分还是使用原始 pandas 循环或者低效实现，这在生产环境是不可接受的。

## Goals / Non-Goals

**Goals:**
- **高性能架构：** 提出一套基于 Numba JIT 的底层实现机制，使大规模 Tick/Bar 数据下的 Trend Scan 计算在毫秒~秒级内完成。
- **与 CUSUM 的无缝串联：** 将 CUSUM 生成的离散时间戳直接映射到原始价格数据索引上，作为回溯窗口扫描的需求触发点。
- **Side 与 t-value 输出体系：** 明确 Trend Scan 除了决定方向 `Side`，还强制输出强度 `t_value` 以供下游在计算 Sample Weights 时进行调用。
- **严格消除前视偏差：** 作为 Primary Model 时，其结构必须保证是基于历史数据的后向滑动（Backward-looking）。

**Non-Goals:**
- **剥离执行层逻辑：** 本设计不涉及订单执行、撮合引擎或交易滑点等。
- **剥离 Meta-Model 构建细节：** 次级模型所使用的具体算法（XGB/RandomForest）及超参数配置不在本设计范围内。

## Decisions

**1. 底层计算方式选型：Numba 与原生数学公式剥离**
- **决策**: 弃用 `statsmodels.api.OLS` 以及 `scipy.stats.linregress`，全面拥抱 Numba `@njit` 编译。在 Numba 环境中直接将 OLS 和 t-value 转化为原始的数学累加公式（协方差/方差、残差标准误推导），不涉及任何复杂的外部库对象。
- **原因**: 实验数据表明，调用一次 `statsmodels` 进行 OLS 需要的时间是纯 Numba 数组操作的上千倍。在穷举窗口搜索时（例如 100个事件 x 50个窗口选项），这个差距会被指数级放大。
- **替代方案**: 使用 NumPy `np.polyfit` 等向量化矩阵，但由于回扫窗口的不规整性（每个历史区段无法对齐成完美的二维张量），导致高内存浪费，同时 Numba 的单次扫描循环效率已逼近底层 C 语言。

**2. 扫描矩阵的分发策略：从 $t_{events}$ 出发的 Backward Scan**
- **决策**: 算法不针对整个全量时间序列直接划窗，而是仅在传入的 $t_{events}$ 数组位置点启动计算。由 $t_{events}[i]$ 向过去回溯 $[L_1, L_2, ..., L_k]$ 若干个不同的长度条数，获取切片数据并求对应斜率的 t 统计量。最后使用 `argmax(abs(t_values))` 选择最佳窗口。
- **原因**: 这杜绝了非事件时间点上庞大的无效计算，直接从需求端（CUSUM）出发索要数据。保证绝对的历史拟合。

**3. 副产品复用与下游整合链路**
- **决策**: 封装的模块 `TrendScan` 不能只吐出个 1 和 -1 的 Series，而应返回一个 DataFrame，列包含 `t1_window`（最佳时间窗口长度）、`t_value`（此时的极大化显著性值）、`side`（1 或 -1）。这种解耦可以在后续把 `t_value` 投入到三重屏障前，先做一次阈值清洗（如 $|t| < 1.5$ 则舍弃该信号）。
- **原因**: 实现 AFML 倡导的精细化强度刻画，有助于下游进一步控制 Meta-Labeling 样本的不纯度。

## Risks / Trade-offs

- **Risk 1**: Numba 在非均匀间距的时间序列（例如非连续的日期索引）上处理滑动窗口切片很容易报越界或者类型错配问题。
  - **Mitigation**: 严格要求进入 `TrendScan` Numba 核的必须是单纯的 1D numpy array（连续的 prices）。相关的索引映射应该在外层 Python/Pandas 对齐后再把纯 values `.to_numpy()` 传给 Numba 以避免类型陷阱。
  
- **Risk 2**: 极端无波动的连续横盘期间，所有回溯窗口的 OLS 方差趋近于 0，将导致 t-value 除零错或者无限大。
  - **Mitigation**: 在 Numba 函数底层加上极小值保护 `epsilon = 1e-12`，捕获零方差切片，并默认给这部分窗口返回 `t_value = 0` 及 `side = 0`。
  
- **Risk 3**: 初入坑的用户在第一次运行策略时可能会被长达数秒的 Numba JIT Compilation pause 惊扰到。
  - **Mitigation**: 编写在执行时的首行 `logger.info("JIT Compiling Trend Scan C cores...")` 作为提示，且不开启高危的 `parallel=True`（路径依赖与大量不均衡运算会导致线程频繁启停，overhead 反而不如紧凑单线程）。
