# IF9999 趋势跟踪策略

基于 AFML 方法论的沪深300股指期货趋势跟踪策略开发项目。

## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py     # Dollar Bars 构建与验证
├── 02_feature_engineering.py    # Phase 2: FracDiff + CUSUM
├── 03_trend_scanning.py         # Phase 3: Trend Scanning 标签
├── 04_primary_model_backtest.py # Phase 3.5: Primary Model 回测
├── config.py                    # 配置参数
├── output/
│   ├── bars/                    # 生成的 Bar 数据
│   ├── features/                # 特征和标签输出
│   └── figures/                 # 可视化图表
└── README.md
```

## 快速开始

```bash
# 构建 Dollar Bars 并生成验证图表
uv run python strategies/IF9999/01_dollar_bar_builder.py
```

## Dollar Bars 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TARGET_DAILY_BARS | 6 | 目标每日 Bar 数量（参数优化最优值） |
| EWMA_SPAN | 20 | 动态阈值 EWMA 窗口 |
| CONTRACT_MULTIPLIER | 300 | IF 合约乘数（每点300元） |

## 三刀验证

验证 Dollar Bars 是否更接近 I.I.D. Normal：

1. **独立性（第一刀）**：AC1 ≈ 0，Ljung-Box p > 0.05
2. **同分布（第二刀）**：方差的方差 VoV → 0
3. **正态性（第三刀）**：JB 统计量最低

## 数据源

- **品种**：IF9999.CCFX（沪深300股指期货主力合约）
- **频率**：1 分钟
- **范围**：2020-01-02 至 2026-03-27
- **数据量**：362,161 行

## Phase 2: 特征工程

运行特征工程流程：

```bash
uv run python strategies/IF9999/02_feature_engineering.py
```

### FracDiff 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FRACDIFF_THRES | 1e-4 | FFD 权重截断阈值 |
| FRACDIFF_D_STEP | 0.05 | d 搜索步长 |
| FRACDIFF_MAX_D | 1.0 | d 最大值 |

### CUSUM Filter 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CUSUM_WINDOW | 20 | 动态阈值滚动窗口 |
| CUSUM_MULTIPLIER | 3 | 阈值乘数（控制事件率） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `fracdiff_series.parquet` | FracDiff 平稳化序列 |
| `fracdiff_params.parquet` | 最优 d 值和统计信息 |
| `cusum_events.parquet` | CUSUM 事件点索引 |

## Phase 3: Trend Scanning 标签生成

运行 Trend Scanning 生成 Primary Model 输出：

```bash
uv run python strategies/IF9999/03_trend_scanning.py
```

### Trend Scanning 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TREND_WINDOWS | [5, 10, 20, 30, 50] | 窗口长度范围（Bars） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `trend_labels.parquet` | Trend Scanning 标签（t1, t_value, side） |

### 标签格式

| 列 | 类型 | 说明 |
|----|------|------|
| `t1` | Datetime | 趋势窗口结束时间 |
| `t_value` | float64 | OLS 斜率 t 统计量（样本权重） |
| `side` | int8 | +1（上涨）/ -1（下跌）/ 0（无趋势） |

### 可视化

| 文件 | 说明 |
|------|------|
| `03_trend_distribution.png` | side 和 t_value 分布 |
| `03_trend_example.png` | 趋势窗口示例（高 |t_value| 事件） |

## Phase 3.5: Primary Model 回测验证

运行 Primary Model 回测验证信号有效性：

```bash
uv run python strategies/IF9999/04_primary_model_backtest.py
```

### 回测方法

| 项目 | 说明 |
|------|------|
| 收益计算 | 点数收益 = side × (close_t1 - close_entry) |
| 持仓周期 | 使用 t1 作为平仓时间（Trend Scanning 最优窗口） |
| 分析维度 | 整体统计 + t_value 分位数分析 |

### 统计指标

| 指标 | 说明 |
|------|------|
| 胜率 | 盈利信号比例 |
| 平均收益 | 所有信号平均点数收益 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 总收益 | 所有信号累计点数 |

### t_value 分位数分析

验证假设：高 |t_value| 信号应有更好的表现。

| 分组 | 阈值 |
|------|------|
| Top 10% | |t_value| > 90% 分位数 |
| 10%-50% | |t_value| > 50% 分位数 |
| Bottom 50% | |t_value| ≤ 50% 分位数 |

### 可视化输出

| 文件 | 说明 |
|------|------|
| `04_pnl_distribution.png` | 收益分布直方图 |
| `04_cumulative_pnl.png` | 累积收益曲线 |
| `04_tvalue_vs_pnl.png` | |t_value| vs 收益散点图 |

## 后续阶段

- Phase 4: Meta-Labeling 模型训练

## 参考

- Marcos Lopez de Prado, *Advances in Financial Machine Learning*, 2018