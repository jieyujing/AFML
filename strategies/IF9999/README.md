# IF9999 趋势跟踪策略

基于 AFML 方法论的沪深300股指期货趋势跟踪策略开发项目。

## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py    # Dollar Bars 构建与验证
├── 02_parameter_optimization.py # Dollar Bars 参数优化
├── 03_feature_engineering.py   # Phase 2: FracDiff + CUSUM
├── config.py                   # 配置参数
├── output/
│   ├── bars/                   # 生成的 Bar 数据
│   ├── features/               # Phase 2 特征输出
│   └── figures/                # 可视化图表
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
uv run python strategies/IF9999/03_feature_engineering.py
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

### 输出文件

| 文件 | 说明 |
|------|------|
| `fracdiff_series.parquet` | FracDiff 平稳化序列 |
| `fracdiff_params.parquet` | 最优 d 值和统计信息 |
| `cusum_events.parquet` | CUSUM 事件点索引 |

## 后续阶段

- Phase 3: 标签生成（Trend Scanning）← 下一步
- Phase 4: Meta-Labeling 模型训练

## 参考

- Marcos Lopez de Prado, *Advances in Financial Machine Learning*, 2018