# IF9999 Dollar Bars 构建与验证设计文档

**日期**：2026-03-30
**项目**：strategies/IF9999/
**目标**：使用动态阈值构建 Dollar Bars，验证采样质量，为趋势跟踪策略奠定数据基础

---

## 1. 项目结构

```
strategies/IF9999/
├── 01_dollar_bar_builder.py    # 主脚本：数据加载 → Dollar Bars → 验证 → 可视化
├── config.py                   # 配置参数集中管理
├── output/
│   ├── bars/
│   │   └── dollar_bars.parquet # 输出的 Dollar Bars 数据
│   └── figures/
│       ├── price_comparison.png    # Time vs Dollar Bars 价格走势对比
│       ├── validation_metrics.png  # 三刀验证指标汇总图
│       └── return_distribution.png # 收益率分布直方图对比
└── README.md                   # 项目说明文档
```

---

## 2. 核心流程逻辑

### Step 1: 数据加载
```python
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df['dollar_volume'] = df['close'] * df['volume'] * 300  # IF每点300元
```

### Step 2: 动态阈值计算
```python
daily_dollar = df.groupby(df['datetime'].dt.date)['dollar_volume'].sum()
daily_ewma = daily_dollar.ewm(span=EWMA_SPAN).mean()
threshold_per_bar = daily_ewma / TARGET_DAILY_BARS  # 动态阈值
```

### Step 3: Dollar Bars 构建
使用 AFMLKit 已有的 `DynamicDollarBarKit` 类：
```python
from afmlkit.bar.kit import DynamicDollarBarKit
bars = DynamicDollarBarKit(trades, target_daily_bars=50, ewma_span=20)
```

### Step 4: 三刀验证
- **第一刀（独立性）**：AC1 ≈ 0，Ljung-Box p > 0.05
- **第二刀（同分布）**：方差的方差 VoV → 0
- **第三刀（正态性）**：JB 统计量最低

### Step 5: 可视化输出
生成三张核心验证图表。

---

## 3. 配置参数

```python
# config.py

# 数据路径
DATA_PATH = "/Users/link/Documents/AFMLKIT/data/csv/IF9999.CCFX-2020-1-1-To-2026-03-27-1m.csv"

# 合约参数
CONTRACT_MULTIPLIER = 300  # IF 每点 300 元

# Dollar Bars 参数
TARGET_DAILY_BARS = 50     # 目标每天 50 个 Bars
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数
JB_THRESHOLD = 0.05        # JB 检验 p-value 阈值（参考）

# 输出路径
OUTPUT_DIR = "strategies/IF9999/output"
```

**参数说明**：
- `TARGET_DAILY_BARS = 50`：高流动性期货，50 bars/day 是 AFML 推荐范围（20-50）
- `EWMA_SPAN = 20`：约 20 天平滑，适应月度流动性变化
- `CONTRACT_MULTIPLIER = 300`：沪深300股指期货标准乘数

---

## 4. 可视化图表设计

### 图表 1：价格走势对比 (`price_comparison.png`)
- 左侧：原始 Time Bars（1分钟）收盘价序列
- 右侧：构建的 Dollar Bars 收盘价序列
- 目的：直观对比两种采样方式的信息密度差异

### 图表 2：三刀验证指标 (`validation_metrics.png`)
- 上排：ACF 自相关图（Time vs Dollar Bars 对比）
- 下排：JB 统计量柱状对比 + VoV 数值标注
- 目的：量化验证 Dollar Bars 是否更接近 I.I.D. Normal

### 图表 3：收益率分布直方图 (`return_distribution.png`)
- Time Bars 收益率分布（蓝色）
- Dollar Bars 收益率分布（橙色）
- 叠加正态分布曲线作为参照
- 目的：直观展示正态性改善

**输出格式**：PNG 图片，分辨率 150 DPI，适合文档嵌入

---

## 5. 数据源说明

| 字段 | 说明 |
|------|------|
| datetime | 时间戳，1 分钟频率 |
| open/close/high/low | 价格字段 |
| volume | 成交量（手） |
| open_interest | 持仓量 |
| dominant_future | 当期主力合约代码 |

**数据范围**：2020-01-02 至 2026-03-27，共 362,161 行

---

## 6. 策略定位

本项目是 **IF9999 趋势跟踪策略** 的第一阶段，后续将扩展：
- Phase 2: 特征工程（FracDiff、CUSUM Filter）
- Phase 3: 标签生成（Trend Scanning）
- Phase 4: Meta-Labeling 模型训练

当前阶段聚焦数据采样质量验证。