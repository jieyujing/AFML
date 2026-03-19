# MVP AFML Validation Notebook Design

## Overview

创建一个 Jupyter Notebook，使用 AFMLKit 实现完整的 AFML MVP 验证流程，验证棕榈油期货（P9999.XDCE）的趋势策略是否有效。

---

## Data Source

- **文件**: `data/csv/P9999.XDCE-2023-1-1-To-2026-03-11-1m.csv`
- **类型**: 1 分钟 OHLCV 期货数据
- **字段**: datetime, open, close, high, low, volume, open_interest, dominant_future
- **时间范围**: 2023-01-03 至 2026-03-11（约 3 年）

---

## Notebook Structure

### Section 1: Configuration

所有可调参数集中在 Notebook 开头：

```python
# Dollar Bars
TARGET_DAILY_BARS = 50      # 目标每天采样 Bar 数
EWMA_SPAN = 20              # 动态阈值 EWMA 平滑周期

# CUSUM Filter
CUSUM_THRESHOLD_MULT = 2.0  # 阈值 = sigma * multiplier
VOL_SPAN = 20               # 波动率 EWMA 平滑周期

# Trend Scanning
L_WINDOWS = [10, 20, 30, 50, 100]  # 候选窗口长度

# Triple Barrier
STOP_LOSS_MULT = 1.5        # 止损 = volatility * STOP_LOSS_MULT
TAKE_PROFIT_MULT = 1.5      # 止盈 = volatility * TAKE_PROFIT_MULT
MAX_HOLD_BARS = 50          # 最大持仓 Bar 数
MIN_CLOSE_TIME_SEC = 0.0    # 最小持仓时间（Bar 数据设为 0）
MIN_RET = 0.0               # 最小收益率阈值（不过滤）
```

### Section 2: Data Loading

1. 加载 CSV 文件
2. 解析 datetime 为 index
3. 计算 Dollar Volume = close * volume
4. 数据质量检查（缺失值、异常值）

### Section 3: Dynamic Dollar Bars

**方法**: 使用 AFMLKit 的 `DynamicDollarBarKit`

**步骤**:
1. 将 1min OHLCV 转换为伪 Tick 数据（每根 1min K 线视为一个 "tick"，成交额为 `close * volume`）
2. 应用 `DynamicDollarBarKit`，目标每天 50 根 Bar
3. 输出 Dollar Bars DataFrame（包含 OHLCV）

**验证**:
- 打印总 Bar 数和日均 Bar 数
- 检查是否接近目标 50 根/天

### Section 4: CUSUM Filter

**方法**: 使用 AFMLKit 的 `cusum_filter`

**步骤**:
1. 计算 Dollar Bars 的 log returns
2. 计算滚动波动率（EWMA，span=VOL_SPAN）
3. 设置动态阈值 = volatility * CUSUM_THRESHOLD_MULT
4. 应用 CUSUM filter 获取事件索引

**验证**:
- 打印事件总数
- 可视化事件在价格序列上的分布

### Section 5: Trend Scanning

**方法**: 使用 AFMLKit 的 `trend_scan_labels`

**步骤**:
1. 准备 close price Series（DatetimeIndex）
2. 准备事件时间点 DatetimeIndex
3. 调用 `trend_scan_labels(price_series, t_events, L_windows)`
4. 获取 DataFrame: `side` (+1/-1), `t_value`, `t1` (最优窗口)

**验证**:
- 打印 up/down 信号数量
- t-value 分布统计

### Section 6: Triple Barrier Method

**方法**: 使用 AFMLKit 的 `triple_barrier`

**步骤**:
1. 计算每个事件点的波动率 target（用于动态设置止盈止损宽度）
2. 计算平均 Bar 持续时间（秒）
3. 准备参数：
   - `horizontal_barriers = (STOP_LOSS_MULT, TAKE_PROFIT_MULT)`
   - `vertical_barrier = MAX_HOLD_BARS * avg_bar_duration_seconds`
   - `min_close_time_sec = MIN_CLOSE_TIME_SEC`
   - `min_ret = MIN_RET`
   - `side = None`（用于 side prediction 模式）
4. 调用 `triple_barrier()` 获取标签
5. 输出：labels, touch indices, returns

**验证**:
- 打印标签分布 (+1/-1/0)
- 计算各类平均持仓时间

### Section 7: Simple Backtest

**策略逻辑**:
- 在 Trend Scanning 信号方向上开仓
- 由 TBM 决定平仓时机
- 计算每笔交易收益

**计算**:
1. 每笔交易收益 = side * return
2. 累计收益曲线
3. Sharpe Ratio = mean(return) / std(return) * sqrt(annual_factor)
4. 最大回撤
5. 胜率

### Section 8: Visualization

生成以下图表：

1. **累计收益曲线**: 时间 vs 累计收益
2. **回撤曲线**: 时间 vs 回撤
3. **标签分布**: 饼图或柱状图
4. **持仓时间分布**: 直方图

---

## Output Format

Notebook 最后输出汇总表格：

```
╔══════════════════════════════════════════════════╗
║              MVP Validation Results               ║
╠══════════════════════════════════════════════════╣
║ Data Period:        2023-01-03 to 2026-03-11     ║
║ Dollar Bars:        3,245 根 (avg 49.2/day)      ║
║ CUSUM Events:       892 个                        ║
║ Trend Scan Signals: 487 up / 405 down            ║
║ TBM Labels:         +1:312, -1:289, 0:291        ║
╠══════════════════════════════════════════════════╣
║ Strategy Sharpe:    1.24                          ║
║ Max Drawdown:       18.3%                         ║
║ Win Rate:           51.9%                         ║
║ Total Trades:       892                           ║
╚══════════════════════════════════════════════════╝
```

---

## Technical Notes

### Dollar Bar from 1min OHLCV

由于原始数据是 1 分钟 K 线而非 Tick 数据，需要特殊处理。使用 `TradesData` 类的正确方式：

```python
from afmlkit.bar.data_model import TradesData

# 将每根 1min K 线视为一个 "tick"
# 成交额 = close * volume
timestamps = df.index.astype(np.int64).values  # nanoseconds
prices = df['close'].values
volumes = df['volume'].values

# 创建 TradesData 实例
trades_data = TradesData(
    ts=timestamps,
    px=prices,
    qty=volumes
)
```

### Triple Barrier Method 调用

`triple_barrier` 函数的完整签名和参数：

```python
from afmlkit.label.tbm import triple_barrier

labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
    timestamps=dollar_bars.index.astype(np.int64).values,  # Bar 时间戳（纳秒）
    close=dollar_bars['close'].values,                      # 收盘价序列
    event_idxs=event_indices,                                # 事件在 close 数组中的位置索引
    targets=volatility_targets,                              # 每个事件的波动率目标
    horizontal_barriers=(STOP_LOSS_MULT, TAKE_PROFIT_MULT), # (止损倍数, 止盈倍数)
    vertical_barrier=MAX_HOLD_BARS * avg_bar_duration_sec,  # 最大持仓时间（秒）
    min_close_time_sec=MIN_CLOSE_TIME_SEC,                  # 最小持仓时间
    side=None,                                               # None = side prediction 模式
    min_ret=MIN_RET                                          # 最小收益率阈值
)
```

### 计算平均 Bar 持续时间

Dollar Bars 的持续时间不固定，需要从数据计算：

```python
# 方法 1：从实际 Bar 时间戳计算
bar_timestamps = dollar_bars.index.astype(np.int64).values
avg_bar_duration_sec = np.mean(np.diff(bar_timestamps)) / 1e9  # 纳秒转秒

# 方法 2：基于交易日估算（备用）
# 假设每天交易 6.5 小时，目标 50 根 Bar
# avg_bar_duration_sec = 6.5 * 3600 / 50 ≈ 468 秒
```

### Volatility Estimation

使用 EWMA 波动率作为 TBM 的动态 target：

```python
returns = np.log(close).diff()
volatility = returns.ewm(span=VOL_SPAN).std()
```

### Annualization Factor

Dollar Bars 频率不固定，需要计算实际年度 Bar 数：

```python
total_days = (end_date - start_date).days
bars_per_year = total_bars / total_days * 252
```

---

## Dependencies

- afmlkit (本地包)
- pandas
- numpy
- matplotlib (可视化)
- seaborn (可选，美化图表)

---

## File Location

```
notebooks/MVP_AFML_Validation.ipynb
```

---

## Success Criteria

MVP 验证通过条件：

| 指标 | 阈值 |
|------|------|
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 30% |
| 标签分布 | 三类都有样本，无极端失衡 |
| Win Rate | > 45% |

如果未通过，输出诊断建议下一步调整方向。