# IF9999 Dollar Bars 构建与验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 IF9999 股指期货构建动态阈值 Dollar Bars，通过三刀验证评估采样质量，生成可视化图表。

**Architecture:** 单脚本架构，将 1 分钟 OHLCV 数据转换为 tick-like 格式，使用 Numba 加速的动态阈值算法构建 Dollar Bars，然后进行独立性、同分布、正态性验证并输出图表。

**Tech Stack:** Python, pandas, numpy, numba, scipy (stats), matplotlib, AFMLKit

---

## 文件结构

| 文件 | 负责内容 |
|------|----------|
| `strategies/IF9999/config.py` | 配置参数集中管理 |
| `strategies/IF9999/01_dollar_bar_builder.py` | 主流程：数据加载 → Dollar Bars 构建 → 三刀验证 → 可视化 |
| `strategies/IF9999/output/bars/dollar_bars.parquet` | 输出的 Dollar Bars 数据 |
| `strategies/IF9999/output/figures/*.png` | 三张验证图表 |
| `strategies/IF9999/README.md` | 项目说明文档 |

---

## Task 1: 创建项目目录结构和配置文件

**Files:**
- Create: `strategies/IF9999/config.py`
- Create: `strategies/IF9999/output/bars/` (目录)
- Create: `strategies/IF9999/output/figures/` (目录)

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p strategies/IF9999/output/bars strategies/IF9999/output/figures
```

- [ ] **Step 2: 编写 config.py**

```python
"""
IF9999 策略配置文件
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_PATH = "/Users/link/Documents/AFMLKIT/data/csv/IF9999.CCFX-2020-1-1-To-2026-03-27-1m.csv"

# 合约参数
CONTRACT_MULTIPLIER = 300  # IF 每点 300 元人民币

# Dollar Bars 参数
TARGET_DAILY_BARS = 50     # 目标每天 50 个 Bars
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
```

- [ ] **Step 3: 验证配置文件可导入**

```python
from strategies.IF9999.config import DATA_PATH, CONTRACT_MULTIPLIER
print(f"DATA_PATH: {DATA_PATH}")
print(f"CONTRACT_MULTIPLIER: {CONTRACT_MULTIPLIER}")
```

- [ ] **Step 4: 提交**

```bash
git add strategies/IF9999/config.py
git commit -m "feat(IF9999): add project config file

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: 编写数据加载模块

**Files:**
- Create: `strategies/IF9999/01_dollar_bar_builder.py` (部分)

- [ ] **Step 1: 编写数据加载函数**

```python
"""
01_dollar_bar_builder.py - IF9999 Dollar Bars 构建与验证

流程:
1. 加载 1 分钟 OHLCV 数据
2. 转换为 tick-like 格式（每分钟作为一个采样点）
3. 计算动态阈值（EWMA 日均美元交易量 / 目标 Bar 数）
4. 构建 Dollar Bars
5. 三刀验证（独立性、同分布、正态性）
6. 可视化输出
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    DATA_PATH, CONTRACT_MULTIPLIER, TARGET_DAILY_BARS, EWMA_SPAN,
    ACF_LAGS, BARS_DIR, FIGURES_DIR
)


def load_if_data(data_path: str) -> pd.DataFrame:
    """
    加载 IF9999 1 分钟 OHLCV 数据。

    :param data_path: CSV 文件路径
    :returns: DataFrame with datetime index and OHLCV columns
    """
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # 计算美元交易量（价格 × 成交量 × 合约乘数）
    df['dollar_volume'] = df['close'] * df['volume'] * CONTRACT_MULTIPLIER

    print(f"✅ 加载完成: {len(df)} 行, 时间范围 {df.index.min()} ~ {df.index.max()}")
    return df
```

- [ ] **Step 2: 测试数据加载**

```python
df = load_if_data(DATA_PATH)
assert len(df) > 0
assert 'dollar_volume' in df.columns
print(f"日均美元交易量: {df['dollar_volume'].groupby(df.index.date).sum().mean():,.0f}")
```

- [ ] **Step 3: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py
git commit -m "feat(IF9999): add data loading function

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: 编写动态阈值计算模块

**Files:**
- Modify: `strategies/IF9999/01_dollar_bar_builder.py`

- [ ] **Step 1: 编写动态阈值计算函数**

```python
def compute_dynamic_thresholds(df: pd.DataFrame, target_daily_bars: int, ewma_span: int) -> pd.Series:
    """
    计算动态 Dollar Bar 阈值。

    阈值 = EWMA(日均美元交易量) / 目标每日 Bar 数

    :param df: OHLCV DataFrame with dollar_volume column
    :param target_daily_bars: 目标每日 Bar 数量
    :param ewma_span: EWMA 平滑窗口
    :returns: Series of daily thresholds indexed by date
    """
    # 计算每日总美元交易量
    daily_dollar = df['dollar_volume'].groupby(df.index.date).sum()
    daily_dollar.index = pd.to_datetime(daily_dollar.index)
    daily_dollar.name = 'daily_dollar'

    # 处理零值（非交易日）
    daily_dollar = daily_dollar.replace(0, np.nan).ffill()

    # EWMA 平滑
    daily_ewma = daily_dollar.ewm(span=ewma_span, min_periods=1).mean()

    # 计算阈值
    thresholds = daily_ewma / target_daily_bars
    thresholds.name = 'threshold'

    print(f"✅ 阈值计算完成: EWMA span={ewma_span}, target={target_daily_bars} bars/day")
    print(f"   平均阈值: {thresholds.mean():,.0f} 元/bar")
    return thresholds
```

- [ ] **Step 2: 测试阈值计算**

```python
thresholds = compute_dynamic_thresholds(df, TARGET_DAILY_BARS, EWMA_SPAN)
assert len(thresholds) > 0
print(f"阈值范围: {thresholds.min():,.0f} ~ {thresholds.max():,.0f}")
```

- [ ] **Step 3: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py
git commit -m "feat(IF9999): add dynamic threshold computation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 编写 Dollar Bars 构建模块

**Files:**
- Modify: `strategies/IF9999/01_dollar_bar_builder.py`

- [ ] **Step 1: 编写 Dollar Bars 构建函数（Numba 加速）**

```python
from numba import njit
from numba.typed import List as NumbaList


@njit(nogil=True)
def _dynamic_dollar_bar_indexer(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
) -> NumbaList:
    """
    动态阈值 Dollar Bar indexer。

    :param timestamps: 时间戳数组（纳秒）
    :param prices: 价格数组
    :param volumes: 成交量数组
    :param thresholds: 每个时间点对应的阈值数组
    :returns: Bar close 索引列表
    """
    n = len(prices)
    indices = NumbaList()
    indices.append(0)  # 第一个点作为起始

    cum_dollar = 0.0
    for i in range(n):
        cum_dollar += prices[i] * volumes[i] * CONTRACT_MULTIPLIER
        if cum_dollar >= thresholds[i]:
            indices.append(i)
            cum_dollar = cum_dollar - thresholds[i]

    return indices


def build_dollar_bars(df: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """
    从 1 分钟数据构建 Dollar Bars。

    将每分钟视为一个 tick，使用动态阈值构建 Bars。

    :param df: 1 分钟 OHLCV DataFrame
    :param thresholds: 每日阈值 Series
    :returns: Dollar Bars DataFrame
    """
    # 将阈值映射到每分钟
    df['date'] = df.index.normalize()
    threshold_map = thresholds.to_dict()
    df['threshold'] = df['date'].map(lambda x: threshold_map.get(pd.Timestamp(x), thresholds.mean()))
    df['threshold'] = df['threshold'].fillna(thresholds.mean())

    # 准备数组
    timestamps = df.index.astype(np.int64).values
    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    threshold_arr = df['threshold'].values.astype(np.float64)

    # 构建 Bar 索引
    close_indices = _dynamic_dollar_bar_indexer(timestamps, prices, volumes, threshold_arr)
    close_indices = np.array(close_indices, dtype=np.int64)

    # 从索引构建 OHLCV
    bars = _build_ohlcv_from_indices(df, close_indices)

    print(f"✅ Dollar Bars 构建完成: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def _build_ohlcv_from_indices(df: pd.DataFrame, close_indices: np.ndarray) -> pd.DataFrame:
    """
    从 close 索引构建 OHLCV DataFrame。
    """
    if len(close_indices) < 2:
        return pd.DataFrame()

    bars_data = []
    for i in range(len(close_indices) - 1):
        start_idx = close_indices[i]
        end_idx = close_indices[i + 1]

        segment = df.iloc[start_idx:end_idx + 1]

        bar = {
            'timestamp': df.index[end_idx],
            'open': segment['open'].iloc[0],
            'high': segment['high'].max(),
            'low': segment['low'].min(),
            'close': segment['close'].iloc[-1],
            'volume': segment['volume'].sum(),
            'dollar_volume': segment['dollar_volume'].sum(),
            'n_ticks': len(segment),
        }
        bars_data.append(bar)

    bars_df = pd.DataFrame(bars_data)
    bars_df = bars_df.set_index('timestamp')
    return bars_df
```

- [ ] **Step 2: 测试 Dollar Bars 构建**

```python
bars = build_dollar_bars(df, thresholds)
assert len(bars) > 0
assert 'close' in bars.columns

# 验证每日 Bar 数量
daily_counts = bars.groupby(bars.index.date).size()
print(f"平均每日 Bar 数: {daily_counts.mean():.1f} (目标: {TARGET_DAILY_BARS})")
```

- [ ] **Step 3: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py
git commit -m "feat(IF9999): add Dollar Bars builder with Numba acceleration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 编写三刀验证模块

**Files:**
- Modify: `strategies/IF9999/01_dollar_bar_builder.py`

- [ ] **Step 1: 编写三刀验证函数**

```python
from statsmodels.stats.diagnostic import acorr_ljungbox


def compute_returns(bars: pd.DataFrame, log: bool = True) -> pd.Series:
    """
    计算收益率序列。

    :param bars: Bar DataFrame with close column
    :param log: 是否使用对数收益率
    :returns: 收益率 Series
    """
    if log:
        returns = np.log(bars['close']).diff().dropna()
    else:
        returns = bars['close'].pct_change().dropna()
    returns.name = 'returns'
    return returns


def validate_independence(returns: pd.Series, acf_lags: list = [1, 5, 10]) -> dict:
    """
    第一刀：独立性验证（序列相关性检验）。

    :param returns: 收益率序列
    :param acf_lags: 自相关检验滞后阶数
    :returns: 验证结果字典
    """
    # AC1 计算
    ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]

    # Ljung-Box 检验
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_pvalue = lb_result['lb_pvalue'].values[0]

    result = {
        'AC1': ac1,
        'Ljung_Box_p': lb_pvalue,
        'pass': abs(ac1) < 0.05 and lb_pvalue > 0.05,
        'description': f"AC1={ac1:.4f} (目标≈0), LB p={lb_pvalue:.4f} (目标>0.05)"
    }
    return result


def validate_identically_distributed(returns: pd.Series) -> dict:
    """
    第二刀：同分布验证（方差的方差检验）。

    :param returns: 收益率序列
    :returns: 验证结果字典
    """
    # 按月分组计算方差
    monthly_vars = returns.groupby(returns.index.to_period('M')).var()
    vov = monthly_vars.var()
    mean_var = monthly_vars.mean()

    result = {
        'VoV': vov,
        'mean_variance': mean_var,
        'VoV_ratio': vov / mean_var if mean_var > 0 else np.nan,
        'pass': vov < mean_var * 0.1,  # VoV < 均值的 10%
        'description': f"VoV={vov:.6f}, mean_var={mean_var:.6f}, ratio={vov/mean_var:.4f} (目标→0)"
    }
    return result


def validate_normality(returns: pd.Series) -> dict:
    """
    第三刀：正态性验证（Jarque-Bera 检验）。

    :param returns: 收益率序列
    :returns: 验证结果字典
    """
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Fisher 定义（正态=0）

    result = {
        'JB_stat': jb_stat,
        'JB_p': jb_pvalue,
        'Skewness': skew,
        'Kurtosis': kurt + 3,  # 转换为 Pearson 定义（正态=3）
        'pass': jb_pvalue > 0.05 or (abs(skew) < 0.5 and abs(kurt) < 1),
        'description': f"JB={jb_stat:.2f}, Skew={skew:.4f}, Kurt={kurt+3:.4f} (目标: Skew≈0, Kurt≈3)"
    }
    return result


def run_three_knife_validation(time_returns: pd.Series, dollar_returns: pd.Series) -> dict:
    """
    对 Time Bars 和 Dollar Bars 进行三刀验证对比。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :returns: 完整验证结果字典
    """
    results = {
        'time_bars': {
            'independence': validate_independence(time_returns),
            'identically_distributed': validate_identically_distributed(time_returns),
            'normality': validate_normality(time_returns),
        },
        'dollar_bars': {
            'independence': validate_independence(dollar_returns),
            'identically_distributed': validate_identically_distributed(dollar_returns),
            'normality': validate_normality(dollar_returns),
        }
    }

    # 打印对比结果
    print("\n" + "=" * 60)
    print("  三刀验证结果对比")
    print("=" * 60)
    print(f"\n第一刀（独立性）:")
    print(f"  Time Bars:  {results['time_bars']['independence']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['independence']['description']}")
    print(f"\n第二刀（同分布）:")
    print(f"  Time Bars:  {results['time_bars']['identically_distributed']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['identically_distributed']['description']}")
    print(f"\n第三刀（正态性）:")
    print(f"  Time Bars:  {results['time_bars']['normality']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['normality']['description']}")

    return results
```

- [ ] **Step 2: 测试三刀验证**

```python
time_returns = compute_returns(df.rename(columns={'close': 'close'}))
dollar_returns = compute_returns(bars)

validation_results = run_three_knife_validation(time_returns, dollar_returns)
```

- [ ] **Step 3: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py
git commit -m "feat(IF9999): add three-knife validation functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: 编写可视化模块

**Files:**
- Modify: `strategies/IF9999/01_dollar_bar_builder.py`

- [ ] **Step 1: 编写价格走势对比图**

```python
sns.set_theme(style="whitegrid", context="paper")


def plot_price_comparison(time_df: pd.DataFrame, dollar_bars: pd.DataFrame, save_path: str):
    """
    绘制 Time Bars vs Dollar Bars 价格走势对比图。

    :param time_df: 1 分钟 OHLCV DataFrame
    :param dollar_bars: Dollar Bars DataFrame
    :param save_path: 保存路径
    """
    # 采样 Time Bars 以提高可视化效率（每小时取一个点）
    time_sampled = time_df['close'].resample('H').last()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time Bars
    ax1.plot(time_sampled.index, time_sampled.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Time Bars (1-min, sampled hourly)', fontsize=12)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.tick_params(axis='x', rotation=45)

    # Dollar Bars
    ax2.plot(dollar_bars.index, dollar_bars['close'].values, color='darkorange', linewidth=0.8)
    ax2.set_title(f'Dollar Bars (target={TARGET_DAILY_BARS} bars/day)', fontsize=12)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存价格对比图: {save_path}")
```

- [ ] **Step 2: 编写三刀验证指标图**

```python
def plot_validation_metrics(time_returns: pd.Series, dollar_returns: pd.Series,
                             validation_results: dict, save_path: str):
    """
    绘制三刀验证指标对比图。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :param validation_results: 验证结果字典
    :param save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 8))

    # 子图 1: ACF 对比
    ax1 = fig.add_subplot(2, 2, 1)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(time_returns.dropna(), lags=20, ax=ax1, title='Time Bars ACF', color='steelblue')
    ax1.set_ylim(-0.15, 0.15)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_acf(dollar_returns.dropna(), lags=20, ax=ax2, title='Dollar Bars ACF', color='darkorange')
    ax2.set_ylim(-0.15, 0.15)

    # 子图 3: JB 统计量对比
    ax3 = fig.add_subplot(2, 2, 3)
    jb_values = [
        validation_results['time_bars']['normality']['JB_stat'],
        validation_results['dollar_bars']['normality']['JB_stat']
    ]
    bars_jb = ax3.bar(['Time Bars', 'Dollar Bars'], jb_values, color=['steelblue', 'darkorange'])
    ax3.set_title('Jarque-Bera Statistic (lower is better)')
    ax3.set_ylabel('JB Statistic')
    for bar, val in zip(bars_jb, jb_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                 ha='center', va='bottom', fontsize=10)

    # 子图 4: VoV 对比
    ax4 = fig.add_subplot(2, 2, 4)
    vov_values = [
        validation_results['time_bars']['identically_distributed']['VoV'],
        validation_results['dollar_bars']['identically_distributed']['VoV']
    ]
    bars_vov = ax4.bar(['Time Bars', 'Dollar Bars'], vov_values, color=['steelblue', 'darkorange'])
    ax4.set_title('Variance of Variances (VoV) (lower is better)')
    ax4.set_ylabel('VoV')
    for bar, val in zip(bars_vov, vov_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存验证指标图: {save_path}")
```

- [ ] **Step 3: 编写收益率分布直方图**

```python
def plot_return_distribution(time_returns: pd.Series, dollar_returns: pd.Series, save_path: str):
    """
    绘制收益率分布直方图对比。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :param save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Time Bars 分布
    ax.hist(time_returns.dropna(), bins=50, density=True, alpha=0.6,
            color='steelblue', label='Time Bars', edgecolor='white')

    # Dollar Bars 分布
    ax.hist(dollar_returns.dropna(), bins=50, density=True, alpha=0.6,
            color='darkorange', label='Dollar Bars', edgecolor='white')

    # 正态分布参考曲线
    from scipy.stats import norm
    x = np.linspace(dollar_returns.min(), dollar_returns.max(), 100)
    mean, std = dollar_returns.mean(), dollar_returns.std()
    ax.plot(x, norm.pdf(x, mean, std), 'k--', linewidth=1.5, label='Normal Reference')

    ax.set_title('Return Distribution Comparison')
    ax.set_xlabel('Log Returns')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存收益率分布图: {save_path}")
```

- [ ] **Step 4: 测试可视化模块**

```python
plot_price_comparison(df, bars, os.path.join(FIGURES_DIR, 'price_comparison.png'))
plot_validation_metrics(time_returns, dollar_returns, validation_results,
                        os.path.join(FIGURES_DIR, 'validation_metrics.png'))
plot_return_distribution(time_returns, dollar_returns,
                        os.path.join(FIGURES_DIR, 'return_distribution.png'))
```

- [ ] **Step 5: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py
git commit -m "feat(IF9999): add visualization functions for three-knife validation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: 编写主流程入口

**Files:**
- Modify: `strategies/IF9999/01_dollar_bar_builder.py`

- [ ] **Step 1: 编写 main 函数**

```python
def main():
    """
    IF9999 Dollar Bars 构建与验证主流程。
    """
    print("=" * 60)
    print("  IF9999 Dollar Bars 构建与验证")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs(BARS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载 IF9999 1 分钟数据...")
    df = load_if_data(DATA_PATH)

    # Step 2: 计算动态阈值
    print("\n[Step 2] 计算动态阈值...")
    thresholds = compute_dynamic_thresholds(df, TARGET_DAILY_BARS, EWMA_SPAN)

    # Step 3: 构建 Dollar Bars
    print("\n[Step 3] 构建 Dollar Bars...")
    bars = build_dollar_bars(df, thresholds)

    # Step 4: 保存 Dollar Bars
    bars_path = os.path.join(BARS_DIR, 'dollar_bars.parquet')
    bars.to_parquet(bars_path)
    print(f"✅ Dollar Bars 已保存: {bars_path}")

    # Step 5: 三刀验证
    print("\n[Step 5] 三刀验证...")
    time_returns = compute_returns(df)
    dollar_returns = compute_returns(bars)
    validation_results = run_three_knife_validation(time_returns, dollar_returns)

    # Step 6: 可视化
    print("\n[Step 6] 生成可视化图表...")
    plot_price_comparison(df, bars, os.path.join(FIGURES_DIR, 'price_comparison.png'))
    plot_validation_metrics(time_returns, dollar_returns, validation_results,
                           os.path.join(FIGURES_DIR, 'validation_metrics.png'))
    plot_return_distribution(time_returns, dollar_returns,
                            os.path.join(FIGURES_DIR, 'return_distribution.png'))

    print("\n" + "=" * 60)
    print("  流程完成")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  - Bars: {bars_path}")
    print(f"  - Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行完整流程测试**

```bash
uv run python strategies/IF9999/01_dollar_bar_builder.py
```

Expected output:
- 加载约 362,161 行数据
- 构建约 15,000+ Dollar Bars（6年 × 250天 × 50 bars）
- 生成 3 张图表
- 保存 parquet 文件

- [ ] **Step 3: 验证输出文件**

```bash
ls -la strategies/IF9999/output/bars/
ls -la strategies/IF9999/output/figures/
```

- [ ] **Step 4: 提交**

```bash
git add strategies/IF9999/01_dollar_bar_builder.py strategies/IF9999/output/
git commit -m "feat(IF9999): complete Dollar Bars builder with visualization

- Dynamic threshold Dollar Bars construction
- Three-knife validation (AC1, VoV, JB)
- Price comparison, validation metrics, return distribution charts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: 编写 README 文档

**Files:**
- Create: `strategies/IF9999/README.md`

- [ ] **Step 1: 编写 README.md**

```markdown
# IF9999 趋势跟踪策略

基于 AFML 方法论的沪深300股指期货趋势跟踪策略开发项目。

## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py    # Dollar Bars 构建与验证
├── config.py                   # 配置参数
├── output/
│   ├── bars/                   # 生成的 Bar 数据
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
| TARGET_DAILY_BARS | 50 | 目标每日 Bar 数量 |
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

## 后续阶段

- Phase 2: 特征工程（FracDiff、CUSUM Filter）
- Phase 3: 标签生成（Trend Scanning）
- Phase 4: Meta-Labeling 模型训练

## 参考

- Marcos López de Prado, *Advances in Financial Machine Learning*, 2018
```

- [ ] **Step 2: 提交**

```bash
git add strategies/IF9999/README.md
git commit -m "docs(IF9999): add project README

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 自检清单

- [x] Spec coverage: 设计文档所有章节已覆盖
- [x] Placeholder scan: 无 TBD、TODO 等占位符
- [x] Type consistency: 函数签名和变量名在各任务中一致