# FeatureKit 完整因子列表

AFMLKit `feature` 模块提供的所有可用因子，按类别分类。

---

## 一、价格相关因子 (Price-based)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `Identity` | `input_col` | 返回原列值 | `input_col` |
| `Lag(periods)` | `lag{periods}` | 滞后值 | `periods`, `input_col` |
| `Return(periods)` | `ret{periods}` | 简单/对数收益率 | `periods`, `input_col`, `is_log` |
| `ReturnT(window)` | `ret{window}s` | 按时间窗口计算收益率 | `window`, `input_col`, `is_log` |
| `ROC(periods)` | `roc{periods}` | 变化率 | `periods`, `input_col` |
| `PctChange(window)` | `pctc{window}` | 百分比变化 | `window`, `input_col` |

---

## 二、移动平均线 (Moving Averages)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `SMA(window)` | `sma{window}` | 简单移动平均 | `window`, `input_col` |
| `EWMA(span)` | `ewma{span}` | 指数加权移动平均 | `span`, `input_col` |

---

## 三、动量指标 (Momentum)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `RSIWilder(window)` | `rsiw{window}` | RSI (Wilder方法) | `window`, `input_col` |
| `StochK(length)` | `stochk{length}` | 随机震荡 %K | `length`, `input_cols=[high,low,close]` |
| `ROC(periods)` | `roc{periods}` | 价格变化率 | `periods`, `input_col` |

---

## 四、波动率指标 (Volatility)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `RealizedVolatility(window)` | `rv{window}` | 实现波动率 | `window`, `input_col='ret'` |
| `EWMST(half_life)` | `ewms{half_life}s` | 指数加权移动标准差 | `half_life`, `input_col` |
| `BollingerPercentB(window)` | `bollb{window}` | 布林带 %B | `window`, `num_std`, `input_col` |
| `ParkinsonRange()` | `parkrange` | Parkinson 价格范围 | `input_cols=[high,low]` |
| `ATR(window)` | `atr{window}` | 平均真实范围 | `window`, `input_cols=[high,low,close]` |
| `VarianceRatio14(window)` | `var_ratio_1_4_{window}` | 方差比率 (1bar vs 4bar) | `window`, `input_col` |
| `KurtosisTransform(window)` | `kurt_{window}` | 滚动超额峰度 | `window`, `input_col` |
| `BiPowerVariation(window)` | `bv_{window}` | 双幂变差 (跳稳健) | `window`, `input_col` |

---

## 五、趋势指标 (Trend)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `ADX(length)` | `adx_{length}` | 平均方向指数 | `length`, `input_cols=[high,low,close]` |
| `TrendSlope(window)` | `trend_slope_{window}` | OLS 趋势斜率 (角度) | `window`, `input_col` |
| `HurstExponent(window)` | `hurst{window}` | 赫斯特指数 | `window`, `input_col` |

---

## 六、均值回归 (Mean Reversion)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `VWAPDistance(periods)` | `vwapd{periods}` | 与 VWAP 的距离 | `periods`, `is_log`, `input_cols=[close,volume]` |
| `MeanReversionZScore(window)` | `mr_z_{window}` | 均值回归 Z-Score | `window`, `input_col` |
| `ZScore(window)` | `z{window}` | 标准 Z-Score | `window`, `input_col` |
| `BurstRatio(window)` | `burst{window}` | Burst 比率 | `window`, `input_col` |

---

## 七、成交量相关 (Volume)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `FlowAcceleration(window)` | `flowacc_{window}_{recent}` | 流量加速 | `window`, `recent_periods`, `input_col` |
| `VPIN(window)` | `vpin_{window}` | 成交量同步知情交易概率 | `window`, `input_cols=[volume_buy,volume_sell]` |

---

## 八、相关性指标 (Correlation)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `PriceVolumeCorrelation(window)` | `corr_pv_{window}` | 价量滚动相关性 | `window`, `input_cols=[close,volume]` |

---

## 九、时间特征 (Time-based)

| 因子类 | 输出名 | 描述 |
|--------|--------|------|
| `TimeCues()` | 9个输出 | 时间编码特征 |
| `BarRate(window)` | `bars_per_hour` | K线生成速率 |
| `BarDuration(periods)` | `dur_{periods}bar` | K线持续时间 |
| `BarDurationEWMA(span)` | `dur_ewma{span}` | K线持续时间 EWMA |

**TimeCues 输出详情:**
- `sin_td`, `cos_td` — 日内时间正弦/余弦编码
- `sin_dw`, `cos_dw` — 周内日期正弦/余弦编码
- `asia`, `eu`, `us` — 交易时段标志
- `sess_x` — 交易时段转换标志
- `top_hr` — 整点标志

---

## 十、K线形态 (Candlestick)

| 因子类 | 输出数 | 描述 |
|--------|--------|------|
| `CandleShape()` | 4个 | K线形态指标 |

**输出:**
- `wick_up_ratio` — 上影线比例
- `wick_dn_ratio` — 下影线比例
- `body_ratio` — 实体比例
- `vwap_drift` — VWAP 漂移

---

## 十一、结构性断点 (Structural Break - AFML Ch.17)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `SADFTest` | `sadf` | 上确界 ADF (气泡检测) | `min_window`, `max_window`, `input_col` |
| `QADFTest` | `qadf` | 分位数 ADF | `window`, `quantile` |
| `CADFTest` | `cadf` | 条件 ADF | `min_window`, `max_window`, `quantile_window` |
| `SubMartingaleTest` | `sub_martingale` | 次鞅检验 | `decay`, `window`, `input_col` |
| `SuperMartingaleTest` | `super_martingale` | 超鞅检验 | `decay`, `window`, `input_col` |
| `CUSUMTest` | 6个输出 | CUSUM 结构性断点 | `window_size`, `warmup_period`, `max_age` |

**CUSUMTest 输出:**
- `{base}_score_up/down` — 上下突破分数
- `{base}_flag_up/down` — 突破标志
- `{base}_age_up/down` — 距上次突破年龄

---

## 十二、熵特征 (Entropy - AFML Ch.18)

| 因子类 | 输出名 | 描述 | 参数 |
|--------|--------|------|------|
| `ShannonEntropyTransform` | `shannon_entropy_{window}` | 香农熵 | `window`, `n_bins` |
| `LZEntropyTransform` | `lz_entropy_{window}` | Lempel-Ziv 熵 | `window`, `n_bins` |
| `KontoyiannisEntropyTransform` | `kontoyiannis_entropy_{window}` | Kontoyiannis LZ 熵 | `window`, `n_bins` |
| `EntropyImpliedVolTransform` | `entropy_vol_{window}` | 熵隐含波动率 | `window`, `n_bins`, `annualize` |
| `ApproximateEntropy` | `apen{window}` | 近似熵 | `window`, `m`, `tolerance` |

**熵值解释:**
- 低熵 (< 1 bit): 可预测的重复模式
- 高熵 (> 1.5 bits): 接近随机行为

---

## 十三、微观结构指标 (Microstructure)

| 因子类 | 输出名 | 描述 |
|--------|--------|------|
| `AmihudTransform` | `amihud_{window}` | Amihud 非流动性 |
| `RollSpreadTransform` | `roll_{window}` | Roll 价差估计 |
| `CorwinSchultzTransform` | `cs_{window}` | Corwin-Schultz 价差 |
| `ParkinsonVolatilityTransform` | `parkinson_{window}` | Parkinson 波动率 |

---

## 十四、序列相关 (Serial Correlation)

| 因子类 | 输出名 | 描述 |
|--------|--------|------|
| `SerialCorrelationTransform` | `serial_corr_lag{lag}` | 滚动序列相关性 |
| `LjungBoxTransform` | `ljung_box_{lags}` | Ljung-Box 统计量 |

---

## 十五、日内特征 (Intraday)

| 因子类 | 输出名 | 描述 |
|--------|--------|------|
| `DailyGap` | `daily_gap` | 隔夜跳空 |
| `ORBBreak` | `orb_long`, `orb_short` | 开盘区间突破 |

---

## 十六、其他高级因子

| 因子类 | 输出名 | 描述 |
|--------|--------|------|
| `CrossMARatioTransform` | `cross_ma_ratio_{short}_{long}` | 均线交叉比率 |
| `CrossMASignalTransform` | `cross_ma_signal_{short}_{long}` | 均线交叉信号 |
| `FracDiffExpandingTransform` | `frac_diff_{d}` | 展开式分数差分 |
| `FracDiffRollingTransform` | `frac_diff_roll_{d}` | 滚动分数差分 |
| `PCATransform` | `pca_{n_components}` | PCA 降维特征 |
| `DirRunLen` | `dir_run_len` | 方向性连续长度 |

---

## 使用示例

```python
from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.feature.transforms import (
    SMA, RSIWilder, ATR, RealizedVolatility,
    PriceVolumeCorrelation, VPIN, ADX,
    SADFTest, ShannonEntropyTransform
)

# 创建特征列表
features = [
    Feature(SMA(20, 'close')),
    Feature(RSIWilder(14, 'close')),
    Feature(ATR(14, input_cols=['high', 'low', 'close'])),
    Feature(RealizedVolatility(30, 'ret1')),
    Feature(PriceVolumeCorrelation(8)),
    Feature(VPIN(32)),
    Feature(ADX(14)),
    Feature(SADFTest(min_window=50, max_window=200, input_col='close')),
    Feature(ShannonEntropyTransform('ret1', window=50, n_bins=3)),
]

# 构建 FeatureKit
kit = FeatureKit(features, retain=['close', 'volume', 'high', 'low'])

# 执行计算
result_df = kit.build(df, backend='nb', timeit=True)
```

---

## 性能说明

- **Numba JIT**: 所有核心计算使用 Numba `@njit` 加速
- **双后端**: `backend='nb'` (Numba) 或 `'pd'` (Pandas)
- **缓存**: FeatureKit 自动缓存中间结果
- **数学运算**: Feature 对象支持 `+`, `-`, `*`, `/` 运算
