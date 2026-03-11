# WebUI Feature Engineering Replacement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `scripts/feature_engineering.py` 的完整特征工程流程替换到 WebUI 组件 `webapp/pages/03_feature_engineering.py` 中，实现与脚本一致的完整特征计算能力。

**Architecture:**
采用模块化函数设计，将特征计算逻辑封装为独立函数，在 WebUI 页面中调用。保持与脚本相同的特征计算顺序：(1) 对数收益率 → (2) 波动率特征 → (3) 动量特征 → (4) 分数阶差分 → (5) CUSUM 对齐 → (6) NaN 清理。使用 afmlkit 核心库函数确保计算一致性。

**Tech Stack:** Streamlit, pandas, numpy, numba, afmlkit (内部库), statsmodels

---

## Task 1: 验证 afmlkit 核心函数导入路径

**Files:**
- 验证：`afmlkit/feature/core/volatility.py`
- 验证：`afmlkit/feature/core/momentum.py`
- 验证：`afmlkit/feature/core/correlation.py`
- 验证：`afmlkit/feature/core/ma.py`
- 验证：`afmlkit/feature/core/frac_diff.py`

**Step 1: 测试导入所有需要的函数**

```bash
cd D:\PycharmProjects\AFMLKIT
uv run python -c "
from afmlkit.feature.core.volatility import ewms, parkinson_range, atr, bollinger_percent_b, variance_ratio_1_4_core
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.momentum import rsi_wilder, roc, stoch_k
from afmlkit.feature.core.correlation import rolling_price_volume_correlation
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
print('All imports successful')
"
```

**Step 2: 记录缺失的导入（如有）**

如果任何导入失败，需要先修复 afmlkit 的 `__init__.py` 导出。

**Step 3: 提交**

此任务为验证任务，无需提交代码。

---

## Task 2: 创建特征计算工具模块

**Files:**
- Create: `webapp/utils/feature_calculator.py`
- Test: `tests/webapp/test_feature_calculator.py`

**Step 1: 编写测试**

```python
# tests/webapp/test_feature_calculator.py
import numpy as np
import pandas as pd
import pytest
from webapp.utils.feature_calculator import (
    compute_log_returns,
    compute_volatility_features,
    compute_momentum_features,
    compute_fracdiff_features,
    align_features_with_cusum
)

def test_compute_log_returns():
    """测试对数收益率计算"""
    df = pd.DataFrame({'close': [100.0, 101.0, 102.0, 103.0]})
    result = compute_log_returns(df)
    assert 'log_return' in result.columns
    assert pd.isna(result['log_return'].iloc[0])
    assert not pd.isna(result['log_return'].iloc[1:])

def test_compute_volatility_features():
    """测试波动率特征计算"""
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'log_return': np.random.randn(100) * 0.01
    })
    result = compute_volatility_features(df, spans=[10, 50])
    assert 'vol_ewm_10' in result.columns
    assert 'vol_ewm_50' in result.columns
    assert 'vol_parkinson' in result.columns
    assert 'vol_atr_14' in result.columns
    assert 'vol_bb_pct_b_20' in result.columns
    assert 'trend_variance_ratio_20' in result.columns

def test_compute_momentum_features():
    """测试动量特征计算"""
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'volume': np.random.randint(1000, 2000, 100)
    })
    result = compute_momentum_features(df)
    assert 'ema_short' in result.columns
    assert 'ema_long' in result.columns
    assert 'rsi_14' in result.columns
    assert 'mom_roc_10' in result.columns
    assert 'mom_stoch_k_14' in result.columns
    assert 'corr_pv_10' in result.columns
    assert 'liq_amihud' in result.columns
    assert 'vol_rel_20' in result.columns

def test_compute_fracdiff_features():
    """测试分数阶差分特征计算"""
    df = pd.DataFrame({'close': np.cumsum(np.random.randn(200)) + 100})
    result, optimal_d = compute_fracdiff_features(df, thres=1e-4, d_step=0.05)
    assert 'ffd_log_price' in result.columns
    assert 0.0 <= optimal_d <= 1.0

def test_align_features_with_cusum(tmp_path):
    """测试特征与 CUSUM 事件对齐"""
    # 创建模拟特征数据
    dates = pd.date_range('2023-01-01', periods=100, freq='T')
    features_df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'feature1': np.random.randn(100)
    }, index=dates)
    features_df.index.name = 'timestamp'

    # 创建模拟 CUSUM 标签
    labels_df = pd.DataFrame({
        'bin': [0, 1, 0, 1],
        't1': dates[10:14],
        'avg_uniqueness': np.random.randn(4)
    }, columns=['bin', 't1', 'avg_uniqueness'])

    # 保存到临时文件
    cusum_path = tmp_path / "cusum_sampled.csv"
    labels_df.to_csv(cusum_path)

    # 测试对齐
    result = align_features_with_cusum(features_df, str(cusum_path))
    assert len(result) <= len(features_df)
    assert 'bin' in result.columns
```

**Step 2: 运行测试验证失败**

```bash
uv run pytest tests/webapp/test_feature_calculator.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'webapp.utils.feature_calculator'"

**Step 3: 创建特征计算器模块**

```python
# webapp/utils/feature_calculator.py
"""特征计算器 - 实现与 scripts/feature_engineering.py 一致的特征计算逻辑"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# afmlkit core imports
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.volatility import (
    ewms, parkinson_range, atr, bollinger_percent_b, variance_ratio_1_4_core
)
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.momentum import rsi_wilder, roc, stoch_k
from afmlkit.feature.core.correlation import rolling_price_volume_correlation


# ── Hyper-parameters ──────────────────────────────────────────────────
VOL_SPANS = [10, 50, 100]          # EWM volatility windows
EMA_SHORT_SPAN = 12                # Short EMA for crossover
EMA_LONG_SPAN = 26                 # Long EMA
RSI_WINDOW = 14                    # RSI lookback
FRACDIFF_THRES = 1e-4              # FFD weight truncation threshold
FRACDIFF_D_STEP = 0.05             # Step size for d optimisation


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算对数收益率"""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_volatility_features(
    df: pd.DataFrame,
    spans: List[int] = None
) -> pd.DataFrame:
    """
    计算波动率特征

    包含:
    - 多窗口 EWM 波动率 (vol_ewm_{span})
    - Parkinson 波动率 (vol_parkinson)
    - ATR (vol_atr_14)
    - Bollinger %B (vol_bb_pct_b_20)
    - 方差比率 (trend_variance_ratio_20)
    """
    spans = spans or VOL_SPANS
    df = df.copy()

    log_ret = df["log_return"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64) if "high" in df.columns else None
    low = df["low"].values.astype(np.float64) if "low" in df.columns else None

    # EWM Volatility
    for span in spans:
        col_name = f"vol_ewm_{span}"
        df[col_name] = ewms(log_ret, span)

    # Parkinson volatility (requires high/low)
    if high is not None and low is not None:
        df["vol_parkinson"] = parkinson_range(high, low)

    # ATR (requires high/low/close)
    if high is not None and low is not None:
        df["vol_atr_14"] = atr(high, low, close, window=14)

    # Bollinger %B
    df["vol_bb_pct_b_20"] = bollinger_percent_b(close, window=20, num_std=2.0)

    # Variance Ratio
    df["trend_variance_ratio_20"] = variance_ratio_1_4_core(
        close, window=20, ddof=1, ret_type="log"
    )

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算动量特征

    包含:
    - EMA short/long (ema_short, ema_long)
    - 对数价格距离 EMA (log_dist_ema_short, log_dist_ema_long)
    - EMA 差值 (ema_diff)
    - RSI (rsi_14)
    - ROC (mom_roc_10)
    - Stochastic %K (mom_stoch_k_14)
    - 价量相关性 (corr_pv_10)
    - Amihud 非流动性 (liq_amihud)
    - 相对成交量 (vol_rel_20)
    """
    df = df.copy()

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64) if "high" in df.columns else None
    low = df["low"].values.astype(np.float64) if "low" in df.columns else None
    volume = df["volume"].values.astype(np.float64) if "volume" in df.columns else None
    log_close = np.log(close)

    # EMA short/long
    ema_short = ewma(close, EMA_SHORT_SPAN)
    ema_long = ewma(close, EMA_LONG_SPAN)
    df["ema_short"] = ema_short
    df["ema_long"] = ema_long

    # Log price distance to EMAs
    df["log_dist_ema_short"] = log_close - np.log(ema_short)
    df["log_dist_ema_long"] = log_close - np.log(ema_long)

    # EMA difference
    df["ema_diff"] = ema_short - ema_long

    # RSI (Wilder)
    df["rsi_14"] = rsi_wilder(close, RSI_WINDOW)

    # ROC
    df["mom_roc_10"] = roc(close, period=10)

    # Stochastic %K (requires high/low)
    if high is not None and low is not None:
        df["mom_stoch_k_14"] = stoch_k(close, low, high, length=14)

    # Price-Volume Correlation (requires volume)
    if volume is not None:
        df["corr_pv_10"] = rolling_price_volume_correlation(close, volume, window=10)

    # Amihud Illiquidity (|Return| / Volume)
    if "log_return" in df.columns and volume is not None:
        df["liq_amihud"] = np.abs(df["log_return"]) / (df["volume"] * df["close"]) * 1e6

    # Relative Volume (Volume / SMA(Volume, 20))
    if volume is not None:
        vol_sma_20 = sma(volume, window=20)
        df["vol_rel_20"] = volume / vol_sma_20

    return df


def compute_fracdiff_features(
    df: pd.DataFrame,
    thres: float = FRACDIFF_THRES,
    d_step: float = FRACDIFF_D_STEP
) -> Tuple[pd.DataFrame, float]:
    """
    计算分数阶差分特征 (FFD)

    自动优化 d 参数，应用 FFD 到对数价格序列

    Returns:
        tuple: (包含 ffd_log_price 列的 DataFrame, 最优 d 值)
    """
    df = df.copy()
    log_price = np.log(df["close"])

    # 优化 d 参数
    optimal_d = optimize_d(log_price, thres=thres, d_step=d_step)

    # 应用 FFD
    ffd_series = frac_diff_ffd(log_price, d=optimal_d, thres=thres)
    ffd_series.name = "ffd_log_price"

    # 合并回 DataFrame
    df["ffd_log_price"] = ffd_series

    return df, optimal_d


def align_features_with_cusum(
    features_df: pd.DataFrame,
    cusum_path: str,
    label_cols: List[str] = None
) -> pd.DataFrame:
    """
    将特征对齐到 CUSUM 事件时间戳

    Args:
        features_df: 特征 DataFrame (DatetimeIndex)
        cusum_path: CUSUM 采样文件路径
        label_cols: 要对齐的标签列，默认包含 ['bin', 't1', 'avg_uniqueness', 'return_attribution']

    Returns:
        对齐后的 DataFrame
    """
    # 加载 CUSUM 标签
    labels_df = pd.read_csv(cusum_path, parse_dates=["timestamp"])
    labels_df = labels_df.set_index("timestamp").sort_index()

    # 默认标签列
    if label_cols is None:
        label_cols = ["bin", "t1", "avg_uniqueness", "return_attribution"]
        # 只保留实际存在的列
        label_cols = [c for c in label_cols if c in labels_df.columns]

    # 时间戳对齐
    common_idx = features_df.index.intersection(labels_df.index)
    aligned_features = features_df.loc[common_idx].copy()

    # 连接标签
    aligned = aligned_features.join(labels_df[label_cols], how="inner")

    return aligned


def purge_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理包含 NaN 的行

    Returns:
        tuple: (清理后的 DataFrame, 清理前的行数，清理后的行数)
    """
    n_before = len(df)
    df_clean = df.dropna()
    n_dropped = n_before - len(df_clean)
    return df_clean


def compute_all_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    cusum_path: Optional[str] = None,
    align_to_cusum: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    计算所有特征的入口函数

    Args:
        df: 输入 DataFrame (必须包含 close 列，可选 high/low/volume)
        config: 特征配置字典
        cusum_path: CUSUM 采样文件路径（如果 align_to_cusum=True）
        align_to_cusum: 是否对齐到 CUSUM 事件

    Returns:
        tuple: (特征矩阵 DataFrame, 元数据字典)
    """
    config = config or {}
    metadata = {}

    # 确保索引是 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    df = df.sort_index()

    # 1. 计算对数收益率
    df = compute_log_returns(df)

    # 2. 计算波动率特征
    vol_spans = config.get("volatility", {}).get("spans", VOL_SPANS)
    df = compute_volatility_features(df, spans=vol_spans)

    # 3. 计算动量特征
    df = compute_momentum_features(df)

    # 4. 计算分数阶差分
    fracdiff_config = config.get("fractional_diff", {})
    thres = fracdiff_config.get("threshold", FRACDIFF_THRES)
    d_step = fracdiff_config.get("d_step", FRACDIFF_D_STEP)
    df, optimal_d = compute_fracdiff_features(df, thres=thres, d_step=d_step)
    metadata["optimal_d"] = optimal_d

    # 5. 对齐到 CUSUM 事件（可选）
    if align_to_cusum and cusum_path:
        df = align_features_with_cusum(df, cusum_path)
        metadata["aligned_to_cusum"] = True
    else:
        metadata["aligned_to_cusum"] = False

    # 6. 清理 NaN
    n_before = len(df)
    df = purge_nan_rows(df)
    metadata["rows_before_clean"] = n_before
    metadata["rows_after_clean"] = len(df)
    metadata["rows_dropped"] = n_before - len(df)

    # 7. 最终统计
    metadata["final_shape"] = df.shape
    metadata["feature_columns"] = list(df.columns)

    if "bin" in df.columns:
        metadata["label_distribution"] = df["bin"].value_counts().to_dict()

    return df, metadata
```

**Step 4: 运行测试验证通过**

```bash
uv run pytest tests/webapp/test_feature_calculator.py -v
```

Expected: PASS (所有测试通过)

**Step 5: 提交**

```bash
git add webapp/utils/feature_calculator.py tests/webapp/test_feature_calculator.py
git commit -m "feat: add feature calculator module with full AFML feature set"
```

---

## Task 3: 更新特征工程页面 UI

**Files:**
- Modify: `webapp/pages/03_feature_engineering.py`
- Test: 手动测试 Streamlit 页面

**Step 1: 读取当前文件了解结构**

已读取，了解现有结构。

**Step 2: 修改导入和初始化**

```python
# 在文件顶部添加导入
from webapp.utils.feature_calculator import (
    compute_all_features,
    VOL_SPANS,
    FRACDIFF_THRES,
    FRACDIFF_D_STEP
)
```

**Step 3: 更新特征配置步骤**

替换现有的配置逻辑为：

```python
if step == "1. 特征配置":
    st.markdown("### 1️⃣ 特征配置")

    # 特征类别选择（保持现有）
    selected_categories = st.multiselect(
        "选择要计算的特征类别",
        options=list(FEATURE_CATEGORIES.keys()),
        default=["波动率", "动量"]
    )

    # 参数配置
    st.markdown("#### 参数配置")

    # 波动率参数
    with st.expander("波动率参数", expanded=True):
        vol_spans_input = st.text_input(
            "波动率窗口 (逗号分隔)",
            value="10,50,100",
            help="EWM 波动率计算的时间窗口，多个值用逗号分隔"
        )
        vol_spans = [int(x.strip()) for x in vol_spans_input.split(",") if x.strip().isdigit()]

    # 动量参数
    with st.expander("动量参数", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rsi_window = st.number_input("RSI 窗口", min_value=5, max_value=50, value=14)
        with col2:
            roc_period = st.number_input("ROC 周期", min_value=5, max_value=50, value=10)

    # 高级参数
    with st.expander("高级参数 (FFD)", expanded=False):
        frac_diff_enabled = st.checkbox("启用分数阶差分", value=True)
        col1, col2 = st.columns(2)
        with col1:
            frac_diff_thres = st.number_input(
                "FFD 阈值",
                min_value=0.00001,
                max_value=0.001,
                value=0.0001,
                format="%.5f",
                step=0.00001
            )
        with col2:
            frac_diff_d_step = st.number_input(
                "d 步长",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01
            )

    # CUSUM 对齐选项
    with st.expander("CUSUM 事件对齐", expanded=False):
        align_to_cusum = st.checkbox(
            "对齐到 CUSUM 事件时间戳",
            value=True,
            help="如果启用，特征将与 CUSUM 采样的事件时间戳对齐"
        )
        cusum_path = st.text_input(
            "CUSUM 文件路径",
            value="outputs/dollar_bars/cusum_sampled_bars.csv"
        )

    # 保存配置
    if st.button("保存特征配置"):
        feature_config = {
            'categories': selected_categories,
            'volatility': {'spans': vol_spans},
            'momentum': {
                'rsi_window': rsi_window,
                'roc_period': roc_period
            },
            'fractional_diff': {
                'enabled': frac_diff_enabled,
                'threshold': frac_diff_thres,
                'd_step': frac_diff_d_step
            },
            'cusum': {
                'align_enabled': align_to_cusum,
                'path': cusum_path
            }
        }
        SessionManager.update('feature_config', feature_config)
        st.success("特征配置已保存")
        st.json(feature_config)
```

**Step 4: 更新特征计算步骤**

替换现有的计算逻辑为调用 `compute_all_features`:

```python
elif step == "2. 特征计算":
    st.markdown("### 2️⃣ 特征计算")

    feature_config = SessionManager.get('feature_config')

    if not feature_config:
        st.warning("请先配置特征参数")
    else:
        st.markdown("#### 当前配置")
        st.json(feature_config)

        # 确认对话框
        if not st.button("开始计算特征"):
            st.stop()

        SessionManager.set_processing(True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        info_container = st.empty()

        try:
            # 准备数据
            df = bar_data.copy()
            status_text.text("准备数据...")
            progress_bar.progress(10)

            # 检查 CUSUM 文件是否存在
            cusum_path = feature_config.get('cusum', {}).get('path', '')
            align_enabled = feature_config.get('cusum', {}).get('align_enabled', False)

            if align_enabled and cusum_path:
                if not Path(cusum_path).exists():
                    st.warning(f"CUSUM 文件不存在：{cusum_path}，将使用连续特征模式")
                    align_enabled = False

            # 计算所有特征
            status_text.text("计算特征 (可能需要几分钟)...")
            progress_bar.progress(30)

            features_df, metadata = compute_all_features(
                df=df,
                config=feature_config,
                cusum_path=cusum_path if align_enabled else None,
                align_to_cusum=align_enabled
            )

            progress_bar.progress(80)
            status_text.text("计算完成!")

            # 显示元数据
            with st.expander("📊 计算摘要", expanded=True):
                st.metric("最优 FFD 参数 d", f"{metadata.get('optimal_d', 'N/A'):.4f}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("清理前行数", f"{metadata.get('rows_before_clean', 0):,}")
                with col2:
                    st.metric("清理后行数", f"{metadata.get('rows_after_clean', 0):,}")
                with col3:
                    st.metric("丢弃行数", f"{metadata.get('rows_dropped', 0):,}")

                if metadata.get('label_distribution'):
                    st.markdown("**标签分布:**")
                    label_dist = metadata['label_distribution']
                    for label, count in sorted(label_dist.items()):
                        total = sum(label_dist.values())
                        pct = count / total * 100
                        st.write(f"  bin={int(label):+d}: {count:,} ({pct:.1f}%)")

            progress_bar.progress(90)

            # 保存特征
            SessionManager.update('features', features_df)
            SessionManager.update('feature_metadata', metadata)
            SessionManager.set_processing(False)
            progress_bar.progress(100)

            st.success(f"✅ 特征计算完成! 共 {len(features_df)} 行，{len(features_df.columns)} 个特征")

            # 显示特征列
            st.markdown("#### 特征列表")
            feature_cols = [col for col in features_df.columns
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            st.write(feature_cols)

        except Exception as e:
            SessionManager.set_processing(False)
            st.error(f"特征计算失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())
```

**Step 5: 更新特征预览步骤**

```python
elif step == "3. 特征预览":
    st.markdown("### 3️⃣ 特征预览")

    features = SessionManager.get('features')
    metadata = SessionManager.get('feature_metadata')

    if features is None:
        st.warning("请先计算特征")
    else:
        # 显示元数据（如果有）
        if metadata:
            with st.expander("📊 计算元数据", expanded=False):
                st.json(metadata)

        # 数据统计
        st.markdown("#### 数据统计")
        st.dataframe(features.describe())

        # 特征相关性热力图
        st.markdown("#### 特征相关性")

        feature_cols = [col for col in features.columns
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                                      'bin', 't1', 'avg_uniqueness', 'return_attribution']]

        if len(feature_cols) > 1:
            corr_matrix = features[feature_cols].corr()
            st.dataframe(corr_matrix)

            # 热力图
            from components.charts import render_heatmap
            fig = render_heatmap(corr_matrix, title="特征相关性矩阵")
            st.plotly_chart(fig, use_container_width=True)

        # 特征分布
        st.markdown("#### 特征分布")

        selected_features = st.multiselect(
            "选择特征查看分布",
            feature_cols,
            default=feature_cols[:3] if feature_cols else []
        )

        if selected_features:
            for feat in selected_features:
                if feat in features.columns:
                    fig = render_line_chart(features, [feat], title=feat)
                    st.plotly_chart(fig, use_container_width=True)
```

**Step 6: 手动测试页面**

```bash
cd webapp
uv run streamlit run app.py
```

导航到特征工程页面，测试完整流程。

**Step 7: 提交**

```bash
git add webapp/pages/03_feature_engineering.py
git commit -m "feat: update feature engineering UI to use new calculator module"
```

---

## Task 4: 更新 SessionManager 支持元数据

**Files:**
- Modify: `webapp/session.py`
- Test: `tests/webapp/test_session.py` (如存在)

**Step 1: 添加 feature_metadata 到 KEYS**

```python
# 在 KEYS 列表中添加
KEYS = [
    'raw_data', 'bar_data', 'dollar_bars', 'features', 'labels', 'sample_weights',
    'bar_config', 'feature_config', 'label_config', 'model_config', 'backtest_config',
    'model', 'model_results', 'feature_importance', 'feature_metadata',  # 新增
    # ... 其余
]
```

**Step 2: 在 init_session 中初始化**

```python
# 在 init_session 方法中，feature_config 之后添加
elif key == 'feature_metadata':
    st.session_state[key] = {}
```

**Step 3: 提交**

```bash
git add webapp/session.py
git commit -m "feat: add feature_metadata to session state"
```

---

## Task 5: 添加端到端集成测试

**Files:**
- Create: `tests/webapp/test_feature_engineering_integration.py`

**Step 1: 编写集成测试**

```python
# tests/webapp/test_feature_engineering_integration.py
"""特征工程集成测试 - 验证完整流程"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from webapp.utils.feature_calculator import compute_all_features


@pytest.fixture
def sample_dollar_bars():
    """生成模拟 Dollar Bars 数据"""
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2023-01-01', periods=n, freq='5T')

    # 随机游走生成价格
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成 OHLCV
    data = {
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.002)),
        'close': prices,
        'volume': np.random.exponential(1000, n)
    }

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    return df


@pytest.fixture
def sample_cusum_labels(sample_dollar_bars):
    """生成模拟 CUSUM 标签"""
    # 选择部分时间戳作为事件
    event_idx = np.arange(50, len(sample_dollar_bars), 10)
    events = sample_dollar_bars.iloc[event_idx].copy()

    labels = pd.DataFrame({
        'timestamp': events.index,
        'bin': np.random.choice([-1, 0, 1], len(events)),
        't1': events.index + pd.Timedelta(minutes=30),
        'avg_uniqueness': np.random.randn(len(events)),
        'return_attribution': np.random.randn(len(events)) * 0.01
    })

    return labels


def test_compute_features_without_alignment(sample_dollar_bars):
    """测试不与其他对齐的特征计算"""
    config = {
        'volatility': {'spans': [10, 50]},
        'momentum': {'rsi_window': 14, 'roc_period': 10},
        'fractional_diff': {'enabled': True, 'threshold': 1e-4, 'd_step': 0.05}
    }

    result, metadata = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        align_to_cusum=False
    )

    # 验证特征列存在
    assert 'vol_ewm_10' in result.columns
    assert 'vol_ewm_50' in result.columns
    assert 'rsi_14' in result.columns
    assert 'ffd_log_price' in result.columns

    # 验证元数据
    assert 'optimal_d' in metadata
    assert metadata['rows_after_clean'] <= metadata['rows_before_clean']
    assert not metadata['aligned_to_cusum']


def test_compute_features_with_cusum_alignment(
    sample_dollar_bars,
    sample_cusum_labels,
    tmp_path
):
    """测试与 CUSUM 事件对齐的特征计算"""
    # 保存 CUSUM 标签到临时文件
    cusum_path = tmp_path / "cusum_labels.csv"
    sample_cusum_labels.to_csv(cusum_path, index=False)

    config = {
        'volatility': {'spans': [10]},
        'momentum': {'rsi_window': 14, 'roc_period': 10},
        'fractional_diff': {'enabled': True, 'threshold': 1e-4, 'd_step': 0.05},
        'cusum': {'align_enabled': True, 'path': str(cusum_path)}
    }

    result, metadata = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        cusum_path=str(cusum_path),
        align_to_cusum=True
    )

    # 验证对齐后的行数与 CUSUM 事件数一致
    assert len(result) == len(sample_cusum_labels)

    # 验证标签列存在
    assert 'bin' in result.columns
    assert 'avg_uniqueness' in result.columns

    # 验证元数据
    assert metadata['aligned_to_cusum']


def test_feature_calculator_with_missing_columns(sample_dollar_bars):
    """测试缺失某些列时的容错处理"""
    # 移除 high/low 列
    df = sample_dollar_bars.drop(columns=['high', 'low'])

    config = {}

    # 应该不抛出异常，只是跳过需要 high/low 的特征
    result, metadata = compute_all_features(df=df, config=config, align_to_cusum=False)

    # 验证不依赖 high/low 的特征存在
    assert 'vol_ewm_10' in result.columns
    assert 'rsi_14' in result.columns
    assert 'ffd_log_price' in result.columns

    # 验证依赖 high/low 的特征不存在或为 NaN
    assert 'vol_parkinson' not in result.columns
    assert 'mom_stoch_k_14' not in result.columns
```

**Step 2: 运行集成测试**

```bash
uv run pytest tests/webapp/test_feature_engineering_integration.py -v
```

Expected: PASS

**Step 3: 提交**

```bash
git add tests/webapp/test_feature_engineering_integration.py
git commit -m "test: add integration tests for feature calculator"
```

---

## Task 6: 更新文档和导出功能

**Files:**
- Modify: `webapp/pages/03_feature_engineering.py` (导出部分)
- Create: `webapp/FEATURE_ENGINEERING_GUIDE.md`

**Step 1: 创建特征工程指南**

```markdown
# WebUI 特征工程指南

## 概述

WebUI 特征工程模块实现了与 `scripts/feature_engineering.py` 一致的完整 AFML 特征计算流程。

## 特征列表

### 波动率特征
- `vol_ewm_{span}` - 多窗口 EWM 波动率
- `vol_parkinson` - Parkinson 波动率
- `vol_atr_14` - 平均真实波动幅度
- `vol_bb_pct_b_20` - Bollinger %B
- `trend_variance_ratio_20` - 方差比率

### 动量特征
- `ema_short` / `ema_long` - 短/长期 EMA
- `log_dist_ema_short` / `log_dist_ema_long` - 对数价格距离 EMA
- `ema_diff` - EMA 差值
- `rsi_14` - 相对强弱指标
- `mom_roc_{period}` - 变化率
- `mom_stoch_k_14` - 随机指标 %K
- `corr_pv_10` - 价量相关性

### 流动性特征
- `liq_amihud` - Amihud 非流动性
- `vol_rel_20` - 相对成交量

### 分数阶差分
- `ffd_log_price` - 分数阶差分对数价格 (自动优化 d 参数)

## 使用流程

1. **数据准备**: 确保已生成 Dollar Bars 数据
2. **特征配置**: 选择特征类别和参数
3. **CUSUM 对齐**: 可选，将特征对齐到事件时间戳
4. **特征计算**: 执行计算并查看摘要
5. **特征预览**: 查看统计、相关性、分布
6. **特征导出**: 导出为 CSV/Parquet/HDF5

## CUSUM 对齐

启用 CUSUM 对齐后，特征将与 `outputs/dollar_bars/cusum_sampled_bars.csv` 中的事件时间戳对齐。

如果 CUSUM 文件不存在，系统将自动降级为连续特征模式。

## 配置示例

```yaml
volatility:
  spans: [10, 50, 100]
momentum:
  rsi_window: 14
  roc_period: 10
fractional_diff:
  enabled: true
  threshold: 0.0001
  d_step: 0.05
cusum:
  align_enabled: true
  path: outputs/dollar_bars/cusum_sampled_bars.csv
```
```

**Step 2: 更新导出步骤**

确保导出功能包含元数据信息。

**Step 3: 提交**

```bash
git add webapp/FEATURE_ENGINEERING_GUIDE.md
git commit -m "docs: add feature engineering guide"
```

---

## Task 7: 最终验证和清理

**Files:**
- 全部修改文件

**Step 1: 运行完整测试套件**

```bash
uv run pytest tests/webapp/ -v
```

**Step 2: 验证 Streamlit 应用启动**

```bash
cd webapp
uv run streamlit run app.py --server.headless true
```

**Step 3: 代码风格检查**

```bash
uv run ruff check webapp/utils/feature_calculator.py webapp/pages/03_feature_engineering.py
uv run ruff format webapp/utils/feature_calculator.py webapp/pages/03_feature_engineering.py
```

**Step 4: 提交最终更改**

```bash
git commit -am "chore: final cleanup and verification"
```

---

## 完成检查清单

- [ ] Task 1: 验证所有 afmlkit 导入
- [ ] Task 2: 创建 feature_calculator 模块和测试
- [ ] Task 3: 更新特征工程页面 UI
- [ ] Task 4: 更新 SessionManager
- [ ] Task 5: 添加集成测试
- [ ] Task 6: 创建文档
- [ ] Task 7: 最终验证

---

## 依赖关系

```
afmlkit (internal)
├── feature.core.frac_diff
├── feature.core.volatility
├── feature.core.ma
├── feature.core.momentum
└── feature.core.correlation

webapp
├── utils.feature_calculator (new)
├── pages.03_feature_engineering (modified)
└── session (modified)
```
