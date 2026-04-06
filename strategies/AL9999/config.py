"""
AL9999 策略配置文件
铝期货（上期所）AFML 工作流配置
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_PATH = "/Users/link/Documents/AFMLKIT/data/csv/AL9999.XSGE-2020-1-1-To-2026-04-02-1m.csv"

# ============================================================
# 合约参数
# ============================================================

CONTRACT_MULTIPLIER = 5  # 吨/手（铝期货）
# 注：dollar_volume = close * volume * 5（价格元/吨 × 手数 × 5吨/手）

# ============================================================
# 交易日定义（夜盘归属下一交易日）
# ============================================================

def map_to_trading_date(dt):
    """
    将时间戳映射到交易日期。

    国内期货习惯：夜盘（21:00-01:00）归属下一交易日。

    :param dt: pd.Timestamp 或 datetime
    :returns: 交易日期（pd.Timestamp normalized）
    """
    import pandas as pd
    hour = dt.hour
    if hour >= 21:  # 夜盘 21:00-24:00
        return pd.Timestamp(dt).normalize() + pd.Timedelta(days=1)
    elif hour < 3:  # 夜盘凌晨 00:00-03:00（已属下一交易日）
        return pd.Timestamp(dt).normalize()
    else:  # 日盘
        return pd.Timestamp(dt).normalize()

# ============================================================
# Dollar Bars 参数
# ============================================================

TARGET_DAILY_BARS = 15      # 目标每天 6 个 Bars（初始值，优化后调整）
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数

# ============================================================
# 输出路径
# ============================================================

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ============================================================
# Phase 2: Feature Engineering 参数
# ============================================================

# FracDiff 参数
FRACDIFF_THRES = 1e-4      # FFD 权重截断阈值
FRACDIFF_D_STEP = 0.05     # d 搜索步长
FRACDIFF_MAX_D = 1.0       # d 最大值

# CUSUM Filter 参数
CUSUM_WINDOW = 20          # 动态阈值滚动窗口
CUSUM_MULTIPLIER = 5       # 阈值乘数（控制事件率：越大事件越少）

# ============================================================
# Phase 3: Trend Scanning 参数
# ============================================================

TREND_WINDOWS = [5, 10, 15, 20]  # 趋势窗口: 5/10/15/20 bars ≈ 0.3/0.7/1.0/1.3 个交易日
MIN_T_VALUE = 1.5            # t-value 显著性门槛（过滤纯噪音事件）

# ============================================================
# Phase 2b: tsfresh 特征参数
# ============================================================

TSFRESH_CONFIG = {
    "enabled": True,
    "lookback": 40,                           # 回看 bars 数量
    "fracdiff_cols": ["close", "log_close"],  # 只对这些列做 fracdiff
    "zscore_windows": [5, 10, 20],  # 需 < lookback(40)
    "features": [
        "mean",
        "median",
        "standard_deviation",
        "skewness",
        "kurtosis",
        "minimum",
        "maximum",
        "abs_energy",
        "mean_change",
        "mean_abs_change",
        "count_above_mean",
        "count_below_mean",
        "first_location_of_maximum",
        "first_location_of_minimum",
        "last_location_of_maximum",
        "last_location_of_minimum",
    ],
}

# ============================================================
# Feature Engineering 配置
# ============================================================

FEATURE_CONFIG = {
    # L1: 基础特征
    "momentum": {
        "enabled": True,
        "rsi_windows": [7, 14, 21],
        "roc_periods": [5, 10, 20],
    },
    "volatility": {
        "enabled": True,
        "ewm_spans": [10, 20, 40],
    },
    "trend": {
        "enabled": True,
        "adx_length": 14,
    },
    "reversion": {
        "enabled": True,
        "vwap_windows": [20, 60],
    },
    "time_cues": {
        "enabled": True,
        "has_night_session": True,  # 铝期货有夜盘
    },

    # L2: 高级特征
    "open_interest": {
        "enabled": True,  # 铝期货有持仓量数据
        "windows": [5, 20, 60],
    },
    "theil_imbalance": {
        "enabled": True,
        "windows": [20, 60],
        "split_method": "clv",
        "decomposed": False,
    },
    "microstructure": {
        "enabled": True,
        "windows": [20, 60],
    },
    "cross_ma": {
        "enabled": True,
        "fast_slow_pairs": [(5, 20), (10, 40)],
    },
    "entropy": {
        "enabled": True,
        "windows": [50, 100],
    },

    # L3: 结构特征
    "serial_corr": {
        "enabled": True,
        "lags": [1, 5, 10],
        "ljung_box_window": 20,
    },
    "structural_break": {
        "enabled": True,
        "adf_window": 100,
        "sadf_min_window": 10,
        "sadf_max_window": 60,
    },

    # L4: 成交量分布特征
    "volume_distribution": {
        "enabled": True,
        "frequency": "h",
        "n_bins": 5,
    },

    # Trend Scan 标签（含未来信息，仅用于分析）
    "trend_scan": {
        "enabled": True,
        "windows": [5, 10, 15, 20],
    },
}

# ============================================================
# Phase 4: MA Primary Model 参数
# ============================================================

# PRIMARY_MODEL_TYPE: 'ma' | 'cusum_direction' | 'rf' | 'dma'
# - 'ma': price > MA → long, price < MA → short
# - 'cusum_direction': g_up >= |g_down| → long, else → short
# - 'rf': load side from RF primary model output
# - 'dma': dual moving average crossover at CUSUM events
PRIMARY_MODEL_TYPE = 'dma'

# Phase 4: DMA Primary Model 参数
DMA_PRIMARY_CONFIG = {
    'fast_window': 5,
    'slow_window': 20,
    'ma_type': 'ewma',  # 'ewma' or 'sma'
}

MA_PRIMARY_MODEL = {
    # MA 参数（PRIMARY_MODEL_TYPE='ma' 时使用）
    'ma_type': 'ewma',
    'span': 20,
    # CUSUM Direction 参数
    'cusum_z_min': 0.0,  # Z-score 强度过滤阈值（0=不过滤）
}

# Phase 4: RF Primary Model 参数
RF_PRIMARY_CONFIG = {
    'n_estimators': 1000,
    'max_features': 'sqrt',           # 'sqrt' 或 'log2'：释放特征联合效应
    'n_jobs': -1,
    'random_state': 42,
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.01,
    'holdout_months': 12,
    'feature_prefixes': [
        'feat_rsi_',
        'feat_roc_',
        'feat_stoch_',
        'feat_adx_',
        'feat_vwap_',
        'feat_cross_ma_',
        'feat_shannon_',
        'feat_lz_entropy_',
        'feat_hl_vol_',
    ],
    # Primary model 目标：High Recall，不漏趋势机会
    'min_t_value': 1.5,
    'max_samples_method': 'avgU',
    'sampling_method': 'sequential_bootstrap',          # 'avgU'（原生Bagging）或 'sequential_bootstrap'
    't1_col': 'exit_ts',
    # Primary model 目标：High Recall，不漏趋势机会——不设置信度深渊
    # 'prob_abyss' 已禁用 (None)，让 RF 对所有样本都输出 side
    'prob_abyss': None,
}

# ============================================================
# Phase 4b: TBM (Triple Barrier Method) 参数（已弃用，保留作为参照）
# ============================================================

TBM_CONFIG = {
    'target_ret_col': 'feat_ewm_vol_20',
    'profit_loss_barriers': (2.0, 2.0),
    'vertical_barrier_bars': 80,
    'min_ret': 0.002,
    'min_close_time_sec': 60,
}

# ============================================================
# Phase 6: Meta Model 参数
# ============================================================

META_MODEL_CONFIG = {
    'precision_threshold': 0.51,
    'n_estimators': 1000,
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.05,
    'holdout_months': 6,  # 保留最后 6 个月不参与训练，用于 OOS 验证
}

# ============================================================
# Filter-First 优化参数
# ============================================================

FILTER_FIRST_CONFIG = {
    'threshold_grid': [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
    'shrinkage_min': 0.15,
    'shrinkage_max': 0.50,
    'execution_guard': {
        'enabled': True,
        'min_hold_bars': 2,
        'cooldown_bars': 1,
        'reverse_confirmation_delta': 0.02,
    },
    'short_penalty_delta': 0.04,
    'side_mode': 'both_with_short_penalty',
}

# ============================================================
# 交易成本参数（商品期货）
# ============================================================

COMMISSION_RATE = 0.00005   # 约 5bp（商品期货高于股指）
SLIPPAGE_POINTS = 1.0       # 单边滑点（点数）
