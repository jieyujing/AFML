"""
Y9999 豆油策略配置文件
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_PATH = "/Users/link/Documents/AFMLKIT/data/csv/Y9999.XDCE-2023-1-1-To-2026-03-11-1m.csv"

# 合约参数
CONTRACT_MULTIPLIER = 10  # 豆油每点 10 元人民币

# Dollar Bars 参数
TARGET_DAILY_BARS = 8      # 目标每天 6 个 Bars
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# ============================================================
# Phase 2: Feature Engineering 参数
# ============================================================

# FracDiff 参数
FRACDIFF_THRES = 1e-4
FRACDIFF_D_STEP = 0.05
FRACDIFF_MAX_D = 1.0

# CUSUM Filter 参数
CUSUM_WINDOW = 20
CUSUM_MULTIPLIER = 3

# ============================================================
# Phase 3: Trend Scanning 参数
# ============================================================

TREND_WINDOWS = [5, 10, 20, 30, 50]

# ============================================================
# Feature Engineering 参数
# ============================================================

FEATURE_CONFIG = {
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
    "trend_scan": {
        "enabled": True,
        "windows": [5, 10, 20, 30, 50],
    },
}

# ============================================================
# Phase 4: MA Primary Model 参数
# ============================================================

PRIMARY_MODEL_TYPE = 'ma'

MA_PRIMARY_MODEL = {
    'ma_type': 'ewma',
    'span': 20,
}

# ============================================================
# Phase 4: TBM 参数（最优）
# ============================================================

TBM_CONFIG = {
    'target_ret_col': 'feat_ewm_vol_20',
    'horizontal_barriers': (1.5, 2.5),  # 最优：止损 1.5, 止盈 2.5
    'vertical_barrier_bars': 50,
    'min_ret': 0.002,
    'min_close_time_sec': 60,
}

# ============================================================
# Phase 6: Meta Model 参数
# ============================================================

META_MODEL_CONFIG = {
    'precision_threshold': 0.50,
    'n_estimators': 1000,
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.01,
}