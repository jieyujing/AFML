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
TARGET_DAILY_BARS = 6      # 目标每天 6 个 Bars（参数优化最优值）
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ============================================================
# Phase 2: Feature Engineering 参数
# ============================================================

# FracDiff 参数
FRACDIFF_THRES = 1e-4      # FFD 权重截断阈值
FRACDIFF_D_STEP = 0.05     # d 搜索步长
FRACDIFF_MAX_D = 1.0       # d 最大值

# CUSUM Filter 参数
CUSUM_WINDOW = 20          # 动态阈值滚动窗口
CUSUM_MULTIPLIER = 3       # 阈值乘数（控制事件率：越大事件越少）

# Phase 2 输出路径
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

# ============================================================
# Phase 3: Trend Scanning 参数
# ============================================================

# Trend Scanning 窗口范围（Bars 数量）
TREND_WINDOWS = [5, 10, 20, 30, 50]  # 短期趋势，适合期货

# ============================================================
# Phase 2: Feature Engineering 参数
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
    },

    # L2: 高级特征
    "theil_imbalance": {
        "enabled": True,
        "windows": [20, 60],
        "split_method": "clv",  # 'clv' (推荐), 'bvc', 'direction'
        "decomposed": False,     # True 返回完整分解 (T_buy, T_sell, T_ratio, T_diff)
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

    # Trend Scan 标签（含未来信息）
    "trend_scan": {
        "enabled": True,
        "windows": [5, 10, 20, 30, 50],
    },
}

# ============================================================
# Phase 4: SuperTrend Primary Model 参数
# ============================================================

SUPERTREND_CONFIG = {
    'params_combinations': [
        {'period': 7, 'multiplier': 2.0, 'name': 'fast'},
        {'period': 10, 'multiplier': 2.5, 'name': 'medium'},
        {'period': 14, 'multiplier': 3.0, 'name': 'slow'},
    ],
    'fusion_method': 'or',  # 'or' for High Recall, 'and' for High Precision
    'min_confidence': 0.33,  # 最小置信度阈值
}

# Primary Model 选择
PRIMARY_MODEL_TYPE = 'ma'  # 'lgbm' / 'supertrend' / 'ma' (MA 表现最优)

# ============================================================
# Phase 4: MA Primary Model 参数
# ============================================================

MA_PRIMARY_MODEL = {
    'ma_type': 'ewma',  # 'ewma' or 'sma'
    'span': 20,         # EWMA span 或 SMA window
}

# ============================================================
# Phase 4: TBM (Triple Barrier Method) 参数
# ============================================================

TBM_CONFIG = {
    'target_ret_col': 'feat_ewm_vol_20',  # 波动率估计列（log-return space）
    'profit_loss_barriers': (5.0, 2.0),    # (tp_mult, sl_mult) - 优化后参数
    'vertical_barrier_bars': 30,           # 最大持仓 bars 数量 - 优化后参数
    'min_ret': 0.001,                      # 最小收益门槛
    'min_close_time_sec': 60,              # 最小持仓时间（秒）
}

# ============================================================
# Phase 6: Meta Model 参数
# ============================================================

META_MODEL_CONFIG = {
    'precision_threshold': 0.50,  # 降低阈值提高召回率
    'n_estimators': 1000,
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.05,
}