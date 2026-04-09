"""
AL9999 CTA Trend Development - Simplified Config
基于 cta-trend-development skill 的纯规则版本
"""

import os

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 数据路径
# ============================================================
DATA_PATH = "/Users/link/Documents/AFMLKIT/data/csv/AL9999.XSGE-2020-1-1-To-2026-04-02-1m.csv"

# ============================================================
# 合约参数
# ============================================================
CONTRACT_MULTIPLIER = 5  # 吨/手（铝期货）

# ============================================================
# 交易日定义（夜盘归属下一交易日）
# ============================================================
def map_to_trading_date(dt):
    """
    将时间戳映射到交易日期。
    国内期货习惯：夜盘（21:00-01:00）归属下一交易日。
    """
    import pandas as pd
    hour = dt.hour
    if hour >= 21:
        return pd.Timestamp(dt).normalize() + pd.Timedelta(days=1)
    elif hour < 3:
        return pd.Timestamp(dt).normalize()
    else:
        return pd.Timestamp(dt).normalize()

# ============================================================
# Dollar Bars 参数（Step 0: 数据边界定义）
# ============================================================
TARGET_DAILY_BARS = 15      # 目标每天 bars 数（调整 dollar_bar 阈值用）
DOLLAR_BAR_THRESHOLD = None  # 运行时计算，基于目标 bars 数
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数（Step 6: 稳健性检查）
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数
VOL_WINDOW = 20            # 波动率窗口

# ============================================================
# 输出路径
# ============================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# 确保目录存在
for d in [OUTPUT_DIR, BARS_DIR, FIGURES_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Step 2: Primary Rule - DMA (双均线交叉)
# ============================================================
DMA_PRIMARY_CONFIG = {
    'fast_window': 5,
    'slow_window': 20,
    'ma_type': 'ewma',  # 'ewma' or 'sma'
}

# ============================================================
# Step 3: Exit Rules 参数（固定持有 / 追踪止损 / 反向信号）
# ============================================================
EXIT_CONFIG = {
    # 选项: 'fixed_hold', 'trailing_stop', 'reverse_signal', 'time_based'
    'type': 'reverse_signal',  # 默认使用反向信号出场
    'fixed_hold_bars': 20,      # 如果 fixed_hold，持有 bars 数
    'trailing_stop_atr_multiplier': 2.0,  # 如果追踪止损，ATR 倍数
    'time_max_hold_bars': 60,   # 最大持有时间（bar 数）
}

# ============================================================
# Step 4: Filter 参数（机制明确的过滤器）
# ============================================================
FILTER_CONFIG = {
    'enabled': True,
    # 选项: 'volatility_regime', 'trend_strength', 'time_filter'
    'type': 'volatility_regime',
    'volatility': {
        'lookback': 20,
        'high_threshold': 1.5,   # 高于平均波动 1.5 倍 → 过滤
        'low_threshold': 0.5,    # 低于平均波动 0.5 倍 → 过滤
    },
    'trend_strength': {
        'adx_threshold': 25,     # ADX > 25 才交易（趋势明确）
    },
}

# ============================================================
# Step 5: Position & Risk Control
# ============================================================
POSITION_CONFIG = {
    'position_size': 1.0,      # 固定仓位（手数）
    'max_position': 1.0,       # 最大仓位
    'max_daily_loss': None,    # 日最大亏损（金额），None=不限制
    'max_drawdown': 0.10,      # 最大回撤 10%
}

# ============================================================
# Step 6: Walk-Forward & Cost Pressure Test 参数
# ============================================================
WF_CONFIG = {
    'train_ratio': 0.70,       # 训练集比例
    'test_ratio': 0.30,        # 测试集比例（OOS）
    'n_splits': 5,             # Walk-forward 折数
    'embargo_bars': 5,         # 冷切期（bar 数）
}

COST_PRESSURE_CONFIG = {
    'commission_multipliers': [1.0, 1.5, 2.0],  # 成本压力测试
    'slippage_multipliers': [1.0, 2.0, 3.0],
}

# ============================================================
# Step 5: Primary Model Factory 参数
# ============================================================
PRIMARY_FACTORY_CONFIG = {
    'cusum_rates': [0.05, 0.10, 0.15],
    'fast_windows': [5, 8, 10, 12, 15],
    'slow_windows': [20, 30, 40, 50, 60],
    'pt_sl': 1.0,
    'vertical_bars': [10, 20, 30],
    'top_n_lightweight': 20,
    'top_n_final': 5,
    'oos_test_ratio': 0.30,
    'score_weights': {
        'effective_recall': 0.45,
        'turnover': -0.10,
        'uniqueness': 0.10,
        'sharpe': 0.20,
        'net_pnl': 0.10,
        'mdd': 0.05,
    },
    'k_search_min': 0.001,
    'k_search_max': 10.0,
    'k_tolerance': 1e-4,
}

# ============================================================
# Step 6: 参数扰动（Parameter Perturbation）
# ============================================================
PARAM_PERTURBATION = {
    'fast_window_delta': [-2, -1, 0, 1, 2],  # 主参数上下扰动
    'slow_window_delta': [-5, -2, 0, 2, 5],
    'threshold_delta': [-0.1, 0, 0.1],       # 过滤阈值扰动
}

# ============================================================
# 交易成本参数（商品期货）
# ============================================================
COMMISSION_RATE = 0.00005   # 约 5bp（商品期货）
SLIPPAGE_POINTS = 1.0       # 单边滑点（点数）

# ============================================================
# 最低验收标准（Step 7 通过标准）
# ============================================================
PASS_CRITERIA = {
    'trade_count_min': 30,
    'oos_windows_min': 3,
    'oos_sharpe_min': 0.0,
    'oos_dsr_min': 0.95,
    'sharpe_oos_vs_full': '>=',  # OOS Sharpe >= Full Sharpe
    'parameter_stability': True,  # 参数邻域不塌陷
}

print(f"[CTA CTA] 配置加载完成: {PROJECT_ROOT}")
