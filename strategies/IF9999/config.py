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
TARGET_DAILY_BARS = 4      # 目标每天 4 个 Bars（参数优化最优值）
EWMA_SPAN = 20             # 动态阈值 EWMA 窗口

# 验证参数
ACF_LAGS = [1, 5, 10]      # 自相关检验滞后阶数

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BARS_DIR = os.path.join(OUTPUT_DIR, "bars")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")