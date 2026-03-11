"""
Alpha158 Features - FFD-transformed feature engineering.

This module implements Qlib Alpha158-style features using Fractional Differentiation
(FFD) to create stationary, memory-preserving feature series.

Key design principles:
1. Apply FFD only to log_price to get stationary base series X_tilde
2. Use X_tilde directly for momentum features (no returns calculation)
3. Feed X_tilde into SMA/EMA/Rank operations (they don't increase differencing order)
4. Volume features use raw data (volume is already stationary)

All features use 'ffd_*' prefix for clear identification.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.volatility import ewms


# Default configuration
DEFAULT_VOLATILITY_SPANS = [5, 10, 20]
DEFAULT_MA_WINDOWS = [5, 10, 20]
DEFAULT_EMA_WINDOWS = [5, 10]
DEFAULT_RANK_WINDOW = 20
DEFAULT_FFD_THRES = 1e-4
DEFAULT_FFD_D_STEP = 0.05
