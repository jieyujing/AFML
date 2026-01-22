"""
Rule-Based Strategies for Hybrid AFML Architecture.

This module implements deterministic strategies (like Moving Average Crossover)
to serve as the Primary Model in a Meta-Labeling architecture.

Key difference from ML models:
- Logic is deterministic and interpretable.
- Focus is on High Recall (catching trends) rather than Precision.
"""

import pandas as pd
import numpy as np

class MovingAverageCrossStrategy:
    """
    Classic Fast/Slow MA Crossover Strategy.
    """
    
    def __init__(self, fast_window: int = 20, slow_window: int = 50):
        self.fast_window = fast_window
        self.slow_window = slow_window
        
    def generate_signals(self, close_series: pd.Series) -> pd.Series:
        """
        Generate signals: 1 (Long), -1 (Short), 0 (Neutral).
        
        Logic:
        - Long when Fast MA > Slow MA
        - Short when Fast MA < Slow MA
        - Signal is only generated at the CROSSOVER point (Event-based) 
          OR continuously held?
          
        For Meta-Labeling, we typically want EVENTS (Entry points).
        So we return non-zero only on the bar where the cross happens.
        """
        fast_ma = close_series.rolling(window=self.fast_window).mean()
        slow_ma = close_series.rolling(window=self.slow_window).mean()
        
        # Identify Regime
        regime = pd.Series(0, index=close_series.index)
        regime[fast_ma > slow_ma] = 1
        regime[fast_ma < slow_ma] = -1
        
        # Identify Crossovers (Change in Regime)
        # diff != 0 means a change occurred
        # We take the sign of the regime AFTER the change as the signal direction
        signal = regime.diff().fillna(0)
        
        # signal values will be:
        # 2 (from -1 to 1) -> Long Signal
        # -2 (from 1 to -1) -> Short Signal
        # 0 -> No Change
        
        events = pd.Series(0, index=close_series.index)
        events[signal > 0] = 1
        events[signal < 0] = -1
        
        return events

class BollingerBandsStrategy:
    """
    Mean Reversion Strategy using Bollinger Bands.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, close_series: pd.Series) -> pd.Series:
        """
        Signal:
        - Short when Price > Upper Band (Overbought)
        - Long when Price < Lower Band (Oversold)
        """
        ma = close_series.rolling(self.window).mean()
        std = close_series.rolling(self.window).std()
        
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        
        events = pd.Series(0, index=close_series.index)
        
        # Crossing Upper Band from below (Entry Short)
        # Using raw levels checks
        # Long Entry: Close < Lower
        # Short Entry: Close > Upper
        
        # To make it "Event-based", we can trigger on the break
        
        events[close_series < lower] = 1
        events[close_series > upper] = -1
        
        # Filter to only the FIRST break (to avoid continuous signals in a trend)
        # Shift 1 to check previous state
        prev_close = close_series.shift(1)
        prev_lower = lower.shift(1)
        prev_upper = upper.shift(1)
        
        # Long: Today < Lower AND Yesterday >= Lower
        long_entry = (close_series < lower) & (prev_close >= prev_lower)
        
        # Short: Today > Upper AND Yesterday <= Upper
        short_entry = (close_series > upper) & (prev_close <= prev_upper)
        
        final_events = pd.Series(0, index=close_series.index)
        final_events[long_entry] = 1
        final_events[short_entry] = -1
        
        return final_events
