import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from itertools import product

from .base import PrimaryModelBase, SignalResult


class DualMAStrategy(PrimaryModelBase):
    """双均线策略"""

    def __init__(
        self,
        short_range: Tuple[int, int] = (3, 10),
        long_range: Tuple[int, int] = (15, 50),
        step: int = 2,
        tp_ratio: float = 2.0,
        sl_ratio: float = 1.0,
        time_barrier: Optional[int] = None,
        vol_window: int = 20
    ):
        """
        :param short_range: 短期均线周期范围 (min, max)
        :param long_range: 长期均线周期范围 (min, max)
        :param step: 参数搜索步长
        :param tp_ratio: 止盈倍数（相对波动率）
        :param sl_ratio: 止损倍数（相对波动率）
        :param time_barrier: 时间屏障（事件数），None 表示不使用
        :param vol_window: 波动率计算窗口
        """
        self.short_range = short_range
        self.long_range = long_range
        self.step = step
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio
        self.time_barrier = time_barrier
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        return "双均线策略"

    @property
    def param_grid(self) -> Dict[str, List[int]]:
        """生成参数网格，确保 long > short"""
        short_vals = list(range(
            self.short_range[0],
            self.short_range[1] + 1,
            self.step
        ))
        long_vals = list(range(
            self.long_range[0],
            self.long_range[1] + 1,
            self.step
        ))

        # 过滤无效组合
        valid_combos = [
            (s, l) for s, l in product(short_vals, long_vals)
            if l > s
        ]

        return {
            'short_window': [c[0] for c in valid_combos],
            'long_window': [c[1] for c in valid_combos]
        }

    def generate_signals(
        self,
        data: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 20,
        **kwargs
    ) -> SignalResult:
        """
        生成双均线交叉信号

        :param data: CUSUM采样数据，必须包含 'price' 列
        :param short_window: 短期均线周期
        :param long_window: 长期均线周期
        """
        prices = data['price']

        # 计算均线
        ma_short = prices.rolling(window=short_window, min_periods=1).mean()
        ma_long = prices.rolling(window=long_window, min_periods=1).mean()

        # 持仓状态：短期均线在上则为多头
        position = pd.Series(
            np.where(ma_short > ma_long, 1, -1),
            index=prices.index
        )

        # 信号：持仓变化点
        signal_change = position.diff()
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[signal_change > 0] = 1   # 金叉做多
        signals[signal_change < 0] = -1  # 死叉做空

        # 计算波动率（用于 TBM 动态止盈止损）
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(
            window=self.vol_window,
            min_periods=1
        ).std()

        # 构建 TBM 输入数据
        events_df = data.copy()
        events_df['volatility'] = volatility

        # 计算 TBM 标签
        events_with_labels = self._compute_tbm_labels(
            events_df,
            positions=position,
            tp_ratio=self.tp_ratio,
            sl_ratio=self.sl_ratio,
            time_barrier=self.time_barrier
        )

        return SignalResult(
            signals=signals,
            positions=position,
            events_with_labels=events_with_labels
        )

    def _compute_tbm_labels(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        tp_ratio: float,
        sl_ratio: float,
        time_barrier: Optional[int]
    ) -> pd.DataFrame:
        """计算三重屏障法标签"""
        prices = data['price'].values
        volatility = data['volatility'].values

        results = []
        n = len(prices)

        for i in range(n - 1):
            entry_price = prices[i]
            entry_vol = volatility[i] if not np.isnan(volatility[i]) else 0.02
            direction = positions.iloc[i]  # 1 for long, -1 for short

            # Direction-aware TP/SL
            if direction > 0:  # Long position
                tp = entry_price * (1 + tp_ratio * entry_vol)  # Up = TP
                sl = entry_price * (1 - sl_ratio * entry_vol)  # Down = SL
            else:  # Short position
                tp = entry_price * (1 - tp_ratio * entry_vol)  # Down = TP
                sl = entry_price * (1 + sl_ratio * entry_vol)  # Up = SL

            label = 0
            exit_idx = n - 1

            max_j = min(i + time_barrier, n) if time_barrier else n
            for j in range(i + 1, max_j):
                if direction > 0:  # Long position
                    if prices[j] >= tp:
                        label = 1   # TP hit
                        exit_idx = j
                        break
                    elif prices[j] <= sl:
                        label = -1  # SL hit
                        exit_idx = j
                        break
                else:  # Short position (direction < 0)
                    if prices[j] <= tp:
                        label = 1   # TP hit (price down)
                        exit_idx = j
                        break
                    elif prices[j] >= sl:
                        label = -1  # SL hit (price up)
                        exit_idx = j
                        break
            else:
                # Time barrier or end of data
                if time_barrier and i + time_barrier < n:
                    exit_idx = i + time_barrier
                    # Direction-aware label for time barrier
                    if direction > 0:
                        label = 1 if prices[exit_idx] > entry_price else -1
                    else:
                        label = 1 if prices[exit_idx] < entry_price else -1
                else:
                    exit_idx = n - 1
                    if direction > 0:
                        label = 1 if prices[exit_idx] > entry_price else -1
                    else:
                        label = 1 if prices[exit_idx] < entry_price else -1

            results.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': prices[exit_idx],
                'volatility': entry_vol,
                'direction': direction,
                'label': label,
                'returns': np.log(prices[exit_idx] / entry_price) * direction  # Direction-adjusted returns
            })

        df = pd.DataFrame(results)
        df.index = data.index[:len(results)]

        return df