# AFMLKit — 端到端工作流参考

## 标准 AFML 量化流程

```python
import pandas as pd
import numpy as np
from afmlkit.bar.data_model import TradesData
from afmlkit.bar.kit import DollarBarKit
from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.feature.core.volatility import VolatilityTransform
from afmlkit.sampling.filters import cusum_filter
from afmlkit.label.kit import TBMLabel, SampleWeights

# Step 1: 加载高频交易数据
trades = TradesData.load_trades_h5(
    'data/trades.h5',
    start_time='2023-01-01',
    end_time='2023-12-31',
    enable_multiprocessing=True
)

# Step 2: 构建金额 K 线（比时间 K 线信息含量更均匀）
dollar_bars = DollarBarKit(trades, dollar_thrs=1_000_000)
ohlcv = dollar_bars.build_ohlcv()

# Step 3: 特征工程 —— 波动率估计作为 TBM 目标收益
from afmlkit.feature.core.volatility import VolatilityTransform
vol_feature = Feature(VolatilityTransform('close', 'ewm_vol_50', 50))
kit = FeatureKit([vol_feature], retain=['close'])
features = kit.build(ohlcv, backend='nb', timeit=True)

# Step 4: CUSUM 过滤 —— 事件采样
event_positions = cusum_filter(
    raw_time_series=features['close'].values,
    threshold=features['close_ewm_vol_50'].values  # 动态阈值
)
event_timestamps = features.index[event_positions]

# 将事件索引（纳秒 int64）写入特征 DataFrame
features['event_idx'] = np.nan
features.loc[event_timestamps, 'event_idx'] = event_timestamps.view('int64')

# Step 5: 三重屏障标签
tbm = TBMLabel(
    features=features,
    target_ret_col='close_ewm_vol_50',
    min_ret=0.001,
    horizontal_barriers=(1.0, 1.0),
    vertical_barrier=pd.Timedelta('5D'),
    min_close_time=pd.Timedelta(seconds=1),
    is_meta=False
)
labeled_features, labels_df = tbm.compute_labels(trades)
# labels_df 列：labels, event_idxs, touch_idxs, returns, vertical_touch_weights

# Step 6: 样本权重
weights = tbm.compute_weights(trades, normalized=True)
final_weights = SampleWeights.compute_final_weights(
    avg_uniqueness=weights['avg_uniqueness'],
    return_attribution=weights['return_attribution'],
    vertical_touch_weights=weights['vertical_touch_weights'],
    time_decay_intercept=0.75,       # 0.5=适度衰减, 1.0=无衰减
    labels=labels_df['labels']       # 传入用于类平衡
)

# Step 7: 组装训练集
X = labeled_features.values
y = labels_df['labels'].values
w = final_weights['combined_weights'].values
```

---

## 元标签（Meta-Labeling）工作流

```python
# 前提：已有主模型的方向预测（side）
primary_predictions = primary_model.predict_proba(X)[:, 1]
features_meta = labeled_features.copy()
features_meta['side'] = np.where(primary_predictions > 0.5, 1, -1)

tbm_meta = TBMLabel(
    features=features_meta,
    target_ret_col='close_ewm_vol_50',
    min_ret=0.001,
    horizontal_barriers=(1.0, 1.0),
    vertical_barrier=pd.Timedelta('5D'),
    is_meta=True   # 元标签模式
)
meta_features, meta_labels = tbm_meta.compute_labels(trades)
# 元标签：1=信任主模型预测, 0=不信任
```

---

## HDF5 数据存储工作流

```python
from afmlkit.bar.data_model import TradesData
from afmlkit.bar.io import AddTimeBarH5, TimeBarReader, H5Inspector

# 1. 保存原始交易数据
trades = TradesData(ts, px, qty, id=ids, timestamp_unit='ms', preprocess=True)
trades.save_h5('data/trades.h5')

# 2. 批量生成 1 秒时间 K 线（追加到同一 H5 文件）
processor = AddTimeBarH5('data/trades.h5')
results = processor.process_all(overwrite=False)

# 3. 读取时间 K 线（支持任意 timeframe 重采样）
reader = TimeBarReader('data/trades.h5')
daily_bars = reader.read('2023-01-01', '2023-12-31', timeframe='1D')

# 4. 数据质量检查
inspector = H5Inspector('data/trades.h5')
inspector.get_integrity_summary(verbose=True)
gaps = inspector.inspect_gaps(max_gap=pd.Timedelta(minutes=5))
```

---

## 自定义 Transform 示例

```python
from afmlkit.feature.base import SISOTransform, MISOTransform
import numpy as np
import pandas as pd

# 单输入单输出
class ZScoreTransform(SISOTransform):
    def __init__(self, input_col: str, window: int):
        super().__init__(input_col, f'zscore_{window}')
        self.window = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        col = x[self.requires[0]]
        mu = col.rolling(self.window).mean()
        sigma = col.rolling(self.window).std()
        return ((col - mu) / sigma).rename(self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        # 高性能版本 —— 调用 Numba 函数
        data = self._prepare_input_nb(x)
        result = zscore_nb(data, self.window)       # 自定义 Numba 函数
        return self._prepare_output_nb(x.index, result)

# 多输入单输出
class PriceRatioTransform(MISOTransform):
    def __init__(self, col1: str, col2: str):
        super().__init__([col1, col2], f'{col1}_{col2}_ratio')

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        c1, c2 = self.requires
        return (x[c1] / x[c2]).rename(self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        inputs = self._prepare_input_nb(x)
        result = inputs[self.requires[0]] / inputs[self.requires[1]]
        return self._prepare_output_nb(x.index, result)
```
