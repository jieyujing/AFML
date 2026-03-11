"""数据加载器工具函数"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """加载 CSV 文件

    Args:
        filepath: 文件路径
        **kwargs: 传递给 pd.read_csv 的参数

    Returns:
        DataFrame
    """
    return pd.read_csv(filepath, **kwargs)


def load_parquet(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """加载 Parquet 文件

    Args:
        filepath: 文件路径
        **kwargs: 传递给 pd.read_parquet 的参数

    Returns:
        DataFrame
    """
    return pd.read_parquet(filepath, **kwargs)


def load_hdf5(filepath: Union[str, Path], key: Optional[str] = None) -> pd.DataFrame:
    """加载 HDF5 文件

    Args:
        filepath: 文件路径
        key: HDF5 数据集键名

    Returns:
        DataFrame
    """
    import h5py

    with h5py.File(filepath, 'r') as f:
        if key is None:
            keys = list(f.keys())
            if keys:
                key = keys[0]

        dataset = f[key][:]
        return pd.DataFrame(dataset)


def load_data(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """智能加载数据文件

    Args:
        filepath: 文件路径
        **kwargs: 传递给加载函数的参数

    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        return load_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        return load_parquet(filepath, **kwargs)
    elif suffix in ['.h5', '.hdf5']:
        return load_hdf5(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


def save_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs
):
    """保存数据到文件

    Args:
        df: DataFrame
        filepath: 保存路径
        **kwargs: 传递给保存函数的参数
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        df.to_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif suffix in ['.h5', '.hdf5']:
        key = kwargs.get('key', 'data')
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(key, data=df.values)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


def prepare_ohlc_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """准备 OHLCV 数据

    Args:
        df: 原始 DataFrame
        **kwargs: 列名映射

    Returns:
        标准化的 OHLCV DataFrame
    """
    # 默认列名映射
    column_mapping = kwargs.get('column_mapping', {
        'timestamp': kwargs.get('timestamp_col', 'timestamp'),
        'open': kwargs.get('open_col', 'open'),
        'high': kwargs.get('high_col', 'high'),
        'low': kwargs.get('low_col', 'low'),
        'close': kwargs.get('close_col', 'close'),
        'volume': kwargs.get('volume_col', 'volume')
    })

    # 重命名列
    df_copy = df.copy()
    for new_name, old_name in column_mapping.items():
        if old_name in df_copy.columns:
            df_copy[new_name] = df_copy[old_name]

    # 确保必要的列存在
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_copy.columns]

    if missing_cols:
        raise ValueError(f"缺少必要的列：{missing_cols}")

    # 选择 OHLCV 列
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_cols = [col for col in ohlcv_cols if col in df_copy.columns]

    result = df_copy[ohlcv_cols].copy()

    # 设置索引
    if 'timestamp' in df_copy.columns:
        result.index = pd.to_datetime(df_copy['timestamp'])
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
        # 尝试从第一列解析时间戳
        first_col = df_copy.columns[0]
        try:
            result.index = pd.to_datetime(df_copy[first_col])
        except:
            pass

    # 排序索引
    result = result.sort_index()

    return result


def generate_sample_data(
    n_samples: int = 10000,
    start_date: str = '2023-01-01',
    freq: str = '1min',
    seed: int = 42
) -> pd.DataFrame:
    """生成模拟 OHLCV 数据

    Args:
        n_samples: 样本数量
        start_date: 开始日期
        freq: 时间频率
        seed: 随机种子

    Returns:
        OHLCV DataFrame
    """
    np.random.seed(seed)

    # 生成时间戳
    dates = pd.date_range(start_date, periods=n_samples, freq=freq)

    # 随机游走生成价格
    returns = np.random.normal(0, 0.001, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成 OHLC
    open_prices = prices * (1 + np.random.uniform(-0.001, 0.001, n_samples))
    high_prices = prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples)))
    low_prices = prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples)))
    close_prices = prices

    # 生成成交量
    volume = np.random.exponential(1000, n_samples).astype(int)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    df.index.name = 'timestamp'

    return df


def validate_ohlc_data(df: pd.DataFrame) -> tuple[bool, list]:
    """验证 OHLC 数据

    Args:
        df: DataFrame

    Returns:
        (是否有效，错误列表)
    """
    errors = []

    # 检查必要的列
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"缺少必要的列：{missing_cols}")

    if errors:
        return False, errors

    # 检查数据类型
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"列 {col} 必须是数值类型")

    # 检查 OHLC 关系
    if (df['high'] < df['low']).any():
        errors.append("存在 High < Low 的数据")

    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        errors.append("存在 High < Open/Close 的数据")

    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        errors.append("存在 Low > Open/Close 的数据")

    # 检查空值
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        errors.append(f"存在空值：{null_counts[null_counts > 0].to_dict()}")

    return len(errors) == 0, errors
