"""CSV 数据加载器 - 支持 OHLCV 数据加载和验证"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# 常见列名映射（支持中英文）
COLUMN_ALIASES = {
    'datetime': ['datetime', 'time', 'date', 'timestamp', 'dt', '时间', '日期'],
    'open': ['open', 'o', '开盘', '开盘价'],
    'high': ['high', 'h', '最高', '最高价'],
    'low': ['low', 'l', '最低', '最低价'],
    'close': ['close', 'c', '收盘', '收盘价'],
    'volume': ['volume', 'v', 'vol', '成交量', '量'],
    'open_interest': ['open_interest', 'oi', '持仓量', '未平仓'],
    'amount': ['amount', '金额', '成交额'],
}


class CSVDataloader:
    """
    CSV 数据加载器

    支持从指定目录加载 CSV 文件，自动识别和验证 OHLCV 格式。
    """

    def __init__(self, data_dir: str = "data/csv"):
        """
        初始化 CSV 加载器

        :param data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self._ensure_dir_exists()

    def _ensure_dir_exists(self):
        """确保数据目录存在"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir}")

    def list_available_files(self) -> List[Path]:
        """
        列出目录下所有可用的 CSV 文件

        :return: CSV 文件路径列表
        """
        if not self.data_dir.exists():
            return []

        files = list(self.data_dir.glob("*.csv"))
        return sorted(files, key=lambda x: x.name)

    def load_file(
        self,
        filepath: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
        parse_dates: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        加载 CSV 文件

        :param filepath: 文件路径
        :param column_mapping: 列名映射字典，如 {'datetime': '时间', 'close': '收盘价'}
        :param parse_dates: 是否解析日期列
        :param kwargs: 传递给 pd.read_csv 的其他参数
        :return: 加载的 DataFrame
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在：{filepath}")

        # 读取 CSV
        df = pd.read_csv(filepath, **kwargs)

        # 自动检测列名映射
        if column_mapping is None:
            column_mapping = self._auto_detect_columns(df)

        # 重命名列
        df = self._rename_columns(df, column_mapping)

        # 解析日期并设置为索引
        if parse_dates and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            df = df.sort_index()

        return df

    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        自动检测列名映射

        :param df: 原始 DataFrame
        :return: 列名映射字典
        """
        mapping = {}
        df_columns_lower = df.columns.str.lower().str.strip()

        for standard_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                alias_lower = alias.lower().strip()
                # 精确匹配
                if alias_lower in df_columns_lower:
                    idx = df_columns_lower.get_loc(alias_lower)
                    mapping[standard_name] = df.columns[idx]
                    break
                # 模糊匹配（包含）
                elif df_columns_lower.str.contains(alias_lower, regex=False).any():
                    idx = df_columns_lower[df_columns_lower.str.contains(alias_lower, regex=False)].argmax()
                    mapping[standard_name] = df.columns[idx]
                    break

        return mapping

    def _rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        根据映射重命名列

        :param df: 原始 DataFrame
        :param mapping: 列名映射字典
        :return: 重命名后的 DataFrame
        """
        rename_dict = {}
        for standard_name, original_name in mapping.items():
            if original_name in df.columns and standard_name != original_name:
                rename_dict[original_name] = standard_name

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证 OHLCV 格式

        :param df: DataFrame
        :return: (是否有效，错误列表)
        """
        errors = []
        required_columns = ['open', 'high', 'low', 'close']

        # 检查必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"缺少必要的列：{missing_cols}")
            return False, errors

        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"列 {col} 必须是数值类型")

        # 检查 OHLC 关系
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                errors.append("存在 High < Low 的数据")

        if 'high' in df.columns and 'close' in df.columns:
            if (df['high'] < df['close']).any():
                errors.append("存在 High < Close 的数据")

        if 'low' in df.columns and 'close' in df.columns:
            if (df['low'] > df['close']).any():
                errors.append("存在 Low > Close 的数据")

        # 检查空值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"存在空值：{null_counts[null_counts > 0].to_dict()}")

        # 检查索引
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("索引必须是 DatetimeIndex")

        return len(errors) == 0, errors

    def load_and_validate(
        self,
        filepath: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        加载并验证 CSV 文件

        :param filepath: 文件路径（可以是相对路径）
        :param column_mapping: 列名映射字典
        :return: (DataFrame, 警告列表)
        """
        filepath = Path(filepath)

        # 简化路径处理逻辑
        if not filepath.is_absolute():
            # 1. 先尝试直接使用相对路径
            if filepath.exists():
                pass  # 直接使用原路径
            # 2. 如果不存在，尝试在 data_dir 中查找
            elif (self.data_dir / filepath).exists():
                filepath = self.data_dir / filepath
            # 3. 如果仍不存在，保持原路径让 load_file 抛出异常
            #    或者尝试不带目录名的文件名
            elif not filepath.exists():
                filename_only = Path(filepath.name)
                candidate = self.data_dir / filename_only
                if candidate.exists():
                    filepath = candidate

        df = self.load_file(filepath, column_mapping)
        is_valid, errors = self.validate_ohlcv(df)

        if not is_valid:
            raise ValueError(f"数据验证失败：\n" + "\n".join(errors))

        warnings = []

        # 检查数据连续性（对于 1 分钟数据）
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            time_diffs = df.index.to_series().diff()
            median_diff = time_diffs.median()

            # 如果是 1 分钟数据
            if median_diff <= pd.Timedelta(minutes=2):
                gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
                if len(gaps) > 0:
                    warnings.append(f"检测到 {len(gaps)} 个数据间隔大于 5 分钟")

        return df, warnings

    def get_file_info(self, filepath: Union[str, Path]) -> Dict:
        """
        获取文件基本信息

        :param filepath: 文件路径
        :return: 文件信息字典
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {}

        stat = filepath.stat()

        stat = filepath.stat()

        # 快速读取前几行获取列信息
        df_head = pd.read_csv(filepath, nrows=5)

        return {
            'filename': filepath.name,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'columns': list(df_head.columns),
            'estimated_rows': self._estimate_row_count(filepath)
        }

    def _estimate_row_count(self, filepath: Path) -> int:
        """估算文件行数"""
        with open(filepath, 'rb') as f:
            # 读取前 10KB 估算平均每行长度
            sample = f.read(10240)
            lines = sample.count(b'\n')
            if lines == 0:
                return 0
            avg_line_size = len(sample) / lines
            return int(filepath.stat().st_size / avg_line_size)


def load_csv(
    filepath: Union[str, Path],
    column_mapping: Optional[Dict[str, str]] = None,
    data_dir: str = "data/csv"
) -> pd.DataFrame:
    """
    便捷函数：加载 CSV 文件

    :param filepath: 文件路径（可以是相对路径）
    :param column_mapping: 列名映射字典
    :param data_dir: 默认数据目录
    :return: 加载的 DataFrame
    """
    loader = CSVDataloader(data_dir)

    # 如果是相对路径，尝试在数据目录中查找
    if not Path(filepath).is_absolute():
        filepath = Path(data_dir) / filepath

    return loader.load_file(filepath, column_mapping)


def validate_csv(
    filepath: Union[str, Path],
    column_mapping: Optional[Dict[str, str]] = None,
    data_dir: str = "data/csv"
) -> Tuple[bool, List[str]]:
    """
    便捷函数：验证 CSV 文件格式

    :param filepath: 文件路径
    :param column_mapping: 列名映射字典
    :param data_dir: 默认数据目录
    :return: (是否有效，错误列表)
    """
    loader = CSVDataloader(data_dir)
    df = loader.load_file(filepath, column_mapping)
    return loader.validate_ohlcv(df)
