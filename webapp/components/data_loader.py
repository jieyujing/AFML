"""数据加载组件"""
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import tempfile


def render_data_loader() -> Optional[pd.DataFrame]:
    """渲染数据加载器

    Returns:
        加载的 DataFrame 或 None
    """
    st.markdown("### 📥 数据加载")

    # 数据源选择
    source_option = st.radio(
        "选择数据源",
        ["上传文件", "从目录选择", "示例数据"],
        horizontal=True
    )

    df = None

    if source_option == "上传文件":
        df = _render_file_uploader()
    elif source_option == "从目录选择":
        df = _render_directory_selector()
    else:
        df = _render_sample_data()

    return df


def _render_file_uploader() -> Optional[pd.DataFrame]:
    """渲染文件上传器

    Returns:
        加载的 DataFrame 或 None
    """
    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=['csv', 'parquet', 'h5', 'hdf5'],
        help="支持 CSV、Parquet、HDF5 格式"
    )

    if uploaded_file is None:
        return None

    return _load_file(uploaded_file)


def _render_directory_selector() -> Optional[pd.DataFrame]:
    """渲染目录选择器

    Returns:
        加载的 DataFrame 或 None
    """
    # 默认数据目录
    default_data_dir = Path(__file__).parent.parent.parent / "data"

    data_dir = st.text_input(
        "数据目录路径",
        value=str(default_data_dir) if default_data_dir.exists() else ""
    )

    if not data_dir or not Path(data_dir).exists():
        return None

    # 查找数据文件
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    parquet_files = list(data_path.glob("*.parquet"))
    h5_files = list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5"))

    all_files = csv_files + parquet_files + h5_files

    if not all_files:
        st.warning("目录下没有找到数据文件")
        return None

    # 文件选择
    file_options = [str(f) for f in all_files]
    selected_file = st.selectbox("选择文件", file_options)

    if not selected_file:
        return None

    return _load_file(selected_file)


def _render_sample_data() -> pd.DataFrame:
    """渲染示例数据生成器

    Returns:
        示例 DataFrame
    """
    st.info("生成模拟交易数据用于演示")

    n_samples = st.slider("样本数量", 1000, 100000, 10000)

    if st.button("生成示例数据"):
        # 生成模拟数据
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

        # 生成随机价格数据
        import numpy as np
        np.random.seed(42)

        # 随机游走生成价格
        returns = np.random.normal(0, 0.001, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))

        # 生成成交量
        volume = np.random.exponential(1000, n_samples).astype(int)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
            'close': prices,
            'volume': volume
        }, index=dates)

        df = df[['open', 'high', 'low', 'close', 'volume']]

        st.success(f"生成了 {n_samples} 条模拟数据")
        return df

    return None


def _load_file(filepath) -> Optional[pd.DataFrame]:
    """加载文件

    Args:
        filepath: 文件路径或上传的文件对象

    Returns:
        加载的 DataFrame 或 None
    """
    try:
        if hasattr(filepath, 'name'):
            # 上传的文件对象
            file_name = filepath.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif file_name.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif file_name.endswith(('.h5', '.hdf5')):
                # HDF5 需要特殊处理
                st.warning("HDF5 文件需要指定 key，请确保文件包含有效的 DataFrame")
                return None
            else:
                st.error(f"不支持的文件格式：{file_name}")
                return None
        else:
            # 文件路径
            file_path = Path(filepath)
            file_name = file_path.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif file_name.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif file_name.endswith(('.h5', '.hdf5')):
                import h5py
                with h5py.File(filepath, 'r') as f:
                    keys = list(f.keys())
                    if keys:
                        selected_key = st.selectbox("选择数据集", keys)
                        df = pd.DataFrame(f[selected_key][:])
                    else:
                        st.warning("HDF5 文件中没有数据集")
                        return None
            else:
                st.error(f"不支持的文件格式：{file_name}")
                return None

        return df

    except Exception as e:
        st.error(f"加载文件失败：{str(e)}")
        return None


def render_data_preview(df: pd.DataFrame, max_rows: int = 10):
    """渲染数据预览

    Args:
        df: 要预览的 DataFrame
        max_rows: 最大显示行数
    """
    st.markdown("### 📊 数据预览")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("行数", df.shape[0])
    with col2:
        st.metric("列数", df.shape[1])
    with col3:
        st.metric("内存使用", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    st.markdown("#### 列名")
    st.write(list(df.columns))

    st.markdown("#### 数据样例")
    st.dataframe(df.head(max_rows))

    st.markdown("#### 数据类型")
    dtype_df = pd.DataFrame({
        '类型': df.dtypes,
        '非空值': df.count(),
        '空值': df.isnull().sum()
    })
    st.dataframe(dtype_df)


def render_bar_config() -> dict:
    """渲染 K 线配置编辑器

    Returns:
        K 线配置字典
    """
    st.markdown("### ⚙️ K 线配置")

    col1, col2 = st.columns(2)

    with col1:
        bar_type = st.selectbox(
            "K 线类型",
            ["time", "tick", "volume", "dollar", "cusum", "imbalance"],
            format_func=lambda x: {
                'time': '时间 K 线',
                'tick': 'Tick K 线',
                'volume': '成交量 K 线',
                'dollar': '金额 K 线',
                'cusum': 'CUSUM K 线',
                'imbalance': '不平衡 K 线'
            }.get(x, x)
        )

    with col2:
        if bar_type == 'time':
            interval = st.selectbox(
                "时间间隔",
                ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
            )
            return {'type': bar_type, 'interval': interval}
        elif bar_type in ['volume', 'dollar']:
            threshold = st.number_input("阈值", min_value=100, value=10000)
            return {'type': bar_type, 'threshold': threshold}
        elif bar_type == 'cusum':
            threshold = st.number_input("CUSUM 阈值", min_value=0.01, value=0.05, step=0.01)
            return {'type': bar_type, 'threshold': threshold}
        else:
            return {'type': bar_type}
