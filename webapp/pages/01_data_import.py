"""数据导入与 K 线构建页面"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="数据导入", page_icon="📥", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.data_loader import render_data_loader, render_data_preview
from components.charts import render_candlestick_chart, render_line_chart
from utils.data_loader import prepare_ohlc_data, validate_ohlc_data, generate_sample_data
from utils.csv_loader import CSVDataloader

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("📥 数据导入与 K 线构建")

# 数据源选择
data_source = st.radio(
    "选择数据源",
    ["通用数据加载", "从 data/csv 目录加载"],
    horizontal=True
)

if data_source == "从 data/csv 目录加载":
    st.markdown("### 📂 从 data/csv 目录加载")

    loader = CSVDataloader("data/csv")
    available_files = loader.list_available_files()

    if not available_files:
        st.warning("data/csv 目录下没有找到 CSV 文件")
    else:
        file_options = {f.name: f for f in available_files}
        selected_file = st.selectbox("选择文件", list(file_options.keys()))

        if selected_file:
            filepath = file_options[selected_file]

            # 显示文件信息
            file_info = loader.get_file_info(filepath)
            st.json({
                "文件名": file_info.get('filename', ''),
                "大小": f"{file_info.get('size_mb', 0)} MB",
                "列名": file_info.get('columns', [])
            })

            if st.button("加载文件"):
                try:
                    df, warnings = loader.load_and_validate(filepath)
                    SessionManager.update('raw_data', df)
                    SessionManager.update('csv_warnings', warnings)
                    st.success(f"成功加载 {len(df)} 行数据")
                    if warnings:
                        for w in warnings:
                            st.warning(w)
                except Exception as e:
                    st.error(f"加载失败：{str(e)}")
else:
    st.markdown("### 📥 通用数据加载")

# 步骤选择
step = st.radio("选择步骤", ["1. 数据导入", "2. 数据验证", "3. K 线构建"], horizontal=True)

if step == "1. 数据导入":
    st.markdown("### 1️⃣ 数据导入")

    # 使用数据加载器
    df = render_data_loader()

    if df is not None:
        # 保存到会话
        SessionManager.update('raw_data', df)
        st.success(f"成功加载 {len(df)} 行数据")

        # 显示预览
        render_data_preview(df)

        # 尝试自动识别 OHLCV 列
        st.markdown("### 🔍 自动列识别")

        # 常见的列名映射
        ohlc_patterns = {
            'open': ['open', 'o', '开盘', '开盘价'],
            'high': ['high', 'h', '最高', '最高价'],
            'low': ['low', 'l', '最低', '最低价'],
            'close': ['close', 'c', '收盘', '收盘价'],
            'volume': ['volume', 'v', 'vol', '成交量', 'amount', '金额']
        }

        detected_cols = {}
        for col_name, patterns in ohlc_patterns.items():
            for pattern in patterns:
                if pattern in df.columns.str.lower():
                    detected_cols[col_name] = df.columns[df.columns.str.lower() == pattern][0]
                    break

        if detected_cols:
            st.success("检测到以下列：")
            st.json(detected_cols)

            # 确认列映射
            st.markdown("#### 列映射确认")
            col_mapping = {}
            cols = st.columns(5)

            for i, (target_col, detected) in enumerate(detected_cols.items()):
                with cols[i % 5]:
                    options = [''] + list(df.columns)
                    selected = st.selectbox(
                        target_col.capitalize(),
                        options,
                        index=options.index(detected) if detected in options else 0,
                        key=f"col_map_{target_col}"
                    )
                    if selected:
                        col_mapping[target_col] = selected

            if st.button("确认列映射"):
                SessionManager.update('column_mapping', col_mapping)
                st.success("列映射已保存")

elif step == "2. 数据验证":
    st.markdown("### 2️⃣ 数据验证")

    raw_data = SessionManager.get('raw_data')

    if raw_data is None:
        st.warning("请先导入数据")
    else:
        # 获取列映射
        col_mapping = SessionManager.get('column_mapping', {})

        # 准备 OHLCV 数据
        try:
            ohlc_data = prepare_ohlc_data(raw_data, column_mapping=col_mapping)
            SessionManager.update('prepared_data', ohlc_data)

            # 验证数据
            is_valid, errors = validate_ohlc_data(ohlc_data)

            if is_valid:
                st.success("✅ 数据验证通过")
            else:
                st.error("❌ 数据验证失败：")
                for error in errors:
                    st.write(f"- {error}")

            # 显示数据统计
            st.markdown("#### 数据统计")
            st.dataframe(ohlc_data.describe())

            # 显示 K 线图
            st.markdown("#### K 线图预览")
            fig = render_candlestick_chart(ohlc_data, limit=100)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"处理数据时出错：{str(e)}")

elif step == "3. K 线构建":
    st.markdown("### 3️⃣ K 线构建")

    raw_data = SessionManager.get('raw_data')
    prepared_data = SessionManager.get('prepared_data')

    if prepared_data is None and raw_data is not None:
        # 尝试自动准备数据
        col_mapping = SessionManager.get('column_mapping', {})
        try:
            prepared_data = prepare_ohlc_data(raw_data, column_mapping=col_mapping)
            SessionManager.update('prepared_data', prepared_data)
        except Exception as e:
            st.error(f"准备数据失败：{str(e)}")
            prepared_data = None

    if prepared_data is None:
        st.warning("请先导入并准备数据")
    else:
        # K 线类型选择
        st.markdown("#### K 线类型")
        bar_type = st.selectbox(
            "选择 K 线类型",
            ["time", "tick", "volume", "dollar"],
            format_func=lambda x: {
                'time': '时间 K 线',
                'tick': 'Tick K 线',
                'volume': '成交量 K 线',
                'dollar': '金额 K 线'
            }.get(x, x)
        )

        # K 线参数配置
        st.markdown("#### 参数配置")

        if bar_type == 'time':
            col1, col2 = st.columns(2)
            with col1:
                interval = st.selectbox("时间间隔", ['1min', '5min', '15min', '30min', '1h', '4h', '1d'])
            with col2:
                method = st.selectbox("聚合方法", ['ohlc', 'close'])

            if st.button("构建时间 K 线"):
                try:
                    # 使用 pandas 重采样
                    bar_data = prepared_data.copy()
                    bar_data = bar_data.resample(interval).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                    bar_data = bar_data.dropna()

                    SessionManager.update('bar_data', bar_data)
                    SessionManager.update('bar_config', {'type': bar_type, 'interval': interval})

                    st.success(f"成功构建 {len(bar_data)} 根 K 线")

                    # 显示预览
                    st.markdown("#### K 线数据预览")
                    st.dataframe(bar_data.head(10))

                    # 显示图表
                    fig = render_candlestick_chart(bar_data, limit=100)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"构建 K 线失败：{str(e)}")

        elif bar_type in ['volume', 'dollar']:
            threshold = st.number_input("阈值", min_value=100, value=1000, step=100)

            if st.button(f"构建 {bar_type} K 线"):
                # 这里需要调用 afmlkit.bar 模块
                st.info(f"构建{bar_type}K 线功能需要调用 afmlkit.bar 模块")

                try:
                    from afmlkit.bar import BarKit

                    kit = BarKit(bar_type=bar_type, threshold=threshold)
                    bar_data = kit.build(prepared_data)

                    SessionManager.update('bar_data', bar_data)
                    SessionManager.update('bar_config', {'type': bar_type, 'threshold': threshold})

                    st.success(f"成功构建 {len(bar_data)} 根 K 线")
                    st.dataframe(bar_data.head(10))

                except ImportError:
                    st.warning("afmlkit.bar 模块尚未实现，将使用简单的聚合方法")

                    # 简单的实现
                    bar_data_list = []
                    current_bar = None
                    current_sum = 0

                    for idx, row in prepared_data.iterrows():
                        if current_bar is None:
                            current_bar = {
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume'],
                                'start_time': idx
                            }
                            current_sum = row['volume'] if bar_type == 'volume' else row['close'] * row['volume']
                        else:
                            current_bar['high'] = max(current_bar['high'], row['high'])
                            current_bar['low'] = min(current_bar['low'], row['low'])
                            current_bar['close'] = row['close']
                            current_bar['volume'] += row['volume']
                            current_sum += row['volume'] if bar_type == 'volume' else row['close'] * row['volume']

                            if current_sum >= threshold:
                                bar_data_list.append(current_bar)
                                current_bar = None
                                current_sum = 0

                    if current_bar:
                        bar_data_list.append(current_bar)

                    bar_data = pd.DataFrame(bar_data_list)
                    bar_data.index = pd.to_datetime([b['start_time'] for b in bar_data_list])

                    SessionManager.update('bar_data', bar_data)
                    SessionManager.update('bar_config', {'type': bar_type, 'threshold': threshold})

                    st.success(f"成功构建 {len(bar_data)} 根 K 线")
                    st.dataframe(bar_data.head(10))

        # 下一步操作
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if SessionManager.get('bar_data') is not None:
        if st.button("➡️ 前往特征工程", use_container_width=True):
            navigate_to("2️⃣ 特征工程")
            st.rerun()
